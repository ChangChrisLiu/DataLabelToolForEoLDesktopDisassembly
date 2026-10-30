"""Unified ``(desktop, step, view)`` frame index over the four capture sources.

The index is the canonical map every later module keys on.  Logical steps come
from the OAK ``Camera_1`` captures ordered by filename timestamp (spec 2.2);
``Camera_2`` is paired to them by identical timestamp, while the scanner and the
RealSense carry *nominal* step numbers in their folder names, which are resolved
to logical steps through the matching ``Camera_1`` folder.

Known source anomalies live in ``configs/index_fixes.yaml``; see
:func:`scan_desktop` for how each one is applied and verified.  Nothing here ever
writes to the source roots, and directories are enumerated with ``os.scandir``
at the known depths only - never with a recursive walk (see
:mod:`tda.core.sources` for the layout primitives).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from tda.core.index_fixes import (
    cam2_fix_step,
    unphotographed_steps,
    verify_cam2_fixes,
    verify_declared_missing,
    verify_scan_rule,
    verify_unphotographed,
)
from tda.core.index_report import write_report
from tda.core.model import VIEWS, FrameKey
from tda.core.sources import (
    BURST_RE,
    COMPONENT_CAMERA_DIRS,
    COMPONENTS_DIR,
    DEFAULT_FOLDER_ALIASES,
    SCAN_SHOT_RE,
    STEP_DIR_RE,
    dirs,
    files,
    first_existing,
    iso_from_mtime,
    iso_from_oak_token,
    norm,
    scan_oak_camera,
    scan_oak_step_dir,
    scanner_components_dir,
    scanner_desktop_dir,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FIXES_PATH = "configs/index_fixes.yaml"
DEFAULT_PATHS_PATH = "configs/paths.yaml"
#: 2: the components gallery is indexed (task U10). A version-1 file is still
#: read (``DesktopIndex.gallery`` unset), but ``load-index`` refuses it
#: (:func:`tda.pipeline.stale_index`): it predates the gallery and the
#: RealSense ``rs_extra_from`` fixes.
#: 3: the sheet's ``dupli`` rows are collapsed and the scanner is mapped by
#: ``duplicate_folders`` (task U25); ``load-index`` refuses a version-2 file too,
#: because its steps are the uncollapsed rows.
#: 4: the scanner's all-components scans are the gallery view ``cmp_scan``
#: (task U42); ``load-index`` refuses a version-3 file too: it says nothing
#: about them, and D66's "All components" scanner frame would go nowhere.
INDEX_VERSION = 4
#: The first version whose scan covers the components gallery.
GALLERY_INDEX_VERSION = 2
#: The first version whose steps are the collapsed ones (task U25).
COLLAPSED_INDEX_VERSION = 3
#: The first version whose scan covers the scanner's gallery (task U42).
SCAN_GALLERY_INDEX_VERSION = 4
#: A scanner shot folder holding more images than this is not a scan of the
#: laid-out parts but a sequence (D64's ``allcomponentsseq11``: 176 fringe
#: projection images); a scan is a burst of ten.
SCAN_SHOT_MAX_BURST = 20
#: How far a RealSense file mtime may sit from its OAK cam1 capture time.
RS_TIME_TOLERANCE_S = 30.0

__all__ = [
    "DesktopIndex", "FrameFile", "build_index", "load_fixes", "load_index",
    "load_roots", "respell_paths", "save_index", "scan_desktop", "write_report",
]


# ---------------------------------------------------------------------------
# data types
# ---------------------------------------------------------------------------
@dataclass
class FrameFile:
    """One image of one view at one logical step, plus its companion files."""

    key: FrameKey
    path: str  # main image (scan: P_0.png; oak: rgb_12mp.jpg; rs: original_color.png)
    aux: dict[str, Any] = field(default_factory=dict)  # aligned/depth_npy/... and burst list
    ts: Optional[str] = None  # ISO capture time (OAK: filename token; else file mtime)
    src_step_dir: str = ""  # original folder name, e.g. "016" or "RGB161"


@dataclass
class DesktopIndex:
    """All frames of one desktop, keyed by logical step."""

    desktop: int
    # logical steps = OAK cam1 captures after fixes, minus the dupli rows, plus
    # the steps no camera photographed (U40)
    n_steps: int
    frames: dict[FrameKey, FrameFile] = field(default_factory=dict)
    missing: list[FrameKey] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    #: Did the scan look at the components gallery (task U10)? ``False`` for an
    #: index file written before it existed (version 1): such an index says
    #: nothing about the gallery, and ``load-index`` refuses it (build-index again).
    gallery: bool = True
    #: Were the sheet's ``dupli`` rows collapsed (task U25)? ``False`` for a file
    #: written before (version 2): its steps are the uncollapsed rows, and
    #: ``load-index`` refuses it.
    collapsed: bool = True
    #: Camera_1 captures before the ``dupli`` rows were collapsed (task U25);
    #: ``0`` in an index written before that, where it equals ``n_steps``.
    n_rows: int = 0
    #: The Camera_1 folder of every capture, in capture order (row ``r`` is
    #: ``rows[r - 1]``): how a folder number written elsewhere -- a Label Studio
    #: image name -- finds its row, and through :func:`collapse_rows` its step.
    rows: list[str] = field(default_factory=list)
    #: The sheet's ``dupli`` rows the steps were collapsed by (sheet numbering).
    dupli_rows: list[int] = field(default_factory=list)
    #: The steps no camera photographed (``unphotographed.<d>.steps``, task
    #: U40): every view's captures sit on the other steps, so capture ``c`` is
    #: the ``c``-th sheet row that is not one of theirs (:func:`capture_steps`).
    #: ``[]`` in an index written before, where no desktop declared any.
    unphotographed: list[int] = field(default_factory=list)
    #: Did the scan look at the scanner's all-components scans (``cmp_scan``,
    #: task U42)? ``False`` for a file written before (version 3 or older):
    #: ``load-index`` refuses it.
    scan_gallery: bool = True


#: ``FrameFile.aux`` key of a step's extra shots (task U25): the captures of a
#: ``dupli`` row and the scanner's duplicate folders. A list of
#: ``{"path", "src_step_dir", "ts", "row"}`` -- not frames the app shows.
EXTRA_SHOTS = "extra_shots"
#: A photo of the laid-out parts taken in the sequence -- a sheet row of them
#: (D66 step 13, put there by ``load-index``) or a Camera_1 capture that is no
#: row at all (``oak.<d>.gallery_captures``, D1's 040) -- is gallery shot
#: ``SEQUENCE_SHOT_BASE + number``: out of reach of the Components folders' own
#: shots 1..N, so it is never renumbered under work drawn on it.
SEQUENCE_SHOT_BASE = 1000
#: ``aux`` key naming the sequence capture a gallery shot came from.
SEQUENCE_CAPTURE_KEY = "sequence_capture"


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------
def _load_yaml(path: str) -> dict:
    """Load a YAML config; ``path`` may be relative to the cwd or to the repo root."""
    import yaml

    if not (os.path.isabs(path) or os.path.exists(path)):
        path = os.path.join(REPO_ROOT, path)
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_fixes(path: str = DEFAULT_FIXES_PATH) -> dict:
    """Load the anomaly fix rules from ``index_fixes.yaml``."""
    return _load_yaml(path)


def load_roots(path: str = DEFAULT_PATHS_PATH) -> dict:
    """Load the source roots (``oak_root``/``scanner_root``/``rs_root``) from paths.yaml."""
    cfg = _load_yaml(path)
    return {k: cfg[k] for k in ("oak_root", "scanner_root", "rs_root") if k in cfg}


def collapse_rows(n_rows: int, dupli_rows: Iterable[int]) -> tuple[list[int], set[int]]:
    """``(step_of_row, extra_rows)``: the sheet's rows collapsed into steps (U25).

    A ``dupli`` row repeats the state of the row before it, so it is not a step:
    row ``r`` belongs to step ``r - #(dupli rows <= r)`` and a ``dupli`` row is an
    extra shot of the step before it. ``step_of_row[0]`` is unused; row 1 is
    never a ``dupli`` (nothing precedes it), and rows outside ``2..n_rows`` are
    ignored.
    """
    extra = {int(r) for r in dupli_rows if 1 < int(r) <= int(n_rows)}
    out, seen = [0] * (int(n_rows) + 1), 0
    for r in range(1, int(n_rows) + 1):
        if r in extra:
            seen += 1
        out[r] = r - seen
    return out, extra


def capture_rows(n_captures: int, dupli_rows: Iterable[int],
                 unphotographed: Iterable[int] = ()) -> list[int]:
    """The sheet row of every Camera_1 capture (task U40); index 0 is unused.

    Without ``unphotographed`` steps capture ``c`` is row ``c``. A step no
    camera photographed (``unphotographed.<d>.steps``) is a sheet row without
    a capture, so capture ``c`` is the ``c``-th row that is not one of theirs:
    the rows are ``n_captures + len(unphotographed)``, collapsed by the
    ``dupli`` rows (sheet numbering) as :func:`collapse_rows` does, and a step's
    row is its one row that is not a ``dupli`` row. The steps must lie in
    ``1..n_captures - #dupli + len(unphotographed)`` (the index checks that
    first, :func:`_valid_unphotographed`).
    """
    unph = {int(s) for s in unphotographed}
    if not unph:
        return list(range(int(n_captures) + 1))
    n_sheet = int(n_captures) + len(unph)
    step_of_row, extra = collapse_rows(n_sheet, dupli_rows)
    gone = {r for r in range(1, n_sheet + 1) if r not in extra and step_of_row[r] in unph}
    return [0] + [r for r in range(1, n_sheet + 1) if r not in gone]


def capture_steps(n_captures: int, dupli_rows: Iterable[int],
                  unphotographed: Iterable[int] = ()) -> tuple[list[int], set[int]]:
    """``(step_of_capture, extra_captures)``: the Camera_1 captures on the steps.

    :func:`collapse_rows` over the captures (U25) when no step is declared
    unphotographed; otherwise (U40) capture ``c`` takes the step of its sheet
    row (:func:`capture_rows`) -- so the captures after an unphotographed step
    are one step later each, like the scanner's ``skipped_steps`` -- and it is
    an extra shot when that row is a ``dupli`` row. ``step_of_capture[0]`` is
    unused. Every view that is aligned by Camera_1's captures (OAK2 by
    timestamp, the RealSense by folder number) follows it.
    """
    unph = {int(s) for s in unphotographed}
    if not unph:
        return collapse_rows(n_captures, dupli_rows)
    rows = capture_rows(n_captures, dupli_rows, unph)
    step_of_row, extra = collapse_rows(int(n_captures) + len(unph), dupli_rows)
    return ([0] + [step_of_row[r] for r in rows[1:]],
            {c for c, r in enumerate(rows) if c and r in extra})


def folder_steps(di: "DesktopIndex", view: str) -> dict[int, int]:
    """``{capture folder number: step}`` of one view (task U25).

    The scanner's ``RGB<n>1`` folders are looked up where the index put them
    (an extra shot answers its step); every other view goes through the
    Camera_1 folder of the same number, its row and the collapsed step of that
    row (past the desktop's unphotographed steps, U40). An index written before
    U25 has no ``rows``: the OAK and RealSense numbers are then the steps, and
    ``{}`` says so.
    """
    out: dict[int, int] = {}
    if view == "scan":
        for key, ff in di.frames.items():
            if key.view != "scan":
                continue
            names = [ff.src_step_dir] + [
                shot.get("src_step_dir") for shot in (ff.aux or {}).get(EXTRA_SHOTS) or []
                if isinstance(shot, dict)]
            for name in names:
                if name and name.startswith("RGB") and name.endswith("1") \
                        and name[3:-1].isdigit():
                    out.setdefault(int(name[3:-1]), key.step)
        return out
    if not di.rows:
        return out
    step_of_row, _extra = capture_steps(len(di.rows), di.dupli_rows,
                                        getattr(di, "unphotographed", None) or ())
    for row, folder in enumerate(di.rows, start=1):
        if folder.isdigit():
            out.setdefault(int(folder), step_of_row[row])
    return out


def _pick_capture(tokens: list[str], rule: Optional[str]) -> str:
    """Choose one timestamp out of a re-shot step folder (``latest``/``earliest``)."""
    return min(tokens) if rule == "earliest" else max(tokens)


# ---------------------------------------------------------------------------
# the per-desktop scan
# ---------------------------------------------------------------------------
def scan_desktop(desktop: int, roots: dict, fixes: dict,
                 dupli_rows: Optional[Iterable[int]] = None,
                 sheet_rows: Optional[int] = None) -> DesktopIndex:
    """Build the frame index for one desktop from the four source trees.

    The Camera_1 captures in time order are the sheet's **rows**. ``dupli_rows``
    (the sheet's ``dupli`` rows, :func:`tda.core.logs.dupli_rows_of`; ``None``
    when the sheet is not known) are collapsed away (task U25, spec 2.2): row
    ``r`` is step ``r - #(dupli rows <= r)``, and the capture of a ``dupli`` row
    -- in every OAK and RealSense view -- is an extra shot of the step it repeats
    (``aux["extra_shots"]``), not a frame. The scanner never counted those rows
    the same way, so it is mapped by its own folder numbers and the declared
    ``scan.<d>.duplicate_folders`` (:func:`_add_scanner`).

    ``unphotographed.<d>.steps`` (task U40) are steps no camera photographed
    (D63: the sheet has three rows more than every camera has captures). They
    are sheet rows without a capture, so in every view the ``i``-th capture is
    on the ``i``-th step that is not declared (:func:`capture_steps`; the
    scanner's ``skipped_steps`` rule in all four views) and the declared steps
    are missing frames everywhere. Asserted like the scanner rule
    (:func:`tda.core.index_fixes.verify_unphotographed`).
    """
    fixes = fixes or {}
    oak_fix = (fixes.get("oak") or {}).get(desktop, {}) or {}
    issues: list[str] = []
    frames: dict[FrameKey, FrameFile] = {}

    aliases = (fixes.get("rs") or {}).get("folder_aliases") or list(DEFAULT_FOLDER_ALIASES)
    oak_desktop = os.path.join(roots["oak_root"], f"Desktop {desktop}")
    disassemble = first_existing(oak_desktop, aliases)
    if disassemble is None:
        issues.append(f"oak: no Disassemble folder under {oak_desktop}")
        return DesktopIndex(desktop, 0, frames, [], issues)

    gallery: dict = {"dirs": {}, "ts": {}}
    step_of_dir, step_of_ts, issues_c1 = _build_logical_steps(
        desktop, disassemble, oak_fix, fixes, frames, gallery
    )
    issues.extend(issues_c1)
    n_rows = len(step_of_dir)
    if n_rows == 0:
        issues.append(f"oak: no Camera_1 step folders under {disassemble}")
        return DesktopIndex(desktop, 0, frames, [], issues)
    rows = [src for src, _row in sorted(step_of_dir.items(), key=lambda kv: kv[1])]

    # OAK and RealSense first in row numbering (the sheet's), then collapsed
    leftovers: dict[str, list[str]] = {"oak2": [], "rs": [], "scan": []}
    issues.extend(
        _add_oak_cam2(desktop, disassemble, oak_desktop, oak_fix, step_of_dir, step_of_ts, frames,
                      gallery["ts"], leftover=leftovers["oak2"])
    )
    rs_fix = (fixes.get("rs") or {}).get(desktop, {}) or {}
    issues.extend(_add_realsense(desktop, roots, aliases, step_of_dir, frames, rs_fix,
                                 gallery["dirs"], leftover=leftovers["rs"]))
    issues.extend(_add_components(desktop, roots, oak_desktop, oak_fix, step_of_ts, frames,
                                  rs_fix))
    declared_unph = unphotographed_steps(fixes, desktop)
    dupli, collapse_issues = _valid_dupli(dupli_rows, n_rows, sheet_rows,
                                          unphotographed=len(declared_unph))
    issues.extend(collapse_issues)
    unph, unph_issues = _valid_unphotographed(desktop, declared_unph, n_rows - len(dupli))
    issues.extend(unph_issues)
    n_steps = n_rows - len(dupli) + len(unph)
    if dupli or unph:
        replaces = {int(r) for r in oak_fix.get("dupli_replaces") or []}
        for r in sorted(replaces - set(dupli)):
            issues.append(f"fix:oak[{desktop}] dupli_replaces MISMATCH: row {r} is not a "
                          f"dupli row of the sheet; ignored")
        # the lines so far count capture rows; say which step each one now is
        issues[:] = _restep(issues, capture_steps(n_rows, dupli, unph)[0])
        issues.extend(_collapse(desktop, n_rows, dupli, rows, frames, replaces & set(dupli),
                                unphotographed=unph))

    scan_fix = (fixes.get("scan") or {}).get(desktop, {}) or {}
    issues.extend(_add_scanner(desktop, roots, step_of_dir, frames, n_steps,
                               scan_fix, collapsed=bool(dupli), leftover=leftovers["scan"],
                               unphotographed=unph))
    # the scanner's scans of the laid-out parts (task U42): shots, not steps
    issues.extend(_add_scanner_components(desktop, roots, frames))

    # the gallery's shots are not steps: a camera with fewer of them than
    # another misses nothing, so only the sequence views can miss a frame
    missing = [
        FrameKey(desktop, step, view)
        for step in range(1, n_steps + 1)
        for view in VIEWS
        if FrameKey(desktop, step, view) not in frames
    ]
    issues.extend(verify_declared_missing(desktop, fixes, n_steps, missing))
    issues.extend(verify_scan_rule(desktop, scan_fix, dupli, n_steps, frames, missing,
                                   leftovers["scan"], unphotographed=unph))
    sheet_steps = None if sheet_rows is None else int(sheet_rows) - len(dupli)
    issues.extend(verify_unphotographed(desktop, fixes, unph, n_steps, frames, missing,
                                        leftovers, sheet_steps))
    return DesktopIndex(desktop, n_steps, frames, missing, issues, n_rows=n_rows, rows=rows,
                        dupli_rows=dupli, unphotographed=unph)


def _valid_dupli(dupli_rows: Optional[Iterable[int]], n_rows: int,
                 sheet_rows: Optional[int],
                 unphotographed: int = 0) -> tuple[list[int], list[str]]:
    """The ``dupli`` rows that can be collapsed, and a line for every one that cannot.

    ``unphotographed`` steps (U40) are sheet rows without a Camera_1 capture:
    the sheet then has that many rows more than the captures, and the
    ``dupli`` rows (sheet numbering) reach that much further.
    """
    if dupli_rows is None:
        return [], []
    issues: list[str] = []
    n_sheet = n_rows + int(unphotographed)
    if sheet_rows is not None and int(sheet_rows) != n_sheet:
        if unphotographed:
            issues.append(f"rows: the sheet has {sheet_rows} rows but Camera_1 {n_rows} "
                          f"captures and {unphotographed} unphotographed step(s) make "
                          f"{n_sheet}; the dupli rows are taken as rows of those")
        else:
            issues.append(f"rows: the sheet has {sheet_rows} rows but Camera_1 {n_rows} "
                          f"captures; the dupli rows are taken as capture numbers")
    keep = []
    for r in sorted({int(r) for r in dupli_rows}):
        if 1 < r <= n_sheet:
            keep.append(r)
        else:
            issues.append(f"rows: dupli row {r} has no Camera_1 capture to collapse; ignored")
    return keep, issues


def _valid_unphotographed(desktop: int, declared: list[int],
                          n_photographed: int) -> tuple[list[int], list[str]]:
    """The ``unphotographed`` steps (U40) the index can apply, and its lines.

    ``n_photographed`` is the steps the captures make (Camera_1's captures
    less the ``dupli`` ones); the declared steps add to them, so a step must
    lie in ``1..n_photographed + len(steps)``. One that does not is a
    ``MISMATCH`` and is dropped (and the rest checked again, the bound being one
    lower).
    """
    keep = sorted({int(s) for s in declared})
    issues: list[str] = []
    while keep:
        bound = n_photographed + len(keep)
        bad = [s for s in keep if not 1 <= s <= bound]
        if not bad:
            break
        issues.append(f"fix:unphotographed[{desktop}] MISMATCH: step(s) {bad} are not steps "
                      f"of the {bound} that {n_photographed} photographed steps and "
                      f"{len(keep)} unphotographed ones make; ignored")
        keep = [s for s in keep if s not in bad]
    if keep:
        rest = [s for s in range(1, n_photographed + len(keep) + 1) if s not in keep]
        issues.append(f"fix:unphotographed[{desktop}] steps {keep} applied: no view has a "
                      f"capture of them, the {n_photographed} photographed steps are "
                      f"{_spans(rest)} in every view")
    return keep, issues


def _spans(steps: list[int]) -> str:
    """``[1, 2, 3, 7, 8]`` -> ``"1-3, 7-8"``."""
    out: list[str] = []
    for s in steps:
        if out and out[-1][1] == s - 1:
            out[-1][1] = s
        else:
            out.append([s, s])
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


def _collapse(desktop: int, n_rows: int, dupli: list[int], rows: list[str],
              frames: dict[FrameKey, FrameFile],
              replaces: Optional[set[int]] = None,
              unphotographed: Iterable[int] = ()) -> list[str]:
    """Re-key the OAK and RealSense frames from rows to steps (task U25).

    A ``dupli`` row's frame becomes an extra shot of its step -- or the step's
    frame, when that view has none of its own (the camera missed the row the
    ``dupli`` repeats). A row of ``replaces`` (``oak.<d>.dupli_replaces``) is a
    re-take of a capture that came out wrong -- a part lifted mid-operation, the
    scanner's blue light on the bench -- so it is the step's frame and the
    capture it replaces becomes the extra shot.

    The frames come keyed by Camera_1 capture number; with ``unphotographed``
    steps (U40) a capture is not its sheet row (:func:`capture_rows`), and
    ``dupli`` / ``replaces`` / the extra shots' ``row`` are sheet rows.
    """
    replaces = replaces or set()
    unph = sorted({int(s) for s in unphotographed})
    step_of_row, extra = capture_steps(n_rows, dupli, unph)
    sheet_row = capture_rows(n_rows, dupli, unph)
    capture_of = {r: c for c, r in enumerate(sheet_row) if c}
    issues = []
    if dupli:
        issues.append(
            f"rows: sheet dupli rows {dupli} collapsed: {n_rows} Camera_1 captures are "
            f"{n_rows - len(dupli)} steps; "
            + ", ".join(f"row {r} (cam1 {rows[capture_of[r] - 1]}) -> "
                        + (f"frame of step {step_of_row[capture_of[r]]}, replacing row {r - 1}"
                           if r in replaces
                           else f"extra shot of step {step_of_row[capture_of[r]]}")
                        for r in dupli if r in capture_of)
        )
    for view in ("oak1", "oak2", "rs"):
        mains: dict[int, tuple[int, FrameFile]] = {}
        extras: dict[int, list[tuple[int, FrameFile]]] = {}
        for key in [k for k in frames if k.view == view]:
            ff = frames.pop(key)
            capture = key.step
            inside = 1 <= capture <= n_rows
            step = step_of_row[capture] if inside else capture
            row = sheet_row[capture] if inside else capture
            if capture in extra:
                extras.setdefault(step, []).append((row, ff))
            else:
                mains[step] = (row, ff)
        for step in sorted(set(mains) | set(extras)):
            shots = sorted(extras.get(step, []), key=lambda e: e[0])
            held = mains.get(step)
            better = next(((r, ff) for r, ff in shots if r in replaces), None)
            if better is not None:
                shots.remove(better)
                if held is not None:
                    shots = sorted(shots + [held], key=lambda e: e[0])
                held = better
            elif held is None:
                held = shots.pop(0)
                issues.append(f"{view}: step {step} has no capture of its own; the dupli "
                              f"row {held[0]} capture ({held[1].src_step_dir}) stands in for it")
            main = held[1]
            main.key = FrameKey(desktop, step, view)
            if shots:
                main.aux = dict(main.aux or {})
                main.aux[EXTRA_SHOTS] = [_shot(ff, row) for row, ff in shots]
            frames[main.key] = main
    return issues


_STEP_WORDS = re.compile(r"(logical step |->step |\(step |-> step )(\d+)")


def _restep(lines: list[str], step_of_row: list[int]) -> list[str]:
    """Rewrite the row numbers of issue lines written before the collapse:
    ``logical step 40`` becomes ``logical step 38 (row 40)``."""
    def sub(m: re.Match) -> str:
        row = int(m.group(2))
        if 1 <= row < len(step_of_row) and step_of_row[row] != row:
            return f"{m.group(1)}{step_of_row[row]} (row {row})"
        return m.group(0)
    return [_STEP_WORDS.sub(sub, line) for line in lines]


def _shot(ff: FrameFile, row: Optional[int]) -> dict:
    """One extra shot as stored in ``aux["extra_shots"]``."""
    shot = {"path": ff.path, "src_step_dir": ff.src_step_dir, "ts": ff.ts}
    if row is not None:
        shot["row"] = int(row)
    return shot


def _build_logical_steps(
    desktop: int,
    disassemble: str,
    oak_fix: dict,
    fixes: dict,
    frames: dict[FrameKey, FrameFile],
    gallery: Optional[dict] = None,
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """Order the Camera_1 captures into logical steps and emit their frames.

    Returns ``(step_of_dir, step_of_ts, issues)``.

    ``oak.<d>.cam1_order: folder`` keeps the folder order where the capture
    times disagree (D60: the clocks of both cameras put folder 004 before 003,
    but the photos are rows 3 and 4 in folder order). ``oak.<d>.gallery_captures``
    names Camera_1 folders that photographed the laid-out parts, not a row (D1's
    040): they are not rows, and become shot :data:`SEQUENCE_SHOT_BASE` + folder
    number of the camera's components gallery; ``gallery`` receives
    ``{"dirs": {folder: shot}, "ts": {capture token: shot}}`` so Camera_2 and
    the RealSense put their frame of it there too.
    """
    issues: list[str] = []
    cam1 = scan_oak_camera(os.path.join(disassemble, "Camera_1"), issues)
    keep_rule = oak_fix.get("step1_keep")
    gallery = gallery if gallery is not None else {}
    gallery.setdefault("dirs", {})
    gallery.setdefault("ts", {})
    to_gallery = {str(g).zfill(3) for g in oak_fix.get("gallery_captures") or []}
    for name in sorted(to_gallery - set(cam1)):
        issues.append(f"fix:oak[{desktop}] gallery_captures MISMATCH: Camera_1 has no "
                      f"folder {name}")

    captures: list[tuple[str, str, dict[str, str]]] = []  # (src_dir, ts_token, role_paths)
    for src_dir in sorted(cam1):
        by_ts = cam1[src_dir]
        if not by_ts:
            issues.append(f"oak1: folder {src_dir} holds no recognised capture files")
            continue
        if len(by_ts) > 1:
            rule = keep_rule if src_dir == "001" else None
            chosen = _pick_capture(sorted(by_ts), rule)
            if rule:
                issues.append(
                    f"fix:oak[{desktop}] step1_keep={rule}: cam1/{src_dir} had "
                    f"{len(by_ts)} captures, kept {chosen}"
                )
            else:
                issues.append(
                    f"oak1: folder {src_dir} holds {len(by_ts)} captures "
                    f"{sorted(by_ts)}; kept the latest ({chosen})"
                )
            captures.append((src_dir, chosen, by_ts[chosen]))
        else:
            ts = next(iter(by_ts))
            captures.append((src_dir, ts, by_ts[ts]))

    for src_dir, ts, role_paths in [c for c in captures if c[0] in to_gallery]:
        shot = SEQUENCE_SHOT_BASE + int(src_dir)
        key = FrameKey(desktop, shot, "cmp_oak1")
        frames[key] = _oak_frame(key, ts, role_paths, src_dir)
        frames[key].aux[SEQUENCE_CAPTURE_KEY] = f"Camera_1/{src_dir}"
        gallery["dirs"][src_dir] = shot
        gallery["ts"][ts] = shot
        issues.append(f"fix:oak[{desktop}] gallery_captures: Camera_1 {src_dir} ({ts}) "
                      f"photographs the laid-out parts, not a row: components-gallery "
                      f"shot {shot}")
    captures = [c for c in captures if c[0] not in to_gallery]

    by_ts_order = sorted(captures, key=lambda c: (c[1], c[0]))
    order_source = (fixes.get("general") or {}).get("step_order_source", "oak1_timestamp")
    by_folder = oak_fix.get("cam1_order") == "folder"
    ordered = by_ts_order if order_source == "oak1_timestamp" and not by_folder \
        else sorted(captures)
    folder_order = [c[0] for c in captures]
    if by_folder:
        swapped = [f"{src}->row {i + 1}" for i, (src, _, _) in enumerate(by_ts_order)
                   if folder_order[i] != src]
        if swapped:
            issues.append(f"fix:oak[{desktop}] cam1_order=folder applied: the rows follow "
                          f"the Camera_1 folder numbers; the capture times alone would "
                          f"give {', '.join(swapped)}")
        else:
            issues.append(f"fix:oak[{desktop}] cam1_order=folder declared but the capture "
                          f"times already follow the folder order")
    elif folder_order != [c[0] for c in ordered]:
        moved = [
            f"{src}->step {i + 1}"
            for i, (src, _, _) in enumerate(ordered)
            if folder_order[i] != src
        ]
        if oak_fix.get("cam1_reorder_by_ts"):
            issues.append(
                f"fix:oak[{desktop}] cam1_reorder_by_ts applied: {', '.join(moved)}"
            )
        else:
            issues.append(
                f"oak1: capture timestamps are not in folder order ({', '.join(moved)}); "
                "no cam1_reorder_by_ts fix declared"
            )
    elif oak_fix.get("cam1_reorder_by_ts"):
        issues.append(
            f"fix:oak[{desktop}] cam1_reorder_by_ts declared but folder order already "
            "matches timestamp order"
        )

    step_of_dir: dict[str, int] = {}
    step_of_ts: dict[str, int] = {}
    for i, (src_dir, ts, role_paths) in enumerate(ordered):
        step = i + 1
        step_of_dir[src_dir] = step
        step_of_ts[ts] = step
        key = FrameKey(desktop, step, "oak1")
        frames[key] = _oak_frame(key, ts, role_paths, src_dir)
    return step_of_dir, step_of_ts, issues


def _oak_frame(
    key: FrameKey, ts: str, role_paths: dict[str, str], src_dir: str
) -> FrameFile:
    """Wrap one OAK capture's files into a :class:`FrameFile`."""
    aux = {role: path for role, path in sorted(role_paths.items()) if role != "main"}
    return FrameFile(
        key=key,
        path=role_paths.get("main", ""),
        aux=aux,
        ts=iso_from_oak_token(ts),
        src_step_dir=src_dir,
    )


def _add_oak_cam2(
    desktop: int,
    disassemble: str,
    oak_desktop: str,
    oak_fix: dict,
    step_of_dir: dict[str, int],
    step_of_ts: dict[str, int],
    frames: dict[FrameKey, FrameFile],
    gallery_ts: Optional[dict[str, int]] = None,
    leftover: Optional[list[str]] = None,
) -> list[str]:
    """Pair Camera_2 captures onto the logical steps, applying the D24/D10 fixes.

    A capture paired with a Camera_1 ``gallery_captures`` folder goes to the same
    shot of Camera_2's components gallery. ``leftover`` receives every capture
    that found no step (``folder (token)``), for the assertion of U40.
    """
    gallery_ts = gallery_ts or {}
    leftover = leftover if leftover is not None else []
    issues: list[str] = []
    cam2 = scan_oak_camera(os.path.join(disassemble, "Camera_2"), issues)
    captures: list[tuple[str, str, dict[str, str]]] = []
    for src_dir in sorted(cam2):
        by_ts = cam2[src_dir]
        tokens = sorted(by_ts)
        if src_dir == "001" and len(tokens) > 1 and oak_fix.get("step1_keep"):
            rule = oak_fix["step1_keep"]
            chosen = _pick_capture(tokens, rule)
            issues.append(
                f"fix:oak[{desktop}] step1_keep={rule}: cam2/{src_dir} had "
                f"{len(tokens)} captures, kept {chosen}"
            )
            tokens = [chosen]
        captures.extend((src_dir, ts, by_ts[ts]) for ts in tokens)

    extra = oak_fix.get("cam2_extra_from")
    if extra:
        extra_dir = os.path.join(oak_desktop, *extra.split("/"))
        found = scan_oak_step_dir(extra_dir, issues) if os.path.isdir(extra_dir) else {}
        if found:
            captures.extend((extra, ts, paths) for ts, paths in sorted(found.items()))
        else:
            issues.append(f"fix:oak[{desktop}] cam2_extra_from {extra}: nothing found")

    assigned: dict[str, list[int]] = {}
    for src_dir, ts, role_paths in sorted(captures, key=lambda c: (c[1], c[0])):
        if ts in gallery_ts:
            key = FrameKey(desktop, gallery_ts[ts], "cmp_oak2")
            frames[key] = _oak_frame(key, ts, role_paths, src_dir)
            frames[key].aux[SEQUENCE_CAPTURE_KEY] = f"Camera_2/{src_dir}"
            issues.append(f"oak2: {src_dir} ({ts}) pairs a Camera_1 gallery capture: "
                          f"components-gallery shot {gallery_ts[ts]}")
            continue
        step = step_of_ts.get(ts)
        if step is None:
            step = cam2_fix_step(oak_fix, src_dir, step_of_dir)
        if step is None:
            issues.append(
                f"oak2: capture {ts} in folder {src_dir} has no cam1 partner; skipped"
            )
            leftover.append(f"{src_dir} ({ts})")
            continue
        key = FrameKey(desktop, step, "oak2")
        if key in frames:
            issues.append(
                f"oak2: step {step} already filled by folder {frames[key].src_step_dir}; "
                f"dropped duplicate from {src_dir} ({ts})"
            )
            leftover.append(f"{src_dir} ({ts})")
            continue
        frames[key] = _oak_frame(key, ts, role_paths, src_dir)
        assigned.setdefault(src_dir, []).append(step)

    for src_dir, steps in sorted(assigned.items()):
        if len(steps) > 1:
            issues.append(
                f"oak2: folder {src_dir} split by timestamp into logical steps "
                f"{sorted(steps)}"
            )
    issues.extend(verify_cam2_fixes(desktop, oak_fix, step_of_dir, assigned))
    return issues


def scan_step(number: int, duplicate_folders: Iterable[int],
              skipped_steps: Iterable[int] = ()) -> int:
    """The step scanner folder ``RGB<number>1`` belongs to (task U25, spec 2.2).

    ``number - #(duplicate folders <= number)``: a duplicate folder photographs
    the state of the step before it again, so it is not counted, and the folder
    itself lands on that step (as an extra shot). Without duplicates folder
    ``n`` is step ``n`` -- which already covers a scanner that skipped a
    ``dupli`` row, because the collapsed steps do not count those rows either.

    ``skipped_steps`` (``scan.<d>.skipped_steps``, U25 round 2) are steps the
    scanner never photographed (D41, D61): the folders from there on are one
    step later each -- the ``b``-th folder lands on the ``b``-th step that was
    not skipped. The desktop's ``unphotographed`` steps (U40) are passed here
    too, joined with its own.
    """
    base = int(number) - sum(1 for d in duplicate_folders if int(d) <= int(number))
    for k in sorted({int(s) for s in skipped_steps}):
        if k <= base:
            base += 1
    return base


def _add_scanner(
    desktop: int,
    roots: dict,
    step_of_dir: dict[str, int],
    frames: dict[FrameKey, FrameFile],
    n_steps: Optional[int] = None,
    scan_fix: Optional[dict] = None,
    collapsed: bool = False,
    leftover: Optional[list[str]] = None,
    unphotographed: Iterable[int] = (),
) -> list[str]:
    """Add the scanner (top-down) frames.

    A desktop whose rows were collapsed (it has ``dupli`` rows) or that declares
    ``scan.<d>.duplicate_folders``, ``scan.<d>.skipped_steps`` or steps no
    camera photographed (``unphotographed``, U40 -- skipped steps of the
    scanner too, the union with its own) maps folder ``RGB<n>1`` with
    :func:`scan_step`; a duplicate folder is an extra shot of its step, or the
    step's frame when nothing else photographed it (D1's ``RGB331``). Every
    other desktop keeps the old rule: folder ``n`` is the step of Camera_1
    folder ``n``.

    No timestamp cross-check is possible here: the scanner PNGs were copied to F:
    in one batch (all mtimes are the copy date, not the capture date), so
    ``ts`` is informational only and must not be compared with the OAK capture
    times the way :func:`_add_realsense` does.
    """
    scan_fix = scan_fix or {}
    unphotographed = {int(s) for s in unphotographed}
    own = (collapsed or "duplicate_folders" in scan_fix or "skipped_steps" in scan_fix
           or bool(unphotographed))
    duplicates = {int(n) for n in (scan_fix.get("duplicate_folders") or [])}
    skipped = {int(n) for n in (scan_fix.get("skipped_steps") or [])} | unphotographed
    issues: list[str] = []
    base = scanner_desktop_dir(roots["scanner_root"], desktop, issues)
    if base is None:
        issues.append(f"scan: no scanner folder found for desktop {desktop}")
        return issues
    mains: dict[int, FrameFile] = {}
    extras: dict[int, list[FrameFile]] = {}
    entries = [e for e in dirs(base, issues)
               if e.name.startswith("RGB") and e.name.endswith("1") and e.name[3:-1].isdigit()]
    for entry in sorted(entries, key=lambda e: int(e.name[3:-1])):
        name = entry.name
        nominal = int(name[3:-1])
        if own:
            step = scan_step(nominal, duplicates, skipped)
            if not 1 <= step <= (n_steps or 0):
                issues.append(f"scan: {name} would be step {step} of {n_steps}; skipped")
                if leftover is not None:
                    leftover.append(name)
                continue
        else:
            step = step_of_dir.get(f"{nominal:03d}")
            if step is None:
                issues.append(f"scan: {name} (nominal step {nominal}) has no cam1 folder; "
                              f"skipped")
                continue
        burst = []
        for f in files(entry.path, issues):
            m = BURST_RE.match(f.name)
            if m:
                burst.append((int(m.group(1)), norm(f.path), f.stat().st_mtime))
        if not burst:
            issues.append(f"scan: {name} holds no P_k.png; skipped")
            continue
        burst.sort()
        main = next((b for b in burst if b[0] == 0), burst[0])
        if main[0] != 0:
            issues.append(f"scan: {name} has no P_0.png; using P_{main[0]}.png")
        if len(burst) < 9:
            issues.append(f"scan: {name} burst has only {len(burst)} images")
        key = FrameKey(desktop, step, "scan")
        ff = FrameFile(key=key, path=main[1], aux={"burst": [b[1] for b in burst]},
                       ts=iso_from_mtime(main[2]), src_step_dir=name)
        if own and nominal in duplicates:
            extras.setdefault(step, []).append(ff)
        else:
            mains[step] = ff
    replacing = {int(n) for n in (scan_fix.get("replacing_folders") or [])}
    for n in sorted(replacing - duplicates):
        issues.append(f"fix:scan[{desktop}] replacing_folders MISMATCH: folder {n} is not "
                      f"a duplicate folder; ignored")
    for step in sorted(set(mains) | set(extras)):
        shots = extras.get(step, [])
        main_ff = mains.get(step)
        better = next((ff for ff in shots if int(ff.src_step_dir[3:-1]) in replacing), None)
        if better is not None:
            # a re-shot of a scan that came out wrong is the step's frame (U25)
            shots = [ff for ff in shots if ff is not better] + ([main_ff] if main_ff else [])
            main_ff = better
        elif main_ff is None:
            main_ff = shots.pop(0)
            issues.append(f"scan: step {step} has no other scanner capture; duplicate "
                          f"folder {main_ff.src_step_dir} stands in for it")
        if shots:
            main_ff.aux = dict(main_ff.aux)
            main_ff.aux[EXTRA_SHOTS] = [_shot(ff, None) for ff in shots]
        frames[main_ff.key] = main_ff
    return issues


def _add_realsense(
    desktop: int,
    roots: dict,
    aliases: list[str],
    step_of_dir: dict[str, int],
    frames: dict[FrameKey, FrameFile],
    rs_fix: Optional[dict] = None,
    gallery_dirs: Optional[dict[str, int]] = None,
    leftover: Optional[list[str]] = None,
) -> list[str]:
    """Add the RealSense frames, aligning by folder number *and* file time.

    The frame of a Camera_1 ``gallery_captures`` folder (by folder number, or
    the ``rs_extra_from`` of its number) goes to the same shot of the
    RealSense's components gallery. ``leftover`` receives every folder that
    found no step, for the assertion of U40.

    Spec 2.2 aligns the RealSense by folder number plus file time.  Unlike the
    scanner, these files were never re-copied, so ``original_color.png``'s mtime
    is a real capture time and can be checked against the OAK cam1 capture time
    of the step the folder number resolved to; a gap beyond
    :data:`RS_TIME_TOLERANCE_S` is reported rather than silently accepted.

    ``rs_extra_from`` (``configs/index_fixes.yaml``, like ``cam2_extra_from``):
    the last frame of the teardown filed under ``Components`` (D1 step 40, D24
    step 43) fills its nominal step ``as_step``, time-checked like any other.
    """
    issues: list[str] = []
    gallery_dirs = gallery_dirs or {}
    rs_desktop = os.path.join(roots["rs_root"], f"Desktop {desktop}")
    base = first_existing(rs_desktop, aliases)
    if base is None:
        issues.append(f"rs: no {'/'.join(aliases)} folder under {rs_desktop}")
        return issues
    for entry in sorted(dirs(base, issues), key=lambda e: e.name):
        if not STEP_DIR_RE.match(entry.name):
            continue
        nominal = int(entry.name)
        step = step_of_dir.get(f"{nominal:03d}")
        if step is None and f"{nominal:03d}" in gallery_dirs:
            issues.extend(_rs_gallery(desktop, entry.path, entry.name,
                                      gallery_dirs[f"{nominal:03d}"], frames))
            continue
        if step is None:
            issues.append(
                f"rs: folder {entry.name} (nominal step {nominal}) has no cam1 folder; skipped"
            )
            if leftover is not None:
                leftover.append(entry.name)
            continue
        found = {f.name: f for f in files(entry.path, issues)}
        main = found.get("original_color.png")
        if main is None:
            issues.append(f"rs: folder {entry.name} has no original_color.png; skipped")
            continue
        aux = {}
        if "depth_raw.npy" in found:
            aux["depth_npy"] = norm(found["depth_raw.npy"].path)
        key = FrameKey(desktop, step, "rs")
        ts = iso_from_mtime(main.stat().st_mtime)
        frames[key] = FrameFile(
            key=key, path=norm(main.path), aux=aux, ts=ts, src_step_dir=entry.name
        )
        issues.extend(_check_rs_time(desktop, step, entry.name, ts, frames))
    issues.extend(_rs_extra(desktop, rs_desktop, rs_fix or {}, step_of_dir, frames,
                            gallery_dirs))
    return issues


def _rs_gallery(desktop: int, path: str, src: str, shot: int,
                frames: dict[FrameKey, FrameFile]) -> list[str]:
    """The RealSense frame of a Camera_1 ``gallery_captures`` folder, as a gallery shot."""
    main = os.path.join(path, "original_color.png")
    if not os.path.isfile(main):
        return [f"rs: {src} (a gallery capture) has no original_color.png; skipped"]
    aux = {SEQUENCE_CAPTURE_KEY: f"RealSense/{src}"}
    depth = os.path.join(path, "depth_raw.npy")
    if os.path.isfile(depth):
        aux["depth_npy"] = norm(depth)
    key = FrameKey(desktop, shot, "cmp_rs")
    frames[key] = FrameFile(key=key, path=norm(main), aux=aux,
                            ts=iso_from_mtime(os.stat(main).st_mtime), src_step_dir=src)
    return [f"rs: {src} is the frame of a Camera_1 gallery capture: components-gallery "
            f"shot {shot}"]


def _rs_extra(desktop: int, rs_desktop: str, rs_fix: dict, step_of_dir: dict[str, int],
              frames: dict[FrameKey, FrameFile],
              gallery_dirs: Optional[dict[str, int]] = None) -> list[str]:
    """Apply a declared ``rs_extra_from`` fix; every outcome is an issue line."""
    folder = rs_fix.get("rs_extra_from")
    if not folder or "as_step" not in rs_fix:
        return []
    head = f"fix:rs[{desktop}] rs_extra_from {folder}"
    nominal = int(rs_fix["as_step"])
    step = step_of_dir.get(f"{nominal:03d}")
    path = os.path.join(rs_desktop, *str(folder).split("/"))
    main = os.path.join(path, "original_color.png")
    if step is None and f"{nominal:03d}" in (gallery_dirs or {}):
        shot = gallery_dirs[f"{nominal:03d}"]
        return [f"{head} (as Camera_1 {nominal:03d}, a gallery capture)"] + _rs_gallery(
            desktop, path, str(folder), shot, frames)
    if step is None:
        return [f"{head} MISMATCH: nominal step {nominal} has no cam1 folder"]
    key = FrameKey(desktop, step, "rs")
    if key in frames:
        return [f"{head} MISMATCH: logical step {step} already has folder "
                f"{frames[key].src_step_dir}; not applied"]
    if not os.path.isfile(main):
        return [f"{head}: nothing found"]
    aux = {}
    depth = os.path.join(path, "depth_raw.npy")
    if os.path.isfile(depth):
        aux["depth_npy"] = norm(depth)
    ts = iso_from_mtime(os.stat(main).st_mtime)
    frames[key] = FrameFile(key=key, path=norm(main), aux=aux, ts=ts,
                            src_step_dir=str(folder))
    return [f"{head} -> logical step {step}"] + _check_rs_time(
        desktop, step, str(folder), ts, frames)


def _add_components(
    desktop: int,
    roots: dict,
    oak_desktop: str,
    oak_fix: dict,
    step_of_ts: dict[str, int],
    frames: dict[FrameKey, FrameFile],
    rs_fix: Optional[dict] = None,
) -> list[str]:
    """Index the components gallery (task U10, spec 3.2b): one view per camera.

    Every capture in ``Components`` is a **shot** of the parts laid out after
    the teardown, numbered ``1..N`` per camera in **capture-time** order (the
    OAK filename token; the RealSense file time, folder name breaking a tie --
    folder names are not in time order on five desktops). The cameras are not
    matched against each other (the layout was re-arranged between shots, and
    nothing needs a shot of one camera to be "the same" as another's). The
    source folder is kept in ``src_step_dir``.

    What is **not** a shot: a capture of the teardown that was filed under
    ``Components`` by mistake. The declared ones fill their sequence step
    (``cam2_extra_from`` for D24's Camera_2, ``rs_extra_from`` for the
    RealSense of D1 and D24) and are skipped here; an undeclared OAK capture
    whose timestamp token is a sequence capture's, or a RealSense frame whose
    file time lies within :data:`RS_TIME_TOLERANCE_S` of a sequence OAK
    capture, is reported and left out.
    """
    rs_fix = rs_fix or {}
    issues: list[str] = []
    base = os.path.join(oak_desktop, COMPONENTS_DIR)
    extra = oak_fix.get("cam2_extra_from") or ""
    for view, names in COMPONENT_CAMERA_DIRS.items():
        cam_dir = first_existing(base, names) if os.path.isdir(base) else None
        if cam_dir is None:
            continue
        captures: list[tuple[str, str, dict[str, str]]] = []
        for src_dir, by_ts in sorted(scan_oak_camera(cam_dir, issues).items()):
            rel = f"{COMPONENTS_DIR}/{os.path.basename(cam_dir)}/{src_dir}"
            for ts, role_paths in sorted(by_ts.items()):
                if ts in step_of_ts or rel == extra:
                    issues.append(f"{view}: {rel} ({ts}) is a capture of the teardown "
                                  f"(step {step_of_ts.get(ts, '?')}), not a gallery shot")
                    continue
                if "main" not in role_paths:
                    issues.append(f"{view}: {rel} ({ts}) has no rgb_12mp.jpg; skipped")
                    continue
                captures.append((rel, ts, role_paths))
        for shot, (rel, ts, role_paths) in enumerate(
                sorted(captures, key=lambda c: (c[1], c[0])), start=1):
            key = FrameKey(desktop, shot, view)
            frames[key] = _oak_frame(key, ts, role_paths, rel)

    teardown = sorted(ff.ts for k, ff in frames.items() if k.view == "oak1" and ff.ts)
    rs_base = os.path.join(roots["rs_root"], f"Desktop {desktop}", COMPONENTS_DIR)
    if not os.path.isdir(rs_base):
        return issues
    used = str(rs_fix.get("rs_extra_from") or "")
    found_shots: list[tuple[str, str, FrameFile]] = []   # (ts, folder, frame)
    for entry in dirs(rs_base, issues):
        if not STEP_DIR_RE.match(entry.name):
            continue
        rel = f"{COMPONENTS_DIR}/{entry.name}"
        if rel == used:
            continue          # the sequence's own frame (rs_extra_from, _add_realsense)
        found = {f.name: f for f in files(entry.path, issues)}
        main = found.get("original_color.png")
        if main is None:
            issues.append(f"cmp_rs: {rel} has no original_color.png; skipped")
            continue
        ts = iso_from_mtime(main.stat().st_mtime)
        near = _nearest_seconds(ts, teardown)
        if near is not None and near <= RS_TIME_TOLERANCE_S:
            issues.append(f"cmp_rs: {rel} was taken {near:.0f}s from a teardown capture "
                          f"(tolerance {RS_TIME_TOLERANCE_S:.0f}s): a frame of the "
                          f"teardown, not a gallery shot; left out")
            continue
        aux = {}
        if "depth_raw.npy" in found:
            aux["depth_npy"] = norm(found["depth_raw.npy"].path)
        found_shots.append((ts, entry.name, FrameFile(
            key=FrameKey(desktop, 0, "cmp_rs"), path=norm(main.path), aux=aux, ts=ts,
            src_step_dir=rel)))
    # capture time first, folder name second: on D6, D7, D10, D12 and D18 the
    # folder 005 holds the FIRST layout photo, so the names would rotate the shots
    for shot, (_ts, _name, ff) in enumerate(sorted(found_shots, key=lambda s: s[:2]),
                                             start=1):
        ff.key = FrameKey(desktop, shot, "cmp_rs")
        frames[ff.key] = ff
    return issues


def scan_shot_k(name: str) -> Optional[int]:
    """The ``k`` of a scanner all-components shot folder, ``None`` for no shot (U42).

    ``allcomponentsrgb21_color`` -> 2 and ``allcompoentsrgb11`` -> 1, the step
    folders' ``RGB<k>1`` convention; a bare ``1`` (``allRGB1_color``, D52's
    ``allcomponentsrgb1``) or no digit at all is 1.
    """
    m = SCAN_SHOT_RE.match(name.lower())
    if m is None:
        return None
    digits = m.group(1)
    if len(digits) > 1 and digits.endswith("1"):
        return int(digits[:-1])
    return int(digits or 1)


def _add_scanner_components(
    desktop: int,
    roots: dict,
    frames: dict[FrameKey, FrameFile],
) -> list[str]:
    """Index the scanner's scans of the laid-out parts: view ``cmp_scan`` (task U42).

    After the teardown the parts were also laid out on the scanner's bed and
    scanned, one to five times (:func:`tda.core.sources.scanner_components_dir`).
    Each shot folder (``all...rgb<k>1[_color]``, spelled a dozen ways) is a
    burst ``P_0..P_9`` like a step scan's, and is indexed like one: ``P_0`` the
    main image, ``aux["burst"]`` every image -- so ``build-cache`` picks among
    them exactly as it does for the scanner's steps -- ``ts`` the main image's
    file time (the copy date, informational) and ``src_step_dir`` the folder
    from the group down. The shots are numbered ``1..N`` by ``(k, name)``:
    the position, so they stay contiguous whatever a folder is called.

    Not a shot: a folder whose name is not an RGB scan's -- D64's
    ``allcomponentsseq11`` / ``21`` are fringe-projection sequences (176 images
    under projected stripes) -- or that holds more than
    :data:`SCAN_SHOT_MAX_BURST` images. Each is reported and left out.
    """
    issues: list[str] = []
    root = roots["scanner_root"]
    base = scanner_components_dir(root, desktop, issues)
    if base is None:
        # silent, like a desktop without a Components folder (_add_components):
        # a gallery has no missing frames, and the report counts its shots
        return issues
    found: list[tuple[int, str, FrameFile]] = []
    for entry in sorted(dirs(base, issues), key=lambda e: e.name):
        name = entry.name
        burst = []
        for f in files(entry.path, issues):
            m = BURST_RE.match(f.name)
            if m:
                burst.append((int(m.group(1)), norm(f.path), f.stat().st_mtime))
        k = scan_shot_k(name)
        if k is None or len(burst) > SCAN_SHOT_MAX_BURST:
            why = ("its name is not an all-components RGB scan's" if k is None
                   else f"more than {SCAN_SHOT_MAX_BURST} images")
            issues.append(f"cmp_scan: {name} is not an RGB scan ({why}, {len(burst)} "
                          f"images); left out")
            continue
        if not burst:
            issues.append(f"cmp_scan: {name} holds no P_k.png; left out")
            continue
        burst.sort()
        main = next((b for b in burst if b[0] == 0), burst[0])
        if main[0] != 0:
            issues.append(f"cmp_scan: {name} has no P_0.png; using P_{main[0]}.png")
        if len(burst) < 9:
            issues.append(f"cmp_scan: {name} burst has only {len(burst)} images")
        rel = norm(os.path.relpath(entry.path, root))
        found.append((k, name, FrameFile(
            key=FrameKey(desktop, 0, "cmp_scan"), path=main[1],
            aux={"burst": [b[1] for b in burst]}, ts=iso_from_mtime(main[2]),
            src_step_dir=rel)))
    for shot, (k, name, ff) in enumerate(sorted(found, key=lambda s: s[:2]), start=1):
        if k != shot:
            issues.append(f"cmp_scan: {name} (k={k}) is shot {shot}")
        ff.key = FrameKey(desktop, shot, "cmp_scan")
        frames[ff.key] = ff
    return issues


def _nearest_seconds(ts: str, others: list[str]) -> Optional[float]:
    """Seconds from ``ts`` to the closest of ``others`` (ISO strings), or ``None``."""
    if not others:
        return None
    at = datetime.fromisoformat(ts)
    return min(abs((at - datetime.fromisoformat(o)).total_seconds()) for o in others)


def _check_rs_time(
    desktop: int,
    step: int,
    src_dir: str,
    rs_ts: str,
    frames: dict[FrameKey, FrameFile],
) -> list[str]:
    """Report a RealSense file time that disagrees with the OAK cam1 capture time."""
    oak1 = frames.get(FrameKey(desktop, step, "oak1"))
    if oak1 is None or not oak1.ts:
        return []
    delta = abs(
        (datetime.fromisoformat(rs_ts) - datetime.fromisoformat(oak1.ts)).total_seconds()
    )
    if delta <= RS_TIME_TOLERANCE_S:
        return []
    return [
        f"rs time mismatch: folder {src_dir} -> logical step {step}: "
        f"original_color.png mtime {rs_ts} is {delta:.0f}s from the oak1 capture "
        f"time {oak1.ts} (tolerance {RS_TIME_TOLERANCE_S:.0f}s)"
    ]


# ---------------------------------------------------------------------------
# whole-dataset build & persistence
# ---------------------------------------------------------------------------
def build_index(
    desktops: Iterable[int],
    roots: dict,
    fixes_path: str = DEFAULT_FIXES_PATH,
    progress: bool = False,
    sheets: Optional[dict] = None,
) -> dict[int, DesktopIndex]:
    """Scan every desktop in ``desktops`` and return ``{desktop: DesktopIndex}``.

    ``sheets`` (:func:`tda.core.logs.sheet_rows_by_desktop`) supplies each
    desktop's ``dupli`` rows; a desktop without one is indexed uncollapsed.
    """
    fixes = load_fixes(fixes_path)
    out: dict[int, DesktopIndex] = {}
    for desktop in desktops:
        sheet = (sheets or {}).get(int(desktop))
        idx = scan_desktop(int(desktop), roots, fixes,
                           dupli_rows=None if sheet is None else sheet.dupli_rows,
                           sheet_rows=None if sheet is None else sheet.n_rows)
        out[int(desktop)] = idx
        if progress:
            n_missing = len(idx.missing)
            rows = (f" ({idx.n_rows} rows, dupli {idx.dupli_rows})" if idx.dupli_rows
                    else "")
            if idx.unphotographed:
                rows += f" (unphotographed {idx.unphotographed})"
            print(
                f"[index] Desktop {desktop:>2}: {idx.n_steps:>3} steps{rows}, "
                f"{len(idx.frames):>4} frames, {n_missing:>3} missing, "
                f"{len(idx.issues)} issues",
                flush=True,
            )
    return out


def _respelled(value: Any, spell: Callable[[str], str]) -> Any:
    """``spell`` applied to a path, every path of a list (a burst) and every value
    of a dict (an extra shot); anything that is not a path is left as it is."""
    if isinstance(value, str):
        return spell(value) if value else value
    if isinstance(value, list):
        return [_respelled(v, spell) for v in value]
    if isinstance(value, dict):
        return {k: _respelled(v, spell) for k, v in value.items()}
    return value


def respell_paths(
    idx: dict[int, DesktopIndex],
    spell_path: Callable[[str], str],
    spell_text: Optional[Callable[[str], str]] = None,
) -> None:
    """Rewrite, in place, every path the scan found with ``spell_path``.

    ``build-index`` scans the raw drive where it is *today* and must record the
    paths in the spelling the dataset was recorded with
    (:meth:`tda.core.rawroot.RawRoot.to_recorded`): the frame's own path, every
    ``aux`` path (``burst`` lists included), and -- through ``spell_text`` --
    the folders the issues name.
    """
    for di in idx.values():
        for ff in di.frames.values():
            ff.path = _respelled(ff.path, spell_path)
            ff.aux = {k: _respelled(v, spell_path) for k, v in (ff.aux or {}).items()}
        if spell_text is not None:
            di.issues = [spell_text(text) for text in di.issues]


def _frame_to_json(ff: FrameFile) -> dict:
    return {
        "desktop": ff.key.desktop,
        "step": ff.key.step,
        "view": ff.key.view,
        "path": ff.path,
        "aux": ff.aux,
        "ts": ff.ts,
        "src_step_dir": ff.src_step_dir,
    }


def keep_previous(index_path: str, report_path: Optional[str] = None,
                  when: Optional[datetime] = None) -> list[str]:
    """Copy an index file (and its report) aside before ``build-index`` overwrites it.

    ``<stem>_v<version>_<YYYYMMDD><ext>`` next to it -- ``index_v2_20261024.json``,
    ``index_report_v2_20261024.md`` -- so going back to the previous index needs
    no manual copy (U25 round 2, N7). A copy that is already there with the same
    bytes is not made again; a different one of the same name gets the time
    appended. Returns the copies made.
    """
    import filecmp
    import shutil

    when = when or datetime.now()
    if not os.path.isfile(index_path):
        return []
    version = "unknown"
    try:
        with open(index_path, encoding="utf-8") as fh:
            m = re.search(r'"version"\s*:\s*(\d+)', fh.read(4096))
        if m:
            version = m.group(1)
    except OSError:
        pass
    tag = f"v{version}_{when:%Y%m%d}"
    kept: list[str] = []
    for path in (index_path, report_path):
        if not path or not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(path)
        dest = f"{stem}_{tag}{ext}"
        if os.path.exists(dest):
            if filecmp.cmp(path, dest, shallow=False):
                continue
            dest = f"{stem}_{tag}_{when:%H%M%S}{ext}"
        shutil.copy2(path, dest)
        kept.append(dest)
    return kept


def save_index(idx: dict[int, DesktopIndex], path: str) -> None:
    """Write the index to JSON (creating parent directories)."""
    payload = {
        "version": INDEX_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "desktops": {
            str(d): {
                "desktop": di.desktop,
                "n_steps": di.n_steps,
                "n_rows": di.n_rows,
                "rows": list(di.rows),
                "dupli_rows": list(di.dupli_rows),
                "unphotographed": list(getattr(di, "unphotographed", None) or []),
                "frames": [_frame_to_json(ff) for ff in di.frames.values()],
                "missing": [[k.desktop, k.step, k.view] for k in di.missing],
                "issues": di.issues,
            }
            for d, di in sorted(idx.items())
        },
    }
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)


def load_index(path: str) -> dict[int, DesktopIndex]:
    """Read back an index written by :func:`save_index`."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    try:
        version = int(payload.get("version", 1))
    except (TypeError, ValueError):
        version = 1
    gallery = version >= GALLERY_INDEX_VERSION
    collapsed = version >= COLLAPSED_INDEX_VERSION
    scan_gallery = version >= SCAN_GALLERY_INDEX_VERSION
    out: dict[int, DesktopIndex] = {}
    for d, raw in payload.get("desktops", {}).items():
        frames: dict[FrameKey, FrameFile] = {}
        for rec in raw["frames"]:
            key = FrameKey(rec["desktop"], rec["step"], rec["view"])
            frames[key] = FrameFile(
                key=key,
                path=rec["path"],
                aux=rec["aux"],
                ts=rec["ts"],
                src_step_dir=rec["src_step_dir"],
            )
        out[int(d)] = DesktopIndex(
            desktop=raw["desktop"],
            n_steps=raw["n_steps"],
            frames=frames,
            missing=[FrameKey(*m) for m in raw["missing"]],
            issues=list(raw["issues"]),
            gallery=gallery,
            collapsed=collapsed,
            n_rows=int(raw.get("n_rows") or 0),
            rows=list(raw.get("rows") or []),
            dupli_rows=[int(r) for r in raw.get("dupli_rows") or []],
            unphotographed=[int(s) for s in raw.get("unphotographed") or []],
            scan_gallery=scan_gallery,
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    """CLI: build the index over F: and write index.json + index_report.md."""
    import argparse

    ap = argparse.ArgumentParser(description="Build the unified TDA frame index.")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=66)
    ap.add_argument("--out", default=None, help="index JSON (default: <cache_dir>/index.json)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--paths", default=DEFAULT_PATHS_PATH)
    ap.add_argument("--fixes", default=DEFAULT_FIXES_PATH)
    args = ap.parse_args(argv)

    from tda.core import rawroot

    cfg = _load_yaml(args.paths)
    raw = rawroot.configure(cfg)
    # scanned where the raw drive is today; what it finds is recorded in the
    # spelling the dataset was recorded with (respell_paths below)
    roots = {k: rawroot.resolve_raw(cfg[k]) for k in ("oak_root", "scanner_root", "rs_root")}
    if any(v is None for v in roots.values()):
        print(f"[index] {raw.message}; nothing was scanned")
        return 1
    if raw.status == rawroot.MOVED:
        print(f"[index] {raw.log_line}")
    cache_dir = cfg.get("cache_dir", "cache")
    out = args.out or os.path.join(cache_dir, "index.json")
    report = args.report or os.path.join(cache_dir, "index_report.md")

    from tda.core.logs import sheet_rows_by_desktop

    started = datetime.now()
    sheets = sheet_rows_by_desktop(os.path.join(cfg.get("raw_logs_dir", "raw_logs"), "drive"))
    idx = build_index(range(args.first, args.last + 1), roots, args.fixes, progress=True,
                      sheets=sheets)
    respell_paths(idx, raw.to_recorded, raw.respell)   # stored: the recorded spelling
    save_index(idx, out)
    write_report(idx, report)
    print(
        f"[index] {len(idx)} desktops, {sum(d.n_steps for d in idx.values())} steps, "
        f"{sum(len(d.frames) for d in idx.values())} frames in "
        f"{(datetime.now() - started).total_seconds():.1f}s"
    )
    print(f"[index] wrote {out}\n[index] wrote {report}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
