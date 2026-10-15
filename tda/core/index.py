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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from tda.core.index_fixes import (
    cam2_fix_step,
    verify_cam2_fixes,
    verify_declared_missing,
)
from tda.core.index_report import write_report
from tda.core.model import VIEWS, FrameKey
from tda.core.sources import (
    BURST_RE,
    COMPONENT_CAMERA_DIRS,
    COMPONENTS_DIR,
    DEFAULT_FOLDER_ALIASES,
    STEP_DIR_RE,
    dirs,
    files,
    first_existing,
    iso_from_mtime,
    iso_from_oak_token,
    norm,
    scan_oak_camera,
    scan_oak_step_dir,
    scanner_desktop_dir,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FIXES_PATH = "configs/index_fixes.yaml"
DEFAULT_PATHS_PATH = "configs/paths.yaml"
#: 2: the components gallery is indexed (task U10). A version-1 file is still
#: read (``DesktopIndex.gallery`` unset), but ``load-index`` refuses it
#: (:func:`tda.pipeline.stale_index`): it predates the gallery and the
#: RealSense ``rs_extra_from`` fixes.
INDEX_VERSION = 2
#: The first version whose scan covers the components gallery.
GALLERY_INDEX_VERSION = 2
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
    n_steps: int  # logical steps = OAK cam1 captures after fixes
    frames: dict[FrameKey, FrameFile] = field(default_factory=dict)
    missing: list[FrameKey] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    #: Did the scan look at the components gallery (task U10)? ``False`` for an
    #: index file written before it existed (version 1): such an index says
    #: nothing about the gallery, and ``load-index`` refuses it (build-index again).
    gallery: bool = True


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


def _pick_capture(tokens: list[str], rule: Optional[str]) -> str:
    """Choose one timestamp out of a re-shot step folder (``latest``/``earliest``)."""
    return min(tokens) if rule == "earliest" else max(tokens)


# ---------------------------------------------------------------------------
# the per-desktop scan
# ---------------------------------------------------------------------------
def scan_desktop(desktop: int, roots: dict, fixes: dict) -> DesktopIndex:
    """Build the frame index for one desktop from the four source trees."""
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

    step_of_dir, step_of_ts, issues_c1 = _build_logical_steps(
        desktop, disassemble, oak_fix, fixes, frames
    )
    issues.extend(issues_c1)
    n_steps = len(step_of_dir)
    if n_steps == 0:
        issues.append(f"oak: no Camera_1 step folders under {disassemble}")
        return DesktopIndex(desktop, 0, frames, [], issues)

    issues.extend(
        _add_oak_cam2(desktop, disassemble, oak_desktop, oak_fix, step_of_dir, step_of_ts, frames)
    )
    rs_fix = (fixes.get("rs") or {}).get(desktop, {}) or {}
    issues.extend(_add_scanner(desktop, roots, step_of_dir, frames))
    issues.extend(_add_realsense(desktop, roots, aliases, step_of_dir, frames, rs_fix))
    issues.extend(_add_components(desktop, roots, oak_desktop, oak_fix, step_of_ts, frames,
                                  rs_fix))

    # the gallery's shots are not steps: a camera with fewer of them than
    # another misses nothing, so only the sequence views can miss a frame
    missing = [
        FrameKey(desktop, step, view)
        for step in range(1, n_steps + 1)
        for view in VIEWS
        if FrameKey(desktop, step, view) not in frames
    ]
    issues.extend(verify_declared_missing(desktop, fixes, step_of_dir, missing))
    return DesktopIndex(desktop, n_steps, frames, missing, issues)


def _build_logical_steps(
    desktop: int,
    disassemble: str,
    oak_fix: dict,
    fixes: dict,
    frames: dict[FrameKey, FrameFile],
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """Order the Camera_1 captures into logical steps and emit their frames.

    Returns ``(step_of_dir, step_of_ts, issues)``.
    """
    issues: list[str] = []
    cam1 = scan_oak_camera(os.path.join(disassemble, "Camera_1"), issues)
    keep_rule = oak_fix.get("step1_keep")

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

    by_ts_order = sorted(captures, key=lambda c: (c[1], c[0]))
    order_source = (fixes.get("general") or {}).get("step_order_source", "oak1_timestamp")
    ordered = by_ts_order if order_source == "oak1_timestamp" else sorted(captures)
    folder_order = [c[0] for c in captures]
    if folder_order != [c[0] for c in ordered]:
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
) -> list[str]:
    """Pair Camera_2 captures onto the logical steps, applying the D24/D10 fixes."""
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
        step = step_of_ts.get(ts)
        if step is None:
            step = cam2_fix_step(oak_fix, src_dir, step_of_dir)
        if step is None:
            issues.append(
                f"oak2: capture {ts} in folder {src_dir} has no cam1 partner; skipped"
            )
            continue
        key = FrameKey(desktop, step, "oak2")
        if key in frames:
            issues.append(
                f"oak2: step {step} already filled by folder {frames[key].src_step_dir}; "
                f"dropped duplicate from {src_dir} ({ts})"
            )
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


def _add_scanner(
    desktop: int,
    roots: dict,
    step_of_dir: dict[str, int],
    frames: dict[FrameKey, FrameFile],
) -> list[str]:
    """Add the scanner (top-down) frames; step number = ``int(folder[3:-1])``.

    No timestamp cross-check is possible here: the scanner PNGs were copied to F:
    in one batch (all mtimes are the copy date, not the capture date), so
    ``ts`` is informational only and must not be compared with the OAK capture
    times the way :func:`_add_realsense` does.
    """
    issues: list[str] = []
    base = scanner_desktop_dir(roots["scanner_root"], desktop, issues)
    if base is None:
        issues.append(f"scan: no scanner folder found for desktop {desktop}")
        return issues
    for entry in sorted(dirs(base, issues), key=lambda e: e.name):
        name = entry.name
        if not (name.startswith("RGB") and name.endswith("1") and name[3:-1].isdigit()):
            continue
        nominal = int(name[3:-1])
        step = step_of_dir.get(f"{nominal:03d}")
        if step is None:
            issues.append(f"scan: {name} (nominal step {nominal}) has no cam1 folder; skipped")
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
        frames[key] = FrameFile(
            key=key,
            path=main[1],
            aux={"burst": [b[1] for b in burst]},
            ts=iso_from_mtime(main[2]),
            src_step_dir=name,
        )
    return issues


def _add_realsense(
    desktop: int,
    roots: dict,
    aliases: list[str],
    step_of_dir: dict[str, int],
    frames: dict[FrameKey, FrameFile],
    rs_fix: Optional[dict] = None,
) -> list[str]:
    """Add the RealSense frames, aligning by folder number *and* file time.

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
        if step is None:
            issues.append(
                f"rs: folder {entry.name} (nominal step {nominal}) has no cam1 folder; skipped"
            )
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
    issues.extend(_rs_extra(desktop, rs_desktop, rs_fix or {}, step_of_dir, frames))
    return issues


def _rs_extra(desktop: int, rs_desktop: str, rs_fix: dict, step_of_dir: dict[str, int],
              frames: dict[FrameKey, FrameFile]) -> list[str]:
    """Apply a declared ``rs_extra_from`` fix; every outcome is an issue line."""
    folder = rs_fix.get("rs_extra_from")
    if not folder or "as_step" not in rs_fix:
        return []
    head = f"fix:rs[{desktop}] rs_extra_from {folder}"
    nominal = int(rs_fix["as_step"])
    step = step_of_dir.get(f"{nominal:03d}")
    path = os.path.join(rs_desktop, *str(folder).split("/"))
    main = os.path.join(path, "original_color.png")
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
) -> dict[int, DesktopIndex]:
    """Scan every desktop in ``desktops`` and return ``{desktop: DesktopIndex}``."""
    fixes = load_fixes(fixes_path)
    out: dict[int, DesktopIndex] = {}
    for desktop in desktops:
        idx = scan_desktop(int(desktop), roots, fixes)
        out[int(desktop)] = idx
        if progress:
            n_missing = len(idx.missing)
            print(
                f"[index] Desktop {desktop:>2}: {idx.n_steps:>3} steps, "
                f"{len(idx.frames):>4} frames, {n_missing:>3} missing, "
                f"{len(idx.issues)} issues",
                flush=True,
            )
    return out


def _respelled(value: Any, spell: Callable[[str], str]) -> Any:
    """``spell`` applied to a path, or to every path of a list (a burst)."""
    if isinstance(value, str):
        return spell(value) if value else value
    if isinstance(value, list):
        return [spell(v) if isinstance(v, str) and v else v for v in value]
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


def save_index(idx: dict[int, DesktopIndex], path: str) -> None:
    """Write the index to JSON (creating parent directories)."""
    payload = {
        "version": INDEX_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "desktops": {
            str(d): {
                "desktop": di.desktop,
                "n_steps": di.n_steps,
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
        gallery = int(payload.get("version", 1)) >= GALLERY_INDEX_VERSION
    except (TypeError, ValueError):
        gallery = False
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

    started = datetime.now()
    idx = build_index(range(args.first, args.last + 1), roots, args.fixes, progress=True)
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
