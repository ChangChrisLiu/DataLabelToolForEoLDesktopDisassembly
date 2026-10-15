"""The index half of the data pipeline behind ``python -m tda.cli`` (spec 2.2-2.5).

:mod:`tda.cli` holds the argument parsing and the printing, this module and
:mod:`tda.pipeline_logs` hold the work, so every step is callable from a test or
from the GUI without going through ``argparse``.

The pipeline runs in one order, because each step needs what the previous one
wrote::

    build-index -> load-index -> import-logs -> import-ls

Here: the shared configuration helpers, ``index.json`` -> ``frame`` rows and
pose segments, the pose-segment split at the ``reorient`` steps, and the
counters behind ``status``. The Drive-sheet import lives next door in
:mod:`tda.pipeline_logs`.

Nothing here writes outside ``D:`` -- :meth:`tda.core.db.Db.backup` is the one
command that may reach the raw drive, and only into ``backup_dir`` as
:func:`ready_backup_dest` finds it today.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Iterable, Optional

from tda.core import db_pose, rawroot
from tda.core.db import Db
from tda.core.pose_breaks import describe_discard, describe_uncarried
from tda.core.db_backup import DEFAULT_KEEP
from tda.core.db_status import VIEW_COUNTERS
from tda.core.index import DesktopIndex, load_index
from tda.core.model import (
    ALL_VIEWS,
    COMPONENT_CAMERA,
    COMPONENT_VIEWS,
    VIEWS,
    FrameKey,
    StepType,
    is_component_view,
)
from tda.core.truth_inputs import HW_INFERRED, TRUTH_AUX_KEYS, infer_hw

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS_PATH = "configs/paths.yaml"
INDEX_NAME = "index.json"
INDEX_REPORT_NAME = "index_report.md"
LOGS_REPORT_NAME = "import_logs_issues.md"
LS_SUMMARY_NAME = "ls_import_summary.json"
LS_EXPORT_NAME = "labelstudio/humansignal_annotated_projects_export.json"
DRIVE_SUBDIR = "drive"
ALL_DESKTOPS = range(1, 67)
#: How many pose issues one desktop's meta keeps (newest last).
POSE_ISSUE_LIMIT = 50
#: ``op_log.annotator`` of a re-cut the pipeline ran, rather than a human.
SPLIT_ANNOTATOR = "cli:split-pose-segments"

#: ``print`` by default; tests and the GUI pass their own sink.
Log = Callable[[str], None]

__all__ = [
    "backup_dest", "backup_keep", "cache_file", "desktops_without_steps", "drive_dir", "index_path",
    "load_index_into_db", "load_paths", "ls_export_path", "merge_desktop_meta", "open_db",
    "parse_desktops", "read_index", "ready_backup_dest", "require", "split_pose_segments",
    "status_rows", "gallery_rows", "settle_gallery", "STALE_INDEX", "StaleIndex", "stale_index",
]


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def load_paths(path: str = DEFAULT_PATHS_PATH) -> dict:
    """Load ``paths.yaml``; a relative path may be given from the repo root."""
    import yaml

    if not (os.path.isabs(path) or os.path.exists(path)):
        path = str(REPO_ROOT / path)
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def require(paths: dict, key: str) -> str:
    """One path from ``paths.yaml``, with a readable error when it is missing."""
    value = paths.get(key)
    if not value:
        raise ValueError(f"paths.yaml defines no {key}")
    return str(value)


def parse_desktops(spec: Optional[str]) -> Optional[set[int]]:
    """Parse ``"13"`` / ``"1-66"`` / ``"1-3,13,60-61"``; ``None`` means "all"."""
    if spec is None:
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        first, _, last = part.partition("-")
        try:
            lo = int(first)
            hi = int(last) if last else lo
        except ValueError:
            raise ValueError(f"not a desktop range: {part!r}") from None
        if hi < lo:
            raise ValueError(f"empty desktop range: {part!r}")
        out.update(range(lo, hi + 1))
    if not out:
        raise ValueError(f"no desktops in {spec!r}")
    return out


def wanted(desktop: int, desktops: Optional[set[int]]) -> bool:
    """Is this desktop in the ``--desktops`` selection (``None`` means all)?"""
    return desktops is None or desktop in desktops


def index_path(paths: dict) -> str:
    """``<cache_dir>/index.json``."""
    return os.path.join(paths.get("cache_dir", "cache"), INDEX_NAME)


def cache_file(paths: dict, name: str) -> str:
    """A file inside ``cache_dir`` (the directory is created on demand)."""
    cache = paths.get("cache_dir", "cache")
    os.makedirs(cache, exist_ok=True)
    return os.path.join(cache, name)


def drive_dir(paths: dict) -> str:
    """``<raw_logs_dir>/drive`` -- the exported Drive sheets."""
    return os.path.join(paths.get("raw_logs_dir", "raw_logs"), DRIVE_SUBDIR)


def ls_export_path(paths: dict) -> str:
    """``<raw_logs_dir>/labelstudio/humansignal_annotated_projects_export.json``."""
    return os.path.join(paths.get("raw_logs_dir", "raw_logs"), *LS_EXPORT_NAME.split("/"))


def _within(target: Path, root: Path) -> bool:
    """Is ``target`` ``root`` itself or something under it?

    ``resolve()`` has already followed symlinks and ``subst`` drive mappings, so
    two spellings of one directory arrive here as one path -- except for its
    *case*, which ``resolve()`` can only repair for components that exist on
    disk, and ``backup_dir`` usually does not exist before the first backup.
    ``os.path.normcase`` finishes the job (and is a no-op on a case-sensitive
    filesystem). The separator is appended before the prefix test so that
    ``backups_old`` is not read as being inside ``backups``.
    """
    root_text = os.path.normcase(str(root)).rstrip("\\/")
    target_text = os.path.normcase(str(target)).rstrip("\\/")
    return target_text == root_text or target_text.startswith(root_text + os.sep)


def backup_dest(paths: dict, dest: Optional[str] = None,
                raw: Optional["rawroot.RawRoot"] = None) -> str:
    """Where a backup may go: ``backup_dir`` itself or a folder inside it.

    ``backup_dir`` is the one place on the read-only raw drive this tool
    writes to (spec 3.5), so an explicit ``--dest`` is confined to it rather
    than trusted. A configuration without a ``backup_dir`` raises ``ValueError``
    from :func:`require` -- ``--dest`` narrows that setting, it never replaces
    it.

    A ``backup_dir`` on the raw drive follows that drive to the letter it has
    *today* (:func:`tda.core.rawroot.backup_root`), and so does a ``--dest``
    spelled with the recorded letter. Beyond re-checking that the raw root still
    holds the dataset nothing here touches the disk; see
    :func:`ready_backup_dest` for the answer a copy can actually be written to.
    ``raw`` is an answer already re-checked by the caller.
    """
    require(paths, "backup_dir")
    raw = raw or rawroot.checked(paths)
    root = Path(rawroot.backup_root(paths, raw)).resolve()
    if dest is None:
        return str(root)
    target = Path(raw.resolve(dest) or dest).resolve()
    if not _within(target, root):
        raise ValueError(f"--dest must be inside the configured backup_dir ({root})")
    return str(target)


def ready_backup_dest(paths: dict, dest: Optional[str] = None) -> str:
    """:func:`backup_dest`, checked and made usable -- or ``BackupUnavailable``.

    The one gate every backup goes through, the CLI's and the window's exit
    copy alike. It **never creates a folder on a volume that is not the raw
    data one**: on the raw drive only the ``TDA_backups`` leaf may be made; a
    ``backup_dir`` anywhere else must already exist
    (:func:`tda.core.rawroot.backup_target`). ``BackupUnavailable`` is an
    ``OSError``, so every caller already reports it as "backup failed: ..."
    with the reason in both languages.
    """
    # One answer for both halves, re-checked now: the drive the process found
    # at start-up may have been unplugged, and its letter given to another
    # volume, since.
    raw = rawroot.checked(paths)
    out = backup_dest(paths, dest, raw)  # refuses a bad --dest before any write
    rawroot.backup_target(paths, raw)
    return out


def backup_keep(paths: dict, prune: bool = True) -> Optional[int]:
    """How many backups to keep: ``paths.yaml``'s ``backup_keep``, or the default.

    ``prune=False`` (``backup --no-prune``) answers ``None``, which
    :func:`tda.core.db_backup.prune_backups` reads as "no limit".
    """
    if not prune:
        return None
    try:
        return int(paths.get("backup_keep", DEFAULT_KEEP))
    except (TypeError, ValueError):
        raise ValueError(
            f"paths.yaml's backup_keep must be a whole number of files, not "
            f"{paths.get('backup_keep')!r}"
        ) from None


def open_db(paths: dict, override: Optional[str] = None) -> Db:
    """Open the database named by ``--db`` or by ``paths.yaml``."""
    return Db(override or require(paths, "db_path"))


def read_index(paths: dict, path: Optional[str] = None) -> dict[int, DesktopIndex]:
    """Read ``index.json``; an absent file yields an empty index."""
    target = path or index_path(paths)
    if not os.path.exists(target):
        return {}
    return load_index(target)


# --------------------------------------------------------------------------- #
# desktop meta
# --------------------------------------------------------------------------- #
def merge_desktop_meta(db: Db, desktop: int, updates: dict) -> None:
    """Update part of a desktop's meta, keeping what other steps wrote.

    :meth:`tda.core.db.Db.upsert_desktop` replaces the whole row, so the stored
    meta is read back first: ``load-index`` must not drop the brand
    ``import-logs`` wrote, and vice versa. A ``None`` in ``updates`` clears its
    field.
    """
    meta = db.get_desktop(desktop) or {}
    meta.pop("id", None)
    meta.update(updates)
    db.upsert_desktop(desktop, {k: v for k, v in meta.items() if v is not None})


def add_desktop_issues(db: Db, desktop: int, key: str, lines: Iterable[str]) -> None:
    """Append issue lines to one meta list, without repeating what is there."""
    meta = db.get_desktop(desktop) or {}
    kept = list(meta.get(key) or [])
    for line in lines:
        if line not in kept:
            kept.append(line)
    merge_desktop_meta(db, desktop, {key: kept[-POSE_ISSUE_LIMIT:]})


# --------------------------------------------------------------------------- #
# load-index
# --------------------------------------------------------------------------- #
#: Why ``load-index`` refuses an index file older than the gallery (task U10).
STALE_INDEX = ("is a version-1 index, written before the components gallery and the "
               "RealSense fixes (rs_extra_from: D1 step 40, D24 step 43): loading it would "
               "mark frames the current index has as missing again. Nothing was loaded. "
               "Run build-index first, then load-index again.")


class StaleIndex(ValueError):
    """An index file written before the components gallery (version 1, task U10)."""


def stale_index(index: dict[int, DesktopIndex]) -> list[int]:
    """The desktops of ``index`` read from a version-1 file (``gallery`` unset)."""
    return sorted(d for d, di in index.items() if not getattr(di, "gallery", True))


def load_index_into_db(
    db: Db,
    index: dict[int, DesktopIndex],
    desktops: Optional[set[int]] = None,
    log: Optional[Log] = None,
) -> dict:
    """Write ``index.json`` into the database: frames + pose segment 1.

    Every indexed frame becomes a ``frame`` row carrying its path, aux files and
    capture time; every key in :attr:`~tda.core.index.DesktopIndex.missing` gets
    a row with ``missing=1`` so the logical step still exists in every view
    (spec 2.2). Each view then gets one pose segment covering steps ``1..n`` --
    immediately re-cut at the ``reorient`` steps when the logs are already in
    (see :func:`split_pose_segments`).

    The frames' own ``pose_segment`` column is left unset on purpose: a frame
    resolves its segment through the step range, so re-cutting the segments
    never has to rewrite thousands of frame rows.

    One desktop is one transaction **and** one try/except: a malformed entry --
    a frame key the index cannot decode, a step count that is not a number --
    rolls its own desktop back and is listed in ``failed``, rather than ending a
    66-desktop run at number 40 and leaving the caller to guess which ones
    landed. The caller exits non-zero when ``failed`` is non-empty.

    An index file written before the components gallery (version 1) is refused
    outright with :class:`StaleIndex` before anything is written: it lacks the
    frames the index fixes added since (the RealSense ``rs_extra_from`` steps of
    D1 and D24), and loading it would mark them missing again (task U10).
    """
    if stale_index(index):
        raise StaleIndex(f"the index {STALE_INDEX}")
    counts: dict = {"desktops": 0, "frames": 0, "missing": 0, "segments": 0,
                    "skipped": 0, "failed": []}
    for desktop in sorted(index):
        if not wanted(desktop, desktops):
            continue
        try:
            _load_one(db, desktop, index[desktop], counts, log)
            counts["desktops"] += 1  # only a desktop that really landed counts
        except Exception as exc:  # one bad entry must not end the run
            counts["failed"].append(
                f"D{desktop:02d}: {type(exc).__name__}: {exc}"
            )
            if log:
                log(f"[load-index] D{desktop:02d}: FAILED, {type(exc).__name__}: {exc}; "
                    f"nothing was written for it")
    return counts


def _merged_aux(db: Db, ff) -> dict:
    """The index's ``aux`` for one frame, with the truth table's keys kept.

    ``aux`` has two writers. The index puts there what it found on the source
    drive; :func:`tda.core.truth_inputs.frame_hw` puts back the image size it
    measured, where it measured it, and the cached copy it read
    (:data:`~tda.core.truth_inputs.TRUTH_AUX_KEYS`). Re-building the index
    replaced the whole object, so every measurement was thrown away and the next
    compile went to the source drive again -- and inferred the view's nominal
    size instead whenever that drive was detached, which silently changes the
    canvas every compiled mask is in.

    Only while the row still names the same file, in every view: a measurement
    is of a photo, not of a frame key. A gallery shot re-pointed at another
    photo, a sequence frame the index fixes now take from elsewhere, and a row
    that was missing (its path is gone, see :func:`_missing_aux`) all start
    over, and :func:`~tda.core.truth_inputs.frame_hw` measures the new file
    (task U10) -- keeping ``hw_source = measured`` there would never let it.
    """
    row = db.get_frame(ff.key) or {}
    if not row or row.get("path") != ff.path:
        return dict(ff.aux or {})
    held = row.get("aux") or {}
    kept = {k: held[k] for k in TRUTH_AUX_KEYS if k in held}
    return dict(ff.aux or {}) | kept


def _missing_aux(row: Optional[dict]) -> dict:
    """What a row that turns ``missing`` keeps of the truth table's keys.

    Only an inferred size (the view's nominal one, which does not depend on a
    file): a measured size and the copy it was read off belong to a photo the
    row no longer names, and a later index that has a photo there again --
    perhaps another one -- must have it measured (task U10).
    """
    held = (row or {}).get("aux") or {}
    if held.get("hw_source") == HW_INFERRED and "hw" in held:
        return {"hw": held["hw"], "hw_source": HW_INFERRED}
    return {}


def _load_one(
    db: Db, desktop: int, di: DesktopIndex, counts: dict, log: Optional[Log]
) -> None:
    """One desktop's frames and pose segments, in one transaction.

    The components gallery (task U10) is checked first: a shot whose photo the
    new index changes or no longer has, while somebody's work sits on it, fails
    the desktop (:class:`GalleryConflict`) before anything is written.
    """
    with db.transaction():
        refused = gallery_refusals(db, desktop, di)
        if refused:
            raise GalleryConflict(
                "the new index would move annotated components-gallery shots: "
                + "; ".join(refused) + " -- re-check the Components folders, or move the "
                "work off those shots first")
        merge_desktop_meta(
            db, desktop, {"index_issues": list(di.issues), "index_n_steps": di.n_steps}
        )
        frames = di.frames
        # a sheet row of the laid-out parts the gallery already holds is not
        # put back into the sequence just to be moved out again (U10)
        held = _held_by_gallery(db, desktop, di)
        repointed: list[str] = []
        for ff in frames.values():
            if ff.key in held:
                continue
            if is_component_view(ff.key.view):
                old = (db.get_frame(ff.key) or {}).get("path")
                if old and old != ff.path:
                    repointed.append(f"{ff.key.view} shot {ff.key.step}")
            db.upsert_frame(ff.key, ff.path, _merged_aux(db, ff), ff.ts,
                            {"missing": False})
        for key in di.missing:
            if key not in held:
                # an inferred size stays (a re-load leaves a loaded desktop as
                # it was); a measured one goes with the photo it was read off
                db.upsert_frame(key, None, _missing_aux(db.get_frame(key)), None,
                                {"missing": True})
        dropped = _drop_stale_shots(db, desktop, di) if di.gallery else []
        if di.n_steps < 1:
            counts["frames"] += len(frames)
            counts["missing"] += len(di.missing)
            counts["skipped"] += 1
            if log:
                log(f"[load-index] D{desktop:02d}: no steps in the index, no pose segment")
            return
        gallery = settle_gallery(db, desktop)   # before the sequence is seeded
        _seed_pose_segments(db, desktop, di.n_steps)
        segments = split_pose_segments(db, desktop)
    counts["frames"] += len(frames)
    counts["missing"] += len(di.missing)
    counts["segments"] += sum(segments.values())
    counts["gallery"] = counts.get("gallery", 0) + sum(gallery["shots"].values())
    if log:
        extra = f", {len(di.missing)} missing" if di.missing else ""
        cut = f", {max(segments.values())} pose segments" if segments else ""
        shots = ""
        if any(gallery["shots"].values()):
            shots = ", gallery " + " ".join(
                f"{view} {n}" for view, n in gallery["shots"].items() if n)
        log(
            f"[load-index] D{desktop:02d}: {di.n_steps} steps, "
            f"{len(frames)} frames{extra}{cut}{shots}"
        )
        for line in gallery["moved"]:
            log(f"[load-index] D{desktop:02d}: {line}")
        if repointed:
            log(f"[load-index] D{desktop:02d}: gallery shot(s) now on another photo: "
                f"{', '.join(repointed)} (nobody had worked on them); build-cache "
                f"copies the new photo")
        if dropped:
            log(f"[load-index] D{desktop:02d}: gallery shot(s) no longer in the index, "
                f"dropped: {', '.join(dropped)}")


# --------------------------------------------------------------------------- #
# the components gallery (task U10, spec 3.2b)
# --------------------------------------------------------------------------- #
#: A sheet row that photographed the laid-out parts rather than a step of the
#: teardown: typed ``ignore`` and named for all the components (D66's step 13,
#: "All components").
ALL_COMPONENTS_RE = re.compile(r"\ball\s+(?:the\s+)?(?:components?|parts)\b", re.IGNORECASE)
#: Where a sequence view's frame of such a row goes: its camera's gallery.
GALLERY_OF: dict[str, str] = {camera: view for view, camera in COMPONENT_CAMERA.items()}
#: ``op_log.annotator`` of the whole-image ROI a gallery shot is seeded with.
GALLERY_ANNOTATOR = "cli:load-index"
#: ``aux`` key naming the sheet step a relocated gallery shot came from.
SHEET_STEP_KEY = "sheet_step"
#: A sheet row's photo is shot ``SHEET_SHOT_BASE + sheet step`` (D66: 1013) of
#: its camera's gallery. Keyed by the sheet step it never moves, and the
#: Components folders' own shots (1..N, a handful per camera) can never reach
#: it -- a folder that appears on the drive later cannot collide with it, and a
#: relocated photo is never renumbered under the shapes drawn on it. The window
#: and the guide count shots by position (第 k/N 张), not by this number.
SHEET_SHOT_BASE = 1000


class GalleryConflict(ValueError):
    """A re-index would move a components-gallery shot somebody worked on."""


def gallery_rows(db: Db, desktop: int) -> list[int]:
    """The steps of ``desktop``'s sheet that are layout photos, not teardown steps."""
    return [int(rec.step) for rec in db.steps(desktop)
            if rec.step_type == StepType.IGNORE.value
            and ALL_COMPONENTS_RE.search(str(rec.raw_name or ""))]


#: ``cli check``'s word for a relocated sheet shot whose sheet row was retyped.
ORPHAN_SHEET_SHOT = "sheet_shot_without_its_row"


def orphan_sheet_shots(db: Db, desktop: int, view: str) -> list[str]:
    """Gallery shots of ``view`` whose sheet row is no longer a layout photo (U10).

    A sheet row's photo moved into the gallery (``aux.sheet_step``) stays there
    when the row is later retyped or renamed in S1 -- nothing moves a shot
    back: the step has no sequence frame until the next ``load-index`` puts
    one back, and from then on the photo is in both. ``cli check`` says so;
    the remedy is to retype the row back (it is the photo of the laid-out
    parts). Kept as a report, not repaired: the shot may carry work.
    """
    rows = set(gallery_rows(db, desktop))
    out = []
    for row in db.frames_for(desktop, view):
        step = (row.get("aux") or {}).get(SHEET_STEP_KEY)
        if step is not None and int(step) not in rows:
            out.append(f"{ORPHAN_SHEET_SHOT}:{view} shot {int(row['step'])} (sheet step "
                       f"{int(step)} is no longer an ignore \"All components\" row, but "
                       f"its photo is still this shot -- retype the row back in S1)")
    return out


def shot_work(db: Db, key: FrameKey) -> list[str]:
    """What sits on one gallery shot: the frame's own work and its segment's shapes.

    :meth:`tda.core.db.Db.frame_work` plus every keyframe of the shot's pose
    segment (a gallery shot is a segment of its own, so a shape drawn there may
    be anchored anywhere in it).
    """
    work = list(db.frame_work(key))
    seg = (db.pose_segment_for(key) or {}).get("seg")
    if seg:
        n = db.conn.execute(
            "SELECT COUNT(*) FROM shape_keyframe WHERE desktop=? AND view=? "
            "AND pose_segment=? AND anchor_step<>?",
            (key.desktop, key.view, int(seg), key.step)).fetchone()[0]
        if n:
            work.append(f"{n} shape keyframe(s) in its segment")
    return work


def _sheet_shot_row(db: Db, desktop: int, view: str, sheet_step: int) -> Optional[dict]:
    """The gallery row holding a sheet step's photo in ``view``, or ``None``."""
    return next((row for row in db.frames_for(desktop, view)
                 if (row.get("aux") or {}).get(SHEET_STEP_KEY) == sheet_step), None)


def gallery_refusals(db: Db, desktop: int, di: DesktopIndex) -> list[str]:
    """Why loading ``di`` would move work on a gallery shot; ``[]`` when it would not.

    A shot changes its photo when the index has another file at its number (a
    folder removed or added in front of it), and loses it when the index has
    none (the last shots of a shorter folder). Either is fine on a shot nobody
    touched -- it is re-pointed or dropped -- and refused on one with work
    (:func:`shot_work`). A sheet row's photo (shot ``SHEET_SHOT_BASE + step``)
    is the sheet step's sequence photo in the index, held to the same rule.
    An index written before the gallery existed says nothing about it and is
    never a reason.
    """
    if not getattr(di, "gallery", True):
        return []
    out: list[str] = []
    for view in COMPONENT_VIEWS:
        index = {k.step: ff for k, ff in di.frames.items() if k.view == view}
        for row in db.frames_for(desktop, view):
            if (row.get("aux") or {}).get(SHEET_STEP_KEY) is not None:
                continue
            shot = int(row["step"])
            ff = index.get(shot)
            if ff is not None and ff.path == row.get("path"):
                continue
            work = shot_work(db, FrameKey(desktop, shot, view))
            if not work:
                continue
            what = (f"the index has another photo there ({ff.path})" if ff is not None
                    else "its photo is no longer in the index")
            out.append(f"D{desktop:02d} {view} shot {shot} ({row.get('path')}): {what}, "
                       f"and it holds {', '.join(work)}")
    for step in gallery_rows(db, desktop):
        for camera, view in GALLERY_OF.items():
            row = _sheet_shot_row(db, desktop, view, step)
            ff = di.frames.get(FrameKey(desktop, step, camera))
            if row is None or ff is None or ff.path == row.get("path"):
                continue
            work = shot_work(db, FrameKey(desktop, int(row["step"]), view))
            if work:
                out.append(f"D{desktop:02d} {view} shot {row['step']} (sheet step {step}): "
                           f"the index has another photo for it ({ff.path}), and it holds "
                           f"{', '.join(work)}")
    return out


def _held_by_gallery(db: Db, desktop: int, di: DesktopIndex) -> set[FrameKey]:
    """Sequence keys of sheet steps whose photo the gallery already holds as is.

    Loading them again would only put the row back for :func:`settle_gallery`
    to move out again -- and say so on every run.
    """
    held: set[FrameKey] = set()
    for step in gallery_rows(db, desktop):
        relocated = False
        for camera, view in GALLERY_OF.items():
            row = _sheet_shot_row(db, desktop, view, step)
            if row is None:
                continue
            relocated = True
            ff = di.frames.get(FrameKey(desktop, step, camera))
            if ff is None or ff.path == row.get("path"):
                held.add(FrameKey(desktop, step, camera))
        if relocated:
            held |= {FrameKey(desktop, step, v) for v in VIEWS if v not in GALLERY_OF}
    return held


def _drop_stale_shots(db: Db, desktop: int, di: DesktopIndex) -> list[str]:
    """Drop the gallery rows the index no longer has (folder shots only).

    Only ever called once :func:`gallery_refusals` found nothing, so every row
    dropped here is one nobody worked on; its empty segment goes with it.
    """
    dropped: list[str] = []
    for view in COMPONENT_VIEWS:
        index = {k.step for k in di.frames if k.view == view}
        for row in db.frames_for(desktop, view):
            shot = int(row["step"])
            if shot in index or (row.get("aux") or {}).get(SHEET_STEP_KEY) is not None:
                continue
            if db.delete_unannotated_frame(FrameKey(desktop, shot, view)):
                db.delete_empty_pose_segment(desktop, view, shot)
                dropped.append(f"{view} shot {shot}")
    return dropped


def settle_gallery(db: Db, desktop: int) -> dict:
    """Move layout-photo rows into the gallery and give every shot its segment.

    Idempotent, and called by both ``load-index`` and ``import-logs`` -- the
    rule needs the step table, which may arrive after the frames.

    1. **A sheet row of the parts** (:func:`gallery_rows`): its step keeps its
       number and its ``ignore`` type, but its frames are not frames of the
       teardown -- each sequence view's photo of it becomes shot
       ``SHEET_SHOT_BASE + step`` of its camera's gallery view (scanner: none;
       ``aux["sheet_step"]`` says where it came from) and the sequence rows are
       dropped. A row somebody annotated is never dropped (it is reported
       instead). The measured size of a photo already there is kept.
    2. **One pose segment per shot**, ``[k, k]``, with the whole image as its
       ROI. The layout was re-arranged between shots, so a shape drawn on one
       must reach no other -- and a pose segment is exactly the scope a shape,
       a layer order and a pairwise layering live in.

    Returns ``{"moved": [lines], "shots": {view: n}}``; ``moved`` names only
    what actually moved this time.
    """
    moved: list[str] = []
    with db.transaction():
        for step in gallery_rows(db, desktop):
            for view in VIEWS:
                key = FrameKey(desktop, step, view)
                row = db.get_frame(key)
                if row is None:
                    continue
                if db.frame_is_annotated(key):
                    moved.append(f"step {step} ({view}) is a photo of the laid-out parts "
                                 f"but has annotations; left in the sequence")
                    continue
                target = GALLERY_OF.get(view)
                if target is not None and row.get("path") and not row.get("missing"):
                    shot = _gallery_shot(db, desktop, target, step)
                    there = FrameKey(desktop, shot, target)
                    held = db.get_frame(there) or {}
                    aux = dict(row.get("aux") or {})
                    aux[SHEET_STEP_KEY] = step
                    if held.get("path") == row["path"]:
                        # the same photo: keep what the truth table measured on it
                        kept = held.get("aux") or {}
                        aux |= {k: kept[k] for k in TRUTH_AUX_KEYS if k in kept}
                    db.upsert_frame(there, row["path"], aux, row.get("ts"),
                                    {"missing": False})
                    where = f"{target} shot {shot}"
                else:
                    where = "nowhere (no gallery for this camera)"
                db.delete_unannotated_frame(key)
                moved.append(f"step {step} ({view}) is a photo of the laid-out parts: "
                             f"moved to {where}")
        shots = {view: _seed_gallery_segments(db, desktop, view) for view in COMPONENT_VIEWS}
    return {"moved": moved, "shots": shots}


def _gallery_shot(db: Db, desktop: int, view: str, sheet_step: int) -> int:
    """The shot a sheet row's photo has: ``SHEET_SHOT_BASE + sheet_step``.

    A photo an earlier build put at another number moves there when nobody
    worked on it (its empty segment goes with it); one with work stays where
    it is.
    """
    want = SHEET_SHOT_BASE + int(sheet_step)
    row = _sheet_shot_row(db, desktop, view, sheet_step)
    if row is None or int(row["step"]) == want:
        return want
    old = FrameKey(desktop, int(row["step"]), view)
    if shot_work(db, old):
        return old.step
    db.delete_unannotated_frame(old)
    db.delete_empty_pose_segment(desktop, view, old.step)
    return want


def _seed_gallery_segments(db: Db, desktop: int, view: str) -> int:
    """Segment ``[k, k]`` with a whole-image ROI for every shot; the number of shots.

    Nothing that is there is rewritten: a segment keeps its row (and so the
    shapes drawn against it), and a stored ROI is left alone.
    """
    shots = [int(row["step"]) for row in db.frames_for(desktop, view) if not row.get("missing")]
    have = {int(s["seg"]): s for s in db.pose_segments(desktop, view)}
    for shot in shots:
        seg = have.get(shot)
        if seg is None:
            db.set_pose_segment(desktop, view, shot, shot, shot, shot, None, None)
        elif (seg.get("start_step"), seg.get("end_step")) != (shot, shot):
            db.update_pose_segment(desktop, view, shot, start_step=shot, end_step=shot,
                                   ref_step=shot)
        if not (seg or {}).get("roi"):
            height, width = _gallery_hw(db, FrameKey(desktop, shot, view))
            db.set_pose_segment_roi(desktop, view, shot, [0, 0, width, height],
                                    annotator=GALLERY_ANNOTATOR, hw=(height, width))
    return len(shots)


def _gallery_hw(db: Db, key: FrameKey) -> tuple[int, int]:
    """The shot's recorded size, else its camera's nominal one."""
    recorded = ((db.get_frame(key) or {}).get("aux") or {}).get("hw")
    try:
        height, width = (int(v) for v in recorded)
        if height > 0 and width > 0:
            return height, width
    except (TypeError, ValueError):
        pass
    return infer_hw(key.view)


# --------------------------------------------------------------------------- #
# pose segments
# --------------------------------------------------------------------------- #
def _seed_pose_segments(db: Db, desktop: int, n_steps: int) -> None:
    """Give every view a segment covering ``1..n_steps`` without erasing anything.

    A view with no segment yet gets segment 1. A view that already has segments
    keeps them -- only the last one's ``end_step`` follows a re-built index --
    because rewriting segment 1 wholesale would drop the chassis corners a human
    clicked (spec 2.4). :func:`split_pose_segments` then re-derives the cuts.
    """
    for view in VIEWS:
        segments = db.pose_segments(desktop, view)
        if not segments:
            db.set_pose_segment(desktop, view, 1, 1, n_steps, n_steps, None, None)
        elif segments[-1]["end_step"] != n_steps:
            db.update_pose_segment(desktop, view, segments[-1]["seg"], end_step=n_steps)


def split_pose_segments(db: Db, desktop: int) -> dict[str, int]:
    """Re-cut every view's pose segments: ``reorient`` steps ∪ that view's breaks.

    Flipping the chassis breaks the pose in all four views at once (spec 2.5),
    so a step typed ``reorient`` opens a new segment everywhere: segment ``k+1``
    starts *at* the reorient step itself.  Since v1.5 a view also has breaks of
    its **own** -- the camera was knocked, or the chassis slid, in front of one
    lens -- which is why the boundary list is computed per view from the
    ``accepted`` rows of ``pose_break``.  A reorient step that also carries a
    manual break is one boundary, not an empty segment between two.

    The function is idempotent and is called by both ``load-index`` and
    ``import-logs``: it re-derives the whole boundary list from the recorded
    steps and the stored breaks every time, so re-running either command over a
    view a human has since cut by hand leaves that cut exactly where it is.

    The rows are moved by :meth:`tda.core.db.Db.apply_recut`, one transaction
    per view -- keyframes to the segment of their own anchor, the layer order
    and both ROIs copied into every piece, the chassis corners kept only where
    the reference step still is, frozen frames queued for a re-check.  What a
    re-cut could not keep is recorded in the desktop's ``pose_issues`` meta.

    Returns ``{view: number of segments}`` for the views that have any.
    """
    out: dict[str, int] = {}
    issues: list[str] = []
    for view in VIEWS:
        result = db.recut_view(desktop, view, annotator=SPLIT_ANNOTATOR)
        if not result["ranges"]:
            continue            # a view nothing photographed has nothing to cut
        issues += _recut_issues(view, result)
        out[view] = len(result["ranges"])
    if issues:
        add_desktop_issues(db, desktop, "pose_issues", issues)
    return out


def _recut_issues(view: str, result: dict) -> list[str]:
    """The audit lines of one view's re-cut: moved references, discarded rows."""
    lines = [
        _ref_moved(view, move["seg"], move["old"], move["start"], move["end"],
                   move["ref_step"])
        for move in result["ref_moves"]
        # A segment that never had a reference step and carried nothing drawn
        # against one lost nothing, so it has nothing to report: saying "the
        # reference step moved from None to 42" only teaches the annotator to
        # skim this list.
        if move["dropped"] or move["old"].get("ref_step") is not None
    ]
    # One renderer for every caller (:func:`tda.core.pose_breaks.describe_discard`).
    # Rendering it here by hand is what made a discarded *layer order* -- whose
    # payload is a list of layer keys, not a dict of column names -- raise a
    # TypeError out of ``split_pose_segments``, and with it roll back the whole
    # desktop of a ``load-index`` or an ``import-logs``.
    lines += [describe_discard(view, d) for d in result["discarded"]]
    if result["uncarried"]:
        lines.append(f"{view}: {describe_uncarried(result['uncarried'])}")
    return lines


def _ref_moved(view: str, seg: int, old: dict, start: int, end: int, ref: int) -> str:
    """The audit line for a pose segment whose reference frame moved.

    Only ever called when something was actually lost: a reference step that
    really moved, or geometry drawn against one that is gone either way. A
    segment with no previous reference is worded as what it is, rather than as
    a move "from None".

    The two ROIs are **not** in that loss: they are rectangles in the view's own
    image rather than geometry drawn against the reference frame, so a re-cut
    copies them into every piece (:mod:`tda.core.db_pose`).
    """
    lost = " the stored corners/homography were dropped;" \
        if db_pose._has_ref_geometry(old) else ""
    moved = (
        f"gave it reference step {ref}" if old["ref_step"] is None
        else f"moved the reference step from {old['ref_step']} to {ref}"
    )
    return (
        f"{view} pose segment {seg}: a new cut moved the range to [{start}-{end}] and "
        f"{moved};{lost} shapes anchored to it need re-checking"
    )


# --------------------------------------------------------------------------- #
# import-ls order guard
# --------------------------------------------------------------------------- #
def desktops_without_steps(db: Db, desktops: Iterable[int]) -> list[int]:
    """Which of ``desktops`` have no step table yet (the pipeline-order guard)."""
    return [d for d in sorted(set(desktops)) if not db.steps(d)]


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
#: The per-desktop counters ``status`` shows (a subset of the repository's).
STATUS_TOTALS = ("steps", "actions", "instances", "events")


def status_rows(db: Db, desktops: Optional[set[int]] = None) -> list[dict]:
    """Per-desktop counters: steps, instances and the four views' frame counters.

    Each view carries ``frames`` (rows in the index, missing ones included),
    ``missing``, ``keyframes`` (drawn or imported shapes), ``verified``
    (frames a human signed off), and ``done`` / ``work``: the window's
    ``[done/total]``, which leaves out the frames the view has no image for
    (task U2c).
    """
    totals = {name: db.count_per_desktop(name) for name in STATUS_TOTALS}
    per_view = {name: db.count_per_view(name) for name in VIEW_COUNTERS}
    ids = set(db.desktop_ids()) | set(totals["steps"]) | {d for d, _v in per_view["frames"]}
    out = []
    for desktop in sorted(d for d in ids if wanted(d, desktops)):
        meta = db.get_desktop(desktop) or {}
        out.append({
            "desktop": desktop,
            "brand": meta.get("brand_model_raw") or meta.get("brand") or "",
            **{name: totals[name].get(desktop, 0) for name in STATUS_TOTALS},
            # the sequence's four views and the components gallery's three
            "views": {
                view: {name: per_view[name].get((desktop, view), 0) for name in VIEW_COUNTERS}
                for view in ALL_VIEWS
            },
        })
    return out
