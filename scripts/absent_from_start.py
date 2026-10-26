"""Controller: 「这台没有这个零件」 -- a logged part the machine does not have (task U34).

The same op as the app's state-combo entry on the first frame and the Steps menu entry
(:func:`tda.core.absent.mark_absent`): the part is ``removed`` / ``elsewhere`` from
step 0 (two hand-written step-0 events), the stored automatic events are recompiled,
the graph meta re-stamped and the frozen frames whose work changes queued for a
re-check -- one ``absent_from_start`` op-log row with its inverse. ``--present``
writes the inverse 「其实有」 through the same path.

The case it was written for: D34 has no optical drive, yet its sheet logs one (step
37 "Try to remove the optical drive", failed; "optical drive screw" rows), so
``optical_drive.01`` stood installed from frame 1 and every frame asked for it.

Usage::

    python scripts/absent_from_start.py --desktop 34 --instance optical_drive.01
                                        [--apply] [--present] [--db PATH] [--out DIR]

Without ``--apply`` it is a dry run: the op is carried out on a backup-API copy of
``--db`` in ``--out`` (``dryrun_<stamp>.sqlite``), never on ``--db`` itself, and the
before / after below are that copy's. ``--db`` defaults to the LIVE database;
``--apply`` on it refuses while the annotator app is running (and when ``psutil`` is
missing, so that cannot be told) or holds the lock, and first takes the app's own
safety backup (``backup_dir`` of ``paths.yaml``). Somebody else's abandoned lock it
takes over is put back as it was afterwards; only a lock in its own name is removed
(one a killed run left behind included).

What it prints, before and after: the part's state and placement at the first and
the last step, the sheet's actions on it, the hand-written events, how many frames of
each sequence view ask for its geometry (what the compiler, ``Space`` and ``cli
check``'s ``missing_shape`` go by: it has no shape, so every one of them is a missing
shape) and on which frames the task card of the reverse walk has a row for it, the
state warnings about it, and the frozen frames the op queued. Before writing it lists
what else still names the part (:func:`tda.core.absent.residues`: single-frame
labels, review flags, open conflicts, pair overrides, layer orders -- kept as they
are); a 「被线挡住」 reason refuses, as in the app. After ``--apply`` it says to run
``python -m tda.cli check`` on the four views, which recompiles the part's old rows.
Output is UTF-8 whatever the console or pipe.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tda import pipeline as P  # noqa: E402
from tda.core import absent as AB  # noqa: E402
from tda.core.db import Db  # noqa: E402
from tda.core.model import VIEWS, FrameKey  # noqa: E402
from tda.core.states import needs_geom, validate_events  # noqa: E402
from tda.core.taxonomy import load_taxonomy  # noqa: E402
from tda.core.truth_inputs import (  # noqa: E402
    InputCache,
    annotatable_steps,
    events_of,
    instances_of,
    pose_segment_of,
    state_of,
)

LIVE = r"D:\DataSet\annotations\tda.sqlite"
OUT_DIR = r"D:\DataSet\.cache\tmp\u34"
ANNOTATOR = "controller"


def snapshot(src: str, dst: str) -> None:
    """A copy through the SQLite backup API (the source opened read-only)."""
    if os.path.exists(dst):
        os.remove(dst)
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    b = sqlite3.connect(dst)
    try:
        s.backup(b)
        ok = b.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        b.close()
        s.close()
    if ok != "ok":
        raise RuntimeError(f"snapshot {dst}: integrity {ok}")


def app_running() -> bool:
    """Is ``python -m tda.cli app`` running? (argv, not a substring of a shell line)

    Fail-closed (U34 round 2, N6): without ``psutil`` nobody can tell, and the
    answer that keeps the live database safe is "yes".
    """
    try:
        import psutil
    except ImportError:
        return True
    for p in psutil.process_iter(["cmdline"]):
        try:
            args = list(p.info.get("cmdline") or [])
        except Exception:  # noqa: BLE001 - a process that went away
            continue
        if any(a == "tda.cli" and i + 1 < len(args) and args[i + 1] == "app"
               for i, a in enumerate(args)):
            return True
    return False


def describe(db: Db, tax, desktop: int, key: str) -> dict:
    """What the machine says about ``key`` now (pure reads)."""
    cache = InputCache()
    instances = instances_of(db, desktop, cache)
    steps = sorted(int(r.step) for r in db.steps(desktop))
    out: dict = {"instance": key, "absent": AB.is_absent(db, desktop, key)}
    if key not in instances or not steps:
        out["missing"] = True
        return out
    for name, step in (("first", steps[0]), ("last", steps[-1])):
        held = state_of(db, tax, desktop, step, cache).get(key)
        out[f"state_{name}"] = (None if held is None
                                else f"step {step}: {held.state} / {held.placement}")
    out["actions"] = [f"step {a.step}: {a.verb} ({a.result})"
                      for a in sorted(db.actions(desktop), key=lambda a: (a.step, a.idx))
                      if a.target == key]
    out["manual_events"] = [f"step {e.step}: {e.attr} {e.old} -> {e.new} [{e.evidence_view}]"
                            for e in db.events(desktop) if e.target == key and not e.auto]
    log = events_of(db, tax, desktop, cache)
    out["derived_events"] = len([e for e in log if e.target == key and e.auto])
    out["warnings"] = [m for m in validate_events(instances, log, tax) if key in m]
    views: dict = {}
    from tda.ui.session_tasks import task_card_for   # Qt-free; late: the UI package
    for view in VIEWS:
        frames = annotatable_steps(db, desktop, view, [int(r["step"]) for r in
                                                       db.frames_for(desktop, view)])
        if not frames:
            continue
        asking = []
        rois: dict = {}
        for step in frames:
            seg = pose_segment_of(db, FrameKey(desktop, step, view))
            if seg not in rois:
                rois[seg] = db.bench_roi(desktop, view, seg)
            if key in needs_geom(instances, state_of(db, tax, desktop, step, cache), tax,
                                 bench_roi=rois[seg]):
                asking.append(step)
        # the card of every frame as the reverse walk shows it: diffed against
        # the next frame (the start frame against nothing -- "draw everything")
        carded = []
        for i, step in enumerate(frames):
            neighbour = frames[i + 1] if i + 1 < len(frames) else None
            card = task_card_for(db, tax, desktop, view, step, neighbour)
            if any(r.get("instance") == key for r in card):
                carded.append(step)
        views[view] = {"frames": len(frames), "asking": len(asking),
                       "asking_steps": f"{asking[0]}-{asking[-1]}" if asking else "",
                       "card": carded,
                       "frozen": len(db.frozen_steps_in(desktop, view, frames))}
    out["views"] = views
    return out


def show(log, title: str, facts: dict) -> None:
    log(f"  {title}:")
    for name in ("absent", "state_first", "state_last", "actions", "manual_events",
                 "derived_events", "warnings"):
        if name in facts:
            log(f"    {name}: {facts[name]}")
    for view, v in (facts.get("views") or {}).items():
        why = (" -- with no shape anywhere each is a missing_shape (the card's "
               "「这一帧还缺形状」, Space refused)" if v["asking"] else "")
        log(f"    {view}: {v['asking']}/{v['frames']} frames ask for its geometry"
            f"{' (' + v['asking_steps'] + ')' if v['asking_steps'] else ''}{why}; "
            f"card change rows at {v['card'] or 'no frame'}; frozen frames {v['frozen']}")


def utf8_streams() -> None:
    """Print Chinese whatever the console or pipe is (U34 round 2, N1).

    Redirected or piped, Windows gives Python a cp1252 ``stdout``, and the first
    line naming 「这台没有这个零件」 raised ``UnicodeEncodeError`` -- before the
    plan, but after the lock and the dry-run copy were made.
    """
    for stream in (sys.stdout, sys.stderr):
        if str(getattr(stream, "encoding", "") or "").lower().replace("-", "") == "utf8":
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass   # not a text stream that can be reconfigured: leave it


def main(argv: Optional[list[str]] = None) -> int:
    utf8_streams()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--desktop", type=int, required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--apply", action="store_true", help="write (default: a dry run on a copy)")
    ap.add_argument("--present", action="store_true",
                    help="the inverse, 「其实有」: take a part said absent back")
    ap.add_argument("--db", default=LIVE)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--paths", default=str(ROOT / "configs" / "paths.yaml"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    mode = "apply" if args.apply else "dryrun"
    logf = open(out / f"absent_{mode}_D{args.desktop:02d}_{stamp}.log", "w", encoding="utf-8")

    def log(text: str = "") -> None:
        print(text, flush=True)
        logf.write(text + "\n")
        logf.flush()

    src = os.path.abspath(args.db)
    is_live = os.path.normcase(src) == os.path.normcase(os.path.abspath(LIVE))
    if args.apply and is_live and app_running():
        log("the annotator app is running (or psutil is missing, so it cannot be told): "
            "close it first -- nothing touched")
        return 2
    if args.apply:
        target = src
    else:
        target = str(out / f"dryrun_{stamp}.sqlite")
        snapshot(src, target)
        log(f"DRY RUN: carried out on a copy, {target}")
    from tda.core.db_backup import (
        acquire_lock_file,
        lock_path_for,
        read_lock_file,
        release_lock_file,
    )
    # somebody else's abandoned lock (older than the TTL) is taken over by
    # acquire_lock_file; it is put back as it was afterwards (U34 round 2, N6).
    # A lock in this script's own name -- one a killed run left behind -- is
    # not put back: it would lock the annotator out for the TTL (R2-N1)
    lock = lock_path_for(target)
    held = read_lock_file(target)
    prior = (lock.read_bytes() if lock.exists()
             and (held or {}).get("annotator") != ANNOTATOR else None)
    try:
        acquire_lock_file(target, ANNOTATOR)
    except RuntimeError as exc:
        log(f"locked: {exc} -- nothing touched")
        return 2
    db = None
    try:
        config = P.load_paths(args.paths)
        db = Db(target)
        tax = load_taxonomy()
        d, key = int(args.desktop), str(args.instance)
        before = describe(db, tax, d, key)
        log(f"D{d:02d} {key} ({'其实有' if args.present else '这台没有这个零件'})")
        show(log, "before", before)
        # what else still names it (U34 round 2, N5): a 「被线挡住」 reason refuses
        # (the plan below says so), the rest is listed and stays
        found = AB.residues(db, d, key)
        if not args.present:
            lines = AB.residue_lines(found)
            log("  still naming it (kept as they are): " + ("none" if not lines else ""))
            for line in lines:
                log(f"    - {line}")
        try:
            plan = (AB.plan_present if args.present else AB.plan_absent)(db, tax, d, key)
        except AB.AbsentError as refused:
            log(f"  refused: {refused} -- nothing written")
            return 1
        if args.apply and is_live:
            from tda.cli_common import safety_backup
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                ok = safety_backup(config, db, "absent", "U34")
            for line in said.getvalue().splitlines():
                log(line)
            if not ok:
                return 1
        write = AB.mark_present if args.present else AB.mark_absent
        op_id, patch = write(db, tax, d, key, ANNOTATOR)
        after = describe(db, tax, d, key)
        log(f"  wrote op {op_id}: step-0 events {len(plan.before)} -> {len(patch.after)}; "
            f"frozen frames queued for a re-check: "
            f"{ {v: len(s) for v, s in patch.rechecks.items()} or 'none'}")
        show(log, "after", after)
        summary = {"desktop": d, "instance": key, "present": args.present, "apply": args.apply,
                   "target": target, "op": op_id, "rechecks": patch.rechecks,
                   "residues": found, "before": before, "after": after}
        path = out / f"absent_summary_{mode}_D{d:02d}_{stamp}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"summary -> {path}")
        if args.apply:
            # the compiled rows of the part are a cache the next refresh drops
            # (U34 round 2, N7): check recompiles every frame of the four views
            log(f"run: python -m tda.cli check --desktop {d} --views scan,oak1,oak2,rs")
    finally:
        if db is not None:
            db.close()
        if prior is None:
            release_lock_file(target)
        else:
            lock.write_bytes(prior)
        logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
