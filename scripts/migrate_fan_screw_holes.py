"""Controller (task W3e): the fan screw holes -- keep the hole pixels, take the fan screws out.

The user, 2026-10-31: 「D07 的风扇部分们就是 CPU 风扇和他的 5 个螺丝，是拆掉的，我标出来的一开始是
螺丝孔，然后随着标注的推进，把螺丝装紧进去」 and 「之前我标的部分可以变成螺丝孔？那些螺丝孔也是需要的，
不需要删掉」. On the seven heatsink + fan desktops D07 D08 D11 D14 D16 D18 D19 the importer made
the fan's five screws ``captive`` (an ``unscrew`` leaves them ``loosened`` in the fan), so the
annotator drew, under each screw's label, the screw **hole** on every frame after the screw
was out. In fact the screws come out at their own ``unscrew`` step (D03 already says so).
This script makes the truth say so and keeps the hole pixels, relabelled to the new class
``screw_hole`` (``configs/taxonomy.yaml``).

Per desktop, per fan screw ``s`` (a ``screw`` that fastens the desktop's ``cpu_cooler`` of kind
``fan`` and is still ``captive``), everything through the app's own paths:

1. ``screw_hole.NN`` (NN = the screw's number) is created with
   :func:`tda.core.extra.write_batch` (one ``add_extra_instances`` op): ``attrs.of = s``,
   ``added_by: controller``, ``reason``; ``parent`` = the screw's parent (the fan),
   ``attached`` -- so the hole leaves the chassis with the fan, where the user's masks stop.
   The stored automatic events are recompiled (the spec-3.3 cascade takes the hole out with
   the fan).
2. The screw's **latest-anchored** keyframe of each chain that reaches its hole frames (the
   hole, drawn first in the reverse walk) is copied to the hole -- same view, pose segment,
   anchor, placement and pixels, ``source = "w3e:hole"``. The screw keeps all its keyframes:
   its earlier "screw drawn in" keyframe is its shape before the unscrew; where the hole
   keyframe is its only one, it keeps using it there (the same footprint).
   Where the screw has such an earlier keyframe and the hole shape reaches outside it, the hole
   gets one more keyframe at that anchor: the hole shape **clipped to the screw head**
   (``source = "w3e:hole_under_head"``). Without it the hole's few pixels outside the head
   would show above the fan on the frames before the unscrew and take them from the fan's
   confirmed rows; with it the hole is fully covered there and no other row changes.
   The hole enters every layer order it has a keyframe in **directly below its screw**
   (:func:`tda.core.zorder_floor.entered`, the default the app now uses for the class), so
   once the screw is out the hole has exactly the pixels the screw had.
   Every single-frame override (「只改这一帧」) of the screw at a step >= its unscrew is moved
   to the hole (D11 ``.01-.03 @15``, D16 ``.01-.04 @13``): per-frame hole pixels.
3. :func:`tda.ui.session_captive.commit_captive_fix` ``(…, s, UNSCREWED, "controller")`` at the
   unscrew step on the scanner view -- the ``Shift+T`` op (``captive_fix``), so a forced
   re-import replays it.
4. Every frozen frame of every view is queued and the queues drained
   (:meth:`~tda.core.truth.TruthService.run_pending_rechecks`); every frame whose cache held a
   row of a fan screw is then refreshed (:meth:`~tda.core.truth.TruthService.refresh_range`,
   digest-guarded), so no stale cache row of a screw that is gone is left (the auto rows of
   oak1 / oak2 / rs) and those frames gain the hole's; a frame nobody ever compiled stays
   uncompiled. Then exactly two
   kinds of conflict this run opened are resolved ``accept_new``: (a) a fan screw's frozen row
   at a step >= its unscrew with no geometry now (the row goes), (b) a ``screw_hole`` row a
   frame gained (old none, new present). Any other conflict stays open and is listed.
5. Every frame this run demoted (verified -> needs_review) with no open conflict is re-signed
   (:meth:`~tda.core.truth.TruthService.verify_frame` as ``controller``); refusals are listed.
6. **Verified and printed** against the rows read before anything was written: every new hole
   row at a step >= the unscrew equals the screw's old row byte for byte (``visible_rle_json``:
   the user's pixels, relabelled); on the frames before the unscrew every hole row shows 0 px;
   every row of every other instance (the fan screws' rows before their unscrew included) is
   unchanged; no open conflict, no pending re-check, every frame back to its review status; the
   heatsink's screws untouched; each fan screw's events ``fastened -> removed`` and
   ``in_chassis -> on_bench`` at its unscrew step. One ``note`` op per desktop records the run.

A desktop with no captive fan screw left (already migrated) is skipped.

Usage::

    python scripts/migrate_fan_screw_holes.py [--apply] [--db PATH] [--desktops 7,8]
                                              [--out DIR] [--paths configs/paths.yaml]

Without ``--apply`` it is a dry run: everything is carried out on a backup-API copy of ``--db``
in ``--out`` (``dryrun_<stamp>.sqlite``), never on ``--db`` itself. ``--db`` defaults to the
LIVE database; ``--apply`` on it refuses while the annotator app runs (or ``psutil`` is
missing) or holds the lock, probes ``BEGIN IMMEDIATE``, and first takes the app's own safety
backup (``backup_dir`` of ``paths.yaml``). Exit 0 when every check holds, 1 when one fails or
a desktop is refused, 2 when nothing was touched (app open, locked, busy). Every run writes
``<out>/fan_holes_<mode>_<stamp>.log`` and ``fan_holes_summary_<mode>_<stamp>.json``.

**If an ``--apply`` run fails part-way, restore the safety backup the log names; do not
re-run on top of it.**
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tda import pipeline as P  # noqa: E402
from tda.core import masks  # noqa: E402
from tda.core import rawroot  # noqa: E402
from tda.core.captive_fix import UNSCREWED, CaptiveFixError, unscrew_step  # noqa: E402
from tda.core.compiler import select_keyframe  # noqa: E402
from tda.core.db import Db  # noqa: E402
from tda.core.dbrows import now_iso  # noqa: E402
from tda.core.extra import (  # noqa: E402
    ADDED_AT,
    ADDED_BY,
    REASON_ATTR,
    ExtraBatch,
    recompile_events,
    write_batch,
)
from tda.core.model import (  # noqa: E402
    ALL_VIEWS,
    HOLE_OF_ATTR,
    SCREW_HOLE_CLASS,
    FrameKey,
    FrameOverride,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    VIEWS,
    ZOrderRec,
    is_provisional,
)
from tda.core.states import IN_CHASSIS, ON_BENCH, REMOVED  # noqa: E402
from tda.core.taxonomy import load_taxonomy  # noqa: E402
from tda.core.truth import TruthService  # noqa: E402
from tda.core.truth_resolve import ACCEPT_NEW, StaleConflictError  # noqa: E402
from tda.core.zorder_floor import entered  # noqa: E402

LIVE = r"D:\DataSet\annotations\tda.sqlite"
OUT_DIR = r"D:\DataSet\.cache\tmp\fan_screw_holes"
ANNOTATOR = "controller"
#: The seven heatsink + fan desktops whose fan screws are captive by the importer's guess.
DESKTOPS = (7, 8, 11, 14, 16, 18, 19)
#: ``shape_keyframe.source`` of the hole's own shape (a copy of the screw's latest keyframe)
SOURCE_HOLE = "w3e:hole"
#: ... and of the hole clipped to an earlier screw head (see the module docstring, step 2)
SOURCE_UNDER_HEAD = "w3e:hole_under_head"
#: ``attrs["reason"]`` of every hole
REASON = ("W3e: the fan screw's hole, drawn under the screw's label on the frames after "
          "the screw came out")
#: The ``compiled_mask`` columns a row is compared on (``input_hash`` moves with any input).
ROW_COLUMNS = ("visible_rle_json", "occlusion_ratio", "visibility", "placement", "status",
               "geom_type", "box_json", "verified_by", "verified_at")
CONFLICT_SCREW_GONE = "screw_gone"
CONFLICT_HOLE_GAINED = "hole_gained"
FASTENED = "fastened"
FAN_CLASS, FAN_KIND = "cpu_cooler", "fan"
MASK = "mask"

_ORDINAL = re.compile(r"\.(\d+)$")


class PlanError(ValueError):
    """A desktop the migration cannot do as described; nothing of it is written."""


# --------------------------------------------------------------------------- #
# the plan (pure reads)
# --------------------------------------------------------------------------- #
@dataclass
class HolePlan:
    """What one fan screw's hole gets."""

    screw: str
    hole: str
    unscrew: int
    parent: Optional[str]
    #: ``(keyframe to add, "hole" | "under_head", hole px outside the head or 0)``
    keyframes: list[tuple[ShapeKeyframe, str, int]] = field(default_factory=list)
    #: the screw's single-frame overrides at steps >= the unscrew, moved to the hole
    overrides: list[FrameOverride] = field(default_factory=list)

    def record(self, desktop: int, stamp: str) -> InstanceRec:
        return InstanceRec(
            key=self.hole, desktop=desktop, cls=SCREW_HOLE_CLASS,
            attrs={HOLE_OF_ATTR: self.screw, ADDED_BY: ANNOTATOR, ADDED_AT: stamp,
                   REASON_ATTR: REASON},
            parent=self.parent, attached=True, raw_names=[])


def _fans(instances: dict[str, InstanceRec]) -> set[str]:
    return {k for k, r in instances.items() if not is_provisional(k)
            and r.cls == FAN_CLASS and str((r.attrs or {}).get("kind") or "") == FAN_KIND}


def fan_screws(instances: dict[str, InstanceRec]) -> list[str]:
    """The captive screws that fasten the desktop's fan, sorted."""
    fans = _fans(instances)
    return sorted(k for k, r in instances.items()
                  if not is_provisional(k) and r.cls == "screw" and r.fastens in fans
                  and (r.attrs or {}).get("captive"))


def _views(db: Db, desktop: int) -> list[str]:
    return [view for view in ALL_VIEWS if db.frames_for(desktop, view)]


def _union(kf: ShapeKeyframe):
    out = None
    for part in kf.parts:
        if part.rle is None:
            continue
        mask = masks.decode_rle(part.rle).astype(bool)
        out = mask if out is None else (out | mask)
    return out


def _copy(kf: ShapeKeyframe, hole: str, anchor: int, parts: list[ShapePart],
          source: str) -> ShapeKeyframe:
    return replace(kf, id=None, instance=hole, anchor_step=int(anchor), parts=parts,
                   source=source, draft_id=None, version=1, edit_count=0, edit_time_ms=0)


def plan_desktop(db: Db, desktop: int) -> list[HolePlan]:
    """The holes of one desktop; ``[]`` when no captive fan screw is left.

    Raises :class:`PlanError` for anything the description above does not cover -- a
    hole key that exists, a screw never unscrewed, a hole shape of several parts, a hole
    clipped to nothing, a screw missing from its layer order, a pair override naming it.
    """
    instances = db.instances(desktop)
    screws = fan_screws(instances)
    if not screws:
        return []
    actions = db.actions(desktop)
    views = _views(db, desktop)
    plans: list[HolePlan] = []
    for s in screws:
        rec = instances[s]
        number = _ORDINAL.search(s)
        if number is None:
            raise PlanError(f"{s}: the key has no number")
        hole = f"{SCREW_HOLE_CLASS}.{number.group(1)}"
        if hole in instances:
            raise PlanError(f"{hole} exists already")
        at = unscrew_step(actions, s)
        if at is None:
            raise PlanError(f"{s}: the log never unscrews it")
        plan = HolePlan(screw=s, hole=hole, unscrew=int(at), parent=rec.parent)
        for view in views:
            chains: dict[tuple[int, str], list[ShapeKeyframe]] = {}
            for kf in db.keyframes(desktop, view, s):
                chains.setdefault((kf.pose_segment, kf.placement), []).append(kf)
            for (seg, placement), chain in sorted(chains.items()):
                last = max(kf.anchor_step for kf in chain)
                if last < at:
                    continue          # no frame of this chain shows the hole
                if placement != IN_CHASSIS or any(kf.geom_type != MASK for kf in chain):
                    raise PlanError(f"{s} {view}/{seg}: a {placement} or box chain reaches "
                                    f"its hole frames")
                latest = select_keyframe(chain, last)
                if len(latest.parts) != 1 or latest.parts[0].rle is None:
                    raise PlanError(f"{s} {view}/{seg}: the hole keyframe has "
                                    f"{len(latest.parts)} parts")
                part = latest.parts[0]
                plan.keyframes.append((_copy(latest, hole, last, [ShapePart(
                    part.name, dict(part.rle), part.box)], SOURCE_HOLE), "hole", 0))
                shape = masks.decode_rle(part.rle).astype(bool)
                for anchor in sorted({kf.anchor_step for kf in chain if kf.anchor_step < at}):
                    head_kf = select_keyframe(chain, anchor)
                    head = _union(head_kf)
                    if head is None:
                        raise PlanError(f"{s} {view}/{seg}@{anchor}: the screw shape is empty")
                    leak = int((shape & ~head).sum())
                    if not leak:
                        continue
                    clipped = shape & head
                    if not clipped.any():
                        raise PlanError(f"{s} {view}/{seg}@{anchor}: the hole lies wholly "
                                        f"outside the screw head")
                    plan.keyframes.append((_copy(latest, hole, anchor, [ShapePart(
                        part.name, masks.encode_rle_boxed(clipped))], SOURCE_UNDER_HEAD),
                        "under_head", leak))
                order = db.zorder(desktop, view, seg).order
                if not any(entry[0] == s for entry in order):
                    raise PlanError(f"{s} is not in the layer order of {view}/{seg}")
                if any(s in (po.above, po.below)
                       for po in db.pair_overrides(desktop, view, seg)):
                    raise PlanError(f"{s} {view}/{seg}: a pair override names it")
        for view, step in db.conn.execute(
                "SELECT view, step FROM frame_override WHERE desktop=? AND instance=? "
                "ORDER BY view, step", (int(desktop), s)).fetchall():
            if int(step) >= at:
                key = FrameKey(int(desktop), int(step), str(view))
                plan.overrides.append(db.frame_overrides(key)[s])
        plans.append(plan)
    return plans


# --------------------------------------------------------------------------- #
# reads for the checks
# --------------------------------------------------------------------------- #
def rows_of(db: Db, desktop: int) -> dict[tuple[int, str, str], tuple]:
    """``(step, view, instance) -> the compared columns`` of every compiled row."""
    cols = ", ".join(ROW_COLUMNS)
    return {(int(r[0]), str(r[1]), str(r[2])): tuple(r[3:]) for r in db.conn.execute(
        f"SELECT step, view, instance, {cols} FROM compiled_mask WHERE desktop=?",
        (int(desktop),)).fetchall()}


def statuses_of(db: Db, desktop: int) -> dict[tuple[str, int], str]:
    return {(str(v), int(s)): str(st) for v, s, st in db.conn.execute(
        "SELECT view, step, review_status FROM frame WHERE desktop=?", (int(desktop),))}


def _area(rle_json: Optional[str]) -> int:
    if not rle_json:
        return 0
    return int(masks.rle_area(json.loads(rle_json)))


# --------------------------------------------------------------------------- #
# one desktop
# --------------------------------------------------------------------------- #
def migrate_desktop(db: Db, tax, truth: TruthService, desktop: int,
                    log: Callable[[str], None]) -> dict:
    """Carry out steps 1-6 on one desktop; returns its numbers (``ok`` says whether every
    check held). ``{"skipped": ...}`` for a desktop with nothing left to do."""
    from tda.ui.session_api import SessionRefusal          # late: the UI package
    from tda.ui.session_captive import commit_captive_fix

    d = int(desktop)
    try:
        plans = plan_desktop(db, d)
    except PlanError as refused:
        log(f"D{d:02d}: refused -- {refused}; nothing written")
        return {"refused": str(refused), "ok": False}
    if not plans:
        log(f"D{d:02d}: no captive fan screw left (already migrated?) -- skipped")
        return {"skipped": "no captive fan screw", "ok": True}
    instances = db.instances(d)
    heatsink_screws = {k: r for k, r in instances.items()
                       if r.cls == "screw" and k not in {p.screw for p in plans}
                       and str(r.key).startswith("screw.cpu_cooler")}
    old_rows = rows_of(db, d)
    old_status = statuses_of(db, d)
    max_conflict = int(db.conn.execute("SELECT COALESCE(MAX(id), 0) FROM conflict")
                       .fetchone()[0])
    views = _views(db, d)
    at = {p.screw: p.unscrew for p in plans}
    holes = {p.hole: p for p in plans}
    out: dict = {"screws": {p.screw: p.unscrew for p in plans}, "holes": sorted(holes)}
    for p in plans:
        log(f"D{d:02d} {p.screw} -> {p.hole}: unscrew @{p.unscrew}, parent {p.parent}; "
            f"keyframes " + ", ".join(f"{kf.view}/{kf.pose_segment}@{kf.anchor_step} {what}"
                                      + (f" ({leak} px outside the head)" if leak else "")
                                      for kf, what, leak in p.keyframes)
            + f"; overrides moved {[(fo.frame.view, fo.frame.step) for fo in p.overrides]}")

    # 1-2. the holes, their shapes, their layers, their per-frame pixels -- one transaction
    stamp = now_iso()
    written_kfs: list[int] = []
    with db.transaction():
        records = [p.record(d, stamp) for p in plans]
        batch_op = write_batch(db, ExtraBatch(desktop=d, records=records), ANNOTATOR)
        table = db.instances(d)
        for p in plans:
            for kf, _what, _leak in p.keyframes:
                written_kfs.append(db.add_keyframe(kf))
            for fo in p.overrides:
                db.set_frame_override(FrameOverride(fo.frame, p.hole, fo.visible_rle,
                                                    fo.visibility, fo.occluder))
                db.delete_frame_override(fo.frame, p.screw)
        segments = sorted({(kf.view, kf.pose_segment) for p in plans
                           for kf, _w, _l in p.keyframes})
        for view, seg in segments:
            rec = db.zorder(d, view, seg)
            order = [tuple(e) for e in rec.order]
            for p in plans:
                names = [kf.parts[0].name for kf, _w, _l in p.keyframes
                         if (kf.view, kf.pose_segment) == (view, seg)]
                if names:
                    order = entered(order, p.hole, names[0], table)
            db.set_zorder(ZOrderRec(d, view, seg, order, version=rec.version + 1))
        recompile_events(db, d, tax)
    out["keyframes"] = Counter(what for p in plans for _kf, what, _l in p.keyframes)
    out["overrides_moved"] = sum(len(p.overrides) for p in plans)

    # 3. the flips, as Shift+T writes them
    flipped: list[str] = []
    for p in plans:
        view = "scan" if "scan" in views else views[0]
        try:
            commit_captive_fix(db, truth, FrameKey(d, p.unscrew, view), p.screw, UNSCREWED,
                               ANNOTATOR)
            flipped.append(p.screw)
        except (CaptiveFixError, SessionRefusal) as refused:
            log(f"  D{d:02d} {p.screw}: the flip was refused -- {refused}")
    out["flipped"] = len(flipped)

    # 4. every frozen frame re-checked, every frame refreshed, the two kinds closed
    rechecked: dict = {}
    for view in views:
        truth.queue_rechecks_for_view(d, view)
        tot = truth.run_pending_rechecks(d, view)
        rechecked[view] = {k: tot.get(k) for k in ("updated", "conflicts", "standing")}
    refreshed: dict = {}
    # the frames whose cache held a row of a fan screw: their stale rows go and the hole's
    # come in. A frame nobody ever compiled stays uncompiled (filling a cache is no part of
    # this run), and the components gallery has neither a hole nor a removed screw
    reach: dict[str, set[int]] = {}
    for step, view, inst in old_rows:
        if inst in at:
            reach.setdefault(view, set()).add(int(step))
    for view in sorted(reach):
        tot = truth.refresh_range(d, view, sorted(reach[view]))
        refreshed[view] = {k: tot.get(k) for k in ("updated", "conflicts", "standing")}
    closed: Counter = Counter()
    left: list = []
    for _round in range(3):
        stale = False
        left = []
        for c in truth.open_conflicts(d):
            if int(c["id"]) <= max_conflict:
                left.append((c["view"], c["step"], c["instance"], "open before the run"))
                continue
            kind = _kind(c, at, holes)
            if kind is None:
                left.append((c["view"], c["step"], c["instance"],
                             f"old {'none' if c['old_rle'] is None else 'present'}, "
                             f"new {'none' if c['new_rle'] is None else 'present'}"))
                continue
            try:
                truth.resolve_conflict(int(c["id"]), ACCEPT_NEW, ANNOTATOR)
                closed[kind] += 1
            except StaleConflictError:
                stale = True
        if not stale:
            break
    out["rechecks"], out["refresh"] = rechecked, refreshed
    out["closed"] = dict(closed)
    out["left_open"] = left

    # 5. re-sign what this run demoted
    resigned: list = []
    refused: list = []
    still_open = {(c["view"], int(c["step"])) for c in truth.open_conflicts(d)}
    now_status = statuses_of(db, d)
    for (view, step), status in sorted(old_status.items()):
        if status != "verified" or now_status.get((view, step)) == "verified":
            continue
        if (view, step) in still_open:
            refused.append((view, step, "open conflict"))
            continue
        try:
            truth.verify_frame(FrameKey(d, step, view), ANNOTATOR)
            resigned.append((view, step))
        except ValueError as exc:
            refused.append((view, step, str(exc)[:200]))
    out["resigned"], out["resign_refused"] = len(resigned), refused

    db.log_op(d, "-", "note", {
        "note": "controller W3e 2026-10-31: the fan on top of the heatsink has removable "
                "screws (user: D07「CPU 风扇和他的 5 个螺丝，是拆掉的」); the hole masks drawn "
                "under each screw's label after its unscrew are relabelled screw_hole.NN "
                "(「那些螺丝孔也是需要的，不需要删掉」), the five fan screws captive -> "
                "unscrewed at their unscrew step; the heatsink's screws stay captive",
        "holes": {p.hole: p.screw for p in plans}, "extra_op": batch_op,
        "keyframes": written_kfs,
        "overrides_moved": [[fo.frame.view, fo.frame.step, p.screw, p.hole]
                            for p in plans for fo in p.overrides],
        "flipped": flipped}, {}, ANNOTATOR)

    # 6. the checks
    out.update(check_desktop(db, d, plans, views, old_rows, old_status, heatsink_screws))
    ok = (out["flipped"] == len(plans) and not left and not refused
          and out["pixel_mismatch"] == 0 and out["hole_rows_missing"] == 0
          and out["hole_px_before_unscrew"] == 0 and out["other_rows_changed"] == 0
          and out["unexpected_rows"] == 0 and out["screw_rows_left"] == 0
          and out["open_conflicts"] == 0 and out["pending_rechecks"] == 0
          and not out["status_changed"] and not out["heatsink_changed"]
          and not out["event_problems"])
    out["ok"] = bool(ok)
    log(f"D{d:02d}: holes {len(plans)}, keyframes {dict(out['keyframes'])}, overrides moved "
        f"{out['overrides_moved']}, flipped {out['flipped']}; closed {out['closed']}, other "
        f"open {len(left)}; re-signed {len(resigned)} frame(s), refused {len(refused)}")
    log(f"  pixel identity: {out['pixel_checked']} hole rows at/after the unscrew compared, "
        f"{out['pixel_mismatch']} mismatch(es) ({out['pixel_checked_scan']} on scan); "
        f"{out['hole_rows_missing']} missing; hole px before the unscrew "
        f"{out['hole_px_before_unscrew']} over {out['hole_rows_before']} rows")
    log(f"  other rows changed {out['other_rows_changed']}, unexpected new rows "
        f"{out['unexpected_rows']}, fan-screw rows left at/after the unscrew "
        f"{out['screw_rows_left']}; open conflicts {out['open_conflicts']}, pending "
        f"rechecks {out['pending_rechecks']}; frames off their status "
        f"{len(out['status_changed'])}; heatsink screws changed {out['heatsink_changed']}; "
        f"event problems {out['event_problems']} -> {'OK' if ok else 'FAILED'}")
    for item in left[:12]:
        log(f"    left open: {item}")
    for item in refused[:12]:
        log(f"    re-sign refused: {item}")
    for item in out["examples"][:12]:
        log(f"    {item}")
    return out


def _kind(conflict: dict, at: dict[str, int], holes: dict[str, HolePlan]) -> Optional[str]:
    """Which of the two expected kinds a conflict is, or ``None``."""
    inst, step = str(conflict["instance"]), int(conflict["step"])
    old, new = conflict.get("old_rle"), conflict.get("new_rle")
    if inst in at and step >= at[inst] and new is None and old is not None:
        return CONFLICT_SCREW_GONE
    if inst in holes and old is None and new is not None:
        return CONFLICT_HOLE_GAINED
    return None


def check_desktop(db: Db, d: int, plans: list[HolePlan], views: list[str],
                  old_rows: dict, old_status: dict, heatsink_screws: dict) -> dict:
    """Step 6: the numbers the run is judged by (pure reads)."""
    new_rows = rows_of(db, d)
    at = {p.screw: p.unscrew for p in plans}
    hole_of = {p.screw: p.hole for p in plans}
    holes = set(hole_of.values())
    examples: list[str] = []
    checked = checked_scan = mismatch = missing = 0
    for (step, view, inst), old in sorted(old_rows.items()):
        if inst not in at or step < at[inst]:
            continue
        new = new_rows.get((step, view, hole_of[inst]))
        if new is None:
            missing += 1
            examples.append(f"no hole row: {view} {step} {hole_of[inst]}")
            continue
        checked += 1
        checked_scan += int(view == "scan")
        if new[0] != old[0]:
            mismatch += 1
            examples.append(f"pixels differ: {view} {step} {inst} -> {hole_of[inst]} "
                            f"({_area(old[0])} px -> {_area(new[0])} px)")
    before_px = before_rows = 0
    for (step, view, inst), new in new_rows.items():
        if inst not in holes:
            continue
        screw = next(p.screw for p in plans if p.hole == inst)
        if step < at[screw]:
            before_rows += 1
            before_px += _area(new[0])
    changed = unexpected = screw_left = 0
    for key, old in old_rows.items():
        step, view, inst = key
        if inst in at and step >= at[inst]:
            continue
        if new_rows.get(key) != old:
            changed += 1
            if len(examples) < 40:
                examples.append(f"row changed: {view} {step} {inst}")
    for key in new_rows:
        step, view, inst = key
        if key in old_rows:
            if inst in at and step >= at[inst]:
                screw_left += 1
                examples.append(f"screw row left: {view} {step} {inst}")
            continue
        if inst not in holes:
            unexpected += 1
            examples.append(f"new row: {view} {step} {inst}")
    now_status = statuses_of(db, d)
    status_changed = sorted(k for k, v in old_status.items() if now_status.get(k) != v)
    open_conflicts = len(db.conflicts(d, open_only=True))
    pending = sum(len(db.rechecks(d, view)) for view in views)
    instances = db.instances(d)
    heatsink_changed = sorted(k for k, r in heatsink_screws.items() if instances.get(k) != r)
    problems: list[str] = []
    events = [e for e in db.events(d)]
    for p in plans:
        mine = [e for e in events if e.target == p.screw and e.step == p.unscrew]
        if not any(e.attr == "state" and e.old == FASTENED and e.new == REMOVED for e in mine):
            problems.append(f"{p.screw}: no fastened -> removed at {p.unscrew}")
        if not any(e.attr == "placement" and e.old == IN_CHASSIS and e.new == ON_BENCH
                   for e in mine):
            problems.append(f"{p.screw}: no in_chassis -> on_bench at {p.unscrew}")
        if (instances[p.screw].attrs or {}).get("captive"):
            problems.append(f"{p.screw}: still captive")
    hole_events = sorted({(e.target, e.step, e.attr, e.new) for e in events if e.target in holes})
    return {"pixel_checked": checked, "pixel_checked_scan": checked_scan,
            "pixel_mismatch": mismatch, "hole_rows_missing": missing,
            "hole_px_before_unscrew": before_px, "hole_rows_before": before_rows,
            "other_rows_changed": changed, "unexpected_rows": unexpected,
            "screw_rows_left": screw_left, "status_changed": status_changed,
            "open_conflicts": open_conflicts, "pending_rechecks": pending,
            "heatsink_changed": heatsink_changed, "event_problems": problems,
            "hole_events": [list(e) for e in hole_events], "examples": examples}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
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
    """Is ``python -m tda.cli app`` running? (argv tokens; fail-closed without psutil)."""
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


def parse_desktops(text: Optional[str]) -> list[int]:
    if not text:
        return list(DESKTOPS)
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: Optional[list[str]] = None) -> int:
    _utf8_streams()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write (default: a dry run on a copy)")
    ap.add_argument("--db", default=LIVE)
    ap.add_argument("--desktops", default=None, help="default: 7,8,11,14,16,18,19")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--paths", default=str(ROOT / "configs" / "paths.yaml"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    mode = "apply" if args.apply else "dryrun"
    logf = open(out / f"fan_holes_{mode}_{stamp}.log", "w", encoding="utf-8")

    def log(text: str = "") -> None:
        print(text, flush=True)
        logf.write(text + "\n")
        logf.flush()

    src = os.path.abspath(args.db)
    is_live = os.path.normcase(src) == os.path.normcase(os.path.abspath(LIVE))
    if args.apply and is_live and app_running():
        log("the annotator app is running (or psutil is missing, so it cannot be told): "
            "close it first -- nothing touched")
        logf.close()
        return 2
    if args.apply:
        target = src
    else:
        target = str(out / f"dryrun_{stamp}.sqlite")
        snapshot(src, target)
        log(f"DRY RUN: carried out on a copy, {target}")
    from tda.core.db_backup import acquire_lock_file, release_lock_file
    try:
        acquire_lock_file(target, ANNOTATOR)
    except RuntimeError as exc:
        log(f"locked: {exc} -- nothing touched")
        logf.close()
        return 2
    db = None
    code = 0
    try:
        config = P.load_paths(args.paths)
        rawroot.configure(config)
        db = P.open_db(dict(config, db_path=target), target)
        try:
            db.conn.execute("BEGIN IMMEDIATE")
            db.conn.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            log(f"DB busy ({exc}); is the annotator open? nothing touched")
            return 2
        if args.apply and is_live:
            from tda.cli_common import safety_backup
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                ok = safety_backup(config, db, "fan_holes", "W3e")
            for line in said.getvalue().splitlines():
                log(line)
            if not ok:
                return 1
            log("  on ANY error from here on: restore this backup over the live database "
                "and do not re-run on top of the half-done run")
        tax = load_taxonomy()
        truth = TruthService(db, tax, cache_dir=config.get("cache_dir"))
        summary: dict = {"target": target, "apply": args.apply, "stamp": stamp,
                         "desktops": {}}
        for d in parse_desktops(args.desktops):
            t0 = time.time()
            res = migrate_desktop(db, tax, truth, d, log)
            res["seconds"] = round(time.time() - t0, 1)
            summary["desktops"][d] = res
            if not res.get("ok"):
                code = 1
        path = out / f"fan_holes_summary_{mode}_{stamp}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")
        log(f"summary -> {path}")
        log("all checks hold" if code == 0 else "SOME CHECK FAILED -- see above")
    finally:
        if db is not None:
            db.close()
        release_lock_file(target)
        logf.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
