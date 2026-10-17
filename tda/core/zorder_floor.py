"""The chassis and the motherboard are floor layers (task U17).

A newly drawn instance goes on top of its ``(view, pose segment)`` layer order
(spec 4.2) -- except the two parts everything else lies on.  The reverse walk
starts on the last frame, where the annotator draws the chassis, the
motherboard, the PSU, its latch and a cover in whatever order they like; on
D2 that stored ``chassis, psu.01, psu_latch.01, cover.01, motherboard.01``.
The motherboard's amodal shape runs under the chassis-edge cover plate and the
PSU latch, so painting it above them handed the plate's upper third and a
corner of the latch to the motherboard's *visible* mask, on ~45 frames of each
of D2, D4 and D5.  So:

* :func:`floor_index` -- where a newly drawn instance enters the order: a
  ``chassis`` at the very bottom, a ``motherboard`` directly above the lowest
  chassis (its whole block, when the chassis has several parts) or at the
  bottom when there is none, anything else on top.  The class is the instance
  record's (``InstanceRec.cls``), never the key's prefix.  Only
  :func:`tda.ui.session_edit._append_to_zorder` asks: a move the annotator
  makes (``Ctrl+Up/Down``, ``L``) is theirs, and the compiler paints whatever
  order is stored.
* :func:`floor_problems` -- ``cli check``'s ``zorder_floor:<instance>``.
  Stored orders are **reported, never rewritten** (U6: no silent reorder): a
  chassis above any part that is not a floor layer, or a motherboard above
  such a part that it overlaps in pixels on at least one frame of the segment.
  An order that is merely unusual -- the board above a part it never touches --
  paints the same pixels either way and is not a problem; neither is a pair the
  annotator settled with ``L`` (a pair override, either way round), whose
  pixels follow the override and not the order.

Qt-free, so ``cli check`` can import it.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from tda.core.db import Db
from tda.core.model import FrameKey
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import InputCache, pose_segment_of

__all__ = ["CHASSIS", "FLOOR_CLASSES", "MOTHERBOARD", "ZORDER_FLOOR", "floor_index",
           "floor_inversions", "floor_problems"]

#: The two floor classes (``configs/taxonomy.yaml``).
CHASSIS = "chassis"
MOTHERBOARD = "motherboard"
FLOOR_CLASSES = frozenset({CHASSIS, MOTHERBOARD})
#: The problem ``cli check`` reports for a floor layer stored too high.
ZORDER_FLOOR = "zorder_floor"
#: How many names one problem line spells out before "+N more".
_NAMED = 5

Entry = tuple[str, str]


def floor_index(order: Sequence[Entry], cls: Optional[str],
                cls_of: Mapping[str, str]) -> int:
    """Where a newly drawn instance of class ``cls`` enters ``order`` (bottom first).

    ``cls_of`` maps instance keys to their taxonomy class.  A ``chassis`` goes
    in at 0; a ``motherboard`` right after the block of the lowest chassis
    entry (the run of that chassis instance's parts starting there), or at 0
    without one; every other class -- and an instance with no record -- at
    ``len(order)``, on top, as before.
    """
    if cls == CHASSIS:
        return 0
    if cls != MOTHERBOARD:
        return len(order)
    lowest = next((i for i, entry in enumerate(order) if cls_of.get(entry[0]) == CHASSIS),
                  None)
    if lowest is None:
        return 0
    end = lowest
    while end < len(order) and order[end][0] == order[lowest][0]:
        end += 1
    return end


def floor_inversions(order: Sequence[Entry],
                     cls_of: Mapping[str, str]) -> dict[str, list[str]]:
    """``floor instance -> non-floor instances it sits above``, from the order alone.

    "Sits above" is per entry: some entry of the floor instance is higher than
    some entry of the other one (a multi-part instance counts by its highest
    and lowest parts).  Keys with no instance record paint nothing and are left
    out.  Both lists come in order position, bottom first.
    """
    top: dict[str, int] = {}
    bottom: dict[str, int] = {}
    for index, entry in enumerate(order):
        top[entry[0]] = index
        bottom.setdefault(entry[0], index)
    out: dict[str, list[str]] = {}
    for floor in sorted(bottom, key=bottom.get):
        if cls_of.get(floor) not in FLOOR_CLASSES:
            continue
        under = [inst for inst in sorted(bottom, key=bottom.get)
                 if inst in cls_of and cls_of[inst] not in FLOOR_CLASSES
                 and bottom[inst] < top[floor]]
        if under:
            out[floor] = under
    return out


def _named(names: Sequence[str]) -> str:
    shown = ", ".join(names[:_NAMED])
    return shown if len(names) <= _NAMED else f"{shown} +{len(names) - _NAMED} more"


def _segment_steps(db: Db, desktop: int, view: str,
                   cache: InputCache) -> dict[int, list[int]]:
    """``pose segment -> its frames``, last step first: what ``check`` compiled."""
    out: dict[int, list[int]] = {}
    for row in db.frames_for(desktop, view):
        if row.get("missing"):
            continue
        step = int(row["step"])
        out.setdefault(pose_segment_of(db, FrameKey(desktop, step, view), cache),
                       []).append(step)
    return {seg: sorted(steps, reverse=True) for seg, steps in out.items()}


def _touch(one, two) -> bool:
    """Do two compiled instances' amodal masks share a pixel?  Inside their windows."""
    h, w = one.amodal.shape
    x0, y0, x1, y1 = 0, 0, w, h
    for window in (one.amodal_window, two.amodal_window):
        if window is not None:
            x0, y0 = max(x0, int(window[0])), max(y0, int(window[1]))
            x1, y1 = min(x1, int(window[2])), min(y1, int(window[3]))
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(np.any(one.amodal[y0:y1, x0:x1] & two.amodal[y0:y1, x0:x1]))


def _overlaps(truth, desktop: int, view: str, steps: Iterable[int],
              pairs: set[tuple[str, str]], cache: InputCache) -> dict[tuple[str, str], int]:
    """``(board, part) -> a step where their amodal masks overlap``, for those that do.

    Frames are compiled one at a time (:meth:`TruthService.compile`) and the
    walk stops as soon as every pair has its frame.  A pair counts only where
    both are mask layers of the same placement group and no pair override
    between them held there.
    """
    pending = set(pairs)
    hits: dict[tuple[str, str], int] = {}
    for step in steps:
        if not pending:
            break
        frame = truth.compile(FrameKey(desktop, int(step), view), cache)
        for board, part in sorted(pending):
            one, two = frame.instances.get(board), frame.instances.get(part)
            if one is None or two is None or one.amodal is None or two.amodal is None:
                continue
            if one.placement != two.placement:
                continue
            held = frame.pairs.get(one.placement, ())
            if (board, part) in held or (part, board) in held:
                continue   # the annotator's L decides these pixels, not the order
            if _touch(one, two):
                hits[(board, part)] = int(step)
                pending.discard((board, part))
    return hits


def floor_problems(db: Db, tax: Taxonomy, desktop: int, view: str,
                   truth=None) -> list[str]:
    """``zorder_floor:<instance> (segment S, above ...)`` per offending floor layer.

    One line per floor instance and pose segment: a chassis names every
    non-floor part under it, a motherboard the ones it overlaps, each with the
    step of an overlapping frame (``part@step``).  ``truth`` is the
    :class:`~tda.core.truth.TruthService` to compile with (``cli check`` passes
    its own); only segments whose order has a motherboard above a non-floor
    part are compiled at all.
    """
    segs = [int(r[0]) for r in db.conn.execute(
        "SELECT pose_segment FROM zorder WHERE desktop=? AND view=? ORDER BY pose_segment",
        (int(desktop), str(view))).fetchall()]
    if not segs:
        return []
    cls_of = {key: rec.cls for key, rec in db.instances(int(desktop)).items()}
    cache = InputCache()
    steps_of: Optional[dict[int, list[int]]] = None
    out: list[str] = []
    for seg in segs:
        found = floor_inversions(db.zorder(int(desktop), str(view), seg).order, cls_of)
        boards = {(floor, part) for floor, under in found.items()
                  if cls_of[floor] == MOTHERBOARD for part in under}
        hits: dict[tuple[str, str], int] = {}
        if boards:
            if steps_of is None:
                steps_of = _segment_steps(db, int(desktop), str(view), cache)
            if truth is None:
                from tda.core.truth import TruthService
                truth = TruthService(db, tax)
            hits = _overlaps(truth, int(desktop), str(view), steps_of.get(seg, []), boards,
                             cache)
        for floor, under in found.items():
            if cls_of[floor] == CHASSIS:
                named = list(under)
            else:
                named = [f"{part}@{hits[(floor, part)]}" for part in under
                         if (floor, part) in hits]
            if named:
                out.append(f"{ZORDER_FLOOR}:{floor} (segment {seg}, above {_named(named)})")
    return out
