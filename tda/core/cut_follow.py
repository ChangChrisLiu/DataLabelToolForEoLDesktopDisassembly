"""A cut keyframe follows its state boundary (task U11).

U8 round 3 made a shape commit stop at an open state boundary: before ``Enter``
or ``Ctrl+K`` writes, the version in force is cut at the highest open ✂ frame
below the current one, with a copy of the *pre-edit* shape -- the **cut
keyframe** -- so the edit reaches the current state run only.  The copy keeps
the frames of the other state, below the run.

On D2/scan the annotator then walked the run of an opened cover down one frame
at a time with ``T``.  Every ``T`` moved the run's lower edge one frame down;
the cut stayed where it was, so the frame that had just joined the run was
still covered by the copy -- the old shape -- and was redrawn, at every frame
from 42 down to 37.  The fix is the obvious one: **when an assertion moves a
part's state boundary, a cut keyframe anchored at that boundary moves with it**,
in every view, in the same op.

* A cut keyframe is marked ``ShapeKeyframe.source == "cut"`` (:data:`CUT`).
  The mark lives as long as its pixels are the copy: a commit that re-traces
  it with other pixels stamps the annotator's own source over it
  (:func:`tda.ui.session_edit._commit_shape`), and from then on it is a shape
  like any other and nothing here touches it.
* :func:`mask_states` is the part's state at every step where it is a mask in
  the chassis -- the only steps a ✂ row, and so a cut, can stand on.
* :func:`plan_cut_moves` compares the states before and after an assertion and
  says what becomes of each cut keyframe of the part whose boundary moved:

  - **extended** (the run grew down over the cut's frame ``b``): the cut goes
    down to the new lower edge ``b'``, so the run's version covers the frames
    that joined it and the far side keeps the copy;
  - **shrunk** (the run's bottom frame went over to the far side): the cut goes
    up to the new lower edge, the same way;
  - **gone** -- the run reaches the first frame of the cut's pose segment, or
    swallowed a neighbouring run that has versions of its own: every frame of
    the cut joined the run, so the cut is dropped when its shape is still in
    another keyframe of the chain (nothing is lost), and kept -- and said -- when
    it is the only copy of that shape left;
  - a run that vanished altogether, or a version of the annotator's own in the
    way, leaves the cut where it is: it still covers far-side frames only.

  Keyframes that are not cut copies are never touched.

The module is Qt-free and writes nothing: the session applies the moves inside
the assertion's own op (:mod:`tda.ui.session_state`), so one ``Ctrl+Z`` takes
back the state and the moves together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tda.core import masks
from tda.core.db import Db
from tda.core.model import FrameKey, Placement, ShapeKeyframe, ShapePart
from tda.core.split_rows import reverse_walk
from tda.core.states import needs_geom
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import InputCache, instances_of, pose_segment_of, state_of

__all__ = ["CUT", "CutMove", "cut_views", "is_cut", "mask_states", "plan_cut_moves",
           "same_parts", "same_shape"]

#: ``ShapeKeyframe.source`` of a cut keyframe: the pre-edit shape copied to the
#: far side of a state boundary by a commit that stopped there (U8 round 3).
CUT = "cut"
IN_CHASSIS = Placement.IN_CHASSIS.value
_MASK = "mask"


def is_cut(kf: Optional[ShapeKeyframe]) -> bool:
    """Is ``kf`` a cut copy whose pixels nobody has redrawn yet?"""
    return kf is not None and kf.source == CUT


def _part_mask(rle: Optional[dict]) -> Optional[np.ndarray]:
    return None if not rle else masks.decode_rle(rle)


def same_shape(one: ShapeKeyframe, two: ShapeKeyframe) -> bool:
    """Do two keyframes carry the same geometry (parts, boxes, pixels)?"""
    return one.geom_type == two.geom_type and same_parts(one.parts, two.parts)


def same_parts(one: list[ShapePart], two: list[ShapePart]) -> bool:
    """Are two part lists the same geometry?

    Compared as stored first -- a cut is a verbatim copy, so the common answer
    costs nothing -- and decoded only when the encodings differ, because two
    encoders can spell the same pixels differently.
    """
    if len(one) != len(two):
        return False
    for a, b in zip(one, two):
        if a.name != b.name or (a.box is None) != (b.box is None):
            return False
        if a.box is not None and tuple(a.box) != tuple(b.box):
            return False
        if a.rle == b.rle:
            continue
        ma, mb = _part_mask(a.rle), _part_mask(b.rle)
        if ma is None or mb is None or ma.shape != mb.shape or not np.array_equal(ma, mb):
            return False
    return True


def mask_states(db: Db, tax: Taxonomy, desktop: int, target: str,
                cache: Optional[InputCache] = None) -> dict[int, Optional[str]]:
    """``step -> state`` of ``target`` at every step of the table.

    The state where the part is a mask in the chassis, ``None`` everywhere else
    (not there yet, on the bench, gone, a state that is not drawn): a ✂ row --
    and therefore a cut -- only ever stands between two mask frames. Pass a
    fresh ``cache`` (or none) on each side of a write: an
    :class:`~tda.core.truth_inputs.InputCache` keeps the event log it read.
    """
    cache = cache if cache is not None else InputCache()
    instances = instances_of(db, desktop, cache)
    out: dict[int, Optional[str]] = {}
    for step in sorted({int(r.step) for r in db.steps(desktop)}):
        fs = state_of(db, tax, desktop, step, cache)
        held = fs.get(target)
        if (held is None or held.placement != IN_CHASSIS
                or needs_geom(instances, fs, tax).get(target) != _MASK):
            out[step] = None
        else:
            out[step] = held.state
    return out


def cut_views(db: Db, desktop: int, target: str) -> list[str]:
    """The views where ``target`` has a cut keyframe, sorted."""
    rows = db.conn.execute(
        "SELECT DISTINCT view FROM shape_keyframe WHERE desktop=? AND instance=? "
        "AND source=? ORDER BY view", (int(desktop), str(target), CUT)).fetchall()
    return [str(r["view"]) for r in rows]


@dataclass
class CutMove:
    """What becomes of one cut keyframe of one view.

    ``new`` is its anchor afterwards, ``None`` when it goes; ``kept`` says the
    boundary went but the cut stays, because its shape is in no other keyframe
    of the chain. ``reach`` are the view's steps whose selected keyframe
    changes -- the frames to re-check.
    """

    view: str
    keyframe_id: int
    old: int
    new: Optional[int]
    kept: bool = False
    reach: list[int] = field(default_factory=list)

    @property
    def moved(self) -> bool:
        return self.new is not None and self.new != self.old

    @property
    def dropped(self) -> bool:
        return self.new is None


def _between(walk: list[int], low: int, high: int) -> list[int]:
    """Walk frames ``low < f <= high``."""
    return [f for f in walk if low < f <= high]


def _plan_one(cut: ShapeKeyframe, chain: list[ShapeKeyframe], walk: list[int],
              before: dict, after: dict) -> Optional[CutMove]:
    """:class:`CutMove` of one cut, ``None`` when its boundary did not move."""
    b = int(cut.anchor_step)
    if b not in walk:
        return None
    i = walk.index(b)
    if i + 1 >= len(walk):
        return None
    n = walk[i + 1]
    down, up = before.get(b), before.get(n)
    if down is None or up is None or down == up:
        return None                     # not on a boundary: not ours to move
    if after.get(b) == down and after.get(n) == up:
        return None                     # the boundary stayed where it was
    others = sorted(int(kf.anchor_step) for kf in chain)
    if after.get(b) == up:
        # extended: b joined the run above; its lower edge is the next frame
        # down that is still on the far side
        j = i - 1
        while j >= 0 and after.get(walk[j]) == up:
            j -= 1
        new = walk[j] if j >= 0 and after.get(walk[j]) == down else None
        if new is not None and not any(new <= a < b for a in others):
            return CutMove(cut.view, int(cut.id), b, new, reach=_between(walk, new, b))
        # every frame of the cut is in the run now: it is bookkeeping, unless
        # its shape is nowhere else
        if any(same_shape(cut, kf) for kf in chain):
            low = max([a for a in others if a < b], default=0)
            return CutMove(cut.view, int(cut.id), b, None, reach=_between(walk, low, b))
        return CutMove(cut.view, int(cut.id), b, b, kept=True)
    if after.get(n) == down:
        # shrunk: the run's bottom frames went over to the far side; its new
        # lower edge is the last of them under a frame that stayed in the run
        j = i + 1
        while j < len(walk) and after.get(walk[j]) == down and before.get(walk[j]) == up:
            j += 1
        if j < len(walk) and after.get(walk[j]) == up and before.get(walk[j]) == up:
            new = walk[j - 1]
            if not any(b < a <= new for a in others):
                return CutMove(cut.view, int(cut.id), b, new, reach=_between(walk, b, new))
        return None                     # the run vanished, or a version in the way
    return None


def plan_cut_moves(db: Db, tax: Taxonomy, desktop: int, target: str,
                   before: dict[int, Optional[str]], after: dict[int, Optional[str]],
                   cache: Optional[InputCache] = None) -> list[CutMove]:
    """What every cut keyframe of ``target`` does after a change of its states.

    ``before`` / ``after`` are :func:`mask_states` on either side of the write.
    Each view is walked on its own frames (:func:`~tda.core.split_rows.reverse_walk`)
    and within the cut's own pose segment, since a cut is a version of one
    segment's chain. Moves only -- nothing is written.
    """
    cache = cache if cache is not None else InputCache()
    out: list[CutMove] = []
    for view in cut_views(db, desktop, target):
        walk_all = reverse_walk(db, desktop, view)
        segment = {f: pose_segment_of(db, FrameKey(desktop, f, view), cache)
                   for f in walk_all}
        kfs = db.keyframes(desktop, view, target)
        for cut in [kf for kf in kfs if is_cut(kf) and kf.placement == IN_CHASSIS]:
            chain = [kf for kf in kfs if kf.id != cut.id
                     and kf.pose_segment == cut.pose_segment
                     and kf.placement == cut.placement]
            walk = [f for f in walk_all if segment[f] == cut.pose_segment]
            move = _plan_one(cut, chain, walk, before, after)
            if move is not None:
                out.append(move)
    return out
