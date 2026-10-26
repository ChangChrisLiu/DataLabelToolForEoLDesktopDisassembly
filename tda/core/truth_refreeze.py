"""「重签」: re-sign confirmed rows that trail today's compile within the tolerance (task U32).

The truth table freezes a confirmed frame's rows (spec 3.4).  A later change to
the inputs that moves a frozen row **beyond** the re-tracing tolerance
(:func:`tda.core.masks.is_conflict`: a 2 px band, more than max(2 % of the
area, 20 px)) opens a conflict a human settles.  A change **within** it opens
nothing -- it is the same annotation, by the app's own definition -- and the
frozen row kept its old pixels for ever, because nothing in the app ever
rewrote one (``verify_frame`` leaves a frozen row that agrees exactly as it
is).  Two things followed, both found on the real data:

* confirmed visible masks of *different* parts overlap.  One part was
  re-traced and accepted, and its neighbour lost a few pixels to it -- under 2 %
  of the neighbour, so its frozen row kept them (D2 ``cover.01`` /
  ``motherboard.01``: 5 227 px on every frame from 17 to 37);
* after a layer-order correction the part that *gains* pixels keeps its old
  row (under 2 % of a screw cover or a drive cage), while the part that lost
  them was settled by a conflict: the pixels belong to nobody -- a hole
  (U31-report section 7).

:func:`plan_frame` finds, on one confirmed frame, the frozen rows whose stored
value differs from today's compile **and** are within the tolerance -- exactly
the rows a refresh skips as "agreeing" -- and :meth:`RefreezeMixin.refreeze`
writes the compile into them: RLE (or box), ``visibility``,
``occlusion_ratio``, the compile's ``input_hash``, ``verified`` still, signed
by whoever asked.  Nothing else moves:

* a row **beyond** the tolerance, or with a conflict open, is a conflict and
  stays one (:func:`~tda.core.truth_refusals.frozen_disagreements`, the
  comparison ``refresh`` and ``verify_frame`` use, decides which);
* a row whose changing pixels touch such a row -- its frozen mask, or the
  pixels the compile now gives it -- **waits** for that conflict (``deferred``):
  re-signed now, it would take pixels the disputed row still holds (an overlap
  the conflict's answer then removes) or give up pixels the disputed row does
  not hold yet (a hole); found on D22, whose drive cage was raised over the
  front-panel connector with that connector's 22 conflicts still open;
* a row a re-signing wrote is measured against the row **a person signed**
  before it (:mod:`tda.core.truth_anchor`, round 2): beyond the tolerance from
  that one it is a conflict for the ordinary refresh, and ``drifted`` here --
  re-signing again and again cannot walk a row away from a signature;
* an instance the compile has and the frozen set lacks (or holds only as an
  ``auto`` row) is **reported, not written**: the frame then shows a different
  set of parts than the one confirmed, which the app's rule answers by sending
  it back for review, not by signing a part nobody looked at;
* the frame stays ``verified``; its digest is stamped for the current inputs
  when no disagreement stands on it (as a refresh would), left as it was
  otherwise.

Each frame is **one op**, ``refreeze_frame``, whose inverse is the old rows as
they were stored -- column for column, signer and time included -- and the
digest row before; :func:`undo_refreeze` puts them back byte for byte (and is
itself an op whose inverse is the re-signed rows).  Running it twice changes
nothing the second time.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from tda.core import dbrows as R
from tda.core import masks
from tda.core.compiler import CompiledFrame, CompiledInstance
from tda.core.model import FrameKey
from tda.core.truth_anchor import (
    REFREEZE_OP,
    anchor_history,
    human_rows,
    rows_differ as _rows_differ,
    signed_by_hand,
)
from tda.core.truth_conflicts import (
    GEOM_BOX,
    GEOM_MASK,
    box_sym_diff,
    disagreement,
    label_changes,
    rounded_box,
    row_values,
)
from tda.core.truth_fresh import digest_of
from tda.core.truth_inputs import InputCache, gather
from tda.core.truth_refusals import frozen_disagreements

__all__ = [
    "OP_KIND", "RESTORE_KIND", "UNDO_KIND", "UNSEEN", "FramePlan", "RefreezeMixin",
    "RowChange", "overlap_px", "plan_frame", "refreeze_history", "refreeze_ops",
    "signed_by_hand", "undo_refreeze", "undo_status", "UNDONE_ALREADY", "WOULD_RESTORE",
]

#: The op of one re-signed frame, and the op of putting it back.
OP_KIND = REFREEZE_OP
UNDO_KIND = "undo_refreeze"
#: What either op's inverse does: write these raw rows (and this digest) back.
RESTORE_KIND = "restore_rows"

VERIFIED = "verified"
#: The labels that say a part cannot be seen at all; a re-signed label moving
#: into or out of these is listed on its own (U32 review N3).
UNSEEN = frozenset({"occluded_full", "out_of_view"})


# --------------------------------------------------------------------------- #
# the plan of one frame
# --------------------------------------------------------------------------- #
@dataclass
class RowChange:
    """One frozen row to re-sign: as stored, as the compile has it, and how far apart."""

    instance: str
    #: Every ``compiled_mask`` column as stored (:data:`tda.core.db_truthraw.RAW_COLUMNS`).
    old: dict
    #: The same columns from the compile; ``verified_by`` / ``verified_at`` are
    #: filled when it is written.
    new: dict
    #: Pixels in exactly one of the two (no tolerance band): what the row gains
    #: and loses.
    px: int
    #: What differs: ``pixels``, ``geometry`` (mask <-> box), ``visibility``,
    #: ``occlusion_ratio``.
    fields: list[str] = field(default_factory=list)

    @property
    def label(self) -> Optional[tuple[Optional[str], Optional[str]]]:
        """``(old visibility, new visibility)`` when the label changes, else ``None``."""
        if "visibility" not in self.fields:
            return None
        return (self.old.get("visibility"), self.new.get("visibility"))

    @property
    def flips(self) -> bool:
        """Does the label cross 看得见 / 看不见 (e.g. ``too_small`` -> ``occluded_full``)?"""
        label = self.label
        if label is None or None in label:
            return False
        return (label[0] in UNSEEN) != (label[1] in UNSEEN)


@dataclass
class FramePlan:
    """What :meth:`RefreezeMixin.refreeze` does (or did) to one confirmed frame."""

    step: int
    #: The input digest the plan was made from; the write is refused if it moved.
    digest: str = ""
    input_hash: str = ""
    changes: list[RowChange] = field(default_factory=list)
    #: Frozen rows beyond the tolerance (or gone from the frame): conflicts, untouched.
    disputed: list[str] = field(default_factory=list)
    #: Frozen rows with a conflict open about them: untouched.
    open_conflicts: list[str] = field(default_factory=list)
    #: In the compile, no truth row at all: reported, not written.
    missing: list[str] = field(default_factory=list)
    #: In the compile with only an ``auto`` row: reported, not written.
    unsigned: list[str] = field(default_factory=list)
    #: ``auto`` rows the compile no longer has (a refresh drops them): left alone.
    leftover: list[str] = field(default_factory=list)
    #: Rows within the tolerance whose changing pixels touch a disputed (or
    #: open-conflict) row: they wait for that conflict; re-signed by a later run.
    deferred: list[str] = field(default_factory=list)
    #: Rows a re-signing wrote whose compile is within the tolerance of the
    #: stored row but **beyond** it from the row a person signed before the
    #: re-signings (:func:`~tda.core.truth_anchor.signed_by_hand`): never
    #: re-signed again; the ordinary refresh queues each as a conflict (round 2, B1).
    drifted: list[str] = field(default_factory=list)
    #: Pixels shared by two frozen visible masks (every pair counted), before
    #: and after; ``None`` unless measured.
    overlap_before: Optional[int] = None
    overlap_after: Optional[int] = None
    #: The ``refreeze_frame`` op written, once applied.
    op: Optional[int] = None
    #: The inputs or the rows moved between the plan and the write, twice: skipped.
    moved: bool = False

    @property
    def rows(self) -> list[str]:
        return [c.instance for c in self.changes]

    @property
    def px(self) -> int:
        return sum(c.px for c in self.changes)

    @property
    def stands(self) -> bool:
        """Does anything stay in disagreement on this frame once it is re-signed?"""
        return bool(self.disputed or self.open_conflicts or self.missing or self.unsigned
                    or self.leftover or self.deferred or self.drifted)

    @property
    def flips(self) -> list[RowChange]:
        """The re-signed rows whose label crosses 看得见 / 看不见."""
        return [c for c in self.changes if c.flips]


def _union(a, b):
    if a is None or b is None:
        return b if a is None else a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _mask_px(stored_rle: Optional[dict], compiled_inst: CompiledInstance) -> int:
    """Pixels in exactly one of the stored visible mask and the compiled one."""
    visible = compiled_inst.visible
    if stored_rle is None:
        return 0 if visible is None else masks.area(visible)
    old = masks.RunLengths(stored_rle)
    if visible is None:
        return old.area()
    if tuple(old.hw) != tuple(int(v) for v in visible.shape):
        return max(old.area(), masks.area(visible))
    window = _union(old.bbox(), masks.bbox_in(visible, compiled_inst.window))
    if window is None:
        return 0
    x0, y0, x1, y1 = window
    return int(np.count_nonzero(old.window(window) ^ visible[y0:y1, x0:x1]))


def _box_of(row_geom: str, rle: Optional[dict], box) -> Optional[tuple]:
    if row_geom == GEOM_BOX:
        return rounded_box(box)
    return None if rle is None else masks.RunLengths(rle).bbox()


def _boxes_px(a: Optional[tuple], b: Optional[tuple]) -> int:
    """Pixels covered by exactly one of two boxes, either of which may be absent."""
    if a is None and b is None:
        return 0
    if a is None or b is None:
        q = a or b
        return int((q[2] - q[0]) * (q[3] - q[1]))
    return box_sym_diff(a, b)


def _difference(row: dict, compiled_inst: CompiledInstance, values) -> tuple[int, list[str]]:
    """``(pixels, fields)`` between a stored row and the compile; ``(0, [])`` when equal."""
    fields: list[str] = []
    stored_geom = row.get("geom_type") or GEOM_MASK
    if stored_geom != values.geom_type:
        fields.append("geometry")
        px = _boxes_px(_box_of(stored_geom, row.get("visible_rle"), row.get("box")),
                       _box_of(values.geom_type, values.visible_rle, values.box))
    elif values.geom_type == GEOM_BOX:
        px = 0
        if [float(v) for v in row.get("box") or ()] != [float(v) for v in values.box or ()]:
            # a sub-pixel move still re-signs the row; it is at least one pixel
            px = max(1, _boxes_px(rounded_box(row.get("box")), rounded_box(values.box)))
            fields.append("pixels")
    else:
        stored_rle = row.get("visible_rle")
        same = (stored_rle is None and values.visible_rle is None) or (
            stored_rle is not None and values.visible_rle is not None
            and list(stored_rle.get("size") or ()) == list(values.visible_rle["size"])
            and masks.rle_counts(stored_rle) == masks.rle_counts(values.visible_rle))
        px = 0 if same else _mask_px(stored_rle, compiled_inst)
        if px:
            fields.append("pixels")
    if (row.get("visibility") or None) != (values.visibility or None):
        fields.append("visibility")
    held = row.get("occlusion_ratio")
    if held is None or float(held) != float(values.occlusion_ratio):
        fields.append("occlusion_ratio")
    return int(px), fields


def _new_raw(values, input_hash: str) -> dict:
    """The ``compiled_mask`` columns of a re-signed row (signer and time still open)."""
    return {
        "visible_rle_json": R.dumps(values.visible_rle),
        "occlusion_ratio": float(values.occlusion_ratio),
        "visibility": values.visibility,
        "placement": values.placement,
        "status": VERIFIED,
        "input_hash": input_hash,
        "verified_by": None,
        "verified_at": None,
        "geom_type": values.geom_type,
        "box_json": R.dumps(None if values.box is None else [float(v) for v in values.box]),
    }


def _paint_rle(canvas: np.ndarray, rle: Optional[dict]) -> None:
    """OR one stored mask into ``canvas`` (same size), inside its own box."""
    if not rle:
        return
    runs = masks.RunLengths(rle)
    box = runs.bbox()
    if box is None or tuple(runs.hw) != tuple(canvas.shape):
        return
    x0, y0, x1, y1 = box
    canvas[y0:y1, x0:x1] |= runs.window(box)


def _paint_box(canvas: np.ndarray, box) -> None:
    rounded = rounded_box(box)
    if rounded is not None:
        x0, y0, x1, y1 = rounded
        canvas[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True


def _canvas_hw(stored: dict, compiled: CompiledFrame) -> Optional[tuple[int, int]]:
    for inst in compiled.instances.values():
        if inst.visible is not None:
            return tuple(int(v) for v in inst.visible.shape)
    for row in stored.values():
        if row.get("visible_rle"):
            return tuple(int(v) for v in row["visible_rle"]["size"])
    return None


def _defer_near_conflicts(plan: FramePlan, stored: dict, compiled: CompiledFrame,
                          alone: set[str]) -> None:
    """Hold back the changes whose changing pixels touch a row left to its conflict.

    ``alone`` are the frozen rows in disagreement (or with a conflict open);
    what they cover is their frozen mask (box) and what the compile gives
    them now.  A change is the pixels in exactly one of its stored and new
    mask -- for a box or a mask <-> box change, the box around both.  Those
    touching what a disputed row covers move from ``changes`` to ``deferred``.
    """
    if not alone or not plan.changes:
        return
    hw = _canvas_hw(stored, compiled)
    if hw is None:
        return
    held = np.zeros(hw, dtype=bool)
    for instance in alone:
        row = stored.get(instance) or {}
        if row.get("geom_type") == GEOM_BOX:
            _paint_box(held, row.get("box"))
        else:
            _paint_rle(held, row.get("visible_rle"))
        now = compiled.instances.get(instance)
        if now is not None and now.visible is not None and tuple(now.visible.shape) == hw:
            held |= now.visible
        elif now is not None and now.box is not None:
            _paint_box(held, now.box)
    keep: list[RowChange] = []
    for change in plan.changes:
        old_rle, new_rle = R.loads(change.old["visible_rle_json"]), R.loads(change.new["visible_rle_json"])
        if change.old["geom_type"] == GEOM_MASK and change.new["geom_type"] == GEOM_MASK:
            moved = np.zeros(hw, dtype=bool)
            _paint_rle(moved, old_rle)
            fresh = np.zeros(hw, dtype=bool)
            _paint_rle(fresh, new_rle)
            moved ^= fresh
        else:
            moved = np.zeros(hw, dtype=bool)
            for geom, rle, box in ((change.old["geom_type"], old_rle, R.loads(change.old["box_json"])),
                                   (change.new["geom_type"], new_rle, R.loads(change.new["box_json"]))):
                _paint_box(moved, _box_of(geom, rle, box))
        if bool((moved & held).any()):
            plan.deferred.append(change.instance)
        else:
            keep.append(change)
    plan.changes = keep


def refreeze_history(db, desktop: int, view: str) -> dict[int, list[dict]]:
    """The ``refreeze_frame`` ops of one view, per step, oldest first (cached).

    :func:`tda.core.truth_anchor.anchor_history`, under the name this module
    has always exported.
    """
    return anchor_history(db, desktop, view)


def plan_frame(db, key: FrameKey, compiled: CompiledFrame,
               overrides: Optional[dict] = None, measure: bool = False,
               history: Optional[list[dict]] = None) -> FramePlan:
    """The frozen rows of one frame that today's compile would re-sign, and why the rest not.

    Writes nothing.  ``compiled`` must be what the frame's current inputs
    compile to; ``overrides`` its frame overrides (read here when left out).
    A row is re-signed when it is ``verified``, carries another ``input_hash``
    than ``compiled``, has no conflict open, is **not** among the
    disagreements :func:`~tda.core.truth_refusals.frozen_disagreements` finds
    (beyond the tolerance, a label a human stated, a part gone) -- and its
    stored value still differs from the compile: pixels, the mask/box kind,
    ``visibility`` or ``occlusion_ratio`` -- unless its changing pixels touch
    a row left to its conflict, when it waits (``deferred``,
    :func:`_defer_near_conflicts`).  ``measure`` also counts the pixels shared
    by two frozen visible masks before and after (:func:`overlap_px`).

    **The reference is the row a person signed** (round 2, B1).  The
    tolerance is not transitive: re-signing a row moves what the next
    comparison is made against, so a row re-signed again and again could walk
    any distance from what anybody confirmed without a conflict ever opening.
    :func:`~tda.core.truth_refusals.frozen_disagreements` therefore measures a
    row a re-signing wrote against the row a person signed before it
    (:mod:`tda.core.truth_anchor`) -- for the refresh, ``verify_frame`` and
    this plan alike -- and such a row beyond the tolerance from that one is a
    conflict the ordinary refresh queues.  The plan lists it as ``drifted``
    (rather than ``disputed``) when only the person's row, not the stored one,
    is beyond the tolerance from the compile: never re-signed again.  A stored
    row is never more than one tolerance from a person's signature, and the
    compile cannot move further than one tolerance from it unnoticed.
    ``history`` is the frame's ``refreeze_frame`` ops (read here when left out).
    """
    stored = db.compiled(key)
    raw = db.compiled_raw(key)
    if overrides is None:
        overrides = db.frame_overrides(key)
    if history is None:
        history = refreeze_history(db, key.desktop, key.view).get(int(key.step), [])
    plan = FramePlan(step=int(key.step), input_hash=str(compiled.input_hash))
    plan.open_conflicts = sorted({str(inst) for _cid, inst in db.open_conflicts_at(key)})
    disputed = {b.instance for b in frozen_disagreements(db, key, compiled, overrides, stored)}
    signed = human_rows(db, key, disputed) if history else {}
    for instance in sorted(signed):
        # beyond the tolerance from what a person signed, within it from what
        # the last re-signing wrote: the drift a re-signing must not carry on
        compiled_inst = compiled.instances.get(instance)
        row = stored.get(instance)
        if compiled_inst is not None and row is not None and disagreement(
                row, compiled_inst) is None and not label_changes(
                row, compiled_inst, overrides.get(instance)):
            plan.drifted.append(instance)
    plan.disputed = sorted(disputed - set(plan.open_conflicts) - set(plan.drifted))
    plan.drifted = sorted(set(plan.drifted) - set(plan.open_conflicts))
    plan.leftover = sorted(inst for inst, row in stored.items()
                           if inst not in compiled.instances and row["status"] != VERIFIED)
    for instance in sorted(compiled.instances):
        row = stored.get(instance)
        if row is None:
            plan.missing.append(instance)
            continue
        if row["status"] != VERIFIED:
            plan.unsigned.append(instance)
            continue
        if instance in disputed or instance in plan.open_conflicts:
            continue
        if row["input_hash"] == compiled.input_hash:
            continue                                   # it *is* this compilation
        compiled_inst = compiled.instances[instance]
        values = row_values(compiled_inst)
        px, fields = _difference(row, compiled_inst, values)
        if not fields:
            continue
        plan.changes.append(RowChange(instance, dict(raw[instance]),
                                      _new_raw(values, compiled.input_hash), px, fields))
    _defer_near_conflicts(plan, stored, compiled,
                          disputed | set(plan.open_conflicts) | set(plan.drifted))
    if measure:
        frozen = {inst: row.get("visible_rle") for inst, row in stored.items()
                  if row["status"] == VERIFIED and (row.get("geom_type") or GEOM_MASK) == GEOM_MASK}
        plan.overlap_before = overlap_px(frozen.values())
        after = dict(frozen)
        for change in plan.changes:
            after[change.instance] = (R.loads(change.new["visible_rle_json"])
                                      if change.new["geom_type"] == GEOM_MASK else None)
        plan.overlap_after = overlap_px(after.values())
    return plan


def overlap_px(rles: Iterable[Optional[dict]]) -> int:
    """Pixels shared by two of these masks, each pair counted once.

    A pixel under ``n`` masks counts ``n (n - 1) / 2`` -- the sum of the
    pairwise intersections, which is how U30-PACK counted the overlapping
    confirmed masks (262 408 px on 98 frames).  Each mask is painted inside its
    own bounding box off the run lengths; masks of another canvas size than the
    first are left out.
    """
    count: Optional[np.ndarray] = None
    for rle in rles:
        if not rle:
            continue
        runs = masks.RunLengths(rle)
        box = runs.bbox()
        if box is None:
            continue
        if count is None:
            count = np.zeros(runs.hw, dtype=np.uint16)
        elif tuple(count.shape) != tuple(runs.hw):
            continue
        x0, y0, x1, y1 = box
        count[y0:y1, x0:x1] += runs.window(box)
    if count is None:
        return 0
    many = count[count > 1].astype(np.int64)
    return int((many * (many - 1) // 2).sum())


# --------------------------------------------------------------------------- #
# writing, and taking it back
# --------------------------------------------------------------------------- #
class _Moved(Exception):
    """The inputs or the rows changed after the plan was made."""


def refreeze_ops(db, desktop: int, view: str, run: Optional[str] = None) -> list[dict]:
    """The ``refreeze_frame`` ops of one view, oldest first; one run's, or the newest run's.

    ``run`` is the tag every op of one :meth:`RefreezeMixin.refreeze` call
    carries; ``"last"`` (or ``None``) picks the newest run of the view.
    """
    ops = [op for op in db.ops_of_kind(int(desktop), OP_KIND) if op["view"] == view]
    if not ops:
        return []
    tag = run
    if tag in (None, "", "last"):
        tag = (ops[-1].get("payload") or {}).get("run")
    return [op for op in ops if (op.get("payload") or {}).get("run") == tag]


#: :func:`undo_status`'s two answers (a refusal is a ``ValueError``).
WOULD_RESTORE, UNDONE_ALREADY = "restore", "undone"


def undo_status(db, op_id: int) -> tuple[FrameKey, str, list[str]]:
    """What :func:`undo_refreeze` would do with one op -- and nothing is written.

    ``(frame, WOULD_RESTORE or UNDONE_ALREADY, the instances it puts back)``;
    ``ValueError`` for an op that is not a re-signing or its undo, or whose
    rows somebody changed since (the refusals :func:`undo_refreeze` raises).
    What ``refreeze --undo --dry-run`` lists (round 2, B2).
    """
    op = db.get_op(int(op_id))
    if op is None or op["kind"] not in (OP_KIND, UNDO_KIND):
        raise ValueError(f"op {op_id} is not a {OP_KIND} (or {UNDO_KIND}) row")
    payload, inverse = op.get("payload") or {}, op.get("inverse") or {}
    key = FrameKey(int(op["desktop"]), int(payload["step"]), str(op["view"]))
    written, wanted = dict(payload.get("rows") or {}), dict(inverse.get("rows") or {})
    now = db.compiled_raw(key)
    if all(not _rows_differ(now.get(i), row) for i, row in wanted.items()):
        return key, UNDONE_ALREADY, sorted(wanted)
    changed = sorted(i for i, row in written.items() if _rows_differ(now.get(i), row))
    if changed:
        raise ValueError(
            f"frame {key.desktop}/{key.view}/step {key.step}: {', '.join(changed)} changed "
            f"after op {op_id} re-signed it; not undone")
    return key, WOULD_RESTORE, sorted(wanted)


def undo_refreeze(db, op_id: int, annotator: str) -> Optional[int]:
    """Put back the rows (and the digest) one ``refreeze_frame`` op replaced.

    The rows are restored **byte for byte** -- every column as it was stored,
    the RLE text, the signer and the time they signed -- and so is the frame's
    digest row (or its absence).  Only while the rows are still exactly what
    the op wrote: a frame somebody touched since (a conflict accepted, the
    frame confirmed again) is refused with ``ValueError`` and nothing is
    written.  An op already undone -- its rows are the old ones again -- is a
    no-op and returns ``None``.  Otherwise one ``undo_refreeze`` op is logged,
    whose inverse is the re-signed rows, and its id is returned.  The frame's
    review status is not touched either way.

    An ``undo_refreeze`` op is undone the same way (it puts the re-signed rows
    back).  :func:`undo_status` asks the same questions without writing.
    """
    key, status, _instances = undo_status(db, op_id)
    if status == UNDONE_ALREADY:
        return None
    op = db.get_op(int(op_id))
    payload, inverse = op.get("payload") or {}, op.get("inverse") or {}
    written, wanted = dict(payload.get("rows") or {}), dict(inverse.get("rows") or {})
    with db.transaction():
        for instance, row in sorted(wanted.items()):
            db.put_compiled_raw(key, instance, row)
        db.put_frame_digest_raw(key, inverse.get("digest"))
        return db.log_op(
            key.desktop, key.view, UNDO_KIND,
            {"step": key.step, "undoes": int(op_id), "instances": sorted(wanted),
             "rows": wanted, "digest": inverse.get("digest")},
            {"kind": RESTORE_KIND, "step": key.step, "rows": written,
             "digest": payload.get("digest")},
            annotator,
        )


Progress = Callable[[FramePlan], None]


class RefreezeMixin:
    """「重签」 for :class:`~tda.core.truth.TruthService` (task U32)."""

    def refreeze_steps(self, desktop: int, view: str,
                       steps: Optional[Iterable[int]] = None) -> list[int]:
        """The confirmed frames of a view (``review_status`` verified, with an image)."""
        wanted = None if steps is None else {int(s) for s in steps}
        return sorted(int(row["step"]) for row in self.db.frames_for(desktop, view)
                      if row.get("review_status") == VERIFIED and not row.get("missing")
                      and (wanted is None or int(row["step"]) in wanted))

    def refreeze(self, desktop: int, view: str, steps: Optional[Iterable[int]] = None,
                 annotator: str = "system", dry_run: bool = False, measure: bool = False,
                 plans: Optional[Iterable[FramePlan]] = None, run: Optional[str] = None,
                 progress: Optional[Progress] = None) -> dict:
        """Re-sign every frozen row of the confirmed frames that trails the compile (U32).

        ``steps`` narrows the frames (default: every confirmed frame of the
        view).  ``dry_run`` compiles and plans and writes nothing.  ``plans``
        are the frames of an earlier dry run to write: each is written as
        planned while its inputs and its rows are still what the plan saw, and
        planned again otherwise (the window plans, asks, then writes -- one
        compilation per frame, not two).  ``measure`` counts the overlapping
        pixels before and after (:func:`overlap_px`); ``progress`` is called
        with each frame's plan as it is done.

        One transaction and one ``refreeze_frame`` op per frame that changes;
        a frame with nothing to re-sign writes nothing at all.  Returns
        ``{"frames": [FramePlan], "considered", "frames_rewritten", "rewritten",
        "px", "disputed", "open_conflicts", "missing", "unsigned", "deferred",
        "drifted", "flips", "moved", "overlap_before", "overlap_after", "ops",
        "run", "dry_run"}``.
        """
        tag = run or f"{R.now_iso()}#{uuid.uuid4().hex[:6]}"
        cache = InputCache()
        given = {int(p.step): p for p in plans or ()}
        todo = sorted(given) if plans is not None else self.refreeze_steps(desktop, view, steps)
        frames: list[FramePlan] = []
        for step in todo:
            key = FrameKey(int(desktop), int(step), str(view))
            plan = given.get(int(step)) or self._plan(key, cache, measure)
            if not dry_run and plan.changes:
                try:
                    self._write_plan(key, plan, annotator, tag)
                except _Moved:
                    plan = self._plan(key, InputCache(), measure)
                    try:
                        if plan.changes:
                            self._write_plan(key, plan, annotator, tag)
                    except _Moved:
                        plan.moved = True
            frames.append(plan)
            if progress is not None:
                progress(plan)
        return self._summary(frames, tag, dry_run)

    def undo_refreeze(self, op_id: int, annotator: str = "system") -> Optional[int]:
        """:func:`undo_refreeze` on this service's database."""
        return undo_refreeze(self.db, op_id, annotator)

    # --------------------------------------------------------------- internals
    def _plan(self, key: FrameKey, cache: InputCache, measure: bool) -> FramePlan:
        inputs = gather(self.db, self.tax, key, cache, self.cache_dir)
        compiled = self._compile_inputs(key, inputs)
        plan = plan_frame(self.db, key, compiled, inputs.frame_overrides, measure)
        plan.digest = digest_of(inputs, self.compiler_version)
        return plan

    def _write_plan(self, key: FrameKey, plan: FramePlan, annotator: str, run: str) -> None:
        """Write one frame's plan in one transaction, or raise :class:`_Moved`."""
        with self.db.transaction():
            if self.inputs_digest(key) != plan.digest:
                raise _Moved()
            now = self.db.compiled_raw(key)
            busy = {inst for _cid, inst in self.db.open_conflicts_at(key)}
            if any(_rows_differ(now.get(c.instance), c.old) or c.instance in busy
                   for c in plan.changes):
                raise _Moved()
            before = self.db.frame_digest(key)
            stamp = R.now_iso()
            written: dict[str, dict] = {}
            for change in plan.changes:
                row = dict(change.new, verified_by=str(annotator), verified_at=stamp)
                self.db.put_compiled_raw(key, change.instance, row)
                written[change.instance] = row
            if not plan.stands:
                # every frozen row now is this compilation or agrees with it
                # byte for byte: what a refresh stamps (and nothing it would not)
                self._stamp(key, plan.digest)
            after = self.db.frame_digest(key)
            plan.op = self.db.log_op(
                key.desktop, key.view, OP_KIND,
                {"step": key.step, "run": run, "instances": sorted(written),
                 "rows": written, "digest": after, "input_hash": plan.input_hash,
                 "px": {c.instance: c.px for c in plan.changes},
                 "fields": {c.instance: list(c.fields) for c in plan.changes},
                 "overlap_before": plan.overlap_before, "overlap_after": plan.overlap_after},
                {"kind": RESTORE_KIND, "step": key.step,
                 "rows": {c.instance: dict(c.old) for c in plan.changes}, "digest": before},
                str(annotator),
            )

    @staticmethod
    def _summary(frames: list[FramePlan], run: str, dry_run: bool) -> dict:
        def total(name: str) -> Optional[int]:
            values = [getattr(f, name) for f in frames if getattr(f, name) is not None]
            return sum(values) if values else None

        changed = [f for f in frames if f.changes and not f.moved]
        return {
            "frames": frames, "considered": len(frames), "frames_rewritten": len(changed),
            "rewritten": sum(len(f.changes) for f in changed),
            "px": sum(f.px for f in changed),
            "disputed": sum(len(f.disputed) for f in frames),
            "open_conflicts": sum(len(f.open_conflicts) for f in frames),
            "missing": sum(len(f.missing) for f in frames),
            "unsigned": sum(len(f.unsigned) for f in frames),
            "deferred": sum(len(f.deferred) for f in frames),
            "drifted": sum(len(f.drifted) for f in frames),
            # (step, instance, old label, new label) of the labels crossing
            # 看得见 / 看不见 (review N3), in frame order
            "flips": [(f.step, c.instance, *c.label) for f in changed for c in f.flips],
            "moved": sum(1 for f in frames if f.moved),
            "overlap_before": total("overlap_before"), "overlap_after": total("overlap_after"),
            "ops": [f.op for f in frames if f.op is not None], "run": run,
            "dry_run": bool(dry_run),
        }
