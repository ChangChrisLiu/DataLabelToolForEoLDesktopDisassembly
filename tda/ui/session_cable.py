"""「被线挡住」 in the session (task U15): ``I`` and ``Ctrl+I`` on a sequence frame.

Cables are not annotated, and a part in its place is sometimes under one. The
annotator says so on the frame (:mod:`tda.core.occluder` is the rule): ``I``
cycles the selected part 完全 -> 部分 -> 撤回, one ``set_frame_override`` op per
press; ``Ctrl+I`` writes one of the three over a run of frames as **one** op,
``set_frame_overrides`` -- one row per frame and one inverse, so one ``Ctrl+Z``.

完全 and 部分 are refused where the label would describe nothing: a part with
no shape on the frame (完全 keeps the amodal shape, so there has to be one -- it
may be carried from another frame as usual), a part that is not on the frame
at all (taken out), one lying on the bench (not in its place), and the part
whose pixels are being drawn. **撤回 is never refused**: on a frame that can no
longer carry a label it holds (the step table moved, the shape went) the next
``I`` is 撤回 whatever the label was, and ``Ctrl+I`` 撤回 runs over any frame.
The components gallery has no cable label at all -- ``I`` means 「看不见」
there (task U10).

**画框外** (task U39) is ``Ctrl+I``'s fourth choice and nobody else's: the part
is in the machine but outside this view's picture (:data:`OFF_FRAME`, the
gallery's 「看不见」 row). It needs no shape -- only a part taken out, or one
on the bench, is refused -- and ``I`` never sets it, only takes it back (the
press after it is 撤回); 撤回 over a span takes it back with the cable labels.

On a confirmed frame the label is the annotator's own statement about that
frame, so the row is re-signed with it rather than queued for Review
(:func:`accept_own_label`); the undo does the same with the old row.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tda.core import masks
from tda.core.db import Db
from tda.core.model import OCCLUDER_CABLE, FrameKey, is_component_view, is_context_key
from tda.core.occluder import (
    FULL,
    OFF_FRAME,
    PARTIAL,
    cable_label,
    frame_label,
    kept_mask,
    next_label,
    occluded_override,
    off_frame,
    off_frame_applies,
    shape_applies,
)
from tda.core.truth import TruthService
from tda.core.truth_conflicts import payload_geometry, payload_labels
from tda.core.truth_inputs import (
    HW_MEASURED,
    InputCache,
    annotatable_steps,
    frame_hw,
    pose_segment_of,
    view_steps,
)
from tda.ui.commands import Op
from tda.ui.gallery import GALLERY_REFUSALS
from tda.ui.session_api import SessionRefusal
from tda.ui.session_ops import _write_override, frozen_among, override_state, settle

__all__ = [
    "CABLE_BY", "CABLE_EDITING", "CABLE_GALLERY", "CABLE_NEEDS_SHAPE", "CABLE_NOW_PARTIAL",
    "CABLE_ON_BENCH", "CABLE_OTHER_SEGMENT", "CABLE_REMOVED", "CableMixin", "LABEL_ZH",
    "OFF_FRAME", "OFF_FRAME_CHOICE", "OFF_FRAME_ON_BENCH", "OFF_FRAME_TAG", "OFF_FRAME_ZH",
    "OP_RANGE", "TAG_ZH", "accept_own_label", "apply_cable", "cable_one", "cable_range",
    "cable_reason", "label_reason", "open_about", "range_bounds",
]

#: The op ``Ctrl+I`` logs: one instance's override on several frames.
OP_RANGE = "set_frame_overrides"
#: The key both directions of a cable op carry: who said it, so that the undo
#: and the redo re-sign a confirmed frame in the same name (:func:`apply_cable`).
CABLE_BY = "cable_by"
#: Pixels committed for this frame only on a part 被线挡住（完全） (Alt+Enter).
CABLE_NOW_PARTIAL = ("{label}：这一帧画出了看得见的部分，被线挡住改成「部分」"
                     "（Ctrl+Z 撤销）")
#: The three steps of the cycle as the status line and the dialog say them.
LABEL_ZH: dict[Optional[str], str] = {FULL: "完全", PARTIAL: "部分", None: "撤回"}
#: The instance table's tag of each label (the Vis column is 46 px wide).
TAG_ZH = {FULL: "线挡", PARTIAL: "线半挡"}

#: No shape applies here (the ``OVERRIDE_NEEDS_SHAPE`` idea, U2d).
CABLE_NEEDS_SHAPE = "先画出它的形状（可以从别的帧带过来），再按 ` 键标被线挡住"
#: The frame does not need the part at all.
CABLE_REMOVED = "这一帧不要它的形状（已拆下，或这个状态不画）：没有可标的"
#: On the staging area: a rectangle, and not in its place.
CABLE_ON_BENCH = "它放在台面上（台面框），不在原位：被线挡住只标原位的零件"
#: Its pixels are being drawn.
CABLE_EDITING = "它正在画、还没提交：先 Q 提交或 Esc 放弃，再按 ` 键"
#: ``Ctrl+``` on a gallery shot: `````` means 看不见 there.
CABLE_GALLERY = GALLERY_REFUSALS["cable_range"]
#: ``Ctrl+``` to a step of another pose segment.
CABLE_OTHER_SEGMENT = ("第 {step} 步在另一个位姿段（这一段是第 {low}–{high} 步）："
                       "Ctrl+` 只在一个位姿段里")

#: 画框外 (task U39): the word the panels and the status line use ...
OFF_FRAME_ZH = "画框外"
#: ... the dialog's fourth choice ...
OFF_FRAME_CHOICE = "画框外（这个视角拍不到）"
#: ... and the instance table's tag (the Vis column is 46 px wide).
OFF_FRAME_TAG = "框外"
#: 画框外 on a part lying on the staging area: the bench is in the picture.
OFF_FRAME_ON_BENCH = "它放在台面上（台面框），在画面里：画框外只标原位的零件"

_REASONS = {"removed": CABLE_REMOVED, "bench": CABLE_ON_BENCH,
            "no_shape": CABLE_NEEDS_SHAPE}
#: 画框外 needs no shape: only these two refuse it (task U39).
_OFF_FRAME_REASONS = {"removed": CABLE_REMOVED, "bench": OFF_FRAME_ON_BENCH}


def cable_reason(db: Db, tax, key: FrameKey, instance: str,
                 cache: Optional[InputCache] = None) -> str:
    """Why a 完全/部分 label cannot go on ``instance`` at ``key``, ``""`` when it can."""
    return _REASONS.get(shape_applies(db, tax, key, instance, cache), "")


def label_reason(db: Db, tax, key: FrameKey, instance: str, label: Optional[str],
                 cache: Optional[InputCache] = None, keyframes=None) -> str:
    """Why ``label`` cannot go on ``instance`` at ``key``, ``""`` when it can.

    完全 / 部分 want a shape in its place (:func:`cable_reason`); 画框外 (task
    U39) only the part in its place (:func:`~tda.core.occluder.off_frame_applies`);
    撤回 is never refused.
    """
    if label is None:
        return ""
    if label == OFF_FRAME:
        return _OFF_FRAME_REASONS.get(off_frame_applies(db, tax, key, instance, cache), "")
    return _REASONS.get(shape_applies(db, tax, key, instance, cache, keyframes), "")


def range_bounds(db: Db, key: FrameKey, cache: Optional[InputCache] = None
                 ) -> tuple[int, int]:
    """``(low, high)``: the annotatable steps of ``key``'s pose segment ``Ctrl+I`` reaches."""
    cache = cache if cache is not None else InputCache()
    seg = pose_segment_of(db, key, cache)
    steps = [s for s in annotatable_steps(db, key.desktop, key.view,
                                          view_steps(db, key.desktop, key.view))
             if pose_segment_of(db, FrameKey(key.desktop, s, key.view), cache) == seg]
    if not steps:
        return key.step, key.step
    return min(steps + [key.step]), max(steps + [key.step])


def _refuse_gallery(key: FrameKey) -> None:
    if is_component_view(key.view):
        raise SessionRefusal(CABLE_GALLERY)


def _hand_mask(fo) -> bool:
    """Does ``fo`` hold a single-frame mask with pixels, drawn by hand (Alt+Enter)?"""
    mask = kept_mask(fo)
    return mask is not None and masks.rle_area(mask) > 0


def _measured(db: Db, key: FrameKey) -> bool:
    """Has this frame's image size been read off the image already? (``frame_hw``)"""
    row = db.get_frame(key) or {}
    return ((row.get("aux") or {}).get("hw_source")) == HW_MEASURED


def cable_one(db: Db, truth: TruthService, key: FrameKey, instance: str,
              label: Optional[str], annotator: str = "system",
              occluder: str = OCCLUDER_CABLE) -> dict:
    """Set ``label`` (完全 / 部分 / 画框外 / ``None`` = 撤回) on this frame: one op.

    One ``set_frame_override`` op. Returns what every edit returns plus
    ``label``, ``changed``, ``was`` (what the frame said before,
    :func:`~tda.core.occluder.frame_label`), ``replaced_hand_mask`` (完全 or
    画框外 took the place of a single-frame mask drawn by hand -- the op's
    inverse holds it) and ``accepted`` (the frozen frames whose row now
    carries the label, :func:`accept_own_label`); nothing is written (and no
    op made) when the frame already says exactly that.
    """
    _refuse_gallery(key)
    why = label_reason(db, truth.tax, key, instance, label)
    if why:
        raise SessionRefusal(why)
    previous = db.frame_overrides(key).get(instance)
    was = frame_label(previous)
    hw = frame_hw(db, key, truth.cache_dir) if label == FULL else (0, 0)
    after = occluded_override(key, instance, previous, label, hw, occluder)
    common = {"desktop": key.desktop, "view": key.view, "step": key.step,
              "instance": instance, "steps": [key.step], CABLE_BY: annotator}
    payload = common | override_state(after)
    inverse = common | override_state(previous)
    if _same(payload, inverse):
        return _settled(db, truth, key, [key.step], None, label, changed=False, was=was)
    disputed = disputed_before(db, truth, key.desktop, key.view, instance, [key.step])
    with db.transaction():
        _write_override(db, key, instance, payload)
        db.log_op(key.desktop, key.view, "set_frame_override", payload, inverse, annotator)
    out = _settled(db, truth, key, [key.step], Op("set_frame_override", payload, inverse),
                   label, was=was,
                   replaced_hand_mask=label in (FULL, OFF_FRAME) and _hand_mask(previous))
    return _signed(db, truth, key.desktop, key.view, instance, [key.step], annotator,
                   disputed, out, key.step)


def open_about(db: Db, desktop: int, view: str, instance: str, steps) -> set[int]:
    """The steps among ``steps`` with a conflict about ``instance`` open right now."""
    wanted = {int(s) for s in steps}
    return {int(r["step"]) for r in db.conflicts(desktop, view, open_only=True)
            if str(r["instance"]) == str(instance) and int(r["step"]) in wanted}


def disputed_before(db: Db, truth: TruthService, desktop: int, view: str, instance: str,
                    steps) -> set[int]:
    """The frozen frames among ``steps`` that already disagree about ``instance``.

    Asked **before** a cable write, so that :func:`accept_own_label` signs
    only the difference that write makes (review round 2, B1). A frozen frame
    the inputs moved under -- a re-trace of the shape elsewhere, still waiting
    in the re-check queue -- would otherwise disagree for both reasons at once
    after the write, and the annotator's label would sign the re-trace too,
    unseen. So every frozen frame in reach is compiled now, whatever its digest
    says (what the queued re-check would do): a difference already there opens
    its conflict, for Review, and the frame counts as disputed.
    """
    for step in frozen_among(db, desktop, view, steps):
        truth.refresh(FrameKey(desktop, step, view), ignore_digest=True)
    return open_about(db, desktop, view, instance, steps)


def _reveals(conflict: dict) -> bool:
    """Does accepting ``conflict`` put pixels where the frozen row had none?

    The frozen side's visible mask is empty (a 完全 the frame was confirmed
    with) and the new side's is not (部分 / 撤回): pixels nobody confirmed --
    the shape may have been redrawn elsewhere since, and an empty row cannot
    disagree with that (review round 4).
    """
    old = payload_geometry(conflict.get("old_rle"))[1]
    new = payload_geometry(conflict.get("new_rle"))[1]
    return ((old is None or masks.rle_area(old) == 0)
            and new is not None and masks.rle_area(new) > 0)


def _confirmed_once(db: Db, conflict: dict) -> bool:
    """Are the pixels ``conflict`` would reveal the ones the row was confirmed with?

    What ``Ctrl+Z`` of a 完全 brings back when nothing moved in between: the
    row the 完全 replaced was signed by a human, and the ``accept_new`` that
    replaced it kept it as its old side. Pixels redrawn since are not that.
    """
    new = masks.rle_counts(payload_geometry(conflict.get("new_rle"))[1])
    earlier = [c for c in db.conflicts(conflict["desktop"], conflict["view"], open_only=False)
               if int(c["step"]) == int(conflict["step"]) and c["instance"] == conflict["instance"]
               and c.get("resolution") == "accept_new" and int(c["id"]) != int(conflict["id"])]
    if not earlier or new is None:
        return False
    last = max(earlier, key=lambda c: int(c["id"]))
    return masks.rle_counts(payload_geometry(last.get("old_rle"))[1]) == new


def accept_own_label(db: Db, truth: TruthService, desktop: int, view: str, instance: str,
                     steps, annotator: str, disputed=(), held=None,
                     current: Optional[int] = None) -> tuple[list[int], list[int]]:
    """Keep a confirmed frame confirmed under a 被线挡住 label its annotator just set.

    Writing a label on a frozen frame makes its re-compilation disagree with
    the frozen row -- the digit keys leave that disagreement in the conflict
    queue, and ``Space`` waits for Review. For the cable label the annotator
    *is* the authority on this frame (task U15, ruled), so each frozen frame
    among ``steps`` is compiled now and the conflict about ``instance`` it
    raises is accepted in this annotator's name -- provided the difference is
    only the visible mask and the ``visibility`` label, and the frame did not
    already disagree about the instance before the write (``disputed``, from
    :func:`disputed_before`): that difference is somebody else's question,
    and accepting it would answer it unread. Its conflict stays open, brought
    up to date with these inputs.

    Nor does it sign pixels a write **reveals** (review round 4): a frame
    confirmed with an empty visible mask (完全) that now shows pixels (部分,
    撤回) is signed only when it is ``current`` -- the frame the annotator is
    looking at; any other is ``withheld``, left with its conflict for Review,
    because its row never said what is under the cable and the shape may have
    been redrawn since -- unless they are exactly the pixels the row was
    confirmed with before a 完全 replaced them (``Ctrl+Z`` with nothing moved
    in between, :func:`_confirmed_once`). Hiding pixels (完全) or changing a
    label over pixels that were confirmed is signed on every frame.

    Every frozen frame compiled here is taken off the re-check queue: it was
    compared with the very inputs a re-check would read, and showing it as
    "recheck" until the sweeper passes was showing a question already answered.
    The undo and the redo of the op go through here too (:func:`apply_cable`),
    so ``Ctrl+Z`` leaves the frame confirmed with its old row. ``held`` is the
    compilation the write's own settle made of the frame in hand.

    Returns ``(signed steps, withheld steps)``.
    """
    done: list[int] = []
    withheld: list[int] = []
    for step in frozen_among(db, desktop, view, steps):
        key = FrameKey(desktop, step, view)
        gen = db.recheck_generation(desktop, view, step)
        made = truth.refresh(key, want_compiled=True, held=held)["compiled"]
        if step not in disputed:
            for cid, name in db.open_conflicts_at(key):
                if name != instance:
                    continue
                conflict = db.get_conflict(cid) or {}
                labels = payload_labels(conflict.get("new_rle"))
                if {c.get("field") for c in labels} - {"visibility"}:
                    continue                     # a state or placement: not ours
                if (step != current and _reveals(conflict)
                        and not _confirmed_once(db, conflict)):
                    withheld.append(step)        # pixels nobody confirmed: Review's
                    continue
                truth.resolve_conflict(cid, "accept_new", annotator, prepared=made)
                done.append(step)
        if gen is not None:
            db.clear_recheck(desktop, view, step, gen)   # just compared: not owed one
    return done, withheld


def _signed(db: Db, truth: TruthService, desktop: int, view: str, instance: str, steps,
            annotator: str, disputed, out: dict, current: Optional[int]) -> dict:
    """``out`` after :func:`accept_own_label`: what it signed, what is still owed a check."""
    out["accepted"], out["withheld"] = accept_own_label(
        db, truth, desktop, view, instance, steps, annotator, disputed, out.get("frame"),
        current)
    out["conflicts"] = len(open_about(db, desktop, view, instance, steps))
    queued = set(db.rechecks(desktop, view))
    out["rechecks"] = [s for s in out.get("rechecks") or [] if s in queued]
    return out


def apply_cable(apply):
    """An undo/redo handler that keeps confirmed frames confirmed (:func:`accept_own_label`).

    Wraps the plain override handler: an op without :data:`CABLE_BY` -- a
    digit key, a single-frame mask -- is applied exactly as before.
    """
    def run(db: Db, truth: TruthService, payload: dict, current: Optional[int] = None) -> dict:
        by = payload.get(CABLE_BY)
        if not by:
            return apply(db, truth, payload, current)
        desktop, view, instance = payload["desktop"], payload["view"], payload["instance"]
        steps = ([int(f["step"]) for f in payload["frames"]] if "frames" in payload
                 else [int(payload["step"])])
        disputed = disputed_before(db, truth, desktop, view, instance, steps)
        stats = apply(db, truth, payload, current)
        return _signed(db, truth, desktop, view, instance, steps, by, disputed, stats,
                       current)

    return run


def cable_range(db: Db, truth: TruthService, key: FrameKey, instance: str,
                label: Optional[str], to_step: int, annotator: str = "system",
                occluder: str = OCCLUDER_CABLE) -> dict:
    """``label`` on every frame from this one to ``to_step`` (either way): one op.

    The frames are this view's annotatable ones (no ``ignore`` step, no frame
    without an image) inside this frame's pose segment; a ``to_step`` in
    another segment is refused. 完全/部分 is refused, naming them, when the
    part has no shape on some of them -- nothing is written then; 画框外 (task
    U39) wants no shape, only the part in its place on every frame. 撤回 goes
    over the frames that carry a 被线挡住 label or 画框外 and leaves the others
    alone.

    Returns what every edit returns plus ``label``, ``changed``, ``frames``
    (the steps written), ``withdrawn`` (what 撤回 took back, the labels of
    :func:`~tda.core.occluder.frame_label`), ``measured`` (frames whose image
    size had to be read off the image first -- once per frame, ever) and
    ``accepted`` (the frozen frames whose row now carries the label,
    :func:`accept_own_label`).
    """
    _refuse_gallery(key)
    cache = InputCache()
    low, high = range_bounds(db, key, cache)
    to_step = int(to_step)
    if not low <= to_step <= high:
        raise SessionRefusal(CABLE_OTHER_SEGMENT.format(step=to_step, low=low, high=high))
    first, last = sorted((key.step, to_step))
    seg = pose_segment_of(db, key, cache)
    steps = [s for s in annotatable_steps(db, key.desktop, key.view, range(first, last + 1))
             if pose_segment_of(db, FrameKey(key.desktop, s, key.view), cache) == seg]
    if label is not None:
        # once for the run; 画框外 reads no shape at all
        shapes = None if label == OFF_FRAME else db.keyframes(key.desktop, key.view, instance)
        refused: dict[str, list[int]] = {}
        for step in steps:
            why = label_reason(db, truth.tax, FrameKey(key.desktop, step, key.view),
                               instance, label, cache, shapes)
            if why:
                refused.setdefault(why, []).append(step)
        if refused:
            raise SessionRefusal("；".join(
                f"第 {_steps_text(found)} 步：{why}" for why, found in refused.items()))
    frames, before, measured = [], [], 0
    withdrawn: set[str] = set()
    for step in steps:
        here = FrameKey(key.desktop, step, key.view)
        previous = db.frame_overrides(here).get(instance)
        if label is None and frame_label(previous) is None:
            continue                     # 撤回 leaves a frame with no such label alone
        if label is None:
            withdrawn.add(str(frame_label(previous)))
        hw = (0, 0)
        if label == FULL:
            measured += not _measured(db, here)
            hw = frame_hw(db, here, truth.cache_dir)
        after = occluded_override(here, instance, previous, label, hw, occluder)
        new, old = override_state(after), override_state(previous)
        if _same(new, old):
            continue
        frames.append({"step": step} | new)
        before.append({"step": step} | old)
    written = [f["step"] for f in frames]
    if not frames:
        return _settled(db, truth, key, steps, None, label, changed=False, frames=[],
                        measured=measured, accepted=[], withdrawn=[])
    common = {"desktop": key.desktop, "view": key.view, "step": key.step,
              "instance": instance, "steps": written, CABLE_BY: annotator}
    payload = common | {"frames": frames}
    inverse = common | {"frames": before}
    disputed = disputed_before(db, truth, key.desktop, key.view, instance, written)
    with db.transaction():
        for state in frames:
            _write_override(db, FrameKey(key.desktop, state["step"], key.view), instance,
                            state)
        db.log_op(key.desktop, key.view, OP_RANGE, payload, inverse, annotator)
    out = _settled(db, truth, key, written, Op(OP_RANGE, payload, inverse), label,
                   frames=written, measured=measured, withdrawn=sorted(withdrawn))
    return _signed(db, truth, key.desktop, key.view, instance, written, annotator,
                   disputed, out, key.step)


def _same(a: dict, b: dict) -> bool:
    """Do two override states say the same thing (the RLE compared by content)?"""
    if bool(a["exists"]) != bool(b["exists"]):
        return False
    if not a["exists"]:
        return True
    return (a["visibility"] == b["visibility"] and a["occluder"] == b["occluder"]
            and masks.rle_counts(a["visible_rle"]) == masks.rle_counts(b["visible_rle"])
            and (a["visible_rle"] is None) == (b["visible_rle"] is None))


def _steps_text(steps: list[int]) -> str:
    """``3、5–7`` for the refusal that names the frames."""
    runs: list[list[int]] = []
    for step in sorted(steps):
        if runs and step == runs[-1][-1] + 1:
            runs[-1].append(step)
        else:
            runs.append([step])
    return "、".join(f"{r[0]}" if len(r) == 1 else f"{r[0]}–{r[-1]}" for r in runs)


def _settled(db: Db, truth: TruthService, key: FrameKey, steps, op: Optional[Op],
             label: Optional[str], changed: bool = True, **extra) -> dict:
    """:func:`tda.ui.session_edit._result`'s answer for a cable write (no op: no settle)."""
    if op is None:
        out = {"affected": [], "compiled": [], "rechecks": [], "conflicts": 0,
               "problems": {}, "frame": None, "op": None}
    else:
        affected = annotatable_steps(db, key.desktop, key.view, steps)
        stats = settle(db, truth, key.desktop, key.view, affected, key.step)
        out = {"affected": affected, "compiled": stats["compiled"],
               "rechecks": stats["rechecks"], "conflicts": stats["conflicts"],
               "problems": stats["problems"], "frame": stats["frame"], "op": op}
    out.update({"label": label, "changed": changed and op is not None}, **extra)
    return out


class CableMixin:
    """「被线挡住」 for :class:`tda.ui.session.AnnotationSession` (task U15)."""

    # -- reads ---------------------------------------------------------------
    def cable_label(self, instance: str) -> Optional[str]:
        """The 被线挡住 label of ``instance`` on the open frame, or ``None``."""
        return cable_label(self.db.frame_overrides(self.current()).get(str(instance)))

    def cable_labels(self) -> dict[str, str]:
        """``instance -> 完全/部分 label`` of every part 被线挡住 on the open frame."""
        try:
            found = self.db.frame_overrides(self.current()) or {}
        except Exception:  # noqa: BLE001 - a missing table is "none set"
            return {}
        return {name: label for name, fo in found.items()
                if (label := cable_label(fo)) is not None}

    def frame_label(self, instance: str) -> Optional[str]:
        """完全 / 部分 / 画框外 of ``instance`` on the open frame, or ``None`` (U39).

        What ``Ctrl+```'s dialog starts on (:func:`~tda.core.occluder.frame_label`).
        """
        return frame_label(self.db.frame_overrides(self.current()).get(str(instance)))

    def off_frame_parts(self) -> frozenset[str]:
        """The parts said 画框外 on the open frame (task U39); none on a gallery shot.

        There the same row is the gallery's own 「看不见」 (task U10), and on a
        框外零件 it is 「不在这张里」 (task U18): neither is listed.
        """
        if self.gallery:
            return frozenset()
        try:
            found = self.db.frame_overrides(self.current()) or {}
        except Exception:  # noqa: BLE001 - a missing table is "none set"
            return frozenset()
        return frozenset(name for name, fo in found.items()
                         if off_frame(fo) and not is_context_key(name))

    def cable_reasons(self) -> dict[str, str]:
        """``instance -> occluder`` of the labels :meth:`cable_labels` lists.

        ``cable`` for 被线挡住; ``viewpoint`` for 角度看不到, which Review's
        可见性审查 sets (task U24) -- the panels say which.
        """
        try:
            found = self.db.frame_overrides(self.current()) or {}
        except Exception:  # noqa: BLE001 - a missing table is "none set"
            return {}
        return {name: str(fo.occluder) for name, fo in found.items()
                if cable_label(fo) is not None}

    def cable_carry(self, instance: str) -> str:
        """Why this frame cannot carry a 被线挡住 label of ``instance``, ``""`` when it can.

        Read off the compiled frame the session already holds: not on this
        frame, on the bench, no shape.
        """
        found = self.compiled().instances.get(str(instance))
        if found is None:
            return CABLE_REMOVED
        if found.placement == "on_bench":
            return CABLE_ON_BENCH
        if found.keyframe_id is None:
            return CABLE_NEEDS_SHAPE
        return ""

    def cable_gate(self, instance: str) -> tuple[Optional[str], str]:
        """``(the label I would set, why it cannot)`` -- ``why`` is ``""`` when it may.

        撤回 always may: after 部分, after 画框外 (task U39: the key takes it
        back and never sets it), and on a frame that can no longer carry the
        label it holds, whatever that is -- the step table moved under it, the
        shape went (never via 部分 then). The pixels being drawn are the
        window's to judge first (it knows whether they are dirty);
        :meth:`cycle_cable` refuses them too, for 完全 and 部分.
        """
        if self.gallery:
            return None, CABLE_GALLERY
        held = self.db.frame_overrides(self.current()).get(str(instance))
        nxt = next_label(held)
        if nxt is None:
            return None, ""
        why = self.cable_carry(instance)
        if why and cable_label(held) is not None:
            return None, ""                      # a label it cannot carry: take it back
        return nxt, why

    def cable_bounds(self) -> tuple[int, int]:
        """``(low, high)`` of the steps ``Ctrl+I`` can reach from the open frame."""
        return range_bounds(self.db, self.current())

    # -- writes --------------------------------------------------------------
    def _refuse_cable_edit(self, instance: str, label: Optional[str]) -> None:
        if label is not None and self.layer.instance == str(instance) and self.layer.changed():
            raise SessionRefusal(CABLE_EDITING)

    def cycle_cable(self, instance: str) -> dict:
        """``I`` on a sequence frame: 完全 -> 部分 -> 撤回 (one op per press).

        The result says, besides what :func:`cable_one` does, ``stale`` -- why
        the frame could not carry the label any more, when that is what made
        this press a 撤回 -- and ``hand_mask_lost``: a 部分 or a 撤回 after a 完全
        that replaced a single-frame mask drawn by hand, which is not back
        (``Ctrl+Z`` to that 完全 finds it).
        """
        key = self.current()
        nxt, why = self.cable_gate(instance)
        self._refuse_cable_edit(instance, nxt)
        if why:
            raise SessionRefusal(why)
        # 画框外 needs no shape (U39): taking it back is no sign of a stale label
        stale = (self.cable_carry(instance)
                 if nxt is None and self.cable_label(instance) is not None else "")
        lost = nxt != FULL and self._hand_mask_lost(key, str(instance))
        result = cable_one(self.db, self.truth, key, str(instance), nxt, self.annotator)
        result.update(stale=stale, hand_mask_lost=lost)
        return self._after_edit(result)

    def _hand_mask_lost(self, key: FrameKey, instance: str) -> bool:
        """Did this cycle's 完全 replace a single-frame mask drawn by hand? (undo history)"""
        for op in reversed(self.undo_stack.ops):
            p = op.payload
            if op.kind != "set_frame_override" or (
                    p.get("desktop"), p.get("view"), p.get("step"), p.get("instance")) != (
                    key.desktop, key.view, key.step, instance):
                continue
            if not p.get(CABLE_BY) or not p.get("occluder"):
                return False                     # the cycle began after this write
            if p.get("visibility") == FULL:
                inverse = op.inverse
                return (inverse.get("visible_rle") is not None
                        and masks.rle_area(inverse["visible_rle"]) > 0
                        and not inverse.get("occluder"))
        return False

    def set_cable(self, instance: str, label: Optional[str]) -> dict:
        """One label on this frame, whatever it held (the cycle's one step, named)."""
        self._refuse_cable_edit(instance, label)
        return self._after_edit(cable_one(self.db, self.truth, self.current(), str(instance),
                                          label, self.annotator))

    def set_cable_range(self, instance: str, label: Optional[str], to_step: int) -> dict:
        """``Ctrl+I``: ``label`` from this frame to ``to_step``, one op (one ``Ctrl+Z``).

        ``label`` is 完全 / 部分 / 画框外 (task U39) or ``None`` (撤回).
        """
        self._refuse_cable_edit(instance, label)
        return self._after_edit(cable_range(self.db, self.truth, self.current(),
                                            str(instance), label, int(to_step),
                                            self.annotator))

    # -- what the canvas draws -----------------------------------------------
    def cable_outlines(self) -> list[tuple[str, list[list[tuple[float, float]]]]]:
        """``[(instance, polygons)]``: the amodal outline of every 完全 part (U15).

        Its visible mask is empty, so the overlay paints nothing; the canvas
        draws this dashed instead, so the annotator sees it is there but
        hidden. Image coordinates; parts hidden with ``H`` are left out.
        Cached on the compiled frame, which a write replaces.
        """
        compiled = self.compiled()
        hidden = frozenset(self._hidden)
        held = self.__dict__.get("_cable_outline_cache")
        if held is not None and held[0] is compiled and held[1] == hidden:
            return held[2]
        out: list[tuple[str, list[list[tuple[float, float]]]]] = []
        for instance, label in sorted(self.cable_labels().items()):
            found = compiled.instances.get(instance)
            if (label != FULL or instance in self._hidden or found is None
                    or found.amodal is None):
                continue
            polygons = _outline(found.amodal, getattr(found, "amodal_window", None))
            if polygons:
                out.append((instance, polygons))
        # the compilation itself is held, not its id: a freed one's id is reused
        self._cable_outline_cache = (compiled, hidden, out)
        return out


def _outline(amodal: np.ndarray, window) -> list[list[tuple[float, float]]]:
    """Outer contours of ``amodal`` (inside ``window`` when known), image coordinates."""
    x0, y0 = 0, 0
    mask = amodal
    if window is not None:
        x0, y0, x1, y1 = (int(v) for v in window)
        if x1 <= x0 or y1 <= y0:
            return []
        mask = amodal[y0:y1, x0:x1]
    out = []
    for flat in masks.mask_to_polygons(mask):
        out.append([(float(flat[i]) + x0, float(flat[i + 1]) + y0)
                    for i in range(0, len(flat) - 1, 2)])
    return out
