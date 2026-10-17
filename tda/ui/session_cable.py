"""「被线挡住」 in the session (task U15): ``I`` and ``Ctrl+I`` on a sequence frame.

Cables are not annotated, and a part in its place is sometimes under one. The
annotator says so on the frame (:mod:`tda.core.occluder` is the rule): ``I``
cycles the selected part 完全 -> 部分 -> 撤回, one ``set_frame_override`` op per
press; ``Ctrl+I`` writes one of the three over a run of frames as **one** op,
``set_frame_overrides`` -- one row per frame and one inverse, so one ``Ctrl+Z``.

Both are refused where the label would describe nothing: a part with no shape
on the frame (完全 keeps the amodal shape, so there has to be one -- it may be
carried from another frame as usual), a part that is not on the frame at all
(taken out), one lying on the bench (not in its place), and the part whose
pixels are being drawn. 撤回 is never refused for a missing shape: it is how a
stored label that lost its shape is repaired. The components gallery has no
cable label at all -- ``I`` means 「看不见」 there (task U10).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tda.core import masks
from tda.core.db import Db
from tda.core.model import OCCLUDER_CABLE, FrameKey, is_component_view
from tda.core.occluder import (
    FULL,
    PARTIAL,
    cable_label,
    next_label,
    occluded_override,
    shape_applies,
)
from tda.core.truth import TruthService
from tda.core.truth_inputs import (
    InputCache,
    annotatable_steps,
    frame_hw,
    pose_segment_of,
    view_steps,
)
from tda.ui.commands import Op
from tda.ui.gallery import GALLERY_REFUSALS
from tda.ui.session_api import SessionRefusal
from tda.ui.session_ops import _write_override, override_state, settle

__all__ = [
    "CABLE_EDITING", "CABLE_GALLERY", "CABLE_NEEDS_SHAPE", "CABLE_ON_BENCH",
    "CABLE_OTHER_SEGMENT", "CABLE_REMOVED", "CableMixin", "LABEL_ZH", "OP_RANGE",
    "TAG_ZH", "cable_one", "cable_range", "cable_reason", "range_bounds",
]

#: The op ``Ctrl+I`` logs: one instance's override on several frames.
OP_RANGE = "set_frame_overrides"
#: The three steps of the cycle as the status line and the dialog say them.
LABEL_ZH: dict[Optional[str], str] = {FULL: "完全", PARTIAL: "部分", None: "撤回"}
#: The instance table's tag of each label (the Vis column is 46 px wide).
TAG_ZH = {FULL: "线挡", PARTIAL: "线半挡"}

#: No shape applies here (the ``OVERRIDE_NEEDS_SHAPE`` idea, U2d).
CABLE_NEEDS_SHAPE = "先画出它的形状（可以从别的帧带过来），再按 I 标被线挡住"
#: The frame does not need the part at all.
CABLE_REMOVED = "这一帧不要它的形状（已拆下，或这个状态不画）：没有可标的"
#: On the staging area: a rectangle, and not in its place.
CABLE_ON_BENCH = "它放在台面上（台面框），不在原位：被线挡住只标原位的零件"
#: Its pixels are being drawn.
CABLE_EDITING = "它正在画、还没提交：先 Enter 提交或 Esc 放弃，再按 I"
#: ``Ctrl+I`` on a gallery shot: ``I`` means 看不见 there.
CABLE_GALLERY = GALLERY_REFUSALS["cable_range"]
#: ``Ctrl+I`` to a step of another pose segment.
CABLE_OTHER_SEGMENT = ("第 {step} 步在另一个位姿段（这一段是第 {low}–{high} 步）："
                       "Ctrl+I 只在一个位姿段里")

_REASONS = {"removed": CABLE_REMOVED, "bench": CABLE_ON_BENCH,
            "no_shape": CABLE_NEEDS_SHAPE}


def cable_reason(db: Db, tax, key: FrameKey, instance: str,
                 cache: Optional[InputCache] = None) -> str:
    """Why a 完全/部分 label cannot go on ``instance`` at ``key``, ``""`` when it can."""
    return _REASONS.get(shape_applies(db, tax, key, instance, cache), "")


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


def cable_one(db: Db, truth: TruthService, key: FrameKey, instance: str,
              label: Optional[str], annotator: str = "system",
              occluder: str = OCCLUDER_CABLE) -> dict:
    """Set ``label`` (完全 / 部分 / ``None`` = 撤回) on this frame: one ``set_frame_override`` op.

    Returns what every edit returns plus ``label`` and ``changed``; nothing is
    written (and no op made) when the frame already says exactly that.
    """
    _refuse_gallery(key)
    if label is not None:
        why = cable_reason(db, truth.tax, key, instance)
        if why:
            raise SessionRefusal(why)
    previous = db.frame_overrides(key).get(instance)
    after = occluded_override(key, instance, previous, label,
                              frame_hw(db, key, truth.cache_dir), occluder)
    common = {"desktop": key.desktop, "view": key.view, "step": key.step,
              "instance": instance, "steps": [key.step]}
    payload = common | override_state(after)
    inverse = common | override_state(previous)
    if _same(payload, inverse):
        return _settled(db, truth, key, [key.step], None, label, changed=False)
    with db.transaction():
        _write_override(db, key, instance, payload)
        db.log_op(key.desktop, key.view, "set_frame_override", payload, inverse, annotator)
    return _settled(db, truth, key, [key.step], Op("set_frame_override", payload, inverse),
                    label)


def cable_range(db: Db, truth: TruthService, key: FrameKey, instance: str,
                label: Optional[str], to_step: int, annotator: str = "system",
                occluder: str = OCCLUDER_CABLE) -> dict:
    """``label`` on every frame from this one to ``to_step`` (either way): one op.

    The frames are this view's annotatable ones (no ``ignore`` step, no frame
    without an image) inside this frame's pose segment; a ``to_step`` in
    another segment is refused. 完全/部分 is refused, naming them, when the
    part has no shape on some of them -- nothing is written then. 撤回 goes
    over the frames that carry a 被线挡住 label and leaves the others alone.

    Returns what every edit returns plus ``label``, ``changed`` and ``frames``
    (the steps written).
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
        refused: dict[str, list[int]] = {}
        for step in steps:
            why = cable_reason(db, truth.tax, FrameKey(key.desktop, step, key.view),
                               instance, cache)
            if why:
                refused.setdefault(why, []).append(step)
        if refused:
            raise SessionRefusal("；".join(
                f"第 {_steps_text(found)} 步：{why}" for why, found in refused.items()))
    frames, before = [], []
    for step in steps:
        here = FrameKey(key.desktop, step, key.view)
        previous = db.frame_overrides(here).get(instance)
        if label is None and cable_label(previous) is None:
            continue                     # 撤回 leaves a frame with no cable label alone
        after = occluded_override(here, instance, previous, label,
                                  frame_hw(db, here, truth.cache_dir), occluder)
        new, old = override_state(after), override_state(previous)
        if _same(new, old):
            continue
        frames.append({"step": step} | new)
        before.append({"step": step} | old)
    written = [f["step"] for f in frames]
    if not frames:
        return _settled(db, truth, key, steps, None, label, changed=False, frames=[])
    common = {"desktop": key.desktop, "view": key.view, "step": key.step,
              "instance": instance, "steps": written}
    payload = common | {"frames": frames}
    inverse = common | {"frames": before}
    with db.transaction():
        for state in frames:
            _write_override(db, FrameKey(key.desktop, state["step"], key.view), instance,
                            state)
        db.log_op(key.desktop, key.view, OP_RANGE, payload, inverse, annotator)
    return _settled(db, truth, key, written, Op(OP_RANGE, payload, inverse), label,
                    frames=written)


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

    def cable_gate(self, instance: str) -> tuple[Optional[str], str]:
        """``(the label I would set, why it cannot)`` -- ``why`` is ``""`` when it may.

        Read off the compiled frame the session already holds: no shape, not
        on this frame, on the bench. The pixels being drawn are the window's to
        judge first (it knows whether they are dirty); :meth:`cycle_cable`
        refuses them too.
        """
        if self.gallery:
            return None, CABLE_GALLERY
        nxt = next_label(self.db.frame_overrides(self.current()).get(str(instance)))
        if nxt is None:
            return None, ""                      # 撤回 is always allowed
        found = self.compiled().instances.get(str(instance))
        if found is None:
            return nxt, CABLE_REMOVED
        if found.placement == "on_bench":
            return nxt, CABLE_ON_BENCH
        if found.keyframe_id is None:
            return nxt, CABLE_NEEDS_SHAPE
        return nxt, ""

    def cable_bounds(self) -> tuple[int, int]:
        """``(low, high)`` of the steps ``Ctrl+I`` can reach from the open frame."""
        return range_bounds(self.db, self.current())

    # -- writes --------------------------------------------------------------
    def _refuse_cable_edit(self, instance: str) -> None:
        if self.layer.instance == str(instance) and self.layer.changed():
            raise SessionRefusal(CABLE_EDITING)

    def cycle_cable(self, instance: str) -> dict:
        """``I`` on a sequence frame: 完全 -> 部分 -> 撤回 (one op per press)."""
        self._refuse_cable_edit(instance)
        nxt, why = self.cable_gate(instance)
        if why:
            raise SessionRefusal(why)
        return self._after_edit(cable_one(self.db, self.truth, self.current(), str(instance),
                                          nxt, self.annotator))

    def set_cable(self, instance: str, label: Optional[str]) -> dict:
        """One label on this frame, whatever it held (the cycle's one step, named)."""
        self._refuse_cable_edit(instance)
        return self._after_edit(cable_one(self.db, self.truth, self.current(), str(instance),
                                          label, self.annotator))

    def set_cable_range(self, instance: str, label: Optional[str], to_step: int) -> dict:
        """``Ctrl+I``: ``label`` from this frame to ``to_step``, one op (one ``Ctrl+Z``)."""
        self._refuse_cable_edit(instance)
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
