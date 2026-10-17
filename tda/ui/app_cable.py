"""The window's 「被线挡住」 (task U15): ``I`` and ``Ctrl+I`` on a sequence frame.

Cables are not annotated. A part in its place under one keeps its shape -- drawn
here or carried from another frame -- and the frame says why it is not seen:
``I`` on the selected part cycles 被线挡住（完全）-> 被线挡住（部分）-> 撤回
(:mod:`tda.core.occluder`), ``Ctrl+I`` sets one of them from this frame to a
step of the same pose segment in one op. On a components-gallery shot ``I`` is
still 「看不见」 (:mod:`tda.ui.app_gallery`, which hands the sequence to
:meth:`CableMixin.act_cable`); the palette's 「看不见」 button is the same slot,
so it is live on a sequence frame and greyed with these refusals.

Where it shows: the instance table's tag (线挡 / 线半挡), the card row's
「⚠ 被线挡住（完全/部分）」, and on the canvas the dashed amodal outline of a
完全 part, whose visible mask is empty.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tda.core.occluder import CYCLE, FULL, PARTIAL
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.app_state import REMOVED_ROW
from tda.ui.canvas.overlay import palette_color
from tda.ui.session_cable import CABLE_EDITING, CABLE_GALLERY, LABEL_ZH

__all__ = ["CABLE_DONE", "CABLE_NO_TARGET", "CableMixin", "CableRangeDialog", "cable_note",
           "range_note"]

#: ``I`` / ``Ctrl+I`` on a sequence frame with nothing to act on. It names the
#: gallery's meaning too: on a gallery shot the same key is 「看不见」 (U10).
PICK_FIRST = ("I 在拆解序列里标「被线挡住」：先在实例表选中零件"
              "（「不在这张里」只用在零件合照里）")
CABLE_NO_TARGET = f"{PICK_FIRST} / select a part first"
#: What each step of the cycle says afterwards, ``{label}`` the part.
CABLE_DONE = {
    FULL: "{label} 被线挡住（完全）：形状保留，这一帧看不见它（再按 I 改成部分，Ctrl+Z 撤销）",
    PARTIAL: "{label} 被线挡住（部分）：看得见的像素照旧（再按 I 撤回，Ctrl+Z 撤销）",
    None: "{label}：撤回被线挡住（Ctrl+Z 撤销）",
}


def cable_note(label: str, value: Optional[str]) -> str:
    """The status line after ``I``."""
    return CABLE_DONE[value].format(label=label)


def range_note(label: str, value: Optional[str], steps: list[int]) -> str:
    """The status line after ``Ctrl+I``: the frames written and what they now say."""
    first, last = min(steps), max(steps)
    where = f"第 {first} 步" if first == last else f"第 {first}–{last} 步"
    count = f"（{len(steps)} 帧）" if len(steps) > 1 and last - first + 1 != len(steps) else ""
    if value is None:
        return f"{label} {where}{count}：撤回被线挡住（Ctrl+Z 一次撤销）"
    return f"{label} {where}{count}都标为被线挡住（{LABEL_ZH[value]}）（Ctrl+Z 一次撤销）"


class CableRangeDialog(QDialog):
    """``Ctrl+I``: which label, and the step the span runs to (U15)."""

    def __init__(self, parent: Optional[QWidget], label: str, step: int,
                 value: Optional[str], low: int, high: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("一段帧被线挡住 / hidden by a cable over a span")
        self.setModal(True)
        self.label_box = QComboBox()
        for choice in CYCLE:
            self.label_box.addItem(f"被线挡住（{LABEL_ZH[choice]}）" if choice else
                                   "撤回被线挡住", choice)
        self.label_box.setCurrentIndex(list(CYCLE).index(value) if value in CYCLE else 0)
        self.step_box = QSpinBox()
        self.step_box.setRange(int(low), int(high))
        self.step_box.setValue(int(step))
        self.step_box.setPrefix("一直到第 ")
        self.step_box.setSuffix(" 步")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{label}：从第 {step} 步起（这一位姿段第 {low}–{high} 步）"))
        layout.addWidget(self.label_box)
        layout.addWidget(self.step_box)
        layout.addWidget(QLabel("这一段每一帧都这样标；没有形状的帧不行（先画出形状或从别的帧带过来）。"))
        layout.addWidget(buttons)
        self.step_box.setFocus()
        self.step_box.selectAll()

    def answer(self) -> tuple[Optional[str], int]:
        return self.label_box.currentData(), int(self.step_box.value())


class CableMixin:
    """「被线挡住」 for :class:`tda.ui.app.MainWindow` (task U15)."""

    # ------------------------------------------------------------- gates
    def _cable_target(self, instance: Optional[str]) -> Optional[str]:
        """The part ``I`` acts on: the one asked for, selected, edited, on the card."""
        return self._cable_pick(instance)[0]

    def _cable_pick(self, instance: Optional[str]) -> tuple[Optional[str], bool]:
        """``(part, from the card)``: the card's highlighted row is the last resort.

        The card highlights its first open row by itself, so a part found only
        there was not chosen by anybody: when it cannot take the label the
        answer is "select a part" (:meth:`cable_refusal`), not its own reason.
        """
        for found in (instance, self.instances.selected_instance(),
                      getattr(self.session, "editing_instance", None)):
            if found:
                return str(found), False
        found = self.task_card.current_instance()
        return (str(found), True) if found else (None, False)

    def cable_refusal(self, instance: Optional[str] = None,
                      dirty: Optional[bool] = None, cached: bool = True) -> str:
        """Why ``I`` would do nothing on this sequence frame, ``""`` if it may.

        The gates of :meth:`act_cable` in its order and words, for the key and
        the palette's 「看不见」 button (``dirty``: the editing layer has pixels
        nobody committed, when the caller knows it). The session's own answer
        -- no shape, not on this frame, on the bench -- is asked once per
        frame, part and re-read of the instance list (``cached``), as the other
        「状态」 buttons ask theirs; the key asks it afresh.
        """
        if not compat.is_open(self.session):
            return CABLE_NO_TARGET
        if getattr(self.session, "gallery", False):
            return CABLE_GALLERY
        if instance is None and self.instances.selected_is_removed():
            return REMOVED_ROW
        target, from_card = self._cable_pick(instance)
        if target is None:
            return CABLE_NO_TARGET
        label = self._label_of(target)[0] or target
        if getattr(self.session, "editing_instance", None) == target and (
                self.has_uncommitted_edit() if dirty is None else dirty):
            return f"{label}：{CABLE_EDITING}"
        gate = getattr(self.session, "cable_gate", None)
        if not callable(gate):
            return "这个会话不支持 / not supported by this session"
        if cached:
            why = self._why_once("cable", target, lambda part: gate(part)[1])
        else:
            try:
                why = gate(target)[1]
            except Exception:  # noqa: BLE001 - the write itself will say what is wrong
                why = ""
        if why and from_card:
            return f"{PICK_FIRST}；任务卡上亮着的 {label}：{why}"
        return f"{label}：{why}" if why else ""

    # ------------------------------------------------------------- I
    @S.guard
    def act_cable(self, instance: Optional[str] = None) -> None:
        """``I`` on a sequence frame: the part's 被线挡住 label, the next one round."""
        why = self.cable_refusal(instance, cached=False)
        if why:
            self.report(why)
            return
        target = self._cable_target(instance)
        label = self._label_of(target)[0] or target
        self._drop_clean_layer(target)
        try:
            result = self.session.cycle_cable(target)
        except ValueError as refused:            # SessionRefusal is one
            self.report(f"{label}：{refused}")
            return
        self._after_cable(cable_note(label, result.get("label")))

    def _drop_clean_layer(self, target: str) -> None:
        """An untouched layer of the part goes first: there is nothing to lose."""
        if getattr(self.session, "editing_instance", None) == target \
                and not self.has_uncommitted_edit():
            self.session.clear_edit()
            self.set_sam_instance(None)
            self._sync_editing_layer()

    # ------------------------------------------------------------- Ctrl+I
    @S.guard
    def act_cable_range(self) -> None:
        """``Ctrl+I``: the selected part's 被线挡住 label from this frame to step N."""
        if getattr(self.session, "gallery", False):
            self.report(CABLE_GALLERY)
            return
        if self.instances.selected_is_removed():
            self.report(REMOVED_ROW)
            return
        target = self._cable_target(None)
        if target is None:
            self.report(CABLE_NO_TARGET.replace("I 在", "Ctrl+I 在", 1))
            return
        label = self._label_of(target)[0] or target
        if getattr(self.session, "editing_instance", None) == target \
                and self.has_uncommitted_edit():
            self.report(f"{label}：{CABLE_EDITING.replace('再按 I', '再按 Ctrl+I')}")
            return
        low, high = self.session.cable_bounds()
        current = self.session.cable_label(target)
        answer = self.ask_cable_range(target, current or FULL, low, high)
        if answer is None:
            return
        value, to_step = answer
        self._drop_clean_layer(target)
        try:
            result = self.session.set_cable_range(target, value, int(to_step))
        except ValueError as refused:            # SessionRefusal is one
            self.report(f"{label}：{refused}")
            return
        frames = result.get("frames") or []
        if not frames:
            self.report(f"{label}：这一段本来就是这样，没有改")
            return
        self._after_cable(range_note(label, value, frames))

    def ask_cable_range(self, instance: str, value: Optional[str], low: int,
                        high: int) -> Optional[tuple[Optional[str], int]]:
        """Show the ``Ctrl+I`` dialog; ``None`` when the annotator cancelled.

        A method so a test can answer it without a modal loop (as
        :meth:`~tda.ui.app_state.StateMixin.ask_state_range`).
        """
        label = self._label_of(instance)[0] or instance
        dialog = CableRangeDialog(self, label, self.session.current().step, value, low, high)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.answer()

    # ------------------------------------------------------------- after
    def _after_cable(self, note: str) -> None:
        """Repaint what reads the label, and say what changed."""
        self.refresh_overlay()
        self.instances.refresh()
        self.task_card.refresh()
        self.timeline.refresh_statuses()
        self.update_status()
        self.report(note, hold_ms=6000)
        self.flash_visibility_note(note)

    def refresh_cable_outlines(self) -> None:
        """The dashed outlines of the parts 被线挡住（完全） on this frame (the canvas)."""
        outlines = getattr(self.session, "cable_outlines", None)
        setter = getattr(self.canvas, "set_cable_outlines", None)
        if setter is None:
            return
        found: list = []
        if callable(outlines) and compat.is_open(self.session):
            try:
                found = outlines()
            except Exception:  # noqa: BLE001 - an outline is never worth a failure
                found = []
        setter([(polygons, palette_color(name)) for name, polygons in found])
