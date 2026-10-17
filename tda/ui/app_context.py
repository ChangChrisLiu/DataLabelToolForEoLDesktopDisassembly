"""The window's 框外零件 (task U18, spec 3.2c): make one, draw it, say it is not here.

* **making one** -- the canvas' right-click menu has 「框外零件…」 (and so has the
  「＋ 添加零件」 dialog of Steps mode, which comes back to Annotate first): the
  dialog (:class:`~tda.ui.panels.context_object.ContextObjectDialog`) asks
  what it is, the session makes it (one op, ``Ctrl+Z`` takes it back), and the
  window selects it and starts its edit layer at once -- brush, polygon,
  circle, SAM point or box, whichever is armed; SAM gets no prompt box for it
  (the guesses are about the machine's parts, inside the ROI). ``Enter``
  commits a keyframe as for any part.
* **I** on one -- 「不在这张里」 (the gallery's 看不见: ``out_of_view``, no
  pixels), again to take it back; ``Ctrl+I`` the same over a span
  (:class:`ContextRangeDialog`). The instance list says 「不在」.

The right button is SAM's negative point while a SAM tool is drawing a part;
the menu does not open then.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMenu,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tda.core.model import is_context_key
from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui import context_text as T

__all__ = ["ContextMixin", "ContextRangeDialog", "NO_FRAME", "WRONG_MODE"]

#: 「框外零件…」 outside Annotate mode, and with no frame to draw on.
WRONG_MODE = "切到标注模式再新建框外零件 / switch to Annotate mode"
NO_FRAME = "这一帧没有图像：框外零件要画在有图的帧上 / no image here"
#: The SAM tools take the right button as a negative point.
_SAM_TOOLS = ("sam_point", "sam_box")


class ContextRangeDialog(QDialog):
    """``Ctrl+I`` on a 框外零件: 「不在」 or 撤回, and the step the span runs to."""

    def __init__(self, parent: Optional[QWidget], label: str, step: int,
                 absent: bool, low: int, high: int) -> None:
        super().__init__(parent)
        self.setWindowTitle(T.RANGE_TITLE)
        self.setModal(True)
        self.what = QComboBox()
        self.what.addItem("不在这些帧里", True)
        self.what.addItem("撤回「不在」", False)
        self.what.setCurrentIndex(0 if absent else 1)
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
        layout.addWidget(self.what)
        layout.addWidget(self.step_box)
        layout.addWidget(QLabel("这一段里画着它的帧都标「不在」（没画它的帧本来就没有它）。"))
        layout.addWidget(buttons)
        self.step_box.setFocus()
        self.step_box.selectAll()

    def answer(self) -> tuple[bool, int]:
        return bool(self.what.currentData()), int(self.step_box.value())


class ContextMixin:
    """框外零件 for :class:`tda.ui.app.MainWindow` (task U18)."""

    def _init_context(self) -> None:
        """The canvas' right-click menu."""
        self._last_context_class: Optional[str] = None
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._on_canvas_menu)

    # ------------------------------------------------------------- the menu
    def canvas_menu(self) -> QMenu:
        """The canvas' right-click menu (not shown yet)."""
        menu = QMenu(self.canvas)
        action = menu.addAction(T.MENU_TEXT, lambda: self.act_context_object())
        action.setToolTip(T.BUTTON_TIP)
        menu.setToolTipsVisible(True)
        return menu

    def canvas_menu_wanted(self) -> bool:
        """Does a right click open the menu now? Not while SAM takes it as a point."""
        if self.mode != A.MODE_ANNOTATE or self.is_flashing():
            return False
        return not (self._tool_name in _SAM_TOOLS
                    and getattr(self.session, "editing_instance", None) is not None)

    @S.guard
    def _on_canvas_menu(self, pos) -> None:
        if not self.canvas_menu_wanted():
            return
        self.canvas_menu().exec(self.canvas.mapToGlobal(pos))

    # ------------------------------------------------------------- make one
    def context_refusal(self) -> str:
        """Why 「框外零件…」 cannot run now, ``""`` if it may."""
        if self.mode != A.MODE_ANNOTATE:
            return WRONG_MODE
        if not compat.is_open(self.session) or self.session.image() is None:
            return NO_FRAME
        return ""

    def ask_context_object(self) -> Optional[dict]:
        """Show the dialog; ``None`` when cancelled (a method, so a test answers it)."""
        from tda.ui.panels.context_object import ContextObjectDialog

        dialog = ContextObjectDialog(self.session.tax, self, cls=self._last_context_class)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.values()

    @S.guard
    def act_context_object(self) -> Optional[str]:
        """「框外零件…」: name it, make it, and start drawing it. Returns its key."""
        why = self.context_refusal()
        if why:
            self.report(why)
            return None
        if not self.can_leave_edit():
            return None
        answer = self.ask_context_object()
        if not answer:
            return None
        try:
            key = self.session.add_context_object(**answer)
        except ValueError as refused:            # SessionRefusal is one
            self.report(str(refused))
            return None
        self._last_context_class = str(answer.get("cls") or "") or None
        self.instances.refresh()
        self.task_card.refresh()
        self.on_request_edit(key)
        label = self._label_of(key)[0] or key
        self.report(T.CREATED.format(label=label))
        return key

    @S.guard
    def on_context_changed(self) -> None:
        """Steps mode deleted a 框外零件, written through: the frame in hand follows."""
        changed = getattr(self.session, "context_changed", None)
        if callable(changed) and compat.is_open(self.session):
            changed()

    def open_context_from_steps(self) -> None:
        """The 「＋ 添加零件」 dialog's 「框外零件…」: to Annotate, then as from the canvas."""
        self.set_mode(A.MODE_ANNOTATE)
        if self.mode == A.MODE_ANNOTATE:
            self.act_context_object()

    # ------------------------------------------------------------- I
    def context_target(self, instance: Optional[str] = None) -> Optional[str]:
        """The 框外零件 ``I`` acts on: the one asked for, selected or edited -- if it is one."""
        for found in (instance, self.instances.selected_instance(),
                      getattr(self.session, "editing_instance", None)):
            if found:
                return str(found) if is_context_key(found) else None
        return None

    def context_absent_refusal(self, instance: str, dirty: Optional[bool] = None) -> str:
        """Why ``I`` would do nothing on this 框外零件, ``""`` if it may."""
        gate = getattr(self.session, "context_absent_gate", None)
        if not callable(gate):
            return "这个会话不支持 / not supported by this session"
        marked, why = gate(instance)
        if why:
            return why
        if not marked and getattr(self.session, "editing_instance", None) == instance and (
                self.has_uncommitted_edit() if dirty is None else dirty):
            return f"{self._label_of(instance)[0] or instance} 正在画：先 Esc 放弃，再按 I"
        return ""

    @S.guard
    def act_context_absent(self, instance: str) -> None:
        """``I`` on a 框外零件: 「不在这张里」, again to take it back."""
        why = self.context_absent_refusal(instance)
        if why:
            self.report(why)
            return
        label = self._label_of(instance)[0] or instance
        editing = getattr(self.session, "editing_instance", None) == instance
        try:
            result = self.session.toggle_context_absent(instance)
        except ValueError as refused:
            self.report(f"{label}：{refused}")
            return
        if editing:
            self.set_sam_instance(None)
            self._sync_editing_layer()
        self._after_context(T.ABSENT_DONE.format(label=label) if result.get("absent")
                            else T.ABSENT_BACK.format(label=label))

    @S.guard
    def act_context_absent_range(self, instance: str) -> None:
        """``Ctrl+I`` on a 框外零件: 「不在」 (or 撤回) from this frame to step N."""
        label = self._label_of(instance)[0] or instance
        low, high = self.session.cable_bounds()
        marked = instance in self.session.context_absent_here()
        answer = self.ask_context_range(instance, not marked, low, high)
        if answer is None:
            return
        absent, to_step = answer
        try:
            result = self.session.set_context_absent_range(instance, absent, int(to_step))
        except ValueError as refused:
            self.report(f"{label}：{refused}")
            return
        frames = result.get("frames") or []
        if not frames:
            self.report(f"{label}：这一段本来就是这样，没有改")
            return
        first, last = min(frames), max(frames)
        where = f"第 {first} 步" if first == last else f"第 {first}–{last} 步"
        self._sync_editing_layer()
        self._after_context((T.ABSENT_RANGE_DONE if absent else T.ABSENT_RANGE_OFF)
                            .format(label=label, where=where))

    def ask_context_range(self, instance: str, absent: bool, low: int,
                          high: int) -> Optional[tuple[bool, int]]:
        """Show the ``Ctrl+I`` dialog; ``None`` when cancelled (a test answers it)."""
        label = self._label_of(instance)[0] or instance
        dialog = ContextRangeDialog(self, label, self.session.current().step, absent,
                                    low, high)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.answer()

    def _after_context(self, note: str) -> None:
        """Repaint what reads the label, and say what changed."""
        self.refresh_overlay()
        self.instances.refresh()
        self.task_card.refresh()
        self.timeline.refresh_statuses()
        self.update_status()
        self.report(note, hold_ms=6000)
