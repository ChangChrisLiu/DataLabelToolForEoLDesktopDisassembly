"""``T`` and the instance list's state combo: the state the picture shows (task U8).

On D13/scan ``ram_latch.01`` is opened at step 14 by the log and is visibly
shut again on a few middle frames.  There is no verb for that and there must
not be one -- a clip springing back is not a disassembly step -- so the
annotator says what the frame shows: select the part, press ``T`` (or pick the
state in the instance list), and the session writes "at this step it is
closed" with the next step put back as it was
(:meth:`tda.ui.session_state.StateMixin.assert_state`).  The task card then
shows the ✂ rows of the new boundaries and ``Ctrl+K`` draws them, exactly as
for a boundary the step table made.

The trial annotator's attempt -- re-drawing the shape at frame 21 with
``Enter`` -- re-traced the version covering 14-41 and raised 48 conflicts.  The
status line therefore says, after every change, which frames now hold which
state and whether this frame's shape has to be drawn again with ``Ctrl+K``.
"""
from __future__ import annotations

from typing import Optional

from tda.ui import app_support as S
from tda.ui import session_api as api
from tda.ui.class_names import state_zh

__all__ = ["NO_TARGET", "SPLIT_NOW", "StateMixin", "state_note"]

#: ``T`` with nothing selected and nothing being drawn.
NO_TARGET = ("T 改的是实例表里选中的零件（或正在画的零件）：先在右下实例表点一行 / "
             "select a part in the instance table first")
#: Said after a change that leaves an open ✂ row on this frame.
SPLIT_NOW = "这帧的形状要重画：Ctrl+K 从这帧起新版本"
#: Prefix of a refusal.
REFUSED = "改不了状态："


def state_note(label: str, patch, split_here: bool,
               run: Optional[tuple[int, int]] = None,
               after: Optional[tuple[int, str]] = None) -> str:
    """The status line after an assertion, in the annotator's words.

    ``内存卡扣 ram_latch.01 在第 21 步：关上（之前为 打开）；第 22 步起恢复 打开``;
    ``run`` (the steps now sharing the state) and ``after`` (the first step past
    them) make a run of frames read as one: ``…；第 20–21 步都是 关上；第 22 步起
    恢复 打开``.  When this frame's card now asks for a new shape version,
    :data:`SPLIT_NOW`.
    """
    text = (f"{label} 在第 {patch.step} 步：{state_zh(patch.state)}"
            f"（之前为 {state_zh(patch.previous)}）")
    if run is None and patch.next_step is not None and patch.restores is not None:
        run = (patch.step, patch.step)
        after = (patch.next_step, patch.restores) if patch.restores != patch.state else None
    if run is not None and run[0] != run[1]:
        text += f"；第 {run[0]}–{run[1]} 步都是 {state_zh(patch.state)}"
    if after is not None:
        back = "恢复 " if after[1] == patch.previous else ""
        text += f"；第 {after[0]} 步起{back}{state_zh(after[1])}"
    if split_here:
        text += f"。{SPLIT_NOW}"
    return text + "（Ctrl+Z 撤销）"


class StateMixin:
    """``T`` / the state combo, for :class:`tda.ui.app.MainWindow`."""

    def _init_state(self) -> None:
        self.instances.sigAssertState.connect(self.on_state_picked)

    def _state_target(self) -> Optional[str]:
        """The part ``T`` is about: the instance list's selection, else the edit's."""
        return (self.instances.selected_instance()
                or getattr(self.session, "editing_instance", None))

    @S.guard
    def act_cycle_state(self) -> None:
        """``T``: the selected part's state at this frame becomes the next one."""
        if not self._editable_row():
            return
        instance = self._state_target()
        if not instance:
            self.report(NO_TARGET)
            return
        if not callable(getattr(self.session, "assert_state", None)):
            self.report(f"{REFUSED}这个会话不支持 / not supported by this session")
            return
        nxt = self.session.next_state(instance)
        if nxt is None:
            self.report(REFUSED + self.session.state_gate(instance)[1])
            return
        self.assert_state_here(instance, nxt)

    @S.guard
    def on_state_picked(self, instance: str, state: str) -> None:
        """The instance list's combo: the same gesture as ``T``, to a named state."""
        self.assert_state_here(instance, state)

    def assert_state_here(self, instance: str, state: str) -> Optional[dict]:
        """Write the state, repaint what reads it, and say what changed."""
        key = self.session.current()
        try:
            result = self.session.assert_state(instance, state)
        except ValueError as refused:            # SessionRefusal is one
            self.logger.info("assert_state refused %s step %s -> %s: %s",
                             instance, key.step, state, refused)
            self.report(f"{REFUSED}{refused}")
            self.instances.refresh()             # the combo goes back to what holds
            return None
        label = self._label_of(instance)[0] or instance
        patch = result.get("patch")
        if not result.get("changed"):
            self.report(f"{label} 在第 {key.step} 步本来就是 {state_zh(state)}，没有改")
            return result
        self.logger.info("assert_state %s step %s: %s -> %s (next %s: %s)", instance,
                         key.step, patch.previous, patch.state, patch.next_step,
                         patch.restores)
        self.task_card.refresh()
        self.timeline.refresh_statuses()
        self.forget_card_hints()
        split_here = any(
            str(row.get("instance")) == str(instance)
            and row.get("kind") == api.KIND_SPLIT_KEYFRAME and not row.get("done")
            for row in self.task_card.rows())
        note = state_note(label, patch, split_here, result.get("run"), result.get("after"))
        self.update_status()
        self.report(note, hold_ms=6000)
        self.flash_visibility_note(note)
        return result
