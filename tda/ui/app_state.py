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

from tda.ui import app_actions as A
from tda.ui import app_support as S
from tda.ui.class_names import state_zh

__all__ = ["CUT_DROPPED", "CUT_KEPT", "CUT_MOVED", "NO_TARGET", "SPLIT_NOW", "StateMixin",
           "cut_text", "state_note"]

#: ``T`` with nothing selected and nothing being drawn.
NO_TARGET = ("T 改的是实例表里选中的零件（或正在画的零件）：先在右下实例表点一行 / "
             "select a part in the instance table first")
#: The tail of every "the shape has to be redrawn" sentence (U8 round 3): no
#: order to remember, the commit cuts the version at the run's edges itself.
SPLIT_NOW = "工具会自动在状态边界切开版本"
#: Prefix of a refusal.
REFUSED = "改不了状态："


def redraw_text(state: str, run: tuple[int, int], redraw) -> str:
    """Which key draws the run's shape, or ``""`` when nothing has to be drawn.

    ``redraw`` is :func:`tda.ui.session_state.redraw_needed`'s ``{"top",
    "below"}`` (``True`` reads as a ✂ at the top). Any frame of the run takes
    ``Enter`` -- the commit cuts the version at both edges -- except the run's
    top frame when it is an open ✂ row: there ``Enter`` is refused and
    ``Ctrl+K`` is the key. A ✂ row only *below* the run is on the other
    state's side (the clip still open at 38 under a closed 39-41): Ctrl+K there
    would draw the wrong frames, so it is not offered.
    """
    if redraw is True:
        redraw = {"top": True, "below": False}
    if not redraw or not (redraw.get("top") or redraw.get("below")):
        return ""
    first, last = run
    look = f"「{state_zh(state)}」"
    if first == last:
        key = "Ctrl+K" if redraw.get("top") else "Enter"
        return f"这帧的形状要重画：画{look}后 {key}；{SPLIT_NOW}"
    where = f"在第 {first}–{last} 步任一帧画{look}后 Enter"
    if redraw.get("top"):
        where += f"（第 {last} 步是 ✂ 行，那里用 Ctrl+K）"
    return f"这几帧的形状要重画：{where}；{SPLIT_NOW}"


def cut_note(result: dict, steps_text) -> str:
    """What a commit that stopped at a state boundary did (U8 round 3), or ``""``."""
    below, top = result.get("cut_below"), result.get("cut_top")
    if below is None and top is None:
        return ""
    parts = []
    if below is not None:
        parts.append(f"已先在第 {below} 步切开旧版本（形状不变）")
    if top is not None:
        parts.append(f"新版本只到第 {top} 步，后面的帧保持原来的形状")
    reach = result.get("edit_reach") or []
    if reach:
        parts.append(f"这次修改只影响第 {steps_text(reach)} 步")
    return "，".join(parts)


#: After an assertion that moved a cut keyframe with its boundary (U11).
CUT_MOVED = "形状版本边界也随之移到第 {step} 步"
#: ... that dropped one: every frame of it joined the run, its shape is elsewhere.
CUT_DROPPED = "第 {step} 步切开的旧版本用不着了，已删掉"
#: ... that left one where it was although its boundary went: its shape is nowhere else.
CUT_KEPT = "第 {step} 步切开时留下的旧形状没删（别处没有这份形状），请看一眼"


def cut_text(cuts, view: Optional[str] = None) -> str:
    """What became of the cut keyframes that followed the boundary (U11), or ``""``.

    ``cuts`` are :class:`~tda.core.cut_follow.CutMove`; the view on screen is
    named first when several views moved (they share the step numbers).
    """
    cuts = sorted(cuts or [], key=lambda m: (m.view != view, m.view))
    parts = []
    moved = [m for m in cuts if m.moved and not m.kept]
    if moved:
        parts.append(CUT_MOVED.format(step=moved[0].new))
    parts += [CUT_DROPPED.format(step=m.old) for m in cuts if m.dropped][:1]
    parts += [CUT_KEPT.format(step=m.old) for m in cuts if m.kept][:1]
    return "；".join(parts)


def state_note(label: str, patch, redraw=None,
               run: Optional[tuple[int, int]] = None,
               after: Optional[tuple[int, str]] = None, unseen: bool = False,
               cuts=None, view: Optional[str] = None) -> str:
    """The status line after an assertion, in the annotator's words.

    ``内存卡扣 ram_latch.01 在第 21 步：关上（之前为 打开）；第 22 步起恢复 打开``;
    ``run`` (the steps now sharing the state) and ``after`` (the first step past
    them) make a run of frames read as one: ``…；第 20–21 步都是 关上；第 22 步起
    恢复 打开``.  ``unseen``: the restore step has no image in this view, so the
    log's value is kept there without anyone having looked.  ``redraw``: where
    the run still shows the other state's shape -- then :func:`redraw_text`.
    ``cuts``: the cut keyframes that followed the boundary -- :func:`cut_text`.
    """
    text = (f"{label} 在第 {patch.step} 步：{state_zh(patch.state)}"
            f"（之前为 {state_zh(patch.previous)}）")
    if run is None and patch.next_step is not None and patch.restores is not None:
        run = (patch.step, patch.step)
        after = (patch.next_step, patch.restores) if patch.restores != patch.state else None
    if run is not None and run[0] != run[1]:
        text += f"；第 {run[0]}–{run[1]} 步都是 {state_zh(patch.state)}"
    if after is not None:
        back = "恢复 " if after[1] == patch.previous else "是 "
        text += f"；第 {after[0]} 步起{back}{state_zh(after[1])}"
    if unseen and patch.next_step is not None:
        text += (f"（第 {patch.next_step} 步在这个视角没有图像，那里按日志算"
                 f"{'' if patch.restores is None else ' ' + state_zh(patch.restores)}）")
    moved = cut_text(cuts, view)
    if moved:
        text += f"。{moved}"
    redo = redraw_text(patch.state, run or (patch.step, patch.step), redraw)
    if redo:
        text += f"。{redo}"
    return text + "（Ctrl+Z 撤销）"


class StateMixin:
    """``T`` / the state combo, for :class:`tda.ui.app.MainWindow`."""

    def _init_state(self) -> None:
        self.instances.sigAssertState.connect(self.on_state_picked)

    def _state_target(self) -> Optional[str]:
        """The part ``T`` is about: the instance list's selection, else the edit's."""
        return (self.instances.selected_instance()
                or getattr(self.session, "editing_instance", None))

    def _refuse_state(self, text: str) -> None:
        """Say why, over whatever the last change said.

        A change holds its line for six seconds and its canvas note for four;
        a refusal right after it -- ``T`` on the next part, which has nothing
        to set -- used to be swallowed by both. An empty report clears the
        hold, and the note goes.
        """
        self.report("")
        self.report(text)
        if getattr(self, "_vis_note", ("", None))[0]:
            self.expire_visibility_note()

    @S.guard
    def act_cycle_state(self) -> None:
        """``T``: the selected part's state at this frame becomes the next one."""
        if not self._editable_row():
            return
        instance = self._state_target()
        if not instance:
            self._refuse_state(NO_TARGET)
            return
        if not callable(getattr(self.session, "assert_state", None)):
            self._refuse_state(f"{REFUSED}这个会话不支持 / not supported by this session")
            return
        nxt = self.session.next_state(instance)
        if nxt is None:
            self._refuse_state(REFUSED + self.session.state_gate(instance)[1])
            return
        self.assert_state_here(instance, nxt)

    @S.guard
    def on_state_picked(self, instance: str, state: str) -> None:
        """The instance list's combo: the same gesture as ``T``, to a named state.

        Annotate mode only, like ``T``: Review shows the frame the queue points
        at and Steps is about the log, and a state written from either landed
        on a frame nobody was annotating.
        """
        if self.mode != A.MODE_ANNOTATE:
            self.instances.refresh()             # the combo goes back to what holds
            self._refuse_state(f"{REFUSED}只在标注模式可用 / Annotate mode only")
            return
        self.assert_state_here(instance, state)

    def assert_state_here(self, instance: str, state: str) -> Optional[dict]:
        """Write the state, repaint what reads it, and say what changed."""
        key = self.session.current()
        try:
            result = self.session.assert_state(instance, state)
        except ValueError as refused:            # SessionRefusal is one
            self.logger.info("assert_state refused %s step %s -> %s: %s",
                             instance, key.step, state, refused)
            self.instances.refresh()             # the combo goes back to what holds
            self._refuse_state(f"{REFUSED}{refused}")
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
        note = state_note(label, patch, result.get("redraw"), result.get("run"),
                          result.get("after"), unseen=bool(result.get("restore_unseen")),
                          cuts=result.get("cuts"), view=key.view)
        self.update_status()
        self.report(note, hold_ms=6000)
        self.flash_visibility_note(note)
        return result
