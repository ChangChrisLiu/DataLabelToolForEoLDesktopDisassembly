"""``T`` and the instance list's state combo: the state the picture shows (task U8).

On D13/scan ``ram_latch.01`` is opened at step 14 by the log and is visibly
shut again on a few middle frames.  There is no verb for that and there must
not be one -- a clip springing back is not a disassembly step -- so the
annotator says what the frame shows: select the part, press ``T`` (or pick the
state in the instance list), and the session writes "at this step it is
closed" with the next step put back as it was
(:meth:`tda.ui.session_state.StateMixin.assert_state`).  The task card then
shows the ✂ rows of the new boundaries and ``Enter`` draws them -- from this
frame on (task U16) -- exactly as for a boundary the step table made.

The trial annotator's attempt -- re-drawing the shape at frame 21 with
``Enter`` -- re-traced the version covering 14-41 and raised 48 conflicts.  The
status line therefore says, after every change, which frames now hold which
state and whether this frame's shape has to be drawn again.

Task U11: on D2/scan the annotator walked an opened cover down from 45 to 37
pressing ``T`` at every frame.  ``Ctrl+T`` says it once -- "from this frame to
step N it is open" (:class:`StateRangeDialog`) -- and the status line hints at
it when ``T`` is pressed on consecutive frames for the same part.
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

from tda.core.absent import CHOICE as ABSENT_CHOICE
from tda.core.absent import residue_lines, residues
from tda.ui import app_actions as A
from tda.ui import app_support as S
from tda.ui.class_names import state_zh

__all__ = ["ABSENT_ASK", "ABSENT_DONE", "CONFIRM_DIRTY", "CUT_CONFIRMED", "CUT_DROPPED",
           "CUT_KEPT", "CUT_MOVED", "HINT_LOG", "HINT_RANGE", "NO_TARGET", "NO_TARGET_FOR",
           "REMOVED_ROW", "SPLIT_NOW", "StateMixin", "StateRangeDialog", "absent_note",
           "cut_text", "range_note", "state_note"]

#: ``T`` with nothing selected and nothing being drawn.
NO_TARGET = ("T 改的是实例表里选中的零件（或正在画的零件）：先在右下实例表点一行 / "
             "select a part in the instance table first")
#: ... as each key of the selected part says it (``Ctrl+T``, ``Ctrl+Shift+K``).
NO_TARGET_FOR = {
    "cycle_state": NO_TARGET,
    "state_range": NO_TARGET.replace("T 改的", "Ctrl+T 改的", 1),
    "confirm_cut_copy": NO_TARGET.replace("T 改的", "Ctrl+Shift+K 确认的", 1),
}
#: A key that edits a part, on a row of the instance table's ``Removed`` list.
REMOVED_ROW = "已移除的零件不可编辑 / this part has been removed"
#: Answers about a part kept by the window at most (:meth:`StateMixin._why_once`).
_WHY_KEPT = 64
#: The tail of every "the shape has to be redrawn" sentence (U8 round 3): no
#: order to remember, the commit cuts the version at the run's edges itself.
SPLIT_NOW = "工具会自动在状态边界切开版本"
#: Prefix of a refusal.
REFUSED = "改不了状态："
#: ``Ctrl+Shift+K`` done (U11 round 2).
CUT_CONFIRMED = ("{label}：切开时留下的形状就是这几帧的样子（到第 {step} 步），✂ 完成"
                 "（Ctrl+Z 撤销）")
#: ``Ctrl+Shift+K`` over painted, uncommitted pixels.
CONFIRM_DIRTY = ("正在画的像素还没提交：画的就是这一帧的样子按 Q；要确认副本先 Esc 放弃")
#: ``T`` on the walk's next frame, same part, same state (U11): there is a key for a run.
HINT_RANGE = "同一状态持续多帧？Ctrl+T 一次设到某一步"
#: ... from the third frame on: perhaps it is the log that misses the action.
HINT_LOG = "日志本身漏了这一步的动作？去 Steps 模式加动作更稳"
#: 「一开始就不在（这台没有它）」 (task U34): the question, and the status line.
ABSENT_TITLE = "这台没有这个零件 / not in this machine"
ABSENT_ASK = ("{label}：这台机器根本没有它？\n"
              "从第 1 帧起它就不在机器里：每一帧的任务卡都不再要它，cli check 和导出里也没有它。"
              "日志里对它的动作留着，但不再算。\n"
              "它的螺丝、接头不跟着变，要单独看。\n"
              "Ctrl+Z 撤销；Steps → Instances 右键「其实有」也能改回来。")
ABSENT_DONE = "{label}：一开始就不在（这台没有它），每一帧都不再要它"
#: Above what still names it (U34 round 2, N5): said, not refused.
ABSENT_RESIDUES = "还有这些记着它（原样留着，不影响这次）："
ABSENT_RECHECK = "；{n} 个已确认帧会重新核对"


def absent_note(label: str, patch) -> str:
    """The status line after 「一开始就不在」 (task U34)."""
    text = ABSENT_DONE.format(label=label)
    if getattr(patch, "frames", 0):
        text += ABSENT_RECHECK.format(n=patch.frames)
    return text + "（Ctrl+Z 撤销）"


def redraw_text(state: str, run: tuple[int, int], redraw) -> str:
    """Which key draws the run's shape, or ``""`` when nothing has to be drawn.

    ``redraw`` is :func:`tda.ui.session_state.redraw_needed`'s ``{"top",
    "below"}`` (``True`` reads as a ✂ at the top). The key is ``Enter`` on the
    run's top frame -- the first of it the reverse walk reaches: ``Enter`` is
    "from this frame on" (task U16), so from there it covers the whole run,
    and the commit cuts the version at the run's lower edge itself. On a lower
    frame of the run it would leave the frames above it with the other
    state's shape. A ✂ row only *below* the run is on the other state's side
    (the clip still open at 38 under a closed 39-41): drawing there would draw
    the wrong frames, so it is not offered.
    """
    if redraw is True:
        redraw = {"top": True, "below": False}
    if not redraw or not (redraw.get("top") or redraw.get("below")):
        return ""
    first, last = run
    look = f"「{state_zh(state)}」"
    if first == last:
        return f"这帧的形状要重画：画{look}后 Q；{SPLIT_NOW}"
    return (f"这几帧的形状要重画：在第 {last} 步画{look}后 Q，第 {first}–{last} 步"
            f"都用它；{SPLIT_NOW}")


def cut_note(result: dict, steps_text) -> str:
    """What a commit that stopped at a state boundary did (U8 round 3), or ``""``.

    The forward walk's two (U29) read the same: ``cut_behind`` is where the
    old version was cut, ``cut_ahead`` where the new one stops.
    """
    below, top = result.get("cut_below"), result.get("cut_top")
    if below is None:
        below = result.get("cut_behind")
    if top is None:
        top = result.get("cut_ahead")
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


def range_note(label: str, patch, redraw=None,
               run: Optional[tuple[int, int]] = None,
               after: Optional[tuple[int, str]] = None, unseen: bool = False,
               cuts=None, view: Optional[str] = None) -> str:
    """The status line after ``Ctrl+T`` (U11): the run the span is now part of.

    ``主板螺丝盖板 cover.02 第 38–45 步都是 打开；第 46 步起 拆下``; ``run`` and
    ``after`` are :func:`tda.ui.session_state.run_around`'s (the span itself
    when absent), then what became of the cut keyframes and what is left to draw.
    """
    first, last = run if run is not None else (patch.step, patch.top)
    value = state_zh(patch.state)
    text = (f"{label} 第 {first}–{last} 步都是 {value}" if first != last
            else f"{label} 第 {first} 步是 {value}")
    if after is None and patch.next_step is not None and patch.restores not in (
            None, patch.state):
        after = (patch.next_step, patch.restores)
    if after is not None:
        text += f"；第 {after[0]} 步起 {state_zh(after[1])}"
    if unseen and patch.next_step is not None:
        text += f"（第 {patch.next_step} 步在这个视角没有图像，那里按日志算）"
    moved = cut_text(cuts, view)
    if moved:
        text += f"。{moved}"
    redo = redraw_text(patch.state, (first, last), redraw)
    if redo:
        text += f"。{redo}"
    return text + "（Ctrl+Z 撤销）"


def state_note(label: str, patch, redraw=None,
               run: Optional[tuple[int, int]] = None,
               after: Optional[tuple[int, str]] = None, unseen: bool = False,
               cuts=None, view: Optional[str] = None, hints=()) -> str:
    """The status line after an assertion, in the annotator's words.

    ``内存卡扣 ram_latch.01 在第 21 步：关上（之前为 打开）；第 22 步起恢复 打开``;
    ``run`` (the steps now sharing the state) and ``after`` (the first step past
    them) make a run of frames read as one: ``…；第 20–21 步都是 关上；第 22 步起
    恢复 打开``.  ``unseen``: the restore step has no image in this view, so the
    log's value is kept there without anyone having looked.  ``redraw``: where
    the run still shows the other state's shape -- then :func:`redraw_text`.
    ``cuts``: the cut keyframes that followed the boundary -- :func:`cut_text`.
    ``hints``: :data:`HINT_RANGE` / :data:`HINT_LOG` while a run is walked.
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
    text += "（Ctrl+Z 撤销）"
    return "。".join([text, *hints]) if hints else text


class StateRangeDialog(QDialog):
    """``Ctrl+T``: the state, and the step the span runs to (U11)."""

    def __init__(self, parent: Optional[QWidget], label: str, step: int,
                 choices: list[str], state: Optional[str], to_step: int,
                 low: int, high: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("一段帧的状态 / state over a span")
        self.setModal(True)
        self.state_box = QComboBox()
        for value in choices:
            self.state_box.addItem(state_zh(value), value)
        if state in choices:
            self.state_box.setCurrentIndex(choices.index(state))
        self.step_box = QSpinBox()
        self.step_box.setRange(int(low), int(high))
        self.step_box.setValue(int(to_step))
        self.step_box.setPrefix("一直到第 ")
        self.step_box.setSuffix(" 步")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{label}：从第 {step} 步起"))
        layout.addWidget(self.state_box)
        layout.addWidget(self.step_box)
        layout.addWidget(QLabel("这一段每一帧都设成这个状态；之后一步恢复原来的状态。"
                                "跨过日志自己的动作时不行（去 Steps 模式改）。"))
        layout.addWidget(buttons)
        self.step_box.setFocus()
        self.step_box.selectAll()

    def answer(self) -> tuple[str, int]:
        return str(self.state_box.currentData()), int(self.step_box.value())


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

    @S.guard
    def act_confirm_cut_copy(self) -> None:
        """``Ctrl+Shift+K``: the cut copy on this frame is its shape (U11 round 2)."""
        if not self._editable_row():
            return
        instance = self._state_target()
        if not instance:
            self._refuse_state(NO_TARGET_FOR["confirm_cut_copy"])
            return
        if not callable(getattr(self.session, "confirm_cut_copy", None)):
            self._refuse_state("这个会话不支持 / not supported by this session")
            return
        editing = getattr(self.session, "editing_instance", None)
        if editing is not None:
            self._note_edit_facts(refresh=False)
            if self.layer_facts()[2]:
                self._refuse_state(CONFIRM_DIRTY)
                return
            if editing == instance:
                self.act_clear_edit()            # the untouched layer goes first
        try:
            result = self.session.confirm_cut_copy(instance)
        except ValueError as refused:            # SessionRefusal is one
            self._refuse_state(str(refused))
            return
        label = self._label_of(instance)[0] or instance
        self.task_card.refresh()
        self.timeline.refresh_statuses()
        self.forget_card_hints()
        self.update_status()
        self.report(CUT_CONFIRMED.format(label=label, step=result.get("confirmed_cut")),
                    hold_ms=6000)

    @S.guard
    def act_state_range(self) -> None:
        """``Ctrl+T``: the selected part's state from this frame to a given step (U11)."""
        if not self._editable_row():
            return
        instance = self._state_target()
        if not instance:
            self._refuse_state(NO_TARGET_FOR["state_range"])
            return
        if not callable(getattr(self.session, "assert_state_range", None)):
            self._refuse_state(f"{REFUSED}这个会话不支持 / not supported by this session")
            return
        choices, why = self.session.state_gate(instance)
        if not choices:
            self._refuse_state(REFUSED + why)
            return
        state, to_step = self.session.state_range_default(instance)
        answer = self.ask_state_range(instance, choices, state, to_step)
        if answer is None:
            return
        self.assert_state_range_here(instance, *answer)

    # ------------------------------------------------- the palette's 「状态」
    def state_button_refusal(self, name: str, dirty: bool) -> str:
        """Why the 「状态」 button ``name`` cannot run now, ``""`` if it may (task U13).

        Its slot's gates in the slot's order and words, read off what the
        panels already hold; ``dirty`` is the guide's cached "the editing layer
        has uncommitted pixels". What only the session can judge -- a class
        with no state to change, a part that is not a two-way cover, a shape
        that is not a cut copy -- is asked once per frame, part and re-read of
        the instance list (:meth:`_why_once`). A refusal found only when the
        write is tried (the log's own action in the way, say) still is: the
        button stays live and its click says it, as the key does.
        """
        if name == "mark_absent":
            return self.absent_refusal(dirty=dirty)
        if name == "correct_verb":
            return self.verb_button_refusal(dirty)
        if name not in NO_TARGET_FOR:
            return ""
        if self.instances.selected_is_removed():
            return REMOVED_ROW
        instance = self._state_target()
        if not instance:
            return NO_TARGET_FOR[name]
        if name == "confirm_cut_copy":
            if getattr(self.session, "editing_instance", None) is not None and dirty:
                return CONFIRM_DIRTY
            if self._card_cut_copy(instance):
                return ""                         # the card's ✂ row says so already
            gate = getattr(self.session, "cut_copy_gate", None)
            return self._why_once("cut_copy", instance, gate) if callable(gate) else ""
        row = self.instances.row_of(instance)
        # 「一开始就不在」 is the combo's, never T's (U34 round 2): it enables nothing here
        if row is not None and [c for c in row.get("state_choices") or ()
                                if c != ABSENT_CHOICE]:
            return ""
        gate = getattr(self.session, "state_gate", None)
        return (self._why_once("state", instance, lambda part: gate(part)[1])
                if callable(gate) else "")

    def _why_once(self, kind: str, instance: str, ask) -> str:
        """``ask(instance)`` -- a session's "why not" -- once per frame, part and rows.

        The palette is refreshed on every status update; the session's answer
        costs a read of the event log. It changes only with a write, and every
        write ends in a re-read of the instance list
        (:meth:`~tda.ui.panels.instances.InstanceListPanel.generation`).
        """
        key = (kind, str(instance), self.session.current(), self.instances.generation())
        cache = self.__dict__.setdefault("_why_cache", {})
        if key not in cache:
            if len(cache) >= _WHY_KEPT:
                cache.clear()
            try:
                cache[key] = str(ask(instance) or "")
            except Exception:  # noqa: BLE001 - a grey button is never worth a failure
                cache[key] = ""
        return cache[key]

    def ask_state_range(self, instance: str, choices: list[str], state: Optional[str],
                        to_step: int) -> Optional[tuple[str, int]]:
        """Show the ``Ctrl+T`` dialog; ``None`` when the annotator cancelled.

        A method rather than an inline dialog so a test can answer it without a
        modal event loop (as :meth:`ask_pose_split`).
        """
        steps = self.session.steps() or [to_step]
        label = self._label_of(instance)[0] or instance
        dialog = StateRangeDialog(self, label, self.session.current().step, choices, state,
                                  to_step, min(steps), max(steps))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.answer()

    def assert_state_range_here(self, instance: str, state: str,
                                to_step: int) -> Optional[dict]:
        """Write the span, repaint what reads it, and say what changed (U11)."""
        key = self.session.current()
        self._t_last = None                      # a span is not a walk of T
        try:
            result = self.session.assert_state_range(instance, state, int(to_step))
        except ValueError as refused:            # SessionRefusal is one
            self.logger.info("assert_state_range refused %s step %s..%s -> %s: %s",
                             instance, key.step, to_step, state, refused)
            self.instances.refresh()
            self._refuse_state(f"{REFUSED}{refused}")
            return None
        label = self._label_of(instance)[0] or instance
        patch = result.get("patch")
        if not result.get("changed"):
            low, high = sorted((key.step, int(to_step)))
            self.report(f"{label} 第 {low}–{high} 步本来就都是 {state_zh(state)}，没有改")
            return result
        self.logger.info("assert_state_range %s steps %s..%s: %s (next %s: %s)", instance,
                         patch.step, patch.top, patch.state, patch.next_step,
                         patch.restores)
        note = range_note(label, patch, result.get("redraw"), result.get("run"),
                          result.get("after"), unseen=bool(result.get("restore_unseen")),
                          cuts=result.get("cuts"), view=key.view)
        self._after_state_change(note)
        return result

    def _after_state_change(self, note: str) -> None:
        """Repaint what reads the state and say ``note`` (both gestures)."""
        self.task_card.refresh()
        self.timeline.refresh_statuses()
        self.forget_card_hints()
        self.update_status()
        self.report(note, hold_ms=6000)
        self.flash_visibility_note(note)

    def _walk_hints(self, key, instance: str, state: str) -> list[str]:
        """:data:`HINT_RANGE` from the second ``T`` in a row on the walk, :data:`HINT_LOG`
        from the third: same part, same state, the frame next to the last one."""
        last = getattr(self, "_t_last", None)
        walk = list(self.session.available_steps())
        count = 1
        if (last is not None and last[:4] == (key.desktop, key.view, instance, state)
                and last[4] in walk and key.step in walk
                and abs(walk.index(last[4]) - walk.index(key.step)) == 1):
            count = last[5] + 1
        self._t_last = (key.desktop, key.view, instance, state, key.step, count)
        return [HINT_RANGE, HINT_LOG][:max(0, min(count - 1, 2))]

    # ------------------------------------- 「一开始就不在（这台没有它）」 (U34)
    def ask_absent_from_start(self, label: str, instance: str, notes=()) -> bool:
        """The confirmation (a method, so a test answers it); ``True``: go ahead.

        ``notes``: what still names the part and stays (U34 round 2, N5).
        """
        from PySide6.QtWidgets import QMessageBox

        text = ABSENT_ASK.format(label=label)
        if notes:
            text += "\n" + "\n".join([ABSENT_RESIDUES, *(f"  · {line}" for line in notes)])
        answer = QMessageBox.question(
            self, ABSENT_TITLE, text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def absent_from_start_here(self, instance: str) -> Optional[dict]:
        """「一开始就不在」 from the state combo: ask, write, say what changed (U34)."""
        label = self._label_of(instance)[0] or instance
        if not callable(getattr(self.session, "mark_absent_from_start", None)):
            self._refuse_state(f"{REFUSED}这个会话不支持 / not supported by this session")
            return None
        why = self.session.absent_gate(instance)
        if why:
            self.instances.refresh()             # the combo goes back to what holds
            self._refuse_state(f"{REFUSED}{why}")
            return None
        notes = residue_lines(residues(self.session.db, self.session.current().desktop,
                                       instance))
        if not self.ask_absent_from_start(label, instance, notes):
            self.instances.refresh()
            self.report(f"{label}：没有改 / not changed")
            return None
        try:
            result = self.session.mark_absent_from_start(instance)
        except ValueError as refused:            # SessionRefusal is one
            self.logger.info("absent_from_start refused %s: %s", instance, refused)
            self.instances.refresh()
            self._refuse_state(f"{REFUSED}{refused}")
            return None
        self.logger.info("absent_from_start %s (%s frozen frames to re-check)", instance,
                         result["patch"].frames)
        self._t_last = None
        self._after_state_change(absent_note(label, result["patch"]))
        return result

    def assert_state_here(self, instance: str, state: str) -> Optional[dict]:
        """Write the state, repaint what reads it, and say what changed.

        The state combo's 「一开始就不在」 (task U34) is no state: it goes to
        :meth:`absent_from_start_here`.
        """
        if state == ABSENT_CHOICE:
            return self.absent_from_start_here(instance)
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
        note = state_note(label, patch, result.get("redraw"), result.get("run"),
                          result.get("after"), unseen=bool(result.get("restore_unseen")),
                          cuts=result.get("cuts"), view=key.view,
                          hints=self._walk_hints(key, instance, state))
        self._after_state_change(note)
        return result
