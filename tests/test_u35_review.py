"""Task U35 in the window: 「这一帧的说明」 in Review, and where a note shows elsewhere.

* the box under the Review panel's queues: 「保存」 / 「清除」 (tool buttons --
  the panel still has no push buttons), the status line, ``Ctrl+Z`` / ``Ctrl+Y``;
* typed and not saved: saved when the frame is left -- ``PgDn``, a timeline
  click, ``Enter`` (confirming steps the frame on), leaving Review, any other
  move -- and the status line says so; typing survives a refresh of the frame;
* the keys: ``Q`` / ``W`` / ``S`` typed into the box are letters, never the
  tools or the commit, in Review and in Annotate, and the window does not say
  「快捷键没生效」 about them; no key was added;
* the read-only 「说明：…」 line in the task card (Annotate, not Review), the
  timeline's ✎ and the Steps table's ✎ next to the step's thumbnail.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QPushButton, QToolButton

import app_scene
from tda.core.frame_notes import OP_KIND
from tda.core.model import FrameKey
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.app_keys import KEY_SWALLOWED, NOTE_KEY_SWALLOWED
from tda.ui.panels.frame_note import CLEAR_TEXT, SAVE_TEXT, TITLE, UNSAVED, FrameNoteBox
from tda.ui.panels.review import ReviewPanel
from tda.ui.panels.steptable_models import NOTE_GLYPH as STEPS_GLYPH
from tda.ui.panels.steptable_models import STEP_COLUMNS
from tda.ui.panels.timeline import NOTE_GLYPH, NOTE_ROLE

DESKTOP, VIEW = app_scene.DESKTOP, app_scene.VIEW
NOTE = "screw.psu.03 被硬盘架挡住，按看不见处理\n第二行：画面右下角反光"


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    session = app_scene.make_session(tmp_path)
    win = MainWindow(session, app_scene.make_paths(tmp_path), "tester",
                     sam_queue=app_scene.StubSamQueue())
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    if win.roi_editing:
        win.act_clear_edit()
    QApplication.processEvents()
    yield win
    app_scene.close_window(win)


def key(step: int) -> FrameKey:
    return FrameKey(DESKTOP, step, VIEW)


def box(win) -> FrameNoteBox:
    return win.review.note_box


def stored(win, step: int):
    return win.session.db.frame_note(key(step))


def type_into(win, text: str) -> None:
    """Type into the note box the way the annotator does: focus, then keys."""
    edit = box(win).edit
    edit.setFocus()
    QApplication.processEvents()
    edit.setPlainText(text)


def review(win) -> int:
    win.set_mode(A.MODE_REVIEW)
    QApplication.processEvents()
    return int(win.session.current().step)


# --------------------------------------------------------------------------- #
# the box
# --------------------------------------------------------------------------- #
def test_the_box_sits_under_the_queues_and_only_reports(qapp):
    panel = ReviewPanel()
    note = panel.note_box
    assert note.title.text() == TITLE
    assert (note.save_button.text(), note.clear_button.text()) == (SAVE_TEXT, CLEAR_TEXT)
    assert isinstance(note.edit, QPlainTextEdit) and note.edit.tabChangesFocus()
    assert note.edit.property(A.TYPING_PROPERTY) is True
    assert isinstance(note.save_button, QToolButton)
    assert panel.findChildren(QPushButton) == []        # still no push buttons (U2b)
    assert not note.edit.isEnabled()                     # no frame, no typing

    saved, cleared = [], []
    panel.sigNoteSave.connect(lambda k, t: saved.append((k, t)))
    panel.sigNoteClear.connect(cleared.append)
    note.show_frame(key(3), None)
    assert not note.save_button.isEnabled() and not note.clear_button.isEnabled()
    note.edit.setPlainText("  hello \n")
    assert note.is_dirty() and note.status.text() == UNSAVED
    note.save_button.click()
    assert saved == [(key(3), "hello")]
    assert note.edit.toPlainText() == "  hello \n"        # the panel writes nothing itself

    note.show_frame(key(3), {"text": "hello", "author": "alice",
                             "updated_at": "2026-10-26T06:03:00+00:00"})
    assert not note.is_dirty() and note.status.text() == "alice · 2026-10-26 06:03 UTC"
    note.clear_button.click()
    assert cleared == [key(3)] and note.edit.toPlainText() == ""
    panel.deleteLater()


def test_save_writes_one_op_says_so_and_ctrl_z_takes_it_back(window):
    step = review(window)
    type_into(window, "  " + NOTE + "\n")
    assert box(window).is_dirty()
    box(window).save_button.click()
    QApplication.processEvents()
    assert stored(window, step)["text"] == NOTE
    assert stored(window, step)["author"] == "tester"
    assert window.status_message() == f"第 {step} 帧的说明已保存（Ctrl+Z 撤销）"
    assert not box(window).is_dirty() and box(window).status.text().startswith("tester · ")
    assert window.focusWidget() is window.canvas         # the Review keys work again
    assert window.session.undo_stack.ops[-1].kind == OP_KIND

    window.act_undo()
    assert stored(window, step) is None
    assert window.status_message() == f"撤销：第 {step} 帧的说明 / undo"
    assert box(window).edit.toPlainText() == "" and not box(window).is_dirty()
    window.act_redo()
    assert stored(window, step)["text"] == NOTE
    assert window.status_message() == f"重做：第 {step} 帧的说明 / redo"
    assert box(window).edit.toPlainText() == NOTE

    box(window).clear_button.click()
    assert stored(window, step) is None
    assert window.status_message() == f"第 {step} 帧的说明已清除（Ctrl+Z 撤销）"
    window.act_undo()
    assert stored(window, step)["text"] == NOTE
    ops = window.session.db.ops_of_kind(DESKTOP, OP_KIND)
    assert len(ops) == 2 and all(o["annotator"] == "tester" for o in ops)


def test_typing_survives_a_refresh_of_the_same_frame(window):
    step = review(window)
    type_into(window, "还没写完")
    window.session._announce()                            # the frame re-announced
    QApplication.processEvents()
    assert box(window).edit.toPlainText() == "还没写完" and box(window).is_dirty()
    assert stored(window, step) is None


# --------------------------------------------------------------------------- #
# leaving the frame
# --------------------------------------------------------------------------- #
def test_pgdn_saves_what_was_typed_and_says_so(window):
    step = review(window)
    type_into(window, NOTE)
    window.act_step(-1)
    QApplication.processEvents()
    now = int(window.session.current().step)
    assert now < step
    assert stored(window, step)["text"] == NOTE
    assert window.status_message().endswith(
        f"第 {step} 帧的说明改过没保存：离开时已自动保存（Ctrl+Z 撤销）")
    assert box(window).key() == key(now) and box(window).edit.toPlainText() == ""
    assert not box(window).is_dirty()
    # the op is on this view's stack: Ctrl+Z takes it back from the next frame
    window.act_undo()
    assert stored(window, step) is None


def test_a_move_that_goes_nowhere_saves_it_without_saying_it_was_left(window):
    step = review(window)
    assert step == max(window.session.available_steps())       # PgUp has nowhere to go
    type_into(window, "最后一帧")
    window.act_step(1)
    QApplication.processEvents()
    assert int(window.session.current().step) == step
    assert stored(window, step)["text"] == "最后一帧"
    assert window.status_message().endswith(f"第 {step} 帧的说明已保存（Ctrl+Z 撤销）")


def test_another_view_clears_the_undo_stack_so_ctrl_z_is_not_promised(window):
    """Round 2, N1: the note is saved before the view is left; opening the other
    view empties the stack -- the line must not promise Ctrl+Z."""
    step = review(window)
    type_into(window, "换视角前写的")
    window.act_set_view("oak1")
    QApplication.processEvents()
    assert window.session.view == "oak1"
    assert stored(window, step)["text"] == "换视角前写的"
    assert not [op for op in window.session.undo_stack.ops if op.kind == OP_KIND]
    said = window.status_message()
    assert said.endswith(f"第 {step} 帧的说明改过没保存：离开时已自动保存（换了视角或机器，"
                         f"撤销记录已清空；要改回在复查里改）")
    assert "Ctrl+Z" not in said
    window.act_undo()                                     # nothing of the note's to undo
    assert stored(window, step)["text"] == "换视角前写的"


def test_a_timeline_click_saves_it_too(window):
    step = review(window)
    type_into(window, "时间轴")
    target = window.session.available_steps()[0]
    window.timeline_goto(target)
    QApplication.processEvents()
    assert stored(window, step)["text"] == "时间轴"
    assert box(window).key() == key(target)


def test_a_move_the_window_did_not_make_is_saved_by_the_panel(window):
    step = review(window)
    type_into(window, "后台")
    other = window.session.available_steps()[0]
    window.session.goto(other, force=True)                # not through leave_frame
    assert stored(window, step)["text"] == "后台"
    QApplication.processEvents()                          # the line is said from the loop
    # after whatever the arrival said (the card's problems here), joined
    assert window.status_message().endswith(
        f"第 {step} 帧的说明改过没保存：离开时已自动保存（Ctrl+Z 撤销）")


def test_enter_confirms_and_saves_the_note_first(window):
    session = window.session
    session.sweeper_enabled = False
    session.goto(12)
    app_scene.seed_shapes(session, 12)
    session.goto(13)
    review(window)
    type_into(window, "确认前写的")
    window.canvas.setFocus()
    assert window.act_confirm()
    assert stored(window, 13)["text"] == "确认前写的"
    assert session.current().step == 12 and session.frame_status(13) == "verified"
    assert window.status_message().endswith("；第 13 帧的说明改过没保存：离开时已自动保存（Ctrl+Z 撤销）")


def _confirmable(window, steps=(12, 13)) -> None:
    session = window.session
    session.sweeper_enabled = False
    session.goto(min(steps))
    app_scene.seed_shapes(session, min(steps))


def test_enter_on_a_queue_entry_says_the_note_after_the_confirmation(window, monkeypatch):
    """Round 2, N3: Enter on a selected entry opens its frame (through leave_frame,
    which saves the note) and then confirms -- a second frame change; the note's
    line comes after the confirmation's, from the event loop."""
    _confirmable(window)
    window.session.goto(13)
    review(window)
    type_into(window, "队列确认前")
    monkeypatch.setattr(window.review, "selected_step", lambda: 12)
    window.canvas.setFocus()
    assert window.act_confirm()
    assert stored(window, 13)["text"] == "队列确认前"
    assert window.session.frame_status(12) == "verified"
    QApplication.processEvents()
    assert window.status_message() == ("step 12 confirmed；第 13 帧的说明改过没保存：离开时已自动保存"
                                       "（Ctrl+Z 撤销）")


def test_confirming_the_last_frame_of_the_walk_does_not_say_it_was_left(window):
    """Round 2, N3: the first step has nowhere to go back to -- Space confirms in place."""
    session = window.session
    session.sweeper_enabled = False
    first = min(session.available_steps())
    session.goto(first)
    app_scene.seed_shapes(session, first)
    review(window)
    type_into(window, "第一帧")
    window.canvas.setFocus()
    assert window.act_confirm()
    assert session.current().step == first
    assert stored(window, first)["text"] == "第一帧"
    assert window.status_message() == (f"step {first} confirmed；第 {first} 帧的说明已保存"
                                       f"（Ctrl+Z 撤销）")


def test_leaving_review_saves_it(window):
    step = review(window)
    type_into(window, "切走")
    window.set_mode(A.MODE_ANNOTATE)
    assert stored(window, step)["text"] == "切走"
    assert window.status_message().startswith(f"第 {step} 帧的说明改过没保存")


# --------------------------------------------------------------------------- #
# the keys
# --------------------------------------------------------------------------- #
def test_q_w_s_typed_into_the_box_are_letters_in_review_and_in_annotate(window):
    review(window)
    window._tool_name = "eraser"      # W would make it the brush, S SAM point
    edit = box(window).edit
    edit.setFocus()
    QApplication.processEvents()
    QTest.keyClicks(edit, "qws")
    assert edit.toPlainText() == "qws"
    window.report("")

    # the box with the annotate-mode keys live: Q commits, W is the brush, S SAM
    window.mode = A.MODE_ANNOTATE
    before = (window._tool_name, len(window.session.undo_stack))
    edit.setFocus()
    QApplication.processEvents()
    QTest.keyClicks(edit, "QWS")
    assert edit.toPlainText() == "qwsQWS"
    assert (window._tool_name, len(window.session.undo_stack)) == before
    for letter in (Qt.Key.Key_Q, Qt.Key.Key_W, Qt.Key.Key_S):
        event = QKeyEvent(QEvent.Type.KeyPress, letter, Qt.KeyboardModifier.NoModifier,
                          chr(int(letter)).lower())
        assert window.handle_key(event) is False          # the box keeps it
    assert window.status_message() != KEY_SWALLOWED       # typing here is the point
    assert window._tool_name == "eraser"

    # the same keys reach the window's filter from any text field: a plain one
    # (no typing property) still says 「快捷键没生效」
    from PySide6.QtWidgets import QLineEdit

    field = QLineEdit(window)
    field.show()
    field.setFocus()
    QApplication.processEvents()
    QTest.keyClicks(field, "W")
    assert field.text() == "W" and window.status_message() == KEY_SWALLOWED
    assert window._tool_name == "eraser"
    field.deleteLater()
    window.mode = A.MODE_REVIEW


def test_a_window_key_that_types_nothing_says_where_the_focus_is(window):
    """Round 2, N4: PgDn, Ctrl+S, an F key in the box are not text -- said once
    (Tab gets back to the canvas); letters, digits, Space and Enter are text."""
    step = review(window)
    edit = box(window).edit
    edit.setFocus()
    QApplication.processEvents()
    window.report("")
    QTest.keyClicks(edit, "a1 b")
    QTest.keyClick(edit, Qt.Key.Key_Return)
    assert window.status_message() == ""                  # typing: nothing said
    QTest.keyClick(edit, Qt.Key.Key_PageDown)
    assert window.status_message() == NOTE_KEY_SWALLOWED
    assert int(window.session.current().step) == step     # the frame did not move
    assert edit.toPlainText() == "a1 b\n"

    # once per focus episode; back in the box, Ctrl+S says it again
    window.canvas.setFocus()
    QApplication.processEvents()
    assert window.status_message() == ""                  # stale once the canvas has the keys
    edit.setFocus()
    QApplication.processEvents()
    QTest.keyClick(edit, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
    assert window.status_message() == NOTE_KEY_SWALLOWED
    for key_, mods, text in ((Qt.Key.Key_F2, Qt.KeyboardModifier.NoModifier, ""),
                             (Qt.Key.Key_PageUp, Qt.KeyboardModifier.NoModifier, ""),
                             (Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier, "")):
        assert not A.types_text(key_, mods, text), key_
    for key_, mods, text in ((Qt.Key.Key_Q, Qt.KeyboardModifier.NoModifier, "q"),
                             (Qt.Key.Key_Q, Qt.KeyboardModifier.ShiftModifier, "Q"),
                             (Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier, "1"),
                             (Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " "),
                             (Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier, "\r"),
                             (Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier, "")):
        assert A.types_text(key_, mods, text), key_


def test_tab_in_the_box_goes_back_to_the_canvas_and_the_focus_is_shown(window):
    review(window)
    edit = box(window).edit
    assert ":focus" in edit.styleSheet() and "border: 2px" in edit.styleSheet()
    edit.setFocus()
    QApplication.processEvents()
    QTest.keyClicks(edit, "x")
    QTest.keyClick(edit, Qt.Key.Key_Tab)
    QApplication.processEvents()
    assert window.focusWidget() is window.canvas
    assert edit.toPlainText() == "x"                      # no tab typed


def test_tab_leaves_the_box_on_a_frame_with_no_image_too(qapp, tmp_path):
    """Round 3, R2-N1: a frame with no image shows the placeholder, not the canvas
    (D1 scan step 2 is one); Tab still takes the keyboard out of the box -- to the
    timeline -- and PgDn then walks on."""
    session = app_scene.make_session(tmp_path, missing=(13,))
    win = MainWindow(session, app_scene.make_paths(tmp_path), "tester",
                     sam_queue=app_scene.StubSamQueue())
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    try:
        if win.roi_editing:
            win.act_clear_edit()
        win.set_mode(A.MODE_REVIEW)
        win.timeline_goto(13)
        QApplication.processEvents()
        assert session.current().step == 13 and session.image() is None
        assert win.stack.currentWidget() is not win.canvas
        edit = box(win).edit
        type_into(win, "这一帧没有图像")
        QTest.keyClick(edit, Qt.Key.Key_Tab)
        QApplication.processEvents()
        assert not edit.hasFocus() and win.focusWidget() is not edit
        assert win.focusWidget() is win.timeline.list_widget()
        assert edit.toPlainText() == "这一帧没有图像"           # no tab typed
        QTest.keyClick(win.focusWidget(), Qt.Key.Key_PageDown)
        QApplication.processEvents()
        assert session.current().step == 12                # the frame key works again
        assert stored(win, 13)["text"] == "这一帧没有图像"    # saved on the way out
        # 「保存」 on such a frame leaves the box the same way
        win.timeline_goto(13)
        type_into(win, "再改一次")
        box(win).save_button.click()
        QApplication.processEvents()
        assert win.focusWidget() is not edit
    finally:
        app_scene.close_window(win)


def test_a_note_that_cannot_be_saved_on_close_goes_into_the_log(qapp, tmp_path, monkeypatch):
    """Round 2, N6: the window closes; the error and the text are in the log."""
    session = app_scene.make_session(tmp_path)
    win = MainWindow(session, app_scene.make_paths(tmp_path), "tester",
                     sam_queue=app_scene.StubSamQueue())
    win.show()
    QApplication.processEvents()
    if win.roi_editing:
        win.act_clear_edit()
    win.set_mode(A.MODE_REVIEW)
    type_into(win, "关窗前没存的说明")

    def refuse(*_args, **_kwargs):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(session, "set_frame_note", refuse)
    errors: list[str] = []
    real = win.report_error
    monkeypatch.setattr(win, "report_error", lambda text: (errors.append(text), real(text)))
    win.close()
    assert win.closed is True
    [line] = [e for e in errors if "frame note" in e]
    assert "disk gone" in line and "关窗前没存的说明" in line
    logged = "".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in Path(tmp_path).rglob("tda_app.log*"))
    assert "关窗前没存的说明" in logged
    win.hide()
    win.deleteLater()
    QApplication.processEvents()


def test_no_key_was_added():
    assert not [a for a in A.ACTIONS if "note" in a.name or "note" in a.slot
                or "说明" in a.label_zh]


# --------------------------------------------------------------------------- #
# where a note shows
# --------------------------------------------------------------------------- #
def test_the_card_shows_the_note_in_annotate_only(window):
    step = int(window.session.current().step)
    assert window.task_card.note_text() == ""
    window.session.set_frame_note(NOTE)
    QApplication.processEvents()
    line = window.task_card.note_line
    assert window.task_card.note_text() == "说明：screw.psu.03 被硬盘架挡住，按看不见处理…"
    assert NOTE in line.toolTip() and "tester" in line.toolTip()
    assert line.isVisible()
    review(window)
    assert window.task_card.note_text() == ""             # Review has the box instead
    window.set_mode(A.MODE_ANNOTATE)
    assert window.task_card.note_text().startswith("说明：")
    other = [s for s in window.session.available_steps() if s != step][0]
    window.timeline_goto(other)
    assert window.task_card.note_text() == "" and not line.isVisible()


def test_the_timeline_marks_the_frame(window):
    step = int(window.session.current().step)
    assert window.timeline.noted_steps() == []
    window.session.set_frame_note(NOTE)
    assert window.timeline.noted_steps() == [step]
    lw = window.timeline.list_widget()
    item = next(lw.item(r) for r in range(lw.count())
                if int(lw.item(r).data(int(Qt.ItemDataRole.UserRole))) == step)
    assert item.text() == f"{NOTE_GLYPH} Step {step}"
    assert item.toolTip() == "✎ 这一帧有说明：screw.psu.03 被硬盘架挡住，按看不见处理…"
    assert item.data(NOTE_ROLE)
    window.act_undo()
    assert window.timeline.noted_steps() == [] and item.text() == f"Step {step}"


def test_the_steps_table_marks_the_step_next_to_its_thumbnail(window):
    step = int(window.session.current().step)
    window.session.set_frame_note(NOTE)
    window.session.set_frame_note("oak1 的说明", key=FrameKey(DESKTOP, step, "oak1"))
    window.set_mode(A.MODE_STEPS)
    model = window.steps_panel.steps_model
    after = next(i for i, c in enumerate(STEP_COLUMNS) if c.kind == "thumb" and c.offset == 0)
    before = next(i for i, c in enumerate(STEP_COLUMNS) if c.kind == "thumb" and c.offset == -1)
    row = model.first_row_of(step)
    cell = model.index(row, after)
    assert model.data(cell, Qt.ItemDataRole.DisplayRole) == STEPS_GLYPH
    tip = model.data(cell, Qt.ItemDataRole.ToolTipRole)
    assert f"{STEPS_GLYPH} scan：{NOTE}" in tip and f"{STEPS_GLYPH} oak1：oak1 的说明" in tip
    assert model.data(model.index(row, before), Qt.ItemDataRole.DisplayRole) is None
    other = model.first_row_of(step - 1)
    assert model.data(model.index(other, after), Qt.ItemDataRole.DisplayRole) is None
    assert not window._steps_dirty                        # a note is no Steps edit

    # round 2, N7: a long note is cut to 120 characters in the tooltip
    window.session.set_frame_note("长" * 200, step=step - 1)
    window.steps_panel.refresh_notes()
    tip = model.data(model.index(other, after), Qt.ItemDataRole.ToolTipRole)
    assert f"{STEPS_GLYPH} scan：{'长' * 120}…" in tip and "长" * 121 not in tip
