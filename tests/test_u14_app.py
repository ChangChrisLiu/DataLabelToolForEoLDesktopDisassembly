"""Task U14 (window): ``Shift+T`` on a screw, the verb button, the palette and the card.

The synthetic D13 scene of the window tests (steps 1-14): the four CPU-cooler
screws are captive (a Dell), unscrewed at 3-6 and out with the cooler at 13.
``Shift+T`` on one says it was taken out when it was unscrewed
(「其实拧下来了」); on one that is not captive, that it stayed in its part
(「其实留在零件里」).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import QApplication

from app_scene import (
    COOLER,
    DESKTOP,
    LAST_STEP,
    SCREWS,
    StubSamQueue,
    close_window,
    make_paths,
    make_session,
    seed_shapes,
)
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.app_verb import CAPTIVE_STAGED
from tda.ui.panels.instances import VERB_STAYS_IN, VERB_TAKEN_OUT, VERB_UNSCREWED
from tda.ui.panels.taskcard import captive_hint

SCREW = SCREWS[0]                      # unscrewed at 3
BOARD_SCREW = "screw.motherboard.01"   # never unscrewed in the scene
COVER = "cover.01"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()
    seed_shapes(win.session, LAST_STEP)
    win.session.goto(10, force=True)
    QApplication.processEvents()
    yield win
    close_window(win)


def captive(win: MainWindow, target: str = SCREW):
    return win.db.instances(DESKTOP)[target].attrs.get("captive")


def press(win: MainWindow, spec: str) -> bool:
    combo = QKeySequence.fromString(spec)[0]
    return win.handle_key(QKeyEvent(QKeyEvent.Type.KeyPress, int(combo.key()),
                                    combo.keyboardModifiers()))


def select_removed(win: MainWindow, target: str) -> None:
    win.instances.show_removed.setChecked(True)
    QApplication.processEvents()
    table = win.instances.table()
    row = next(i for i in range(table.rowCount())
               if table.item(i, 1) is not None and table.item(i, 1).text() == target)
    table.setCurrentCell(row, 1)
    assert win.instances.selected_removed_instance() == target


def test_shift_t_on_a_loosened_screw_takes_it_out_and_says_from_when(window):
    assert window.instances.select_instance(SCREW)
    assert press(window, "Shift+T") is True
    assert captive(window) is False
    label = window._label_of(SCREW)[0]
    message = window.status_message()
    assert message.startswith(f"{label} 其实拧下来了：从第 3 步起算拧下拿走")
    assert message.endswith("（Ctrl+Z 撤销）")
    assert SCREW not in {r["key"] for r in window.instances.rows()}
    window.act_undo()
    assert captive(window) is True
    assert SCREW in {r["key"] for r in window.instances.rows()}
    window.act_redo()
    assert captive(window) is False


def test_the_button_says_which_way_and_does_what_the_key_does(window):
    button = window.instances.verb_button
    window.instances.select_instance(BOARD_SCREW)
    assert not button.isEnabled()                       # the log never unscrews it
    window.instances.select_instance(SCREW)
    assert button.isEnabled() and button.text() == VERB_UNSCREWED
    window.instances.select_instance(COVER)
    assert button.text() == VERB_TAKEN_OUT              # covers keep theirs (U9)
    window.instances.select_instance(SCREW)
    button.click()
    QApplication.processEvents()
    assert captive(window) is False
    # out of the machine now: the Removed rows offer the way back
    select_removed(window, SCREW)
    assert button.isEnabled() and button.text() == VERB_STAYS_IN
    button.click()
    QApplication.processEvents()
    assert captive(window) is True
    parent = window._label_of(COOLER)[0]
    assert f"其实留在零件里：第 3 步拧松后留在 {parent} 里" in window.status_message()


def test_shift_t_on_a_removed_screw_row_keeps_it_in_its_part(window):
    window.instances.select_instance(SCREW)
    window.act_correct_verb()
    select_removed(window, SCREW)
    window.act_correct_verb()
    assert captive(window) is True


def test_the_card_row_of_a_loosened_screw_names_the_key(window):
    assert captive_hint() == "拧松后留在零件里；画面里没有它就 Shift+T"
    window.session.goto(12, force=True)                 # the cooler and its screws are back
    QApplication.processEvents()
    texts = {r["instance"]: t for r, t in zip(window.task_card.rows(),
                                              window.task_card.row_texts())}
    assert all(captive_hint() in texts[key] for key in SCREWS)
    window.task_card.sigRequestEdit.emit(SCREW)         # click the row, press the key
    QApplication.processEvents()
    window.act_correct_verb()
    assert captive(window) is False
    texts = {r["instance"]: t for r, t in zip(window.task_card.rows(),
                                              window.task_card.row_texts())}
    assert SCREW not in texts and captive_hint() in texts[SCREWS[1]]


def test_the_palette_button_keeps_its_caption_and_says_screws_too(window):
    button = window.palette.button("correct_verb")
    assert button.name == "其实拆下"
    assert "螺丝：其实拧下来了 / 其实留在零件里" in button.toolTip()
    window.instances.select_instance(SCREW)
    window.refresh_guidance()
    assert button.isEnabled() and button.reason() == ""
    window.instances.select_instance(BOARD_SCREW)
    window.refresh_guidance()
    assert not button.isEnabled() and button.reason().startswith("日志里没有拧下")
    window.handle_key(QKeyEvent(QKeyEvent.Type.KeyPress, int(Qt.Key.Key_T),
                                Qt.KeyboardModifier.ShiftModifier))
    assert window.status_message() == f"改不了：{button.reason()}"
    window.instances.select_instance(SCREW)
    window.refresh_guidance()
    from PySide6.QtTest import QTest
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     button.rect().center())
    QApplication.processEvents()
    assert captive(window) is False


def test_the_status_line_counts_the_confirmed_frames_re_checked(window):
    seed_shapes(window.session, 12)                     # the screws drawn at 12
    window.session.goto(12, force=True)
    assert window.session.confirm_frame() is True
    window.session.goto(10, force=True)
    QApplication.processEvents()
    window.instances.select_instance(SCREW)
    window.act_correct_verb()
    assert "；已确认的 1 帧重新核对，其中 1 帧确认时画着它、现在算已拿走，会开冲突" in \
        window.status_message()
    window.session.truth.run_pending_rechecks(DESKTOP, "scan")   # the conflict opens
    select_removed(window, SCREW)
    window.act_correct_verb()                                    # Shift+T again, not Ctrl+Z
    message = window.status_message()
    assert "；已确认的 1 帧重新核对，1 帧又和确认时一样，会关闭 1 个冲突" in message
    assert "会开冲突" not in message


def test_refused_while_steps_mode_holds_edits_not_applied(window):
    window.set_mode(A.MODE_STEPS)
    panel = window.steps_panel
    panel.data.apply_instance_edit(SCREWS[1], "captive", False)   # the gate missed it
    assert not window._steps_dirty and panel.has_unapplied_edits()
    window.set_mode(A.MODE_ANNOTATE)
    window.session.goto(10, force=True)
    window.instances.select_instance(SCREW)
    window.refresh_guidance()
    button = window.palette.button("correct_verb")
    assert not button.isEnabled() and button.reason() == CAPTIVE_STAGED   # review N4
    window.act_correct_verb()
    assert window.status_message() == f"改不了：{CAPTIVE_STAGED}"
    assert captive(window) is True
    window.instances.select_instance(COVER)               # a cover's button stays live
    window.refresh_guidance()
    assert button.isEnabled() and button.reason() == ""
    window.instances.select_instance(SCREW)
    window._mark_steps_dirty()                            # the flag alone refuses too
    panel.data.apply_instance_edit(SCREWS[1], "captive", True)
    window.act_correct_verb()
    assert window.status_message() == f"改不了：{CAPTIVE_STAGED}"
    # a cover's verb is not the instance table: U9's correction is not refused
    window.instances.select_instance(COVER)
    window.act_correct_verb()
    assert [a.verb for a in window.db.actions(DESKTOP) if a.target == COVER] == ["remove"]


def test_a_write_through_delete_in_steps_is_neither_a_move_nor_an_edit(window):
    """Steps deletes an orphaned screw at once; the captive snapshot keeps in step."""
    window.set_mode(A.MODE_STEPS)
    panel = window.steps_panel
    model = panel.steps_model
    target = next(c for c in range(model.columnCount())
                  if model.headerData(c, Qt.Orientation.Horizontal) == "Target")
    assert model.setData(model.index(3, target), SCREW, Qt.ItemDataRole.EditRole)  # step 4
    panel.delete_instance(SCREWS[1])
    assert SCREWS[1] not in window.db.instances(DESKTOP)
    assert not panel.database_moved()
    panel.apply()
    assert not panel.has_unapplied_edits() and not panel.database_moved()


def test_the_steps_panel_is_re_read_after_a_flip(window):
    window.set_mode(A.MODE_STEPS)
    panel = window.steps_panel
    window.set_mode(A.MODE_ANNOTATE)
    window.session.goto(10, force=True)
    window.instances.select_instance(SCREW)
    window.act_correct_verb()
    window.set_mode(A.MODE_STEPS)
    assert panel.data.instances[SCREW].attrs["captive"] is False
    panel.apply()                                         # an Apply keeps the flip
    assert captive(window) is False


def test_review_mode_refuses_the_button(window):
    window.set_mode(A.MODE_REVIEW)
    window.on_verb_picked(SCREW, "unscrewed")
    assert "只在标注模式可用" in window.status_message()
    assert captive(window) is True
