"""Task U9 (window): ``Shift+T``, the verb button and the card's hint.

The synthetic D13 scene of the window tests (steps 1-14): ``cover.01`` is
opened by the log at 2 and ``drive_latch.01`` at 7, both drawn at 14. Saying
one of them was taken out removes it from every frame from its step on; a part
in the ``Removed`` rows goes the other way.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_scene import (
    DESKTOP,
    LAST_STEP,
    StubSamQueue,
    close_window,
    make_paths,
    make_session,
    seed_shapes,
)
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.app_verb import NO_VERB_TARGET
from tda.ui.panels.instances import VERB_ONLY_OPENED, VERB_TAKEN_OUT

COVER = "cover.01"         # the log opens it at 2
LATCH = "drive_latch.01"   # ... and this one at 7


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
    win.session.goto(LAST_STEP, force=True)
    QApplication.processEvents()
    yield win
    close_window(win)


def verb(win: MainWindow, target: str) -> list[tuple]:
    return [(a.step, a.verb) for a in win.db.actions(DESKTOP) if a.target == target]


def card_instances(win: MainWindow) -> set[str]:
    return {str(r["instance"]) for r in win.task_card.rows()}


def test_shift_t_is_bound_in_annotate_mode_only():
    action = A.action_named("correct_verb")
    assert action.keys == ("Shift+T",) and action.slot == "act_correct_verb"
    assert action.modes == (A.MODE_ANNOTATE,)
    shift = Qt.KeyboardModifier.ShiftModifier
    assert A.action_for(Qt.Key.Key_T, shift, A.MODE_ANNOTATE) is action
    assert A.action_for(Qt.Key.Key_T, shift, A.MODE_STEPS) is None
    assert A.action_for(Qt.Key.Key_T, shift, A.MODE_REVIEW) is None
    assert A.action_for(Qt.Key.Key_T, Qt.KeyboardModifier.NoModifier,
                        A.MODE_ANNOTATE).name == "cycle_state"


def test_shift_t_takes_the_selected_part_out_and_says_where_from(window):
    assert window.instances.select_instance(COVER)
    window.act_correct_verb()
    assert verb(window, COVER) == [(2, "remove")]
    message = window.status_message()
    assert "cover.01 在第 2 步是拆下的（不是打开）：从第 2 步起它不在机箱里了" in message
    assert "这些帧上它的形状不再使用（13 帧）" in message and "Ctrl+Z" in message
    assert COVER not in card_instances(window)
    assert COVER not in {r["key"] for r in window.instances.rows()}
    window.act_undo()
    assert verb(window, COVER) == [(2, "open")]
    assert COVER in card_instances(window)
    window.act_redo()
    assert verb(window, COVER) == [(2, "remove")]


def test_shift_t_acts_on_the_part_a_card_click_started(window):
    """Click the row, press the key: the untouched edit goes with the part."""
    window.on_request_edit(LATCH)
    window.instances.table().setCurrentCell(-1, -1)
    window.act_correct_verb()
    assert verb(window, LATCH) == [(7, "remove")]
    assert window.session.editing_instance is None
    assert LATCH not in card_instances(window)


def test_a_part_with_pixels_not_committed_is_refused(window):
    window.on_request_edit(COVER)
    window.act_tool("brush")
    viewport = window.canvas.viewport()
    centre = viewport.rect().center()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     centre)
    QTest.mouseMove(viewport, centre + QPoint(8, 0))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, centre + QPoint(8, 0))
    QApplication.processEvents()
    window.instances.table().setCurrentCell(-1, -1)
    window.act_correct_verb()
    assert "正在画、还没提交" in window.status_message()
    assert verb(window, COVER) == [(2, "open")]


def test_the_button_is_the_same_gesture_both_ways(window):
    button = window.instances.verb_button
    assert button.focusPolicy() == Qt.FocusPolicy.NoFocus
    window.instances.select_instance("chassis")
    assert not button.isEnabled()
    window.instances.select_instance(COVER)
    assert button.isEnabled() and button.text() == VERB_TAKEN_OUT
    button.click()
    QApplication.processEvents()
    assert verb(window, COVER) == [(2, "remove")]
    # B: the removed row offers the way back
    window.instances.show_removed.setChecked(True)
    QApplication.processEvents()
    table = window.instances.table()
    row = next(i for i in range(table.rowCount())
               if table.item(i, 1) is not None and table.item(i, 1).text() == COVER)
    table.setCurrentCell(row, 1)
    assert window.instances.selected_removed_instance() == COVER
    assert button.isEnabled() and button.text() == VERB_ONLY_OPENED
    button.click()
    QApplication.processEvents()
    assert verb(window, COVER) == [(2, "open")]
    message = window.status_message()
    assert "cover.01 在第 2 步只是打开了（不是拆下）：从第 2 步起它还在机箱里" in message
    assert COVER in card_instances(window)


def test_shift_t_on_a_removed_row_puts_it_back(window):
    window.instances.select_instance(COVER)
    window.act_correct_verb()
    window.instances.show_removed.setChecked(True)
    QApplication.processEvents()
    table = window.instances.table()
    row = next(i for i in range(table.rowCount())
               if table.item(i, 1) is not None and table.item(i, 1).text() == COVER)
    table.setCurrentCell(row, 1)
    window.act_correct_verb()
    assert verb(window, COVER) == [(2, "open")]


def test_refusals_are_said(window):
    window.instances.table().setCurrentCell(-1, -1)
    window.act_correct_verb()
    assert window.status_message() == NO_VERB_TARGET
    window.instances.select_instance("ram_latch.01")
    window.act_correct_verb()
    assert window.status_message().startswith("改不了：") and "只能打开" in \
        window.status_message()
    window.set_mode(A.MODE_REVIEW)
    window.on_verb_picked(COVER, "remove")
    assert "只在标注模式可用" in window.status_message()
    assert not window.instances.verb_button.isEnabled()
    assert verb(window, COVER) == [(2, "open")]


def test_the_steps_panel_is_re_read_after_a_correction(window):
    window.set_mode(A.MODE_STEPS)
    panel = window.steps_panel
    window.set_mode(A.MODE_ANNOTATE)
    window.session.goto(LAST_STEP, force=True)
    window.instances.select_instance(COVER)
    window.act_correct_verb()
    window.set_mode(A.MODE_STEPS)
    assert [a.verb for a in panel.data.row(2).actions] == ["remove"]
    # an Apply from there now keeps the correction
    panel.apply()
    assert verb(window, COVER) == [(2, "remove")]
