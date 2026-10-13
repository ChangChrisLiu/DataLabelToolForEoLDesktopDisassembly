"""Task U8 (window): ``T`` and the instance list's state combo.

The synthetic D13 scene of the window tests (steps 1-14): ``ram_latch.01`` is
opened by the log at 14, ``ram_latch.02`` stays closed throughout.  A
correction "``ram_latch.02`` is open at 12" makes a boundary the step table
does not have, and everything on screen has to treat it like one it does.
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
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.app_guide import BANNER_SPLIT
from tda.ui.app_state import NO_TARGET, SPLIT_NOW
from tda.ui.panels.instances import STATE_NONE

LOGGED = "ram_latch.01"        # the log opens it at 14: frame 13 is a ✂ row
CORRECTED = "ram_latch.02"     # closed throughout; asserted open at 12
STEP = 12


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    answer_roi(win)
    seed_shapes(win.session, LAST_STEP)
    yield win
    close_window(win)


def answer_roi(win: MainWindow) -> None:
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()


def goto(win: MainWindow, step: int) -> None:
    win.session.goto(step, force=True)
    QApplication.processEvents()


def manual(win: MainWindow, target: str) -> list[tuple]:
    return [(e.step, e.old, e.new) for e in win.db.events(DESKTOP)
            if not e.auto and e.target == target]


def card_row(win: MainWindow, instance: str):
    return next((r for r in win.task_card.rows() if r["instance"] == instance), None)


def brush_stroke(win: MainWindow) -> None:
    win.act_tool("brush")
    viewport = win.canvas.viewport()
    centre = viewport.rect().center()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, centre)
    QTest.mouseMove(viewport, centre + QPoint(8, 0))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, centre + QPoint(8, 0))
    QApplication.processEvents()


def test_t_is_bound_in_annotate_mode_only():
    action = A.action_named("cycle_state")
    assert action.keys == ("T",) and action.slot == "act_cycle_state"
    assert action.modes == (A.MODE_ANNOTATE,)
    assert A.action_for(Qt.Key.Key_T, Qt.KeyboardModifier.NoModifier,
                        A.MODE_STEPS) is None


def test_t_sets_the_next_state_and_says_what_holds_where(window):
    goto(window, STEP)
    assert window.instances.select_instance(CORRECTED)
    window.act_cycle_state()
    assert manual(window, CORRECTED) == [(STEP, "closed", "open"), (STEP + 1, "open", "closed")]
    message = window.status_message()
    assert "ram_latch.02 在第 12 步：打开（之前为 关上）；第 13 步起恢复 关上" in message
    assert SPLIT_NOW in message and "Ctrl+Z" in message
    assert window.canvas.banner_text().startswith("内存卡扣 ram_latch.02 在第 12 步")
    row = next(r for r in window.instances.rows() if r["key"] == CORRECTED)
    assert row["state"] == "open"
    assert window.instances.state_combo.currentText() == "状态：打开"
    # T again: back to closed, and the two events are gone
    window.act_cycle_state()
    assert manual(window, CORRECTED) == []


def test_a_run_of_frames_is_said_as_one(window):
    """Walking backwards over a run: the line names the run and where it ends."""
    goto(window, STEP)
    window.instances.select_instance(CORRECTED)
    window.act_cycle_state()
    goto(window, STEP - 1)
    window.instances.select_instance(CORRECTED)
    window.act_cycle_state()
    assert manual(window, CORRECTED) == [(STEP - 1, "closed", "open"),
                                         (STEP + 1, "open", "closed")]
    message = window.status_message()
    assert ("ram_latch.02 在第 11 步：打开（之前为 关上）；第 11–12 步都是 打开；"
            "第 13 步起恢复 关上") in message
    # frame 11 is diffed against 12, where it is open too: nothing to draw here
    assert SPLIT_NOW not in message


def test_t_with_nothing_selected_says_where_to_click(window):
    goto(window, STEP)
    window.instances.table().setCurrentCell(-1, -1)
    window.act_cycle_state()
    assert window.status_message() == NO_TARGET


def test_t_is_refused_in_words_where_there_is_nothing_to_set(window):
    goto(window, STEP)
    window.instances.select_instance("chassis")
    window.act_cycle_state()
    assert window.status_message().startswith("改不了状态：") and "只有一种" in \
        window.status_message()
    # the log opens ram_latch.01 at 14: "open at 13" would contradict it
    goto(window, 13)
    window.instances.select_instance(LOGGED)
    window.act_cycle_state()
    assert "Steps 模式" in window.status_message()
    assert manual(window, LOGGED) == []


def test_t_acts_on_the_part_being_drawn_when_nothing_is_selected(window):
    goto(window, STEP)
    window.on_request_edit(CORRECTED)
    window.instances.table().setCurrentCell(-1, -1)
    window.act_cycle_state()
    assert manual(window, CORRECTED) == [(STEP, "closed", "open"), (STEP + 1, "open", "closed")]
    assert window.session.editing_instance == CORRECTED     # the layer survives


def test_the_state_combo_is_the_same_gesture_and_never_takes_the_focus(window):
    goto(window, STEP)
    combo = window.instances.state_combo
    assert combo.focusPolicy() == Qt.FocusPolicy.NoFocus
    window.instances.select_instance("chassis")
    assert not combo.isEnabled() and combo.currentText() == STATE_NONE
    window.instances.select_instance(CORRECTED)
    assert combo.isEnabled()
    assert [combo.itemData(i) for i in range(combo.count())] == ["closed", "open"]
    assert combo.currentText() == "状态：关上"
    combo.activated.emit(1)                     # what a click on 打开 sends
    QApplication.processEvents()
    assert manual(window, CORRECTED) == [(STEP, "closed", "open"), (STEP + 1, "open", "closed")]
    assert "在第 12 步：打开" in window.status_message()
    combo.activated.emit(1)                     # the state that holds: nothing
    assert len(manual(window, CORRECTED)) == 2


def test_ctrl_z_takes_the_state_back(window):
    goto(window, STEP)
    window.instances.select_instance(CORRECTED)
    window.act_cycle_state()
    window.act_undo()
    assert manual(window, CORRECTED) == []
    window.act_redo()
    assert manual(window, CORRECTED) == [(STEP, "closed", "open"), (STEP + 1, "open", "closed")]


@pytest.mark.parametrize("instance,step", [(LOGGED, 13), (CORRECTED, STEP)],
                         ids=["log_boundary", "manual_boundary"])
def test_a_manual_boundary_gets_the_split_banner_a_log_boundary_gets(window, instance, step):
    """E: Enter instead of Ctrl+K on the new boundary is warned exactly the same."""
    if instance == CORRECTED:
        goto(window, step)
        window.instances.select_instance(instance)
        window.act_cycle_state()
    goto(window, step)
    row = card_row(window, instance)
    assert row is not None and row["kind"] == api.KIND_SPLIT_KEYFRAME
    assert row["done"] is False
    window.task_card.sigRequestEdit.emit(instance)
    brush_stroke(window)
    if instance == CORRECTED:
        # the change's own note holds the canvas for VIS_NOTE_MS, and it names
        # Ctrl+K too; once it has gone the banner is the edit's again
        assert SPLIT_NOW in window.canvas.banner_text()
        window.expire_visibility_note()
    plan = window.guide.plan()
    assert plan.phase == G.PHASE_DRAW_PIXELS
    assert plan.action == "commit_split"
    assert "Ctrl+K" in plan.now and "从这帧起新版本" in plan.now
    assert any(G.COMMIT_SPLIT in str(line) for line in plan.steps)
    assert window.canvas.banner_text().endswith(BANNER_SPLIT)
    assert window.palette.button("commit_split").suggested()
    assert not window.palette.button("commit").suggested()
