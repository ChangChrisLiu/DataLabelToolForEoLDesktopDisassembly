"""Task U39 in the main window: 画框外 is ``Ctrl+```'s fourth choice, and only that.

The 14-step D13 window scene (:mod:`app_scene`), every part of step 14 drawn,
the window on step 12 -- where the cooler and its four captive screws have no
shape. Pinned: the dialog lists 「画框外（这个视角拍不到）」 fourth and starts on it
for a part that is 画框外; the span is one op and one ``Ctrl+Z``; the status
line, the instance table (框外) and the card row (「画框外」, done) say it; the
grave key's cycle never reaches it and takes it back; ``Space`` goes through
once nothing else is missing. No key was added.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import (
    COOLER,
    DESKTOP,
    LAST_STEP,
    SCREWS,
    VIEW,
    StubSamQueue,
    close_window,
    make_paths,
    make_session,
    seed_shapes,
)
from test_app_actions import key_event
from tda.core.model import FrameKey
from tda.core.occluder import FULL, OFF_FRAME, PARTIAL, frame_label
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.app_cable import OFF_FRAME_OFF, CableRangeDialog, choice_text
from tda.ui.panels.taskcard import OFF_FRAME_CHIP, row_view
from tda.ui.session_cable import OFF_FRAME_CHOICE, OFF_FRAME_TAG, OP_RANGE

CHASSIS = "chassis"
STEP = 12


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def answer_roi(win: MainWindow) -> None:
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()


@pytest.fixture
def window(qapp, tmp_path: Path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(1400, 900)
    answer_roi(win)
    seed_shapes(win.session, LAST_STEP)
    win.session.goto(STEP, force=True)
    QApplication.processEvents()
    yield win
    close_window(win)


def said(win: MainWindow, step: int = STEP, instance: str = COOLER):
    return frame_label(win.db.frame_overrides(FrameKey(DESKTOP, step, VIEW)).get(instance))


def vis_cell(win: MainWindow, instance: str) -> str:
    table = win.instances.table()
    for row in range(table.rowCount()):
        item = table.item(row, 1)
        if item is not None and item.text() == instance:
            return table.item(row, 3).text()
    raise AssertionError(f"no row for {instance}")


def card_row(win: MainWindow, instance: str):
    return next((r for r in win.task_card.rows() if r.get("instance") == instance), None)


# --------------------------------------------------------------------------- #
# the dialog
# --------------------------------------------------------------------------- #
def test_the_dialog_offers_off_frame_fourth(qapp):
    dialog = CableRangeDialog(None, "CPU 散热器", STEP, FULL, 1, LAST_STEP)
    box = dialog.label_box
    assert [box.itemText(i) for i in range(box.count())] == [
        "被线挡住（完全）", "被线挡住（部分）", "撤回（被线挡住 / 画框外）", OFF_FRAME_CHOICE]
    assert OFF_FRAME_CHOICE == "画框外（这个视角拍不到）"
    assert [box.itemData(i) for i in range(box.count())] == [FULL, PARTIAL, None, OFF_FRAME]
    assert box.currentIndex() == 0
    box.setCurrentIndex(3)
    assert dialog.answer() == (OFF_FRAME, STEP)
    # a part already 画框外 opens on it
    assert CableRangeDialog(None, "x", STEP, OFF_FRAME, 1, LAST_STEP).label_box.currentIndex() == 3
    assert choice_text(OFF_FRAME) == OFF_FRAME_CHOICE
    dialog.deleteLater()


def test_no_key_was_added():
    assert A.action_named("cable_range").keys == ("Ctrl+`",)
    assert A.action_named("mark_absent").keys == ("`",)
    assert "画框外" in A.action_named("cable_range").label_zh


# --------------------------------------------------------------------------- #
# Ctrl+`: 画框外 over a span
# --------------------------------------------------------------------------- #
def test_ctrl_grave_sets_off_frame_over_a_span_in_one_op(window, monkeypatch):
    asked: dict = {}

    def answer(instance, value, low, high):
        asked.update(instance=instance, value=value)
        return OFF_FRAME, 10

    monkeypatch.setattr(window, "ask_cable_range", answer)
    assert window.instances.select_instance(COOLER)
    window.handle_key(key_event("Ctrl+`"))
    assert asked == {"instance": COOLER, "value": FULL}
    assert [said(window, s) for s in (9, 10, 11, 12, 13)] == [None, OFF_FRAME, OFF_FRAME,
                                                               OFF_FRAME, None]
    assert "第 10–12 步都标为画框外" in window.status_message()
    ops = [op for op in window.db.ops(DESKTOP, VIEW, limit=100) if op["kind"] == OP_RANGE]
    assert len(ops) == 1
    # every panel says it
    assert vis_cell(window, COOLER) == OFF_FRAME_TAG == "框外"
    row = card_row(window, COOLER)
    assert row is not None and row["done"] and row["off_frame"]
    assert row_view(row)["chip"] == OFF_FRAME_CHIP
    # the dialog starts on what the frame says now
    asked.clear()
    window.instances.select_instance(COOLER)
    window.handle_key(key_event("Ctrl+`"))
    assert asked["value"] == OFF_FRAME
    window.act_undo()
    assert all(said(window, s) is None for s in (10, 11, 12))


def test_ctrl_grave_withdraw_says_what_it_took_back(window, monkeypatch):
    monkeypatch.setattr(window, "ask_cable_range", lambda *a: (OFF_FRAME, 11))
    assert window.instances.select_instance(COOLER)
    window.handle_key(key_event("Ctrl+`"))
    monkeypatch.setattr(window, "ask_cable_range", lambda *a: (None, 11))
    window.instances.select_instance(COOLER)
    window.handle_key(key_event("Ctrl+`"))
    assert "第 11–12 步：撤回画框外" in window.status_message()
    assert said(window, 11) is None and said(window, 12) is None


def test_ctrl_grave_names_the_frames_the_part_is_taken_out_on(window, monkeypatch):
    monkeypatch.setattr(window, "ask_cable_range", lambda *a: (OFF_FRAME, LAST_STEP))
    assert window.instances.select_instance(COOLER)
    window.handle_key(key_event("Ctrl+`"))
    assert "第 13–14 步" in window.status_message()
    assert said(window, STEP) is None


def test_a_drawn_part_off_frame_keeps_the_hidden_shape_warning(window, monkeypatch):
    """The U2b round-3 rule: a drawn shape a label hides is said (⚠), and here why."""
    monkeypatch.setattr(window, "ask_cable_range", lambda *a: (OFF_FRAME, STEP))
    assert window.instances.select_instance(CHASSIS)
    window.handle_key(key_event("Ctrl+`"))
    assert said(window, STEP, CHASSIS) == OFF_FRAME
    table = window.instances.table()
    line = next(i for i in range(table.rowCount())
                if table.item(i, 1) is not None and table.item(i, 1).text() == CHASSIS)
    cell = table.item(line, 3)
    assert cell.text().startswith("⚠")
    assert "已设为不可见" in cell.toolTip() and "画框外" in cell.toolTip()


def test_drawing_with_q_on_an_off_frame_frame_keeps_the_label_and_warns(window, monkeypatch):
    """Round 2, ruled: Q does not take 画框外 back (labels are labels on the
    sequence, U10 round 2); the ⚠ cell says the drawn shape is hidden and why."""
    from app_scene import cell as app_cell
    from tda.ui import session_api as api

    monkeypatch.setattr(window, "ask_cable_range", lambda *a: (OFF_FRAME, STEP))
    assert window.instances.select_instance(COOLER)
    window.handle_key(key_event("Ctrl+`"))
    assert said(window) == OFF_FRAME
    session = window.session
    session.begin_edit(COOLER)
    session.set_editing_mask(app_cell(40))
    result = session.commit_edit(api.SCOPE_KEYFRAME)
    session.clear_edit()
    assert not result.get("cleared_absent")
    assert said(window) == OFF_FRAME                         # the label stays
    window.instances.refresh()
    window.task_card.refresh()
    QApplication.processEvents()
    table = window.instances.table()
    line = next(i for i in range(table.rowCount())
                if table.item(i, 1) is not None and table.item(i, 1).text() == COOLER)
    cell = table.item(line, 3)
    assert cell.text().startswith("⚠")
    assert "已设为不可见" in cell.toolTip() and "画框外" in cell.toolTip()
    row = card_row(window, COOLER)
    assert row is not None and row["done"] and row["off_frame"]
    sentence = row_view(row)["sentence"]
    assert sentence.startswith("画框外") and "⚠ 已设为不可见" in sentence


# --------------------------------------------------------------------------- #
# the grave key's cycle
# --------------------------------------------------------------------------- #
def test_the_grave_cycle_never_sets_it(window):
    seen = []
    for _ in range(3):
        assert window.instances.select_instance(CHASSIS)
        window.handle_key(key_event("`"))
        seen.append(said(window, STEP, CHASSIS))
    assert seen == [FULL, PARTIAL, None]


def test_the_grave_key_takes_it_back(window, monkeypatch):
    monkeypatch.setattr(window, "ask_cable_range", lambda *a: (OFF_FRAME, STEP))
    assert window.instances.select_instance(COOLER)
    window.handle_key(key_event("Ctrl+`"))
    assert said(window) == OFF_FRAME
    window.instances.select_instance(COOLER)
    window.handle_key(key_event("`"))
    assert said(window) is None
    label = window._label_of(COOLER)[0] or COOLER
    assert OFF_FRAME_OFF.format(label=label) in window.status_message()
    window.act_undo()
    assert said(window) == OFF_FRAME


# --------------------------------------------------------------------------- #
# Space
# --------------------------------------------------------------------------- #
def test_space_is_not_held_up_by_it(window, monkeypatch):
    key = FrameKey(DESKTOP, STEP, VIEW)
    window.handle_key(key_event("Space"))
    assert window.db.get_frame(key)["review_status"] != "verified"   # five shapes missing
    window.session.goto(STEP, force=True)
    QApplication.processEvents()
    monkeypatch.setattr(window, "ask_cable_range", lambda *a: (OFF_FRAME, STEP))
    for part in (COOLER, *SCREWS):
        assert window.instances.select_instance(part)
        window.handle_key(key_event("Ctrl+`"))
        assert said(window, STEP, part) == OFF_FRAME
    assert all(r["done"] for r in window.task_card.rows())
    window.handle_key(key_event("Space"))
    assert window.db.get_frame(key)["review_status"] == "verified"
