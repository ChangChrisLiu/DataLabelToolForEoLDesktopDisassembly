"""Task U34 (session and window): 「一开始就不在（这台没有它）」 from the ``T`` picker.

The annotator stood on frame 1 of D34, selected the optical drive the sheet
logs and the machine does not have, pressed ``T`` -- and could choose
「挪开」 only: ``removed`` is not a state a part *in the picture* can be set to
(:func:`tda.core.state_assert.assertable_states`). On the first frame of a part
nobody has drawn anywhere the picker now offers one more entry; it asks, then
writes the op (:mod:`tda.core.absent`), and the part leaves every frame's card.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from absent_scene import DRIVE, add_drive
from app_scene import (
    DESKTOP,
    LAST_STEP,
    StubSamQueue,
    close_window,
    make_paths,
    make_session,
    seed_shapes,
)
from tda.core import absent as AB
from tda.ui.app import MainWindow
from tda.ui.app_state import ABSENT_DONE
from tda.ui.session_api import SessionRefusal


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    add_drive(session.db, session.tax, DESKTOP)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()
    seed_shapes(win.session, LAST_STEP, skip=(DRIVE,))
    win.asked = []

    def answer(label: str, instance: str, notes=()) -> bool:
        win.asked.append(instance)
        win.notes = list(notes)
        return getattr(win, "answer", True)

    win.ask_absent_from_start = answer
    yield win
    close_window(win)


def goto(win: MainWindow, step: int) -> None:
    win.session.goto(step, force=True)
    QApplication.processEvents()


def manual(win: MainWindow, key: str = DRIVE) -> list[tuple]:
    return [(e.step, e.attr, e.new) for e in win.db.events(DESKTOP)
            if not e.auto and e.target == key]


def row(win: MainWindow, key: str = DRIVE):
    return next((r for r in win.session.instance_rows() if r["key"] == key), None)


def missing(win: MainWindow) -> list[str]:
    return [p for p in win.session.current_problems() if p == f"missing_shape:{DRIVE}"]


# --------------------------------------------------------------------------- #
# where it is offered
# --------------------------------------------------------------------------- #
def test_the_picker_offers_it_on_the_first_frame_only(window):
    goto(window, 1)
    assert row(window)["state_choices"] == ["installed", "displaced", AB.CHOICE]
    # a part somebody drew is in the machine: not offered
    assert AB.CHOICE not in row(window, "cover.01")["state_choices"]
    window.instances.select_instance(DRIVE)
    combo = window.instances.state_combo
    texts = [combo.itemText(i) for i in range(combo.count())]
    assert texts == ["状态：装着", "状态：挪开", AB.CAPTION]
    assert combo.currentText() == "状态：装着"
    goto(window, 2)
    assert AB.CHOICE not in row(window)["state_choices"]
    assert window.session.next_state(DRIVE) == "displaced"
    with pytest.raises(SessionRefusal, match="只在第一帧"):
        window.session.mark_absent_from_start(DRIVE)
    assert manual(window) == []


def test_a_one_state_part_gets_it_too_and_the_combo_shows_what_holds(window):
    """A plugged connector has no state ``T`` could set; on frame 1 it has this one."""
    plug = "connector.01"                       # unplugged at 8: never drawn at 14
    goto(window, 1)
    assert row(window, plug)["state_choices"] == [AB.CHOICE]
    window.instances.select_instance(plug)
    combo = window.instances.state_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["状态：插上", AB.CAPTION]
    assert combo.currentText() == "状态：插上" and combo.isEnabled()
    # T has no state to set on it, as before (round 2: the choice is not T's)
    assert window.session.next_state(plug) is None
    window.act_cycle_state()
    assert window.asked == [] and window.status_message().startswith("改不了状态：")
    assert "只有一种" in window.status_message()
    assert window.state_button_refusal("cycle_state", False) != ""
    combo.activated.emit(1)                     # the combo's 「一开始就不在」
    QApplication.processEvents()
    assert window.asked == [plug] and AB.is_absent(window.db, DESKTOP, plug)


def test_picking_it_asks_then_the_part_leaves_every_frame(window):
    goto(window, 1)
    assert missing(window)                     # the card's 「这一帧还缺形状」
    window.instances.select_instance(DRIVE)
    combo = window.instances.state_combo
    combo.activated.emit(combo.count() - 1)    # what a click on the last entry sends
    QApplication.processEvents()
    assert window.asked == [DRIVE]
    assert manual(window) == [(0, "state", "removed"), (0, "placement", "elsewhere")]
    message = window.status_message()
    assert ABSENT_DONE.format(label="").strip("：") in message and "Ctrl+Z" in message
    assert row(window) is None and missing(window) == []
    for step in (2, 9, 11, LAST_STEP):
        goto(window, step)
        assert row(window) is None and missing(window) == []
        assert all(r.get("instance") != DRIVE for r in window.task_card.rows())
    ops = window.db.ops(DESKTOP, "scan", limit=3)
    assert ops[0]["kind"] == AB.OP_KIND


def test_answering_no_writes_nothing(window):
    goto(window, 1)
    window.answer = False
    window.instances.select_instance(DRIVE)
    window.assert_state_here(DRIVE, AB.CHOICE)
    assert window.asked == [DRIVE] and manual(window) == []
    assert "没有改" in window.status_message()


def test_t_only_toggles_the_states_and_never_asks(window):
    """Round 2 (ruling): the choice is the combo's and Steps', not a stop of ``T``."""
    goto(window, 1)
    for expected in ("displaced", "installed", "displaced", "installed"):
        window.instances.select_instance(DRIVE)
        assert window.session.next_state(DRIVE) == expected
        window.act_cycle_state()
        assert row(window)["state"] == expected
    assert window.asked == [] and not AB.is_absent(window.db, DESKTOP, DRIVE)
    assert all(step > 0 for step, _attr, _new in manual(window))


def test_ctrl_z_and_ctrl_y(window):
    goto(window, 1)
    window.instances.select_instance(DRIVE)
    window.assert_state_here(DRIVE, AB.CHOICE)
    assert AB.is_absent(window.db, DESKTOP, DRIVE)
    window.act_undo()
    assert manual(window) == [] and row(window) is not None and missing(window)
    window.act_redo()
    assert AB.is_absent(window.db, DESKTOP, DRIVE) and row(window) is None


def test_the_removed_rows_say_what_it_is(window):
    goto(window, 1)
    window.assert_state_here(DRIVE, AB.CHOICE)
    window.instances.show_removed.setChecked(True)
    QApplication.processEvents()
    from tda.ui import app_compat as compat

    gone = {r["key"]: r for r in compat.removed_rows(window.session)}
    assert gone[DRIVE]["state"] == AB.SHORT and gone[DRIVE]["verb_fix"] is None


def test_the_question_lists_residues_and_a_cable_label_refuses(window):
    """Round 2 (N5): what still names it is said; a 「被线挡住」 reason stops it."""
    from tda.core.model import FrameKey, FrameOverride

    goto(window, 1)
    window.db.set_frame_override(FrameOverride(FrameKey(DESKTOP, 2, "scan"), DRIVE,
                                               visibility="occluded_partial"))
    window.answer = False
    window.assert_state_here(DRIVE, AB.CHOICE)
    assert window.notes == ["单帧可见性标签 1 处：scan 第 2 帧"]
    window.db.set_frame_override(FrameOverride(FrameKey(DESKTOP, 3, "scan"), DRIVE,
                                               visibility="occluded_full", occluder="cable"))
    window.asked.clear()
    window.answer = True
    window.assert_state_here(DRIVE, AB.CHOICE)
    assert window.asked == [] and manual(window) == []
    assert "标着「被线挡住」" in window.status_message()


def test_a_part_being_drawn_with_pixels_is_refused(window):
    goto(window, 1)
    import numpy as np

    window.session.begin_edit(DRIVE)
    mask = np.zeros((64, 64), dtype=bool)
    mask[1:5, 1:5] = True
    window.session.set_editing_mask(mask)
    with pytest.raises(SessionRefusal, match="正在画"):
        window.session.mark_absent_from_start(DRIVE)
    assert manual(window) == []
