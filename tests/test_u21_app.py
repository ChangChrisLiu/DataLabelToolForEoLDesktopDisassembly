"""Task U21: where the annotator says 「其实是硬盘架」 / 「其实是卡扣」 (offscreen Qt).

* Annotate mode, the D13 window (``drive_latch.01`` "SSD shield locker" opened
  at 7, ``drive_cage.01`` "SSD shield" out at 11): ``Shift+T``, the instance
  list's button and the palette's 「其实拆下」 offer the other class on a drive
  latch / cage row; with a verb offered too a small picker asks which; the
  reclass is confirmed first, written as one undoable op, and refused while
  Steps holds edits;
* Steps mode: the Instances row's right-click 「其实是硬盘架…」, written through.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import StubSamQueue, close_window, make_paths
from session_scene import DESKTOP, LAST_STEP, make_session, seed_shapes
from tda.core.db import Db
from tda.core.reclass import OP_RECLASS
from tda.ui import app_verb as V
from tda.ui.app import MainWindow
from tda.ui.panels.instances import RECLASS, VERB_OR_RECLASS

LOCKER = "drive_latch.01"     # opened at 7, still in at 14
CAGE = "drive_cage.01"        # out at 11
LATCH = "drive_latch.02"      # what the cage becomes


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
    win.asked = {"pick": [], "confirm": []}

    def pick(instance, choices):
        win.asked["pick"].append((instance, list(choices)))
        return win.answer_pick

    def confirm(instance, lines):
        win.asked["confirm"].append((instance, list(lines)))
        return win.answer_confirm

    win.ask_verb_or_reclass = pick
    win.ask_reclass_confirm = confirm
    win.answer_pick, win.answer_confirm = None, True
    QApplication.processEvents()
    yield win
    close_window(win)


def cls_of(win, key):
    rec = win.db.instances(DESKTOP).get(key)
    return None if rec is None else rec.cls


def select_removed(win, key):
    win.instances.show_removed.setChecked(True)
    QApplication.processEvents()
    table = win.instances.table()
    for row in range(table.rowCount()):
        win.instances.table().setCurrentCell(row, 1)
        if win.instances.selected_removed_instance() == key:
            return
    raise AssertionError(f"{key} is not in the Removed rows")


# --------------------------------------------------------------------------- #
# the rows and the button
# --------------------------------------------------------------------------- #
def test_the_rows_offer_the_other_class(window):
    rows = {r["key"]: r for r in window.session.instance_rows()}
    assert rows[LOCKER]["reclass"] == "drive_cage" and rows[LOCKER]["verb_fix"] == "remove"
    assert rows["cover.01"]["reclass"] is None
    window.instances.select_instance(LOCKER)
    assert window.instances.verb_button.text() == VERB_OR_RECLASS
    assert window.instances.verb_button.isEnabled()
    select_removed(window, CAGE)
    assert window.instances.verb_offer() == (CAGE, RECLASS)
    assert window.instances.verb_button.text() == "其实是卡扣"
    assert window.instances.verb_button.isEnabled()
    # the palette's 「其实拆下」 is live on it (it greyed a cage before)
    assert window.state_button_refusal("correct_verb", False) == ""


# --------------------------------------------------------------------------- #
# Shift+T on a latch that also offers a verb: the picker
# --------------------------------------------------------------------------- #
def test_shift_t_asks_which_and_the_verb_still_works(window):
    window.instances.select_instance(LOCKER)
    window.answer_pick = "remove"
    window.act_correct_verb()
    [(instance, choices)] = window.asked["pick"]
    assert instance == LOCKER
    assert choices == [("remove", "其实拆下来了"), (RECLASS, "其实是硬盘架")]
    assert [(a.step, a.verb) for a in window.db.actions(DESKTOP) if a.target == LOCKER] == \
        [(7, "remove")]
    assert not window.asked["confirm"]


def test_cancelling_the_picker_changes_nothing(window):
    window.instances.select_instance(LOCKER)
    before = window.db.actions(DESKTOP)
    window.answer_pick = None
    window.act_correct_verb()
    assert window.db.actions(DESKTOP) == before and cls_of(window, LOCKER) == "drive_latch"


def test_a_latch_only_opened_cannot_be_the_cage(window):
    window.instances.select_instance(LOCKER)
    window.answer_pick = RECLASS
    window.act_correct_verb()
    assert not window.asked["confirm"]                 # refused before the confirmation
    assert cls_of(window, LOCKER) == "drive_latch"
    assert window.status_message().startswith(
        "改不了：第 7 步 drive_latch.01 是「打开」：硬盘架不会只是打开。")


# --------------------------------------------------------------------------- #
# the cage's button: no verb, no picker -- the confirmation, the op, Ctrl+Z
# --------------------------------------------------------------------------- #
def test_the_cage_becomes_a_latch_and_ctrl_z_takes_it_back(window):
    select_removed(window, CAGE)
    window.instances.verb_button.click()
    QApplication.processEvents()
    assert not window.asked["pick"]
    [(instance, lines)] = window.asked["confirm"]
    assert instance == CAGE
    assert lines[0].startswith("把 drive_cage.01（SSD shield） 改成卡扣 drive_latch.02：")
    assert lines[-1] == "立即写入，不经 Apply（标注模式里 Ctrl+Z 撤销）。"
    assert cls_of(window, CAGE) is None and cls_of(window, LATCH) == "drive_latch"
    assert [(a.step, a.verb) for a in window.db.actions(DESKTOP) if a.target == LATCH] == \
        [(11, "remove")]
    assert window.status_message().startswith("drive_cage.01 其实是卡扣：现在是 drive_latch.02")
    assert window.status_message().endswith("（Ctrl+Z 撤销）")
    assert len(window.db.ops_of_kind(DESKTOP, OP_RECLASS)) == 1
    window.act_undo()
    assert cls_of(window, CAGE) == "drive_cage" and cls_of(window, LATCH) is None
    window.act_redo()
    assert cls_of(window, CAGE) is None and cls_of(window, LATCH) == "drive_latch"


def test_a_declined_confirmation_writes_nothing(window):
    select_removed(window, CAGE)
    window.answer_confirm = False
    window.act_correct_verb()
    assert cls_of(window, CAGE) == "drive_cage"
    assert not window.db.ops_of_kind(DESKTOP, OP_RECLASS)


def test_refused_while_steps_holds_edits(window):
    select_removed(window, CAGE)
    window._steps_dirty = True
    window.act_correct_verb()
    assert not window.asked["confirm"] and cls_of(window, CAGE) == "drive_cage"
    assert window.status_message() == f"改不了：{V.RECLASS_STAGED}"


# --------------------------------------------------------------------------- #
# Steps mode: the right-click
# --------------------------------------------------------------------------- #
@pytest.fixture
def panel(qapp, tmp_path):
    from reclass_scenes import build
    from tda.ui.panels.steptable import StepTablePanel

    db = Db(str(tmp_path / "steps.sqlite"))
    scene = build(db, locker_verb="open")
    widget = StepTablePanel(db, 1, taxonomy=scene.tax, cache_dir=tmp_path / "cache")
    widget.asked = []

    def ask(key, lines):
        widget.asked.append((key, list(lines)))
        return True

    widget.ask_reclass = ask
    saved: list[int] = []
    widget.sigSaved.connect(saved.append)
    widget.saved = saved
    yield widget
    widget.deleteLater()
    db.close()


def _entries(panel, key) -> dict:
    menu = panel.instances_menu(panel.instances_model.keys.index(key))
    return {a.text(): a for a in menu.actions()}


def test_the_right_click_offers_it_on_the_two_classes_only(panel):
    assert "其实是硬盘架… / Reclass as drive_cage" in _entries(panel, "drive_latch.02")
    assert not [t for t in _entries(panel, "storage_drive.ssd.01") if t.startswith("其实是")]


def test_the_right_click_reclasses_written_through(panel):
    _entries(panel, "drive_latch.02")["其实是硬盘架… / Reclass as drive_cage"].trigger()
    [(key, lines)] = panel.asked
    assert key == "drive_latch.02" and lines[-1] == "立即写入，不经 Apply（Steps 里没有撤销）。"
    assert "drive_cage.01" in panel.data.instances and "drive_latch.02" not in panel.data.instances
    assert panel.status.text().startswith("drive_latch.02 其实是硬盘架：现在是 drive_cage.01")
    assert "drive_latch.01 已挂到 drive_cage.01 上，随机架一起拆下（第 5 步）" in panel.status.text()
    assert panel.saved == [1]
    assert not panel.has_unapplied_edits()
    # and the cage offers the way back
    assert "其实是卡扣… / Reclass as drive_latch" in _entries(panel, "drive_cage.01")


def test_the_right_click_is_refused_with_edits_staged(panel):
    panel.data.apply_edit(2, "notes", "a note")
    _entries(panel, "drive_latch.02")["其实是硬盘架… / Reclass as drive_cage"].trigger()
    assert not panel.asked
    assert panel.status.text().startswith("Rejected: 先 Apply 或 Revert：Steps 里还有没写入的修改")
    assert "drive_latch.02" in panel.db.instances(1)
