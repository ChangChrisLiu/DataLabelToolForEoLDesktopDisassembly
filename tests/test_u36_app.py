"""Task U36 in the window: the 可见性审查 tab over runs and across the views.

The scene is :mod:`vis_runs_scene` (the scanner and oak1 on the same confirmed
steps). Pinned: the summary at the top of the queue and the caption saying the
order; under the list, for the selected line, the frame strip, the other
views (the one that sees the part highlighted), the suggestion -- also the
guide's sentence and the palette's border, while the key stays the
annotator's; 「从这帧起另算」 cutting the line at the frame on screen; and
「同步到其他视角」 offered after an answer, writing, and undone by one Ctrl+Z.
No new key.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import vis_review_scene as V
import vis_runs_scene as S
from app_scene import StubSamQueue, close_window, make_paths
from test_app_actions import key_event
from tda.core.model import OCCLUDER_VIEWPOINT
from tda.ui import app_actions as A
from tda.ui import session_api as api
from tda.ui import visreview_text as T
from tda.ui.app import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    session = S.build(tmp_path)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(1400, 900)
    win.show()
    QApplication.processEvents()
    if win.roi_editing:
        win.act_clear_edit()
    QApplication.processEvents()
    yield win
    close_window(win)
    session.close(force=True)


def review(win: MainWindow) -> None:
    win.session.goto(9, force=True)
    win.set_mode(A.MODE_REVIEW)
    win.review.show_visibility()
    QApplication.processEvents()


def vis_list(win: MainWindow):
    return win.review.list_for(api.QUEUE_VISIBILITY)


def lines(win: MainWindow) -> list[str]:
    lw = vis_list(win)
    return [lw.item(i).text() for i in range(lw.count())]


def pick(win: MainWindow, instance: str, step: int = None) -> None:
    lw = vis_list(win)
    row = next(i for i in range(lw.count())
               if lw.item(i).text().startswith(instance + " ")
               and (step is None or f"第 {step}" in lw.item(i).text()))
    lw.setCurrentRow(row)
    QApplication.processEvents()
    win.review.load_strip_now()


def press(win: MainWindow, spec: str) -> bool:
    done = win.handle_key(key_event(spec))
    QApplication.processEvents()
    return done


def here(win: MainWindow) -> int:
    return int(win.session.current().step)


def test_the_summary_sits_on_top_of_the_queue_in_its_order(window):
    review(window)
    summary = window.review.run_summary
    assert summary.isVisible()
    rows = summary.rows()
    assert [r[0] for r in rows] == [line.split(" ")[0] for line in lines(window)]
    marked = next(r for r in rows if r[0] == S.MARKED)
    assert marked[1:] == ("5/1", "4/1", "—", "未确认")
    row = [r[0] for r in rows].index(S.MARKED)
    assert summary.bold(row, "scan") and not summary.bold(row, "oak1")
    assert "零件按最长的一段排，最长的在前" in summary.caption()
    # round 2 (N2): folded to its title, which still says the order
    assert summary.expanded() and summary.table.isVisible()
    summary.toggle.click()
    QApplication.processEvents()
    assert not summary.expanded() and not summary.table.isVisible()
    assert summary.toggle.isVisible() and "最长的在前" in summary.caption()
    summary.toggle.click()
    QApplication.processEvents()
    assert summary.table.isVisible()
    # a click on a summary row selects that part's line and opens its frame
    summary._clicked([r[0] for r in rows].index(S.ANGLE), 0)
    QApplication.processEvents()
    assert window.review.selected_visibility()["instance"] == S.ANGLE
    assert window.instances.selected_instance() == S.ANGLE


def test_the_selected_line_shows_its_frames_and_the_other_views(window):
    review(window)
    pick(window, S.ANGLE)
    detail = window.review.run_detail
    assert detail.rows() == [("scan（这里）", "0", "5", "手动 5"), ("oak1", "5", "0", "·"),
                             ("oak2", "—", "—", "未标注"), ("rs", "—", "—", "只有草稿，没确认")]
    assert detail.highlighted() == ["oak1"]
    assert detail.strip_steps() == [10, 12, 14]
    assert all(not p.isNull() for p in detail.strip_pixmaps())
    assert detail.suggestion() == ("建议：角度看不到（完全，Z）——oak1 同样的帧看得见，"
                                   "这里没有挡住它的零件")
    # the guide says it, and the palette's border is on 角度看不到 -- the key is theirs
    window.refresh_guidance()
    plan = window.guide.plan()
    assert plan.action == "vis_review_viewpoint"
    assert plan.now.startswith("建议：角度看不到（完全，Z）") and "还是你按键决定" in plan.now
    assert window.palette.button("vis_review_viewpoint").suggested
    assert window.db.frame_overrides(S.key(10)).get(S.ANGLE).occluder is None   # nothing set
    # a line hidden by a part: Space is the default
    pick(window, V.HIDDEN)
    window.refresh_guidance()
    assert window.guide.plan().action == "vis_review_computed"
    assert window.review.run_detail.highlighted() == []


def test_split_here_cuts_the_line_and_the_keys_answer_the_rest(window):
    review(window)
    pick(window, S.ANGLE)
    assert here(window) == 10
    QTest.mouseClick(window.review.run_detail.split_button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert window.status_message().startswith("可见性审查：第 10 帧已经是这一段的第一帧")
    window.session.goto(12, force=True)
    QApplication.processEvents()
    QTest.mouseClick(window.review.run_detail.split_button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert [line for line in lines(window) if line.startswith(S.ANGLE)] == [
        "connector.05 第 10–11 帧（2 帧） · occluded_full（手动）",
        "connector.05 第 12–14 帧（3 帧） · occluded_full（手动）"]
    assert window.review.selected_visibility()["steps"] == [12, 13, 14]
    assert window.status_message().startswith("从第 12 帧起另算：")
    assert press(window, "Z")
    assert window.db.frame_overrides(S.key(12))[S.ANGLE].occluder == OCCLUDER_VIEWPOINT
    assert window.db.frame_overrides(S.key(11))[S.ANGLE].occluder is None
    assert [line for line in lines(window) if line.startswith(S.ANGLE)] == [
        "connector.05 第 10–11 帧（2 帧） · occluded_full（手动）"]


def test_sync_is_offered_after_an_answer_and_taken_back_by_one_ctrl_z(window):
    """Space on the latch a screw hides here and in oak1: the computed stamp
    goes to oak1's five frames; one Ctrl+Z takes all five back."""
    review(window)
    sync = window.review.run_detail.sync_button
    assert not sync.isEnabled()
    pick(window, V.HIDDEN)
    assert press(window, "Space")
    assert "其他视角还有 5 帧同样看不见、还没审" in window.status_message()
    assert sync.isEnabled() and "oak1 5 帧" in sync.toolTip()
    QTest.mouseClick(sync, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    said = window.status_message()
    assert said.startswith("已同步到其他视角：") and "oak1 第 10–14 帧，共 5 帧" in said
    assert "手动标的原因换成了" not in said
    assert not sync.isEnabled()
    assert all(window.db.instance_flags(S.key(s, S.OTHER), V.HIDDEN)["verdict"] == "computed"
               and window.db.frame_overrides(S.key(s, S.OTHER)).get(V.HIDDEN) is None
               for s in range(10, 15))
    assert press(window, "Ctrl+Z")                     # the sync, all five frames
    assert all(window.db.instance_flags(S.key(s, S.OTHER), V.HIDDEN) is None
               for s in range(10, 15))
    assert window.db.instance_flags(S.key(10), V.HIDDEN)["verdict"] == "computed"


def test_the_offer_goes_out_when_its_verdict_is_undone(window):
    """Round 2 (N8): Ctrl+Z of the verdict leaves nothing to copy -- the button goes dark."""
    review(window)
    sync = window.review.run_detail.sync_button
    pick(window, V.HIDDEN)
    assert press(window, "Space")
    assert sync.isEnabled()
    assert press(window, "Ctrl+Z")
    assert not sync.isEnabled()
    assert press(window, "Ctrl+Y")                     # the verdict back: so is the offer
    assert sync.isEnabled()


def test_no_offer_when_the_other_views_have_nothing_to_take(window):
    review(window)
    pick(window, S.ANGLE)
    assert press(window, "Z")                          # oak1 sees it: nothing to copy
    assert "同步" not in window.status_message()
    assert not window.review.run_detail.sync_button.isEnabled()
    window.act_sync_views()
    assert window.status_message() == T.SYNC_IDLE
    # the cable marks of oak1 are a person's: Z on connector.04 offers nothing either
    pick(window, S.MARKED)
    assert press(window, "Z")
    assert not window.review.run_detail.sync_button.isEnabled()


def test_the_default_dock_keeps_five_lines_of_the_list(qapp, tmp_path):
    """Round 2 (N2): a fresh app folder, 1920 x 1000 -- the list keeps five
    lines under the summary (three parts high) and the detail's buttons show."""
    session = S.build(tmp_path)
    paths = make_paths(tmp_path)
    paths["app_dir"] = str(tmp_path / "fresh_app")
    win = MainWindow(session, paths, "tester", sam_queue=StubSamQueue())
    try:
        win.resize(1920, 1000)
        win.show()
        QApplication.processEvents()
        if win.roi_editing:
            win.act_clear_edit()
        review(win)
        lw = vis_list(win)
        lw.setCurrentRow(0)
        QApplication.processEvents()
        assert lw.viewport().height() >= 5 * lw.sizeHintForRow(0) - 1
        assert win.review.run_summary.table.height() <= (
            win.review.run_summary.table.horizontalHeader().height() + 3 * 16 + 8)
        assert win.review.run_detail.split_button.isVisible()
    finally:
        close_window(win)
        session.close(force=True)


def test_no_new_key():
    """The two buttons are the panel's alone: no binding calls their slots."""
    slots = {action.slot for action in A.ACTIONS}
    assert "act_split_run" not in slots and "act_sync_views" not in slots
