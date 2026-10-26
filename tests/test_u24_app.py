"""Task U24 in the window: no label keys while annotating; Review's 「可见性审查」.

The scene is :mod:`vis_review_scene` (frames 14-10 confirmed, five runs to
audit, 17 rows). Pinned: Annotate mode ignores ``Ctrl+1..7`` / ``Ctrl+V`` (the
grave key still says 被线挡住 there); Review binds the verdicts; the fifth
tab counts runs and frames, says how far the view is and 「可见性审查完成」 at
the end; selecting a line opens its frame and selects the part; each key
answers the **whole line** (a run) as one op, moves to the next line, says
what it did and is taken back by one ``Ctrl+Z`` in Review; the 按帧 listing
answers one frame; ``Space`` on the other tabs still accepts the frame; the
queue is not built while annotating; the palette's three buttons; and a part
角度看不到 reads 「角度」 in the instance table.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import vis_review_scene as V
from app_scene import StubSamQueue, close_window, make_paths
from test_app_actions import key_event
from tda.core.occluder import FULL, PARTIAL
from tda.ui import app_actions as A
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui import visreview_text as T
from tda.ui.app import MainWindow
from tda.ui.app_visreview import NOT_IN_REVIEW, OTHER_TAB

#: Task U36: the frame count, 「全程」, and the part with the longest run first.
LINES = [
    "connector.03 第 10–14 帧（5 帧） · too_small（自动） · 全程",
    "ram_latch.03 第 10–14 帧（5 帧） · occluded_full（被螺丝挡住，自动） · 全程",
    "ram_module.01 第 10–14 帧（5 帧） · occluded_partial（被螺丝挡住，自动） · 全程",
    "cpu.01 第 13 帧 · occluded_full（原因: 线缆，手动）",
    "psu.01 第 12 帧 · out_of_view（手动）",
]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    session = V.build(tmp_path)
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
    """Review, the 可见性审查 tab on top, standing on frame 9 (in no run)."""
    win.session.goto(9, force=True)
    win.set_mode(A.MODE_REVIEW)
    win.review.show_visibility()
    QApplication.processEvents()


def vis_list(win: MainWindow):
    return win.review.list_for(api.QUEUE_VISIBILITY)


def lines(win: MainWindow) -> list[str]:
    lw = vis_list(win)
    return [lw.item(i).text() for i in range(lw.count())]


def pick(win: MainWindow, instance: str) -> None:
    """Select the first line of ``instance`` as a click would."""
    lw = vis_list(win)
    row = next(i for i in range(lw.count()) if lw.item(i).text().startswith(instance + " "))
    lw.setCurrentRow(row)
    QApplication.processEvents()


def press(win: MainWindow, spec: str) -> bool:
    done = win.handle_key(key_event(spec))
    QApplication.processEvents()
    return done


def tilde() -> QKeyEvent:
    """``Shift+``` as Windows hands it to Qt on a US layout."""
    return QKeyEvent(QEvent.Type.KeyPress, int(Qt.Key.Key_AsciiTilde),
                     Qt.KeyboardModifier.ShiftModifier, "~")


def here(win: MainWindow) -> int:
    return int(win.session.current().step)


# --------------------------------------------------------------------------- #
# the keys
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", [*(f"Ctrl+{n}" for n in range(1, 8)), "Ctrl+V",
                                  *(str(n) for n in range(3, 8)), "Z", "Shift+Z",
                                  "Shift+Space"])
def test_annotate_mode_ignores_the_label_keys(window, spec):
    window.set_mode(A.MODE_ANNOTATE)
    key = window.session.current()
    before = dict(window.db.frame_overrides(key))
    event = key_event(spec)
    assert A.action_for(event.key(), event.modifiers(), A.MODE_ANNOTATE) is None
    assert press(window, spec) is False
    assert window.db.frame_overrides(key) == before


def test_annotate_mode_keeps_the_grave_key_for_the_cable(window):
    calls: list[str] = []
    window.act_mark_absent = lambda: calls.append("absent")
    window.set_mode(A.MODE_ANNOTATE)
    assert window.handle_key(QKeyEvent(QEvent.Type.KeyPress, int(Qt.Key.Key_QuoteLeft),
                                       Qt.KeyboardModifier.NoModifier, "`")) is True
    assert calls == ["absent"]
    # the label slots, called anyway (a stray dispatch), say where labels are set
    window.act_set_visibility("out_of_view")
    assert window.status_message() == NOT_IN_REVIEW


@pytest.mark.parametrize("spec,name", [
    ("Space", "vis_review_computed"), ("Shift+Space", "vis_review_computed"),
    ("3", "visibility_3"), ("Ctrl+3", "visibility_3"), ("Ctrl+V", "cycle_visibility"),
    ("`", "vis_review_cable"), ("Z", "vis_review_viewpoint"),
    ("Shift+Z", "vis_review_viewpoint_partial"), ("Ctrl+Z", "undo"), ("Ctrl+Y", "redo"),
    ("Return", "review_accept"), ("Enter", "review_accept"),
])
def test_review_mode_binds_the_verdicts(spec, name):
    event = key_event(spec)
    assert A.action_for(event.key(), event.modifiers(), A.MODE_REVIEW).name == name


def test_shift_grave_arrives_as_a_tilde_and_is_the_partial_cable(window):
    event = tilde()
    assert A.action_for(event.key(), event.modifiers(), A.MODE_REVIEW).name == \
        "vis_review_cable_partial"
    assert A.action_for(event.key(), event.modifiers(), A.MODE_ANNOTATE) is None


# --------------------------------------------------------------------------- #
# the tab
# --------------------------------------------------------------------------- #
def test_the_fifth_tab_counts_rows_and_says_how_far_the_view_is(window):
    review(window)
    tabs = window.review.tabs()
    assert tabs.tabText(tabs.currentIndex()) == "可见性审查 (17)"
    assert lines(window) == LINES
    assert window.review.visibility_status() == "还要审 5 段 / 17 帧（已审 0 / 17 帧）"
    assert window.guide.plan().now == G.VIS_REVIEW
    assert not window.review.per_frame()                        # 按段 by default


def test_selecting_a_line_opens_its_frame_and_selects_the_part(window):
    review(window)
    window.session.goto(14, force=True)
    QApplication.processEvents()
    pick(window, V.OUT)
    assert here(window) == V.OUT_STEP
    assert window.instances.selected_instance() == V.OUT
    # a run the frame on screen is in: no jump
    window.session.goto(13, force=True)
    QApplication.processEvents()
    pick(window, V.HIDDEN)
    assert here(window) == 13 and window.instances.selected_instance() == V.HIDDEN
    # the selection survives the frame change it caused
    assert window.review.selected_visibility()["instance"] == V.HIDDEN


def ops(win: MainWindow) -> int:
    return [op["kind"] for op in win.db.ops(V.DESKTOP, V.VIEW, limit=500)].count(
        "review_visibility")


def test_space_confirms_the_whole_line_and_moves_to_the_next(window):
    review(window)
    pick(window, V.HIDDEN)
    assert here(window) == 10
    assert press(window, "Space")
    assert f"内存卡扣 {V.HIDDEN} 第 10–14 帧 → 确认（按计算） 完全遮挡" in window.status_message()
    assert all(window.db.instance_flags(V.key(s), V.HIDDEN)["verdict"] == "computed"
               for s in range(10, 15))
    assert ops(window) == 1
    assert not any(line.startswith(V.HIDDEN) for line in lines(window))
    # the next line -- the one now where the answered one was -- and its part,
    # on the frame on screen (10 is in its run: no jump)
    assert window.review.selected_visibility()["instance"] == V.HALF
    assert here(window) == 10 and window.instances.selected_instance() == V.HALF
    assert window.review.visibility_status() == "还要审 4 段 / 12 帧（已审 5 / 17 帧）"
    assert press(window, "Ctrl+Z")
    assert lines(window) == LINES


def test_shift_space_is_spaces_alias(window):
    review(window)
    pick(window, V.TINY)
    assert press(window, "Shift+Space")
    assert "第 10–14 帧 → 确认（按计算）" in window.status_message()
    assert not any(line.startswith(V.TINY) for line in lines(window))
    # the next line (U36's order: the next longest run)
    assert window.review.selected_visibility()["instance"] == V.HIDDEN
    assert window.review.visibility_status() == "还要审 4 段 / 12 帧（已审 5 / 17 帧）"
    assert press(window, "Ctrl+Z")
    assert lines(window) == LINES


def test_z_marks_the_whole_run_angle_and_one_ctrl_z_takes_it_back(window):
    review(window)
    pick(window, V.HALF)
    assert press(window, "Z")
    for step in range(10, 15):
        fo = window.db.frame_overrides(V.key(step))[V.HALF]
        assert (fo.visibility, fo.occluder) == (FULL, "viewpoint")
        assert window.session.frame_status(step) == "verified"
    assert ops(window) == 1
    assert "第 10–14 帧 → 角度看不到（完全）" in window.status_message()
    tabs = window.review.tabs()
    assert tabs.tabText(tabs.currentIndex()) == "可见性审查 (12)"
    assert window.review.selected_visibility() is None or \
        window.review.selected_visibility()["instance"] != V.HALF
    assert press(window, "Ctrl+Z")                               # one undo: all five
    assert all(window.db.frame_overrides(V.key(s)).get(V.HALF) is None
               and window.db.instance_flags(V.key(s), V.HALF) is None for s in range(10, 15))
    assert tabs.tabText(tabs.currentIndex()) == "可见性审查 (17)"
    assert press(window, "Ctrl+Y")
    assert all(window.db.frame_overrides(V.key(s))[V.HALF].occluder == "viewpoint"
               for s in range(10, 15))


@pytest.mark.parametrize("key,label,occluder", [
    ("Shift+Z", PARTIAL, "viewpoint"), ("`", FULL, "cable"), ("tilde", PARTIAL, "cable"),
])
def test_the_halves_answer_the_whole_run_too(window, key, label, occluder):
    review(window)
    pick(window, V.HALF)
    if key == "tilde":
        assert window.handle_key(tilde()) is True                  # Shift+`
        QApplication.processEvents()
    else:
        assert press(window, key)
    assert all((window.db.frame_overrides(V.key(s))[V.HALF].visibility,
                window.db.frame_overrides(V.key(s))[V.HALF].occluder) == (label, occluder)
               for s in range(10, 15))
    assert ops(window) == 1


def test_the_per_frame_listing_answers_one_frame(window):
    review(window)
    window.review.set_per_frame(True)
    QApplication.processEvents()
    assert window.review.per_frame()
    assert len(lines(window)) == 17
    assert "ram_module.01 第 12 帧 · occluded_partial（被螺丝挡住，自动）" in lines(window)
    # the counts are the same either way
    assert window.review.visibility_status() == "还要审 5 段 / 17 帧（已审 0 / 17 帧）"
    lw = vis_list(window)
    lw.setCurrentRow(lines(window).index("ram_module.01 第 12 帧 · occluded_partial"
                                         "（被螺丝挡住，自动）"))
    QApplication.processEvents()
    assert here(window) == 12
    assert press(window, "Z")
    assert window.db.frame_overrides(V.key(12))[V.HALF].occluder == "viewpoint"
    assert all(window.db.frame_overrides(V.key(s)).get(V.HALF) is None
               for s in (10, 11, 13, 14))
    assert ops(window) == 1
    # the next line is ram_module.01's frame 13; back to 按段: the run is split
    assert window.review.selected_visibility()["steps"] == [13]
    window.review.set_per_frame(False)
    QApplication.processEvents()
    assert [line for line in lines(window) if line.startswith(V.HALF)] == [
        "ram_module.01 第 10–11 帧（2 帧） · occluded_partial（被螺丝挡住，自动）",
        "ram_module.01 第 13–14 帧（2 帧） · occluded_partial（被螺丝挡住，自动）"]
    assert window.review.selected_visibility()["steps"] == [13, 14]   # kept, by frame


def test_digits_and_ctrl_v_set_a_label_by_hand(window):
    review(window)
    pick(window, V.OUT)
    assert press(window, "1")
    assert window.db.frame_overrides(V.key(V.OUT_STEP))[V.OUT].visibility == "visible"
    assert window.db.compiled(V.key(V.OUT_STEP))[V.OUT]["visibility"] == "visible"
    assert "手动设为 可见" in window.status_message()
    assert window.session.frame_status(V.OUT_STEP) == "verified"
    pick(window, V.CABLED)
    assert press(window, "Ctrl+V")                 # occluded_full -> the next value
    fo = window.db.frame_overrides(V.key(V.CABLE_STEP))[V.CABLED]
    assert (fo.visibility, fo.occluder) == (api.VISIBILITY_VALUES[
        api.VISIBILITY_VALUES.index(FULL) + 1], None)


def test_space_on_the_other_tabs_still_accepts_the_frame(window, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(window, "act_confirm", lambda: calls.append("accept") or True)
    window.set_mode(A.MODE_REVIEW)
    window.review.tabs().setCurrentIndex(list(api.QUEUE_NAMES).index(api.QUEUE_CONFLICTS))
    assert press(window, "Space")
    assert calls == ["accept"]


def test_a_verdict_key_on_another_tab_or_with_nothing_picked_says_where(window):
    window.set_mode(A.MODE_REVIEW)
    window.review.tabs().setCurrentIndex(list(api.QUEUE_NAMES).index(api.QUEUE_CONFLICTS))
    assert press(window, "Z")
    assert window.status_message() == OTHER_TAB
    review(window)
    vis_list(window).setCurrentRow(-1)
    assert press(window, "Z")
    assert window.status_message() == T.PICK_A_ROW


def test_when_nothing_is_left_the_tab_says_so(window):
    session = window.session
    for entry in session.visibility_queue():
        session.goto(entry["steps"][0], force=True)
        session.review_visibility(entry["instance"], "computed", steps=entry["steps"])
    review(window)
    assert lines(window) == []
    assert window.review.visibility_status() == "可见性审查完成（17 帧都审过了）"
    assert window.guide.plan().now == G.VIS_REVIEW_DONE


def test_the_last_answer_says_the_review_is_done(window):
    session = window.session
    for entry in session.visibility_queue()[1:]:
        session.goto(entry["steps"][0], force=True)
        session.review_visibility(entry["instance"], "computed", steps=entry["steps"])
    review(window)
    pick(window, V.TINY)
    assert press(window, "Shift+Space")
    assert window.status_message().endswith(f"；{T.DONE}")


def test_the_queue_is_not_built_while_annotating(window, monkeypatch):
    built: list[int] = []
    real = window.session.review._build_visibility
    monkeypatch.setattr(window.session.review, "_build_visibility",
                        lambda: built.append(1) or real())
    window.set_mode(A.MODE_ANNOTATE)
    for step in (12, 11, 10):
        window.session.goto(step, force=True)
        QApplication.processEvents()
    assert built == []
    review(window)
    assert built and len(lines(window)) == 5


# --------------------------------------------------------------------------- #
# the palette, the instance table, the generated tables
# --------------------------------------------------------------------------- #
def test_the_palette_has_the_three_answers_in_review(window):
    review(window)
    vis_list(window).setCurrentRow(-1)
    window.refresh_guidance()
    palette = window.palette
    for name in ("vis_review_computed", "vis_review_cable", "vis_review_viewpoint"):
        button = palette.button(name)
        assert not button.isHidden() and not button.isEnabled(), name
        assert "选一条" in button.reason()
    window.session.goto(9, force=True)
    QApplication.processEvents()
    pick(window, V.HALF)
    window.refresh_guidance()
    button = palette.button("vis_review_viewpoint")
    assert button.isEnabled() and button.key == "Z"
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert all(window.db.frame_overrides(V.key(s))[V.HALF].occluder == "viewpoint"
               for s in range(10, 15))
    window.refresh_guidance()
    assert palette.button("undo").isEnabled()
    window.set_mode(A.MODE_ANNOTATE)
    assert palette.button("vis_review_viewpoint").isHidden()


def test_a_part_out_of_the_cameras_sight_reads_angle_in_annotate(window):
    review(window)
    pick(window, V.HALF)
    press(window, "Z")                                          # frame 10
    window.set_mode(A.MODE_ANNOTATE)
    window.session.goto(10, force=True)
    QApplication.processEvents()
    table = window.instances.table()
    line = next(i for i in range(table.rowCount())
                if table.item(i, 1) is not None and table.item(i, 1).text() == V.HALF)
    cell = table.item(line, window.instances.COLUMNS.index("Vis"))
    assert cell.text() == "角度" and "角度看不到" in cell.toolTip()
    row = next(r for r in window.instances.rows() if r["key"] == V.HALF)
    assert row["occluder"] == "viewpoint"


def test_the_task_card_says_angle_not_cable():
    from tda.ui.panels.taskcard import row_view

    base = {"instance": "x.01", "kind": api.KIND_ADD_SHAPE, "done": True, "cls": "cover",
            "raw_names": [], "cable": FULL}
    assert "角度看不到（完全）" in row_view(dict(base, occluder="viewpoint"))["sentence"]
    assert "被线挡住（完全）" in row_view(dict(base, occluder="cable"))["sentence"]


def test_the_cheat_sheet_and_the_guide_table_list_the_verdicts():
    html = A.cheat_sheet_html()
    assert "<h4>可见性审查（Review 模式）</h4>" in html
    for keys in ("Space / Shift+Space", "1 / Ctrl+1", "Ctrl+V", "`", "Shift+`", "Z",
                 "Shift+Z"):
        assert f"<tr><td><b>{keys}</b></td><td>可见性审查" in html, keys
    table = A.shortcut_markdown()
    assert "| `1 / Ctrl+1` | 可见性审查：这一段手动设为 visible（记为审过） | " \
           "可见性审查（Review 模式） |" in table
    assert "| `Z` | 可见性审查：这一段角度看不到（完全）" in table
    assert "| `Space / Shift+Space` | 可见性审查：选中的这一段按计算的对" in table
    assert "| `Return / Enter` | 接受队列选中的这一帧 |" in table
