"""Task U11 (window): ``Ctrl+T``, the status lines, and the hints of a walk.

The 42-step window scene with D2's situation four steps earlier: ``ram_latch.04``
has no ``open`` action (closed 1-41 by the log) and the board takes it out at
42; it has a closed shape drawn before the walk (one version, 1-41).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app_scene import DESKTOP, StubSamQueue, close_window, make_paths, make_session
from tda.core import masks
from tda.core.model import ShapeKeyframe, ShapePart
from tda.ui import app_actions as A
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.app_state import HINT_LOG, HINT_RANGE, StateRangeDialog, cut_text, range_note

CLIP = "ram_latch.04"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    session = make_session(tmp_path, last_step=42)
    session.sweeper_enabled = False
    db = session.db
    for key, rec in db.instances(DESKTOP).items():
        if key.startswith("ram_latch."):
            rec.parent, rec.attached = "motherboard.01", True
            db.upsert_instance(rec)
    db.replace_steps(DESKTOP, db.steps(DESKTOP),
                     [a for a in db.actions(DESKTOP)
                      if not (a.target == CLIP and a.verb == "open")])
    closed = np.zeros((64, 64), dtype=bool)
    closed[40:50, 40:50] = True
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=CLIP, desktop=DESKTOP, view="scan", pose_segment=1,
        anchor_step=42, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(closed))]))
    session._invalidate()
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()
    yield win
    close_window(win)


def goto(win: MainWindow, step: int) -> None:
    win.session.goto(step, force=True)
    QApplication.processEvents()


def manual(win: MainWindow, target: str = CLIP) -> list[tuple]:
    return [(e.step, e.old, e.new) for e in win.db.events(DESKTOP)
            if not e.auto and e.target == target]


def press_t(win: MainWindow, step: int, instance: str = CLIP) -> str:
    goto(win, step)
    assert win.instances.select_instance(instance), step
    win.act_cycle_state()
    return win.status_message()


def draw_open(win: MainWindow, step: int) -> dict:
    goto(win, step)
    shape = np.zeros((64, 64), dtype=bool)
    shape[4:14, 4:30] = True
    win.session.begin_edit(CLIP)
    win.session.set_editing_mask(shape)
    result = win.session.commit_edit(api.SCOPE_KEYFRAME)
    win.session.clear_edit()
    return result


def test_ctrl_t_is_bound_in_annotate_mode_only():
    action = A.action_named("state_range")
    assert action.keys == ("Ctrl+T",) and action.slot == "act_state_range"
    assert action.modes == (A.MODE_ANNOTATE,)
    found = A.action_for(Qt.Key.Key_T, Qt.KeyboardModifier.ControlModifier, A.MODE_ANNOTATE)
    assert found is not None and found.name == "state_range"
    assert A.action_for(Qt.Key.Key_T, Qt.KeyboardModifier.ControlModifier,
                        A.MODE_STEPS) is None
    # nothing else had it
    assert [a.name for a in A.ACTIONS if "Ctrl+T" in a.keys] == ["state_range"]


def test_ctrl_t_asks_once_and_says_the_run(window, monkeypatch):
    win = window
    asked = {}

    def answer(instance, choices, state, to_step):
        asked.update(instance=instance, choices=choices, state=state, to_step=to_step)
        return "open", 34

    monkeypatch.setattr(win, "ask_state_range", answer)
    goto(win, 41)
    assert win.instances.select_instance(CLIP)
    win.act_state_range()
    assert asked == {"instance": CLIP, "choices": ["closed", "open"], "state": "open",
                     "to_step": 1}
    assert manual(win) == [(34, "closed", "open")]
    message = win.status_message()
    assert f"{CLIP} 第 34–41 步都是 打开；第 42 步起 拆下" in message
    assert "Ctrl+Z" in message
    win.act_undo()
    assert manual(win) == []


def test_ctrl_t_cancelled_writes_nothing(window, monkeypatch):
    win = window
    monkeypatch.setattr(win, "ask_state_range", lambda *a: None)
    goto(win, 41)
    win.instances.select_instance(CLIP)
    win.act_state_range()
    assert manual(win) == []


def test_ctrl_t_across_the_logs_action_says_go_to_steps(window, monkeypatch):
    win = window
    monkeypatch.setattr(win, "ask_state_range", lambda *a: ("closed", 10))
    goto(win, 21)
    win.instances.select_instance("ram_latch.01")
    win.act_state_range()
    message = win.status_message()
    assert message.startswith("改不了状态：") and "Steps 模式" in message
    assert "跨过了日志第 14 步" in message
    assert manual(win, "ram_latch.01") == []


def test_the_dialog_offers_the_states_and_the_step(qapp):
    dialog = StateRangeDialog(None, "内存卡扣 ram_latch.04", 41, ["closed", "open"], "open",
                              1, 1, 42)
    assert [dialog.state_box.itemText(i) for i in range(2)] == ["关上", "打开"]
    assert dialog.answer() == ("open", 1)
    dialog.step_box.setValue(34)
    dialog.state_box.setCurrentIndex(0)
    assert dialog.answer() == ("closed", 34)
    assert dialog.step_box.prefix() == "一直到第 " and dialog.step_box.suffix() == " 步"
    dialog.deleteLater()


def test_t_after_a_commit_says_the_version_boundary_moved_too(window):
    win = window
    press_t(win, 41)
    assert draw_open(win, 41)["cut_below"] == 40
    message = press_t(win, 40)
    assert "形状版本边界也随之移到第 39 步" in message
    assert "这几帧的形状要重画" not in message          # the run needs no redraw


def test_a_walk_of_t_gets_the_hints(window):
    win = window
    first = press_t(win, 41)
    assert HINT_RANGE not in first and HINT_LOG not in first
    second = press_t(win, 40)
    assert HINT_RANGE in second and HINT_LOG not in second
    third = press_t(win, 39)
    assert HINT_RANGE in third and HINT_LOG in third
    # another frame, not the next one of the walk: the count starts again
    again = press_t(win, 20)
    assert HINT_RANGE not in again and HINT_LOG not in again


def test_the_hint_texts_are_short():
    assert HINT_RANGE == "同一状态持续多帧？Ctrl+T 一次设到某一步"
    assert HINT_LOG == "日志本身漏了这一步的动作？去 Steps 模式加动作更稳"


def test_the_cut_texts():
    from tda.core.cut_follow import CutMove

    moved = [CutMove("oak1", 1, 40, 39), CutMove("scan", 2, 41, 38)]
    assert cut_text(moved, "scan") == "形状版本边界也随之移到第 38 步"
    assert cut_text([CutMove("scan", 3, 39, None)], "scan") == (
        "第 39 步切开的旧版本用不着了，已删掉")
    assert cut_text([CutMove("scan", 3, 39, 39, kept=True)], "scan") == (
        "第 39 步切开时留下的旧形状没删（别处没有这份形状），请看一眼")
    assert cut_text([], "scan") == ""


def test_the_range_note_reads_as_the_brief():
    class Patch:
        step, last, top, state = 38, 45, 45, "open"
        next_step, restores, previous = 46, "removed", "closed"

    assert range_note("cover.02", Patch(), run=(38, 45), after=(46, "removed")) == (
        "cover.02 第 38–45 步都是 打开；第 46 步起 拆下（Ctrl+Z 撤销）")


# --------------------------------------------------------------------------- #
# D: the documents quote what the window says
# --------------------------------------------------------------------------- #
def _squash(text: str) -> str:
    return "".join(text.split())


@pytest.fixture
def d13_window(qapp, tmp_path: Path):
    """The whole D13 sheet: ram_latch.01 opened at 14, out with the board at 42."""
    session = make_session(tmp_path, last_step=42)
    session.sweeper_enabled = False
    db = session.db
    for key, rec in db.instances(DESKTOP).items():
        if key.startswith("ram_latch."):
            rec.parent, rec.attached = "motherboard.01", True
            db.upsert_instance(rec)
    shape = np.zeros((64, 64), dtype=bool)
    shape[40:50, 40:50] = True                    # the open clip
    db.add_keyframe(ShapeKeyframe(
        id=None, instance="ram_latch.01", desktop=DESKTOP, view="scan", pose_segment=1,
        anchor_step=42, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(shape))]))
    session._invalidate()
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()
    yield win
    close_window(win)


def test_the_quickstart_example_is_what_the_window_says(d13_window, monkeypatch):
    win, latch = d13_window, "ram_latch.01"
    repo = Path(__file__).resolve().parents[1]
    quick = _squash((repo / "docs" / "trial_quickstart.md").read_text(encoding="utf-8"))
    said = press_t(win, 41, latch)
    assert _squash("在第 41 步：关上（之前为 打开）；第 42 步起是 拆下") in quick
    assert _squash("在第 41 步：关上（之前为 打开）；第 42 步起是 拆下") in _squash(said)
    asked = {}

    def answer(instance, choices, state, to_step):
        asked.update(state=state, to_step=to_step)
        return "closed", 39

    monkeypatch.setattr(win, "ask_state_range", answer)
    win.act_state_range()
    assert asked == {"state": "closed", "to_step": 15}       # hand-set; the log opens at 14
    line = win.status_message().removesuffix("（Ctrl+Z 撤销）")
    assert line.startswith("内存卡扣 ram_latch.01 第 39–41 步都是 关上；第 42 步起 拆下")
    assert _squash(line) in quick, line
    goto(win, 41)
    win.session.begin_edit(latch)
    closed = np.zeros((64, 64), dtype=bool)
    closed[40:44, 40:56] = True
    win.session.set_editing_mask(closed)
    win.act_commit()
    if win._pending_warning is not None:                    # "too big" for a clip
        win.act_commit()
    assert win.status_message() == ("已先在第 38 步切开旧版本（形状不变），"
                                    "这次修改只影响第 39–41 步")
    assert _squash(win.status_message()) in quick
    moved = press_t(win, 38, latch)
    assert "形状版本边界也随之移到第 37 步" in moved
    assert _squash("形状版本边界也随之移到第 37 步") in quick
    guide = _squash((repo / "docs" / "annotation_guide.md").read_text(encoding="utf-8"))
    assert _squash("`Ctrl+T` 选状态、填\"一直到第 N 步\"，一次设完") in guide
    assert _squash(A.action_named("state_range").label_zh) in guide        # the key table
    spec = _squash((repo / "docs" / "superpowers" / "specs"
                    / "2026-09-17-teardown-annotator-design.md").read_text(encoding="utf-8"))
    assert _squash("**v1.12（2026-10-14）**：**切开的版本跟着状态边界走") in spec
