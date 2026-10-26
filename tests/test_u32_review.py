"""Task U32 in Review: 「重签这一帧」 / 「重签全部」 (two buttons, no key).

The panel only reports which one was clicked; the window plans through the
session (nothing written), says so when there is nothing to re-sign, otherwise
asks -- the question names the frame count -- and on yes re-signs through the
same service as ``cli refreeze`` and says what it did on the status line.

The scene is the shared D13 window scene: frames 13 and 12 confirmed, then the
chassis re-traced one column wider -- within the tolerance, so no conflict,
and both frames' chassis rows keep the old pixels.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QPushButton, QToolButton

import app_scene
from tda.core import masks
from tda.core.model import FrameKey, ShapePart
from tda.core.truth_conflicts import row_values
from tda.core.truth_refreeze import OP_KIND
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.panels.review import REFREEZE_ALL, REFREEZE_FRAME, ReviewPanel

DESKTOP, VIEW = app_scene.DESKTOP, app_scene.VIEW
CHASSIS = "chassis"


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


def _stale_rows(win) -> None:
    """Frames 13 and 12 confirmed, then the chassis one column wider: within tolerance."""
    session = win.session
    session.sweeper_enabled = False
    session.goto(12)
    app_scene.seed_shapes(session, 12)
    session.goto(13)
    assert session.confirm_frame() and session.confirm_frame()
    [kf] = [k for k in session.db.keyframes(DESKTOP, VIEW, CHASSIS)]
    shape = masks.decode_rle(kf.parts[0].rle)
    wider = shape | np.roll(shape, 1, axis=1)
    kf.parts = [ShapePart("main", masks.encode_rle(wider))]
    session.db.update_keyframe(kf)
    session._invalidate()
    stats = session.truth.refresh_range(DESKTOP, VIEW, [12, 13])
    assert stats["conflicts"] == 0                     # the same annotation, by the tolerance
    for step in (12, 13):
        assert session.db.compiled(key(step))[CHASSIS]["input_hash"] \
            != session.truth.compile(key(step)).input_hash


def _answer(win, yes: bool) -> list[list[str]]:
    asked: list[list[str]] = []

    def ask(lines):
        asked.append(list(lines))
        return yes

    win.ask_refreeze_confirm = ask
    return asked


def test_the_panel_has_two_resign_buttons_that_only_report(qapp):
    panel = ReviewPanel()
    seen: list[str] = []
    panel.sigRefreeze.connect(seen.append)
    assert panel.refreeze_frame_button.text() == "重签这一帧"
    assert panel.refreeze_all_button.text() == "重签全部"
    panel.refreeze_frame_button.click()
    panel.refreeze_all_button.click()
    assert seen == [REFREEZE_FRAME, REFREEZE_ALL]
    assert isinstance(panel.refreeze_all_button, QToolButton)
    assert panel.findChildren(QPushButton) == []        # still no resolution buttons (U2b)
    panel.deleteLater()


def test_resign_has_no_key():
    assert not [a for a in A.ACTIONS if "refreeze" in a.name or "refreeze" in a.slot
                or "重签" in a.label_zh]


def test_resign_all_plans_asks_with_the_frame_count_and_resigns(window):
    _stale_rows(window)
    window.set_mode(A.MODE_REVIEW)
    db = window.session.db
    asked = _answer(window, yes=True)

    window.review.refreeze_all_button.click()
    QApplication.processEvents()

    assert len(asked) == 1
    assert asked[0][0].startswith("重签 2 帧（2 行，"), asked[0]
    assert asked[0][1] == "帧：12–13"
    assert window.status_message().startswith("已重签 2 帧、2 行"), window.status_message()
    for step in (12, 13):
        compiled = window.session.truth.compile(key(step))
        row = db.compiled(key(step))[CHASSIS]
        assert row["visible_rle"]["counts"] == \
            row_values(compiled.instances[CHASSIS]).visible_rle["counts"]
        assert row["status"] == "verified" and row["verified_by"] == "tester"
        assert db.get_frame(key(step))["review_status"] == "verified"
    assert [op["annotator"] for op in db.ops_of_kind(DESKTOP, OP_KIND)] == ["tester"] * 2

    # a second click finds nothing to do and asks nothing
    window.review.refreeze_all_button.click()
    assert len(asked) == 1
    assert "没有要重签的行" in window.status_message()


def test_answering_no_writes_nothing(window):
    _stale_rows(window)
    window.set_mode(A.MODE_REVIEW)
    db = window.session.db
    raw = {s: db.compiled_raw(key(s)) for s in (12, 13)}
    asked = _answer(window, yes=False)

    window.review.refreeze_all_button.click()

    assert len(asked) == 1
    assert "重签已取消" in window.status_message()
    assert {s: db.compiled_raw(key(s)) for s in (12, 13)} == raw
    assert db.ops_of_kind(DESKTOP, OP_KIND) == []


def test_resign_this_frame_is_the_frame_on_screen_only(window):
    _stale_rows(window)
    window.set_mode(A.MODE_REVIEW)
    session = window.session
    asked = _answer(window, yes=True)
    session.goto(10)                                   # not confirmed
    window.review.refreeze_frame_button.click()
    assert asked == []
    assert "这一帧还没确认" in window.status_message()

    session.goto(12)
    window.review.refreeze_frame_button.click()
    assert asked[-1][0].startswith("重签 1 帧（1 行，"), asked[-1]
    ops = session.db.ops_of_kind(DESKTOP, OP_KIND)
    assert [op["payload"]["step"] for op in ops] == [12]
    # frame 13 still trails the compile
    assert session.db.compiled(key(13))[CHASSIS]["input_hash"] \
        != session.truth.compile(key(13)).input_hash
