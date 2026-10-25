"""Task U29 (B): ``Q`` (``Ctrl+K``) from the middle of a state run.

The ledger's item (U8/U11 era): committing a new shape version from the middle
of a run cuts the version at the highest open ✂ below, and the frames of the
run *above* the commit frame -- still on the old version -- were not flagged.

What happens now, pinned here on the D13 window scene (14 steps, every part
drawn anchored at 14). ``ram_latch.01`` is closed on 1-13 and opened by the log
at 14; ``Ctrl+T`` says it is open on 8-12 as well, so its single version V is
in force across three open ✂ rows: 13 (open at 14, closed at 13), 12 (closed
at 13, open at 12) and 7 (open at 8, closed at 7). Frames 14 down to 8 are
confirmed. ``Q`` on 10, the middle of the run 8-12:

* the version is cut at 7 (pixels unchanged), the new one covers 8-10;
* 11-12 -- the rest of the run, above the commit frame -- keep V: their stored
  rows and review status are byte-identical, nothing is re-checked or demoted
  (the user's ruling of U16: ``Q`` is "from this frame on", the frames behind
  in the walk stay as they are), and the ✂ at 12 stays open on the task card;
* the frames the new version reaches (8-10, all confirmed) are re-checked:
  10 at once (it is on screen), 8-9 through the queue, and conflict;
* the status line and the canvas say the run was split and name 11-12 (2 of
  them confirmed) with the key that reaches them -- ``Ctrl+Q``, which writes
  the shape up to the run's top (12) and not across its ✂.

Nothing in that was wrong; U16's 「仍是旧样子」 note closed the gap.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import app_scene
from session_scene import DESKTOP, VIEW, rect
from test_app_actions import key_event
from tda.core.model import FrameKey
from tda.ui import app_actions as A
from tda.ui import session_api as api
from tda.ui.app import MainWindow

LATCH = "ram_latch.01"
OPENED = rect(40, 2, 62, 12)
ABOVE = [11, 12, 13, 14]
SAID = ("已先在第 7 步切开旧版本（形状不变），这次修改只影响第 8–10 步。"
        "已从这帧起新版本；第 11、12 帧（其中已确认 2 帧）仍是旧样子，要一起改用 Ctrl+Q")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    made = app_scene.make_session(tmp_path)
    made.sweeper_enabled = False
    win = MainWindow(made, app_scene.make_paths(tmp_path), "tester",
                     sam_queue=app_scene.StubSamQueue())
    win.resize(1400, 900)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    if win.roi_editing:
        win.act_clear_edit()
    app_scene.seed_shapes(made, 1)
    made.goto(12, force=True)
    made.assert_state_range(LATCH, "open", 8)
    made.goto(14, force=True)
    for step in range(14, 7, -1):
        assert made.current().step == step
        assert made.confirm_frame() is True, step
    QApplication.processEvents()
    yield win
    app_scene.close_window(win)


def goto(win: MainWindow, step: int) -> None:
    win.session.goto(step, force=True)
    QApplication.processEvents()


def paint(win: MainWindow, mask: np.ndarray) -> None:
    win.on_request_edit(LATCH)
    win.set_editing_mask(mask, undoable=True)
    QApplication.processEvents()


def card_row(session, step: int, instance: str = LATCH):
    session.goto(step, force=True)
    found = [r for r in session.task_card() if r["instance"] == instance]
    return found[0] if found else None


def stored(session, steps) -> dict:
    """``step -> (review status, compiled rows)`` exactly as stored -- nothing refreshed."""
    out = {}
    for step in steps:
        key = FrameKey(DESKTOP, step, VIEW)
        out[step] = ((session.db.get_frame(key) or {}).get("review_status"),
                     session.db.compiled(key))
    return out


def anchors(session) -> list[int]:
    return sorted(kf.anchor_step for kf in session.db.keyframes(DESKTOP, VIEW, LATCH))


def shape_at(session, step: int) -> np.ndarray:
    return session.truth.compile(FrameKey(DESKTOP, step, VIEW)).instances[LATCH].amodal


def test_the_scene_has_three_open_boundaries_across_one_version(window):
    session = window.session
    for step, transition in ((13, ["open", "closed"]), (12, ["closed", "open"]),
                             (7, ["open", "closed"])):
        row = card_row(session, step)
        assert row["kind"] == api.KIND_SPLIT_KEYFRAME and not row["done"], step
        assert row["transition"] == transition, step
    assert anchors(session) == [14]


@pytest.mark.parametrize("key", ["slot", "Ctrl+K"])
def test_q_in_the_middle_of_a_run_keeps_the_frames_above_and_says_so(window, key):
    session = window.session
    old = shape_at(session, 11)
    before = stored(session, ABOVE)
    goto(window, 10)
    paint(window, OPENED)
    if key == "slot":
        window.act_commit()
    else:
        assert window.handle_key(key_event(key))
    if window._pending_warning is not None:          # a latch this size is "too big"
        window.act_commit()
    QApplication.processEvents()

    assert session.editing_instance is None, window.status_message()
    assert window.status_message() == SAID
    assert window.canvas.banner_text() == SAID.split("。", 1)[1]
    assert anchors(session) == [7, 10, 14]
    # the frames above, in the run and beyond it: untouched, still confirmed
    assert stored(session, ABOVE) == before
    assert all(session.frame_status(s) == api.STATUS_VERIFIED for s in ABOVE)
    assert not set(session.db.rechecks(DESKTOP, VIEW)) & set(ABOVE)
    for step in (11, 12):
        assert np.array_equal(shape_at(session, step), old), step
    assert card_row(session, 12)["done"] is False             # its ✂ is still open
    # the frames the new version reaches are re-checked: 10 now, 8-9 queued
    assert sorted(session.db.rechecks(DESKTOP, VIEW)) == [8, 9]
    assert [int(c["step"]) for c in session.db.conflicts(DESKTOP, VIEW, open_only=True)] == [10]
    for step in (8, 9, 10):
        assert np.array_equal(shape_at(session, step), OPENED), step
    # ... and the cut below kept the old shape, pixel for pixel
    assert np.array_equal(shape_at(session, 7), old)


def test_ctrl_q_after_it_reaches_the_rest_of_the_run_and_stops_at_its_top(window):
    session = window.session
    old = shape_at(session, 13)
    goto(window, 10)
    paint(window, OPENED)
    window.act_commit()
    if window._pending_warning is not None:
        window.act_commit()
    # what the note points at: Ctrl+Q right away writes it into V, up to 12
    # (asked first: 11 and 12 are confirmed; the size warning answered too)
    for _ in range(4):
        window.handle_key(key_event("Ctrl+Q"))
        QApplication.processEvents()
        if not (window._all_asked or window._pending_warning is not None):
            break
    assert anchors(session) == [7, 12, 14], window.status_message()
    for step in (8, 10, 11, 12):
        assert np.array_equal(shape_at(session, step), OPENED), step
    for step in (13, 14):
        assert np.array_equal(shape_at(session, step), old), step
