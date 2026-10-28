"""Task U38 in the window: a step with no photo shows none, and Space / Enter refuse it.

The scene is :mod:`app_scene` with step ``LAST_STEP - 1`` missing, and what D61's
scanner step 26 had: a row flagged ``missing`` whose ``aux`` still names the
copy an earlier ``build-cache`` left under its number -- a copy that exists,
with its thumbnail. Pinned: the canvas shows the no-image placeholder, not the
copy; the timeline has no thumbnail for it; ``Space`` (and ``Enter`` in Review,
the same action) is refused with one status line saying why; and the
photographed frame next to it confirms exactly as before.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import (DESKTOP, LAST_STEP, VIEW, AnnotationSession, StubSamQueue,
                       close_window, make_db, make_paths, seed_shapes)
from test_u38_missing_frames import _stale_copy
from tda.core.model import FrameKey
from tda.core.truth import TruthService
from tda.core.truth_inputs import HW_MEASURED
from tda.core.truth_refusals import FRAME_MISSING
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.panels.taskcard import NO_PHOTO_SENTENCE

GAP = LAST_STEP - 1
REFUSED = (f"step {GAP} 没有照片（漏拍），不能确认 / step {GAP} has no photo "
           f"(a missed shot) and cannot be confirmed")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    db, paths, tax = make_db(tmp_path, missing=(GAP,))
    key = FrameKey(DESKTOP, GAP, VIEW)
    stale, _thumb = _stale_copy(Path(paths["cache_dir"]), key)
    db.upsert_frame(key, "", {"hw": [64, 64], "hw_source": HW_MEASURED, "cache_path": stale},
                    None, {"missing": True})
    session = AnnotationSession(db, tax, TruthService(db, tax), paths["cache_dir"], "tester")
    session.open(DESKTOP, VIEW)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(1400, 900)
    win.show()
    QApplication.processEvents()
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()
    QApplication.processEvents()
    yield win
    close_window(win)
    session.close(force=True)


def press(win: MainWindow, name: str):
    result = win.dispatch(A.action_named(name))
    QApplication.processEvents()
    return result


def go(win: MainWindow, step: int) -> None:
    win.session.goto(step, force=True)
    QApplication.processEvents()


def test_the_missing_step_shows_the_placeholder_not_the_copy_at_its_number(window):
    key = FrameKey(DESKTOP, GAP, VIEW)
    assert Path(window.session.db.get_frame(key)["aux"]["cache_path"]).is_file()
    go(window, GAP)
    assert window.session.current().step == GAP
    assert window.session.image() is None
    assert window.session.image_path(GAP) is None
    assert window.session.thumb_path(GAP) is None        # the timeline draws no picture
    assert window.tools_enabled is False
    assert window.stack.currentWidget() is window.placeholder_label
    # the photographed neighbour still has both
    assert window.session.image_path(LAST_STEP) is not None
    assert window.session.thumb_path(LAST_STEP) is not None


def test_space_on_the_missing_step_is_refused_in_one_line(window):
    db = window.session.db
    key = FrameKey(DESKTOP, GAP, VIEW)
    go(window, GAP)
    assert window.act_confirm() is False
    assert window.status_message() == REFUSED
    assert window.session.current().step == GAP          # nothing to step back from
    assert db.compiled(key) == {}
    assert db.get_frame(key)["review_status"] in (None, "unlabeled")
    # the pane: the refusal and its one reason -- not the compiler's sixty
    # "missing shape" lines about a step nobody photographed
    rows = window.task_card.problem_rows()
    assert "cannot be verified" in rows[0]["text"]
    assert [r["code"] for r in rows[1:]] == [FRAME_MISSING]
    assert rows[1]["text"] == NO_PHOTO_SENTENCE and rows[1]["instance"] == ""
    # the Space key itself goes the same way -- to verify_frame, not to the
    # window's generic "no image" -- and the greyed button gives the same reason
    window.report("")
    press(window, "confirm")
    assert window.status_message() == REFUSED
    window.refresh_guidance()
    button = window.palette.button("confirm")
    assert not button.isEnabled() and button.reason() == REFUSED
    assert db.compiled(key) == {}


def test_enter_in_review_is_the_same_gate(window):
    go(window, GAP)
    window.set_mode(A.MODE_REVIEW)
    QApplication.processEvents()
    window.report("")
    press(window, "review_accept")
    assert window.status_message() == REFUSED
    assert window.session.db.compiled(FrameKey(DESKTOP, GAP, VIEW)) == {}


def test_a_missing_frame_never_sits_in_the_needs_review_queue(window):
    """Round 2: Enter, Space and R all refuse it, so it could never leave the queue."""
    from tda.ui import session_api as api

    session = window.session
    db = session.db
    before = session.queues()
    assert before[api.QUEUE_NEEDS_REVIEW] == []
    db.set_frame_flags(FrameKey(DESKTOP, GAP, VIEW), review_status="needs_review")
    db.set_frame_flags(FrameKey(DESKTOP, 5, VIEW), review_status="needs_review")
    session.review.invalidate()
    after = session.queues()
    assert after[api.QUEUE_NEEDS_REVIEW] == [{"step": 5}]
    for name in (api.QUEUE_CONFLICTS, api.QUEUE_MISSING_SHAPE, api.QUEUE_UNEXPLAINED):
        assert after[name] == before[name], name
    assert session.frame_status(GAP) == "missing"
    window.review.refresh()
    QApplication.processEvents()
    assert window.review.list_for(api.QUEUE_NEEDS_REVIEW).count() == 1


def test_space_on_a_frame_without_a_path_says_the_same(window):
    """Round 2: no path and not flagged missing is no photo too -- same gate, same line."""
    db = window.session.db
    key = FrameKey(DESKTOP, 5, VIEW)
    db.upsert_frame(key, "", {"hw": [64, 64]}, None, {"missing": False})
    go(window, 5)
    assert window.session.image() is None
    window.report("")
    press(window, "confirm")
    assert window.status_message() == REFUSED.replace(f"step {GAP}", "step 5")
    assert db.compiled(key) == {} or all(r["status"] != "verified"
                                         for r in db.compiled(key).values())


def test_the_steps_table_draws_no_thumbnail_for_it(qapp, tmp_path):
    """Steps mode reads the scanner copy by step number too: not for a missed shot."""
    from tda.ui.panels.steptable import StepTableModel
    from tda.ui.steps_model import StepTableData

    db, paths, tax = make_db(tmp_path, missing=(GAP,))
    try:
        _stale_copy(Path(paths["cache_dir"]), FrameKey(DESKTOP, GAP, VIEW))
        data = StepTableData.load(db, DESKTOP, tax)
        assert data.no_photo == {GAP}
        model = StepTableModel(data, paths["cache_dir"])
        assert model._thumb(GAP) is None
        assert model._thumb(LAST_STEP) is not None
    finally:
        db.close()


def test_the_photographed_frame_confirms_and_the_walk_steps_over_the_gap(window):
    """(e) Nothing changes for a frame with a photo."""
    session = window.session
    seed_shapes(session, LAST_STEP)
    go(window, LAST_STEP)
    assert window.act_confirm() is True
    assert window.status_message().startswith(f"step {LAST_STEP} confirmed")
    assert session.db.get_frame(FrameKey(DESKTOP, LAST_STEP, VIEW))["review_status"] == \
        "verified"
    assert session.current().step == GAP - 1            # the missing step is skipped
