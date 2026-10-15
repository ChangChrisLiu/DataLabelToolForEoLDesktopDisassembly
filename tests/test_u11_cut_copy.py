"""Task U11 round 2: a cut copy is a placeholder that confines edits, never an answer.

A ✂ row whose frame shows a ``source="cut"`` keyframe is not done: the card
says 「待画：这帧上是另一状态的形状副本，画好后 Ctrl+K」 (the banner too),
``Ctrl+K`` with the frame's shape drawn clears the mark and completes it, an
untouched ``Ctrl+K`` is refused, and ``Ctrl+Shift+K`` (「这帧就是这个形状」)
takes the mark off without pixels. ``cli check`` counts the open row
(``shape_predates_state``) and every other confirmed frame under the copy
(``cut_copy_in_force``). A boundary with no cut copy (the log's own) is as
it was.

Scene: the D13 sheet, 256 px, every part drawn once (anchor 42), ``ram_latch.04``
without its ``open`` action -- closed 1-41, out with the board at 42 (D2's cover).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from session_scene import DESKTOP, VIEW, draw, make_session, rect, seed_shapes
from tda.core.cut_follow import CUT
from tda.core.split_rows import cut_copies_in_force, stale_split_rows
from tda.ui import app_actions as A
from tda.ui import session_api as api
from tda.ui.panels.taskcard import CUT_COPY_SENTENCE, row_view
from tda.ui.session_api import SessionRefusal

CLIP = "ram_latch.04"
HW = (256, 256)
OPEN = rect(2, 2, 30, 12, HW)
CLOSED2 = rect(40, 40, 70, 60, HW)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def scene(qapp, tmp_path):
    made = make_session(tmp_path, last_step=42, hw=HW)
    made.sweeper_enabled = False
    db = made.db
    db.replace_steps(DESKTOP, db.steps(DESKTOP),
                     [a for a in db.actions(DESKTOP)
                      if not (a.target == CLIP and a.verb == "open")])
    made._invalidate()
    seed_shapes(made, 33, anchor=42)
    yield made
    made.close(force=True)


def anchors(s, instance: str = CLIP) -> list[tuple[int, str]]:
    return sorted((kf.anchor_step, kf.source) for kf in s.db.keyframes(DESKTOP, VIEW, instance))


def row(s, step: int, instance: str = CLIP) -> dict:
    s.goto(step, force=True)
    return next(r for r in s.task_card() if r["instance"] == instance)


def amodal(s, step: int, instance: str = CLIP) -> np.ndarray:
    s.goto(step, force=True)
    return s.compiled().instances[instance].amodal.copy()


def open_at_41_and_draw(s) -> np.ndarray:
    """``T`` open at 41 and the open shape drawn: the closed shape is cut to 40."""
    closed = amodal(s, 41)
    s.goto(41, force=True)
    s.assert_state(CLIP, "open")
    assert draw(s, CLIP, OPEN, api.SCOPE_KEYFRAME)["cut_below"] == 40
    s.clear_edit()
    return closed


def test_the_row_over_a_cut_copy_is_open_and_says_so(scene):
    s = scene
    open_at_41_and_draw(s)
    found = row(s, 40)
    assert found["kind"] == api.KIND_SPLIT_KEYFRAME
    assert found["done"] is False and found["cut_copy"] is True
    view = row_view(found)
    assert view["sentence"].startswith("待画：这帧上是另一状态的形状副本，画好后 Ctrl+K")
    assert view["sentence"].startswith(CUT_COPY_SENTENCE) and view["chip"] == "待画"
    assert s.cut_copy_here(CLIP) is True
    s.goto(41, force=True)
    assert s.cut_copy_here(CLIP) is False


def test_ctrl_k_with_the_frames_shape_clears_the_mark(scene):
    s = scene
    open_at_41_and_draw(s)
    s.goto(40, force=True)
    s.begin_edit(CLIP)
    with pytest.raises(SessionRefusal, match="另一状态的形状副本"):
        s.commit_edit(api.SCOPE_SPLIT)                 # untouched: refused
    s.clear_edit()
    assert anchors(s) == [(40, CUT), (42, "manual")]
    result = draw(s, CLIP, CLOSED2, api.SCOPE_SPLIT)
    s.clear_edit()
    assert result["no_split"] is True                   # the copy ends here: re-traced
    assert anchors(s) == [(40, "manual"), (42, "manual")]
    assert row(s, 40)["done"] is True and "cut_copy" not in row(s, 40)
    assert (amodal(s, 40) == CLOSED2).all() and (amodal(s, 41) == OPEN).all()


def test_ctrl_shift_k_says_the_copy_is_this_frames_shape(scene):
    s = scene
    closed = open_at_41_and_draw(s)
    s.goto(40, force=True)
    result = s.confirm_cut_copy(CLIP)
    assert result["confirmed_cut"] == 40
    assert anchors(s) == [(40, "manual"), (42, "manual")]
    assert row(s, 40)["done"] is True
    assert (amodal(s, 40) == closed).all()               # not a pixel changed
    op = s.db.ops(DESKTOP, VIEW, limit=1)[0]
    assert op["kind"] == "commit_keyframe" and op["payload"]["scope"] == "confirm_cut_copy"
    assert s.undo() is True                              # the mark comes back
    assert anchors(s) == [(40, CUT), (42, "manual")] and row(s, 40)["done"] is False
    assert s.redo() is True
    assert anchors(s) == [(40, "manual"), (42, "manual")]


def test_ctrl_shift_k_is_refused_where_there_is_no_copy_or_pixels_wait(scene):
    s = scene
    open_at_41_and_draw(s)
    s.goto(41, force=True)
    with pytest.raises(SessionRefusal, match="不是切开时留下的副本"):
        s.confirm_cut_copy(CLIP)
    s.goto(40, force=True)
    s.begin_edit(CLIP)
    s.set_editing_mask(CLOSED2)
    with pytest.raises(SessionRefusal, match="还没提交"):
        s.confirm_cut_copy(CLIP)
    assert anchors(s) == [(40, CUT), (42, "manual")]
    s.clear_edit()
    s.begin_edit(CLIP)                                   # untouched: closed, then confirmed
    s.confirm_cut_copy()
    assert s.editing_instance is None and anchors(s)[0] == (40, "manual")


def test_check_counts_the_row_and_the_confirmed_frames_under_the_copy(scene):
    s = scene
    s.goto(41, force=True)
    for _ in range(4):
        assert s.confirm_frame() is True                 # 41, 40, 39, 38 frozen
    s.goto(41, force=True)
    s.assert_state(CLIP, "open")
    draw(s, CLIP, OPEN, api.SCOPE_KEYFRAME)              # cut at 40, the closed shape
    s.clear_edit()

    def mine(rows):
        return [r for r in rows if r[1] == CLIP]

    stale = mine(stale_split_rows(s.db, s.tax, DESKTOP, VIEW))
    assert stale == [(40, CLIP)]                         # the open row: shape_predates_state
    under = mine(cut_copies_in_force(s.db, s.tax, DESKTOP, VIEW, skip=stale))
    assert under == [(38, CLIP), (39, CLIP)]             # the other confirmed frames
    s.goto(40, force=True)
    s.confirm_cut_copy(CLIP)
    assert mine(stale_split_rows(s.db, s.tax, DESKTOP, VIEW)) == []
    assert mine(cut_copies_in_force(s.db, s.tax, DESKTOP, VIEW)) == []


def test_cli_check_names_both(scene, capsys):
    import argparse

    from tda.cli_app import _check_one

    s = scene
    s.goto(41, force=True)
    for _ in range(4):
        assert s.confirm_frame() is True
    s.goto(41, force=True)
    s.assert_state(CLIP, "open")
    draw(s, CLIP, OPEN, api.SCOPE_KEYFRAME)
    s.clear_edit()
    _check_one(s.db, s.tax, argparse.Namespace(limit=500), DESKTOP, VIEW)
    out = capsys.readouterr().out
    assert f"shape_predates_state:{CLIP} (step 40)" in out
    assert f"cut_copy_in_force:{CLIP} (step 39)" in out
    assert f"cut_copy_in_force:{CLIP} (step 38)" in out
    assert f"cut_copy_in_force:{CLIP} (step 40)" not in out     # counted once


def test_a_log_boundary_without_a_copy_is_as_it_was(qapp, tmp_path):
    """D13's own boundary (ram_latch.01 opened at 14): ✂ at 13, no copy, U8's rules."""
    s = make_session(tmp_path, last_step=24, hw=HW)
    s.sweeper_enabled = False
    try:
        seed_shapes(s, 13, anchor=24)
        found = row(s, 13, "ram_latch.01")
        assert found["done"] is False and "cut_copy" not in found
        assert row_view(found)["sentence"] != CUT_COPY_SENTENCE
        s.goto(13, force=True)
        assert s.cut_copy_here("ram_latch.01") is False
        s.begin_edit("ram_latch.01")
        result = s.commit_edit(api.SCOPE_SPLIT)          # untouched Ctrl+K still cuts
        s.clear_edit()
        assert result.get("unchanged_split") is True
        assert row(s, 13, "ram_latch.01")["done"] is True
    finally:
        s.close(force=True)


def test_ctrl_shift_k_is_bound_in_annotate_mode_only():
    action = A.action_named("confirm_cut_copy")
    assert action.keys == ("Ctrl+Shift+K",) and action.slot == "act_confirm_cut_copy"
    assert action.modes == (A.MODE_ANNOTATE,)
    mods = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
    found = A.action_for(Qt.Key.Key_K, mods, A.MODE_ANNOTATE)
    assert found is not None and found.name == "confirm_cut_copy"
    assert [a.name for a in A.ACTIONS if "Ctrl+Shift+K" in a.keys] == ["confirm_cut_copy"]


# --------------------------------------------------------------------------- #
# the window: the banner says the row's sentence, the key does the session's work
# --------------------------------------------------------------------------- #
def test_the_window_banner_and_key(qapp, tmp_path: Path):
    from app_scene import StubSamQueue, close_window, make_paths
    from app_scene import make_session as make_window_session
    from tda.core import masks
    from tda.core.model import ShapeKeyframe, ShapePart
    from tda.ui.app import MainWindow
    from tda.ui.app_guide import BANNER_CUT_COPY

    session = make_window_session(tmp_path, last_step=42)
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
    try:
        if win.roi_editing:
            win.wait_for_roi_proposal()
            win.act_commit()
        if win.roi_editing:
            win.act_clear_edit()
        win.session.goto(41, force=True)
        win.session.assert_state(CLIP, "open")
        shape = np.zeros((64, 64), dtype=bool)
        shape[4:14, 4:30] = True
        win.session.begin_edit(CLIP)
        win.session.set_editing_mask(shape)
        win.session.commit_edit(api.SCOPE_KEYFRAME)      # cut at 40
        win.session.clear_edit()
        win.session.goto(40, force=True)
        QApplication.processEvents()
        win.task_card.refresh()
        win.task_card.sigRequestEdit.emit(CLIP)          # click the row: the copy loads
        QApplication.processEvents()
        win.expire_visibility_note()
        assert win.canvas.banner_text().endswith(BANNER_CUT_COPY)
        assert BANNER_CUT_COPY.startswith("待画：这帧上是另一状态的形状副本，画好后 Ctrl+K")
        win.act_confirm_cut_copy()                       # Ctrl+Shift+K
        assert win.session.editing_instance is None
        assert "✂ 完成" in win.status_message()
        assert sorted((kf.anchor_step, kf.source)
                      for kf in db.keyframes(DESKTOP, "scan", CLIP)) == [
            (40, "manual"), (42, "manual")]
        win.act_undo()
        assert (40, CUT) in [(kf.anchor_step, kf.source)
                             for kf in db.keyframes(DESKTOP, "scan", CLIP)]
    finally:
        close_window(win)
