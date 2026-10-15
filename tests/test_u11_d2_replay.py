"""Task U11, end to end: the D2/scan cover walk, replayed.

What happened on desktop 2 / scan (2026-10-14, 23:40-23:55): ``cover.02`` is in
the log from 1 to 45 and gone at 46, and nobody wrote down that it was opened.
The annotator stood on 45, said "open" with ``T``, drew it and pressed
``Enter``; then walked down to 37 pressing ``T`` at every frame -- and, because
the cut the first commit made stayed at 44 while the run grew, re-drew almost
every frame. Seven keyframes for two shapes.

The replay runs four steps earlier on the D13 sheet: ``ram_latch.04`` with its
``open`` action taken out is closed from 1 to 41 and leaves with the board at
42 (D2's 45/46 are 41/42 here, D2's 42 is 38, D2's 37 is 33). The latch has a
closed shape drawn before the walk (one version, 1-41), as the cover had.

After U11: ``T`` at 41 + ``Enter``; ``T`` at 40, 39, 38; the moved shape at 38 +
``Ctrl+K``; ``T`` at 37, 36, 35, 34 -- two commits, and the moved shape is in
force on 34-38 without a redraw, the first open shape on 39-41, the cut copy
(the closed shape) at 33. ``Ctrl+T`` makes the same in three gestures.

Round 2: the copy at 33 is a placeholder, so its ✂ row stays open until the
closed shape is drawn there (``Ctrl+K``, which clears the mark) or the copy is
said to be right (``Ctrl+Shift+K``).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from session_scene import DESKTOP, VIEW, draw, make_session, rect, seed_shapes
from tda.core.cut_follow import CUT
from tda.ui import session_api as api
from tda.ui.session_api import SessionRefusal

LATCH = "ram_latch.04"
LAST = 42
HW = (256, 256)
OPEN = rect(2, 2, 30, 12, HW)        # drawn at 41 (D2: 45)
MOVED = rect(4, 4, 34, 14, HW)       # the cover swung further: drawn at 38 (D2: 42)
CLOSED2 = rect(40, 40, 70, 60, HW)   # the closed shape, drawn at 33 at the end (D2: 37)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _scene(tmp_path, seed: bool = True):
    made = make_session(tmp_path, last_step=LAST, hw=HW)
    made.sweeper_enabled = False
    db = made.db
    db.replace_steps(DESKTOP, db.steps(DESKTOP),
                     [a for a in db.actions(DESKTOP)
                      if not (a.target == LATCH and a.verb == "open")])
    made._invalidate()
    if seed:
        seed_shapes(made, 33, anchor=LAST)
    else:
        seed_shapes(made, 33, anchor=LAST, skip=(LATCH,))
    return made


@pytest.fixture
def scene(qapp, tmp_path):
    made = _scene(tmp_path)
    yield made
    made.close(force=True)


def amodal(s, step: int) -> np.ndarray:
    s.goto(step, force=True)
    return s.compiled().instances[LATCH].amodal.copy()


def anchors(s) -> list[tuple[int, str]]:
    return sorted((kf.anchor_step, kf.source) for kf in s.db.keyframes(DESKTOP, VIEW, LATCH))


def manual(s) -> list[tuple]:
    return [(e.step, e.old, e.new) for e in s.db.events(DESKTOP)
            if not e.auto and e.target == LATCH]


def t(s, step: int, state: str = "open") -> dict:
    s.goto(step, force=True)
    return s.assert_state(LATCH, state)


def commits(s) -> int:
    return sum(1 for op in s.db.ops(DESKTOP, VIEW, limit=200) if op["kind"] == "commit_keyframe")


def check_the_result(s, closed: np.ndarray) -> None:
    assert manual(s) == [(34, "closed", "open")]
    assert anchors(s) == [(33, CUT), (38, "manual"), (42, "manual")]
    for step in range(34, 39):
        assert (amodal(s, step) == MOVED).all(), step
    for step in range(39, 42):
        assert (amodal(s, step) == OPEN).all(), step
    for step in (20, 33):
        assert (amodal(s, step) == closed).all(), step
    row = card_row(s, 33)
    assert row["kind"] == api.KIND_SPLIT_KEYFRAME
    assert row["done"] is False and row["cut_copy"] is True   # a placeholder (round 2)


def card_row(s, step: int) -> dict:
    s.goto(step, force=True)
    return next(r for r in s.task_card() if r["instance"] == LATCH)


def draw_the_closed_shape_at_33(s) -> None:
    """The end of both walks: the ✂ row at 33 asks, Ctrl+K answers."""
    s.goto(33, force=True)
    s.begin_edit(LATCH)
    with pytest.raises(SessionRefusal, match="另一状态的形状副本"):
        s.commit_edit(api.SCOPE_SPLIT)          # untouched: it would sign the copy
    s.clear_edit()
    result = draw(s, LATCH, CLOSED2, api.SCOPE_SPLIT)
    s.clear_edit()
    assert result["no_split"] is True and result["cut_below"] is None
    assert anchors(s) == [(33, "manual"), (38, "manual"), (42, "manual")]
    assert card_row(s, 33)["done"] is True
    assert (amodal(s, 20) == CLOSED2).all() and (amodal(s, 34) == MOVED).all()


def test_the_t_walk_needs_no_redraw(scene):
    s = scene
    closed = amodal(s, 41)
    t(s, 41)
    s.goto(41, force=True)
    assert draw(s, LATCH, OPEN, api.SCOPE_KEYFRAME)["cut_below"] == 40
    s.clear_edit()
    for step in (40, 39, 38):
        moved = t(s, step)["cuts"]
        assert [(m.old, m.new) for m in moved] == [(step, step - 1)]
        assert (amodal(s, step) == OPEN).all(), step          # no redraw
    s.goto(38, force=True)
    result = draw(s, LATCH, MOVED, api.SCOPE_SPLIT)
    s.clear_edit()
    assert result["cut_below"] is None                       # the cut is at 37 already
    for step in (37, 36, 35, 34):
        moved = t(s, step)["cuts"]
        assert [(m.old, m.new) for m in moved] == [(step, step - 1)]
        assert (amodal(s, step) == MOVED).all(), step         # no redraw
    check_the_result(s, closed)
    assert commits(s) == 2
    draw_the_closed_shape_at_33(s)


def test_ctrl_t_then_enter_then_ctrl_k_is_the_same_in_three_gestures(scene):
    s = scene
    closed = amodal(s, 41)
    s.goto(41, force=True)
    result = s.assert_state_range(LATCH, "open", 34)
    assert result["changed"] and result["patch"].span == list(range(34, 42))
    s.goto(41, force=True)
    assert draw(s, LATCH, OPEN, api.SCOPE_KEYFRAME)["cut_below"] == 33
    s.clear_edit()
    s.goto(38, force=True)
    draw(s, LATCH, MOVED, api.SCOPE_SPLIT)
    s.clear_edit()
    check_the_result(s, closed)
    assert commits(s) == 2
    assert [op["kind"] for op in s.db.ops(DESKTOP, VIEW, limit=3)] == [
        "commit_keyframe", "commit_keyframe", "assert_state"]
    draw_the_closed_shape_at_33(s)


def test_without_a_closed_shape_the_copy_is_the_first_open_one(qapp, tmp_path):
    """No shape before the walk: Enter at 41 draws the first one (no cut), and
    Ctrl+K at 38 cuts it at 37 -- the copy that follows the run down is the
    open shape of 41, so the closed frames below 34 still need their own
    shape. Its ✂ row reads done: the version changes there (see the report)."""
    s = _scene(tmp_path, seed=False)
    try:
        t(s, 41)
        s.goto(41, force=True)
        assert draw(s, LATCH, OPEN, api.SCOPE_KEYFRAME)["cut_below"] is None
        s.clear_edit()
        for step in (40, 39, 38):
            assert t(s, step)["cuts"] == []
        s.goto(38, force=True)
        assert draw(s, LATCH, MOVED, api.SCOPE_SPLIT)["cut_below"] == 37
        s.clear_edit()
        for step in (37, 36, 35, 34):
            t(s, step)
        assert anchors(s) == [(33, CUT), (38, "manual"), (41, "manual")]
        assert (amodal(s, 33) == OPEN).all() and (amodal(s, 34) == MOVED).all()
    finally:
        s.close(force=True)
