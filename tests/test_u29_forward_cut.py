"""Task U29 (C): the forward walk stops a shape commit at an open state boundary too.

The reverse walk has done it since U8 round 3 (:func:`tda.ui.session_edit.plan_cuts`:
a commit never crosses an open ✂ -- the version in force is cut at the boundary
below, a re-trace stops at the one above, and 改全部 on a ✂ row is refused).
The forward walk (``browse_forward``: a repair walk, from frame k towards the
later steps) wrote straight across: ``Q`` on a carried frame pulled the old
version back to k-1 and gave the new one its whole anchor, through the frames
of the next state. Now it mirrors the reverse walk.

The D13 scene of the U8 tests, 24 steps: ``ram_latch.01`` (closed on 1-13, the
log opens it at 14) drawn once, anchored at 24, and said closed at 20-21 with
``T``. Its one version V crosses three open boundaries: 13|14, 19|20, 21|22.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from session_scene import DESKTOP, VIEW, cell, make_session, rect, seed_shapes
from tda.core.model import FrameKey
from tda.ui import session_api as api
from tda.ui import session_edit as edit
from tda.ui.session_api import SessionRefusal

LATCH = "ram_latch.01"
LAST = 24
NEW = rect(40, 2, 62, 12)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(qapp, tmp_path):
    made = make_session(tmp_path, last_step=LAST)
    made.sweeper_enabled = False
    seed_shapes(made, 22, anchor=LAST)
    made.goto(21)
    made.assert_state(LATCH, "closed")
    made.goto(20)
    made.assert_state(LATCH, "closed")
    made.browse_forward()
    yield made
    made.close(force=True)


def anchors(session) -> list[tuple[int, str]]:
    return sorted((kf.anchor_step, kf.source)
                  for kf in session.db.keyframes(DESKTOP, VIEW, LATCH))


def shape_at(session, step: int) -> np.ndarray:
    return session.truth.compile(FrameKey(DESKTOP, step, VIEW)).instances[LATCH].amodal


def commit(session, step: int, scope: str, mask=NEW) -> dict:
    session.goto(step, force=True)
    session.begin_edit(LATCH)
    session.set_editing_mask(mask)
    return session.commit_edit(scope)


def test_the_scene(session):
    walk = edit.reverse_walk(session.db, DESKTOP, VIEW)
    opened = edit.open_split_steps(session.db, session.tax, DESKTOP, VIEW, LATCH,
                                   walk=walk)
    assert opened == [13, 19, 21]
    assert [a for a, _src in anchors(session)] == [LAST]


def test_q_going_forward_stops_at_the_boundary_ahead(session):
    """Q on 17 (carried from 16): the new version is 17-19, the closed 20-21 and the
    open 22-24 keep the old shape through a cut copy at the old anchor."""
    old = shape_at(session, 22)
    session.goto(17, force=True)
    session.begin_edit(LATCH)
    session.set_editing_mask(NEW)
    assert session.enter_scope(LATCH) == api.SCOPE_SPLIT        # carried from behind
    preview = session.preview(api.SCOPE_SPLIT)
    result = session.commit_edit(api.SCOPE_SPLIT)
    assert result["cut_ahead"] == 19 and result["cut_behind"] is None
    assert result["cut_below"] is None and result["cut_top"] is None
    assert result["edit_reach"] == [17, 18, 19] == preview["steps"]
    assert anchors(session) == [(16, "manual"), (19, "manual"), (LAST, "cut")]
    for step in (17, 18, 19):
        assert np.array_equal(shape_at(session, step), NEW), step
    for step in (1, 14, 16, 20, 21, 22, 24):
        assert np.array_equal(shape_at(session, step), old), step
    # the copy is a placeholder: walking on, 20's ✂ row is not done
    session.goto(20, force=True)
    row = [r for r in session.task_card() if r["instance"] == LATCH][0]
    assert row["kind"] == api.KIND_SPLIT_KEYFRAME and not row["done"] and row["cut_copy"]
    # one op, one undo
    assert session.undo()
    assert anchors(session) == [(LAST, "manual")]
    assert np.array_equal(shape_at(session, 17), old)


def test_q_on_the_frame_before_a_boundary_writes_that_frame_only(session):
    session.goto(19, force=True)
    result = commit(session, 19, api.SCOPE_SPLIT)
    assert result["cut_ahead"] == 19 and result["edit_reach"] == [19]
    assert [a for a, _s in anchors(session)] == [18, 19, LAST]


def test_the_retrace_going_forward_keeps_both_neighbouring_states(session):
    """改全部 on 21 (the closed run 20-21, no ✂ row on 21's forward card): the run is
    written, the open frames behind keep V pulled back to 19, the open ones ahead the
    cut copy at 24."""
    old = shape_at(session, 22)
    session.goto(21, force=True)
    session.begin_edit(LATCH)
    session.set_editing_mask(NEW)
    preview = session.preview(api.SCOPE_KEYFRAME)
    result = session.commit_edit(api.SCOPE_KEYFRAME)
    assert (result["cut_behind"], result["cut_ahead"]) == (19, 21)
    assert result["edit_reach"] == [20, 21] == preview["steps"]
    assert anchors(session) == [(19, "manual"), (21, "manual"), (LAST, "cut")]
    for step in (20, 21):
        assert np.array_equal(shape_at(session, step), NEW), step
    for step in (14, 19, 22, 24):
        assert np.array_equal(shape_at(session, step), old), step
    assert session.undo()
    assert anchors(session) == [(LAST, "manual")]


def test_the_retrace_on_a_forward_boundary_row_is_refused_with_the_same_sentence(session):
    """20's forward card holds the ✂ row (open at 19, closed at 20): 改全部 there would
    rewrite the open frames behind -- the reverse walk's sentence, those frames named."""
    session.goto(20, force=True)
    row = [r for r in session.task_card() if r["instance"] == LATCH][0]
    assert row["kind"] == api.KIND_SPLIT_KEYFRAME and not row["done"]
    session.begin_edit(LATCH)
    session.set_editing_mask(NEW)
    with pytest.raises(SessionRefusal) as refused:
        session.commit_edit(api.SCOPE_KEYFRAME)
    said = str(refused.value)
    # the run behind that state holds over (review N5): 14-19, not 1-19 (1-13 are closed)
    assert said == edit.ACROSS_BOUNDARY.format(first=14, last=19, what="（打开）")
    assert said == "这一帧是状态边界（✂）：改全部会把第 14–19 步（打开）的形状一起改掉。用 Q（从这帧起新版本）"
    assert anchors(session) == [(LAST, "manual")]
    # Q there is the split at the boundary, as in the reverse walk
    result = session.commit_edit(api.SCOPE_SPLIT)
    assert result["edit_reach"] == [20, 21] and result["cut_ahead"] == 21


def test_the_reverse_sentence_is_unchanged(session):
    session.browse_reverse()
    session.goto(21, force=True)
    session.begin_edit(LATCH)
    session.set_editing_mask(NEW)
    with pytest.raises(SessionRefusal) as refused:
        session.commit_edit(api.SCOPE_KEYFRAME)
    assert str(refused.value) == ("这一帧是状态边界（✂）：改全部会把第 22–24 步（打开）"
                                  "的形状一起改掉。用 Q（从这帧起新版本）")


def test_the_status_line_says_where_a_forward_commit_stopped():
    from tda.ui.app_state import cut_note

    text = cut_note({"cut_ahead": 19, "cut_behind": None, "cut_below": None,
                     "cut_top": None, "edit_reach": [17, 18, 19]},
                    lambda s: f"{s[0]}–{s[-1]}")
    assert text == "新版本只到第 19 步，后面的帧保持原来的形状，这次修改只影响第 17–19 步"
    text = cut_note({"cut_ahead": 21, "cut_behind": 19, "edit_reach": [20, 21]},
                    lambda s: f"{s[0]}–{s[-1]}")
    assert text.startswith("已先在第 19 步切开旧版本（形状不变），新版本只到第 21 步")


@pytest.mark.parametrize("step,said", [
    (19, "第 20–21 步（关上）"),            # was 20-24: 22-24 are open again
    (13, "第 14–19 步（打开）"),            # was 14-24
])
def test_the_reverse_sentence_names_the_run_its_state_holds_over(session, step, said):
    """Review N5: the frames named are the ones in the state named, up to the next ✂."""
    session.browse_reverse()
    session.goto(step, force=True)
    session.begin_edit(LATCH)
    session.set_editing_mask(NEW)
    with pytest.raises(SessionRefusal) as refused:
        session.commit_edit(api.SCOPE_KEYFRAME)
    assert f"改全部会把{said}的形状一起改掉" in str(refused.value)


def test_a_shape_drawn_by_nobody_is_not_cut_going_forward(qapp, tmp_path):
    """No version in force (a first draw), and a re-trace inside one run: nothing cut."""
    made = make_session(tmp_path, last_step=LAST)
    made.sweeper_enabled = False
    made.browse_forward()
    try:
        result = commit(made, 18, api.SCOPE_KEYFRAME, cell(60))
        assert result["cut_ahead"] is None and result["cut_behind"] is None
        assert len(made.db.keyframes(DESKTOP, VIEW, LATCH)) == 1
    finally:
        made.close(force=True)
