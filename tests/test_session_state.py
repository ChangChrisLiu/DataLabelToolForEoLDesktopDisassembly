"""Task U8 (session): the state of a part at the frame in hand, undoably.

``AnnotationSession.assert_state`` writes the core's patch, logs it with its
inverse, settles the frames it reaches and goes on the undo stack like any
other edit; the task card then shows the ✂ rows of the new boundaries exactly
as it shows the step table's.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from session_scene import DESKTOP, VIEW, cell, draw, make_session, rect, seed_shapes
from tda.core.model import FrameKey
from tda.core.states import needs_geom
from tda.core.truth_inputs import instances_of, state_of
from tda.ui import session_api as api
from tda.ui.session_api import SessionRefusal

LATCH = "ram_latch.01"
LAST = 24


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(qapp, tmp_path):
    made = make_session(tmp_path, last_step=LAST)
    made.sweeper_enabled = False       # the re-check queue is inspected, not drained
    seed_shapes(made, 22, anchor=LAST)   # frames 22-24 complete
    yield made
    made.close(force=True)


def manual(session, target: str = LATCH) -> list[tuple]:
    return [(e.step, e.old, e.new) for e in session.db.events(DESKTOP)
            if not e.auto and e.target == target]


def card_row(session, step: int, instance: str = LATCH):
    session.goto(step, force=True)
    found = [r for r in session.task_card() if r["instance"] == instance]
    return found[0] if found else None


def test_the_rows_say_which_parts_can_change_state_here(session):
    session.goto(21)
    rows = {r["key"]: r for r in session.instance_rows()}
    assert rows[LATCH]["state"] == "open"
    assert rows[LATCH]["state_choices"] == ["closed", "open"]
    assert rows["chassis"]["state_choices"] == []
    assert rows["motherboard.01"]["state_choices"] == ["installed", "displaced"]
    assert session.next_state(LATCH) == "closed"


def test_an_assertion_is_one_op_with_its_inverse(session):
    session.goto(21)
    result = session.assert_state(LATCH, "closed")
    assert result["changed"] and result["affected"] == [21, 22]
    ops = session.db.ops(DESKTOP, VIEW, limit=5)
    assert ops[0]["kind"] == "assert_state"
    payload, inverse = ops[0]["payload"], ops[0]["inverse"]
    assert payload["view"] == VIEW and payload["target"] == LATCH
    assert [(e["step"], e["old"], e["new"]) for e in payload["events"]] == [
        (21, "open", "closed"), (22, "closed", "open")]
    assert inverse["events"] == [] and inverse["steps"] == payload["steps"] == [21, 22]
    assert session.undo_stack.ops[-1].kind == "assert_state"
    assert session.dirty


def test_undo_and_redo_walk_the_assertions(session):
    session.goto(21)
    session.assert_state(LATCH, "closed")
    session.goto(20)
    session.assert_state(LATCH, "closed")
    assert manual(session) == [(20, "open", "closed"), (22, "closed", "open")]
    assert session.undo() is True
    assert manual(session) == [(21, "open", "closed"), (22, "closed", "open")]
    assert session.undo() is True
    assert manual(session) == []
    assert session.redo() and session.redo()
    assert manual(session) == [(20, "open", "closed"), (22, "closed", "open")]
    rows = {r["key"]: r for r in session.instance_rows()}
    assert rows[LATCH]["state"] == "closed"           # the panel reads it straight


def test_asserting_the_state_that_holds_is_no_op(session):
    session.goto(21)
    before = len(session.undo_stack)
    result = session.assert_state(LATCH, "open")
    assert result["changed"] is False
    assert len(session.undo_stack) == before
    assert session.db.ops(DESKTOP, VIEW, limit=1) == [] or \
        session.db.ops(DESKTOP, VIEW, limit=1)[0]["kind"] != "assert_state"


def test_the_gesture_is_gated_where_there_is_nothing_to_set(session):
    session.goto(21)
    with pytest.raises(SessionRefusal, match="不在画面里"):
        session.assert_state("ram_module.01", "displaced")   # out at step 18
    with pytest.raises(SessionRefusal, match="只有一种"):
        session.assert_state("chassis", "present")
    with pytest.raises(SessionRefusal, match="只能设为"):
        session.assert_state(LATCH, "removed")
    assert session.state_choices("ram_module.01") == []
    session.goto(13)
    with pytest.raises(SessionRefusal, match="Steps 模式"):
        session.assert_state(LATCH, "open")       # the log opens it at 14
    assert manual(session) == []


def test_a_frozen_frame_the_change_reaches_is_queued_for_a_re_check(session):
    db = session.db
    session.goto(22)
    assert session.confirm_frame() is True            # 22 frozen, now on 21
    assert session.current().step == 21
    result = session.assert_state(LATCH, "closed")
    assert 22 in result["rechecks"] and 22 in db.rechecks(DESKTOP, VIEW)
    # the frozen frame agrees with the inputs: the state is not a pixel
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    assert db.get_frame(FrameKey(DESKTOP, 22, VIEW))["review_status"] == "verified"
    assert not db.open_conflicts_at(FrameKey(DESKTOP, 22, VIEW))


def test_the_corrected_frames_get_their_split_rows_and_ctrl_k_makes_the_versions(session):
    """The D13 episode end to end: closed at 20-21, open on either side."""
    session.goto(21)
    session.assert_state(LATCH, "closed")
    session.goto(20)
    session.assert_state(LATCH, "closed")

    # reverse walk: 21 is diffed against 22 (open -> closed), 20 against 21
    # (no change), 19 against 20 (closed -> open)
    row = card_row(session, 21)
    assert row["kind"] == api.KIND_SPLIT_KEYFRAME and not row["done"]
    assert row["transition"] == ["open", "closed"]
    assert card_row(session, 20) is None
    assert card_row(session, 22) is None
    row = card_row(session, 19)
    assert row["kind"] == api.KIND_SPLIT_KEYFRAME and not row["done"]
    assert row["transition"] == ["closed", "open"]

    # closed and open both need a mask: nothing stops being drawn
    instances = instances_of(session.db, DESKTOP)
    for step in (19, 20, 21, 22):
        state = state_of(session.db, session.tax, DESKTOP, step)
        assert needs_geom(instances, state, session.tax).get(LATCH) == "mask", step

    session.goto(21)
    result = draw(session, LATCH, cell(60), api.SCOPE_SPLIT)
    assert result["anchor_step"] == 21 and not result["no_split"]
    assert card_row(session, 21)["done"] is True
    assert card_row(session, 19)["done"] is False     # the closed version reaches 19
    session.goto(19)
    result = draw(session, LATCH, cell(61), api.SCOPE_SPLIT)
    assert result["anchor_step"] == 19 and not result["no_split"]
    assert card_row(session, 19)["done"] is True

    anchors = sorted(kf.anchor_step for kf in session.db.keyframes(DESKTOP, VIEW, LATCH))
    assert anchors == [19, 21, LAST]
    for step, index in ((18, 61), (19, 61), (20, 60), (21, 60), (22, None)):
        session.goto(step, force=True)
        inst = session.compiled().instances[LATCH]
        if index is not None:
            assert (inst.amodal == cell(index)).all(), step


def test_ctrl_k_at_the_far_boundary_starts_from_the_version_in_force(session):
    """Open question of the brief, pinned: at 19 the layer is the *closed* shape.

    The version in force at 19 after the split at 21 is the closed one, so
    that is what begin_edit loads; the open shape of the keyframe above (22-24)
    is not copied in.
    """
    session.goto(21)
    session.assert_state(LATCH, "closed")
    session.goto(20)
    session.assert_state(LATCH, "closed")
    session.goto(21)
    draw(session, LATCH, cell(60), api.SCOPE_SPLIT)
    session.goto(19)
    session.begin_edit(LATCH)
    assert (session.editing_mask() == cell(60)).all()
    session.clear_edit()


# --------------------------------------------------------------------------- #
# round 2
# --------------------------------------------------------------------------- #
def test_a_frame_with_no_image_here_is_refused_and_points_elsewhere(qapp, tmp_path):
    """Minor 3: nobody can see the state on a missing frame; the restore may land there."""
    made = make_session(tmp_path, last_step=LAST, missing=(20,))
    made.sweeper_enabled = False
    try:
        made.goto(20, force=True)
        with pytest.raises(SessionRefusal, match="第 20 步在 scan 没有可用的图像") as refused:
            made.assert_state(LATCH, "closed")
        assert "第 19 步" in str(refused.value) and "视角" in str(refused.value)
        assert manual(made) == []
        made.goto(19, force=True)
        result = made.assert_state(LATCH, "closed")
        assert result["restore_unseen"] is True        # 20 has no image in this view
        assert manual(made) == [(19, "open", "closed"), (20, "closed", "open")]
    finally:
        made.close(force=True)


def test_confirmed_frames_that_still_owe_a_split_are_listed(qapp, tmp_path):
    """Minor 4: what ``cli check`` counts -- an open ✂ row on a confirmed frame."""
    from tda.ui.session_tasks import stale_split_rows

    made = make_session(tmp_path, last_step=LAST)
    made.sweeper_enabled = False
    try:
        seed_shapes(made, 19, anchor=LAST)              # frames 19-24 complete
        made.goto(22, force=True)
        for _ in range(4):
            assert made.confirm_frame() is True         # 22, 21, 20, 19 frozen

        def latch_rows() -> list[tuple[int, str]]:
            # the seed gives every part one shape, so the log's own boundaries
            # (the socket lever at 20, ...) are open ✂ rows too: only the latch
            # is this test's business
            return [r for r in stale_split_rows(made.db, made.tax, DESKTOP, VIEW)
                    if r[1] == LATCH]

        assert latch_rows() == []
        for step in (21, 20):
            made.goto(step, force=True)
            made.assert_state(LATCH, "closed")
        assert latch_rows() == [(19, LATCH), (21, LATCH)]
        made.goto(21, force=True)
        draw(made, LATCH, cell(60), api.SCOPE_SPLIT)
        assert latch_rows() == [(19, LATCH)]
        made.goto(19, force=True)
        draw(made, LATCH, cell(61), api.SCOPE_SPLIT)
        assert latch_rows() == []
    finally:
        made.close(force=True)


# --------------------------------------------------------------------------- #
# round 2b: the end of a part's life
# --------------------------------------------------------------------------- #
def test_ctrl_k_cuts_the_boundary_unchanged_so_enter_reaches_only_the_run(qapp, tmp_path):
    """Closed at 39-41, the board takes the clip at 42; ✂ lands at 38.

    At 38 the loaded shape is already the open one: Ctrl+K with the layer
    untouched cuts the version there, and Enter at 41 then re-traces 39-41
    only -- conflicts on the frozen frames of that run, 33-38 untouched.
    """
    # 256 px, so a latch is not `visible_tiny` and a re-trace is past tolerance
    made = make_session(tmp_path, last_step=42, hw=(256, 256))
    made.sweeper_enabled = False
    db = made.db
    try:
        seed_shapes(made, 33, anchor=42)                 # frames 33-42 complete
        made.goto(42, force=True)
        for _ in range(10):
            assert made.confirm_frame() is True          # 42 ... 33 frozen
        closed_shape = rect(2, 2, 30, 12, (256, 256))
        for step in (41, 40, 39):
            made.goto(step, force=True)
            made.assert_state(LATCH, "closed")
        assert manual(made) == [(39, "open", "closed")]
        row = card_row(made, 38)
        assert row["kind"] == api.KIND_SPLIT_KEYFRAME and not row["done"]
        before = {s: _amodal(made, s) for s in (33, 36, 38, 39, 41)}

        made.goto(38, force=True)
        made.begin_edit(LATCH)
        assert not made.layer.changed()
        result = made.commit_edit(api.SCOPE_SPLIT)       # Ctrl+K, nothing painted
        made.clear_edit()
        assert result["changed"] and result.get("unchanged_split") is True
        assert not result["no_split"] and result["anchor_step"] == 38
        assert sorted(kf.anchor_step for kf in db.keyframes(DESKTOP, VIEW, LATCH)) == [38, 42]
        assert card_row(made, 38)["done"] is True
        made.truth.run_pending_rechecks(DESKTOP, VIEW)
        assert db.conflicts(DESKTOP, VIEW) == []         # same pixels: nothing moved

        made.goto(41, force=True)
        result = draw(made, LATCH, closed_shape, api.SCOPE_KEYFRAME)   # Enter
        made.clear_edit()
        assert result["affected"] == [39, 40, 41]
        made.truth.run_pending_rechecks(DESKTOP, VIEW)
        assert sorted({c["step"] for c in db.conflicts(DESKTOP, VIEW)}) == [39, 40, 41]
        for step in (33, 36, 38):
            assert (_amodal(made, step) == before[step]).all(), step
        for step in (39, 41):
            assert (_amodal(made, step) == closed_shape).all(), step
    finally:
        made.close(force=True)


def test_ctrl_k_with_nothing_changed_and_no_open_row_is_still_a_no_op(session):
    session.goto(21)
    assert not session.open_split_here(LATCH)            # open on both sides
    session.begin_edit(LATCH)
    result = session.commit_edit(api.SCOPE_SPLIT)
    session.clear_edit()
    assert result["changed"] is False and "unchanged_split" not in result
    assert [kf.anchor_step for kf in session.db.keyframes(DESKTOP, VIEW, LATCH)] == [LAST]


def _amodal(session, step: int):
    session.goto(step, force=True)
    return session.compiled().instances[LATCH].amodal.copy()
