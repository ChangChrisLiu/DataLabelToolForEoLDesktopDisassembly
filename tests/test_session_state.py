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

from session_scene import DESKTOP, VIEW, cell, draw, make_session, seed_shapes
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
