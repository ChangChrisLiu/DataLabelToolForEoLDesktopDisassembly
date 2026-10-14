"""Task U8 (core): "at step k this part is in state s" as hand-written events.

The scene is the real D13 sheet up to step 24: ``ram_latch.01`` is opened at
step 14 by the log, and the D13/scan pictures show it closed again around
frames 19-21 -- the correction this module exists for.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest

from session_scene import DESKTOP, seed_db
from tda.core import state_assert as SA
from tda.core.db import Db
from tda.core.model import StateEvent
from tda.core.states import INITIAL_STEP, state_at, validate_events
from tda.core.taxonomy import load_taxonomy
from tda.core.truth_inputs import events_of
from tda.ui.class_names import state_zh

LATCH = "ram_latch.01"
LAST = 24


@pytest.fixture
def scene(tmp_path):
    tax = load_taxonomy()
    db = Db(str(tmp_path / "tda.sqlite"))
    seed_db(db, tax, tmp_path / "cache", last_step=LAST)
    return db, tax


def manual(db: Db, target: str = LATCH) -> list[tuple]:
    """The target's hand-written events as ``(step, old, new)``, stored order."""
    return [(e.step, e.old, e.new) for e in db.events(DESKTOP)
            if not e.auto and e.target == target]


def rows(db: Db) -> list[tuple]:
    """Every hand-written event row, every field but the id, stored order."""
    return [tuple(asdict(e).values()) for e in db.events(DESKTOP) if not e.auto]


def states(db: Db, tax, steps, target: str = LATCH) -> dict[int, str]:
    instances = db.instances(DESKTOP)
    log = events_of(db, tax, DESKTOP)
    return {s: state_at(instances, log, s, tax)[target].state for s in steps}


def test_the_log_opens_the_latch_at_14(scene):
    db, tax = scene
    assert states(db, tax, (13, 14, 20, 24)) == {13: "closed", 14: "open", 20: "open",
                                                  24: "open"}


def test_walking_backwards_leaves_exactly_two_events(scene):
    """21, then 20, then 19: one run, open -> closed at 19 and back at 22."""
    db, tax = scene
    for step in (21, 20, 19):
        patch = SA.assert_state(db, tax, DESKTOP, LATCH, step, "closed", evidence_view="scan")
        assert patch.changed and patch.previous == "open"
        assert patch.next_step == step + 1
    assert manual(db) == [(19, "open", "closed"), (22, "closed", "open")]
    stored = [e for e in db.events(DESKTOP) if not e.auto]
    assert all(e.evidence_view == "scan" and e.attr == "state" for e in stored)
    assert states(db, tax, range(17, 24)) == {
        17: "open", 18: "open", 19: "closed", 20: "closed", 21: "closed",
        22: "open", 23: "open"}
    assert validate_events(db.instances(DESKTOP), events_of(db, tax, DESKTOP), tax) == []


def test_walking_forwards_ends_in_the_same_two_events(scene):
    db, tax = scene
    for step in (19, 20, 21):
        SA.assert_state(db, tax, DESKTOP, LATCH, step, "closed", evidence_view="scan")
    assert manual(db) == [(19, "open", "closed"), (22, "closed", "open")]


def test_the_two_frames_of_the_trial_20_and_21(scene):
    """The probe's expectation: ``open->closed @20``, ``closed->open @22``."""
    db, tax = scene
    for step in (21, 20):
        SA.assert_state(db, tax, DESKTOP, LATCH, step, "closed", evidence_view="scan")
    assert manual(db) == [(20, "open", "closed"), (22, "closed", "open")]


def test_asserting_what_holds_writes_nothing(scene):
    db, tax = scene
    before = rows(db)
    patch = SA.assert_state(db, tax, DESKTOP, LATCH, 18, "open")
    assert not patch.changed and patch.steps == [] and patch.after == []
    assert rows(db) == before
    # ... and inside a closed run, closed again changes nothing either
    SA.assert_state(db, tax, DESKTOP, LATCH, 20, "closed")
    after = rows(db)
    assert not SA.assert_state(db, tax, DESKTOP, LATCH, 20, "closed").changed
    assert rows(db) == after


def test_reopening_the_middle_of_a_run_splits_it_and_closing_it_merges_back(scene):
    db, tax = scene
    for step in (19, 20, 21):
        SA.assert_state(db, tax, DESKTOP, LATCH, step, "closed")
    SA.assert_state(db, tax, DESKTOP, LATCH, 20, "open")
    assert manual(db) == [(19, "open", "closed"), (20, "closed", "open"),
                          (21, "open", "closed"), (22, "closed", "open")]
    SA.assert_state(db, tax, DESKTOP, LATCH, 20, "closed")
    assert manual(db) == [(19, "open", "closed"), (22, "closed", "open")]
    # shrinking from either end
    SA.assert_state(db, tax, DESKTOP, LATCH, 19, "open")
    SA.assert_state(db, tax, DESKTOP, LATCH, 21, "open")
    assert manual(db) == [(20, "open", "closed"), (21, "closed", "open")]
    SA.assert_state(db, tax, DESKTOP, LATCH, 20, "open")
    assert manual(db) == []


def test_a_correction_at_the_step_of_the_derived_event_folds_after_it(scene):
    """Closed at 14, the step the log opens it: the manual event comes after."""
    db, tax = scene
    SA.assert_state(db, tax, DESKTOP, LATCH, 14, "closed")
    assert manual(db) == [(14, "open", "closed"), (15, "closed", "open")]
    assert states(db, tax, (13, 14, 15)) == {13: "closed", 14: "closed", 15: "open"}
    assert validate_events(db.instances(DESKTOP), events_of(db, tax, DESKTOP), tax) == []


def test_the_last_step_has_no_next_one_to_restore(scene):
    db, tax = scene
    patch = SA.assert_state(db, tax, DESKTOP, LATCH, LAST, "closed")
    assert patch.next_step is None and patch.restores is None
    assert manual(db) == [(LAST, "open", "closed")]


def test_the_inverse_restores_the_exact_rows(scene):
    """Unrelated hand-written rows are untouched, and the inverse is exact."""
    db, tax = scene
    db.add_events([
        StateEvent(DESKTOP, INITIAL_STEP, "ram_latch.02", "state", "closed", "open",
                   evidence_view=None, auto=False),
        StateEvent(DESKTOP, 20, "ram_latch.03", "state", "closed", "open",
                   evidence_view="oak1", auto=False),
    ])
    SA.assert_state(db, tax, DESKTOP, LATCH, 21, "closed", evidence_view="scan")
    start = rows(db)
    patch = SA.assert_state(db, tax, DESKTOP, LATCH, 20, "closed", evidence_view="scan")
    assert [(e.step, e.old, e.new) for e in patch.before] == [(21, "open", "closed")]
    assert [(e.step, e.old, e.new) for e in patch.after] == [(20, "open", "closed")]
    inverse = patch.inverse()
    SA.write_events(db, inverse["desktop"], inverse["target"], inverse["steps"],
                    inverse["events"])
    assert rows(db) == start
    payload = patch.payload()
    SA.write_events(db, payload["desktop"], payload["target"], payload["steps"],
                    payload["events"])
    assert manual(db) == [(20, "open", "closed"), (22, "closed", "open")]
    assert manual(db, "ram_latch.02") == [(INITIAL_STEP, "closed", "open")]
    assert manual(db, "ram_latch.03") == [(20, "closed", "open")]
    assert payload["state"] == "closed" and inverse["state"] == "open"
    assert payload["reach"] == [20, 21]


@pytest.mark.parametrize("target,step,state,words", [
    (LATCH, 20, "plugged", "没有「plugged」这个状态"),
    (LATCH, 20, "removed", "由步骤表决定"),
    ("nothing.01", 20, "open", "没有零件 nothing.01"),
    (LATCH, 99, "closed", "第 99 步不在"),
    # the sheet takes the SSD out at step 10
    ("storage_drive.ssd.01", 12, "displaced", "已经拆下"),
])
def test_refusals_say_why_and_write_nothing(scene, target, step, state, words):
    db, tax = scene
    before = rows(db)
    with pytest.raises(SA.StateAssertError) as refused:
        SA.assert_state(db, tax, DESKTOP, target, step, state)
    assert words in str(refused.value)
    assert isinstance(refused.value, ValueError)
    assert rows(db) == before


def test_a_correction_that_breaks_a_derived_event_is_refused(scene):
    """Open at 13: the log's own ``closed -> open`` at 14 would no longer hold.

    When the action itself happened a step earlier, the step table is what is
    wrong -- a manual event would only paper over it.
    """
    db, tax = scene
    before = rows(db)
    with pytest.raises(SA.StateAssertError) as refused:
        SA.assert_state(db, tax, DESKTOP, LATCH, 13, "open", name=state_zh)
    text = str(refused.value)
    assert "日志第 14 步把 ram_latch.01 从「关上」变成「打开」" in text
    assert "已经是「打开」" in text and "Steps 模式" in text
    assert "step" not in text and "current state" not in text    # no validator English
    assert rows(db) == before


def test_assertable_states_come_from_the_taxonomy(scene):
    _db, tax = scene
    assert SA.assertable_states(tax, "ram_latch") == ["closed", "open"]
    assert SA.assertable_states(tax, "cover") == ["closed", "open"]
    assert SA.assertable_states(tax, "cable_clip") == ["closed", "open"]
    assert SA.assertable_states(tax, "screw") == ["fastened", "loosened"]
    assert SA.assertable_states(tax, "motherboard") == ["installed", "displaced"]
    # one mask-bearing state, or none: nothing to choose
    assert SA.assertable_states(tax, "connector") == []
    assert SA.assertable_states(tax, "chassis") == []
    assert SA.assertable_states(tax, "cable") == []
    assert SA.assertable_states(tax, "no_such_class") == []


def test_a_rewrite_of_one_part_leaves_the_derived_log_alone(scene):
    db, tax = scene
    derived = [e for e in events_of(db, tax, DESKTOP) if e.auto]
    SA.assert_state(db, tax, DESKTOP, LATCH, 20, "closed")
    assert [e for e in events_of(db, tax, DESKTOP) if e.auto] == derived


def test_s1_validates_the_merged_log_not_the_stored_row_order(tmp_path):
    """A correction at the step of a derived event is valid in S1 too.

    ``Apply`` rewrites the derived rows, so they get newer ids than the
    hand-written one at the same step; validating in stored-id order folded
    the correction *before* the action it corrects and reported it.
    """
    from steps_fixtures import seeded_db
    from tda.ui.steps_model import StepTableData

    tax = load_taxonomy()
    db = seeded_db(str(tmp_path / "s1.sqlite"), tax)
    try:
        SA.assert_state(db, tax, 13, LATCH, 14, "closed", evidence_view="scan")
        data = StepTableData.load(db, 13, tax)
        messages = data.save(db)
        assert not [m for m in messages if LATCH in m], messages
        assert [(e.step, e.old, e.new) for e in db.events(13)
                if not e.auto and e.target == LATCH] == [(14, "open", "closed"),
                                                          (15, "closed", "open")]
    finally:
        db.close()


def _dupli(db, step: int) -> None:
    steps = db.steps(DESKTOP)
    for rec in steps:
        if rec.step == step:
            rec.step_type = "dupli"
    db.replace_steps(DESKTOP, steps, db.actions(DESKTOP))


def test_a_duplicate_frame_takes_the_state_of_the_frame_it_repeats(scene):
    """B2: the restore goes past ``dupli`` steps; 22 repeats 21, 23 restores."""
    db, tax = scene
    _dupli(db, 22)
    patch = SA.assert_state(db, tax, DESKTOP, LATCH, 21, "closed")
    assert patch.inherits == [22] and patch.next_step == 23
    assert patch.reach == [21, 22, 23]
    assert manual(db) == [(21, "open", "closed"), (23, "closed", "open")]
    assert states(db, tax, (20, 21, 22, 23)) == {20: "open", 21: "closed",
                                                  22: "closed", 23: "open"}
    assert validate_events(db.instances(DESKTOP), events_of(db, tax, DESKTOP), tax) == []


def test_an_assertion_at_a_duplicate_frame_points_at_the_frame_it_repeats(scene):
    db, tax = scene
    _dupli(db, 22)
    before = rows(db)
    with pytest.raises(SA.StateAssertError,
                       match="第 22 步在步骤表里是重复帧（dupli），和第 21 步是同一个样子"):
        SA.assert_state(db, tax, DESKTOP, LATCH, 22, "closed")
    assert rows(db) == before


def test_a_run_extends_over_a_duplicate_frame(scene):
    """Closed at 20, then 21 too: the run carries through dupli 22 to 23."""
    db, tax = scene
    _dupli(db, 22)
    SA.assert_state(db, tax, DESKTOP, LATCH, 20, "closed")
    assert manual(db) == [(20, "open", "closed"), (21, "closed", "open")]
    SA.assert_state(db, tax, DESKTOP, LATCH, 21, "closed")
    assert manual(db) == [(20, "open", "closed"), (23, "closed", "open")]
    # and back: open at 21 again ends the run at 20, 22 follows 21
    SA.assert_state(db, tax, DESKTOP, LATCH, 21, "open")
    assert manual(db) == [(20, "open", "closed"), (21, "closed", "open")]
    assert states(db, tax, (20, 21, 22, 23)) == {20: "closed", 21: "open",
                                                  22: "open", 23: "open"}


# --------------------------------------------------------------------------- #
# B1: the last frame before a removal
# --------------------------------------------------------------------------- #
@pytest.fixture
def whole(tmp_path):
    """The whole sheet: the board leaves at 42 and takes the clips with it."""
    tax = load_taxonomy()
    db = Db(str(tmp_path / "whole.sqlite"))
    seed_db(db, tax, tmp_path / "cache", last_step=42)
    return db, tax


def test_a_removal_is_legal_from_any_state_but_removed():
    """``validate_events``: the ``old`` of a move into ``removed`` is not checked."""
    tax = load_taxonomy()
    from tda.core.model import InstanceRec

    instances = {"ram_latch.01": InstanceRec(key="ram_latch.01", desktop=1, cls="ram_latch")}
    open_then_out = [
        StateEvent(1, 14, "ram_latch.01", "state", "closed", "open"),
        StateEvent(1, 41, "ram_latch.01", "state", "open", "closed", auto=False),
        StateEvent(1, 42, "ram_latch.01", "state", "open", "removed"),  # recorded open
    ]
    assert validate_events(instances, open_then_out, tax) == []
    back = open_then_out + [StateEvent(1, 43, "ram_latch.01", "state", "removed", "open",
                                       auto=False)]
    assert [p.split(": ", 2)[-1] for p in validate_events(instances, back, tax)] == [
        "'removed' is final, an instance cannot leave it"]
    wrong_old = [StateEvent(1, 14, "ram_latch.01", "state", "open", "closed")]
    assert "the current state at that step is 'closed'" in validate_events(
        instances, wrong_old, tax)[0]


def test_walking_back_from_the_removal_leaves_one_event(whole):
    """41, 40, 39 closed: one event at 39; the board's step takes it out closed."""
    db, tax = whole
    for step in (41, 40, 39):
        patch = SA.assert_state(db, tax, DESKTOP, LATCH, step, "closed")
        assert patch.changed and patch.next_step == step + 1
    assert manual(db) == [(39, "open", "closed")]
    assert states(db, tax, range(38, 43)) == {38: "open", 39: "closed", 40: "closed",
                                              41: "closed", 42: "removed"}
    assert validate_events(db.instances(DESKTOP), events_of(db, tax, DESKTOP), tax) == []


@pytest.mark.parametrize("target,step,state", [
    (LATCH, 41, "closed"),                         # the host cascade at 42
    ("ram_latch.02", 41, "closed"),
    ("screw.motherboard.01", 35, "loosened"),      # its own removal at 36
    ("screw.motherboard.06", 40, "loosened"),      # ... at 41
    ("motherboard.01", 41, "displaced"),           # the board itself at 42
])
def test_the_frame_before_a_removal_can_be_corrected(whole, target, step, state):
    db, tax = whole
    patch = SA.assert_state(db, tax, DESKTOP, target, step, state)
    assert [(e.step, e.old, e.new) for e in patch.after] == [(step, patch.previous, state)]
    assert states(db, tax, (step, step + 1), target) == {step: state, step + 1: "removed"}
    assert validate_events(db.instances(DESKTOP), events_of(db, tax, DESKTOP), tax) == []


def test_s1_apply_takes_a_correction_before_a_removal(tmp_path):
    """The same validator runs in S1: closed at 41 must not read as a problem there."""
    from steps_fixtures import seeded_db
    from tda.ui.steps_model import StepTableData

    tax = load_taxonomy()
    db = seeded_db(str(tmp_path / "s1.sqlite"), tax)
    try:
        SA.assert_state(db, tax, 13, LATCH, 41, "closed", evidence_view="scan")
        messages = StepTableData.load(db, 13, tax).save(db)
        assert not [m for m in messages if LATCH in m], messages
    finally:
        db.close()
