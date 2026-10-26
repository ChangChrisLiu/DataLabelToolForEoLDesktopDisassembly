"""Task U34 (core): 「这台没有这个零件」 -- a logged part the machine does not have.

The op writes two hand-written step-0 events (``installed -> removed``,
``in_chassis -> elsewhere``) and its inverse takes them away; from then on the
part needs geometry nowhere, the sheet's own actions on it derive nothing, a
later event about it is a no-op without a warning, its children stay as the
sheet says, and the task card never has a row -- ✂ or other -- for it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from absent_scene import DISPLACED_AT, DRIVE, FAILED_AT, RIDER, SCREWS, add_drive
from session_scene import DESKTOP, VIEW, seed_db
from tda.core import absent as AB
from tda.core.db import Db
from tda.core.extra import plan_extras, write_batch
from tda.core.graph import validate_sequence
from tda.core.model import InstanceRec, ShapeKeyframe, ShapePart, StateEvent
from tda.core.states import (
    INITIAL_STEP,
    absent_keys,
    event_problems,
    events_from_actions,
    gallery_state,
    initial_overrides,
    is_gallery_part,
    needs_geom,
    state_at,
    validate_events,
)
from tda.core.taxonomy import load_taxonomy
from tda.core.truth_inputs import events_of, instances_of, state_of
from tda.ui import session_api as api
from tda.ui.session_tasks import task_card_for

LAST = 14


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def db(tmp_path: Path, tax):
    made = Db(str(tmp_path / "tda.sqlite"))
    seed_db(made, tax, tmp_path / "cache", last_step=LAST)
    add_drive(made, tax, DESKTOP)
    yield made
    made.close()


def manual(db: Db, key: str = DRIVE) -> list[tuple]:
    return [(e.step, e.attr, e.old, e.new, e.evidence_view) for e in db.events(DESKTOP)
            if not e.auto and e.target == key]


def held(db: Db, tax, step: int, key: str = DRIVE):
    return state_of(db, tax, DESKTOP, step).get(key)


def needs(db: Db, tax, step: int) -> dict:
    return needs_geom(instances_of(db, DESKTOP), state_of(db, tax, DESKTOP, step), tax)


# --------------------------------------------------------------------------- #
# the op and its inverse
# --------------------------------------------------------------------------- #
def test_the_op_is_two_step_0_events_and_one_op_with_its_inverse(db, tax):
    assert held(db, tax, 1).state == "installed"
    assert held(db, tax, DISPLACED_AT).state == "displaced"   # the sheet's push
    op_id, patch = AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    assert manual(db) == [
        (INITIAL_STEP, "state", "installed", "removed", AB.EVIDENCE),
        (INITIAL_STEP, "placement", "in_chassis", "elsewhere", AB.EVIDENCE)]
    op = db.get_op(op_id)
    assert op["kind"] == AB.OP_KIND == "absent_from_start" and op["view"] == AB.OP_VIEW
    assert op["payload"]["instance"] == DRIVE and op["payload"]["absent"] is True
    assert [e["new"] for e in op["payload"]["events"]] == ["removed", "elsewhere"]
    assert op["inverse"] == {"desktop": DESKTOP, "instance": DRIVE, "absent": False,
                             "events": []}
    assert patch.absent and patch.before == [] and len(patch.after) == 2
    assert absent_keys(db.events(DESKTOP)) == {DRIVE}
    assert AB.is_absent(db, DESKTOP, DRIVE)


def test_from_step_0_on_it_is_removed_elsewhere_and_needs_nothing(db, tax):
    AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    for step in range(1, LAST + 1):
        inst = held(db, tax, step)
        assert (inst.state, inst.placement, inst.absent) == ("removed", "elsewhere", True)
        assert DRIVE not in needs(db, tax, step)
    # not on the bench either: a view with a bench ROI would ask for its box there
    inst = held(db, tax, LAST)
    assert DRIVE not in needs_geom(instances_of(db, DESKTOP),
                                   state_of(db, tax, DESKTOP, LAST), tax,
                                   bench_roi=(0, 0, 64, 64))


def test_the_inverse_takes_the_events_away(db, tax):
    AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    op_id, patch = AB.mark_present(db, tax, DESKTOP, DRIVE, "tester")
    assert manual(db) == [] and not AB.is_absent(db, DESKTOP, DRIVE)
    assert patch.absent is False and patch.after == []
    assert db.get_op(op_id)["payload"]["absent"] is False
    assert held(db, tax, 1).state == "installed"
    assert held(db, tax, DISPLACED_AT).state == "displaced"   # the sheet counts again
    assert DRIVE in needs(db, tax, 1)


def test_the_inverse_puts_back_a_hand_written_initial_state(db, tax):
    """What stood at step 0 before the op comes back with 「其实有」."""
    db.add_events([StateEvent(DESKTOP, INITIAL_STEP, DRIVE, "state", "installed",
                              "displaced", evidence_view=None, auto=False)])
    assert held(db, tax, 1).state == "displaced"
    _op, patch = AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    assert patch.after[0].old == "displaced"        # the state it held before step 1
    AB.mark_present(db, tax, DESKTOP, DRIVE, "tester")
    assert manual(db) == [(INITIAL_STEP, "state", "installed", "displaced", None)]
    assert held(db, tax, 1).state == "displaced"


def test_the_stored_automatic_events_are_recompiled_with_it(db, tax):
    stored = [(e.step, e.attr, e.new) for e in db.events(DESKTOP)
              if e.auto and e.target == DRIVE]
    assert stored == [(DISPLACED_AT, "state", "displaced")]
    AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    assert [e for e in db.events(DESKTOP) if e.auto and e.target == DRIVE] == []


# --------------------------------------------------------------------------- #
# the sheet's actions on it: no resurrection, no warning
# --------------------------------------------------------------------------- #
def test_a_later_displace_and_a_failed_remove_are_no_ops(db, tax):
    AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    log = events_of(db, tax, DESKTOP)
    assert [e for e in log if e.target == DRIVE and e.auto] == []
    assert [m for m in validate_events(instances_of(db, DESKTOP), log, tax)
            if DRIVE in m] == []
    for step in (FAILED_AT, DISPLACED_AT, LAST):
        assert held(db, tax, step).state == "removed"


def test_a_stale_or_hand_written_later_event_cannot_resurrect_it(db, tax):
    """Any event about it after step 0 is folded as a no-op and never warned about."""
    AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    instances = instances_of(db, DESKTOP)
    log = [*events_of(db, tax, DESKTOP),
           StateEvent(DESKTOP, DISPLACED_AT, DRIVE, "state", "installed", "displaced",
                      auto=True),
           StateEvent(DESKTOP, 12, DRIVE, "state", "removed", "removed", auto=True),
           StateEvent(DESKTOP, 13, DRIVE, "state", "displaced", "installed",
                      evidence_view=VIEW, auto=False)]
    for step in (DISPLACED_AT, 12, 13, LAST):
        inst = state_at(instances, log, step, tax)[DRIVE]
        assert (inst.state, inst.placement) == ("removed", "elsewhere")
    assert [p for p in event_problems(instances, log, tax) if p.event.target == DRIVE] == []


def test_its_screws_and_its_rider_stay_as_the_sheet_says(db, tax):
    """No cascade: the annotator decides the dependents separately."""
    AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    assert held(db, tax, 6, SCREWS[0]).state == "fastened"
    assert held(db, tax, 7, SCREWS[0]).state == "removed"       # its own unscrew
    rider = held(db, tax, LAST, RIDER)
    assert (rider.state, rider.placement, rider.left_with) == ("fastened", "in_chassis", None)
    assert RIDER in needs(db, tax, LAST)


def test_a_parent_leaving_takes_nothing_out_through_an_absent_child(tax):
    """The derived log and the closure both stop at a part the machine does not have."""
    instances = {
        "chassis": InstanceRec("chassis", 1, "chassis"),
        "drive_cage.01": InstanceRec("drive_cage.01", 1, "drive_cage"),
        DRIVE: InstanceRec(DRIVE, 1, "optical_drive", parent="drive_cage.01", attached=True),
        RIDER: InstanceRec(RIDER, 1, "screw", parent=DRIVE, attached=True,
                           attrs={"role": "optical_drive"}),
    }
    from tda.core.model import ActionRec

    actions = [ActionRec(1, 3, 0, "drive_cage.01", "remove")]
    initial = {DRIVE: "removed"}
    derived = events_from_actions(instances, actions, tax, initial=initial)
    assert [e.target for e in derived] == ["drive_cage.01", "drive_cage.01"]
    marks = [StateEvent(1, 0, DRIVE, "state", "installed", "removed", AB.EVIDENCE, False)]
    final = state_at(instances, [*derived, *marks], 5, tax)
    assert final[RIDER].state == "fastened" and final[RIDER].placement == "in_chassis"
    assert final[DRIVE].absent and final[DRIVE].left_with is None


def test_the_7_4_replay_does_not_replay_its_actions(db, tax):
    instances = instances_of(db, DESKTOP)
    actions = db.actions(DESKTOP)
    before = [p for p in validate_sequence(instances, [], actions, tax) if DRIVE in p]
    assert before        # "failed remove ... has no unmet constraint - missing edge?"
    AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    initial = initial_overrides(e for e in db.events(DESKTOP) if not e.auto)
    assert [p for p in validate_sequence(instances, [], actions, tax, initial=initial)
            if DRIVE in p] == []


def test_it_never_joins_the_components_gallery(db, tax):
    """Even when the sheet "removes" it successfully: it was never there to lay out."""
    from tda.core.model import ActionRec

    steps, actions = db.steps(DESKTOP), list(db.actions(DESKTOP))
    actions = [a for a in actions if a.target != DRIVE]
    actions.append(ActionRec(DESKTOP, FAILED_AT, 1, DRIVE, "remove"))
    db.replace_steps(DESKTOP, steps, actions)
    instances = instances_of(db, DESKTOP)
    final = state_at(instances, events_of(db, tax, DESKTOP), 10**6, tax)
    assert is_gallery_part(instances, final, DRIVE)
    AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    log = events_of(db, tax, DESKTOP)
    final = state_at(instances, log, 10**6, tax)
    assert not is_gallery_part(instances, final, DRIVE)
    assert DRIVE not in needs_geom(instances, gallery_state(instances, log, tax), tax)


# --------------------------------------------------------------------------- #
# the task card: no row, ✂ or other
# --------------------------------------------------------------------------- #
def _card_rows(db, tax, key: str = DRIVE) -> dict[int, list[str]]:
    out = {}
    for step in range(1, LAST + 1):
        neighbour = step + 1 if step < LAST else None
        rows = [r["kind"] for r in task_card_for(db, tax, DESKTOP, VIEW, step, neighbour)
                if r.get("instance") == key]
        if rows:
            out[step] = rows
    return out


def test_the_card_never_asks_for_it_and_has_no_cut_row(db, tax):
    before = _card_rows(db, tax)
    # the sheet's push at 11: the reverse walk's frame 10 splits its shape (✂)
    assert before.get(DISPLACED_AT - 1) == [api.KIND_SPLIT_KEYFRAME]
    assert all(DRIVE in needs(db, tax, s) for s in range(1, LAST + 1))
    AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    assert _card_rows(db, tax) == {}
    assert all(DRIVE not in needs(db, tax, s) for s in range(1, LAST + 1))
    from tda.core.split_rows import open_split_steps, reverse_walk

    walk = reverse_walk(db, DESKTOP, VIEW)
    assert open_split_steps(db, tax, DESKTOP, VIEW, DRIVE, 1, LAST + 1, walk=walk) == []


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_a_part_with_a_shape_anywhere_is_in_the_machine(db, tax):
    from tda.core import masks
    import numpy as np

    mask = np.zeros((64, 64), dtype=bool)
    mask[2:9, 2:9] = True
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=DRIVE, desktop=DESKTOP, view="oak1", pose_segment=1,
        anchor_step=LAST, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(mask))]))
    with pytest.raises(AB.AbsentError, match="已经画过形状（oak1 1 个）"):
        AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    assert manual(db) == []


def test_the_other_refusals_write_nothing(db, tax):
    assert "机箱" in AB.refusal(db, tax, DESKTOP, "chassis")
    assert "没有零件" in AB.refusal(db, tax, DESKTOP, "nothing.01")
    batch = plan_extras(instances_of(db, DESKTOP), tax, DESKTOP, "misc_part", 1,
                        annotator="tester")
    write_batch(db, batch, "tester")
    assert "手动添加的零件" in AB.refusal(db, tax, DESKTOP, batch.keys[0])
    AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    with pytest.raises(AB.AbsentError, match="已经是「一开始就不在」"):
        AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    AB.mark_present(db, tax, DESKTOP, DRIVE, "tester")
    with pytest.raises(AB.AbsentError, match="没有标成"):
        AB.mark_present(db, tax, DESKTOP, DRIVE, "tester")


# --------------------------------------------------------------------------- #
# frozen frames
# --------------------------------------------------------------------------- #
def test_the_frozen_frames_it_was_in_are_queued_for_a_re_check(db, tax):
    from tda.core.model import FrameKey

    for step in (12, 13):
        db.set_frame_flags(FrameKey(DESKTOP, step, VIEW), review_status="verified")
    _op, patch = AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    assert patch.rechecks == {VIEW: [12, 13]} and patch.frames == 2
    assert set(db.rechecks(DESKTOP, VIEW)) >= {12, 13}
    # and the way back reaches the same frames (it is in them again)
    _op, back = AB.mark_present(db, tax, DESKTOP, DRIVE, "tester")
    assert back.rechecks == {VIEW: [12, 13]}
