"""Task U33: ``import-logs --force`` turns D36's wrong clip into the cable event.

The database D36 was imported into before U33 holds ``cable_clip.01`` (cls
``cable_clip``, raw name "Remove 1 isolated cable from motherboard", step 2's
``release``) and -- on the live database -- one hand-set state event on it:
the annotator answered the card's request for the clip with 「这一步之后是
closed」 on step 42. The scene below writes exactly that state on top of the
real sheet (``tests/fixtures/logs/desktop_36.csv``), then re-imports it:

* with no keyframe, the clip goes, its hand-set event with it (in the
  ``dropped_instance`` op row), step 2 targets ``cable:isolated.01`` with
  ``remove``, and the instance / action / state-event tables are exactly what a
  fresh import of the sheet writes;
* with a keyframe, the clip is kept and the import says why, in words;
* a second ``--force`` changes nothing; D23 and D63 keep their clips.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from tda.core import masks
from tda.core.db import Db
from tda.core.model import ActionRec, InstanceRec, ShapeKeyframe, ShapePart, StateEvent
from tda.core.states import events_from_actions
from tda.core.taxonomy import load_taxonomy
from tda.pipeline_logs import DROP_OP_KIND, cable_event_clips, import_logs_into_db

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
D36 = 36
ROW = "Remove 1 isolated cable from motherboard"
CLIP = "cable_clip.01"
NODE = "cable:isolated.01"
#: The hand-set event the live database carries (op 10836, assert_state).
ASSERTED = StateEvent(D36, 42, CLIP, "state", "open", "closed", evidence_view="scan",
                      auto=False)


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def _import(db: Db, tax, desktop: int = D36, force: bool = False, lines=None):
    run = import_logs_into_db(db, str(FIXTURES), tax, desktops={desktop}, force=force,
                              log=(lines.append if lines is not None else None))
    [one] = run.runs
    assert one.status == "imported", one.issues
    return one


def _tables(db: Db, desktop: int = D36) -> dict:
    """The three tables U33 may touch, without row ids."""
    return {
        "instance": {k: asdict(v) for k, v in db.instances(desktop).items()},
        "action": sorted((a.step, a.idx, a.target, a.verb, a.tool, a.direction, a.result)
                         for a in db.actions(desktop)),
        "state_event": [(e.step, e.target, e.attr, e.old, e.new, e.evidence_view, e.auto)
                        for e in db.events(desktop)],
    }


def _as_before_u33(db: Db, tax, asserted: bool = True) -> None:
    """Write what the pre-U33 importer left in the database for D36's step 2."""
    steps, actions = db.steps(D36), db.actions(D36)
    old = [replace(a, target=CLIP, verb="release") if a.step == 2 else a for a in actions]
    db.replace_steps(D36, steps, old)
    clip = InstanceRec(key=CLIP, desktop=D36, cls="cable_clip", attrs={"sheet_no": 1},
                       raw_names=[ROW])
    db.upsert_instance(clip)
    instances = db.instances(D36)
    db.replace_events(D36, events_from_actions(instances, old, tax), auto_only=True)
    if asserted:
        db.add_events([ASSERTED])


@pytest.fixture
def fresh(tmp_path, tax) -> dict:
    """What a first import of the sheet writes today: the reference tables."""
    db = Db(str(tmp_path / "fresh.sqlite"))
    try:
        _import(db, tax)
        return _tables(db)
    finally:
        db.close()


@pytest.fixture
def before(tmp_path, tax):
    db = Db(str(tmp_path / "d36.sqlite"))
    _import(db, tax)
    _as_before_u33(db, tax)
    yield db
    db.close()


# --------------------------------------------------------------------------- #
# the scene really is the pre-U33 state
# --------------------------------------------------------------------------- #
def test_the_scene_has_the_wrong_clip(before):
    assert CLIP in before.instances(D36)
    step2 = [a for a in before.actions(D36) if a.step == 2]
    assert [(a.target, a.verb) for a in step2] == [(CLIP, "release")]
    clip_events = [(e.step, e.old, e.new, e.auto) for e in before.events(D36)
                   if e.target == CLIP]
    assert clip_events == [(2, "closed", "open", True), (42, "open", "closed", False)]


# --------------------------------------------------------------------------- #
# --force: no keyframe -> the cable event, cleanly
# --------------------------------------------------------------------------- #
def test_force_turns_the_clip_into_the_cable_event(before, tax, fresh):
    lines: list[str] = []
    run = _import(before, tax, force=True, lines=lines)
    assert CLIP not in before.instances(D36)
    step2 = [a for a in before.actions(D36) if a.step == 2]
    assert [(a.target, a.verb, a.result) for a in step2] == [(NODE, "remove", "success")]
    # nothing names the clip any more, the cable's own event is there
    assert not [e for e in before.events(D36) if e.target == CLIP]
    assert [(e.step, e.old, e.new, e.auto) for e in before.events(D36)
            if e.target == NODE] == [(2, "routed", "removed", True)]
    # the instance / action / state-event tables are what a fresh import writes
    assert _tables(before) == fresh
    said = (f"dropped instance {CLIP} (the sheet's {ROW!r} is a cable event now "
            f"({NODE}), not a part; 1 hand-set state event(s) went with it)")
    assert run.dropped == [said]
    assert any(said in line for line in lines)
    assert run.kept_for_review == 0


def test_the_drop_op_keeps_the_record_and_the_hand_set_event(before, tax):
    _import(before, tax, force=True)
    [op] = [r for r in before.ops(D36, "-") if r["kind"] == DROP_OP_KIND]
    payload, inverse = op["payload"], op["inverse"]
    assert payload["instance"] == CLIP and payload["cable"] == NODE
    assert payload["record"]["cls"] == "cable_clip"
    assert payload["record"]["raw_names"] == [ROW]
    want = [{**asdict(ASSERTED)}]
    assert [{k: e[k] for k in want[0]} for e in payload["events"]] == want
    assert inverse["events"] == payload["events"]
    assert "cable event now" in payload["reason"]


def test_a_clip_with_no_hand_set_event_goes_the_same_way(tmp_path, tax, fresh):
    db = Db(str(tmp_path / "plain.sqlite"))
    try:
        _import(db, tax)
        _as_before_u33(db, tax, asserted=False)
        run = _import(db, tax, force=True)
        assert run.dropped == [f"dropped instance {CLIP} (the sheet's {ROW!r} is a cable "
                               f"event now ({NODE}), not a part)"]
        assert _tables(db) == fresh
        [op] = [r for r in db.ops(D36, "-") if r["kind"] == DROP_OP_KIND]
        assert op["payload"]["events"] == []
    finally:
        db.close()


def test_a_second_force_changes_nothing(before, tax):
    _import(before, tax, force=True)
    tables = _tables(before)
    drops = len([r for r in before.ops(D36, "-") if r["kind"] == DROP_OP_KIND])
    run = _import(before, tax, force=True)
    assert _tables(before) == tables
    assert not run.dropped
    assert len([r for r in before.ops(D36, "-") if r["kind"] == DROP_OP_KIND]) == drops


# --------------------------------------------------------------------------- #
# --force: a keyframe keeps it, and the import says so
# --------------------------------------------------------------------------- #
def test_a_clip_with_a_keyframe_is_kept_and_reported(before, tax):
    mask = np.zeros((64, 64), dtype=bool)
    mask[8:16, 8:16] = True
    before.add_keyframe(ShapeKeyframe(
        id=None, instance=CLIP, desktop=D36, view="scan", pose_segment=1,
        anchor_step=42, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(mask))]))
    lines: list[str] = []
    run = _import(before, tax, force=True, lines=lines)
    assert CLIP in before.instances(D36)
    assert not run.dropped
    assert run.kept_for_review == 1 and run.kept_carrying_work == 1
    said = (f"D36: instance {CLIP} was the sheet's {ROW!r}, a cable event now ({NODE}), "
            f"but is kept (1 shape_keyframe, 1 state_event) - if that work is not of "
            f"a part, undo it (the keyframe / frame override / confirmed row / conflict "
            f"/ relation that holds it), then run import-logs --force again")
    assert said in run.issues
    assert any(said in line for line in lines)
    # the step is the cable event all the same, and the hand-set answer stays
    step2 = [a for a in before.actions(D36) if a.step == 2]
    assert [(a.target, a.verb) for a in step2] == [(NODE, "remove")]
    assert [(e.step, e.new, e.auto) for e in before.events(D36) if e.target == CLIP] == \
        [(42, "closed", False)]


def test_the_advice_works_undo_the_holding_work_and_force_again(before, tax, fresh):
    """Round 2 (N2): S1 cannot delete it -- its hand-set event is a reference --
    but undoing the work and re-importing drops it, event and all."""
    from tda.core.model import FrameKey, FrameOverride
    from tda.ui.steps_delete import check_deletable
    from tda.ui.steps_model import StepTableData
    from tda.ui.steps_values import EditError

    held = FrameOverride(FrameKey(D36, 42, "scan"), CLIP, None, "occluded_full",
                         occluder="cable")
    before.set_frame_override(held)
    run = _import(before, tax, force=True)
    assert CLIP in before.instances(D36) and not run.dropped
    assert any("but is kept (1 frame_override, 1 state_event)" in i for i in run.issues)
    with pytest.raises(EditError):       # what the old advice sent the annotator to
        check_deletable(StepTableData.load(before, D36, tax), before, CLIP)
    before.delete_frame_override(held.frame, CLIP)
    run = _import(before, tax, force=True)
    assert CLIP not in before.instances(D36)
    assert run.dropped and "1 hand-set state event(s) went with it" in run.dropped[0]
    assert _tables(before) == fresh


def test_an_ordinary_vanished_part_still_holds_on_its_hand_set_event(before, tax):
    """Only a clip whose row is a cable event lets go of its events (U33)."""
    ghost = InstanceRec(key="psu.02", desktop=D36, cls="psu", raw_names=["Power module"])
    before.upsert_instance(ghost)
    before.add_events([StateEvent(D36, 30, "psu.02", "state", "installed", "displaced",
                                  evidence_view="scan", auto=False)])
    run = _import(before, tax, force=True)
    assert "psu.02" in before.instances(D36) and CLIP not in before.instances(D36)
    assert ("D36: instance psu.02 vanished from the sheet but is kept (1 state_event) "
            "- merge or delete it in S1") in run.issues


def test_the_match_is_by_every_raw_name(tax):
    """A part another row still names is not a cable, whatever else it carries."""
    from tda.core.logs import import_log, read_desktop_csv

    rows, meta = read_desktop_csv(FIXTURES / "desktop_36.csv")
    li = import_log(D36, rows, meta, tax)
    stored = {
        CLIP: InstanceRec(key=CLIP, desktop=D36, cls="cable_clip", raw_names=[ROW]),
        "cable_clip.02": InstanceRec(key="cable_clip.02", desktop=D36, cls="cable_clip",
                                     raw_names=[ROW, "Heatsink"]),
        "cable_clip.03": InstanceRec(key="cable_clip.03", desktop=D36, cls="cable_clip"),
    }
    assert cable_event_clips(stored, li) == {CLIP: (NODE, ROW)}


# --------------------------------------------------------------------------- #
# the clips that are clips stay clips
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("desktop,step", [(23, 2), (63, 3)])
def test_d23_and_d63_keep_their_clip_through_force(tmp_path, tax, desktop, step):
    db = Db(str(tmp_path / f"d{desktop}.sqlite"))
    try:
        _import(db, tax, desktop)
        before = _tables(db, desktop)
        run = _import(db, tax, desktop, force=True)
        assert not run.dropped
        assert _tables(db, desktop) == before
        acts = [a for a in db.actions(desktop) if a.step == step]
        assert [(a.target, a.verb) for a in acts] == [(CLIP, "release")]
        assert db.instances(desktop)[CLIP].cls == "cable_clip"
        assert not [a for a in db.actions(desktop) if a.target.startswith("cable:")]
    finally:
        db.close()
