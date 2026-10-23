"""Task U20: a drive cage's own latch -- the rule and the states it gives.

On the Dell OptiPlex desktops the HDD cage ("HDD shield") has a latch of its
own that the sheets mostly never name. The ruling: if a cage exists, its latch
exists with it -- adopted from the sheet when the sheet opens one and never
removes it (D13), kept when S1 added one (D5), else implied (D6): closed, open
on the frame before the cage comes out, out with the cage.

Synthetic scenes here; the import fixtures are in ``test_u20_import.py``.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from tda.core.db import Db
from tda.core.implied import (
    CAGE_LATCH_RULE,
    IMPLIED_ATTR,
    IMPLIED_RULE_ATTR,
    NOTE_ATTR,
    OP_DELETE_IMPLIED,
    cage_latch_note,
    cage_latches,
    is_cage_latch,
    is_implied,
)
from tda.core.model import ActionRec, InstanceRec, StepRec
from tda.core.state_assert import StateAssertError, assert_range, assert_state
from tda.core.states import (
    CAGE_LATCH_EVIDENCE,
    cage_latch_opens,
    event_problems,
    events_from_actions,
    gone_with_parent,
    is_cage_latch_event,
    state_at,
    validate_events,
)
from tda.core.taxonomy import load_taxonomy
from tda.core.truth_inputs import events_of, state_of

D = 6
CAGE = "drive_cage.01"
LOCKER = "drive_latch.01"     # D5/D6's separate "HDD shield locker"
DRIVE = "storage_drive.hdd.01"
LATCH = "drive_latch.02"      # the cage's own, implied


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def inst(key: str, cls: str, **kw) -> InstanceRec:
    return InstanceRec(key=key, desktop=D, cls=cls, **kw)


def d6_instances(locker_of: str = "hdd") -> dict[str, InstanceRec]:
    """D6's drive bay as the sheet names it: the locker, the drive, the cage."""
    return {
        "chassis": inst("chassis", "chassis"),
        LOCKER: inst(LOCKER, "drive_latch", attrs={"of": locker_of}),
        DRIVE: inst(DRIVE, "storage_drive", attrs={"kind": "hdd"}),
        CAGE: inst(CAGE, "drive_cage", attrs={"of": "hdd"}),
    }


def d6_actions(locker_verb: str = "remove") -> list[ActionRec]:
    """Locker at 10 (taken out, as Shift+T said on D5/D6), drive at 13, cage at 14."""
    return [
        ActionRec(D, 10, 0, LOCKER, locker_verb),
        ActionRec(D, 13, 0, DRIVE, "remove"),
        ActionRec(D, 14, 0, CAGE, "remove"),
    ]


def with_plan(instances: dict, actions: list, tax, **kw) -> dict:
    """``instances`` with :func:`cage_latches` carried out, as the importer does."""
    plan = cage_latches(instances, actions, tax, **kw)
    out = {k: replace(r, attrs=dict(r.attrs)) for k, r in instances.items()}
    for e in plan.adopted:
        out[e["latch"]] = replace(out[e["latch"]], parent=e["cage"], attached=True)
    for rec in plan.created:
        out[rec.key] = rec
    return out


# --------------------------------------------------------------------------- #
# the three cases
# --------------------------------------------------------------------------- #
def test_d6_gets_an_implied_latch_on_its_cage(tax):
    plan = cage_latches(d6_instances(), d6_actions(), tax)
    assert not plan.adopted and not plan.kept
    [rec] = plan.created
    assert rec.key == LATCH and rec.cls == "drive_latch" and rec.desktop == D
    assert (rec.parent, rec.attached) == (CAGE, True)
    assert rec.attrs == {IMPLIED_ATTR: True, IMPLIED_RULE_ATTR: CAGE_LATCH_RULE, "of": "hdd",
                         NOTE_ATTR: "drive_cage.01 自带的卡扣（日志里没有）"}
    assert is_implied(rec) and is_cage_latch(rec)
    assert plan.opens == {LATCH: 13}
    assert plan.implied_lines() == [
        "implied instance drive_latch.02: drive_cage.01 自带的卡扣（日志里没有） - closed, "
        "open at step 13, out with drive_cage.01 at 14"]


def test_an_opened_never_removed_latch_is_adopted_d13(tax):
    plan = cage_latches(d6_instances(), d6_actions("open"), tax)
    assert plan.adopted == [{"latch": LOCKER, "cage": CAGE, "opened": 10}]
    assert not plan.created
    assert plan.lines() == [
        "cage latch adopted: drive_latch.01 is drive_cage.01's own latch (opened at step "
        "10, never removed on its own) - parent drive_cage.01, attached"]


def test_a_latch_added_in_s1_on_the_cage_is_kept_d5(tax):
    added = inst(LATCH, "drive_latch", attrs={"added_by": "chang", "of": "hdd"},
                 parent=CAGE, attached=True)
    plan = cage_latches(d6_instances(), d6_actions(), tax, extras={LATCH: added})
    assert plan.kept == [{"latch": LATCH, "cage": CAGE, "how": "added in S1"}]
    assert not plan.created and not plan.adopted
    assert plan.lines() == ["cage latch kept: drive_latch.02 (added in S1) is "
                            "drive_cage.01's own latch"]


def test_an_added_latch_elsewhere_does_not_count_and_its_key_is_skipped(tax):
    other = inst(LATCH, "drive_latch", attrs={"added_by": "chang"})   # no parent
    plan = cage_latches(d6_instances(), d6_actions(), tax, extras={LATCH: other})
    assert [rec.key for rec in plan.created] == ["drive_latch.03"]


@pytest.mark.parametrize("of", ["ssd", "drive_cage.02"])
def test_a_latch_of_something_else_is_not_adopted(tax, of):
    plan = cage_latches(d6_instances(locker_of=of), d6_actions("open"), tax)
    assert not plan.adopted and [rec.key for rec in plan.created] == [LATCH]


def test_a_latch_with_no_of_is_adopted_and_one_naming_the_cage_preferred(tax):
    instances = d6_instances(locker_of="")
    instances["drive_latch.03"] = inst("drive_latch.03", "drive_latch",
                                       attrs={"of": CAGE})
    actions = d6_actions("open") + [ActionRec(D, 5, 0, "drive_latch.03", "open")]
    plan = cage_latches(instances, actions, tax)
    assert [e["latch"] for e in plan.adopted] == ["drive_latch.03"]
    # alone, the one with no `of` would do
    plan = cage_latches(d6_instances(locker_of=""), d6_actions("open"), tax)
    assert [e["latch"] for e in plan.adopted] == [LOCKER]


def test_a_latch_opened_after_the_cage_left_is_not_its_latch(tax):
    actions = [ActionRec(D, 13, 0, DRIVE, "remove"), ActionRec(D, 14, 0, CAGE, "remove"),
               ActionRec(D, 15, 0, LOCKER, "open")]
    plan = cage_latches(d6_instances(), actions, tax)
    assert not plan.adopted and [rec.key for rec in plan.created] == [LATCH]


# --------------------------------------------------------------------------- #
# the states it gives
# --------------------------------------------------------------------------- #
def test_closed_then_open_at_c_minus_1_then_out_with_the_cage(tax):
    instances = with_plan(d6_instances(), d6_actions(), tax)
    events = events_from_actions(instances, d6_actions(), tax)
    mine = [e for e in events if e.target == LATCH]
    assert [(e.step, e.attr, e.old, e.new, e.auto) for e in mine] == [
        (13, "state", "closed", "open", True),
        (14, "state", "open", "removed", True),
        (14, "placement", "in_chassis", "on_bench", True),
    ]
    assert is_cage_latch_event(mine[0]) and mine[0].evidence_view == CAGE_LATCH_EVIDENCE
    assert not any(is_cage_latch_event(e) for e in mine[1:])
    for step, want in ((0, "closed"), (1, "closed"), (12, "closed"), (13, "open"),
                       (14, "removed"), (40, "removed")):
        assert state_at(instances, events, step, tax)[LATCH].state == want, step
    assert gone_with_parent(state_at(instances, events, 14, tax), LATCH)
    assert not validate_events(instances, events, tax)
    assert cage_latch_opens(instances, d6_actions(), tax) == [(13, LATCH)]


def test_the_open_event_follows_the_actions_of_its_step(tax):
    instances = with_plan(d6_instances(), d6_actions(), tax)
    events = events_from_actions(instances, d6_actions(), tax)
    at13 = [e.target for e in events if e.step == 13]
    assert at13.index(DRIVE) < at13.index(LATCH)


@pytest.mark.parametrize("cage_verb", [None, "displace", "failed"])
def test_a_cage_the_log_never_removes_gets_no_latch(tax, cage_verb):
    """It never comes back, which is the one frame the ruling asks for its latch on."""
    tail = []
    if cage_verb == "failed":
        tail = [ActionRec(D, 14, 0, CAGE, "remove", result="failed")]
    elif cage_verb:
        tail = [ActionRec(D, 14, 0, CAGE, cage_verb)]
    actions = d6_actions("open")[:2] + tail
    plan = cage_latches(d6_instances(), actions, tax)
    assert not plan          # nothing adopted either: the opened locker stays the log's
    plan = cage_latches(d6_instances(), actions, tax, deleted=_deleted())
    assert plan.unmatched[0]["why"] == "the log no longer removes drive_cage.01"


@pytest.mark.parametrize("cage_verb", [None, "displace"])
def test_a_latch_whose_cage_stops_being_removed_stays_closed(tax, cage_verb):
    """An implied latch kept after the cage's removal was edited out in Steps."""
    instances = with_plan(d6_instances(), d6_actions(), tax)
    actions = d6_actions()[:2] + ([ActionRec(D, 14, 0, CAGE, cage_verb)] if cage_verb else [])
    events = events_from_actions(instances, actions, tax)
    assert not [e for e in events if e.target == LATCH]
    assert cage_latch_opens(instances, actions, tax) == []
    for step in (1, 13, 14, 30):
        held = state_at(instances, events, step, tax)[LATCH]
        assert (held.state, held.placement) == ("closed", "in_chassis")


def test_two_cages_get_one_latch_each_in_removal_order(tax):
    instances = d6_instances()
    instances["drive_cage.02"] = inst("drive_cage.02", "drive_cage", attrs={"of": "optical"})
    actions = d6_actions() + [ActionRec(D, 8, 0, "drive_cage.02", "remove")]
    plan = cage_latches(instances, actions, tax)
    assert [(r.key, r.parent, r.attrs["of"]) for r in plan.created] == [
        ("drive_latch.02", "drive_cage.02", "optical"), ("drive_latch.03", CAGE, "hdd")]
    assert plan.opens == {"drive_latch.02": 7, "drive_latch.03": 13}
    done = with_plan(instances, actions, tax)
    events = events_from_actions(done, actions, tax)
    assert state_at(done, events, 7, tax)["drive_latch.02"].state == "open"
    assert state_at(done, events, 7, tax)["drive_latch.03"].state == "closed"
    assert state_at(done, events, 13, tax)["drive_latch.03"].state == "open"


def test_two_cages_and_one_opened_latch_adopt_once(tax):
    instances = d6_instances()
    instances["drive_cage.02"] = inst("drive_cage.02", "drive_cage", attrs={"of": "hdd"})
    actions = d6_actions("open") + [ActionRec(D, 20, 0, "drive_cage.02", "remove")]
    plan = cage_latches(instances, actions, tax)
    assert plan.adopted == [{"latch": LOCKER, "cage": CAGE, "opened": 10}]
    assert [(r.key, r.parent) for r in plan.created] == [("drive_latch.02", "drive_cage.02")]


def test_the_rule_is_idempotent_on_its_own_result(tax):
    first = with_plan(d6_instances(), d6_actions(), tax)
    again = cage_latches(first, d6_actions(), tax)
    assert not again.created
    assert again.kept == [{"latch": LATCH, "cage": CAGE, "how": "implied"}]
    adopted = with_plan(d6_instances(), d6_actions("open"), tax)
    again = cage_latches(adopted, d6_actions("open"), tax)
    assert again.adopted == [{"latch": LOCKER, "cage": CAGE, "opened": 10}]
    assert not again.created


def test_the_key_it_had_comes_back_and_a_kept_key_is_never_reused(tax):
    stored = with_plan(d6_instances(), d6_actions(), tax)
    stored["drive_latch.05"] = replace(stored[LATCH], key="drive_latch.05")   # an older copy
    del stored[LATCH]
    plan = cage_latches(d6_instances(), d6_actions(), tax, stored=stored)
    # made again under its key, and said to be kept, not created (review N3)
    assert not plan.created and [rec.key for rec in plan.carried] == ["drive_latch.05"]
    assert [rec.key for rec in plan.made] == ["drive_latch.05"]
    assert plan.kept == [{"latch": "drive_latch.05", "cage": CAGE, "how": "implied"}]
    assert plan.opens == {"drive_latch.05": 13} and not plan.implied_lines()
    assert plan.lines() == ["cage latch kept: drive_latch.05 (implied) is drive_cage.01's "
                            "own latch"]
    # a stored drive_latch key that is not this cage's latch is skipped over
    held = {"drive_latch.02": inst("drive_latch.02", "drive_latch")}
    plan = cage_latches(d6_instances(), d6_actions(), tax, stored=held)
    assert [rec.key for rec in plan.created] == ["drive_latch.03"]


def test_drafts_and_context_objects_are_never_cages(tax):
    instances = {"ls:HDD shield#1": inst("ls:HDD shield#1", "drive_cage"),
                 "ctx.drive_cage.01": inst("ctx.drive_cage.01", "drive_cage",
                                           attrs={"context": True})}
    assert not cage_latches(instances, [], tax)


# --------------------------------------------------------------------------- #
# a deletion is remembered per cage
# --------------------------------------------------------------------------- #
def _deleted(cage: str = CAGE, latch: str = LATCH, op: int = 7) -> list[dict]:
    return [{"id": op, "payload": {"instance": latch, "rule": CAGE_LATCH_RULE,
                                   "cage": cage}}]


def test_a_latch_deleted_in_steps_is_not_made_again(tax):
    plan = cage_latches(d6_instances(), d6_actions(), tax, deleted=_deleted())
    assert not plan.created
    assert plan.kept_deleted == [{"latch": LATCH, "cage": CAGE, "ops": [7]}]
    assert plan.lines() == ["cage latch kept deleted: drive_latch.02 of drive_cage.01 was "
                            "deleted in Steps - not made again"]


def test_a_deletion_follows_a_merged_cage_and_reports_what_it_no_longer_fits(tax):
    plan = cage_latches(d6_instances(), d6_actions(), tax,
                        deleted=_deleted(cage="drive_cage.09"),
                        rename={"drive_cage.09": CAGE})
    assert plan.kept_deleted and not plan.created
    plan = cage_latches(d6_instances(), d6_actions(), tax, deleted=_deleted(cage="drive_cage.09"))
    assert [rec.key for rec in plan.created] == [LATCH]
    assert plan.unmatched == [{"latch": LATCH, "cage": "drive_cage.09", "ops": [7],
                               "why": "the sheet has no drive cage drive_cage.09"}]
    plan = cage_latches(d6_instances(), d6_actions("open"), tax, deleted=_deleted())
    assert plan.unmatched[0]["why"] == "drive_cage.01 has its latch drive_latch.01 now"
    assert plan.lines()[-1] == ("cage latch delete not matched: drive_latch.02 of "
                                "drive_cage.01 (drive_cage.01 has its latch drive_latch.01 now)")


def test_only_cage_latch_deletions_count(tax):
    other = [{"id": 3, "payload": {"instance": "motherboard.01", "rule": None}}]
    plan = cage_latches(d6_instances(), d6_actions(), tax, deleted=other)
    assert [rec.key for rec in plan.created] == [LATCH] and not plan.unmatched


# --------------------------------------------------------------------------- #
# a manual T / Ctrl+T wins over the default
# --------------------------------------------------------------------------- #
@pytest.fixture
def d6db(tmp_db_path, tax):
    db = Db(tmp_db_path)
    steps = [StepRec(D, s, "normal" if s > 1 else "initial", f"step {s}")
             for s in range(1, 16)]
    db.replace_steps(D, steps, d6_actions())
    for rec in with_plan(d6_instances(), d6_actions(), tax).values():
        db.upsert_instance(rec)
    yield db
    db.close()


def _run(db, tax) -> dict[int, str]:
    return {s: state_of(db, tax, D, s)[LATCH].state for s in range(1, 16)}


def test_t_on_an_earlier_frame_says_it_was_open_already(d6db, tax):
    patch = assert_state(d6db, tax, D, LATCH, 12, "open")
    assert patch.changed and patch.next_step == 13
    assert [(e.step, e.old, e.new) for e in patch.after] == [(12, "closed", "open")]
    run = _run(d6db, tax)
    assert [run[s] for s in (11, 12, 13, 14)] == ["closed", "open", "open", "removed"]
    events = events_of(d6db, tax, D)
    assert not event_problems(d6db.instances(D), events, tax)


def test_ctrl_t_from_the_frame_it_was_first_open_up_to_the_default(d6db, tax):
    assert_range(d6db, tax, D, LATCH, 9, 12, "open")
    run = _run(d6db, tax)
    assert [run[s] for s in (8, 9, 12, 13, 14)] == ["closed", "open", "open", "open",
                                                     "removed"]
    # ... and over it: the default changes nothing then
    assert_range(d6db, tax, D, LATCH, 5, 13, "open")
    run = _run(d6db, tax)
    assert [run[s] for s in (4, 5, 13, 14)] == ["closed", "open", "open", "removed"]
    assert not validate_events(d6db.instances(D), events_of(d6db, tax, D), tax)


def test_t_on_the_default_frame_says_it_was_still_closed(d6db, tax):
    assert_state(d6db, tax, D, LATCH, 13, "closed")
    assert _run(d6db, tax)[13] == "closed" and _run(d6db, tax)[14] == "removed"
    assert not validate_events(d6db.instances(D), events_of(d6db, tax, D), tax)


def test_a_closed_span_over_the_default_says_where_to_press_t(d6db, tax):
    with pytest.raises(StateAssertError, match="机架卡扣 drive_latch.02 默认在那一帧是「open」"):
        assert_range(d6db, tax, D, LATCH, 10, 13, "closed")


def test_a_log_action_still_refuses_a_span_across_it(tmp_db_path, tax):
    """The exemption is the default's alone: an adopted latch's own ``open`` refuses."""
    db = Db(tmp_db_path)
    try:
        steps = [StepRec(D, s, "normal", f"step {s}") for s in range(1, 16)]
        db.replace_steps(D, steps, d6_actions("open"))
        for rec in with_plan(d6_instances(), d6_actions("open"), tax).values():
            db.upsert_instance(rec)
        assert db.instances(D)[LOCKER].parent == CAGE
        with pytest.raises(StateAssertError, match="日志第 10 步"):
            assert_range(db, tax, D, LOCKER, 5, 12, "open")
        with pytest.raises(StateAssertError, match="动作发生在哪一步"):
            assert_state(db, tax, D, LOCKER, 9, "open")
    finally:
        db.close()


def test_the_delete_op_kind_is_named_once():
    assert OP_DELETE_IMPLIED == "delete_implied"
    assert cage_latch_note(CAGE) == "drive_cage.01 自带的卡扣（日志里没有）"
