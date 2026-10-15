"""Task U9 (core + session): 「其实拆下来了」 / 「其实只是打开了」, undoably.

The D13 scene (steps 1-14): ``cover.01`` ("CPU Fan Cover") is opened at step
2 and ``drive_latch.01`` ("SSD shield locker") at step 7 -- the two removable
classes the scene has. The correction rewrites the verb of the action that
opened the part through the step table's own path, logs it with its inverse,
and settles every frame of the view from that step on.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from session_scene import DESKTOP, VIEW, make_session, seed_shapes
from tda.core.model import FrameKey
from tda.core.states import validate_events
from tda.core.truth_inputs import events_of, instances_of, state_of
from tda.core.verb_fix import OP_CORRECT_VERB, VerbFixError, offers, plan_verb_fix
from tda.ui.session_api import SessionRefusal
from tda.ui.steps_model import StepTableData

COVER = "cover.01"        # opened at 2
LATCH = "drive_latch.01"  # opened at 7
LAST = 14


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(qapp, tmp_path):
    made = make_session(tmp_path, last_step=LAST)
    made.sweeper_enabled = False       # the re-check queue is inspected, not drained
    seed_shapes(made, LAST, anchor=LAST, skip=(LATCH,))   # everything but the latch drawn
    yield made
    made.close(force=True)


def verbs(session, target: str) -> list[tuple]:
    return [(a.step, a.verb) for a in session.db.actions(DESKTOP) if a.target == target]


def snapshot(session) -> tuple:
    """Everything the correction may change: steps, actions, the merged event log.

    The merged log rather than the stored automatic rows: the scene was seeded
    without them (``events_of`` never reads them), and the S1 save writes them.
    """
    db = session.db
    actions = [(a.step, a.idx, a.target, a.verb, a.tool, a.direction, a.result,
                a.failure_reason, a.difficulty) for a in db.actions(DESKTOP)]
    steps = [(s.step, s.step_type, s.raw_name, s.notes) for s in db.steps(DESKTOP)]
    events = [(e.step, e.target, e.attr, e.old, e.new, e.auto)
              for e in events_of(db, session.tax, DESKTOP)]
    return steps, actions, events


def card(session) -> list[tuple]:
    return [(r["instance"], r["kind"], r.get("verb_fix")) for r in session.task_card()]


def held(session, step: int, target: str = COVER) -> str:
    return state_of(session.db, session.tax, DESKTOP, step)[target].state


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_the_open_covers_and_latches_are_offered_and_nothing_else(session):
    assert session.current().step == LAST
    assert session.verb_offers() == {COVER: "remove", LATCH: "remove"}
    rows = {r["key"]: r for r in session.instance_rows()}
    assert rows[COVER]["verb_fix"] == "remove" and rows[LATCH]["verb_fix"] == "remove"
    assert rows["ram_latch.01"]["verb_fix"] is None      # open, but moulded on
    assert rows["cover.02"]["verb_fix"] is None          # a cover, but closed
    session.goto(1)
    assert session.verb_offers() == {}                   # not opened yet


def test_offers_follow_the_parts_own_action():
    """The pure gate: the state must be what the part's own open/remove left."""
    from tda.core.model import ActionRec, InstanceRec
    from tda.core.states import InstState
    from tda.core.taxonomy import load_taxonomy

    tax = load_taxonomy()
    inst = {k: InstanceRec(key=k, desktop=1, cls=c) for k, c in
            (("cover.01", "cover"), ("psu_latch.01", "psu_latch"), ("ram_latch.01", "ram_latch"),
             ("cover.02", "cover"))}
    acts = [ActionRec(1, 3, 0, "cover.01", "open"), ActionRec(1, 4, 0, "psu_latch.01", "remove"),
            ActionRec(1, 5, 0, "ram_latch.01", "open")]
    state = {"cover.01": InstState("open", "in_chassis"),
             "psu_latch.01": InstState("removed", "on_bench"),
             "ram_latch.01": InstState("open", "in_chassis"),
             "cover.02": InstState("open", "in_chassis")}          # opened by hand (T)
    assert offers(tax, inst, state, acts, 6) == {"cover.01": "remove", "psu_latch.01": "open"}
    assert offers(tax, inst, state, acts, 3) == {"cover.01": "remove"}


# --------------------------------------------------------------------------- #
# A: 其实拆下来了
# --------------------------------------------------------------------------- #
def test_taken_out_rewrites_the_opening_action_and_logs_its_inverse(session):
    result = session.correct_verb(COVER)
    fix = result["fix"]
    assert (fix.step, fix.old, fix.new, fix.at) == (2, "open", "remove", LAST)
    assert verbs(session, COVER) == [(2, "remove")]
    assert result["affected"] == list(range(2, LAST + 1))
    assert result["lost"] == list(range(2, LAST + 1))      # drawn at 14, used from 2 on
    op = session.db.ops(DESKTOP, VIEW, limit=1)[0]
    assert op["kind"] == OP_CORRECT_VERB == session.undo_stack.ops[-1].kind
    payload, inverse = op["payload"], op["inverse"]
    assert (payload["verb"], inverse["verb"]) == ("remove", "open")
    assert payload["step"] == inverse["step"] == 2 and payload["target"] == COVER
    assert payload["view"] == VIEW and payload["reach"] == result["affected"]
    assert session.dirty
    # the part is out from 2 on: off the card and off the instance list
    assert held(session, 1) == "closed" and held(session, 2) == held(session, LAST) == "removed"
    assert COVER not in {i for i, _k, _f in card(session)}
    assert COVER not in {r["key"] for r in session.instance_rows()}
    assert not validate_events(instances_of(session.db, DESKTOP),
                               events_of(session.db, session.tax, DESKTOP), session.tax)


def test_undo_and_redo_are_exact(session):
    before, card_before = snapshot(session), card(session)
    session.correct_verb(COVER)
    session.correct_verb(LATCH)
    after = snapshot(session)
    assert session.undo() and session.undo()
    assert snapshot(session) == before and card(session) == card_before
    assert session.redo() and session.redo()
    assert snapshot(session) == after
    assert COVER not in {i for i, _k, _f in card(session)}


def test_a_frozen_frame_in_reach_is_queued_for_a_re_check(session):
    db = session.db
    session.goto(LAST)
    seed_shapes(session, LAST, anchor=LAST)          # the latch too: 14 can be confirmed
    assert session.confirm_frame() is True           # 14 frozen, now on 13
    assert session.current().step == LAST - 1
    result = session.correct_verb(COVER)
    assert result["compiled"] == [LAST - 1]
    assert LAST in result["rechecks"] and LAST in db.rechecks(DESKTOP, VIEW)


def test_refusals_say_why(session):
    with pytest.raises(SessionRefusal, match="只能打开、不会单独拆下"):
        session.correct_verb("ram_latch.01")
    with pytest.raises(SessionRefusal, match="不是能打开也能拆下的"):
        session.correct_verb("screw.motherboard.01")
    with pytest.raises(SessionRefusal, match="是「关上」"):
        session.correct_verb("cover.02")
    with pytest.raises(SessionRefusal, match="已经是「打开」了"):
        session.correct_verb(COVER, to="open")
    assert verbs(session, COVER) == [(2, "open")]
    assert not [o for o in session.db.ops(DESKTOP, VIEW, limit=20)
                if o["kind"] == OP_CORRECT_VERB]


def test_an_open_set_by_hand_points_to_t(session):
    """psu_latch.01 is never opened by the log; T says it is open at 14."""
    session.assert_state("psu_latch.01", "open")
    assert session.verb_offers().get("psu_latch.01") is None
    with pytest.raises(SessionRefusal, match="手动设的状态（T）"):
        session.correct_verb("psu_latch.01")


def test_a_hand_written_state_after_the_step_has_to_go_first(session):
    """T closed the cover at 10; a removal at 2 could not be followed by that."""
    session.goto(10)
    session.assert_state(COVER, "closed")
    session.goto(LAST)
    with pytest.raises(SessionRefusal, match="第 10 步有手动设的状态（T：打开 → 关上）"):
        session.correct_verb(COVER)
    assert verbs(session, COVER) == [(2, "open")]


def test_a_later_action_of_the_part_keeps_it_open(session):
    data = StepTableData.load(session.db, DESKTOP, session.tax)
    data.add_action(12, target=COVER, verb="remove")
    data.save(session.db)
    session.goto(10)
    with pytest.raises(SessionRefusal, match="日志第 12 步还有它的动作（remove）"):
        session.correct_verb(COVER)


def test_the_core_refuses_without_writing(session):
    db, tax = session.db, session.tax
    with pytest.raises(VerbFixError, match="没有零件"):
        plan_verb_fix(db, tax, DESKTOP, "cover.99", LAST)
    fix = plan_verb_fix(db, tax, DESKTOP, LATCH, LAST)
    assert (fix.step, fix.old, fix.new) == (7, "open", "remove")
    assert verbs(session, LATCH) == [(7, "open")]


def test_an_untouched_edit_of_the_part_is_dropped_and_a_touched_one_refuses(session):
    session.begin_edit(COVER)
    mask = session.editing_mask().copy()
    mask[0:3, 0:3] = ~mask[0:3, 0:3]
    session.set_editing_mask(mask)
    with pytest.raises(SessionRefusal, match="正在画、还没提交"):
        session.correct_verb(COVER)
    assert verbs(session, COVER) == [(2, "open")]
    session.clear_edit()
    session.begin_edit(COVER)
    session.correct_verb(COVER)
    assert session.editing_instance is None


# --------------------------------------------------------------------------- #
# B: 其实只是打开了
# --------------------------------------------------------------------------- #
def test_only_opened_puts_a_removed_part_back(session):
    data = StepTableData.load(session.db, DESKTOP, session.tax)
    data.apply_edit(7, "verb", "remove")              # the sheet said "remove ..."
    data.save(session.db)
    session.goto(LAST)
    assert session.verb_offers()[LATCH] == "open"
    assert LATCH not in {i for i, _k, _f in card(session)}
    result = session.correct_verb(LATCH)
    fix = result["fix"]
    assert (fix.step, fix.old, fix.new) == (7, "remove", "open")
    assert verbs(session, LATCH) == [(7, "open")]
    assert result["lost"] == [] and result["todo"] == list(range(7, LAST + 1))
    assert held(session, LAST, LATCH) == "open"
    assert LATCH in {i for i, _k, _f in card(session)}
    op = session.db.ops(DESKTOP, VIEW, limit=1)[0]
    assert (op["payload"]["verb"], op["inverse"]["verb"]) == ("open", "remove")
    assert session.undo()
    assert verbs(session, LATCH) == [(7, "remove")]


def test_a_part_that_left_with_its_parent_is_not_its_own_removal(session):
    db = session.db
    rec = db.instances(DESKTOP)[LATCH]
    rec.parent, rec.attached = "drive_cage.01", True      # out with the cage at 11
    db.upsert_instance(rec)
    data = StepTableData.load(db, DESKTOP, session.tax)
    data.save(db)                                          # the cascade goes in the log
    session.goto(LAST)
    assert held(session, LAST, LATCH) == "removed"
    assert LATCH not in session.verb_offers()
    with pytest.raises(SessionRefusal, match="跟着 drive_cage.01 一起拆下的"):
        session.correct_verb(LATCH)


def test_the_frame_in_hand_is_compiled_without_the_part(session):
    session.correct_verb(COVER)
    compiled = session.compiled()
    assert COVER not in compiled.instances
    assert FrameKey(DESKTOP, LAST, VIEW) == compiled.key


# --------------------------------------------------------------------------- #
# review round
# --------------------------------------------------------------------------- #
def test_an_added_part_open_from_the_start_points_to_steps_mode(session):
    """m1: T cannot say "removed"; an added part leaves by an action of its own."""
    data = StepTableData.load(session.db, DESKTOP, session.tax)
    [added] = data.add_extras("psu_latch", 1, state="open")
    data.save(session.db)
    session.goto(LAST)
    assert held(session, LAST, added) == "open"
    with pytest.raises(SessionRefusal, match="S1 添加的零件") as refused:
        session.correct_verb(added)
    assert "Steps 模式" in str(refused.value) and "remove" in str(refused.value)


def test_a_re_close_by_t_also_offers_steps_mode(session):
    """m2: the T correction after the step can go, or the removal can move."""
    session.goto(10)
    session.assert_state(COVER, "closed")
    session.goto(LAST)
    with pytest.raises(SessionRefusal, match="或在 Steps 模式里改拆下的那一步"):
        session.correct_verb(COVER)


def test_the_correction_leaves_the_graph_alone_and_keeps_its_annotator(session):
    """m4: no relations op from Shift+T; undo and redo carry who made it."""
    session.correct_verb(COVER)
    kinds = [r["kind"] for r in session.db.conn.execute(
        "SELECT kind FROM op_log WHERE desktop=? ORDER BY id", (DESKTOP,))]
    assert kinds == [OP_CORRECT_VERB]
    op = session.db.ops(DESKTOP, VIEW, limit=1)[0]
    assert op["payload"]["annotator"] == op["inverse"]["annotator"] == session.annotator
    assert op["annotator"] == session.annotator
    assert session.undo() and session.redo()
    assert verbs(session, COVER) == [(2, "remove")]
