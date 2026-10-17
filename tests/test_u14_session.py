"""Task U14 (core + session): 「其实拧下来了」 / 「其实留在零件里」, undoably.

The D13 scene (steps 1-14) is a Dell: the importer marks its four CPU-cooler
screws captive, the log unscrews them at 3-6 (``loosened``, in the chassis) and
takes the cooler out at 13, and they go with it. ``Shift+T`` on one of them
says it was taken out when it was unscrewed (D3's fan screws): its ``captive``
flag flips through the step table's own path, the derived log follows, and
every confirmed frame its truth changes on -- in every view -- is re-checked.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from session_scene import COOLER, DESKTOP, SCREWS, VIEW, make_session, seed_shapes
from tda.core.cache import VIEW_EXT, cache_path
from tda.core.captive_fix import (
    OP_CAPTIVE_FIX,
    STAYS_IN,
    UNSCREWED,
    CaptiveFixError,
    captive_offers,
    plan_captive_fix,
)
from tda.core.model import FrameKey
from tda.core.states import validate_events
from tda.core.truth_inputs import events_of, instances_of, state_of
from tda.ui.session_api import SessionRefusal
from tda.ui.steps_model import StepTableData

SCREW = SCREWS[0]          # unscrewed at 3
SECOND = SCREWS[1]         # unscrewed at 4
BOARD_SCREW = "screw.motherboard.01"   # the scene stops before it is unscrewed
LAST = 14


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(qapp, tmp_path):
    from tda.core.extra import recompile_events

    made = make_session(tmp_path, last_step=LAST)
    made.sweeper_enabled = False       # the re-check queue is inspected, not drained
    with made.db.transaction():        # the stored derived log, as an import leaves it
        recompile_events(made.db, DESKTOP, made.tax)
    seed_shapes(made, LAST, anchor=LAST)
    yield made
    made.close(force=True)


def held(session, step: int, target: str = SCREW) -> tuple:
    inst = state_of(session.db, session.tax, DESKTOP, step)[target]
    return inst.state, inst.placement


def states(session, target: str = SCREW) -> list[tuple]:
    return [held(session, step, target) for step in range(1, LAST + 1)]


def record(session, target: str = SCREW) -> dict:
    rec = session.db.instances(DESKTOP)[target]
    return {"captive": rec.attrs.get("captive"), "captive_source": rec.attrs.get("captive_source"),
            "parent": rec.parent, "attached": rec.attached}


def snapshot(session) -> tuple:
    """Everything a flip may change: the instance table, the steps and both event logs."""
    db = session.db
    instances = {k: asdict(v) for k, v in db.instances(DESKTOP).items()}
    actions = [(a.step, a.idx, a.target, a.verb, a.result) for a in db.actions(DESKTOP)]
    stored = sorted((e.step, e.target, e.attr, e.old, e.new, e.auto) for e in db.events(DESKTOP))
    merged = [(e.step, e.target, e.attr, e.old, e.new, e.auto)
              for e in events_of(db, session.tax, DESKTOP)]
    return instances, actions, stored, merged


LOOSE = ("loosened", "in_chassis")
OUT = ("removed", "on_bench")
FASTENED = ("fastened", "in_chassis")


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_the_screws_the_log_unscrews_are_offered_by_their_flag(session):
    assert session.captive_offers() == {key: UNSCREWED for key in SCREWS}
    # the covers and latches keep their own offer (U9), untouched
    assert session.verb_offers() == {"cover.01": "remove", "drive_latch.01": "remove"}
    session.goto(8)
    rows = {r["key"]: r for r in session.instance_rows()}
    assert rows[SCREW]["verb_fix"] == UNSCREWED
    assert rows[BOARD_SCREW]["verb_fix"] is None          # never unscrewed in the scene


def test_the_pure_gate():
    from tda.core.model import ActionRec, InstanceRec

    inst = {"s.1": InstanceRec("s.1", 1, "screw", attrs={"captive": True}),
            "s.2": InstanceRec("s.2", 1, "screw", attrs={"captive": False}),
            "s.3": InstanceRec("s.3", 1, "screw"),
            "s.4": InstanceRec("s.4", 1, "screw", attrs={"captive": True}),
            "c.1": InstanceRec("c.1", 1, "cover"),
            "ls:screw.1": InstanceRec("ls:screw.1", 1, "screw")}
    acts = [ActionRec(1, 2, 0, "s.1", "unscrew"), ActionRec(1, 3, 0, "s.2", "unscrew"),
            ActionRec(1, 4, 0, "s.3", "unscrew"), ActionRec(1, 5, 0, "s.4", "unscrew",
                                                            result="failed"),
            ActionRec(1, 6, 0, "c.1", "open"), ActionRec(1, 7, 0, "ls:screw.1", "unscrew")]
    assert captive_offers(inst, acts) == {"s.1": UNSCREWED, "s.2": STAYS_IN, "s.3": STAYS_IN}


# --------------------------------------------------------------------------- #
# 其实拧下来了
# --------------------------------------------------------------------------- #
def test_unscrewed_takes_it_out_at_its_unscrew_and_logs_the_inverse(session):
    before = states(session)
    assert before[1:3] == [FASTENED, LOOSE] and before[11] == LOOSE and before[12] == OUT
    session.goto(8)
    result = session.correct_captive(SCREW)
    fix = result["fix"]
    assert (fix.step, fix.old, fix.new, fix.at, fix.direction) == (3, True, False, 8, UNSCREWED)
    assert fix.parent == COOLER
    # out of the machine from its unscrew on, every step
    assert states(session)[:2] == before[:2]
    assert states(session)[2:] == [OUT] * (LAST - 2)
    assert record(session) == {"captive": False, "captive_source": "manual",
                               "parent": COOLER, "attached": False}
    assert result["reach"] == {VIEW: list(range(3, LAST + 1))}
    assert result["affected"] == list(range(3, LAST + 1)) and result["compiled"] == [8]
    op = session.db.ops(DESKTOP, VIEW, limit=1)[0]
    assert op["kind"] == OP_CAPTIVE_FIX == session.undo_stack.ops[-1].kind
    payload, inverse = op["payload"], op["inverse"]
    assert (payload["captive"], payload["captive_source"], payload["attached"]) == \
        (False, "manual", False)
    assert (inverse["captive"], inverse["captive_source"], inverse["attached"]) == \
        (True, "heuristic", True)
    assert payload["target"] == inverse["target"] == SCREW
    assert payload["step"] == 3 and payload["reach"] == result["reach"]
    # the stored derived log is recompiled: the unscrew at 3 now removes it
    stored = {(e.step, e.attr, e.old, e.new) for e in session.db.events(DESKTOP)
              if e.auto and e.target == SCREW}
    assert (3, "state", "fastened", "removed") in stored
    assert not any(e[0] == 13 for e in stored)               # not with the cooler any more
    assert not validate_events(instances_of(session.db, DESKTOP),
                               events_of(session.db, session.tax, DESKTOP), session.tax)
    assert SCREW not in {r["key"] for r in session.instance_rows()}
    assert session.dirty


def test_undo_and_redo_are_table_identical(session):
    session.goto(8)
    before = snapshot(session)
    session.correct_captive(SCREW)
    session.correct_captive(SECOND)
    after = snapshot(session)
    assert after != before
    assert session.undo() and session.undo()
    assert snapshot(session) == before
    assert states(session) == [FASTENED, FASTENED] + [LOOSE] * 10 + [OUT] * 2
    assert session.redo() and session.redo()
    assert snapshot(session) == after
    assert states(session, SECOND)[3:] == [OUT] * (LAST - 3)


# --------------------------------------------------------------------------- #
# 其实留在零件里
# --------------------------------------------------------------------------- #
def test_stays_in_puts_it_back_in_its_part(session):
    session.goto(8)
    session.correct_captive(SCREW)
    session.goto(LAST)
    assert session.captive_offers()[SCREW] == STAYS_IN
    result = session.correct_captive(SCREW)
    fix = result["fix"]
    assert (fix.step, fix.new, fix.direction, fix.parent) == (3, True, STAYS_IN, COOLER)
    assert record(session) == {"captive": True, "captive_source": "manual",
                               "parent": COOLER, "attached": True}
    assert states(session)[2:12] == [LOOSE] * 10 and states(session)[12:] == [OUT] * 2
    inst = state_of(session.db, session.tax, DESKTOP, LAST)[SCREW]
    assert inst.left_with == COOLER                           # out inside the cooler


def test_stays_in_takes_the_part_it_fastens_when_it_has_no_parent(session):
    db = session.db
    rec = db.instances(DESKTOP)[SECOND]
    rec.attrs["captive"], rec.parent, rec.attached = False, None, False
    db.upsert_instance(rec)
    StepTableData.load(db, DESKTOP, session.tax).save(db)    # the log follows
    assert held(session, 6, SECOND) == OUT
    before = snapshot(session)
    session.goto(10)
    result = session.correct_captive(SECOND, STAYS_IN)
    assert result["fix"].filled and result["fix"].parent == COOLER
    assert record(session, SECOND) == {"captive": True, "captive_source": "manual",
                                       "parent": COOLER, "attached": True}
    assert held(session, 6, SECOND) == LOOSE
    assert session.undo()
    assert snapshot(session) == before
    assert record(session, SECOND)["parent"] is None


def test_stays_in_needs_a_part_to_stay_in(session):
    db = session.db
    rec = db.instances(DESKTOP)[SECOND]
    rec.attrs["captive"], rec.parent, rec.attached, rec.fastens = False, None, False, None
    db.upsert_instance(rec)
    with pytest.raises(SessionRefusal, match="没写拧在哪个零件上"):
        session.correct_captive(SECOND)
    rec.fastens = "ghost.part.01"                         # names no part of this machine
    db.upsert_instance(rec)
    with pytest.raises(SessionRefusal, match="fastens 写的是「ghost.part.01」，这台机器上找不到"):
        session.correct_captive(SECOND)
    assert not db.ops_of_kind(DESKTOP, OP_CAPTIVE_FIX)


# --------------------------------------------------------------------------- #
# confirmed frames, every view
# --------------------------------------------------------------------------- #
def _oak1_frames(session, steps) -> None:
    """Frame rows (and images) for oak1, so the desktop has a second view."""
    db = session.db
    for step in steps:
        key = FrameKey(DESKTOP, step, "oak1")
        path = cache_path(str(session.cache_dir), key, VIEW_EXT["oak1"])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, np.full((64, 64, 3), 40 + step, dtype=np.uint8))
        db.upsert_frame(key, path, {"cache_path": path, "hw": [64, 64]}, None)


def test_confirmed_frames_are_re_checked_in_every_view_and_a_drawn_one_conflicts(session):
    db = session.db
    seed_shapes(session, 12, anchor=12)              # the screws drawn, loosened, at 12
    session.goto(12)
    assert session.confirm_frame() is True
    _oak1_frames(session, range(1, LAST + 1))
    db.conn.execute("UPDATE frame SET review_status='verified' WHERE desktop=? AND view=? "
                    "AND step IN (2, 9)", (DESKTOP, "oak1"))
    db.conn.commit()
    session.goto(10)
    result = session.correct_captive(SCREW)
    assert result["reach"]["oak1"] == list(range(3, LAST + 1))
    assert result["frozen"] == {VIEW: [12], "oak1": [9]}     # oak1 2 is before the unscrew
    assert result["queued"] == {"oak1": [9]}
    assert 9 in db.rechecks(DESKTOP, "oak1") and 2 not in db.rechecks(DESKTOP, "oak1")
    assert 12 in result["rechecks"] and 12 in db.rechecks(DESKTOP, VIEW)
    # 12 was confirmed with the screw drawn in the chassis: it is out there now
    assert ("scan", 12) in result["clashes"]
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    opened = [c for c in session.truth.open_conflicts(DESKTOP, VIEW)
              if c["instance"] == SCREW]
    assert [c["step"] for c in opened] == [12]
    # an undo re-checks them again (12 was drained above)
    assert 12 not in db.rechecks(DESKTOP, VIEW)
    assert session.undo()
    assert 12 in db.rechecks(DESKTOP, VIEW) and 9 in db.rechecks(DESKTOP, "oak1")
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    assert not [c for c in session.truth.open_conflicts(DESKTOP, VIEW)
                if c["instance"] == SCREW]                   # it agrees again: closed


def test_a_second_flip_says_it_closes_the_conflict_the_first_one_opened(session):
    """Shift+T again rather than Ctrl+Z: the signed row matches again (review F3)."""
    seed_shapes(session, 12, anchor=12)
    session.goto(12)
    assert session.confirm_frame() is True
    session.goto(10)
    first = session.correct_captive(SCREW)
    assert first["clashes"] == [("scan", 12)] and first["closes"] == []
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    assert [c["step"] for c in session.truth.open_conflicts(DESKTOP, VIEW)
            if c["instance"] == SCREW] == [12]
    second = session.correct_captive(SCREW)              # back in, not an undo
    assert second["fix"].new is True
    assert second["clashes"] == [] and second["closes"] == [("scan", 12)]
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    assert not [c for c in session.truth.open_conflicts(DESKTOP, VIEW)
                if c["instance"] == SCREW]


def test_a_confirmed_frame_that_now_lacks_it_is_owed(session):
    """Not captive, drawn nowhere: confirmed at 12 without it; captive, it is due there."""
    db = session.db
    session.goto(8)
    session.correct_captive(SCREW)
    seed_shapes(session, 12, anchor=12)               # everything there but the screw
    session.goto(12)
    assert SCREW not in session.compiled().instances
    assert session.confirm_frame() is True
    session.goto(10)
    result = session.correct_captive(SCREW)
    assert result["frozen"] == {VIEW: [12]} and result["clashes"] == []
    assert result["owed"] == [("scan", 12)]


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_refusals_say_why(session):
    with pytest.raises(SessionRefusal, match="不是螺丝"):
        session.correct_captive("cover.01")
    with pytest.raises(SessionRefusal, match="没有拧下 screw.motherboard.01 的动作"):
        session.correct_captive(BOARD_SCREW)
    with pytest.raises(SessionRefusal, match="已经是「留在零件里」了"):
        session.correct_captive(SCREW, STAYS_IN)
    with pytest.raises(SessionRefusal, match="没有零件"):
        session.correct_captive("screw.cpu_cooler.99")
    assert not session.db.ops_of_kind(DESKTOP, OP_CAPTIVE_FIX)
    assert session.captive_gate(BOARD_SCREW).startswith("日志里没有拧下")
    assert session.captive_gate(SCREW) == ""


def test_a_hand_written_state_after_the_unscrew_has_to_go_first(session):
    """T fastened it again at 10; a screw taken out at 3 cannot be."""
    session.goto(10)
    session.assert_state(SCREW, "fastened")
    with pytest.raises(SessionRefusal, match="第 10 步有手动设的状态（T：拧松 → 拧紧），拧下拿走以后"):
        session.correct_captive(SCREW)
    assert record(session)["captive"] is True


def test_a_screw_with_pixels_not_committed_is_refused_and_an_untouched_edit_dropped(session):
    session.goto(10)
    session.begin_edit(SCREW)
    mask = session.editing_mask().copy()
    mask[0:3, 0:3] = ~mask[0:3, 0:3]
    session.set_editing_mask(mask)
    with pytest.raises(SessionRefusal, match="正在画、还没提交"):
        session.correct_captive(SCREW)
    assert record(session)["captive"] is True
    session.clear_edit()
    session.begin_edit(SCREW)
    session.correct_captive(SCREW)
    assert session.editing_instance is None


def test_the_core_refuses_without_writing(session):
    db, tax = session.db, session.tax
    with pytest.raises(CaptiveFixError, match="没有零件"):
        plan_captive_fix(db, tax, DESKTOP, "screw.99", LAST)
    fix = plan_captive_fix(db, tax, DESKTOP, SECOND, LAST)
    assert (fix.step, fix.new) == (4, False)
    assert record(session, SECOND)["captive"] is True


def test_a_gallery_shot_offers_nothing_and_refuses(qapp, tmp_path):
    import gallery_scene as GS
    from tda.ui.gallery import NO_VERB

    made = GS.make_session(tmp_path)
    try:
        assert made.captive_offers() == {}
        with pytest.raises(SessionRefusal, match="零件合照不改步骤表"):
            made.correct_captive(SCREW)
        assert made.captive_gate(SCREW) == NO_VERB
        # one sentence for covers and screws: no button names of either (review N5)
        assert "「其实…」" in NO_VERB and "其实拆下来了" not in NO_VERB
        assert not made.db.ops_of_kind(GS.DESKTOP, OP_CAPTIVE_FIX)
    finally:
        made.close(force=True)


# --------------------------------------------------------------------------- #
# the card
# --------------------------------------------------------------------------- #
def test_the_card_marks_a_loosened_captive_screw(session):
    session.goto(12)                       # the cooler and its screws are back in
    marked = {r["instance"] for r in session.task_card() if r.get("verb_fix") == UNSCREWED}
    assert marked == set(SCREWS)
    session.goto(LAST)
    assert not [r for r in session.task_card() if r.get("verb_fix") == UNSCREWED]
    session.goto(12)
    session.correct_captive(SCREW)
    marked = {r["instance"] for r in session.task_card() if r.get("verb_fix") == UNSCREWED}
    assert SCREW not in marked
