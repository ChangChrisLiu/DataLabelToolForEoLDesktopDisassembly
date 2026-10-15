"""Task U11 (B): "from this frame to step N the part is in state s" -- ``Ctrl+T``.

The core first (:func:`tda.core.state_assert.assert_range`, on the D13 sheet up
to step 24: ``ram_latch.01`` is opened by the log at 14), then the session's
op: one undo, the cut keyframes following the span's lower edge, the refusal
across a derived event, and the dialog's defaults.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from session_scene import DESKTOP, VIEW, draw, make_session, rect, seed_db, seed_shapes
from tda.core import state_assert as SA
from tda.core.cut_follow import CUT
from tda.core.db import Db
from tda.core.states import state_at, validate_events
from tda.core.taxonomy import load_taxonomy
from tda.core.truth_inputs import events_of
from tda.ui import session_api as api
from tda.ui import session_edit as edit
from tda.ui.class_names import state_zh
from tda.ui.session_api import SessionRefusal

LATCH = "ram_latch.01"          # opened by the log at 14
LAST = 24


@pytest.fixture
def core(tmp_path):
    tax = load_taxonomy()
    db = Db(str(tmp_path / "tda.sqlite"))
    seed_db(db, tax, tmp_path / "cache", last_step=LAST)
    return db, tax


def manual(db: Db, target: str = LATCH) -> list[tuple]:
    return [(e.step, e.old, e.new) for e in db.events(DESKTOP)
            if not e.auto and e.target == target]


def states(db: Db, tax, steps, target: str = LATCH) -> dict[int, str]:
    instances = db.instances(DESKTOP)
    log = events_of(db, tax, DESKTOP)
    return {s: state_at(instances, log, s, tax)[target].state for s in steps}


def _dupli(db, step: int) -> None:
    steps = db.steps(DESKTOP)
    for rec in steps:
        if rec.step == step:
            rec.step_type = "dupli"
    db.replace_steps(DESKTOP, steps, db.actions(DESKTOP))


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def test_a_span_is_two_events_like_a_walk_of_t(core, tmp_path):
    db, tax = core
    patch = SA.assert_range(db, tax, DESKTOP, LATCH, 21, 19, "closed", evidence_view="scan")
    assert patch.changed and (patch.step, patch.last, patch.top) == (19, 21, 21)
    assert patch.span == [19, 20, 21] and patch.next_step == 22
    assert manual(db) == [(19, "open", "closed"), (22, "closed", "open")]
    assert states(db, tax, range(17, 24)) == {
        17: "open", 18: "open", 19: "closed", 20: "closed", 21: "closed",
        22: "open", 23: "open"}
    assert patch.payload()["last"] == 21 and patch.reach == [19, 20, 21, 22]
    assert validate_events(db.instances(DESKTOP), events_of(db, tax, DESKTOP), tax) == []
    # the same as the walk of T: 21, 20, 19
    other = Db(str(tmp_path / "walk.sqlite"))
    seed_db(other, tax, tmp_path / "walk_cache", last_step=LAST)
    for step in (21, 20, 19):
        SA.assert_state(other, tax, DESKTOP, LATCH, step, "closed", evidence_view="scan")
    assert manual(other) == manual(db)


def test_the_hand_written_events_inside_the_span_go(core):
    db, tax = core
    SA.assert_state(db, tax, DESKTOP, LATCH, 20, "closed")
    SA.assert_state(db, tax, DESKTOP, LATCH, 17, "closed")
    assert manual(db) == [(17, "open", "closed"), (18, "closed", "open"),
                          (20, "open", "closed"), (21, "closed", "open")]
    SA.assert_range(db, tax, DESKTOP, LATCH, 16, 22, "closed")
    assert manual(db) == [(16, "open", "closed"), (23, "closed", "open")]


def test_a_span_shrinking_a_run_from_its_bottom(core):
    db, tax = core
    SA.assert_range(db, tax, DESKTOP, LATCH, 16, 22, "closed")
    SA.assert_range(db, tax, DESKTOP, LATCH, 18, 16, "open")
    assert manual(db) == [(19, "open", "closed"), (23, "closed", "open")]


def test_a_span_across_a_derived_event_is_the_logs_business(core):
    db, tax = core
    with pytest.raises(SA.StateAssertError) as refused:
        SA.plan_range(db, tax, DESKTOP, LATCH, 20, 10, "closed", name=state_zh)
    text = str(refused.value)
    assert "第 10–20 步跨过了日志第 14 步" in text and "Steps 模式" in text
    assert "「关上」→「打开」" in text
    assert manual(db) == []
    # starting *at* the log's step is what a single T there does
    SA.assert_range(db, tax, DESKTOP, LATCH, 14, 16, "closed")
    assert states(db, tax, (13, 14, 16, 17)) == {13: "closed", 14: "closed",
                                                 16: "closed", 17: "open"}


def test_what_holds_already_writes_nothing(core):
    db, tax = core
    patch = SA.assert_range(db, tax, DESKTOP, LATCH, 15, 20, "open")
    assert not patch.changed and manual(db) == []


def test_refusals_of_t_hold_for_a_span(core):
    db, tax = core
    with pytest.raises(SA.StateAssertError, match="已经拆下"):
        SA.plan_range(db, tax, DESKTOP, "cpu_cooler.fan.01", 10, 14, "displaced")
    with pytest.raises(SA.StateAssertError, match="不在 D13 的步骤表里"):
        SA.plan_range(db, tax, DESKTOP, LATCH, 20, 99, "closed")
    _dupli(db, 18)
    with pytest.raises(SA.StateAssertError, match="重复帧"):
        SA.plan_range(db, tax, DESKTOP, LATCH, 18, 20, "closed")
    patch = SA.assert_range(db, tax, DESKTOP, LATCH, 19, 17, "closed")   # 18 inside
    assert manual(db) == [(17, "open", "closed"), (20, "closed", "open")]
    assert patch.inherits == []


def test_a_dupli_after_the_span_follows_it(core):
    db, tax = core
    _dupli(db, 22)
    patch = SA.assert_range(db, tax, DESKTOP, LATCH, 19, 21, "closed")
    assert patch.inherits == [22] and patch.next_step == 23
    assert manual(db) == [(19, "open", "closed"), (23, "closed", "open")]


# --------------------------------------------------------------------------- #
# session
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


HW = (256, 256)
CLIP = "ram_latch.04"      # D2's cover: closed 1-41 by the log, out with the board at 42


@pytest.fixture
def scene(qapp, tmp_path):
    made = make_session(tmp_path, last_step=42, hw=HW)
    made.sweeper_enabled = False
    db = made.db
    db.replace_steps(DESKTOP, db.steps(DESKTOP),
                     [a for a in db.actions(DESKTOP)
                      if not (a.target == CLIP and a.verb == "open")])
    made._invalidate()
    seed_shapes(made, 33, anchor=42)
    yield made
    made.close(force=True)


def anchors(s, instance: str = CLIP) -> list[tuple[int, str]]:
    return sorted((kf.anchor_step, kf.source) for kf in s.db.keyframes(DESKTOP, VIEW, instance))


def test_one_op_one_undo(scene):
    s = scene
    s.goto(41, force=True)
    depth = len(s.undo_stack)
    result = s.assert_state_range(CLIP, "open", 34)
    assert result["changed"] and len(s.undo_stack) == depth + 1
    assert manual(s.db, CLIP) == [(34, "closed", "open")]
    assert result["run"] == (34, 41) and result["after"] == (42, "removed")
    assert set(range(34, 42)) <= set(result["affected"])
    op = s.db.ops(DESKTOP, VIEW, limit=1)[0]
    assert op["kind"] == "assert_state" and op["payload"]["last"] == 41
    assert s.undo() and manual(s.db, CLIP) == []
    assert s.redo() and manual(s.db, CLIP) == [(34, "closed", "open")]


def test_the_cut_follows_a_span_in_the_same_op(scene):
    s = scene
    s.goto(41, force=True)
    s.assert_state(CLIP, "open")
    draw(s, CLIP, rect(2, 2, 30, 12, HW), api.SCOPE_KEYFRAME)
    s.clear_edit()
    assert anchors(s) == [(40, CUT), (42, "manual")]
    s.goto(40, force=True)
    result = s.assert_state_range(CLIP, "open", 34)
    assert [(m.old, m.new) for m in result["cuts"]] == [(40, 33)]
    assert anchors(s) == [(33, CUT), (42, "manual")]
    assert s.undo() is True
    assert anchors(s) == [(40, CUT), (42, "manual")]
    assert manual(s.db, CLIP) == [(41, "closed", "open")]


def test_a_span_across_the_logs_own_action_is_refused_in_the_session(scene):
    s = scene
    s.goto(21, force=True)
    with pytest.raises(SessionRefusal, match="Steps 模式"):
        s.assert_state_range("ram_latch.01", "closed", 10)
    assert manual(s.db, "ram_latch.01") == []


def test_the_dialog_defaults(scene):
    s = scene
    s.goto(41, force=True)
    assert s.state_range_default(CLIP) == ("open", 1)          # no action below: step 1
    s.assert_state(CLIP, "open")
    assert s.state_range_default(CLIP) == ("open", 1)          # hand-set: keep it
    s.goto(21, force=True)
    assert s.state_range_default("ram_latch.01") == ("closed", 15)   # the log opens it at 14
    s.browsing = edit.FORWARD
    assert s.state_range_default("ram_latch.01") == ("closed", 41)   # the board takes it at 42
