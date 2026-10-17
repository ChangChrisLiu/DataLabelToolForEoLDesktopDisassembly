"""Task U18 (core + session): 「框外零件」 -- objects outside the machine, drawn on their own.

The D13 scene (steps 1-14) with every chassis part of step 14 drawn. A context
object is an instance ``ctx.<class>.NN`` with ``attrs.context``: no state, no
action, no relation. It is made with one undoable op, drawn like any part (its
shapes in their own chain, carried down the walk), compiled in a placement
group of its own -- never occluding or occluded by the machine's parts -- and
never asked for: no task-card row, no ``missing_shape``, no ``Space`` refusal.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import asdict

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from session_scene import CHASSIS, DESKTOP, VIEW, make_session, rect, seed_shapes
from tda.core import context as C
from tda.core.compiler import marked_absent
from tda.core.model import (
    CONTEXT_PLACEMENT,
    FrameKey,
    InstanceRec,
    is_context,
    is_extra,
    machine_only,
)
from tda.core.states import needs_geom
from tda.core.truth_inputs import gather, instances_of, state_of
from tda.core.truth_refusals import blocking_reasons
from tda.ui.session_api import SCOPE_KEYFRAME, SessionRefusal

LAST = 14
STEP = 12
BRACKET = "ctx.cooler_bracket.01"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(qapp, tmp_path):
    made = make_session(tmp_path, last_step=LAST)
    made.sweeper_enabled = False
    seed_shapes(made, LAST, anchor=LAST)
    made.goto(STEP, force=True)
    yield made
    made.close(force=True)


def make_bracket(session, label: str = "旁边的支架") -> str:
    return session.add_context_object("cooler_bracket", label)


def draw_here(session, instance: str, mask: np.ndarray) -> dict:
    session.begin_edit(instance)
    session.set_editing_mask(mask)
    return session.commit_edit(SCOPE_KEYFRAME)


# --------------------------------------------------------------------------- #
# what it is
# --------------------------------------------------------------------------- #
def test_key_format_and_ordinal_per_class_among_context_objects(session):
    tax, db = session.tax, session.db
    stored = db.instances(DESKTOP)
    first = C.plan_context(stored, tax, DESKTOP, "cooler_bracket", "", annotator="t")
    assert first.key == "ctx.cooler_bracket.01"
    assert first.attrs["context"] is True and first.attrs["added_by"] == "t"
    assert "added_at" in first.attrs and first.raw_names == []
    assert "label" not in first.attrs                 # optional for a taxonomy class
    C.write_context(db, first, "t")
    second = C.plan_context(db.instances(DESKTOP), tax, DESKTOP, "cooler_bracket", "b")
    assert second.key == "ctx.cooler_bracket.02"
    # the ordinal runs per class: another class starts at 01
    other = C.plan_context(db.instances(DESKTOP), tax, DESKTOP, "other", "茶杯")
    assert other.key == "ctx.other.01" and other.attrs["label"] == "茶杯"
    assert is_context(first) and not is_extra(first)


def test_other_needs_a_label_and_the_class_must_be_known(session):
    tax, stored = session.tax, session.db.instances(DESKTOP)
    with pytest.raises(C.ContextError):
        C.plan_context(stored, tax, DESKTOP, "other", "  ")
    with pytest.raises(C.ContextError):
        C.plan_context(stored, tax, DESKTOP, "teacup", "x")
    assert "other" in C.class_choices(tax) and C.class_choices(tax)[-1] == "other"


def test_making_one_is_one_logged_op_that_undo_takes_back(session):
    ops_before = len(session.db.ops_of_kind(DESKTOP, C.OP_ADD))
    key = make_bracket(session)
    assert key == BRACKET
    assert is_context(session.db.instances(DESKTOP)[key])
    logged = session.db.ops_of_kind(DESKTOP, C.OP_ADD)
    assert len(logged) == ops_before + 1
    assert logged[-1]["inverse"] == {"delete": key}
    assert session.undo_stack.ops[-1].kind == "add_context"
    assert session.undo()
    assert key not in session.db.instances(DESKTOP)
    assert session.redo()
    assert key in session.db.instances(DESKTOP)


def test_undo_takes_the_object_and_its_keyframes_back(session):
    key = make_bracket(session)
    result = draw_here(session, key, rect(40, 40, 56, 56))
    kfs = session.db.keyframes(DESKTOP, VIEW, key)
    assert len(kfs) == 1 and kfs[0].placement == CONTEXT_PLACEMENT
    assert kfs[0].anchor_step == STEP                  # drawn here, carried down the walk
    assert result["changed"] is not False
    assert session.undo() and session.undo()           # the commit, then the creation
    assert key not in session.db.instances(DESKTOP)
    assert session.db.keyframes(DESKTOP, VIEW, key) == []
    assert all(entry[0] != key for entry in session.db.zorder(DESKTOP, VIEW, 1).order)


def test_the_creation_op_takes_its_shapes_with_it_and_its_redo_brings_them_back(session):
    """The op's own inverse, applied while a shape is still there (no LIFO help)."""
    from tda.ui.session_context import apply_context

    key = make_bracket(session)
    draw_here(session, key, rect(40, 40, 56, 56))
    before = [asdict(k) for k in session.db.keyframes(DESKTOP, VIEW, key)]
    op = next(o for o in session.undo_stack.ops if o.kind == "add_context")
    apply_context(session.db, session.truth, op.inverse, STEP)
    assert key not in session.db.instances(DESKTOP)
    assert session.db.keyframes(DESKTOP, VIEW, key) == []
    apply_context(session.db, session.truth, op.payload, STEP)
    assert key in session.db.instances(DESKTOP)
    assert [asdict(k) for k in session.db.keyframes(DESKTOP, VIEW, key)] == before


def test_removing_takes_every_row_that_names_it_and_restoring_puts_them_back(session):
    key = make_bracket(session)
    draw_here(session, key, rect(40, 40, 56, 56))
    db = session.db
    session.toggle_context_absent(key)                 # a frame override on it
    before_kf = [asdict(k) for k in db.keyframes(DESKTOP, VIEW, key)]
    before_z = list(db.zorder(DESKTOP, VIEW, 1).order)
    with db.transaction():
        record = C.remove_context(db, DESKTOP, key)
    assert key not in db.instances(DESKTOP) and db.keyframes(DESKTOP, VIEW, key) == []
    assert all(e[0] != key for e in db.zorder(DESKTOP, VIEW, 1).order)
    assert key not in db.frame_overrides(FrameKey(DESKTOP, STEP, VIEW))
    with db.transaction():
        C.restore_context(db, record)
    assert [asdict(k) for k in db.keyframes(DESKTOP, VIEW, key)] == before_kf
    assert list(db.zorder(DESKTOP, VIEW, 1).order) == before_z
    assert marked_absent(db.frame_overrides(FrameKey(DESKTOP, STEP, VIEW))[key])


# --------------------------------------------------------------------------- #
# compiled on its own
# --------------------------------------------------------------------------- #
def test_it_is_compiled_where_a_shape_applies_and_nowhere_else(session):
    key = make_bracket(session)
    draw_here(session, key, rect(40, 40, 56, 56))
    for step in range(1, LAST + 1):
        inputs = gather(session.db, session.tax, FrameKey(DESKTOP, step, VIEW))
        if step <= STEP:
            assert inputs.needs.get(key) == "mask", step
            assert inputs.placements[key] == CONTEXT_PLACEMENT
        else:
            assert key not in inputs.needs, step          # the walk passed it: not there
    compiled = session.truth.compile(FrameKey(DESKTOP, STEP, VIEW))
    assert compiled.painted.get(CONTEXT_PLACEMENT) == [key]
    assert key not in compiled.painted.get("in_chassis", [])
    assert not [p for p in compiled.problems if key in p]


def test_it_neither_occludes_nor_is_occluded_by_the_machine(session):
    key = make_bracket(session)
    compiled = session.truth.compile(FrameKey(DESKTOP, STEP, VIEW))
    chassis_before = compiled.instances[CHASSIS].visible.copy()
    # drawn right over the chassis' cell, on top of the layer order
    draw_here(session, key, compiled.instances[CHASSIS].amodal.copy() | rect(40, 40, 56, 56))
    after = session.truth.compile(FrameKey(DESKTOP, STEP, VIEW))
    assert np.array_equal(after.instances[CHASSIS].visible, chassis_before)
    assert after.instances[key].occlusion_ratio == 0.0
    # and its pixels are its own whatever the machine's order says
    assert after.instances[key].visible.sum() == after.instances[key].amodal.sum()


def test_context_objects_layer_among_themselves_by_draw_order_and_with_l(session):
    one = make_bracket(session, "a")
    draw_here(session, one, rect(30, 30, 50, 50))
    two = session.add_context_object("other", "茶杯")
    draw_here(session, two, rect(40, 40, 60, 60))
    compiled = session.truth.compile(FrameKey(DESKTOP, STEP, VIEW))
    assert compiled.painted[CONTEXT_PLACEMENT] == [one, two]   # drawn second: on top
    overlap = rect(40, 40, 50, 50)
    assert not (compiled.instances[one].visible & overlap).any()
    # L between the two: the first one above the second, in pixels
    session.begin_edit(one)
    result = session.commit_edit(f"zorder:above:{two}")
    assert result["op"].kind == "set_pair_override"
    again = session.truth.compile(FrameKey(DESKTOP, STEP, VIEW))
    assert again.instances[one].visible[overlap].all()
    assert not again.instances[two].visible[overlap].any()


def test_layering_across_the_two_families_is_refused(session):
    key = make_bracket(session)
    draw_here(session, key, rect(40, 40, 56, 56))
    with pytest.raises(SessionRefusal):
        session.set_zorder_move(key, CHASSIS)
    with pytest.raises(SessionRefusal):
        session.set_zorder_move(CHASSIS, key)
    session.begin_edit(key)
    with pytest.raises(SessionRefusal):
        session.commit_edit(f"zorder:above:{CHASSIS}")


# --------------------------------------------------------------------------- #
# never asked for
# --------------------------------------------------------------------------- #
def test_no_state_no_card_no_missing_shape_no_space_refusal(session):
    key = make_bracket(session)
    instances = instances_of(session.db, DESKTOP)
    state = state_of(session.db, session.tax, DESKTOP, STEP)
    assert key in instances and key not in state            # no state at all
    assert key not in needs_geom(instances, state, session.tax)
    assert all(row.get("instance") != key for row in session.task_card())
    # the start frame, every machine part drawn; the object not drawn anywhere:
    # nothing about it blocks Space
    session.goto(LAST, force=True)
    frame = FrameKey(DESKTOP, LAST, VIEW)
    compiled = session.truth.compile(frame)
    assert key not in compiled.instances
    reasons = blocking_reasons(session.db, frame, lambda: (compiled, None, None),
                               thorough=True)
    assert reasons == []
    # drawn: confirming the frame works, and freezes its row with the frame
    draw_here(session, key, rect(40, 40, 56, 56))
    session.truth.verify_frame(frame, "tester")
    assert session.db.compiled(frame)[key]["status"] == "verified"
    assert session.db.compiled(frame)[key]["placement"] == CONTEXT_PLACEMENT


def test_the_card_of_every_frame_is_the_one_it_was(session):
    cards = {}
    for step in (STEP, STEP - 1, 3):
        session.goto(step, force=True)
        cards[step] = session.task_card()
    session.goto(STEP, force=True)
    key = make_bracket(session)
    draw_here(session, key, rect(40, 40, 56, 56))
    for step, card in cards.items():
        session.goto(step, force=True)
        assert [r.get("instance") for r in session.task_card()] == \
            [r.get("instance") for r in card]


def test_t_and_shift_t_refuse_it_in_its_own_words(session):
    key = make_bracket(session)
    draw_here(session, key, rect(40, 40, 56, 56))
    choices, why = session.state_gate(key)
    assert choices == [] and "框外零件" in why
    assert "框外零件" in session.verb_gate(key)
    with pytest.raises(SessionRefusal):
        session.correct_verb(key)
    assert session.is_screw(key) is False


# --------------------------------------------------------------------------- #
# I / Ctrl+I: 「不在这张里」
# --------------------------------------------------------------------------- #
def test_i_says_it_is_not_in_this_frame_and_again_takes_it_back(session):
    key = make_bracket(session)
    draw_here(session, key, rect(40, 40, 56, 56))
    frame = FrameKey(DESKTOP, STEP, VIEW)
    assert session.toggle_context_absent(key) == {"absent": True}
    assert marked_absent(session.db.frame_overrides(frame)[key])
    compiled = session.compiled()
    assert compiled.instances[key].visible is None
    assert compiled.instances[key].visibility == "out_of_view"
    row = next(r for r in session.instance_rows() if r["key"] == key)
    assert row["context"] is True and row["absent"] is True
    assert session.toggle_context_absent(key) == {"absent": False}
    assert key not in session.db.frame_overrides(frame)
    # where no shape of it applies it is not there anyway
    session.goto(LAST, force=True)
    with pytest.raises(SessionRefusal):
        session.toggle_context_absent(key)


def test_ctrl_i_marks_a_span_in_one_op(session):
    key = make_bracket(session)
    draw_here(session, key, rect(40, 40, 56, 56))
    result = session.set_context_absent_range(key, True, 8)
    assert result["frames"] == [8, 9, 10, 11, 12]
    for step in range(8, STEP + 1):
        fo = session.db.frame_overrides(FrameKey(DESKTOP, step, VIEW)).get(key)
        assert marked_absent(fo), step
    assert session.undo()
    assert all(key not in session.db.frame_overrides(FrameKey(DESKTOP, s, VIEW))
               for s in range(8, STEP + 1))


def test_the_instance_rows_list_it_last_drawn_here_or_not(session):
    key = make_bracket(session)
    rows = session.instance_rows()
    assert rows[-1]["key"] == key and rows[-1]["has_shape"] is False
    assert all(not r.get("context") for r in rows[:-1])
    draw_here(session, key, rect(40, 40, 56, 56))
    rows = session.instance_rows()
    assert rows[-1]["key"] == key and rows[-1]["has_shape"] is True
    assert rows[-1]["label"] == "旁边的支架"


def test_a_forced_re_import_keeps_it_and_says_so(session):
    from session_scene import FIXTURES
    from tda.pipeline_logs import import_logs_into_db

    key = make_bracket(session)
    draw_here(session, key, rect(40, 40, 56, 56))
    said: list[str] = []
    run = import_logs_into_db(session.db, str(FIXTURES), session.tax, desktops={DESKTOP},
                              force=True, log=said.append)
    [one] = run.runs
    assert one.status == "imported", one.issues
    assert not [d for d in one.dropped if key in d]
    assert key in session.db.instances(DESKTOP)
    assert session.db.keyframes(DESKTOP, VIEW, key)
    assert key not in one.extras_kept
    assert any("context object" in line and key in line for line in said)


# --------------------------------------------------------------------------- #
# the machine's table
# --------------------------------------------------------------------------- #
def test_machine_only_and_the_graph_ignore_it(session):
    from tda.core.graph_derive import settled_instances
    from tda.core.graph_infer import real_instances
    from tda.core.implied import implied_instances

    key = session.add_context_object("motherboard", "另一块主板")
    stored = session.db.instances(DESKTOP)
    assert key in stored and key not in machine_only(stored)
    assert key not in settled_instances(stored)
    assert key not in real_instances(stored, "motherboard")
    assert all(rec.key != key for rec in implied_instances(stored, [], session.tax))
    # nothing about it is a state event
    from tda.core.states import events_from_actions

    assert all(e.target != key for e in events_from_actions(stored, [], session.tax))
