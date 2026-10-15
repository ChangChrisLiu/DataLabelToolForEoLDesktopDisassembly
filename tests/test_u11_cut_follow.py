"""Task U11 (A): a cut keyframe is marked, and follows its state boundary.

U8 round 3 cuts the version in force at the open ✂ frame below a commit, with a
copy of the pre-edit shape. On D2/scan the annotator then walked the run of an
opened cover down with ``T``; the cut stayed put, so every frame that joined the
run still showed the copy and was redrawn. Here the copy is marked
(``source="cut"``) and moves with the boundary, in the assertion's own op.

The scene is the D13 sheet at 256 px with every part drawn once (anchor 42),
and the D2 cover's situation -- present 1-45, gone at 46, no ``open`` action --
four steps earlier: ``ram_latch.04`` with its ``open`` action (step 17) taken
out of the table, so the log has it closed from 1 to 41 and the board takes it
out at 42.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from session_scene import DESKTOP, VIEW, draw, make_session, rect, seed_shapes
from tda.core import masks
from tda.core.cut_follow import CUT, is_cut, plan_cut_moves, same_shape
from tda.core.model import FrameKey, ShapeKeyframe, ShapePart
from tda.ui import session_api as api
from tda.ui import session_edit as edit

LATCH = "ram_latch.04"          # closed throughout, out with the board at 42
LAST = 42
HW256 = (256, 256)
OPEN = rect(2, 2, 30, 12, HW256)        # the shape drawn "open" at 41
MOVED = rect(4, 4, 34, 14, HW256)       # the cover swung a little further
CLOSED_AGAIN = rect(6, 6, 20, 20, HW256)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def scene(qapp, tmp_path):
    made = make_session(tmp_path, last_step=LAST, hw=HW256)
    made.sweeper_enabled = False
    drop_open_action(made, LATCH)
    seed_shapes(made, 33, anchor=LAST)    # LATCH's closed shape: one version, 1-41
    yield made
    made.close(force=True)


def drop_open_action(session, instance: str) -> None:
    """The sheet as D2's: nobody wrote down that the part was opened."""
    db = session.db
    db.replace_steps(DESKTOP, db.steps(DESKTOP),
                     [a for a in db.actions(DESKTOP)
                      if not (a.target == instance and a.verb == "open")])
    session._invalidate()


def kfs(session, instance: str = LATCH, view: str = VIEW) -> list[ShapeKeyframe]:
    return sorted(session.db.keyframes(DESKTOP, view, instance),
                  key=lambda kf: (kf.anchor_step, kf.id))


def anchors(session, instance: str = LATCH, view: str = VIEW) -> list[tuple[int, str]]:
    return [(kf.anchor_step, kf.source) for kf in kfs(session, instance, view)]


def amodal(session, step: int, instance: str = LATCH) -> np.ndarray:
    session.goto(step, force=True)
    return session.compiled().instances[instance].amodal.copy()


def manual(session, target: str = LATCH) -> list[tuple]:
    return [(e.step, e.old, e.new) for e in session.db.events(DESKTOP)
            if not e.auto and e.target == target]


def assert_at(session, step: int, state: str = "open", instance: str = LATCH) -> dict:
    session.goto(step, force=True)
    return session.assert_state(instance, state)


# --------------------------------------------------------------------------- #
# A1: the mark
# --------------------------------------------------------------------------- #
def test_the_copy_a_commit_cuts_is_marked_and_keeps_the_draft_reference(scene):
    s = scene
    seeded = kfs(s)[0]
    seeded.source, seeded.draft_id = "ls_adopted", 77     # an adopted Label Studio draft
    s.db.update_keyframe(seeded)
    closed = amodal(s, 41)
    assert_at(s, 41)                                   # ✂ at 40 (closed under open)
    s.goto(41, force=True)
    result = draw(s, LATCH, OPEN, api.SCOPE_KEYFRAME)
    s.clear_edit()
    assert result["cut_below"] == 40
    assert anchors(s) == [(40, CUT), (42, "manual")]
    cut = kfs(s)[0]
    assert is_cut(cut) and cut.draft_id == 77 and (amodal(s, 40) == closed).all()
    # the op carries the mark: undo and redo bring the same row back
    assert s.undo() and anchors(s) == [(42, "ls_adopted")]
    assert s.redo() and anchors(s) == [(40, CUT), (42, "manual")]
    assert kfs(s)[0].draft_id == 77


def test_a_redraw_with_other_pixels_makes_it_the_annotators_shape(scene):
    s = scene
    assert_at(s, 41)
    s.goto(41, force=True)
    draw(s, LATCH, OPEN, api.SCOPE_KEYFRAME)
    s.clear_edit()
    s.goto(40, force=True)                              # the closed side: the copy
    draw(s, LATCH, CLOSED_AGAIN, api.SCOPE_KEYFRAME)
    s.clear_edit()
    assert anchors(s) == [(40, "manual"), (42, "manual")]
    assert s.undo() and anchors(s) == [(40, CUT), (42, "manual")]


def test_a_rewrite_with_the_same_pixels_keeps_the_mark(scene):
    """``commit_edit`` itself (the session skips an untouched layer) with the copy."""
    s = scene
    assert_at(s, 41)
    s.goto(41, force=True)
    draw(s, LATCH, OPEN, api.SCOPE_KEYFRAME)
    s.clear_edit()
    same = masks.decode_rle(kfs(s)[0].parts[0].rle)
    edit.commit_edit(s.db, s.truth, FrameKey(DESKTOP, 40, VIEW), LATCH, same,
                     api.SCOPE_KEYFRAME)
    assert anchors(s)[0] == (40, CUT)


def test_same_shape_reads_pixels_not_spelling():
    one = ShapeKeyframe(None, LATCH, DESKTOP, VIEW, 1, 5,
                        parts=[ShapePart("main", masks.encode_rle(OPEN))])
    two = ShapeKeyframe(None, LATCH, DESKTOP, VIEW, 1, 9,
                        parts=[ShapePart("main", masks.encode_rle_boxed(OPEN))])
    three = ShapeKeyframe(None, LATCH, DESKTOP, VIEW, 1, 9,
                          parts=[ShapePart("main", masks.encode_rle(MOVED))])
    assert same_shape(one, two) and not same_shape(one, three)


# --------------------------------------------------------------------------- #
# A2: the planner on its own (core, no session)
# --------------------------------------------------------------------------- #
HW32 = (32, 32)


def _add(db, anchor: int, shape: np.ndarray, source: str = "manual",
         view: str = VIEW, instance: str = LATCH, segment: int = 1) -> int:
    return db.add_keyframe(ShapeKeyframe(
        id=None, instance=instance, desktop=DESKTOP, view=view, pose_segment=segment,
        anchor_step=anchor, placement="in_chassis", geom_type="mask", source=source,
        parts=[ShapePart("main", masks.encode_rle(shape))]))


def _states(runs: dict, last: int = LAST) -> dict:
    """``step -> state``: closed by default, the runs given, nothing at ``last``."""
    out = {s: "closed" for s in range(1, last)}
    for (low, high), value in runs.items():
        out.update({s: value for s in range(low, high + 1)})
    out[last] = None
    return out


@pytest.fixture
def core(tmp_path):
    from session_scene import seed_db
    from tda.core.db import Db
    from tda.core.taxonomy import load_taxonomy

    tax = load_taxonomy()
    db = Db(str(tmp_path / "tda.sqlite"))
    seed_db(db, tax, tmp_path / "cache", last_step=LAST, hw=HW32)
    return db, tax


def _plan(core, before, after):
    db, tax = core
    return [(m.old, m.new, m.kept, m.reach)
            for m in plan_cut_moves(db, tax, DESKTOP, LATCH, before, after)]


def test_core_extend_moves_the_cut_down_one_frame(core):
    db, _tax = core
    _add(db, 40, rect(1, 1, 9, 9, HW32), source=CUT)
    _add(db, 42, rect(2, 2, 12, 6, HW32))
    before = _states({(41, 41): "open"})
    after = _states({(40, 41): "open"})
    assert _plan(core, before, after) == [(40, 39, False, [40])]


def test_core_a_span_carries_it_all_the_way(core):
    db, _tax = core
    _add(db, 40, rect(1, 1, 9, 9, HW32), source=CUT)
    _add(db, 42, rect(2, 2, 12, 6, HW32))
    assert _plan(core, _states({(41, 41): "open"}), _states({(34, 41): "open"})) == [
        (40, 33, False, list(range(34, 41)))]


def test_core_shrink_moves_it_up(core):
    db, _tax = core
    _add(db, 37, rect(1, 1, 9, 9, HW32), source=CUT)
    _add(db, 42, rect(2, 2, 12, 6, HW32))
    before = _states({(38, 41): "open"})
    assert _plan(core, before, _states({(39, 41): "open"})) == [(37, 38, False, [38])]
    assert _plan(core, before, _states({(41, 41): "open"})) == [
        (37, 40, False, [38, 39, 40])]


def test_core_only_cut_copies_move_and_only_at_a_moved_boundary(core):
    db, _tax = core
    _add(db, 40, rect(1, 1, 9, 9, HW32))                     # the annotator's: stays
    _add(db, 42, rect(2, 2, 12, 6, HW32))
    assert _plan(core, _states({(41, 41): "open"}), _states({(40, 41): "open"})) == []
    _add(db, 20, rect(3, 3, 9, 9, HW32), source=CUT)         # not on a boundary
    assert _plan(core, _states({(41, 41): "open"}), _states({(40, 41): "open"})) == []


def test_core_the_run_reaching_the_first_frame_drops_a_pure_cut(core):
    db, _tax = core
    closed = rect(1, 1, 9, 9, HW32)
    _add(db, 39, closed, source=CUT)
    _add(db, 41, rect(2, 2, 12, 6, HW32))
    _add(db, 42, closed)                                     # the version it was cut from
    before = _states({(40, 41): "open"})
    after = _states({(1, 41): "open"})
    assert _plan(core, before, after) == [(39, None, False, list(range(1, 40)))]


def test_core_a_cut_whose_shape_is_nowhere_else_is_kept(core):
    db, _tax = core
    _add(db, 39, rect(1, 1, 9, 9, HW32), source=CUT)
    _add(db, 41, rect(2, 2, 12, 6, HW32))
    _add(db, 42, rect(4, 4, 14, 8, HW32))                    # re-traced since the cut
    assert _plan(core, _states({(40, 41): "open"}), _states({(1, 41): "open"})) == [
        (39, 39, True, [])]


def test_core_a_merge_over_a_version_of_its_own_drops_a_pure_cut(core):
    """38 was open on its own (37 cut, 38 drawn); 39 joins: the runs merge."""
    db, _tax = core
    closed = rect(1, 1, 9, 9, HW32)
    _add(db, 37, closed, source=CUT)
    _add(db, 38, rect(5, 5, 9, 9, HW32))
    _add(db, 39, closed, source=CUT)
    _add(db, 41, rect(2, 2, 12, 6, HW32))
    _add(db, 42, closed)
    before = _states({(38, 38): "open", (40, 41): "open"})
    after = _states({(38, 41): "open"})
    assert _plan(core, before, after) == [(39, None, False, [39])]


def test_core_a_vanished_run_leaves_the_cut_alone(core):
    db, _tax = core
    _add(db, 40, rect(1, 1, 9, 9, HW32), source=CUT)
    _add(db, 41, rect(2, 2, 12, 6, HW32))
    _add(db, 42, rect(4, 4, 14, 8, HW32))
    assert _plan(core, _states({(41, 41): "open"}), _states({})) == []


def test_core_every_view_on_its_own_walk(core):
    db, tax = core
    for step in range(1, LAST + 1):              # oak1 has the frames, but not 39
        if step != 39:
            db.upsert_frame(FrameKey(DESKTOP, step, "oak1"), f"oak1_{step}.png",
                            {"hw": [32, 32]}, None, None)
    for view in (VIEW, "oak1"):
        _add(db, 40, rect(1, 1, 9, 9, HW32), source=CUT, view=view)
        _add(db, 42, rect(2, 2, 12, 6, HW32), view=view)
    moves = plan_cut_moves(db, tax, DESKTOP, LATCH, _states({(41, 41): "open"}),
                           _states({(39, 41): "open"}))
    assert sorted((m.view, m.old, m.new, tuple(m.reach)) for m in moves) == [
        ("oak1", 40, 38, (40,)), ("scan", 40, 38, (39, 40))]


# --------------------------------------------------------------------------- #
# A2: through the session -- one op, undo, other views
# --------------------------------------------------------------------------- #
def enter_open_at_41(s) -> np.ndarray:
    """``T`` open at 41 and the open shape drawn there; the closed shape it had."""
    closed = amodal(s, 41)
    assert_at(s, 41)
    s.goto(41, force=True)
    result = draw(s, LATCH, OPEN, api.SCOPE_KEYFRAME)
    s.clear_edit()
    assert result["cut_below"] == 40
    return closed


def test_t_on_the_cut_frame_moves_the_cut_with_the_boundary(scene):
    s = scene
    closed = enter_open_at_41(s)
    result = assert_at(s, 40)
    assert [(m.view, m.old, m.new) for m in result["cuts"]] == [(VIEW, 40, 39)]
    assert anchors(s) == [(39, CUT), (42, "manual")]
    assert (amodal(s, 40) == OPEN).all()               # no redraw: the run's version
    assert (amodal(s, 39) == closed).all()             # the far side keeps the copy
    assert 40 in result["affected"]
    s.goto(39, force=True)
    card = [r for r in s.task_card() if r["instance"] == LATCH]
    # round 2: the copy is a placeholder -- the row asks for the frame's shape
    assert card and card[0]["kind"] == api.KIND_SPLIT_KEYFRAME
    assert card[0]["done"] is False and card[0]["cut_copy"] is True


def test_shrinking_the_run_moves_it_back_up(scene):
    s = scene
    closed = enter_open_at_41(s)
    assert_at(s, 40)
    result = assert_at(s, 40, "closed")
    assert [(m.old, m.new) for m in result["cuts"]] == [(39, 40)]
    assert anchors(s) == [(40, CUT), (42, "manual")]
    assert (amodal(s, 40) == closed).all() and (amodal(s, 41) == OPEN).all()


def test_the_state_and_the_move_are_one_op(scene):
    s = scene
    enter_open_at_41(s)
    depth = len(s.undo_stack)
    assert_at(s, 40)
    assert len(s.undo_stack) == depth + 1
    op = s.db.ops(DESKTOP, VIEW, limit=1)[0]
    assert op["kind"] == "assert_state"
    assert [k["anchor_step"] for k in op["payload"]["keyframes"]] == [39]
    assert [k["anchor_step"] for k in op["inverse"]["keyframes"]] == [40]
    assert op["payload"]["cut_reach"] == {VIEW: [40]}
    assert s.undo() is True
    assert manual(s) == [(41, "closed", "open")]
    assert anchors(s) == [(40, CUT), (42, "manual")]
    assert (amodal(s, 40) != OPEN).any()
    assert s.redo() is True
    assert manual(s) == [(40, "closed", "open")]
    assert anchors(s) == [(39, CUT), (42, "manual")]
    assert (amodal(s, 40) == OPEN).all()


def test_a_dropped_cut_comes_back_under_its_own_id(scene):
    """The run reaches over a version of its own (38): the pure cut at 39 goes."""
    s = scene
    enter_open_at_41(s)                                 # cut 40 = the closed shape
    assert_at(s, 38)
    s.goto(38, force=True)
    draw(s, LATCH, MOVED, api.SCOPE_SPLIT)              # cut 37 (closed again), 38 drawn
    s.clear_edit()
    assert [a for a, _src in anchors(s)] == [37, 38, 40, 42]
    assert_at(s, 40)                                    # 40 joins 41: the cut goes to 39
    assert [a for a, _src in anchors(s)] == [37, 38, 39, 42]
    cut_id = next(kf.id for kf in kfs(s) if kf.anchor_step == 39)
    result = assert_at(s, 39)                           # 38-41 one run now
    assert [(m.old, m.new, m.kept) for m in result["cuts"]] == [(39, None, False)]
    assert anchors(s) == [(37, CUT), (38, "manual"), (42, "manual")]
    assert (amodal(s, 39) == OPEN).all()                # the run's version above took it
    assert (amodal(s, 38) == MOVED).all()
    assert s.undo() is True
    assert any(kf.id == cut_id and kf.anchor_step == 39 and is_cut(kf) for kf in kfs(s))
    assert s.redo() is True
    assert all(kf.id != cut_id for kf in kfs(s))


def test_a_cut_redrawn_by_the_annotator_does_not_move(scene):
    s = scene
    enter_open_at_41(s)
    s.goto(40, force=True)
    draw(s, LATCH, CLOSED_AGAIN, api.SCOPE_KEYFRAME)    # the closed side, by hand
    s.clear_edit()
    result = assert_at(s, 40)
    assert result["cuts"] == []
    assert anchors(s) == [(40, "manual"), (42, "manual")]


def test_a_cut_in_another_view_moves_too_and_its_frozen_frame_is_rechecked(scene):
    s = scene
    db = s.db
    for step in range(1, LAST + 1):
        key = FrameKey(DESKTOP, step, "oak1")
        db.upsert_frame(key, f"oak1_{step}.png", {"hw": [256, 256]}, None, None)
    db.conn.execute("UPDATE frame SET review_status='verified' WHERE desktop=? AND "
                    "view='oak1' AND step BETWEEN 38 AND 41", (DESKTOP,))
    _add(db, 40, rect(1, 1, 9, 9, HW256), source=CUT, view="oak1")
    _add(db, 42, rect(2, 2, 12, 6, HW256), view="oak1")
    enter_open_at_41(s)
    result = assert_at(s, 40)
    assert sorted((m.view, m.old, m.new) for m in result["cuts"]) == [
        ("oak1", 40, 39), (VIEW, 40, 39)]
    assert [a for a, _src in anchors(s, view="oak1")] == [39, 42]
    assert 40 in db.rechecks(DESKTOP, "oak1")
    assert s.undo() is True
    assert [a for a, _src in anchors(s, view="oak1")] == [40, 42]
