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
from tda.core.cut_follow import CUT, is_cut, same_shape
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
