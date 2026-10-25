"""Task U29 (A): the undo of a 完全 cable label on a part under 20 visible pixels.

The truth table's comparison has a floor (spec 3.4: a tolerant symmetric
difference above ``max(2 % of the area, 20 px)``): a re-compilation that differs
from a frozen row by fewer pixels is "the same annotation". A 完全 row has an
empty visible mask, and its label is the one its (emptied) geometry gives, so
on a part of 16 px the undo found nothing to disagree with -- neither pixels
(16 <= 20) nor a label (the row was not "forced") -- and the frozen row stayed
``occluded_full`` with no pixels under a frame that no longer says so.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from session_scene import CHASSIS, DESKTOP, VIEW, make_session, rect, seed_shapes
from tda.core import masks
from tda.core.model import FrameKey, ShapePart
from tda.core.occluder import FULL, PARTIAL, cable_label

STEP = 14
TINY = (40, 40, 44, 44)          # 4 x 4 = 16 px: under the 20 px floor


def _tiny_chassis(session) -> None:
    db = session.db
    for kf in db.keyframes(DESKTOP, VIEW, CHASSIS):
        kf.parts = [ShapePart("main", masks.encode_rle(rect(*TINY)))]
        db.update_keyframe(kf)
    session.refresh_all()


@pytest.fixture
def tiny(tmp_path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    seed_shapes(session, 1)                       # every part of the scene, drawn to 14
    _tiny_chassis(session)
    for step in (STEP, STEP - 1, STEP - 2):
        session.goto(step, force=True)
        assert session.confirm_frame(), step
    session.goto(STEP, force=True)
    return session


def rows(session, steps=(STEP,)) -> dict:
    return {s: session.db.compiled(FrameKey(DESKTOP, s, VIEW))[CHASSIS] for s in steps}


def open_conflicts(session) -> list[dict]:
    return [dict(c) for c in session.db.conflicts(DESKTOP, VIEW, open_only=True)]


def same_row(a: dict, b: dict) -> bool:
    return (a["visibility"] == b["visibility"] and a["status"] == b["status"]
            and masks.rle_counts(a["visible_rle"]) == masks.rle_counts(b["visible_rle"])
            and abs(float(a["occlusion_ratio"]) - float(b["occlusion_ratio"])) < 1e-9)


# --------------------------------------------------------------------------- #
# the rule (tda.core.truth_conflicts)
# --------------------------------------------------------------------------- #
def _inst(visible, visibility, ratio, box=None):
    from tda.core.compiler import CompiledInstance

    amodal = None if visible is None else rect(*TINY)
    return CompiledInstance("x", visible, box, amodal, ratio, visibility, "in_chassis", 1,
                            window=None if visible is None else (0, 0, 64, 64))


def _row(mask, visibility, ratio):
    return {"geom_type": "mask", "visible_rle": masks.encode_rle(mask), "box": None,
            "visibility": visibility, "occlusion_ratio": ratio, "placement": "in_chassis"}


def test_nothing_seen_against_something_seen_is_a_label_change_under_the_floor():
    import numpy as np

    from tda.core.truth_conflicts import disagreement, label_changes, seen_differs

    empty = np.zeros((64, 64), dtype=bool)
    row = _row(empty, "occluded_full", 1.0)                 # a 完全 row (or fully hidden)
    shown = _inst(rect(*TINY), "too_small", 0.0)
    assert disagreement(row, shown) is None                 # 16 px: under the floor
    assert seen_differs(row, shown)
    assert label_changes(row, shown) == [
        {"field": "visibility", "old": "occluded_full", "new": "too_small"}]
    # the other way round: a confirmed sliver now wholly covered
    assert label_changes(_row(rect(*TINY), "too_small", 0.0),
                         _inst(empty, "occluded_full", 1.0)) == [
        {"field": "visibility", "old": "too_small", "new": "occluded_full"}]


def test_a_sliver_under_the_same_label_and_a_retrace_are_still_noise():
    import numpy as np

    from tda.core.truth_conflicts import label_changes

    empty = np.zeros((64, 64), dtype=bool)
    # a few pixels of a large part peek out: still occluded_full -- nothing to say
    assert label_changes(_row(empty, "occluded_full", 1.0),
                         _inst(rect(40, 40, 42, 42), "occluded_full", 0.99)) == []
    # pixels on both sides: a label crossing (too_small -> visible_tiny) is noise
    assert label_changes(_row(rect(40, 40, 44, 44), "too_small", 0.0),
                         _inst(rect(40, 40, 46, 46), "visible_tiny", 0.0)) == []


# --------------------------------------------------------------------------- #
# the session
# --------------------------------------------------------------------------- #
def test_the_part_is_under_the_floor(tiny):
    row = rows(tiny)[STEP]
    assert masks.rle_area(row["visible_rle"]) == 16
    assert row["visibility"] != FULL and row["status"] == "verified"


def test_undo_of_full_on_a_tiny_part_restores_the_computed_row(tiny):
    session = tiny
    before = rows(session)[STEP]
    result = session.cycle_cable(CHASSIS)                     # 完全
    assert result["accepted"] == [STEP]
    assert rows(session)[STEP]["visibility"] == FULL
    assert session.undo()
    assert cable_label(session.db.frame_overrides(session.current()).get(CHASSIS)) is None
    after = rows(session)[STEP]
    assert same_row(after, before), (after["visibility"], before["visibility"])
    assert open_conflicts(session) == []
    assert session.frame_status(STEP) == "verified"
    # and the redo hides it again, signed
    assert session.redo()
    assert rows(session)[STEP]["visibility"] == FULL and open_conflicts(session) == []


def test_off_after_partial_and_full_cycle_on_a_tiny_part(tiny):
    """完全 -> 部分 -> 撤回 by the key ends where it began, row for row."""
    session = tiny
    before = rows(session)[STEP]
    session.cycle_cable(CHASSIS)
    session.cycle_cable(CHASSIS)
    assert rows(session)[STEP]["visibility"] == PARTIAL
    session.cycle_cable(CHASSIS)
    assert same_row(rows(session)[STEP], before)
    assert open_conflicts(session) == []


def test_ctrl_i_undo_over_tiny_confirmed_frames(tiny):
    """The run: every frame back to its computed row; the off-screen ones too,
    since the pixels are exactly those the rows were confirmed with."""
    session = tiny
    steps = (STEP - 2, STEP - 1, STEP)
    before = rows(session, steps)
    result = session.set_cable_range(CHASSIS, FULL, STEP - 2)
    assert sorted(result["accepted"]) == list(steps)
    assert session.undo()
    after = rows(session, steps)
    assert all(same_row(after[s], before[s]) for s in steps), {
        s: after[s]["visibility"] for s in steps}
    assert open_conflicts(session) == []
    assert all(session.frame_status(s) == "verified" for s in steps)


@pytest.mark.parametrize("verdict", ["viewpoint", "cable"])
def test_a_full_verdict_in_the_visibility_review_undoes_on_a_tiny_part(tmp_path, verdict):
    """Review's Z / ` (U24) take the same path: the 16 px connector of the U24 scene."""
    import vis_review_scene as V
    from tda.core import vis_review as VR

    scene = V.build(tmp_path)
    try:
        k = V.key(14)
        scene.goto(14, force=True)
        before = scene.db.compiled(k)[V.TINY]
        assert masks.rle_area(before["visible_rle"]) == 16
        verdicts = {"viewpoint": VR.VERDICT_VIEWPOINT, "cable": VR.VERDICT_CABLE}
        result = scene.review_visibility(V.TINY, verdicts[verdict], FULL)
        assert result["accepted"] == [14]
        assert scene.db.compiled(k)[V.TINY]["visibility"] == FULL
        assert scene.undo()
        back = scene.db.compiled(k)[V.TINY]
        assert same_row(back, before), back["visibility"]
        assert [c for c in scene.db.conflicts(V.DESKTOP, V.VIEW, open_only=True)] == []
        assert scene.frame_status(14) == "verified"
    finally:
        scene.close(force=True)


def test_off_over_tiny_frames_confirmed_full_signs_only_the_frame_on_screen(tmp_path):
    """完全 confirmed with Space (no history to say what is under it): 撤回 from 14
    signs 14 and leaves 12 and 13 to Review -- the rule of review round 4 holds
    under the floor too, instead of the rows silently staying 完全."""
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    seed_shapes(session, 1)
    _tiny_chassis(session)
    steps = (STEP - 2, STEP - 1, STEP)
    for step in (STEP, STEP - 1, STEP - 2):
        session.goto(step, force=True)
        session.cycle_cable(CHASSIS)                     # unconfirmed: nothing to sign
        assert session.confirm_frame(), step
    session.goto(STEP, force=True)
    assert all(r["visibility"] == FULL for r in rows(session, steps).values())
    result = session.set_cable_range(CHASSIS, None, STEP - 2)
    assert result["accepted"] == [STEP] and sorted(result["withheld"]) == [STEP - 2, STEP - 1]
    assert sorted(int(c["step"]) for c in open_conflicts(session)) == [STEP - 2, STEP - 1]
    assert rows(session)[STEP]["visibility"] != FULL
    assert masks.rle_area(rows(session)[STEP]["visible_rle"]) == 16
