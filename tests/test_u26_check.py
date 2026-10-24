"""Task U26, item 1: ``check`` stops counting expected occlusions as problems.

A part covered completely by labelled parts (or a hand / tool occluder) has an
empty visible mask and the computed label ``occluded_full``: that is what the
layers should give. ``check`` used to list each such row as
``empty_visible:<part>`` -- 100 to 300 per verified desktop on the scanner --
and exit 1 on every finished machine. Now it prints one line,
「n 个零件完全被挡住（正常）」, and an empty visible mask is a problem only when
it has no reason: nothing drawn, or a shape wholly outside the picture that
nobody labelled 「看不见」.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import vis_review_scene as V
from compiler_scenes import KEY, covered, kf, rect, run, zo
from tda.cli_app import _check_one
from tda.cli_common import EXIT_ERROR, EXIT_OK
from tda.core import masks
from tda.core.empty_visible import (
    COVERED,
    NO_SHAPE,
    OUT_OF_VIEW,
    EmptyVisible,
    classify,
    empty_visible,
    split_problems,
)
from tda.core.model import FrameOverride, OccluderMask, ShapeKeyframe, ShapePart, Similarity

FULL = "occluded_full"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------- #
# the rule, on compiled frames
# --------------------------------------------------------------------------- #
def test_a_part_covered_by_a_labelled_part_is_expected():
    out = run(**covered())
    assert "empty_visible:small" in out.problems          # the window's note is unchanged
    [row] = empty_visible(out)
    assert row == EmptyVisible(KEY.step, "small", COVERED, FULL)
    assert row.expected


def test_a_part_covered_by_a_hand_is_expected():
    hand = OccluderMask(KEY, "hand", masks.encode_rle(rect(10, 10, 50, 50)))
    out = run({"small": [kf("small", {"main": (20, 20, 30, 30)})]}, zo([("small", "main")]),
              occluders=[hand])
    [row] = empty_visible(out)
    assert (row.reason, row.visibility, row.expected) == (COVERED, FULL, True)


def test_an_undrawn_part_is_a_problem():
    nothing = ShapeKeyframe(id=1, instance="ghost", desktop=1, view="scan", pose_segment=1,
                            anchor_step=10, placement="in_chassis", geom_type="mask",
                            parts=[ShapePart("main", masks.encode_rle(np.zeros((64, 64), bool)))])
    out = run({"ghost": [nothing]}, zo([("ghost", "main")]))
    assert "empty_visible:ghost" in out.problems
    [row] = empty_visible(out)
    assert (row.reason, row.expected) == (NO_SHAPE, False)
    assert row.problem() == (f"empty_visible:ghost (step {KEY.step}): "
                             f"形状在这一帧是空的（没画，或整个在画面外）")


def test_a_part_outside_the_picture_is_a_problem_until_it_is_labelled_absent():
    """The registration moves the shape off the frame: nothing to be covered."""
    away = Similarity(tx=500.0)
    keyframes = {"far": [kf("far", {"main": (20, 20, 30, 30)})]}
    out = run(keyframes, zo([("far", "main")]), transform=away)
    [row] = empty_visible(out)
    assert (row.reason, row.expected) == (NO_SHAPE, False)

    # 「看不见」 says so: the compiler has no note for it, and the rule agrees
    absent = FrameOverride(KEY, "far", visible_rle=None, visibility="out_of_view")
    labelled = run(keyframes, zo([("far", "main")]), transform=away,
                   frame_overrides={"far": absent})
    assert empty_visible(labelled) == []
    assert labelled.instances["far"].visibility == OUT_OF_VIEW
    assert classify(labelled.instances["far"]) == OUT_OF_VIEW


def test_split_problems_drops_the_expected_and_explains_the_rest():
    problems = ["missing_shape:a", "empty_visible:b", "empty_visible:c", "zorder_missing:d/main",
                "empty_visible:b"]
    rows = [EmptyVisible(1, "b", COVERED, FULL), EmptyVisible(1, "c", NO_SHAPE, "too_small"),
            EmptyVisible(2, "b", COVERED, FULL)]
    kept, expected = split_problems(problems, rows)
    assert kept == ["missing_shape:a",
                    "empty_visible:c (step 1): 形状在这一帧是空的（没画，或整个在画面外）",
                    "zorder_missing:d/main"]
    assert expected == [rows[0], rows[2]]
    # a line with no row to match is kept as it was
    assert split_problems(["empty_visible:x"], []) == (["empty_visible:x"], [])


# --------------------------------------------------------------------------- #
# cli check, on the U24 scene
# --------------------------------------------------------------------------- #
def _check(session, capsys) -> tuple[int, str]:
    code = _check_one(session.db, session.tax, argparse.Namespace(limit=500),
                      V.DESKTOP, V.VIEW)
    return code, capsys.readouterr().out


def test_check_prints_one_line_for_a_covered_part_and_exits_zero(qapp, tmp_path, capsys):
    """``ram_latch.03`` lies wholly under a screw on all 14 frames: normal, not a problem."""
    session = V.build(tmp_path, confirm=())
    try:
        code, out = _check(session, capsys)
    finally:
        session.close(force=True)
    assert "[check] 1 个零件完全被挡住（正常）：14 个零件帧，不算问题" in out
    assert "empty_visible" not in out
    assert "problems: 0" in out
    assert code == EXIT_OK


def test_check_still_lists_an_undrawn_part(qapp, tmp_path, capsys):
    session = V.build(tmp_path, confirm=())
    try:
        # a newer shape of connector.03 with nothing in it, on every frame
        session.db.add_keyframe(ShapeKeyframe(
            id=None, instance=V.TINY, desktop=V.DESKTOP, view=V.VIEW, pose_segment=1,
            anchor_step=14, placement="in_chassis", geom_type="mask", version=5,
            parts=[ShapePart("main", masks.encode_rle(np.zeros(V.HW, bool)))]))
        code, out = _check(session, capsys)
    finally:
        session.close(force=True)
    listed = [line.strip() for line in out.splitlines() if "- empty_visible:" in line]
    assert listed and all(line.startswith(f"- empty_visible:{V.TINY} (step ")
                          and line.endswith("形状在这一帧是空的（没画，或整个在画面外）")
                          for line in listed)
    assert f"[check] {len(listed)} 个零件帧什么都看不见、也没有原因（empty_visible" in out
    assert "[check] 1 个零件完全被挡住（正常）" in out       # the covered one stays a note
    assert f"problems: {len(listed)}" in out
    assert code == EXIT_ERROR


def test_check_says_nothing_of_a_part_labelled_absent(qapp, tmp_path, capsys):
    """``psu.01`` is 「看不见」 on frame 12: its label explains the empty frame."""
    session = V.build(tmp_path, confirm=())
    try:
        code, out = _check(session, capsys)
        frame = session.truth.compile(V.key(V.OUT_STEP)).instances[V.OUT]
    finally:
        session.close(force=True)
    assert frame.visibility == OUT_OF_VIEW
    assert V.OUT not in out and code == EXIT_OK
