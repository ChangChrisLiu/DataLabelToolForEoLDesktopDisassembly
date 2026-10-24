"""Task U28: 「拆前 / 拆后」 -- which step, which two scanner frames, which window.

The 14-step D13 scene of the window tests (its 64x64 scanner frames paint a
bright square one cell further along each step, so two consecutive frames
really differ), every chassis part drawn in the scanner view unless a test says
otherwise.  The four captive CPU-fan screws are attached to the fan: unscrewed
(loosened) at 3-6, carried out with the fan at 13.

* the step: a removed part, a displaced one, a captive screw that leaves with
  its cooler, a cover that only opens, the chassis that never changes;
* the window: the part's own scanner outline when there is one, else the
  largest region that differs (one blob, two blobs, below the floor, the ROI,
  the frame border); a pose break or no difference shows the frames whole;
* the captions and the log's words; the caches.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import app_scene as AS
from app_scene import COOLER, DESKTOP, LAST_STEP, SCREWS, make_db, seed_shapes
from tda.core import masks
from tda.core.model import ActionRec, FrameKey, ShapeKeyframe, ShapePart
from tda.core.truth import TruthService
from tda.core.truth_inputs import events_of
from tda.ui.locator_pair import (
    MIN_SIDE_FRAC,
    NO_CHANGE,
    NO_DIFF,
    PAD,
    POSE_BREAK,
    change_window,
    leaving_steps,
)
from tda.ui.locator_source import LocatorSource
from tda.ui.panels.locator import crop_rect
from tda.ui.session import AnnotationSession

SSD = "storage_drive.ssd.01"          # removed at 10
COVER = "cover.01"                    # opened at 2, never removed in 14 steps
PSU = "psu.01"
SCREW = SCREWS[2]                     # screw.cpu_cooler.03


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def make_scene(tmp_path: Path, *, drawn: bool = True, skip=(), **kwargs):
    """``(session on the scanner view, LocatorSource)``."""
    db, paths, tax = make_db(tmp_path, **kwargs)
    session = AnnotationSession(db, tax, TruthService(db, tax), paths["cache_dir"], "tester")
    session.sweeper_enabled = False
    session.open(DESKTOP, "scan")
    if drawn:
        seed_shapes(session, LAST_STEP, skip=skip)
    return session, LocatorSource(db, tax, session.truth, paths["cache_dir"])


@pytest.fixture
def scene(qapp, tmp_path):
    session, source = make_scene(tmp_path)
    yield session, source
    session.close(force=True)


def leaving_of(session, instance: str):
    db, tax = session.db, session.tax
    return leaving_steps(db.instances(DESKTOP), events_of(db, tax, DESKTOP), tax).get(instance)


# --------------------------------------------------------------------------- #
# which step
# --------------------------------------------------------------------------- #
def test_a_removed_part_leaves_at_its_remove_step(scene):
    session, source = scene
    found = leaving_of(session, SSD)
    assert (found.step, found.kind, found.old, found.new) == (10, "removed", "installed",
                                                             "removed")
    assert found.left_with is None
    pair = source.pair(DESKTOP, SSD)
    assert pair.asked == (9, 10) and pair.steps == (9, 10)
    assert pair.captions == ("拆前 第 9 帧", "拆后 第 10 帧")
    assert pair.lines() == ["固态硬盘 storage_drive.ssd.01（装着 → 拆下）", "第 10 帧：SSD · remove"]


def test_a_displaced_part_leaves_at_its_displace_step(qapp, tmp_path):
    session, source = make_scene(tmp_path)
    try:
        db = session.db
        actions = db.actions(DESKTOP) + [
            ActionRec(DESKTOP, 9, 1, PSU, "displace"),
            ActionRec(DESKTOP, 14, 1, PSU, "remove"),
        ]
        db.replace_steps(DESKTOP, db.steps(DESKTOP), actions)
        found = leaving_of(session, PSU)
        assert (found.step, found.kind, found.new) == (9, "displaced", "displaced")
        pair = source.pair(DESKTOP, PSU)          # the action table moved: rebuilt
        assert pair.steps == (8, 9)
        assert pair.log_line == "第 9 帧：SSD connector 2 · displace"
    finally:
        session.close(force=True)


def test_a_captive_screw_leaves_with_its_cooler_not_when_it_is_loosened(scene):
    session, source = scene
    found = leaving_of(session, SCREW)
    assert found.step == 13 and found.kind == "removed"
    assert found.old == "loosened" and found.left_with == COOLER
    pair = source.pair(DESKTOP, SCREW)
    assert pair.steps == (12, 13)
    assert pair.label_line == "散热器螺丝 screw.cpu_cooler.03（拧松 → 拆下）"
    assert pair.log_line == "第 13 帧：CPU fan · remove"     # the cooler's own line
    assert pair.with_line == "随 CPU 风扇 一起拆走"


def test_a_part_that_only_opens_has_its_first_change(scene):
    session, source = scene
    found = leaving_of(session, COVER)
    assert (found.step, found.kind, found.old, found.new) == (2, "changed", "closed", "open")
    pair = source.pair(DESKTOP, COVER)
    assert pair.captions == ("拆前 第 1 帧", "拆后 第 2 帧")
    assert pair.lines() == ["导风罩 cover.01（关上 → 打开）", "第 2 帧：CPU Fan Cover · open"]


def test_a_part_that_never_changes_has_no_pair(scene):
    session, source = scene
    assert leaving_of(session, "chassis") is None
    pair = source.pair(DESKTOP, "chassis")
    assert pair.leaving is None and pair.note == NO_CHANGE
    assert pair.steps == (None, None) and pair.images == (None, None)
    assert pair.lines() == ["机箱 chassis"]


def test_a_card_row_that_names_a_step_is_not_a_part(scene):
    _session, source = scene
    assert source.own_shapes(DESKTOP, "scan", "Remove the cover") == (False, False)
    assert source.own_shapes(DESKTOP, "scan", "ctx.cup.01") == (False, False)
    assert source.own_shapes(DESKTOP, "scan", "chassis") == (True, True)
    assert source.own_shapes(DESKTOP, "oak1", "chassis") == (True, False)


# --------------------------------------------------------------------------- #
# the window: the difference of two pictures
# --------------------------------------------------------------------------- #
def _pictures(hw=(300, 400)):
    rng = np.random.default_rng(7)
    base = rng.integers(60, 80, size=(*hw, 3), dtype=np.uint8)
    return base, base.copy()


def test_one_blob_is_the_window():
    before, after = _pictures()
    after[100:140, 200:260] = 230
    frac = change_window(before, after)
    x0, y0, x1, y1 = frac[0] * 400, frac[1] * 300, frac[2] * 400, frac[3] * 300
    assert abs(x0 - 200) <= 4 and abs(x1 - 260) <= 4
    assert abs(y0 - 100) <= 4 and abs(y1 - 140) <= 4


def test_of_two_blobs_the_largest_wins():
    before, after = _pictures()
    after[20:40, 20:40] = 230              # 400 px
    before[200:260, 250:350] = 10          # 6000 px: what left
    frac = change_window(before, after)
    assert frac[0] * 400 >= 240 and frac[1] * 300 >= 190


def test_a_difference_below_the_floor_is_none():
    before, after = _pictures()
    assert change_window(before, after) is None                # nothing at all
    after[150:152, 150:152] = 250                              # a speck, opened away
    assert change_window(before, after) is None
    after[100:140, 200:260] = before[100:140, 200:260] + 12    # under the level
    assert change_window(before, after) is None


def test_the_roi_limits_the_search_and_the_border_is_left_out():
    before, after = _pictures()
    after[100:140, 200:260] = 230
    after[100:140, 20:60] = 230
    left = change_window(before, after, keep=(0.0, 0.0, 0.3, 1.0))
    assert left[2] <= 0.3
    edge = _pictures()
    edge[1][:, 395:] = 250                 # the frame edge flickers: no window
    assert change_window(*edge) is None


def test_the_crop_pads_the_window_and_keeps_a_sixth_of_the_frame():
    big = crop_rect((1000.0, 1000.0, 2000.0, 1600.0), (3040, 4032), aspect=1.0,
                    fill=1.0 / PAD, min_frac=MIN_SIDE_FRAC)
    assert abs((big[2] - big[0]) - 1000 * PAD) < 1.0      # the long side, padded
    assert abs((big[2] - big[0]) - (big[3] - big[1])) < 1.0   # the cell's shape
    screw = crop_rect((2000.0, 1500.0, 2030.0, 1530.0), (3040, 4032), aspect=1.0,
                      fill=1.0 / PAD, min_frac=MIN_SIDE_FRAC)
    assert abs((screw[2] - screw[0]) - 4032 / 6.0) < 1.0


# --------------------------------------------------------------------------- #
# the window on a scene
# --------------------------------------------------------------------------- #
def test_the_window_is_the_parts_own_scanner_outline_when_there_is_one(scene):
    _session, source = scene
    pair = source.pair(DESKTOP, COVER)          # drawn in the scanner view
    geometry, _why, _where = source.geometry(DESKTOP, 1, COVER)
    assert pair.window_source == "part" and pair.window == tuple(geometry.box)
    assert pair.geometry is not None and pair.note == ""


def test_the_window_is_what_changed_when_it_was_never_drawn(scene):
    _session, source = scene
    before = source.stats["diffs"]
    pair = source.pair(DESKTOP, SSD)            # out at 10: not drawn at 14
    assert pair.window_source == "diff" and pair.note == ""
    x0, y0, x1, y1 = pair.window
    assert 18 <= y0 and y1 <= 34                 # the moving square's rows (20-32)
    assert source.stats["diffs"] == before + 1


def test_a_new_scanner_shape_is_seen_on_the_next_pair(scene):
    session, source = scene
    assert source.pair(DESKTOP, SSD).window_source == "diff"
    shape = np.zeros(AS.HW, dtype=bool)
    shape[40:50, 40:56] = True
    session.db.add_keyframe(ShapeKeyframe(
        id=None, instance=SSD, desktop=DESKTOP, view="scan", pose_segment=1,
        anchor_step=9, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(shape))]))
    session.truth.refresh_range(DESKTOP, "scan", range(1, LAST_STEP + 1))   # as a commit does
    pair = source.pair(DESKTOP, SSD)
    assert pair.window_source == "part"
    assert pair.window[0] <= 40 and pair.window[2] >= 56


def test_two_identical_frames_are_shown_whole_with_a_note(qapp, tmp_path):
    session, source = make_scene(tmp_path, skip=(SSD,))
    try:
        cache = Path(session.cache_dir)
        nine = session.images.image_path(FrameKey(DESKTOP, 9, "scan"))
        ten = session.images.image_path(FrameKey(DESKTOP, 10, "scan"))
        cv2.imwrite(ten, cv2.imread(nine))
        assert cache.exists()
        pair = source.pair(DESKTOP, SSD)
        assert pair.window is None and pair.note == NO_DIFF
        assert pair.images[0] is not None and pair.images[1] is not None
    finally:
        session.close(force=True)


def test_a_pose_break_between_the_two_frames_shows_them_whole(qapp, tmp_path):
    # the scanner was moved at 8: frames 7 and 8 are two pose segments
    session, source = make_scene(tmp_path, two_segments=True)
    try:
        before = source.stats["diffs"]
        pair = source.pair(DESKTOP, "connector.01")       # unplugged at 8
        assert pair.steps == (AS.SEGMENT_CUT - 1, AS.SEGMENT_CUT)
        assert pair.window is None and pair.note == POSE_BREAK
        assert source.stats["diffs"] == before            # no difference taken
        assert pair.images[0] is not None and pair.images[1] is not None
    finally:
        session.close(force=True)


def test_a_missing_scanner_frame_falls_back_and_says_so(scene):
    session, source = scene
    session.db.set_frame_flags(FrameKey(DESKTOP, 10, "scan"), missing=True)
    pair = source.pair(DESKTOP, SSD)
    assert pair.asked == (9, 10) and pair.steps == (9, 9)
    assert pair.captions == ("拆前 第 9 帧", "拆后 第 9 帧（第 10 帧扫描仪没有图）")
    assert pair.note == NO_DIFF                 # the same picture twice


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
def test_a_repeated_pair_comes_from_the_caches(scene):
    _session, source = scene
    first = source.pair(DESKTOP, SSD)
    stats = dict(source.stats)
    again = source.pair(DESKTOP, SSD)
    assert source.stats == stats                 # no picture, no difference, no build
    assert again.window == first.window and again.images[0] is first.images[0]
    # the cooler leaves at the screw's step: the same two frames, one difference
    source.pair(DESKTOP, SCREW)
    diffs = source.stats["diffs"]
    source.pair(DESKTOP, COOLER)
    assert source.stats["diffs"] == diffs
    assert source.stats["pairs"] == stats["pairs"] + 2
