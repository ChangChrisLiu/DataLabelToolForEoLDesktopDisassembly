"""Task U17: the chassis and the motherboard are floor layers.

A newly drawn instance goes on top of the ``(view, pose segment)`` layer order
-- except a ``chassis``, which goes to the very bottom, and a ``motherboard``,
which goes directly above the lowest chassis (at the bottom without one).  On
D2 the annotator drew the chassis, the PSU, its latch and the chassis-edge
cover plate before the motherboard on the start frame; the board then painted
over the plate's upper third and a corner of the latch on ~45 frames.

Stored orders are not rewritten: ``cli check`` reports ``zorder_floor:<inst>``
for a chassis above any non-floor part, or a motherboard above one it overlaps
in pixels on some frame of the segment.

Scene: the D13 sheet (``tests/session_scene.py``), whose start frame has a
chassis, ``motherboard.01``, ``psu.01``, ``psu_latch.01`` and ``cover.01``.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from session_scene import (
    BOARD,
    CHASSIS,
    DESKTOP,
    HW,
    LAST_STEP,
    VIEW,
    draw,
    make_session,
    rect,
    seed_db,
)
from tda.cli_common import EXIT_ERROR
from tda.core import masks
from tda.core.db import Db
from tda.core.model import FrameKey, InstanceRec, PairOverride, ShapeKeyframe, ShapePart, ZOrderRec
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.zorder_floor import ZORDER_FLOOR, floor_index, floor_inversions, floor_problems
from tda.ui import session_api as api
from tda.ui.session_edit import _append_to_zorder

PSU = "psu.01"
LATCH = "psu_latch.01"
COVER = "cover.01"
SEG = 1   # the scene has no pose segments: every frame is in the default one
KEY = FrameKey(DESKTOP, LAST_STEP, VIEW)

#: D2's start frame in miniature: the board under the chassis-edge plate's
#: lower rows and under a corner of the PSU latch; the PSU beside it.
FRAME = rect(0, 0, 64, 64)
BOARD_MASK = rect(8, 16, 48, 60)
COVER_MASK = rect(4, 4, 60, 24)       # rows 16-23 lie over the board
LATCH_MASK = rect(42, 40, 56, 52)     # its left corner lies over the board
PSU_MASK = rect(50, 26, 63, 38)       # beside the board, touching nothing


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def db(tmp_path: Path):
    """The seeded D13 database, no session: ``_append_to_zorder`` on its own."""
    tax = load_taxonomy()
    made = Db(str(tmp_path / "tda.sqlite"))
    seed_db(made, tax, tmp_path / "cache")
    yield made
    made.close()


@pytest.fixture
def session(qapp, tmp_path: Path):
    made = make_session(tmp_path)
    made.sweeper_enabled = False
    made.goto(LAST_STEP)
    yield made
    made.close(force=True)


def order_of(db: Db) -> list[str]:
    return [inst for inst, _part in db.zorder(DESKTOP, VIEW, SEG).order]


def append(db: Db, *instances: str) -> None:
    for inst in instances:
        _append_to_zorder(db, KEY, SEG, inst)


# --------------------------------------------------------------------------- #
# _append_to_zorder: the three cases
# --------------------------------------------------------------------------- #
def test_a_chassis_drawn_after_other_parts_goes_to_the_bottom(db):
    append(db, PSU, COVER, CHASSIS)
    assert order_of(db) == [CHASSIS, PSU, COVER]


def test_a_motherboard_drawn_after_others_goes_directly_above_the_chassis(db):
    """D2's own drawing order: chassis, PSU, latch, cover, then the board."""
    append(db, CHASSIS, PSU, LATCH, COVER, BOARD)
    assert order_of(db) == [CHASSIS, BOARD, PSU, LATCH, COVER]


def test_a_motherboard_without_a_chassis_goes_to_the_bottom(db):
    append(db, PSU, COVER, BOARD)
    assert order_of(db) == [BOARD, PSU, COVER]
    append(db, CHASSIS)                 # ... and a chassis drawn later goes under it
    assert order_of(db) == [CHASSIS, BOARD, PSU, COVER]


def test_every_other_class_still_goes_on_top(db):
    append(db, CHASSIS, BOARD, PSU, COVER)
    assert order_of(db) == [CHASSIS, BOARD, PSU, COVER]
    append(db, LATCH)
    assert order_of(db)[-1] == LATCH


def test_the_motherboard_goes_above_the_chassis_whole_block(db):
    """A multi-part chassis is one block: the board does not split it."""
    db.set_zorder(ZOrderRec(DESKTOP, VIEW, SEG,
                            [(CHASSIS, "main"), (CHASSIS, "rim"), (PSU, "main")]))
    append(db, BOARD)
    assert db.zorder(DESKTOP, VIEW, SEG).order == [
        (CHASSIS, "main"), (CHASSIS, "rim"), (BOARD, "main"), (PSU, "main")]


def test_the_class_comes_from_the_instance_record_not_the_key(db):
    db.upsert_instance(InstanceRec("chassis.lid.01", DESKTOP, "cover"))
    db.upsert_instance(InstanceRec("board.extra.01", DESKTOP, "motherboard"))
    append(db, CHASSIS, PSU, "chassis.lid.01", "board.extra.01")
    assert order_of(db) == [CHASSIS, "board.extra.01", PSU, "chassis.lid.01"]


def test_an_instance_already_in_the_order_is_never_moved(db):
    """No silent reorder: an existing order keeps a chassis wherever it is."""
    db.set_zorder(ZOrderRec(DESKTOP, VIEW, SEG, [(PSU, "main"), (CHASSIS, "main")]))
    assert _append_to_zorder(db, KEY, SEG, CHASSIS) is None
    assert order_of(db) == [PSU, CHASSIS]


def test_the_payload_carries_both_whole_orders(db):
    append(db, CHASSIS, PSU, COVER)
    state = _append_to_zorder(db, KEY, SEG, BOARD)
    assert state["before"]["order"] == [[CHASSIS, "main"], [PSU, "main"], [COVER, "main"]]
    assert state["after"]["order"] == [[CHASSIS, "main"], [BOARD, "main"], [PSU, "main"],
                                       [COVER, "main"]]
    assert state["after"]["version"] == state["before"]["version"] + 1


def test_floor_index_on_its_own():
    cls_of = {"c": "chassis", "m": "motherboard", "p": "psu"}
    assert floor_index([("p", "main")], "chassis", cls_of) == 0
    assert floor_index([("p", "main")], "motherboard", cls_of) == 0
    assert floor_index([("p", "main"), ("c", "main")], "motherboard", cls_of) == 2
    assert floor_index([("c", "main"), ("p", "main")], "psu", cls_of) == 2
    assert floor_index([("c", "main")], None, cls_of) == 1     # no record: on top


# --------------------------------------------------------------------------- #
# through the session: undo, and the D2-shaped scene
# --------------------------------------------------------------------------- #
def test_undo_restores_the_order_from_before_the_floor_insert(session):
    db = session.db
    draw(session, PSU, PSU_MASK, api.SCOPE_KEYFRAME)
    draw(session, COVER, COVER_MASK, api.SCOPE_KEYFRAME)
    before = list(db.zorder(DESKTOP, VIEW, SEG).order)
    draw(session, BOARD, BOARD_MASK, api.SCOPE_KEYFRAME)
    assert order_of(db) == [BOARD, PSU, COVER]
    after = list(db.zorder(DESKTOP, VIEW, SEG).order)

    assert session.undo() is True
    assert db.zorder(DESKTOP, VIEW, SEG).order == before
    assert db.keyframes(DESKTOP, VIEW, BOARD) == []
    assert session.redo() is True
    assert db.zorder(DESKTOP, VIEW, SEG).order == after

    draw(session, CHASSIS, FRAME, api.SCOPE_KEYFRAME)
    assert order_of(db) == [CHASSIS, BOARD, PSU, COVER]
    assert session.undo() is True
    assert db.zorder(DESKTOP, VIEW, SEG).order == after


def test_the_d2_scene_leaves_the_plate_and_the_latch_their_pixels(session):
    for inst, mask in ((CHASSIS, FRAME), (PSU, PSU_MASK), (LATCH, LATCH_MASK),
                       (COVER, COVER_MASK), (BOARD, BOARD_MASK)):
        draw(session, inst, mask, api.SCOPE_KEYFRAME)
    assert order_of(session.db) == [CHASSIS, BOARD, PSU, LATCH, COVER]

    frame = session.compiled().instances
    assert np.array_equal(frame[COVER].visible, COVER_MASK)
    assert np.array_equal(frame[LATCH].visible, LATCH_MASK)
    assert np.array_equal(frame[BOARD].visible, BOARD_MASK & ~COVER_MASK & ~LATCH_MASK)
    assert floor_problems(session.db, session.tax, DESKTOP, VIEW, session.truth) == []

    # D2's stored order, for contrast: the board above the three it was drawn after
    session.set_zorder_move(BOARD, COVER)
    assert order_of(session.db) == [CHASSIS, PSU, LATCH, COVER, BOARD]
    frame = session.compiled().instances
    assert int((frame[COVER].visible & BOARD_MASK).sum()) == 0     # the plate lost rows 16-23
    assert int(frame[COVER].visible.sum()) == int((COVER_MASK & ~BOARD_MASK).sum())
    assert floor_problems(session.db, session.tax, DESKTOP, VIEW, session.truth) == [
        f"{ZORDER_FLOOR}:{BOARD} (segment {SEG}, above {LATCH}@{LAST_STEP}, "
        f"{COVER}@{LAST_STEP})"]


# --------------------------------------------------------------------------- #
# cli check: zorder_floor, reported and not repaired
# --------------------------------------------------------------------------- #
def shapes(db: Db, **by_instance: np.ndarray) -> None:
    """One keyframe per instance, anchored at the last step (every frame selects it)."""
    for inst, mask in by_instance.items():
        db.add_keyframe(ShapeKeyframe(
            id=None, instance=inst, desktop=DESKTOP, view=VIEW, pose_segment=SEG,
            anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(mask))]))


def stored(db: Db, *instances: str) -> None:
    db.set_zorder(ZOrderRec(DESKTOP, VIEW, SEG, [(inst, "main") for inst in instances]))


def problems_of(db: Db, tmp_path: Path) -> list[str]:
    tax = load_taxonomy()
    return floor_problems(db, tax, DESKTOP, VIEW,
                          TruthService(db, tax, cache_dir=str(tmp_path / "cache")))


def test_check_reports_a_motherboard_above_a_part_it_overlaps(db, tmp_path):
    shapes(db, **{CHASSIS: FRAME, BOARD: BOARD_MASK, COVER: COVER_MASK, PSU: PSU_MASK})
    stored(db, CHASSIS, PSU, COVER, BOARD)
    assert problems_of(db, tmp_path) == [
        f"{ZORDER_FLOOR}:{BOARD} (segment {SEG}, above {COVER}@{LAST_STEP})"]   # not the PSU


def test_check_is_quiet_about_a_motherboard_above_a_part_it_never_touches(db, tmp_path):
    shapes(db, **{CHASSIS: FRAME, BOARD: BOARD_MASK, PSU: PSU_MASK,
                  COVER: rect(2, 2, 8, 8)})
    stored(db, CHASSIS, PSU, COVER, BOARD)
    assert floor_inversions(db.zorder(DESKTOP, VIEW, SEG).order,
                            {k: r.cls for k, r in db.instances(DESKTOP).items()}) == {
        BOARD: [PSU, COVER]}                      # unusual, but it paints the same pixels
    assert problems_of(db, tmp_path) == []


def test_check_reports_a_chassis_above_any_part_overlap_or_not(db, tmp_path):
    shapes(db, **{CHASSIS: rect(0, 0, 4, 4), BOARD: BOARD_MASK, PSU: PSU_MASK})
    stored(db, PSU, BOARD, CHASSIS)
    assert problems_of(db, tmp_path) == [
        f"{ZORDER_FLOOR}:{CHASSIS} (segment {SEG}, above {PSU})"]   # the board is a floor


def test_check_leaves_a_pair_the_annotator_settled_with_l(db, tmp_path):
    shapes(db, **{CHASSIS: FRAME, BOARD: BOARD_MASK, COVER: COVER_MASK})
    stored(db, CHASSIS, COVER, BOARD)
    db.set_pair_override(PairOverride(DESKTOP, VIEW, SEG, COVER, BOARD))
    assert problems_of(db, tmp_path) == []


def test_cli_check_prints_it_exits_one_and_changes_nothing(db, tmp_path, capsys):
    from tda.cli_app import _check_one

    shapes(db, **{CHASSIS: FRAME, BOARD: BOARD_MASK, COVER: COVER_MASK,
                  LATCH: LATCH_MASK})
    stored(db, CHASSIS, LATCH, COVER, BOARD)
    before = db.zorder(DESKTOP, VIEW, SEG)
    code = _check_one(db, load_taxonomy(), argparse.Namespace(limit=500), DESKTOP, VIEW)
    out = capsys.readouterr().out
    assert (f"  - {ZORDER_FLOOR}:{BOARD} (segment {SEG}, above {LATCH}@{LAST_STEP}, "
            f"{COVER}@{LAST_STEP})") in out
    assert f"[check] 1 layer order(s) put the chassis" in out
    assert code == EXIT_ERROR
    after = db.zorder(DESKTOP, VIEW, SEG)
    assert (after.order, after.version) == (before.order, before.version)   # reported only
