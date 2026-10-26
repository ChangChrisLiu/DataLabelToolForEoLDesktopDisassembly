"""Task U15: 「被线挡住」 -- a part in its place, hidden by something nobody labels.

The rule (:mod:`tda.core.occluder`), the session's ``I`` and ``Ctrl+I``
(:mod:`tda.ui.session_cable`) on the D13 scene of the session tests, the schema
(version 5: ``frame_override.occluder``, an older file migrated on open), the
places that copy an override (merge, ``keep_old``, the digit keys), and
``cli check``'s ``occluder_without_shape``.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from session_scene import (
    CHASSIS,
    COOLER,
    DESKTOP,
    VIEW,
    make_session,
    rect,
    seed_shapes,
)
from tda.core import masks
from tda.core.db import SCHEMA_VERSION, Db
from tda.core.model import (
    OCCLUDER_CABLE,
    OCCLUDER_OTHER,
    FrameKey,
    FrameOverride,
    ShapeKeyframe,
    ShapePart,
)
from tda.core.occluder import (
    FULL,
    PARTIAL,
    cable_label,
    kept_mask,
    next_label,
    occluded_override,
    occluders_without_shape,
)
from tda.core.truth_conflicts import row_visibility, was_forced
from tda.ui.session_api import SessionRefusal
from tda.ui.session_cable import (
    CABLE_EDITING,
    CABLE_GALLERY,
    CABLE_NEEDS_SHAPE,
    CABLE_REMOVED,
    OP_RANGE,
)

STEP = 14                       # the scene's last step: the session opens on it


@pytest.fixture
def drawn(tmp_path):
    """The D13 scene to step 14, every chassis part drawn (anchored at 14)."""
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    seed_shapes(session, STEP)
    assert session.current().step == STEP
    return session


def override(session, step: int = STEP, instance: str = CHASSIS):
    return session.db.frame_overrides(FrameKey(DESKTOP, step, VIEW)).get(instance)


def ops(session, kind: str) -> list[dict]:
    return [op for op in session.db.ops(DESKTOP, VIEW, limit=1000) if op["kind"] == kind]


def visible_area(session, instance: str = CHASSIS) -> int:
    found = session.compiled().instances[instance]
    return 0 if found.visible is None else int(np.count_nonzero(found.visible))


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def test_the_cycle_is_full_partial_off():
    key = FrameKey(DESKTOP, STEP, VIEW)
    assert next_label(None) == FULL
    full = occluded_override(key, CHASSIS, None, FULL, (8, 8))
    assert (full.visibility, full.occluder) == (FULL, OCCLUDER_CABLE)
    assert masks.rle_area(full.visible_rle) == 0 and full.visible_rle["size"] == [8, 8]
    assert cable_label(full) == FULL and next_label(full) == PARTIAL
    partial = occluded_override(key, CHASSIS, full, PARTIAL, (8, 8))
    assert (partial.visibility, partial.occluder, partial.visible_rle) == (
        PARTIAL, OCCLUDER_CABLE, None)              # 完全's empty mask goes
    assert next_label(partial) is None
    assert occluded_override(key, CHASSIS, partial, None, (8, 8)) is None
    # a bare label from the 3 key is not a cable label: I starts at 完全
    bare = FrameOverride(key, CHASSIS, None, FULL)
    assert cable_label(bare) is None and next_label(bare) == FULL


def test_a_hand_drawn_mask_outlives_partial_and_off():
    key = FrameKey(DESKTOP, STEP, VIEW)
    drawn = masks.encode_rle(rect(1, 1, 4, 4, (8, 8)))
    mine = FrameOverride(key, CHASSIS, drawn, PARTIAL, OCCLUDER_CABLE)
    assert kept_mask(mine) == drawn
    off = occluded_override(key, CHASSIS, mine, None, (8, 8))
    assert (off.visible_rle, off.visibility, off.occluder) == (drawn, None, None)
    with pytest.raises(ValueError):
        occluded_override(key, CHASSIS, None, FULL, (8, 8), occluder="hand")


# --------------------------------------------------------------------------- #
# I on a sequence frame
# --------------------------------------------------------------------------- #
def test_i_cycles_full_partial_off_one_op_each(drawn):
    session = drawn
    seen = visible_area(session)
    assert seen > 0

    result = session.cycle_cable(CHASSIS)
    assert result["label"] == FULL and result["changed"]
    fo = override(session)
    assert (fo.visibility, fo.occluder) == (FULL, OCCLUDER_CABLE)
    assert masks.rle_area(fo.visible_rle) == 0
    inst = session.compiled().instances[CHASSIS]
    assert visible_area(session) == 0 and inst.visibility == FULL
    assert inst.amodal is not None and inst.amodal.any()        # the shape stays
    row = session.db.compiled(session.current())[CHASSIS]
    assert row["visibility"] == FULL and masks.rle_area(row["visible_rle"]) == 0
    # a manual label with its empty mask is consistent: nothing forced, no conflict
    assert row_visibility(row) == FULL and not was_forced(row)

    assert session.cycle_cable(CHASSIS)["label"] == PARTIAL
    fo = override(session)
    assert (fo.visibility, fo.occluder, fo.visible_rle) == (PARTIAL, OCCLUDER_CABLE, None)
    assert visible_area(session) == seen                          # pixels kept
    assert session.compiled().instances[CHASSIS].visibility == PARTIAL

    assert session.cycle_cable(CHASSIS)["label"] is None
    assert override(session) is None
    assert visible_area(session) == seen

    logged = sorted(ops(session, "set_frame_override"), key=lambda op: op["id"])
    assert [(op["payload"]["exists"], op["payload"]["visibility"], op["payload"]["occluder"])
            for op in logged] == [(True, FULL, "cable"), (True, PARTIAL, "cable"),
                                  (False, None, None)]
    assert [op["inverse"]["visibility"] for op in logged] == [None, FULL, PARTIAL]
    assert logged[0]["inverse"]["exists"] is False


def test_the_rows_and_the_card_say_it_and_the_row_stays_done(drawn):
    session = drawn
    session.cycle_cable(CHASSIS)
    [row] = [r for r in session.instance_rows() if r["key"] == CHASSIS]
    assert row["cable"] == FULL and row["visibility"] == FULL
    card = {r["instance"]: r for r in session.task_card()}
    assert card[CHASSIS]["cable"] == FULL and card[CHASSIS]["done"]
    assert "invisible" not in card[CHASSIS]          # the reason replaces "set invisible"
    session.cycle_cable(CHASSIS)
    card = {r["instance"]: r for r in session.task_card()}
    assert card[CHASSIS]["cable"] == PARTIAL and card[CHASSIS]["done"]
    others = [r for r in session.instance_rows() if r["key"] != CHASSIS]
    assert all(r["cable"] is None for r in others)


def test_undo_and_redo_walk_the_cycle_back(drawn):
    session = drawn
    session.cycle_cable(CHASSIS)
    session.cycle_cable(CHASSIS)
    assert cable_label(override(session)) == PARTIAL
    assert session.undo()
    assert cable_label(override(session)) == FULL and visible_area(session) == 0
    assert session.undo()
    assert override(session) is None and visible_area(session) > 0
    assert session.redo()
    assert cable_label(override(session)) == FULL
    assert masks.rle_area(override(session).visible_rle) == 0


def test_space_is_not_blocked_by_it(drawn):
    session = drawn
    session.cycle_cable(CHASSIS)
    assert not [p for p in session.current_problems() if CHASSIS in p]
    assert session.confirm_frame()


def test_a_part_with_no_shape_is_refused(tmp_path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    with pytest.raises(SessionRefusal) as err:
        session.cycle_cable(CHASSIS)
    assert str(err.value) == CABLE_NEEDS_SHAPE
    assert override(session) is None and not ops(session, "set_frame_override")
    assert session.cable_gate(CHASSIS) == (FULL, CABLE_NEEDS_SHAPE)


def test_a_removed_part_is_refused(drawn):
    # the cooler is lifted out at step 13: frame 14 does not have it
    assert drawn.cable_gate(COOLER)[1] == CABLE_REMOVED
    with pytest.raises(SessionRefusal):
        drawn.cycle_cable(COOLER)


def test_the_part_being_drawn_is_refused(drawn):
    session = drawn
    session.begin_edit(CHASSIS)
    mask = session.compiled().instances[CHASSIS].amodal.copy()
    mask[:4, :4] = True
    session.set_editing_mask(mask)
    with pytest.raises(SessionRefusal) as err:
        session.cycle_cable(CHASSIS)
    assert str(err.value) == CABLE_EDITING
    assert override(session) is None


def test_off_is_allowed_where_the_shape_went(drawn):
    """撤回 is how a label that lost its shape is repaired: never refused for it."""
    session = drawn
    session.cycle_cable(CHASSIS)
    session.cycle_cable(CHASSIS)            # 部分: the next press is 撤回
    for kf in session.db.keyframes(DESKTOP, VIEW, CHASSIS):
        session.db.delete_keyframe(kf.id)
    session._invalidate()
    assert session.cable_gate(CHASSIS) == (None, "")
    session.cycle_cable(CHASSIS)
    assert override(session) is None


# --------------------------------------------------------------------------- #
# the digit keys and the other writers of an override
# --------------------------------------------------------------------------- #
def test_a_digit_replaces_the_label_and_its_reason(drawn):
    session = drawn
    session.cycle_cable(CHASSIS)
    session.set_visibility(CHASSIS, "visible")
    fo = override(session)
    assert (fo.visibility, fo.occluder, fo.visible_rle) == ("visible", None, None)
    assert visible_area(session) > 0
    assert session.undo()
    assert cable_label(override(session)) == FULL


def test_clear_visibility_takes_the_reason_and_its_empty_mask(drawn):
    session = drawn
    session.cycle_cable(CHASSIS)
    session.clear_visibility(CHASSIS)
    assert override(session) is None


def test_a_single_frame_mask_keeps_the_label_and_the_reason(drawn):
    session = drawn
    session.cycle_cable(CHASSIS)
    session.cycle_cable(CHASSIS)            # 部分
    session.begin_edit(CHASSIS)
    session.set_editing_mask(rect(10, 10, 30, 30))
    session.commit_edit("frame_override")
    fo = override(session)
    assert (fo.visibility, fo.occluder) == (PARTIAL, OCCLUDER_CABLE)
    assert masks.rle_area(fo.visible_rle) > 0


def test_the_store_refuses_an_unknown_occluder(tmp_path):
    db = Db(str(tmp_path / "x.sqlite"))
    key = FrameKey(DESKTOP, 1, VIEW)
    with pytest.raises(ValueError):
        db.set_frame_override(FrameOverride(key, CHASSIS, None, FULL, "hand"))
    db.set_frame_override(FrameOverride(key, CHASSIS, None, PARTIAL, OCCLUDER_OTHER))
    assert db.frame_overrides(key)[CHASSIS].occluder == OCCLUDER_OTHER
    # the whole row: a label written without its reason clears the old one
    db.set_frame_override(FrameOverride(key, CHASSIS, None, PARTIAL))
    assert db.frame_overrides(key)[CHASSIS].occluder is None
    assert [fo.instance for fo in db.occluded_overrides(DESKTOP, VIEW)] == []
    db.close()


# --------------------------------------------------------------------------- #
# Ctrl+I: a run of frames, one op
# --------------------------------------------------------------------------- #
def test_ctrl_i_writes_a_run_as_one_op(drawn):
    session = drawn
    result = session.set_cable_range(CHASSIS, FULL, 10)
    assert result["frames"] == [10, 11, 12, 13, 14]
    for step in range(10, 15):
        assert cable_label(override(session, step)) == FULL, step
    assert override(session, 9) is None
    [op] = ops(session, OP_RANGE)
    assert [f["step"] for f in op["payload"]["frames"]] == [10, 11, 12, 13, 14]
    assert all(not f["exists"] for f in op["inverse"]["frames"])
    assert session.undo()
    assert all(override(session, s) is None for s in range(10, 15))
    assert session.redo()
    assert all(cable_label(override(session, s)) == FULL for s in range(10, 15))
    # 撤回 over part of it: only the frames that carry the label
    session.set_cable_range(CHASSIS, None, 12)
    assert [cable_label(override(session, s)) for s in range(10, 15)] == [
        FULL, FULL, None, None, None]


def test_ctrl_i_names_the_frames_without_a_shape(tmp_path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    # the chassis is drawn up to step 12 only
    session.db.add_keyframe(ShapeKeyframe(
        id=None, instance=CHASSIS, desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=12, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(rect(4, 4, 40, 40)))]))
    session._invalidate()
    with pytest.raises(SessionRefusal) as err:
        session.set_cable_range(CHASSIS, FULL, 10)
    assert "第 13–14 步" in str(err.value) and CABLE_NEEDS_SHAPE in str(err.value)
    assert not ops(session, OP_RANGE)
    assert all(override(session, s) is None for s in range(10, 15))
    session.goto(12, force=True)
    assert session.set_cable_range(CHASSIS, PARTIAL, 10)["frames"] == [10, 11, 12]


def test_ctrl_i_stays_inside_the_pose_segment(drawn):
    session = drawn
    db = session.db
    db.set_pose_segment(DESKTOP, VIEW, 1, 1, 7, 7, None, None)
    db.set_pose_segment(DESKTOP, VIEW, 2, 8, 14, 14, None, None)
    assert session.cable_bounds() == (8, 14)
    with pytest.raises(SessionRefusal) as err:
        session.set_cable_range(CHASSIS, FULL, 5)
    assert "另一个位姿段" in str(err.value) and "8–14" in str(err.value)


def test_the_gallery_refuses_both(tmp_path):
    import gallery_scene as GS
    from tda.core.truth import TruthService
    from tda.ui.session import AnnotationSession

    db, tax, cache_dir = GS.make_db(tmp_path)
    session = AnnotationSession(db, tax, TruthService(db, tax), str(cache_dir), "tester")
    session.sweeper_enabled = False
    session.open(GS.DESKTOP, "cmp_oak1")
    part = next(iter(session.compiled().instances))
    for call in (lambda: session.cycle_cable(part),
                 lambda: session.set_cable_range(part, FULL, session.current().step)):
        with pytest.raises(SessionRefusal) as err:
            call()
        assert str(err.value) == CABLE_GALLERY


# --------------------------------------------------------------------------- #
# the schema: version 5, an older file migrated on open
# --------------------------------------------------------------------------- #
def test_a_v4_file_without_the_column_opens_and_gains_it(tmp_path):
    path = str(tmp_path / "old.sqlite")
    schema = (Path(__file__).resolve().parents[1] / "tda" / "core" / "schema.sql"
              ).read_text(encoding="utf-8")
    old = re.sub(r"\n\s*--[^\n]*\n\s*--[^\n]*\n\s*occluder\s+TEXT,", "", schema)
    assert "occluder        TEXT" not in old and old != schema
    conn = sqlite3.connect(path)
    with conn:
        conn.executescript(old)
        conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', '4')")
        conn.execute("INSERT INTO desktop(id) VALUES(?)", (DESKTOP,))
        conn.execute("INSERT INTO frame_override(desktop, step, view, instance, visibility) "
                     "VALUES(?, 3, ?, ?, 'occluded_full')", (DESKTOP, VIEW, CHASSIS))
    assert "occluder" not in {r[1] for r in conn.execute("PRAGMA table_info(frame_override)")}
    conn.close()

    db = Db(path)
    try:
        columns = {r["name"] for r in db.conn.execute("PRAGMA table_info(frame_override)")}
        assert "occluder" in columns
        fo = db.frame_overrides(FrameKey(DESKTOP, 3, VIEW))[CHASSIS]
        assert (fo.visibility, fo.occluder) == ("occluded_full", None)   # a bare label
        assert db.conn.execute("SELECT value FROM meta WHERE key='schema_version'"
                               ).fetchone()[0] == str(SCHEMA_VERSION)
        assert SCHEMA_VERSION >= 5          # 5 here; 6 since U35 (frame_note)
        db.set_frame_override(FrameOverride(FrameKey(DESKTOP, 3, VIEW), CHASSIS, None,
                                            FULL, OCCLUDER_CABLE))
        assert db.frame_overrides(FrameKey(DESKTOP, 3, VIEW))[CHASSIS].occluder == "cable"
    finally:
        db.close()
    Db(path).close()                     # opening it again changes nothing


# --------------------------------------------------------------------------- #
# merge, keep_old, cli check
# --------------------------------------------------------------------------- #
def test_a_merge_carries_the_reason(tmp_db_path):
    from merge_scenes import DESKTOP as MD, GONE, KEEP, build
    from tda.core.merge_instances import merge_instances, undo_merge

    db = Db(tmp_db_path)
    try:
        scene = build(db)
        key = scene.key(2)
        db.set_frame_override(FrameOverride(key, GONE, masks.encode_rle(
            np.zeros((64, 64), dtype=bool)), FULL, OCCLUDER_CABLE))
        report = merge_instances(db, scene.tax, MD, GONE, KEEP, "tester")
        moved = db.frame_overrides(key)
        assert GONE not in moved and moved[KEEP].occluder == OCCLUDER_CABLE
        assert moved[KEEP].visibility == FULL
        undo_merge(db, scene.tax, db.get_op(report.op_id), "tester")
        back = db.frame_overrides(key)
        assert KEEP not in back and back[GONE].occluder == OCCLUDER_CABLE
    finally:
        db.close()


def test_keep_old_keeps_a_reason_its_label_still_fits(drawn):
    session = drawn
    truth, db = session.truth, session.db
    key = session.current()
    session.cycle_cable(CHASSIS)
    session.cycle_cable(CHASSIS)                 # 部分, then confirm the frame
    assert session.confirm_frame()
    # a re-trace elsewhere moves the chassis under the confirmed row
    kf = db.keyframes(DESKTOP, VIEW, CHASSIS)[0]
    kf.parts = [ShapePart("main", masks.encode_rle(rect(0, 0, 50, 50)))]
    db.update_keyframe(kf)
    truth.refresh(key)
    [conflict] = [c for c in db.conflicts(DESKTOP, open_only=True)
                  if c["instance"] == CHASSIS and int(c["step"]) == key.step]
    truth.resolve_conflict(conflict["id"], "keep_old", "tester")
    fo = db.frame_overrides(key)[CHASSIS]
    assert fo.visibility == PARTIAL and fo.occluder == OCCLUDER_CABLE


def test_cli_check_reports_a_reason_without_a_shape(drawn):
    session = drawn
    db, tax = session.db, session.tax
    session.cycle_cable(CHASSIS)
    assert occluders_without_shape(db, tax, DESKTOP, VIEW) == []
    # the store can hold one the window refuses to write: the cooler at 14
    db.set_frame_override(FrameOverride(FrameKey(DESKTOP, 14, VIEW), COOLER, None,
                                        PARTIAL, OCCLUDER_CABLE))
    for kf in db.keyframes(DESKTOP, VIEW, CHASSIS):
        db.delete_keyframe(kf.id)
    assert occluders_without_shape(db, tax, DESKTOP, VIEW) == [(14, CHASSIS), (14, COOLER)]
