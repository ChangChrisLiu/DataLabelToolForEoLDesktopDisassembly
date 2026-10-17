"""Task U15, review round: the reviewer's scenes as tests.

* M1 -- a label on a frame that can no longer carry it (the step table moved the
  part to the bench or out, the shape went): ``cli check`` reports it, ``I``
  takes it straight back (never via 部分) and says why; 撤回 is never refused.
* M2 -- ``keep_old`` keeps the reason only with the very label it was given for.
* M3 -- 完全 over a hand-drawn single-frame mask says so; a later 部分 / 撤回
  does not claim it is back.
* M4 -- pixels committed for this frame only on a 完全 part make it 部分.
* M5 -- a 完全 part's COCO annotation is a crowd region.
* M6 -- the empty mask is built from the run lengths, once per size.
* M8 -- on a confirmed frame the label is re-signed, both ways.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
from tda.core.export import export_coco
from tda.core.model import OCCLUDER_CABLE, FrameKey, FrameOverride, ShapeKeyframe, ShapePart
from tda.core.occluder import (
    FULL,
    PARTIAL,
    cable_label,
    empty_rle,
    occluders_without_shape,
)
from tda.ui import session_edit as edit
from tda.ui.app_cable import HAND_NOT_BACK, HAND_REPLACED, cable_note
from tda.ui.session_cable import (
    CABLE_NEEDS_SHAPE,
    CABLE_ON_BENCH,
    CABLE_REMOVED,
    OP_RANGE,
)

STEP = 14
BENCH_ROI = [0, 48, 24, 64]


@pytest.fixture
def drawn(tmp_path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    seed_shapes(session, STEP)
    return session


def label_at(session, step: int, instance: str = CHASSIS):
    return cable_label(session.db.frame_overrides(FrameKey(DESKTOP, step, VIEW)).get(instance))


def override(session, step: int = STEP, instance: str = CHASSIS):
    return session.db.frame_overrides(FrameKey(DESKTOP, step, VIEW)).get(instance)


def _cooler_drawn_to_12(session) -> None:
    session.db.add_keyframe(ShapeKeyframe(
        id=None, instance=COOLER, desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=12, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(rect(30, 30, 50, 50)))]))


def _take_the_cooler_out_at_12(session) -> None:
    """Steps mode: the log is corrected -- the cooler came out at 12, not 13."""
    db = session.db
    actions = db.actions(DESKTOP)
    for n, action in enumerate(a for a in actions if a.target == COOLER and a.step == 13):
        action.step, action.idx = 12, 90 + n
    db.replace_steps(DESKTOP, db.steps(DESKTOP), actions)
    session.refresh_all()
    session.goto(12, force=True)


# --------------------------------------------------------------------------- #
# M1: a label the frame can no longer carry
# --------------------------------------------------------------------------- #
def test_a_part_moved_to_the_bench_under_its_label_is_reported_and_taken_back(tmp_path):
    """p5: the cooler 完全 at 12, then the log takes it out at 12 -- onto the bench."""
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    seed_shapes(session, STEP)
    db = session.db
    _cooler_drawn_to_12(session)
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=COOLER, desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=STEP, placement="on_bench", geom_type="box",
        parts=[ShapePart("main", None, (2.0, 50.0, 20.0, 62.0))]))
    db.set_pose_segment(DESKTOP, VIEW, 1, 1, STEP, STEP, None, None)
    with db.conn:
        db.conn.execute("UPDATE pose_segment SET bench_roi_json=? WHERE desktop=? AND view=?",
                        (json.dumps(BENCH_ROI), DESKTOP, VIEW))
    session.refresh_all()
    session.goto(12, force=True)
    session.cycle_cable(COOLER)
    assert label_at(session, 12, COOLER) == FULL
    assert occluders_without_shape(db, session.tax, DESKTOP, VIEW) == []

    _take_the_cooler_out_at_12(session)
    assert session.compiled().instances[COOLER].placement == "on_bench"
    assert occluders_without_shape(db, session.tax, DESKTOP, VIEW) == [(12, COOLER)]
    assert session.cable_gate(COOLER) == (None, "")          # straight to 撤回
    result = session.cycle_cable(COOLER)
    assert result["label"] is None and result["stale"] == CABLE_ON_BENCH
    assert override(session, 12, COOLER) is None
    assert occluders_without_shape(db, session.tax, DESKTOP, VIEW) == []
    # its bench box is back in the truth table
    assert session.compiled().instances[COOLER].box is not None
    note = cable_note("cooler", None, result)
    assert "已经不能标被线挡住" in note and "台面" in note


def test_a_full_label_whose_shape_went_is_taken_back_not_made_partial(drawn):
    """p7_stale_full."""
    session = drawn
    session.cycle_cable(CHASSIS)                                   # 完全
    for kf in session.db.keyframes(DESKTOP, VIEW, CHASSIS):
        session.db.delete_keyframe(kf.id)
    session._invalidate()
    assert occluders_without_shape(session.db, session.tax, DESKTOP, VIEW) == [(STEP, CHASSIS)]
    assert session.cable_gate(CHASSIS) == (None, "")
    result = session.cycle_cable(CHASSIS)
    assert result["label"] is None and result["stale"] == CABLE_NEEDS_SHAPE
    assert override(session) is None
    # with no label left, the shape refusal is what I says
    assert session.cable_gate(CHASSIS) == (FULL, CABLE_NEEDS_SHAPE)


def test_a_label_on_a_part_taken_out_earlier_is_reported_and_taken_back(drawn):
    """p7_removed: no staging area, so the cooler is simply gone from 12."""
    session = drawn
    _cooler_drawn_to_12(session)
    session.refresh_all()
    session.goto(12, force=True)
    session.cycle_cable(COOLER)
    _take_the_cooler_out_at_12(session)
    assert COOLER not in session.compiled().instances
    assert occluders_without_shape(session.db, session.tax, DESKTOP, VIEW) == [(12, COOLER)]
    result = session.cycle_cable(COOLER)
    assert result["label"] is None and result["stale"] == CABLE_REMOVED
    assert occluders_without_shape(session.db, session.tax, DESKTOP, VIEW) == []
    # ... or from a frame where it is still there: Ctrl+I 撤回 reaches it
    session.undo()
    session.goto(11, force=True)
    assert session.set_cable_range(COOLER, None, 12)["frames"] == [12]
    assert occluders_without_shape(session.db, session.tax, DESKTOP, VIEW) == []


def test_off_is_never_refused_not_even_while_the_part_is_drawn(drawn):
    session = drawn
    session.cycle_cable(CHASSIS)
    session.cycle_cable(CHASSIS)                                   # 部分: next is 撤回
    session.begin_edit(CHASSIS)
    session.set_editing_mask(rect(0, 0, 9, 9))
    assert session.cable_gate(CHASSIS) == (None, "")
    session.cycle_cable(CHASSIS)
    assert override(session) is None
    session.set_cable_range(CHASSIS, None, STEP)                  # nothing left: no-op


def test_cli_check_names_the_way_back(drawn, tmp_path, capsys):
    from tda.cli_app import _check_one

    session = drawn
    session.cycle_cable(CHASSIS)
    for kf in session.db.keyframes(DESKTOP, VIEW, CHASSIS):
        session.db.delete_keyframe(kf.id)

    class Args:
        limit = 1000

    _check_one(session.db, session.tax, Args(), DESKTOP, VIEW)
    out = capsys.readouterr().out
    assert f"occluder_without_shape:{CHASSIS} (step {STEP})" in out
    assert "按 I 撤回（或从零件还在的帧按 Ctrl+I 撤回）" in out and f"{CHASSIS}@{STEP}" in out


# --------------------------------------------------------------------------- #
# M2: keep_old
# --------------------------------------------------------------------------- #
def test_keep_old_drops_a_reason_the_kept_label_was_not_given_for(drawn):
    """p1e: a frozen row is kept over a 完全; the frozen label never had a cable."""
    session = drawn
    db, truth = session.db, session.truth
    key = session.current()
    assert session.confirm_frame()
    frozen = db.compiled(key)[CHASSIS]["visibility"]
    assert frozen != FULL
    # written at the store, not through I: the conflict is left for Review
    db.set_frame_override(FrameOverride(key, CHASSIS, empty_rle((64, 64)), FULL,
                                        OCCLUDER_CABLE))
    truth.refresh(key)
    [cid] = [c["id"] for c in db.conflicts(DESKTOP, VIEW, open_only=True)
             if c["instance"] == CHASSIS]
    truth.resolve_conflict(cid, "keep_old", "tester")
    fo = db.frame_overrides(key)[CHASSIS]
    assert fo.visibility == frozen and fo.occluder is None


# --------------------------------------------------------------------------- #
# M3 / M4: a hand-drawn single-frame mask
# --------------------------------------------------------------------------- #
def test_full_over_a_hand_drawn_mask_says_so_and_the_rest_does_not_claim_it_back(drawn):
    """p8_handmask."""
    session = drawn
    session.begin_edit(CHASSIS)
    session.set_editing_mask(rect(2, 2, 6, 6))
    session.commit_edit("frame_override")
    full = session.cycle_cable(CHASSIS)
    assert full["label"] == FULL and full["replaced_hand_mask"]
    assert HAND_REPLACED in cable_note("chassis", FULL, full)
    partial = session.cycle_cable(CHASSIS)
    assert partial["label"] == PARTIAL and partial["hand_mask_lost"]
    assert HAND_NOT_BACK in cable_note("chassis", PARTIAL, partial)
    off = session.cycle_cable(CHASSIS)
    assert off["label"] is None and off["hand_mask_lost"]
    assert override(session) is None
    assert HAND_NOT_BACK in cable_note("chassis", None, off)
    # Ctrl+Z three times finds the hand-drawn mask
    for _ in range(3):
        session.undo()
    fo = override(session)
    assert fo.occluder is None and masks.rle_area(fo.visible_rle) == 16


def test_a_cycle_without_a_hand_mask_says_nothing_about_one(drawn):
    session = drawn
    assert not session.cycle_cable(CHASSIS)["replaced_hand_mask"]
    assert not session.cycle_cable(CHASSIS)["hand_mask_lost"]
    assert not session.cycle_cable(CHASSIS)["hand_mask_lost"]


def test_pixels_for_this_frame_only_on_a_full_part_make_it_partial(drawn):
    """p8b."""
    session = drawn
    key = session.current()
    session.cycle_cable(CHASSIS)
    session.begin_edit(CHASSIS)
    session.set_editing_mask(rect(2, 2, 5, 5))
    result = session.commit_edit("frame_override")
    assert result["cable_partial"]
    fo = override(session)
    assert (fo.visibility, fo.occluder) == (PARTIAL, OCCLUDER_CABLE)
    assert masks.rle_area(fo.visible_rle) == 9
    inst = session.compiled().instances[CHASSIS]
    assert inst.visibility == PARTIAL and int(inst.visible.sum()) == 9
    assert [r["cable"] for r in session.instance_rows() if r["key"] == CHASSIS] == [PARTIAL]
    assert session.cable_outlines() == []
    session.undo()
    assert cable_label(override(session)) == FULL
    # an empty mask for this frame only says nothing is seen: still 完全
    out = edit.commit_edit(session.db, session.truth, key, CHASSIS,
                           np.zeros((64, 64), dtype=bool), "frame_override",
                           annotator="tester")
    assert not out["cable_partial"]
    assert cable_label(override(session)) == FULL


# --------------------------------------------------------------------------- #
# M5: a crowd region in COCO
# --------------------------------------------------------------------------- #
def test_a_full_part_is_a_crowd_region_and_a_perfect_detector_scores_1(drawn, tmp_path):
    """p5_coco: segm AP of a perfect visible detector is 1.0 with the crowd flag."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    session = drawn
    session.cycle_cable(CHASSIS)
    out = tmp_path / "coco.json"
    doc = export_coco(session.db, session.tax, [DESKTOP], VIEW, str(out), only_verified=False)
    images = {im["id"]: im["extra"]["step"] for im in doc["images"]}
    [hidden] = [a for a in doc["annotations"] if images[a["image_id"]] == STEP
                and a["attributes"]["instance_key"] == CHASSIS]
    assert hidden["iscrowd"] == 1 and hidden["area"] == 0
    assert set(hidden["segmentation"]) == {"size", "counts"}          # compressed RLE
    assert isinstance(hidden["segmentation"]["counts"], str)
    assert all(a["iscrowd"] == 0 for a in doc["annotations"] if a is not hidden)

    def ap(path) -> float:
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(path))
            data = json.loads(path.read_text(encoding="utf-8"))
            dets = [{"image_id": a["image_id"], "category_id": a["category_id"], "score": 1.0,
                     "segmentation": a["segmentation"]}
                    for a in data["annotations"] if a["area"] > 0]
            ev = COCOeval(gt, gt.loadRes(copy.deepcopy(dets)), "segm")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        return float(ev.stats[0])

    assert ap(out) == pytest.approx(1.0)
    # the control: the same file without the crowd flag punishes the detector
    plain = copy.deepcopy(doc)
    for ann in plain["annotations"]:
        ann["iscrowd"] = 0
    other = tmp_path / "plain.json"
    other.write_text(json.dumps(plain), encoding="utf-8")
    assert ap(other) < 1.0


# --------------------------------------------------------------------------- #
# M6: the empty mask
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hw", [(1, 1), (8, 8), (64, 64), (3040, 4032), (4032, 3040)])
def test_the_empty_mask_is_the_encoders_and_is_built_without_an_array(hw):
    assert empty_rle(hw) == masks.encode_rle(np.zeros(hw, dtype=bool))
    assert masks.rle_area(empty_rle(hw)) == 0


def test_the_empty_mask_of_a_12mp_frame_costs_nothing():
    empty_rle((3041, 4033))                     # a size nobody asked for yet
    started = time.perf_counter()
    for _ in range(40):
        empty_rle((3041, 4033))
    assert time.perf_counter() - started < 0.01


def test_ctrl_i_reads_the_shapes_once_and_encodes_no_frame(drawn, monkeypatch):
    session = drawn
    reads, encodes = [], []
    real_keyframes, real_encode = session.db.keyframes, masks.encode_rle
    monkeypatch.setattr(session.db, "keyframes",
                        lambda *a, **k: reads.append(a) or real_keyframes(*a, **k))
    monkeypatch.setattr(masks, "encode_rle",
                        lambda m, *a, **k: encodes.append(np.asarray(m).shape)
                        or real_encode(m, *a, **k))
    from tda.ui import session_cable as SC

    result = SC.cable_range(session.db, session.truth, session.current(), CHASSIS, FULL, 1,
                            "tester")
    assert result["frames"] == list(range(1, STEP + 1))
    assert len([a for a in reads if CHASSIS in a]) <= 1
    assert result["measured"] == 0                      # the scene's frames are measured


# --------------------------------------------------------------------------- #
# M8: a confirmed frame stays confirmed, both ways
# --------------------------------------------------------------------------- #
def open_conflicts(session) -> list[dict]:
    return [dict(c) for c in session.db.conflicts(DESKTOP, VIEW, open_only=True)]


def test_i_on_a_confirmed_frame_keeps_it_confirmed_and_ctrl_z_too(drawn):
    session = drawn
    db = session.db
    key = session.current()
    assert session.confirm_frame()
    session.goto(STEP, force=True)
    before = db.compiled(key)[CHASSIS]
    assert before["status"] == "verified" and db.get_frame(key)["review_status"] == "verified"

    result = session.cycle_cable(CHASSIS)
    assert result["accepted"] == [STEP]
    assert open_conflicts(session) == []
    row = db.compiled(key)[CHASSIS]
    assert row["status"] == "verified" and row["visibility"] == FULL
    assert masks.rle_area(row["visible_rle"]) == 0 and row["verified_by"] == "tester"
    assert override(session).occluder == OCCLUDER_CABLE
    assert db.get_frame(key)["review_status"] == "verified"
    assert session.frame_status(STEP) == "verified"

    assert session.undo()
    assert open_conflicts(session) == []
    back = db.compiled(key)[CHASSIS]
    assert back["status"] == "verified" and back["visibility"] == before["visibility"]
    assert masks.rle_counts(back["visible_rle"]) == masks.rle_counts(before["visible_rle"])
    assert db.get_frame(key)["review_status"] == "verified"
    assert session.redo()
    assert open_conflicts(session) == [] and db.compiled(key)[CHASSIS]["visibility"] == FULL
    # Space has nothing to wait for
    assert session.confirm_frame()


def test_ctrl_i_over_confirmed_frames_keeps_them_confirmed(tmp_path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    seed_shapes(session, 1)                  # every part of the scene, drawn to 14
    db = session.db
    for step in (STEP, STEP - 1, STEP - 2):
        session.goto(step, force=True)
        assert session.confirm_frame()
    session.goto(STEP, force=True)
    result = session.set_cable_range(CHASSIS, FULL, STEP - 2)
    assert sorted(result["accepted"]) == [STEP - 2, STEP - 1, STEP]
    assert open_conflicts(session) == []
    for step in (STEP - 2, STEP - 1, STEP):
        row = db.compiled(FrameKey(DESKTOP, step, VIEW))[CHASSIS]
        assert row["status"] == "verified" and row["visibility"] == FULL
    assert [op["kind"] for op in db.ops(DESKTOP, VIEW, limit=50)].count(OP_RANGE) == 1
    assert session.undo()
    assert open_conflicts(session) == []
    assert all(db.compiled(FrameKey(DESKTOP, s, VIEW))[CHASSIS]["visibility"] != FULL
               for s in (STEP - 2, STEP - 1, STEP))


def _seed_all(session, steps) -> None:
    """The reviewer's scene (p10): every part of 12-14 its own cell, anchored at its last."""
    from session_scene import cell, chassis_instances
    from tda.core.model import ZOrderRec

    keys = sorted({k for st in steps for k in chassis_instances(session, st)})
    order = []
    for index, key in enumerate(keys):
        anchor = max(st for st in steps if key in chassis_instances(session, st))
        session.db.add_keyframe(ShapeKeyframe(
            id=None, instance=key, desktop=DESKTOP, view=VIEW, pose_segment=1,
            anchor_step=anchor, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(cell(index)))]))
        order.append((key, "main"))
    session.db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, order))
    session.refresh_all()


@pytest.mark.parametrize("label", [PARTIAL, FULL])
def test_a_retrace_waiting_for_its_recheck_is_not_signed_by_the_label(tmp_path, label):
    """Review round 2, B1 (p10c / p10f): 12-14 confirmed; the chassis re-traced at 11
    queues 12/13/14 for a re-check; Ctrl+I from 14 to 12 before the sweeper runs. The
    re-trace (36 px -> 400 px) is somebody's question for Review: it opens its
    conflicts first, the label signs nothing on those frames, and is still written."""
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    db = session.db
    _seed_all(session, (12, 13, 14))
    for step in (14, 13, 12):
        session.goto(step, force=True)
        assert session.confirm_frame(), step
    session.goto(11, force=True)
    session.begin_edit(CHASSIS)
    session.set_editing_mask(rect(10, 40, 30, 60))
    queued = session.commit_edit("keyframe")["rechecks"]
    assert {12, 13, 14} <= set(queued) | set(db.rechecks(DESKTOP, VIEW))
    session.goto(STEP, force=True)                 # arriving compiles 14: its conflict
    assert [c["step"] for c in open_conflicts(session)] == [STEP]
    frozen = {s: db.compiled(FrameKey(DESKTOP, s, VIEW))[CHASSIS] for s in (12, 13, 14)}

    result = session.set_cable_range(CHASSIS, label, 12)
    assert result["accepted"] == []
    assert sorted(int(c["step"]) for c in open_conflicts(session)
                  if c["instance"] == CHASSIS) == [12, 13, 14]
    for step in (12, 13, 14):
        row = db.compiled(FrameKey(DESKTOP, step, VIEW))[CHASSIS]
        assert row == frozen[step], step                   # nothing signed
        assert label_at(session, step) == label            # the label is written
    resolved = [c for c in db.conflicts(DESKTOP, VIEW, open_only=False)
                if c["status"] != "open" and c.get("resolution") == "accept_new"]
    assert resolved == []
    # 12 and 13 were compiled just now: not shown as waiting for a re-check
    assert not {12, 13, 14} & set(db.rechecks(DESKTOP, VIEW))
    assert all(session.frame_status(s) == "conflict" for s in (12, 13, 14))
    # the undo goes the same way: nothing signed, the conflicts still Review's
    assert session.undo()
    assert all(label_at(session, s) is None for s in (12, 13, 14))
    assert all(db.compiled(FrameKey(DESKTOP, s, VIEW))[CHASSIS] == frozen[s]
               for s in (12, 13, 14))
    assert sorted(int(c["step"]) for c in open_conflicts(session)) == [12, 13, 14]
    assert not {12, 13, 14} & set(db.rechecks(DESKTOP, VIEW))


def test_a_signed_frame_is_not_left_waiting_for_a_recheck(tmp_path):
    """The plain case over several frames: re-signed, and off the re-check queue."""
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    db = session.db
    _seed_all(session, (12, 13, 14))
    for step in (14, 13, 12):
        session.goto(step, force=True)
        assert session.confirm_frame(), step
    session.goto(STEP, force=True)
    result = session.set_cable_range(CHASSIS, FULL, 12)
    assert sorted(result["accepted"]) == [12, 13, 14] and result["rechecks"] == []
    assert open_conflicts(session) == [] and db.rechecks(DESKTOP, VIEW) == []
    assert all(session.frame_status(s) == "verified" for s in (12, 13, 14))
    assert session.undo()
    assert open_conflicts(session) == [] and db.rechecks(DESKTOP, VIEW) == []
    assert all(session.frame_status(s) == "verified" for s in (12, 13, 14))


def test_the_digit_keys_still_ask_review(drawn):
    session = drawn
    assert session.confirm_frame()
    session.goto(STEP, force=True)
    session.set_visibility(CHASSIS, FULL)
    assert [c["instance"] for c in open_conflicts(session)] == [CHASSIS]


def test_a_conflict_already_open_is_not_answered_by_the_label(drawn):
    """Somebody else's question about the part stays Review's."""
    session = drawn
    db = session.db
    key = session.current()
    assert session.confirm_frame()
    session.goto(STEP, force=True)
    kf = db.keyframes(DESKTOP, VIEW, CHASSIS)[0]
    kf.parts = [ShapePart("main", masks.encode_rle(rect(0, 0, 40, 40)))]
    db.update_keyframe(kf)
    session.truth.refresh(key)
    assert [c["instance"] for c in open_conflicts(session)] == [CHASSIS]
    result = session.cycle_cable(CHASSIS)
    assert result["accepted"] == []
    assert [c["instance"] for c in open_conflicts(session)] == [CHASSIS]
