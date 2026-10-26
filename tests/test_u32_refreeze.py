"""Task U32 -- 「重签」: re-sign confirmed rows that trail today's compile within the tolerance.

A confirmed frame's rows are frozen.  An input change that moves one **beyond**
the re-tracing tolerance opens a conflict; one **within** it opened nothing and
the row kept its old pixels for ever.  Two consequences were found on the real
data, and both are rebuilt here on a 200 x 200 scene (a motherboard, a cover
and a screw, three confirmed frames):

* **an overlap**: the cover is re-traced 10 px wider -- a conflict on the
  cover, accepted -- while the motherboard loses the same 600 px, under 2 % of
  its area: its frozen row keeps them, and two confirmed masks share 600 px on
  every frame (D2 ``cover.01`` / ``motherboard.01``);
* **a hole**: the layer order is corrected so the cover lies over the screw --
  a conflict on the screw, accepted -- while the cover gains the screw's
  400 px, under 2 % of it: nobody's frozen mask holds them (U31 section 7).

:meth:`tda.core.truth.TruthService.refreeze` writes the compile into exactly
the rows within the tolerance; a row beyond it stays a conflict; the frame
stays confirmed; one op per frame whose inverse puts the rows back byte for
byte; a second run changes nothing; ``cli check`` counts the rows
(``stale_within_tolerance``); ``cli refreeze --dry-run`` writes nothing; and
Review's two buttons plan, ask (naming the frame count) and re-sign.
"""
from __future__ import annotations

import hashlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
import yaml

from tda.core import masks
from tda.core.db import Db
from tda.core.model import (
    FrameKey,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    StepRec,
    ZOrderRec,
)
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_conflicts import row_values
from tda.core.truth_refreeze import OP_KIND, UNDO_KIND, overlap_px, plan_frame

DESKTOP = 1
VIEW = "scan"
HW = (200, 200)
STEPS = (1, 2, 3)
BOARD = "motherboard.01"
COVER = "cover.01"
SCREW = "screw.motherboard.01"
ANNOTATOR = "tester"

BOARD_RECT = (10, 10, 190, 190)          # 32 400 px: 2 % is 648
SMALL_COVER = (20, 20, 80, 80)           # 3 600 px: 2 % is 72
WIDER_COVER = (20, 20, 90, 80)           # +600 px, all of it the board's
FAR_SCREW = (150, 150, 170, 170)
BIG_COVER = (20, 20, 170, 170)           # 22 500 px: 2 % is 450
SCREW_UNDER = (40, 40, 60, 60)           # 400 px


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #
def rect(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros(HW, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _kf(instance: str, box) -> ShapeKeyframe:
    return ShapeKeyframe(id=None, instance=instance, desktop=DESKTOP, view=VIEW,
                         pose_segment=1, anchor_step=STEPS[-1], placement="in_chassis",
                         geom_type="mask", parts=[ShapePart("main", masks.encode_rle(rect(*box)))])


def build(db: Db, cover=SMALL_COVER, screw=FAR_SCREW,
          order=(BOARD, SCREW, COVER)) -> dict[str, ShapeKeyframe]:
    """Three confirmed-able frames of a board, a cover and a screw; returns the keyframes."""
    db.upsert_instance(InstanceRec(key=BOARD, desktop=DESKTOP, cls="motherboard"))
    db.upsert_instance(InstanceRec(key=COVER, desktop=DESKTOP, cls="cover"))
    db.upsert_instance(InstanceRec(key=SCREW, desktop=DESKTOP, cls="screw",
                                   attrs={"role": "motherboard", "head": "PH2",
                                          "captive": False},
                                   parent=BOARD, fastens=BOARD))
    db.replace_steps(DESKTOP, [StepRec(DESKTOP, 1, "initial", "initial state"),
                               StepRec(DESKTOP, 2, "normal", "look"),
                               StepRec(DESKTOP, 3, "normal", "look again")], [])
    for step in STEPS:
        db.upsert_frame(FrameKey(DESKTOP, step, VIEW), f"s{step:03d}.jpg",
                        {"hw": [HW[0], HW[1]]}, None)
    db.set_pose_segment(DESKTOP, VIEW, 1, 1, STEPS[-1], STEPS[-1], None, None)
    kfs = {BOARD: _kf(BOARD, BOARD_RECT), COVER: _kf(COVER, cover), SCREW: _kf(SCREW, screw)}
    for kf in kfs.values():
        db.add_keyframe(kf)
    set_order(db, order)
    return kfs


def set_order(db: Db, order, version: int = 1) -> None:
    db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, [(i, "main") for i in order], version=version))


def key(step: int) -> FrameKey:
    return FrameKey(DESKTOP, step, VIEW)


def confirm_all(svc: TruthService) -> None:
    for step in STEPS:
        svc.verify_frame(key(step), ANNOTATOR)


def retrace(db: Db, kf: ShapeKeyframe, box) -> None:
    kf.parts = [ShapePart("main", masks.encode_rle(rect(*box)))]
    db.update_keyframe(kf)


def accept_all(db: Db, svc: TruthService, instance: str) -> int:
    done = 0
    for conflict in db.conflicts(DESKTOP, VIEW, open_only=True):
        if conflict["instance"] == instance:
            svc.resolve_conflict(int(conflict["id"]), "accept_new", ANNOTATOR)
            done += 1
    return done


def frozen_masks(db: Db, step: int) -> dict[str, dict]:
    return {i: r["visible_rle"] for i, r in db.compiled(key(step)).items()
            if r["status"] == "verified" and r["visible_rle"] is not None}


def owner_count(db: Db, step: int, x: int, y: int) -> int:
    """How many frozen visible masks hold pixel ``(x, y)``."""
    return sum(1 for rle in frozen_masks(db, step).values() if masks.rle_contains(rle, x, y))


def ops(db: Db, kind: str) -> list[dict]:
    return db.ops_of_kind(DESKTOP, kind)


@pytest.fixture
def db(tmp_path: Path):
    made = Db(str(tmp_path / "tda.sqlite"))
    yield made
    made.close()


@pytest.fixture
def svc(db: Db) -> TruthService:
    return TruthService(db, load_taxonomy())


def overlap_scene(db: Db, svc: TruthService, accept: bool = True) -> dict:
    """The D2 case: the cover re-traced 10 px wider over the board, the cover's conflicts
    accepted (``accept``); the board's frozen rows still hold the 600 px."""
    kfs = build(db)
    svc.refresh_range(DESKTOP, VIEW, STEPS)
    confirm_all(svc)
    retrace(db, kfs[COVER], WIDER_COVER)
    stats = svc.refresh_range(DESKTOP, VIEW, STEPS)
    assert stats["conflicts"] == 3                      # the cover, beyond the tolerance
    assert {c["instance"] for c in db.conflicts(DESKTOP, VIEW)} == {COVER}
    if accept:
        assert accept_all(db, svc, COVER) == 3
    return kfs


# --------------------------------------------------------------------------- #
# consequence 1: an overlap
# --------------------------------------------------------------------------- #
def test_the_overlap_a_retrace_within_tolerance_left_is_removed(db, svc):
    overlap_scene(db, svc)
    for step in STEPS:
        assert overlap_px(frozen_masks(db, step).values()) == 600
        assert owner_count(db, step, 85, 50) == 2          # the cover's new strip ...
    plan = svc.refreeze(DESKTOP, VIEW, dry_run=True, measure=True)
    assert plan["rewritten"] == 3 and plan["frames_rewritten"] == 3
    assert [f.rows for f in plan["frames"]] == [[BOARD]] * 3   # not the screw, not the cover
    assert {(f.overlap_before, f.overlap_after) for f in plan["frames"]} == {(600, 0)}

    done = svc.refreeze(DESKTOP, VIEW, annotator="controller", measure=True)

    assert done["rewritten"] == 3 and done["px"] == 3 * 600
    assert (done["overlap_before"], done["overlap_after"]) == (1800, 0)
    for step in STEPS:
        assert overlap_px(frozen_masks(db, step).values()) == 0
        assert owner_count(db, step, 85, 50) == 1          # ... is the cover's alone now
        compiled = svc.compile(key(step))
        row = db.compiled(key(step))[BOARD]
        assert row["visible_rle"]["counts"] == row_values(compiled.instances[BOARD]).visible_rle["counts"]
        assert row["input_hash"] == compiled.input_hash
        assert row["status"] == "verified" and row["verified_by"] == "controller"
        assert db.get_frame(key(step))["review_status"] == "verified"
    assert len(ops(db, OP_KIND)) == 3                      # one op per frame
    assert ops(db, OP_KIND)[0]["annotator"] == "controller"
    # every frozen row is this compilation now: nothing for a refresh to do
    again = svc.refresh_range(DESKTOP, VIEW, STEPS)
    assert again["conflicts"] == 0 and again["updated"] == 0


# --------------------------------------------------------------------------- #
# consequence 2: a hole after a layer-order move
# --------------------------------------------------------------------------- #
def test_the_hole_a_layer_order_move_left_is_filled(db, svc):
    build(db, cover=BIG_COVER, screw=SCREW_UNDER, order=(BOARD, COVER, SCREW))
    svc.refresh_range(DESKTOP, VIEW, STEPS)
    confirm_all(svc)
    set_order(db, (BOARD, SCREW, COVER), version=2)      # the cover lies over the screw
    stats = svc.refresh_range(DESKTOP, VIEW, STEPS)
    assert {c["instance"] for c in db.conflicts(DESKTOP, VIEW)} == {SCREW}
    assert stats["conflicts"] == 3
    assert accept_all(db, svc, SCREW) == 3
    for step in STEPS:
        assert owner_count(db, step, 50, 50) == 0          # the screw's 400 px: nobody's
        assert masks.rle_area(db.compiled(key(step))[SCREW]["visible_rle"]) == 0

    done = svc.refreeze(DESKTOP, VIEW, measure=True)

    assert done["rewritten"] == 3 and [f.rows for f in done["frames"]] == [[COVER]] * 3
    assert done["px"] == 3 * 400
    for step in STEPS:
        assert owner_count(db, step, 50, 50) == 1          # the cover's now
        frozen = frozen_masks(db, step)
        union = np.zeros(HW, dtype=bool)
        for rle in frozen.values():
            union |= masks.decode_rle(rle)
        assert union.sum() == rect(*BOARD_RECT).sum()      # no hole, no gap
        assert overlap_px(frozen.values()) == 0
        assert db.get_frame(key(step))["review_status"] == "verified"


# --------------------------------------------------------------------------- #
# beyond the tolerance: a conflict, and it stays one
# --------------------------------------------------------------------------- #
def test_a_row_beyond_the_tolerance_is_left_alone_and_stays_a_conflict(db, svc):
    kfs = build(db)
    svc.refresh_range(DESKTOP, VIEW, STEPS)
    confirm_all(svc)
    retrace(db, kfs[COVER], WIDER_COVER)          # not refreshed: nothing queued yet
    before = {s: db.compiled_raw(key(s)) for s in STEPS}
    digests = {s: db.frame_digest(key(s)) for s in STEPS}

    done = svc.refreeze(DESKTOP, VIEW)

    for step, plan in zip(STEPS, done["frames"]):
        assert plan.disputed == [COVER] and plan.stands
        # the board's strip is what the cover's conflict is about: re-signed now,
        # a keep_old on the cover would leave it nobody's -- it waits
        assert plan.rows == [] and plan.deferred == [BOARD]
        assert db.compiled_raw(key(step)) == before[step]                 # byte for byte
        assert db.frame_digest(key(step)) == digests[step]                # nothing stamped
    assert done["deferred"] == 3 and done["ops"] == []
    # ... and the refresh still raises it
    stats = svc.refresh_range(DESKTOP, VIEW, STEPS)
    assert stats["conflicts"] == 3
    assert {c["instance"] for c in db.conflicts(DESKTOP, VIEW)} == {COVER}
    # with the conflict open, a second run leaves it to Review as well
    again = svc.refreeze(DESKTOP, VIEW)
    assert again["rewritten"] == 0 and again["open_conflicts"] == 3 and again["deferred"] == 3
    assert all(db.compiled_raw(key(s))[COVER] == before[s][COVER] for s in STEPS)
    assert len(db.conflicts(DESKTOP, VIEW, open_only=True)) == 3
    # settled: the board follows
    assert accept_all(db, svc, COVER) == 3
    last = svc.refreeze(DESKTOP, VIEW, measure=True)
    assert last["rewritten"] == 3 and last["deferred"] == 0
    assert last["overlap_after"] == 0


def test_a_row_far_from_a_conflict_is_resigned_beside_it(db, svc):
    """A disputed part elsewhere on the frame holds back only what touches it."""
    kfs = build(db)
    svc.refresh_range(DESKTOP, VIEW, STEPS)
    confirm_all(svc)
    retrace(db, kfs[SCREW], (150, 150, 171, 170))   # the screw: 20 px more, within its band
    retrace(db, kfs[COVER], WIDER_COVER)             # the cover: beyond
    done = svc.refreeze(DESKTOP, VIEW, dry_run=True)
    for plan in done["frames"]:
        assert plan.disputed == [COVER] and plan.deferred == [BOARD]
        assert plan.rows == [SCREW]


def test_a_part_the_frame_gained_is_reported_not_signed(db, svc):
    overlap_scene(db, svc)
    db.delete_compiled(key(2), SCREW)            # as if the screw were new on this frame
    plan = svc.refreeze(DESKTOP, VIEW, steps=[2], dry_run=True)
    [frame] = plan["frames"]
    assert frame.missing == [SCREW] and frame.stands
    svc.refreeze(DESKTOP, VIEW, steps=[2])
    assert SCREW not in db.compiled(key(2))                  # not invented
    assert db.compiled(key(2))[BOARD]["input_hash"] == svc.compile(key(2)).input_hash


def test_only_confirmed_frames_are_touched(db, svc):
    overlap_scene(db, svc)
    db.set_frame_flags(key(2), review_status="needs_review")
    done = svc.refreeze(DESKTOP, VIEW)
    assert [f.step for f in done["frames"]] == [1, 3]


# --------------------------------------------------------------------------- #
# the op, its undo, idempotence
# --------------------------------------------------------------------------- #
def test_undo_through_the_inverse_restores_the_rows_byte_for_byte(db, svc):
    overlap_scene(db, svc)
    raw = {s: db.compiled_raw(key(s)) for s in STEPS}
    digests = {s: db.frame_digest(key(s)) for s in STEPS}
    done = svc.refreeze(DESKTOP, VIEW)
    assert len(done["ops"]) == 3
    assert all(db.compiled_raw(key(s)) != raw[s] for s in STEPS)
    op = db.get_op(done["ops"][0])
    assert op["kind"] == OP_KIND and op["inverse"]["kind"] == "restore_rows"
    assert set(op["inverse"]["rows"]) == {BOARD}

    for op_id in reversed(done["ops"]):
        assert svc.undo_refreeze(op_id, ANNOTATOR) is not None

    for step in STEPS:
        assert db.compiled_raw(key(step)) == raw[step]          # every column, every row
        assert db.frame_digest(key(step)) == digests[step]
        assert db.get_frame(key(step))["review_status"] == "verified"
    assert len(ops(db, UNDO_KIND)) == 3
    # undone already: nothing written
    assert svc.undo_refreeze(done["ops"][0], ANNOTATOR) is None
    assert len(ops(db, UNDO_KIND)) == 3
    # and the undo is an op too: its inverse puts the re-signed rows back
    redo = ops(db, UNDO_KIND)[-1]
    assert svc.undo_refreeze(int(redo["id"]), ANNOTATOR) is not None
    assert db.compiled(key(1))[BOARD]["input_hash"] == svc.compile(key(1)).input_hash


def test_undo_refuses_a_frame_somebody_changed_since(db, svc):
    overlap_scene(db, svc)
    done = svc.refreeze(DESKTOP, VIEW, steps=[1])
    row = dict(db.compiled_raw(key(1))[BOARD], verified_by="somebody else")
    db.put_compiled_raw(key(1), BOARD, row)
    with pytest.raises(ValueError, match="changed after op"):
        svc.undo_refreeze(done["ops"][0], ANNOTATOR)
    assert db.compiled_raw(key(1))[BOARD] == row


def test_refreeze_is_idempotent(db, svc):
    overlap_scene(db, svc)
    first = svc.refreeze(DESKTOP, VIEW)
    assert first["rewritten"] == 3
    count = len(db.ops(DESKTOP, VIEW, limit=10_000))
    raw = {s: db.compiled_raw(key(s)) for s in STEPS}

    second = svc.refreeze(DESKTOP, VIEW)

    assert second["rewritten"] == 0 and second["ops"] == []
    assert len(db.ops(DESKTOP, VIEW, limit=10_000)) == count
    assert {s: db.compiled_raw(key(s)) for s in STEPS} == raw


def test_a_plan_is_written_as_planned_and_replanned_when_the_rows_moved(db, svc):
    overlap_scene(db, svc)
    plan = svc.refreeze(DESKTOP, VIEW, dry_run=True)
    # frame 1's board row moves under the plan (as an accept_new would)
    compiled = svc.compile(key(1))
    values = row_values(compiled.instances[BOARD])
    db.put_compiled(key(1), BOARD, values.visible_rle, values.occlusion_ratio,
                    values.visibility, values.placement, "verified", compiled.input_hash,
                    verified_by="somebody")
    done = svc.refreeze(DESKTOP, VIEW, plans=plan["frames"])
    assert [f.rows for f in done["frames"]] == [[], [BOARD], [BOARD]]
    assert db.compiled(key(1))[BOARD]["verified_by"] == "somebody"   # left as it was


# --------------------------------------------------------------------------- #
# round 2, B1: re-signing again and again does not walk away from a signature
# --------------------------------------------------------------------------- #
def set_screw(db: Db, kf: ShapeKeyframe, columns: int) -> None:
    """The far screw, ``columns`` columns wider than confirmed (3 columns: at the threshold)."""
    retrace(db, kf, (150, 150, 170 + columns, 170))


def screw_drift(db: Db, original: np.ndarray, step: int = 2) -> int:
    """Tolerant difference between what is stored for the screw and what was confirmed."""
    stored = masks.decode_rle(db.compiled(key(step))[SCREW]["visible_rle"])
    return masks.tolerant_sym_diff(original, stored)


def test_repeated_resigning_does_not_drift(db, svc):
    """The review's ratchet: five re-traces of 3 columns each, a refresh and a re-sign after each.

    Before round 2 the stored screw went 400 -> 700 px with no conflict at
    all (260 px of tolerant difference from what was confirmed, 13 times the
    threshold).  Now the stored row stays within one tolerance of the
    confirmed one and the conflict opens in the second round.
    """
    kfs = build(db)
    svc.refresh_range(DESKTOP, VIEW, STEPS)
    confirm_all(svc)
    original = masks.decode_rle(db.compiled(key(2))[SCREW]["visible_rle"]).copy()
    threshold = max(0.02 * original.sum(), 20)
    per_round, resigned = [], []
    for i in range(1, 6):
        set_screw(db, kfs[SCREW], 3 * i)
        per_round.append(svc.refresh_range(DESKTOP, VIEW, STEPS)["conflicts"])
        done = svc.refreeze(DESKTOP, VIEW)
        resigned.append(sum(SCREW in f.rows for f in done["frames"]))
        assert screw_drift(db, original) <= threshold          # at most one tolerance, unseen
    assert per_round[0] == 0 and resigned[0] == 3              # round 1: within, re-signed
    assert per_round[1] == 3                                   # round 2: the conflict opens
    assert resigned[1:] == [0, 0, 0, 0]
    opened = [c for c in db.conflicts(DESKTOP, VIEW, open_only=True) if c["instance"] == SCREW]
    assert len(opened) == 3                                    # one per frozen row (U7)


@pytest.mark.parametrize("past", [False, True])
def test_the_threshold_is_the_refreshs_own(db, svc, past):
    """The review's 3.3: at the threshold re-signed, one pixel past it a conflict, as refresh."""
    kfs = build(db)
    svc.refresh_range(DESKTOP, VIEW, STEPS)
    confirm_all(svc)
    shape = rect(150, 150, 173, 170)
    if past:
        shape[160, 176] = True
    kfs[SCREW].parts = [ShapePart("main", masks.encode_rle(shape))]
    db.update_keyframe(kfs[SCREW])
    before = {s: db.compiled_raw(key(s))[SCREW] for s in STEPS}
    done = svc.refreeze(DESKTOP, VIEW)
    for plan in done["frames"]:
        assert (SCREW in plan.disputed) is past and (SCREW in plan.rows) is not past
    if past:
        assert all(db.compiled_raw(key(s))[SCREW] == before[s] for s in STEPS)
    assert svc.refresh_range(DESKTOP, VIEW, STEPS)["conflicts"] == (3 if past else 0)


def test_without_resigning_the_same_edits_raise_a_conflict(db, svc):
    """The review's control: the same five edits, no re-signing -- conflicts as before."""
    kfs = build(db)
    svc.refresh_range(DESKTOP, VIEW, STEPS)
    confirm_all(svc)
    conflicts = 0
    for i in range(1, 6):
        set_screw(db, kfs[SCREW], 3 * i)
        conflicts += svc.refresh_range(DESKTOP, VIEW, STEPS)["conflicts"]
    assert conflicts >= 3


def test_a_drifted_row_is_listed_and_left_and_a_person_resets_the_anchor(db, svc):
    kfs = build(db)
    svc.refresh_range(DESKTOP, VIEW, STEPS)
    confirm_all(svc)
    set_screw(db, kfs[SCREW], 3)
    first = svc.refreeze(DESKTOP, VIEW)
    assert [f.rows for f in first["frames"]] == [[BOARD, SCREW]] * 3   # the board gives way
    set_screw(db, kfs[SCREW], 6)                  # no refresh yet: nothing is queued
    before = {s: db.compiled_raw(key(s))[SCREW] for s in STEPS}
    plan = svc.refreeze(DESKTOP, VIEW, dry_run=True)
    assert [f.drifted for f in plan["frames"]] == [[SCREW]] * 3 and plan["drifted"] == 3
    assert all(f.stands and SCREW not in f.rows for f in plan["frames"])
    svc.refreeze(DESKTOP, VIEW)
    assert {s: db.compiled_raw(key(s))[SCREW] for s in STEPS} == before
    # the ordinary refresh raises it -- measured against what the person signed
    assert svc.refresh_range(DESKTOP, VIEW, STEPS)["conflicts"] == 3
    assert {c["instance"] for c in db.conflicts(DESKTOP, VIEW, open_only=True)} == {SCREW}
    # accepting it is a person's signature: the next small move re-signs again
    assert accept_all(db, svc, SCREW) == 3
    set_screw(db, kfs[SCREW], 9)
    assert svc.refresh_range(DESKTOP, VIEW, STEPS)["conflicts"] == 0
    last = svc.refreeze(DESKTOP, VIEW)
    assert all(SCREW in f.rows for f in last["frames"])


def test_a_label_crossing_seen_and_unseen_is_listed(db, svc):
    """Review N3: a sliver the cover now hides -- within the band, but the label flips."""
    kfs = build(db, cover=(20, 20, 168, 172), screw=FAR_SCREW, order=(BOARD, SCREW, COVER))
    svc.refresh_range(DESKTOP, VIEW, STEPS)
    confirm_all(svc)
    shown = db.compiled(key(1))[SCREW]["visibility"]
    retrace(db, kfs[COVER], (20, 20, 170, 172))   # the last two screw columns covered too
    assert svc.refresh_range(DESKTOP, VIEW, STEPS)["conflicts"] == 0
    plan = svc.refreeze(DESKTOP, VIEW, dry_run=True)
    assert shown != "occluded_full"
    assert plan["flips"] == [(s, SCREW, shown, "occluded_full") for s in STEPS]
    from tda.cli_refreeze import frame_line
    from tda.ui.app_refreeze import refreeze_question

    assert f"label crosses seen/unseen: {SCREW} {shown} -> occluded_full" \
        in frame_line(plan["frames"][0])
    assert any(line.startswith("标签在看得见 / 看不见之间翻转") and
               f"第 1 帧 {SCREW}：{shown} → occluded_full" in line
               for line in refreeze_question(plan, (DESKTOP, VIEW)))


def test_overlap_counts_every_pair_once():
    a, b, c = (masks.encode_rle(rect(*box)) for box in
               ((0, 0, 10, 10), (5, 0, 15, 10), (8, 0, 20, 10)))
    # a&b 50, b&c 70, a&c 20 -- the 2 x 10 px under all three count three pairs
    assert overlap_px([a, b, c]) == 50 + 70 + 20
    assert overlap_px([a, None, masks.encode_rle(np.zeros(HW, dtype=bool))]) == 0


def test_plan_frame_is_what_cli_check_counts(db, svc):
    overlap_scene(db, svc)
    compiled = svc.compile(key(2))
    assert [c.instance for c in plan_frame(db, key(2), compiled).changes] == [BOARD]


# --------------------------------------------------------------------------- #
# the command line
# --------------------------------------------------------------------------- #
def _make_env(tmp_path: Path, accept: bool = True) -> dict:
    db_path = tmp_path / "annotations" / "tda.sqlite"
    db_path.parent.mkdir(parents=True)
    (tmp_path / "backups").mkdir()
    cfg = {"cache_dir": str(tmp_path / "cache"), "db_path": str(db_path),
           "backup_dir": str(tmp_path / "backups"), "raw_logs_dir": str(tmp_path / "raw")}
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    made = Db(str(db_path))
    try:
        overlap_scene(made, TruthService(made, load_taxonomy()), accept=accept)
    finally:
        made.close()
    return {"paths": str(tmp_path / "paths.yaml"), "db": db_path,
            "backups": tmp_path / "backups"}


@pytest.fixture
def env(tmp_path: Path) -> dict:
    """The overlap scene (the cover's conflicts accepted) behind a paths.yaml."""
    return _make_env(tmp_path)


@pytest.fixture
def env_open(tmp_path: Path) -> dict:
    """The same with the cover's three conflicts still open."""
    return _make_env(tmp_path, accept=False)


def run(env: dict, *argv: str) -> int:
    from tda.cli import main

    return main(["--paths", env["paths"], *argv])


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_cli_check_counts_the_rows_refreeze_would_rewrite(env, capsys):
    from tda.cli_common import EXIT_OK

    assert run(env, "check", "--desktop", str(DESKTOP), "--view", VIEW) == EXIT_OK
    out = capsys.readouterr().out
    assert "motherboard.01@1" in out and "cli refreeze" in out
    assert out.strip().splitlines()[-1] == "stale_within_tolerance: 3 (deferred 0)"

    assert run(env, "refreeze", "--desktop", str(DESKTOP)) == EXIT_OK
    capsys.readouterr()
    assert run(env, "check", "--desktop", str(DESKTOP), "--view", VIEW) == EXIT_OK
    assert capsys.readouterr().out.strip().splitlines()[-1] == \
        "stale_within_tolerance: 0 (deferred 0)"


def test_cli_check_counts_the_rows_waiting_for_a_conflict_apart(env_open, capsys):
    """Review N1: the board waits for the cover's open conflicts -- counted, apart."""
    from tda.cli_common import EXIT_ERROR

    assert run(env_open, "check", "--desktop", str(DESKTOP), "--view", VIEW) == EXIT_ERROR
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "stale_within_tolerance: 0 (deferred 3)"
    assert "open conflicts: 3" in out
    assert "另有 3 条同样落后、但挨着开着的冲突" in out and "motherboard.01@1" in out


def test_cli_dry_run_writes_nothing(env, capsys):
    from tda.cli_common import EXIT_OK

    before = sha(env["db"])
    assert run(env, "refreeze", "--desktop", str(DESKTOP), "--dry-run") == EXIT_OK
    out = capsys.readouterr().out
    assert sha(env["db"]) == before
    assert not list(env["backups"].glob("*.sqlite"))
    assert "would re-sign 3 rows on 3 frames" in out
    assert "  step 1: 1 rows re-signed (motherboard.01 600), 600 px changed; " \
           "overlap 600 -> 0 px (600 removed)" in out
    assert "dry run: nothing was written" in out
    assert not Path(str(env["db"]) + ".lock").exists()      # the lock was let go


def test_cli_run_backs_up_then_undo_last_puts_it_back(env, capsys):
    from tda.cli_common import EXIT_OK

    made = Db(str(env["db"]))
    raw = {s: made.compiled_raw(key(s)) for s in STEPS}
    made.close()

    assert run(env, "refreeze", "--desktop", str(DESKTOP), "--view", VIEW) == EXIT_OK
    out = capsys.readouterr().out
    assert "backed the database up first" in out
    assert list(env["backups"].glob("tda_*.sqlite"))
    assert "re-signed 3 rows on 3 frames" in out and "undo with --undo last" in out
    made = Db(str(env["db"]))
    assert [op["annotator"] for op in ops(made, OP_KIND)] == ["cli:refreeze"] * 3
    made.close()

    assert run(env, "refreeze", "--desktop", str(DESKTOP), "--undo", "last") == EXIT_OK
    assert "3 frame(s) put back" in capsys.readouterr().out
    made = Db(str(env["db"]))
    try:
        assert {s: made.compiled_raw(key(s)) for s in STEPS} == raw
    finally:
        made.close()


def test_cli_undo_dry_run_lists_and_writes_nothing(env, capsys):
    """Review B2: ``--undo last --dry-run`` used to undo -- without the backup."""
    from tda.cli_common import EXIT_ERROR, EXIT_OK

    assert run(env, "refreeze", "--desktop", str(DESKTOP), "--view", VIEW) == EXIT_OK
    capsys.readouterr()
    backups = len(list(env["backups"].glob("*.sqlite")))
    before = sha(env["db"])

    assert run(env, "refreeze", "--desktop", str(DESKTOP), "--view", VIEW,
               "--undo", "last", "--dry-run") == EXIT_OK
    out = capsys.readouterr().out
    assert sha(env["db"]) == before
    assert len(list(env["backups"].glob("*.sqlite"))) == backups
    for step in STEPS:
        assert f"  step {step}: would put back motherboard.01" in out
    assert "3 frame(s) would put back, 0 already undone, 0 refused" in out
    assert "dry run: nothing was written" in out
    # an undo takes back a whole run: --steps is refused, nothing written
    assert run(env, "refreeze", "--desktop", str(DESKTOP), "--undo", "last",
               "--steps", "1") == EXIT_ERROR
    assert "takes no --steps" in capsys.readouterr().out
    assert sha(env["db"]) == before
    # the real undo still works afterwards
    assert run(env, "refreeze", "--desktop", str(DESKTOP), "--undo", "last") == EXIT_OK
    assert "3 frame(s) put back" in capsys.readouterr().out


def test_cli_refreeze_refuses_while_the_lock_is_held(env, capsys):
    from tda.cli_common import EXIT_LOCKED
    from tda.core.db_backup import acquire_lock_file

    acquire_lock_file(str(env["db"]), "somebody")
    before = sha(env["db"])
    assert run(env, "refreeze", "--desktop", str(DESKTOP)) == EXIT_LOCKED
    assert sha(env["db"]) == before
