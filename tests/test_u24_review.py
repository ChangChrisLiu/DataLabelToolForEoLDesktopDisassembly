"""Task U24: 「可见性审查」 -- the visibility labels audited once, in Review.

The scene (:mod:`vis_review_scene`): frames 14-10 confirmed, with a part wholly
under another (computed ``occluded_full``), one half under another
(``occluded_partial``), a 4 px part (``too_small``), a part 被线挡住 on frame 13
and one set ``out_of_view`` by hand on frame 12.

Pinned here, without a window: what the queue lists and how it compresses
runs and names the parts that hide one; each verdict's op, its stamp and its
undo -- one frame, and a whole run as one op (a 40-frame run is one ``Z``);
that confirmed frames stay confirmed, a difference that was already there
opens a conflict instead of being signed, and revealed pixels are signed on
the frame on screen only; the new reason ``viewpoint``; and that the queue
costs nothing when nothing was written.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import vis_review_scene as V
from session_scene import DESKTOP, VIEW, cell, rect
from tda.core import masks
from tda.core import vis_review as VR
from tda.core.model import (
    OCCLUDER_CABLE,
    OCCLUDER_VIEWPOINT,
    OCCLUDERS,
    FrameKey,
    FrameOverride,
)
from tda.core.occluder import FULL, PARTIAL, occluded_override
from tda.ui import session_api as api
from tda.ui import visreview_text as T
from tda.ui.session_api import SessionRefusal
from tda.ui.session_visreview import OP_KIND

STEPS = [10, 11, 12, 13, 14]


@pytest.fixture
def scene(tmp_path):
    session = V.build(tmp_path)
    yield session
    session.close(force=True)


def lines(session) -> list[str]:
    return [T.entry_text(e) for e in session.visibility_queue()]


def entry(session, instance: str) -> dict:
    return next(e for e in session.visibility_queue() if e["instance"] == instance)


def ops(session) -> int:
    return [op["kind"] for op in session.db.ops(DESKTOP, VIEW, limit=500)].count(OP_KIND)


def open_conflicts(session) -> list[tuple[int, str]]:
    return sorted((int(c["step"]), str(c["instance"]))
                  for c in session.db.conflicts(DESKTOP, VIEW, open_only=True))


# --------------------------------------------------------------------------- #
# the queue
# --------------------------------------------------------------------------- #
def test_the_queue_lists_every_label_to_audit_one_line_per_run(scene):
    assert lines(scene) == [
        "connector.03 第 10–14 帧 · too_small（自动）",
        "cpu.01 第 13 帧 · occluded_full（原因: 线缆，手动）",
        "psu.01 第 12 帧 · out_of_view（手动）",
        "ram_latch.03 第 10–14 帧 · occluded_full（被螺丝挡住，自动）",
        "ram_module.01 第 10–14 帧 · occluded_partial（被螺丝挡住，自动）",
    ]
    hidden = entry(scene, V.HIDDEN)
    assert hidden["steps"] == STEPS and hidden["occluded_by"] == [V.COVER]
    assert hidden["manual"] is False and hidden["occluder"] is None
    assert entry(scene, V.HALF)["occluded_by"] == [V.HALF_COVER]
    cabled = entry(scene, V.CABLED)
    assert (cabled["manual"], cabled["occluder"], cabled["steps"]) == (True, OCCLUDER_CABLE,
                                                                       [V.CABLE_STEP])
    assert scene.visibility_progress() == (0, 17)


def test_frames_nobody_confirmed_are_not_in_the_queue(scene):
    queued = {s for e in scene.visibility_queue() for s in e["steps"]}
    assert queued == set(STEPS)
    # the same labels exist on the unconfirmed frames below 10
    scene.goto(9, force=True)
    assert scene.compiled().instances[V.HIDDEN].visibility == FULL
    assert scene.db.get_frame(V.key(9))["review_status"] != "verified"


def test_a_reviewed_frame_in_the_middle_splits_the_run(scene):
    """A frame that left the queue is a gap: the line becomes two."""
    scene.goto(12, force=True)
    scene.review_visibility(V.HIDDEN, VR.VERDICT_COMPUTED)
    assert [e["steps"] for e in scene.visibility_queue()
            if e["instance"] == V.HIDDEN] == [[10, 11], [13, 14]]


def test_runs_compress_over_steps_without_an_image():
    rows = [{"step": s, "instance": "a.01", "visibility": FULL, "hand_visibility": None,
             "occluder": None, "placement": "in_chassis", "input_hash": "h"}
            for s in (3, 5, 6, 8)]
    # step 4 has no image (not annotatable): 3 and 5 are consecutive; 7 is a
    # frame not in the queue, so 6 and 8 are not
    runs = VR.runs_of(rows, [1, 2, 3, 5, 6, 7, 8])
    assert [r.steps for r in runs] == [[3, 5, 6], [8]]
    named = VR.runs_of(rows, [3, 5, 6, 8], {(5, "a.01"): ("b.01",)})
    assert [(r.steps, r.occluded_by) for r in named] == [([3], ()), ([5], ("b.01",)),
                                                        ([6, 8], ())]


def test_the_occluders_are_named_off_the_stored_rows_without_compiling(scene, monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("the queue compiled a frame")

    monkeypatch.setattr(scene.truth, "compile", refuse)
    monkeypatch.setattr(scene.truth, "refresh", refuse)
    found = VR.occluders_on_frame(scene.db, V.key(12),
                                  {V.HIDDEN: "in_chassis", V.HALF: "in_chassis"})
    assert found == {V.HIDDEN: (V.COVER,), V.HALF: (V.HALF_COVER,)}


def test_an_occluder_layer_is_named_too(scene):
    from tda.core.model import OccluderMask

    hidden = scene.db.compiled(V.key(12))
    box = masks.rle_bbox(next(kf for kf in scene.db.keyframes(DESKTOP, VIEW, V.HALF)
                              ).parts[0].rle)
    hand = rect(box[0], box[1], box[0] + 4, box[1] + 4, V.HW)
    scene.db.set_occluder(OccluderMask(V.key(12), "hand", masks.encode_rle(hand)))
    found = VR.occluders_on_frame(scene.db, V.key(12), {V.HALF: "in_chassis"})
    assert found[V.HALF] == (V.HALF_COVER, "@hand")
    assert T.occluded_by_text(found[V.HALF]) == "被螺丝、手挡住"
    assert hidden  # the rows were there to read


def test_the_queue_is_rebuilt_only_after_a_write_and_names_each_row_once(scene, monkeypatch):
    calls = {"rows": 0, "names": 0}
    real_rows, real_names = VR.review_rows, VR.occluders_on_frame
    import tda.ui.session_queues as Q

    def rows(*args, **kwargs):
        calls["rows"] += 1
        return real_rows(*args, **kwargs)

    def names(*args, **kwargs):
        calls["names"] += 1
        return real_names(*args, **kwargs)

    monkeypatch.setattr(Q, "review_rows", rows)
    monkeypatch.setattr(Q, "occluders_on_frame", names)
    scene.review._vis_memo = None
    scene.review._vis_names.clear()
    first = scene.visibility_queue()
    assert calls == {"rows": 1, "names": 5}            # one per confirmed frame
    assert scene.visibility_queue() == first
    scene.visibility_progress()
    assert calls == {"rows": 1, "names": 5}            # nothing written: nothing read
    scene.goto(12, force=True)
    scene.review_visibility(V.OUT, VR.VERDICT_COMPUTED)
    scene.visibility_queue()
    assert calls["rows"] == 2 and calls["names"] == 5  # the names are kept per row


def test_queues_list_it_only_while_live(scene):
    assert api.QUEUE_VISIBILITY not in scene.queues()
    scene.set_visibility_live(True)
    assert len(scene.queues()[api.QUEUE_VISIBILITY]) == 5
    scene.set_visibility_live(False)
    assert api.QUEUE_VISIBILITY not in scene.queues()


def test_the_components_gallery_has_no_visibility_review(tmp_path):
    import gallery_scene as GS

    session = GS.make_session(tmp_path)
    try:
        session.set_visibility_live(True)
        assert session.visibility_queue() == []
        assert api.QUEUE_VISIBILITY not in session.queues() or \
            session.queues()[api.QUEUE_VISIBILITY] == []
        with pytest.raises(SessionRefusal, match="零件合照"):
            session.review_visibility("chassis", VR.VERDICT_COMPUTED)
    finally:
        session.close(force=True)


# --------------------------------------------------------------------------- #
# the stamp
# --------------------------------------------------------------------------- #
def test_space_stamps_the_row_as_it_is_and_ctrl_z_takes_the_stamp_back(scene):
    scene.goto(12, force=True)
    before = scene.db.compiled(V.key(12))[V.HIDDEN]
    epoch = scene._epoch
    result = scene.review_visibility(V.HIDDEN, VR.VERDICT_COMPUTED)
    assert result["moved"] == [] and ops(scene) == 1
    flags = scene.db.instance_flags(V.key(12), V.HIDDEN)
    assert {k: flags[k] for k in ("vis_reviewed", "by", "verdict", "visibility",
                                  "occluder")} == {
        "vis_reviewed": True, "by": "tester", "verdict": "computed",
        "visibility": FULL, "occluder": None}
    assert flags["at"].startswith("20")
    assert scene.db.compiled(V.key(12))[V.HIDDEN] == before   # nothing re-signed
    assert scene._epoch == epoch                               # nothing recompiled
    assert scene.visibility_progress() == (1, 17)
    assert scene.undo()
    assert scene.db.instance_flags(V.key(12), V.HIDDEN) is None
    assert scene.visibility_progress() == (0, 17)
    assert scene.redo()
    assert scene.db.instance_flags(V.key(12), V.HIDDEN)["verdict"] == "computed"


def test_shift_space_stamps_the_whole_run_in_one_op(scene):
    scene.goto(14, force=True)
    scene.review_visibility(V.TINY, VR.VERDICT_COMPUTED, steps=STEPS)
    assert ops(scene) == 1
    assert all(scene.db.instance_flags(V.key(s), V.TINY)["vis_reviewed"] for s in STEPS)
    assert V.TINY not in {e["instance"] for e in scene.visibility_queue()}
    assert scene.undo()
    assert all(scene.db.instance_flags(V.key(s), V.TINY) is None for s in STEPS)
    assert entry(scene, V.TINY)["steps"] == STEPS


def test_a_stamp_keeps_other_flags_and_the_undo_restores_them_byte_for_byte(scene):
    scene.db.set_instance_flags(V.key(12), V.HIDDEN, {"low_contrast": True})
    scene.goto(12, force=True)
    scene.review_visibility(V.HIDDEN, VR.VERDICT_COMPUTED)
    flags = scene.db.instance_flags(V.key(12), V.HIDDEN)
    assert flags["low_contrast"] is True and flags["vis_reviewed"] is True
    scene.undo()
    assert scene.db.instance_flags(V.key(12), V.HIDDEN) == {"low_contrast": True}


def test_a_row_whose_label_changed_after_its_stamp_comes_back(scene):
    scene.goto(V.OUT_STEP, force=True)
    scene.review_visibility(V.OUT, VR.VERDICT_COMPUTED)
    assert V.OUT not in {e["instance"] for e in scene.visibility_queue()}
    # the label moves under the stamp: written some other way, and its
    # conflict accepted in Review
    scene.set_visibility(V.OUT, "too_small")
    assert V.OUT not in {e["instance"] for e in scene.visibility_queue()}  # frozen still
    [cid] = [c["id"] for c in scene.db.conflicts(DESKTOP, VIEW, open_only=True)
             if c["instance"] == V.OUT]
    assert scene.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW) == "resolved"
    assert scene.db.compiled(V.key(V.OUT_STEP))[V.OUT]["visibility"] == "too_small"
    assert entry(scene, V.OUT)["visibility"] == "too_small"


# --------------------------------------------------------------------------- #
# verdicts that set the label: the frame stays confirmed (U15's M8 path)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("verdict,label,occluder", [
    (VR.VERDICT_LABEL, "visible", None),
    (VR.VERDICT_LABEL, "out_of_view", None),
    (VR.VERDICT_CABLE, FULL, OCCLUDER_CABLE),
    (VR.VERDICT_CABLE, PARTIAL, OCCLUDER_CABLE),
    (VR.VERDICT_VIEWPOINT, FULL, OCCLUDER_VIEWPOINT),
    (VR.VERDICT_VIEWPOINT, PARTIAL, OCCLUDER_VIEWPOINT),
])
def test_a_verdict_sets_the_label_keeps_the_frame_confirmed_and_undoes(scene, verdict, label,
                                                                       occluder):
    k = V.key(14)
    scene.goto(14, force=True)
    before = scene.db.compiled(k)[V.HALF]
    result = scene.review_visibility(V.HALF, verdict, label)
    # signed where the row changed; 部分 over a computed 部分 changes no value
    assert result["moved"] == [14] and result["accepted"] == (
        [] if label == before["visibility"] else [14])
    assert ops(scene) == 1
    fo = scene.db.frame_overrides(k)[V.HALF]
    assert (fo.visibility, fo.occluder) == (label, occluder)
    row = scene.db.compiled(k)[V.HALF]
    assert row["status"] == "verified" and row["visibility"] == label
    if label == FULL:
        assert masks.rle_area(row["visible_rle"]) == 0       # nothing seen, shape kept
    assert open_conflicts(scene) == []
    assert scene.db.get_frame(k)["review_status"] == "verified"
    assert scene.frame_status(14) == "verified"
    stamp = scene.db.instance_flags(k, V.HALF)
    assert (stamp["verdict"], stamp["visibility"], stamp["occluder"]) == (verdict, label,
                                                                           occluder)
    assert V.HALF not in {e["instance"] for e in scene.visibility_queue()
                          if 14 in e["steps"]}

    assert scene.undo()
    assert scene.db.frame_overrides(k).get(V.HALF) is None
    assert scene.db.instance_flags(k, V.HALF) is None
    back = scene.db.compiled(k)[V.HALF]
    assert back["status"] == "verified" and back["visibility"] == before["visibility"]
    assert masks.rle_counts(back["visible_rle"]) == masks.rle_counts(before["visible_rle"])
    assert open_conflicts(scene) == [] and scene.frame_status(14) == "verified"
    assert scene.redo()
    assert scene.db.frame_overrides(k)[V.HALF].occluder == occluder
    assert open_conflicts(scene) == [] and scene.frame_status(14) == "verified"


def test_viewpoint_over_a_cable_label_changes_the_reason_only(scene):
    k = V.key(V.CABLE_STEP)
    scene.goto(V.CABLE_STEP, force=True)
    scene.review_visibility(V.CABLED, VR.VERDICT_VIEWPOINT, FULL)
    fo = scene.db.frame_overrides(k)[V.CABLED]
    assert (fo.visibility, fo.occluder) == (FULL, OCCLUDER_VIEWPOINT)
    assert open_conflicts(scene) == []
    assert V.CABLED not in {e["instance"] for e in scene.visibility_queue()}
    scene.undo()
    assert scene.db.frame_overrides(k)[V.CABLED].occluder == OCCLUDER_CABLE


def test_a_difference_that_was_already_there_opens_a_conflict_instead(scene):
    """The part re-traced at 9 (改全部) out from under its cover reaches the
    confirmed 10-14 (queued for a re-check, the sweeper off); arriving at 13
    opens its conflict. A verdict there: the re-trace is Review's question, so
    the label does not sign it -- the label is written, the conflict stays
    open."""
    scene.goto(9, force=True)
    scene.begin_edit(V.HIDDEN)
    scene.set_editing_mask(rect(200, 200, 230, 230, V.HW))
    scene.commit_edit(api.SCOPE_KEYFRAME)
    assert 13 in scene.db.rechecks(DESKTOP, VIEW)
    frozen = scene.db.compiled(V.key(13))[V.HIDDEN]
    scene.goto(13, force=True)
    assert (13, V.HIDDEN) in open_conflicts(scene)
    result = scene.review_visibility(V.HIDDEN, VR.VERDICT_VIEWPOINT, PARTIAL)
    assert result["accepted"] == []
    assert (13, V.HIDDEN) in open_conflicts(scene)
    assert scene.db.compiled(V.key(13))[V.HIDDEN] == frozen          # nothing signed
    assert scene.db.frame_overrides(V.key(13))[V.HIDDEN].occluder == OCCLUDER_VIEWPOINT
    assert scene.frame_status(13) == "conflict"
    assert scene.db.instance_flags(V.key(13), V.HIDDEN)["verdict"] == "viewpoint"
    # the stamp is for a label the stored row does not have yet: still listed
    assert 13 in {s for e in scene.visibility_queue() if e["instance"] == V.HIDDEN
                  for s in e["steps"]}


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_a_frame_nobody_confirmed_is_refused(scene):
    scene.goto(5, force=True)
    with pytest.raises(SessionRefusal, match="还没确认"):
        scene.review_visibility(V.HIDDEN, VR.VERDICT_COMPUTED)
    assert ops(scene) == 0


def test_a_verdict_over_a_run_is_one_op_and_one_undo(scene):
    """The coordinator's ruling: a line is a run, every key answers all of it."""
    scene.goto(10, force=True)
    before = {s: scene.db.compiled(V.key(s))[V.HALF] for s in STEPS}
    result = scene.review_visibility(V.HALF, VR.VERDICT_VIEWPOINT, FULL, steps=STEPS)
    assert result["moved"] == STEPS and sorted(result["accepted"]) == STEPS
    assert ops(scene) == 1 and open_conflicts(scene) == []
    for step in STEPS:
        fo = scene.db.frame_overrides(V.key(step))[V.HALF]
        assert (fo.visibility, fo.occluder) == (FULL, OCCLUDER_VIEWPOINT)
        row = scene.db.compiled(V.key(step))[V.HALF]
        assert row["status"] == "verified" and row["visibility"] == FULL
        assert scene.frame_status(step) == "verified"
        assert scene.db.instance_flags(V.key(step), V.HALF)["verdict"] == "viewpoint"
    assert V.HALF not in {e["instance"] for e in scene.visibility_queue()}
    assert scene.undo()
    for step in STEPS:
        assert scene.db.frame_overrides(V.key(step)).get(V.HALF) is None
        assert scene.db.instance_flags(V.key(step), V.HALF) is None
        assert scene.db.compiled(V.key(step))[V.HALF]["visibility"] == \
            before[step]["visibility"]
        assert scene.frame_status(step) == "verified"
    assert open_conflicts(scene) == [] and entry(scene, V.HALF)["steps"] == STEPS


def test_pixels_a_label_reveals_are_signed_on_the_frame_on_screen_only(scene):
    """U15 review round 4, per frame: the half-hidden run set 被线挡住（完全）
    (empty rows, signed), the part then redrawn elsewhere at 9 (改全部: the
    empty rows still agree), then a digit over the run from frame 12 -- the
    label shows pixels nobody confirmed; 12, on screen, is signed, the others
    keep their conflicts for Review."""
    scene.goto(10, force=True)
    scene.review_visibility(V.HALF, VR.VERDICT_CABLE, FULL, steps=STEPS)
    scene.goto(9, force=True)
    scene.begin_edit(V.HALF)
    scene.set_editing_mask(rect(200, 200, 230, 230, V.HW))
    scene.commit_edit(api.SCOPE_KEYFRAME)
    scene.goto(12, force=True)
    assert open_conflicts(scene) == []
    result = scene.review_visibility(V.HALF, VR.VERDICT_LABEL, PARTIAL, steps=STEPS)
    assert result["accepted"] == [12] and sorted(result["withheld"]) == [10, 11, 13, 14]
    assert open_conflicts(scene) == [(s, V.HALF) for s in (10, 11, 13, 14)]
    assert masks.rle_area(scene.db.compiled(V.key(12))[V.HALF]["visible_rle"]) == 900
    assert all(masks.rle_area(scene.db.compiled(V.key(s))[V.HALF]["visible_rle"]) == 0
               for s in (10, 11, 13, 14))
    assert ops(scene) == 2
    assert scene.undo()
    assert open_conflicts(scene) == []


def test_a_run_is_refused_as_a_whole_naming_the_frames(scene, monkeypatch):
    import tda.ui.session_visreview as SV

    real = SV.shape_applies

    def no_shape_on_12(db, tax, key, *args, **kwargs):
        return "no_shape" if key.step == 12 else real(db, tax, key, *args, **kwargs)

    monkeypatch.setattr(SV, "shape_applies", no_shape_on_12)
    scene.goto(10, force=True)
    with pytest.raises(SessionRefusal) as refused:
        scene.review_visibility(V.HALF, VR.VERDICT_VIEWPOINT, FULL, steps=STEPS)
    assert str(refused.value) == f"第 12 帧：{T.VIEWPOINT_NEEDS_SHAPE}"
    assert ops(scene) == 0
    assert all(scene.db.frame_overrides(V.key(s)).get(V.HALF) is None for s in STEPS)


def test_a_forty_frame_viewpoint_run_is_one_z_one_op_one_undo(tmp_path):
    """The motherboard drawn 5 px wide over the whole sheet: ``too_small`` on
    frames 2-41, all confirmed -- one line; one Z over it. (5 x 30 px: a part
    under the conflict floor of 20 px would not be compared back on the undo,
    see the U24 report.)"""
    from session_scene import make_session, seed_shapes

    session = make_session(tmp_path, last_step=42, hw=V.HW)
    session.sweeper_enabled = False
    try:
        keys = seed_shapes(session, 1, grid=8, anchor=42, refresh=False)
        board = "motherboard.01"
        ys, xs = np.nonzero(cell(keys.index(board), V.HW))
        V.keyframe(session, board, rect(int(xs.min()), int(ys.min()), int(xs.min()) + 5,
                                        int(ys.min()) + 30, V.HW), anchor=42)
        session.refresh_all()
        for step in range(41, 1, -1):
            session.goto(step, force=True)
            assert session.confirm_frame(), step
        session.set_visibility_live(True)
        [line] = [e for e in session.visibility_queue() if e["instance"] == board]
        run = list(range(2, 42))
        assert line["steps"] == run and line["visibility"] == "too_small"
        session.goto(2, force=True)
        result = session.review_visibility(board, VR.VERDICT_VIEWPOINT, FULL,
                                           steps=line["steps"])
        assert result["moved"] == run and sorted(result["accepted"]) == run
        assert ops(session) == 1 and open_conflicts(session) == []
        assert all(session.db.frame_overrides(V.key(s))[board].occluder == OCCLUDER_VIEWPOINT
                   and session.db.compiled(V.key(s))[board]["visibility"] == FULL
                   and session.frame_status(s) == "verified" for s in run)
        assert board not in {e["instance"] for e in session.visibility_queue()}
        assert session.undo()
        assert all(session.db.frame_overrides(V.key(s)).get(board) is None
                   and session.db.instance_flags(V.key(s), board) is None
                   and session.db.compiled(V.key(s))[board]["visibility"] == "too_small"
                   for s in run)
        assert open_conflicts(session) == []
        assert [e["steps"] for e in session.visibility_queue()
                if e["instance"] == board] == [run]
    finally:
        session.close(force=True)


def test_viewpoint_needs_a_shape_in_its_place(scene, monkeypatch):
    import tda.ui.session_visreview as SV

    scene.goto(14, force=True)
    for why, said in (("no_shape", T.VIEWPOINT_NEEDS_SHAPE), ("bench", T.VIEWPOINT_BENCH),
                      ("removed", T.VIEWPOINT_REMOVED)):
        monkeypatch.setattr(SV, "shape_applies", lambda *a, _w=why, **k: _w)
        with pytest.raises(SessionRefusal) as refused:
            scene.review_visibility(V.HALF, VR.VERDICT_VIEWPOINT, FULL)
        assert str(refused.value) == said
    assert ops(scene) == 0


def test_a_label_must_be_a_visibility_value(scene):
    scene.goto(14, force=True)
    with pytest.raises(SessionRefusal):
        scene.review_visibility(V.HALF, VR.VERDICT_LABEL, "seen")


# --------------------------------------------------------------------------- #
# viewpoint, store level
# --------------------------------------------------------------------------- #
def test_viewpoint_is_an_occluder_the_store_and_the_rule_take(tmp_path):
    from tda.core.db import Db

    assert OCCLUDERS == ("cable", "other_unlabeled", "viewpoint")
    db = Db(str(tmp_path / "t.sqlite"))
    k = FrameKey(1, 3, "oak2")
    db.set_frame_override(FrameOverride(k, "a.01", None, PARTIAL, OCCLUDER_VIEWPOINT))
    assert db.frame_overrides(k)["a.01"].occluder == OCCLUDER_VIEWPOINT
    made = occluded_override(k, "a.01", None, FULL, (4, 4), OCCLUDER_VIEWPOINT)
    assert (made.visibility, made.occluder, masks.rle_area(made.visible_rle)) == (
        FULL, OCCLUDER_VIEWPOINT, 0)
    with pytest.raises(ValueError):
        db.set_frame_override(FrameOverride(k, "a.01", None, FULL, "angle"))
    db.close()


def test_the_stamp_helpers():
    stamp = VR.make_stamp("chang", VR.VERDICT_VIEWPOINT, FULL, OCCLUDER_VIEWPOINT, at="t")
    assert stamp == {"vis_reviewed": True, "by": "chang", "at": "t", "verdict": "viewpoint",
                     "visibility": FULL, "occluder": "viewpoint"}
    assert VR.stamp_covers(stamp, FULL, OCCLUDER_VIEWPOINT)
    assert not VR.stamp_covers(stamp, PARTIAL, OCCLUDER_VIEWPOINT)
    assert not VR.stamp_covers(stamp, FULL, OCCLUDER_CABLE)
    assert VR.stamp_covers({"vis_reviewed": True}, "anything", None)   # taken at its word
    assert not VR.stamp_covers({}, FULL, None) and not VR.stamp_covers(None, FULL, None)
    assert VR.with_stamp({"x": 1, "verdict": "old"}, stamp) == {"x": 1, **stamp}
    assert VR.with_stamp({"vis_reviewed": True, "by": "a"}, None) is None
    with pytest.raises(ValueError):
        VR.make_stamp("a", "maybe", FULL, None)
