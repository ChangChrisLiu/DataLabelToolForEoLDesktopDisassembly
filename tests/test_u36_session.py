"""Task U36 in the session: runs first-class, the other views, 从这帧起另算, 同步到其他视角.

The scene is :mod:`vis_runs_scene` -- the U24 scene seen from the scanner and
from oak1 on the same confirmed steps 10-14 (rs a draft, oak2 nothing). Pinned:
the queue's lines (frame count, 「全程」, the longest run first) and the
summary per part and view; what every view says on a run's steps, 「—」 and
drafts included; the suggested reason, both kinds and from both views; the
cut of a run and its undoing; the sync -- its op and its inverse, that it
never overwrites a verdict nor puts a reason where a part covers the part,
and that the other view's frames stay confirmed; the frame strip; and a frame
with no image inside a run.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import vis_review_scene as V
import vis_runs_scene as S
from session_scene import DESKTOP, VIEW
from tda.core import vis_review as VR
from tda.core.model import OCCLUDER_CABLE, OCCLUDER_VIEWPOINT
from tda.core.occluder import FULL
from tda.ui import visreview_text as T
from tda.ui.session_api import SessionRefusal
from tda.ui.session_visruns import OP_SYNC, crop_strip

STEPS = S.STEPS


@pytest.fixture
def scene(tmp_path):
    session = S.build(tmp_path)
    session.set_visibility_live(True)
    yield session
    session.close(force=True)


def lines(session) -> list[str]:
    return [T.entry_text(e) for e in session.visibility_queue()]


def entry(session, instance: str, step: int = None) -> dict:
    return next(e for e in session.visibility_queue() if e["instance"] == instance
                and (step is None or step in e["steps"]))


def detail(session, instance: str):
    return session.visibility_detail(entry(session, instance))


def cells(found) -> list[tuple]:
    return [T.view_cells(row) for row in found["views"]]


def sync_ops(session) -> int:
    return [op["kind"] for op in session.db.ops(DESKTOP, VIEW, limit=500)].count(OP_SYNC)


def open_conflicts(session, view: str) -> list:
    return session.db.conflicts(DESKTOP, view, open_only=True)


# --------------------------------------------------------------------------- #
# the queue
# --------------------------------------------------------------------------- #
def test_the_lines_are_runs_with_their_length_the_longest_first(scene):
    assert lines(scene) == [
        "connector.03 第 10–14 帧（5 帧） · too_small（自动） · 全程",
        "connector.04 第 10–14 帧（5 帧） · occluded_full（原因: 线缆，手动） · 全程",
        "connector.05 第 10–14 帧（5 帧） · occluded_full（手动） · 全程",
        "ram_latch.03 第 10–14 帧（5 帧） · occluded_full（被螺丝挡住，自动） · 全程",
        "ram_module.01 第 10–14 帧（5 帧） · occluded_partial（被螺丝挡住，自动） · 全程",
        "cpu.01 第 13 帧 · occluded_full（原因: 线缆，手动）",
        "psu.01 第 12 帧 · out_of_view（手动）",
    ]
    first = scene.visibility_queue()[0]
    assert (first["a"], first["b"], first["n"], first["whole"]) == (10, 14, 5, True)


def test_the_summary_says_per_view_how_long_each_part_is_hidden(scene):
    summary = {row["instance"]: row for row in scene.visibility_summary()}
    assert [row["instance"] for row in scene.visibility_summary()] == [
        e["instance"] for e in scene.visibility_queue()]           # the queue's order
    marked = summary[S.MARKED]
    assert marked["views"]["scan"] == (5, 1, 5) and marked["views"][S.OTHER] == (4, 1, 4)
    assert marked["views"]["oak2"] is None and marked["states"]["oak2"] == "none"
    assert marked["views"][S.DRAFT] is None and marked["states"][S.DRAFT] == "unconfirmed"
    assert summary[S.ANGLE]["views"][S.OTHER] == (0, 0, 0)         # seen in oak1
    assert [T.summary_cell(marked["views"][v], marked["states"][v])
            for v in ("scan", "oak1", "oak2", "rs")] == ["5/1", "4/1", "—", "未确认"]


def test_a_frame_with_no_image_inside_a_run_does_not_break_it(scene):
    scene.db.set_frame_flags(S.key(12), missing=True)
    scene.review.invalidate()
    found = entry(scene, V.HIDDEN)
    assert found["steps"] == [10, 11, 13, 14]
    assert T.entry_text(found).startswith("ram_latch.03 第 10–14 帧（4 帧）")


# --------------------------------------------------------------------------- #
# the other views
# --------------------------------------------------------------------------- #
def test_the_other_views_on_the_same_steps_with_dashes_and_drafts(scene):
    found = detail(scene, S.ANGLE)
    assert cells(found) == [("scan（这里）", "0", "5", "手动 5"), ("oak1", "5", "0", "·"),
                            ("oak2", "—", "—", "未标注"), ("rs", "—", "—", "只有草稿，没确认")]
    assert [row["view"] for row in found["views"] if row["highlight"]] == ["oak1"]
    marked = detail(scene, S.MARKED)
    assert cells(marked)[1] == ("oak1", "1", "4", "线缆 3，角度 1")
    hidden = detail(scene, V.HIDDEN)
    assert cells(hidden)[:2] == [("scan（这里）", "0", "5", "被螺丝挡住"),
                                 ("oak1", "0", "5", "被螺丝挡住")]
    assert not any(row["highlight"] for row in hidden["views"])


def test_the_suggestion_from_the_scanner_both_kinds(scene):
    angle = detail(scene, S.ANGLE)["suggestion"]
    assert (angle["kind"], angle["action"], angle["views"]) == (
        "viewpoint", "vis_review_viewpoint", [S.OTHER])
    assert T.suggestion_text(angle, entry(scene, S.ANGLE)) == (
        "建议：角度看不到（完全，Z）——oak1 同样的帧看得见，这里没有挡住它的零件")
    hidden = detail(scene, V.HIDDEN)["suggestion"]
    assert (hidden["kind"], hidden["action"]) == ("blocked", "vis_review_computed")
    assert T.suggestion_text(hidden, entry(scene, V.HIDDEN)) == "建议：按计算确认（Space）——被螺丝挡住"
    cable = detail(scene, S.MARKED)["suggestion"]
    assert (cable["verdict"], cable["key"]) == ("cable", "`")
    assert detail(scene, V.OUT)["suggestion"] is None               # out_of_view: no default


def test_the_suggestion_from_the_other_camera(scene):
    scene.open(DESKTOP, S.OTHER, force=True)
    scene.set_visibility_live(True)
    hidden = scene.visibility_detail(entry(scene, V.HIDDEN))["suggestion"]
    assert (hidden["kind"], hidden["verdict"], hidden["scan_too"]) == ("blocked", "computed", True)
    assert T.suggestion_text(hidden, entry(scene, V.HIDDEN)) == (
        "建议：按计算确认（Space）——被螺丝挡住，扫描仪视角也看不见")
    marked = entry(scene, S.MARKED)
    assert marked["steps"] == [10, 11, 12]                   # 13 answered, 14 seen
    assert scene.visibility_detail(marked)["views"][0]["hidden"] == 3     # the scanner


def test_reading_the_views_writes_nothing_and_compiles_nothing(scene, monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("compiled")

    monkeypatch.setattr(scene.truth, "compile", refuse)
    monkeypatch.setattr(scene.truth, "refresh", refuse)
    changes = scene.db.conn.total_changes
    for e in scene.visibility_queue():
        scene.visibility_detail(e)
    scene.visibility_summary()
    assert scene.db.conn.total_changes == changes


# --------------------------------------------------------------------------- #
# 从这帧起另算
# --------------------------------------------------------------------------- #
def test_a_run_is_cut_at_the_open_frame_and_the_keys_answer_the_rest(scene):
    run = entry(scene, S.ANGLE)
    out = scene.split_visibility_run(S.ANGLE, 12, run["steps"])
    assert out == {"merged": False, "step": 12, "steps": [12, 13, 14]}
    assert [e["steps"] for e in scene.visibility_queue() if e["instance"] == S.ANGLE] == [
        [10, 11], [12, 13, 14]]
    scene.goto(12, force=True)
    scene.review_visibility(S.ANGLE, VR.VERDICT_VIEWPOINT, FULL, steps=[12, 13, 14])
    assert [e["steps"] for e in scene.visibility_queue() if e["instance"] == S.ANGLE] == [
        [10, 11]]
    assert scene.db.frame_overrides(S.key(11))[S.ANGLE].occluder is None
    assert scene.db.frame_overrides(S.key(12))[S.ANGLE].occluder == OCCLUDER_VIEWPOINT
    # nothing was written by the cut itself: one op, the verdict's
    assert [op["kind"] for op in scene.db.ops(DESKTOP, VIEW, limit=50)].count(
        "review_visibility") == 1


def test_a_cut_is_refused_off_the_run_or_at_its_start_and_taken_back_there(scene):
    run = entry(scene, S.ANGLE)
    with pytest.raises(SessionRefusal, match="不在选中的这一段里"):
        scene.split_visibility_run(S.ANGLE, 9, run["steps"])
    with pytest.raises(SessionRefusal, match="已经是这一段的第一帧"):
        scene.split_visibility_run(S.ANGLE, 10, run["steps"])
    scene.split_visibility_run(S.ANGLE, 13, run["steps"])
    second = entry(scene, S.ANGLE, 13)
    assert second["steps"] == [13, 14]
    assert scene.split_visibility_run(S.ANGLE, 13, second["steps"]) == {"merged": True,
                                                                        "step": 13}
    assert entry(scene, S.ANGLE)["steps"] == STEPS
    # the cut is the annotator's for as long as the view is open
    scene.split_visibility_run(S.ANGLE, 12, STEPS)
    scene.open(DESKTOP, VIEW, force=True)
    scene.set_visibility_live(True)
    assert entry(scene, S.ANGLE)["steps"] == STEPS


# --------------------------------------------------------------------------- #
# 同步到其他视角
# --------------------------------------------------------------------------- #
def test_nothing_to_sync_before_a_verdict(scene):
    plan = scene.sync_visibility_plan(S.MARKED, STEPS)
    assert plan["refused"] == "no_verdict"
    with pytest.raises(SessionRefusal, match="还没有判定"):
        scene.sync_visibility(S.MARKED, STEPS)
    assert sync_ops(scene) == 0


def test_the_sync_copies_the_verdict_and_never_overwrites_one(scene):
    """Z on the scanner's 被线挡住 run of connector.04, then the sync: oak1's
    10-12 (cable marks, no verdict) take 角度看不到; 13 (answered in oak1) and
    14 (seen there) are left alone. One op; the frames stay confirmed."""
    scene.goto(10, force=True)
    scene.review_visibility(S.MARKED, VR.VERDICT_VIEWPOINT, FULL, steps=STEPS)
    answered = scene.db.instance_flags(S.key(13, S.OTHER), S.MARKED)
    plan = scene.sync_visibility_plan(S.MARKED, STEPS)
    assert sorted(plan["views"][S.OTHER]) == [10, 11, 12] and plan["replaced"] == 3
    assert plan["skipped"]["verdict"] == 1 and plan["skipped"]["different"] == 1
    result = scene.sync_visibility(S.MARKED, STEPS)
    assert result["written"] == {S.OTHER: [10, 11, 12]} and result["count"] == 3
    assert sync_ops(scene) == 1
    for step in (10, 11, 12):
        k = S.key(step, S.OTHER)
        fo = scene.db.frame_overrides(k)[S.MARKED]
        assert (fo.visibility, fo.occluder) == (FULL, OCCLUDER_VIEWPOINT)
        stamp = scene.db.instance_flags(k, S.MARKED)
        assert (stamp["verdict"], stamp["visibility"], stamp["occluder"]) == (
            "viewpoint", FULL, OCCLUDER_VIEWPOINT)
        assert scene.db.get_frame(k)["review_status"] == "verified"
        assert scene.db.compiled(k)[S.MARKED]["status"] == "verified"
    assert scene.db.instance_flags(S.key(13, S.OTHER), S.MARKED) == answered
    assert scene.db.frame_overrides(S.key(14, S.OTHER)).get(S.MARKED) is None
    assert open_conflicts(scene, S.OTHER) == []
    # nothing left to sync: a second press has nothing to copy
    with pytest.raises(SessionRefusal, match="没有可以同步的帧"):
        scene.sync_visibility(S.MARKED, STEPS)
    assert T.sync_note("接口", result).startswith(
        "已同步到其他视角：接口 oak1 第 10–12 帧，共 3 帧，记为审过（Ctrl+Z 撤销）")


def test_the_sync_is_undone_and_redone_as_one_op(scene):
    scene.goto(10, force=True)
    scene.review_visibility(S.MARKED, VR.VERDICT_VIEWPOINT, FULL, steps=STEPS)
    before = {s: (scene.db.frame_overrides(S.key(s, S.OTHER)).get(S.MARKED),
                  scene.db.instance_flags(S.key(s, S.OTHER), S.MARKED)) for s in STEPS}
    scene.sync_visibility(S.MARKED, STEPS)
    assert scene.undo()
    for step in STEPS:
        fo, flags = before[step]
        now = scene.db.frame_overrides(S.key(step, S.OTHER)).get(S.MARKED)
        assert (now is None) == (fo is None)
        if fo is not None:
            assert (now.visibility, now.occluder) == (fo.visibility, fo.occluder)
        assert scene.db.instance_flags(S.key(step, S.OTHER), S.MARKED) == flags
        assert scene.db.get_frame(S.key(step, S.OTHER))["review_status"] == "verified"
    assert scene.db.frame_overrides(S.key(10, S.OTHER))[S.MARKED].occluder == OCCLUDER_CABLE
    assert open_conflicts(scene, S.OTHER) == []
    # the scanner's own verdict is the op before: still there
    assert scene.db.instance_flags(S.key(10), S.MARKED)["verdict"] == "viewpoint"
    assert scene.redo()
    assert scene.db.frame_overrides(S.key(11, S.OTHER))[S.MARKED].occluder == OCCLUDER_VIEWPOINT
    assert open_conflicts(scene, S.OTHER) == []


def test_no_reason_goes_where_a_part_covers_it_but_the_computed_verdict_does(scene):
    scene.goto(10, force=True)
    scene.review_visibility(V.HIDDEN, VR.VERDICT_VIEWPOINT, FULL, steps=STEPS)
    plan = scene.sync_visibility_plan(V.HIDDEN, STEPS)
    assert plan["count"] == 0 and plan["skipped"]["covered"] == 5
    with pytest.raises(SessionRefusal, match="那边有零件挡着它 5 帧"):
        scene.sync_visibility(V.HIDDEN, STEPS)
    assert scene.undo()
    scene.review_visibility(V.HIDDEN, VR.VERDICT_COMPUTED, steps=STEPS)
    result = scene.sync_visibility(V.HIDDEN, STEPS)
    assert result["written"] == {S.OTHER: STEPS}
    for step in STEPS:
        stamp = scene.db.instance_flags(S.key(step, S.OTHER), V.HIDDEN)
        assert (stamp["verdict"], stamp["visibility"]) == ("computed", FULL)
        assert scene.db.frame_overrides(S.key(step, S.OTHER)).get(V.HIDDEN) is None


def test_a_label_that_is_not_hidden_is_not_synced(scene):
    scene.goto(12, force=True)
    scene.review_visibility(V.OUT, VR.VERDICT_LABEL, "visible")
    with pytest.raises(SessionRefusal, match="只用于看不见的标签"):
        scene.sync_visibility(V.OUT, [12])


# --------------------------------------------------------------------------- #
# the frame strip
# --------------------------------------------------------------------------- #
def test_the_strip_is_the_first_middle_and_last_frame_cut_around_the_part(scene):
    frames = scene.visibility_strip(V.HIDDEN, STEPS)
    assert [f["step"] for f in frames] == [10, 12, 14]
    for found in frames:
        image = found["image"]
        assert image.dtype == np.uint8 and image.ndim == 3 and max(image.shape[:2]) <= 96
        # the part's box is drawn
        assert ((image[:, :, 0] == 255) & (image[:, :, 1] == 200)).any()
    assert [f["step"] for f in scene.visibility_strip(V.CABLED, [13])] == [13]
    scene.db.set_frame_flags(S.key(12), missing=True)
    scene.db.conn.execute("UPDATE frame SET path = 'nowhere.png', aux_json = '{}' "
                          "WHERE desktop = ? AND step = 12 AND view = ?", (DESKTOP, VIEW))
    import shutil
    from tda.core.cache import cached_image_path
    gone = cached_image_path(scene.cache_dir, S.key(12))
    shutil.move(gone, gone + ".away")
    scene.images.clear()
    assert scene.visibility_strip(V.HIDDEN, [12])[0]["image"] is None


def test_crop_strip_fits_and_frames_the_box():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cut = crop_strip(image, (100, 100, 300, 140), size=96)
    assert max(cut.shape[:2]) == 96 and ((cut[:, :, 0] == 255)).any()
    whole = crop_strip(image, None, size=96)
    assert whole.shape[:2] == (64, 96)
    assert crop_strip(None, (0, 0, 1, 1)) is None
    half = crop_strip(np.zeros((200, 300, 3), dtype=np.uint8), (100, 100, 300, 140), scale=0.5)
    assert ((half[:, :, 0] == 255)).any()
