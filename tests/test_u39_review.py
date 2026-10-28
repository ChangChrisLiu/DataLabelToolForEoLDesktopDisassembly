"""Task U39 in the visibility review: 画框外 is 看不见, for a reason of its own.

The U36 scene (:mod:`vis_runs_scene`: the scanner and oak1 confirmed on 10-14)
with one part the scanner leaves out of its picture on 10-14 -- written with
``Ctrl+```'s 画框外 on confirmed frames, which stay confirmed -- and the
covered RAM latch left out of oak1's picture. Pinned: the queue's line says
「原因: 画框外，手动」; the summary counts it as 看不见; the other-views table
says 「画框外 n」 for it, here and from the other view; the suggestion never
turns it into 角度看不到 (the part is not in the picture at all); and
「同步到其他视角」 refuses it -- the edge of one camera's picture is its own --
including a digit key's ``out_of_view``.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

import vis_review_scene as V
import vis_runs_scene as S
from session_scene import DESKTOP, VIEW, chassis_instances
from tda.core import vis_review as VR
from tda.core import vis_runs as RU
from tda.core.model import FrameKey
from tda.core.occluder import OFF_FRAME
from tda.ui import visreview_text as T
from tda.ui.session_api import SessionRefusal
from tda.ui.session_visruns import OP_SYNC

STEPS = S.STEPS
OUT = "out_of_view"
_TAKEN = {V.HIDDEN, V.COVER, V.HALF, V.HALF_COVER, V.TINY, V.CABLED, V.OUT, S.ANGLE,
          S.MARKED, "chassis"}


@pytest.fixture
def scene(tmp_path):
    session = S.build(tmp_path)
    # a part in its place on 10-14 (still there at 14), plainly seen in both views
    part = next(k for k in chassis_instances(session, 14)
                if k not in _TAKEN and k in chassis_instances(session, 10))
    # the scanner's frames 10-14 leave PART out of the picture (Ctrl+` 画框外)
    session.goto(14, force=True)
    result = session.set_cable_range(part, OFF_FRAME, 10)
    assert result["frames"] == STEPS
    # ... and oak1's leave the covered RAM latch out of its own
    session.open(DESKTOP, S.OTHER, force=True)
    session.goto(14, force=True)
    session.set_cable_range(V.HIDDEN, OFF_FRAME, 10)
    session.open(DESKTOP, VIEW, force=True)
    session.goto(10, force=True)
    session.set_visibility_live(True)
    session.part = part
    yield session
    session.close(force=True)


def entry(session, instance: str) -> dict:
    return next(e for e in session.visibility_queue() if e["instance"] == instance)


def cells(session, instance: str) -> dict:
    found = session.visibility_detail(entry(session, instance))
    return {row["view"]: T.view_cells(row) for row in found["views"]}


def test_the_frames_stay_confirmed_and_the_queue_says_off_frame(scene):
    part = scene.part
    for step in STEPS:
        key = FrameKey(DESKTOP, step, VIEW)
        assert scene.db.get_frame(key)["review_status"] == "verified"
        assert scene.db.compiled_label(key, part)["visibility"] == OUT
    run = entry(scene, part)
    assert (run["steps"], run["visibility"], run["manual"], run["occluder"]) == (
        STEPS, OUT, True, None)
    assert T.entry_text(run) == (f"{part} 第 10–14 帧（5 帧） · out_of_view"
                                 f"（原因: 画框外，手动） · 全程")
    # the digit key's out_of_view on psu.01 is the same statement
    assert T.entry_text(entry(scene, V.OUT)) == "psu.01 第 12 帧 · out_of_view（原因: 画框外，手动）"
    # 被线挡住 / 角度看不到 lines are as they were
    assert T.entry_text(entry(scene, V.CABLED)) == "cpu.01 第 13 帧 · occluded_full（原因: 线缆，手动）"


def test_the_summary_counts_it_as_hidden(scene):
    summary = {row["instance"]: row for row in scene.visibility_summary()}
    assert summary[scene.part]["views"]["scan"] == (5, 1, 5)


def test_the_other_views_table_says_off_frame_and_no_viewpoint_is_suggested(scene):
    part = scene.part
    found = scene.visibility_detail(entry(scene, part))
    scan = next(r for r in found["views"] if r["view"] == VIEW)
    oak1 = next(r for r in found["views"] if r["view"] == S.OTHER)
    assert scan["reasons"] == {RU.OFF_FRAME_REASON: 5} and scan["hidden"] == 5
    assert T.view_cells(scan) == ("scan（这里）", "0", "5", "画框外 5")
    assert oak1["seen"] == 5 and oak1["highlight"]               # oak1 sees it
    # seen elsewhere and nothing over it here -- still not 角度看不到: it is not
    # in this picture at all
    assert found["suggestion"] is None
    assert T.suggestion_text(found["suggestion"], entry(scene, part)) == ""
    # the other way round: oak1 left the latch out, the scanner has it covered
    latch = cells(scene, V.HIDDEN)
    assert latch[S.OTHER][2:] == ("5", "画框外 5")
    assert latch[VIEW][2] == "5"


def test_the_sync_never_carries_off_frame_to_another_view(scene):
    part = scene.part
    scene.goto(10, force=True)
    scene.review_visibility(part, VR.VERDICT_COMPUTED, steps=STEPS)
    plan = scene.sync_visibility_plan(part, STEPS)
    assert plan["refused"] == RU.OFF_FRAME_REASON and plan["count"] == 0
    with pytest.raises(SessionRefusal) as refused:
        scene.sync_visibility(part, STEPS)
    assert str(refused.value) == T.SYNC_REFUSED[RU.OFF_FRAME_REASON]
    assert "画框外不同步" in str(refused.value)
    # a digit key's out_of_view is 画框外 too: never synced either
    scene.review_visibility(S.ANGLE, VR.VERDICT_LABEL, OUT, steps=STEPS)
    with pytest.raises(SessionRefusal, match="画框外不同步"):
        scene.sync_visibility(S.ANGLE, STEPS)
    assert not [op for op in scene.db.ops(DESKTOP, VIEW, limit=500) if op["kind"] == OP_SYNC]
    # nothing reached oak1
    for step in STEPS:
        assert scene.db.frame_overrides(FrameKey(DESKTOP, step, S.OTHER)).get(part) is None
        assert not (scene.db.instance_flags(FrameKey(DESKTOP, step, S.OTHER), part) or {})


def test_the_occlusion_sync_is_unchanged(scene):
    """U36's sync of a 完全 run still goes where it went (only out_of_view left)."""
    assert RU.SYNC_LABELS == VR.OCCLUDED_LABELS
    scene.goto(10, force=True)
    scene.review_visibility(V.HIDDEN, VR.VERDICT_COMPUTED, steps=STEPS)
    # oak1's latch is 画框外 now -- a person's word there, so it is left alone
    plan = scene.sync_visibility_plan(V.HIDDEN, STEPS)
    assert plan["refused"] == "" and plan["count"] == 0
    assert plan["skipped"]["different"] == 5
