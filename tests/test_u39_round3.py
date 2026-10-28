"""Task U39 round 3: the reviewer's three points, each pinned.

1. **The review asks the compiler's question.** 画框外 is a hand
   ``out_of_view`` with **no pixels of its own** (:func:`tda.core.occluder.off_frame`).
   A digit key's ``out_of_view`` over a single-frame mask drawn by hand is not:
   the compiler paints that part as before, and the queue line and the
   other-views table say 「手动」 as they did, not 「画框外」.
2. **Only a part in its place is 画框外.** A part lying on the bench is in the
   picture (the staging area is): with a pixel-less ``out_of_view`` it is still
   painted, its missing box is still ``bench_missing``, its card row is not
   done and the instance table does not call it 框外.
3. **The scanner package skips only what is out of the picture.** A painted
   part with no annotation is passed over when the compiler labelled it
   ``out_of_view``; any other one still fails loudly (``KeyError``), never
   hidden.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

import compiler_scenes as C
import vis_review_scene as V
from session_scene import COOLER, DESKTOP, VIEW, chassis_instances, make_session
from tda.core import masks
from tda.core import vis_review as VR
from tda.core import vis_runs as RU
from tda.core.model import FrameKey, FrameOverride
from tda.core.occluder import OFF_FRAME
from tda.ui import session_api as api
from tda.ui import visreview_text as T
from tda.ui.session_api import SessionRefusal
from tda.ui.session_cable import OFF_FRAME_ON_BENCH
from tda.ui.session_coverage import coverage
from tda.ui.session_tasks import OFF_FRAME_KEY

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_scanpack as S  # noqa: E402

OUT = "out_of_view"
_TAKEN = {V.HIDDEN, V.COVER, V.HALF, V.HALF_COVER, V.TINY, V.CABLED, V.OUT, "chassis"}


def painted(compiled) -> set[str]:
    return {inst for layers in compiled.layers.values() for inst, _part in layers}


# --------------------------------------------------------------------------- #
# 1. a hand out_of_view over a hand-drawn mask is not 画框外
# --------------------------------------------------------------------------- #
@pytest.fixture
def review(tmp_path):
    session = V.build(tmp_path, confirm=())
    plain = [k for k in chassis_instances(session, 14)
             if k not in _TAKEN and k in chassis_instances(session, 10)]
    masked, outside = plain[0], plain[1]
    session.goto(14, force=True)
    # a mask drawn for this frame only, then a digit key's out_of_view on it
    session.begin_edit(masked)
    drawn = session.compiled().instances[masked].amodal.copy()
    drawn[int(np.nonzero(drawn)[0].min()), :] = False     # a little less than the shape
    session.set_editing_mask(drawn)
    session.commit_edit(api.SCOPE_FRAME_OVERRIDE)
    session.clear_edit()
    session.set_visibility(masked, OUT)
    # ... and a part 画框外 the way Ctrl+` writes it
    session.set_cable_range(outside, OFF_FRAME, 14)
    assert session.confirm_frame()
    session.goto(14, force=True)
    session.set_visibility_live(True)
    session.masked, session.outside = masked, outside
    yield session
    session.close(force=True)


def test_a_hand_out_of_view_with_pixels_is_painted_and_not_off_frame(review):
    masked, outside = review.masked, review.outside
    fo = review.db.frame_overrides(FrameKey(DESKTOP, 14, VIEW))[masked]
    assert fo.visibility == OUT and fo.visible_rle is not None      # the hand mask is kept
    compiled = review.compiled()
    assert masked in painted(compiled) and outside not in painted(compiled)
    assert compiled.instances[masked].visibility == OUT             # the label still hides it


def test_the_queue_says_off_frame_only_for_the_pixel_less_one(review):
    entries = {e["instance"]: e for e in review.visibility_queue()}
    masked, outside = entries[review.masked], entries[review.outside]
    assert (masked["manual"], masked["off_frame"]) == (True, False)
    assert (outside["manual"], outside["off_frame"]) == (True, True)
    assert T.entry_text(masked) == f"{review.masked} 第 14 帧 · out_of_view（手动） · 全程"
    assert T.entry_text(outside) == (f"{review.outside} 第 14 帧 · out_of_view"
                                     f"（原因: 画框外，手动） · 全程")
    rows = {r["instance"]: r for r in review.db.vis_review_rows(DESKTOP, VIEW)}
    assert rows[review.masked]["hand_pixels"] is True
    assert rows[review.outside]["hand_pixels"] is False
    assert not VR.off_frame_row(rows[review.masked]) and VR.off_frame_row(rows[review.outside])


def test_the_other_views_table_says_manual_for_the_one_with_pixels(review):
    def scan_row(instance):
        entry = next(e for e in review.visibility_queue() if e["instance"] == instance)
        found = review.visibility_detail(entry)
        return next(r for r in found["views"] if r["view"] == VIEW)

    assert scan_row(review.masked)["reasons"] == {"manual": 1}
    assert scan_row(review.outside)["reasons"] == {RU.OFF_FRAME_REASON: 1}
    assert T.view_cells(scan_row(review.masked))[3] == "手动 1"


def test_two_neighbouring_frames_one_with_pixels_are_two_runs():
    def row(step, pixels):
        return {"step": step, "instance": "a.01", "visibility": OUT,
                "hand_visibility": OUT, "occluder": None, "hand_pixels": pixels}

    runs = VR.runs_of([row(1, False), row(2, True), row(3, True)], [1, 2, 3])
    assert [(r.steps, r.off_frame) for r in runs] == [([1], True), ([2, 3], False)]


# --------------------------------------------------------------------------- #
# 2. a part on the bench is never 画框外
# --------------------------------------------------------------------------- #
def test_the_compiler_paints_a_bench_part_whatever_its_label():
    scene = C.two_squares()                                         # B over A
    over = {"B": FrameOverride(C.KEY, "B", None, OUT)}
    chassis = C.run(scene["keyframes"], scene["zorder"], frame_overrides=over)
    assert "B" not in painted(chassis)
    assert int(np.count_nonzero(chassis.instances["A"].visible)) == 32 * 32  # A whole
    bench = {inst: [C.kf(inst, {"main": box}, kf_id=i, placement=C.BENCH)]
             for i, (inst, box) in enumerate((("A", C.A_BOX), ("B", C.B_BOX)), 1)}
    placed = C.run(bench, scene["zorder"], frame_overrides=over,
                   placements={"A": C.BENCH, "B": C.BENCH})
    assert "B" in painted(placed)                                    # still a layer
    assert placed.instances["B"].visibility == OUT and placed.instances["B"].visible is None
    overlap = 20 * 20                                                # (20..40)^2
    assert int(np.count_nonzero(placed.instances["A"].visible)) == 32 * 32 - overlap


@pytest.fixture
def bench(tmp_path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    session.db.set_pose_segment(DESKTOP, VIEW, 1, 1, 14, 14, None, None)
    session.db.set_pose_segment_bench_roi(DESKTOP, VIEW, 1, [0, 40, 64, 64])
    session.goto(14, force=True)                     # the cooler lies on the bench here
    session.db.set_frame_override(FrameOverride(FrameKey(DESKTOP, 14, VIEW), COOLER, None, OUT))
    session._invalidate()
    session.review.invalidate()
    yield session
    session.close(force=True)


def test_a_bench_part_with_a_pixel_less_out_of_view_is_still_bench_missing(bench):
    assert f"bench_missing:{COOLER}" in bench.compiled().problems
    row = next(r for r in bench.task_card() if r["instance"] == COOLER)
    assert row["kind"] == api.KIND_ADD_BENCH_BOX
    assert not row["done"] and not row.get(OFF_FRAME_KEY)
    assert COOLER in coverage(bench.db, bench.tax, DESKTOP, VIEW, [14])[14].bench_missing
    assert COOLER not in bench.off_frame_parts()
    assert not next(r for r in bench.instance_rows() if r["key"] == COOLER).get("off_frame")
    # and Ctrl+` refuses to write 画框外 on it at all
    with pytest.raises(SessionRefusal, match=OFF_FRAME_ON_BENCH):
        bench.set_cable_range(COOLER, OFF_FRAME, 14)


# --------------------------------------------------------------------------- #
# 3. the scanner package skips only an out_of_view part
# --------------------------------------------------------------------------- #
def ann(instance: str, mask) -> dict:
    return {"attributes": {"instance_key": instance}, "segmentation": masks.encode_rle(mask)}


def test_the_label_map_skips_a_painted_out_of_view_part():
    bench = {inst: [C.kf(inst, {"main": box}, kf_id=i, placement=C.BENCH)]
             for i, (inst, box) in enumerate((("A", C.A_BOX), ("B", C.B_BOX)), 1)}
    compiled = C.run(bench, C.two_squares()["zorder"],
                     frame_overrides={"B": FrameOverride(C.KEY, "B", None, OUT)},
                     placements={"A": C.BENCH, "B": C.BENCH})
    assert "B" in painted(compiled) and compiled.instances["B"].visibility == OUT
    label, _visible = S.frame_label_map([ann("A", compiled.instances["A"].visible)], C.HW,
                                        {"A": 1, "B": 2}, compiled, set())
    assert int(np.count_nonzero(label == 1)) == 32 * 32 - 20 * 20
    assert not np.any(label == 2)


def test_any_other_painted_part_without_an_annotation_still_fails_loudly():
    scene = C.two_squares()
    compiled = C.run(scene["keyframes"], scene["zorder"])          # B seen, not annotated
    assert compiled.instances["B"].visibility != OUT and "B" in painted(compiled)
    with pytest.raises(KeyError, match="B"):
        S.frame_label_map([ann("A", compiled.instances["A"].visible)], C.HW,
                          {"A": 1, "B": 2}, compiled, set())
