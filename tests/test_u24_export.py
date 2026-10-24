"""Task U24: what the visibility review leaves in the exports, ``status`` and ``check``.

* COCO: ``attributes.visibility_reviewed`` -- true where Review's 可见性审查
  stamped the row for the label and the reason it has; ``occluder`` carries the
  new ``viewpoint`` (角度看不到), whose 完全 is a crowd region with its amodal
  shape, like a cable's.
* VLM: a part 角度看不到 is "not visible from this viewpoint" -- not "hidden by".
* ``status``: 「可见性审查 n/m」 per view; ``check``: nothing new.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import contextlib
import io

import pytest

import vis_review_scene as V
from session_scene import DESKTOP, VIEW
from tda.core import vis_review as VR
from tda.core.export import export_coco
from tda.core.export import vlm_reasoning as R
from tda.core.export.coco import load_ctx
from tda.core.export.vlm import _read_view
from tda.core.export.vlm_tasks import seen_as, unseen_reason
from tda.core.occluder import FULL, PARTIAL

STEPS = [10, 11, 12, 13, 14]


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory):
    """The scene, with two verdicts: Z on the half-hidden part at 14, Space on
    the tiny part at 12 (the wholly hidden part, with no reason, is not
    exported at all: nothing of it is seen)."""
    session = V.build(tmp_path_factory.mktemp("u24_export"))
    session.goto(14, force=True)
    session.review_visibility(V.HALF, VR.VERDICT_VIEWPOINT, FULL)
    session.goto(12, force=True)
    session.review_visibility(V.TINY, VR.VERDICT_COMPUTED)
    yield session
    session.close(force=True)


@pytest.fixture(scope="module")
def coco(reviewed, tmp_path_factory):
    out = tmp_path_factory.mktemp("u24_coco") / "c.json"
    return export_coco(reviewed.db, reviewed.tax, [DESKTOP], VIEW, str(out),
                       truth=reviewed.truth)


def ann(doc: dict, step: int, instance: str) -> dict:
    image = next(i["id"] for i in doc["images"] if i["extra"]["step"] == step)
    return next(a for a in doc["annotations"]
                if a["image_id"] == image and a["attributes"]["instance_key"] == instance)


def test_coco_says_which_labels_a_human_audited(coco):
    assert ann(coco, 12, V.TINY)["attributes"]["visibility_reviewed"] is True
    assert ann(coco, 11, V.TINY)["attributes"]["visibility_reviewed"] is False
    assert ann(coco, 14, "chassis")["attributes"]["visibility_reviewed"] is False
    assert ann(coco, V.CABLE_STEP, V.CABLED)["attributes"]["visibility_reviewed"] is False
    assert all(isinstance(a["attributes"]["visibility_reviewed"], bool)
               for a in coco["annotations"])


def test_coco_carries_viewpoint_like_a_cable(coco):
    hidden = ann(coco, 14, V.HALF)
    attrs = hidden["attributes"]
    assert (attrs["visibility"], attrs["occluder"], attrs["visibility_reviewed"]) == (
        FULL, "viewpoint", True)
    assert hidden["iscrowd"] == 1 and hidden["area"] == 0
    assert hidden["amodal_segmentation"] and hidden["bbox"] == hidden["amodal_bbox"]
    assert ann(coco, V.CABLE_STEP, V.CABLED)["attributes"]["occluder"] == "cable"
    assert ann(coco, 13, V.HALF)["attributes"]["occluder"] is None


def test_a_stamp_that_no_longer_covers_the_label_exports_false(reviewed, tmp_path):
    """The stamp records the label it was given for; a moved label is not reviewed."""
    key = V.key(12)
    flags = reviewed.db.instance_flags(key, V.TINY)
    reviewed.db.set_instance_flags(key, V.TINY, dict(flags, visibility="visible_tiny"))
    try:
        doc = export_coco(reviewed.db, reviewed.tax, [DESKTOP], VIEW,
                          str(tmp_path / "c.json"), truth=reviewed.truth)
        assert ann(doc, 12, V.TINY)["attributes"]["visibility_reviewed"] is False
    finally:
        reviewed.db.set_instance_flags(key, V.TINY, flags)


def test_the_vlm_says_not_visible_from_this_viewpoint(reviewed):
    ctx = load_ctx(reviewed.db, reviewed.tax, DESKTOP, VIEW)
    frames, _order = _read_view(reviewed.db, ctx, DESKTOP, VIEW, set(), True)
    row = frames[14].rows[V.HALF]
    assert row["occluder"] == "viewpoint"
    assert seen_as(row) == {"visibility": FULL, "occluder": "viewpoint"}
    assert unseen_reason(V.HALF, row, "oak2") == (
        "ram_module.01 is in place, not visible from this viewpoint, in view oak2")
    assert "hidden by" not in unseen_reason(V.HALF, row, VIEW)
    # the cable keeps its own words
    assert unseen_reason("x.01", {"visibility": FULL, "occluder": "cable"}, VIEW) == (
        "x.01 is in place, hidden by a cable, in view scan")
    partial = R.observe(V.HALF, "installed", "oak2", [1, 2, 3, 4], PARTIAL,
                        occluder="viewpoint")
    assert partial["evidence"]["occluder"] == "viewpoint"


def test_status_says_how_far_each_view_is(reviewed):
    from tda import pipeline as P
    from tda.cli import format_desktop, format_status

    [row] = P.status_rows(reviewed.db, {DESKTOP})
    scan = row["views"][VIEW]
    assert (scan["vis_reviewed"], scan["vis_total"]) == (2, 17)
    assert (row["views"]["oak1"]["vis_reviewed"], row["views"]["oak1"]["vis_total"]) == (0, 0)
    detail = format_desktop(reviewed.db, row)
    scan_line = next(line for line in detail.splitlines() if line.startswith(f"  {VIEW} "))
    assert scan_line.rstrip().endswith("可见性审查 2/17")
    table = format_status([row])
    assert "可见性审查 / visibility review" in table
    assert f"D{DESKTOP:02d}      | scan 2/17 | oak1 0/0 | oak2 0/0 | rs 0/0" in table


def test_status_prints_nothing_about_it_with_nothing_to_audit():
    from tda.cli import format_vis_review

    views = {v: {"vis_reviewed": 0, "vis_total": 0} for v in ("scan", "oak1", "oak2", "rs")}
    assert format_vis_review([{"desktop": 1, "views": views}]) == []


def test_check_reports_nothing_new(reviewed):
    from tda.cli_app import _check_one

    class Args:
        limit = 1000

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _check_one(reviewed.db, reviewed.tax, Args(), DESKTOP, VIEW)
    said = out.getvalue()
    assert "viewpoint" not in said and "可见性" not in said
    assert "occluder_without_shape" not in said
