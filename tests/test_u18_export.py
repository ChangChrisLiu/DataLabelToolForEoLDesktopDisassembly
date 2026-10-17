"""Task U18: the 框外零件 in the exports, ``cli check`` and ``status``.

COCO writes a context object under its class (``other`` its own category,
listed only when used) with ``attributes.context = true``, its ``label``, its
``instance_key`` and no state or placement; under ``--roi-crop`` it is kept only
where it reaches into the ROI. The VLM export never sees one. ``cli check`` has
no ``missing_shape`` for it and ``status`` counts it apart.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication
from pycocotools.coco import COCO

from session_scene import CHASSIS, DESKTOP, VIEW, make_session, rect, seed_shapes
from tda.core.export import export_coco, export_vlm
from tda.core.export.coco import category_ids, load_ctx, other_category
from tda.core.export.vlm import _read_view
from tda.core.model import FrameKey
from tda.ui.session_api import SCOPE_KEYFRAME

STEP = 14


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def draw(session, instance, mask) -> None:
    session.begin_edit(instance)
    session.set_editing_mask(mask)
    session.commit_edit(SCOPE_KEYFRAME)


@pytest.fixture
def scene(qapp, tmp_path):
    """The D13 scene at 14, drawn; a bracket inside the ROI's corner, a cup outside."""
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    seed_shapes(session, STEP)
    session.goto(STEP, force=True)
    bracket = session.add_context_object("cooler_bracket", "台面上的支架")
    draw(session, bracket, rect(36, 40, 48, 52))
    cup = session.add_context_object("other", "茶杯")
    draw(session, cup, rect(52, 52, 62, 62))
    yield session, bracket, cup
    session.close(force=True)


def _ann(doc: dict, step: int, instance: str) -> dict:
    images = {im["id"]: im for im in doc["images"]}
    for ann in doc["annotations"]:
        if (images[ann["image_id"]]["extra"]["step"] == step
                and ann["attributes"]["instance_key"] == instance):
            return ann
    raise AssertionError(f"no annotation for {instance} at {step}")


def test_coco_writes_them_under_their_class_with_their_own_attributes(scene, tmp_path):
    session, bracket, cup = scene
    out = tmp_path / "coco.json"
    doc = export_coco(session.db, session.tax, [DESKTOP], VIEW, str(out), only_verified=False)
    ann = _ann(doc, STEP, bracket)
    assert ann["category_id"] == category_ids(session.tax)["cooler_bracket"]
    assert ann["attributes"]["context"] is True
    assert ann["attributes"]["label"] == "台面上的支架"
    assert ann["attributes"]["instance_key"] == bracket
    assert ann["attributes"]["state"] is None and ann["attributes"]["placement"] is None
    assert ann["area"] == 12 * 12
    # `other` is a category of its own, one past the taxonomy, listed because used
    other = _ann(doc, STEP, cup)
    assert other["category_id"] == len(session.tax.classes) + 1
    assert other_category(session.tax) in doc["categories"]
    assert other["attributes"]["label"] == "茶杯"
    # the machine's parts are written as they always were
    chassis = _ann(doc, STEP, CHASSIS)
    assert "context" not in chassis["attributes"] and "label" not in chassis["attributes"]
    assert chassis["attributes"]["state"] is not None
    # still a COCO file
    coco = COCO(str(out))
    assert coco.annToMask(coco.anns[ann["id"]]).sum() == 12 * 12


def test_no_other_category_without_an_other_object(scene, tmp_path):
    session, bracket, cup = scene
    session.db.delete_keyframe(session.db.keyframes(DESKTOP, VIEW, cup)[0].id)
    doc = export_coco(session.db, session.tax, [DESKTOP], VIEW, str(tmp_path / "c.json"),
                      only_verified=False)
    assert len(doc["categories"]) == len(session.tax.classes)
    assert all(a["attributes"]["instance_key"] != cup for a in doc["annotations"])


def test_roi_crop_keeps_one_only_where_it_reaches_into_the_roi(scene, tmp_path):
    session, bracket, cup = scene
    db = session.db
    with db.conn:
        db.conn.execute("UPDATE pose_segment SET roi_json=? WHERE desktop=? AND view=?",
                        (json.dumps([0, 0, 44, 50]), DESKTOP, VIEW))
    if not db.pose_segments(DESKTOP, VIEW):
        db.set_pose_segment(DESKTOP, VIEW, 1, 1, STEP, STEP, None, None)
        with db.conn:
            db.conn.execute("UPDATE pose_segment SET roi_json=? WHERE desktop=? AND view=?",
                            (json.dumps([0, 0, 44, 50]), DESKTOP, VIEW))
    doc = export_coco(db, session.tax, [DESKTOP], VIEW, str(tmp_path / "roi.json"),
                      only_verified=False, roi_crop=True)
    kept = _ann(doc, STEP, bracket)                   # 36-48 x 40-52 meets 0-44 x 0-50
    assert kept["segmentation"]["size"] == [50, 44]
    assert kept["area"] == (44 - 36) * (50 - 40)
    with pytest.raises(AssertionError):
        _ann(doc, STEP, cup)                          # wholly outside: dropped


def test_the_vlm_export_never_sees_them(scene, tmp_path):
    session, bracket, cup = scene
    ctx = load_ctx(session.db, session.tax, DESKTOP, VIEW)
    assert bracket not in ctx.instances and bracket in ctx.context
    frames, _order = _read_view(session.db, ctx, DESKTOP, VIEW, set(), False)
    assert bracket not in frames[STEP].rows and cup not in frames[STEP].rows
    assert session.confirm_frame()
    out = tmp_path / "vlm.jsonl"
    export_vlm(session.db, session.tax, [DESKTOP], VIEW, str(out))
    text = out.read_text(encoding="utf-8")
    assert text and "ctx." not in text


def test_the_summary_line_counts_them_apart(scene, tmp_path):
    from tda.cli_app import _context_count

    session, bracket, cup = scene
    doc = export_coco(session.db, session.tax, [DESKTOP], VIEW, str(tmp_path / "c.json"),
                      only_verified=False)
    assert _context_count(doc) == sum(1 for a in doc["annotations"]
                                      if a["attributes"].get("context"))
    assert _context_count(doc) >= 2


def test_check_and_status_leave_them_out_of_the_machine(scene):
    from tda.cli import format_desktop
    from tda.pipeline import status_rows

    session, bracket, cup = scene
    db = session.db
    for step in range(1, STEP + 1):
        compiled = session.truth.compile(FrameKey(DESKTOP, step, VIEW))
        assert not [p for p in compiled.problems if "ctx." in p], step
    [row] = status_rows(db, {DESKTOP})
    machine = [k for k in db.instances(DESKTOP) if not k.startswith("ctx.")]
    assert row["instances"] == len(machine)
    assert row["context"] == 2
    keyframes = row["views"][VIEW]["keyframes"]
    assert keyframes == sum(1 for kf in db.keyframes(DESKTOP, VIEW)
                            if not kf.instance.startswith("ctx."))
    text = format_desktop(db, row)
    assert "框外零件" in text and bracket in text and cup in text
