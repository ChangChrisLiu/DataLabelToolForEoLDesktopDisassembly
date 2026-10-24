"""Task U15: the exports say 「被线挡住」.

COCO: every annotation carries ``attributes.occluder``; a part with an occluder
carries its whole shape (``amodal_segmentation`` / ``amodal_bbox``), and one
hidden completely is exported with an empty visible segmentation -- where a part
with nothing visible and no reason given is still dropped. VLM: an
``observe`` of a part under a cable says so, and V14 abstains with the reason.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from session_scene import CHASSIS, DESKTOP, VIEW, make_session, seed_shapes
from tda.core import masks
from tda.core.export import export_coco, export_vlm
from tda.core.export import vlm_reasoning as R
from tda.core.export.coco import load_ctx
from tda.core.export.vlm import _read_view
from tda.core.export.vlm_tasks import seen_as, unseen_reason
from tda.core.model import FrameKey, FrameOverride
from tda.core.occluder import FULL, PARTIAL

STEP = 14


@pytest.fixture
def scene(tmp_path):
    """The D13 scene at 14, drawn; the chassis 完全, a second part 部分."""
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    seed_shapes(session, STEP)
    other = next(r["key"] for r in session.instance_rows()
                 if r["key"] != CHASSIS and r["has_shape"])
    session.cycle_cable(CHASSIS)
    session.cycle_cable(other)
    session.cycle_cable(other)
    return session, other


def _ann(doc: dict, step: int, instance: str) -> dict:
    images = {im["id"]: im for im in doc["images"]}
    for ann in doc["annotations"]:
        if (images[ann["image_id"]]["extra"]["step"] == step
                and ann["attributes"]["instance_key"] == instance):
            return ann
    raise AssertionError(f"no annotation for {instance} at {step}")


def test_coco_keeps_a_part_hidden_by_a_cable(scene, tmp_path):
    session, other = scene
    out = tmp_path / "coco.json"
    doc = export_coco(session.db, session.tax, [DESKTOP], VIEW, str(out), only_verified=False)
    hidden = _ann(doc, STEP, CHASSIS)
    attrs = hidden["attributes"]
    assert (attrs["visibility"], attrs["occluder"]) == (FULL, "cable")
    assert hidden["area"] == 0 and masks.rle_area(hidden["segmentation"]) == 0
    amodal = hidden["amodal_segmentation"]
    assert masks.rle_area(amodal) > 0
    assert hidden["amodal_bbox"] == hidden["bbox"]                 # where the part is
    compiled = session.compiled().instances[CHASSIS].amodal
    assert np.array_equal(masks.decode_rle(amodal), compiled)

    partly = _ann(doc, STEP, other)
    assert (partly["attributes"]["visibility"], partly["attributes"]["occluder"]) == (
        PARTIAL, "cable")
    assert partly["area"] > 0 and masks.rle_area(partly["amodal_segmentation"]) > 0

    plain = [a for a in doc["annotations"] if a["attributes"]["instance_key"]
             not in (CHASSIS, other)]
    assert plain and all(a["attributes"]["occluder"] is None for a in plain)
    # every mask annotation has its amodal shape now (U26), a reason or not
    assert all(masks.rle_area(a["amodal_segmentation"]) > 0 for a in plain)
    # the chassis elsewhere carries no reason
    assert _ann(doc, STEP - 1, CHASSIS)["attributes"]["occluder"] is None
    # and the file is still a COCO file
    coco = COCO(str(out))
    assert coco.annToMask(coco.anns[hidden["id"]]).sum() == 0
    assert coco_mask.area(_coco(amodal)) == masks.rle_area(amodal)


def _coco(rle: dict) -> dict:
    counts = rle["counts"]
    return {"size": rle["size"], "counts": counts.encode("ascii") if isinstance(counts, str)
            else counts}


def test_coco_roi_crop_keeps_the_hidden_part(scene, tmp_path):
    session, _other = scene
    db = session.db
    db.set_pose_segment(DESKTOP, VIEW, 1, 1, STEP, STEP, None, None)
    with db.conn:
        db.conn.execute("UPDATE pose_segment SET roi_json=? WHERE desktop=? AND view=?",
                        (json.dumps([0, 0, 40, 48]), DESKTOP, VIEW))
    doc = export_coco(db, session.tax, [DESKTOP], VIEW, str(tmp_path / "roi.json"),
                      only_verified=False, roi_crop=True)
    hidden = _ann(doc, STEP, CHASSIS)
    assert hidden["segmentation"]["size"] == [48, 40] and hidden["area"] == 0
    assert hidden["amodal_segmentation"]["size"] == [48, 40]
    assert hidden["bbox"] == hidden["amodal_bbox"]


def test_a_bare_label_carries_no_reason(scene, tmp_path):
    """The 3 key's bare 完全遮挡 is exported as it always was: no reason, its pixels.

    Its amodal shape it has like every mask annotation (U26).
    """
    session, other = scene
    session.cycle_cable(other)                       # 部分 -> 撤回
    session.set_visibility(other, FULL)              # ... and a bare label instead
    doc = export_coco(session.db, session.tax, [DESKTOP], VIEW,
                      str(tmp_path / "c.json"), only_verified=False)
    bare = _ann(doc, STEP, other)
    assert bare["attributes"]["visibility"] == FULL
    assert bare["attributes"]["occluder"] is None
    assert bare["area"] > 0                          # its pixels were never taken away
    assert bare["iscrowd"] == 0 and bare["amodal_area"] >= bare["area"]


def test_nothing_visible_and_no_reason_is_still_dropped():
    from tda.core.export.coco import _annotation

    empty = masks.encode_rle(np.zeros((16, 16), dtype=bool))
    whole = np.zeros((16, 16), dtype=bool)
    whole[2:8, 3:12] = True
    row = {"visible_rle": empty, "geom_type": "mask"}
    bare = {"visibility": FULL, "occluder": None}
    assert _annotation(1, 1, 1, row, bare, None, False) is None
    assert _annotation(1, 1, 1, row, bare, None, False, masks.encode_rle(whole)) is None
    hidden = _annotation(1, 1, 1, row, {"visibility": FULL, "occluder": "cable"}, None,
                         False, masks.encode_rle(whole))
    assert hidden["bbox"] == hidden["amodal_bbox"] == [3, 2, 9, 6] and hidden["area"] == 0


# --------------------------------------------------------------------------- #
# VLM
# --------------------------------------------------------------------------- #
def test_the_vlm_rows_and_words_carry_the_reason(scene):
    session, other = scene
    ctx = load_ctx(session.db, session.tax, DESKTOP, VIEW)
    frames, _order = _read_view(session.db, ctx, DESKTOP, VIEW, set(), False)
    rows = frames[STEP].rows
    assert rows[CHASSIS]["occluder"] == "cable" and rows[other]["occluder"] == "cable"
    assert "occluder" not in frames[STEP - 1].rows[CHASSIS]
    assert CHASSIS not in frames[STEP].pointable          # nothing to point at
    assert other in frames[STEP].pointable                 # 部分 is still answerable
    assert seen_as(rows[other]) == {"visibility": PARTIAL, "occluder": "cable"}
    assert seen_as({"visibility": "visible"}) == {"visibility": "visible"}
    assert unseen_reason(CHASSIS, rows[CHASSIS], VIEW) == (
        "chassis is in place, hidden by a cable, in view scan")
    assert unseen_reason(CHASSIS, {"visibility": FULL}, VIEW) == (
        "chassis is occluded_full in view scan")
    step = R.observe(other, "installed", VIEW, [1, 2, 3, 4], PARTIAL, occluder="cable")
    assert step["evidence"]["occluder"] == "cable"
    assert "occluder" not in R.observe(other, "x", VIEW, [1, 2, 3, 4], PARTIAL)["evidence"]


def test_a_vlm_observation_of_a_part_under_a_cable_says_so(scene, tmp_path):
    session, other = scene
    assert session.confirm_frame()                  # V1 speaks of confirmed frames
    out = tmp_path / "vlm.jsonl"
    export_vlm(session.db, session.tax, [DESKTOP], VIEW, str(out), tasks=("V1",))
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    [v1] = [r for r in records if r["label"]["step"] == STEP]
    evidence = v1["label"]["evidence"]["attributes"]
    assert evidence[other]["occluder"] == "cable"
    assert CHASSIS not in evidence                  # nothing of it to point at
    assert all("occluder" not in attrs for name, attrs in evidence.items() if name != other)
    observed = {s["target"]: s for s in v1["label"]["rationale"]["steps"]
                if s["op"] == "observe"}
    assert observed[other]["evidence"]["occluder"] == "cable"
    assert all("occluder" not in s["evidence"] for name, s in observed.items()
               if name != other)
