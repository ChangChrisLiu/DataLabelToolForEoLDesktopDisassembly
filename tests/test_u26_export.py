"""Task U26, item 2: every mask annotation of the COCO export carries its amodal shape.

U15 found the exporter had never written the amodal (whole) shape and gave it
to a part 被线挡住 only. The dataset's point is layered amodal masks, so now
every mask annotation has ``amodal_segmentation`` (compressed RLE, the same
coordinates as ``segmentation``), ``amodal_bbox`` and ``amodal_area``;
``--no-amodal`` leaves them out for size, ``--roi-crop`` crops them too, and a
part 被线挡住（完全） is still a crowd region with its amodal box.

Scene: :mod:`vis_review_scene` -- a part wholly under another (not exported:
nothing of it is seen), one half under another, a 4 px one, a cable on
``cpu.01`` at 13, ``psu.01`` 「看不见」 at 12.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import yaml
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

import vis_review_scene as V
from tda.cli import main
from tda.cli_common import EXIT_OK
from tda.core import masks
from tda.core.export import export_coco
from tda.core.model import FrameKey

AMODAL = ("amodal_segmentation", "amodal_bbox", "amodal_area")
#: x0, y0, x1, y1: the left half of the 256x256 canvas, cutting cells in two
ROI = (0, 0, 120, 256)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    session = V.build(tmp_path_factory.mktemp("u26_export"))
    yield session
    session.close(force=True)


def export(session, tmp_path, name: str = "c.json", **kw) -> tuple[dict, str]:
    out = tmp_path / name
    doc = export_coco(session.db, session.tax, [V.DESKTOP], V.VIEW, str(out),
                      only_verified=False, truth=session.truth, **kw)
    return doc, str(out)


def ann(doc: dict, step: int, instance: str) -> dict:
    image = next(i["id"] for i in doc["images"] if i["extra"]["step"] == step)
    return next(a for a in doc["annotations"]
                if a["image_id"] == image and a["attributes"]["instance_key"] == instance)


def decode(rle: dict) -> np.ndarray:
    return masks.decode_rle(rle)


def test_every_mask_annotation_carries_its_amodal_shape(scene, tmp_path):
    doc, _out = export(scene, tmp_path)
    assert doc["info"]["amodal"] is True
    anns = doc["annotations"]
    assert anns and all(isinstance(a["segmentation"], dict) for a in anns)
    for a in anns:
        amodal = a["amodal_segmentation"]
        assert amodal["size"] == a["segmentation"]["size"]
        assert isinstance(amodal["counts"], str)                  # compressed RLE
        assert a["amodal_area"] == masks.rle_area(amodal) > 0
        assert a["amodal_bbox"] == masks.rle_bbox_xywh(amodal)
        visible = decode(a["segmentation"])
        assert not (visible & ~decode(amodal)).any()              # visible inside amodal
    # the shape is the compiler's, from the keyframe -- not the visible mask
    half = ann(doc, 14, V.HALF)
    compiled = scene.truth.compile(V.key(14)).instances[V.HALF].amodal
    assert np.array_equal(decode(half["amodal_segmentation"]), compiled)
    assert half["amodal_area"] > half["area"]
    assert half["attributes"]["occlusion_ratio"] == pytest.approx(
        1 - half["area"] / half["amodal_area"])


def test_a_part_hidden_by_a_cable_keeps_the_crowd_rule(scene, tmp_path):
    doc, _out = export(scene, tmp_path)
    hidden = ann(doc, V.CABLE_STEP, V.CABLED)
    assert hidden["attributes"]["occluder"] == "cable"
    assert hidden["iscrowd"] == 1 and hidden["area"] == 0
    assert hidden["bbox"] == hidden["amodal_bbox"]
    assert hidden["amodal_area"] > 0
    # the part wholly under a labelled screw, with no reason, is still not exported
    assert not [a for a in doc["annotations"] if a["attributes"]["instance_key"] == V.HIDDEN]


def test_no_amodal_leaves_the_fields_out_and_nothing_else(scene, tmp_path):
    full, _ = export(scene, tmp_path, "full.json")
    slim, _ = export(scene, tmp_path, "slim.json", amodal=False)
    assert slim["info"]["amodal"] is False
    assert not any(field in a for a in slim["annotations"] for field in AMODAL)
    # the same annotations otherwise -- the cable's 完全 one included, its box
    # still the amodal one
    stripped = [{k: v for k, v in a.items() if k not in AMODAL} for a in full["annotations"]]
    assert slim["annotations"] == stripped
    hidden = ann(slim, V.CABLE_STEP, V.CABLED)
    assert hidden["iscrowd"] == 1 and hidden["bbox"] == ann(full, V.CABLE_STEP,
                                                             V.CABLED)["amodal_bbox"]


def test_roi_crop_crops_the_amodal_shape_too(tmp_path):
    scene = V.build(tmp_path / "scene")          # its own: this one gets a segment
    try:
        db = scene.db
        db.set_pose_segment(V.DESKTOP, V.VIEW, 1, 1, 14, 14, None, None)
        with db.conn:
            db.conn.execute("UPDATE pose_segment SET roi_json=? WHERE desktop=? AND view=?",
                            (json.dumps(list(ROI)), V.DESKTOP, V.VIEW))
        assert db.pose_segment_for(FrameKey(V.DESKTOP, 14, V.VIEW)).get("roi") == list(ROI)
        full, _ = export(scene, tmp_path, "full.json")
        cropped, out = export(scene, tmp_path, "roi.json", roi_crop=True)
    finally:
        scene.close(force=True)
    x0, y0, x1, y1 = ROI
    assert cropped["annotations"]
    cut = 0
    for a in cropped["annotations"]:
        assert a["amodal_segmentation"]["size"] == a["segmentation"]["size"] == [y1 - y0,
                                                                                x1 - x0]
        step = next(i["extra"]["step"] for i in cropped["images"] if i["id"] == a["image_id"])
        whole = ann(full, step, a["attributes"]["instance_key"])
        expected = decode(whole["amodal_segmentation"])[y0:y1, x0:x1]
        assert np.array_equal(decode(a["amodal_segmentation"]), expected)
        assert a["amodal_area"] == int(expected.sum())
        cut += a["amodal_area"] < whole["amodal_area"]
    assert cut                                       # some shape did reach past the ROI
    COCO(out)                                        # and the file still loads


def test_pycocotools_reads_the_amodal_fields(scene, tmp_path):
    doc, out = export(scene, tmp_path)
    coco = COCO(out)
    assert len(coco.anns) == len(doc["annotations"])
    for a in doc["annotations"]:
        rle = dict(a["amodal_segmentation"], counts=a["amodal_segmentation"]["counts"].encode())
        assert int(coco_mask.area(rle)) == a["amodal_area"]
        assert [int(v) for v in coco_mask.toBbox(rle)] == a["amodal_bbox"]
        # the visible segmentation is untouched
        assert int(coco.annToMask(coco.anns[a["id"]]).sum()) == a["area"]


def test_the_amodal_shapes_cost_one_compilation_per_exported_frame(scene, tmp_path,
                                                                   monkeypatch):
    """Not stored, so derived on every export -- once per frame, and only with them.

    Without them the U2 property holds: an export over an untouched view
    compiles nothing (``tests/test_session_fresh.py``).
    """
    import tda.core.truth as truth_mod

    export(scene, tmp_path, "warm.json")            # the truth table is current now
    original = truth_mod.compile_frame
    steps: list[int] = []
    monkeypatch.setattr(truth_mod, "compile_frame",
                        lambda key, *a, **k: (steps.append(key.step), original(key, *a, **k))[1])
    doc, _ = export(scene, tmp_path, "again.json")
    assert sorted(steps) == sorted(i["extra"]["step"] for i in doc["images"])
    steps.clear()
    export(scene, tmp_path, "slim.json", amodal=False)
    # the one 被线挡住（完全） part still needs its box: its frame only
    assert steps == [V.CABLE_STEP]


def test_the_cli_flag(scene, tmp_path, capsys):
    paths = tmp_path / "paths.yaml"
    root = tmp_path / "backups"
    root.mkdir()
    paths.write_text(yaml.safe_dump({"db_path": scene.db.path, "cache_dir": str(tmp_path),
                                     "backup_dir": str(root)}), encoding="utf-8")
    sizes = {}
    for name, flag in (("with", []), ("without", ["--no-amodal"])):
        out = tmp_path / f"{name}.json"
        code = main(["--paths", str(paths), "export-coco", "--desktops", str(V.DESKTOP),
                     "--view", V.VIEW, "--out", str(out), *flag])
        capsys.readouterr()
        assert code == EXIT_OK
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["info"]["amodal"] is (name == "with")
        assert all(("amodal_segmentation" in a) is (name == "with")
                   for a in doc["annotations"])
        sizes[name] = out.stat().st_size
    assert sizes["without"] < sizes["with"]
