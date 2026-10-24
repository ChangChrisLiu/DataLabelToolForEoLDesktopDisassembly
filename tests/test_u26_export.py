"""Task U26, item 2: every mask annotation of the COCO export carries its amodal shape.

U15 found the exporter had never written the amodal (whole) shape and gave it
to a part 被线挡住 only. The dataset's point is layered amodal masks, so now
every mask annotation has ``amodal_segmentation`` (compressed RLE, the same
coordinates as ``segmentation``), ``amodal_bbox`` and ``amodal_area``;
``--no-amodal`` leaves them out for size, ``--roi-crop`` crops them too, and a
part 被线挡住（完全） is still a crowd region with its amodal box.

Rulings on the first round: a part the labelled parts cover completely is
exported the same way, as a crowd region with its amodal shape (``--no-hidden``
drops it), and ``visible ⊆ amodal`` holds in the file -- a visible mask drawn
outside the shape widens it (``info.amodal_union_rows``).

Scene: :mod:`vis_review_scene` -- a part wholly under another, one half under
another, a 4 px one, a cable on ``cpu.01`` at 13, ``psu.01`` 「看不见」 at 12.
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
from tda.core.model import FrameKey, FrameOverride

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
    assert doc["info"]["amodal_union_rows"] == 0          # nothing was drawn outside


def test_a_part_hidden_by_a_cable_keeps_the_crowd_rule(scene, tmp_path):
    doc, _out = export(scene, tmp_path)
    hidden = ann(doc, V.CABLE_STEP, V.CABLED)
    assert hidden["attributes"]["occluder"] == "cable"
    assert hidden["iscrowd"] == 1 and hidden["area"] == 0
    assert hidden["bbox"] == hidden["amodal_bbox"]
    assert hidden["amodal_area"] > 0


def test_a_part_covered_by_labelled_parts_is_a_crowd_region(scene, tmp_path):
    """``ram_latch.03`` lies wholly under a screw on all 14 frames: exported like a 完全."""
    doc, _out = export(scene, tmp_path)
    assert doc["info"]["hidden"] is True
    covered = [a for a in doc["annotations"] if a["attributes"]["instance_key"] == V.HIDDEN]
    steps = {i["id"]: i["extra"]["step"] for i in doc["images"]}
    assert sorted(steps[a["image_id"]] for a in covered) == list(range(1, 15))
    for a in covered:
        attrs = a["attributes"]
        assert (attrs["visibility"], attrs["occluder"]) == ("occluded_full", None)
        assert a["iscrowd"] == 1 and a["area"] == 0
        assert masks.rle_area(a["segmentation"]) == 0
        assert a["bbox"] == a["amodal_bbox"] and a["amodal_area"] > 0
        compiled = scene.truth.compile(V.key(steps[a["image_id"]])).instances[V.HIDDEN]
        assert np.array_equal(decode(a["amodal_segmentation"]), compiled.amodal)

    # --no-hidden: as before U26 -- the covered part goes, the cable's 完全 stays
    slim, _ = export(scene, tmp_path, "slim.json", hidden=False)
    assert slim["info"]["hidden"] is False
    assert not [a for a in slim["annotations"] if a["attributes"]["instance_key"] == V.HIDDEN]
    assert ann(slim, V.CABLE_STEP, V.CABLED)["iscrowd"] == 1
    assert len(slim["annotations"]) == len(doc["annotations"]) - len(covered)


def test_a_visible_mask_drawn_outside_the_shape_widens_it(tmp_path):
    """A 「只改这一帧」 mask away from the shape (D2 S15 ``ssd.02``): amodal ∪ visible."""
    scene = V.build(tmp_path / "scene", confirm=())    # its own: this one gets an override
    try:
        key = V.key(5)
        shape = scene.truth.compile(key).instances[V.HALF].amodal
        elsewhere = np.zeros(V.HW, dtype=bool)
        elsewhere[240:250, 240:250] = True             # a corner no shape reaches
        assert not (elsewhere & shape).any()
        scene.db.set_frame_override(FrameOverride(key, V.HALF, masks.encode_rle(elsewhere)))
        doc, _ = export(scene, tmp_path, "c.json")
    finally:
        scene.close(force=True)
    widened = ann(doc, 5, V.HALF)
    assert np.array_equal(decode(widened["segmentation"]), elsewhere)
    assert np.array_equal(decode(widened["amodal_segmentation"]), shape | elsewhere)
    assert widened["amodal_area"] == int(shape.sum()) + 100
    assert doc["info"]["amodal_union_rows"] == 1
    for a in doc["annotations"]:                       # and it holds everywhere
        assert not (decode(a["segmentation"]) & ~decode(a["amodal_segmentation"])).any()


def test_no_amodal_leaves_out_what_u26_added_and_nothing_else(scene, tmp_path):
    """Review N1: a row with a reason keeps the two fields U15 / U24 wrote."""
    full, _ = export(scene, tmp_path, "full.json")
    slim, _ = export(scene, tmp_path, "slim.json", amodal=False)
    assert slim["info"]["amodal"] is False
    assert not any("amodal_area" in a for a in slim["annotations"])
    reasoned = [a for a in slim["annotations"] if a["attributes"]["occluder"]]
    assert reasoned and all("amodal_segmentation" in a and "amodal_bbox" in a
                            for a in reasoned)
    assert not any(field in a for a in slim["annotations"] if not a["attributes"]["occluder"]
                   for field in AMODAL)

    # the same annotations otherwise -- the cable's 完全 one included, its box
    # still the amodal one
    def u15(a: dict) -> dict:
        kept = ("amodal_segmentation", "amodal_bbox") if a["attributes"]["occluder"] else ()
        return {k: v for k, v in a.items() if k not in AMODAL or k in kept}

    assert slim["annotations"] == [u15(a) for a in full["annotations"]]
    hidden = ann(slim, V.CABLE_STEP, V.CABLED)
    assert hidden["iscrowd"] == 1 and hidden["bbox"] == ann(full, V.CABLE_STEP,
                                                             V.CABLED)["amodal_bbox"]


#: ``export_coco`` of this scene (confirmed frames 10-14) with main's code at
#: a48f1ad, the commit U26 branched from -- written by
#: ``.cache/u26/make_pre_fixture.py`` from a ``git archive`` of it: 5 images,
#: 172 annotations, one (the cable's 完全 on ``cpu.01``) with the two U15 fields.
PRE_U26_SHA256 = "41ea27a2be6396cc664be8b2c126944fc919178723af50b32b49a3d55d90f240"
PRE_U26_BYTES = 117_748
#: The ``info`` keys U26 added; the pre-U26 file has ``version`` "1".
U26_INFO = ("amodal", "amodal_union_rows", "hidden")


def test_no_amodal_no_hidden_writes_the_pre_u26_file(tmp_path):
    """Review N1's invariant: byte for byte, but for what ``info`` says of the switches.

    The info block records the two switches and the format's version ("2"),
    so it differs by design; with it put back as it was, the file is the one
    main wrote, to the byte (the real D11, with five rows 被线挡住, is checked
    the same way in the U26 report).
    """
    import hashlib

    scene = V.build(tmp_path / "scene")
    try:
        out = tmp_path / "slim.json"
        doc = export_coco(scene.db, scene.tax, [V.DESKTOP], V.VIEW, str(out),
                          truth=scene.truth, amodal=False, hidden=False)
    finally:
        scene.close(force=True)
    assert doc["info"]["version"] == "2"
    assert {k: doc["info"][k] for k in U26_INFO} == {
        "amodal": False, "amodal_union_rows": 0, "hidden": False}
    info = {k: v for k, v in doc["info"].items() if k not in U26_INFO}
    info["version"] = "1"
    # written the way the exporter writes (text mode: CRLF here, as main's was)
    assert out.read_text(encoding="utf-8") == json.dumps(doc, ensure_ascii=True, indent=1)
    redone = tmp_path / "pre.json"
    redone.write_text(json.dumps(dict(doc, info=info), ensure_ascii=True, indent=1),
                      encoding="utf-8")
    pre = redone.read_bytes()
    assert len(pre) == PRE_U26_BYTES
    assert hashlib.sha256(pre).hexdigest() == PRE_U26_SHA256


def test_with_visible_unites_fills_in_and_refuses_two_sizes():
    """Review N3: no shape -> the visible mask; two sizes -> an error, not silence."""
    from tda.core.export.coco import _with_visible

    def rle(y0, y1, x0, x1, hw=(16, 16)):
        m = np.zeros(hw, dtype=bool)
        m[y0:y1, x0:x1] = True
        return masks.encode_rle(m)

    shape, inside, outside = rle(2, 8, 2, 8), rle(3, 5, 3, 5), rle(10, 12, 10, 12)
    assert _with_visible(shape, inside) == (shape, False)
    united, grew = _with_visible(shape, outside)
    assert grew and masks.rle_area(united) == 36 + 4
    assert _with_visible(None, inside) == (inside, True)            # N3a
    empty = masks.encode_rle(np.zeros((16, 16), dtype=bool))
    assert _with_visible(None, empty) == (None, False)              # neither: null triple
    assert _with_visible(None, None) == (None, False)
    with pytest.raises(ValueError, match="two frames"):            # N3b
        _with_visible(shape, rle(3, 5, 3, 5, hw=(16, 20)))


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
    slim, _ = export(scene, tmp_path, "slim.json", amodal=False)
    # the parts hidden completely still need their boxes: ram_latch.03 is
    # under a screw on every frame
    assert sorted(steps) == sorted(i["extra"]["step"] for i in slim["images"])
    steps.clear()
    export(scene, tmp_path, "slimmer.json", amodal=False, hidden=False)
    # ... and without it only the 被线挡住（完全） part does: its frame only
    assert steps == [V.CABLE_STEP]


def test_the_cli_flags(scene, tmp_path, capsys):
    paths = tmp_path / "paths.yaml"
    root = tmp_path / "backups"
    root.mkdir()
    paths.write_text(yaml.safe_dump({"db_path": scene.db.path, "cache_dir": str(tmp_path),
                                     "backup_dir": str(root)}), encoding="utf-8")
    sizes, counts = {}, {}
    for name, flag in (("with", []), ("without", ["--no-amodal"]),
                       ("unhidden", ["--no-hidden"])):
        out = tmp_path / f"{name}.json"
        code = main(["--paths", str(paths), "export-coco", "--desktops", str(V.DESKTOP),
                     "--view", V.VIEW, "--out", str(out), *flag])
        printed = capsys.readouterr().out
        assert code == EXIT_OK
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["info"]["amodal"] is (name != "without")
        assert doc["info"]["hidden"] is (name != "unhidden")
        # --no-amodal keeps the two U15 fields on a row with a reason (N1)
        assert all(("amodal_segmentation" in a)
                   is (name != "without" or bool(a["attributes"]["occluder"]))
                   for a in doc["annotations"])
        assert all(("amodal_area" in a) is (name != "without") for a in doc["annotations"])
        crowd = sum(a["iscrowd"] for a in doc["annotations"])
        assert f"[export-coco] {crowd} annotation(s) are parts hidden completely" in printed
        sizes[name], counts[name] = out.stat().st_size, len(doc["annotations"])
    assert sizes["without"] < sizes["with"]
    # ram_latch.03 on each confirmed frame (the CLI exports confirmed rows only)
    assert counts["unhidden"] == counts["with"] - len(V.CONFIRMED)
