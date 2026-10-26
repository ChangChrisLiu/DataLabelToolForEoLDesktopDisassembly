"""Task U30-PACK: the scanner-view segmentation package for the 3D collaborators.

``scripts/export_scanpack.py`` on two scenes built from the real D2 and D13
sheets (``tests/fixtures/logs``): one 128 x 128 picture per step, a two-image
burst per step on a stand-in "raw drive" with a cache manifest choosing between
them (step 3 chooses ``P_1`` while the frame row still names ``P_0``; step 4's
cache copy is re-encoded), every part a square of its own on a large chassis,
then

* ``HIDDEN`` wholly under ``COVER`` (a crowd row, ``occluded_full``),
* ``HALF`` half under ``HALF_COVER``,
* ``OUTSIDE`` drawn below the ROI,
* ``CABLED`` partly under a cable occluder nobody labels on frame 2,
* ``PATCHED`` given a 「只改这一帧」 mask on frame 3 that reaches over
  ``NEIGHBOR`` -- two confirmed masks claiming the same pixels,
* ``STALE_UPPER`` drawn over a strip of ``STALE_LOWER``, whose confirmed row on
  frame 4 still holds that strip (a row confirmed before the upper part grew,
  as on D2's cover/motherboard) -- an overlap with no per-frame mask, settled
  by the layer order alone,

every frame confirmed (D2: all but the last), and one frame per scene left out
with ``exclude`` (round 2). Pinned: the label map is the union of the visible
masks with the right ids, visible ⊆ amodal, ``clean`` implies equal masks and
no shared pixel, ``occlusion_ratio`` is the masks', the manifest's counts and
hashes, each picture is the chosen source file byte for byte, the exclusion,
the README (example, definitions, Known issues), no internal path in a shipped
file, the output is deterministic, and the build refuses confirmed rows the
inputs no longer agree with.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_scanpack as S  # noqa: E402
from tda.core import masks  # noqa: E402
from tda.core.db import Db  # noqa: E402
from tda.core.logs import import_log, read_desktop_csv  # noqa: E402
from tda.core.model import (  # noqa: E402
    FrameKey,
    FrameOverride,
    OccluderMask,
    ShapeKeyframe,
    ShapePart,
    ZOrderRec,
)
from tda.core.states import needs_geom  # noqa: E402
from tda.core.taxonomy import load_taxonomy  # noqa: E402
from tda.core.truth import TruthService  # noqa: E402
from tda.core.truth_inputs import instances_of, state_of  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "logs"
VIEW = "scan"
HW = (128, 128)
SIDE = 16
ROI = [0, 0, 128, 112]
LAST = {2: 10, 13: 14}
#: The frame each scene leaves out: the frame before a part leaves (D2: the fan
#: connector unplugged at 7; D13: the SSD removed at 10).
EXCLUDED = {2: 6, 13: 9}
EXCLUDE_REASON = "the fixture's layer order is wrong on this frame"
NOTE = "A fixture note for the Known issues."
CHASSIS = "chassis"
HIDDEN, COVER = "screw.motherboard.01", "ram_module.01"
HALF, HALF_COVER = "screw.motherboard.02", "ram_module.02"
OUTSIDE = "screw.motherboard.03"
CABLED, CABLE_STEP = "screw.motherboard.04", 2
PATCHED, NEIGHBOR, PATCH_STEP = "ram_latch.03", "ram_latch.04", 3
STALE_LOWER, STALE_UPPER, STALE_STEP = "screw.motherboard.05", "screw.motherboard.06", 4
CELLS = {COVER: 32, HIDDEN: 33, HALF_COVER: 34, HALF: 35, PATCHED: 40, NEIGHBOR: 41,
         CABLED: 42, STALE_LOWER: 44, STALE_UPPER: 45, OUTSIDE: 60}
#: The columns of STALE_LOWER's cell STALE_UPPER is drawn over (the 12 px left
#: keep it ``visible`` rather than ``visible_tiny``).
STRIP = 2
CHOSEN_P1_STEP, REENCODED_STEP = 3, 4


def rect(x0, y0, x1, y1) -> np.ndarray:
    mask = np.zeros(HW, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def cell(index: int) -> np.ndarray:
    col, row = index % 8, index // 8
    x0, y0 = col * SIDE + 1, row * SIDE + 1
    return rect(x0, y0, x0 + SIDE - 2, y0 + SIDE - 2)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #
def _pictures(root: Path, desktop: int, last: int) -> dict:
    """Two burst images per step on the stand-in raw drive, the cache, its manifest."""
    raw_paths, manifest = {}, {}
    cache = root / "cache" / VIEW / f"D{desktop:02d}"
    cache.mkdir(parents=True, exist_ok=True)
    for step in range(1, last + 1):
        folder = root / "raw" / f"D{desktop:02d}" / f"RGB{step}1"
        folder.mkdir(parents=True, exist_ok=True)
        burst = []
        for p in (0, 1):
            img = np.full((*HW, 3), 40 + 7 * step + 50 * p, dtype=np.uint8)
            img[8:24, 8 + step:24 + step] = 230
            path = folder / f"P_{p}.png"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            burst.append(path.as_posix())
        pick = 1 if step == CHOSEN_P1_STEP else 0
        dest = cache / f"s{step:03d}.png"
        if step == REENCODED_STEP:     # same pixels, other bytes
            cv2.imwrite(str(dest), cv2.imread(burst[pick], cv2.IMREAD_UNCHANGED),
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            shutil.copyfile(burst[pick], dest)
        manifest[str(step)] = {
            "chosen": pick, "reason": "p0_dark" if pick else "p0",
            "metrics": [{"path": b, "p_index": i} for i, b in enumerate(burst)],
            "src": burst[pick]}
        raw_paths[step] = burst
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return raw_paths


def build_scene(root: Path, desktop: int) -> Path:
    """The scene's database (closed), its cache under ``root / cache``."""
    last = LAST[desktop]
    tax = load_taxonomy()
    db = Db(str(root / "tda.sqlite"))
    rows, meta = read_desktop_csv(FIXTURES / f"desktop_{desktop:02d}.csv")
    imported = import_log(desktop, rows, meta, tax)
    db.replace_steps(desktop, [s for s in imported.steps if s.step <= last],
                     [a for a in imported.actions if a.step <= last])
    cooler = next(k for k in imported.instances if k.startswith("cpu_cooler."))
    for key, rec in imported.instances.items():
        if key.startswith("screw.cpu_cooler."):       # captive: rides out with the cooler
            rec.parent, rec.attached, rec.fastens = cooler, True, cooler
        db.upsert_instance(rec)
    db.upsert_desktop(desktop, {"brand": "Dell", "model_family": "Optiplex 7020"})
    raw_paths = _pictures(root, desktop, last)
    for step in range(1, last + 1):
        db.upsert_frame(FrameKey(desktop, step, VIEW), raw_paths[step][0],
                        {"hw": list(HW), "burst": raw_paths[step]}, None)
    db.set_pose_segment(desktop, VIEW, 1, 1, last, last, None, None)
    db.set_pose_segment_roi(desktop, VIEW, 1, ROI, hw=HW)

    insts = instances_of(db, desktop)
    state = state_of(db, tax, desktop, 1)
    keys = sorted(k for k, kind in needs_geom(insts, state, tax).items()
                  if kind == "mask" and state[k].placement == "in_chassis" and k != CHASSIS)
    free = iter(i for i in range(56) if i not in CELLS.values())
    shapes = {CHASSIS: rect(0, 0, 128, 112)}
    for key in keys:
        shapes[key] = cell(CELLS[key] if key in CELLS else next(free))
    shapes[COVER] = shapes[COVER] | shapes[HIDDEN]
    ys, xs = np.nonzero(cell(CELLS[HALF]))
    shapes[HALF_COVER] = shapes[HALF_COVER] | rect(int(xs.min()), int(ys.min()),
                                                   int(xs.min()) + 7, int(ys.max()) + 1)
    ys, xs = np.nonzero(cell(CELLS[STALE_LOWER]))
    shapes[STALE_UPPER] = shapes[STALE_UPPER] | rect(int(xs.max()) + 1 - STRIP, int(ys.min()),
                                                     int(xs.max()) + 3, int(ys.max()) + 1)
    for key, mask in shapes.items():
        db.add_keyframe(ShapeKeyframe(
            id=None, instance=key, desktop=desktop, view=VIEW, pose_segment=1,
            anchor_step=last, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(mask))]))
    top = (COVER, HALF_COVER, STALE_UPPER)
    order = [CHASSIS] + [k for k in keys if k not in top] + list(top)
    db.set_zorder(ZOrderRec(desktop, VIEW, 1, [(k, "main") for k in order]))
    ys, xs = np.nonzero(cell(CELLS[CABLED]))
    db.set_occluder(OccluderMask(FrameKey(desktop, CABLE_STEP, VIEW), "cable", masks.encode_rle(
        rect(int(xs.min()), int(ys.min()), int(xs.min()) + 5, int(ys.max()) + 1))))
    patch = np.roll(cell(CELLS[PATCHED]), 6, axis=1)
    db.set_frame_override(FrameOverride(FrameKey(desktop, PATCH_STEP, VIEW), PATCHED,
                                        visible_rle=masks.encode_rle(patch)))

    truth = TruthService(db, tax, cache_dir=str(root / "cache"))
    steps = list(range(1, last + 1))
    truth.refresh_range(desktop, VIEW, steps)
    confirm = steps[:-1] if desktop == 2 else steps
    for step in confirm:
        truth.verify_frame(FrameKey(desktop, step, VIEW), "tester")
    # STALE_LOWER's confirmed row on frame 4 still holds the strip STALE_UPPER
    # covers today: confirmed before the upper part grew (same inputs digest,
    # as a row inside the re-tracing tolerance keeps)
    key = FrameKey(desktop, STALE_STEP, VIEW)
    row = db.compiled(key)[STALE_LOWER]
    db.put_compiled(key, STALE_LOWER, masks.encode_rle(cell(CELLS[STALE_LOWER])),
                    row["occlusion_ratio"], row["visibility"], row["placement"], "verified",
                    row["input_hash"], verified_by=row["verified_by"])
    db.close()
    return root / "tda.sqlite"


def below_raw(path):
    """The stand-in raw drive's ``tail``: the part of a path below ``.../raw/``."""
    text = str(path).replace("\\", "/")
    return text.split("/raw/", 1)[1] if "/raw/" in text else None


def make_pack(root: Path, db_path: Path, desktop: int, name: str = "pack", **kw) -> dict:
    kw.setdefault("exclude", {(desktop, EXCLUDED[desktop]): EXCLUDE_REASON})
    kw.setdefault("notes", [NOTE])
    kw.setdefault("tail", below_raw)
    kw.setdefault("expected_hw", HW)
    return S.build_pack(str(db_path), str(root / name), [desktop],
                        cache_dir=str(root / "cache"), work_dir=str(root / f"work_{name}"),
                        **kw)


@pytest.fixture(scope="module", params=[2, 13], ids=["D2", "D13"])
def built(request, tmp_path_factory):
    desktop = request.param
    root = tmp_path_factory.mktemp(f"u30_d{desktop}")
    db_path = build_scene(root, desktop)
    before = sha(db_path)
    summary = make_pack(root, db_path, desktop)
    pack = root / "pack"
    ddir = pack / f"D{desktop:02d}"
    coco = json.loads((ddir / "annotations.json").read_text(encoding="utf-8"))
    instances = json.loads((ddir / "instances.json").read_text(encoding="utf-8"))
    frames = json.loads((ddir / "frames.json").read_text(encoding="utf-8"))
    return {"desktop": desktop, "root": root, "db": db_path, "db_sha_before": before,
            "summary": summary, "pack": pack, "ddir": ddir, "coco": coco,
            "instances": instances, "frames": frames,
            "step_of": {i["id"]: i["extra"]["step"] for i in coco["images"]}}


def _label(built, img) -> np.ndarray:
    out = cv2.imread(str(built["ddir"] / img["label_file"]), cv2.IMREAD_UNCHANGED)
    assert out is not None and out.dtype == np.uint16 and out.shape == HW
    return out


def _anns(built, img) -> list[dict]:
    return [a for a in built["coco"]["annotations"] if a["image_id"] == img["id"]]


def _ann(built, step: int, key: str) -> dict:
    img = next(i for i in built["coco"]["images"] if i["extra"]["step"] == step)
    return next(a for a in _anns(built, img) if a["attributes"]["instance_key"] == key)


def _id(built, key: str) -> int:
    return next(v["id"] for v in built["instances"]["instances"].values() if v["key"] == key)


# --------------------------------------------------------------------------- #
# the label map
# --------------------------------------------------------------------------- #
def test_label_map_is_the_union_of_the_visible_masks_and_the_ids_match(built):
    ids = {v["key"]: v["id"] for v in built["instances"]["instances"].values()}
    assert sorted(ids.values()) == list(range(1, len(ids) + 1))
    assert built["coco"]["images"], "nothing exported"
    for img in built["coco"]["images"]:
        label = _label(built, img)
        anns = _anns(built, img)
        claims = np.zeros(HW, dtype=np.uint8)
        owner = np.zeros(HW, dtype=np.int64)
        for ann in anns:
            assert ann["instance_id"] == ids[ann["attributes"]["instance_key"]]
            seen = masks.decode_rle(ann["segmentation"]).astype(bool)
            assert ann["area"] == int(seen.sum())
            claims += seen
            owner[seen] = ann["instance_id"]
            assert ann["label_px"] == int((label == ann["instance_id"]).sum())
            # every pixel the map gives the part is one of its visible pixels
            assert not ((label == ann["instance_id"]) & ~seen).any()
        assert np.array_equal(label > 0, claims > 0)
        alone = claims == 1
        assert np.array_equal(label[alone], owner[alone])
        for ann in anns:
            seen = masks.decode_rle(ann["segmentation"]).astype(bool)
            assert ann["shared_px"] == int((claims[seen] > 1).sum())


def test_two_confirmed_masks_on_one_pixel_go_to_the_frames_own_correction(built):
    patched = _ann(built, PATCH_STEP, PATCHED)
    neighbour = _ann(built, PATCH_STEP, NEIGHBOR)
    assert patched["shared_px"] > 0 and neighbour["shared_px"] > 0
    assert patched["label_px"] == patched["area"]
    assert neighbour["label_px"] < neighbour["area"]
    assert not neighbour["clean"] and not patched["clean"]
    assert patched["amodal_is_shape"] is False          # the patch reaches out of the shape
    # on the other frames the neighbour is a plain clean square
    assert _ann(built, 1, NEIGHBOR)["clean"]


def test_without_a_per_frame_mask_the_layer_order_owns_the_shared_pixels(built):
    """D2's cover over the motherboard: an older confirmed row under a newer upper part."""
    lower = _ann(built, STALE_STEP, STALE_LOWER)
    upper = _ann(built, STALE_STEP, STALE_UPPER)
    shared = STRIP * (SIDE - 2)
    assert lower["shared_px"] == upper["shared_px"] == shared
    assert upper["label_px"] == upper["area"]                       # higher layer wins
    assert lower["label_px"] == lower["area"] - shared
    # the lower row is its whole square again -- equal to its shape, labelled
    # visible -- and still not clean: the map gave the strip away
    assert lower["area"] == lower["amodal_area"] and lower["visibility"] == "visible"
    assert not lower["clean"]
    # the upper part is uncovered and owns every pixel: only shared_px keeps it out
    assert upper["area"] == upper["amodal_area"] and upper["visibility"] == "visible"
    assert upper["label_px"] == upper["area"] and upper["amodal_is_shape"]
    assert not upper["clean"]
    assert _ann(built, 1, STALE_UPPER)["clean"]                     # no overlap there
    assert built["summary"]["checks"]["clean_but_for_shared_px"] >= 1


# --------------------------------------------------------------------------- #
# the annotations
# --------------------------------------------------------------------------- #
def test_visible_lies_inside_amodal_and_the_occluders_add_up(built):
    for ann in built["coco"]["annotations"]:
        seen = masks.decode_rle(ann["segmentation"]).astype(bool)
        whole = masks.decode_rle(ann["amodal_segmentation"]).astype(bool)
        assert not (seen & ~whole).any()
        assert ann["amodal_area"] == int(whole.sum())
        hidden = sum(o["px"] for o in ann["occluders"]) + ann["occluded_unlabelled_px"]
        assert hidden == ann["amodal_area"] - ann["area"]
        assert ann["visibility"] == ann["attributes"]["visibility"]


def test_occlusion_ratio_is_the_masks_and_the_confirmed_value_stays_in_attributes(built):
    differs = 0
    for ann in built["coco"]["annotations"]:
        assert ann["occlusion_ratio"] == pytest.approx(1 - ann["area"] / ann["amodal_area"],
                                                       abs=1e-12)
        differs += ann["occlusion_ratio"] != pytest.approx(
            ann["attributes"]["occlusion_ratio"], abs=1e-9)
    # the per-frame mask reaches outside its shape: today's amodal is wider than
    # the one the row was confirmed against
    patched = _ann(built, PATCH_STEP, PATCHED)
    assert patched["attributes"]["occlusion_ratio"] == 0.0 and patched["occlusion_ratio"] > 0
    assert differs == built["summary"]["checks"]["occlusion_ratio_differs_from_confirmed"]
    # a part 'visible' by the ladder may still be partly covered (< 0.3)
    lower = _ann(built, 1, STALE_LOWER)
    assert lower["visibility"] == "visible" and 0 < lower["occlusion_ratio"] < 0.3


def test_clean_implies_equal_masks_visible_label_inside_the_roi_and_no_shared_pixel(built):
    clean = [a for a in built["coco"]["annotations"] if a["clean"]]
    assert clean
    for ann in clean:
        seen = masks.decode_rle(ann["segmentation"])
        whole = masks.decode_rle(ann["amodal_segmentation"])
        assert np.array_equal(seen, whole)
        assert ann["visibility"] == "visible" and ann["iscrowd"] == 0
        x, y, w, h = ann["amodal_bbox"]
        assert x >= ROI[0] and y >= ROI[1] and x + w <= ROI[2] and y + h <= ROI[3]
        assert ann["occluders"] == [] and ann["occluded_unlabelled_px"] == 0
        assert ann["label_px"] == ann["area"] and ann["amodal_is_shape"]
        assert ann["shared_px"] == 0 and ann["occlusion_ratio"] == 0.0


def test_the_scenes_parts_are_classified_as_drawn(built):
    cover, hidden = _ann(built, 1, COVER), _ann(built, 1, HIDDEN)
    assert hidden["iscrowd"] == 1 and hidden["area"] == 0 and not hidden["clean"]
    assert hidden["visibility"] == "occluded_full"
    assert hidden["occluders"] == [{"id": _id(built, COVER), "px": hidden["amodal_area"]}]
    assert cover["clean"]
    half = _ann(built, 1, HALF)
    assert not half["clean"] and half["occluders"][0]["id"] == _id(built, HALF_COVER)
    outside = _ann(built, 1, OUTSIDE)
    assert outside["area"] == outside["amodal_area"] > 0 and not outside["clean"]
    assert built["summary"]["checks"]["clean_but_for_roi"] >= 1
    cabled = _ann(built, CABLE_STEP, CABLED)
    assert cabled["occluded_unlabelled_px"] > 0 and not cabled["clean"]
    assert _ann(built, 1, CABLED)["clean"]
    chassis = _ann(built, 1, CHASSIS)
    assert chassis["occlusion_ratio"] > 0 and not chassis["clean"]


def test_only_confirmed_and_not_excluded_frames_are_exported(built):
    desktop, last = built["desktop"], LAST[built["desktop"]]
    gone = EXCLUDED[desktop]
    exported = sorted(i["extra"]["step"] for i in built["coco"]["images"])
    confirmed = range(1, last) if desktop == 2 else range(1, last + 1)
    assert exported == [s for s in confirmed if s != gone]
    frames = built["frames"]["frames"]
    if desktop == 2:
        assert frames[str(last)]["exported"] is False and frames[str(last)]["image"] is None
        assert frames[str(last)]["excluded"] is None
    assert all(a["attributes"]["verified"] for a in built["coco"]["annotations"])
    assert built["coco"]["info"]["version"] == "2"


def test_an_excluded_frame_is_out_of_the_files_and_the_counts(built):
    desktop, gone = built["desktop"], EXCLUDED[built["desktop"]]
    ddir = built["ddir"]
    assert not (ddir / "images" / f"S{gone:02d}.png").exists()
    assert not (ddir / "labels" / f"S{gone:02d}.png").exists()
    assert gone not in built["step_of"].values()
    entry = built["frames"]["frames"][str(gone)]
    assert entry["exported"] is False and entry["excluded"] == EXCLUDE_REASON
    assert built["coco"]["info"]["excluded_frames"] == [{"step": gone, "reason": EXCLUDE_REASON}]
    seq = json.loads((ddir / "sequence.json").read_text(encoding="utf-8"))
    row = next(s for s in seq["steps"] if s["step"] == gone)
    assert row["frame"] is None and row["excluded"] == EXCLUDE_REASON
    manifest = json.loads((built["pack"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["exclusions"] == [{"desktop": desktop, "step": gone,
                                       "reason": EXCLUDE_REASON}]
    assert manifest["per_desktop"][0]["frames"] == len(built["coco"]["images"])
    with open(built["pack"] / "clean_parts.csv", encoding="utf-8", newline="") as handle:
        assert all(int(r["step"]) != gone for r in csv.DictReader(handle))
    # never a part's before or after frame: the one before it is
    leaving = [v["leaving"] for v in built["instances"]["instances"].values()
               if v["leaving"] and v["leaving"]["step"] == gone + 1]
    assert leaving
    assert all(lv["before_frame"] == gone - 1 and lv["after_frame"] == gone + 1
               for lv in leaving)
    readme = (built["pack"] / "README.md").read_text(encoding="utf-8")
    assert f"D{desktop:02d} S{gone:02d} is left out" in readme and EXCLUDE_REASON in readme


def test_an_exclusion_must_name_a_packed_frame(built, tmp_path):
    with pytest.raises(S.PackError, match="not packed"):
        make_pack(tmp_path, built["db"], built["desktop"], exclude={(99, 1): "x"})
    with pytest.raises(S.PackError, match="no frame"):
        make_pack(tmp_path, built["db"], built["desktop"], name="p2",
                  exclude={(built["desktop"], 77): "x"})


RULED = "under the closed cage on these frames"


def test_not_clean_rules_parse():
    assert S.parse_not_clean(["D28:connector.06, connector.07@1-10=" + RULED,
                              "d4:screw.motherboard.01@S03,5-6=why"]) == [
        {"desktop": 28, "parts": ["connector.06", "connector.07"],
         "steps": list(range(1, 11)), "reason": RULED},
        {"desktop": 4, "parts": ["screw.motherboard.01"], "steps": [3, 5, 6],
         "reason": "why"}]
    for bad in ("D28:connector.06@1-10", "D28@1-10=x", "D28:connector.06@10-1=x",
                "D28:connector.06@a=x"):
        with pytest.raises(S.PackError):
            S.parse_not_clean([bad])
    assert S._steps_text([1, 2, 3, 7, 9, 10]) == "S01-S03, S07, S09-S10"


def test_a_part_ruled_not_clean_leaves_the_clean_set(built, tmp_path):
    """Final build, decision 1: D28's plugs under the closed cage on S01-S10."""
    assert all(a["clean_note"] is None for a in built["coco"]["annotations"])
    desktop = built["desktop"]
    rule = S.parse_not_clean([f"D{desktop}:{NEIGHBOR}@1-2={RULED}"])
    summary = make_pack(tmp_path, built["db"], desktop, not_clean=rule)
    coco = json.loads((tmp_path / "pack" / f"D{desktop:02d}" / "annotations.json")
                      .read_text(encoding="utf-8"))
    step_of = {i["id"]: i["extra"]["step"] for i in coco["images"]}
    mine = {step_of[a["image_id"]]: a for a in coco["annotations"]
            if a["attributes"]["instance_key"] == NEIGHBOR}
    for step in (1, 2):
        assert mine[step]["clean"] is False and mine[step]["clean_note"] == RULED
    assert mine[4]["clean"] is True and mine[4]["clean_note"] is None
    assert summary["totals"]["clean"] == built["summary"]["totals"]["clean"] - 2
    assert summary["checks"]["clean_but_ruled_not_clean"] == 2
    assert summary["not_clean"] == [{"desktop": desktop, "parts": [NEIGHBOR],
                                     "steps": "S01-S02", "reason": RULED,
                                     "annotations": 2, "were_clean": 2}]
    with open(tmp_path / "pack" / "clean_parts.csv", encoding="utf-8", newline="") as handle:
        assert not [r for r in csv.DictReader(handle)
                    if r["instance_key"] == NEIGHBOR and int(r["step"]) in (1, 2)]
    readme = (tmp_path / "pack" / "README.md").read_text(encoding="utf-8")
    assert f"D{desktop:02d} S01-S02 `{NEIGHBOR}` (2 annotations)" in readme
    assert "`clean_note`" in readme.split("## What this is")[0]


def test_a_not_clean_rule_must_match(built, tmp_path):
    desktop = built["desktop"]
    with pytest.raises(S.PackError, match="matches no annotation"):
        make_pack(tmp_path, built["db"], desktop,
                  not_clean=S.parse_not_clean([f"D{desktop}:no.such.part@1-2=x"]))
    with pytest.raises(S.PackError, match="not packed"):
        make_pack(tmp_path, built["db"], desktop, name="p2",
                  not_clean=S.parse_not_clean(["D99:connector.06@1=x"]))


def test_exclusions_parse():
    assert S.parse_exclusions(["D2:1,D4:1=why not", "d13:s5"]) == {
        (2, 1): "why not", (4, 1): "why not", (13, 5): S.DEFAULT_EXCLUDE_REASON}
    with pytest.raises(S.PackError):
        S.parse_exclusions(["D2-1"])


def test_model_names_are_normalised_and_the_sheet_text_kept():
    sheet = {
        ("Dell", "Optiplex 7020"): "Dell OptiPlex 7020",
        ("Dell", "Optiplex 990"): "Dell OptiPlex 990",
        ("HP", "Compoq 8100 Elite Small Form Fador"): "HP Compaq 8100 Elite SFF",
        ("HP", "Compoq 8300 Elite Small Form Fador"): "HP Compaq 8300 Elite SFF",
        ("HP", "EliteDesk 800 GI SFF"): "HP EliteDesk 800 G1 SFF",
        ("HP", "Compoq 8200 Elite Small Form Fador"): "HP Compaq 8200 Elite SFF",
        ("HP", "ProDesk 400 GI SFF"): "HP ProDesk 400 G1 SFF",
    }
    for (brand, family), model in sheet.items():
        raw = f"{brand} {family}"
        assert S.model_names({"brand": brand, "model_family": family,
                              "brand_model_raw": raw}) == (model, raw, None)
    for family in ("ProDesk 4300 GI SFF", "ProDesk 4300", "Compoq 4300 Elite Small Form Fador"):
        model, raw, note = S.model_names({"brand": "HP", "model_family": family})
        assert model == "HP Compaq Pro 4300 SFF"          # the name only (R2-N4)
        assert raw == f"HP {family}"
        assert "nameplate" in note and f"'{family}'" in note
    assert S.model_names({"brand": "Acme", "model_family": "X1"}) == ("Acme X1", "Acme X1",
                                                                       None)


def test_the_model_note_travels_beside_the_name(built):
    for path in ("frames.json", "instances.json", "sequence.json"):
        head = json.loads((built["ddir"] / path).read_text(encoding="utf-8"))
        assert head["model"] == "Dell OptiPlex 7020" and head["model_note"] is None
    assert built["coco"]["info"]["model_note"] is None


# --------------------------------------------------------------------------- #
# the pictures
# --------------------------------------------------------------------------- #
def test_each_picture_is_the_chosen_source_file_byte_for_byte(built):
    raw = built["root"] / "raw" / f"D{built['desktop']:02d}"
    for step, entry in built["frames"]["frames"].items():
        if not entry["exported"]:
            continue
        step = int(step)
        pick = 1 if step == CHOSEN_P1_STEP else 0
        source = raw / f"RGB{step}1" / f"P_{pick}.png"
        image = built["ddir"] / entry["image"]
        assert sha(image) == sha(source) == entry["sha256"]
        assert entry["file_name"] == f"P_{pick}.png"
        assert entry["source_folder"] == f"D{built['desktop']:02d}/RGB{step}1"
        assert entry["burst_choice"]["p_index"] == pick
        assert (entry["width"], entry["height"]) == (HW[1], HW[0])
        assert entry["roi"] == ROI
        assert entry["frame_record_file"] == ("P_0.png" if pick else None)
        assert entry["identical_to_annotated_copy"] is (step != REENCODED_STEP)
        assert entry["annotated_copy_same_pixels"] is True
        assert entry["review_status"] == "verified"
    s4 = built["frames"]["frames"][str(REENCODED_STEP)]
    assert sha(built["ddir"] / s4["image"]) != sha(
        built["root"] / "cache" / VIEW / f"D{built['desktop']:02d}" / f"s{REENCODED_STEP:03d}.png")
    qa = built["summary"]["qa"]
    assert [r["step"] for r in qa["frame_row_names_other_image"]] == [CHOSEN_P1_STEP]
    assert [r["step"] for r in qa["cache_differs_from_raw"]] == [REENCODED_STEP]
    readme = (built["pack"] / "README.md").read_text(encoding="utf-8")
    assert f"D{built['desktop']:02d} S{CHOSEN_P1_STEP:02d}" in readme


def test_frames_carry_the_step_name_and_its_actions(built):
    frames = built["frames"]["frames"]
    two = frames["2"]
    assert two["step_name"]
    assert [a["target"] for a in two["actions"]] and two["actions"][0]["result"] == "success"
    assert two["actions"][0]["target_id"] == _id(built, two["actions"][0]["target"])


# --------------------------------------------------------------------------- #
# instances and sequence
# --------------------------------------------------------------------------- #
def test_the_captive_screws_leave_with_the_cooler(built):
    insts = {v["key"]: v for v in built["instances"]["instances"].values()}
    cooler = next(k for k in insts if k.startswith("cpu_cooler."))
    r = insts[cooler]["leaving"]["step"]
    assert insts[cooler]["leaving"]["kind"] == "removed"
    assert insts[cooler]["leaving"]["verb"] == "remove"
    screw = insts["screw.cpu_cooler.01"]
    assert screw["captive"] in (True, None) and screw["attached"] is True
    assert screw["leaving"]["step"] == r and screw["leaving"]["left_with"] == cooler
    assert screw["leaving"]["left_with_id"] == insts[cooler]["id"]
    assert screw["leaving"]["verb"] == "remove"
    assert (screw["leaving"]["before_frame"], screw["leaving"]["after_frame"]) == (r - 1, r)
    assert screw["states"][str(r - 1)]["placement"] == "in_chassis"
    assert screw["states"][str(r)]["state"] == "removed"
    seq = json.loads((built["ddir"] / "sequence.json").read_text(encoding="utf-8"))
    runs = seq["timeline"][str(screw["id"])]["runs"]
    assert runs[0]["from"] == 1 and runs[-1]["to"] == LAST[built["desktop"]]
    assert {"step": r, "id": screw["id"], "key": "screw.cpu_cooler.01", "kind": "removed",
            "verb": "remove", "left_with": cooler} in seq["leaving"]
    assert [s["step"] for s in seq["steps"]] == list(range(1, LAST[built["desktop"]] + 1))


def test_instances_carry_class_descriptions_and_no_bookkeeping(built):
    for inst in built["instances"]["instances"].values():
        assert inst["class_description"]
        assert inst["category_id"] is not None
        assert not set(inst["attrs"]) & set(S.DROP_ATTRS)


# --------------------------------------------------------------------------- #
# the package
# --------------------------------------------------------------------------- #
def test_manifest_counts_and_hashes(built):
    pack = built["pack"]
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["package"] == "scanpack"
    on_disk = sorted(p.relative_to(pack).as_posix() for p in pack.rglob("*")
                     if p.is_file() and p.name != "manifest.json")
    assert sorted(manifest["files"]) == on_disk
    for rel, rec in manifest["files"].items():
        assert sha(pack / rel) == rec["sha256"] and (pack / rel).stat().st_size == rec["bytes"]
    (desk,) = manifest["per_desktop"]
    anns = built["coco"]["annotations"]
    assert desk["annotations"] == len(anns)
    assert desk["frames"] == len(built["coco"]["images"])
    assert desk["crowd"] == sum(a["iscrowd"] for a in anns)
    assert desk["clean"] == sum(a["clean"] for a in anns) == manifest["totals"]["clean"]
    assert desk["instances"] == len(built["instances"]["instances"])
    check = desk["confirmed_vs_today"]
    assert check["rows_compared"] == len(anns)
    assert check["pixel_identical"] + check["differ_within_tolerance"] == len(anns)
    assert check["differ_within_tolerance"] >= 1     # STALE_LOWER's row on frame 4
    assert (desk["pose_segments"], desk["rois"]) == (1, 1)
    assert manifest["frame_size"] == {"width": HW[1], "height": HW[0]}
    assert manifest["tier"] == "gold"
    assert manifest["notes"] == [NOTE]
    with open(pack / "clean_parts.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == desk["clean"]
    assert sum(manifest["clean_by_class"].values()) == desk["clean"]
    by_key = {(built["step_of"][a["image_id"]], a["instance_id"]): a for a in anns}
    for row in rows:
        ann = by_key[(int(row["step"]), int(row["instance_id"]))]
        assert ann["clean"] and int(row["area"]) == ann["area"]
        assert row["class"] == built["instances"]["instances"][row["instance_id"]]["class"]


INTERNAL_WORDS = ("spec 8", "spec §", "rehearsal", "TDA ", "cache_file_name",
                  "source_recorded_path", "added_by", "truth_check",
                  "identical_to_fresh_compile", "within_app_tolerance", "max_px_diff")


def test_the_shipped_files_carry_no_internal_paths_or_jargon(built):
    root = str(built["root"]).replace("\\", "/")
    for path in built["pack"].rglob("*"):
        if path.suffix not in (".json", ".md", ".csv", ".txt"):
            continue
        text = path.read_text(encoding="utf-8")
        for word in INTERNAL_WORDS:
            assert word not in text, (path.name, word)
        assert root not in text and str(built["root"]) not in text, path.name
        if path.name != "annotations.json":  # RLE counts are arbitrary characters
            assert not re.search(r"\bU\d\d\b", text), path.name
            # no drive letter anywhere, frames.json's source folders included (R2-N6)
            assert not re.search(r"\b[A-Za-z]:[/\\]", text), path.name
    for entry in built["frames"]["frames"].values():
        assert not re.match(r"^([A-Za-z]:|/)", entry["source_folder"]), entry["source_folder"]


def test_a_source_folder_outside_the_dataset_root_is_refused(built, tmp_path):
    with pytest.raises(S.PackError, match="not below the raw dataset's root"):
        make_pack(tmp_path, built["db"], built["desktop"], tail=lambda p: None)


def test_the_readme_states_the_rules(built):
    text = (built["pack"] / "README.md").read_text(encoding="utf-8")
    summary = text.split("## What this is")[0]
    assert "shared_px == 0" in summary and "之后" in summary and "S01" in summary
    assert "label map" in summary and f"{HW[1]}×{HW[0]}" in summary
    assert "6. no other confirmed mask claims any of its pixels (`shared_px == 0`)" in text
    assert "## Known issues" in text and NOTE in text
    assert "## Confirmed and derived fields" in text
    assert "Initial conditions" in text
    assert f"{HW[1]} x {HW[0]} for every frame here" in text
    assert "`gold` on every annotation here" in text
    assert "Each desktop here was scanned from one camera pose and has one ROI" in text
    assert "none is of a part set down on the bench" in text
    assert "same_pose_segment" in text and "confirmed_vs_today" in text
    # R2-N1: no reason is claimed for the classes with few clean rows
    assert "Large parts" not in text and "大件" not in text


def test_the_visibility_table_lists_the_rules_in_the_order_they_are_tried(built):
    text = (built["pack"] / "README.md").read_text(encoding="utf-8")
    table = text.split("| Value | Rule (first match wins) |")[1].split("\n\n")[0]
    values = re.findall(r"^\| `(\w+)` \|", table, flags=re.M)
    assert values == ["out_of_view", "occluded_full", "too_small", "visible_tiny",
                      "occluded_partial", "visible"]
    assert "| `out_of_view` | no geometry on the frame |" in table
    assert ">= 0.95" in table and "< 6 px" in table and "< 0.3" in table
    assert "6 to 11 px" in table


def test_the_input_database_is_not_written(built):
    assert sha(built["db"]) == built["db_sha_before"]


def test_the_output_is_deterministic(built):
    again = make_pack(built["root"], built["db"], built["desktop"], name="again")
    one, two = built["pack"], built["root"] / "again"
    files = sorted(p.relative_to(one).as_posix() for p in one.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(two).as_posix() for p in two.rglob("*") if p.is_file())
    for rel in files:
        assert (one / rel).read_bytes() == (two / rel).read_bytes(), rel
    assert again["totals"] == built["summary"]["totals"]


def test_the_readme_example_runs(built, monkeypatch, capsys):
    text = (built["pack"] / "README.md").read_text(encoding="utf-8")
    code = text.split("```python\n", 1)[1].split("```", 1)[0]
    assert len(code.strip().splitlines()) <= 22
    # as shipped: it names a desktop the package has (the final build had no D02)
    assert f'desk = "D{built["desktop"]:02d}"' in code
    monkeypatch.chdir(built["pack"])
    exec(compile(code, "README example", "exec"), {})
    out = capsys.readouterr().out
    assert "label map == visible mask: True" in out
    assert "hidden pixels (amodal minus visible): 0" in out


def test_a_full_package_is_not_overwritten_without_force(built):
    with pytest.raises(S.PackError, match="--force"):
        make_pack(built["root"], built["db"], built["desktop"])


# --------------------------------------------------------------------------- #
# the confirmed rows must still be what the inputs make
# --------------------------------------------------------------------------- #
def test_confirmed_rows_the_inputs_no_longer_make_are_refused(tmp_path):
    db_path = build_scene(tmp_path, 13)
    db = Db(str(db_path))
    kf = next(k for k in db.keyframes(13, VIEW, COVER))
    kf.parts = [ShapePart("main", masks.encode_rle(cell(CELLS[COVER])))]   # HIDDEN uncovered
    db.update_keyframe(kf)
    db.close()
    with pytest.raises(S.PackError, match="disagree"):
        make_pack(tmp_path, db_path, 13)


# --------------------------------------------------------------------------- #
# round 3: the README's fixed claims are checked at build time (R2-N3)
# --------------------------------------------------------------------------- #
def test_a_frame_of_another_size_stops_the_build(built, tmp_path):
    with pytest.raises(S.PackError, match="the README says 1600x1600"):
        make_pack(tmp_path, built["db"], built["desktop"], expected_hw=(1600, 1600))


def test_another_tier_stops_the_build(built, tmp_path):
    with pytest.raises(S.PackError, match="tier 'gold', the README says 'silver'"):
        make_pack(tmp_path, built["db"], built["desktop"], expected_tier="silver")


def test_a_picture_chosen_for_another_reason_than_a_dark_p0_stops_the_build(tmp_path):
    db_path = build_scene(tmp_path, 13)
    manifest = tmp_path / "cache" / VIEW / "D13" / "manifest.json"
    records = json.loads(manifest.read_text(encoding="utf-8"))
    records[str(CHOSEN_P1_STEP)]["reason"] = "p0_saturated"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(S.PackError, match="the README explains only a dark P_0"):
        make_pack(tmp_path, db_path, 13)


# --------------------------------------------------------------------------- #
# round 3: the README's data-driven sentences, every branch (R2-N1, R2-N2)
# --------------------------------------------------------------------------- #
def _summary(**over) -> dict:
    base = {
        "annotations_by_class": {"chassis": 40, "screw": 100, "ram_module": 30},
        "clean_by_class": {"screw": 90, "ram_module": 2},
        "placement_counts": {"in_chassis": 170},
        "per_desktop": [{"desktop": 4, "pose_segments": 1, "rois": 1}],
        "checks": {"amodal_box_outside_roi": 12, "clean_but_for_roi": 5},
        "amodal_box_outside_roi_by_class": {"chassis": 10, "psu": 2},
    }
    base.update(over)
    return base


def test_a_model_note_is_a_paragraph_below_the_table_not_a_row_of_it():
    desk = {"desktop": 30, "model": "HP Compaq Pro 4300 SFF", "model_note": "check it",
            "frames": 1, "instances_annotated": 1, "annotations": 1, "crowd": 0, "clean": 1,
            "clean_instances": 1, "pose_segments": 1, "rois": 1}
    summary = _summary(
        pack_version="1", desktops=[30], per_desktop=[desk],
        totals={"frames": 1, "annotations": 1, "crowd": 0, "clean": 1, "overlap_frames": 0,
                "overlap_px": 0, "amodal_widened_rows": 0},
        checks={"amodal_box_outside_roi": 0, "clean_but_for_roi": 0,
                "occlusion_ratio_differs_from_confirmed": 0, "occluded_full_with_pixels": 0},
        visibility_counts={"visible": 1}, burst_sizes={"10": 1}, burst_reasons={"p0": 1},
        frame_size={"width": 1600, "height": 1600}, tier="gold", state_values=["installed"],
        placement_values=["in_chassis"], frame_record_names_other_image=[], exclusions=[],
        notes=[], amodal_box_outside_roi_by_class={})
    text = S.readme_text(summary)
    assert "| D30 | HP Compaq Pro 4300 SFF * |" in text
    total_row = next(line for line in text.splitlines() if line.startswith("| **Total**"))
    after = text.split(total_row, 1)[1]
    assert after.startswith("\n\n\\* D30: check it.\n\n*Parts* counts")
    # the final build's case: nothing overlaps, nothing is left out (no "0 frames")
    assert "No two confirmed masks overlap in this package" in text
    assert "本包里确认的可见掩码互不重叠" in text and "在 0 帧上" not in text
    assert "on 0 frames" not in text and "On 0 frames" not in text
    assert "No two confirmed visible masks claim the same pixel in this package." in text
    assert "No frame is left out of this package." in text and "none (see" not in text
    assert 'desk = "D30"' in text
    desk["model_note"] = None
    plain = S.readme_text(summary)
    total_row = next(line for line in plain.splitlines() if line.startswith("| **Total**"))
    assert plain.split(total_row, 1)[1].startswith("\n\n*Parts* counts")


def test_the_overlap_cause_follows_the_counts(built):
    """Final build (e): after the re-sign only per-frame masks may overlap; say which."""
    t = built["summary"]["totals"]
    # the scene has both: PATCHED's per-frame mask and STALE_LOWER's old row
    assert 0 < t["overlap_px_without_per_frame_mask"] < t["overlap_px"]
    assert t["overlap_frames_without_per_frame_mask"] == 1
    text = (built["pack"] / "README.md").read_text(encoding="utf-8")
    assert S.overlap_cause(built["summary"]) in text

    def cause(total, plain, frames=1):
        return S.overlap_cause({"totals": {
            "overlap_px": total, "overlap_px_without_per_frame_mask": plain,
            "overlap_frames_without_per_frame_mask": frames}})

    only_corrected = cause(50, 0)
    assert "every one of them is a mask corrected on that frame only" in only_corrected
    assert "do not overlap anywhere" in only_corrected
    assert cause(50, 50).startswith("they are rows confirmed at different times")
    assert cause(50, 20, 2).startswith("30 of them are a mask corrected")
    assert "20 (on 2 frames) are rows confirmed" in cause(50, 20, 2)
    assert S.overlap_intro({"totals": {"overlap_px": 0, "overlap_frames": 0}}) == \
        "No two confirmed visible masks claim the same pixel in this package."


def test_the_few_clean_classes_are_named_without_a_reason():
    en, zh = S.few_clean_text(_summary())
    assert en.startswith("Classes with few or no clean rows (under 15 % of their annotations): "
                         "chassis (0 of 40), ram_module (2 of 30).")
    assert "Large" not in en and "sits on" not in en and "大件" not in zh
    assert "chassis（0/40）" in zh and "ram_module（2/30）" in zh
    en, zh = S.few_clean_text(_summary(clean_by_class={"chassis": 40, "screw": 90,
                                                       "ram_module": 30}))
    assert en == "Every class has at least 15 % of its annotations clean." and "15%" in zh


def test_the_roi_paragraph_reads_right_in_every_branch():
    plain = S.roi_paragraph(_summary())
    assert "Each desktop here was scanned from one camera pose and has one ROI" in plain
    assert "none is of a part set down on the bench" in plain
    assert ("on 12 annotations (chassis 10, psu 2) the amodal box reaches past it, so by "
            "rule 3 they are not clean; for 5 of them it is the only reason.") in plain

    bench = S.roi_paragraph(_summary(placement_counts={"in_chassis": 160, "on_bench": 5}))
    assert "5 annotations are of parts set down on the bench" in bench
    assert "none is" not in bench and "Every annotation in this package is of a part in " \
        "the chassis" not in bench

    inside = S.roi_paragraph(_summary(checks={"amodal_box_outside_roi": 0,
                                              "clean_but_for_roi": 0},
                                      amodal_box_outside_roi_by_class={}))
    assert "Every amodal box lies inside its frame's ROI" in inside
    assert "0 annotations" not in inside and "(none)" not in inside

    other_rule = S.roi_paragraph(_summary(checks={"amodal_box_outside_roi": 1,
                                                  "clean_but_for_roi": 0},
                                          amodal_box_outside_roi_by_class={"psu": 1}))
    assert "on 1 annotation (psu 1) the amodal box reaches past it, so by rule 3 it is " \
        "not clean; each of them also fails another rule." in other_rule

    poses = S.roi_paragraph(_summary(per_desktop=[
        {"desktop": 4, "pose_segments": 2, "rois": 2},
        {"desktop": 13, "pose_segments": 1, "rois": 1}]))
    assert "D04 (2 camera poses, 2 ROIs) has more than one" in poses
    assert "one camera pose and has one ROI" not in poses


# --------------------------------------------------------------------------- #
# round 3: a missing neighbour frame is not a pose change (R2-N5)
# --------------------------------------------------------------------------- #
def test_same_pose_segment_is_null_when_a_frame_is_missing():
    from tda.ui.locator_pair import Leaving

    leaving = {"first": Leaving(2, "changed", "closed", "open"),
               "last": Leaving(9, "removed", "installed", "removed"),
               "middle": Leaving(5, "removed", "installed", "removed")}
    out = S._leaving([], leaving, [3, 4, 5, 6], seg_of=lambda step: 1 if step < 5 else 2)
    assert (out["first"]["before_frame"], out["first"]["same_pose_segment"]) == (None, None)
    assert (out["last"]["after_frame"], out["last"]["same_pose_segment"]) == (None, None)
    assert out["middle"]["same_pose_segment"] is False      # 4 and 5: two poses
