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

and every frame confirmed (D2: all but the last). Pinned: the label map is the
union of the visible masks with the right ids, visible ⊆ amodal, ``clean``
implies equal masks, the manifest's counts and hashes, each picture is the
chosen source file byte for byte, the output is deterministic, the README's
example runs, and the build refuses confirmed rows the inputs no longer agree
with.
"""
from __future__ import annotations

import csv
import hashlib
import json
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
CHASSIS = "chassis"
HIDDEN, COVER = "screw.motherboard.01", "ram_module.01"
HALF, HALF_COVER = "screw.motherboard.02", "ram_module.02"
OUTSIDE = "screw.motherboard.03"
CABLED, CABLE_STEP = "screw.motherboard.04", 2
PATCHED, NEIGHBOR, PATCH_STEP = "ram_latch.03", "ram_latch.04", 3
CELLS = {COVER: 32, HIDDEN: 33, HALF_COVER: 34, HALF: 35, PATCHED: 40, NEIGHBOR: 41,
         CABLED: 42, OUTSIDE: 60}
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
def _pictures(root: Path, desktop: int, last: int) -> tuple[dict, dict]:
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
    return raw_paths, manifest


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
    db.upsert_desktop(desktop, {"brand": "Dell", "model_family": f"fixture D{desktop}"})
    raw_paths, _ = _pictures(root, desktop, last)
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
    left = rect(int(xs.min()), int(ys.min()), int(xs.min()) + 7, int(ys.max()) + 1)
    shapes[HALF_COVER] = shapes[HALF_COVER] | left
    for key, mask in shapes.items():
        db.add_keyframe(ShapeKeyframe(
            id=None, instance=key, desktop=desktop, view=VIEW, pose_segment=1,
            anchor_step=last, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(mask))]))
    order = [CHASSIS] + [k for k in keys if k not in (COVER, HALF_COVER)] + [COVER, HALF_COVER]
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
    db.close()
    return root / "tda.sqlite"


def make_pack(root: Path, db_path: Path, desktop: int, name: str = "pack", **kw) -> dict:
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
            "instances": instances, "frames": frames}


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
        assert ann["occlusion_ratio"] == ann["attributes"]["occlusion_ratio"]


def test_clean_implies_equal_masks_visible_label_and_inside_the_roi(built):
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
    cabled = _ann(built, CABLE_STEP, CABLED)
    assert cabled["occluded_unlabelled_px"] > 0 and not cabled["clean"]
    assert _ann(built, 1, CABLED)["clean"]
    chassis = _ann(built, 1, CHASSIS)
    assert chassis["occlusion_ratio"] > 0 and not chassis["clean"]


def test_only_confirmed_frames_are_exported(built):
    last = LAST[built["desktop"]]
    exported = sorted(i["extra"]["step"] for i in built["coco"]["images"])
    frames = built["frames"]["frames"]
    if built["desktop"] == 2:
        assert exported == list(range(1, last))
        assert frames[str(last)]["exported"] is False and frames[str(last)]["image"] is None
        assert not (built["ddir"] / "images" / f"S{last:02d}.png").exists()
    else:
        assert exported == list(range(1, last + 1))
    assert all(a["attributes"]["verified"] for a in built["coco"]["annotations"])
    assert built["coco"]["info"]["version"] == "2"


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
        assert (entry["width"], entry["height"]) == (HW[1], HW[0])
        assert entry["roi"] == ROI
        assert entry["frame_row_path"] == (
            (raw / f"RGB{step}1" / "P_0.png").as_posix() if pick else None)
        assert entry["cache_identical"] is (step != REENCODED_STEP)
        assert entry["cache_pixels_identical"] is True
        assert entry["review_status"] == "verified"
    s4 = built["frames"]["frames"][str(REENCODED_STEP)]
    assert sha(built["ddir"] / s4["image"]) != sha(
        built["root"] / "cache" / VIEW / f"D{built['desktop']:02d}" / f"s{REENCODED_STEP:03d}.png")
    qa = built["summary"]["qa"]
    assert [r["step"] for r in qa["frame_row_names_other_image"]] == [CHOSEN_P1_STEP]
    assert [r["step"] for r in qa["cache_differs_from_raw"]] == [REENCODED_STEP]


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


def test_instances_carry_class_descriptions(built):
    for inst in built["instances"]["instances"].values():
        assert inst["class_description"]
        assert inst["category_id"] is not None


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
    assert desk["truth_check"]["rows"] == len(anns)
    with open(pack / "clean_parts.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == desk["clean"]
    assert sum(manifest["clean_by_class"].values()) == desk["clean"]
    by_key = {(a["image_id"] % 100_000 // 10, a["instance_id"]): a for a in anns}
    for row in rows:
        ann = by_key[(int(row["step"]), int(row["instance_id"]))]
        assert ann["clean"] and int(row["area"]) == ann["area"]
        assert row["class"] == built["instances"]["instances"][row["instance_id"]]["class"]


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
    code = code.replace('desk = "D02"', f'desk = "D{built["desktop"]:02d}"')
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
