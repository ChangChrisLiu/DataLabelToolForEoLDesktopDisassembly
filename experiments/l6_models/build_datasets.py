"""Tile L6's verified scan frames and assemble the four RF-DETR training sets.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.build_datasets

1. **Verified tiles** -- every frame of ``gt/gt_det.json`` (the L6 export),
   cropped to its ROI (the stored pose-segment ROI; D27's proposed one), cut
   by L2's ``tiles.py`` (work scale 1.0, 640 tiles, stride 512, grey 114),
   labelled by L2's ``tile_labels`` (a box is kept in a tile when >= 50 % of
   it lies inside), JPEG 95 -- L4's ``build_datasets.tile_verified`` with L6's
   names (``L6Dnn_scan_sNNN_oy_ox``).
2. **Training sets** (``decision_rule.txt``), RF-DETR's Roboflow-YOLO layout
   under ``dataset_rfdetr/<set>/`` (images hard-linked, labels copied byte for
   byte, ``valid`` = 150 random training tiles, seed 0, selects nothing):

   * ``v1h`` -- L2's 4 873 LS tiles minus D19's (v1 without D19);
   * ``v2h`` -- L4's ``all`` fold minus D19's LS tiles (v2 without D19): L2's
     LS tiles minus D13 scan minus D19, + L4's 920 verified tiles;
   * ``F``   -- L2's LS tiles minus the LS scan tiles of every desktop with
     verified scan frames (D8, D10, D13, D19) + the L6 verified tiles of the 18
     training desktops (not D19, D27);
   * ``v3``  -- the same LS tiles + the L6 verified tiles of all 20 desktops.

``dataset/manifest.json``: per-desktop / per-class tile-label counts, per set
the tile counts and L3's tiles hash (sorted by name, (name, sha256(image),
sha256(label))).
"""
from __future__ import annotations

from experiments.l6_models import env  # noqa: F401  (side effects first)

import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import cv2

from experiments.l2_detector import data as D
from experiments.l2_detector import tiles as T
from experiments.l2_detector.build_dataset import tile_labels
from experiments.l3_detector.build_datasets import label_of, tiles_hash
from experiments.l4_detector import gt as G4
from experiments.l6_models.export_gt import GT_DET, HELD_OUT

VDS = env.ROOT / "dataset" / "verified"
RF = env.ROOT / "dataset_rfdetr"
L2DS = Path("D:/DataSet/experiments_out/l2_detector/dataset")
L4DS = Path("D:/DataSet/experiments_out/l4_detector/dataset")
MANIFEST = env.ROOT / "dataset" / "manifest.json"


def tile_verified(frames) -> tuple[list[dict], Counter]:
    rows, lab = [], Counter()
    t0 = time.perf_counter()
    for f in frames:
        img = D.read_rgb(f.image)
        if img is None:
            raise SystemExit(f"unreadable {f.image}")
        crop = T.work_crop(img, f.roi, "scan")
        for tile in T.make_tiles(crop):
            labels = tile_labels(f.boxes, f.roi, "scan", tile)
            name = f"L6D{f.desktop:02d}_scan_s{f.step:03d}_{tile.oy:04d}_{tile.ox:04d}"
            ip = VDS / "images" / "scan" / f"{name}.jpg"
            lp = VDS / "labels" / "scan" / f"{name}.txt"
            ip.parent.mkdir(parents=True, exist_ok=True)
            lp.parent.mkdir(parents=True, exist_ok=True)
            if not ip.exists():
                cv2.imwrite(str(ip), cv2.cvtColor(tile.rgb, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            lp.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                                  for c, x, y, w, h in labels), encoding="utf-8")
            for c, *_ in labels:
                lab[(f.desktop, D.DET_CLASSES[c])] += 1
            rows.append({"image": ip.as_posix(), "desktop": f.desktop, "step": f.step,
                         "ox": tile.ox, "oy": tile.oy, "n": len(labels), "roi": list(f.roi)})
    print(f"[l6] {len(rows)} verified tiles from {len(frames)} frames in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    return rows, lab


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def rf_set(root: Path, train: list[Path], val: list[Path]) -> None:
    names = [p.name for p in train]
    if len(set(names)) != len(names):
        raise SystemExit(f"{root.name}: duplicate tile names")
    for split, items in (("train", train), ("valid", val)):
        for src in items:
            link(src, root / split / "images" / src.name)
            lab = root / split / "labels" / f"{src.stem}.txt"
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_bytes(label_of(src).read_bytes())
    cls = "".join(f"  {i}: {c}\n" for i, c in enumerate(D.DET_CLASSES))
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: train/images\nval: valid/images\n"
        f"nc: {len(D.DET_CLASSES)}\nnames:\n{cls}", encoding="utf-8")


def main() -> int:
    frames = G4.load(GT_DET)
    if list(D.DET_CLASSES) != list(G4.DET_CLASSES):
        raise SystemExit("class lists disagree")
    vrows, vlab = tile_verified(frames)
    desktops = sorted({f.desktop for f in frames})
    vimg = {d: [Path(r["image"]) for r in vrows if r["desktop"] == d] for d in desktops}

    ls = json.loads((L2DS / "manifest.json").read_text(encoding="utf-8"))
    if len(ls) != 4873:
        raise SystemExit(f"L2 manifest has {len(ls)} tiles, expected 4 873")
    replaced = {(d, "scan") for d in desktops}             # verified scan frames replace drafts
    ls_v1h = [Path(m["image"]) for m in ls if m["desktop"] not in HELD_OUT]
    ls_l6 = [Path(m["image"]) for m in ls
             if m["desktop"] not in HELD_OUT and (m["desktop"], m["view"]) not in replaced]
    dropped = Counter(f"D{m['desktop']}:{m['view']}" for m in ls
                      if not (m["desktop"] not in HELD_OUT
                              and (m["desktop"], m["view"]) not in replaced))
    l4_all = sorted((L4DS.parent / "dataset_rfdetr" / "all" / "train" / "images").glob("*.jpg"))
    if len(l4_all) != 5547:
        raise SystemExit(f"L4 all fold has {len(l4_all)} tiles, expected 5 547")
    ls_by_name = {Path(m["image"]).name: m for m in ls}
    l4_ver = {p.name for p in (L4DS / "verified" / "images" / "scan").glob("*.jpg")}
    v2h = []
    for p in l4_all:          # resolve the hard links to their sources (same bytes, own labels)
        if p.name in l4_ver:
            v2h.append(L4DS / "verified" / "images" / "scan" / p.name)
        else:
            m = ls_by_name[p.name]
            if m["desktop"] in HELD_OUT:
                continue
            v2h.append(Path(m["image"]))
    train_d = [d for d in desktops if d not in HELD_OUT]
    sets = {
        "v1h": (ls_v1h, "COCO rf-detr-small.pth",
                "L2's 4 873 LS tiles minus D19 (v1's data without D19)"),
        "v2h": (v2h, "v1h", "L4's all fold (L2 LS minus D13 scan, + L4's 920 verified tiles) "
                            "minus D19's LS tiles"),
        "F": (ls_l6 + [p for d in train_d for p in vimg[d]], "v2h",
              "L2 LS minus D19 and minus the LS scan tiles of D8/D10/D13/D19, + the L6 "
              "verified tiles of the 18 training desktops"),
        "v3": (ls_l6 + [p for d in desktops for p in vimg[d]], "shipped v2",
               "L2 LS minus the LS scan tiles of D8/D10/D13/D19, + the L6 verified tiles "
               "of all 20 desktops"),
    }
    out = {}
    for name, (train, init, rule) in sets.items():
        rng = random.Random(0)
        val = rng.sample(train, 150)
        rf_set(RF / name, train, val)
        n_ver = sum(1 for p in train if p.name.startswith(("L4D", "L6D")))
        out[name] = {"init": init, "rule": rule, "tiles": len(train), "verified_tiles": n_ver,
                     "ls_tiles": len(train) - n_ver, "sha256": tiles_hash(train),
                     "held_out_in_train": sorted({p.name[:6] for p in train
                                                  if p.name.startswith(("D19", "D27", "L6D19",
                                                                        "L6D27", "L4D19"))})}
        print(f"  {name}: {out[name]['ls_tiles']} LS + {n_ver} verified = {len(train)} tiles; "
              f"{out[name]['held_out_in_train'] or 'no held-out tile'}", flush=True)
    for name in ("v1h", "v2h", "F"):
        if out[name]["held_out_in_train"]:
            raise SystemExit(f"{name} holds tiles of a held-out desktop")
    box_cnt = Counter((f.desktop, b.cls) for f in frames for b in f.boxes)
    manifest = {
        "gt": str(GT_DET),
        "frames": {f"D{d}": sum(1 for f in frames if f.desktop == d) for d in desktops},
        "tiles": {f"D{d}": len(vimg[d]) for d in desktops},
        "tiles_without_labels": sum(1 for r in vrows if r["n"] == 0),
        "frame_boxes": {c: {f"D{d}": box_cnt[(d, c)] for d in desktops} for c in D.DET_CLASSES},
        "tile_labels": {c: {f"D{d}": vlab[(d, c)] for d in desktops} for c in D.DET_CLASSES},
        "verified_sha256": tiles_hash([Path(r["image"]) for r in vrows]),
        "ls_dropped": dict(dropped),
        "tiles_hash_rule": "sha256 over tiles sorted by name of (name, sha256(image), sha256(label))",
        "tiling": {"roi": "pose_segment.roi_json (stored); D27: tda.core.cache.suggest_roi_over",
                   "work_scale": 1.0, "tile": T.TILE, "stride": T.STRIDE, "pad": T.PAD_VALUE},
        "sets": out,
        "rows": vrows,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("rows", "frame_boxes",
                                                                      "tile_labels")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
