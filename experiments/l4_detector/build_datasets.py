"""Tile the verified scan frames and assemble L4's RF-DETR folds.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.build_datasets

1. **Verified tiles** -- every verified scan frame of D2/3/4/5/13 (``gt.py``),
   cropped to the pose segment's stored ROI (what the app crops to), cut by
   L2's ``tiles.py`` (work scale 1.0, 640 tiles, stride 512, grey 114 pad),
   labelled by L2's ``build_dataset.tile_labels`` (a box is kept in a tile when
   >= 50 % of it lies inside).  JPEG quality 95, as L2.  Written once to
   ``dataset/verified/{images,labels}/scan/L4Dnn_scan_sNNN_oy_ox``.
2. **Folds** (``decision_rule.txt``), RF-DETR's Roboflow-YOLO layout under
   ``dataset_rfdetr/<fold>/``: every image a hard link (same volume, no copy),
   every label file copied byte for byte; ``valid`` = 150 random *training*
   tiles (seed 0; RF-DETR's per-epoch eval needs one; nothing is selected).

   * ``hold13`` -- L2's LS tiles not of D13 + verified D2/3/4/5;
   * ``hold5``  -- L2's LS tiles minus L2's D13 scan tiles + verified D2/3/4/13;
   * ``all``    -- L2's LS tiles minus L2's D13 scan tiles + all verified.

``manifest.json``: per-desktop / per-class tile-label counts and frame counts of
the verified set; per fold the tile counts and the SHA-256 tiles hash (L3's rule:
sorted by name, (name, sha256(image), sha256(label))).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l4_detector import env  # noqa: E402
from experiments.l4_detector import gt as G  # noqa: E402
from experiments.l2_detector import data as D  # noqa: E402
from experiments.l2_detector import tiles as T  # noqa: E402
from experiments.l2_detector.build_dataset import tile_labels  # noqa: E402
from experiments.l3_detector.build_datasets import label_of, tiles_hash  # noqa: E402

VDS = env.OUT / "dataset" / "verified"
RF = env.OUT / "dataset_rfdetr"
L2DS = env.L2OUT / "dataset"
FOLDS = {"hold13": 13, "hold5": 5, "all": None}


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
            name = f"L4D{f.desktop:02d}_scan_s{f.step:03d}_{tile.oy:04d}_{tile.ox:04d}"
            ip = VDS / "images" / "scan" / f"{name}.jpg"
            lp = VDS / "labels" / "scan" / f"{name}.txt"
            ip.parent.mkdir(parents=True, exist_ok=True)
            lp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ip), cv2.cvtColor(tile.rgb, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            lp.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                                  for c, x, y, w, h in labels), encoding="utf-8")
            for c, *_ in labels:
                lab[(f.desktop, D.DET_CLASSES[c])] += 1
            rows.append({"image": ip.as_posix(), "desktop": f.desktop, "view": "scan",
                         "step": f.step, "ox": tile.ox, "oy": tile.oy, "n": len(labels),
                         "roi": list(f.roi)})
    print(f"[l4] {len(rows)} verified tiles from {len(frames)} frames in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    return rows, lab


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def rf_fold(root: Path, train: list[Path], val: list[Path]) -> None:
    for split, items in (("train", train), ("valid", val)):
        for src in items:
            link(src, root / split / "images" / src.name)
            lab = root / split / "labels" / f"{src.stem}.txt"
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_bytes(label_of(src).read_bytes())
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(D.DET_CLASSES))
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: train/images\nval: valid/images\n"
        f"nc: {len(D.DET_CLASSES)}\nnames:\n{names}", encoding="utf-8")


def main() -> int:
    frames = G.load()
    if list(D.DET_CLASSES) != list(G.DET_CLASSES):
        raise SystemExit("class lists disagree")
    vrows, vlab = tile_verified(frames)
    ls = json.loads((L2DS / "manifest.json").read_text(encoding="utf-8"))
    hold13_ls = [Path(p) for p in (L2DS / "hold13_train.txt").read_text(encoding="utf-8").split()]
    ls_no_d13scan = [Path(m["image"]) for m in ls
                     if not (m["desktop"] == 13 and m["view"] == "scan")]
    if len(hold13_ls) != 4015 or len(ls_no_d13scan) != 4873 - 246:
        raise SystemExit(f"unexpected L2 tile counts {len(hold13_ls)} / {len(ls_no_d13scan)}")
    vimg = {d: [Path(r["image"]) for r in vrows if r["desktop"] == d] for d in G.DESKTOPS}
    folds = {}
    for fold, held in FOLDS.items():
        base = hold13_ls if held == 13 else ls_no_d13scan
        ver = [p for d in G.DESKTOPS if d != held for p in vimg[d]]
        train = base + ver
        rng = random.Random(0)
        val = rng.sample(train, 150)
        rf_fold(RF / fold, train, val)
        folds[fold] = {"held_out": held, "ls_tiles": len(base), "verified_tiles": len(ver),
                       "tiles": len(train), "sha256": tiles_hash(train),
                       "verified_sha256": tiles_hash(ver),
                       "verified_desktops": [d for d in G.DESKTOPS if d != held],
                       "ls_rule": ("L2 hold13_train.txt (no D13)" if held == 13 else
                                   "L2 manifest minus D13 scan (verified labels replace them)")}
        print(f"  {fold}: {len(base)} LS + {len(ver)} verified = {len(train)} tiles", flush=True)
    frame_cnt = Counter(f.desktop for f in frames)
    box_cnt = Counter((f.desktop, b.cls) for f in frames for b in f.boxes)
    other = sorted({c for (_d, c) in box_cnt if c not in G.DET_CLASSES})
    manifest = {
        "source_db": env.DB,
        "frames": {f"D{d}": frame_cnt[d] for d in G.DESKTOPS},
        "tiles": {f"D{d}": len(vimg[d]) for d in G.DESKTOPS},
        "tiles_without_labels": sum(1 for r in vrows if r["n"] == 0),
        "frame_boxes": {c: {f"D{d}": box_cnt[(d, c)] for d in G.DESKTOPS}
                        for c in list(G.DET_CLASSES) + other},
        "other_classes_not_trained": other,
        "tile_labels": {c: {f"D{d}": vlab[(d, c)] for d in G.DESKTOPS} for c in G.DET_CLASSES},
        "verified_sha256": tiles_hash([Path(r["image"]) for r in vrows]),
        "tiles_hash_rule": "sha256 over tiles sorted by name of (name, sha256(image), "
                           "sha256(label))",
        "tiling": {"roi": "pose_segment.roi_json (stored)", "work_scale": 1.0,
                   "tile": T.TILE, "stride": T.STRIDE, "pad": T.PAD_VALUE},
        "folds": folds,
        "rows": vrows,
    }
    (env.OUT / "dataset" / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                                     encoding="utf-8")
    show = {k: v for k, v in manifest.items() if k != "rows"}
    print(json.dumps(show, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
