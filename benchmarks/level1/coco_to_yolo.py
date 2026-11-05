#!/usr/bin/env python
"""Convert make_coco.py splits to an Ultralytics YOLO segmentation dataset.

    python coco_to_yolo.py --coco runs/coco_visible --out runs/yolo_visible [--link auto|hardlink|symlink|copy]

Layout written (the pack itself is never written to):
    <out>/images/<split>/D<nn>_S<nn>.png   link (or copy) of the pack picture
    <out>/labels/<split>/D<nn>_S<nn>.txt   one line per instance: class x1 y1 x2 y2 ... (normalised polygon)
    <out>/data.yaml                        Ultralytics dataset file (names = pack class names, contiguous ids)
    <out>/category_map.json                YOLO class index -> pack category id (used by predict_yolo.py)

Only `iscrowd = 0` rows become labels (YOLO has no ignore regions; the ignore regions are therefore
background during training, as in most COCO-style YOLO training). A mask with several pieces (an occluded
part) is joined into one polygon the way Ultralytics' own COCO converter does (`merge_multi_segment`);
holes are not representable in a YOLO polygon and are filled.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l1common import contiguous_categories, dump_json, image_root, load_json  # noqa: E402


def link_file(src, dst, mode):
    if dst.exists():
        return "exists"
    modes = ("hardlink", "symlink", "copy") if mode == "auto" else (mode,)
    for m in modes:
        try:
            if m == "hardlink":
                os.link(src, dst)
            elif m == "symlink":
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
            return m
        except OSError:
            continue
    raise OSError(f"could not link or copy {src} -> {dst}")


def mask_to_polygon(rle, min_piece_px=4, eps=1.0):
    """RLE -> one polygon (flat list of x, y pixel coords) or None."""
    import cv2
    from pycocotools import mask as mask_utils
    m = mask_utils.decode(rle).astype(np.uint8)
    if m.sum() == 0:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
    for c in contours:
        if cv2.contourArea(c) < min_piece_px and len(contours) > 1:
            continue
        if eps > 0:
            c = cv2.approxPolyDP(c, eps, True)
        if len(c) >= 3:
            pieces.append(c.reshape(-1, 2).astype(np.float64))
    if not pieces:
        c = max(contours, key=len).reshape(-1, 2).astype(np.float64)
        if len(c) < 3:  # a 1-2 px sliver: use its box
            x0, y0 = c.min(0)
            x1, y1 = c.max(0) + 1
            c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        pieces = [c]
    if len(pieces) == 1:
        return pieces[0]
    from ultralytics.data.converter import merge_multi_segment
    merged = merge_multi_segment([p.reshape(-1).tolist() for p in pieces])
    return np.concatenate(merged, axis=0)


def convert_split(split, coco_path, out, to_idx, link_mode, override_root):
    coco = load_json(coco_path)
    root = image_root(coco, coco_path, override_root)
    img_dir, lab_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    by_img = {}
    for a in coco["annotations"]:
        if not a["iscrowd"]:
            by_img.setdefault(a["image_id"], []).append(a)
    modes = {}
    n_lines = 0
    for im in coco["images"]:
        src = (root / im["file_name"]).resolve()
        if not src.exists():
            sys.exit(f"missing image {src}; is --image-root right?")
        stem = im["file_name"].replace("/images/", "_").replace("/", "_").rsplit(".", 1)[0]
        dst = img_dir / f"{stem}{src.suffix}"
        m = link_file(src, dst, link_mode)
        modes[m] = modes.get(m, 0) + 1
        w, h = im["width"], im["height"]
        lines = []
        for a in by_img.get(im["id"], []):
            poly = mask_to_polygon(a["segmentation"])
            if poly is None:
                continue
            poly = poly / np.array([w, h], dtype=np.float64)
            poly = np.clip(poly, 0.0, 1.0)
            lines.append(f"{to_idx[a['category_id']]} " + " ".join(f"{v:.6f}" for v in poly.reshape(-1)))
        (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_lines += len(lines)
    return len(coco["images"]), n_lines, modes, coco


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, help="folder with train.json / val.json / test.json (make_coco.py --out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--link", choices=("auto", "hardlink", "symlink", "copy"), default="auto",
                    help="how images are placed (auto: hardlink, else symlink, else copy)")
    ap.add_argument("--image-root", default=None, help="override info.image_root of the COCO files")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args(argv)
    coco_dir, out = Path(args.coco), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cats = None
    present = {}
    for split in args.splits:
        p = coco_dir / f"{split}.json"
        if not p.exists():
            print(f"skip {split}: {p} not found")
            continue
        if cats is None:
            cats = load_json(p)["categories"]
            ids, to_idx, names = contiguous_categories(cats)
        n_img, n_lab, modes, _ = convert_split(split, p, out, to_idx, args.link, args.image_root)
        present[split] = n_img
        print(f"{split:5s}: {n_img} images ({modes}), {n_lab} polygons")
    # Ultralytics needs a val entry: an empty val split falls back to train (train_yolo.py then runs with val=False)
    val_split = "val" if present.get("val") else "train"
    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": f"images/{val_split}",
        "test": "images/test" if present.get("test") else None,
        "names": {i: names[cid] for i, cid in enumerate(ids)},
    }
    import yaml
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({k: v for k, v in data.items() if v is not None}, f, sort_keys=False)
    dump_json({"yolo_index_to_category_id": {str(i): cid for i, cid in enumerate(ids)},
               "names": {str(cid): names[cid] for cid in ids}, "val_is_train": val_split == "train",
               "coco_dir": str(coco_dir.resolve())}, out / "category_map.json", indent=1)
    print(f"done in {time.time() - t0:.1f}s -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
