#!/usr/bin/env python
"""Run a trained YOLO segmentation model on a make_coco.py split and write COCO-format predictions.

    python predict_yolo.py --gt coco/test.json --weights runs/yolo/train/weights/best.pt \
        --cmap runs/yolo_visible/category_map.json --out runs/yolo/preds_test.json [--imgsz 1280]

Each prediction: image_id, category_id (pack id), score, bbox [x, y, w, h] and segmentation (RLE) in the
full 1600 x 1600 frame. Defaults follow COCO practice: conf 0.001, NMS IoU 0.7, at most 100 per image.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l1common import dump_json, image_root, load_json, rle_encode  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="split JSON (images to run on)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cmap", required=True, help="category_map.json from coco_to_yolo.py")
    ap.add_argument("--out", required=True, help="predictions JSON")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    import torch
    from ultralytics import YOLO
    from ultralytics.utils import ops

    coco = load_json(args.gt)
    root = image_root(coco, args.gt, args.image_root)
    cmap = {int(k): int(v) for k, v in load_json(args.cmap)["yolo_index_to_category_id"].items()}
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.weights)
    images = coco["images"]
    preds = []
    t0 = time.time()
    for i in range(0, len(images), args.batch):
        chunk = images[i:i + args.batch]
        paths = [str(root / im["file_name"]) for im in chunk]
        results = model.predict(paths, imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det,
                                device=device, retina_masks=False, verbose=False)
        for im, r in zip(chunk, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            masks = r.masks.data if r.masks is not None else None
            h, w = r.orig_shape
            for j0 in range(0, len(xyxy), 25):
                mm = None
                if masks is not None:
                    mm = (ops.scale_masks(masks[j0:j0 + 25][None].float(), (h, w))[0] > 0.5).cpu().numpy()
                for j in range(j0, min(j0 + 25, len(xyxy))):
                    x0, y0, x1, y1 = [float(v) for v in xyxy[j]]
                    p = {"image_id": im["id"], "category_id": cmap[int(cls[j])], "score": round(float(conf[j]), 5),
                         "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]}
                    if mm is not None:
                        p["segmentation"] = rle_encode(mm[j - j0])
                    preds.append(p)
        if (i // args.batch) % 25 == 0:
            print(f"{min(i + args.batch, len(images))}/{len(images)} images, {len(preds)} predictions, "
                  f"{time.time() - t0:.0f}s", flush=True)
    dump_json(preds, args.out)
    meta = {"weights": str(args.weights), "gt": str(args.gt), "images": len(images), "predictions": len(preds),
            "seconds": round(time.time() - t0, 1), "args": vars(args)}
    dump_json(meta, Path(args.out).with_suffix(".meta.json"), indent=1)
    print(f"{len(preds)} predictions on {len(images)} images in {meta['seconds']}s -> {args.out}")


if __name__ == "__main__":
    main()
