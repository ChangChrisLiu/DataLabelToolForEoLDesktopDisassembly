#!/usr/bin/env python
"""Run a train_maskrcnn.py checkpoint on a make_coco.py split and write COCO-format predictions.

    python predict_maskrcnn.py --gt coco/test.json --ckpt runs/maskrcnn_visible/best.pt \
        --out runs/maskrcnn_visible/preds_test.json

The frame is scaled as in training (`size` from the checkpoint), boxes and mask probabilities are mapped back
to the full 1600 x 1600 frame, masks thresholded at 0.5. Score threshold 0.001, at most 100 per image.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l1common import dump_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--size", type=int, default=None, help="default: the checkpoint's training size")
    ap.add_argument("--score-thresh", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from maskrcnn_lib import build_model, predict

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    size = args.size or ck["size"]
    label_to_cat = {int(k): int(v) for k, v in ck["label_to_category_id"].items()}
    model = build_model(ck["num_classes"], size, ck["arch"], "none", args.score_thresh, args.max_det)
    model.load_state_dict(ck["model"])
    model.to(device)
    t0 = time.time()
    preds = predict(model, args.gt, size, device, label_to_cat, args.image_root, amp=not args.no_amp)
    dump_json(preds, args.out)
    meta = {"ckpt": str(args.ckpt), "epoch": ck.get("epoch"), "gt": str(args.gt), "predictions": len(preds),
            "size": size, "seconds": round(time.time() - t0, 1), "args": vars(args)}
    dump_json(meta, Path(args.out).with_suffix(".meta.json"), indent=1)
    print(f"{len(preds)} predictions in {meta['seconds']}s -> {args.out}")


if __name__ == "__main__":
    main()
