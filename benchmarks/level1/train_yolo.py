#!/usr/bin/env python
"""Baseline (a): Ultralytics YOLO instance segmentation, trained on a coco_to_yolo.py dataset.

    python train_yolo.py --data runs/yolo_visible/data.yaml --run runs/yolo11n_seg_visible \
        [--model yolo11n-seg.pt] [--imgsz 1280] [--epochs 100] [--batch 8] [--seed 0] [--device 0]

Pretrained weights: `--model` is a local .pt file or an Ultralytics asset name (yolo11n-seg.pt, yolo11s-seg.pt,
yolo26n-seg.pt, ...). An asset name is looked up in / downloaded to `--weights-dir` (default
<run>/../weights; set L1_WEIGHTS_DIR to share one folder); offline nodes need it there beforehand.
The run folder gets run.json (arguments, environment, timing), train.log, and Ultralytics' own output in
<run>/train/ (weights/best.pt, weights/last.pt, results.csv, args.yaml).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l1common import Logger, dump_json, env_report, set_seed  # noqa: E402


def resolve_weights(model, weights_dir):
    p = Path(model)
    if p.exists():
        return str(p.resolve())
    weights_dir.mkdir(parents=True, exist_ok=True)
    target = weights_dir / p.name
    if target.exists():
        return str(target)
    from ultralytics.utils.downloads import attempt_download_asset
    return str(Path(attempt_download_asset(str(target))).resolve())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml written by coco_to_yolo.py")
    ap.add_argument("--run", required=True, help="run folder")
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="'0', '0,1', 'cpu' (default: GPU 0 if available)")
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--extra", default="{}", help="JSON dict of further Ultralytics train() arguments")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from <run>/train/weights/last.pt")
    args = ap.parse_args(argv)

    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(args.weights_dir or os.environ.get("L1_WEIGHTS_DIR") or run.parent / "weights").resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    # Ultralytics settings and any helper download (its AMP check fetches a tiny model) stay out of $HOME
    os.environ.setdefault("YOLO_CONFIG_DIR", str(weights_dir / "ultralytics_cfg"))
    data = str(Path(args.data).resolve())
    log = Logger(run / "train.log")
    set_seed(args.seed)

    import torch
    from ultralytics import YOLO

    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    cmap = json.loads((Path(data).parent / "category_map.json").read_text(encoding="utf-8"))
    weights = resolve_weights(args.model, weights_dir)
    os.chdir(weights_dir)
    log("weights", weights, "device", device, "data", data)
    extra = json.loads(args.extra)
    meta = {"baseline": "ultralytics_yolo_seg", "args": vars(args), "weights": weights, "device": device,
            "env": env_report(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    dump_json(meta, run / "run.json", indent=1)

    t0 = time.time()
    last = run / "train" / "weights" / "last.pt"
    if args.resume and last.exists():
        log("resuming from", last)
        YOLO(str(last)).train(resume=True)
    else:
        _train(YOLO(weights), args, data, device, run, cmap, extra)
    secs = time.time() - t0
    best = run / "train" / "weights" / "best.pt"
    meta.update({"train_seconds": round(secs, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "best": str(best) if best.exists() else None, "last": str(last) if last.exists() else None})
    dump_json(meta, run / "run.json", indent=1)
    log(f"training done in {secs / 60:.1f} min; best {best if best.exists() else last}")


def _train(model, args, data, device, run, cmap, extra):
    model.train(data=data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, workers=args.workers,
                seed=args.seed, deterministic=True, device=device, project=str(run), name="train",
                exist_ok=True, patience=args.patience, val=not cmap.get("val_is_train", False), verbose=True,
                **extra)


if __name__ == "__main__":
    main()
