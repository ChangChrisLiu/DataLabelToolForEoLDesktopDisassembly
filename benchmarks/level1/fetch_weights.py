#!/usr/bin/env python
"""Download the pretrained weights the baselines start from, for machines whose compute nodes are offline.

    python fetch_weights.py --yolo-dir <folder>      # Mask R-CNN weights go to $TORCH_HOME/hub/checkpoints

* torchvision Mask R-CNN R50-FPN v2 and v1 COCO weights (maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth,
  maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth) -> $TORCH_HOME/hub/checkpoints/ (set TORCH_HOME first).
* Ultralytics yolo11n-seg.pt / yolo11s-seg.pt, plus the small detection models its AMP self-check loads
  (yolo11n.pt, yolo26n.pt) -> --yolo-dir (pass the same folder as train_yolo.py --weights-dir / L1_WEIGHTS_DIR).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-dir", default=os.environ.get("L1_WEIGHTS_DIR", "weights"))
    ap.add_argument("--yolo", nargs="+", default=["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11n.pt", "yolo26n.pt"])
    args = ap.parse_args(argv)
    if "TORCH_HOME" not in os.environ:
        print("note: TORCH_HOME is not set; torchvision uses ~/.cache/torch")
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_Weights
    for w in (MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1, MaskRCNN_ResNet50_FPN_Weights.COCO_V1):
        w.get_state_dict(progress=True)
        print("ok", w.url)
    d = Path(args.yolo_dir)
    d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(d / "ultralytics_cfg"))
    from ultralytics.utils.downloads import attempt_download_asset
    for name in args.yolo:
        print("ok", attempt_download_asset(str(d / name)))


if __name__ == "__main__":
    main()
