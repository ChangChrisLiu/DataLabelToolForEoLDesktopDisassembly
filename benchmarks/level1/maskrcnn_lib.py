"""Dataset, model and inference shared by train_maskrcnn.py and predict_maskrcnn.py (torchvision Mask R-CNN)."""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l1common import contiguous_categories, image_root, load_json, rle_encode  # noqa: E402


def read_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def resize_image(img, size):
    """Scale so that the longer side is `size`; returns (image, scale)."""
    import cv2
    h, w = img.shape[:2]
    s = size / max(h, w)
    if abs(s - 1.0) < 1e-9:
        return img, 1.0
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp), s


class CocoInstanceDataset:
    """COCO split -> (image tensor CxHxW float 0..1, target dict) at `size`; iscrowd rows are left out."""

    def __init__(self, coco_path, size, train=False, image_root_override=None):
        self.coco = load_json(coco_path)
        self.root = image_root(self.coco, coco_path, image_root_override)
        self.size = size
        self.train = train
        self.cat_ids, to_idx, self.names = contiguous_categories(self.coco["categories"])
        self.cat_to_label = {cid: i + 1 for cid, i in to_idx.items()}  # 0 = background
        self.label_to_cat = {v: k for k, v in self.cat_to_label.items()}
        self.images = self.coco["images"]
        self.anns = {}
        for a in self.coco["annotations"]:
            if not a["iscrowd"]:
                self.anns.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        import cv2
        import torch
        from pycocotools import mask as mask_utils
        im = self.images[i]
        img, s = resize_image(read_rgb(self.root / im["file_name"]), self.size)
        h, w = img.shape[:2]
        boxes, labels, masks = [], [], []
        for a in self.anns.get(im["id"], []):
            x, y, bw, bh = a["bbox"]
            box = [x * s, y * s, (x + bw) * s, (y + bh) * s]
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            m = mask_utils.decode(a["segmentation"])
            if s != 1.0:
                m = (cv2.resize(m * 255, (w, h), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8)
            boxes.append(box)
            labels.append(self.cat_to_label[a["category_id"]])
            masks.append(m)
        if self.train and random.random() < 0.5:  # horizontal flip (torch seeds `random` per worker)
            img = img[:, ::-1]
            masks = [m[:, ::-1] for m in masks]
            boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]
        t_img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.from_numpy(np.ascontiguousarray(np.stack(masks))) if masks
            else torch.zeros((0, h, w), dtype=torch.uint8),
            "image_id": im["id"],
        }
        return t_img, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(num_classes, size, arch="v2", weights="coco", score_thresh=0.001, detections=100):
    """Mask R-CNN ResNet-50-FPN with a `num_classes` (+ background) head.

    weights: 'coco' (torchvision COCO weights, heads replaced), 'imagenet' (backbone only), 'none',
    or a path to a state dict / checkpoint of this script."""
    import torch
    from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_Weights,
                                              maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.models import ResNet50_Weights

    fn = maskrcnn_resnet50_fpn_v2 if arch == "v2" else maskrcnn_resnet50_fpn
    w_coco = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if arch == "v2" else MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    kw = dict(min_size=size, max_size=size, box_score_thresh=score_thresh, box_detections_per_img=detections)
    if weights == "coco":
        model = fn(weights=w_coco, **kw)
    elif weights == "imagenet":
        model = fn(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1, num_classes=num_classes + 1, **kw)
    else:
        model = fn(weights=None, weights_backbone=None, num_classes=num_classes + 1, **kw)
    if model.roi_heads.box_predictor.cls_score.out_features != num_classes + 1:
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes + 1)
        in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, num_classes + 1)
    if weights not in ("coco", "imagenet", "none"):
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ck.get("model", ck))
    return model


@__import__("torch").no_grad()
def predict(model, coco_path, size, device, label_to_cat, image_root_override=None, log=print, amp=True):
    """Run on every image of a split; returns a COCO results list in full-frame coordinates."""
    import torch
    import torch.nn.functional as F
    coco = load_json(coco_path)
    root = image_root(coco, coco_path, image_root_override)
    model.eval()
    preds = []
    t0 = time.time()
    use_amp = amp and device.type == "cuda"
    for n, im in enumerate(coco["images"]):
        orig = read_rgb(root / im["file_name"])
        H, W = orig.shape[:2]
        img, s = resize_image(orig, size)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model([t])[0]
        boxes = out["boxes"].float().cpu().numpy() / s
        scores = out["scores"].float().cpu().numpy()
        labels = out["labels"].cpu().numpy()
        masks = out.get("masks")
        for j0 in range(0, len(boxes), 25):
            mm = None
            if masks is not None and len(masks):
                chunk = masks[j0:j0 + 25].float()
                if s != 1.0:
                    chunk = F.interpolate(chunk, size=(H, W), mode="bilinear", align_corners=False)
                mm = (chunk[:, 0] > 0.5).cpu().numpy()
            for j in range(j0, min(j0 + 25, len(boxes))):
                x0, y0, x1, y1 = [float(v) for v in boxes[j]]
                p = {"image_id": im["id"], "category_id": int(label_to_cat[int(labels[j])]),
                     "score": round(float(scores[j]), 5),
                     "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]}
                if mm is not None:
                    p["segmentation"] = rle_encode(mm[j - j0])
                preds.append(p)
        if n % 50 == 0:
            log(f"predict {n + 1}/{len(coco['images'])} images, {len(preds)} predictions, {time.time() - t0:.0f}s")
    return preds
