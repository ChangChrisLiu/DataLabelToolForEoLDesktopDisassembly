"""Overlays of the worst masks per group: GT (green) vs each model's shown mask (red).

::

    python -m experiments.l5_sam.overlays --csv baseline_all --prompt point_box \\
        --model stock=D:/DataSet/models/weights/sam2.1_hiera_large.pt [--model f13=...] [--desktop 13]

Selection: per group the K masks with the lowest stock IoU for ``--prompt``, one
per instance.  Each tile is zoomed on the object (3x its longest side, at least
64 px) with the prompt drawn in yellow.  Writes ``OUT/overlays/<csv>_<group>.jpg``.
"""
from __future__ import annotations

from experiments.l5_sam import env  # noqa: F401

import argparse

import cv2
import numpy as np
import pandas as pd

from experiments.l5_sam import aggregate as A
from experiments.l5_sam import common as C
from experiments.l5_sam import gt as G

TILE = 256


def tile(crop, win, rec, gt_mask, pred, label, prompt):
    x0, y0, x1, y1 = rec.bbox
    longest = max(x1 - x0, y1 - y0)
    half = max(32, int(1.5 * longest))
    cx, cy = (x0 + x1) // 2 - win.x0, (y0 + y1) // 2 - win.y0
    s = win.side
    a0, b0 = max(0, cx - half), max(0, cy - half)
    a1, b1 = min(s, cx + half), min(s, cy + half)
    native = crop if not win.oversize else cv2.resize(crop, (s, s), interpolation=cv2.INTER_LINEAR)
    img = native[b0:b1, a0:a1].copy()
    f = TILE / max(img.shape[:2])
    img = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)), interpolation=cv2.INTER_NEAREST)
    bgr = img[:, :, ::-1].copy()
    for m, col in ((gt_mask, (0, 255, 0)), (pred, (0, 0, 255))):
        mm = cv2.resize(m[b0:b1, a0:a1].astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(bgr, cnts, -1, col, 1)
    if prompt in ("point", "point_box"):
        px, py = rec.point[0] - win.x0 - a0, rec.point[1] - win.y0 - b0
        cv2.circle(bgr, (int((px + 0.5) * f), int((py + 0.5) * f)), 3, (0, 255, 255), -1)
    if prompt in ("box", "point_box", "box_loose"):
        bx = (C.loose_box(rec.bbox) if prompt == "box_loose"
              else C.jittered_box(rec.bbox, C.seed_of(rec.desktop, rec.step, rec.instance)))
        p0 = (int((bx[0] - win.x0 - a0) * f), int((bx[1] - win.y0 - b0) * f))
        p1 = (int((bx[2] - win.x0 - a0) * f), int((bx[3] - win.y0 - b0) * f))
        cv2.rectangle(bgr, p0, p1, (0, 255, 255), 1)
    out = np.zeros((TILE + 18, TILE, 3), np.uint8)
    out[:bgr.shape[0], :bgr.shape[1]] = bgr
    cv2.putText(out, label, (2, TILE + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--prompt", default="point_box")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--desktop", type=int)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args(argv)

    import torch  # noqa: F401
    from tda.models.sam_service import SamRequest, SamService

    env.limit_threads()
    d = A.load(args.csv)
    d = d[(d["model"] == "stock") & (d["prompt"] == args.prompt)]
    if args.desktop:
        d = d[d["desktop"] == args.desktop]
    recs, images = G.load()
    by_key = {(r.desktop, r.step, r.instance): r for r in recs}
    models = []
    for spec in args.model:
        name, path = spec.split("=", 1)
        models.append((name, SamService(checkpoint=path, config=env.SAM2_CONFIG, device="cuda")))
    out_dir = env.OUT / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_cache: dict = {}
    for g in A.GROUPS:
        s = d[A.in_group(d, g)].sort_values("iou").drop_duplicates("inst").head(args.k)
        rows = []
        for _, row in s.iterrows():
            rec = by_key[(int(row["desktop"]), int(row["step"]), row["instance"])]
            if rec.frame not in frames_cache:
                frames_cache[rec.frame] = G.load_frames(images, {rec.frame})[rec.frame]
            frame = frames_cache[rec.frame]
            win = C.crop_window(rec.bbox, rec.frame_hw)
            crop = C.crop_image(frame, win)
            gtm = rec.mask_in(win.rect)
            seed = C.seed_of(rec.desktop, rec.step, rec.instance)
            jbox = C.box_in_crop(C.jittered_box(rec.bbox, seed), win)
            px, py = C.to_crop((rec.point[0] + 0.5, rec.point[1] + 0.5), win)
            pts = [] if args.prompt in ("box", "box_loose") else [(px, py, 1)]
            box = None if args.prompt == "point" else jbox
            if args.prompt == "box_loose":
                box = C.box_in_crop(C.loose_box(rec.bbox), win)
            tiles = []
            for name, svc in models:
                res = svc.predict(SamRequest(image_crop=crop, points=pts, box=box, multimask=True))
                pred = C.paste_mask(res.mask, win)
                tiles.append(tile(crop, win, rec, gtm, pred,
                                  f"{name} D{rec.desktop}s{rec.step} {rec.cls[:10]} {C.iou(pred, gtm):.2f}",
                                  args.prompt))
            rows.append(np.concatenate(tiles, axis=1))
        if not rows:
            continue
        grid = []
        per_row = max(1, 4 // len(models))
        for i in range(0, len(rows), per_row):
            chunk = rows[i:i + per_row]
            while len(chunk) < per_row:
                chunk.append(np.zeros_like(rows[0]))
            grid.append(np.concatenate(chunk, axis=1))
        tag = f"{args.csv}_{args.prompt}_{g}" + (f"_D{args.desktop}" if args.desktop else "")
        cv2.imwrite(str(out_dir / f"{tag}.jpg"), np.concatenate(grid, axis=0),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[overlay] {tag}.jpg ({len(rows)} masks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
