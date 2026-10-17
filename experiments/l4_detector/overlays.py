"""Overlay sheets: verified boxes vs two models' detections on held-out frames.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.overlays \\
        --a l3hold13 --b v2_hold13 --cls screw 13:12 13:30 ...

Each ``desktop:step`` becomes ``overlays/<a>_vs_<b>_<cls>_D<d>_s<step>.jpg``:
the ROI crop of the frame twice, side by side -- left model ``a``, right model
``b`` -- with the verified boxes of ``cls`` in green and the model's
detections of ``cls`` at conf >= 0.10 in red (centre outside every green box)
or cyan (centre inside one), each with its confidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l4_detector import env  # noqa: E402
from experiments.l4_detector import gt as G  # noqa: E402
from experiments.l4_detector.infer import dets_path  # noqa: E402


def panel(img, roi, gts, rows, title):
    x0, y0, x1, y1 = roi
    out = cv2.cvtColor(img[y0:y1, x0:x1].copy(), cv2.COLOR_RGB2BGR)
    for b in gts:
        cv2.rectangle(out, (b[0] - x0, b[1] - y0), (b[2] - x0 - 1, b[3] - y0 - 1), (0, 200, 0), 1)
    for r in rows:
        cx, cy = 0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3])
        inside = any(g[0] <= cx < g[2] and g[1] <= cy < g[3] for g in gts)
        col = (255, 255, 0) if inside else (0, 0, 255)
        p0 = (int(r[0]) - x0, int(r[1]) - y0)
        cv2.rectangle(out, p0, (int(r[2]) - x0, int(r[3]) - y0), col, 1)
        cv2.putText(out, f"{r[5]:.2f}", (p0[0], max(10, p0[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, col, 1, cv2.LINE_AA)
    cv2.putText(out, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--cls", default="screw")
    ap.add_argument("frames", nargs="+")
    args = ap.parse_args(argv)
    from experiments.l2_detector import data as D

    fby = G.index(G.load())
    da = json.loads(dets_path(args.a, "scan").read_text(encoding="utf-8"))
    db = json.loads(dets_path(args.b, "scan").read_text(encoding="utf-8"))
    out = env.OUT / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    for key in args.frames:
        d, s = (int(v) for v in key.split(":"))
        f = fby[(d, s)]
        img = D.read_rgb(f.image)
        gts = [b.box for b in f.boxes if b.cls == args.cls]
        k = f"{d}:scan:{s}"
        ra = [r for r in da.get(k, []) if r[4] == args.cls and r[5] >= 0.10]
        rb = [r for r in db.get(k, []) if r[4] == args.cls and r[5] >= 0.10]
        sheet = np.hstack([panel(img, f.roi, gts, ra, f"{args.a} D{d} s{s}"),
                           np.full((f.roi[3] - f.roi[1], 8, 3), 255, np.uint8),
                           panel(img, f.roi, gts, rb, f"{args.b} D{d} s{s}")])
        p = out / f"{args.a}_vs_{args.b}_{args.cls}_D{d}_s{s:03d}.jpg"
        cv2.imwrite(str(p), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
