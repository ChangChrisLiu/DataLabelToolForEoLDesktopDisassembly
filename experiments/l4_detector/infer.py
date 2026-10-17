"""Tiled RF-DETR inference on the verified scan frames (or L3's D13 OAK frames).

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.infer --model production
    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.infer --model v2_hold13 --desktops 13
    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.infer --model l3hold13 --set oak

L3's ``RFDetector`` (L2's tiling and merge, rfdetr's forward in fp16) -- the
same tiles and merge the app runs (``tda.models.detector`` is their port).
``--conf`` is the score floor before the merge: 0.001 = L2/L3's evaluation
protocol (the default), 0.10 = the app's ``merge_floor``.

* ``--set scan``: the verified scan frames (``gt.py``), cropped to the pose
  segment's stored ROI, read from the file the app shows;
* ``--set oak``: L3's held-out OAK protocol for the regression check -- D13
  oak1 / oak2 frames with ``ls:`` drafts, ROI = the drafts' union + 10 %
  (L2's ``roi_of``), drafts rebuilt from the L4 DB copy.

Writes ``dets/<model>[_cNN]/<set>.json`` (``{"d:view:step": [[x0, y0, x1, y1,
cls, conf], ...]}``) and ``<set>_timing.json`` (per frame: prep / net / merge /
total seconds, tiles).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l4_detector import env  # noqa: E402

MODELS = {
    "production": env.PRODUCTION,
    "l3hold13": env.L3_HOLD13,
    "v2_hold13": env.OUT / "rfdetr_small" / "runs" / "hold13" / "last_ema.pth",
    "v2_hold5": env.OUT / "rfdetr_small" / "runs" / "hold5" / "last_ema.pth",
    "v2_all": env.OUT / "rfdetr_small" / "runs" / "all" / "last_ema.pth",
}


def tag_of(model: str, conf: float) -> str:
    return model if conf == 0.001 else f"{model}_c{int(round(conf * 100)):02d}"


def dets_path(model: str, dset: str, conf: float = 0.001) -> Path:
    return env.OUT / "dets" / tag_of(model, conf) / f"{dset}.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--set", dest="dset", default="scan", choices=("scan", "oak"))
    ap.add_argument("--desktops", default="2,3,4,5,13")
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args(argv)
    w = MODELS[args.model]
    if not w.exists():
        raise SystemExit(f"missing weights {w}")
    env.bind_l2(env.OUT / "l2bind")
    from experiments.l2_detector import data as D
    from experiments.l3_detector.infer import RFDetector

    det = RFDetector(w, half=True, conf=args.conf)
    out: dict[str, list] = {}
    tim = []
    t_all = time.perf_counter()
    if args.dset == "scan":
        from experiments.l4_detector import gt as G

        want = {int(d) for d in args.desktops.split(",")}
        for f in G.load():
            if f.desktop not in want:
                continue
            img = D.read_rgb(f.image)
            t = {}
            dets = det.detect(img, f.roi, "scan", timings=t)
            tim.append({"key": f"{f.desktop}:scan:{f.step}", **t})
            out[f"{f.desktop}:scan:{f.step}"] = [d.row() for d in dets]
    else:
        boxes = D.load_boxes()
        by = D.index(boxes)
        con = D.connect()
        for view in ("oak1", "oak2"):
            steps = sorted(s for (d, v, s) in by if d == 13 and v == view)
            roi = D.roi_of(boxes, 13, view)
            for step in steps:
                path = D.frame_path(con, 13, view, step)
                img = D.read_rgb(path) if path else None
                if img is None:
                    print(f"  missing D13 {view} s{step}")
                    continue
                t = {}
                dets = det.detect(img, roi, view, timings=t)
                tim.append({"key": f"13:{view}:{step}", **t})
                out[f"13:{view}:{step}"] = [d.row() for d in dets]
            print(f"  D13 {view}: {len(steps)} frames", flush=True)
        con.close()
    p = dets_path(args.model, args.dset, args.conf)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out), encoding="utf-8")
    p.with_name(f"{args.dset}_timing.json").write_text(json.dumps(tim), encoding="utf-8")
    import torch

    print(f"wrote {p} ({len(out)} frames, {time.perf_counter() - t_all:.0f} s, "
          f"peak CUDA {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
