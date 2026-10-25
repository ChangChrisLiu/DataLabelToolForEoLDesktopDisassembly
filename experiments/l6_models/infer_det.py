"""Tiled RF-DETR inference on L6's verified scan frames (L4's ``infer.py`` with L6's frames).

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.infer_det --model F
    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.infer_det --model v2 --desktops 19,27

L3's ``RFDetector`` (L2's tiling and merge, rfdetr's forward in fp16 -- what
``tda.models.detector`` ports), every box >= ``--conf`` (0.001 = L2/L3/L4's
evaluation protocol) into the merge; each frame cropped to its ROI (``gt.py``
of L6: the stored one, D27's proposed one).  Writes
``dets/<model>[_cNN]/scan.json`` (``{"d:scan:step": [[x0, y0, x1, y1, cls,
conf], ...]}``) and ``scan_timing.json``.  The app is checked before the model
loads and every 10 frames; when it is open the script exits with code 3 and
writes nothing (``run_queue`` starts it again).
"""
from __future__ import annotations

from experiments.l6_models import env  # noqa: F401  (side effects first)

import argparse
import json
import sys
import time
from pathlib import Path

from experiments.l4_detector import gt as G4
from experiments.l6_models.export_gt import GT_DET, HELD_OUT
from experiments.l6_models.train_det import RUNS, SHIPPED_V2

MODELS = {
    "v1": env.WEIGHTS / "tda_det_rfdetr_small_all.pt",
    "v2": SHIPPED_V2,
    "v1h": RUNS / "v1h" / "last_ema.pth",
    "v2h": RUNS / "v2h" / "last_ema.pth",
    "F": RUNS / "F" / "last_ema.pth",
    "v3": RUNS / "v3" / "last_ema.pth",
}
EXIT_INTERRUPTED = 3


def tag_of(model: str, conf: float) -> str:
    return model if conf == 0.001 else f"{model}_c{int(round(conf * 100)):02d}"


def dets_path(model: str, conf: float = 0.001) -> Path:
    return env.ROOT / "dets" / tag_of(model, conf) / "scan.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--desktops", default=",".join(str(d) for d in HELD_OUT))
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args(argv)
    w = MODELS[args.model]
    if not w.exists():
        raise SystemExit(f"missing weights {w}")
    if env.app_running():
        return EXIT_INTERRUPTED
    from experiments.l2_detector import data as D
    from experiments.l3_detector.infer import RFDetector

    det = RFDetector(w, half=True, conf=args.conf)
    want = {int(d) for d in args.desktops.split(",")}
    frames = [f for f in G4.load(GT_DET) if f.desktop in want]
    out: dict[str, list] = {}
    tim = []
    t_all = time.perf_counter()
    for i, f in enumerate(frames):
        if i % 10 == 0 and env.app_running():
            print("[l6] the app is open: inference stops, nothing written", flush=True)
            return EXIT_INTERRUPTED
        img = D.read_rgb(f.image)
        t = {}
        dets = det.detect(img, f.roi, "scan", timings=t)
        tim.append({"key": f"{f.desktop}:scan:{f.step}", **t})
        out[f"{f.desktop}:scan:{f.step}"] = [d.row() for d in dets]
    p = dets_path(args.model, args.conf)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out), encoding="utf-8")
    p.with_name("scan_timing.json").write_text(json.dumps(tim), encoding="utf-8")
    import torch

    print(f"[l6] wrote {p} ({len(out)} frames of D{sorted(want)}, "
          f"{time.perf_counter() - t_all:.0f} s, peak CUDA "
          f"{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
