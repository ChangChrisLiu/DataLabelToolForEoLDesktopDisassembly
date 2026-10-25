"""Inference cost in the app's terms, with the new models wired in through the app's own overrides.

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l6_models.perf_app [--det v2,v3] [--sam]

Nothing in ``configs/`` changes.  For each detector a copy of
``configs/detector.yaml`` with ``model`` (and ``cache_root``, so the app's
cache under ``D:/DataSet/cache/det`` is never touched) replaced is written to
``ROOT/configs/`` and named by ``TDA_DETECTOR`` -- the app's override -- and
read by ``tda.models.detector.load_detector_config()``; the model is built by
the app's ``DetectionEngine`` (``make_detector`` -> ``RFDetrTileDetector``:
the manifest check, the class check, fp16, the warm-up).  On every frame of
D19 and D27 (1600 x 1600 scanner frames, the ROI crop, ``neighbour`` = the
next step as in the reverse walk):

* ``detect`` alone (the frame already decoded) -- ms per frame;
* ``process`` cold (read + detect + the candidates' dE, written to the cache)
  and warm (``answer_from_cache``) -- ms per frame;
* the cache file per desktop/view (bytes) and the detections stored per frame.

SAM: ``configs/sam.yaml`` (shipped) and a copy whose ``head`` is the v3 head,
named by ``TDA_SAM_HEAD`` and read by ``load_sam_head_config()``; each a
``SamService(heads=...)``; on 200 random D19/D27 masks (seed 0), alternating:
``set_image`` ms and ``predict`` ms for a box and a lone click (with the
mask's class, so the routing applies), and which head answered.  The head
files' sizes.  Writes ``ROOT/perf/perf_app.json``.  The GPU rule: refuses to
start while the app is open.
"""
from __future__ import annotations

from experiments.l6_models import env  # noqa: F401  (side effects first)

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from experiments.l4_detector import gt as G4
from experiments.l6_models.export_gt import GT_DET, HELD_OUT

PERF = env.ROOT / "perf"
CFG = env.ROOT / "configs"
DET_MODELS = {
    "v2": env.WEIGHTS / "tda_det_rfdetr_small_all_v2.pt",
    "v3": env.WEIGHTS / "tda_det_rfdetr_small_all_v3.pt",
    "F": env.WEIGHTS / "tda_det_rfdetr_small_l6_F.pth",
}
SAM_HEADS = {
    "shipped": None,
    "v3": env.WEIGHTS / "sam2.1_tda_head_v3.pt",
}


def det_config_file(name: str, model: Path) -> Path:
    data = yaml.safe_load((env.REPO / "configs" / "detector.yaml").read_text(encoding="utf-8"))
    data["model"] = str(model).replace("\\", "/")
    data["cache_root"] = str(env.ROOT / "det_cache").replace("\\", "/")
    CFG.mkdir(parents=True, exist_ok=True)
    out = CFG / f"detector_{name}.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def sam_config_file(name: str, head: Path) -> Path:
    data = yaml.safe_load((env.REPO / "configs" / "sam.yaml").read_text(encoding="utf-8"))
    data["head"] = str(head).replace("\\", "/")
    CFG.mkdir(parents=True, exist_ok=True)
    out = CFG / f"sam_{name}.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def q(vals, p):
    return round(float(np.percentile(vals, p)), 1) if vals else None


def det_perf(name: str, model: Path) -> dict:
    import shutil

    import cv2
    import torch

    from tda.models.det_engine import DetectionEngine, PlanItem
    from tda.models.detector import load_detector_config

    cfg_path = det_config_file(name, model)
    os.environ["TDA_DETECTOR"] = str(cfg_path)
    config = load_detector_config()
    if config is None:
        raise SystemExit(f"{cfg_path}: the app's loader turned the detector off")
    engine = DetectionEngine(config)
    t0 = time.perf_counter()
    engine.prepare()
    cache_dir = engine.cache.dir
    if cache_dir.exists():
        shutil.rmtree(cache_dir)           # a cold start for this model
    torch.cuda.reset_peak_memory_stats()
    described = engine.load_model()
    load_s = time.perf_counter() - t0
    frames = [f for f in G4.load(GT_DET) if f.desktop in HELD_OUT]
    by = {(f.desktop, f.step): f for f in frames}
    detect_ms, cold_ms, warm_ms, stored, sizes = [], [], [], [], {}
    for d in HELD_OUT:
        engine.open_view(d, "scan")
        steps = sorted(s for (dd, s) in by if dd == d)
        for s in steps:
            f = by[(d, s)]
            img = cv2.cvtColor(cv2.imread(f.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            torch.cuda.synchronize()
            t = time.perf_counter()
            engine.model.detect(img, tuple(f.roi), "scan", s)
            torch.cuda.synchronize()
            detect_ms.append(1000 * (time.perf_counter() - t))
        for s in steps:
            f = by[(d, s)]
            nb = by.get((d, s + 1))
            item = PlanItem(step=s, candidates=(f.image,), crop=tuple(f.roi),
                            neighbour=s + 1 if nb else None,
                            neighbour_candidates=(nb.image,) if nb else ())
            r = engine.process(item)
            cold_ms.append(1000 * r.seconds)
            stored.append(len(r.dets))
        p = engine.flush()
        sizes[f"D{d}"] = p.stat().st_size if p else 0
        engine.open_view(d, "scan")
        for s in steps:
            f = by[(d, s)]
            nb = by.get((d, s + 1))
            item = PlanItem(step=s, candidates=(f.image,), crop=tuple(f.roi),
                            neighbour=s + 1 if nb else None,
                            neighbour_candidates=(nb.image,) if nb else ())
            r = engine.answer_from_cache(item)
            if r is not None:
                warm_ms.append(1000 * r.seconds)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    engine.release()
    out = {"model": str(model), "config": str(cfg_path), "described": described,
           "load_s": round(load_s, 1), "frames": len(frames),
           "detect_ms": {"median": q(detect_ms, 50), "p90": q(detect_ms, 90)},
           "process_cold_ms": {"median": q(cold_ms, 50), "p90": q(cold_ms, 90)},
           "cache_warm_ms": {"median": q(warm_ms, 50), "p90": q(warm_ms, 90), "n": len(warm_ms)},
           "stored_dets_per_frame": round(statistics.mean(stored), 1) if stored else None,
           "cache_bytes": sizes, "peak_cuda_mb": round(peak),
           "model_bytes": model.stat().st_size}
    del engine
    torch.cuda.empty_cache()
    return out


def sam_perf(n: int = 200) -> dict:
    import torch

    from experiments.l5_sam import common as C
    from experiments.l5_sam import gt as G5
    from experiments.l6_models.train_sam import load_recs
    from tda.models.sam_heads import load_sam_head_config
    from tda.models.sam_service import SamRequest, SamService

    services = {}
    for name, head in SAM_HEADS.items():
        if head is None:
            os.environ["TDA_SAM_HEAD"] = str(env.REPO / "configs" / "sam.yaml")
        else:
            os.environ["TDA_SAM_HEAD"] = str(sam_config_file(name, head))
        cfg = load_sam_head_config()
        if not cfg.wanted:
            raise SystemExit(f"{name}: the head configuration is off ({cfg.off})")
        svc = SamService(heads=cfg)
        if not svc.tuned_loaded:
            raise SystemExit(f"{name}: the tuned head did not load ({svc.head_off_reason})")
        services[name] = (svc, str(cfg.head))
    recs, images = load_recs()
    recs = [r for r in recs if r.desktop in HELD_OUT]
    rng = np.random.default_rng(0)
    pick = [recs[i] for i in sorted(rng.choice(len(recs), size=min(n, len(recs)), replace=False))]
    frames = {}
    res = {k: {"set_image": [], "box": [], "point": [], "heads": {}} for k in services}

    def timed(fn, *a):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*a)
        torch.cuda.synchronize()
        return out, 1000 * (time.perf_counter() - t)

    for r in pick:
        if r.frame not in frames:
            frames.clear()
            frames.update(G5.load_frames(images, {r.frame}))
        win = C.crop_window(r.bbox, r.frame_hw)
        crop = C.crop_image(frames[r.frame], win)
        box = C.box_in_crop(C.jittered_box(r.bbox, 1), win)
        px, py = C.to_crop((r.point[0] + 0.5, r.point[1] + 0.5), win)
        for name, (svc, _h) in services.items():
            _, ms = timed(svc.set_image, crop)
            res[name]["set_image"].append(ms)
            for kind, req in (("box", SamRequest(image_crop=crop, box=box, multimask=True,
                                                 instance_cls=r.cls)),
                              ("point", SamRequest(image_crop=crop, points=[(px, py, 1)],
                                                   multimask=True, instance_cls=r.cls))):
                out, ms = timed(svc.predict, req)
                res[name][kind].append(ms)
                key = f"{kind}:{out.head}"
                res[name]["heads"][key] = res[name]["heads"].get(key, 0) + 1
    out = {}
    for name, (svc, head) in services.items():
        out[name] = {"head": head, "head_bytes": Path(head).stat().st_size,
                     "set_image_ms": {"median": q(res[name]["set_image"], 50),
                                      "p90": q(res[name]["set_image"], 90)},
                     "decode_box_ms": {"median": q(res[name]["box"], 50),
                                       "p90": q(res[name]["box"], 90)},
                     "decode_point_ms": {"median": q(res[name]["point"], 50),
                                         "p90": q(res[name]["point"], 90)},
                     "answered_by": res[name]["heads"], "masks": len(pick)}
    out["peak_cuda_gb_both_services"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det", default="v2,v3")
    ap.add_argument("--sam", action="store_true")
    args = ap.parse_args(argv)
    if env.app_running():
        print("[l6] the app is open: no measurement", flush=True)
        return 3
    PERF.mkdir(parents=True, exist_ok=True)
    path = PERF / "perf_app.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name in [n for n in args.det.split(",") if n]:
        result.setdefault("detector", {})[name] = det_perf(name, DET_MODELS[name])
        print(f"[l6] detector {name}: {json.dumps(result['detector'][name])}", flush=True)
    if args.sam:
        result["sam"] = sam_perf()
        print(f"[l6] SAM: {json.dumps(result['sam'])}", flush=True)
    result["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
