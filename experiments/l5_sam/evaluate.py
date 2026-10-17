"""Evaluate SAM 2.1 checkpoints on verified masks exactly as the app prompts it.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l5_sam.evaluate \\
        --model stock=D:/DataSet/models/weights/sam2.1_hiera_large.pt \\
        [--model f13=D:/DataSet/models/weights/sam2.1_hiera_large_tda_f13.pt] \\
        --desktops 13 --out f13

Every model is the app's own :class:`tda.models.sam_service.SamService`
(bf16, candidates ranked by predicted IoU; the app shows candidate 0).  Per
mask: sam_compare's window, one embedding per model, then the prompts of
:data:`experiments.l5_sam.common.PROMPTS`.  Several models run in the same
process, alternating per mask, so their timings are comparable (gate G5).

Writes ``OUT/eval/<out>.csv`` (one row per mask x model x prompt).
"""
from __future__ import annotations

from experiments.l5_sam import env  # noqa: F401  (side effects first)

import argparse
import csv
import time
from collections import defaultdict

import numpy as np

from experiments.l5_sam import common as C
from experiments.l5_sam import gt as G

FIELDS = ["desktop", "step", "instance", "cls", "area", "size_bucket", "visibility",
          "occlusion", "drawn", "groups", "win_side", "oversize", "model", "prompt",
          "iou", "bf", "oracle_iou", "n_cand", "score", "single_iou", "single_bf",
          "pred_area", "embed_ms", "decode_ms", "status"]


def _sync_ms(torch, fn, *a, **k):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*a, **k)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def eval_mask(torch, svc, rec: G.Rec, crop: np.ndarray, win: C.Window, gt: np.ndarray,
              embed_ms: float) -> list[dict]:
    from tda.models.sam_service import SamRequest

    seed = C.seed_of(rec.desktop, rec.step, rec.instance)
    jbox = C.box_in_crop(C.jittered_box(rec.bbox, seed), win)
    lbox = C.box_in_crop(C.loose_box(rec.bbox), win)
    px, py = C.to_crop((rec.point[0] + 0.5, rec.point[1] + 0.5), win)
    pt = (px, py, 1)
    first = {"box": ([], jbox), "point": ([pt], None), "point_box": ([pt], jbox),
             "box_loose": ([], lbox)}
    rows = []
    shown_pb = None
    for prompt, (pts, box) in first.items():
        res, ms = _sync_ms(torch, svc.predict,
                           SamRequest(image_crop=crop, points=list(pts), box=box, multimask=True))
        cands = [C.paste_mask(m, win) for m in res.candidates]
        ious = [C.iou(m, gt) for m in cands]
        shown = cands[0]
        row = {"prompt": prompt, "iou": ious[0], "bf": C.boundary_f1(shown, gt),
               "oracle_iou": max(ious), "n_cand": len(cands), "score": res.score,
               "pred_area": int(shown.sum()), "embed_ms": embed_ms, "decode_ms": ms,
               "status": "ok"}
        if prompt in C.GATED_PROMPTS:
            single = svc.predict(SamRequest(image_crop=crop, points=list(pts), box=box,
                                            multimask=False))
            sm = C.paste_mask(single.mask, win)
            row["single_iou"] = C.iou(sm, gt)
            row["single_bf"] = C.boundary_f1(sm, gt)
        if prompt == "point_box":
            shown_pb = (res.candidates[0], shown)
        rows.append(row)

    # refine: one corrective click on point_box's shown mask (the app's refine path)
    crop_mask, native = shown_pb
    click = C.corrective_click(native, gt)
    if click is None:
        rows.append({"prompt": "refine", "iou": 1.0, "bf": C.boundary_f1(native, gt),
                     "oracle_iou": 1.0, "n_cand": 1, "score": float("nan"),
                     "pred_area": int(native.sum()), "embed_ms": embed_ms, "decode_ms": 0.0,
                     "status": "no_error"})
    else:
        cx, cy = C.to_crop((win.x0 + click[0] + 0.5, win.y0 + click[1] + 0.5), win)
        req = SamRequest(image_crop=crop, points=[pt, (cx, cy, click[2])], box=jbox,
                         mask_input=crop_mask, multimask=False)
        res, ms = _sync_ms(torch, svc.predict, req)
        m = C.paste_mask(res.mask, win)
        v = C.iou(m, gt)
        rows.append({"prompt": "refine", "iou": v, "bf": C.boundary_f1(m, gt), "oracle_iou": v,
                     "n_cand": 1, "score": res.score, "pred_area": int(m.sum()),
                     "embed_ms": embed_ms, "decode_ms": ms, "status": "ok"})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True, help="name=checkpoint")
    ap.add_argument("--desktops", type=int, nargs="+", default=list(G.DESKTOPS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    import torch

    env.limit_threads()
    free = env.gpu_free_gb()
    if free < env.MIN_FREE_GB + 4:
        raise SystemExit(f"only {free:.1f} GiB free on the GPU; need {env.MIN_FREE_GB + 4:.0f}")
    from tda.models.sam_service import SamService

    models = []
    for spec in args.model:
        name, path = spec.split("=", 1)
        t0 = time.perf_counter()
        svc = SamService(checkpoint=path, config=env.SAM2_CONFIG, device="cuda")
        print(f"[eval] {name}: {path} loaded in {time.perf_counter() - t0:.1f} s", flush=True)
        models.append((name, svc))
    torch.cuda.reset_peak_memory_stats()

    recs, images = G.load()
    recs = [r for r in recs if r.desktop in set(args.desktops)]
    if args.limit:
        rng = np.random.default_rng(0)
        recs = [recs[i] for i in sorted(rng.choice(len(recs), args.limit, replace=False))]
    by_frame = defaultdict(list)
    for r in recs:
        by_frame[r.frame].append(r)
    frames = G.load_frames(images, set(by_frame))
    print(f"[eval] {len(recs)} masks on {len(by_frame)} frames, desktops {args.desktops}", flush=True)

    out = env.OUT / "eval" / f"{args.out}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for fi, key in enumerate(sorted(by_frame), 1):
            if env.stop_requested():
                print("[eval] STOP file present: stopping", flush=True)
                break
            frame = frames[key]
            for rec in by_frame[key]:
                win = C.crop_window(rec.bbox, rec.frame_hw)
                crop = C.crop_image(frame, win)
                gtm = rec.mask_in(win.rect)
                base = {"desktop": rec.desktop, "step": rec.step, "instance": rec.instance,
                        "cls": rec.cls, "area": rec.area, "size_bucket": G.size_bucket(rec.area),
                        "visibility": rec.visibility, "occlusion": round(rec.occlusion, 4),
                        "drawn": int(rec.drawn), "groups": "|".join(rec.group),
                        "win_side": win.side, "oversize": int(win.oversize)}
                for name, svc in models:
                    _, embed_ms = _sync_ms(torch, svc.set_image, crop)
                    for row in eval_mask(torch, svc, rec, crop, win, gtm, embed_ms):
                        full = {**base, "model": name, **row}
                        for k in ("iou", "bf", "oracle_iou", "score", "single_iou", "single_bf",
                                  "embed_ms", "decode_ms"):
                            if isinstance(full.get(k), float):
                                full[k] = round(full[k], 5)
                        w.writerow(full)
                n += 1
            if fi % 10 == 0:
                el = time.perf_counter() - t0
                print(f"[eval] {fi}/{len(by_frame)} frames, {n} masks, {el:.0f} s, "
                      f"ETA {el / fi * (len(by_frame) - fi):.0f} s", flush=True)
    print(f"[eval] {n} masks in {time.perf_counter() - t0:.0f} s -> {out}")
    print(f"[eval] peak VRAM allocated {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB, "
          f"reserved {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB "
          f"({len(models)} model(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
