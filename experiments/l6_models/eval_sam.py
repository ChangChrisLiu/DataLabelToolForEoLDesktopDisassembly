"""L5's SAM harness on L6's held-out masks: stock, the shipped tuned head's checkpoint, the L6 fold.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.eval_sam --out heldout
    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.eval_sam --out v3_all \\
        --model stock=... --model v3=... --desktops 19 27

Every mask of the chosen desktops (``gt/gt_sam.pkl``), every model the app's
own ``SamService`` (stock head only, ``heads=None``: each checkpoint is one
model), sam_compare's window, one embedding per model and mask, then L5's
``evaluate.eval_mask`` -- the prompts box / point / point_box / box_loose /
refine, unchanged.  Models run in one process, alternating per mask, so their
timings compare.  Rows go to ``sam/eval/<out>.csv`` (L5's columns) as they are
made; a run that finds the CSV resumes after the last complete mask.

The GPU rule: the app is checked before the models load and every 10 frames;
when it is open the script exits with code 3 (the rows so far are kept).
"""
from __future__ import annotations

from experiments.l6_models import env  # noqa: F401  (side effects first)

import argparse
import csv
import sys
import time
from collections import defaultdict

import numpy as np

from experiments.l5_sam import common as C
from experiments.l5_sam import gt as G5
from experiments.l5_sam.evaluate import FIELDS, _sync_ms, eval_mask
from experiments.l6_models.export_gt import HELD_OUT
from experiments.l6_models.train_sam import SAM_OUT, checkpoint_path, load_recs

EXIT_INTERRUPTED = 3
DEFAULT_MODELS = {
    "stock": env.WEIGHTS / "sam2.1_hiera_large.pt",
    "f5_v2": env.WEIGHTS / "sam2.1_hiera_large_tda_f5_v2.pt",
    "F": checkpoint_path("F"),
}


def done_keys(path) -> dict:
    """``(desktop, step, instance) -> {(model, prompt)}`` of the rows already written."""
    if not path.exists():
        return {}
    rows = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[(int(r["desktop"]), int(r["step"]), r["instance"])].add((r["model"], r["prompt"]))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", default=None, help="name=checkpoint")
    ap.add_argument("--desktops", type=int, nargs="+", default=list(HELD_OUT))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    specs = ([tuple(s.split("=", 1)) for s in args.model] if args.model
             else [(k, str(v)) for k, v in DEFAULT_MODELS.items()])
    if env.app_running():
        return EXIT_INTERRUPTED

    import torch

    env.limit_threads()
    free = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
    if free < 4 + 1.5 * len(specs):
        print(f"[l6] only {free:.1f} GiB free: SAM evaluation waits", flush=True)
        return EXIT_INTERRUPTED
    from tda.models.sam_service import SamService

    out = SAM_OUT / "eval" / f"{args.out}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    have = done_keys(out)
    expect = {(m, p) for m, _ in specs for p in C.PROMPTS}
    recs, images = load_recs()
    recs = [r for r in recs if r.desktop in set(args.desktops)]
    todo = [r for r in recs if not expect <= have.get((r.desktop, r.step, r.instance), set())]
    print(f"[l6] {len(recs)} masks of D{args.desktops}; {len(recs) - len(todo)} already done",
          flush=True)
    if not todo:
        return 0
    models = []
    for name, path in specs:
        t0 = time.perf_counter()
        svc = SamService(checkpoint=path, config=env.l5env.SAM2_CONFIG, device="cuda")
        print(f"[l6] {name}: {path} loaded in {time.perf_counter() - t0:.1f} s", flush=True)
        models.append((name, svc))
    torch.cuda.reset_peak_memory_stats()
    by_frame = defaultdict(list)
    for r in todo:
        by_frame[r.frame].append(r)
    new_file = not out.exists()
    t0 = time.perf_counter()
    n = 0
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for fi, key in enumerate(sorted(by_frame), 1):
            if fi % 10 == 1 and fi > 1 and env.app_running():
                print("[l6] the app is open: SAM evaluation pauses", flush=True)
                return EXIT_INTERRUPTED
            frame = G5.load_frames(images, {key})[key]
            for rec in by_frame[key]:
                win = C.crop_window(rec.bbox, rec.frame_hw)
                crop = C.crop_image(frame, win)
                gtm = rec.mask_in(win.rect)
                base = {"desktop": rec.desktop, "step": rec.step, "instance": rec.instance,
                        "cls": rec.cls, "area": rec.area, "size_bucket": G5.size_bucket(rec.area),
                        "visibility": rec.visibility, "occlusion": round(rec.occlusion, 4),
                        "drawn": int(rec.drawn), "groups": "|".join(rec.group),
                        "win_side": win.side, "oversize": int(win.oversize)}
                rows = []
                for name, svc in models:
                    _, embed_ms = _sync_ms(torch, svc.set_image, crop)
                    for row in eval_mask(torch, svc, rec, crop, win, gtm, embed_ms):
                        full = {**base, "model": name, **row}
                        for k in ("iou", "bf", "oracle_iou", "score", "single_iou", "single_bf",
                                  "embed_ms", "decode_ms"):
                            if isinstance(full.get(k), float):
                                full[k] = round(full[k], 5)
                        rows.append(full)
                w.writerows(rows)          # a mask's rows together: resume never splits one
                n += 1
            fh.flush()
            if fi % 10 == 0:
                el = time.perf_counter() - t0
                print(f"[l6] {fi}/{len(by_frame)} frames, {n} masks, {el:.0f} s, "
                      f"ETA {el / fi * (len(by_frame) - fi):.0f} s", flush=True)
    print(f"[l6] {n} masks in {time.perf_counter() - t0:.0f} s -> {out}; peak VRAM "
          f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB ({len(models)} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
