"""Fine-tune SAM 2.1's prompt encoder + mask decoder on L6's verified masks, in resumable chunks.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.train_sam --fold F
    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.train_sam --fold v3

L5's recipe v2 (``experiments/l5_sam/train.py``: its Sampler, forward,
three-round loss, optimiser and schedule, imported unchanged, ``loose_frac``
0.5), on the masks of ``gt/gt_sam.pkl``: fold ``F`` = the 18 training desktops,
``v3`` = all 20.  4 epochs; an epoch = every training mask once as the primary
target (``len // 4`` steps).

What is new is the GPU rule (``decision_rule.txt``, OPERATIONS).  The run is
cut into chunks of :data:`env.SAM_CHUNK` steps; before each chunk the app is
checked, and when it is open (or STOP exists) the script saves its state and
exits with code 3, giving the GPU back; ``run_queue.py`` waits and starts it
again, and it resumes from ``state.pt``: the trainable modules' weights, the
AdamW state, the step, the running loss sums and the torch RNG.  To make a
resumed run the same run, the data order of epoch ``e`` comes from
``default_rng((SEED, e))`` and the crop sampler is re-seeded at the start of
chunk ``c`` from ``default_rng((SEED, 1000 + c))`` (L5 drew both from one
stream; the distribution is the same).

The GPU is shared with the detector's training: this process caps itself at
:data:`MEM_CAP_GB`, and refuses to start when that would leave less than
:data:`KEEP_FREE_GB` free for the annotator's app to start into.

Output: ``D:/DataSet/models/weights/sam2.1_hiera_large_tda_l6_<fold>.pt`` (a
full state dict, as L5's) and ``ROOT/sam/train/<fold>/{log.jsonl,
manifest.json}``.
"""
from __future__ import annotations

from experiments.l6_models import env  # noqa: F401  (side effects first)

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np

from experiments.l5_sam import gt as G5
from experiments.l5_sam import train as T5
from experiments.l6_models.export_gt import GT_SAM, HELD_OUT

SAM_OUT = env.ROOT / "sam"
EXIT_INTERRUPTED = 3
MEM_CAP_GB = 16.0
KEEP_FREE_GB = 5.0
RECIPE = "v2"


def fold_desktops(fold: str, recs) -> tuple[int, ...]:
    ds = sorted({r.desktop for r in recs})
    if fold == "F":
        return tuple(d for d in ds if d not in HELD_OUT)
    return tuple(ds)


def checkpoint_path(fold: str):
    return env.WEIGHTS / f"sam2.1_hiera_large_tda_l6_{fold}.pt"


def load_recs():
    with open(GT_SAM, "rb") as fh:
        blob = pickle.load(fh)
    return [G5.Rec(**r) for r in blob["recs"]], blob["images"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", required=True, choices=("F", "v3"))
    ap.add_argument("--epochs", type=float, default=T5.EPOCHS)
    args = ap.parse_args(argv)
    out = SAM_OUT / "train" / args.fold
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        print(f"[l6] SAM {args.fold} is already trained")
        return 0
    env.wait_for_gpu(f"SAM train {args.fold}")
    if env.stop_requested():
        return EXIT_INTERRUPTED

    import torch

    env.limit_threads()
    free = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    cap = min(MEM_CAP_GB, free - KEEP_FREE_GB)
    if cap < 12:
        print(f"[l6] only {free:.1f} GiB free: not starting SAM training (needs 12 + "
              f"{KEEP_FREE_GB})", flush=True)
        return EXIT_INTERRUPTED
    torch.cuda.set_per_process_memory_fraction(cap / total_gb)
    print(f"[l6] {free:.1f} GiB free; this process capped at {cap:.1f} GiB", flush=True)

    recs, images = load_recs()
    ds = fold_desktops(args.fold, recs)
    recs = [r for r in recs if r.desktop in ds]
    spe = len(recs) // T5.CROPS_PER_STEP
    n_steps = int(round(args.epochs * spe))
    model, trainable = T5.build_model(str(env.l5env.SAM2_CKPT))
    model.sam_mask_decoder.train()
    model.sam_prompt_encoder.train()
    opt = torch.optim.AdamW(trainable, lr=T5.LR, weight_decay=T5.WD)
    state_p = out / "state.pt"
    step = 0
    ep_sum: dict = defaultdict(float)
    ep_n: dict = defaultdict(int)
    train_s = 0.0
    chunks = 0
    if state_p.exists():
        st = torch.load(str(state_p), map_location="cpu", weights_only=False)
        model.sam_prompt_encoder.load_state_dict(st["prompt_encoder"])
        model.sam_mask_decoder.load_state_dict(st["mask_decoder"])
        opt.load_state_dict(st["opt"])
        step = int(st["step"])
        ep_sum.update(st["ep_sum"])
        ep_n.update(st["ep_n"])
        train_s = float(st["train_s"])
        chunks = int(st.get("chunks", 0))
        torch.set_rng_state(st["torch_rng"])
        torch.cuda.set_rng_state(st["cuda_rng"])
        print(f"[l6] resumed SAM {args.fold} at step {step}/{n_steps}", flush=True)
    else:
        random.seed(T5.SEED)
        np.random.seed(T5.SEED)
        torch.manual_seed(T5.SEED)
    frames = G5.load_frames(images, {r.frame for r in recs})
    sampler = T5.Sampler(recs, frames, T5.SEED, loose_frac=T5.RECIPES[RECIPE])
    print(f"[l6] SAM {args.fold}: desktops {ds}, {len(recs)} masks, {len(frames)} frames, "
          f"{spe} steps/epoch, {n_steps} steps", flush=True)

    def lr_at(s):
        if s < T5.WARMUP:
            return T5.LR * (s + 1) / T5.WARMUP
        p = (s - T5.WARMUP) / max(1, n_steps - T5.WARMUP)
        return T5.LR * (T5.LR_FLOOR + (1 - T5.LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * p)))

    orders: dict[int, np.ndarray] = {}

    def order_of(e: int) -> np.ndarray:
        if e not in orders:
            orders[e] = np.random.default_rng((T5.SEED, e)).permutation(len(recs))
        return orders[e]

    def chunk_batches(s0: int, s1: int):
        for s in range(s0, s1):
            e, i = divmod(s, spe)
            kind = T5.PROMPT_KINDS[s % len(T5.PROMPT_KINDS)]
            idx = order_of(e)[i * T5.CROPS_PER_STEP:(i + 1) * T5.CROPS_PER_STEP]
            yield [sampler.crop(int(j), kind) for j in idx]

    def save_state() -> None:
        tmp = state_p.with_name("state.pt.partial")
        torch.save({"prompt_encoder": model.sam_prompt_encoder.state_dict(),
                    "mask_decoder": model.sam_mask_decoder.state_dict(),
                    "opt": opt.state_dict(), "step": step, "ep_sum": dict(ep_sum),
                    "ep_n": dict(ep_n), "train_s": train_s, "chunks": chunks,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()},
                   str(tmp))
        os.replace(tmp, state_p)

    log = open(out / "log.jsonl", "a", encoding="utf-8")
    acc: dict = defaultdict(float)
    acc_n = 0
    while step < n_steps:
        app = env.app_processes()
        if app or env.stop_requested():
            print(f"[l6] {'the app is open' if app else 'STOP'}: SAM {args.fold} pauses at "
                  f"step {step}", flush=True)
            log.write(json.dumps({"event": "interrupted", "step": step,
                                  "why": "app" if app else "STOP",
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
            log.close()
            return EXIT_INTERRUPTED
        c = step // env.SAM_CHUNK
        s1 = min((c + 1) * env.SAM_CHUNK, n_steps)
        sampler.rng = np.random.default_rng((T5.SEED, 1000 + c))
        t0 = time.perf_counter()
        for batch in T5.prefetch(chunk_batches(step, s1)):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss, st = T5.train_step(torch, model, batch)
            if not torch.isfinite(loss):
                raise SystemExit(f"[l6] non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, T5.CLIP)
            opt.step()
            e = step // spe
            ep_sum[e] += float(loss.detach())
            ep_n[e] += 1
            step += 1
            for k, v in st.items():
                acc[k] += v
            acc["grad_norm"] += float(gn)
            acc_n += 1
            if step % 50 == 0 or step == n_steps:
                rec = {k: round(v / acc_n, 5) for k, v in acc.items()}
                rec.update(step=step, lr=lr_at(step), s=round(train_s + time.perf_counter() - t0, 1),
                           mem_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2))
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[l6] {step}/{n_steps} loss {rec['loss_r1']:.3f}/{rec['loss_r2']:.3f}/"
                      f"{rec['loss_r3']:.3f} r1 IoU {rec['r1_iou']:.3f} r3 IoU {rec['r3_iou']:.3f}"
                      f" | peak {rec['mem_gb']} GiB", flush=True)
                acc.clear()
                acc_n = 0
        train_s += time.perf_counter() - t0
        chunks += 1
        save_state()
    log.close()

    ckpt = checkpoint_path(args.fold)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tmp = ckpt.with_name(ckpt.name + ".partial")
    torch.save({"model": sd}, str(tmp))
    os.replace(tmp, ckpt)
    epoch_means = [ep_sum[e] / max(1, ep_n[e]) for e in sorted(ep_sum)]
    manifest = {
        "fold": args.fold, "desktops": ds, "masks": len(recs), "steps": step,
        "steps_per_epoch": spe, "epochs": args.epochs, "lr": T5.LR,
        "epoch_mean_loss": epoch_means, "train_s": round(train_s, 1), "chunks": chunks,
        "peak_alloc_gb_this_process": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "checkpoint": str(ckpt), "init": str(env.l5env.SAM2_CKPT), "config": env.l5env.SAM2_CONFIG,
        "gt": str(GT_SAM),
        "recipe": {"name": RECIPE, "loose_frac": T5.RECIPES[RECIPE], "loose_max": T5.LOOSE_MAX,
                   "wd": T5.WD, "warmup": T5.WARMUP, "lr_floor": T5.LR_FLOOR, "clip": T5.CLIP,
                   "crops_per_step": T5.CROPS_PER_STEP, "max_targets": T5.MAX_TARGETS_PER_CROP,
                   "rounds": T5.ROUNDS, "loss": [T5.W_FOCAL, T5.W_DICE, T5.W_IOU],
                   "ctx": T5.CTX_RANGE, "shift": T5.SHIFT, "seed": T5.SEED,
                   "chunk_steps": env.SAM_CHUNK,
                   "rng": "epoch order default_rng((seed, epoch)); sampler re-seeded "
                          "default_rng((seed, 1000 + chunk)) per chunk"},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[l6] saved {ckpt} after {step} steps ({train_s:.0f} s of training, {chunks} chunks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
