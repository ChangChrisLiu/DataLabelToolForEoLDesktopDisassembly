"""Fine-tune RF-DETR small on L2's LS tiles + the verified scan tiles, one L4 fold.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.train_rfdetr --fold hold13

Recipe: L3's (``experiments/l3_detector/train_rfdetr.py``: RECIPE, resolution
640, rfdetr's own fine-tuning defaults, ``last_ema.pth`` is the model), fixed in
``decision_rule.txt`` before any result.  Only the initial weights differ:

* ``hold13`` starts from L3's hold13 RF-DETR small (never saw D13; the
  production model was trained on D13's LS drafts);
* ``hold5`` and ``all`` start from the production checkpoint
  (``tda_det_rfdetr_small_all.pt``, which never saw D5).

The head already has the 8 classes (+1), so it is loaded as is.

Sharing the machine with the annotator: BELOW_NORMAL priority (``env``), 4
CPU threads, <= 4 loader workers; after every epoch the peak CUDA memory of
this process and the card's total use (``nvidia-smi``) go to
``vram.jsonl``; when ``D:/DataSet/.cache/tmp/l4/STOP`` exists the run stops
after the current epoch (Lightning's ``should_stop``), rfdetr writes
``last_ema.pth`` as at a normal end, and ``l4_train.json`` says
``stopped_early``.  ``--resume <last.ckpt>`` continues such a run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l4_detector import env  # noqa: E402
from experiments.l3_detector.train_rfdetr import RECIPE, RESOLUTION  # noqa: E402

RF = env.OUT / "dataset_rfdetr"
INIT = {"hold13": env.L3_HOLD13, "hold5": env.PRODUCTION, "all": env.PRODUCTION}


def runs_dir() -> Path:
    return env.OUT / "rfdetr_small" / "runs"


def weights_of(fold: str) -> Path:
    return runs_dir() / fold / "last_ema.pth"


def gpu_total_mb() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=20).strip().splitlines()[0]
        used, total, util = (int(float(v)) for v in out.split(","))
        return {"card_used_mb": used, "card_total_mb": total, "card_util_pct": util}
    except Exception as exc:  # noqa: BLE001
        return {"nvidia_smi": str(exc)}


def _watch_callback(log_path: Path):
    import torch
    from pytorch_lightning import Callback

    class L4Watch(Callback):
        """VRAM after every epoch; the controller's stop file between epochs."""

        def __init__(self) -> None:
            self.stopped = False
            self.t0 = time.perf_counter()

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            rec = {"epoch": int(trainer.current_epoch), "t_s": round(time.perf_counter() - self.t0, 1),
                   "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20),
                   "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20)}
            rec.update(gpu_total_mb())
            if env.stop_requested():
                trainer.should_stop = True
                self.stopped = True
                rec["stop_file"] = True
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[l4] epoch {rec['epoch']}: {json.dumps(rec)}", flush=True)

    return L4Watch()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=sorted(INIT))
    ap.add_argument("--batch", type=int, default=RECIPE["batch_size"])
    ap.add_argument("--resume", default=None, help="a last.ckpt to continue a stopped run")
    args = ap.parse_args(argv)
    env.limit_threads()
    import rfdetr
    import rfdetr.training as rt

    rec = dict(RECIPE)
    rec["num_workers"] = min(rec["num_workers"], env.MAX_WORKERS)
    if args.batch != rec["batch_size"]:
        rec["grad_accum_steps"] = max(1, 16 // args.batch)   # effective batch stays 16
        rec["batch_size"] = args.batch
    if args.resume:
        rec["resume"] = args.resume
    init = INIT[args.fold]
    if not init.exists():
        raise SystemExit(f"missing init weights {init}")
    out = runs_dir() / args.fold
    out.mkdir(parents=True, exist_ok=True)
    watch = _watch_callback(out / "vram.jsonl")
    original = rt.build_trainer

    def build_trainer(*a, **kw):
        trainer = original(*a, **kw)
        trainer.callbacks.append(watch)
        return trainer

    rt.build_trainer = build_trainer
    model = rfdetr.RFDETRSmall(pretrain_weights=str(init), resolution=RESOLUTION)
    t0 = time.perf_counter()
    model.train(dataset_dir=str(RF / args.fold), dataset_file="yolo", output_dir=str(out),
                resolution=RESOLUTION, **rec)
    dt = time.perf_counter() - t0
    manifest = json.loads((env.OUT / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    (out / "l4_train.json").write_text(json.dumps(
        {"model": "rfdetr_small", "init": str(init), "fold": args.fold,
         "resolution": RESOLUTION, "recipe": rec, "seconds": round(dt, 1),
         "stopped_early": watch.stopped, "tiles": manifest["folds"][args.fold]}, indent=1),
        encoding="utf-8")
    print(f"[l4] rfdetr_small fold {args.fold} trained in {dt:.0f} s"
          f"{' (STOPPED EARLY)' if watch.stopped else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
