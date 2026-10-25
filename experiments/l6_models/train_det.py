"""Fine-tune RF-DETR small on one L6 training set, one epoch at a time when it has to.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.train_det --set v1h
    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.train_det --set v1h --resume

Recipe: L3/L4's (``experiments/l3_detector/train_rfdetr.py`` RECIPE, resolution
640, ``last_ema.pth`` is the model), fixed in ``decision_rule.txt``.  Initial
weights per set: ``v1h`` the COCO checkpoint, ``v2h`` v1h's result, ``F``
v2h's result, ``v3`` the shipped v2.

**The GPU rule.**  The script refuses to start while the annotator's app is
open (``env.wait_for_gpu`` blocks first).  After every epoch a callback checks
the app (and the STOP file); if it is open, the run stops at that epoch
boundary -- rfdetr has just written ``last.ckpt`` (every epoch, the ModelCheckpoint
``filename="last"``) -- and the script exits with code 3.  ``--resume``
continues from that ``last.ckpt`` (Lightning restores the weights, the
optimizer, the schedule, the EMA callback's average and the epoch counter).
``run_queue.py`` does the waiting and the resuming.  When the 30th epoch is
done the script writes ``l6_train.json`` and copies ``last_ema.pth`` to
``D:/DataSet/models/weights/tda_det_rfdetr_small_l6_<set>.pth`` (fold sets).
"""
from __future__ import annotations

from experiments.l6_models import env  # noqa: F401  (side effects first)

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiments.l3_detector.train_rfdetr import RECIPE, RESOLUTION

RF = env.ROOT / "dataset_rfdetr"
RUNS = env.ROOT / "runs"
SHIPPED_V2 = env.WEIGHTS / "tda_det_rfdetr_small_all_v2.pt"
INIT = {"v1h": env.WEIGHTS / "rf-detr-small.pth",
        "v2h": RUNS / "v1h" / "last_ema.pth",
        "F": RUNS / "v2h" / "last_ema.pth",
        "v3": SHIPPED_V2}
EXIT_INTERRUPTED = 3


def weights_of(name: str) -> Path:
    return RUNS / name / "last_ema.pth"


def published(name: str) -> Path:
    return env.WEIGHTS / f"tda_det_rfdetr_small_l6_{name}.pth"


def done(name: str) -> bool:
    return (RUNS / name / "l6_train.json").exists()


def gpu_total_mb() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=20).strip().splitlines()[0]
        used, total, util = (int(float(v)) for v in out.split(","))
        return {"card_used_mb": used, "card_total_mb": total, "card_util_pct": util}
    except Exception as exc:  # noqa: BLE001
        return {"nvidia_smi": str(exc)}


def _watch(log_path: Path, epochs: int):
    import torch
    from pytorch_lightning import Callback

    class L6Watch(Callback):
        """VRAM after every epoch; stop at the epoch boundary when the app is open."""

        def __init__(self) -> None:
            self.interrupted = False
            self.last_epoch = -1
            self.t0 = time.perf_counter()

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            self.last_epoch = int(trainer.current_epoch)
            rec = {"epoch": self.last_epoch, "t_s": round(time.perf_counter() - self.t0, 1),
                   "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20),
                   "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20)}
            rec.update(gpu_total_mb())
            if self.last_epoch < epochs - 1:
                app = env.app_processes()
                if app or env.stop_requested():
                    trainer.should_stop = True
                    self.interrupted = True
                    rec["interrupted"] = "app" if app else "STOP"
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[l6] epoch {rec['epoch']}: {json.dumps(rec)}", flush=True)

    return L6Watch()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="name", required=True, choices=sorted(INIT))
    ap.add_argument("--resume", action="store_true", help="continue from runs/<set>/last.ckpt")
    args = ap.parse_args(argv)
    if done(args.name):
        print(f"[l6] {args.name} is already trained")
        return 0
    env.wait_for_gpu(f"train {args.name}")
    if env.stop_requested():
        return EXIT_INTERRUPTED
    env.limit_threads()
    import rfdetr
    import rfdetr.training as rt

    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(RECIPE)
    rec["num_workers"] = min(rec["num_workers"], env.MAX_WORKERS)
    ckpt = out / "last.ckpt"
    if args.resume and not ckpt.exists():
        raise SystemExit(f"nothing to resume: {ckpt} is missing")
    if ckpt.exists():               # an interrupted run is always continued, never restarted
        rec["resume"] = str(ckpt)
        args.resume = True
    init = INIT[args.name]
    if not init.exists():
        raise SystemExit(f"missing init weights {init}")
    watch = _watch(out / "vram.jsonl", rec["epochs"])
    original = rt.build_trainer

    def build_trainer(*a, **kw):
        trainer = original(*a, **kw)
        trainer.callbacks.append(watch)
        return trainer

    rt.build_trainer = build_trainer
    chunks = out / "chunks.jsonl"
    t0 = time.perf_counter()
    with chunks.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "start", "resume": bool(args.resume),
                             "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    model = rfdetr.RFDETRSmall(pretrain_weights=str(init), resolution=RESOLUTION)
    model.train(dataset_dir=str(RF / args.name), dataset_file="yolo", output_dir=str(out),
                resolution=RESOLUTION, **rec)
    dt = time.perf_counter() - t0
    with chunks.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "end", "seconds": round(dt, 1),
                             "last_epoch": watch.last_epoch, "interrupted": watch.interrupted,
                             "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    if watch.interrupted or watch.last_epoch < rec["epochs"] - 1:
        print(f"[l6] {args.name}: stopped after epoch {watch.last_epoch} "
              f"({dt:.0f} s this chunk); resume with --resume", flush=True)
        return EXIT_INTERRUPTED
    total = 0.0
    for line in chunks.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("event") == "end":
            total += float(r["seconds"])
    manifest = json.loads((env.ROOT / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    info = {"model": "rfdetr_small", "set": args.name, "init": str(init),
            "resolution": RESOLUTION, "recipe": {k: v for k, v in rec.items() if k != "resume"},
            "seconds": round(total, 1), "data": manifest["sets"][args.name],
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.name != "v3":
        dst = published(args.name)
        shutil.copyfile(weights_of(args.name), dst)
        info["published"] = str(dst)
    (out / "l6_train.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"[l6] rfdetr_small {args.name} trained ({total:.0f} s of training)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
