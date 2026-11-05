#!/usr/bin/env python
"""End-to-end smoke test of the kit: 2 train desktops, 1 test desktop, 1 epoch, small images.

    python smoke.py --data-root <hf download> --work <scratch folder> [--device cpu] [--skip-yolo] [--skip-maskrcnn]

Runs make_coco -> eval self-test -> (coco_to_yolo -> train_yolo -> predict_yolo -> eval) and
(train_maskrcnn -> predict_maskrcnn -> eval), and writes <work>/smoke_summary.md. The scores are meaningless
(one epoch on two desktops); the point is that every stage runs and writes its files.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(step, cmd, times, log):
    print(f"\n=== {step}: {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n=== {step}: {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        p = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    times[step] = round(time.time() - t0, 1)
    print(f"    {step}: exit {p.returncode}, {times[step]} s", flush=True)
    if p.returncode != 0:
        sys.exit(f"{step} failed (exit {p.returncode}); see {log}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--pack", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--train-desktops", default="1,2")
    ap.add_argument("--test-desktops", default="4")
    ap.add_argument("--device", default=None, help="e.g. cpu, 0 (YOLO) / cuda (Mask R-CNN); default: GPU if free")
    ap.add_argument("--yolo-model", default="yolo11n-seg.pt")
    ap.add_argument("--yolo-imgsz", type=int, default=320)
    ap.add_argument("--yolo-conf", type=float, default=1e-5,
                    help="smoke only: after 1 epoch (still in warm-up) YOLO scores stay below the usual 0.001")
    ap.add_argument("--maskrcnn-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-yolo", action="store_true")
    ap.add_argument("--skip-maskrcnn", action="store_true")
    args = ap.parse_args(argv)

    py = sys.executable
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    log = work / "smoke.log"
    times = {}
    pack = args.pack or str(Path(args.data_root) / "annotations" / "scanner")
    splits = args.splits or str(Path(args.data_root) / "splits.json")
    coco = work / "coco"
    run("make_coco", [py, HERE / "make_coco.py", "--pack", pack, "--splits", splits, "--out", coco,
                      "--train-desktops", args.train_desktops, "--val-n", "0", "--test-desktops", args.test_desktops],
        times, log)
    run("eval_selftest", [py, HERE / "eval_coco.py", "--gt", coco / "test.json", "--selftest",
                          "--out", work / "selftest"], times, log)
    results = {}
    if not args.skip_yolo:
        ydata, yrun = work / "yolo_data", work / "runs" / "yolo"
        dev = [] if args.device is None else ["--device", "cpu" if args.device == "cpu" else args.device.replace("cuda:", "").replace("cuda", "0")]
        run("coco_to_yolo", [py, HERE / "coco_to_yolo.py", "--coco", coco, "--out", ydata], times, log)
        run("train_yolo", [py, HERE / "train_yolo.py", "--data", ydata / "data.yaml", "--run", yrun,
                           "--model", args.yolo_model, "--weights-dir", work / "weights", "--imgsz", args.yolo_imgsz,
                           "--epochs", 1, "--batch", 4, "--workers", args.workers, *dev], times, log)
        w = yrun / "train" / "weights" / "best.pt"
        if not w.exists():
            w = yrun / "train" / "weights" / "last.pt"
        run("predict_yolo", [py, HERE / "predict_yolo.py", "--gt", coco / "test.json", "--weights", w,
                             "--cmap", ydata / "category_map.json", "--out", yrun / "preds_test.json",
                             "--imgsz", args.yolo_imgsz, "--conf", args.yolo_conf, *dev], times, log)
        run("eval_yolo", [py, HERE / "eval_coco.py", "--gt", coco / "test.json", "--pred", yrun / "preds_test.json",
                          "--out", yrun / "eval_test", "--title", "Smoke: YOLO seg, 1 epoch"], times, log)
        results["yolo"] = json.loads((yrun / "eval_test" / "metrics.json").read_text(encoding="utf-8"))
    if not args.skip_maskrcnn:
        mrun = work / "runs" / "maskrcnn"
        dev = [] if args.device is None else ["--device", "cpu" if args.device == "cpu" else
                                               (args.device if args.device.startswith("cuda") else f"cuda:{args.device}")]
        run("train_maskrcnn", [py, HERE / "train_maskrcnn.py", "--train", coco / "train.json", "--run", mrun,
                               "--size", args.maskrcnn_size, "--epochs", 1, "--batch", 2, "--workers", args.workers,
                               "--log-every", 10, *dev], times, log)
        run("predict_maskrcnn", [py, HERE / "predict_maskrcnn.py", "--gt", coco / "test.json", "--ckpt",
                                 mrun / "best.pt", "--out", mrun / "preds_test.json", *dev], times, log)
        run("eval_maskrcnn", [py, HERE / "eval_coco.py", "--gt", coco / "test.json", "--pred",
                              mrun / "preds_test.json", "--out", mrun / "eval_test",
                              "--title", "Smoke: Mask R-CNN, 1 epoch"], times, log)
        results["maskrcnn"] = json.loads((mrun / "eval_test" / "metrics.json").read_text(encoding="utf-8"))

    lines = ["# Level-1 smoke test", "",
             f"train desktops {args.train_desktops}, test desktops {args.test_desktops}, 1 epoch; "
             "scores are meaningless, only that every stage ran.", "",
             "| step | seconds |", "|---|---|"] + [f"| {k} | {v} |" for k, v in times.items()]
    lines += ["", "| baseline | bbox AP | bbox AP50 | segm AP | segm AP50 |", "|---|---|---|---|---|"]
    for name, res in results.items():
        b, s = res["metrics"].get("bbox"), res["metrics"].get("segm")
        f = lambda m, k: "n/a" if not m else f"{100 * m['summary'][k]:.2f}"  # noqa: E731
        lines.append(f"| {name} | {f(b, 'AP')} | {f(b, 'AP50')} | {f(s, 'AP')} | {f(s, 'AP50')} |")
    (work / "smoke_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump({"seconds": times, "results": {k: {t: (m["summary"] if m else None) for t, m in v["metrics"].items()}
                                             for k, v in results.items()}},
              open(work / "smoke_summary.json", "w", encoding="utf-8"), indent=1)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
