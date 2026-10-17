"""Copy L4's ``all`` model to ``D:/DataSet/models/weights`` as v2, with its manifest.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.publish_production

L3's ``publish_production`` adapted: source ``<OUT>/rfdetr_small/runs/all/
last_ema.pth`` (trained by ``train_rfdetr --fold all`` from the production
checkpoint, the fold recipe unchanged); target
``tda_det_rfdetr_small_all_v2.pt`` + ``.json``: the training data (L2's LS
tiles minus D13's scan tiles + the verified scan tiles of D2/3/4/5/13) with
both tile hashes, the init checkpoint and its hash, the recipe, the tiling /
merge / guess rule, the held-out evaluation summary (``decision.json``), the
source commit and date.  The weight is never committed to git.  Only run when
``decision.json`` says the gate passed (``--force`` overrides, for the record).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l4_detector import env  # noqa: E402
from experiments.l2_detector import data as D  # noqa: E402
from experiments.l2_detector import tiles as T  # noqa: E402
from experiments.l3_detector.publish_production import git, sha256  # noqa: E402

NAME = "tda_det_rfdetr_small_all_v2"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    decision = json.loads((env.OUT / "decision.json").read_text(encoding="utf-8"))
    if not decision["gate"]["pass"] and not args.force:
        raise SystemExit("the gate did not pass: nothing published")
    run = env.OUT / "rfdetr_small" / "runs" / "all"
    src = run / "last_ema.pth"
    meta = json.loads((run / "l4_train.json").read_text(encoding="utf-8"))
    if meta.get("stopped_early"):
        raise SystemExit("the all run was stopped early: not a production model")
    ds = json.loads((env.OUT / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    fold = ds["folds"]["all"]
    prod = json.loads(env.PRODUCTION.with_suffix(".json").read_text(encoding="utf-8"))
    dst = env.WEIGHTS / f"{NAME}.pt"
    shutil.copyfile(src, dst)
    status = git("status", "--porcelain", "--", "experiments/l4_detector",
                 "experiments/l3_detector", "experiments/l2_detector")
    enabled = [c for c, v in decision["classes"].items() if v.get("enable")]
    manifest = {
        "model": "rfdetr_small",
        "file": str(dst),
        "sha256": sha256(dst),
        "bytes": dst.stat().st_size,
        "source_run": str(run),
        "init_weights": {"file": str(env.PRODUCTION), "sha256": prod["sha256"],
                         "what": "the v1 production model (L3, tda_det_rfdetr_small_all.pt), "
                                 "itself fine-tuned from rf-detr-small.pth (COCO)",
                         "v1_init": prod.get("init_weights")},
        "training": {
            "desktops": "LS drafts of every desktop-view with >= 20 screw+connector drafts "
                        "(L2's rule) minus D13 scan, + the verified scan frames of D2, D3, D4, "
                        "D5, D13 (no hold-out)",
            "tiles": fold["tiles"],
            "tiles_sha256": fold["sha256"],
            "ls_tiles": fold["ls_tiles"],
            "ls_rule": fold["ls_rule"],
            "ls_tile_source": str(env.L2OUT / "dataset" / "manifest.json"),
            "verified_tiles": fold["verified_tiles"],
            "verified_sha256": fold["verified_sha256"],
            "verified_source": f"{env.DB} (copy of annotations/tda.sqlite, 2026-10-16): "
                               "compiled visible masks of review_status='verified' scan frames",
            "verified_tile_manifest": str(env.OUT / "dataset" / "manifest.json"),
            "tiles_hash_rule": "sha256 over tiles sorted by name of (name, sha256(image), "
                               "sha256(label))",
            "recipe": meta.get("recipe"),
            "resolution": meta.get("resolution", T.TILE),
            "seconds": meta.get("seconds"),
        },
        "classes": list(D.DET_CLASSES),
        "inference": {
            "tile": T.TILE, "stride": T.STRIDE, "pad_value": T.PAD_VALUE,
            "work_scale": T.WORK_SCALE,
            "roi": "the pose segment's stored ROI (the verified tiles were cut at it; "
                   "L2's LS tiles at the drafts' union + 10 %)",
            "merge": "drop cut-off boxes contained >= 60 % in a whole one, class-wise NMS 0.5",
            "guess": f"classes {['screw'] + enabled}: conf >= 0.10, skip parts already drawn "
                     "(centre +-2 px or IoU >= 0.3), rank by native mean dE j->k in the box",
            "precision": "fp16",
        },
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": bool(status),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation": {"report": str(env.OUT / "compare.md"),
                       "decision": decision,
                       "note": "leave-one-desktop-out fold models (F13 from L3 hold13, F5 from "
                               "v1 production); this all-desktop model has no held-out score "
                               "of its own"},
        "licence": prod.get("licence", "Apache-2.0 (Roboflow RF-DETR)"),
    }
    (env.WEIGHTS / f"{NAME}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "evaluation"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
