"""Write L6's candidates to the weights directory, with manifests the app's loaders accept.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.publish det
    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.publish sam [--dry-run]

``det``: ``runs/v3/last_ema.pth`` -> ``tda_det_rfdetr_small_all_v3.pt`` + ``.json``
(L4's manifest layout: file size and SHA-256, the 8 head classes in the
model's order -- ``tda.models.detector.check_manifest`` compares both --, the
init checkpoint, the data digest (tiles hash, the COCO export's and the
ground truth's SHA-256), the recipe, the shipped ``conf`` and the guess
classes the report recommends, gate D's verdict and numbers).

``sam``: ``sam2.1_hiera_large_tda_l6_v3.pt`` -> the head delta
``sam2.1_tda_head_v3.pt`` + ``.json``, by L5's ``export_head`` checks (the
same tensor names and shapes as stock, every differing tensor inside the
prompt encoder / mask decoder, the image encoder byte-identical), read back
the way the app reads it (``tda.models.sam_heads.read_head``) and compared
tensor by tensor; the manifest carries gate S's verdict and numbers.

Nothing in ``configs/`` changes: the controller ships.
"""
from __future__ import annotations

from experiments.l6_models import env  # noqa: F401  (side effects first)

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from experiments.l5_sam.export_head import compare, same_bytes, sha256_file
from experiments.l6_models.export_gt import COCO, GT_DET, GT_SAM, MANIFEST as GT_MANIFEST

DET_NAME = "tda_det_rfdetr_small_all_v3"
#: The controller's round-3 ruling (REPORT.md, round 2 / 3): screws at 0.15, every
#: other class at ``conf``; read by ``tda.models.detector.check_manifest_floors``.
CONF_BY_CLASS = {"screw": 0.15}
HEAD_NAME = "sam2.1_tda_head_v3"
SAM_FULL = env.WEIGHTS / "sam2.1_hiera_large_tda_l6_v3.pt"
STOCK = env.WEIGHTS / "sam2.1_hiera_large.pt"
SHIPPED_V2 = env.WEIGHTS / "tda_det_rfdetr_small_all_v2.pt"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(env.REPO), text=True).strip()


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _round2_summary() -> dict:
    """F at the shipped screw floor (``round2_floor.py``): rank-1 and screw recall / precision."""
    r = _json(env.ROOT / "eval" / "round2_floor.json") or {}
    f = (r.get("floors") or {}).get(f"{CONF_BY_CLASS['screw']:.2f}")
    if not f:
        return {}
    return {"floor": CONF_BY_CLASS["screw"], "fold_model": "F (held out D19, D27)",
            "rank1_pooled": f["rank1_pooled"],
            "screw": {k: {"recall": v["recall"], "precision": v["precision"]}
                      for k, v in f["counts"].items() if k.endswith(":screw")},
            "per_frame": f["per_frame"],
            "comparators_rank1": r.get("comparator", {}).get("rank1")}


def publish_det() -> int:
    run = env.ROOT / "runs" / "v3"
    meta = _json(run / "l6_train.json")
    if not meta:
        raise SystemExit("v3 has not finished training")
    src = run / "last_ema.pth"
    dst = env.WEIGHTS / f"{DET_NAME}.pt"
    shutil.copyfile(src, dst)
    gate = _json(env.ROOT / "eval" / "gate_det.json") or {}
    ds = _json(env.ROOT / "dataset" / "manifest.json")
    gtm = _json(GT_MANIFEST)
    v2 = _json(SHIPPED_V2.with_suffix(".json"))
    extra = [c for c, v in (gate.get("extra_classes") or {}).items() if v.get("arm")]
    guess = ["screw", "connector", "ram_module", "psu_latch", "cpu"] + extra
    manifest = {
        "model": "rfdetr_small",
        "file": str(dst),
        "sha256": sha256_file(dst),
        "bytes": dst.stat().st_size,
        "source_run": str(run),
        "published_by": "experiments/l6_models/publish.py (a candidate; configs/detector.yaml "
                        "is the controller's to change)",
        "init_weights": {"file": str(SHIPPED_V2), "sha256": (v2 or {}).get("sha256"),
                         "what": "the shipped v2 (L4), itself v1 (L3, COCO + L2's LS tiles) "
                                 "fine-tuned on L4's all fold"},
        "training": {
            "desktops": "L2's LS tiles minus the LS scan tiles of D8/D10/D13/D19 (their verified "
                        "frames replace the drafts) + the annotator's verified scan frames of all "
                        "20 verified desktops (D2-D17, D19-D21 Dell; D27 HP ProDesk 400 G1 SFF): "
                        "boxes of the compiled visible masks, in_chassis, visible / "
                        "occluded_partial / visible_tiny / too_small; each frame once",
            "frames": ds["frames"], "ls_tiles": ds["sets"]["v3"]["ls_tiles"],
            "verified_tiles": ds["sets"]["v3"]["verified_tiles"],
            "tiles": ds["sets"]["v3"]["tiles"],
            "tiles_sha256": ds["sets"]["v3"]["sha256"],
            "verified_tiles_sha256": ds["verified_sha256"],
            "tiles_hash_rule": ds["tiles_hash_rule"],
            "data_digest": {"coco_export": str(COCO), "coco_sha256": gtm["coco_sha256"],
                            "coco_command": " ".join((gtm.get("export") or {}).get("command", [])[2:]),
                            "ground_truth": str(GT_DET), "ground_truth_sha256": gtm["gt_det_sha256"],
                            "db_snapshot": str(env.LIVE_COPY) + " (backup-API copy of "
                                           "annotations/tda.sqlite, 2026-10-24 21:33)"},
            "tile_manifest": str(env.ROOT / "dataset" / "manifest.json"),
            "recipe": meta["recipe"], "resolution": meta["resolution"],
            "seconds": meta["seconds"],
        },
        "classes": ["screw", "connector", "ram_latch", "psu_latch", "cpu_socket_lever",
                    "drive_latch", "ram_module", "cpu"],
        "conf_by_class": dict(CONF_BY_CLASS),
        "inference": {
            "tile": 640, "stride": 512, "pad_value": 114,
            "work_scale": {"scan": 1.0, "oak1": 1.0, "oak2": 1.0, "rs": 2.0},
            "roi": "the pose segment's stored ROI (D27 was tiled at the app's proposed ROI: it "
                   "has none stored)",
            "merge": "drop cut-off boxes contained >= 60 % in a whole one, class-wise NMS 0.5",
            "conf": 0.05, "merge_floor": 0.05, "conf_by_class": dict(CONF_BY_CLASS),
            "conf_by_class_note": "applied where detections are read (the guess, the U28 "
                                  "chips); the cache keeps every box >= merge_floor",
            "guess_classes_recommended": guess,
            "precision": "fp16",
        },
        "evaluation": {
            "report": "experiments/l6_models/REPORT.md",
            "gate": "decision_rule.txt gate D, fold F (v2h -> 18 desktops) held out on D19 and "
                    "D27; this all-desktop model has no held-out score of its own",
            "verdict": gate.get("verdict"),
            "D1": (gate.get("gate") or {}).get("D1"), "D2": (gate.get("gate") or {}).get("D2"),
            "D3_pass": ((gate.get("gate") or {}).get("D3") or {}).get("pass"),
            "extra_classes": gate.get("extra_classes"),
            "round2_screw_floor": _round2_summary(),
            "ruling": "controller, round 3: shipped with conf_by_class {screw: 0.15} over "
                      "gate D (pooled screw rank-1 14/15 = the comparators', all 48 removal "
                      "events 32 vs 22, connectors 11 vs 4, ram_module 4 vs 1)",
        },
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain", "--", "experiments/l6_models")),
        "date": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "licence": "Apache-2.0 (Roboflow RF-DETR)",
    }
    out = dst.with_suffix(".json")
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[l6] wrote {dst} ({dst.stat().st_size} bytes) and {out.name}")
    return 0


def publish_sam(dry_run: bool = False) -> int:
    import torch

    from tda.models.sam_heads import FINGERPRINT_KEYS, HEAD_FORMAT, HEAD_MODULES, fingerprint, \
        read_head

    if not SAM_FULL.exists():
        raise SystemExit(f"{SAM_FULL} is not there")
    stock = torch.load(str(STOCK), map_location="cpu", weights_only=True, mmap=True)["model"]
    tuned = torch.load(str(SAM_FULL), map_location="cpu", weights_only=True, mmap=True)["model"]
    found = compare(stock, tuned)
    differ = found["differ"]
    print(f"[l6] {len(differ)} tensors differ ({found['params'] / 1e6:.3f} M parameters: "
          f"{found['per_module']}); unchanged in the head modules: {found['unchanged_in_modules']}")
    if found["outside"]:
        raise SystemExit(f"tensors outside {HEAD_MODULES} differ: {found['outside'][:10]}")
    if found["encoder_differ"]:
        raise SystemExit("the image encoder is not byte-identical")
    keys = list(FINGERPRINT_KEYS) + found["unchanged_in_modules"]
    base_print = fingerprint(stock, keys)
    if dry_run:
        return 0
    out = env.WEIGHTS / f"{HEAD_NAME}.pt"
    stock_sha, source_sha = sha256_file(STOCK), sha256_file(SAM_FULL)
    blob = {"format": HEAD_FORMAT, "modules": list(HEAD_MODULES),
            "base": {"file": STOCK.name, "sha256": stock_sha, "fingerprint": base_print},
            "source": {"file": SAM_FULL.name, "sha256": source_sha},
            "state_dict": {k: tuned[k].detach().clone().contiguous() for k in differ}}
    partial = out.with_name(out.name + ".partial")
    torch.save(blob, str(partial))
    os.replace(partial, out)
    back = read_head(out)
    assert sorted(back["state_dict"]) == differ
    for k in differ:
        assert same_bytes(back["state_dict"][k], tuned[k]), k
    assert back["base"]["fingerprint"] == base_print
    gate = _json(env.ROOT / "sam" / "tables" / "heldout_gate.json") or {}
    train = _json(env.ROOT / "sam" / "train" / "v3" / "manifest.json") or {}
    gtm = _json(GT_MANIFEST) or {}
    manifest = {
        "format": HEAD_FORMAT,
        "what": "SAM 2.1 Hiera-L prompt encoder + mask decoder fine-tuned on the annotator's "
                "verified scan masks of all 20 verified desktops (L6, L5's recipe v2), as a "
                "delta on the stock checkpoint",
        "head": {"file": out.name, "sha256": sha256_file(out), "bytes": out.stat().st_size},
        "source": {"file": SAM_FULL.name, "sha256": source_sha, "bytes": SAM_FULL.stat().st_size},
        "base": {"file": STOCK.name, "sha256": stock_sha, "bytes": STOCK.stat().st_size,
                 "fingerprint": base_print},
        "sam2_config": env.l5env.SAM2_CONFIG,
        "modules": list(HEAD_MODULES),
        "tensors": {"count": len(differ), "params": found["params"],
                    "per_module": found["per_module"], "names": differ},
        "unchanged_in_modules": found["unchanged_in_modules"],
        "identical": {"image_encoder_tensors": len(found["encoder"]),
                      "other_tensors": len(found["others"]), "checked": "byte for byte"},
        "training": {"experiment": "L6 (experiments/l6_models/REPORT.md)",
                     "recipe": "L5 v2 (half of the box prompts pushed out by up to 25 %), "
                               "resumable 250-step chunks",
                     "trained_on_desktops": train.get("desktops"), "masks": train.get("masks"),
                     "held_out": None, "train_manifest": train,
                     "data_digest": {"coco_sha256": gtm.get("coco_sha256"),
                                     "ground_truth": str(GT_SAM)}},
        "evaluation": {"gate": "decision_rule.txt gate S, fold S_F (18 desktops) held out on D19 "
                               "and D27, routed as configs/sam.yaml; this all-desktop head has no "
                               "held-out score of its own",
                       "verdict": gate.get("verdict"), "S1": (gate.get("gate") or {}).get("S1"),
                       "S2": (gate.get("gate") or {}).get("S2"),
                       "S3": (gate.get("gate") or {}).get("S3")},
        "routing": "unchanged: configs/sam.yaml route + stock_point_classes",
        "exported_by": "experiments/l6_models/publish.py",
        "exported_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"[l6] wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {out.with_suffix('.json').name}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=("det", "sam"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return publish_det() if args.what == "det" else publish_sam(args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
