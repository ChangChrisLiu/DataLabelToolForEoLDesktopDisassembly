#!/usr/bin/env python
"""COCO evaluation (pycocotools) of bbox and / or segm predictions against a make_coco.py split.

    python eval_coco.py --gt coco/test.json --pred preds.json --out runs/x/eval [--iou-types bbox segm]
    python eval_coco.py --gt coco/test.json --selftest            # GT as predictions -> AP must be 1.0

Predictions: a COCO results list, `[{"image_id", "category_id", "score", "bbox": [x, y, w, h],
"segmentation": RLE}, ...]` (bbox is needed for bbox eval, segmentation for segm eval; one file may hold both).

Reported per IoU type: AP (IoU 0.50:0.05:0.95), AP50, AP75, APs / APm / APl, AR1 / AR10 / AR100,
ARs / ARm / ARl (maxDets 100, standard COCO area ranges), per-class AP and AP50, and AP / AP50 binned by the
ground truth's `occlusion_ratio` (see --occlusion-bins). Writes <out>/metrics.json and <out>/metrics.md.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l1common import dump_json, load_json  # noqa: E402

STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
DEFAULT_OCC_BINS = "0,0.01,0.3,0.6,1.0"


def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


def load_gt(gt_path):
    from pycocotools.coco import COCO
    data = load_json(gt_path)
    gt = COCO()
    gt.dataset = data
    _quiet(gt.createIndex)
    return gt


def prepare_dets(preds, iou_type):
    """Copy predictions for one IoU type: segm drops bbox so that area / bbox come from the mask."""
    out = []
    for p in preds:
        q = {"image_id": p["image_id"], "category_id": p["category_id"], "score": float(p["score"])}
        if iou_type == "bbox":
            if "bbox" not in p:
                continue
            q["bbox"] = [float(v) for v in p["bbox"]]
        else:
            seg = p.get("segmentation")
            if not seg:
                continue
            seg = dict(seg)
            if isinstance(seg.get("counts"), bytes):
                seg["counts"] = seg["counts"].decode("ascii")
            q["segmentation"] = seg
        out.append(q)
    return out


def _mean_valid(x):
    x = x[x > -1]
    return float(np.mean(x)) if x.size else -1.0


def run_cocoeval(gt, dets, iou_type, cat_ids=None):
    from pycocotools.cocoeval import COCOeval
    # loadRes mutates its input (adds area / id / bbox): give it copies
    dt, _ = _quiet(gt.loadRes, [dict(d) for d in dets]) if dets else (None, "")
    if dt is None:
        return None
    E = COCOeval(gt, dt, iou_type)
    if cat_ids is not None:
        E.params.catIds = cat_ids
    _quiet(E.evaluate)
    _quiet(E.accumulate)
    _, text = _quiet(E.summarize)
    return E, text


def per_class(E, gt):
    prec = E.eval["precision"]  # [T, R, K, A, M]
    t50 = int(np.argmin(np.abs(E.params.iouThrs - 0.5)))
    rows = {}
    for k, cid in enumerate(E.params.catIds):
        name = gt.cats[cid]["name"]
        n_gt = len(gt.getAnnIds(catIds=[cid], iscrowd=False))
        rows[name] = {
            "category_id": int(cid),
            "gt_instances": n_gt,
            "AP": _mean_valid(prec[:, :, k, 0, -1]),
            "AP50": _mean_valid(prec[t50, :, k, 0, -1]),
        }
    return rows


def make_bins(edges, fully_covered=False):
    """Edges -> [(lo, hi, closed_hi, label)]: half-open bins, the last one closed; optionally a final
    'fully covered' bin of GT with occlusion ratio exactly 1 (no visible pixel; amodal ground truth only)."""
    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        bins.append((lo, hi, last, f"[{lo:g}, {hi:g}{']' if last else ')'}"))
    if fully_covered:
        bins.append((1.0, 1.0, True, "fully covered (= 1)"))
    return bins


def occlusion_binned(gt, dets, iou_type, bins):
    """AP / AP50 with GT outside the occlusion bin ignored (as COCO does for area ranges).

    Implemented through COCO's area-range machinery: each GT's 'area' is replaced by its occlusion
    ratio (x 1e6) and the bin is the only area range; unmatched detections count as false positives in
    every bin (they carry no occlusion ratio)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    out = []
    if not any("occlusion_ratio" in a for a in gt.dataset["annotations"]):
        return out
    for lo, hi, closed, label in bins:
        ds = copy.copy(gt.dataset)
        anns = []
        n_in = 0
        for a in gt.dataset["annotations"]:
            b = dict(a)
            r = float(a.get("occlusion_ratio") or 0.0)
            b["area"] = r * 1e6
            if not a["iscrowd"] and lo <= r and (r < hi or (closed and r <= hi)):
                n_in += 1
            anns.append(b)
        ds["annotations"] = anns
        g = COCO()
        g.dataset = ds
        _quiet(g.createIndex)
        dt, _ = _quiet(g.loadRes, [dict(p) for p in dets])
        for a in dt.dataset["annotations"]:
            a["area"] = lo * 1e6  # inside the bin -> unmatched detections are false positives
        E = COCOeval(g, dt, iou_type)
        hi_v = hi * 1e6 if closed else hi * 1e6 - 1e-3
        E.params.areaRng = [[lo * 1e6, hi_v]]
        E.params.areaRngLbl = ["bin"]
        _quiet(E.evaluate)
        _quiet(E.accumulate)
        prec = E.eval["precision"]
        t50 = int(np.argmin(np.abs(E.params.iouThrs - 0.5)))
        out.append({"bin": label, "gt_instances": n_in,
                    "AP": _mean_valid(prec[:, :, :, 0, -1]), "AP50": _mean_valid(prec[t50, :, :, 0, -1])})
    return out


def evaluate(gt_path, preds, iou_types, occ_bins):
    gt = load_gt(gt_path)
    result = {"gt": str(gt_path), "images": len(gt.getImgIds()),
              "gt_instances": len(gt.getAnnIds(iscrowd=False)),
              "gt_ignore_regions": len(gt.getAnnIds(iscrowd=True)),
              "predictions": len(preds), "masks": gt.dataset.get("info", {}).get("masks"), "metrics": {}}
    valid_imgs = set(gt.getImgIds())
    dropped = sum(1 for p in preds if p["image_id"] not in valid_imgs)
    if dropped:
        print(f"warning: {dropped} predictions on images not in the GT were dropped", file=sys.stderr)
        preds = [p for p in preds if p["image_id"] in valid_imgs]
    for t in iou_types:
        dets = prepare_dets(preds, t)
        if not dets:  # pycocotools cannot load an empty result list; every AP / AR is 0 by definition
            print(f"warning: no predictions usable for {t}; all scores 0", file=sys.stderr)
            pc = {gt.cats[c]["name"]: {"category_id": int(c), "gt_instances": len(gt.getAnnIds(catIds=[c], iscrowd=False)),
                                       "AP": 0.0, "AP50": 0.0} for c in sorted(gt.cats)}
            for v in pc.values():
                if v["gt_instances"] == 0:
                    v["AP"] = v["AP50"] = -1.0
            result["metrics"][t] = {"summary": {n: 0.0 for n in STAT_NAMES}, "per_class": pc, "summary_text": "",
                                    "no_predictions": True}
            continue
        E, text = run_cocoeval(gt, dets, t)
        entry = {"summary": dict(zip(STAT_NAMES, [float(v) for v in E.stats])),
                 "per_class": per_class(E, gt), "summary_text": text}
        if occ_bins:
            fully = gt.dataset.get("info", {}).get("masks") == "amodal"
            entry["by_occlusion"] = occlusion_binned(gt, dets, t, make_bins(occ_bins, fully))
        result["metrics"][t] = entry
    return result


def fmt(v):
    return "n/a" if v is None or v < 0 else f"{100 * v:.1f}"


def to_markdown(res, title):
    lines = [f"# {title}", "",
             f"GT `{Path(res['gt']).name}` ({res.get('masks')} masks): {res['images']} images, "
             f"{res['gt_instances']} instances, {res['gt_ignore_regions']} ignore regions; "
             f"{res['predictions']} predictions. COCO AP x 100, IoU 0.50:0.05:0.95, maxDets 100.", ""]
    types = [t for t, m in res["metrics"].items() if m]
    lines.append("| IoU type | " + " | ".join(STAT_NAMES) + " |")
    lines.append("|---|" + "---|" * len(STAT_NAMES))
    for t in types:
        s = res["metrics"][t]["summary"]
        lines.append(f"| {t} | " + " | ".join(fmt(s[n]) for n in STAT_NAMES) + " |")
    lines += ["", "Per class (AP / AP50):", ""]
    lines.append("| class | GT | " + " | ".join(f"{t} AP | {t} AP50" for t in types) + " |")
    lines.append("|---|---|" + "---|---|" * len(types))
    classes = list(res["metrics"][types[0]]["per_class"]) if types else []
    for c in classes:
        cells = []
        for t in types:
            pc = res["metrics"][t]["per_class"][c]
            cells += [fmt(pc["AP"]), fmt(pc["AP50"])]
        n = res["metrics"][types[0]]["per_class"][c]["gt_instances"]
        lines.append(f"| {c} | {n} | " + " | ".join(cells) + " |")
    occ_ref = next((res["metrics"][t]["by_occlusion"] for t in types if res["metrics"][t].get("by_occlusion")), None)
    if occ_ref:
        lines += ["", "By ground-truth occlusion ratio (other GT ignored; AP / AP50):", ""]
        lines.append("| occlusion ratio | GT | " + " | ".join(f"{t} AP | {t} AP50" for t in types) + " |")
        lines.append("|---|---|" + "---|---|" * len(types))
        for i, b in enumerate(occ_ref):
            cells = []
            for t in types:
                occ = res["metrics"][t].get("by_occlusion") or []
                bb = occ[i] if i < len(occ) else {"AP": 0.0 if res["metrics"][t].get("no_predictions") else -1,
                                                  "AP50": 0.0 if res["metrics"][t].get("no_predictions") else -1}
                cells += [fmt(bb["AP"]), fmt(bb["AP50"])]
            lines.append(f"| {b['bin']} | {b['gt_instances']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def gt_as_predictions(gt_path):
    data = load_json(gt_path)
    return [{"image_id": a["image_id"], "category_id": a["category_id"], "score": 1.0,
             "bbox": a["bbox"], "segmentation": a["segmentation"]}
            for a in data["annotations"] if not a["iscrowd"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="a make_coco.py split, e.g. coco/test.json")
    ap.add_argument("--pred", help="COCO results JSON")
    ap.add_argument("--out", help="output folder for metrics.json / metrics.md")
    ap.add_argument("--iou-types", nargs="+", default=["bbox", "segm"], choices=["bbox", "segm"])
    ap.add_argument("--occlusion-bins", default=DEFAULT_OCC_BINS,
                    help=f"bin edges of the GT occlusion ratio, '' to skip (default {DEFAULT_OCC_BINS})")
    ap.add_argument("--title", default=None)
    ap.add_argument("--selftest", action="store_true", help="feed the GT as predictions; every AP must be 1.0")
    args = ap.parse_args(argv)
    bins = [float(x) for x in args.occlusion_bins.split(",")] if args.occlusion_bins else []

    t0 = time.time()
    if args.selftest:
        preds = gt_as_predictions(args.gt)
        title = args.title or "Self-test: ground truth as predictions"
    else:
        if not args.pred:
            ap.error("--pred is required unless --selftest")
        preds = load_json(args.pred)
        title = args.title or f"COCO evaluation of {Path(args.pred).name}"
    res = evaluate(args.gt, preds, args.iou_types, bins)
    res["seconds"] = round(time.time() - t0, 1)
    md = to_markdown(res, title)
    print(md)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        dump_json(res, out / "metrics.json", indent=1)
        (out / "metrics.md").write_text(md, encoding="utf-8")
        print(f"-> {out / 'metrics.json'}, {out / 'metrics.md'}")
    if args.selftest:
        bad = []
        for t, m in res["metrics"].items():
            if m is None:
                bad.append(f"{t}: no predictions")
                continue
            for n in ("AP", "AP50", "AP75", "APs", "APm", "APl"):
                v = m["summary"][n]
                if v != -1 and abs(v - 1.0) > 1e-9:
                    bad.append(f"{t} {n} = {v}")
            for c, pc in m["per_class"].items():
                if pc["AP"] != -1 and abs(pc["AP"] - 1.0) > 1e-9:
                    bad.append(f"{t} class {c} AP = {pc['AP']}")
            for b in m.get("by_occlusion", []):
                if b["AP"] != -1 and abs(b["AP"] - 1.0) > 1e-9:
                    bad.append(f"{t} occlusion {b['bin']} AP = {b['AP']}")
        # control: the evaluator must notice missing predictions (every second one dropped)
        half = evaluate(args.gt, preds[::2], ["segm"], [])["metrics"]["segm"]["summary"]["AP"]
        print(f"control: every second GT prediction dropped -> segm AP {half:.3f} (must be < 1)")
        if not half < 1 - 1e-6:
            bad.append(f"control with half the predictions gave AP {half}")
        if bad:
            print("SELFTEST FAILED:\n  " + "\n  ".join(bad))
            sys.exit(1)
        print(f"SELFTEST PASSED: AP = 1.0 for {', '.join(res['metrics'])} ({res['seconds']} s)")


if __name__ == "__main__":
    main()
