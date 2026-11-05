#!/usr/bin/env python
"""Merge the per-desktop scanner annotations into standard COCO train / val / test files.

Input : the scanner pack (Hugging Face `annotations/scanner/`, one `D<nn>/annotations.json` per desktop)
        and `splits.json` (12 test / 54 train desktops).
Output: <out>/train.json, val.json, test.json, stats.json, stats.md, make_coco_args.json

* test  = the desktops `splits.json` lists as test (D04 D10 D15 D20 D24 D30 D31 D32 D46 D48 D49 D65);
* val   = `--val-n` train desktops drawn with `random.Random(--val-seed).sample(sorted(train), n)`
          (default 6 with seed 0 -> D03 D21 D33 D35 D41 D60, see README), or `--val-desktops`;
* train = the remaining train desktops.

Image ids are the pack's own (already unique across desktops, checked); annotation ids are renumbered
1..N per output file. `file_name` is `D<nn>/images/S<nn>.png`, relative to `info.image_root`
(the absolute path of the pack folder on this machine; override with `--image-root-in-json`).

Masks (`--masks`):
  visible (default)  `segmentation` = the confirmed visible mask; fully covered parts (pack crowd rows:
                     empty mask, bbox = amodal box) stay `iscrowd = 1`, i.e. ignore regions.
  amodal             `segmentation` / `bbox` / `area` = `amodal_segmentation` / `amodal_bbox` / `amodal_area`,
                     and every row has `iscrowd = 0` (a fully covered part's amodal shape is real ground truth).

Filters: `--clean-only` (keep the pack's `clean` rows as positives), `--min-area N` (on the chosen mask),
`--classes a,b,c` (category subset; other classes are dropped together with their category entries).
A row removed by --clean-only or --min-area becomes an ignore region (`iscrowd = 1`, the default
`--filtered ignore`) or is deleted (`--filtered drop`). Category ids are always the pack's ids.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l1common import TEST_DESKTOPS, dname, dump_json, load_json, parse_desktops  # noqa: E402

EXTRA_ANN_FIELDS = ("instance_id", "visibility", "occlusion_ratio", "clean", "shared_px")


def pick_splits(args):
    sp = load_json(args.splits) if Path(args.splits).exists() else None
    if sp is not None:
        test = sorted(int(d) for d in sp["test"])
        train_all = sorted(int(d) for d in sp["train"])
    else:
        print(f"warning: {args.splits} not found; using the built-in test list", file=sys.stderr)
        test = list(TEST_DESKTOPS)
        train_all = sorted(int(p.name[1:]) for p in Path(args.pack).glob("D[0-9][0-9]") if int(p.name[1:]) not in test)
    if sorted(test) != sorted(TEST_DESKTOPS):
        print(f"warning: the test desktops in {args.splits} differ from the published 12: {test}", file=sys.stderr)
    if args.val_desktops is not None:
        val = parse_desktops(args.val_desktops)
        bad = [d for d in val if d not in train_all]
        if bad:
            sys.exit(f"--val-desktops must be train desktops; not train: {bad}")
    else:
        val = sorted(random.Random(args.val_seed).sample(train_all, args.val_n)) if args.val_n > 0 else []
    train = [d for d in train_all if d not in val]
    # explicit overrides (smoke tests, ablations)
    if args.train_desktops is not None:
        train = parse_desktops(args.train_desktops)
    if args.test_desktops is not None:
        test = parse_desktops(args.test_desktops)
    for a, b, na, nb in ((train, test, "train", "test"), (val, test, "val", "test"), (train, val, "train", "val")):
        both = set(a) & set(b)
        if both:
            sys.exit(f"desktops in both {na} and {nb}: {sorted(both)}")
    return {"train": train, "val": val, "test": test}


def load_clean_csv(pack):
    p = Path(pack) / "clean_parts.csv"
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as f:
        return {(int(r["desktop"]), int(r["step"]), int(r["instance_id"])) for r in csv.DictReader(f)}


def convert_split(name, desktops, args, keep_cats, clean_csv, pack_root, all_categories):
    images, anns = [], []
    seen_img = set()
    categories = all_categories
    reasons = Counter()
    clean_mismatch = 0
    next_ann = 1
    for d in desktops:
        path = Path(args.pack) / dname(d) / "annotations.json"
        if not path.exists():
            sys.exit(f"missing {path}: download the scanner pack (see README)")
        coco = load_json(path)
        if [(c["id"], c["name"]) for c in coco["categories"]] != [(c["id"], c["name"]) for c in categories]:
            sys.exit(f"{path}: category table differs from the first desktop's")
        img_step = {}
        for im in coco["images"]:
            if im["id"] in seen_img:
                sys.exit(f"duplicate image id {im['id']} ({path})")
            seen_img.add(im["id"])
            step = int(im.get("extra", {}).get("step", 0))
            img_step[im["id"]] = step
            images.append({
                "id": im["id"],
                "file_name": f"{dname(d)}/{im['file_name']}",
                "width": im["width"],
                "height": im["height"],
                "desktop": d,
                "step": step,
                "roi": im.get("extra", {}).get("roi"),
            })
        for a in coco["annotations"]:
            if keep_cats is not None and a["category_id"] not in keep_cats:
                reasons["dropped_other_class"] += 1
                continue
            if args.masks == "amodal":
                seg, bbox, area, crowd = a["amodal_segmentation"], a["amodal_bbox"], a["amodal_area"], 0
            else:
                seg, bbox, area, crowd = a["segmentation"], a["bbox"], a["area"], int(a["iscrowd"])
            if clean_csv is not None:
                in_csv = (d, img_step[a["image_id"]], a["instance_id"]) in clean_csv
                clean_mismatch += int(in_csv != bool(a.get("clean")))
            filtered = None
            if args.clean_only and not a.get("clean"):
                filtered = "not_clean"
            elif args.min_area and area < args.min_area and not crowd:
                filtered = "below_min_area"
            if filtered:
                reasons[filtered] += 1
                if args.filtered == "drop":
                    continue
                crowd = 1
            if crowd and int(a["iscrowd"]) and args.masks == "visible":
                reasons["fully_covered_crowd"] += 1
            out = {
                "id": next_ann,
                "image_id": a["image_id"],
                "category_id": a["category_id"],
                "segmentation": seg,
                "bbox": [float(v) for v in bbox],
                "area": float(area),
                "iscrowd": int(crowd),
                "desktop": d,
                "instance_key": a.get("attributes", {}).get("instance_key"),
                "pack_ann_id": a["id"],
            }
            for k in EXTRA_ANN_FIELDS:
                if k in a:
                    out[k] = a[k]
            if filtered:
                out["ignore_reason"] = filtered
            anns.append(out)
            next_ann += 1
    if keep_cats is not None:
        categories = [c for c in categories if c["id"] in keep_cats]
    if clean_csv is not None and clean_mismatch:
        print(f"warning: {name}: {clean_mismatch} rows where the clean flag and clean_parts.csv disagree", file=sys.stderr)
    info = {
        "description": f"E3D scanner view, Level-1 {name} split ({args.masks} masks)",
        "split": name,
        "desktops": [dname(d) for d in desktops],
        "masks": args.masks,
        "clean_only": bool(args.clean_only),
        "min_area": args.min_area,
        "filtered": args.filtered,
        "image_root": pack_root,
        "made_by": "benchmarks/level1/make_coco.py",
        "date_created": time.strftime("%Y-%m-%d"),
    }
    return {"info": info, "licenses": [], "images": images, "annotations": anns, "categories": categories}, reasons


def split_stats(coco, reasons):
    names = {c["id"]: c["name"] for c in coco["categories"]}
    pos, ign = Counter(), Counter()
    imgs_with = defaultdict(set)
    for a in coco["annotations"]:
        (ign if a["iscrowd"] else pos)[names[a["category_id"]]] += 1
        if not a["iscrowd"]:
            imgs_with[names[a["category_id"]]].add(a["image_id"])
    per_img = Counter(a["image_id"] for a in coco["annotations"] if not a["iscrowd"])
    return {
        "desktops": coco["info"]["desktops"],
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "instances": sum(pos.values()),
        "ignore_regions": sum(ign.values()),
        "max_instances_per_image": max(per_img.values()) if per_img else 0,
        "instances_per_class": {n: pos.get(n, 0) for n in sorted(names.values())},
        "ignore_per_class": {n: ign.get(n, 0) for n in sorted(names.values())},
        "images_per_class": {n: len(imgs_with.get(n, ())) for n in sorted(names.values())},
        "filter_counts": dict(reasons),
    }


def stats_markdown(stats, args):
    splits = [s for s in ("train", "val", "test") if s in stats]
    lines = [f"# Level-1 COCO files: `--masks {args.masks}`"
             + (" `--clean-only`" if args.clean_only else "")
             + (f" `--min-area {args.min_area}`" if args.min_area else ""), ""]
    lines.append("| | " + " | ".join(splits) + " |")
    lines.append("|---|" + "---|" * len(splits))
    for key, label in (("images", "images"), ("instances", "instances (iscrowd 0)"),
                       ("ignore_regions", "ignore regions (iscrowd 1)"), ("max_instances_per_image", "max instances / image")):
        lines.append(f"| {label} | " + " | ".join(str(stats[s][key]) for s in splits) + " |")
    lines.append("| desktops | " + " | ".join(" ".join(stats[s]["desktops"]) for s in splits) + " |")
    lines += ["", "Instances (ignore regions) per class:", ""]
    lines.append("| class | " + " | ".join(splits) + " |")
    lines.append("|---|" + "---|" * len(splits))
    classes = sorted(next(iter(stats.values()))["instances_per_class"])
    for c in classes:
        lines.append(f"| {c} | " + " | ".join(
            f"{stats[s]['instances_per_class'][c]} ({stats[s]['ignore_per_class'][c]})" for s in splits) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=".", help="folder of the Hugging Face download (default: .)")
    ap.add_argument("--pack", default=None, help="scanner pack folder (default: <data-root>/annotations/scanner)")
    ap.add_argument("--splits", default=None, help="splits.json (default: <data-root>/splits.json)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--masks", choices=("visible", "amodal"), default="visible")
    ap.add_argument("--clean-only", action="store_true", help="only the pack's clean rows are positives")
    ap.add_argument("--min-area", type=float, default=0, help="rows with a smaller mask area are filtered")
    ap.add_argument("--filtered", choices=("ignore", "drop"), default="ignore",
                    help="a row filtered by --clean-only / --min-area becomes an ignore region (default) or is deleted")
    ap.add_argument("--classes", default=None, help="comma-separated class names to keep (default: all 24)")
    ap.add_argument("--val-n", type=int, default=6)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--val-desktops", default=None, help="explicit val desktops (train desktops), e.g. 7,11")
    ap.add_argument("--train-desktops", default=None, help="override the train desktops (smoke tests)")
    ap.add_argument("--test-desktops", default=None, help="override the test desktops (smoke tests)")
    ap.add_argument("--image-root-in-json", default=None,
                    help="value written to info.image_root (default: absolute path of --pack)")
    args = ap.parse_args(argv)
    args.pack = args.pack or str(Path(args.data_root) / "annotations" / "scanner")
    args.splits = args.splits or str(Path(args.data_root) / "splits.json")
    if not (Path(args.pack) / "D01").exists() and not any(Path(args.pack).glob("D[0-9][0-9]")):
        sys.exit(f"no D<nn> folders in {args.pack}")
    pack_root = args.image_root_in_json or str(Path(args.pack).resolve())

    splits = pick_splits(args)
    first = load_json(Path(args.pack) / dname((splits["train"] or splits["test"])[0]) / "annotations.json")
    cat_by_name = {c["name"]: c["id"] for c in first["categories"]}
    keep_cats = None
    if args.classes:
        names = [n.strip() for n in args.classes.split(",") if n.strip()]
        unknown = [n for n in names if n not in cat_by_name]
        if unknown:
            sys.exit(f"unknown classes {unknown}; known: {sorted(cat_by_name)}")
        keep_cats = {cat_by_name[n] for n in names}
    clean_csv = load_clean_csv(args.pack)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    t0 = time.time()
    for name in ("train", "val", "test"):
        coco, reasons = convert_split(name, splits[name], args, keep_cats, clean_csv, pack_root,
                                      first["categories"])
        dump_json(coco, out / f"{name}.json")
        stats[name] = split_stats(coco, reasons)
        s = stats[name]
        print(f"{name:5s}: {len(splits[name]):2d} desktops, {s['images']:5d} images, {s['instances']:6d} instances, "
              f"{s['ignore_regions']:5d} ignore regions -> {out / (name + '.json')}")
    dump_json(stats, out / "stats.json", indent=1)
    (out / "stats.md").write_text(stats_markdown(stats, args), encoding="utf-8")
    dump_json({**vars(args), "splits": {k: [dname(d) for d in v] for k, v in splits.items()}},
              out / "make_coco_args.json", indent=1)
    print(f"done in {time.time() - t0:.1f}s; stats -> {out / 'stats.md'}")


if __name__ == "__main__":
    main()
