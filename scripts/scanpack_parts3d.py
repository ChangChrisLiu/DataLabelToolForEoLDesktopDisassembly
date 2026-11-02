"""Take each annotated part's 3D points out of a reconstruction of the same scanner frame.

Shipped with the scanner annotations (`annotations/scanner/tools/parts3d.py` on the dataset repository) for the
3D collaborators; written for the round-3 release (2026-11-02). Needs only numpy and Pillow (huggingface_hub only for
`where --download`). The guide is `annotations/scanner/HOW_TO_EXTRACT_3D_PARTS.md`.

Subcommands
-----------
plan     one row per (desktop, part): how to get that part's points -- the best frame, the route, the frames involved.
where    the raw scanner picture each annotated frame was drawn on (dataset path), and optionally downloads it.
extract  route 1 (and the clean set): the points under a part's pixels in the label map of one frame, from an
         *organized* point cloud of that frame (H x W x 3, or H x W x 6 with colour; NaN = no point) -> one PLY per part.
diff     route 2 (disassembly order): the points of a part that leaves at step r = the points of frame r-1 inside its
         amodal mask on r-1 that are gone on frame r (same camera pose: compared pixel by pixel).

Coordinates: image pixels of the 1600 x 1600 scanner picture `frames.json` names, x = column, y = row, origin top
left. "Organized" means cloud[y, x] is the 3D point seen at pixel (x, y) of that picture; if your reconstruction is not
organized, project its points into the picture with the scanner calibration first (see the guide).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HF_REPO = "ChangChrisLiu/E3D-Desktop-Disassembly"
RAW_PREFIX = "raw/scanner/"
DRIVE_PREFIX = "UGA DATA/"          # source_folder is written below the scanner drive's root


# --------------------------------------------------------------------------- reading the package
def desktops_of(pack: Path, wanted: str | None) -> list[str]:
    names = sorted(p.name for p in pack.iterdir() if p.is_dir() and p.name.startswith("D") and p.name[1:].isdigit())
    if not wanted:
        return names
    keep: set[int] = set()
    for part in wanted.split(","):
        a, _, b = part.strip().lstrip("Dd").partition("-")
        keep.update(range(int(a), int(b.lstrip("Dd") or a) + 1))
    return [n for n in names if int(n[1:]) in keep]


class Desktop:
    """One `D<NN>/` folder: annotations, instances, frames, label maps."""

    def __init__(self, pack: Path, name: str):
        self.dir = pack / name
        self.name = name
        self.coco = json.loads((self.dir / "annotations.json").read_text(encoding="utf-8"))
        self.inst = json.loads((self.dir / "instances.json").read_text(encoding="utf-8"))["instances"]
        self.frames = json.loads((self.dir / "frames.json").read_text(encoding="utf-8"))["frames"]
        self.images = {im["id"]: im for im in self.coco["images"]}
        self.by_step = {int(im["extra"]["step"]): im for im in self.coco["images"]}
        self.key_to_id = {v["key"]: int(k) for k, v in self.inst.items()}

    def instance(self, ref: str) -> tuple[int, dict]:
        iid = int(ref) if ref.isdigit() else self.key_to_id.get(ref)
        if iid is None or str(iid) not in self.inst:
            raise SystemExit(f"{self.name}: no part {ref!r} (keys: {', '.join(sorted(self.key_to_id))})")
        return iid, self.inst[str(iid)]

    def anns_of(self, iid: int) -> list[dict]:
        return [a for a in self.coco["annotations"] if int(a["instance_id"]) == iid]

    def label_map(self, step: int) -> np.ndarray:
        im = self.by_step.get(step)
        if im is None:
            raise SystemExit(f"{self.name}: frame S{step:02d} is not in the package")
        return np.array(Image.open(self.dir / im["label_file"]))

    def ann(self, iid: int, step: int) -> dict | None:
        im = self.by_step.get(step)
        if im is None:
            return None
        return next((a for a in self.coco["annotations"]
                     if int(a["instance_id"]) == iid and a["image_id"] == im["id"]), None)

    def raw_path(self, step: int) -> str | None:
        fr = self.frames.get(str(step)) or {}
        folder, name = fr.get("source_folder"), fr.get("file_name")
        if not folder or not name:
            return None
        if folder.startswith(DRIVE_PREFIX):
            folder = folder[len(DRIVE_PREFIX):]
        return f"{RAW_PREFIX}{folder}/{name}"


def decode_rle(rle: dict) -> np.ndarray:
    """COCO RLE (compressed string or list counts) -> bool mask H x W; no pycocotools needed."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        vals, p, s = [], 0, counts.encode("ascii")
        while p < len(s):
            x, k, more = 0, 0, True
            while more:
                c = s[p] - 48
                x |= (c & 0x1F) << (5 * k)
                more = bool(c & 0x20)
                p += 1
                k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(vals) > 2:
                x += vals[-2]
            vals.append(x)
        counts = vals
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for n in counts:
        if val:
            flat[pos:pos + n] = True
        pos += n
        val = not val
    return flat.reshape((w, h)).T          # column-major


# --------------------------------------------------------------------------- plan
PLAN_COLUMNS = ["desktop", "instance_id", "key", "class", "of", "route", "best_frame", "best_frame_px",
                "clean_frames", "frames_seen", "frames_crowd", "leaving_step", "leaving_kind", "leaving_verb",
                "left_with", "before_frame", "after_frame", "same_pose_segment", "raw_best_frame", "note"]


def plan_rows(d: Desktop) -> list[dict]:
    rows = []
    for sid, rec in sorted(d.inst.items(), key=lambda kv: int(kv[0])):
        iid = int(sid)
        anns = d.anns_of(iid)
        if not anns:
            continue
        step_of = {a["image_id"]: int(d.images[a["image_id"]]["extra"]["step"]) for a in anns}
        clean = sorted((a for a in anns if a.get("clean")), key=lambda a: (-int(a["area"]), step_of[a["image_id"]]))
        seen = sorted((a for a in anns if int(a["area"]) > 0), key=lambda a: (-int(a["area"]), step_of[a["image_id"]]))
        crowd = [a for a in anns if a.get("iscrowd")]
        lv = rec.get("leaving") or {}
        notes = []
        if clean:
            route, best = "clean", clean[0]
        elif lv.get("kind") in ("removed", "displaced") and lv.get("before_frame") and lv.get("after_frame") \
                and lv.get("same_pose_segment"):
            route, best = "diff", next((a for a in anns if step_of[a["image_id"]] == lv["before_frame"]), seen[0] if seen else None)
            notes.append("never unoccluded: take the points that disappear when it leaves (route 2)")
        elif seen:
            route, best = "visible", seen[0]
            notes.append("never unoccluded and no clean leaving pair: visible pixels only (partial surface)")
        else:
            route, best = "none", None
            notes.append("covered on every frame: no points")
        if lv.get("left_with"):
            notes.append(f"leaves inside {lv['left_with']}: route 2 removes both together")
        if lv.get("kind") == "changed":
            notes.append(f"never leaves; changes {lv.get('old')} -> {lv.get('new')} at step {lv.get('step')}")
        if rec["class"] == "drive_cage" and (rec.get("attrs") or {}).get("of") == "optical":
            notes.append("optical drive with its bracket (one instance)")
        if rec["class"] in ("chassis", "motherboard"):
            notes.append("large part under most others: use the frame where the most of it is visible")
        bstep = step_of[best["image_id"]] if best else None
        rows.append({
            "desktop": d.name, "instance_id": iid, "key": rec["key"], "class": rec["class"],
            "of": (rec.get("attrs") or {}).get("of") or "", "route": route,
            "best_frame": bstep or "", "best_frame_px": int(best["area"]) if best else 0,
            "clean_frames": " ".join(str(s) for s in sorted(step_of[a["image_id"]] for a in clean)),
            "frames_seen": len(seen), "frames_crowd": len(crowd),
            "leaving_step": lv.get("step") or "", "leaving_kind": lv.get("kind") or "", "leaving_verb": lv.get("verb") or "",
            "left_with": lv.get("left_with") or "", "before_frame": lv.get("before_frame") or "",
            "after_frame": lv.get("after_frame") or "",
            "same_pose_segment": "" if lv.get("same_pose_segment") is None else str(bool(lv.get("same_pose_segment"))).lower(),
            "raw_best_frame": d.raw_path(bstep) if bstep else "", "note": "; ".join(notes)})
    return rows


def cmd_plan(a) -> int:
    pack = Path(a.pack)
    out = Path(a.out)
    rows = []
    for name in desktops_of(pack, a.desktops):
        rows += plan_rows(Desktop(pack, name))
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[r["route"]] = by.get(r["route"], 0) + 1
    print(f"[parts3d] {len(rows)} parts on {len({r['desktop'] for r in rows})} desktops -> {out}; routes {by}")
    return 0


# --------------------------------------------------------------------------- where / fetch
def cmd_where(a) -> int:
    pack = Path(a.pack)
    d = Desktop(pack, f"D{int(a.desktop.lstrip('Dd')):02d}")
    steps = sorted(int(s) for s in d.frames) if not a.frames else [int(s) for s in a.frames.split(",")]
    for s in steps:
        rel = d.raw_path(s)
        print(f"{d.name} S{s:02d}\t{rel}")
        if a.download and rel:
            from huggingface_hub import hf_hub_download    # only for --download
            got = hf_hub_download(HF_REPO, rel, repo_type="dataset", local_dir=a.download)
            print(f"   -> {got}")
    return 0


# --------------------------------------------------------------------------- extract (route 1 / clean)
def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None) -> None:
    n = len(xyz)
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
            "property float x", "property float y", "property float z"]
    if rgb is not None:
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    head += ["end_header"]
    with path.open("wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        if rgb is None:
            f.write(xyz.astype("<f4").tobytes())
        else:
            rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            f.write(rec.tobytes())


def load_cloud(path: str, hw: tuple[int, int]) -> np.ndarray:
    cloud = np.load(path)
    if cloud.ndim != 3 or cloud.shape[:2] != hw or cloud.shape[2] not in (3, 6):
        raise SystemExit(f"{path}: expected an organized cloud of shape {hw[0]} x {hw[1]} x 3 (or x 6 with colour), "
                         f"got {cloud.shape}; see the guide (project an unorganized cloud into the picture first)")
    return cloud


def points_under(cloud: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    sel = cloud[mask]
    ok = np.isfinite(sel[:, :3]).all(axis=1)
    sel = sel[ok]
    rgb = sel[:, 3:6].clip(0, 255).astype("u1") if sel.shape[1] == 6 else None
    return sel[:, :3], rgb


def cmd_extract(a) -> int:
    pack = Path(a.pack)
    d = Desktop(pack, f"D{int(a.desktop.lstrip('Dd')):02d}")
    step = int(a.frame)
    labels = d.label_map(step)
    cloud = load_cloud(a.cloud, labels.shape)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = [d.instance(r)[0] for r in a.instances.split(",")] if a.instances else \
        sorted(int(v) for v in np.unique(labels) if v)
    rows = []
    for iid in wanted:
        rec = d.inst[str(iid)]
        ann = d.ann(iid, step)
        mask = labels == iid
        xyz, rgb = points_under(cloud, mask)
        name = f"{d.name}_S{step:02d}_{rec['key']}.ply"
        if len(xyz):
            write_ply(out / name, xyz, rgb)
        rows.append({"part": rec["key"], "class": rec["class"], "pixels": int(mask.sum()), "points": len(xyz),
                     "clean": bool(ann and ann.get("clean")), "visibility": (ann or {}).get("visibility"),
                     "occlusion_ratio": (ann or {}).get("occlusion_ratio"), "file": name if len(xyz) else None})
    (out / f"{d.name}_S{step:02d}.extract.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r in rows:
        print(f"  {r['part']:<28} {r['class']:<16} px {r['pixels']:>7}  points {r['points']:>7}  "
              f"{'clean' if r['clean'] else (r['visibility'] or '-')}")
    print(f"[parts3d] {len(rows)} parts of {d.name} S{step:02d} -> {out}")
    return 0


# --------------------------------------------------------------------------- diff (route 2)
def cmd_diff(a) -> int:
    pack = Path(a.pack)
    d = Desktop(pack, f"D{int(a.desktop.lstrip('Dd')):02d}")
    iid, rec = d.instance(a.part)
    lv = rec.get("leaving") or {}
    before, after = lv.get("before_frame"), lv.get("after_frame")
    if lv.get("kind") not in ("removed", "displaced") or not before or not after:
        raise SystemExit(f"{rec['key']}: no leaving pair (kind {lv.get('kind')!r}, frames {before}, {after}); "
                         f"use extract (route 1) instead")
    if not lv.get("same_pose_segment"):
        raise SystemExit(f"{rec['key']}: frames S{before:02d} and S{after:02d} were taken from different camera poses; "
                         f"register the two clouds first, then compare by nearest neighbour")
    ann = d.ann(iid, before)
    if ann is None:
        raise SystemExit(f"{rec['key']}: no annotation on S{before:02d}")
    amodal = decode_rle(ann["amodal_segmentation"])
    hw = amodal.shape
    cb, ca = load_cloud(a.cloud_before, hw), load_cloud(a.cloud_after, hw)
    pb, pa = cb[..., :3], ca[..., :3]
    has_b = np.isfinite(pb).all(axis=2)
    has_a = np.isfinite(pa).all(axis=2)
    moved = np.zeros(hw, dtype=bool)
    both = has_b & has_a
    moved[both] = np.linalg.norm(pb[both] - pa[both], axis=1) > a.tol
    gone = amodal & has_b & (~has_a | moved)
    xyz, rgb = points_under(cb, gone)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{d.name}_{rec['key']}_route2_S{before:02d}-S{after:02d}.ply"
    if len(xyz):
        write_ply(out / name, xyz, rgb)
    others = [v["key"] for v in d.inst.values()
              if (v.get("leaving") or {}).get("step") == lv.get("step") and v["key"] != rec["key"]]
    print(f"[parts3d] {rec['key']}: {len(xyz)} points that are on S{before:02d} inside its amodal mask "
          f"({int(amodal.sum())} px) and gone or moved more than {a.tol} on S{after:02d} -> {out / name}")
    if others:
        print(f"[parts3d]   also changing at step {lv.get('step')}: {', '.join(sorted(others))} -- their points can "
              f"fall inside this mask too; subtract them if they overlap")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="one row per part: route, best frame, frames involved")
    p.add_argument("--pack", required=True, help="the annotations/scanner folder")
    p.add_argument("--desktops", help="e.g. 1-66 or 3,4,22")
    p.add_argument("--out", default="parts_3d_plan.csv")
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("where", help="dataset path of the raw picture of each frame")
    p.add_argument("--pack", required=True)
    p.add_argument("--desktop", required=True)
    p.add_argument("--frames", help="e.g. 1,12,13 (default all)")
    p.add_argument("--download", metavar="DIR", help="download the pictures into DIR (needs huggingface_hub)")
    p.set_defaults(func=cmd_where)
    p = sub.add_parser("extract", help="route 1: points under the label map of one frame")
    p.add_argument("--pack", required=True)
    p.add_argument("--desktop", required=True)
    p.add_argument("--frame", required=True, help="step number of the frame (S<nn>)")
    p.add_argument("--cloud", required=True, help="organized cloud .npy, H x W x 3 or x 6, NaN = no point")
    p.add_argument("--instances", help="keys or ids, comma separated (default: every part in the label map)")
    p.add_argument("--out", default="parts3d_out")
    p.set_defaults(func=cmd_extract)
    p = sub.add_parser("diff", help="route 2: the points that disappear when the part leaves")
    p.add_argument("--pack", required=True)
    p.add_argument("--desktop", required=True)
    p.add_argument("--part", required=True, help="instance key or id")
    p.add_argument("--cloud-before", required=True, help="organized cloud of frame leaving.before_frame")
    p.add_argument("--cloud-after", required=True, help="organized cloud of frame leaving.after_frame")
    p.add_argument("--tol", type=float, default=0.002, help="a point that moved more than this (cloud units) counts as gone")
    p.add_argument("--out", default="parts3d_out")
    p.set_defaults(func=cmd_diff)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
