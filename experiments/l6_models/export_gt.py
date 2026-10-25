"""Ground truth for L6: ``export-coco`` of every verified scanner frame, from the L6 copy.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.export_gt            # export + build
    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.export_gt --build    # build only

1. **Export.**  :data:`env.EXPORT_DB` is made once from :data:`env.LIVE_COPY`
   with SQLite's backup API (``export-coco`` refreshes the truth table and
   takes the single-user lock, i.e. it writes, so it never sees the copy every
   other L6 read uses).  Then the app's own exporter, in the app's env::

       python -m tda.cli export-coco --db <EXPORT_DB> --desktops <the 20> \\
           --view scan --no-hidden --out <ROOT>/coco/coco_scan_verified.json

   (``--only-verified`` and the amodal fields are the exporter's defaults;
   ``--no-hidden`` leaves out the parts covered completely, which have no
   visible pixel to learn from.)
2. **Build** (CPU only).  The image entries whose ``extra.review_status`` is
   ``verified`` (a frame with confirmed rows but not confirmed as a whole is
   left out), and of their annotations the ones L4/L5 trained on: not
   ``iscrowd``, not a context object, ``placement == in_chassis``,
   ``visibility`` in :data:`KEEP_VIS`, a non-empty segmentation.

   * ``gt/gt_det.json`` -- L4's ``gt.Frame`` list (box = the bbox of the
     visible mask, exclusive x1/y1; ROI = the pose segment's stored ROI; image
     = the file the app shows), read by L4's evaluation code unchanged;
   * ``gt/gt_sam.pkl`` -- L5's ``gt.Rec`` list (the visible mask bit-packed in
     its bbox, the distance-transform click point, the ``drawn`` flag from the
     op log), read by L5's harness and the L6 SAM trainer;
   * ``gt/manifest.json`` -- counts per desktop / class, the COCO file's
     SHA-256, and a cross-check of every frame against a direct read of the
     same rows in :data:`env.LIVE_COPY` (L4's ``gt.build``): the same boxes,
     classes, visibilities and areas, or the differences listed.
"""
from __future__ import annotations

from experiments.l6_models import env  # noqa: F401  (side effects first)

import argparse
import hashlib
import json
import os
import pickle
import sqlite3
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.l4_detector import gt as G4
from experiments.l5_sam import gt as G5

VIEW = "scan"
KEEP_VIS = G4.KEEP_VIS
COCO = env.ROOT / "coco" / "coco_scan_verified.json"
GT_DIR = env.ROOT / "gt"
GT_DET = GT_DIR / "gt_det.json"
GT_SAM = GT_DIR / "gt_sam.pkl"
MANIFEST = GT_DIR / "manifest.json"
EXPORT_INFO = env.ROOT / "logs" / "export_coco.json"
HELD_OUT = (19, 27)
EXTRA_CLASSES = ("ram_latch", "drive_latch", "cable_clip", "cpu_socket_lever")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verified_desktops() -> list[int]:
    con = env.connect()
    ds = [int(r[0]) for r in con.execute(
        "SELECT DISTINCT desktop FROM frame WHERE view=? AND review_status='verified' "
        "ORDER BY desktop", (VIEW,))]
    con.close()
    return ds


def _ranges(ds: list[int]) -> str:
    out, start, prev = [], None, None
    for d in ds:
        if start is None:
            start = prev = d
        elif d == prev + 1:
            prev = d
        else:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = d
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(out)


def export() -> dict:
    if not env.EXPORT_DB.exists():
        src = sqlite3.connect(env.DB_URI, uri=True)
        dst = sqlite3.connect(str(env.EXPORT_DB))
        src.backup(dst)
        dst.close()
        src.close()
        print(f"[l6] {env.EXPORT_DB} made from {env.LIVE_COPY}", flush=True)
    ds = verified_desktops()
    COCO.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(env.APP_ENV_PY), "-m", "tda.cli", "export-coco", "--db", str(env.EXPORT_DB),
           "--desktops", _ranges(ds), "--view", VIEW, "--no-hidden", "--out", str(COCO)]
    print("[l6] " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(env.REPO), env=dict(os.environ), capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    log = env.ROOT / "logs" / "export_coco.log"
    log.write_text(res.stdout + "\n--- stderr ---\n" + res.stderr, encoding="utf-8")
    print(res.stdout[-3000:], flush=True)
    if res.returncode != 0:
        print(res.stderr[-3000:])
        raise SystemExit(f"export-coco failed ({res.returncode}); see {log}")
    info = {"command": cmd, "seconds": round(time.perf_counter() - t0, 1),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"), "desktops": ds,
            "stdout_tail": res.stdout[-1500:]}
    EXPORT_INFO.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def _drawn_steps(con) -> dict[tuple[int, str], set[int]]:
    """``(desktop, instance) -> steps`` a keyframe was committed on (L5's rule, every desktop)."""
    out: dict[tuple[int, str], set[int]] = defaultdict(set)
    for r in con.execute("SELECT payload_json FROM op_log WHERE kind='commit_keyframe' "
                         "AND view=?", (VIEW,)):
        try:
            p = json.loads(r["payload_json"] or "{}")
        except ValueError:
            continue
        if p.get("view") != VIEW or p.get("desktop") is None:
            continue
        out[(int(p["desktop"]), str(p.get("instance")))].add(int(p.get("step", -1)))
    return out


_SUGGESTED: dict[int, dict] = {}
_STORED_ROI = G4.roi_of


def roi_for(con, desktop: int, step: int) -> tuple[tuple[int, int, int, int], str]:
    """The stored pose-segment ROI, or -- where none is stored (D27 on 2026-10-24) --
    the box the app would propose for the segment: ``suggest_roi_over`` on its
    first / middle / last frame with an image (``tda.ui.app_roi._roi_sample_steps``)."""
    roi = _STORED_ROI(con, desktop, step)
    if roi is not None:
        return roi, "stored"
    if desktop not in _SUGGESTED:
        import cv2

        from tda.core.cache import ROI_SAMPLE_FRAMES, suggest_roi_over

        seg = con.execute("SELECT start_step, end_step FROM pose_segment WHERE desktop=? AND "
                          "view=? AND start_step<=? AND end_step>=? ORDER BY seg LIMIT 1",
                          (desktop, VIEW, step, step)).fetchone()
        steps = [int(r["step"]) for r in con.execute(
            "SELECT step FROM frame WHERE desktop=? AND view=? ORDER BY step", (desktop, VIEW))]
        if seg is not None:
            steps = [s for s in steps if int(seg["start_step"]) <= s <= int(seg["end_step"])]
        steps = [s for s in steps if G4.image_path(con, desktop, s)]
        if len(steps) > ROI_SAMPLE_FRAMES:
            steps = [steps[0], steps[len(steps) // 2], steps[-1]]
        imgs = [cv2.imread(G4.image_path(con, desktop, s), cv2.IMREAD_COLOR) for s in steps]
        box = tuple(int(v) for v in suggest_roi_over([i for i in imgs if i is not None], VIEW))
        _SUGGESTED[desktop] = {"roi": box, "sample_steps": steps}
        print(f"[l6] D{desktop} {VIEW}: no stored ROI; the app's proposal from steps {steps} "
              f"is {box}", flush=True)
    return tuple(_SUGGESTED[desktop]["roi"]), "suggest_roi_over"


def interior_point_crop(crop: np.ndarray, x0: int, y0: int) -> tuple[int, int]:
    """L5's ``interior_point`` computed on the bbox crop (same pixel: the zero ring
    around the crop stands for everything outside it)."""
    x, y = G5.interior_point(crop)
    return x0 + x, y0 + y


def build(check: bool = True) -> dict:
    from tda.core import masks

    doc = json.loads(COCO.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in doc["categories"]}
    images = {im["id"]: im for im in doc["images"]}
    anns = defaultdict(list)
    for a in doc["annotations"]:
        anns[a["image_id"]].append(a)
    con = env.connect()
    drawn = _drawn_steps(con)
    frames: list[G4.Frame] = []
    recs: list[dict] = []
    img_paths: dict[tuple[int, int], str] = {}
    skipped = Counter()
    roi_source = Counter()
    not_verified_frames = []
    for img_id in sorted(images, key=lambda i: (images[i]["extra"]["desktop"],
                                               images[i]["extra"]["step"])):
        im = images[img_id]
        d, s = int(im["extra"]["desktop"]), int(im["extra"]["step"])
        if im["extra"].get("review_status") != "verified":
            not_verified_frames.append(f"D{d}s{s}")
            continue
        roi, how = roi_for(con, d, s)
        roi_source[(d, how)] += 1
        path = G4.image_path(con, d, s)
        if path is None:
            raise SystemExit(f"D{d} scan s{s}: no image")
        img_paths[(d, s)] = path
        fr = G4.Frame(desktop=d, step=s, roi=roi, image=path)
        for a in sorted(anns.get(img_id, []), key=lambda a: a["attributes"]["instance_key"]):
            at = a["attributes"]
            vis = str(at.get("visibility") or "")
            if a.get("iscrowd"):
                skipped["crowd"] += 1
                continue
            if at.get("context"):
                skipped["context"] += 1
                continue
            if at.get("placement") != "in_chassis":
                skipped[f"placement:{at.get('placement')}"] += 1
                continue
            if vis not in KEEP_VIS:
                skipped[vis or "none"] += 1
                continue
            seg = a.get("segmentation")
            if not seg or not isinstance(seg, dict):
                skipped[f"{vis}:no_mask"] += 1
                continue
            area = int(masks.rle_area(seg))
            box = masks.rle_bbox(seg)
            if area <= 0 or box is None:
                skipped[f"{vis}:empty"] += 1
                continue
            inst = str(at["instance_key"])
            cls = cats[a["category_id"]]
            box = tuple(int(v) for v in box)
            fr.boxes.append(G4.GtBox(desktop=d, step=s, instance=inst, cls=cls, visibility=vis,
                                     area=area, box=box))
            m = masks.decode_rle(seg).astype(bool)
            x0, y0, x1, y1 = box
            crop = m[y0:y1, x0:x1]
            recs.append({
                "desktop": d, "step": s, "instance": inst, "cls": cls, "visibility": vis,
                "occlusion": float(at.get("occlusion_ratio") or 0.0), "area": area,
                "bbox": box, "point": interior_point_crop(crop, x0, y0),
                "drawn": s in drawn.get((d, inst), set()),
                "frame_hw": (int(m.shape[0]), int(m.shape[1])),
                "packed": np.packbits(crop)})
        frames.append(fr)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    G4.save(frames, GT_DET)
    with open(GT_SAM, "wb") as fh:
        pickle.dump({"recs": recs, "images": img_paths}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # spot-check the crop click point against L5's full-frame one
    rng = np.random.default_rng(0)
    for i in rng.choice(len(recs), size=min(40, len(recs)), replace=False):
        r = G5.Rec(**recs[int(i)])
        full = np.zeros(r.frame_hw, bool)
        x0, y0, x1, y1 = r.bbox
        full[y0:y1, x0:x1] = r.mask_crop()
        if G5.interior_point(full) != tuple(r.point):
            raise SystemExit(f"click point differs for {r.key} s{r.step}")

    cross = cross_check(frames, con) if check else {}
    con.close()
    desktops = sorted({f.desktop for f in frames})
    per_d = Counter(f.desktop for f in frames)
    box_cnt = Counter((f.desktop, b.cls) for f in frames for b in f.boxes)
    inst_cnt = defaultdict(set)
    for f in frames:
        for b in f.boxes:
            inst_cnt[b.cls].add((f.desktop, b.instance))
    classes = sorted({c for _d, c in box_cnt}, key=lambda c: -sum(
        v for (d, cc), v in box_cnt.items() if cc == c))
    manifest = {
        "coco": str(COCO), "coco_sha256": sha256_file(COCO),
        "coco_info": doc.get("info"),
        "source_db": str(env.LIVE_COPY),
        "frames": {f"D{d}": per_d[d] for d in desktops}, "n_frames": len(frames),
        "frames_with_verified_rows_not_verified": not_verified_frames,
        "boxes": {c: {f"D{d}": box_cnt[(d, c)] for d in desktops if box_cnt[(d, c)]}
                  for c in classes},
        "boxes_total": {c: sum(v for (d, cc), v in box_cnt.items() if cc == c) for c in classes},
        "instances": {c: len(inst_cnt[c]) for c in classes},
        "desktops_with_class": {c: len({d for d, _i in inst_cnt[c]}) for c in classes},
        "masks": len(recs),
        "roi_source": {f"D{d}:{how}": n for (d, how), n in sorted(roi_source.items())
                       if how != "stored"} or "every frame's stored pose-segment ROI",
        "suggested_rois": {f"D{d}": v for d, v in _SUGGESTED.items()},
        "boxes_centre_outside_roi": {
            f"D{d}": n for d, n in Counter(
                f.desktop for f in frames for b in f.boxes
                if not (f.roi[0] <= (b.box[0] + b.box[2]) / 2 < f.roi[2]
                        and f.roi[1] <= (b.box[1] + b.box[3]) / 2 < f.roi[3])).items()},
        "skipped": dict(skipped),
        "kept_visibility": dict(Counter(b.visibility for f in frames for b in f.boxes)),
        "extra_classes": {c: {"boxes": sum(v for (d, cc), v in box_cnt.items() if cc == c),
                              "instances": len(inst_cnt[c]),
                              "desktops": sorted({d for d, _i in inst_cnt[c]}),
                              "held_out_boxes": {f"D{h}": box_cnt[(h, c)] for h in HELD_OUT}}
                          for c in EXTRA_CLASSES},
        "cross_check": cross,
        "gt_det": str(GT_DET), "gt_det_sha256": sha256_file(GT_DET),
        "gt_sam": str(GT_SAM),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def cross_check(frames: list, con) -> dict:
    """The same frames read straight from the copy's compiled_mask rows (L4's ``gt.build``)."""
    saved, saved_roi = G4.DESKTOPS, G4.roi_of
    G4.DESKTOPS = tuple(sorted({f.desktop for f in frames}))
    G4.roi_of = lambda c, d, s, view=VIEW: roi_for(c, d, s)[0]
    try:
        direct = G4.build(con)
    finally:
        G4.DESKTOPS, G4.roi_of = saved, saved_roi
    a = {(f.desktop, f.step): f for f in frames}
    b = {(f.desktop, f.step): f for f in direct}
    only_db = sorted(set(b) - set(a))
    only_coco = sorted(set(a) - set(b))
    diffs = []
    same = 0
    for k in sorted(set(a) & set(b)):
        sa = {(x.instance, x.cls, x.visibility, x.area, tuple(x.box)) for x in a[k].boxes}
        sb = {(x.instance, x.cls, x.visibility, x.area, tuple(x.box)) for x in b[k].boxes}
        if sa == sb and tuple(a[k].roi) == tuple(b[k].roi):
            same += 1
        else:
            diffs.append({"frame": f"D{k[0]}s{k[1]}", "only_coco": sorted(map(str, sa - sb))[:5],
                          "only_db": sorted(map(str, sb - sa))[:5]})
    out = {"frames_same": same, "frames_differ": len(diffs), "only_in_db": [f"D{d}s{s}" for d, s in only_db],
           "only_in_coco": [f"D{d}s{s}" for d, s in only_coco], "differences": diffs[:30]}
    print(f"[l6] cross-check against the DB rows: {same} frames identical, {len(diffs)} differ, "
          f"{len(only_db)} only in the DB, {len(only_coco)} only in the export", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="skip the export, build from the COCO file")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args(argv)
    info = {}
    if not args.build:
        info = export()
    elif EXPORT_INFO.exists():
        info = json.loads(EXPORT_INFO.read_text(encoding="utf-8"))
    man = build(check=not args.no_check)
    if info:
        man["export"] = info
        MANIFEST.write_text(json.dumps(man, indent=1), encoding="utf-8")
    show = {k: v for k, v in man.items() if k not in ("cross_check",)}
    print(json.dumps(show, indent=1)[:6000])
    print(json.dumps(man["cross_check"], indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
