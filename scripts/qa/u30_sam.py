"""U30-QA, pass 4: SAM agreement triage (GPU; only while the annotator is closed).

For every (frame, instance) whose stored label is ``visible`` or
``occluded_partial``: prompt the shipped SAM service -- SAM 2.1 large with the
fine-tuned head of ``configs/sam.yaml``, built the way the app builds it -- with
one box, the part's amodal bounding box, on a production-like viewport around it
(L5's window: 1.4 x the box, 512..1024 px, the app's multimask first prompt and
its shown candidate), and record IoU(SAM, stored visible) and IoU(SAM, amodal).
The stock head answers the same prompt on the same embedding too, because the
shipped head was trained on the verified scan masks of D2-D17, D19-D21 and D27:
on those it may simply repeat the annotation.

**Triage, not truth**: SAM given a box segments *something* in the box -- a
low IoU says "look", nothing more.

Before every desktop the script looks for a running ``tda.cli ... app``
process and stops (exit 3) if there is one; results are per desktop
(``<root>/sam/D<nn>.json``) and a re-run skips the desktops already done.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tda.core import masks as M  # noqa: E402
import u30_common as U  # noqa: E402

LABELS = ("visible", "occluded_partial")
MIN_WINDOW, MAX_WINDOW, CONTEXT = 512, 1024, 1.4


class AppOpened(RuntimeError):
    """The annotator was started while a desktop was being prompted."""


def app_running() -> bool:
    import psutil

    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = " ".join(p.info.get("cmdline") or [])
        except Exception:  # noqa: BLE001
            continue
        if name.startswith("python") and "tda.cli" in cmd and " app" in cmd:
            return True
    return False


def window(bbox, hw):
    """L5's crop_window: ``(x0, y0, x1, y1, scale)``."""
    h, w = hw
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    longest = max(x1 - x0, y1 - y0)
    side = int(round(longest * CONTEXT))
    scale = 1.0
    if side > MAX_WINDOW:
        side = min(int(round(longest * 1.1)), min(h, w))
        scale = MAX_WINDOW / float(side)
    side = max(MIN_WINDOW, side)
    side = min(side, min(h, w))
    wx0 = max(0, min(int(round(cx - side / 2.0)), w - side))
    wy0 = max(0, min(int(round(cy - side / 2.0)), h - side))
    return wx0, wy0, wx0 + side, wy0 + side, scale


def iou_w(sam, ref_w, ref_out: int) -> float:
    """IoU of a window-sized SAM mask with a reference whose window part is ``ref_w``
    and which has ``ref_out`` more pixels outside the window."""
    inter = int((sam & ref_w).sum())
    union = int((sam | ref_w).sum()) + ref_out
    return inter / union if union else 0.0


def build_service():
    from tda.cli_common import load_paths
    from tda.models import sam_heads, sam_service

    paths = load_paths(str(REPO / "configs" / "paths.yaml"))
    checkpoint = sam_service.checkpoint_in(paths)
    heads = sam_heads.load_sam_head_config(weights_dir=paths.get("weights_dir"))
    svc = (sam_service.SamService(checkpoint=str(checkpoint), heads=heads)
           if checkpoint is not None else sam_service.SamService(heads=heads))
    return svc


def run_desktop(svc, d: int) -> list[dict]:
    from tda.models.sam_service import SamRequest

    res = U.load(d)
    rows = [r for r in res["rows"]
            if r.get("visibility") in LABELS and r.get("amodal_bbox")
            and (r.get("placement") or r.get("placement_state")) == "in_chassis"
            and (r.get("vis_area") or 0) > 0]
    rows.sort(key=lambda r: (r["step"], r["inst"]))
    out = []
    img_step, rgb = None, None
    routing = svc._routing
    for n, r in enumerate(rows):
        if n and n % 150 == 0 and app_running():
            raise AppOpened(f"D{d:02d}")
        step, inst = r["step"], r["inst"]
        if step != img_step:
            bgr = U.color(res, step)
            rgb = None if bgr is None else np.ascontiguousarray(bgr[:, :, ::-1])
            img_step = step
        if rgb is None:
            continue
        rl = res["rles"][(step, inst)]
        vis = U.decode(rl["visible"])
        am = U.decode(rl["amodal"])
        x0, y0, x1, y1, scale = window(r["amodal_bbox"], rgb.shape[:2])
        patch = rgb[y0:y1, x0:x1]
        side = x1 - x0
        if scale < 1.0:
            patch = cv2.resize(patch, (MAX_WINDOW, MAX_WINDOW), interpolation=cv2.INTER_AREA)
        patch = np.ascontiguousarray(patch)
        bx0, by0, bx1, by1 = r["amodal_bbox"]
        box = ((min(max(bx0, x0), x1) - x0) * scale, (min(max(by0, y0), y1) - y0) * scale,
               (min(max(bx1, x0), x1) - x0) * scale, (min(max(by1, y0), y1) - y0) * scale)
        vis_w, am_w = vis[y0:y1, x0:x1], am[y0:y1, x0:x1]
        vis_out = int(vis.sum()) - int(vis_w.sum())
        am_out = int(am.sum()) - int(am_w.sum())
        req = SamRequest(image_crop=patch, box=box, multimask=True, instance_cls=r.get("cls"))
        got = {}
        for head in ("tuned", "stock"):
            svc._routing = routing if head == "tuned" else None
            try:
                ans = svc.predict(req)
            finally:
                svc._routing = routing
            cands = []
            for m in ans.candidates:
                if m.shape != (side, side):
                    m = cv2.resize(m.astype(np.uint8), (side, side),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                cands.append(m)
            shown = cands[0]
            full = np.zeros(vis.shape, bool, order="F")
            full[y0:y1, x0:x1] = shown
            got[head] = {
                "head": ans.head, "score": round(float(ans.score), 4),
                "iou_vis": round(iou_w(shown, vis_w, vis_out), 4),
                "iou_amodal": round(iou_w(shown, am_w, am_out), 4),
                "oracle_vis": round(max(iou_w(c, vis_w, vis_out) for c in cands), 4),
                "area": int(shown.sum()),
                "rle": M.encode_rle(full, window=(x0, y0, x1, y1)) if head == "tuned" else None,
            }
        out.append({"desktop": d, "step": step, "inst": inst, "cls": r.get("cls"),
                    "visibility": r.get("visibility"), "occ": r.get("occ_stored"),
                    "vis_area": r.get("vis_area"), "amodal_area": r.get("amodal_area"),
                    "kf_source": r.get("kf_source"), "window": [x0, y0, x1, y1],
                    "tuned": got["tuned"], "stock": got["stock"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--desktops", default=None)
    ap.add_argument("--force", action="store_true", help="redo desktops already done")
    args = ap.parse_args(argv)
    out_dir = os.path.join(U.QA_ROOT, "sam")
    os.makedirs(out_dir, exist_ok=True)
    wanted = U.DESKTOPS if not args.desktops else [int(x) for x in args.desktops.split(",")]
    todo = [d for d in wanted
            if args.force or not os.path.isfile(os.path.join(out_dir, f"D{d:02d}.json"))]
    if not todo:
        print("nothing to do")
        return 0
    if app_running():
        print("the annotator is open: not touching the GPU (exit 3)")
        return 3
    svc = build_service()
    print(f"SAM heads: {svc.heads}, off reason: {svc.head_off_reason!r}, "
          f"tuned file: {svc.tuned_file}", flush=True)
    for d in todo:
        if app_running():
            print(f"the annotator opened: stopping before D{d:02d} (exit 3)", flush=True)
            return 3
        t0 = time.time()
        try:
            rows = run_desktop(svc, d)
        except AppOpened:
            print(f"the annotator opened during D{d:02d}: stopped, nothing written for it "
                  f"(exit 3)", flush=True)
            return 3
        with open(os.path.join(out_dir, f"D{d:02d}.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        tv = [r["tuned"]["iou_vis"] for r in rows]
        sv = [r["stock"]["iou_vis"] for r in rows]
        print(f"D{d:02d}: {len(rows)} prompts, median IoU(SAM, visible) tuned "
              f"{np.median(tv) if tv else float('nan'):.3f} / stock "
              f"{np.median(sv) if sv else float('nan'):.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
