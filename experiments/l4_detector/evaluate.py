"""Score one model's detections on the verified scan frames: detector metrics and the guess.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.evaluate --model production --desktops 2,3,4,5,13

Detector metrics -- L2's ``det_eval.evaluate`` unchanged (AP50, R50@0.10,
hit@0.10 = a same-class detection at conf >= 0.10 whose box centre lies in the
GT box, FP/frame@0.10), per desktop, against the verified boxes (``gt.py``);
plus hit / precision curves over conf (precision = 1 - L2's FP share: a
detection is false when its centre is in no same-class GT box and its IoU
with every one is < 0.3).

The guess (``decision_rule.txt``) on the removal events of ``gt.events``:
candidates = detections of the event's class on frame j with conf >= the
threshold, centre inside the ROI, minus the app's ``already_drawn`` against
the other same-class instances visible on j; ranked by the app's
``box_change(j, k)``; rank-1 hit = its centre in the target's visible mask on
j.  Thresholds 0.05 / 0.10 / 0.25.  The difference map's rank 1 (L2's M0app:
``diff_delta_e(k, j, roi, max_side=1600)`` + ``diff_blobs``) is computed once
per event and cached.

Writes ``eval/<model>/det_metrics.csv``, ``curves.csv`` and ``guess_events.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l4_detector import env  # noqa: E402
from experiments.l4_detector import gt as G  # noqa: E402
from experiments.l4_detector.infer import dets_path  # noqa: E402

THRESHOLDS = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50)
GUESS_CONFS = (0.05, 0.10, 0.25)
EVAL = env.OUT / "eval"
M0_CACHE = EVAL / "m0app.json"


def by_frame(frames) -> dict:
    return {(f.desktop, "scan", f.step): f.boxes for f in frames}


def det_rows(dets: dict, frames, desktops) -> list[dict]:
    from experiments.l2_detector.det_eval import evaluate

    by = by_frame(frames)
    rows = []
    for d in desktops:
        sub = {k: v for k, v in dets.items() if int(k.split(":")[0]) == d}
        if sub:
            rows += evaluate(sub, d, by)
    if len(desktops) > 1:
        sub = {k: v for k, v in dets.items() if int(k.split(":")[0]) in desktops}
        rows += evaluate(sub, "pooled", by)
    return rows


def curves(dets: dict, frames, desktops) -> list[dict]:
    from experiments.l2_detector.det_eval import centre_in, iou_mat

    by = by_frame(frames)
    acc = defaultdict(lambda: {"n_gt": 0, "hit": 0, "n_det": 0, "fp": 0, "frames": 0})
    for key, rows in dets.items():
        d, _v, s = key.split(":")
        d, s = int(d), int(s)
        if d not in desktops:
            continue
        gts = by.get((d, "scan", s), [])
        for cls in G.DET_CLASSES:
            g = np.asarray([b.box for b in gts if b.cls == cls], float).reshape(-1, 4)
            dr = [r for r in rows if r[4] == cls]
            db = np.asarray([r[:4] for r in dr], float).reshape(-1, 4)
            dc = np.asarray([r[5] for r in dr], float)
            for thr in THRESHOLDS:
                for grp in (d, "pooled"):
                    a = acc[(grp, cls, thr)]
                    a["frames"] += 1
                    a["n_gt"] += len(g)
                    keep = dc >= thr
                    a["n_det"] += int(keep.sum())
                    if not keep.any():
                        continue
                    cin = centre_in(db[keep], g)
                    if len(g):
                        a["hit"] += int(np.sum(cin.any(axis=0)))
                        fp = ~cin.any(axis=1) & (iou_mat(db[keep], g).max(axis=1) < 0.3)
                    else:
                        fp = np.ones(int(keep.sum()), bool)
                    a["fp"] += int(fp.sum())
    out = []
    for (grp, cls, thr), a in sorted(acc.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
        if grp == "pooled" and len(desktops) == 1:
            continue
        out.append({"desktop": grp, "cls": cls, "conf": thr, "n_gt": a["n_gt"] // 1,
                    "hit": round(100 * a["hit"] / a["n_gt"], 1) if a["n_gt"] else "",
                    "precision": round(100 * (1 - a["fp"] / a["n_det"]), 1) if a["n_det"] else "",
                    "det_per_frame": round(a["n_det"] / max(1, a["frames"]), 2),
                    "fp_per_frame": round(a["fp"] / max(1, a["frames"]), 2)})
    return out


class Frames:
    """Decoded frames of the verified set, a few at a time."""

    def __init__(self, frames, keep: int = 6) -> None:
        from experiments.l2_detector import data as D

        self.read = D.read_rgb
        self.path = {(f.desktop, f.step): f.image for f in frames}
        self.cache: dict = {}
        self.order: list = []
        self.keep = keep

    def get(self, d: int, s: int):
        k = (d, s)
        if k not in self.cache:
            self.cache[k] = self.read(self.path[k])
            self.order.append(k)
            while len(self.order) > self.keep:
                self.cache.pop(self.order.pop(0), None)
        return self.cache[k]


def m0app(events, frames, con) -> dict:
    """The difference map's rank 1 per event (model-independent), cached."""
    from experiments.l1_localise import methods as M
    from tda.core.diffmap import diff_blobs, diff_delta_e

    cache = json.loads(M0_CACHE.read_text(encoding="utf-8")) if M0_CACHE.exists() else {}
    todo = [e for e in events if "a1" not in cache.get(e.key, {})]
    if todo:
        fr = Frames(frames)
        for ev in todo:
            img_j, img_k = fr.get(ev.desktop, ev.j), fr.get(ev.desktop, ev.step)
            mask = G.visible_mask(con, ev.desktop, ev.j, ev.target)
            props = M.blobs_to_props(diff_blobs(diff_delta_e(img_k, img_j, roi=ev.roi,
                                                             max_side=M.MAX_SIDE)))
            hits = [bool(mask[int(round(p.point[1])), int(round(p.point[0]))]) for p in props]
            # a1: the rank-1 blob's pixel count -- what the U2h gate compares with the bands
            cache[ev.key] = {"in1": int(bool(hits[:1]) and hits[0]), "in3": int(any(hits[:3])),
                             "p1": [round(v, 1) for v in props[0].point] if props else None,
                             "a1": int(props[0].area) if props else None}
        M0_CACHE.parent.mkdir(parents=True, exist_ok=True)
        M0_CACHE.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return cache


def guess(dets: dict, frames, desktops, con) -> list[dict]:
    from tda.models.detector import box_change
    from tda.ui.app_detect import already_drawn

    evs, _why = G.events(frames, con)
    evs = [e for e in evs if e.desktop in desktops]
    m0 = m0app(evs, frames, con)
    fby = G.index(frames)
    fr = Frames(frames)
    rows = []
    for ev in evs:
        cls = ev.cls
        row = {"ev": ev.key, "desktop": ev.desktop, "step": ev.step, "verb": ev.verb,
               "target": ev.target, "cls": cls, "width": round(ev.width, 1),
               "small": int(cls in G.SMALL_CLASSES),
               "M0app_in1": m0[ev.key]["in1"], "M0app_in3": m0[ev.key]["in3"]}
        drow = dets.get(f"{ev.desktop}:scan:{ev.j}")
        if cls in G.DET_CLASSES and drow is not None:
            mask = G.visible_mask(con, ev.desktop, ev.j, ev.target)
            img_j, img_k = fr.get(ev.desktop, ev.j), fr.get(ev.desktop, ev.step)
            drawn = G.drawn_on_j(fby, ev, cls)
            roi = ev.roi
            cands = []
            for r in drow:
                if r[4] != cls or r[5] < min(GUESS_CONFS):
                    continue
                box = tuple(float(int(round(v))) for v in r[:4])
                cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
                if not (roi[0] <= cx < roi[2] and roi[1] <= cy < roi[3]):
                    continue
                if already_drawn(box, drawn):
                    continue
                hit = bool(mask[min(mask.shape[0] - 1, int(cy)), min(mask.shape[1] - 1, int(cx))])
                cands.append({"conf": float(r[5]), "box": box, "hit": hit,
                              "change": box_change(img_j, img_k, tuple(int(v) for v in box))})
            for thr in GUESS_CONFS:
                pool = [c for c in cands if c["conf"] >= thr]
                pool.sort(key=lambda c: -c["change"])     # stable: conf order on ties
                tag = f"c{int(round(thr * 100)):02d}"
                row[f"n_{tag}"] = len(pool)
                row[f"ceil_{tag}"] = int(any(c["hit"] for c in pool))
                row[f"in1_{tag}"] = int(bool(pool) and pool[0]["hit"])
                row[f"in3_{tag}"] = int(any(c["hit"] for c in pool[:3]))
                row[f"rank_{tag}"] = next((i + 1 for i, c in enumerate(pool) if c["hit"]), 0)
                if thr == 0.10:
                    row["top3"] = json.dumps([[list(map(int, c["box"])), round(c["conf"], 3),
                                               round(c["change"], 2), int(c["hit"])]
                                              for c in pool[:3]])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--desktops", default="2,3,4,5,13")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--skip-guess", action="store_true")
    args = ap.parse_args(argv)
    env.bind_l2(env.OUT / "l2bind")
    desktops = [int(d) for d in args.desktops.split(",")]
    p = dets_path(args.model, "scan", args.conf)
    dets = json.loads(p.read_text(encoding="utf-8"))
    frames = G.load()
    tag = p.parent.name
    out = EVAL / tag
    out.mkdir(parents=True, exist_ok=True)
    rows = det_rows(dets, frames, desktops)
    write_csv(out / "det_metrics.csv", rows)
    write_csv(out / "curves.csv", curves(dets, frames, desktops))
    for r in rows:
        if r["cls"] in ("screw", "connector", "ram_latch") or r["desktop"] == "pooled":
            print(f"  {tag} D{r['desktop']} {r['cls']:>16} n={r['n_gt']:>4} AP50={r['AP50']} "
                  f"hit={r['hit@0.10']} FP/f={r['FP/frame@0.10']}")
    if not args.skip_guess:
        con = G.connect()
        g = guess(dets, frames, desktops, con)
        con.close()
        write_csv(out / "guess_events.csv", g)
        sc = [r for r in g if r["cls"] == "screw"]
        print(f"  {tag} screw events {len(sc)}: rank-1 "
              f"{sum(r.get('in1_c10', 0) for r in sc)}, ceiling "
              f"{sum(r.get('ceil_c10', 0) for r in sc)}, M0app {sum(r['M0app_in1'] for r in sc)}")
    print(f"[l4] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
