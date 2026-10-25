"""Round 2: a screw-only confidence floor on the fold model F, from the cached detections.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.round2_floor

No retraining, no GPU.  The detections are ``dets/<model>/scan.json`` of
round 1 (every box >= 0.001 into the merge, L4's protocol).  For each screw
floor f in 0.05 (round 1), 0.10, 0.15, 0.20, 0.25, F's screw boxes below f are
dropped and every other class stays at 0.05, then:

* per class and desktop, recall / precision at 0.05 (L4's curves, raw counts:
  ``eval_det.counts``) -- for screws this is recall / precision at f;
* L4's guess (``evaluate.guess``: the app's ``already_drawn`` and
  ``box_change`` ranking) rank-1 per class on the removal events of the armed
  classes (screw 15, connector 25, ram_module 5, psu_latch 1, cpu 2), read at
  conf 0.05 -- a screw event's candidates are then the screw boxes >= f;
* per frame: the detections the app would store (>= 0.05, every class: the
  floor applies after the cache, so the cache is unchanged), the armed-class
  candidates left after the floor, and the screw FP among them.

The comparators are round 1's, at 0.05: v2h on D19, the shipped v2 on D27.
The criterion (the controller's, round 2): the smallest floor at which F's
screw rank-1 is >= the comparator's on BOTH desktops (D19 6/7, D27 8/8) while
connectors keep >= 10/25 and ram_module >= 3/5.  Writes
``eval/round2_floor.json`` and prints the table.
"""
from __future__ import annotations

from experiments.l6_models import env  # noqa: F401  (side effects first)

import json
import sys
from collections import defaultdict

from experiments.l4_detector import evaluate as E4
from experiments.l4_detector import gt as G4
from experiments.l6_models import eval_det as ED
from experiments.l6_models.export_gt import GT_DET, HELD_OUT
from experiments.l6_models.infer_det import dets_path

#: The controller's floors (0.10-0.25), round 1's 0.05, and a finer sweep below
#: 0.10 (the one D27 screw removal F loses above 0.078 is scored 0.078).
FLOORS = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20, 0.25)
ARMED = ED.ARMED
CMP = {19: "v2h", 27: "v2"}
NEED = {"screw": {19: 6, 27: 8}, "connector": 10, "ram_module": 3}


def floored(dets: dict, floor: float, cls: str = "screw") -> dict:
    return {k: [r for r in rows if not (r[4] == cls and r[5] < floor)] for k, rows in dets.items()}


def rank1(rows: list[dict], tag: str = "c05") -> dict:
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["cls"] not in ARMED:
            continue
        k = (int(r["desktop"]), r["cls"])
        out[k][0] += int(r.get(f"in1_{tag}", 0) or 0)
        out[k][1] += 1
    return dict(out)


def per_frame(dets: dict, frames, floor: float) -> dict:
    """Stored (>= 0.05, every class), armed candidates after the floor, screw FP after it."""
    from experiments.l2_detector.det_eval import centre_in, iou_mat
    import numpy as np

    by = E4.by_frame(frames)
    acc = {d: {"frames": 0, "stored": 0, "candidates": 0, "screw_cand": 0, "screw_fp": 0}
           for d in HELD_OUT}
    for key, rows in dets.items():
        d, _v, s = key.split(":")
        d, s = int(d), int(s)
        if d not in acc:
            continue
        a = acc[d]
        a["frames"] += 1
        a["stored"] += sum(1 for r in rows if r[5] >= 0.05)
        cand = [r for r in rows if r[4] in ARMED and r[5] >= (floor if r[4] == "screw" else 0.05)]
        a["candidates"] += len(cand)
        sc = np.asarray([r[:4] for r in cand if r[4] == "screw"], float).reshape(-1, 4)
        g = np.asarray([b.box for b in by.get((d, "scan", s), []) if b.cls == "screw"],
                       float).reshape(-1, 4)
        a["screw_cand"] += len(sc)
        if len(sc):
            if len(g):
                fp = ~centre_in(sc, g).any(axis=1) & (iou_mat(sc, g).max(axis=1) < 0.3)
            else:
                fp = np.ones(len(sc), bool)
            a["screw_fp"] += int(fp.sum())
    return {f"D{d}": {"frames": a["frames"],
                      "stored_per_frame": round(a["stored"] / a["frames"], 2),
                      "armed_candidates_per_frame": round(a["candidates"] / a["frames"], 2),
                      "screw_candidates_per_frame": round(a["screw_cand"] / a["frames"], 2),
                      "screw_fp_per_frame": round(a["screw_fp"] / a["frames"], 2)}
            for d, a in acc.items()}


def main() -> int:
    frames = G4.load(GT_DET)
    G4.DESKTOPS = tuple(sorted({f.desktop for f in frames}))
    E4.EVAL = ED.EVAL
    E4.M0_CACHE = ED.EVAL / "m0app.json"
    desktops = list(HELD_OUT)
    con = env.connect()
    raw = {m: json.loads(dets_path(m).read_text(encoding="utf-8")) for m in ("F", "v2h", "v2")}
    cmp_r1, cmp_cnt = {}, {}
    for m in ("v2h", "v2"):
        r1 = rank1(E4.guess(raw[m], frames, desktops, con))
        c = ED.counts(raw[m], frames, desktops)
        for d, mm in CMP.items():
            if mm == m:
                cmp_r1.update({k: v for k, v in r1.items() if k[0] == d})
                cmp_cnt.update({k: v for k, v in c.items() if k[0] == d})
    cmp_pf = {}
    for d, m in CMP.items():
        cmp_pf.update({k: v for k, v in per_frame(raw[m], frames, 0.05).items() if k == f"D{d}"})
    results = {"comparators": {f"D{d}": m for d, m in CMP.items()},
               "comparator": {"rank1": {f"D{d}:{c}": v for (d, c), v in cmp_r1.items()},
                              "counts": {f"D{d}:{c}": v for (d, c), v in cmp_cnt.items()
                                         if c in ARMED},
                              "per_frame": cmp_pf},
               "floors": {}}
    chosen = None
    for f in FLOORS:
        dets = floored(raw["F"], f)
        r1 = rank1(E4.guess(dets, frames, desktops, con))
        c = ED.counts(dets, frames, desktops)
        sc = {d: r1.get((d, "screw"), [0, 0])[0] for d in HELD_OUT}
        conn = sum(r1.get((d, "connector"), [0, 0])[0] for d in HELD_OUT)
        ram = sum(r1.get((d, "ram_module"), [0, 0])[0] for d in HELD_OUT)
        ok = (all(sc[d] >= NEED["screw"][d] for d in HELD_OUT) and conn >= NEED["connector"]
              and ram >= NEED["ram_module"])
        results["floors"][f"{f:.2f}"] = {
            "rank1": {f"D{d}:{cc}": v for (d, cc), v in r1.items()},
            "rank1_pooled": {cc: [sum(r1.get((d, cc), [0, 0])[0] for d in HELD_OUT),
                                  sum(r1.get((d, cc), [0, 0])[1] for d in HELD_OUT)]
                             for cc in ARMED},
            "counts": {f"D{d}:{cc}": {**v, "recall": ED._r(ED.recall(v)),
                                       "precision": ED._r(ED.precision(v))}
                       for (d, cc), v in c.items() if cc in ARMED},
            # the cache holds every box >= 0.05 whatever the floor: per_frame floors itself
            "per_frame": per_frame(raw["F"], frames, f),
            "meets_criterion": ok}
        if ok and chosen is None:
            chosen = f
    con.close()
    results["chosen_floor"] = chosen
    (ED.EVAL / "round2_floor.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    # ---- the table -------------------------------------------------------------
    lines = ["| screw floor | screw D19 R/P | screw D27 R/P | rank-1 screw D19 | D27 | "
             "connector (25) | ram_module (5) | psu_latch | cpu | all 48 | meets |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def rp(cnt, d, cc):
        v = cnt[f"D{d}:{cc}"]
        return f"{ED._fmt(v['recall'])}/{ED._fmt(v['precision'])}"

    for f, r in results["floors"].items():
        pr = r["rank1_pooled"]
        tot = sum(v[0] for v in pr.values())
        lines.append(
            f"| F, {f} | {rp(r['counts'], 19, 'screw')} | {rp(r['counts'], 27, 'screw')} | "
            f"{r['rank1'].get('D19:screw', [0, 0])[0]}/7 | {r['rank1'].get('D27:screw', [0, 0])[0]}/8 | "
            f"{pr['connector'][0]} | {pr['ram_module'][0]} | {pr['psu_latch'][0]}/1 | "
            f"{pr['cpu'][0]}/2 | {tot} | {'yes' if r['meets_criterion'] else 'no'} |")
    cr = results["comparator"]["rank1"]
    cc = results["comparator"]["counts"]

    def crp(d, k):
        v = cc[f"D{d}:{k}"]
        return f"{ED._fmt(ED.recall(v))}/{ED._fmt(ED.precision(v))}"

    def pooled(k):
        return sum(cr.get(f"D{d}:{k}", [0, 0])[0] for d in HELD_OUT)

    lines.append(f"| comparators, 0.05 | {crp(19, 'screw')} | {crp(27, 'screw')} | "
                 f"{cr['D19:screw'][0]}/7 | {cr['D27:screw'][0]}/8 | {pooled('connector')} | "
                 f"{pooled('ram_module')} | {pooled('psu_latch')}/1 | {pooled('cpu')}/2 | "
                 f"{sum(pooled(k) for k in ARMED)} | – |")
    lines.append("")
    lines.append("| | D19 stored / armed cand. / screw cand. / screw FP per frame | D27 |")
    lines.append("|---|---|---|")
    for f, r in results["floors"].items():
        p = r["per_frame"]
        lines.append(f"| F, {f} | " + " | ".join(
            f"{p[k]['stored_per_frame']} / {p[k]['armed_candidates_per_frame']} / "
            f"{p[k]['screw_candidates_per_frame']} / {p[k]['screw_fp_per_frame']}"
            for k in ("D19", "D27")) + " |")
    p = results["comparator"]["per_frame"]
    lines.append("| comparators, 0.05 | " + " | ".join(
        f"{p[k]['stored_per_frame']} / {p[k]['armed_candidates_per_frame']} / "
        f"{p[k]['screw_candidates_per_frame']} / {p[k]['screw_fp_per_frame']}"
        for k in ("D19", "D27")) + " |")
    lines.append(f"\nchosen floor: {chosen}")
    print("\n".join(lines))
    (ED.EVAL / "round2_floor_table.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
