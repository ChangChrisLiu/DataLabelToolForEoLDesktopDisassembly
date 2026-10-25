"""Score the detectors on D19 / D27 with L4's evaluation code, and apply gate D.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.eval_det

For every model with detections (``dets/<model>/scan.json``: v2, v2h, F, and
v1 / v1h / v3 when present):

* L4's ``evaluate.det_rows`` (L2's det_eval: AP50, R50@0.10, hit@0.10,
  FP/frame@0.10) and ``evaluate.curves`` (hit / precision over conf), per
  desktop -- unchanged, fed L6's ground truth;
* :func:`counts` -- the same curves logic (L4's ``curves``) kept as raw counts
  at conf 0.05: n_gt, hits, n_det, FP per (desktop, class) -- what the gate
  reads;
* L4's ``evaluate.guess`` on the removal events of D19 / D27 (the app's
  ``already_drawn`` and ``box_change``; conf 0.05 / 0.10 / 0.25) with the
  difference map's rank 1, cached in ``eval/m0app.json``.

Then gate D of ``decision_rule.txt`` (comparators: D19 F vs v2h, D27 F vs v2),
and the same tests with the shipped v2 as D19's comparator (reported, not
gated), the extra classes' rule, and ``eval/gate_det.json`` + ``eval/det.md``.
CPU only.
"""
from __future__ import annotations

from experiments.l6_models import env  # noqa: F401  (side effects first)

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.l4_detector import evaluate as E4
from experiments.l4_detector import gt as G4
from experiments.l6_models.export_gt import GT_DET, HELD_OUT
from experiments.l6_models.infer_det import dets_path

EVAL = env.ROOT / "eval"
CONF = 0.05
ARMED = ("screw", "connector", "ram_module", "psu_latch", "cpu")
EXTRA = ("ram_latch", "drive_latch", "cpu_socket_lever")
CMP = {19: "v2h", 27: "v2"}
CMP_SHIPPED = {19: "v2", 27: "v2"}
NEW = "F"


def counts(dets: dict, frames, desktops, conf: float = CONF) -> dict:
    """``{(desktop, cls): {n_gt, hits, n_det, fp, frames}}`` at ``conf`` (L4's curves, raw)."""
    from experiments.l2_detector.det_eval import centre_in, iou_mat

    by = E4.by_frame(frames)
    acc = defaultdict(lambda: {"n_gt": 0, "hits": 0, "n_det": 0, "fp": 0, "frames": 0})
    for key, rows in dets.items():
        d, _v, s = key.split(":")
        d, s = int(d), int(s)
        if d not in desktops:
            continue
        gts = by.get((d, "scan", s), [])
        for cls in G4.DET_CLASSES:
            g = np.asarray([b.box for b in gts if b.cls == cls], float).reshape(-1, 4)
            dr = [r for r in rows if r[4] == cls and r[5] >= conf]
            db = np.asarray([r[:4] for r in dr], float).reshape(-1, 4)
            a = acc[(d, cls)]
            a["frames"] += 1
            a["n_gt"] += len(g)
            a["n_det"] += len(db)
            if not len(db):
                continue
            cin = centre_in(db, g)
            if len(g):
                a["hits"] += int(np.sum(cin.any(axis=0)))
                fp = ~cin.any(axis=1) & (iou_mat(db, g).max(axis=1) < 0.3)
            else:
                fp = np.ones(len(db), bool)
            a["fp"] += int(fp.sum())
    return dict(acc)


def pooled(cnt: dict, keys) -> dict:
    out = {"n_gt": 0, "hits": 0, "n_det": 0, "fp": 0}
    for k in keys:
        for f in out:
            out[f] += cnt.get(k, {}).get(f, 0)
    return out


def recall(c: dict):
    return 100.0 * c["hits"] / c["n_gt"] if c["n_gt"] else None


def precision(c: dict):
    return 100.0 * (1 - c["fp"] / c["n_det"]) if c["n_det"] else None


def compare(new: dict, cmp: dict) -> dict:
    """The decision rule's WORSE / BETTER for one pair of count blocks."""
    tol = max(1.0, 0.02 * new["n_gt"])
    dh = new["hits"] - cmp["hits"]
    r_worse = -dh > tol
    r_better = dh > tol
    pn, pc = precision(new), precision(cmp)
    if pn is None or pc is None:
        dp = None
        p_worse = p_better = False
        if pn is not None and pc is None:   # detections where the comparator had none
            p_worse = new["fp"] >= 2
        if pc is not None and pn is None:
            p_better = cmp["fp"] >= 2
    else:
        dp = pn - pc
        p_worse = (-dp > 5.0) and (new["fp"] - cmp["fp"] >= 2)
        p_better = (dp > 5.0) and (cmp["fp"] - new["fp"] >= 2)
    return {"n_gt": new["n_gt"], "hits_new": new["hits"], "hits_cmp": cmp["hits"],
            "tol_boxes": round(tol, 2),
            "recall_new": _r(recall(new)), "recall_cmp": _r(recall(cmp)),
            "precision_new": _r(pn), "precision_cmp": _r(pc), "d_precision": _r(dp),
            "n_det_new": new["n_det"], "n_det_cmp": cmp["n_det"],
            "fp_new": new["fp"], "fp_cmp": cmp["fp"],
            "recall_worse": r_worse, "recall_better": r_better,
            "precision_worse": p_worse, "precision_better": p_better,
            "WORSE": r_worse or p_worse,
            "BETTER": (r_better and not p_worse) or (p_better and not r_worse)}


def _r(v, nd=1):
    return None if v is None else round(float(v), nd)


def gate(cnts: dict, cmp_of: dict) -> dict:
    """Gate D with the comparators ``cmp_of`` (desktop -> model)."""
    new = cnts[NEW]
    per = {}
    for d in HELD_OUT:
        for c in ARMED + EXTRA:
            per[f"D{d}:{c}"] = compare(new.get((d, c), pooled({}, [])),
                                       cnts[cmp_of[d]].get((d, c), pooled({}, [])))
    overall = {}
    for c in ARMED + EXTRA:
        n = pooled(new, [(d, c) for d in HELD_OUT])
        m = pooled({}, [])
        for d in HELD_OUT:
            for f in m:
                m[f] += cnts[cmp_of[d]].get((d, c), {}).get(f, 0)
        overall[c] = compare(n, m)
    sc27 = compare(pooled(new, [(27, "screw"), (27, "connector")]),
                   pooled(cnts[cmp_of[27]], [(27, "screw"), (27, "connector")]))
    d1_worse = [c for c in ARMED if per[f"D19:{c}"]["WORSE"]]
    d2_better = [c for c in ARMED if overall[c]["BETTER"]]
    d3 = sc27["BETTER"] and not per["D27:screw"]["WORSE"] and not per["D27:connector"]["WORSE"]
    res = {"comparators": {f"D{d}": m for d, m in cmp_of.items()},
           "D1": {"pass": not d1_worse, "worse_on_D19": d1_worse},
           "D2": {"pass": len(d2_better) >= 2, "better_overall": d2_better},
           "D3": {"pass": d3, "screw+connector_D27": sc27,
                  "screw_D27_worse": per["D27:screw"]["WORSE"],
                  "connector_D27_worse": per["D27:connector"]["WORSE"]},
           "per_desktop": per, "overall": overall}
    res["pass"] = res["D1"]["pass"] and res["D2"]["pass"] and res["D3"]["pass"]
    return res


def extra_rule(cnts: dict, guess_rows: dict, gate_pass: bool) -> dict:
    out = {}
    for c in EXTRA:
        rec = {}
        ok = gate_pass
        for d in HELD_OUT:
            cc = cnts[NEW].get((d, c), pooled({}, []))
            r = recall(cc)
            rec[f"D{d}"] = {"n_gt": cc["n_gt"], "recall": _r(r)}
            if cc["n_gt"] >= 10 and (r is None or r < 60.0):
                ok = False
        ev = [r for r in guess_rows.get(NEW, []) if r["cls"] == c]
        if ev:
            det1 = sum(int(r.get("in1_c05", 0)) for r in ev)
            m01 = sum(int(r["M0app_in1"]) for r in ev)
            rec["events"] = {"n": len(ev), "rank1_F_c05": det1, "rank1_diffmap": m01}
            ok = ok and det1 >= m01
        else:
            rec["events"] = {"n": 0}
        rec["arm"] = ok
        out[c] = rec
    return out


def md_table(rows: list[list], head: list[str]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join("–" if v is None else str(v) for v in r) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-guess", action="store_true")
    args = ap.parse_args(argv)
    frames = G4.load(GT_DET)
    desktops = list(HELD_OUT)
    G4.DESKTOPS = tuple(sorted({f.desktop for f in frames}))   # events: every desktop's actions
    E4.EVAL = EVAL
    E4.M0_CACHE = EVAL / "m0app.json"
    models = [m for m in ("v1", "v1h", "v2", "v2h", "F", "v3") if dets_path(m).exists()]
    for need in ("v2", "v2h", "F"):
        if need not in models:
            raise SystemExit(f"no detections of {need}")
    cnts, guess_rows, det_tables = {}, {}, {}
    con = env.connect()
    for m in models:
        dets = json.loads(dets_path(m).read_text(encoding="utf-8"))
        out = EVAL / m
        out.mkdir(parents=True, exist_ok=True)
        rows = E4.det_rows(dets, frames, desktops)
        E4.write_csv(out / "det_metrics.csv", rows)
        E4.write_csv(out / "curves.csv", E4.curves(dets, frames, desktops))
        det_tables[m] = rows
        cnts[m] = counts(dets, frames, desktops)
        if not args.skip_guess:
            g = E4.guess(dets, frames, desktops, con)
            E4.write_csv(out / "guess_events.csv", g)
            guess_rows[m] = g
        print(f"[l6] {m}: scored", flush=True)
    con.close()
    res = {"conf": CONF, "models": models, "gate": gate(cnts, CMP),
           "sensitivity_shipped_v2_on_D19": gate(cnts, CMP_SHIPPED)}
    res["verdict"] = "PASS" if res["gate"]["pass"] else "FAIL"
    res["extra_classes"] = extra_rule(cnts, guess_rows, res["gate"]["pass"])
    res["counts"] = {m: {f"D{d}:{c}": v for (d, c), v in cnts[m].items()} for m in models}
    # the guess, rank-1 hits per desktop / class / conf
    gsum = {}
    for m, rows in guess_rows.items():
        for r in rows:
            for tag in ("c05", "c10", "c25"):
                k = f"{m}|D{r['desktop']}|{r['cls']}|{tag}"
                s = gsum.setdefault(k, {"n": 0, "rank1": 0, "top3": 0, "ceiling": 0, "diffmap": 0})
                s["n"] += 1
                s["rank1"] += int(r.get(f"in1_{tag}", 0) or 0)
                s["top3"] += int(r.get(f"in3_{tag}", 0) or 0)
                s["ceiling"] += int(r.get(f"ceil_{tag}", 0) or 0)
                s["diffmap"] += int(r["M0app_in1"])
    res["guess"] = gsum
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "gate_det.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ---- markdown --------------------------------------------------------------
    lines = [f"# Detector, held out D19 / D27 (conf {CONF})\n"]
    head = ["class", "desktop", "n_gt"] + [f"{m} R/P" for m in models]
    rows = []
    for c in ARMED + EXTRA:
        for d in desktops:
            cells = []
            for m in models:
                cc = cnts[m].get((d, c), pooled({}, []))
                cells.append(f"{_fmt(recall(cc))}/{_fmt(precision(cc))}")
            rows.append([c, f"D{d}", cnts["F"].get((d, c), {}).get("n_gt", 0)] + cells)
    lines.append(md_table(rows, head))
    g = res["gate"]
    lines.append(f"\n## Gate D (D19: F vs {CMP[19]}, D27: F vs {CMP[27]}): **{res['verdict']}**\n")
    lines.append(f"- D1 no armed class worse on D19: {'PASS' if g['D1']['pass'] else 'FAIL'} "
                 f"(worse: {g['D1']['worse_on_D19'] or 'none'})")
    lines.append(f"- D2 >= 2 armed classes better overall: {'PASS' if g['D2']['pass'] else 'FAIL'} "
                 f"(better: {g['D2']['better_overall'] or 'none'})")
    lines.append(f"- D3 D27 screw+connector better, neither worse: {'PASS' if g['D3']['pass'] else 'FAIL'}")
    s = res["sensitivity_shipped_v2_on_D19"]
    lines.append(f"\nSensitivity, shipped v2 as D19's comparator: D1 {s['D1']} D2 {s['D2']} "
                 f"D3 {s['D3']['pass']} -> {'PASS' if s['pass'] else 'FAIL'}")
    lines.append("\n## Guess rank-1 (conf 0.05) on removal events\n")
    gr = []
    for k, v in sorted(gsum.items()):
        m, d, c, tag = k.split("|")
        if tag == "c05":
            gr.append([m, d, c, v["n"], v["rank1"], v["top3"], v["ceiling"], v["diffmap"]])
    lines.append(md_table(gr, ["model", "desktop", "class", "n", "rank1", "top3", "ceiling",
                               "diff map rank1"]))
    lines.append("\n## Extra classes\n")
    lines.append("```\n" + json.dumps(res["extra_classes"], indent=1) + "\n```")
    (EVAL / "det.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(lines))
    return 0


def _fmt(v):
    return "–" if v is None else f"{v:.1f}"


if __name__ == "__main__":
    sys.exit(main())
