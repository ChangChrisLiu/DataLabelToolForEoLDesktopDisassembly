"""The SAM gate (``decision_rule.txt``, gate S): the shipped two-head setup vs. the same with the L6 head.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.gate_sam [--csv heldout] [--new F]

Reads ``sam/eval/<csv>.csv`` (``eval_sam.py``: models stock, f5_v2, F on every
D19 / D27 mask) and composes the two setups per mask and prompt -- exact,
since both heads decode the same embedding of the same image encoder:

* ``shipped`` = route(stock, f5_v2), ``new`` = route(stock, <new>);
* route = the tuned head for every prompt except ``point`` on a mask whose
  class is in ``configs/sam.yaml`` ``stock_point_classes`` (the stock head);
  ``refine`` follows ``point_box`` (a box), so the tuned head.

Then, per L5's groups (small, screw, connector, latch, large) and the gated
prompts point and box: paired dIoU new - shipped, 95 % CI by L5's cluster
bootstrap over instances (``aggregate.Boot``, 2 000, seed 0); WORSE / BETTER
and S1-S3 as the rule says.  Also reported: every prompt, boundary F, per
class, stock alone, "tuned everywhere" (F vs f5_v2 unrouted), latency.
Writes ``sam/tables/<csv>_gate.json`` and ``<csv>.md``.  CPU only.
"""
from __future__ import annotations

from experiments.l6_models import env  # noqa: F401  (side effects first)

import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

from experiments.l5_sam import common as C
from experiments.l5_sam.aggregate import Boot, ci, in_group
from experiments.l6_models.train_sam import SAM_OUT

GROUPS = ("small", "screw", "connector", "latch", "large")
GATED = ("point", "box")
PROMPTS = C.PROMPTS
SHIPPED_TUNED = "f5_v2"


def stock_point_classes() -> set[str]:
    cfg = yaml.safe_load((env.REPO / "configs" / "sam.yaml").read_text(encoding="utf-8"))
    return set(cfg["stock_point_classes"])


def load(name: str) -> pd.DataFrame:
    d = pd.read_csv(SAM_OUT / "eval" / f"{name}.csv")
    d["groups"] = d["groups"].fillna("")
    d["inst"] = "D" + d["desktop"].astype(str) + ":" + d["instance"]
    d["mkey"] = d["inst"] + ":" + d["step"].astype(str)
    return d


def routed(d: pd.DataFrame, tuned: str, name: str, listed: set[str]) -> pd.DataFrame:
    """One setup's rows: ``tuned`` everywhere but point-only prompts of listed classes."""
    use_stock = (d["prompt"] == "point") & d["cls"].isin(listed)
    pick = ((d["model"] == tuned) & ~use_stock) | ((d["model"] == "stock") & use_stock)
    out = d[pick].copy()
    out["head"] = np.where(out["model"] == "stock", "stock", "tuned")
    out["model"] = name
    return out


def paired(d: pd.DataFrame, a: str, b: str, metric: str) -> pd.DataFrame:
    x = d[d["model"] == a].set_index(["mkey", "prompt"])
    y = d[d["model"] == b].set_index(["mkey", "prompt"])
    j = x[["inst", "desktop", "groups", "cls", metric]].join(y[[metric]], rsuffix="_new",
                                                               how="inner")
    j["delta"] = j[f"{metric}_new"] - j[metric]
    return j.reset_index()


def cells(j: pd.DataFrame, insts: list[str]) -> dict:
    """``(group, prompt) -> (point, samples, n masks, cmp mean, new mean)`` on ``j``."""
    boot = Boot(insts)
    out = {}
    for g in GROUPS + ("all",):
        jg = j[in_group(j, g)]
        for p in PROMPTS:
            jp = jg[jg["prompt"] == p]
            pt, smp = boot.cell(jp)
            out[(g, p)] = (pt, smp, int(jp["mkey"].nunique()), float(jp.iloc[:, -3].mean())
                           if len(jp) else float("nan"), float(jp.iloc[:, -2].mean())
                           if len(jp) else float("nan"))
        avg_pt = [out[(g, p)][0] for p in GATED]
        avg_smp = np.mean(np.stack([out[(g, p)][1] for p in GATED]), axis=0)
        out[(g, "avg")] = (float(np.mean(avg_pt)), avg_smp, out[(g, "point")][2], None, None)
    sc = j[in_group(j, "screw") | in_group(j, "connector")]
    pts, smps = [], []
    for p in GATED:
        pt, smp = boot.cell(sc[sc["prompt"] == p])
        out[("screw+connector", p)] = (pt, smp, int(sc[sc["prompt"] == p]["mkey"].nunique()),
                                       None, None)
        pts.append(pt)
        smps.append(smp)
    out[("screw+connector", "avg")] = (float(np.mean(pts)), np.mean(np.stack(smps), axis=0),
                                       out[("screw+connector", "point")][2], None, None)
    return out


def worse(c: dict, g: str) -> bool:
    vals = [c[(g, p)][0] for p in GATED if c[(g, p)][2] > 0]
    return any(v < -0.01 for v in vals if not np.isnan(v))


def better(c: dict, g: str) -> bool:
    pt, smp = c[(g, "avg")][0], c[(g, "avg")][1]
    if np.isnan(pt):
        return False
    return pt >= 0.01 and ci(smp)[0] > 0


def fmt(v, nd=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{nd}f}" \
        if isinstance(v, float) else str(v)


def gate(d: pd.DataFrame, cmp: str, new: str, metric: str = "iou") -> dict:
    j = paired(d, cmp, new, metric)
    per = {}
    for name, ds in (("D19", [19]), ("D27", [27]), ("overall", [19, 27])):
        jj = j[j["desktop"].isin(ds)]
        per[name] = cells(jj, sorted(jj["inst"].unique()))
    s1_worse = [g for g in GROUPS if worse(per["D19"], g)]
    s2_better = [g for g in GROUPS if better(per["overall"], g)]
    s3 = (better(per["D27"], "screw+connector") and not worse(per["D27"], "screw")
          and not worse(per["D27"], "connector"))
    res = {"S1": {"pass": not s1_worse, "worse_on_D19": s1_worse},
           "S2": {"pass": len(s2_better) >= 2, "better_overall": s2_better},
           "S3": {"pass": s3, "screw+connector_D27_better": better(per["D27"], "screw+connector"),
                  "screw_D27_worse": worse(per["D27"], "screw"),
                  "connector_D27_worse": worse(per["D27"], "connector")}}
    res["pass"] = res["S1"]["pass"] and res["S2"]["pass"] and res["S3"]["pass"]
    table = {}
    for name, c in per.items():
        for (g, p), (pt, smp, n, a, b) in c.items():
            lo, hi = ci(smp)
            table[f"{name}|{g}|{p}"] = {"n": n, "delta": None if np.isnan(pt) else round(pt, 4),
                                        "ci": [None if np.isnan(lo) else round(lo, 4),
                                               None if np.isnan(hi) else round(hi, 4)],
                                        "cmp": None if a is None or np.isnan(a) else round(a, 4),
                                        "new": None if b is None or np.isnan(b) else round(b, 4)}
    res["cells"] = table
    return res


def md_cells(res: dict, title: str) -> list[str]:
    lines = [f"\n### {title}\n", "| set | group | n | " + " | ".join(
        f"{p} (cmp→new, Δ [CI])" for p in PROMPTS) + " | avg(point, box) Δ [CI] |",
             "|" + "---|" * (len(PROMPTS) + 4)]
    for name in ("D19", "D27", "overall"):
        for g in GROUPS + ("all", "screw+connector"):
            cs = []
            for p in PROMPTS:
                c = res["cells"].get(f"{name}|{g}|{p}")
                if c is None:
                    cs.append("–")
                    continue
                cs.append(f"{c['cmp'] if c['cmp'] is not None else '–'}→"
                          f"{c['new'] if c['new'] is not None else '–'} "
                          f"({fmt(c['delta'])} [{fmt(c['ci'][0])}, {fmt(c['ci'][1])}])")
            a = res["cells"][f"{name}|{g}|avg"]
            lines.append(f"| {name} | {g} | {a['n']} | " + " | ".join(cs)
                         + f" | {fmt(a['delta'])} [{fmt(a['ci'][0])}, {fmt(a['ci'][1])}] |")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="heldout")
    ap.add_argument("--new", default="F")
    args = ap.parse_args(argv)
    raw = load(args.csv)
    listed = stock_point_classes()
    d = pd.concat([raw, routed(raw, SHIPPED_TUNED, "shipped", listed),
                   routed(raw, args.new, "new", listed)], ignore_index=True)
    res = {"csv": args.csv, "new": args.new, "stock_point_classes": sorted(listed),
           "masks": {f"D{k}": int(v) for k, v in raw.drop_duplicates("mkey")
                     .groupby("desktop").size().items()},
           "gate": gate(d, "shipped", "new")}
    res["verdict"] = "PASS" if res["gate"]["pass"] else "FAIL"
    res["boundary_f"] = gate(d, "shipped", "new", "bf")
    res["tuned_everywhere"] = gate(d, SHIPPED_TUNED, args.new)     # unrouted, for the record
    res["shipped_vs_stock"] = gate(d, "stock", "shipped")
    res["new_vs_stock"] = gate(d, "stock", "new")
    # per class (point / box / point_box IoU), shipped -> new
    pcs = {}
    for (desk, c), s in d[d["model"].isin(["shipped", "new"])].groupby(["desktop", "cls"]):
        row = {"n": int(s["mkey"].nunique())}
        for p in ("point", "box", "point_box"):
            q = s[s["prompt"] == p]
            a = q[q["model"] == "shipped"]["iou"].mean()
            b = q[q["model"] == "new"]["iou"].mean()
            row[p] = [round(float(a), 3), round(float(b), 3)]
        pcs[f"D{desk}:{c}"] = row
    res["per_class"] = pcs
    lat = {}
    for m in ("stock", SHIPPED_TUNED, args.new):
        s = raw[raw["model"] == m]
        lat[m] = {"set_image_med_ms": round(float(s.drop_duplicates("mkey")["embed_ms"].median()), 2),
                  "decode_first_med_ms": round(float(s[s["prompt"].isin(C.GATED_PROMPTS)
                                                       & (s["status"] == "ok")]["decode_ms"].median()), 2),
                  "decode_refine_med_ms": round(float(s[(s["prompt"] == "refine")
                                                        & (s["status"] == "ok")]["decode_ms"].median()), 2)}
    res["latency"] = lat
    out = SAM_OUT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.csv}_gate.json").write_text(json.dumps(res, indent=1, default=float),
                                              encoding="utf-8")
    g = res["gate"]
    lines = [f"# SAM heads on held-out D19 / D27 ({args.csv}): shipped route(stock, f5_v2) vs "
             f"route(stock, {args.new})\n",
             f"masks: {res['masks']}; stock_point_classes: {', '.join(sorted(listed))}\n",
             f"## Gate S: **{res['verdict']}**\n",
             f"- S1 no group worse on D19: {'PASS' if g['S1']['pass'] else 'FAIL'} "
             f"(worse: {g['S1']['worse_on_D19'] or 'none'})",
             f"- S2 >= 2 groups better overall: {'PASS' if g['S2']['pass'] else 'FAIL'} "
             f"(better: {g['S2']['better_overall'] or 'none'})",
             f"- S3 D27 screw+connector better, neither worse: {'PASS' if g['S3']['pass'] else 'FAIL'} "
             f"({g['S3']})"]
    lines += md_cells(g, "IoU, shipped → new")
    lines += md_cells(res["boundary_f"], "Boundary F, shipped → new")
    lines += md_cells(res["tuned_everywhere"], f"IoU, unrouted: f5_v2 → {args.new}")
    lines += md_cells(res["shipped_vs_stock"], "IoU, stock → shipped")
    lines.append("\n### Latency (median ms, same process)\n")
    lines.append("```\n" + json.dumps(lat, indent=1) + "\n```")
    (out / f"{args.csv}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
