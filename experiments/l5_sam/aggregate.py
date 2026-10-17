"""Tables, cluster-bootstrap deltas and the decision rule's gate.

::

    # stock baseline tables (every desktop)
    python -m experiments.l5_sam.aggregate baseline --csv baseline_all
    # one fold: stock vs fine-tuned on the held-out desktop, with the gate
    python -m experiments.l5_sam.aggregate fold --csv f13 --ft f13 --desktop 13

Writes ``OUT/tables/<name>.md`` and, for a fold, ``OUT/tables/<name>_gate.json``.
"""
from __future__ import annotations

from experiments.l5_sam import env  # noqa: F401

import argparse
import json

import numpy as np
import pandas as pd

from experiments.l5_sam import common as C

GROUPS = ("small", "screw", "connector", "latch", "large")
SMALL_GROUPS = ("small", "screw", "connector", "latch")
GATED = C.GATED_PROMPTS
ALL_PROMPTS = C.PROMPTS
N_BOOT = 2000


def load(name: str) -> pd.DataFrame:
    d = pd.read_csv(env.OUT / "eval" / f"{name}.csv")
    d["groups"] = d["groups"].fillna("")
    d["inst"] = "D" + d["desktop"].astype(str) + ":" + d["instance"]
    d["mkey"] = d["inst"] + ":" + d["step"].astype(str)
    return d


def in_group(d: pd.DataFrame, g: str) -> pd.Series:
    if g == "all":
        return pd.Series(True, index=d.index)
    return d["groups"].str.split("|").apply(lambda s: g in s)


def fmt(v, nd=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #
def baseline(name: str) -> str:
    d = load(name)
    d = d[d["model"] == "stock"]
    lines = [f"# Stock SAM 2.1 baseline ({name}): {d['mkey'].nunique()} masks, "
             f"{d['inst'].nunique()} instances\n"]
    lines.append("Shown candidate (top predicted IoU of 3, the app's) — IoU / boundary F (2 px); "
                 "oracle = best of 3; single = multimask=False.\n")
    hdr = "| group | n | " + " | ".join(f"{p} IoU / BF" for p in ALL_PROMPTS) + " | box oracle | point oracle | pb oracle | pb single |"
    lines += [hdr, "|" + "---|" * (len(ALL_PROMPTS) + 6)]
    for g in GROUPS + ("all",):
        s = d[in_group(d, g)]
        cells = []
        for p in ALL_PROMPTS:
            q = s[s["prompt"] == p]
            cells.append(f"{fmt(q['iou'].mean())} / {fmt(q['bf'].mean())}")
        orc = [fmt(s[s["prompt"] == p]["oracle_iou"].mean()) for p in GATED]
        single = fmt(s[s["prompt"] == "point_box"]["single_iou"].mean())
        lines.append(f"| {g} | {s['mkey'].nunique()} | " + " | ".join(cells) + " | "
                     + " | ".join(orc) + f" | {single} |")
    lines.append("\n## Per desktop (point_box shown IoU / box / point)\n")
    lines += ["| desktop | n | small | screw | connector | latch | large | all |", "|---|---|---|---|---|---|---|---|"]
    for dk in sorted(d["desktop"].unique()):
        s = d[d["desktop"] == dk]
        cells = []
        for g in GROUPS + ("all",):
            q = s[in_group(s, g)]
            cells.append("/".join(fmt(q[q["prompt"] == p]["iou"].mean(), 2) for p in ("point_box", "box", "point")))
        lines.append(f"| D{dk} | {s['mkey'].nunique()} | " + " | ".join(cells) + " |")
    lines.append("\n## Per class (sorted by point_box IoU)\n")
    lines += ["| class | n | inst | box | point | point_box | box_loose | refine | pb BF | pb oracle |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for c, s in d.groupby("cls"):
        r = [c, s["mkey"].nunique(), s["inst"].nunique()]
        r += [s[s["prompt"] == p]["iou"].mean() for p in ALL_PROMPTS]
        r += [s[s["prompt"] == "point_box"]["bf"].mean(), s[s["prompt"] == "point_box"]["oracle_iou"].mean()]
        rows.append(r)
    for r in sorted(rows, key=lambda r: r[5]):
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | " + " | ".join(fmt(v) for v in r[3:]) + " |")
    lines.append("\n## Per size bucket\n")
    lines += ["| size | n | box | point | point_box | box_loose | refine | pb BF |", "|---|---|---|---|---|---|---|---|"]
    for b in ("<400", "400-2k", "2k-20k", ">20k"):
        s = d[d["size_bucket"] == b]
        lines.append(f"| {b} | {s['mkey'].nunique()} | " + " | ".join(
            fmt(s[s["prompt"] == p]["iou"].mean()) for p in ALL_PROMPTS) + f" | {fmt(s[s['prompt'] == 'point_box']['bf'].mean())} |")
    lines.append("\n## Visible vs occluded, drawn frame vs propagated (point_box IoU)\n")
    pb = d[d["prompt"] == "point_box"]
    lines.append(f"- visibility: " + ", ".join(f"{k} {fmt(v)} (n={n})" for k, v, n in
                 [(k, s['iou'].mean(), len(s)) for k, s in pb.groupby('visibility')]))
    lines.append(f"- drawn frame {fmt(pb[pb['drawn'] == 1]['iou'].mean())} (n={int((pb['drawn'] == 1).sum())}), "
                 f"other frames {fmt(pb[pb['drawn'] == 0]['iou'].mean())}")
    occ = pb["occlusion"] > 0.02
    lines.append(f"- occlusion ratio > 0.02: {fmt(pb[occ]['iou'].mean())} (n={int(occ.sum())}); "
                 f"unoccluded {fmt(pb[~occ]['iou'].mean())}")
    lines.append("\n## Latency (ms, median / p90)\n")
    emb = d.drop_duplicates("mkey")["embed_ms"]
    lines.append(f"- set_image: {fmt(emb.median(), 1)} / {fmt(emb.quantile(0.9), 1)}")
    for p in ALL_PROMPTS:
        q = d[(d["prompt"] == p) & (d["status"] == "ok")]["decode_ms"]
        lines.append(f"- decode {p}: {fmt(q.median(), 1)} / {fmt(q.quantile(0.9), 1)}")
    text = "\n".join(lines) + "\n"
    out = env.OUT / "tables" / f"{name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------- #
# fold: paired deltas, bootstrap over instances, the gate
# --------------------------------------------------------------------------- #
def paired(d: pd.DataFrame, a: str, b: str, metric: str) -> pd.DataFrame:
    x = d[d["model"] == a].set_index(["mkey", "prompt"])
    y = d[d["model"] == b].set_index(["mkey", "prompt"])
    j = x[["inst", "groups", "cls", "size_bucket", metric]].join(y[[metric]], rsuffix="_ft", how="inner")
    j["delta"] = j[f"{metric}_ft"] - j[metric]
    return j.reset_index()


class Boot:
    """Cluster bootstrap over instances, shared weights for every cell."""

    def __init__(self, insts: list[str], seed: int = 0) -> None:
        self.insts = sorted(set(insts))
        self.pos = {k: i for i, k in enumerate(self.insts)}
        rng = np.random.default_rng(seed)
        n = len(self.insts)
        self.w = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)  # B, n

    def cell(self, j: pd.DataFrame) -> tuple[float, np.ndarray]:
        """(point estimate, bootstrap samples) of the mask-weighted mean delta."""
        if j.empty:
            return float("nan"), np.full(N_BOOT, np.nan)
        s = np.zeros(len(self.insts))
        c = np.zeros(len(self.insts))
        g = j.groupby("inst")["delta"].agg(["sum", "count"])
        for k, row in g.iterrows():
            s[self.pos[k]] = row["sum"]
            c[self.pos[k]] = row["count"]
        with np.errstate(invalid="ignore", divide="ignore"):
            samples = (self.w @ s) / (self.w @ c)
        return float(s.sum() / c.sum()), samples


def ci(samples: np.ndarray) -> tuple[float, float]:
    s = samples[~np.isnan(samples)]
    if s.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def fold(name: str, ft: str, desktop: int, stock: str = "stock") -> tuple[str, dict]:
    d = load(name)
    d = d[d["desktop"] == desktop]
    res = {"fold": ft, "desktop": desktop, "cells": {}}
    boots = Boot(list(d["inst"].unique()))
    lines = [f"# Fold {ft}: held-out D{desktop} — stock vs fine-tuned ({d['mkey'].nunique()} masks, "
             f"{d['inst'].nunique()} instances)\n"]
    est = {}
    for metric in ("iou", "bf"):
        j = paired(d, stock, ft, metric)
        lines.append(f"\n## {metric.upper()} of the shown candidate: stock → fine-tuned, Δ [95 % CI over instances]\n")
        lines += ["| group | n | " + " | ".join(ALL_PROMPTS) + " | prompt-avg Δ |",
                  "|" + "---|" * (len(ALL_PROMPTS) + 3)]
        for g in GROUPS + ("all",):
            jg = j[in_group(j, g)]
            cells = []
            avg_samples = []
            avg_pt = []
            for p in ALL_PROMPTS:
                jp = jg[jg["prompt"] == p]
                pt, smp = boots.cell(jp)
                est[(metric, g, p)] = (pt, smp)
                lo, hi = ci(smp)
                cells.append(f"{fmt(jp[metric].mean())}→{fmt(jp[f'{metric}_ft'].mean())} "
                             f"({'+' if pt >= 0 else ''}{fmt(pt)} [{fmt(lo)}, {fmt(hi)}])")
                if p in GATED:
                    avg_samples.append(smp)
                    avg_pt.append(pt)
            apt = float(np.mean(avg_pt))
            asmp = np.mean(np.stack(avg_samples), axis=0)
            est[(metric, g, "avg")] = (apt, asmp)
            lo, hi = ci(asmp)
            lines.append(f"| {g} | {jg['mkey'].nunique()} | " + " | ".join(cells)
                         + f" | {'+' if apt >= 0 else ''}{fmt(apt)} [{fmt(lo)}, {fmt(hi)}] |")
        # macro over the small groups
        for p in ("avg", "refine"):
            pts = [est[(metric, g, p)][0] for g in SMALL_GROUPS]
            smp = np.mean(np.stack([est[(metric, g, p)][1] for g in SMALL_GROUPS]), axis=0)
            est[(metric, "macro", p)] = (float(np.mean(pts)), smp)
            lo, hi = ci(smp)
            lines.append(f"\nmacro (small, screw, connector, latch) {p}: Δ{metric.upper()} "
                         f"{'+' if np.mean(pts) >= 0 else ''}{fmt(float(np.mean(pts)))} [{fmt(lo)}, {fmt(hi)}]")
        # per class
        lines.append(f"\n### per class, {metric.upper()} (point_box | box | point)\n")
        lines += ["| class | n | stock | fine-tuned | Δ pb | Δ box | Δ point |", "|---|---|---|---|---|---|---|"]
        for c, jc in j.groupby("cls"):
            pb = jc[jc["prompt"] == "point_box"]
            dl = [jc[jc["prompt"] == p]["delta"].mean() for p in ("point_box", "box", "point")]
            lines.append(f"| {c} | {pb['mkey'].nunique()} | {fmt(pb[metric].mean())} | "
                         f"{fmt(pb[f'{metric}_ft'].mean())} | " + " | ".join(fmt(v) for v in dl) + " |")
    # oracle / single for the record
    lines.append("\n## Oracle best-of-3 and single-mask IoU (all masks)\n")
    lines += ["| prompt | stock oracle | ft oracle | stock single | ft single |", "|---|---|---|---|---|"]
    for p in GATED:
        s = d[(d["prompt"] == p)]
        a, b = s[s["model"] == stock], s[s["model"] == ft]
        lines.append(f"| {p} | {fmt(a['oracle_iou'].mean())} | {fmt(b['oracle_iou'].mean())} | "
                     f"{fmt(a['single_iou'].mean())} | {fmt(b['single_iou'].mean())} |")
    # latency
    lat = {}
    for m in (stock, ft):
        s = d[d["model"] == m]
        lat[m] = {"embed_med": float(s.drop_duplicates("mkey")["embed_ms"].median()),
                  "decode_med": float(s[(s["prompt"].isin(GATED)) & (s["status"] == "ok")]["decode_ms"].median()),
                  "refine_med": float(s[(s["prompt"] == "refine") & (s["status"] == "ok")]["decode_ms"].median())}
    lines.append("\n## Latency (ms, median, same process)\n")
    lines += ["| model | set_image | first-prompt decode | refine decode |", "|---|---|---|---|"]
    for m in (stock, ft):
        lines.append(f"| {m} | {fmt(lat[m]['embed_med'], 1)} | {fmt(lat[m]['decode_med'], 2)} | "
                     f"{fmt(lat[m]['refine_med'], 2)} |")

    # ---- the gate -----------------------------------------------------------
    def e(metric, g, p):
        return est[(metric, g, p)][0]

    def lo(metric, g, p):
        return ci(est[(metric, g, p)][1])[0]

    g1 = (e("iou", "macro", "avg") >= 0.02 and lo("iou", "macro", "avg") > 0
          and e("bf", "macro", "avg") >= 0.01 and lo("bf", "macro", "avg") > 0)
    g2_groups = {g: e("iou", g, "avg") for g in SMALL_GROUPS}
    g2_cells = {f"{g}/{p}": e("iou", g, p) for g in SMALL_GROUPS for p in GATED}
    g2 = all(v >= -0.01 for v in g2_groups.values()) and all(v >= -0.03 for v in g2_cells.values())
    g3_cells = {p: e("iou", "large", p) for p in GATED}
    g3 = all(v >= -0.01 for v in g3_cells.values())
    g4 = e("iou", "macro", "refine") >= -0.01
    g5 = (lat[ft]["decode_med"] <= 1.10 * lat[stock]["decode_med"]
          and lat[ft]["embed_med"] <= 1.10 * lat[stock]["embed_med"])
    gate = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5}
    res.update(gate=gate, passed=all(gate.values()),
               macro_iou=(e("iou", "macro", "avg"), *ci(est[("iou", "macro", "avg")][1])),
               macro_bf=(e("bf", "macro", "avg"), *ci(est[("bf", "macro", "avg")][1])),
               macro_refine_iou=e("iou", "macro", "refine"),
               g2_groups=g2_groups, g2_cells=g2_cells, g3_cells=g3_cells, latency=lat,
               all_iou_avg=e("iou", "all", "avg"))
    lines.append("\n## Gate (decision_rule.txt)\n")
    lines.append(f"- G1 gain: macro ΔIoU {fmt(res['macro_iou'][0])} [{fmt(res['macro_iou'][1])}, "
                 f"{fmt(res['macro_iou'][2])}] (need ≥ +0.02, lo > 0); macro ΔBF {fmt(res['macro_bf'][0])} "
                 f"[{fmt(res['macro_bf'][1])}, {fmt(res['macro_bf'][2])}] (need ≥ +0.01, lo > 0) → "
                 f"{'PASS' if g1 else 'FAIL'}")
    lines.append(f"- G2 no small group harmed: groups {', '.join(f'{k} {fmt(v)}' for k, v in g2_groups.items())}; "
                 f"worst cell {min(g2_cells.items(), key=lambda kv: kv[1])[0]} "
                 f"{fmt(min(g2_cells.values()))} → {'PASS' if g2 else 'FAIL'}")
    lines.append(f"- G3 large parts: {', '.join(f'{k} {fmt(v)}' for k, v in g3_cells.items())} → "
                 f"{'PASS' if g3 else 'FAIL'}")
    lines.append(f"- G4 refine: macro ΔIoU {fmt(res['macro_refine_iou'])} → {'PASS' if g4 else 'FAIL'}")
    lines.append(f"- G5 latency: decode {fmt(lat[ft]['decode_med'], 2)} vs {fmt(lat[stock]['decode_med'], 2)} ms, "
                 f"set_image {fmt(lat[ft]['embed_med'], 1)} vs {fmt(lat[stock]['embed_med'], 1)} ms → "
                 f"{'PASS' if g5 else 'FAIL'}")
    lines.append(f"\n**Fold {ft}: {'PASS' if res['passed'] else 'FAIL'}**")
    text = "\n".join(lines) + "\n"
    out = env.OUT / "tables" / f"{name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    (env.OUT / "tables" / f"{name}_gate.json").write_text(json.dumps(res, indent=1, default=float),
                                                         encoding="utf-8")
    return text, res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("baseline", "fold"))
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ft")
    ap.add_argument("--desktop", type=int)
    args = ap.parse_args(argv)
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.mode == "baseline":
        print(baseline(args.csv))
    else:
        text, _res = fold(args.csv, args.ft, args.desktop)
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
