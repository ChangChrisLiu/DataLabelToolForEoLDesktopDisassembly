"""L4's tables and the decision (``decision_rule.txt``), from the evaluation CSVs.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.compare

Reads ``eval/<model>/{det_metrics,curves,guess_events}.csv`` and
``eval/oak_check.csv``; writes ``compare.md`` and ``decision.json``.  Arms per
held-out desktop: D13 -- L3 hold13 (comparator), production (not held out: it
saw D13's LS drafts), v2 hold13; D5 -- production (comparator), v2 hold5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l4_detector import env  # noqa: E402
from experiments.l4_detector import gt as G  # noqa: E402

EVAL = env.OUT / "eval"
FOLDS = {13: {"comparator": "l3hold13", "v2": "v2_hold13", "extra": ["production"]},
         5: {"comparator": "production", "v2": "v2_hold5", "extra": []}}
LABEL = {"production": "production", "l3hold13": "L3 hold13", "v2_hold13": "v2 F13",
         "v2_hold5": "v2 F5", "v2_all": "v2 all"}


def det(model: str) -> pd.DataFrame:
    p = EVAL / model / "det_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def guess(model: str) -> pd.DataFrame:
    p = EVAL / model / "guess_events.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def curves(model: str) -> pd.DataFrame:
    p = EVAL / model / "curves.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def cell(df: pd.DataFrame, desktop, cls: str, col: str):
    if df.empty:
        return None
    s = df[(df.desktop.astype(str) == str(desktop)) & (df.cls == cls)]
    if s.empty or pd.isna(s.iloc[0][col]) or s.iloc[0][col] == "":
        return None
    return float(s.iloc[0][col])


def fmt(v, nd=1) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def boot_hit(model_a: str, model_b: str, desktop: int, cls: str = "screw", n: int = 2000):
    """Paired bootstrap over frames of hit@0.10(b) - hit@0.10(a) (frames resampled)."""
    from experiments.l2_detector.det_eval import centre_in
    from experiments.l4_detector.infer import dets_path

    fr = [f for f in G.load() if f.desktop == desktop]
    da = json.loads(dets_path(model_a, "scan").read_text(encoding="utf-8"))
    db = json.loads(dets_path(model_b, "scan").read_text(encoding="utf-8"))
    per = []
    for f in fr:
        g = np.asarray([b.box for b in f.boxes if b.cls == cls], float).reshape(-1, 4)
        row = []
        for d in (da, db):
            rows = [r for r in d.get(f"{f.desktop}:scan:{f.step}", []) if r[4] == cls and r[5] >= 0.10]
            b = np.asarray([r[:4] for r in rows], float).reshape(-1, 4)
            row.append(int(centre_in(b, g).any(axis=0).sum()) if len(g) and len(b) else 0)
        per.append((len(g), row[0], row[1]))
    per = np.asarray(per, float)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n):
        s = per[rng.integers(0, len(per), len(per))]
        diffs.append(100 * (s[:, 2].sum() - s[:, 1].sum()) / max(1, s[:, 0].sum()))
    d0 = 100 * (per[:, 2].sum() - per[:, 1].sum()) / max(1, per[:, 0].sum())
    return round(d0, 1), round(float(np.percentile(diffs, 2.5)), 1), round(float(np.percentile(diffs, 97.5)), 1)


def main() -> int:
    lines = ["# L4 -- the detector against the annotator's verified scan masks", ""]
    decision: dict = {"gate": {}, "classes": {}, "notes": []}

    # ---------------------------------------------------------------- baseline
    prod, l3 = det("production"), det("l3hold13")
    lines += ["## 1. Baseline: the production model on the verified frames "
              "(hit@0.10 / AP50 / FP per frame @0.10)", "",
              "D2-D5 were never trained on (no LS drafts); D13 was (its LS drafts are the "
              "same frames). L3 hold13 never saw D13.", ""]
    head = "| class | n D2-5 | production D2-5 pooled | " + " | ".join(
        f"prod D{d}" for d in (2, 3, 4, 5)) + " | n D13 | prod D13 (seen) | L3 hold13 D13 |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for c in G.DET_CLASSES:
        n25 = sum((cell(prod, d, c, "n_gt") or 0) for d in (2, 3, 4, 5))
        hits, aps, fps, ns = [], [], [], []
        for d in (2, 3, 4, 5):
            n = cell(prod, d, c, "n_gt") or 0
            h = cell(prod, d, c, "hit@0.10")
            if n and h is not None:
                hits.append(h * n)
                ns.append(n)
        pooled = (sum(hits) / sum(ns)) if ns else None
        per = [f"{fmt(cell(prod, d, c, 'hit@0.10'))}" for d in (2, 3, 4, 5)]
        lines.append(
            f"| {c} | {int(n25)} | {fmt(pooled)} | " + " | ".join(per)
            + f" | {int(cell(prod, 13, c, 'n_gt') or 0)} | "
            f"{fmt(cell(prod, 13, c, 'hit@0.10'))} / {fmt(cell(prod, 13, c, 'AP50'))} / "
            f"{fmt(cell(prod, 13, c, 'FP/frame@0.10'), 2)} | "
            f"{fmt(cell(l3, 13, c, 'hit@0.10'))} / {fmt(cell(l3, 13, c, 'AP50'))} / "
            f"{fmt(cell(l3, 13, c, 'FP/frame@0.10'), 2)} |")
    lines.append("")

    # ---------------------------------------------------------------- folds
    lines += ["## 2. Held-out folds (hit@0.10 / AP50 / FP per frame @0.10)", ""]
    for d, arm in FOLDS.items():
        models = [arm["comparator"]] + arm["extra"] + [arm["v2"]]
        dfs = {m: det(m) for m in models}
        if dfs[arm["v2"]].empty:
            lines += [f"D{d}: {arm['v2']} not evaluated", ""]
            continue
        hdr = f"| D{d} class | n | " + " | ".join(LABEL[m] for m in models) + " | v2 − comparator hit |"
        lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        for c in G.DET_CLASSES:
            n = cell(dfs[models[0]], d, c, "n_gt")
            if not n:
                continue
            vals = [f"{fmt(cell(dfs[m], d, c, 'hit@0.10'))} / {fmt(cell(dfs[m], d, c, 'AP50'))} / "
                    f"{fmt(cell(dfs[m], d, c, 'FP/frame@0.10'), 2)}" for m in models]
            hv, hc = cell(dfs[arm["v2"]], d, c, "hit@0.10"), cell(dfs[arm["comparator"]], d, c, "hit@0.10")
            diff = None if hv is None or hc is None else hv - hc
            lines.append(f"| {c} | {int(n)} | " + " | ".join(vals) + f" | {fmt(diff)} |")
        try:
            b = boot_hit(arm["comparator"], arm["v2"], d)
            lines += ["", f"Screw hit@0.10, v2 − comparator on D{d}: {b[0]:+.1f} pts "
                      f"(paired bootstrap over frames, 95 % CI {b[1]:+.1f} .. {b[2]:+.1f})."]
            decision["notes"].append({f"screw_hit_boot_D{d}": b})
        except Exception as exc:  # noqa: BLE001
            lines += ["", f"(bootstrap failed: {exc})"]
        lines.append("")

    # ---------------------------------------------------------------- guess
    lines += ["## 3. The guess on the held-out desktops' removal events", "",
              "Rank-1 / top-3 / candidate ceiling at conf >= 0.10, after the app's "
              "already-drawn skip, ranked by the app's dE; M0app = the difference map's rank 1.", ""]
    hdr = "| desktop | class | n | M0app top-1 | " + " | ".join(
        ["comparator top-1 / top-3 / ceil", "v2 top-1 / top-3 / ceil"]) + " |"
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    pooled = {"comp": 0, "v2": 0, "n": 0}
    per_class = {}
    for d, arm in FOLDS.items():
        gc, gv = guess(arm["comparator"]), guess(arm["v2"])
        if gv.empty or gc.empty:
            continue
        gc, gv = gc[gc.desktop == d], gv[gv.desktop == d]
        for c in G.SMALL_CLASSES + ("ram_module", "cpu"):
            sc, sv = gc[gc.cls == c], gv[gv.cls == c]
            if sc.empty:
                continue
            def trip(s):
                return (f"{int(s.in1_c10.sum())} / {int(s.in3_c10.sum())} / {int(s.ceil_c10.sum())}"
                        if "in1_c10" in s else "–")
            lines.append(f"| D{d} | {c} | {len(sc)} | {int(sc.M0app_in1.sum())} | {trip(sc)} | {trip(sv)} |")
            pc = per_class.setdefault(c, {"n": 0, "m0": 0, "comp": 0, "v2": 0,
                                          "v2_c05": 0, "v2_c25": 0, "comp_c05": 0, "comp_c25": 0})
            pc["n"] += len(sc)
            pc["m0"] += int(sc.M0app_in1.sum())
            pc["comp"] += int(sc.in1_c10.sum())
            pc["v2"] += int(sv.in1_c10.sum())
            for t in ("c05", "c25"):
                pc[f"v2_{t}"] += int(sv[f"in1_{t}"].sum())
                pc[f"comp_{t}"] += int(sc[f"in1_{t}"].sum())
            if c == "screw":
                pooled["comp"] += int(sc.in1_c10.sum())
                pooled["v2"] += int(sv.in1_c10.sum())
                pooled["n"] += len(sc)
    lines.append("")
    decision["guess_per_class_pooled"] = per_class

    # ---------------------------------------------------------------- oak
    oak = EVAL / "oak_check.csv"
    oak_ok = None
    if oak.exists():
        o = pd.read_csv(oak)
        lines += ["## 4. OAK regression check (D13 oak1 / oak2, LS drafts, L3's protocol)", "",
                  "| view | class | n | L3 hold13 hit / AP50 / FP | v2 F13 hit / AP50 / FP |",
                  "|---|---|---|---|---|"]
        oak_ok = True
        for view in ("oak1", "oak2"):
            for c in G.DET_CLASSES:
                a = o[(o.model == "l3hold13") & (o.view == view) & (o.cls == c)]
                b = o[(o.model == "v2_hold13") & (o.view == view) & (o.cls == c)]
                if a.empty or b.empty:
                    continue
                a, b = a.iloc[0], b.iloc[0]
                lines.append(f"| {view} | {c} | {a.n_gt} | {a['hit@0.10']} / {a.AP50} / {a['FP/frame@0.10']} | "
                             f"{b['hit@0.10']} / {b.AP50} / {b['FP/frame@0.10']} |")
                if c == "screw":
                    ok = float(b["hit@0.10"]) >= float(a["hit@0.10"]) - 3
                    decision["gate"][f"G3_{view}"] = {"l3hold13": float(a["hit@0.10"]),
                                                      "v2": float(b["hit@0.10"]), "pass": ok}
                    oak_ok = oak_ok and ok
        lines.append("")

    # ---------------------------------------------------------------- gate
    g1 = {}
    for d, arm in FOLDS.items():
        hv = cell(det(arm["v2"]), d, "screw", "hit@0.10")
        hc = cell(det(arm["comparator"]), d, "screw", "hit@0.10")
        if hv is None or hc is None:
            continue
        g1[f"D{d}"] = {"comparator": hc, "v2": hv, "pass": hv >= hc - 2}
    decision["gate"]["G1"] = g1
    decision["gate"]["G2"] = {"comparator_top1": pooled["comp"], "v2_top1": pooled["v2"],
                              "screw_events": pooled["n"], "pass": pooled["v2"] >= pooled["comp"]}
    passed = (bool(g1) and all(v["pass"] for v in g1.values()) and decision["gate"]["G2"]["pass"]
              and bool(oak_ok))
    decision["gate"]["pass"] = passed
    decision["gate"]["folds_evaluated"] = sorted(g1)
    # classes
    for c in G.SMALL_CLASSES[1:] + ("ram_module", "cpu"):
        hits = {f"D{d}": cell(det(arm["v2"]), d, c, "hit@0.10") for d, arm in FOLDS.items()}
        pc = per_class.get(c, {"n": 0, "m0": 0, "v2": 0})
        ok_hit = all(v is not None and v >= 60 for v in hits.values()) and len(hits) == 2
        ok_guess = pc["n"] > 0 and pc["v2"] >= pc["m0"]
        decision["classes"][c] = {"hit": hits, "events": pc["n"], "v2_top1": pc["v2"],
                                  "M0app_top1": pc["m0"], "hit_rule": ok_hit,
                                  "guess_safeguard": ok_guess, "enable": bool(passed and ok_hit and ok_guess)}
    # ---------------------------------------------------------------- curves
    lines += ["## 4b. Operating point: hit % / precision % over conf (held-out desktop)", ""]
    for c in ("screw", "connector", "ram_latch", "psu_latch", "drive_latch", "cpu_socket_lever"):
        hdr = "| D | model | " + " | ".join(f"{t:g}" for t in (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50)) + " |"
        rows = []
        for d, arm in FOLDS.items():
            for m in (arm["comparator"], arm["v2"]):
                cv = curves(m)
                if cv.empty:
                    continue
                s = cv[(cv.desktop.astype(str) == str(d)) & (cv.cls == c)]
                if s.empty:
                    continue
                rows.append(f"| D{d} | {LABEL[m]} | " + " | ".join(
                    f"{r.hit} / {r.precision}" for r in s.sort_values("conf").itertuples()) + " |")
        if rows:
            lines += [f"**{c}**", "", hdr, "|" + "---|" * (hdr.count("|") - 1)] + rows + [""]

    # ---------------------------------------------------------------- timing
    lines += ["## 5. Time per scan frame (tiled detection, median / p90 s; the app's own "
              "machine under the annotator's load)", "",
              "| model | floor | frames | tiles | net | total |", "|---|---|---|---|---|---|"]
    for tag in sorted(p.name for p in (env.OUT / "dets").iterdir() if p.is_dir()):
        tp = env.OUT / "dets" / tag / "scan_timing.json"
        if not tp.exists() or tag == "oracle":
            continue
        t = json.loads(tp.read_text(encoding="utf-8"))
        if not t:
            continue
        tot = np.asarray([r["total_s"] for r in t])
        net = np.asarray([r["net_s"] for r in t])
        floor = "0.10" if tag.endswith("_c10") else "0.001"
        lines.append(f"| {tag} | {floor} | {len(t)} | {int(np.median([r['n_tiles'] for r in t]))} | "
                     f"{np.median(net):.3f} / {np.percentile(net, 90):.3f} | "
                     f"{np.median(tot):.3f} / {np.percentile(tot, 90):.3f} |")
    lines.append("")
    lines += ["## 6. Decision", "", "```", json.dumps(decision, indent=1), "```", ""]
    (env.OUT / "compare.md").write_text("\n".join(lines), encoding="utf-8")
    (env.OUT / "decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
