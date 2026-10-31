"""Score a model's answers against an export-vlm file (W3b).

``python scripts/vlm_score.py --truth <vlm_scan.jsonl> --pred <model.jsonl>
[--chat <chat.jsonl>] --out <dir> [--name <model>] [--seed 0] [--resamples 1000]``

``--pred`` is one JSON object per line, ``{"id", "raw_text"}``. Every record
of ``--truth`` is graded twice by :func:`tda.core.export.vlm_check.grade`:
**strict** (the reply is one JSON object) and **lenient** (the first object
in it, else the prose); a record with no prediction counts as a parse failure.
Both parse-failure rates are reported.

Aggregated per task, per capability layer (L1 V1 V2 V8 V14 / L2 V4 / L3 V6 V7
V16 / change V3 V12 V10 / cross V15) and per question template, each with a
95 % interval from a **cluster bootstrap over desktops** (desktops resampled
with replacement, ``--resamples`` times, seeded). Beside the primary score:

* V2's headline is the accuracy on ``prior_is_correct = false`` -- the
  questions the class alone does not answer;
* V6's ``blocked_rate`` is over the frames with ``blocked_rate_defined`` only,
  and its exact-instance match rides along as the secondary number;
* V7 adds violation-free rate, required-action recall and goal-reached rate;
* records whose truth rests on a captive screw's tightness
  (``image_undecidable``) are also scored without;
* the export's text-only shortcut upper bound per task (``<truth>.summary.json``)
  is a standing column.

Writes ``<out>/<name>.score.json`` and ``<out>/<name>.score.md``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vlm_io as IO  # noqa: E402
from tda.core.export.vlm_check import CAPABILITY_LAYERS, grade  # noqa: E402

LAYER_ORDER = ("L1", "L2", "L3", "change", "cross")


def cluster_bootstrap(rows: list[tuple[int, float]], resamples: int, seed: int
                      ) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """``(mean, lo, hi)``: 95 % percentile interval, desktops resampled whole.

    Records of one desktop are not independent (one machine, one operator, one
    lighting), so the unit of resampling is the desktop: each resample draws
    as many desktops as there are, with replacement, and takes the mean over
    all their records. Seeded, so the same inputs give the same interval.
    """
    if not rows:
        return None, None, None
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for desk, value in rows:
        sums[desk] += float(value)
        counts[desk] += 1
    desks = sorted(sums)
    point = sum(sums.values()) / sum(counts.values())
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        total = n = 0.0
        for _ in desks:
            d = desks[rng.randrange(len(desks))]
            total += sums[d]
            n += counts[d]
        stats.append(total / n)
    stats.sort()
    lo = stats[int(0.025 * (len(stats) - 1))]
    hi = stats[int(round(0.975 * (len(stats) - 1)))]
    return point, lo, hi


def _cell(rows, resamples, seed) -> dict:
    mean, lo, hi = cluster_bootstrap(rows, resamples, seed)
    return {"n": len(rows), "mean": _r(mean), "ci95": [_r(lo), _r(hi)],
            "desktops": len({d for d, _ in rows})}


def _r(x):
    return None if x is None else round(float(x), 4)


def score(truth_path, pred_path, chat_path=None, *, resamples: int = 1000,
          seed: int = 0) -> dict:
    from tda.core.taxonomy import load_taxonomy

    tax = load_taxonomy()
    records = list(IO.read_jsonl(truth_path))
    worlds, _ = IO.load_oracle(truth_path)
    summary = IO.load_summary(truth_path)
    preds: dict[str, Optional[str]] = {}
    for row in IO.read_jsonl(pred_path):
        preds[str(row["id"])] = row.get("raw_text")
    transforms: dict[str, list] = {}
    if chat_path:
        for row in IO.read_jsonl(chat_path):
            meta = row.get("meta") or {}
            if meta.get("transforms"):
                transforms[str(row["id"])] = meta["transforms"]
    out: dict = {"records": len(records), "predicted": sum(1 for r in records
                                                           if r["id"] in preds)}
    for mode in ("strict", "lenient"):
        per_task: dict[str, list] = defaultdict(list)
        per_layer: dict[str, list] = defaultdict(list)
        per_template: dict[str, list] = defaultdict(list)
        failures: dict[str, int] = defaultdict(int)
        extras: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for record in records:
            task = record["prompt"]["task"]
            label = record["label"]
            desk = int(label["desktop"])
            world = worlds.get(desk)
            trans = transforms.get(record["id"])
            g = grade(record, preds.get(record["id"]), world, tax, mode=mode,
                      transform=trans[0] if trans else None)
            if not g.parsed:
                failures[task] += 1
            row = (desk, g.score)
            per_task[task].append(row)
            per_layer[CAPABILITY_LAYERS.get(task, "other")].append(row)
            per_template[label.get("template_id", "?")].append(row)
            m = g.metrics
            if not label.get("image_undecidable"):
                extras[task]["decidable"].append(row)
            if task == "V2" and label.get("prior_is_correct") is False:
                extras[task]["prior_wrong"].append(row)
            if task == "V6":
                extras[task]["exact"].append((desk, float(bool(m.get("exact_match")))))
                if m.get("blocked_rate_defined") and g.parsed:
                    extras[task]["blocked_rate"].append((desk, float(bool(m.get("blocked")))))
            if task == "V7" and g.parsed and "violation_free" in m:
                extras[task]["violation_free"].append((desk, float(m["violation_free"])))
                extras[task]["required_recall"].append((desk, float(m["required_recall"])))
                extras[task]["goal_reached"].append((desk, float(m["goal_reached"])))
            if task == "V16" and m.get("localised") is not None:
                extras[task]["localised"].append((desk, float(m["localised"])))
            if task == "V4" and "blockers_f1" in m:
                extras[task]["blockers_f1"].append((desk, float(m["blockers_f1"])))
            if task == "V10" and "bin_correct" in m:
                extras[task]["bin_accuracy"].append((desk, float(m["bin_correct"])))
        out[mode] = {
            "parse_failure_rate": {t: _r(failures[t] / len(rows))
                                   for t, rows in sorted(per_task.items())},
            "parse_failure_rate_all": _r(sum(failures.values()) / max(len(records), 1)),
            "by_task": {t: _cell(rows, resamples, seed) for t, rows in sorted(per_task.items())},
            "by_layer": {l: _cell(per_layer[l], resamples, seed)
                         for l in LAYER_ORDER if per_layer.get(l)},
            "by_template": {t: _cell(rows, resamples, seed)
                            for t, rows in sorted(per_template.items())},
            "extra": {t: {k: _cell(v, resamples, seed) for k, v in sorted(d.items())}
                      for t, d in sorted(extras.items())},
        }
    out["shortcut_upper_bound"] = _shortcuts(summary)
    return out


def _shortcuts(summary: dict) -> dict:
    """The export's text-only upper bound per task: the best of its classifiers."""
    table = {}
    for task, row in (summary.get("shortcuts") or {}).items():
        best = max((row.get(k) or 0.0) for k in ("verb", "class", "verb_class", "template"))
        table[task] = {"majority": row.get("majority"), "best_text_only": _r(best)}
    option = summary.get("v6_option_shortcut") or {}
    if option.get("records"):
        table.setdefault("V6", {})["option_verb"] = option.get("shortcut")
        table["V6"]["random"] = option.get("random")
    return table


def markdown(result: dict, name: str) -> str:
    def fmt(cell):
        if not cell or cell.get("mean") is None:
            return "–"
        lo, hi = cell["ci95"]
        return f"{cell['mean']:.3f} [{lo:.3f}, {hi:.3f}]"

    s, l = result["strict"], result["lenient"]
    short = result.get("shortcut_upper_bound") or {}
    lines = [f"# {name}", "",
             f"{result['predicted']} of {result['records']} records answered. "
             f"Parse failures: strict {s['parse_failure_rate_all']:.1%}, "
             f"lenient {l['parse_failure_rate_all']:.1%}. "
             "Intervals: 95 %, cluster bootstrap over desktops.", "",
             "| task | n | strict | lenient | parse fail (strict / lenient) | "
             "text-only upper bound |",
             "|---|---|---|---|---|---|"]
    for task in sorted(s["by_task"], key=lambda t: int(t[1:])):
        sc = short.get(task) or {}
        bound = sc.get("best_text_only")
        if task == "V6" and sc.get("option_verb") is not None:
            bound = sc["option_verb"]
        lines.append(f"| {task} | {s['by_task'][task]['n']} | {fmt(s['by_task'][task])} | "
                     f"{fmt(l['by_task'][task])} | "
                     f"{s['parse_failure_rate'][task]:.1%} / "
                     f"{l['parse_failure_rate'][task]:.1%} | "
                     f"{'–' if bound is None else f'{bound:.1%}'} |")
    lines += ["", "| layer | n | strict | lenient |", "|---|---|---|---|"]
    for layer, cell in s["by_layer"].items():
        lines.append(f"| {layer} | {cell['n']} | {fmt(cell)} | {fmt(l['by_layer'][layer])} |")
    lines += ["", "| task | metric | strict | lenient |", "|---|---|---|---|"]
    for task, metrics in s["extra"].items():
        for key, cell in metrics.items():
            other = l["extra"].get(task, {}).get(key)
            lines.append(f"| {task} | {key} | {fmt(cell)} | {fmt(other)} |")
    lines += ["", "| template | n | strict |", "|---|---|---|"]
    for tid, cell in s["by_template"].items():
        lines.append(f"| {tid} | {cell['n']} | {fmt(cell)} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--truth", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--chat", default=None,
                    help="the chat JSONL the model was run on (box transforms of --roi)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resamples", type=int, default=1000)
    args = ap.parse_args(argv)
    name = args.name or Path(args.pred).stem
    result = score(args.truth, args.pred, args.chat, resamples=args.resamples,
                   seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.score.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    (out / f"{name}.score.md").write_text(markdown(result, name), encoding="utf-8")
    print(f"[vlm-score] {name}: strict parse failures "
          f"{result['strict']['parse_failure_rate_all']:.1%}, lenient "
          f"{result['lenient']['parse_failure_rate_all']:.1%} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
