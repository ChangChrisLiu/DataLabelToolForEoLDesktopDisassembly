"""Question counts of a built VLM pack, per task, form and unit (owner 2026-11-04).

``python scripts/vlm_stats.py --pack <vlmpack> --out <dir>`` writes ``<dir>/vlm_stats.json``
and ``<dir>/vlm_stats.csv``: one row per ``(task, form, unit)`` with the pool count
(``data/vlm_<unit>.jsonl``) and the ``eval_v3`` count, the task's group (Perception V1 V8 V2
V15, Understanding V3 V12 V10, Planning V4 V6 V7 V16) and its display name; the JSON also
holds per-task pool totals with each form's share.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

GROUPS = {"V1": "Perception", "V8": "Perception", "V2": "Perception", "V15": "Perception",
          "V3": "Understanding", "V12": "Understanding", "V10": "Understanding",
          "V4": "Planning", "V6": "Planning", "V7": "Planning", "V16": "Planning"}
NAMES = {"V1": "Enumeration", "V8": "Grounding", "V2": "State", "V15": "Cross-view",
         "V3": "Action", "V12": "State change", "V10": "Progress", "V4": "Feasibility",
         "V6": "Next action", "V7": "Plan", "V16": "Plan check"}
ORDER = ("V1", "V8", "V2", "V15", "V3", "V12", "V10", "V4", "V6", "V7", "V16")


def form_of(record: dict) -> str:
    """The record's form: V2 state / cannot_tell / count, V3 free / roi / span, V6 open / mc /
    constrained, V7 free / constrained, V8 single / set, V12 by span, V15 moment / cross,
    V16 valid / invalid; ``""`` for a task of one form."""
    task, label = record["prompt"]["task"], record["label"]
    check = label.get("answer_check") or {}
    if task == "V2":
        return ("count" if check.get("derive") == "screw_count" else
                "cannot_tell" if check.get("hidden") else "state")
    if task == "V3":
        return label.get("v3_form") or "free"
    if task == "V6":
        return label.get("v6_form") or "open"
    if task == "V7":
        return label.get("v7_form") or "free"
    if task == "V8":
        return "set" if check.get("type") == "components" else "single"
    if task == "V12":
        return f"span{label.get('v12_span') or 1}"
    if task == "V15":
        return label.get("v15_variant") or "moment"
    if task == "V16":
        return "valid" if (check.get("answer_keys") or label["answer"]).get("valid") else "invalid"
    return ""


def display_name(task: str, form: str) -> str:
    if task == "V2" and form == "count":
        return "Screw count"
    return NAMES.get(task, task)


def count(files) -> Counter:
    out: Counter = Counter()
    for unit, path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[(r["prompt"]["task"], form_of(r), unit)] += 1
    return out


def stats(pack: Path) -> dict:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    ev = manifest.get("eval_version") or "eval_v3"
    units = list(manifest.get("units") or [])
    pool = count([(u, pack / "data" / f"vlm_{u}.jsonl") for u in units
                  if (pack / "data" / f"vlm_{u}.jsonl").exists()])
    subset = count([(u, pack / ev / f"{u}.jsonl") for u in units
                    if (pack / ev / f"{u}.jsonl").exists()])
    rows = []
    for key in sorted(set(pool) | set(subset),
                      key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1], k[2])):
        task, form, unit = key
        rows.append({"task": task, "form": form, "unit": unit,
                     "group": GROUPS.get(task, "-"), "name": display_name(task, form),
                     "pool": pool.get(key, 0), "eval": subset.get(key, 0)})
    totals: dict[str, dict] = {}
    by_task: dict[str, Counter] = defaultdict(Counter)
    for (task, form, _), n in pool.items():
        by_task[task][form] += n
    for task in sorted(by_task, key=lambda t: ORDER.index(t) if t in ORDER else 99):
        total = sum(by_task[task].values())
        totals[task] = {"group": GROUPS.get(task, "-"), "name": NAMES.get(task, task),
                        "pool": total,
                        "eval": sum(n for (t, _, _), n in subset.items() if t == task),
                        "forms": {f or "-": {"pool": n, "share": round(n / total, 4)}
                                  for f, n in sorted(by_task[task].items())}}
    return {"pack": str(pack), "eval_version": ev, "source_commit": manifest.get("source_commit"),
            "rows": rows, "tasks": totals,
            "groups": {g: sum(v["pool"] for v in totals.values() if v["group"] == g)
                       for g in ("Perception", "Understanding", "Planning")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    result = stats(Path(args.pack))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vlm_stats.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                        encoding="utf-8")
    with open(out / "vlm_stats.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "form", "unit", "group", "name", "pool", "eval"])
        w.writeheader()
        w.writerows(result["rows"])
    print(f"[vlm-stats] {len(result['rows'])} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
