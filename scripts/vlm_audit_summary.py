"""Summarise the filled human-audit sheets: error rates with Wilson 95 % intervals (W4-E).

``python tools/vlm_audit_summary.py --csv <filled.csv> [<more.csv> ...]
[--records eval_v1/audit/audit.jsonl] --out <dir> [--name audit]``

The CSVs are what ``vlm_audit_sheet.py``'s pages export (or its skeleton
filled in a spreadsheet): one row per item with ``id, task, verdict, note,
reviewer, ...``; ``verdict`` is ``correct`` / ``wrong`` / ``doubtful`` (the
Chinese labels 正确 / 错误 / 存疑 and a few spellings are accepted; an empty
cell is an item not yet reviewed). Several files -- batches, reviewers -- are
read together.

Per task and overall, two rates with their **Wilson score intervals** (95 %):

* **error rate** -- items marked *wrong* over the items reviewed;
* **error-or-doubt rate** -- *wrong* or *doubtful* over the items reviewed
  (the conservative reading).

The unit is the item. When several reviewers marked the same item, the item
takes the **worst** verdict (wrong > doubtful > correct) in those rates, and
the disagreements are listed apart with the pairwise agreement. The report
also lists every item marked wrong or doubtful with the reviewer's note, and
with ``--records`` the items not yet reviewed.

Wilson rather than Wald: with 0 errors in 300 the Wald interval is [0, 0] and
the Wilson interval [0, 1.3 %], which is what the paper can honestly print.
Writes ``<out>/<name>.summary.md`` and ``<out>/<name>.summary.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

__all__ = ["RANK", "Z95", "markdown", "normalise_verdict", "read_verdicts", "summarise",
           "wilson", "main"]

#: The 97.5th percentile of the standard normal: a two-sided 95 % interval.
Z95 = 1.959963984540054
#: Worse verdicts rank higher; an item with several verdicts takes the highest.
RANK = {"correct": 0, "doubtful": 1, "wrong": 2}
_SPELLINGS = {
    "correct": "correct", "ok": "correct", "right": "correct", "true": "correct", "yes": "correct",
    "y": "correct", "1": "correct", "正确": "correct", "对": "correct",
    "wrong": "wrong", "error": "wrong", "incorrect": "wrong", "false": "wrong", "no": "wrong",
    "n": "wrong", "x": "wrong", "0": "wrong", "错误": "wrong", "错": "wrong",
    "doubtful": "doubtful", "doubt": "doubtful", "unsure": "doubtful", "unclear": "doubtful",
    "?": "doubtful", "存疑": "doubtful", "疑": "doubtful",
}


def wilson(k: int, n: int, z: float = Z95) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """``(p, lo, hi)``: the Wilson score interval of ``k`` successes in ``n`` trials.

    centre = (k + z^2/2) / (n + z^2), half-width = z sqrt(k(n-k)/n + z^2/4) / (n + z^2).
    ``n = 0`` gives ``(None, None, None)``. Never below 0 or above 1.
    """
    if n <= 0:
        return None, None, None
    if k < 0 or k > n:
        raise ValueError(f"k={k} outside 0..{n}")
    p = k / n
    z2 = z * z
    denom = n + z2
    centre = (k + z2 / 2) / denom
    half = z * math.sqrt(k * (n - k) / n + z2 / 4) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def normalise_verdict(text) -> Optional[str]:
    """``correct`` / ``wrong`` / ``doubtful``, or ``None`` for an empty or unknown cell."""
    if text is None:
        return None
    key = str(text).strip().lower()
    if not key:
        return None
    return _SPELLINGS.get(key, _SPELLINGS.get(key.split()[0]))


def read_verdicts(paths: Sequence) -> list[dict]:
    """Every row of every CSV, verdict normalised, ``source`` = the file name."""
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for raw in csv.DictReader(fh):
                row = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
                if not row.get("id"):
                    continue
                row["verdict_raw"] = row.get("verdict", "")
                row["verdict"] = normalise_verdict(row.get("verdict"))
                row["source"] = Path(path).name
                rows.append(row)
    return rows


def _task_order(task: str) -> tuple:
    digits = "".join(ch for ch in task[1:] if ch.isdigit())
    return (int(digits or 0), task)


def _rate(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": _r(p), "ci95": [_r(lo), _r(hi)]}


def _r(x):
    return None if x is None else round(float(x), 5)


def summarise(rows: Iterable[dict], records: Optional[Iterable[dict]] = None) -> dict:
    """The report as data; see the module docstring for the rules."""
    rows = list(rows)
    by_id: dict[str, dict] = {}
    if records is not None:
        for r in records:
            by_id[r["id"]] = {"task": r["prompt"]["task"],
                              "readable_id": r["label"].get("readable_id", "")}
    verdicts: dict[str, list[dict]] = defaultdict(list)
    unknown: list[dict] = []
    for row in rows:
        if row["verdict"] is None:
            if row["verdict_raw"]:
                unknown.append({"id": row["id"], "verdict": row["verdict_raw"],
                                "source": row["source"]})
            continue
        verdicts[row["id"]].append(row)
    reviewers = sorted({r.get("reviewer") or "" for r in rows if r["verdict"]})
    items: dict[str, dict] = {}
    for rid, rs in verdicts.items():
        worst = max(rs, key=lambda r: RANK[r["verdict"]])
        task = by_id.get(rid, {}).get("task") or worst.get("task") or "?"
        items[rid] = {"id": rid, "task": task,
                      "readable_id": by_id.get(rid, {}).get("readable_id") or worst.get("readable_id", ""),
                      "verdict": worst["verdict"],
                      "verdicts": [{"verdict": r["verdict"], "reviewer": r.get("reviewer", ""),
                                    "note": r.get("note", ""), "source": r["source"]} for r in rs],
                      "note": " | ".join(f"{r.get('reviewer') or r['source']}: {r.get('note')}"
                                         if len(rs) > 1 else r.get("note", "")
                                         for r in rs if r.get("note"))}
    per_task: dict[str, dict] = {}
    tasks = sorted({it["task"] for it in items.values()}, key=_task_order)
    wrong_all = doubt_all = 0
    for task in tasks:
        mine = [it for it in items.values() if it["task"] == task]
        n = len(mine)
        wrong = sum(1 for it in mine if it["verdict"] == "wrong")
        doubt = sum(1 for it in mine if it["verdict"] == "doubtful")
        wrong_all += wrong
        doubt_all += doubt
        per_task[task] = {"reviewed": n, "wrong": wrong, "doubtful": doubt,
                          "correct": n - wrong - doubt,
                          "error_rate": _rate(wrong, n),
                          "error_or_doubt_rate": _rate(wrong + doubt, n)}
    n_all = len(items)
    overall = {"reviewed": n_all, "wrong": wrong_all, "doubtful": doubt_all,
               "correct": n_all - wrong_all - doubt_all,
               "error_rate": _rate(wrong_all, n_all),
               "error_or_doubt_rate": _rate(wrong_all + doubt_all, n_all)}
    flagged = sorted((it for it in items.values() if it["verdict"] != "correct"),
                     key=lambda it: (_task_order(it["task"]), it["readable_id"], it["id"]))
    # agreement where more than one reviewer marked an item
    multi = [it for it in items.values() if len(it["verdicts"]) > 1]
    pairs = agree = 0
    for it in multi:
        vs = [v["verdict"] for v in it["verdicts"]]
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                pairs += 1
                agree += vs[i] == vs[j]
    disagreements = [it for it in multi if len({v["verdict"] for v in it["verdicts"]}) > 1]
    not_reviewed = sorted(set(by_id) - set(items)) if by_id else []
    return {"files": sorted({r["source"] for r in rows}), "reviewers": reviewers,
            "rows": len(rows), "items_reviewed": n_all,
            "items_total": len(by_id) if by_id else None,
            "not_reviewed": not_reviewed, "unknown_verdicts": unknown,
            "overall": overall, "by_task": per_task,
            "flagged": [{k: it[k] for k in ("id", "task", "readable_id", "verdict", "note")}
                        for it in flagged],
            "agreement": {"items_with_several_verdicts": len(multi), "pairs": pairs,
                          "agree": agree, "rate": _r(agree / pairs) if pairs else None,
                          "disagreements": [{"id": it["id"], "task": it["task"],
                                             "readable_id": it["readable_id"],
                                             "verdicts": it["verdicts"]} for it in disagreements]}}


def _fmt(cell: dict) -> str:
    if cell["rate"] is None:
        return "–"
    lo, hi = cell["ci95"]
    return f"{cell['rate']:.1%} [{lo:.1%}, {hi:.1%}]"


def markdown(result: dict, name: str = "audit") -> str:
    o = result["overall"]
    total = result["items_total"]
    cover = (f"{result['items_reviewed']} of {total} items reviewed"
             if total is not None else f"{result['items_reviewed']} items reviewed")
    lines = [f"# {name}: human audit summary", "",
             f"{cover}; {result['rows']} CSV rows from {len(result['files'])} file(s): "
             + ", ".join(f"`{f}`" for f in result["files"])
             + (f"; reviewers: {', '.join(r or '(unnamed)' for r in result['reviewers'])}"
                if result["reviewers"] else "") + ".",
             "",
             "Intervals are Wilson score intervals at 95 %. *error* = marked wrong; "
             "*error or doubt* = marked wrong or doubtful. An item with several verdicts "
             "takes the worst one here; the disagreements are listed below.", "",
             "| task | reviewed | correct | wrong | doubtful | error rate [95 %] | "
             "error or doubt [95 %] |", "|---|---|---|---|---|---|---|"]
    for task, cell in result["by_task"].items():
        lines.append(f"| {task} | {cell['reviewed']} | {cell['correct']} | {cell['wrong']} | "
                     f"{cell['doubtful']} | {_fmt(cell['error_rate'])} | "
                     f"{_fmt(cell['error_or_doubt_rate'])} |")
    lines.append(f"| **all** | {o['reviewed']} | {o['correct']} | {o['wrong']} | {o['doubtful']} | "
                 f"{_fmt(o['error_rate'])} | {_fmt(o['error_or_doubt_rate'])} |")
    lines += ["", "## Items marked wrong or doubtful", ""]
    if result["flagged"]:
        lines += ["| task | item | id | verdict | note |", "|---|---|---|---|---|"]
        for it in result["flagged"]:
            note = (it["note"] or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {it['task']} | {it['readable_id']} | `{it['id']}` | {it['verdict']} | {note} |")
    else:
        lines.append("None.")
    agr = result["agreement"]
    if agr["items_with_several_verdicts"]:
        lines += ["", "## Agreement between reviewers", "",
                  f"{agr['items_with_several_verdicts']} items carry more than one verdict; "
                  f"{agr['agree']} of {agr['pairs']} verdict pairs agree"
                  + (f" ({agr['rate']:.1%})." if agr["rate"] is not None else ".")]
        if agr["disagreements"]:
            lines += ["", "| task | item | id | verdicts |", "|---|---|---|---|"]
            for it in agr["disagreements"]:
                vs = "; ".join(f"{v['reviewer'] or v['source']}: {v['verdict']}"
                               + (f" ({v['note']})" if v["note"] else "") for v in it["verdicts"])
                lines.append(f"| {it['task']} | {it['readable_id']} | `{it['id']}` | "
                             f"{vs.replace('|', '/')} |")
    if result["not_reviewed"]:
        lines += ["", f"## Not yet reviewed ({len(result['not_reviewed'])})", "",
                  ", ".join(f"`{i}`" for i in result["not_reviewed"][:60])
                  + (" ..." if len(result["not_reviewed"]) > 60 else "")]
    if result["unknown_verdicts"]:
        lines += ["", f"## Verdict cells that could not be read ({len(result['unknown_verdicts'])})", ""]
        lines += [f"* `{u['id']}` in `{u['source']}`: \"{u['verdict']}\""
                  for u in result["unknown_verdicts"][:40]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--csv", nargs="+", required=True, help="the filled sheets")
    ap.add_argument("--records", default=None,
                    help="the audit sample (audit.jsonl): task per id and what is still open")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="audit")
    args = ap.parse_args(argv)
    import vlm_io as IO

    rows = read_verdicts(args.csv)
    records = list(IO.read_jsonl(args.records)) if args.records else None
    result = summarise(rows, records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / f"{args.name}.summary.md").write_text(markdown(result, args.name), encoding="utf-8")
    o = result["overall"]
    rate = o["error_rate"]
    print(f"[vlm-audit-summary] {o['reviewed']} items reviewed, {o['wrong']} wrong, "
          f"{o['doubtful']} doubtful; error rate "
          + (f"{rate['rate']:.1%} [{rate['ci95'][0]:.1%}, {rate['ci95'][1]:.1%}]"
             if rate["rate"] is not None else "–")
          + f" -> {out / (args.name + '.summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
