"""One table across folds and recipes for the report.

::

    python -m experiments.l5_sam.summary   # -> OUT/tables/summary.md

For each (fold, model) present in ``OUT/eval/{f13,f5,f13_v2,f5_v2}.csv``: the
prompt-averaged IoU / BF (box, point, point_box) per group, the loose box and
the refine IoU, and the gate verdicts from ``OUT/tables/<csv>_gate.json``.
"""
from __future__ import annotations

from experiments.l5_sam import env  # noqa: F401

import json
import sys

from experiments.l5_sam import aggregate as A
from experiments.l5_sam import common as C

RUNS = (("f13", "f13", 13, "v1"), ("f13_v2", "f13_v2", 13, "v2"),
        ("f5", "f5", 5, "v1"), ("f5_v2", "f5_v2", 5, "v2"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lines = ["| held-out | model | " + " | ".join(f"{g} IoU / BF" for g in A.GROUPS)
             + " | loose box: small-macro / large | refine macro | gate |",
             "|" + "---|" * (len(A.GROUPS) + 5)]
    seen_stock = set()
    for csv, ft, desk, recipe in RUNS:
        path = env.OUT / "eval" / f"{csv}.csv"
        if not path.exists():
            continue
        d = A.load(csv)
        d = d[d["desktop"] == desk]
        gate_path = env.OUT / "tables" / f"{csv}_gate.json"
        gate = json.loads(gate_path.read_text("utf-8")) if gate_path.exists() else None
        for model in ("stock", ft):
            if model == "stock" and desk in seen_stock:
                continue
            s = d[d["model"] == model]
            cells = []
            for g in A.GROUPS:
                q = s[A.in_group(s, g) & s["prompt"].isin(C.GATED_PROMPTS)]
                iou = q.groupby("prompt")["iou"].mean().mean()
                bf = q.groupby("prompt")["bf"].mean().mean()
                cells.append(f"{iou:.3f} / {bf:.3f}")
            lb = s[s["prompt"] == "box_loose"]
            macro_lb = sum(lb[A.in_group(lb, g)]["iou"].mean() for g in A.SMALL_GROUPS) / 4
            large_lb = lb[A.in_group(lb, "large")]["iou"].mean()
            rf = s[s["prompt"] == "refine"]
            macro_rf = sum(rf[A.in_group(rf, g)]["iou"].mean() for g in A.SMALL_GROUPS) / 4
            verdict = "–"
            if model != "stock" and gate:
                failed = [k for k, v in gate["gate"].items() if not v]
                verdict = "PASS" if gate["passed"] else "FAIL " + ",".join(failed)
            name = "stock SAM 2.1" if model == "stock" else f"{recipe} ({model})"
            lines.append(f"| D{desk} | {name} | " + " | ".join(cells)
                         + f" | {macro_lb:.3f} / {large_lb:.3f} | {macro_rf:.3f} | {verdict} |")
            if model == "stock":
                seen_stock.add(desk)
    text = ("Prompt-averaged (box, point, point_box) IoU / boundary F of the shown "
            "candidate; groups as decision_rule.txt.\n\n" + "\n".join(lines) + "\n")
    out = env.OUT / "tables" / "summary.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
