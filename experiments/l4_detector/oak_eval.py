"""The OAK regression check: D13 oak1 / oak2 against the LS drafts, L3's held-out protocol.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.oak_eval --models l3hold13,v2_hold13

Detections from ``infer --set oak`` (L2's ROI = the drafts' union + 10 %, every
box >= 0.001 into the merge); GT = the ``ls:`` drafts rebuilt from the L4 DB
copy (identical to L3's ``ls_boxes_all.json`` for D13 oak1 / oak2: 817 boxes);
L2's ``det_eval.evaluate`` unchanged.  Writes ``eval/oak_check.csv``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l4_detector import env  # noqa: E402
from experiments.l4_detector.evaluate import write_csv  # noqa: E402
from experiments.l4_detector.infer import dets_path  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="l3hold13,v2_hold13")
    args = ap.parse_args(argv)
    env.bind_l2(env.OUT / "l2bind")
    from experiments.l2_detector import data as D
    from experiments.l2_detector.det_eval import evaluate

    by = D.index(D.load_boxes())
    rows = []
    for m in args.models.split(","):
        dets = json.loads(dets_path(m, "oak").read_text(encoding="utf-8"))
        for r in evaluate(dets, 13, by):
            if r["n_gt"]:
                rows.append({"model": m, **r})
    write_csv(env.OUT / "eval" / "oak_check.csv", rows)
    for r in rows:
        if r["cls"] in ("screw", "connector", "ram_latch"):
            print(f"  {r['model']:>10} {r['view']} {r['cls']:>10} n={r['n_gt']:>4} "
                  f"AP50={r['AP50']} hit={r['hit@0.10']} FP/f={r['FP/frame@0.10']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
