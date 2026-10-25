"""U30-QA, pass 5b: how far each "drawn below but on top in the picture" finding reaches.

For every high ``visible_changed`` flag of ``u30_analyse.py`` (part ``i`` keeps its
look-alike visible pixels while the part ``m`` that lay over them moves away at
step ``s``), count the frames before ``s`` on which ``m`` is in the same shape
version and ``i``'s stored visible mask still overlaps ``m``'s amodal shape by
>= 30 px: those are the frames whose layer order puts ``m`` under ``i``.
Writes ``<root>/reach.json``; read-only otherwise.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import u30_common as U  # noqa: E402
import u30_analyse as A  # noqa: E402


def main() -> int:
    items = json.load(open(os.path.join(U.QA_ROOT, "all_items.json"), encoding="utf-8"))
    flags = [f for f in items if f["code"] == "visible_changed" and f["severity"] == "high"
             and f.get("movers")]
    by_desk = defaultdict(list)
    for f in flags:
        by_desk[f["desktop"]].append(f)
    out = {}
    for d, fl in sorted(by_desk.items()):
        dk = A.Desk(d)
        rows = []
        for f in fl:
            s0 = f["steps"][0]
            inst = f["inst"]
            for m in f["movers"]:
                kf = (dk.R.get((s0, m)) or {}).get("kf_id")
                frames = []
                for s in dk.steps:
                    if s > s0:
                        break
                    rm, ri = dk.R.get((s, m)), dk.R.get((s, inst))
                    if not rm or not ri or rm.get("kf_id") != kf or not A.present(ri):
                        continue
                    am, vi = dk.amodal(s, m), dk.visible(s, inst)
                    if am is None or vi is None:
                        continue
                    if int((am & vi).sum()) >= 30:
                        frames.append(s)
                    dk.drop_masks(keep_steps=())
                rows.append({"inst": inst, "mover": m, "at": f["frames"], "frames": frames})
        out[d] = rows
        n = sum(len(r["frames"]) for r in rows)
        print(f"D{d:02d}: {len(rows)} part(s) drawn over a part that covers them, {n} part-frames",
              flush=True)
    with open(os.path.join(U.QA_ROOT, "reach.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
