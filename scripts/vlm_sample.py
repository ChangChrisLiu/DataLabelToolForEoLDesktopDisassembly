"""Stratified, seeded draws from an export-vlm file (W3b).

``python scripts/vlm_sample.py --export <vlm_scan.jsonl> --out <dir>
[--purpose audit|repeat] [--size 300] [--per-task 25] [--seed 0]
[--test-desktops 50-66] [--max-share 0.2]``

Two uses, one procedure (minutes 2026-10-31 §3):

* ``audit`` -- the 300-item human audit of the frozen evaluation set: about
  ``--per-task`` per task, positives and negatives balanced where a task has
  both, every test desktop represented. The user marks each row right or
  wrong; 0 errors in 300 bounds the error rate at about 1 % (95 %).
* ``repeat`` -- the 300-item subset the closed models are run on three times
  for the standard deviation. Same strata, a different seed stream, so it is
  not the audit sample.

No task exceeds ``--max-share`` of the draw (V2 was 64 % of the day-1 export).
The test desktops are ``--test-desktops`` when given, else those whose
records say ``split: test``, else every desktop.

Deterministic: the order is a SHA-256 of seed, purpose and record id, never
``random`` or ``hash()``. Writes ``<purpose>.ids.txt``, ``<purpose>.jsonl``
(the records) and ``<purpose>.sheet.md`` (a reviewer sheet: image path,
question, options, answer, and an empty verdict column).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vlm_io as IO  # noqa: E402


def _key(seed: int, purpose: str, text: str) -> str:
    return hashlib.sha256(f"{seed}|{purpose}|{text}".encode("utf-8")).hexdigest()


def draw(records: list[dict], *, size: int = 300, per_task: int = 25, seed: int = 0,
         purpose: str = "audit", test_desktops: Optional[set[int]] = None,
         max_share: float = 0.2) -> list[dict]:
    """The sample, in task order then draw order."""
    if test_desktops is None:
        flagged = {IO.desktop_of(r) for r in records if r["label"].get("split") == "test"}
        test_desktops = flagged or None
    pool = [r for r in records if test_desktops is None
            or IO.desktop_of(r) in test_desktops]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_task[r["prompt"]["task"]].append(r)
    tasks = sorted(by_task, key=lambda t: int(t[1:]))
    if not tasks:
        return []
    cap = max(1, int(max_share * size))
    quota = {t: min(per_task, len(by_task[t]), cap) for t in tasks}
    # what the small tasks cannot fill goes to the others, up to the cap
    while sum(quota.values()) < size:
        room = [t for t in tasks if quota[t] < min(len(by_task[t]), cap)]
        if not room:
            break
        for t in room:
            if sum(quota.values()) >= size:
                break
            quota[t] += 1

    picked: dict[str, list[dict]] = {}
    for task in tasks:
        picked[task] = _stratified(by_task[task], quota[task], seed, purpose)

    # every desktop at least once: swap into the largest task, same stratum first
    desks = sorted({IO.desktop_of(r) for r in pool})
    for desk in desks:
        if any(IO.desktop_of(r) == desk for rows in picked.values() for r in rows):
            continue
        candidates = sorted((r for r in pool if IO.desktop_of(r) == desk),
                            key=lambda r: _key(seed, purpose, r["id"]))
        for cand in candidates:
            task = cand["prompt"]["task"]
            rows = picked.get(task) or []
            counts = defaultdict(int)
            for r in rows:
                counts[IO.desktop_of(r)] += 1
            victims = [i for i, r in enumerate(rows) if counts[IO.desktop_of(r)] > 1
                       and IO.is_positive(r) == IO.is_positive(cand)]
            if victims:
                rows[victims[-1]] = cand
                break
    return [r for task in tasks for r in picked[task]]


def _stratified(rows: list[dict], n: int, seed: int, purpose: str) -> list[dict]:
    """``n`` rows: strata (positive / negative) alternated, desktops round-robin."""
    strata: dict[Optional[bool], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in sorted(rows, key=lambda r: _key(seed, purpose, r["id"])):
        strata[IO.is_positive(r)][IO.desktop_of(r)].append(r)
    order = sorted(strata, key=lambda s: (s is None, s is False))
    queues = {}
    for s in order:
        desks = sorted(strata[s], key=lambda d: _key(seed, purpose, f"desk{d}"))
        queues[s] = (desks, {d: list(v) for d, v in strata[s].items()}, [0])
    out: list[dict] = []
    while len(out) < n and any(any(q[1].values()) for q in queues.values()):
        for s in order:
            if len(out) >= n:
                break
            desks, lists, cursor = queues[s]
            for _ in range(len(desks)):
                d = desks[cursor[0] % len(desks)]
                cursor[0] += 1
                if lists[d]:
                    out.append(lists[d].pop(0))
                    break
    return out


def sheet(rows: list[dict], manifest: dict) -> str:
    """A Markdown table a reviewer fills in."""
    lines = ["| # | id | task | D / view / step | image | question | answer | ok? |",
             "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, start=1):
        label = r["label"]
        paths = " ; ".join(manifest.get(img, {}).get("path", img)
                           for img in r["prompt"].get("images") or [])
        question = r["prompt"]["question"].replace("\n", " ")
        if r["prompt"].get("options"):
            question += " " + " ".join(f"({chr(65 + k)}) {o['verb']} {o['target_label']}"
                                       for k, o in enumerate(r["prompt"]["options"]))
        answer = json.dumps(label["answer"], ensure_ascii=False)
        if len(answer) > 300:
            answer = answer[:297] + "..."
        cells = [str(i), r["id"], r["prompt"]["task"],
                 f"D{label['desktop']:02d} / {label['view']} / {label['step']}",
                 paths, question, answer, ""]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--purpose", default="audit", choices=("audit", "repeat"))
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--per-task", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-desktops", default=None)
    ap.add_argument("--max-share", type=float, default=0.2)
    args = ap.parse_args(argv)
    from vlm_references import _parse_range

    records = list(IO.read_jsonl(args.export))
    test = _parse_range(args.test_desktops) if args.test_desktops else None
    rows = draw(records, size=args.size, per_task=args.per_task, seed=args.seed,
                purpose=args.purpose, test_desktops=test, max_share=args.max_share)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.purpose}.ids.txt").write_text("".join(r["id"] + "\n" for r in rows),
                                                encoding="utf-8")
    IO.write_jsonl(out / f"{args.purpose}.jsonl", rows)
    (out / f"{args.purpose}.sheet.md").write_text(sheet(rows, IO.load_manifest(args.export)),
                                                 encoding="utf-8")
    by_task: dict[str, int] = defaultdict(int)
    for r in rows:
        by_task[r["prompt"]["task"]] += 1
    print(f"[vlm-sample] {args.purpose}: {len(rows)} records from "
          f"{len({IO.desktop_of(r) for r in rows})} desktops -> {out}: "
          + " ".join(f"{t}={n}" for t, n in sorted(by_task.items(), key=lambda x: int(x[0][1:]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
