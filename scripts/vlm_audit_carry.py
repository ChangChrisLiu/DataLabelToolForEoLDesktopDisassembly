"""Carry the first author audit's V2 verdicts over to a redrawn audit sample (spec 6.7).

``python tools/vlm_audit_carry.py --old-csv <old page CSVs ...> --old-records <old audit.jsonl>
--new-records <new audit.jsonl> --out <carried.csv> [--batch 50]``

The questions were redesigned after the first audit (parts named by a red mark instead of an
ordinal, interchangeable parts asked as a group, action tasks multi-view). A **V2** verdict
still says something about a new item when the new item asks the same thing of the same
picture, so it is carried when all of these match:

* desktop, step and view (a verdict judged that picture, not another camera's);
* the part: the same count question (``screw_count`` with the same ``role`` and ``state``),
  or the same instance -- or a group the old instance is a member of (the group's members from
  ``answer_check.members``, else from ``evidence.instances``);
* the gold answer (the ``answer_check.fields`` of both, else the whole answer).

Everything else -- V3 / V4 and every other task (re-asked, now multi-view), a V2 whose answer
changed, an item with no old verdict -- is left blank, to be audited afresh. Nothing is forced
into the new sample.

The output has the audit sheet's columns (``id, task, readable_id, desktop, view, step, batch,
verdict, note, reviewer, answered_at``), one row per new record in the sample's order (``batch``
numbered like the sheet's pages), and one row per carried verdict for a record that several old
verdicts carry to (several reviewers). A carried row keeps the old reviewer and time and
starts its note with ``[carried from <old readable id> (<old id>)]``.
``tools/vlm_audit_summary.py`` reads it together with the new pages' exports. The old files
are only read.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vlm_io as IO  # noqa: E402
from vlm_audit_summary import read_verdicts  # noqa: E402

__all__ = ["CSV_COLUMNS", "carry", "main", "match_key", "part_of", "to_csv"]

#: The audit sheet's columns (``vlm_audit_sheet.CSV_COLUMNS``; a test keeps them equal).
CSV_COLUMNS = ("id", "task", "readable_id", "desktop", "view", "step", "batch", "verdict",
               "note", "reviewer", "answered_at")
DEFAULT_BATCH = 50


def part_of(record: dict) -> Optional[tuple]:
    """What a V2 record asks about: ``("count", role, state)``, ``("state", key, members)``
    (``members`` = the group's member keys, empty for a single part), or ``None``."""
    if record["prompt"]["task"] != "V2":
        return None
    label = record["label"]
    check = label.get("answer_check") or {}
    if check.get("derive") == "screw_count":
        return ("count", str(check.get("role")), str(check.get("state", "present")))
    key = check.get("instance") or check.get("group")
    if not key:
        return None
    members: tuple = ()
    if "@" in str(key) or check.get("members"):
        members = tuple(str(k) for k in (check.get("members")
                                         or (label.get("evidence") or {}).get("instances") or []))
    return ("state", str(key), members)


def _answer(record: dict) -> tuple:
    label = record["label"]
    answer = (label.get("answer_check") or {}).get("answer_keys") or label.get("answer") or {}
    fields = (label.get("answer_check") or {}).get("fields") or sorted(answer)
    return tuple((f, repr(answer.get(f))) for f in fields)


def match_key(record: dict) -> tuple:
    label = record["label"]
    return (int(label.get("desktop", -1)), int(label.get("step", -1)), str(label.get("view", "")))


def _same_part(old: tuple, new: tuple) -> bool:
    if old[0] != new[0]:
        return False
    if old[0] == "count":
        return old == new
    return old[1] == new[1] or old[1] in new[2]


def carry(verdict_rows: Iterable[dict], old_records: Iterable[dict],
          new_records: Iterable[dict]) -> dict:
    """``{"carried": {new id: [old verdict row + old record]}, ...}`` and the counts.

    ``verdict_rows`` are :func:`vlm_audit_summary.read_verdicts` rows (``verdict`` normalised);
    blank ones are skipped, and a page exported twice by one reviewer counts once (the later
    answer).
    """
    old_by_id = {r["id"]: r for r in old_records}
    latest: dict[tuple, dict] = {}
    for row in verdict_rows:
        if row.get("verdict") is None:
            continue
        key = (row["id"], row.get("reviewer", ""))
        if key not in latest or (row.get("answered_at") or "") >= \
                (latest[key].get("answered_at") or ""):
            latest[key] = row
    unknown = sorted({rid for rid, _ in latest if rid not in old_by_id})
    old_v2 = [(row, old_by_id[row["id"]]) for row in latest.values()
              if row["id"] in old_by_id and old_by_id[row["id"]]["prompt"]["task"] == "V2"]
    carried: dict[str, list[tuple[dict, dict]]] = {}
    used: set[tuple] = set()
    new_records = list(new_records)
    for new in new_records:
        new_part = part_of(new)
        if new_part is None:
            continue
        for row, old in old_v2:
            old_part = part_of(old)
            if old_part is None or match_key(old) != match_key(new):
                continue
            if _same_part(old_part, new_part) and _answer(old) == _answer(new):
                carried.setdefault(new["id"], []).append((row, old))
                used.add((row["id"], row.get("reviewer", "")))
    left = sorted({(row["id"], old["label"].get("readable_id", "")) for row, old in old_v2
                   if (row["id"], row.get("reviewer", "")) not in used})
    return {"carried": carried, "unknown_old_ids": unknown,
            "old_verdicts": len(latest), "old_v2_verdicts": len(old_v2),
            "not_carried_v2": left, "new_records": len(new_records)}


def _blank(record: dict, batch: int) -> dict:
    label = record["label"]
    return {"id": record["id"], "task": record["prompt"]["task"],
            "readable_id": label.get("readable_id", ""), "desktop": label.get("desktop", ""),
            "view": label.get("view", ""), "step": label.get("step", ""), "batch": batch,
            "verdict": "", "note": "", "reviewer": "", "answered_at": ""}


def to_csv(new_records: Sequence[dict], carried: dict, batch: int = DEFAULT_BATCH) -> str:
    """The carried CSV: every new record in order, carried verdicts filled in."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator="\r\n")
    writer.writeheader()
    batch = max(1, int(batch))
    for i, record in enumerate(new_records):
        base = _blank(record, i // batch + 1)
        found = carried.get(record["id"]) or []
        if not found:
            writer.writerow(base)
            continue
        for row, old in found:
            note = f"[carried from {old['label'].get('readable_id', '')} ({old['id']})]"
            if row.get("note"):
                note += " " + row["note"]
            writer.writerow(dict(base, verdict=row["verdict"], note=note,
                                 reviewer=row.get("reviewer", ""),
                                 answered_at=row.get("answered_at", "")))
    return buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--old-csv", nargs="+", required=True,
                    help="the CSVs exported from the first audit's pages")
    ap.add_argument("--old-records", required=True, help="the first audit sample (audit.jsonl)")
    ap.add_argument("--new-records", required=True, help="the redrawn audit sample (audit.jsonl)")
    ap.add_argument("--out", required=True, help="the carried CSV")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help="items per page of the new sheet (for the batch column)")
    args = ap.parse_args(argv)
    if Path(args.out).resolve() in {Path(p).resolve() for p in args.old_csv}:
        raise SystemExit("--out must not be one of the --old-csv files (they are only read)")
    old = list(IO.read_jsonl(args.old_records))
    new = list(IO.read_jsonl(args.new_records))
    result = carry(read_verdicts(args.old_csv), old, new)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("﻿" + to_csv(new, result["carried"], args.batch), encoding="utf-8",
                   newline="")
    n_rows = sum(len(v) for v in result["carried"].values())
    print(f"[vlm-audit-carry] {result['old_verdicts']} old verdicts, {result['old_v2_verdicts']} "
          f"on V2; carried {n_rows} to {len(result['carried'])} of {result['new_records']} new "
          f"items -> {out}")
    if result["not_carried_v2"]:
        print(f"[vlm-audit-carry] {len(result['not_carried_v2'])} V2 verdicts not carried (no new "
              "item with the same desktop, step, view, part and answer):")
        for rid, readable in result["not_carried_v2"]:
            print(f"  {readable} ({rid})")
    if result["unknown_old_ids"]:
        print(f"[vlm-audit-carry] warning: {len(result['unknown_old_ids'])} verdict ids are not "
              f"in --old-records: {', '.join(result['unknown_old_ids'][:10])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
