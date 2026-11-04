"""The frozen evaluation subset of the VLM pack (W4-A; paper plan 6.2).

``python scripts/vlm_eval_set.py --export <vlm_scan.jsonl> [<vlm_oak1.jsonl> ... <vlm_multi.jsonl>]
--out <dir>/eval_v2 [--seed 0] [--target-binary 400] [--target-other 300]
[--max-share 0.2] [--audit 300]``

Every model is run on the same subset, drawn once from the export units over
**all** desktops (zero-shot: no training split is spent). Since the redesign of
2026-11-03 the subset is ``eval_v2``: perception (V1 V2 V8 V14 V15) comes from
the four camera units, the action tasks (V3 V4 V6 V7 V10 V12 V16) from the
``multi`` unit, with the same per-task targets as ``eval_v1``; the command line
refuses an ``eval_v1`` path (:func:`refuse_old_eval`). Per task:

* at least ``--target-binary`` records for the yes / no tasks (V4 feasible or
  not, V12 changed or not, V14 answerable or not, V15 same moment or not) and
  at least ``--target-other`` for the rest; V6 is drawn as **pairs** (open +
  multiple choice of one frame, ``label.v6_pair``), so its 300 units are 600
  records;
* the records are allocated **per unit in proportion to the export** (a task
  lives in one unit since the redesign, so this is the identity unless several
  units hold it), then **equally over the task's
  strata** (:func:`strata`), level by level: V4 feasible / infeasible, the
  infeasible half over ``truth_source``; V2 ``prior_is_correct`` false / true,
  then class; V12 changed / unchanged, then the negative's kind; V14 answerable
  / not; V15 same / different, then the view pair; V16 valid / invalid, then
  the corruption family; V3 by verb, V8 by the target's class, V10 by progress
  bin. A stratum that cannot fill its share hands the rest to its siblings;
* inside a stratum the desktops take turns (round-robin in a seeded order),
  and no desktop holds more than ``--max-share`` of a task;
* what the strata could not fill under those caps is topped up from the task's
  remaining records in seeded order.

Deterministic: every order is a SHA-256 of the seed and a record id (or a
desktop number), never ``random`` or ``hash()``. Two runs over the same export
write the same bytes.

Writes ``ids.txt`` (one record id per line, task order), ``<view>.jsonl`` (the
chosen records of each export, in the export's order) with the sidecars the
scorer reads beside it -- ``<view>.images_manifest.jsonl`` and
``<view>.oracle.jsonl`` cut down to what the subset names, ``<view>.summary.json``
copied whole (its shortcut table is the full export's in-sample bound) -- and
the n table ``table.md`` + ``table.json``. With ``--audit N`` the human-audit
sample of :mod:`vlm_sample` (``--purpose audit``) is drawn **from inside** the
subset into ``audit/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vlm_io as IO  # noqa: E402

__all__ = ["BINARY_TASKS", "EVAL_VERSION", "OLD_EVAL_VERSIONS", "TASK_ORDER",
           "allocate_equal", "allocate_proportional", "build", "draw", "refuse_old_eval",
           "strata", "table_markdown", "units_of"]

#: The subset's name; bumped when the draw or its parameters change (eval_v2: the
#: redesign of 2026-11-03 -- marks, part groups, multi-view action tasks).
EVAL_VERSION = "eval_v2"
#: Subsets the current records no longer match: a path through one is refused.
OLD_EVAL_VERSIONS = ("eval_v1",)
#: Tasks whose answer is a yes / no: the larger target.
BINARY_TASKS = frozenset({"V4", "V12", "V14", "V15"})
TARGET_BINARY = 400
TARGET_OTHER = 300
MAX_SHARE = 0.2
SEED = 0
#: Table order (the spec's codes; V5 is not in the default export).
TASK_ORDER = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V10", "V12", "V14",
              "V15", "V16")
#: The export units a record can come from, in the export's order: the four camera views,
#: then ``multi`` (the action tasks, every camera at once).
VIEWS = ("scan", "oak1", "oak2", "rs", "multi")


def refuse_old_eval(*paths) -> None:
    """``SystemExit`` whose message names the current subset when a path goes through
    a superseded subset directory (``eval_v1``): its records are the old questions,
    and scored with the new tools they would mean something else. The interpreter
    prints the message and exits 1."""
    for path in paths:
        if path is None:
            continue
        parts = Path(str(path)).parts
        old = next((p for p in parts for v in OLD_EVAL_VERSIONS
                    if p == v or p.startswith(v + ".")), None)
        if old is not None:
            raise SystemExit(f"[vlm] refused: {path} is under {old}, the subset before the "
                             f"question redesign of 2026-11-03; use {EVAL_VERSION}/ (the same "
                             f"layout, with a multi unit for the action tasks)")


# --------------------------------------------------------------------------- #
# strata
# --------------------------------------------------------------------------- #
def _bool(value) -> str:
    return "yes" if value else "no"


def v2_class(record: dict) -> str:
    """The class a V2 question is about (a screw role for the counts)."""
    check = record["label"]["answer_check"]
    if check.get("derive") == "screw_count":
        return f"screw:{check.get('role')}"
    return str(check.get("class") or "?")


#: The largest share one answer may have inside a V4 class of the subset (final review I1).
V4_MAX_MAJORITY = 0.6


def v4_class(record: dict) -> str:
    """The class a V4 question names: its match class (screw role, connector kind), else
    its taxonomy class."""
    check = record["label"].get("answer_check") or {}
    return str(check.get("match_class") or check.get("target_class") or "?")


def balance_v4(units: Sequence["Unit"], seed: int,
               max_majority: float = V4_MAX_MAJORITY) -> list["Unit"]:
    """V4's pool with each (view, class) cut to at most ``max_majority`` of one answer.

    The minority answer of a class is kept whole and the majority cut, in a seeded order,
    to ``minority * max_majority / (1 - max_majority)``; a class with one answer only
    is left out (a text-only reader would answer it from the class name). The draw then
    splits each class equally between yes and no, so no class of the subset leans past
    ``max_majority`` (the pool's own lean is in the export's ``shortcuts``).
    """
    import math

    cells: dict[tuple, dict[bool, list]] = defaultdict(lambda: {True: [], False: []})
    for u in units:
        feasible = not u.records[0]["label"].get("negative")
        cells[(u.view, v4_class(u.records[0]))][feasible].append(u)
    keep: set = set()
    for (view, cls), sides in cells.items():
        small = min(len(sides[True]), len(sides[False]))
        if small == 0:
            continue
        room = math.floor(small * max_majority / (1.0 - max_majority) + 1e-9)
        for side in (True, False):
            ordered = sorted(sides[side], key=lambda u: _key(seed, "v4bal", u.key))
            keep.update(u.key for u in ordered[:room])
    return [u for u in units if u.key in keep]


def strata(record: dict) -> tuple[str, ...]:
    """The task-specific stratum path of a record (view and desktop apart).

    Level by level, each level split equally: the first element is the
    balance the paper reports on (positive / negative, prior right / wrong),
    the later ones keep the long tail represented.
    """
    task = record["prompt"]["task"]
    label = record["label"]
    answer = label.get("answer") or {}
    if task == "V2":
        return (f"prior_correct={_bool(label.get('prior_is_correct'))}", v2_class(record))
    if task == "V3":
        return (f"verb={answer.get('verb')}",)
    if task == "V4":
        # the class the question names first, then yes / no inside it (final review I1:
        # "motherboard" was "no" 51 of 56 times in eval_v2) -- :func:`balance_v4` keeps
        # each class's yes / no within V4_MAX_MAJORITY before the draw
        feasible = not label.get("negative")
        return ((f"class={v4_class(record)}", f"feasible={_bool(feasible)}")
                + (() if feasible else (f"source={label.get('truth_source')}",)))
    if task == "V8":
        if label.get("answer_check", {}).get("type") == "components":
            return ()  # every visible part of a class: a set answer like V1's, no stratum
        instance = str(answer.get("instance") or "?")
        return (f"class={instance.split('.', 1)[0]}",)
    if task == "V10":
        return (f"progress={answer.get('progress_bin')}",)
    if task == "V12":
        changed = not label.get("negative")
        return ((f"changed={_bool(changed)}",)
                + (() if changed else (f"kind={label.get('negative')}",)))
    if task == "V14":
        return (f"answerable={_bool(not label.get('negative'))}",)
    if task == "V15":
        same = not label.get("negative")
        return (f"same_moment={_bool(same)}", "pair=" + "+".join(label.get("views") or []))
    if task == "V16":
        valid = not label.get("negative")
        return ((f"valid={_bool(valid)}",)
                + (() if valid else (f"corruption={label.get('negative')}",)))
    return ()


def _task_order(task: str) -> tuple:
    digits = "".join(ch for ch in task[1:] if ch.isdigit())
    return (int(digits or 0), task)


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #
class Unit:
    """What the draw picks as one: a record, or V6's open + multiple-choice pair."""

    __slots__ = ("key", "records", "task", "view", "desktop", "path")

    def __init__(self, key: str, records: list[dict], view: str):
        self.key = key
        self.records = records
        self.task = records[0]["prompt"]["task"]
        self.view = view
        self.desktop = IO.desktop_of(records[0])
        self.path = strata(records[0])

    def __repr__(self) -> str:  # pragma: no cover - debugging
        return f"Unit({self.key}, {self.task}, {self.view}, D{self.desktop})"


def units_of(records: Iterable[dict], view: str) -> list[Unit]:
    """Group an export's records into units; V6 pairs share ``label.v6_pair``."""
    pairs: dict[str, list[dict]] = {}
    order: list[str] = []
    for rec in records:
        if rec["prompt"]["task"] == "V6" and rec["label"].get("v6_pair"):
            key = "pair:" + str(rec["label"]["v6_pair"])
        else:
            key = rec["id"]
        if key not in pairs:
            pairs[key] = []
            order.append(key)
        pairs[key].append(rec)
    return [Unit(key, pairs[key], view) for key in order]


# --------------------------------------------------------------------------- #
# allocation
# --------------------------------------------------------------------------- #
def allocate_equal(avail: dict, n: int) -> dict:
    """Split ``n`` equally over the strata, capped by what each holds.

    A stratum that cannot take its share leaves the rest to the others, in
    rounds, until ``n`` is placed or every stratum is exhausted.
    """
    keys = sorted(avail)
    quota = {k: 0 for k in keys}
    remaining = max(0, int(n))
    while remaining > 0:
        room = [k for k in keys if quota[k] < avail[k]]
        if not room:
            break
        share = max(1, remaining // len(room))
        for k in room:
            take = min(share, avail[k] - quota[k], remaining)
            quota[k] += take
            remaining -= take
            if remaining == 0:
                break
    return quota


def allocate_proportional(avail: dict, n: int) -> dict:
    """Split ``n`` in proportion to the strata's sizes (largest remainder)."""
    keys = sorted(avail)
    total = sum(avail[k] for k in keys)
    n = min(int(n), total)
    if total == 0 or n <= 0:
        return {k: 0 for k in keys}
    exact = {k: n * avail[k] / total for k in keys}
    quota = {k: int(exact[k]) for k in keys}
    left = n - sum(quota.values())
    # the largest remainders first; ties by key so the result is a function of
    # the inputs alone
    for k in sorted(keys, key=lambda k: (-(exact[k] - quota[k]), k)):
        if left <= 0:
            break
        if quota[k] < avail[k]:
            quota[k] += 1
            left -= 1
    return quota


def _key(seed: int, *parts) -> str:
    return hashlib.sha256("|".join([str(seed), *map(str, parts)]).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# the draw
# --------------------------------------------------------------------------- #
def _draw_leaf(units: list[Unit], n: int, seed: int, counts: dict[int, int], cap: int
               ) -> list[Unit]:
    """``n`` units, desktops taking turns, none past ``cap`` units of its task.

    Each round goes over the desktops **least used by this task so far** first
    (ties in a seeded order that depends on the stratum), so the strata of one
    task spread over different desktops instead of all starting from the same
    few.
    """
    if not units or n <= 0:
        return []
    by_desk: dict[int, list[Unit]] = defaultdict(list)
    for u in sorted(units, key=lambda u: _key(seed, u.key)):
        by_desk[u.desktop].append(u)
    first = units[0]
    tie = {d: _key(seed, first.task, first.view, "/".join(first.path), "desk", d)
           for d in by_desk}
    out: list[Unit] = []
    while len(out) < n:
        progressed = False
        for d in sorted(by_desk, key=lambda d: (counts[d], tie[d])):
            if len(out) >= n:
                break
            if counts[d] >= cap or not by_desk[d]:
                continue
            out.append(by_desk[d].pop(0))
            counts[d] += 1
            progressed = True
        if not progressed:
            break
    return out


def _draw_tree(units: list[Unit], n: int, level: int, seed: int,
               counts: dict[int, int], cap: int,
               allocate: Callable[[dict, int], dict]) -> list[Unit]:
    """Allocate over the ``level``-th stratum key, then recurse into each."""
    groups: dict[str, list[Unit]] = defaultdict(list)
    leaves: list[Unit] = []
    for u in units:
        if level < len(u.path):
            groups[u.path[level]].append(u)
        else:
            leaves.append(u)
    if not groups:
        return _draw_leaf(leaves, n, seed, counts, cap)
    if leaves:
        groups[""] = leaves  # records with a shorter path: a stratum of their own
    quota = allocate({k: len(v) for k, v in groups.items()}, n)
    out: list[Unit] = []
    for k in sorted(groups):
        if quota[k]:
            out.extend(_draw_tree(groups[k], quota[k], level + 1, seed, counts, cap,
                                  allocate_equal))
    return out


def draw(units: Sequence[Unit], *, seed: int = SEED, target_binary: int = TARGET_BINARY,
         target_other: int = TARGET_OTHER, max_share: float = MAX_SHARE,
         targets: Optional[dict[str, int]] = None) -> dict[str, list[Unit]]:
    """``task -> chosen units``; see the module docstring for the procedure."""
    by_task: dict[str, list[Unit]] = defaultdict(list)
    for u in units:
        by_task[u.task].append(u)
    out: dict[str, list[Unit]] = {}
    for task in sorted(by_task, key=_task_order):
        pool = by_task[task]
        if task == "V4":
            pool = balance_v4(pool, seed)
        n = (targets or {}).get(task)
        if n is None:
            n = target_binary if task in BINARY_TASKS else target_other
        n = min(int(n), len(pool))
        cap = max(1, int(max_share * n)) if n else 0
        counts: dict[int, int] = defaultdict(int)
        # level 0: the view, in proportion; the task's strata below it, equally
        by_view: dict[str, list[Unit]] = defaultdict(list)
        for u in pool:
            by_view[u.view].append(u)
        quota = allocate_proportional({v: len(us) for v, us in by_view.items()}, n)
        chosen: list[Unit] = []
        for view in sorted(by_view):
            if quota[view]:
                chosen.extend(_draw_tree(by_view[view], quota[view], 0, seed, counts, cap,
                                         allocate_equal))
        # the caps may have left strata short: top up from what is left, in
        # seeded order, under the same caps
        if len(chosen) < n:
            taken = {u.key for u in chosen}
            rest = [u for u in pool if u.key not in taken]
            chosen.extend(_draw_leaf(rest, n - len(chosen), seed + 1, counts, cap))
        out[task] = chosen
    return out


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def table_rows(chosen: dict[str, list[Unit]], pool: dict[str, list[Unit]],
               targets: dict[str, int]) -> list[dict]:
    rows = []
    for task in sorted(pool, key=_task_order):
        units = chosen.get(task, [])
        records = [r for u in units for r in u.records]
        desks: dict[int, int] = defaultdict(int)
        views: dict[str, int] = defaultdict(int)
        by_stratum: dict[str, int] = defaultdict(int)
        for u in units:
            desks[u.desktop] += 1
            views[u.view] += 1
            for depth in range(1, len(u.path) + 1):
                by_stratum[" / ".join(u.path[:depth])] += 1
        n_units = len(units)
        rows.append({
            "task": task, "target": targets[task], "units": n_units, "records": len(records),
            "pool_units": len(pool[task]),
            "pool_records": sum(len(u.records) for u in pool[task]),
            "unit": "pair" if task == "V6" else "record",
            "by_view": dict(sorted(views.items(), key=lambda kv: VIEWS.index(kv[0])
                                   if kv[0] in VIEWS else 99)),
            "by_stratum": dict(sorted(by_stratum.items())),
            "desktops": len(desks),
            "max_desktop_share": round(max(desks.values()) / n_units, 4) if n_units else 0.0,
        })
    return rows


def table_markdown(rows: list[dict], *, seed: int, max_share: float) -> str:
    lines = [f"# {EVAL_VERSION}: the evaluation subset", "",
             f"Seed {seed}; no desktop above {max_share:.0%} of a task; views in "
             "proportion to the export, strata equal, desktops round-robin. "
             "`units` is what was drawn (V6: a pair = open + multiple choice); "
             "`records` is what the scorer sees.", "",
             "| task | target | units | records | pool (records) | by view | desktops | "
             "max desktop share |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        views = ", ".join(f"{v} {n}" for v, n in r["by_view"].items())
        lines.append(f"| {r['task']} | {r['target']} | {r['units']} | {r['records']} | "
                     f"{r['pool_records']} | {views} | {r['desktops']} | "
                     f"{r['max_desktop_share']:.1%} |")
    lines += ["", "## Strata (units)", ""]
    for r in rows:
        if not r["by_stratum"]:
            continue
        lines.append(f"**{r['task']}**: " + "; ".join(
            f"{k} {n}" for k, n in r["by_stratum"].items()))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def _view_of(export_path) -> str:
    """``vlm_scan.jsonl`` -> ``scan``; a file named otherwise keeps its stem."""
    stem = Path(export_path).stem
    return stem[4:] if stem.startswith("vlm_") else stem


def _write_sidecars(export_path, out_jsonl: Path, records: list[dict]) -> None:
    """The subset's manifest and oracle (cut down) and the summary (whole)."""
    used = {img for r in records for img in r["prompt"].get("images") or []}
    desks = {IO.desktop_of(r) for r in records}
    src_manifest = IO.sidecar(export_path, "images_manifest.jsonl")
    if src_manifest.exists():
        rows = []
        for row in IO.read_jsonl(src_manifest):
            if row.get("type") == "header" or row.get("image") in used:
                rows.append(row)
        IO.write_jsonl(IO.sidecar(out_jsonl, "images_manifest.jsonl"), rows)
    src_oracle = IO.sidecar(export_path, "oracle.jsonl")
    if src_oracle.exists():
        def keep():
            for row in IO.read_jsonl(src_oracle):
                if row.get("type") == "desktop" and int(row["desktop"]) in desks:
                    yield row
                elif row.get("type") == "frame" and row.get("image") in used:
                    yield row
        IO.write_jsonl(IO.sidecar(out_jsonl, "oracle.jsonl"), keep())
    src_summary = IO.sidecar(export_path, "summary.json")
    if src_summary.exists():
        shutil.copyfile(src_summary, IO.sidecar(out_jsonl, "summary.json"))


def build(exports: Sequence, out_dir, *, seed: int = SEED, target_binary: int = TARGET_BINARY,
          target_other: int = TARGET_OTHER, max_share: float = MAX_SHARE,
          audit: int = 0, audit_per_task: int = 25) -> dict:
    """Draw the subset from ``exports`` and write it under ``out_dir``.

    Returns ``{"version", "seed", "targets", "records", "ids", "rows", "by_view",
    "audit"}``; ``rows`` is the n table.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    units: list[Unit] = []
    per_view: dict[str, tuple] = {}
    for path in exports:
        view = _view_of(path)
        if view in per_view:
            raise ValueError(f"two exports of view {view!r}: {per_view[view][0]} and {path}")
        mine = units_of(IO.read_jsonl(path), view)
        per_view[view] = (str(path), mine)
        units.extend(mine)
    chosen = draw(units, seed=seed, target_binary=target_binary, target_other=target_other,
                  max_share=max_share)
    pool: dict[str, list[Unit]] = defaultdict(list)
    for u in units:
        pool[u.task].append(u)
    targets = {t: (target_binary if t in BINARY_TASKS else target_other) for t in pool}
    ids = [r["id"] for task in sorted(chosen, key=_task_order)
           for u in chosen[task] for r in u.records]
    keep = set(ids)
    if len(keep) != len(ids):
        raise RuntimeError("a record was drawn twice")  # pragma: no cover - by construction
    (out / "ids.txt").write_text("".join(i + "\n" for i in ids), encoding="utf-8",
                                 newline="\n")
    by_view: dict[str, int] = {}
    all_records: list[dict] = []
    for view in sorted(per_view, key=lambda v: VIEWS.index(v) if v in VIEWS else 99):
        path, _ = per_view[view]
        records = [r for r in IO.read_jsonl(path) if r["id"] in keep]
        target = out / f"{view}.jsonl"
        IO.write_jsonl(target, records)
        _write_sidecars(path, target, records)
        by_view[view] = len(records)
        all_records.extend(records)
    rows = table_rows(chosen, pool, targets)
    (out / "table.md").write_text(table_markdown(rows, seed=seed, max_share=max_share),
                                  encoding="utf-8", newline="\n")
    result = {"version": EVAL_VERSION, "seed": seed,
              "targets": {"binary": target_binary, "other": target_other,
                          "binary_tasks": sorted(BINARY_TASKS, key=_task_order)},
              "max_desktop_share": max_share, "records": len(ids),
              "desktops": len({IO.desktop_of(r) for r in all_records}),
              "by_view": by_view, "rows": rows}
    if audit:
        result["audit"] = _audit(all_records, per_view, out / "audit", seed, audit,
                                 audit_per_task, max_share)
    (out / "table.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    return result


def _audit(records: list[dict], per_view: dict, out: Path, seed: int, size: int,
           per_task: int, max_share: float) -> dict:
    """The human-audit sample, drawn from inside the subset (``vlm_sample``)."""
    import vlm_sample

    desks = {IO.desktop_of(r) for r in records}
    rows = vlm_sample.draw(records, size=size, per_task=per_task, seed=seed, purpose="audit",
                           test_desktops=desks, max_share=max_share)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.ids.txt").write_text("".join(r["id"] + "\n" for r in rows),
                                       encoding="utf-8", newline="\n")
    IO.write_jsonl(out / "audit.jsonl", rows)
    manifest: dict[str, dict] = {}
    for path, _ in per_view.values():
        manifest.update(IO.load_manifest(path))
    (out / "audit.sheet.md").write_text(vlm_sample.sheet(rows, manifest), encoding="utf-8",
                                        newline="\n")
    by_task: dict[str, int] = defaultdict(int)
    for r in rows:
        by_task[r["prompt"]["task"]] += 1
    return {"size": len(rows), "desktops": len({IO.desktop_of(r) for r in rows}),
            "by_task": dict(sorted(by_task.items(), key=lambda kv: _task_order(kv[0])))}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--export", nargs="+", required=True,
                    help="the export-vlm JSONL files, one per unit (vlm_<view>.jsonl, vlm_multi.jsonl)")
    ap.add_argument("--out", required=True, help=f"the subset's directory ({EVAL_VERSION})")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--target-binary", type=int, default=TARGET_BINARY)
    ap.add_argument("--target-other", type=int, default=TARGET_OTHER)
    ap.add_argument("--max-share", type=float, default=MAX_SHARE)
    ap.add_argument("--audit", type=int, default=0,
                    help="also draw the human-audit sample of this many records")
    args = ap.parse_args(argv)
    refuse_old_eval(args.out, *args.export)
    result = build(args.export, args.out, seed=args.seed, target_binary=args.target_binary,
                   target_other=args.target_other, max_share=args.max_share,
                   audit=args.audit)
    print(f"[vlm-eval-set] {result['records']} records from {result['desktops']} desktops "
          f"-> {args.out}: " + " ".join(f"{r['task']}={r['records']}" for r in result["rows"]))
    if result.get("audit"):
        print(f"[vlm-eval-set] audit sample: {result['audit']['size']} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
