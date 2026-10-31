"""``pixel-edges``: the user's Review rulings on who blocks whom, frozen as ``covered_by`` edges (task W3a).

::

    python -m tda.cli pixel-edges --db <copy> --desktops 1-66
        [--whitelist configs/covers_whitelist.yaml [--allow-unruled]]
        [--graph-dir <export-graph out>] [--view scan] [--out <evidence.json>]
        [--apply | --undo pixels|review|all]

The logic is :mod:`tda.core.pixel_edges`: a pair the user ruled ``blocks`` in
Review (table ``cover_ruling``) that the confirmed frames show covering becomes
``covered_by(below <- above)``, ``source = review:covers``, ``status =
accepted``. The whitelist is the controller's suggestion layer: its unruled
pairs are listed, and written (``source = pixels:<version>``) only with
``--allow-unruled``. Without either flag the run reads the rulings alone.

**Dry run by default**: it prints the rows it would write and the evidence
file, and writes nothing to the database. ``--apply`` writes; ``--undo pixels``
/ ``review`` / ``all`` deletes the ``pixels:*`` / ``review:*`` / both kinds of
row of the desktops given (a bare ``--undo`` is refused; deleting the review
rows is reversible only because the rulings stay in ``cover_ruling``). Both refuse while
the annotator's app is running, take the single-user lock, probe for another
writer, back the database up first (``tda_pre_pixel_edges_<stamp>.sqlite`` in
``backup_dir``), and log one ``op_log`` row per desktop (kind ``pixel_edges`` /
``pixel_edges_undo``, the inverse in it).

The evidence JSON (``--out``, default ``<graph-dir or cwd>/pixel_edges.json``)
ships with the dataset: whitelist version and thresholds, every edge with its
pixels and frames, the suggestions, the pairs the user rejected, and every pair
that was skipped and why.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from tda.cli_common import EXIT_ERROR, EXIT_OK, desktops as _desktops, session as _session
from tda.cli_guards import app_running, named_backup, writer_busy
from tda.core import pixel_edges as PE
from tda.core.export.graph_covers import RULING_TABLE
from tda.core.graph import edges_from_db, find_deadlocks, constraint_edges
from tda.core.graph_derive import settled_instances
from tda.core.graph_rules import active_edges
from tda.core.model import VIEWS
from tda.core.states import initial_overrides
from tda.core.taxonomy import load_taxonomy

__all__ = ["EVIDENCE_NAME", "cmd_pixel_edges"]

EVIDENCE_NAME = "pixel_edges.json"
TAG = "[pixel-edges]"


def _has_rulings(db) -> bool:
    return db.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                           (RULING_TABLE,)).fetchone() is not None


def _evidence(db, tax, desktop: int, args) -> tuple[list[dict], int]:
    if args.graph_dir:
        found = PE.evidence_from_graph_dir(args.graph_dir, desktop)
        return found if found is not None else ([], 0)
    return PE.evidence_from_db(db, tax, desktop, args.view)


def _row_line(item: dict) -> str:
    frac = f"{item['frac'] * 100:.1f}%" if item.get("frac") is not None else "amodal ?"
    return (f"covered_by({item['below']}, {item['above']})  {item['px']} px, {frac}, "
            f"{item['n_frames']} frames from step {item['first']}  [{item['source']}]")


def _loops(db, tax, desktop: int, items: list[dict]) -> list[str]:
    """Deadlocks the whole set would close together (each was checked alone already)."""
    if not items:
        return []
    settled = settled_instances(db.instances(desktop))
    active = active_edges(constraint_edges(edges_from_db(db, desktop)))
    initial = initial_overrides(e for e in db.events(desktop) if not e.auto)
    new = [PE.edge_of(i) for i in items]
    if find_deadlocks(active, settled, tax, initial=initial):
        return []      # already looped before: reported by `constraints`, not ours
    return [d.label() for d in find_deadlocks([*active, *new], settled, tax, initial=initial)]


def cmd_pixel_edges(args: argparse.Namespace) -> int:
    """Rehearse (default), write or undo the ``covered_by`` edges of Review's rulings."""
    tax = load_taxonomy()
    if args.undo is not None and args.undo not in PE.UNDO_SELECTORS:
        print(f"{TAG} --undo 要说删哪一种 / --undo needs one of: pixels (pixels:* rows), "
              f"review (review:* rows), all; nothing deleted")
        return EXIT_ERROR
    if args.apply and args.undo:
        print(f"{TAG} --apply 和 --undo 只能选一个 / pass one of --apply and --undo")
        return EXIT_ERROR
    if args.allow_unruled and not args.whitelist:
        print(f"{TAG} --allow-unruled 需要 --whitelist / --allow-unruled needs --whitelist")
        return EXIT_ERROR
    whitelist = None
    if args.whitelist:
        try:
            whitelist = PE.load_whitelist(args.whitelist, tax)
        except PE.WhitelistError as exc:
            print(f"{TAG} 白名单被拒 / whitelist refused: {exc}")
            return EXIT_ERROR
    writing = bool(args.apply or args.undo)
    if writing and app_running():
        print(f"{TAG} 标注程序正在运行，先关掉它；什么也没写 / the annotator app is running "
              f"(or psutil is missing, so it cannot be told): close it first; nothing written")
        return EXIT_ERROR
    with _session(args, lock=True) as (paths, db):
        wanted = _desktops(args)
        known = db.desktop_ids()
        chosen = [d for d in known if wanted is None or d in wanted]
        for d in sorted((wanted or set()) - set(known)):
            print(f"{TAG} D{d:02d} 不在数据库里，跳过 / is not in the database; skipped")
        if writing:
            busy = writer_busy(db)
            if busy:
                print(f"{TAG} 数据库正被别人写 / the database is busy ({busy}); nothing written")
                return EXIT_ERROR
        if args.undo:
            return _undo(paths, db, chosen, args.undo)
        return _run(paths, db, tax, chosen, whitelist, args)


def _undo(paths, db, chosen: list[int], which: str) -> int:
    prefixes = PE.UNDO_SELECTORS[which]
    doomed = {d: [e for e in edges_from_db(db, d) if PE.owned(e, prefixes)] for d in chosen}
    doomed = {d: rows for d, rows in doomed.items() if rows}
    if not doomed:
        print(f"{TAG} 没有 {'/'.join(p + '*' for p in prefixes)} 的边，什么也没删 / no such rows; "
              f"nothing deleted")
        return EXIT_OK
    if named_backup(paths, db, "pixel-edges", "pixel_edges_undo") is None:
        return EXIT_ERROR
    for d in sorted(doomed):
        with db.transaction():
            out = PE.undo_edges(db, d, which)
        print(f"{TAG} D{d:02d}: 删了 {out['deleted']} 条 / deleted {out['deleted']} rows; "
              f"graph_version {out['graph_version'] or '-'}")
    return EXIT_OK


def _run(paths, db, tax, chosen: list[int], whitelist, args) -> int:
    has_table = _has_rulings(db)
    if not has_table:
        print(f"{TAG} 数据库里还没有 {RULING_TABLE} 表（复查裁定）/ no {RULING_TABLE} table yet: "
              f"no ruling to freeze")
    plans: list[PE.DesktopPlan] = []
    for d in chosen:
        covers, confirmed = _evidence(db, tax, d, args)
        rulings = PE.rulings_of(db, d) if has_table else {}
        plan = PE.plan_desktop(d, covers, confirmed, db.instances(d), edges_from_db(db, d),
                               db.actions(d), db.events(d), tax, rulings, whitelist, args.view)
        PE.split_suggestions(plan, args.allow_unruled)
        plans.append(plan)
    refused: dict[int, list[str]] = {}
    for plan in plans:
        loops = _loops(db, tax, plan.desktop, plan.to_write(args.allow_unruled))
        if loops:
            refused[plan.desktop] = loops
    _print(plans, args, refused)

    source = f"graph-dir {args.graph_dir}" if args.graph_dir else "database"
    out = Path(args.out) if args.out else Path(args.graph_dir or os.getcwd()) / EVIDENCE_NAME
    doc = PE.evidence_json(plans, whitelist, args.allow_unruled, source, args.view,
                           has_table, bool(args.apply))
    doc["refused_desktops"] = {f"D{d:02d}": loops for d, loops in sorted(refused.items())}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print(f"{TAG} 证据文件 / evidence -> {out}")

    todo = {p.desktop: p.to_write(args.allow_unruled) for p in plans
            if p.to_write(args.allow_unruled) and p.desktop not in refused}
    if not args.apply:
        print(f"{TAG} 演练，什么也没写 / dry run: nothing written "
              f"({sum(len(v) for v in todo.values())} rows would be; pass --apply)")
        return EXIT_ERROR if refused else EXIT_OK
    if not todo:
        print(f"{TAG} 没有要写的边 / nothing to write")
        return EXIT_ERROR if refused else EXIT_OK
    if named_backup(paths, db, "pixel-edges", "pixel_edges") is None:
        return EXIT_ERROR
    for d, items in sorted(todo.items()):
        with db.transaction():
            done = PE.write_edges(db, d, items)
        print(f"{TAG} D{d:02d}: 写了 {done['written']} 条 / wrote {done['written']} rows; "
              f"graph_version {done['graph_version']}")
    return EXIT_ERROR if refused else EXIT_OK


def _print(plans: list[PE.DesktopPlan], args, refused: dict[int, list[str]]) -> None:
    pairs: Counter = Counter()
    why: Counter = Counter()
    for plan in plans:
        rows = plan.to_write(args.allow_unruled)
        if rows or plan.suggested or plan.rejected_by_user or plan.skipped:
            print(f"{TAG} D{plan.desktop:02d}: {len(plan.review)} 条复查裁定 / ruled, "
                  f"{len(plan.whitelist)} whitelist, {len(plan.suggested)} 建议 / suggested, "
                  f"{len(plan.rejected_by_user)} 用户判不挡 / rejected, "
                  f"{len(plan.skipped)} skipped; {plan.confirmed_frames} confirmed frames")
        for item in rows:
            print(f"{TAG}   {'WRITE' if args.apply else 'would write'} {_row_line(item)}")
        for item in plan.skipped:
            if item["why"] in ("would_violate", "reverse_edge", "ruled_no_pixel_evidence",
                               "deadlock"):
                print(f"{TAG}   skipped ({item['why']}) {item['above']} over {item['below']}"
                      + (f": {item['lines'][0]}" if item.get("lines") else ""))
            why[item["why"]] += 1
        for loop in refused.get(plan.desktop, []):
            print(f"{TAG}   REFUSED: together they close a loop {loop}")
        for name in ("review", "whitelist", "suggested"):
            for item in getattr(plan, name):
                pairs[(item.get("whitelist_pair") or "(ruled)", name)] += 1
    for (pair, name), n in sorted(pairs.items()):
        print(f"{TAG} pair {pair}: {n} {name}")
    print(f"{TAG} skipped: " + (", ".join(f"{k} {v}" for k, v in sorted(why.items())) or "none"))


def _add_pixel_edges(sub) -> None:
    p = sub.add_parser(
        "pixel-edges",
        help="freeze Review's 'blocks' rulings (and, if allowed, whitelist pairs) as "
             "covered_by edges from confirmed-frame pixels (task W3a)",
    )
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-66 (default: all)")
    p.add_argument("--whitelist", default=None,
                   help=f"class-pair suggestions, e.g. {PE.DEFAULT_WHITELIST}; listed, written "
                        f"only with --allow-unruled")
    p.add_argument("--allow-unruled", action="store_true",
                   help="also write unruled pairs that match the whitelist (source pixels:<version>)")
    p.add_argument("--graph-dir", default=None,
                   help="read the covers from an export-graph output instead of measuring them")
    p.add_argument("--view", default="scan", choices=list(VIEWS),
                   help="the view whose confirmed frames give the pixels (default scan)")
    p.add_argument("--out", default=None,
                   help=f"evidence JSON (default <graph-dir or cwd>/{EVIDENCE_NAME})")
    p.add_argument("--apply", action="store_true",
                   help="write the rows (default: a dry run that writes nothing to the database)")
    p.add_argument("--undo", nargs="?", const="", default=None, metavar="pixels|review|all",
                   help="delete the desktops' rows of one kind: pixels (source pixels:*, the "
                        "whitelist's), review (review:*, Review's rulings) or all; a bare "
                        "--undo is refused. Deleting the review rows is reversible only "
                        "because the rulings stay in the cover_ruling table: a later "
                        "--apply writes them again")
    p.set_defaults(func=cmd_pixel_edges)
