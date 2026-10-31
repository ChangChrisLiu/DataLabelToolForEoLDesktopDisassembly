"""``set-split``: write the platform-disjoint train / test split into ``desktop.split`` (task W3a).

::

    python -m tda.cli set-split --db <copy> --test 4,10,15,20,65,24,32,30,31,48,49,46
        [--train rest | --train 1-3,5] [--apply]

``test`` for every desktop of ``--test``; ``train`` for every other desktop
**with steps** (``--train rest``, the default) or for the ``--train`` list;
every other desktop is left untouched. Spec 8.1: the exporters take the split
from ``Desktop.split``. Dry run by default; ``--apply`` refuses while the
annotator's app runs, takes the lock, probes for a writer, backs the database
up (``tda_pre_set_split_<stamp>.sqlite``) and logs one ``op_log`` row per desktop
whose split changes (kind ``set_split``, the old value as its inverse).
"""
from __future__ import annotations

import argparse
from typing import Optional

from tda import pipeline as P
from tda.cli_common import EXIT_ERROR, EXIT_OK, session as _session
from tda.cli_guards import app_running, named_backup, open_read_only, writer_busy

__all__ = ["OP_KIND", "split_plan", "cmd_set_split"]

OP_KIND = "set_split"
OP_VIEW = "-"
ANNOTATOR = "cli:set-split"
TEST, TRAIN = "test", "train"
REST = "rest"
TAG = "[set-split]"


def split_plan(db, test: set[int], train: Optional[set[int]]) -> dict[int, tuple[Optional[str], str]]:
    """``desktop -> (old split, new split)`` for every desktop whose split is set by this run.

    ``train=None`` means "every other desktop with steps". Raises ``ValueError``
    for a desktop the database lacks or one named on both sides.
    """
    known = set(db.desktop_ids())
    missing = sorted((test | (train or set())) - known)
    if missing:
        raise ValueError("不在数据库里 / not in the database: "
                         + ", ".join(f"D{d:02d}" for d in missing))
    both = sorted(test & (train or set()))
    if both:
        raise ValueError("同时在 test 和 train 里 / on both sides: "
                         + ", ".join(f"D{d:02d}" for d in both))
    if train is None:
        train = {d for d in known - test if db.steps(d)}
    out: dict[int, tuple[Optional[str], str]] = {}
    for d in sorted(test | train):
        old = (db.get_desktop(d) or {}).get("split")
        out[d] = (old, TEST if d in test else TRAIN)
    return out


def cmd_set_split(args: argparse.Namespace) -> int:
    """Rehearse (default) or write ``desktop.split``."""
    test = P.parse_desktops(args.test) or set()
    if not test:
        print(f"{TAG} --test 不能为空 / --test names no desktop")
        return EXIT_ERROR
    train = None if str(args.train).strip().lower() == REST else (
        P.parse_desktops(args.train) or set())
    if args.apply and app_running():
        print(f"{TAG} 标注程序正在运行，先关掉它；什么也没写 / the annotator app is running "
              f"(or psutil is missing, so it cannot be told): close it first; nothing written")
        return EXIT_ERROR
    if not args.apply:
        # through `mode=ro` (review round 1, M1): a dry run must not migrate an older copy
        db = open_read_only(args, TAG)
        if db is None:
            return EXIT_ERROR
        try:
            return _set_split(None, db, set(test), train, apply=False)
        finally:
            db.close()
    with _session(args, lock=True) as (paths, db):
        return _set_split(paths, db, set(test), train, apply=True)


def _set_split(paths, db, test: set[int], train: Optional[set[int]], apply: bool) -> int:
    try:
        plan = split_plan(db, set(test), train)
    except ValueError as exc:
        print(f"{TAG} {exc}; nothing written")
        return EXIT_ERROR
    changes = {d: v for d, v in plan.items() if v[0] != v[1]}
    counts = {s: sum(1 for _, new in plan.values() if new == s) for s in (TEST, TRAIN)}
    untouched = sorted(set(db.desktop_ids()) - set(plan))
    print(f"{TAG} test {counts[TEST]}, train {counts[TRAIN]}, untouched "
          f"{len(untouched)}" + (f" ({', '.join(f'D{d:02d}' for d in untouched)})"
                                 if untouched else "")
          + f"; {len(changes)} change")
    for d, (old, new) in sorted(changes.items()):
        print(f"{TAG}   D{d:02d}: {old or '-'} -> {new}")
    if not apply:
        print(f"{TAG} 演练，什么也没写 / dry run: nothing written (pass --apply)")
        return EXIT_OK
    if not changes:
        print(f"{TAG} 都已是这样 / already so; nothing written")
        return EXIT_OK
    busy = writer_busy(db)
    if busy:
        print(f"{TAG} 数据库正被别人写 / the database is busy ({busy}); nothing written")
        return EXIT_ERROR
    if named_backup(paths, db, "set-split", "set_split") is None:
        return EXIT_ERROR
    with db.transaction():
        for d, (old, new) in sorted(changes.items()):
            P.merge_desktop_meta(db, d, {"split": new})
            db.log_op(d, OP_VIEW, OP_KIND, {"split": new}, {"split": old}, ANNOTATOR)
    print(f"{TAG} 写了 {len(changes)} 台 / wrote {len(changes)} desktops")
    return EXIT_OK


def _add_set_split(sub) -> None:
    p = sub.add_parser("set-split",
                       help="write desktop.split: test for a list, train for the rest (W3a)")
    p.add_argument("--test", required=True, help="the test desktops, e.g. 4,10,15")
    p.add_argument("--train", default=REST,
                   help="'rest' (default: every other desktop with steps) or a list")
    p.add_argument("--apply", action="store_true",
                   help="write (default: a dry run that writes nothing)")
    p.set_defaults(func=cmd_set_split)
