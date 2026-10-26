"""``refreeze``: 「重签」 -- re-sign confirmed rows that trail today's compile (task U32).

::

    python -m tda.cli refreeze --desktop N [--view scan] [--steps 17-37] [--dry-run]
    python -m tda.cli refreeze --desktop N [--view scan] --undo last|RUN

A confirmed frame's rows are frozen; a later change to the inputs that moves
one **within** the re-tracing tolerance opens no conflict and, until now, left
the row as it was for ever -- confirmed masks of neighbouring parts came to
overlap, and pixels a layer-order fix handed to a big part belonged to nobody.
This writes today's compile into exactly those rows, through
:meth:`tda.core.truth.TruthService.refreeze`: a row beyond the tolerance is a
conflict and stays one, a part the frame gained is reported and not signed,
the frame stays confirmed.  Per frame it prints the rows re-signed, the pixels
that changed and the pixels two confirmed masks shared before and after.

Safety as for the other destructive commands: the single-user lock (exit 3
while somebody else holds it), and a backup into ``backup_dir`` before the
first write.  ``--dry-run`` takes the lock too but writes nothing, not even the
backup.  Every frame is one ``refreeze_frame`` op whose inverse is the old rows
byte for byte; ``--undo last`` (or the run tag a run printed) puts a run back.
"""
from __future__ import annotations

import argparse
from typing import Optional

from tda import pipeline as P
from tda.cli_common import EXIT_ERROR, EXIT_OK, safety_backup, session
from tda.core.taxonomy import load_taxonomy

__all__ = ["DEFAULT_ANNOTATOR", "_add_refreeze", "cmd_refreeze", "frame_line"]

#: Who signs the rows when ``--annotator`` is not given: the lock holder's name.
DEFAULT_ANNOTATOR = "cli:refreeze"
#: How many instances a frame line names before it says "..."
NAMED = 6


def _n(value: Optional[int]) -> str:
    return "-" if value is None else f"{int(value):,}".replace(",", " ")


def frame_line(plan) -> str:
    """One frame of a run: what was (or would be) re-signed and what was left alone."""
    parts = []
    if plan.changes:
        named = ", ".join(f"{c.instance} {_n(c.px)}" for c in plan.changes[:NAMED])
        more = f", ... +{len(plan.changes) - NAMED}" if len(plan.changes) > NAMED else ""
        parts.append(f"{len(plan.changes)} rows re-signed ({named}{more}), {_n(plan.px)} px changed")
    if plan.overlap_before is not None and (plan.overlap_before or plan.overlap_after):
        removed = plan.overlap_before - (plan.overlap_after or 0)
        parts.append(f"overlap {_n(plan.overlap_before)} -> {_n(plan.overlap_after)} px"
                     f" ({_n(removed)} removed)")
    left = []
    if plan.disputed:
        left.append(f"{len(plan.disputed)} beyond the tolerance ({', '.join(plan.disputed[:NAMED])})")
    if plan.open_conflicts:
        left.append(f"{len(plan.open_conflicts)} with an open conflict")
    if plan.deferred:
        left.append(f"{len(plan.deferred)} waiting for those conflicts "
                    f"({', '.join(plan.deferred[:NAMED])})")
    if plan.missing or plan.unsigned:
        gained = plan.missing + plan.unsigned
        left.append(f"{len(gained)} not confirmed on this frame ({', '.join(gained[:NAMED])}): "
                    f"not signed")
    if plan.moved:
        left.append("the inputs moved while re-signing: skipped, run again")
    if left:
        parts.append("left alone: " + "; ".join(left))
    return f"  step {plan.step}: " + "; ".join(parts)


def _report(desktop: int, view: str, result: dict) -> None:
    for plan in result["frames"]:
        if plan.changes or plan.disputed or plan.open_conflicts or plan.missing \
                or plan.unsigned or plan.deferred or plan.moved:
            print(frame_line(plan))
    verb = "would re-sign" if result["dry_run"] else "re-signed"
    print(f"[refreeze] D{desktop:02d} {view}: {result['considered']} confirmed frames, "
          f"{verb} {result['rewritten']} rows on {result['frames_rewritten']} frames, "
          f"{_n(result['px'])} px changed; overlap {_n(result['overlap_before'])} -> "
          f"{_n(result['overlap_after'])} px")
    stood = result["disputed"] + result["open_conflicts"]
    if stood:
        print(f"[refreeze] {stood} confirmed row(s) differ beyond the tolerance or have an "
              f"open conflict: left alone (Review > Conflicts; cli check queues them)")
    if result.get("deferred"):
        print(f"[refreeze] {result['deferred']} row(s) within the tolerance touch those rows: "
              f"they wait for the conflicts (settle them, then run refreeze again)")
    gained = result["missing"] + result["unsigned"]
    if gained:
        print(f"[refreeze] {gained} part(s) in today's compile were never confirmed on their "
              f"frame: not signed (the frame goes back to review on the next refresh)")
    if result["moved"]:
        print(f"[refreeze] {result['moved']} frame(s) changed while being re-signed: skipped; "
              f"run again")
    if result["dry_run"]:
        print("[refreeze] dry run: nothing was written")
    elif result["ops"]:
        print(f"[refreeze] run {result['run']}: {len(result['ops'])} refreeze_frame op(s); "
              f"undo with --undo last")


def _undo(db, desktop: int, view: str, run: str, annotator: str) -> int:
    from tda.core.truth_refreeze import refreeze_ops, undo_refreeze

    ops = refreeze_ops(db, desktop, view, run)
    if not ops:
        print(f"[refreeze] D{desktop:02d} {view}: no refreeze run {run!r} to undo")
        return EXIT_ERROR
    tag = (ops[0].get("payload") or {}).get("run")
    undone = already = 0
    refused = []
    for op in reversed(ops):
        try:
            made = undo_refreeze(db, int(op["id"]), annotator)
        except ValueError as exc:
            refused.append(str(exc))
            continue
        if made is None:
            already += 1
        else:
            undone += 1
    for text in refused:
        print(f"  refused: {text}")
    print(f"[refreeze] D{desktop:02d} {view}: run {tag}: {undone} frame(s) put back, "
          f"{already} already undone, {len(refused)} refused")
    return EXIT_ERROR if refused else EXIT_OK


def cmd_refreeze(args: argparse.Namespace) -> int:
    """Re-sign (or with ``--dry-run`` plan) one desktop's confirmed frames, view by view."""
    from tda.cli_app import _view_list
    from tda.core.truth import TruthService

    views = _view_list(args.views)
    steps = P.parse_desktops(args.steps) if args.steps else None   # same "1-3,7" syntax
    with session(args, lock=True) as (paths, db):
        desktop = int(args.desktop)
        annotator = str(args.annotator or DEFAULT_ANNOTATOR)
        if not args.dry_run:
            why = ("--undo puts re-signed rows back" if args.undo
                   else "it rewrites confirmed rows")
            if not safety_backup(paths, db, "refreeze", why):
                return EXIT_ERROR
        if args.undo:
            return max(_undo(db, desktop, view, str(args.undo), annotator) for view in views)
        truth = TruthService(db, load_taxonomy())
        for view in views:
            result = truth.refreeze(desktop, view, steps=steps, annotator=annotator,
                                    dry_run=bool(args.dry_run), measure=True)
            _report(desktop, view, result)
        return EXIT_OK


def _add_refreeze(sub) -> None:
    from tda.cli_app import _add_view_flags

    p = sub.add_parser("refreeze",
                       help="「重签」: write today's compile into confirmed rows that differ "
                            "from it only within the tolerance")
    p.add_argument("--desktop", type=int, required=True)
    _add_view_flags(p)
    p.add_argument("--steps", default=None,
                   help="only these confirmed frames, e.g. 17-37 or 3,7 (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="plan and print, write nothing (no backup either)")
    p.add_argument("--annotator", default=None,
                   help=f"who signs the re-signed rows (default {DEFAULT_ANNOTATOR})")
    p.add_argument("--undo", default=None, metavar="last|RUN",
                   help="put back the rows of the newest run ('last') or of the run tag a "
                        "run printed, byte for byte")
    p.set_defaults(func=cmd_refreeze)
