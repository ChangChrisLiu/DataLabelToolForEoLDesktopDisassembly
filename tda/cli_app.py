"""The commands that come after the data pipeline: the app, checks and exports.

:mod:`tda.cli` holds the four import commands that seed the database; this
module holds the five that use it -- open the annotator, recompile and count the
problems of one view, build the local image cache, and write the two rehearsal
exports.  They are registered by appending :data:`SUBCOMMANDS` to the parser
there, so ``python -m tda.cli --help`` lists all of them together.

Everything that writes takes the single-user lock through
:func:`tda.cli_common.session` (exit code 3 while somebody else holds it),
including ``check``: recompiling a view writes the ``compiled_mask`` rows it
finds stale.  That import is at module level and points at
:mod:`tda.cli_common`, never at :mod:`tda.cli`: importing the latter from here
loads it a second time under ``python -m tda.cli`` and hands this module a
``Locked`` class that ``main()`` does not catch.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Sequence

from tda import pipeline as P
from tda.cli_common import EXIT_ERROR, EXIT_OK, load_paths, session
from tda.core.model import ALL_VIEWS, COMPONENT_VIEWS, is_component_view
from tda.core.taxonomy import load_taxonomy

__all__ = ["SUBCOMMANDS", "cmd_app", "cmd_build_cache", "cmd_check",
           "cmd_export_coco", "cmd_export_vlm"]


def _desktop_list(spec: str | None) -> list[int]:
    """``--desktops 1-3,13`` -> ``[1, 2, 3, 13]``; nothing given -> ``[]``."""
    wanted = P.parse_desktops(spec) if spec else None
    return sorted(wanted) if wanted else []


#: ``--view components``: the components gallery's three views (task U10).
COMPONENTS_ALIAS = "components"


def _view_list(spec: str) -> list[str]:
    """``"scan,oak1"`` -> ``["scan", "oak1"]``, in order and without repeats.

    ``ValueError`` for a view that does not exist, which :func:`tda.cli.main`
    turns into one line and exit 1. That replaces argparse's ``choices``, which
    cannot describe a comma-separated list and answered "invalid choice:
    'scan,oak1'" for a spelling this now accepts.

    The components gallery's views (task U10) are views like any other here;
    ``components`` names all three.
    """
    known = ", ".join(ALL_VIEWS) + f" ({COMPONENTS_ALIAS} = {', '.join(COMPONENT_VIEWS)})"
    wanted: list[str] = []
    for name in (v.strip() for v in str(spec).split(",") if v.strip()):
        wanted.extend(COMPONENT_VIEWS if name == COMPONENTS_ALIAS else [name])
    if not wanted:
        raise ValueError(f"no view given; the views are {known}")
    unknown = [v for v in wanted if v not in ALL_VIEWS]
    if unknown:
        raise ValueError(f"not a view: {', '.join(unknown)}; the views are {known}")
    out: list[str] = []
    for view in wanted:
        if view not in out:
            out.append(view)
    return out


def _add_view_flags(p, default: str = "scan", plural_first: bool = False) -> None:
    """``--view`` and ``--views``: one option, both spellings, one or many views.

    The exports spelled it ``--view`` and ``build-cache`` spelled it ``--views``,
    and each rejected the other -- a coin toss the annotator loses half the time
    in the middle of a release. They are the same option now, and it takes a
    comma-separated list wherever more than one view makes sense.
    """
    names = ("--views", "--view") if plural_first else ("--view", "--views")
    p.add_argument(*names, dest="views", default=default, metavar="VIEW[,VIEW...]",
                   help=f"one view or a comma-separated list ({', '.join(ALL_VIEWS)}; "
                        f"'{COMPONENTS_ALIAS}' = the components gallery's three); "
                        f"--view and --views are the same option")


def _count(stats: dict, key: str) -> int:
    """How many of something an exporter reported, whether it gave a list or a number."""
    value = stats.get(key, 0)
    if isinstance(value, int):
        return value
    try:
        return len(value)
    except TypeError:
        return 0


def _context_count(stats: dict) -> int:
    """How many of the exported annotations are 框外零件 (task U18), counted apart."""
    found = stats.get("annotations")
    if not isinstance(found, list):
        return 0
    return sum(1 for ann in found
               if isinstance(ann, dict) and (ann.get("attributes") or {}).get("context"))


def _runs(desktops: Sequence[int]) -> list[tuple[int, int]]:
    """Contiguous ``(first, last)`` runs of a desktop list.

    ``tda.core.cache.main`` only speaks first/last, so ``--desktops 1-3,13``
    used to collapse into ``--first 1 --last 13`` and cache ten machines nobody
    asked for -- hours of copying from ``F:``.  One call per run instead.
    """
    runs: list[tuple[int, int]] = []
    for desktop in sorted(set(int(d) for d in desktops)):
        if runs and desktop == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], desktop)
        else:
            runs.append((desktop, desktop))
    return runs


def _pending_rechecks(truth, desktop: int, view: str) -> list[int]:
    """Steps the truth service still owes a re-check, when it tracks them."""
    found = getattr(truth, "pending_rechecks", None)
    if not callable(found):
        return []
    try:
        return [int(s) for s in found(desktop, view)]
    except Exception:  # noqa: BLE001 - a missing table is "nothing pending"
        return []


def _open_conflicts(truth, db, desktops: Sequence[int], view: str) -> int:
    """How many conflicts of these ``(desktop, view)`` are still open.

    An open conflict is a frozen row the recompile disagrees with: a human has
    to choose between the two, and until they do the view's truth table is not
    one to ship (spec 3.4).  ``check`` reported only the conflicts *this run*
    raised, so a queue somebody left behind yesterday read as a clean desktop.

    Worker F1 owns ``TruthService.open_conflicts``; it is preferred as soon as
    it exists, and until then the same question is answered straight from the
    table it will read anyway.
    """
    found = getattr(truth, "open_conflicts", None)
    if not callable(found):
        def found(desktop: int, view: str) -> list:
            return db.conflicts(desktop, view)
    return sum(_how_many(found(int(desktop), str(view))) for desktop in desktops)


def _how_many(answer) -> int:
    """A count, whether the service answered with the rows or with their number.

    ``TruthService.open_conflicts`` returns a list; a future one may well return
    an ``int``, and this adapter is the wrong place to care which.
    """
    if isinstance(answer, int):
        return answer
    try:
        return len(answer)
    except TypeError:
        return 0


def _frame_counts(db, desktops: Sequence[int], view: str) -> tuple[int, int]:
    """``(verified, not verified)`` frames of one view over the selection.

    An export that filters on ``verified`` and finds none writes an empty
    artefact, so it has to say what it looked at: "0 verified, 214 not verified"
    is a diagnosis, "0 images" is a mystery.
    """
    verified = total = 0
    for desktop in desktops:
        frames = [row for row in db.frames_for(int(desktop), str(view))
                  if not row.get("missing")]
        total += len(frames)
        verified += sum(1 for v, _step in db.verified_frames(int(desktop)) if v == view)
    return verified, max(total - verified, 0)


def _nothing_exported(command: str, view: str, out: str, only_verified: bool,
                      also: Sequence[Path] = ()) -> int:
    """Warn, remove the empty artefact **and its siblings**, and fail.

    An export that found nothing used to write the file anyway and exit 0, so a
    day of annotation came back as a COCO with zero images or a zero-byte JSONL
    and nothing at all to say why. The file goes too: an empty artefact left on
    disk is the one that gets shipped, or diffed against, or trained on -- and
    so do the files written beside it, or the next run finds an image manifest
    and a summary describing a JSONL nobody wrote.
    """
    subject = ("0 verified frames" if only_verified else "no exportable frames")
    hint = (" (use --no-only-verified to include unverified frames)"
            if only_verified else "")
    print(f"[{command}] WARNING: {subject} for the requested desktops/{view} "
          f"- nothing exported{hint}")
    for path in [Path(out), *also]:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # a locked or read-only file: say so, do not crash
            print(f"[{command}] could not remove the empty {path}: {exc}")
    return EXIT_ERROR


def _out_paths(out: str | None, views: Sequence[str], paths: dict, pattern: str,
               command: str) -> Optional[dict[str, str]]:
    """Where each view's file goes, or ``None`` when ``--out`` cannot cover them."""
    if out and len(views) > 1:
        print(f"[{command}] refused: --out names one file but {len(views)} views "
              f"were asked for ({', '.join(views)}). Drop --out to write one file "
              f"per view into cache_dir, or export one view at a time.")
        return None
    if out:
        return {views[0]: str(out)}
    return {view: P.cache_file(paths, pattern.format(view=view)) for view in views}


def _call_export(export, *args, allow_conflicts: bool, **kw):
    """Call an exporter, passing ``allow_conflicts`` only when it takes it.

    The keyword is F1's; an exporter that predates it must still run rather
    than die of an unexpected argument.  A ``TypeError`` raised *inside* the
    exporter says nothing about ``allow_conflicts`` and is left alone -- it is
    a bug, and swallowing it would hide it behind a second, argument-less call.
    """
    try:
        return export(*args, allow_conflicts=allow_conflicts, **kw)
    except TypeError as exc:
        if "allow_conflicts" not in str(exc):
            raise
        return export(*args, **kw)


def _prepare_truth(db, tax, desktops: Sequence[int], view: str,
                   refresh: bool = True, ignore_digest: bool = False,
                   on_frame=None) -> dict:
    """Bring the compiled truth of ``desktops`` up to date before it is read.

    Three commands read the truth table and all three were reading it raw:
    ``check`` recompiled but ignored the re-check queue the session fills in the
    background, and the two exports read whatever happened to be stored.  A
    frame whose inputs changed after it was frozen therefore exported its *old*
    geometry.  This is the one place that fixes it:

    * the pending re-checks are drained first (a frozen frame the session
      queued but never got to);
    * unless ``refresh`` is off, every step of the view is recompiled, so the
      automatic rows are current too.  ``refresh`` is **not**
      ``only_verified``: that one is the exporter's filter over what to write,
      and reading them as the same flag made ``--only-verified`` silently skip
      the recompile.

    One :class:`~tda.core.truth.TruthService` is built and returned in
    ``truth``: the exports take it as a keyword argument and refresh lazily
    through the same object, so a frame is never compiled twice per run.

    ``ignore_digest`` compiles every frame whatever its stored digest says.
    ``check`` needs it: a frame whose digest is current is skipped without
    compiling, and a skipped frame reports no problems -- so the second
    ``check`` of an untouched view printed ``problems: 0`` and exited 0 over
    hundreds of missing shapes. The exports keep the digest shortcut.

    ``on_frame`` is handed on to ``refresh_range`` (``check``'s count of the
    rows 「重签」 would rewrite, U32).

    Returns ``{"truth", "desktops", "view", "steps", "updated", "conflicts",
    "closed", "overtaken", "problems", "empty_visible", "pending"}``;
    ``pending`` is what is *still* queued afterwards, which is what the exports
    refuse on, ``closed`` the conflicts closed because the inputs agree with
    the frozen rows again (U6), ``overtaken`` the ones a newer disagreement
    replaced (U7), and ``empty_visible`` the reason behind every
    ``empty_visible`` problem (U26, :mod:`tda.core.empty_visible`).
    """
    from tda.core.truth import TruthService

    truth = TruthService(db, tax)
    total = {"truth": truth, "desktops": list(desktops), "view": str(view),
             "steps": 0, "updated": 0, "conflicts": 0, "closed": 0, "overtaken": 0,
             "problems": [], "empty_visible": [], "pending": []}
    for desktop in desktops:
        runner = getattr(truth, "run_pending_rechecks", None)
        if callable(runner):
            drained = runner(int(desktop), str(view))
            # conflicts the inputs agree with again are closed by the drain
            # too (U6), and the operator is told how many
            if isinstance(drained, dict):
                total["closed"] += int(drained.get("closed", 0))
                total["overtaken"] += int(drained.get("overtaken", 0))
        if refresh:
            steps = [int(row["step"]) for row in db.frames_for(int(desktop), str(view))
                     if not row.get("missing")]
            stats = truth.refresh_range(int(desktop), str(view), steps,
                                        ignore_digest=ignore_digest,
                                        **({} if on_frame is None else {"on_frame": on_frame}))
            total["steps"] += len(steps)
            total["updated"] += int(stats.get("updated", 0))
            total["conflicts"] += int(stats.get("conflicts", 0))
            total["closed"] += int(stats.get("closed", 0))
            total["overtaken"] += int(stats.get("overtaken", 0))
            total["problems"].extend(stats.get("problems") or [])
            total["empty_visible"].extend(stats.get("empty_visible") or [])
        total["pending"].extend(_pending_rechecks(truth, int(desktop), str(view)))
    return total


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
def cmd_app(args: argparse.Namespace) -> int:
    """Open the annotator window; the exit code is the window's.

    The lock is taken by the window rather than by ``session`` here: it has to
    stay held for the whole session and be released on close, next to the exit
    backup, and a refusal has to reach the annotator as a message box.
    """
    from tda.ui import app as app_module

    return int(app_module.main(
        paths=args.paths,
        desktop=None if args.desktop is None else int(args.desktop),
        view=None if args.view is None else str(args.view),
        annotator=str(args.annotator),
        step=None if args.step is None else int(args.step),
        db=args.db,
    ))


def _add_app(sub) -> None:
    p = sub.add_parser("app", help="open the annotator (resumes the last frame)")
    p.add_argument("--desktop", type=int, default=None,
                   help="machine to open (default: where this annotator left off)")
    # one view, not a list: the window opens on a single view. The plural
    # spelling is accepted so that a hand that has just typed `--views` at an
    # export does not get "unrecognized arguments" here.
    p.add_argument("--view", "--views", dest="view", default=None, choices=list(ALL_VIEWS),
                   help="the ONE view to open (default: the last one, else scan; "
                        "cmp_oak1/cmp_oak2/cmp_rs open the components gallery); "
                        "unlike check and the exports this takes no list")
    p.add_argument("--annotator", required=True, help="who is annotating (the lock holder)")
    p.add_argument("--step", type=int, default=None, help="open this logical step")
    p.set_defaults(func=cmd_app)


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def cmd_check(args: argparse.Namespace) -> int:
    """Recompile one view and print what the compiler is unhappy about.

    Exit code 1 when anything is reported, so a shell loop over the desktops
    can stop on the first machine that needs attention. "Anything" is three
    things, not one: the compiler's problems, the conflicts still standing
    open, and the re-checks still queued. A desktop with none of the first and
    six of the second is not a desktop to export, and answering 0 there was how
    an open conflict survived all the way into a release artefact.

    A part covered completely by labelled parts (or a hand, a tool, a label)
    is not one of the problems (task U26): its empty visible mask is what the
    layers should give, and it is one summary line, 「n 个零件完全被挡住（正常）」.
    An empty visible mask with no such reason -- nothing drawn -- still is
    (:mod:`tda.core.empty_visible`).

    The last line, ``stale_within_tolerance: n (deferred m)`` (task U32),
    counts the confirmed rows 「重签」 would rewrite now (``n``) and those as
    far behind that wait for a neighbour's open conflict (``m``): they differ
    from today's compile only within the tolerance, so they open no conflict
    and keep their old pixels.  A counter for the audit, not a problem: the
    exit code ignores it.
    """
    with session(args, lock=True) as (_paths, db):
        desktop = int(args.desktop)
        tax = load_taxonomy()
        worst = EXIT_OK
        for view in _view_list(args.views):
            worst = max(worst, _check_one(db, tax, args, desktop, view))
        return worst


def _check_one(db, tax, args: argparse.Namespace, desktop: int, view: str) -> int:
    """Recompile one ``(desktop, view)`` and report it; the exit code is its own."""
    # 「重签」 (U32): confirmed rows that differ from today's compile only within
    # the tolerance -- no conflict, and no pass ever rewrites them -- counted off
    # the compilation each frame's refresh just made
    from tda.core.truth_refreeze import plan_frame

    resign: list[tuple[int, str]] = []
    waiting: list[tuple[int, str]] = []

    def count_resign(key, one: dict) -> None:
        compiled = one.get("compiled")
        if compiled is None or (db.get_frame(key) or {}).get("review_status") != "verified":
            return
        plan = plan_frame(db, key, compiled)
        resign.extend((key.step, c.instance) for c in plan.changes)
        # behind the compile as well, but waiting for a neighbour's conflict (U32 N1).
        # A row drifted from what a person signed needs no line of its own: the
        # refresh that just ran measured it against that row and queued its
        # conflict, so it is among the open conflicts below
        waiting.extend((key.step, instance) for instance in plan.deferred)

    # every frame compiled, as on a first run: a digest-skipped frame reports
    # no problems, and a check has to see them all every time
    stats = _prepare_truth(db, tax, [desktop], view, refresh=True, ignore_digest=True,
                           on_frame=count_resign)
    # a part covered completely is what the layers should give, not a problem
    # (U26): one summary line; an empty mask with no reason stays listed
    from tda.core.empty_visible import split_problems

    problems, expected = split_problems(list(stats["problems"]), stats["empty_visible"])
    unexplained = [row for row in stats["empty_visible"] if not row.expected]
    standing = _open_conflicts(stats["truth"], db, [desktop], view)
    # a confirmed frame whose card still has an open ✂ row ships the new state
    # next to the old shape (U8 round 2): visible here, not only on the card
    from tda.core.split_rows import (
        CUT_COPY_IN_FORCE,
        SHAPE_PREDATES_STATE,
        cut_copies_in_force,
        stale_split_rows,
    )

    stale = stale_split_rows(db, tax, desktop, view)
    problems.extend(f"{SHAPE_PREDATES_STATE}:{instance} (step {step})"
                    for step, instance in stale)
    # ... and every other confirmed frame a cut copy -- a placeholder, never an
    # answer -- is in force on (U11 round 2); the ✂ frame itself is the line above
    copies = cut_copies_in_force(db, tax, desktop, view, skip=stale)
    problems.extend(f"{CUT_COPY_IN_FORCE}:{instance} (step {step})"
                    for step, instance in copies)
    # a chassis, or a motherboard over a part it lies under, stored too high in
    # a layer order (U17): reported, never reordered
    from tda.core.zorder_floor import ZORDER_FLOOR, floor_problems

    floor = floor_problems(db, tax, desktop, view, stats["truth"])
    problems.extend(floor)
    # a stored 「被线挡住」 reason on a frame that cannot carry it (U15): no shape,
    # taken out, or on the bench -- data to repair (I there: 撤回)
    from tda.core.occluder import OCCLUDER_WITHOUT_SHAPE, occluders_without_shape

    lost = occluders_without_shape(db, tax, desktop, view)
    problems.extend(f"{OCCLUDER_WITHOUT_SHAPE}:{instance} (step {step})"
                    for step, instance in lost)
    if is_component_view(view):
        # a relocated sheet photo whose sheet row was retyped later (U10):
        # first, a per-shot list of missing shapes would hide it past --limit
        problems[:0] = P.orphan_sheet_shots(db, desktop, view)
    unit = "shots" if is_component_view(view) else "steps"   # the gallery's frames (U10)
    print(f"[check] D{desktop:02d} {view}: {stats['steps']} {unit}, "
          f"{stats['updated']} rows written, {stats['conflicts']} conflicts")
    if stats.get("closed"):
        print(f"[check] {stats['closed']} conflict(s) closed: the inputs agree with "
              f"the frozen rows again")
    if stats.get("overtaken"):
        print(f"[check] {stats['overtaken']} conflict(s) replaced by a newer "
              f"disagreement about the same frozen row")
    if stats["pending"]:
        print(f"[check] {len(stats['pending'])} frames are still queued for "
              f"a re-check: {sorted(stats['pending'])[:10]}")
    _print_empty_visible(expected, unexplained)
    if stale:
        print(f"[check] {len(stale)} confirmed frame(s) still carry the shape from "
              f"before their state changed ({SHAPE_PREDATES_STATE}, Q there): "
              + ", ".join(f"{instance}@{step}" for step, instance in stale[:10]))
    if copies:
        print(f"[check] {len(copies)} confirmed frame(s) show a cut copy -- the other "
              f"state's shape kept off the run -- nobody has drawn or confirmed "
              f"({CUT_COPY_IN_FORCE}; Q or Ctrl+Shift+K at the ✂ frame): "
              + ", ".join(f"{instance}@{step}" for step, instance in copies[:10]))
    if floor:
        print(f"[check] {len(floor)} layer order(s) put the chassis, or the motherboard "
              f"over a part lying on it, too high ({ZORDER_FLOOR}; not changed -- select "
              f"it and Ctrl+Down in the instance list): "
              + "; ".join(text.split(":", 1)[1] for text in floor[:10]))
    if lost:
        print(f"[check] {len(lost)} frame(s) say a part is hidden by a cable where it is "
              f"not in its place with a shape -- no shape, taken out, or on the bench "
              f"({OCCLUDER_WITHOUT_SHAPE}): 按 ` 键撤回（或从零件还在的帧按 Ctrl+` 撤回）: "
              + ", ".join(f"{instance}@{step}" for step, instance in lost[:10]))
    if resign:
        frames = len({step for step, _instance in resign})
        print(f"[check] {len(resign)} 条已确认行（{frames} 帧）和今天的编译只差容差以内：不算冲突，"
              f"存的仍是旧像素（stale_within_tolerance；cli refreeze 或 Review 的「重签」改成"
              f"今天的编译）: " + ", ".join(f"{instance}@{step}" for step, instance in resign[:10]))
    if waiting:
        print(f"[check] 另有 {len(waiting)} 条同样落后、但挨着开着的冲突，等冲突处理完再重签: "
              + ", ".join(f"{instance}@{step}" for step, instance in waiting[:10]))
    if getattr(args, "visibility", False) and not is_component_view(view):
        _print_visibility_runs(db, desktop, view)
    for text in problems[: int(args.limit)]:
        print(f"  - {text}")
    if len(problems) > int(args.limit):
        print(f"  ... {len(problems) - int(args.limit)} more")
    print(f"problems: {len(problems)}")
    print(f"open conflicts: {standing}")
    # a counter for the audit, not a problem: the rows are the same annotation
    # by the app's own tolerance, and the exit code stays what it was.  The
    # rows waiting for a neighbour's conflict are counted apart (U32 N1)
    print(f"stale_within_tolerance: {len(resign)} (deferred {len(waiting)})")
    return EXIT_ERROR if (problems or standing or stats["pending"]) else EXIT_OK


def _print_empty_visible(expected: Sequence, unexplained: Sequence) -> None:
    """The empty visible masks: one line for the expected ones, one for the rest (U26).

    ``expected`` are the rows whose emptiness is what the frame should show,
    ``unexplained`` the ones listed with the problems; each is an
    :class:`~tda.core.empty_visible.EmptyVisible`. A part counts once however
    many frames it is hidden on; the frames are given beside it.
    """
    from tda.core.empty_visible import COVERED

    hidden = [row for row in expected if row.reason == COVERED]
    absent = [row for row in expected if row.reason != COVERED]
    if hidden:
        print(f"[check] {len({row.instance for row in hidden})} 个零件完全被挡住（正常）："
              f"{len(hidden)} 个零件帧，不算问题")
    if absent:
        print(f"[check] {len({row.instance for row in absent})} 个零件标了看不见（正常）："
              f"{len(absent)} 个零件帧，不算问题")
    if unexplained:
        print(f"[check] {len(unexplained)} 个零件帧什么都看不见、也没有原因（empty_visible，"
              f"列在下面）: "
              + ", ".join(f"{row.instance}@{row.step}" for row in unexplained[:10]))


def _print_visibility_runs(db, desktop: int, view: str) -> None:
    """``--visibility`` (task U36): every run of the visibility review, for an audit.

    One header line, then one ``visibility_run`` line per run -- part, first
    and last frame, frames, label, the reason a hand label gives (``-`` for
    none), set by hand or not, a verdict (a stamp that covers it) or not, and
    whether it is every confirmed frame of the part in the view. A listing,
    not a problem: the exit code is untouched.
    """
    from tda.core.vis_runs import view_runs

    runs = view_runs(db, desktop, view)
    pending = [run for run in runs if not run.reviewed]
    print(f"[check] visibility_runs D{desktop:02d} {view}: {len(runs)} 段 / "
          f"{sum(len(run.steps) for run in runs)} 帧，待审 {len(pending)} 段 / "
          f"{sum(len(run.steps) for run in pending)} 帧（可见性审查，按零件、按步）")
    for run in runs:
        print(f"  visibility_run {run.instance} a={run.steps[0]} b={run.steps[-1]} "
              f"n={len(run.steps)} label={run.visibility} occluder={run.occluder or '-'} "
              f"manual={'yes' if run.manual else 'no'} verdict={'yes' if run.reviewed else 'no'} "
              f"whole={'yes' if run.whole else 'no'}")


def _add_check(sub) -> None:
    p = sub.add_parser("check", help="recompile one or more views and count the problems")
    p.add_argument("--desktop", type=int, required=True)
    _add_view_flags(p)
    p.add_argument("--limit", type=int, default=20, help="how many problems to list")
    p.add_argument("--visibility", action="store_true", default=False,
                   help="also list the visibility review's runs per part (a-b, n, label, "
                        "occluder, verdict yes/no); quiet by default")
    p.set_defaults(func=cmd_check)


# --------------------------------------------------------------------------- #
# build-cache
# --------------------------------------------------------------------------- #
def cmd_build_cache(args: argparse.Namespace) -> int:
    """Copy the chosen frames from ``F:`` into the local cache.

    A thin wrapper: :func:`tda.core.cache.main` owns the burst selection, the
    manifest and the progress log; this only resolves ``--cache`` from
    ``paths.yaml`` so the two command lines agree on where the cache is.
    """
    from tda.core import cache

    paths = load_paths(args.paths)
    wanted = _desktop_list(args.desktops)
    runs = _runs(wanted) if wanted else [(int(args.first), int(args.last))]
    extra: list[str] = []
    if args.index:
        extra += ["--index", str(args.index)]
    if args.log:
        extra += ["--log", str(args.log)]
    if args.recompute:
        extra.append("--recompute")
    if args.force:
        extra.append("--force")
    # the components gallery's frames come from the database (task U10): it
    # holds the sheet-row photos the index still lists as steps. ``--db``
    # wins over paths.yaml, as for every other command.
    db_path = str(getattr(args, "db", None) or paths.get("db_path") or "")
    if db_path and Path(db_path).is_file():
        extra += ["--db", db_path]
    elif getattr(args, "db", None):
        print(f"[build-cache] no database at {db_path}: the components gallery is "
              f"copied as the index lists it (without the step-table photos, e.g. D66)")

    worst = 0
    views = ",".join(_view_list(args.views))
    for first, last in runs:
        argv = ["--cache", P.require(paths, "cache_dir"), "--views", views,
                "--first", str(first), "--last", str(last), *extra]
        worst = max(worst, int(cache.main(argv)))
    return worst


def _add_build_cache(sub) -> None:
    p = sub.add_parser("build-cache", help="copy the chosen frames into the local cache")
    _add_view_flags(p, plural_first=True)
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-66 or 1-3,13")
    # Kept because tda.core.cache.main speaks this and scripts already use it;
    # --desktops wins when both are given.
    p.add_argument("--first", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--last", type=int, default=66, help=argparse.SUPPRESS)
    p.add_argument("--index", default=None)
    p.add_argument("--log", default=None)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_build_cache)


# --------------------------------------------------------------------------- #
# exports
# --------------------------------------------------------------------------- #
def _export_prologue(args, db, desktops: Sequence[int], view: str, command: str):
    """``(exit code, truth)``: refuse while anything is unresolved, else go.

    An export is a release artefact: shipping a frame whose stored geometry the
    session had already marked stale -- or one a human is still being asked to
    adjudicate -- is worse than not shipping at all, so the command stops and
    says what to run ``check`` on.  The service it built is handed back so the
    exporter refreshes through the same one.

    Two reasons to stop, and the annotator can only overrule the second:
    a queued re-check is a job this command could not finish, while an open
    conflict is a question only they can answer.  ``--allow-conflicts`` says
    they have decided to ship over it.
    """
    stats = _prepare_truth(db, load_taxonomy(), desktops, view,
                           refresh=bool(getattr(args, "refresh", True)))
    verified, unverified = _frame_counts(db, desktops, view)
    print(f"[{command}] {view}: {verified} verified frames, {unverified} not "
          f"verified, {verified + unverified} considered")
    if stats.get("closed"):
        print(f"[{command}] {stats['closed']} conflict(s) closed: the inputs agree "
              f"with the frozen rows again")
    if stats.get("overtaken"):
        print(f"[{command}] {stats['overtaken']} conflict(s) replaced by a newer "
              f"disagreement about the same frozen row")
    if stats["pending"]:
        print(f"[{command}] refused: {len(stats['pending'])} frames are still "
              f"pending a re-check ({sorted(stats['pending'])[:10]}). Run "
              f"'python -m tda.cli check --desktop N --view {view}' first.")
        return EXIT_ERROR, stats["truth"]
    if not bool(getattr(args, "allow_conflicts", False)):
        standing = _open_conflicts(stats["truth"], db, desktops, view)
        if standing:
            print(f"[{command}] refused: {standing} open conflict"
                  f"{'' if standing == 1 else 's'} on the requested desktops. "
                  f"Resolve them in the annotator, or pass --allow-conflicts to "
                  f"export the frozen rows as they stand.")
            return EXIT_ERROR, stats["truth"]
    return None, stats["truth"]


def cmd_export_coco(args: argparse.Namespace) -> int:
    """Write the compiled truth of one view as a COCO file."""
    from tda.core.export.coco import export_coco

    with session(args, lock=True) as (paths, db):
        desktops = _desktop_list(args.desktops)
        if not desktops:
            print("[export-coco] nothing to export: pass --desktops")
            return EXIT_ERROR
        views = _view_list(args.views)
        targets = _out_paths(args.out, views, paths, "coco_{view}.json", "export-coco")
        if targets is None:
            return EXIT_ERROR
        worst = EXIT_OK
        for view in views:
            out = targets[view]
            refused, truth = _export_prologue(args, db, desktops, view, "export-coco")
            if refused is not None:
                worst = max(worst, refused)
                continue
            stats = _call_export(
                export_coco, db, load_taxonomy(), desktops, view, str(out),
                allow_conflicts=bool(args.allow_conflicts),
                only_verified=bool(args.only_verified),
                roi_crop=bool(args.roi_crop), truth=truth,
                # passed only to switch the amodal shapes or the hidden parts
                # off (U26): the defaults are the exporter's own, and one that
                # predates the keywords runs
                **({} if getattr(args, "amodal", True) else {"amodal": False}),
                **({} if getattr(args, "hidden", True) else {"hidden": False}),
            )
            if not _count(stats, "annotations"):
                worst = max(worst, _nothing_exported("export-coco", view, out,
                                                     bool(args.only_verified)))
                continue
            # ``images``/``annotations`` are the *lists*: printing them put the
            # whole COCO document on the terminal.
            outside = _context_count(stats)
            note = (f" ({outside} of them context objects outside the machine)"
                    if outside else "")
            _report_amodal(stats)
            print(f"[export-coco] {_count(stats, 'images')} images, "
                  f"{_count(stats, 'annotations')} annotations{note} -> {out}")
        return worst


def _report_amodal(stats: dict) -> None:
    """The parts hidden completely, and the shapes a visible mask had to widen (U26).

    Before the total, which stays the last line: it is the one a script reads.
    """
    found = stats.get("annotations")
    info = stats.get("info") if isinstance(stats.get("info"), dict) else {}
    if isinstance(found, list):
        crowd = sum(1 for ann in found if isinstance(ann, dict) and ann.get("iscrowd"))
        if crowd:
            print(f"[export-coco] {crowd} annotation(s) are parts hidden completely: "
                  f"empty segmentation, iscrowd 1, bbox = the amodal box")
    union = int(info.get("amodal_union_rows") or 0)
    if union:
        print(f"[export-coco] {union} annotation(s) had visible pixels outside the compiled "
              f"shape: exported with amodal = amodal | visible (their union)")


def _add_export_coco(sub) -> None:
    p = sub.add_parser("export-coco", help="compiled truth -> one COCO json per view")
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-20")
    _add_view_flags(p)
    p.add_argument("--out", default=None,
                   help="one output file; only with a single view (without it each "
                        "view is written to <cache_dir>/coco_<view>.json)")
    p.add_argument("--roi-crop", action="store_true",
                   help="crop to the pose-segment ROI (the amodal shapes too)")
    p.add_argument("--amodal", dest="amodal", action="store_true", default=True,
                   help="give every mask annotation its whole shape: amodal_segmentation, "
                        "amodal_bbox, amodal_area (the default)")
    p.add_argument("--no-amodal", dest="amodal", action="store_false",
                   help="leave the amodal fields out, for a smaller file")
    p.add_argument("--hidden", dest="hidden", action="store_true", default=True,
                   help="export a part the labelled parts cover completely as a crowd "
                        "region with its amodal shape (the default)")
    p.add_argument("--no-hidden", dest="hidden", action="store_false",
                   help="leave the completely covered parts out")
    p.add_argument("--only-verified", dest="only_verified", action="store_true",
                   default=True, help="export confirmed rows only (the default)")
    p.add_argument("--no-only-verified", dest="only_verified", action="store_false",
                   help="export the automatic rows as well")
    _add_refresh_flags(p)
    p.set_defaults(func=cmd_export_coco)


def _add_refresh_flags(p) -> None:
    """What an export does before it reads the truth table, and what stops it."""
    p.add_argument("--refresh", dest="refresh", action="store_true", default=True,
                   help="recompile the requested frames first (the default)")
    p.add_argument("--no-refresh", dest="refresh", action="store_false",
                   help="export what is stored, only draining the re-check queue")
    p.add_argument("--allow-conflicts", dest="allow_conflicts", action="store_true",
                   help="export even though frames of these desktops still carry "
                        "an open conflict. Without it the export refuses: a "
                        "conflict is a frozen row the recompile disagrees with, "
                        "and only a human can say which of the two ships")


def cmd_export_vlm(args: argparse.Namespace) -> int:
    """Write the P0 question set of spec 8.2 for one view as JSONL."""
    from tda.core.export.vlm import TASKS, export_vlm

    with session(args, lock=True) as (paths, db):
        desktops = _desktop_list(args.desktops)
        if not desktops:
            print("[export-vlm] nothing to export: pass --desktops")
            return EXIT_ERROR
        views = _view_list(args.views)
        gallery = [v for v in views if is_component_view(v)]
        if gallery:
            # the VLM questions are about the teardown (task U10): a layout
            # photo of the removed parts answers none of them
            print(f"[export-vlm] the components gallery ({', '.join(gallery)}) has no "
                  f"VLM questions: they are about the teardown sequence; skipped")
            views = [v for v in views if not is_component_view(v)]
            if not views:
                return EXIT_ERROR
        targets = _out_paths(args.out, views, paths, "vlm_{view}.jsonl", "export-vlm")
        if targets is None:
            return EXIT_ERROR
        tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()] or list(TASKS)
        unknown = [t for t in tasks if t not in TASKS]
        if unknown:
            print(f"[export-vlm] not a task: {', '.join(unknown)}; "
                  f"the P0 set is {','.join(TASKS)}")
            return EXIT_ERROR
        worst = EXIT_OK
        for view in views:
            out = targets[view]
            refused, truth = _export_prologue(args, db, desktops, view, "export-vlm")
            if refused is not None:
                worst = max(worst, refused)
                continue
            stats = _call_export(
                export_vlm, db, load_taxonomy(), desktops, view, str(out),
                allow_conflicts=bool(args.allow_conflicts), tasks=tasks,
                only_verified=bool(args.only_verified), truth=truth,
            )
            _report_vlm(stats, view)
            if not _count(stats, "records"):
                worst = max(worst, _nothing_exported("export-vlm", view, out,
                                                     bool(args.only_verified),
                                                     siblings_of_vlm(str(out))))
                continue
            # the total goes last and stays short: it is the line a script reads
            print(f"[export-vlm] {_count(stats, 'records')} records -> {out}")
        return worst


def siblings_of_vlm(out: str) -> Sequence[Path]:
    """The manifest and the summary written beside one VLM JSONL."""
    from tda.core.export.vlm import siblings

    return siblings(out)


def _report_vlm(stats: dict, view: str) -> None:
    """The per-task counts and everything the export declined to answer.

    Three silences, and a release run has to see all three. ``illegal_steps``:
    the logged action there breaks a hard constraint, so the graph and the log
    contradict each other and no affordance answer may be published as ground
    truth (the 18 lines of ``reports/constraints_report.md``).
    ``excluded_desktops``: no graph at all, so every answer would be "nothing is
    blocked". ``v10_excluded``: the log names a target nobody resolved, and V10
    is scored exhaustively, so an incomplete history is a wrong answer.
    """
    by_task = stats.get("by_task") or {}
    if by_task:
        print(f"[export-vlm] {view}: "
              + " ".join(f"{task}={n}" for task, n in sorted(by_task.items())))
        source = stats.get("by_source") or {}
        if source:
            print(f"[export-vlm] {view}: "
                  + " ".join(f"{k}={v}" for k, v in sorted(source.items())))
    illegal = stats.get("illegal_steps") or {}
    if illegal:
        listed = ", ".join(f"D{d}:{sorted(steps)}" for d, steps in sorted(illegal.items()))
        print(f"[export-vlm] {len(illegal)} desktop(s) have steps whose logged "
              f"action the graph forbids; V4/V5/V6/V16 say nothing there: {listed}")
    excluded = stats.get("excluded_desktops") or {}
    if excluded:
        listed = ", ".join(f"D{d}:{why}" for d, why in sorted(excluded.items()))
        print(f"[export-vlm] {len(excluded)} desktop(s) answer no affordance or "
              f"planning question at all: {listed}")
    no_v10 = stats.get("v10_excluded") or {}
    if no_v10:
        listed = ", ".join(f"D{d}:{len(t)}" for d, t in sorted(no_v10.items()))
        print(f"[export-vlm] {len(no_v10)} desktop(s) have logged actions whose "
              f"target is unresolved, so V10 is not asked of them: {listed}")
    _report_shortcuts(stats, view)


def _report_shortcuts(stats: dict, view: str) -> None:
    """How much of this file a model could answer without looking at an image.

    Printed, not buried: a generated benchmark is always at risk of being
    solvable from its own wording, and the only defence is to measure it every
    time and put the number where a release run sees it. Each classifier is
    fitted to the file it scores, so these are upper bounds no text-only model
    can beat.
    """
    shortcuts = stats.get("shortcuts") or {}
    for task in sorted(shortcuts):
        row = shortcuts[task]
        print(f"[export-vlm] {view}: {task} text-only upper bound — "
              f"majority {row['majority']:.1%}, verb {row['verb']:.1%}, "
              f"class {row['class']:.1%}, verb+class {row['verb_class']:.1%}, "
              f"template {row['template']:.1%} ({row['records']} records)")
    option = stats.get("v6_option_shortcut") or {}
    if option.get("records"):
        print(f"[export-vlm] {view}: V6 option-verb shortcut "
              f"{option['shortcut']:.1%} against {option['random']:.1%} random "
              f"({option['records']} records with options)")


def _add_export_vlm(sub) -> None:
    from tda.core.export.vlm_tasks import TASKS

    p = sub.add_parser("export-vlm", help="compiled truth -> one VLM JSONL per view")
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-20")
    _add_view_flags(p)
    p.add_argument("--out", default=None,
                   help="one output file; only with a single view (without it each "
                        "view is written to <cache_dir>/vlm_<view>.jsonl)")
    p.add_argument("--tasks", default=",".join(TASKS),
                   help=f"comma-separated task ids; the P0 set is {','.join(TASKS)}")
    p.add_argument("--only-verified", dest="only_verified", action="store_true",
                   default=False, help="ask only about confirmed rows")
    p.add_argument("--no-only-verified", dest="only_verified", action="store_false")
    _add_refresh_flags(p)
    p.set_defaults(func=cmd_export_vlm)


#: Appended to ``tda.cli.SUBCOMMANDS`` in the order they appear in ``--help``.
SUBCOMMANDS: tuple[Callable[[argparse._SubParsersAction], None], ...] = (
    _add_app,
    _add_check,
    _add_build_cache,
    _add_export_coco,
    _add_export_vlm,
)
