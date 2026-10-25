"""``infer-relations`` and ``adopt-relations``: the relational fields of a database on disk.

::

    python -m tda.cli infer-relations [--desktops 1-66] [--dry-run] [--force]
    python -m tda.cli adopt-relations --desktop 23 [--dry-run] [--keys K1,K2]

``adopt-relations`` (task U27, at the end of this module) makes the relational
fields set by hand before there was a record of them durable: it rehearses
``import-logs --force`` of one desktop, compares, and logs a ``relation_fix``
row (:mod:`tda.core.relation_replay`) for every field the re-import would
change -- under the lock, after a safety backup when there is something to
record; a ``--keys`` naming no instance of the desktop is refused. The rest
of this docstring is about ``infer-relations``.

``import-logs`` runs :func:`~tda.core.graph_rules.infer_relational_fields` as
part of each desktop's transaction, so every freshly imported machine already
has its ``screw.fastens``, its captive ``parent``/``attached`` pair, its latch
``of`` and a ``socket_host`` that names an *instance* rather than a class. This
command is for the database that was imported before that was wired in: it does
the same thing in place, without re-reading a single sheet, so no manual edit to
a step table is touched.

What makes it safe to run on live work:

* it takes the single-user lock and backs the database up first, exactly like
  ``import-logs --force`` (``--dry-run`` does neither, because it writes no
  data -- the database is still opened and schema-checked under the lock);
* a desktop that already carries **verified** frames is reported and refused
  unless ``--force``, because filling a captive screw's ``parent`` changes what
  later frames need geometry for;
* it never overwrites a field that already holds a value, so every human
  correction survives -- which also makes a second run report zero fills;
* one desktop is one transaction, and a desktop that raises is reported and
  skipped rather than ending the run;
* every changed instance gets an ``op_log`` row of kind :data:`OP_KIND` holding
  the old and the new value of each field, so the run is auditable and can be
  undone record by record.

Instance *keys* never change here -- only the relational columns -- so nothing
that points at an instance (keyframes above all) can be orphaned by it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from tda.cli_common import (
    EXIT_ERROR,
    EXIT_OK,
    desktops as _desktops,
    safety_backup as _safety_backup,
    session as _session,
)
from tda.core.db import Db
from tda.core.dbrows import now_iso
from tda.core.graph_infer import (
    AMBIGUOUS,
    NO_CANDIDATE,
    infer_relational_fields,
    unresolved_kind,
    unresolved_relations,
)
from tda.core.implied import OP_KIND as IMPLIED_OP_KIND
from tda.core.implied import implied_instances
from tda.core.model import InstanceRec, is_context, is_extra, is_provisional, machine_only
from tda.core.relation_replay import OP_RELATION_FIX, fix_rows, relation_changes
from tda.core.states import events_from_actions, initial_overrides
from tda.core.taxonomy import Taxonomy, load_taxonomy

__all__ = [
    "ADOPT_ANNOTATOR", "ANNOTATOR", "OP_KIND", "OP_VIEW", "Adoption", "DesktopRelations",
    "RelationsRun", "adopt_relations", "adoption_lines", "infer_relations_into_db",
    "rehearse_import", "sheet_of",
]

#: ``op_log.kind`` of every row this command writes.
OP_KIND = "infer_relations"
#: ``op_log`` is scoped per ``(desktop, view)``; a relational field belongs to
#: no view, so it gets this placeholder rather than an arbitrary real one.
OP_VIEW = "-"
#: ``op_log.annotator`` -- this is machinery, not a person.
ANNOTATOR = "cli:infer-relations"

#: The instance columns the heuristic may fill (``attrs`` carries a latch's
#: ``of``), in the order they are reported.
TRACKED = ("parent", "attached", "mounted_on", "fastens", "socket_host", "attrs")


# --------------------------------------------------------------------------- #
# run records
# --------------------------------------------------------------------------- #
@dataclass
class DesktopRelations:
    """What ``infer-relations`` did for one desktop."""

    desktop: int
    status: str  # applied | refused | failed
    fills: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    changed: int = 0  # instances whose stored row was rewritten
    #: One line per instance ``--add-implied`` created for this desktop.
    implied: list[str] = field(default_factory=list)
    error: str = ""

    def unresolved_of(self, kind: str) -> int:
        """How many unresolved lines are of one kind (ambiguous / no candidate)."""
        return sum(1 for line in self.unresolved if unresolved_kind(line) == kind)


@dataclass
class RelationsRun:
    """The whole run."""

    dry_run: bool = False
    runs: list[DesktopRelations] = field(default_factory=list)
    #: Desktops of the selection that carry nothing frozen, so a second run
    #: narrowed to them would go through. Filled only when the run was refused.
    rest: list[int] = field(default_factory=list)

    @property
    def applied(self) -> list[DesktopRelations]:
        return [r for r in self.runs if r.status == "applied"]

    @property
    def failed(self) -> list[DesktopRelations]:
        return [r for r in self.runs if r.status == "failed"]

    @property
    def refused(self) -> list[DesktopRelations]:
        """Desktops skipped because they carry verified work and ``--force`` was not given."""
        return [r for r in self.runs if r.status == "refused"]

    @property
    def fills(self) -> int:
        return sum(len(r.fills) for r in self.runs)

    @property
    def unresolved(self) -> int:
        return sum(len(r.unresolved) for r in self.runs)

    def unresolved_of(self, kind: str) -> int:
        return sum(r.unresolved_of(kind) for r in self.runs)

    @property
    def changed(self) -> int:
        return sum(r.changed for r in self.runs)

    @property
    def implied(self) -> int:
        return sum(len(r.implied) for r in self.runs)

    def tally(self) -> str:
        """``N unresolved (A ambiguous, C no candidate)`` for a summary line.

        The split is what decides who does the next piece of work: an ambiguous
        reference needs one of several instances picked, a missing one needs an
        instance created first -- and whether a never-removed part gets an
        instance at all is the user's call (decision C7), not this command's.
        """
        return (
            f"{self.unresolved} unresolved ({self.unresolved_of(AMBIGUOUS)} "
            f"{AMBIGUOUS}, {self.unresolved_of(NO_CANDIDATE)} {NO_CANDIDATE})"
        )


# --------------------------------------------------------------------------- #
# what changed
# --------------------------------------------------------------------------- #
def refusal_line(run: RelationsRun) -> str:
    """The one sentence a refused run ends with.

    Said once, not once by the runner and again by the command, and it carries
    the argument that acts on it: the exact ``--desktops`` list of everything in
    the selection that *would* have gone through, ready to paste. Without that
    the annotator's only offered move is ``--force``, which is the one thing the
    refusal exists to slow down.
    """
    listed = ", ".join(f"D{r.desktop:02d}" for r in run.refused)
    rest = (f", or run the rest with --desktops {','.join(str(d) for d in run.rest)}"
            if run.rest else "")
    return (
        f"[infer-relations] refused before touching anything: {listed} carry verified "
        f"frames, whose frozen rows would be re-checked and may raise conflicts. "
        f"Re-run with --force to fill them anyway{rest}."
    )


def _snapshot(rec: InstanceRec) -> dict[str, Any]:
    """The tracked fields of one instance, deep enough to compare afterwards."""
    return {
        name: dict(rec.attrs) if name == "attrs" else getattr(rec, name)
        for name in TRACKED
    }


def _diff(before: dict[str, Any], after: dict[str, Any]) -> tuple[dict, dict]:
    """``(new, old)`` for the fields that differ; both empty when nothing did."""
    names = [name for name in TRACKED if before[name] != after[name]]
    return ({n: after[n] for n in names}, {n: before[n] for n in names})


# --------------------------------------------------------------------------- #
# one desktop
# --------------------------------------------------------------------------- #
def _apply_one(db: Db, tax: Taxonomy, desktop: int, dry_run: bool,
               add_implied: bool = False, reset_declined: bool = False) -> DesktopRelations:
    """Infer, and (unless ``dry_run``) write, one desktop's relational fields.

    The whole desktop lands in one transaction: the implied instances, the
    changed instance rows, an ``op_log`` row per new or changed instance, and --
    only when something actually changed -- the stored automatic state events,
    which are recompiled because the cascade of spec 3.3 reads
    ``attached``/``parent`` off the instance table.
    (:func:`tda.core.truth_inputs.events_of` re-derives that log on every read
    and ignores the stored ``auto=True`` rows, so this is about the copy
    :mod:`tda.ui.steps_model` and the status report read directly.)

    ``add_implied`` runs :func:`tda.core.implied.implied_instances` **before**
    the heuristic, exactly as ``import-logs`` does, so the references the four
    never-lifted motherboards leave behind resolve onto the new instance in the
    same pass. A class the annotator deleted in S1 stays refused unless
    ``reset_declined`` takes that back, for the desktops this run was pointed at
    and inside each desktop's own transaction: a desktop whose inference then
    raises must not come out of it having forgotten a refusal it never acted on.
    """
    declined = set() if reset_declined else db.declined_implied(desktop)
    # the machine's table: a 框外零件 (task U18) neither implies nor is implied,
    # and no heuristic fills a field on one
    instances = machine_only(db.instances(desktop))
    actions = db.actions(desktop)
    new_instances = (implied_instances(instances, actions, tax, declined)
                     if add_implied else [])
    for rec in new_instances:
        instances[rec.key] = rec
    before = {key: _snapshot(rec) for key, rec in instances.items()}
    fills = infer_relational_fields(instances, tax, actions)
    made = {rec.key for rec in new_instances}
    changes = []
    for key, rec in sorted(instances.items()):
        new, old = _diff(before[key], _snapshot(rec))
        if new and key not in made:
            changes.append((key, new, old))
    out = DesktopRelations(
        desktop=desktop, status="applied", fills=fills,
        # `declined` is this desktop's own refusals (empty under --reset-declined,
        # which has just taken them back): a class S1 said no to is not something
        # to keep asking about once per latch
        unresolved=unresolved_relations(instances, tax, actions, declined),
        changed=len(changes),
        implied=[f"implied instance {rec.key}: {rec.attrs.get('note', '')}"
                 for rec in new_instances],
    )
    if dry_run or not (changes or new_instances or reset_declined):
        return out
    with db.transaction():
        if reset_declined:
            # inside the desktop's own transaction: a desktop that then raises
            # must not have forgotten a refusal it did not act on
            db.reset_declined_implied(desktop)
        for rec in new_instances:
            db.upsert_instance(instances[rec.key])
            db.log_op(
                desktop, OP_VIEW, IMPLIED_OP_KIND,
                {"instance": rec.key, "cls": rec.cls, "attrs": dict(instances[rec.key].attrs)},
                {"instance": rec.key}, ANNOTATOR,
            )
        for key, new, old in changes:
            db.upsert_instance(instances[key])
            db.log_op(
                desktop, OP_VIEW, OP_KIND,
                {"instance": key, "fields": new}, {"instance": key, "fields": old},
                ANNOTATOR,
            )
        initial = initial_overrides(e for e in db.events(desktop) if not e.auto)
        db.replace_events(
            desktop, events_from_actions(instances, actions, tax, initial=initial),
            auto_only=True,
        )
    return out


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def verified_frames(db: Db, desktop: int) -> int:
    """How many *frames* of this desktop a human has frozen (spec 3.4).

    One frame is one ``(view, step)``. Counting ``compiled_mask`` rows instead
    -- one per instance of the frame -- is what made this report "43 verified
    frames" for a desktop with two, which reads like a reason to stop rather
    than like the truth. :meth:`tda.core.db.Db.verified_frame_count` owns the
    query now, so the guard and the re-check queue cannot drift apart.
    """
    return db.verified_frame_count(desktop)


def _queue_rechecks(db: Db, desktop: int, log=None) -> int:
    """Queue this desktop's verified frames for a re-check by the truth service.

    The relational fills can change which instances a later frame needs, so
    every frozen frame of the desktop is handed to the persisted re-check queue
    (one request per view). Nothing frozen is lost either way: the truth
    service raises a conflict when a verified row disappears from a recompiled
    frame; queueing only makes the annotator meet it now rather than later.

    The queueing happens whether or not there is anywhere to report it: a
    ``log`` of ``None`` is a caller that prints nothing, not a caller that wants
    the frozen frames left unchecked. Returns how many were queued.
    """
    by_view: dict[str, list[int]] = {}
    for view, step in db.verified_frames(desktop):
        by_view.setdefault(view, []).append(step)
    queued = 0
    for view, steps in by_view.items():
        queued += len(db.add_rechecks(desktop, view, steps))
    if log:
        log(f"[infer-relations] D{desktop:02d}: queued {queued} verified frames for "
            f"re-check (open the desktop in the app, or run `python -m tda.cli check "
            f"--desktop {desktop}`)")
    return queued


def _selected(db: Db, desktops: Optional[set[int]], log) -> list[int]:
    """The desktops to work on, warning about any ``--desktops`` id the DB lacks."""
    known = db.desktop_ids()
    if desktops is None:
        return known
    missing = sorted(desktops - set(known))
    if missing and log:
        for desktop in missing:
            log(f"[infer-relations] D{desktop:02d} is not in the database; skipped")
    return [desktop for desktop in known if desktop in desktops]


def infer_relations_into_db(
    db: Db,
    tax: Taxonomy,
    desktops: Optional[set[int]] = None,
    dry_run: bool = False,
    force: bool = False,
    log=None,
    add_implied: bool = False,
    reset_declined: bool = False,
) -> RelationsRun:
    """Run the heuristic over every desktop in the database, one transaction each.

    A desktop that already carries **verified** compiled rows is reported and
    then refused unless ``force``: filling a captive screw's ``parent`` changes
    what :func:`tda.core.states.needs_geom` answers on every later frame, so a
    row a human froze can stop being produced. That is not silent -- the truth
    service turns a vanished verified row into a conflict on the next refresh --
    but it is not something to spring on an annotator either. ``dry_run``
    reports the count and carries on, because it writes nothing.
    """
    run = RelationsRun(dry_run=dry_run)
    prefix = "[infer-relations]" + (" (dry run)" if dry_run else "")
    selected = _selected(db, desktops, log)

    # Pre-scan: a run that would refuse desktop 40 must not first have rewritten
    # thirty-nine. The whole selection is checked before anything is written, so
    # the annotator can decide once -- --force, or a narrower --desktops.
    frozen_by_desktop = {d: verified_frames(db, d) for d in selected}
    for desktop, frozen in frozen_by_desktop.items():
        if frozen and log:
            log(f"{prefix} D{desktop:02d} has {frozen} verified frames; their frozen "
                f"rows will be re-checked and may raise conflicts")
    if not (force or dry_run) and any(frozen_by_desktop.values()):
        for desktop, frozen in frozen_by_desktop.items():
            if frozen:
                run.runs.append(DesktopRelations(
                    desktop=desktop, status="refused",
                    error=f"{frozen} verified frames; re-run with --force to proceed",
                ))
        run.rest = [d for d in selected if not frozen_by_desktop[d]]
        if log:
            log(refusal_line(run))
        return run

    for desktop in selected:
        frozen = frozen_by_desktop[desktop]
        try:
            one = _apply_one(db, tax, desktop, dry_run, add_implied, reset_declined)
        except Exception as exc:  # one bad desktop must not end the run
            one = DesktopRelations(
                desktop=desktop, status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        run.runs.append(one)
        if log:
            _log_desktop(log, prefix, one)
        # A new implied instance is not in `changed` -- it is a row that did not
        # exist, not one that was rewritten -- but it is the *biggest* change
        # there is for a frozen frame: `needs_geom` gains an instance on every
        # single frame of the desktop.
        if frozen and (one.changed or one.implied) and not dry_run:
            _queue_rechecks(db, desktop, log)
    if log:
        log(f"{prefix} {len(run.applied)} desktops, {run.implied} implied instances, "
            f"{run.fills} fills on {run.changed} instances, {run.tally()}, "
            f"{len(run.refused)} refused, {len(run.failed)} failed")
        if run.failed:
            # the one thing a reader wants after a stack of per-desktop lines:
            # is the rest of the database in the state this command promises?
            log(f"{prefix} {len(run.failed)} desktops failed; every other desktop was "
                f"applied. Re-run for them once the cause is fixed: "
                f"--desktops "
                + ",".join(str(r.desktop) for r in run.failed))
    return run


def _log_desktop(log, prefix: str, one: DesktopRelations) -> None:
    """One desktop's headline plus every line it produced."""
    if one.status == "failed":
        log(f"{prefix} D{one.desktop:02d}: FAILED, {one.error}")
        return
    log(f"{prefix} D{one.desktop:02d}: {len(one.implied)} implied, {len(one.fills)} "
        f"fills on {one.changed} instances, {len(one.unresolved)} unresolved "
        f"({one.unresolved_of(AMBIGUOUS)} {AMBIGUOUS}, "
        f"{one.unresolved_of(NO_CANDIDATE)} {NO_CANDIDATE})")
    for text in one.implied:
        log(f"{prefix}   {text}")
    for text in one.fills:
        log(f"{prefix}   {text}")
    for text in one.unresolved:
        log(f"{prefix}   {text}")


# --------------------------------------------------------------------------- #
# command
# --------------------------------------------------------------------------- #
def cmd_infer_relations(args: argparse.Namespace) -> int:
    """Fill the relational fields of a database that was imported without them."""
    if args.reset_declined and not args.add_implied:
        print("[infer-relations] --reset-declined only means something with "
              "--add-implied: it forgets the implied instances S1 deleted so "
              "that they can be created again.")
        return EXIT_ERROR

    with _session(args, lock=True) as (paths, db):
        if not args.dry_run and not _safety_backup(
                paths, db, "infer-relations",
                "it rewrites the relational fields of every desktop"):
            return EXIT_ERROR  # the line is printed and nothing was written
        run = infer_relations_into_db(
            db, load_taxonomy(), _desktops(args), args.dry_run, args.force, log=print,
            add_implied=args.add_implied, reset_declined=args.reset_declined,
        )
        # the refusal is printed by the runner itself (`log=print`), once
        if run.failed or run.refused:
            return EXIT_ERROR
        return EXIT_OK


def _add_infer_relations(sub) -> None:
    p = sub.add_parser(
        "infer-relations",
        help="fill empty screw.fastens / parent / latch.of / socket_host in place",
    )
    p.add_argument("--desktops", default=None, help="e.g. 13 or 1-66")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be filled but write no data (the database "
                        "is still opened and schema-checked under the lock); no "
                        "backup is taken and a desktop with verified frames is "
                        "reported rather than refused")
    p.add_argument("--force", action="store_true",
                   help="also fill desktops that already carry verified frames, whose "
                        "frozen rows are then re-checked and may raise conflicts")
    p.add_argument("--add-implied", action="store_true",
                   help="first create the instances configs/taxonomy.yaml's "
                        "implied_when_referenced allows: a part the desktop clearly "
                        "has (the motherboard of D49/D62/D63/D64) that its log never "
                        "operates on. OFF here because this command exists to repair "
                        "a database in place; import-logs always does it")
    p.add_argument("--reset-declined", action="store_true",
                   help="with --add-implied, first forget which implied instances "
                        "were deleted in S1 on the selected desktops, so they are "
                        "created again. Without it a deleted one stays deleted, "
                        "through every re-import")
    p.set_defaults(func=cmd_infer_relations)


# --------------------------------------------------------------------------- #
# adopt-relations (task U27)
# --------------------------------------------------------------------------- #
#: ``op_log.annotator`` (and the fix's ``by``) of the rows ``adopt-relations`` writes.
ADOPT_ANNOTATOR = "cli:adopt-relations"


class _Rehearsal(Exception):
    """Raised to roll a rehearsed import back once its result is read."""

    def __init__(self, instances: dict[str, InstanceRec], replay: dict) -> None:
        super().__init__("rehearsal")
        self.instances = instances
        self.replay = replay


@dataclass
class Adoption:
    """What ``adopt-relations`` found -- and, unless a dry run, recorded -- on one desktop."""

    desktop: int
    sheet: str = ""
    #: ``(key, field, what a forced re-import would make, what is stored)``.
    differences: list[tuple[str, str, object, object]] = field(default_factory=list)
    #: Stored parts the sheet no longer makes: a re-import keeps them as they are.
    vanished: list[str] = field(default_factory=list)
    #: ``relation_fix`` rows written (0 on a dry run).
    recorded: int = 0
    #: After recording: ``"key.field = value (why)"`` of every difference a
    #: forced re-import would still make -- the replay refuses the fix.
    remaining: list[str] = field(default_factory=list)
    #: The backup before writing failed: nothing was recorded.
    aborted: bool = False


def sheet_of(directory: str | Path, desktop: int) -> Optional[Path]:
    """The canonical exported sheet of ``desktop`` in ``directory``, or ``None``."""
    from tda.core.logs import iter_desktop_csvs

    for number, path in iter_desktop_csvs(directory):
        if number == int(desktop):
            return path
    return None


def rehearse_import(db: Db, tax: Taxonomy, desktop: int, path: str | Path
                    ) -> tuple[dict[str, InstanceRec], dict]:
    """The instance table ``import-logs --force`` would write for ``desktop``; nothing stays.

    The import's own write (:func:`tda.pipeline_logs._write_import`) runs on
    the sheet inside a transaction that is rolled back once its result is
    read: merges, reclasses, ``Shift+T`` verbs, captive flips, implied
    instances, cage latches, the heuristic and the relation fixes recorded so
    far, exactly as a forced import does them. Nothing is dropped (a vanished
    part is simply not in the result). Returns ``(instances, the
    replay_relations payload)``. Must be the outermost transaction: inside
    another one the rollback would not be ours to make.
    """
    from tda.core.logs import import_log, read_desktop_csv
    from tda.pipeline_logs import DropPlan, _write_import

    if getattr(db, "_tx_depth", 0):
        raise RuntimeError("adopt-relations rehearses an import and cannot run inside "
                           "another transaction")
    rows, meta = read_desktop_csv(path)
    li = import_log(int(desktop), rows, meta, tax)
    try:
        with db.transaction():
            *_rest, relations = _write_import(db, li, tax, db.steps(int(desktop)),
                                              DropPlan(), {})
            raise _Rehearsal(dict(li.instances), relations.payload())
    except _Rehearsal as done:
        return done.instances, done.replay


def _wanted(key: str, rec: InstanceRec, keys: Optional[set[str]]) -> bool:
    """A part whose fields a re-import rewrites (and that ``--keys`` names)."""
    return (not is_provisional(key) and not is_context(rec) and not is_extra(rec)
            and (keys is None or key in keys))


def adopt_relations(db: Db, tax: Taxonomy, desktop: int, path: str | Path,
                    dry_run: bool = False, keys: Optional[set[str]] = None,
                    before_write: Optional[Callable[[], bool]] = None) -> Adoption:
    """Record a ``relation_fix`` for every relational field a forced re-import would change.

    The stored instance table is compared with :func:`rehearse_import`'s, field
    by field over :data:`tda.core.relation_replay.FIELDS`; each difference is
    one row (``old`` = what the re-import would make, ``new`` = what is
    stored, ``by`` = :data:`ADOPT_ANNOTATOR`), all in one transaction. The
    fixes already recorded are part of the rehearsal, so a second run finds
    nothing. Then the import is rehearsed once more and every difference still
    left is reported with the replay's reason (a fix naming a part the sheet
    no longer has). Parts added in S1, 框外零件 and Label Studio drafts are
    never compared: a re-import does not rewrite them. ``keys`` narrows it to
    those parts. ``dry_run`` compares and writes nothing. ``before_write`` is
    called only when there is something to record (the command's backup,
    round 3, R2-N5); ``False`` from it records nothing (``aborted``).
    """
    desktop = int(desktop)
    table = db.instances(desktop)
    stored = {key: rec for key, rec in table.items() if _wanted(key, rec, keys)}
    after, _replay = rehearse_import(db, tax, desktop, path)
    found = Adoption(desktop=desktop, sheet=str(path),
                     differences=relation_changes(after, stored),
                     vanished=sorted(key for key in stored if key not in after))
    if dry_run or not found.differences:
        return found
    if before_write is not None and not before_write():
        found.aborted = True
        return found
    at = now_iso()
    parts = {**after, **table}                     # the raw names of the parts named
    with db.transaction():
        for key, name, old, new in found.differences:
            payload, inverse = fix_rows(desktop, stored[key], name, old, new,
                                        ADOPT_ANNOTATOR, at, parts)
            db.log_op(desktop, OP_VIEW, OP_RELATION_FIX, payload, inverse, ADOPT_ANNOTATOR)
    found.recorded = len(found.differences)
    again, replay = rehearse_import(db, tax, desktop, path)
    why = {(e["key"], e["field"]): e["why"] for e in replay.get("unmatched") or ()}
    adopted = {key: stored[key] for key, *_ in found.differences}
    found.remaining = [f"{key}.{name} = {_said(new)} ({why.get((key, name), 'not replayed')})"
                       for key, name, _old, new in relation_changes(again, adopted)]
    return found


def _said(value: object) -> str:
    return "none" if value is None else str(value)


def adoption_lines(found: Adoption, dry_run: bool = False) -> list[str]:
    """What ``adopt-relations`` prints for one desktop."""
    prefix = "[adopt-relations]" + (" (dry run)" if dry_run else "")
    d = f"D{found.desktop:02d}"
    sheet = Path(found.sheet).name
    lines = []
    if not found.differences:
        lines.append(f"{prefix} {d}: nothing to adopt - a forced re-import of {sheet} "
                     f"keeps every relational field as stored")
    else:
        lines.append(f"{prefix} {d}: {len(found.differences)} relational field(s) differ "
                     f"from what a forced re-import of {sheet} would make:")
        lines.extend(f"{prefix}   {key}.{name} = {_said(new)} (a forced re-import would make "
                     f"it {_said(old)})" for key, name, old, new in found.differences)
    if found.vanished:
        lines.append(f"{prefix}   {len(found.vanished)} stored part(s) the sheet no longer "
                     f"has, kept as they are by a re-import: {', '.join(found.vanished)}")
    if not found.differences:
        return lines
    if dry_run:
        lines.append(f"{prefix} {d}: nothing recorded; run without --dry-run to record "
                     f"them as relation_fix ops")
        return lines
    kept = found.recorded - len(found.remaining)
    lines.append(f"{prefix} {d}: recorded {found.recorded} relation_fix op(s); a forced "
                 f"re-import now keeps {kept} of them")
    lines.extend(f"{prefix}   still lost: {text}" for text in found.remaining)
    return lines


def cmd_adopt_relations(args: argparse.Namespace) -> int:
    """Make the relational fields set before U27 survive ``import-logs --force``."""
    from tda import pipeline as P     # late: the pipeline is heavy

    with _session(args, lock=True) as (paths, db):
        desktop = int(args.desktop)
        if not db.steps(desktop):
            print(f"[adopt-relations] D{desktop:02d} has no steps; there is nothing a "
                  f"re-import would overwrite (run import-logs first)")
            return EXIT_ERROR
        directory = args.dir or P.drive_dir(paths)
        path = sheet_of(directory, desktop)
        if path is None:
            print(f"[adopt-relations] no sheet for D{desktop:02d} in {directory}")
            return EXIT_ERROR
        keys = ({k.strip() for k in str(args.keys).split(",") if k.strip()}
                if args.keys else None)
        unknown = sorted((keys or set()) - set(db.instances(desktop)))
        if unknown:
            print(f"[adopt-relations] D{desktop:02d} has no instance {', '.join(unknown)}; "
                  f"nothing was compared (check --keys)")
            return EXIT_ERROR
        # backed up only when there is something to record (round 3, R2-N5)
        found = adopt_relations(
            db, load_taxonomy(), desktop, path, args.dry_run, keys,
            before_write=lambda: _safety_backup(
                paths, db, "adopt-relations",
                "it records relation fixes every later forced re-import replays"))
        if found.aborted:
            return EXIT_ERROR  # the backup line is printed and nothing was written
        for line in adoption_lines(found, args.dry_run):
            print(line)
        return EXIT_OK


def _add_adopt_relations(sub) -> None:
    p = sub.add_parser(
        "adopt-relations",
        help="record the relational fields set by hand before U27, so import-logs "
             "--force keeps them",
    )
    p.add_argument("--desktop", type=int, required=True, help="one desktop, e.g. 23")
    p.add_argument("--dir", default=None, help="log folder (default <raw_logs_dir>/drive)")
    p.add_argument("--keys", default=None,
                   help="only these parts, comma-separated (e.g. cable_clip.01)")
    p.add_argument("--dry-run", action="store_true",
                   help="list the differences, record nothing (the import is still "
                        "rehearsed under the lock and rolled back); without it the "
                        "database is backed up first when there is something to record")
    p.set_defaults(func=cmd_adopt_relations)
