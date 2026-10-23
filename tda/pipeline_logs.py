"""The Drive-sheet half of the pipeline: sheets -> steps, actions, instances, events.

Second step of the order :mod:`tda.pipeline` documents. One exported sheet per
desktop becomes the step table the annotator works from (spec 2.3), the
instance identities behind it (3.2) and the automatic state-event log (6.3);
the frame index supplies the step durations and catches the two places where
sheet and cameras disagree.

The import is deliberately hard to lose work to. A desktop that already has
steps is skipped unless ``force`` is given, and even a forced run
* says per desktop what it is about to replace,
* carries the imported ``LS:`` note lines over to the new steps,
* keeps the verb the step table carries on every action corrected with
  ``Shift+T`` (:mod:`tda.core.verb_replay`), and the ``captive`` flag of every
  screw flipped with it (:mod:`tda.core.captive_replay`),
* keeps the instances merged in S1 merged (:mod:`tda.core.merge_replay`),
* keeps a part reclassed with 「其实是硬盘架」 / 「其实是卡扣」 under its new
  class and key (:mod:`tda.core.reclass_replay`, task U21),
* keeps a drive cage's implied latch that an annotator deleted in Steps
  deleted (:func:`tda.core.implied.cage_latches`, task U20), and
* leaves the hand-written (``auto=False``) state events alone.

Every drive cage of the sheet gets its own latch (task U20): adopted from the
sheet when it opens one and never removes it, kept when S1 added one, else
implied -- closed, open on the frame before the cage comes out, out with it.

One desktop is one transaction, and a sheet that cannot be read is recorded as
``failed`` and skipped rather than ending the run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from tda.core.db import Db
from tda.core.graph_rules import (
    infer_relational_fields,
    is_provisional,
    unresolved_relations,
)
from tda.core.implied import OP_KIND as IMPLIED_OP_KIND
from tda.core.implied import (
    CAGE_CLASS,
    OP_CAGE_LATCH,
    OP_DELETE_IMPLIED,
    CageLatches,
    cage_latches,
    implied_instances,
    is_implied,
)
from tda.core.index import DesktopIndex
from tda.core.log_report import _expected_steps
from tda.core.logs import LogImport, import_log, iter_desktop_csvs, read_desktop_csv
from tda.core import extra as X
from tda.core.extra import extra_keys
from tda.core.model import ALL_VIEWS, StepRec, StepType, is_context, is_extra
from tda.core.states import INITIAL_STEP, events_from_actions, initial_overrides
from tda.core.taxonomy import Taxonomy
from tda.core.merge_replay import (
    OP_MERGE,
    OP_REPLAY_MERGES,
    MergeReplay,
    effective_merges,
    replay_merges_for_import,
)
from tda.core.verb_replay import OP_REPLAY, Replay, replay_for_import
from tda.core.captive_replay import OP_REPLAY_CAPTIVE, CaptiveReplay, replay_captive_for_import
from tda.core.reclass_replay import OP_REPLAY_RECLASS, ReclassReplay, replay_reclass_for_import
from tda.pipeline import Log, merge_desktop_meta, settle_gallery, split_pose_segments, wanted

#: Mirrors :data:`tda.core.ls_import.NOTES_PREFIX`; repeated so importing the
#: pipeline does not pull in numpy/pycocotools. ``tests/test_cli.py`` pins them
#: together.
LS_NOTES_PREFIX = "LS: "
#: A step whose two bracketing photographs are further apart than this is
#: reported: it is more likely a break than a 30-minute operation.
MAX_STEP_SECONDS = 1800.0
#: Heading of the import report's per-desktop list of inferred relational
#: fields. They are guesses, and every one of them is an S1 question.
INFERRED_HEADING = "inferred relational fields (heuristic - confirm in S1)"
#: ``op_log`` is scoped per ``(desktop, view)``; an identity row belongs to no
#: view, so the implied-instance rows get this placeholder (same as
#: :data:`tda.cli_relations.OP_VIEW`).
OP_VIEW = "-"
#: ``op_log.annotator`` of an implied instance -- this is machinery, not a person.
IMPLIED_ANNOTATOR = "cli:import-logs"

#: Why a desktop was refused. The two have different answers -- one wants
#: ``--force-verified`` and the other ``--force-drop`` -- so the run-level line
#: has to tell them apart rather than assume the older of the two.
REFUSED_VERIFIED = "verified_frames"
REFUSED_IMPLAUSIBLE = "implausible_shrink"
#: The sheet (or the implied-instance rule) now produces a key an annotator
#: added in S1 for a part the log did not name -- ``--adopt-extras`` answers it.
REFUSED_EXTRAS = "extra_keys"

__all__ = [
    "DesktopRun", "INFERRED_HEADING", "LogsRun", "add_cage_latches", "add_implied_instances",
    "brand_model",
    "chassis_type", "desktop_fields", "expected_steps", "extra_clash", "import_logs_into_db",
    "inferred_section", "REFUSED_EXTRAS",
]


# --------------------------------------------------------------------------- #
# desktop metadata
# --------------------------------------------------------------------------- #
#: Lower-cased first token of ``Desktop Brand`` -> the canonical brand.
BRANDS = {
    "hp": "HP", "hewlett-packard": "HP", "compaq": "HP", "compoq": "HP", "compac": "HP",
    "dell": "Dell", "delle": "Dell", "lenovo": "Lenovo", "apple": "Apple",
    "acer": "Acer", "asus": "ASUS", "ibm": "IBM", "gateway": "Gateway",
}
#: Model families that name their brand implicitly (the sheets often drop "Dell").
IMPLIED_BRANDS = {
    "optiplex": "Dell", "precision": "Dell", "inspiron": "Dell", "vostro": "Dell",
    "dimension": "Dell", "elitedesk": "HP", "prodesk": "HP", "pavilion": "HP",
    "thinkcentre": "Lenovo",
}
#: Chassis-type keywords, most specific first (the sheets spell SFF many ways).
CHASSIS_TYPES = (
    ("small form", "sff"), ("sff", "sff"), ("ssf", "sff"), ("sf", "sff"),
    ("cmt", "cmt"), ("twr", "twr"), ("tower", "twr"), ("mt", "mt"), ("dt", "dt"),
)


def brand_model(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Split ``Desktop Brand`` into ``(brand, model_family)``.

    ``"HP EliteDesk 800 G2 TWR"`` -> ``("HP", "EliteDesk 800 G2 TWR")`` and
    ``"Optiplex 7020"`` -> ``("Dell", "Optiplex 7020")``; an unrecognised name
    keeps its whole text as the model and no brand.
    """
    text = " ".join(str(raw or "").split())
    if not text:
        return None, None
    tokens = text.split(" ")
    head = tokens[0].lower()
    if head in BRANDS:
        rest = " ".join(tokens[1:]).strip()
        return BRANDS[head], rest or None
    return IMPLIED_BRANDS.get(head), text


def chassis_type(raw: str) -> Optional[str]:
    """Guess the form factor (``sff``/``mt``/``twr``/``cmt``/``dt``) from the name."""
    words = str(raw or "").lower().replace("-", " ").split()
    text = " ".join(words)
    for needle, value in CHASSIS_TYPES:
        if " " in needle:
            if needle in text:
                return value
        elif needle in words:
            return value
    return None


def desktop_fields(meta: dict) -> dict:
    """The ``desktop`` columns (plus meta_json extras) one sheet's metadata fills."""
    raw = str(meta.get("brand_model_raw") or "").strip()
    brand, model = brand_model(raw)
    return {
        "brand": brand,
        "model_family": model,
        "chassis_type": chassis_type(raw),
        "size": (meta.get("size_raw") or None),
        "date": (meta.get("collection_date") or None),
        "notes": (meta.get("notes") or None),
        "brand_model_raw": raw or None,
        "sheet_desktop_id": meta.get("desktop_id"),
    }


# --------------------------------------------------------------------------- #
# run records
# --------------------------------------------------------------------------- #
@dataclass
class DesktopRun:
    """What ``import-logs`` did for one desktop."""

    desktop: int
    source: str
    status: str  # imported | skipped | refused | failed
    steps: int = 0
    actions: int = 0
    instances: int = 0
    events: int = 0
    durations: int = 0
    ls_notes: int = 0  # steps whose Label Studio notes were carried over
    #: Notes the re-imported sheet had no matching row for (it was renumbered).
    ls_notes_dropped: int = 0
    brand: str = ""
    #: For a ``refused`` run, which refusal it was: :data:`REFUSED_VERIFIED`,
    #: :data:`REFUSED_IMPLAUSIBLE` or :data:`REFUSED_EXTRAS`. Empty for every
    #: other status.
    reason: str = ""
    issues: list[str] = field(default_factory=list)
    #: ``"<key>.<field> = <value>"`` per relational field the heuristic filled.
    fills: list[str] = field(default_factory=list)
    #: ``"unresolved: ..."`` per reference it deliberately did not guess at.
    unresolved: list[str] = field(default_factory=list)
    #: One line per instance :mod:`tda.core.implied` created for this desktop.
    implied: list[str] = field(default_factory=list)
    #: One line per instance a ``--force`` re-import removed because this sheet
    #: no longer produces it (see :func:`_plan_drops`).
    dropped: list[str] = field(default_factory=list)
    #: Vanished instances kept rather than dropped, for any reason.
    kept_for_review: int = 0
    #: Of those, the ones held by work of their own; the rest are held only
    #: because a kept one names them, and need no decision of their own.
    kept_carrying_work: int = 0
    #: Keys of the instances an annotator added in S1 (:mod:`tda.core.extra`)
    #: that the import left alone -- the sheet does not name them.
    extras_kept: list[str] = field(default_factory=list)
    #: One line per added instance ``--adopt-extras`` handed to the sheet.
    adopted: list[str] = field(default_factory=list)
    #: The ``Shift+T`` verb corrections a forced re-import kept (one line) and
    #: the ones the new sheet no longer fits (one line each), task U9 round 2;
    #: then the same for the merges made in S1 (task U12).
    corrections: list[str] = field(default_factory=list)
    #: What the drive-cage latch rule did besides creating (task U20): a latch
    #: of the sheet adopted as a cage's own, one kept, one kept deleted, a
    #: deletion that no longer fits. The created ones are in ``implied``.
    cage_latches: list[str] = field(default_factory=list)


def inferred_section(run: DesktopRun) -> list[str]:
    """The report block listing what the heuristic filled for one desktop.

    Empty when it filled nothing, so a desktop whose sheet already said
    everything adds no noise. Kept here rather than in :mod:`tda.cli` so the
    heading and the bullet format live next to the run record that carries them.
    """
    cage = getattr(run, "cage_latches", [])
    if not (run.fills or run.unresolved or run.implied or cage):
        return []
    lines = [f"**{INFERRED_HEADING}**", ""]
    lines.extend(f"- {text}" for text in run.implied)
    lines.extend(f"- {text}" for text in cage)
    lines.extend(f"- {text}" for text in run.fills)
    lines.extend(f"- {text}" for text in run.unresolved)
    lines.append("")
    return lines


@dataclass
class LogsRun:
    """The whole ``import-logs`` run."""

    directory: str
    runs: list[DesktopRun] = field(default_factory=list)
    report_path: str = ""

    def _with(self, status: str) -> list[DesktopRun]:
        return [r for r in self.runs if r.status == status]

    @property
    def imported(self) -> list[DesktopRun]:
        return self._with("imported")

    @property
    def skipped(self) -> list[DesktopRun]:
        return self._with("skipped")

    @property
    def failed(self) -> list[DesktopRun]:
        return self._with("failed")

    @property
    def refused(self) -> list[DesktopRun]:
        """Desktops the run would not touch, for either of two reasons.

        Each carries a ``reason``: :data:`REFUSED_VERIFIED` (frozen frames were
        compiled from the step table this would replace -- ``--force-verified``)
        or :data:`REFUSED_IMPLAUSIBLE` (the re-imported sheet would shrink the
        desktop past believing -- ``--force-drop``). They want different answers,
        so a caller listing them must group by ``reason``.
        """
        return self._with("refused")

    def refused_of(self, reason: str) -> list[DesktopRun]:
        """The refused desktops of one :data:`REFUSED_VERIFIED`-style reason."""
        return [r for r in self.refused if r.reason == reason]

    @property
    def with_ls_notes(self) -> list[int]:
        """Desktops that **had** Label Studio notes when a forced re-import ran.

        Gated on having had them, not on how many survived: the desktop whose
        sheet was renumbered kept none at all, and that is precisely the one
        whose ``import-ls`` needs re-running.
        """
        return [r.desktop for r in self.imported if r.ls_notes or r.ls_notes_dropped]

    @property
    def dropped_ls_notes(self) -> list[DesktopRun]:
        """Desktops where a renumbered sheet cost some notes their home."""
        return [r for r in self.imported if r.ls_notes_dropped]


# --------------------------------------------------------------------------- #
# the index's contribution
# --------------------------------------------------------------------------- #
def _apply_index(
    steps: list[StepRec], di: Optional[DesktopIndex]
) -> tuple[int, list[str]]:
    """Fill ``duration_s`` from the index and cross-check it against the sheet.

    The duration of step ``k`` is ``ts_k - ts_(k-1)`` of the OAK cam1 captures
    (spec 2.2): the time between the two photographs that bracket the
    operation. Step 1 has no predecessor, and a step whose cam1 frame is missing
    (or whose timestamps run backwards) keeps ``None`` rather than a guess.

    Three disagreements between the sheet and the cameras are reported for the
    manual list of spec 2.2: a differing step count, a ``reorient`` row past the
    last indexed step (no view photographed it, so it opens no pose segment),
    and a gap longer than :data:`MAX_STEP_SECONDS`, which is recorded but is
    more likely a break than an operation.
    """
    if di is None:
        return 0, []
    times: dict[int, datetime] = {}
    for key, ff in di.frames.items():
        if key.view == "oak1" and ff.ts:
            try:
                times[key.step] = datetime.fromisoformat(ff.ts)
            except ValueError:
                pass
    filled, issues = 0, []
    for step in steps:
        before, after = times.get(step.step - 1), times.get(step.step)
        if before is None or after is None:
            continue
        seconds = (after - before).total_seconds()
        if seconds < 0:
            issues.append(
                f"D{step.desktop:02d} step {step.step}: oak1 capture times run backwards "
                f"({seconds:.0f}s); duration left unset"
            )
            continue
        if seconds > MAX_STEP_SECONDS:
            issues.append(
                f"D{step.desktop:02d} step {step.step}: {seconds / 60:.0f} min between the "
                f"oak1 captures of steps {step.step - 1} and {step.step} - a break rather "
                f"than the operation time? (recorded as measured)"
            )
        step.duration_s = seconds
        filled += 1
    if len(steps) != di.n_steps:
        issues.append(
            f"D{di.desktop:02d}: the sheet has {len(steps)} steps but the index has "
            f"{di.n_steps} - needs manual confirmation (spec 2.2)"
        )
    beyond = [
        s.step for s in steps
        if s.step_type == StepType.REORIENT.value and s.step > di.n_steps
    ]
    if beyond:
        issues.append(
            f"D{di.desktop:02d}: reorient steps {beyond} are past the index's last step "
            f"({di.n_steps}), so they open no pose segment - no view photographed them"
        )
    return filled, issues


# --------------------------------------------------------------------------- #
# writing one desktop
# --------------------------------------------------------------------------- #
def carry_ls_notes(steps: list[StepRec], previous: list[StepRec]) -> tuple[int, int]:
    """Copy the ``LS:`` note lines of the stored steps onto the freshly parsed ones.

    The Label Studio import folds the annotators' form fields into
    ``Step.notes`` (:mod:`tda.core.ls_import`), and
    :meth:`~tda.core.db.Db.replace_steps` would drop them on a forced
    re-import, so they are re-appended here.

    A note is carried over only when the step number **and** the raw step name
    agree with the row it came from. Matching on the number alone was wrong in
    the one case that matters: a re-exported sheet with a row inserted moves
    every later operation down by one, and the note describing "CPU Fan Screw 1"
    would have been re-attached to whatever now sits at that number -- silently,
    while the warning still read "(carried over)". A wrong note is worse than a
    missing one, because only the missing one can be rebuilt by re-running
    ``import-ls``. The raw names are compared stripped, since a trailing space
    in a hand-edited sheet is not a renumbering.

    Returns ``(steps that kept a line, notes that could not be carried over)``.
    """
    by_step = {s.step: s for s in steps}
    kept = dropped = 0
    for old in previous:
        lines = [
            line for line in (old.notes or "").splitlines()
            if line.startswith(LS_NOTES_PREFIX)
        ]
        if not lines:
            continue
        step = by_step.get(old.step)
        if step is None or step.raw_name.strip() != old.raw_name.strip():
            dropped += 1
            continue
        base = (step.notes or "").strip()
        step.notes = "\n".join(([base] if base else []) + lines)
        kept += 1
    return kept, dropped


def add_implied_instances(db: Db, li: LogImport, tax: Taxonomy) -> list[str]:
    """Create the instances the sheet implies but never operates on.

    Runs *before* the relational heuristic and inside the caller's transaction:
    the point of an implied ``motherboard.01`` is that the 33 references the
    four never-lifted boards leave behind then resolve onto it. Each one gets
    an ``op_log`` row of kind :data:`~tda.core.implied.OP_KIND`, so the run is
    auditable and a single record can be undone. Returns one report line each.

    A class the annotator has already deleted in S1 is skipped: the desktop's
    ``implied_declined`` meta is the one part of it a ``--force`` re-import does
    not rewrite, which is exactly why the refusal lives there.
    """
    lines = []
    declined = db.declined_implied(li.desktop)  # survives --force, by design
    for rec in implied_instances(li.instances, li.actions, tax, declined):
        li.instances[rec.key] = rec
        db.log_op(
            li.desktop, OP_VIEW, IMPLIED_OP_KIND,
            {"instance": rec.key, "cls": rec.cls, "attrs": dict(rec.attrs)},
            {"instance": rec.key}, IMPLIED_ANNOTATOR,
        )
        lines.append(f"implied instance {rec.key}: {rec.attrs.get('note', '')}")
    return lines


@dataclass
class CageInputs:
    """What the drive-cage latch rule reads off the database, before any write (U20)."""

    extras: dict
    stored: dict
    deleted: list[dict]
    #: ``gone -> gone_record`` of the merges in effect (U20 review, M1).
    merged: dict


def _cage_inputs(db: Db, desktop: int, instances: dict,
                 extras: Optional[dict] = None) -> CageInputs:
    """The parts added in S1, the table being replaced, the deletions and the merges."""
    stored = db.instances(desktop)
    if extras is None:
        extras = {key: rec for key, rec in stored.items()
                  if is_extra(rec) and key not in instances}
    in_effect = {entry["gone"] for entry in effective_merges(db, desktop)}
    merged = {}
    for op in db.ops_of_kind(desktop, OP_MERGE):
        payload = op.get("payload") or {}
        if payload.get("gone") in in_effect and payload.get("gone_record"):
            merged[str(payload["gone"])] = payload["gone_record"]
    return CageInputs(extras, stored, db.ops_of_kind(desktop, OP_DELETE_IMPLIED), merged)


def add_cage_latches(db: Db, li: LogImport, tax: Taxonomy, inputs: CageInputs,
                     rename: Optional[dict[str, str]] = None) -> CageLatches:
    """Give every drive cage of the sheet its own latch (task U20). **In the transaction.**

    Runs right after :func:`add_implied_instances` and before the relational
    heuristic: :func:`tda.core.implied.cage_latches` decides, this carries it
    out on the parsed records -- the adopted latch gets ``parent = cage`` and
    ``attached``, the implied ones (new, or made again under the key an earlier
    import gave them) are added -- and writes **one** ``op_log`` row of kind
    :data:`~tda.core.implied.OP_CAGE_LATCH` (inverse: the adopted ones
    unparented, the newly created keys). Nothing is written when the plan is
    empty: no cage the log removes, and no deletion or merge to report on --
    D22-D33 have a cage and get no row. ``inputs`` were read before the
    transaction opened (:func:`_cage_inputs`): the parts added in S1, the table
    being replaced, the :data:`~tda.core.implied.OP_DELETE_IMPLIED` rows (a
    latch deleted in Steps stays deleted) and the merges in effect (a latch
    merged in Steps stays merged).
    """
    plan = cage_latches(li.instances, li.actions, tax, inputs.extras, inputs.stored,
                        inputs.deleted, rename, inputs.merged)
    for entry in plan.adopted:
        rec = li.instances[entry["latch"]]
        rec.parent, rec.attached = entry["cage"], True
    for rec in plan.made:
        li.instances[rec.key] = rec
    if plan:
        db.log_op(li.desktop, OP_VIEW, OP_CAGE_LATCH, plan.payload(), plan.inverse(),
                  IMPLIED_ANNOTATOR)
    return plan


def _cage_keys(db: Db, li: LogImport, tax: Taxonomy) -> set[str]:
    """The keys the drive-cage latch rule will create, decided before anything is written.

    Asked on the records as the import will see them -- the merges and the
    ``Shift+T`` corrections replayed (both pure) -- because a corrected verb is
    exactly what decides it: D5's "HDD shield locker" opened by the sheet is the
    cage's latch, taken out at step 10 by ``Shift+T`` it is not.
    """
    instances, actions, merges = replay_merges_for_import(db, li.desktop, li.instances,
                                                          li.actions)
    # a latch reclassed as the cage (task U21) is a cage here too
    instances, actions, reclass = replay_reclass_for_import(db, li.desktop, instances,
                                                            actions, tax)
    if not any(rec.cls == CAGE_CLASS for rec in instances.values()):
        return set()
    actions, _replay = replay_for_import(db, li.desktop, actions, instances, tax,
                                         rename=merges.rename, reclassed=reclass.renames)
    got = _cage_inputs(db, li.desktop, instances)
    plan = cage_latches(instances, actions, tax, got.extras, got.stored, got.deleted,
                        merges.rename, got.merged)
    return {rec.key for rec in plan.made}


#: ``relation.source`` this pipeline owns. A derived edge naming a key that is
#: about to go is machinery and goes with it; anything else is somebody's work.
RULE_SOURCE = "rule"


def _references(db: Db, desktop: int, key: str) -> dict[str, int]:
    """Every reason not to delete one instance, per table; empty when there is none.

    :meth:`~tda.core.db.Db.instance_reference_counts` leaves out the two tables
    that have richer per-instance queries of their own, so they are asked here:
    ``shape_keyframe`` across every view, and the ``relation`` rows a human or
    the Label Studio import owns. A ``source="rule"`` edge is deliberately *not*
    a reason -- it was derived from the instance and is re-derived without it.
    """
    counts = dict(db.instance_reference_counts(desktop, key))
    shapes = sum(len(db.keyframes(desktop, view, key)) for view in ALL_VIEWS)
    if shapes:
        counts["shape_keyframe"] = shapes
    held = [r for r in db.relations(desktop)
            if key in (r["target"], r["blocker"]) and r["source"] != RULE_SOURCE]
    if held:
        counts["relation"] = len(held)
    return counts


#: ``op_log.kind`` written for every instance a forced re-import removes.
DROP_OP_KIND = "dropped_instance"

#: The relational columns one instance can name another with. A key that is kept
#: has to keep whatever it points at, or S1 inherits a dangling reference.
REFERRING_FIELDS = ("parent", "mounted_on", "fastens", "socket_host")

#: How small a re-imported step table may get before the sheet is not believed:
#: an export truncated to its header parses perfectly and means nothing.
MIN_STEP_RATIO = 0.6
#: ... and how much of a desktop's instance table one run may remove.
MAX_DROP_RATIO = 0.2
MAX_DROP_COUNT = 5
#: How many of the keys at risk a refusal lists before saying "and N more".
REFUSAL_KEYS_SHOWN = 10


@dataclass
class DropPlan:
    """What a re-import would remove from one desktop, decided before it writes.

    ``dropping`` is what will go, ``holding`` maps every vanished key that stays
    to the reason it stays, and ``refusal`` -- when set -- means the desktop is
    not imported at all.
    """

    dropping: list[str] = field(default_factory=list)
    holding: dict[str, str] = field(default_factory=dict)
    #: Of ``holding``, the keys held by work of their own rather than by another
    #: kept key. Counted apart because they are the ones S1 has to look at.
    carrying_work: int = 0
    refusal: str = ""
    #: Set when drops were found on a path that may not take them (no --force).
    deferred: int = 0


def _would_exist(db: Db, li: LogImport, tax: Taxonomy) -> set[str]:
    """Every key this import will end up having produced, implied ones included.

    :func:`implied_instances` is pure, so asking it here -- before anything is
    written -- costs nothing and lets the whole plan be decided outside the
    transaction. ``add_implied_instances`` asks it again for real, and so does
    :func:`add_cage_latches` for the drive cages' latches (:func:`_cage_keys`):
    without them a latch implied by an earlier import would be planned as
    vanished, dropped and made again in one run.
    """
    implied = implied_instances(li.instances, li.actions, tax,
                               db.declined_implied(li.desktop))
    # the keys a reclass replay gives the parsed parts (task U21): the sheet's
    # "HDD Locker Shield" parses to drive_latch.02, the import writes drive_cage.01
    merged, actions, _merges = replay_merges_for_import(db, li.desktop, li.instances,
                                                        li.actions)
    reclassed, _actions, _report = replay_reclass_for_import(db, li.desktop, merged, actions,
                                                             tax)
    return (set(li.instances) | set(reclassed) | {rec.key for rec in implied}
            | _cage_keys(db, li, tax))


def _hold_reasons(db: Db, desktop: int, vanished: set[str]) -> tuple[dict[str, str], int]:
    """Which vanished keys stay, and why -- including the ones they point at.

    A key carrying a human's work stays. So does any *other* vanished key that a
    staying one names in one of :data:`REFERRING_FIELDS`, transitively: keeping
    a frozen ``psu.02`` while deleting the ``psu.03`` it is mounted on would
    leave S1 a dangling reference, which is exactly the kind of tidy-up this
    function exists to avoid. A surviving *imported* instance can never point at
    a vanished key -- the importer has just rewritten every one of its
    relational fields from the sheet -- so only the vanished set is walked.

    Returns ``(key -> why, how many are held by work of their own)``. The second
    number is the one a human has to act on; the rest are along for the ride.
    """
    stored = db.instances(desktop)
    holding = {
        key: ", ".join(f"{n} {table}" for table, n in sorted(counts.items()))
        for key in sorted(vanished)
        for counts in [_references(db, desktop, key)]
        if counts
    }
    carrying_work = len(holding)
    # an instance S1 added is never dropped (see `_plan_drops`), so whatever it
    # rides on stays too -- its parent is what takes it out of the chassis
    for key, rec in sorted(stored.items()):
        if not is_extra(rec):
            continue
        for field_name in REFERRING_FIELDS:
            ref = getattr(rec, field_name, None)
            if ref in vanished and ref not in holding:
                holding[ref] = f"named by {key}.{field_name}, which is kept"
    growing = True
    while growing:
        growing = False
        for key in sorted(holding):
            rec = stored.get(key)
            for field_name in REFERRING_FIELDS:
                ref = getattr(rec, field_name, None) if rec else None
                if ref in vanished and ref not in holding:
                    holding[ref] = f"named by {key}.{field_name}, which is kept"
                    growing = True
    return holding, carrying_work


def _plan_drops(db: Db, li: LogImport, tax: Taxonomy, previous: list[StepRec],
                force: bool, force_drop: bool) -> DropPlan:
    """Decide what a re-import may remove, before it has written anything.

    The writer only ever upserted, so a key the sheet stopped naming simply
    stayed -- harmless while the identity rule holds still, and fatal the moment
    it changes. The thirteen PSU desktops of the real database were imported
    under the old rule; re-importing them merges ``psu.01`` and ``psu.02``, and
    without a drop the old ``psu.02`` would survive with no actions at all,
    ``installed`` in the chassis for ever.

    Deleting is the only thing this tool does that destroys annotator-visible
    data, so three gates stand in front of it:

    * **only under ``--force``**, which is the only import path that takes a
      safety backup first. A plain ``import-logs`` of a desktop whose steps are
      gone but whose instances are not reports the count and changes nothing.
    * **only if the sheet is believable.** An export truncated to its header
      parses perfectly and yields nothing; a parse *failure* was always safe
      (the desktop rolls back) and a parse *success* that means nothing was not.
      A re-import keeping less than :data:`MIN_STEP_RATIO` of the previous steps,
      or removing more than :data:`MAX_DROP_RATIO` of the real instances or more
      than :data:`MAX_DROP_COUNT` of them, refuses the desktop unless
      ``--force-drop`` says otherwise.
    * **only what nothing holds** -- see :func:`_hold_reasons`.

    Provisional ``ls:*`` keys are never in scope: they belong to ``import-ls``.
    Nor are the instances an annotator added in S1 (:mod:`tda.core.extra`):
    the sheet never named them, so its not naming them now says nothing. And
    never a 框外零件 (task U18): it is not part of the machine the sheet describes.
    """
    stored = db.instances(li.desktop)
    produced = _would_exist(db, li, tax)
    vanished = {key for key, rec in stored.items()
                if key not in produced and not is_provisional(key) and not is_extra(rec)
                and not is_context(rec)}
    if not vanished:
        return DropPlan()
    holding, carrying = _hold_reasons(db, li.desktop, vanished)
    dropping = sorted(vanished - set(holding))
    if not dropping:
        return DropPlan(holding=holding, carrying_work=carrying)
    if not force:
        return DropPlan(holding=holding, carrying_work=carrying,
                        deferred=len(dropping))
    if not force_drop:
        refusal = _implausible(li, previous, stored, dropping)
        if refusal:
            return DropPlan(refusal=refusal)
    return DropPlan(dropping=dropping, holding=holding, carrying_work=carrying)


def _implausible(li: LogImport, previous: list[StepRec], stored: dict,
                 dropping: list[str]) -> str:
    """Why this re-import is not believable, or ``""`` when it is.

    The exemption is "there is nothing to lose", and that is a fact about the
    *instances*, not about the step table. Keying it on ``previous`` alone
    skipped all three checks on exactly the state the ``--force`` gate above was
    written for -- steps gone, instances still there -- so a header-only sheet
    could delete 36 of a desktop's 37 instances and exit 0. A desktop is exempt
    only when it holds neither steps nor real instances.

    The step-ratio check needs a previous step count to be a ratio of; the two
    instance-count checks need only stored real instances, and run whenever
    there are any.
    """
    real = [key for key, rec in stored.items()
            if not is_provisional(key) and not is_implied(rec) and not is_extra(rec)
            and not is_context(rec)]
    if not previous and not real:  # a genuine first import: nothing to lose
        return ""
    drafts = sum(1 for key in stored if is_provisional(key))
    reasons = []
    if previous and len(li.steps) < MIN_STEP_RATIO * len(previous):
        reasons.append(f"{len(li.steps)} steps against {len(previous)} before")
    if real and len(dropping) > MAX_DROP_COUNT:
        reasons.append(f"{len(dropping)} instances would be dropped, more than "
                       f"{MAX_DROP_COUNT}")
    if real and len(dropping) > MAX_DROP_RATIO * len(real):
        reasons.append(f"{len(dropping)} of {len(real)} real instances would be "
                       f"dropped, more than {int(MAX_DROP_RATIO * 100)}%")
    if not reasons:
        return ""
    shown = ", ".join(dropping[:REFUSAL_KEYS_SHOWN])
    more = (f" and {len(dropping) - REFUSAL_KEYS_SHOWN} more"
            if len(dropping) > REFUSAL_KEYS_SHOWN else "")
    drafted = f" (+{drafts} Label Studio drafts)" if drafts else ""
    return (
        f"D{li.desktop:02d}: the re-imported sheet is not believable "
        f"({'; '.join(reasons)}); {len(real)} real instances stored{drafted}, "
        f"{len(li.instances)} in the sheet. Nothing was written. At risk: "
        f"{shown}{more}. Re-run with --force-drop if the sheet really is right."
    )


def _apply_drops(db: Db, li: LogImport, plan: DropPlan) -> tuple[list[str], list[str]]:
    """Carry the plan out inside the caller's transaction; ``(dropped, issues)``.

    Every removal is logged to ``op_log`` with the whole stored record and the
    derived edges that went with it, so a drop can be audited and undone one row
    at a time -- the same promise :func:`add_implied_instances` makes for the
    rows it creates.
    """
    dropped: list[str] = []
    issues = [
        f"D{li.desktop:02d}: instance {key} vanished from the sheet but is kept "
        f"({why}) - merge or delete it in S1"
        for key, why in sorted(plan.holding.items())
    ]
    if plan.deferred:
        issues.append(
            f"D{li.desktop:02d}: {plan.deferred} instances are no longer in the "
            f"sheet; re-run with --force to drop them"
        )
    if not plan.dropping:
        return dropped, issues
    # one read of each table for the whole desktop: the loop below only ever
    # removes rows, so re-reading per key would cost a query per drop to learn
    # nothing the first read did not already say
    stored = db.instances(li.desktop)
    relations = [dict(row) for row in db.relations(li.desktop)]
    for key in plan.dropping:
        rec = stored.get(key)
        edges = [row for row in relations
                 if key in (row["target"], row["blocker"])
                 and row["source"] == RULE_SOURCE]
        for row in edges:
            db.delete_relation(li.desktop, row["type"], row["target"], row["blocker"])
        db.delete_instance(li.desktop, key)
        record = asdict(rec) if rec is not None else {"key": key}
        db.log_op(
            li.desktop, OP_VIEW, DROP_OP_KIND,
            {"instance": key, "record": record, "relations": edges,
             "reason": _drop_reason(rec)},
            {"instance": key, "record": record, "relations": edges},
            IMPLIED_ANNOTATOR,
        )
        dropped.append(f"dropped instance {key} ({_drop_reason(rec)})")
    return dropped, issues


def _drop_reason(rec) -> str:
    """Why this key went, in the words the annotator needs to hear."""
    if rec is not None and is_implied(rec):
        return "implied instance no longer warranted"
    return "no longer in the sheet"


# --------------------------------------------------------------------------- #
# keys an annotator added in S1
# --------------------------------------------------------------------------- #
def extra_clash(db: Db, li: LogImport, tax: Taxonomy) -> dict[str, str]:
    """``key -> "log" | "implied"`` for every added instance this import would take.

    An instance added in S1 (:mod:`tda.core.extra`) holds a key of its class's
    ordinal run -- ``ram_latch.05`` after the log's ``.04``. A sheet re-exported
    with rows "RAM clip 5-8" produces exactly those keys, and so can the
    implied-instance rule (``motherboard.01`` after ``--reset-declined``). The
    writer upserts every produced row, so without this the log would take the
    added rows over in silence: provenance gone, the hand-set initial state now
    applying to the log's part, the shape drawn for one clip now standing for
    another.
    """
    extras = {key for key, rec in db.instances(li.desktop).items() if is_extra(rec)}
    if not extras:
        return {}
    implied = {rec.key for rec in implied_instances(
        li.instances, li.actions, tax, db.declined_implied(li.desktop))}
    out: dict[str, str] = {}
    for key in sorted(extras):
        if key in li.instances:
            out[key] = "log"
        elif key in implied:
            out[key] = "implied"
    return out


def _extras_refusal(desktop: int, clash: dict[str, str]) -> str:
    """The sentence a desktop refused for :data:`REFUSED_EXTRAS` is reported with."""
    named = ", ".join(f"{key} ({'the sheet' if who == 'log' else 'the implied rule'})"
                      for key, who in sorted(clash.items()))
    return (
        f"D{desktop:02d}: this import would take over {len(clash)} instance(s) an "
        f"annotator added in S1 for parts the log did not name: {named}. Nothing was "
        f"written. Re-run with --adopt-extras to hand them to the sheet (their "
        f"provenance and hand-set initial state go, their shapes stay on the key), or "
        f"delete them in S1 first if the sheet means different parts."
    )


def _adopt_extras(db: Db, desktop: int, taken_by: dict[str, str]) -> list[str]:
    """Hand added instances to the sheet. **Inside the caller's transaction.**

    The row itself is overwritten by the upsert that follows (the log's record:
    no provenance, the log's raw names); what has to go here is what the
    addition wrote -- the step-0 initial state the annotator set and the removal
    「拆下于第 k 步」 (:func:`tda.core.extra.initial_events`, task U10) -- which
    would otherwise apply to the log's part, whose actions now say when it
    leaves. Any other hand-set state stays. The shapes stay on the key. One
    ``op_log`` row records the added records and events as they were, as the inverse
    :func:`tda.core.extra.undo_op` applies.
    """
    if not taken_by:
        return []
    stored = db.instances(desktop)
    records = [stored[key] for key in sorted(taken_by)]
    # what the addition wrote: its initial state and the removal
    # 「拆下于第 k 步」 (U10) -- the sheet's part is taken out by the sheet
    events = [e for key in sorted(taken_by) for e in X.initial_events(db, desktop, key)]
    for key in taken_by:
        db.delete_manual_events(desktop, key, up_to_step=INITIAL_STEP,
                                evidence_view=X.ADDED_EVIDENCE)
    payload, inverse = X.adopt_patch(records, events, taken_by)
    db.log_op(desktop, OP_VIEW, X.OP_ADOPT, payload, inverse, IMPLIED_ANNOTATOR)
    return [
        f"adopted {key}: added in S1, now produced by the "
        f"{'sheet' if taken_by[key] == 'log' else 'implied-instance rule'} "
        f"(provenance, hand-set initial state and removal step dropped, shapes kept)"
        for key in sorted(taken_by)
    ]


def _write_import(
    db: Db, li: LogImport, tax: Taxonomy, previous: list[StepRec], plan: DropPlan,
    adopt: Optional[dict[str, str]] = None,
) -> tuple[int, int, int, list[str], list[str], list[str], list[str], list[str], list[str],
           Replay, MergeReplay, CaptiveReplay, CageLatches, ReclassReplay]:
    """Write one desktop's import atomically.

    Returns ``(events, steps with LS notes, notes dropped, fills, unresolved,
    implied, dropped instances, drop issues, adopted, replay, merges, captive,
    cage latches, reclasses)``. ``adopt`` is :func:`extra_clash` when
    ``--adopt-extras`` was given (see :func:`_adopt_extras`).

    Right after the merges, every part reclassed in the app (「其实是硬盘架」 /
    「其实是卡扣」, task U21) is made under its new class and key
    (:mod:`tda.core.reclass_replay`), so the verb replay, the cage rule and
    the inference below all see the cage; one ``replay_reclass`` row says
    which were kept and which no longer fit the sheet.

    Right after the implied instances, every drive cage of the sheet gets its
    own latch (:func:`add_cage_latches`, task U20) -- adopted from the sheet,
    kept, kept deleted or created -- before the relational heuristic, and the
    events below already open a created one on the frame before its cage
    comes out.

    Every merge made with 「合并到…」 that is still in effect is applied to the
    parsed records first (:mod:`tda.core.merge_replay`, task U12): the sheet
    names both halves of a merged part, and without this the merged-away key
    would come back with its step pointing at it again. One ``replay_merges``
    row in the op log says which merges were replayed, which the sheet no
    longer needs and which no longer fit it; the verb replay below follows the
    merged keys.

    Every action corrected with ``Shift+T`` keeps the verb the replaced step
    table carries -- the annotator's last word, undo and redo included -- on
    the parsed action before it is written (:mod:`tda.core.verb_replay`), and
    one ``replay_corrections`` row in the op log says which were kept, which
    the sheet already agrees with, and which no longer fit it.

    Every screw whose ``captive`` was flipped with ``Shift+T`` (task U14) keeps
    the value the replaced instance table carries over the importer's guess,
    matched by key (:mod:`tda.core.captive_replay`), before the relational
    inference below fills a captive screw's ``parent``; one ``replay_captive``
    row says which were kept, which agree and which no longer fit.

    The relational heuristic of spec 7.3 runs here, *before* the instances are
    written and inside the same transaction: ``logs.py`` leaves ``fastens``,
    ``parent``/``attached`` and a latch's ``of`` empty and may put a bare class
    name in ``socket_host``, and the state machine reads those the moment the
    annotator opens the desktop -- a captive cooler screw with no ``parent``
    stays in the chassis after its cooler is gone (spec 3.3). The freshly
    parsed actions go in with them, because the step order is what breaks the
    tie on a desktop that lists two coolers. Filling them
    first also means ``events_from_actions`` below already emits the cascade,
    so the stored automatic log and the one
    :func:`tda.core.truth_inputs.events_of` re-derives on every read agree --
    which is why the instances S1 added (:mod:`tda.core.extra`) are folded in
    too: the sheet does not know them, but the cascade that takes an added RAM
    clip out with its board is part of the same log.
    """
    declined = db.declined_implied(li.desktop)  # read before the transaction writes
    # the merges made in S1 (task U12), read off the instance table this
    # replaces, before anything else looks at the parsed records
    li.instances, li.actions, merges = replay_merges_for_import(
        db, li.desktop, li.instances, li.actions)
    # the parts reclassed in the app (task U21): made under their new class and
    # key here, before the verb replay, which follows the old keys to them
    li.instances, li.actions, reclass = replay_reclass_for_import(
        db, li.desktop, li.instances, li.actions, tax)
    # the verbs of the actions corrected with Shift+T, read off the step table
    # this replaces and put back onto the parsed actions (task U9), so the
    # events and the relational inference below see the corrected verbs
    li.actions, replay = replay_for_import(db, li.desktop, li.actions, li.instances, tax,
                                           rename=merges.rename, reclassed=reclass.renames)
    # the captive flag of the screws flipped with Shift+T (task U14), read off
    # the instance table this replaces, before the inference fills parents
    li.instances, captive = replay_captive_for_import(db, li.desktop, li.instances,
                                                      rename=merges.rename)
    adopt = dict(adopt or {})
    extras = {key: rec for key, rec in db.instances(li.desktop).items()
              if is_extra(rec) and key not in li.instances and key not in adopt}
    initial = {key: state for key, state in initial_overrides(
        e for e in db.events(li.desktop) if not e.auto).items() if key not in adopt}
    # the drive cages' own latches (task U20): the table being replaced (the
    # key an implied latch had), the latches deleted and merged in Steps
    cage_inputs = _cage_inputs(db, li.desktop, li.instances, extras)
    with db.transaction():
        kept, dropped = carry_ls_notes(li.steps, previous)
        merge_desktop_meta(db, li.desktop, desktop_fields(li.meta))
        db.replace_steps(li.desktop, li.steps, li.actions)
        if replay:
            db.log_op(li.desktop, OP_VIEW, OP_REPLAY, replay.payload(),
                      {"actions": replay.inverse}, IMPLIED_ANNOTATOR)
        if captive:
            db.log_op(li.desktop, OP_VIEW, OP_REPLAY_CAPTIVE, captive.payload(),
                      {"instances": captive.inverse}, IMPLIED_ANNOTATOR)
        if merges:
            db.log_op(li.desktop, OP_VIEW, OP_REPLAY_MERGES, merges.payload(),
                      merges.inverse, IMPLIED_ANNOTATOR)
        if reclass:
            db.log_op(li.desktop, OP_VIEW, OP_REPLAY_RECLASS, reclass.payload(),
                      reclass.inverse, IMPLIED_ANNOTATOR)
        adopted = _adopt_extras(db, li.desktop, adopt)
        implied = add_implied_instances(db, li, tax)
        cage = add_cage_latches(db, li, tax, cage_inputs, merges.rename)
        implied += cage.implied_lines()
        fills = infer_relational_fields(li.instances, tax, li.actions)
        for inst in li.instances.values():
            db.upsert_instance(inst)
        # the plan was decided before this transaction opened; carrying it out
        # in here is what makes a desktop that raises keep both its old
        # instances and its old steps
        gone, drop_issues = _apply_drops(db, li, plan)
        events = events_from_actions({**extras, **li.instances}, li.actions, tax,
                                     initial=initial)
        db.replace_events(li.desktop, events, auto_only=True)
        # a sheet row of the laid-out parts leaves the sequence (task U10)
        settle_gallery(db, li.desktop)
        split_pose_segments(db, li.desktop)
    return (len(events), kept, dropped, fills,
            unresolved_relations(li.instances, tax, li.actions, declined), implied,
            gone, drop_issues, adopted, replay, merges, captive, cage, reclass)


def dropped_notes_line(desktop: int, dropped: int) -> str:
    """The one line that says a renumbered sheet cost this desktop its notes.

    It counts *steps*: one step's notes are one carry-over, however many ``LS:``
    lines the Label Studio import folded onto it.
    """
    return (
        f"D{desktop:02d}: {dropped} steps with LS notes could not be carried over "
        f"(sheet renumbered) - re-run import-ls"
    )


def _force_warning(db: Db, desktop: int, previous: list[StepRec]) -> str:
    """What a ``--force`` run is about to overwrite, in one line.

    "carried over" is a promise about the notes whose step number *and* raw name
    survive the re-import; how many did not is only known afterwards and is said
    then (:func:`dropped_notes_line`).
    """
    ls_steps = db.steps_with_note_prefix(desktop, LS_NOTES_PREFIX)
    notes = (
        f"{len(ls_steps)} steps carry Label Studio notes (carried over where the "
        f"step number and name still match); "
        if ls_steps else ""
    )
    return (
        f"[import-logs] D{desktop:02d}: --force, replacing {len(previous)} steps / "
        f"{len(db.actions(desktop))} actions; {notes}the open/remove corrections made "
        f"with Shift+T are replayed where the sheet still has that action, the captive "
        f"flags flipped with Shift+T where it still has that screw, and so are "
        f"the instances merged in S1 (合并到…) and the parts reclassed (其实是硬盘架 / "
        f"其实是卡扣) where the sheet still names them; every other "
        f"manual edit to the step table is lost, and so is every relational field a "
        f"human set on an instance "
        f"(fastens, parent, mounted_on, socket_host): the instances are re-created from "
        f"the sheet and the heuristic fills them again"
    )


def _verified_refusal(db: Db, desktop: int, frozen: int, log: Optional[Log]) -> DesktopRun:
    """Report and refuse a forced re-import of a desktop with frozen frames.

    ``infer-relations`` only fills empty relational fields and still asks twice
    before touching a desktop a human has signed off on; a forced re-import
    rewrites the entire step table underneath those frames, which is strictly
    worse, so it asks too. ``--force-verified`` is the second answer.
    """
    if log:
        log(f"[import-logs] D{desktop:02d} has {frozen} verified frames; a forced "
            f"re-import would rewrite the step table underneath them. Refused, "
            f"nothing written - pass --force-verified to do it anyway")
    return DesktopRun(
        desktop, "", "refused", reason=REFUSED_VERIFIED,
        issues=[f"D{desktop:02d}: {frozen} verified frames; re-run with "
                f"--force-verified to re-import anyway"],
    )


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def import_logs_into_db(
    db: Db,
    directory: str,
    tax: Taxonomy,
    index: Optional[dict[int, DesktopIndex]] = None,
    desktops: Optional[set[int]] = None,
    force: bool = False,
    log: Optional[Log] = None,
    force_verified: bool = False,
    force_drop: bool = False,
    adopt_extras: bool = False,
) -> LogsRun:
    """Import the exported Drive sheets into steps, actions, instances and events.

    A desktop that already has steps is **skipped** unless ``force`` is given:
    the step table is where the annotator resolves the ``?`` targets and the
    compound rows, and :meth:`tda.core.db.Db.replace_steps` would throw that
    work away. A desktop that additionally carries **verified frames** is
    ``refused`` even then, unless ``force_verified`` says so as well: those
    frames were compiled from the step table this would replace. A sheet that
    cannot be read is recorded as ``failed`` and the run carries on to the next
    desktop.

    A desktop whose sheet now produces a key an annotator **added in S1**
    (:func:`extra_clash`) is ``refused`` too -- the rows would be taken over
    in silence -- unless ``adopt_extras`` hands them to the sheet.
    """
    index = index or {}
    run = LogsRun(directory=str(directory))
    for desktop, path in iter_desktop_csvs(directory):
        if not wanted(desktop, desktops):
            continue
        previous = db.steps(desktop)
        if previous and not force:
            run.runs.append(DesktopRun(desktop, str(path), "skipped"))
            if log:
                log(f"[import-logs] D{desktop:02d}: skipped, the database already has "
                    f"steps (use --force to overwrite)")
            continue
        frozen = db.verified_frame_count(desktop) if previous else 0
        if frozen and not force_verified:
            run.runs.append(_verified_refusal(db, desktop, frozen, log))
            continue
        if frozen and log:
            log(f"[import-logs] D{desktop:02d} has {frozen} verified frames; "
                f"--force-verified was given, so they are re-imported over")
        if previous and log:
            log(_force_warning(db, desktop, previous))
        try:
            run.runs.append(_import_one(db, desktop, path, tax, index, previous, log,
                                        force=force, force_drop=force_drop,
                                        adopt_extras=adopt_extras))
        except Exception as exc:  # one unreadable sheet must not end the run
            run.runs.append(DesktopRun(
                desktop, str(path), "failed",
                issues=[f"D{desktop:02d}: {type(exc).__name__}: {exc}"],
            ))
            if log:
                log(f"[import-logs] D{desktop:02d}: FAILED, {type(exc).__name__}: {exc}")
    return run


def _import_one(
    db: Db, desktop: int, path, tax: Taxonomy,
    index: dict[int, DesktopIndex], previous: list[StepRec], log: Optional[Log],
    force: bool = False, force_drop: bool = False, adopt_extras: bool = False,
) -> DesktopRun:
    """Read and write one desktop; raises if the sheet cannot be parsed."""
    rows, meta = read_desktop_csv(path)
    li = import_log(desktop, rows, meta, tax)
    clash = extra_clash(db, li, tax)
    if clash and not adopt_extras:
        refusal = _extras_refusal(desktop, clash)
        if log:
            log(f"[import-logs] refused: {refusal}")
        return DesktopRun(desktop, str(path), "refused", reason=REFUSED_EXTRAS,
                          issues=[refusal])
    plan = _plan_drops(db, li, tax, previous, force, force_drop)
    if plan.refusal:
        if log:
            log(f"[import-logs] refused: {plan.refusal}")
        return DesktopRun(desktop, str(path), "refused",
                          reason=REFUSED_IMPLAUSIBLE, issues=[plan.refusal])
    filled, issues = _apply_index(li.steps, index.get(desktop))
    sheet_id = meta.get("desktop_id")
    if sheet_id is not None and int(sheet_id) != desktop:
        issues.append(
            f"D{desktop:02d}: the sheet's own Desktop ID is {sheet_id}; "
            f"kept the file's number"
        )
    (events, kept, dropped, fills, unresolved, implied,
     gone, drop_issues, adopted, replay, merges, captive, cage, reclass) = _write_import(
        db, li, tax, previous, plan, clash)
    extras_kept = extra_keys(db.instances(desktop))
    if dropped:
        issues.append(dropped_notes_line(desktop, dropped))
    corrections = ([replay.kept_line()] if replay.kept_line() else []) + \
        replay.unmatched_lines() + \
        merges.lines(skip={e["latch"] for e in cage.merged}) + reclass.lines() + \
        ([captive.kept_line()] if captive.kept_line() else []) + captive.unmatched_lines()
    issues.extend(drop_issues)
    if log:
        for text in adopted:
            log(f"[import-logs]   {text}")
        if extras_kept:
            log(f"[import-logs]   kept {len(extras_kept)} instance(s) added in S1 "
                f"(not in the sheet): {', '.join(extras_kept)}")
        outside = sorted(key for key, rec in db.instances(desktop).items()
                         if is_context(rec))
        if outside:
            # 框外零件 (task U18): not part of the machine, untouched by the sheet
            log(f"[import-logs]   kept {len(outside)} context object(s) outside the "
                f"machine: {', '.join(outside)}")
        extra = f", {len(implied)} implied instance(s)" if implied else ""
        log(f"[import-logs] D{desktop:02d}: {len(li.steps)} steps, {len(li.actions)} "
            f"actions, {len(li.instances)} instances, {events} events, "
            f"{filled} durations, {len(fills)} inferred relational fields, "
            f"{len(li.issues) + len(issues)} issues{extra}")
        for text in implied + cage.lines():
            log(f"[import-logs]   {text}")
        for text in gone:
            log(f"[import-logs]   {text}")
        for text in drop_issues:
            log(f"[import-logs]   {text}")
        if dropped:
            log(f"[import-logs]   {dropped_notes_line(desktop, dropped)}")
        for text in corrections:
            log(f"[import-logs]   {text}")
    return DesktopRun(
        desktop=desktop, source=str(path), status="imported", steps=len(li.steps),
        actions=len(li.actions), instances=len(li.instances), events=events,
        durations=filled, ls_notes=kept, ls_notes_dropped=dropped,
        brand=str(li.meta.get("brand_model_raw") or ""),
        issues=list(li.issues) + issues, fills=fills, unresolved=unresolved,
        implied=implied, dropped=gone, kept_for_review=len(plan.holding),
        kept_carrying_work=plan.carrying_work, extras_kept=extras_kept, adopted=adopted,
        corrections=corrections, cage_latches=cage.lines(),
    )


def expected_steps(directory: str) -> dict[int, int]:
    """``desktop -> n_logged_steps`` from the export's summary table, if present."""
    return _expected_steps(Path(directory) / "desktop_meta.csv")
