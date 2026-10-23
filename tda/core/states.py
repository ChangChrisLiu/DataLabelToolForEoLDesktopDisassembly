"""State machine and geometry policy -- steps 1-2 of the compiler (spec 3.3).

The whole module is pure: it turns the recorded action list into an explicit
event log, folds that log into a per-step snapshot, and answers which instances
need geometry at a step.

* :func:`initial_state`      -- every instance at its class default, in the chassis.
* :func:`events_from_actions`-- actions (spec 6.3) -> :class:`~tda.core.model.StateEvent` log.
* :func:`state_at`           -- initial state + every event with ``step <= k``.
* :func:`needs_geom`         -- which instances need a mask or a bench box at a step.
* :func:`diff_states`        -- what changed between two snapshots.
* :func:`validate_events`    -- the legal-transition checks of spec 6.2.

Conventions
-----------
Every instance starts ``in_chassis`` (the chassis itself included). An action
whose ``result`` is not ``"success"`` produces no events at all (spec 6.3), and
an action whose verb does not apply to the target class -- or whose effect is
already the current value -- produces no event either.

Whenever an instance reaches ``removed``, it also leaves the chassis
(``placement: in_chassis -> on_bench``) and drags its attached children with it.
A ``connector`` is the one exception: removing it means the cable was taken out
as a whole, which is a state change only, because an unplugged/removed
connector carries no geometry of its own (spec 6.2). That cascade is enforced
twice over: :func:`events_from_actions` materialises it as explicit events, and
:func:`state_at` closes over it again, so a hand-entered (``auto=False``) parent
removal never leaves an attached child behind asking for a mask. The child is
then *inside* its parent rather than beside it, so :func:`needs_geom` stops
asking for geometry of its own as well (:func:`gone_with_parent`).

Virtual nodes are targets starting with ``cable:`` (or, for actions, anything
absent from ``instances``). They have their own small state machine and are
carried in a :data:`FrameState` once an event mentions them, but they never
carry geometry, because :func:`needs_geom` only answers for real instances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from tda.core.model import (
    ActionRec,
    InstanceRec,
    Placement,
    StateEvent,
    is_cage_latch,
    is_context,
    is_provisional,
)
from tda.core.taxonomy import Taxonomy

__all__ = [
    "AFTER_LAST_STEP",
    "ANY_VIEW",
    "CAGE_LATCH_EVIDENCE",
    "GALLERY_EXPORT_STATE",
    "GALLERY_PLACEMENT",
    "INITIAL_STEP",
    "InstState",
    "FrameState",
    "EventProblem",
    "cage_latch_opens",
    "diff_states",
    "event_problems",
    "events_from_actions",
    "gallery_state",
    "gone_with_parent",
    "initial_overrides",
    "is_cage_latch_event",
    "is_gallery_part",
    "initial_state",
    "is_virtual_target",
    "needs_geom",
    "removal_steps",
    "state_at",
    "validate_events",
]

ATTR_STATE = "state"
ATTR_PLACEMENT = "placement"
REMOVED = "removed"
#: The logical step a hand-written *initial* state event is recorded at: before
#: step 1, so it is folded into every snapshot and is never "the change" of any
#: frame (task U5b -- an instance added in S1 whose picture shows it ``open``).
INITIAL_STEP = 0
CABLE_PREFIX = "cable:"
CABLE_CLASS = "cable"
CONNECTOR_CLASS = "connector"
IN_CHASSIS = Placement.IN_CHASSIS.value
ON_BENCH = Placement.ON_BENCH.value
_PLACEMENTS = frozenset(p.value for p in Placement)
_SUCCESS = "success"
#: ``StateEvent.evidence_view`` of the one derived event no action stands
#: behind: a drive cage's own latch (:func:`tda.core.model.is_cage_latch`, task
#: U20) opens on the frame before its cage comes out (:func:`cage_latch_opens`).
#: It is a default, not a record of what was done, so it gives way to a manual
#: state: :func:`event_problems` does not hold it against a latch the annotator
#: already said was open (``T`` / ``Ctrl+T`` earlier), and a ``Ctrl+T`` span may
#: run across it (:mod:`tda.core.state_assert`).
CAGE_LATCH_EVIDENCE = "rule:drive_cage_latch"
#: The verb whose effect that event has.
_OPEN = "open"


def is_cage_latch_event(event: StateEvent) -> bool:
    """Is ``event`` the derived "the cage's latch is open here" default?"""
    return bool(event.auto) and event.evidence_view == CAGE_LATCH_EVIDENCE


class _AnyView:
    """The default of :func:`needs_geom`'s ``bench_roi``: no view in mind.

    ``None`` already means something -- *this view has no staging area* -- so
    "not asked about a view at all" needs a value of its own rather than a
    second flag nobody would remember to pass.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return "ANY_VIEW"


#: Sentinel for :func:`needs_geom`: ask the geometry policy with no view in mind.
ANY_VIEW = _AnyView()


@dataclass
class InstState:
    """One instance's state and placement at one logical step.

    ``left_with`` is provenance rather than state: the key of the parent whose
    removal dragged this instance out (:func:`_close_attached_cascade`), and
    ``None`` for everything that got where it is on its own -- including a
    child that was removed by its own action *before* its parent followed. It
    is what :func:`gone_with_parent` reads, and it is deliberately excluded
    from equality, so two snapshots still compare on what they *are*.
    """

    state: str
    placement: str
    left_with: Optional[str] = field(default=None, compare=False)
    #: A part of the components gallery (task U10, :func:`gallery_state`): it
    #: is drawn in a layout photo whatever its state and placement say.
    gallery: bool = field(default=False, compare=False)


#: instance key -> its state at one logical step
FrameState = dict[str, InstState]


def is_virtual_target(target: str, instances: dict[str, InstanceRec]) -> bool:
    """Is this action target a virtual node rather than a real instance?"""
    return target.startswith(CABLE_PREFIX) or target not in instances


def _virtual_class(target: str) -> str | None:
    """The virtual-node class of ``target``, or ``None`` when it has none."""
    return CABLE_CLASS if target.startswith(CABLE_PREFIX) else None


# --------------------------------------------------------------------------- #
# 1. state
# --------------------------------------------------------------------------- #
def initial_state(
    instances: dict[str, InstanceRec],
    tax: Taxonomy,
    initial: Optional[Mapping[str, str]] = None,
) -> FrameState:
    """The state before step 1: class defaults, everything still in the chassis.

    ``initial`` overrides the default of the instances it names -- the states
    :func:`initial_overrides` reads off the hand-written step-0 events.

    A 框外零件 (:func:`tda.core.model.is_context`, task U18) has no state at
    all: it is not part of the machine, so no snapshot carries it -- and with
    that no geometry policy, task card, cascade or validation ever sees it.
    """
    over = initial or {}
    return {
        key: InstState(state=over.get(key) or tax.default_state(rec.cls),
                       placement=IN_CHASSIS)
        for key, rec in instances.items()
        if not is_context(rec)
    }


def initial_overrides(events: Iterable[StateEvent]) -> dict[str, str]:
    """``key -> state`` a hand-written event sets before step 1.

    An instance an annotator added in S1 (:mod:`tda.core.extra`) has no action,
    so the only way it can start anywhere but its class default -- a RAM clip
    the picture shows open -- is a manual ``state`` event at
    :data:`INITIAL_STEP`. :func:`state_at` folds that event like any other;
    this is what :func:`events_from_actions` needs so the events it *derives*
    later (the cascade when the host leaves) start from the same value, and
    :func:`validate_events` does not report ``old`` as wrong.
    """
    out: dict[str, str] = {}
    for event in sorted(events, key=lambda e: e.step):
        if (not event.auto and event.step <= INITIAL_STEP
                and event.attr == ATTR_STATE and event.new):
            out[event.target] = str(event.new)
    return out


def _attached_children(instances: dict[str, InstanceRec]) -> dict[str, list[str]]:
    """parent key -> sorted keys of the children that leave the chassis with it."""
    children: dict[str, list[str]] = {}
    for key, rec in instances.items():
        if rec.attached and rec.parent:
            children.setdefault(rec.parent, []).append(key)
    for keys in children.values():
        keys.sort()
    return children


def _cascades_on_removal(rec: InstanceRec) -> bool:
    """Does removing this instance take its attached children out with it?

    Shared by :func:`events_from_actions` and :func:`state_at` so the event log
    and the closure over it can never disagree. A ``connector`` does not
    cascade: removing one means the cable went out as a whole (spec 6.3).
    """
    return rec.cls != CONNECTOR_CLASS


def removal_steps(
    instances: dict[str, InstanceRec],
    actions: Iterable[ActionRec],
    tax: Taxonomy,
    keys: Optional[Iterable[str]] = None,
) -> dict[str, int]:
    """``key -> the step of the first successful action that removes it``.

    Only the instance's **own** actions count (a ``remove``, or whatever verb
    the taxonomy makes ``removed`` for its class); one that never leaves is not
    in the result. ``keys`` limits the answer to those instances.
    """
    wanted = None if keys is None else set(keys)
    out: dict[str, int] = {}
    for action in sorted(actions, key=lambda a: (a.step, a.idx)):
        target = action.target
        if action.result != _SUCCESS or target in out:
            continue
        if wanted is not None and target not in wanted:
            continue
        rec = instances.get(target)
        if rec is None:
            continue
        if tax.apply_verb(rec.cls, rec.attrs, action.verb) == (ATTR_STATE, REMOVED):
            out[target] = int(action.step)
    return out


def cage_latch_opens(
    instances: dict[str, InstanceRec],
    actions: Iterable[ActionRec],
    tax: Taxonomy,
) -> list[tuple[int, str]]:
    """``(step, latch)`` of every drive cage's own latch that opens by default.

    A cage's own latch (:func:`tda.core.model.is_cage_latch`, task U20) is in no
    log, so nothing says when it was opened -- only that it must have been
    before the cage came out. The ruling: it is ``closed`` from the start and
    ``open`` on the frame before its cage's removal step ``c``, i.e. at
    ``c - 1``, and it leaves with the cage at ``c`` by the ordinary cascade. A
    cage the log never removes (or only displaces) keeps its latch closed, and
    so does a cage removed at step 1 (there is no frame before it).
    """
    latches = {key: rec for key, rec in instances.items()
               if is_cage_latch(rec) and rec.parent and rec.parent in instances}
    if not latches:
        return []
    removed = removal_steps(instances, actions, tax, {rec.parent for rec in latches.values()})
    out = []
    for key, rec in latches.items():
        step = removed.get(str(rec.parent))
        if step is not None and step - 1 >= 1:
            out.append((step - 1, key))
    return sorted(out)


def events_from_actions(
    instances: dict[str, InstanceRec],
    actions: list[ActionRec],
    tax: Taxonomy,
    initial: Optional[Mapping[str, str]] = None,
) -> list[StateEvent]:
    """Compile the recorded actions into a chronological state-event log.

    Actions are processed in ``(step, idx)`` order, so ``old`` on every event is
    the value that actually held when the action was performed. All events are
    ``auto=True``: they are derived, not typed in by an annotator.

    ``initial`` is :func:`initial_overrides` of the hand-written events: the
    state an instance with no action starts in when it is not its class
    default. Without it the cascade that takes an ``open`` added clip out with
    its board would be recorded as ``closed -> removed``.

    One derived event has no action behind it: a drive cage's own latch opens
    at the step before its cage comes out (:func:`cage_latch_opens`, task U20),
    after the actions of that step, marked :data:`CAGE_LATCH_EVIDENCE`. Nothing
    is emitted when the latch is already open or gone by then.
    """
    actions = list(actions)
    frame = initial_state(instances, tax, initial)
    cable_states: dict[str, str] = {}
    children = _attached_children(instances)
    events: list[StateEvent] = []
    defaults = cage_latch_opens(instances, actions, tax)

    def open_by_default(step: int, target: str) -> None:
        """The cage's latch is open here unless something said otherwise."""
        inst, rec = frame.get(target), instances[target]
        effect = tax.apply_verb(rec.cls, rec.attrs, _OPEN)
        if inst is None or effect is None or effect[0] != ATTR_STATE:
            return
        if inst.state in (REMOVED, effect[1]) or inst.placement != IN_CHASSIS:
            return
        events.append(StateEvent(desktop=rec.desktop, step=step, target=target,
                                 attr=ATTR_STATE, old=inst.state, new=effect[1],
                                 evidence_view=CAGE_LATCH_EVIDENCE, auto=True))
        inst.state = effect[1]

    def emit(action: ActionRec, target: str, attr: str, new: str) -> None:
        inst = frame[target]
        old = getattr(inst, attr)
        if old == new:
            return
        events.append(
            StateEvent(
                desktop=action.desktop,
                step=action.step,
                target=target,
                attr=attr,
                old=old,
                new=new,
                auto=True,
            )
        )
        setattr(inst, attr, new)

    def detach(action: ActionRec, target: str, seen: set[str]) -> None:
        """Move ``target`` to the bench, then its attached children with it.

        ``seen`` keeps a malformed parent cycle from recursing forever.
        """
        if frame[target].placement == IN_CHASSIS:
            emit(action, target, ATTR_PLACEMENT, ON_BENCH)
        for child in children.get(target, ()):
            if child in seen:
                continue
            seen.add(child)
            emit(action, child, ATTR_STATE, REMOVED)
            detach(action, child, seen)

    for action in sorted(actions, key=lambda a: (a.step, a.idx)):
        while defaults and defaults[0][0] < action.step:
            open_by_default(*defaults.pop(0))
        if action.result != _SUCCESS:
            continue
        if is_virtual_target(action.target, instances):
            cls = _virtual_class(action.target)
            if cls is None:
                continue
            effect = tax.apply_verb(cls, {}, action.verb)
            if effect is None or effect[0] != ATTR_STATE:
                continue
            old = cable_states.get(action.target, tax.default_state(cls))
            if old == effect[1]:
                continue
            events.append(
                StateEvent(
                    desktop=action.desktop,
                    step=action.step,
                    target=action.target,
                    attr=ATTR_STATE,
                    old=old,
                    new=effect[1],
                    auto=True,
                )
            )
            cable_states[action.target] = effect[1]
            continue

        rec = instances[action.target]
        if action.target not in frame:
            continue        # a 框外零件 (task U18): it has no state to change
        effect = tax.apply_verb(rec.cls, rec.attrs, action.verb)
        if effect is None:
            continue
        attr, new = effect
        emit(action, action.target, attr, new)
        if attr == ATTR_STATE and new == REMOVED and _cascades_on_removal(rec):
            detach(action, action.target, {action.target})
    for step, target in defaults:   # after the last action: nothing left to remove it
        open_by_default(step, target)
    return events


def _close_attached_cascade(
    instances: dict[str, InstanceRec],
    frame: FrameState,
    removed_at: Optional[dict[str, int]] = None,
) -> None:
    """Force the spec-3.3 cascade on ``frame`` in place, idempotently.

    Every ``attached`` child of an instance already in state ``removed`` is
    removed as well, transitively; a child still ``in_chassis`` moves to the
    bench (one already ``elsewhere`` stays there -- it is out of every view).
    Running this on a frame that already satisfies the cascade -- anything folded
    from an :func:`events_from_actions` log -- changes nothing about its state,
    but it does record on each claimed child *which* parent took it out
    (``InstState.left_with``), which no event can carry.

    ``removed_at`` maps an instance to the step it reached ``removed`` at, and
    is what keeps the claim honest: a child that left **before** its parent did
    -- its own ``remove`` action at step 5, the parent following at step 9 --
    is lying in the staging area in its own right and is not claimed. Without
    it every already-removed child is claimed, which is the right answer for a
    frame with no history to consult.

    This is the only place ``left_with`` is ever set, and it is gated on
    :func:`_cascades_on_removal`, so the cascade and everything that reads it
    (:func:`gone_with_parent`) cannot drift apart: a ``connector`` parent takes
    nothing with it here and therefore claims nothing there either.
    """
    children = _attached_children(instances)
    if not children:
        return
    steps = removed_at or {}

    def claims(parent: str, child: str) -> bool:
        """Did ``parent`` take ``child`` out, or had the child already gone?"""
        child_step, parent_step = steps.get(child), steps.get(parent)
        if child_step is None or parent_step is None:
            return True
        return child_step >= parent_step

    def walk(parent: str, seen: set[str]) -> None:
        for child in children.get(parent, ()):
            if child in seen:
                continue
            seen.add(child)
            inst = frame.get(child)
            if inst is None or not claims(parent, child):
                continue
            inst.state = REMOVED
            if inst.placement == IN_CHASSIS:
                inst.placement = ON_BENCH
            inst.left_with = parent
            walk(child, seen)

    for key in sorted(children):  # only instances that actually have children
        rec = instances.get(key)
        inst = frame.get(key)
        if rec is None or inst is None or inst.state != REMOVED:
            continue
        if _cascades_on_removal(rec):
            walk(key, {key})


def state_at(
    instances: dict[str, InstanceRec],
    events: list[StateEvent],
    step: int,
    tax: Taxonomy,
    initial: Optional[Mapping[str, str]] = None,
) -> FrameState:
    """The state of every instance at logical step ``step``.

    Initial state, plus every event with ``event.step <= step``, plus the
    attached-child closure of spec 3.3 step 1. :func:`events_from_actions`
    already emits that cascade as explicit events, so the closure is a no-op on
    a compiled log; it is what makes a hand-entered (``auto=False``) parent
    removal drag its attached children along too.

    A virtual ``cable:*`` node enters the snapshot as soon as an event mentions
    it (placement ``in_chassis``, it is nowhere else); it carries no geometry.
    Events on any other unknown key are skipped.

    The step each instance reached ``removed`` at is kept while folding and
    handed to the closure, which is how a child that came out on its own keeps
    its own identity instead of being counted as part of a parent that followed
    it later.

    ``initial`` is :func:`initial_overrides` for a caller whose ``events`` are
    the *derived* log alone (the spec 7.4 replay): the hand-written step-0
    events that carry an added part's initial state are not in it, so the
    snapshot has to start from them. A full log that contains them folds them
    either way.
    """
    frame = initial_state(instances, tax, initial)
    removed_at: dict[str, int] = {}
    for event in sorted(events, key=lambda e: e.step):  # stable: keeps intra-step order
        if event.step > step:
            break
        inst = frame.get(event.target)
        if inst is None:
            cls = _virtual_class(event.target)
            if cls is None:
                continue
            inst = InstState(state=tax.default_state(cls), placement=IN_CHASSIS)
            frame[event.target] = inst
        if event.attr == ATTR_STATE:
            if event.new == REMOVED and inst.state != REMOVED:
                removed_at[event.target] = event.step
            inst.state = event.new
        elif event.attr == ATTR_PLACEMENT:
            inst.placement = event.new
    _close_attached_cascade(instances, frame, removed_at)
    return frame


# --------------------------------------------------------------------------- #
# 2. geometry policy
# --------------------------------------------------------------------------- #
def gone_with_parent(fs: FrameState, key: str) -> bool:
    """Did this instance leave the chassis *inside* its parent?

    A captive cooler screw is ``attached`` to the cooler: once the cooler is
    out, the screw is *in* it, not lying beside it, so it is neither visible in
    the chassis nor a separate thing in the staging area (user decision C7,
    "子零件随父零件一起消失").

    The answer is read off the snapshot rather than recomputed: only
    :func:`_close_attached_cascade` ever sets ``left_with``, and it does so
    exactly for the children it took out -- transitively, never for a child of
    a ``connector`` (:func:`_cascades_on_removal`), and never for one that had
    already left on its own. The parent's own state is re-checked so a frame
    assembled by hand cannot claim a child of a part that is still in place.
    """
    inst = fs.get(key)
    if inst is None or not inst.left_with:
        return False
    parent = fs.get(inst.left_with)
    return parent is None or parent.state == REMOVED or parent.placement != IN_CHASSIS


def needs_geom(
    instances: dict[str, InstanceRec],
    fs: FrameState,
    tax: Taxonomy,
    *,
    bench_roi: object = ANY_VIEW,
) -> dict[str, str]:
    """Which instances need their own geometry in this frame state.

    **The one answer to "what has to be drawn here".** The truth table, the
    ``missing_shape`` queue, the task card, the default keyframe anchor and the
    canvas' "affects N frames" strip all ask this question, and they each used
    to compute their own version of it -- which is how a frame could be reported
    complete by one of them and missing a shape by another.

    ``"mask"`` for an instance still in the chassis whose state is tracked by
    the per-class table of spec 6.2, ``"box"`` for one lying in the bench area
    (the chassis excepted -- it never goes on the bench). Everything else is
    omitted: an unplugged connector, anything ``elsewhere`` or in some other
    placement, any key that is not a real instance (a ``cable:*`` node), and --
    see :func:`gone_with_parent` -- an attached child that went out inside its
    parent, which the annotator must not be asked to draw a second time.

    ``bench_roi`` is the staging area **this view** can see at this frame (spec
    3.3 step 2 / 4.2 item 1, 若该视角有堆放区 ROI). ``None`` means the view has
    none, so a part lying on the bench is not in its picture at all and is not
    its work; a rectangle means it is. Leaving the argument out asks the pure
    policy, with no view in mind, which is what a caller that has not resolved a
    frame yet wants -- the value is never read, only its presence.

    A provisional Label Studio key (:func:`tda.core.model.is_provisional`) is
    draft material rather than a part of the machine and is never asked for
    either: it has no actions, so it would otherwise stand installed in the
    chassis for ever and be a missing shape on every frame of its desktop.
    """
    blind = bench_roi is None
    geom: dict[str, str] = {}
    for key, inst in fs.items():
        rec = instances.get(key)
        if rec is None or is_provisional(key) or gone_with_parent(fs, key):
            continue
        if inst.gallery:
            # a part laid out for the components gallery (task U10) is a mask
            # wherever it is, and the snapshot only marks the ones to draw
            geom[key] = "mask"
            continue
        if inst.placement == IN_CHASSIS:
            kind = "mask"
        elif inst.placement == ON_BENCH:
            if blind:
                continue
            kind = "box"
        else:
            continue
        if tax.needs_mask(rec.cls, inst.state, inst.placement):
            geom[key] = kind
    return geom


# --------------------------------------------------------------------------- #
# the components gallery (task U10, spec 3.2b)
# --------------------------------------------------------------------------- #
#: A step after every step: :func:`state_at` folds the whole log up to it,
#: which is the state at the end of the teardown.
AFTER_LAST_STEP = 1 << 30
#: Where a gallery part is in the frame state: the **mask chain**. A part laid
#: out on the table is drawn like a part in the chassis -- a mask, layered with
#: its neighbours, and a missing one blocks the shot -- but it has no bench ROI
#: and no box; the exports write it as ``removed`` / ``on_bench``
#: (:data:`GALLERY_EXPORT_STATE`).
GALLERY_PLACEMENT = IN_CHASSIS
#: Where everything that is not in the gallery is: out of the picture.
ELSEWHERE = Placement.ELSEWHERE.value
#: ``(state, placement)`` the exports write for a gallery part.
GALLERY_EXPORT_STATE = (REMOVED, ON_BENCH)


def is_gallery_part(instances: dict[str, InstanceRec], final: FrameState, key: str,
                    excluded: Iterable[str] = ()) -> bool:
    """Is ``key`` one of the parts the components gallery asks for?

    Every real instance that is ``removed`` at the end of the teardown and
    left the chassis in its own right -- not inside a parent
    (:func:`gone_with_parent`: the clips on the board are *in* the board) --
    except the classes in ``excluded`` (:data:`tda.core.model.GALLERY_EXCLUDED_CLASSES`),
    which are not identifiable in a layout. A Label Studio draft never is.
    """
    rec = instances.get(key)
    inst = final.get(key)
    if rec is None or inst is None or is_provisional(key):
        return False
    if rec.cls in set(excluded) or inst.state != REMOVED:
        return False
    return not gone_with_parent(final, key)


def gallery_state(instances: dict[str, InstanceRec], events: list[StateEvent],
                  tax: Taxonomy, excluded: Iterable[str] = ()) -> FrameState:
    """The frame state of every components-gallery shot of one desktop.

    One answer for all of them, because a layout photo is not a moment of the
    teardown: the parts on the table are the ones removed by its **last** step.
    Each of them is marked ``gallery`` (:func:`needs_geom` asks for a mask of
    it) in :data:`GALLERY_PLACEMENT`, keeping its ``removed`` state; every other
    instance -- what stayed in the chassis, the excluded classes, the children
    that went out inside a parent -- is ``elsewhere``, out of the picture.
    """
    excluded = frozenset(excluded)
    final = state_at(instances, events, AFTER_LAST_STEP, tax)
    out: FrameState = {}
    for key, inst in final.items():
        if is_gallery_part(instances, final, key, excluded):
            out[key] = InstState(state=inst.state, placement=GALLERY_PLACEMENT,
                                 gallery=True)
        else:
            out[key] = InstState(state=inst.state, placement=ELSEWHERE)
    return out


def diff_states(a: FrameState, b: FrameState) -> list[tuple[str, str, str, str]]:
    """``(key, attr, old, new)`` for every difference, sorted by key.

    Keys missing from either side are ignored: the caller compares two
    snapshots of the same instance table.
    """
    rows: list[tuple[str, str, str, str]] = []
    for key in sorted(a.keys() & b.keys()):
        before, after = a[key], b[key]
        if before.state != after.state:
            rows.append((key, ATTR_STATE, before.state, after.state))
        if before.placement != after.placement:
            rows.append((key, ATTR_PLACEMENT, before.placement, after.placement))
    return rows


# --------------------------------------------------------------------------- #
# validation (spec 6.2 legal transitions)
# --------------------------------------------------------------------------- #
#: Codes of :class:`EventProblem`, one per rule of :func:`validate_events`.
LEAVES_REMOVED = "leaves_removed"
OLD_MISMATCH = "old_mismatch"
UNKNOWN_VALUE = "unknown_value"


@dataclass(frozen=True)
class EventProblem:
    """One illegal event, as data: :func:`validate_events` is its text.

    ``held`` is the value of ``event.attr`` the replay held when the event
    was applied; ``code`` is :data:`LEAVES_REMOVED`, :data:`OLD_MISMATCH` or
    :data:`UNKNOWN_VALUE`. A caller that has to *explain* a problem -- the
    state assertion of task U8, in the annotator's words -- reads these
    instead of parsing the English.
    """

    event: StateEvent
    held: str
    code: str
    text: str


def event_problems(
    instances: dict[str, InstanceRec],
    events: list[StateEvent],
    tax: Taxonomy,
) -> list[EventProblem]:
    """:func:`validate_events` as :class:`EventProblem` records, same order."""
    problems: list[EventProblem] = []
    current: dict[str, InstState] = initial_state(instances, tax)

    for event in sorted(events, key=lambda e: e.step):  # stable: keeps intra-step order
        rec = instances.get(event.target)
        if rec is not None and is_context(rec):
            continue        # a 框外零件 (task U18): no state, nothing to validate
        if rec is not None:
            cls = rec.cls
        else:
            virtual = _virtual_class(event.target)
            if virtual is None:
                continue
            cls = virtual
        inst = current.get(event.target)
        if inst is None:
            inst = InstState(state=tax.default_state(cls), placement=IN_CHASSIS)
            current[event.target] = inst
        if event.attr not in (ATTR_STATE, ATTR_PLACEMENT):
            continue

        held = getattr(inst, event.attr)
        reasons: list[tuple[str, str]] = []
        if event.attr == ATTR_STATE and held == REMOVED and event.new != REMOVED:
            reasons.append((LEAVES_REMOVED,
                            f"{REMOVED!r} is final, an instance cannot leave it"))
        # A removal is legal from any state but ``removed`` itself: the clip a
        # board takes out with it at 42 was recorded leaving ``open`` and is
        # ``closed`` once the annotator says so at 41 (task U8). Its ``old`` is
        # the log's account of the state, not a precondition of the removal.
        into_removed = (event.attr == ATTR_STATE and event.new == REMOVED
                        and held != REMOVED)
        # The cage latch's default "open here" (task U20) is no action: when the
        # annotator already said it was open earlier, it simply holds already.
        default_held = is_cage_latch_event(event) and held == event.new
        if event.old != held and not into_removed and not default_held:
            reasons.append((OLD_MISMATCH,
                            f"the current {event.attr} at that step is {held!r}"))
        if event.attr == ATTR_STATE:
            if event.new not in tax.states_of(cls):
                reasons.append((UNKNOWN_VALUE,
                                f"{event.new!r} is not a state of class {cls!r}"))
        elif event.new not in _PLACEMENTS:
            reasons.append((UNKNOWN_VALUE, f"{event.new!r} is not a known placement"))

        problems.extend(
            EventProblem(event, held, code,
                         f"{event.target}: {event.old} → {event.new} at step "
                         f"{event.step}: {reason}")
            for code, reason in reasons
        )
        setattr(inst, event.attr, event.new)
    return problems


def validate_events(
    instances: dict[str, InstanceRec],
    events: list[StateEvent],
    tax: Taxonomy,
) -> list[str]:
    """Report every illegal event as ``"<key>: <old> → <new> at step <k>: <reason>"``.

    Three things are flagged, in that order per event:

    1. leaving ``removed`` -- a removed part cannot come back (manual events may
       otherwise reverse a transition, e.g. a latch ``open -> closed``);
    2. an ``old`` value that does not match the value actually held at that step
       -- except on a move *into* ``removed``, which is legal from every state
       but ``removed`` itself (a part can be taken out open or closed; the
       recorded ``old`` is the log's account, not a precondition);
    3. a ``new`` value that is not a state of the target's class (or not a known
       placement).

    Events are replayed in ``(step, list)`` order and applied even when flagged,
    so later events are checked against the recorded trajectory. Targets that
    are neither instances nor virtual nodes are ignored, as are attributes other
    than ``state`` and ``placement``. :func:`event_problems` is the same check
    as records.
    """
    return [problem.text for problem in event_problems(instances, events, tax)]
