"""The constraint graph's vocabulary, and the rules that derive its edges.

The spec 7.2 semantics, the :class:`Edge` type, the verb-applicability tables
and the spec 7.3 attribute rules. :mod:`tda.core.graph` (the reasoning layer),
:mod:`tda.core.graph_plan` (the planner) and :mod:`tda.core.graph_templates`
(the family templates) all import from here, which is what keeps the modules
acyclic; :mod:`tda.core.graph` re-exports everything a caller needs.

One function matters here: :func:`propose_edges`, the spec 7.3 attribute rules
(~90% coverage). Everything in this module is pure.

Below it sits :mod:`tda.core.graph_infer`: which instance a name means, and the
heuristics that fill the relational fields those rules read
(:func:`infer_relational_fields`, :func:`unresolved_relations`). They are
re-exported here so the import path callers already use keeps working, and the
dependency runs one way only -- rules import inference, never the reverse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from tda.core.graph_infer import (
    AMBIGUOUS,
    LS_PREFIX,
    NO_CANDIDATE,
    SCREW_ROLE_CLASSES,
    UNRESOLVED,
    infer_relational_fields,
    is_provisional,
    real_instances as _real_instances,
    resolve_ref as _resolve,
    unique_of_class as _unique_of_class,
    unresolved_kind,
    unresolved_relations,
)
from tda.core.model import ActionRec, InstanceRec, is_extra
from tda.core.states import FrameState
from tda.core.taxonomy import Taxonomy

__all__ = [
    "AMBIGUOUS",
    "ACCEPTED",
    "BLOCKED_GATES",
    "BLOCKED_MODES",
    "BLOCKED_REQUIRED",
    "CABLE_DEFAULT_STATE",
    "CABLE_PREFIX",
    "CONTAINED_MODE",
    "LOOSE_MODE",
    "INACTIVE_STATUSES",
    "LOOSE_ATTR",
    "LS_PREFIX",
    "NON_STEPWISE_DESKTOPS",
    "STEPWISE_META",
    "STEPWISE_TASKS",
    "MOTHERBOARD_CLASS",
    "MOUNTED_MODE",
    "MOUNTED_ON_MOTHERBOARD",
    "NO_CANDIDATE",
    "ORPHAN_OF",
    "PROPOSED",
    "REJECTED",
    "SCREW_ROLE_CLASSES",
    "SETTLED_OF",
    "UNRESOLVED",
    "Edge",
    "GATED_VERBS",
    "GATES",
    "HARD_TYPES",
    "REQUIRED_STATES",
    "VerbTarget",
    "gated_verbs",
    "active_edges",
    "is_orphan",
    "blocker_state",
    "board_of",
    "cable_nodes",
    "cable_owner",
    "connector_owner",
    "cooler_on_cpu",
    "fan_leads_by_timing",
    "infer_relational_fields",
    "is_loose",
    "is_provisional",
    "is_stepwise",
    "propose_edges",
    "required_states",
    "unresolved_boards",
    "unresolved_coolers",
    "unresolved_fan_owners",
    "unresolved_kind",
    "unresolved_relations",
    "verb_applies",
    "verb_effect",
]

#: The five hard-constraint edge types of spec 7.2, most-specific first.
HARD_TYPES = ("fastened_by", "connected_to", "locked_by", "covered_by", "blocked_by")

#: Spec 7.2: the blocker must be in one of these states before the target of an
#: edge of this type may be acted on.
#:
#: ``covered_by`` is also satisfied by ``unplugged`` (task W3j, spec v1.55): only
#: a connector has that state, and an unplugged connector has no geometry --
#: ``configs/taxonomy.yaml`` ``connector.needs_mask: {unplugged: false}``, spec
#: §6.2 「拔掉的接头不标」 -- so it is not in the picture and covers nothing.
#: Without it the log replay called a connector unplugged at step 3 "still in
#: the way" of a heatsink removed at step 19, and 18 Review questions were
#: wrong (the user, 2026-11-01: 「第19步的时候这个connector一定已经不在了」,
#: 「这个connector在第12步已经被移走了，为啥你说还没拿走呢？」).
#: ``tests/vlm_checker.py`` transcribes this table by hand and must say the same.
REQUIRED_STATES: dict[str, frozenset[str]] = {
    "fastened_by": frozenset({"loosened", "removed"}),
    "connected_to": frozenset({"unplugged", "removed"}),
    "locked_by": frozenset({"open"}),
    "covered_by": frozenset({"open", "removed", "displaced", "unplugged"}),
    "blocked_by": frozenset({"removed", "displaced", "released"}),
}

#: The ``blocked_by`` mode of the rule-derived "this part is mounted on that
#: board" edge (task W3d).
MOUNTED_MODE = "mounted"
#: The ``blocked_by`` mode of "the blocker is a load the part carries" (task
#: W3g): a drive in the drive cage, the optical drive on top of it.
CONTAINED_MODE = "contained"
#: The ``blocked_by`` mode of a **moment edge** (never stored): a part lying loose in the
#: machine (R16, :data:`LOOSE_ATTR`) holds back every action outside its own way out until
#: it is out (:func:`tda.core.graph.moment_edges`). It gates every verb and only
#: ``removed`` clears it.
LOOSE_MODE = "loose"
#: ``blocked_by`` is the only type that carries a mode (spec 7.1).
#: ``physical_path`` stays first: it is the Relations panel's default.
BLOCKED_MODES = ("physical_path", "tool_access", "cable_tension", MOUNTED_MODE,
                 CONTAINED_MODE)

#: Verbs a hard constraint can block at all (spec 7.2). ``reorient`` is a
#: capture action on the chassis, not a disassembly step, so nothing gates it.
GATED_VERBS = frozenset({"remove", "displace", "open", "unscrew", "disconnect", "release"})

#: Which of those verbs each edge type actually gates. One table, read by
#: :func:`tda.core.graph.applicable_preconditions` (and so by ``legal_actions``
#: and ``validate_sequence``) *and* by :mod:`tda.core.graph_plan`, so the
#: checker and the planner cannot drift apart.
#:
#: Every type used to gate every verb, which said a slim-case PSU could not be
#: swung out of the way until its whole harness was unplugged -- backwards, as
#: swinging it out is how you reach the plugs. Across the 66 sheets that alone
#: produced 34 ``displace psu.01 violates connected_to(...)`` lines, every one
#: of them describing correct work. The distinction is *moving the part* versus
#: *reaching it*:
#:
#: * ``fastened_by`` / ``locked_by`` -- a screwed-down or latched part does not
#:   move, but you can still work on what is plugged into it;
#: * ``covered_by`` / ``blocked_by`` -- no access at all, so every verb waits
#:   (``blocked_by`` by its ``mode``, as before);
#: * ``connected_to`` -- you may displace, open or unscrew a part whose cables
#:   are still plugged; you may not take it away.
GATES: dict[str, frozenset[str]] = {
    "fastened_by": frozenset({"remove", "displace", "open"}),
    "locked_by": frozenset({"remove", "displace", "open"}),
    "covered_by": GATED_VERBS,
    "blocked_by": GATED_VERBS,
    "connected_to": frozenset({"remove"}),
}


#: What each ``blocked_by`` mode actually stops (spec 7.1). The mode was stored,
#: digested and never read, so every manual ``blocked_by`` gated every verb --
#: the same over-binding the ``connected_to`` row of :data:`GATES` removed. The
#: distinction is again *moving the part* versus *reaching it*:
#:
#: * ``cable_tension`` -- a cable under tension lets the part swing aside, it
#:   only stops it leaving the machine, exactly like ``connected_to``;
#: * ``physical_path`` -- the blocker is in the part's way, so it can neither be
#:   taken out nor moved, but its own screws and plugs are still reachable;
#: * ``tool_access`` -- you cannot get at it at all, so every gated verb waits;
#: * ``mounted`` (task W3d) -- the blocker is mounted on the part (a CPU, RAM
#:   stick, cooler or card on the board): as ``physical_path``, the board can
#:   neither be taken out nor moved, but its own screws and plugs stay reachable;
#: * ``contained`` (task W3g) -- the blocker is a load the part carries (the
#:   drives **in** the drive cage, the optical drive **on** it, the user's
#:   rulings B1 / B2 / B5): the cage may still be swung aside and worked on with
#:   them inside -- D30's log does exactly that at step 9 -- but it may not leave
#:   the machine until they are out, so only ``remove`` waits.
BLOCKED_GATES: dict[str, frozenset[str]] = {
    "cable_tension": frozenset({"remove"}),
    "physical_path": frozenset({"remove", "displace"}),
    "tool_access": GATED_VERBS,
    "mounted": frozenset({"remove", "displace"}),
    "contained": frozenset({"remove"}),
    LOOSE_MODE: GATED_VERBS,
}

#: ``blocked_by`` modes whose blocker must reach a narrower set of states than
#: :data:`REQUIRED_STATES` gives the type (task W3d). A part mounted on the
#: board has to come **off**: a RAM stick swung aside is still in its slot, so
#: ``displaced`` does not free the board (the user's 「拆掉了」; in the 60
#: board-removing logs no CPU, RAM stick or card is ever displaced). A drive
#: the cage carries (``contained``, task W3g) likewise: a displaced drive is
#: still in the cage, so only ``removed`` lets the cage leave.
BLOCKED_REQUIRED: dict[str, frozenset[str]] = {
    "mounted": frozenset({"removed"}),
    "contained": frozenset({"removed"}),
    LOOSE_MODE: frozenset({"removed"}),
}


def required_states(edge_type: str, mode: Optional[str] = None) -> frozenset[str]:
    """The states the blocker of an edge of this type (and ``blocked_by`` mode) must be in.

    The one accessor every reader of the required states goes through (task
    W3d): a ``blocked_by`` whose mode has a :data:`BLOCKED_REQUIRED` row gets
    that row, everything else :data:`REQUIRED_STATES` by type (empty for an
    unknown type).
    """
    if edge_type == "blocked_by" and mode in BLOCKED_REQUIRED:
        return BLOCKED_REQUIRED[mode]
    return REQUIRED_STATES.get(edge_type, frozenset())


def gated_verbs(edge_type: str, mode: Optional[str] = None) -> frozenset[str]:
    """Which verbs an edge of this type (and ``blocked_by`` mode) gates.

    An unknown type, and a ``blocked_by`` whose mode is missing or unknown, gate
    them all: gating too much is noisy and visible, gating too little silently
    drops a constraint out of the ground truth, so both err the loud way.
    """
    if edge_type == "blocked_by":
        return BLOCKED_GATES.get(mode or "", GATED_VERBS)
    return GATES.get(edge_type, GATED_VERBS)

CABLE_PREFIX = "cable:"
REMOVED = "removed"
PROPOSED = "proposed"
ACCEPTED = "accepted"
REJECTED = "rejected"

#: The status a human decision falls back to while the rules no longer derive
#: the edge it was about (spec 7.3 gives no word for it, so the two are spelled
#: out rather than encoded in another column -- no migration). See
#: :mod:`tda.core.graph_derive`.
ORPHAN_OF = {ACCEPTED: "accepted_orphan", REJECTED: "rejected_orphan"}
SETTLED_OF = {orphan: settled for settled, orphan in ORPHAN_OF.items()}

#: Statuses that gate nothing: a human said no, or the rule that justified the
#: decision is no longer derived, so the decision is about nothing.
INACTIVE_STATUSES = frozenset({REJECTED, *ORPHAN_OF.values()})

#: The default state of a ``cable:*`` node, mirroring ``taxonomy.yaml``. A
#: cable enters the frame state only once an event names it (see
#: :func:`tda.core.states.state_at`), so an edge may well point at one that is
#: not in the snapshot yet; it is still routed until something releases it.
CABLE_DEFAULT_STATE = "routed"

#: Which states a verb may be applied *from*, where the taxonomy's effect table
#: alone is too permissive. ``remove`` is handled by :data:`REMOVE_FROM_STATES`.
VERB_FROM_STATES: dict[str, frozenset[str]] = {
    "unscrew": frozenset({"fastened"}),
    "disconnect": frozenset({"plugged"}),
    "open": frozenset({"closed"}),
    "release": frozenset({"closed", "routed"}),
    "displace": frozenset({"installed"}),
}

#: Classes that need an intermediate step before ``remove``: a screw has to come
#: loose first. Every other class can be taken straight out of whatever state it
#: is in -- including a ``connector``, which spec 6.3 lets you remove while it
#: is still plugged, meaning the whole cable was pulled out in one go.
REMOVE_FROM_STATES: dict[str, frozenset[str]] = {
    "screw": frozenset({"loosened"}),
}

#: One action as the graph talks about it: ``(verb, target)``.
VerbTarget = tuple[str, str]


@dataclass(frozen=True)
class Edge:
    """One hard constraint: ``blocker`` gates every action on ``target``.

    ``necessity`` is ``required`` or ``recommended``; ``mode`` is only set on
    ``blocked_by``; ``source`` is ``rule`` / ``template`` / ``manual``;
    ``status`` is ``proposed`` / ``accepted`` / ``rejected`` (spec 7.3).
    """

    type: str
    target: str
    blocker: str
    necessity: str = "required"
    mode: Optional[str] = None
    reason: str = ""
    source: str = "rule"
    evidence_step: Optional[int] = None
    status: str = "proposed"

    def label(self) -> str:
        """``fastened_by(motherboard.01, screw.motherboard.03)``."""
        return f"{self.type}({self.target}, {self.blocker})"


# --------------------------------------------------------------------------- #
# shared helpers: edge status, verb applicability, cable nodes
# --------------------------------------------------------------------------- #
def active_edges(edges: Iterable["Edge"]) -> list["Edge"]:
    """Edges that still gate something (spec 7.3: proposed / accepted / rejected).

    Out go the ones a human rejected and the **orphaned decisions**: a decision
    about a rule edge the rules no longer derive gates nothing, because the edge
    it was about is gone (:data:`INACTIVE_STATUSES`).
    """
    return [e for e in edges if e.status not in INACTIVE_STATUSES]


def is_orphan(edge: "Edge") -> bool:
    """Is this a decision whose rule edge is no longer derived?"""
    return edge.status in SETTLED_OF


#: Instance ``attrs`` flag of a part that was lying loose in the machine from
#: the start -- fallen out, not fixed to anything (the owner's R16, ruling of
#: 2026-11-04 afternoon: D43 D60 D65 case fan, D54 SSD and case fan, D45 HDD,
#: D16 SSD, and no others). Data, set by hand with an op record, never
#: inferred. While one is present it has the highest priority: only the actions
#: that take the loose parts out are legal
#: (:func:`tda.core.graph.legal_actions`, ``loose_first``).
LOOSE_ATTR = "loose"


def is_loose(rec: Optional[InstanceRec]) -> bool:
    """Was this part lying loose in the machine from the start (R16)?"""
    return rec is not None and bool((rec.attrs or {}).get(LOOSE_ATTR))


#: Desktop-meta flag of a desktop whose log is **not stepwise**: several parts
#: came out at once, so neither the general rules nor any step-ordered question
#: apply to it (the owner's ruling on D66, 2026-11-04). Its annotations are
#: left as they are; the VLM export skips :data:`STEPWISE_TASKS` there.
STEPWISE_META = "stepwise"
#: The tasks that read the order of the steps (next action, feasibility, plan,
#: plan check) and so are not asked on a non-stepwise desktop.
STEPWISE_TASKS = ("V4", "V6", "V7", "V16")
#: The desktops the owner ruled non-stepwise; the meta flag says the same and
#: is what a reader of the database goes by.
NON_STEPWISE_DESKTOPS = frozenset({66})


def is_stepwise(desktop: int, meta: Optional[dict] = None) -> bool:
    """May step-ordered questions be asked on this desktop? (D66: no.)

    ``meta`` is the desktop's meta row; its ``stepwise: false`` decides, and
    :data:`NON_STEPWISE_DESKTOPS` answers when there is no row to read.
    """
    if meta is not None and STEPWISE_META in meta:
        return meta.get(STEPWISE_META) is not False
    return int(desktop) not in NON_STEPWISE_DESKTOPS


def verb_effect(tax: Taxonomy, cls: str, attrs: dict, verb: str) -> Optional[str]:
    """The state this verb would put the class in, or ``None`` for no effect."""
    effect = tax.apply_verb(cls, attrs, verb)
    if effect is None or effect[0] != "state":
        return None
    return effect[1]


def verb_applies(tax: Taxonomy, cls: str, attrs: dict, verb: str, current: str) -> bool:
    """Can this verb be performed on an instance of ``cls`` in state ``current``?

    Independent of the constraint graph: it only asks whether the verb belongs
    to the class (spec 6.3), does something, and starts from a sane state.
    """
    new = verb_effect(tax, cls, attrs, verb)
    if new is None or current == REMOVED or new == current:
        return False
    allowed = REMOVE_FROM_STATES.get(cls) if verb == "remove" else VERB_FROM_STATES.get(verb)
    return allowed is None or current in allowed


def blocker_state(state: FrameState, blocker: str) -> Optional[str]:
    """A node's state, or ``None`` when it is not a node of this desktop."""
    inst = state.get(blocker)
    if inst is not None:
        return inst.state
    if blocker.startswith(CABLE_PREFIX):
        return CABLE_DEFAULT_STATE
    return None


def cable_nodes(edges: list["Edge"], state: FrameState) -> dict[str, str]:
    """Every virtual cable node this graph mentions, with its current state."""
    keys = {e.blocker for e in edges if e.blocker.startswith(CABLE_PREFIX)}
    keys |= {k for k in state if k.startswith(CABLE_PREFIX)}
    return {k: blocker_state(state, k) or CABLE_DEFAULT_STATE for k in keys}


# --------------------------------------------------------------------------- #
# cables
# --------------------------------------------------------------------------- #
#: ``cable:<owner>`` tokens the logs use that are not taxonomy class names.
CABLE_OWNER_ALIASES = {
    "cpu_fan": "cpu_cooler",
    "cooler_fan": "cpu_cooler",
    "hdd": "storage_drive",
    "ssd": "storage_drive",
    "drive": "storage_drive",
    "optical": "optical_drive",
}

#: Classes whose cables are *not* captive to them, however ``logs.py`` groups
#: them. A drive's SATA leads unplug at both ends, so spec 7.2 says each plug
#: gates only the part it sits on -- unlike a fan lead or a front-panel loom,
#: which are moulded into their part and gate it wherever they plug in.
DETACHABLE_CABLE_OWNERS = frozenset({"storage_drive", "optical_drive"})

#: Connector kinds that are power leads: whatever ``taxonomy_map`` tagged them
#: with, the far end of one is moulded into the PSU, so the PSU harness owns it
#: (spec 7.2: every plug of the harness must be out before the PSU can go).
POWER_LEAD_KINDS = frozenset({"atx_24pin", "cpu_power", "sata_power", "molex"})

#: Connector kinds with a detachable plug at *both* ends, owned by nobody.
TWO_ENDED_KINDS = frozenset({"sata_data"})
#: Instance ``attrs`` flag saying this one connector's cable has a detachable
#: plug at both ends whatever its kind (task W3l): the HP SFF drive power lead
#: runs from a board header (its board end is the sheet's 「Optical drive -
#: motherboard connector」), so the drive-end ``sata_power`` plug is nobody's
#: harness and gates only its own socket host.  Set by the annotator (or the
#: controller) on the instance, never inferred.
TWO_ENDED_ATTR = "two_ended"

#: The class a fan lead can be captive to.
COOLER_CLASS = "cpu_cooler"

#: ``cpu_cooler.kind`` values that actually carry a fan. ``taxonomy.yaml`` lists
#: ``fan`` / ``heatsink`` / ``heatsink_fan``, and a bare ``heatsink`` is a block
#: of metal: nothing is moulded into it, so no fan lead can belong to it.
FAN_CAPABLE_COOLER_KINDS = frozenset({"fan", "heatsink_fan"})


def cable_owner(
    cable_id: str,
    instances: Optional[dict[str, InstanceRec]] = None,
) -> Optional[str]:
    """The part a virtual cable node belongs to (spec 3.1 ``cable.owner``).

    Without ``instances`` this answers the owner *class* (``"cable:psu"`` ->
    ``"psu"``); with them it resolves that class to its instance key, but only
    when the desktop has exactly one instance of it -- an ambiguous or absent
    owner yields ``None`` rather than a guess. A key that is not a cable node
    yields ``None`` either way.

    Note this is the *cable's* owner, in the spec-7.1 sense of "the cable is
    permanently attached to this part", which is narrower than the grouping key
    ``logs.py`` writes: see :data:`DETACHABLE_CABLE_OWNERS`.
    """
    if not cable_id.startswith(CABLE_PREFIX):
        return None
    owner = cable_id[len(CABLE_PREFIX):].strip()
    if not owner:
        return None
    owner = CABLE_OWNER_ALIASES.get(owner, owner)
    if owner in DETACHABLE_CABLE_OWNERS:
        return None
    if instances is None:
        return owner
    return _unique_of_class(instances, owner)


def connector_owner(
    rec: InstanceRec,
    instances: dict[str, InstanceRec],
) -> Optional[str]:
    """The part one connector's cable is captive to, or ``None``.

    The connector's ``kind`` decides first, because it is the physical fact and
    ``logs.py``'s ``cable_owner`` is only a grouping key:

    * a power lead (:data:`POWER_LEAD_KINDS`) belongs to the PSU harness, even
      when the sheet filed it under the drive it feeds -- so the PSU cannot come
      out until every one of its plugs is pulled;
    * a ``sata_data`` lead has a detachable plug at both ends and belongs to
      nobody, so each end gates only its own socket host (spec 7.2);
    * anything else falls back to the tagged cable owner, which still drops the
      drive classes of :data:`DETACHABLE_CABLE_OWNERS` -- an untyped connector
      filed under a drive is one of its SATA leads -- and, for a fan lead, a
      cooler that cannot carry a fan (see :func:`owns_fan_lead`).

    ``attrs[TWO_ENDED_ATTR]`` (task W3l) overrides the kind: the annotator says
    this lead is detachable at both ends (an HP drive power lead off a board
    header), so it belongs to nobody -- the PSU may come out with it plugged.
    """
    if rec.attrs.get(TWO_ENDED_ATTR):
        return None
    kind = str(rec.attrs.get("kind") or "")
    if kind in POWER_LEAD_KINDS:
        return _unique_of_class(instances, "psu")
    if kind in TWO_ENDED_KINDS:
        return None
    owner = cable_owner(rec.cable or "", instances)
    if owner and not owns_fan_lead(instances.get(owner)):
        return None
    return owner


def owns_fan_lead(owner: Optional[InstanceRec]) -> bool:
    """Can this instance be the far end of a cooler fan lead?

    Anything that is not a cooler at all is none of this rule's business and
    answers ``True``; a cooler answers whether its ``kind`` has a fan
    (:data:`FAN_CAPABLE_COOLER_KINDS`).

    ``cable:cpu_fan`` used to resolve onto "the one ``cpu_cooler`` of this
    desktop" and stop there. On twelve machines that one instance is a bare
    ``kind=heatsink`` -- the sheet operated the heatsink and never the fan, so
    the fan was never instantiated -- and the rule bound the fan lead to a part
    with no fan, producing fifteen ``connected_to`` edges that said the heatsink
    could not come out until a connector that is not on it was unplugged, and
    eleven of the thirty sequence violations across the dataset. A missing
    instance is a question for stage S1 (:func:`unresolved_fan_owners`), not
    something to answer with the nearest part of roughly the right class.
    """
    if owner is None or owner.cls != COOLER_CLASS:
        return True
    return str(owner.attrs.get("kind") or "") in FAN_CAPABLE_COOLER_KINDS


#: Connector ``kind`` of a fan lead (``taxonomy.yaml`` ``connector.kind``).
FAN_LEAD_KIND = "fan"
#: Verbs that take a plug out of its socket.
UNPLUG_VERBS = frozenset({"disconnect", "remove"})
#: Classes a fan lead can be moulded into.
FAN_OWNER_CLASSES = frozenset({COOLER_CLASS, "case_fan"})


def _is_fan_lead(rec: InstanceRec) -> bool:
    """A fan-type connector: ``kind = fan``, or filed under a fan's cable."""
    if rec.cls != "connector":
        return False
    if str(rec.attrs.get("kind") or "") == FAN_LEAD_KIND:
        return True
    owner = cable_owner(rec.cable or "")
    return owner in FAN_OWNER_CLASSES


def fan_leads_by_timing(
    instances: dict[str, InstanceRec],
    actions: Iterable[ActionRec],
) -> dict[str, set[str]]:
    """``fan cooler -> the fan leads the log unplugs right before it comes out`` (R1).

    The owner's ruling of 2026-11-04: which part a fan lead belongs to is read
    off **when it is unplugged**. A fan-type connector unplugged in the step(s)
    right before the CPU fan is removed is that fan's lead; one still plugged
    after the fan is out is a case fan's lead and constrains only its socket
    host (the motherboard), which is what the connector rule already says.

    For each settled ``cpu_cooler`` that carries a fan
    (:data:`FAN_CAPABLE_COOLER_KINDS` -- the ``fan`` part of a two-part cooler,
    or a one-part fan cooler; a bare ``heatsink`` owns no lead, so on a
    heatsink-only machine every fan lead is a case fan's) whose removal the
    log records, the **window** is the removal step itself (plugs pulled
    before it inside that step) plus the run of log steps immediately before
    it in which a fan lead is pulled or the fan itself is moved aside. Every
    fan lead whose first unplug falls in the window belongs to that fan.

    A cooler the log never removes gets no entry: the caller then falls back
    to the cable tag (:func:`connector_owner`). Pure; sorted inputs, so the
    answer does not depend on the order the actions were read in.
    """
    ordered = sorted((a for a in actions if a.result == "success"),
                     key=lambda a: (a.step, a.idx))
    steps = sorted({a.step for a in ordered})
    by_step: dict[int, list[ActionRec]] = {}
    for a in ordered:
        by_step.setdefault(a.step, []).append(a)
    leads = {key for key, rec in instances.items()
             if not is_provisional(key) and _is_fan_lead(rec)}
    first_unplug: dict[str, ActionRec] = {}
    for a in ordered:
        if a.target in leads and a.verb in UNPLUG_VERBS:
            first_unplug.setdefault(a.target, a)

    out: dict[str, set[str]] = {}
    for fan in _real_instances(instances, COOLER_CLASS):
        if not owns_fan_lead(instances[fan]):
            continue
        removal = next((a for a in ordered if a.target == fan and a.verb == "remove"), None)
        if removal is None:
            continue
        window = {removal.step}
        pos = steps.index(removal.step)
        for step in reversed(steps[:pos]):
            acts = by_step[step]
            pulls = any(a.target in leads and a.verb in UNPLUG_VERBS for a in acts)
            moves_fan = all(a.target == fan for a in acts)
            if not (pulls or moves_fan):
                break
            window.add(step)
        mine = set()
        for lead, a in first_unplug.items():
            if a.step not in window:
                continue
            if a.step == removal.step and a.idx > removal.idx:
                continue  # pulled after the fan was already out
            mine.add(lead)
        out[fan] = mine
    return out


def unresolved_fan_owners(instances: dict[str, InstanceRec]) -> list[str]:
    """One line per fan lead whose cooler cannot carry a fan, for stage S1.

    The connector keeps its ``socket_host`` edge -- the plug really is in that
    board -- so this is not a lost constraint but a missing *instance*: the
    desktop has a fan, the log never named it, and only a human can say whether
    to create one. Sorted, so a re-run of the same desktop reports the same
    thing in the same order.
    """
    out: list[str] = []
    for key, rec in sorted(instances.items()):
        if rec.cls != "connector" or is_provisional(key):
            continue
        kind = str(rec.attrs.get("kind") or "")
        if kind in POWER_LEAD_KINDS or kind in TWO_ENDED_KINDS:
            continue
        owner_key = cable_owner(rec.cable or "", instances)
        owner = instances.get(owner_key or "")
        if owner_key and not owns_fan_lead(owner):
            out.append(
                f"unresolved fan owner: {key} is a cooler fan lead, but the only "
                f"cpu_cooler on this desktop is {owner_key} "
                f"(kind={str(owner.attrs.get('kind') or '')!r}), which has no fan. "
                f"No owner edge was proposed; create the fan instance if the "
                f"machine has one."
            )
    return out


# --------------------------------------------------------------------------- #
# propose_edges (spec 7.3 item 1)
# --------------------------------------------------------------------------- #
#: Latch classes whose ``of`` attribute names what they lock.
LATCH_OF_CLASSES = ("ram_latch", "drive_latch", "card_latch", "cooler_latch")

#: The latch bar on the chassis that holds the expansion cards (R12), and
#: what it holds: every one of them at once (the owner, 2026-11-04).
CARD_LATCH_CLASS = "card_latch"
CARD_CLASS = "expansion_card"

#: Latch classes that lock a fixed part class (the spec-7.3 template edges).
LATCH_TEMPLATE_HOSTS = {"cpu_socket_lever": "cpu", "psu_latch": "psu"}

#: ``cover.of`` -> what the cover hides. ``("cls", ...)`` matches instance
#: classes, ``("role", ...)`` matches ``screw.role``. ``front_bezel`` and
#: ``other`` name no part, so they are left to the manual panel.
COVER_TARGETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "cpu_cooler": ("cls", ("cpu_cooler",)),
    "motherboard_screws": ("role", ("motherboard",)),
    "ram": ("cls", ("ram_module",)),
    "expansion_slot": ("cls", ("expansion_card",)),
    "drive": ("cls", ("storage_drive", "optical_drive")),
}

#: The two ``cpu_cooler.kind`` values a two-part cooler is made of, bottom first:
#: the heatsink sits on the CPU, the fan on the heatsink (task W3a).
COOLER_STACK = ("heatsink", "fan")
#: The class whose levers the cooler on the CPU sits on.
SOCKET_LEVER_CLASS = "cpu_socket_lever"

#: The board everything in :data:`MOUNTED_ON_MOTHERBOARD` sits on (task W3d).
MOTHERBOARD_CLASS = "motherboard"
#: Classes mounted on the motherboard: each real instance comes off before the
#: board (user rule 2026-10-31, task W3d). The export side's class-level support
#: table (:data:`tda.core.export.graph_covers.SUPPORT`) is built from this tuple.
MOUNTED_ON_MOTHERBOARD = ("cpu", "ram_module", "cpu_cooler", "expansion_card")
#: Classes of :data:`MOUNTED_ON_MOTHERBOARD` whose S1 extras block the board as
#: well (R6, the owner's ruling of 2026-11-04: an added RAM stick also comes
#: off before the board). Their removal is a hand-written state event.
EXTRA_MOUNTED_CLASSES = ("ram_module",)


def cooler_on_cpu(instances: dict[str, InstanceRec]
                  ) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """``(the cooler on the CPU, the fan on top of it, why there is no answer)``.

    The fixed template edge of spec 7.3 says the cooler sits on the CPU. It used
    to resolve "the cooler" with ``unique_of_class``, so the nine desktops whose
    sheet lists a ``cpu_cooler.fan`` *and* a ``cpu_cooler.heatsink`` got no
    ``covered_by(cpu <- cooler)`` at all and every CPU plan opened the socket
    lever with the heatsink still on it (task W3a). Now:

    * exactly one settled ``cpu_cooler`` (any ``kind``) -- it, no fan;
    * two, one ``kind = heatsink`` and one ``kind = fan`` -- the heatsink, and
      the fan sits on the heatsink;
    * any other set (two fans, three parts, a part without a kind next to
      another) -- no answer and one line saying why, never a guess
      (:func:`unresolved_coolers`).

    No cooler at all is ``(None, None, None)``: nothing to say. The pixel
    evidence agrees: on every one of the 63 desktops with confirmed scanner
    frames the cooler this picks covers the socket lever (10-13 k px), and on
    the eight two-part desktops it is always the heatsink, never the fan.
    """
    coolers = _real_instances(instances, COOLER_CLASS)
    if not coolers:
        return None, None, None
    if len(coolers) == 1:
        return coolers[0], None, None
    kinds = {key: str(instances[key].attrs.get("kind") or "") for key in coolers}
    if len(coolers) == len(COOLER_STACK) and sorted(kinds.values()) == sorted(COOLER_STACK):
        bottom, top = (next(k for k in coolers if kinds[k] == kind) for kind in COOLER_STACK)
        return bottom, top, None
    said = ", ".join(f"{key} (kind={kinds[key]!r})" for key in coolers)
    return None, None, (
        f"unresolved cooler: 散热器组合无法判断哪件压在 CPU 上 / cannot tell which "
        f"cpu_cooler sits on the CPU: {said}. No covered_by edge from a cooler to the "
        f"CPU or the socket lever was proposed; in S1 make it one part, or one "
        f"kind=heatsink plus one kind=fan."
    )


def unresolved_coolers(instances: dict[str, InstanceRec]) -> list[str]:
    """The :func:`cooler_on_cpu` line for stage S1, or nothing (task W3a).

    The same channel as :func:`unresolved_fan_owners`: ``constraints`` prints
    it under the desktop and its report lists it with the unresolved references.
    """
    _, _, problem = cooler_on_cpu(instances)
    return [problem] if problem else []


def board_of(instances: dict[str, InstanceRec]) -> tuple[Optional[str], Optional[str]]:
    """``(the motherboard the mounted parts sit on, why there is no answer)`` (task W3d).

    The one settled ``motherboard`` of the desktop (drafts, 框外零件 and
    gallery-only parts excluded, as :func:`tda.core.graph_infer.real_instances`
    does). No board is ``(None, None)``: nothing to say. Two or more real boards
    (never seen on the 66 desktops: every "second board" is an ``ls:`` draft)
    are no answer and one line for S1, never a guess.
    """
    boards = _real_instances(instances, MOTHERBOARD_CLASS)
    if not boards:
        return None, None
    if len(boards) == 1:
        return boards[0], None
    return None, (
        f"unresolved board: 这台机器有不止一块主板，无法判断零件装在哪块上 / more than one "
        f"motherboard on this desktop: {', '.join(boards)}. No blocked_by (mounted) edge from a "
        f"mounted part ({', '.join(MOUNTED_ON_MOTHERBOARD)}) to a board was proposed; "
        f"in S1 keep one board."
    )


def unresolved_boards(instances: dict[str, InstanceRec]) -> list[str]:
    """The :func:`board_of` line for stage S1, or nothing (task W3d).

    The sibling of :func:`unresolved_coolers`, through the same channel:
    ``constraints`` prints it under the desktop and its report lists it.
    """
    _, problem = board_of(instances)
    return [problem] if problem else []


def propose_edges(instances: dict[str, InstanceRec], tax: Taxonomy,
                  actions: Optional[Iterable[ActionRec]] = None) -> list[Edge]:
    """Derive every hard edge the recorded attributes imply (spec 7.3 item 1).

    Rules, in order: ``screw.fastens`` -> ``fastened_by``; the connector rule of
    spec 7.2 -> ``connected_to``; ``latch.of`` plus the fixed CPU-lever and
    PSU-latch edges -> ``locked_by``; ``cover.of`` plus the fixed edges of the
    cooler on the CPU (over the CPU, over every socket lever, and a two-part
    cooler's fan over its heatsink; :func:`cooler_on_cpu`) -> ``covered_by``;
    every part mounted on the motherboard -> ``blocked_by(board <- part,
    mode="mounted")`` (:func:`_mounted_on_board`, below) -- the one rule-derived
    ``blocked_by``. ``cable_clip.holds`` still yields no edge on its own -- a
    clipped cable only blocks a target once a human says which one, so those
    ``blocked_by`` edges stay manual (spec 7.3 item 3).

    Every edge comes back ``necessity="required"``, ``source="rule"``,
    ``status="proposed"``; duplicates are collapsed, keeping the first reason.
    Instances whose relational fields are still empty simply derive nothing --
    run :func:`infer_relational_fields` first to fill the obvious ones.

    An instance added in S1 (:func:`tda.core.model.is_extra`) takes part in an
    edge only through a field that names it or that somebody filled in on it --
    ``fastens``, ``socket_host``, ``cable``, ``of`` -- and never through a
    reading keyed on its class, role or kind (the CPU lever over the CPU, a
    power lead on the PSU, the cooler over the CPU, "this cover hides every
    motherboard screw"). The log never operated it, so such an edge is either
    one the recorded teardown "violates" or a question about it nobody asked;
    a human who wants one adds it in the Relations tab.

    **Everything on the motherboard comes off before the motherboard** (the
    user's rule of 2026-10-31, task W3d): every real instance of
    :data:`MOUNTED_ON_MOTHERBOARD` -- ``cpu``, ``ram_module``, ``cpu_cooler``
    (both parts of a two-part cooler) and ``expansion_card`` -- gives
    ``blocked_by(motherboard <- part, mode="mounted")``, reason
    ``"<part> sits on <board>"``. It is an assembly / protocol rule of this
    dataset (physical for a GPU seated in its slot and screwed to the chassis,
    protocol for a RAM stick). The mode gates ``remove`` and ``displace`` of the
    board (:data:`BLOCKED_GATES`), and its blocker must be ``removed``
    (:data:`BLOCKED_REQUIRED`, :func:`required_states`). Not ``covered_by``,
    which the first round used: that type is satisfied by ``displaced``, the
    taxonomy lets ``displace`` apply to all four classes, and so every board plan
    swung the parts aside instead of taking them off -- the user's 「拆掉了」
    means removed, and in the 60 board-removing logs no CPU, RAM stick or card is
    ever displaced. All 60 logs already obey the rule.
    What it does **not** say is "the board is the last thing out": that holds
    on 56 of the 60 logs, but D01 D02 D35 D36 take the PSU (D01 also the optical
    drive, D02 the cover and its PSU latch) out after the board, so only these
    four classes block it. A board added in S1 takes no part (the class-keyed
    convention above); a **RAM stick added in S1 does** (the owner's R6 ruling
    of 2026-11-04: an added stick also comes off before the board -- every such
    stick is the second module of a 'RAM set' row, taken out with the first),
    see :data:`EXTRA_MOUNTED_CLASSES`; drafts are not real instances at all.
    Two real boards on one desktop give no edge and one S1 line
    (:func:`unresolved_boards`, the channel of :func:`unresolved_coolers`).

    **A card latch locks every expansion card of its desktop** (R12, the
    owner's ruling of 2026-11-04): one latch bar holds all the cards at once,
    so ``locked_by(card <- card_latch)`` for every real ``expansion_card``,
    whatever its ``of`` says (the sheets write the class word ``gpu``, which
    names no instance; D61's bar holds the GPU and the WLAN card).

    ``actions`` (the desktop's log) switches on the **timing rule of R1**
    (:func:`fan_leads_by_timing`): a fan lead unplugged right before the CPU
    fan comes out is that fan's lead, ``connected_to(fan <- lead)``, which is
    how the two-part coolers (one ``fan`` on one ``heatsink``) get the edge the
    cable tag could not give them. Without ``actions`` the cable tag alone
    decides, as before; every caller that holds the log passes it.
    """
    timing = fan_leads_by_timing(instances, actions) if actions is not None else {}
    out: list[Edge] = []
    for rule in (_fastened_by, _connected_to, _locked_by, _covered_by, _mounted_on_board):
        if rule is _connected_to:
            out.extend(_connected_to(instances, tax, timing))
        else:
            out.extend(rule(instances, tax))
    out.extend(_fan_lead_edges(instances, timing))
    return _dedupe(out)


def _dedupe(edges: list[Edge]) -> list[Edge]:
    """Drop repeated ``(type, target, blocker)`` triples, keeping the first."""
    seen: set[tuple[str, str, str]] = set()
    out: list[Edge] = []
    for edge in edges:
        key = (edge.type, edge.target, edge.blocker)
        if key in seen or edge.target == edge.blocker:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _edge(etype: str, target: str, blocker: str, reason: str,
          mode: Optional[str] = None) -> Edge:
    return Edge(type=etype, target=target, blocker=blocker, reason=reason, source="rule",
                mode=mode)


def _fastened_by(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """``screw.fastens = X`` -> ``fastened_by(X, screw)`` (spec 7.1)."""
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        if rec.cls != "screw":
            continue
        target = _resolve(instances, rec.fastens)
        if target:
            out.append(_edge("fastened_by", target, key, f"{key} fastens {target}"))
    return out


def _connected_to(instances: dict[str, InstanceRec], tax: Taxonomy,
                  timing: Optional[dict[str, set[str]]] = None) -> list[Edge]:
    """The connector rule of spec 7.2.

    For a part X, every connector whose ``socket_host`` is X *or* whose cable is
    owned by X must be unplugged before X may be removed. The two ends of a SATA
    data cable carry no owner, so each gates only the part it plugs into; every
    plug of the PSU harness gates the PSU as well as its own socket host. See
    :func:`connector_owner` for how the owner is decided.

    ``timing`` is :func:`fan_leads_by_timing`. Where it names leads of a
    cooler, it -- not the cable tag -- decides which fan leads that cooler owns
    (R1): a lead tagged ``cable:cpu_fan`` that the log unplugs after the fan is
    out is a case fan's, and gates only its socket host. A cooler with no timed
    lead falls back to the tag.
    """
    timing = timing or {}
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        if rec.cls != "connector":
            continue
        host = _resolve(instances, rec.socket_host)
        if host:
            out.append(_edge("connected_to", host, key, f"{key} plugs into {host}"))
        if is_extra(rec) and not rec.cable:
            continue  # its kind alone names no owner (see propose_edges)
        owner = connector_owner(rec, instances)
        if owner and timing.get(owner) and key not in timing[owner]:
            continue  # the log says this lead is not the cooler's (R1 timing)
        if owner and owner != host:
            out.append(
                _edge("connected_to", owner, key, f"{key} is on the {owner} cable")
            )
    return out


def _fan_lead_edges(instances: dict[str, InstanceRec],
                    timing: dict[str, set[str]]) -> list[Edge]:
    """``connected_to(fan <- lead)`` for every lead the R1 timing rule gives a fan.

    Most of them the cable tag already derives (:func:`_connected_to` runs
    first, and :func:`_dedupe` keeps its reason); the new ones are the two-part
    coolers' fan leads, whose ``cable:cpu_fan`` tag resolves to no single
    cooler.
    """
    out: list[Edge] = []
    for fan, leads in sorted(timing.items()):
        for lead in sorted(leads):
            out.append(_edge(
                "connected_to", fan, lead,
                f"{lead} is the lead of {fan}: unplugged right before {fan} comes out (R1)"))
    return out


def _locked_by(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """``latch.of`` -> ``locked_by``, plus the fixed CPU-lever / PSU-latch edges."""
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        target = None
        if rec.cls in LATCH_OF_CLASSES:
            target = _resolve(instances, rec.attrs.get("of"))
        elif rec.cls in LATCH_TEMPLATE_HOSTS and not is_extra(rec):
            target = _unique_of_class(instances, LATCH_TEMPLATE_HOSTS[rec.cls])
        if target:
            out.append(_edge("locked_by", target, key, f"{key} locks {target}"))
        if rec.cls == CARD_LATCH_CLASS and not is_provisional(key):
            for card in _real_instances(instances, CARD_CLASS):
                out.append(_edge("locked_by", card, key,
                                 f"{key} holds every expansion card, {card} too (R12)"))
    return out


def _covered_by(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """``cover.of`` -> ``covered_by``, plus the fixed edges of the cooler on the CPU.

    The cooler on the CPU (:func:`cooler_on_cpu`) covers the CPU, and every
    socket lever too (by class: a lever cannot be opened under a heatsink, and
    no edge said so, so every plan opened the lever first); of a two-part
    cooler the fan covers the heatsink. A part added in S1 takes no part in
    these class readings (see :func:`propose_edges`).
    """
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        if rec.cls != "cover":
            continue
        for target in _cover_targets(instances, rec):
            out.append(_edge("covered_by", target, key, f"{key} covers {target}"))
    on_cpu, fan, _ = cooler_on_cpu(instances)
    if on_cpu is None or is_extra(instances[on_cpu]):
        return out
    cpu = _unique_of_class(instances, "cpu")
    if cpu and not is_extra(instances[cpu]):
        out.append(_edge("covered_by", cpu, on_cpu, f"{on_cpu} sits on {cpu}"))
    for lever in _real_instances(instances, SOCKET_LEVER_CLASS):
        if not is_extra(instances[lever]):
            out.append(_edge("covered_by", lever, on_cpu,
                             f"{on_cpu} sits on the socket lever {lever}"))
    if fan and not is_extra(instances[fan]):
        out.append(_edge("covered_by", on_cpu, fan, f"{fan} sits on {on_cpu}"))
    return out


def _mounted_on_board(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """Every real part of :data:`MOUNTED_ON_MOTHERBOARD` blocks the board (task W3d).

    ``blocked_by(board <- part, mode=MOUNTED_MODE)``, one per settled, non-extra
    instance of the four classes: the part has to be removed before the board
    can be moved or taken out. No edge at all when the desktop has no single
    real board (:func:`board_of`) or the board is an S1 extra (see
    :func:`propose_edges`). A part added in S1 takes part only when its class
    is in :data:`EXTRA_MOUNTED_CLASSES` (R6).
    """
    board, _ = board_of(instances)
    if board is None or is_extra(instances[board]):
        return []
    out: list[Edge] = []
    for cls in MOUNTED_ON_MOTHERBOARD:
        for key in _real_instances(instances, cls):
            if not is_extra(instances[key]) or cls in EXTRA_MOUNTED_CLASSES:
                out.append(_edge("blocked_by", board, key, f"{key} sits on {board}",
                                 mode=MOUNTED_MODE))
    return out


def _cover_targets(instances: dict[str, InstanceRec], cover: InstanceRec) -> list[str]:
    """Which instances this cover hides, from its ``of`` attribute."""
    of = str(cover.attrs.get("of") or "")
    if not of:
        return []
    if of in instances:  # `of` already names one instance
        return [of]
    rule = COVER_TARGETS.get(of)
    if rule is None:
        return []
    kind, wanted = rule

    def hit(rec: InstanceRec) -> bool:
        if is_extra(rec):
            return False  # a class-wide reading; an added part is named or left out
        if kind == "cls":
            return rec.cls in wanted
        return rec.cls == "screw" and rec.attrs.get("role") in wanted

    return [key for key, rec in sorted(instances.items()) if hit(rec)]
