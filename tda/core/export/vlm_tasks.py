"""One generator per VLM task of spec 8.2, P0 set (task B6).

**The final round (owner decisions of 2026-11-04).** V4 V6 V7 V16 are asked under the rules
of the moment -- the stored graph plus dynamic occlusion of the scanner frame and loose parts
first (:class:`Moment`, :func:`tda.core.graph.moment_edges`) -- and not at all on a desktop
that is not stepwise (D66). V6's gold is the frontier, every action possible now, grouped
(open / mc / constrained); V7 plans under the rules (free / constrained, the plug count);
V16's validity is a replay; V14 is merged into V2 (``cannot_tell``); every record carries
``rationale.text``, its reasoning in words. The text below keeps the history.

``V1 V2 V3 V4 V5 V6 V7 V8 V10 V12 V14 V15 V16`` -- thirteen questions, each
with a structured answer and an ``answer_check`` block saying how a checker
re-derives it. **A question whose answer cannot be checked by program is not
emitted.** W3b (spec v1.48) added V7-cpu, took V5 out of the default
:data:`TASKS`, made V6 group-equivalent, asked V15-moment without the
``verified`` gate, and taught V2 / V3 / V12 / V14 that a 2D image shows a
screw's presence, never its tightness (:data:`SCREW_PRESENCE`, ``captive``).

What the graph knows, and what it does not
------------------------------------------
The constraint graph records physical *necessity* and it is incomplete: there
are no ``blocked_by`` edges in the database at all today, so a cooler screw
under a fan shroud has no edge saying the shroud is in the way. Three truths
follow, and every affordance and planning task here is built on them:

* what the graph **forbids** is certainly impossible -- a required precondition
  is a physical fact, and an unmet one cannot be worked around;
* what the **state** forbids is certainly impossible too, and for a different
  reason: an already-unplugged connector cannot be disconnected and a part that
  has left the machine cannot be removed. No edge is involved, and the answer is
  in the picture rather than in the graph;
* what the graph **permits** is only an **upper bound** -- it is the set of
  actions no *recorded* constraint rules out, not the set of actions that work;
* what the log **demonstrates** -- a successful action recorded at that state --
  is certainly possible.

The third one is there because the first two were not enough. Built from graph
negatives alone, V4 was 83 % solvable from the question text: the verbs barely
overlapped (`disconnect` was positive 987 times and negative never, because no
edge gates a connector), so "which verb is it?" answered the question without
looking at the image. A state negative is drawn to match its positive's verb and
class, which is what makes the two indistinguishable on paper.

So V4's positives are demonstrated actions and never merely permitted ones;
V5 answers with a bounded set (what must be in, what must not be in, and the
permitted upper bound named as such); V6 is graded on the demonstrated
reference and on how often an answer is certainly blocked; V16's valid plans are
logged suffixes only. Nothing here ever calls the permitted set "every action
that is physically possible".

Two sources of truth
--------------------
* **perception** (V1, V2, V8, V12, V14, V15) reads the compiled truth table and
  is emitted **only from VERIFIED frames**, only about instances that have a
  compiled row in *this* view;
* **planning / history** (V3, V4, V5, V6, V10, V16) reads the step log, the
  state machine and the constraint graph, so it exists for every desktop whose
  steps have been imported.

Where the log and the graph contradict each other -- the 18 violations of
``reports/constraints_report.md`` -- nothing is emitted: a ground truth that
argues with itself is worse than a gap, and :func:`illegal_steps` hands the list
to the caller to report.

Everything is pure and Qt-free; :mod:`tda.core.export.vlm` owns the loop, the
files and the refusals.
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Sequence

from tda.core.export import vlm_reasoning as R
from tda.core.export.vlm_groups import (
    INITIALISMS,
    Namer,
    PartGroup,
    _words,
    class_label,
    class_phrase,
    mark_box,
    match_class,
    part_groups,
    plural,
    present,
    target_class,
    to_do,
)
from tda.core.export.coco import (
    ANSWERABLE,
    VERIFIED,
    DesktopCtx,
    row_bbox_xywh,
)
from tda.core.graph import (
    LOOSE_SOURCE,
    OCCLUSION_SOURCE,
    Edge,
    applicable_preconditions,
    find_deadlocks,
    gated_verbs,
    legal_actions,
    loose_parts,
    moment_edges,
    remaining_plan,
    unmet,
)
from tda.core.graph_rules import (
    CABLE_PREFIX,
    LOOSE_MODE,
    is_stepwise,
    active_edges,
    blocker_state,
    cable_nodes,
    required_states,
    verb_applies,
    verb_effect,
)
from tda.core.implied import is_implied
from tda.core.model import (
    NO_CHANGE_STEP_TYPES,
    ActionRec,
    FrameKey,
    InstanceRec,
    VIEWS,
    StepType,
)
from tda.core.model import is_provisional
from tda.core.states import FrameState, InstState, gone_with_parent, initial_overrides
from tda.core.taxonomy import Taxonomy

__all__ = [
    "ACTION_TASKS",
    "MARKED_ACTION_TASKS",
    "MULTI_VIEW",
    "Namer",
    "PERCEPTION_TASKS_SINGLE",
    "PartGroup",
    "UNMARKED_TASKS",
    "V3_MIN_PX",
    "mark_box",
    "part_groups",
    "question_label",
    "ALL_TASKS",
    "BIG_PARTS",
    "CHASSIS_CLASS",
    "CHECK_TYPES",
    "GENERATORS",
    "IMAGE_ID_SALT",
    "LAYERS",
    "PERCEPTION_TASKS",
    "PLANNING_TASKS",
    "TASKS",
    "TRUTH_SOURCES",
    "FrameData",
    "TaskCtx",
    "class_label",
    "exclusion_reason",
    "illegal_steps",
    "instance_label",
    "match_class",
    "opaque_image_id",
    "plain_name",
    "record",
    "required_actions",
    "row_verified",
    "unresolved_action_targets",
]

#: Every task this module can generate, in the order records are written per
#: frame. ``--tasks V5`` still reaches V5 through this tuple.
#: V14 is gone (owner 2026-11-04): merged into V2, whose state questions answer
#: ``cannot_tell`` on a part this view cannot see.
ALL_TASKS = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V10", "V12", "V15", "V16")

#: The default task set (W3b, spec 8.2 v1.48): V5 is out of it. Its legal set is
#: the graph's upper bound, a median of 37 actions per step (p90 57, max 76), so
#: "list what can be done now" is graded on one demonstrated element and on a
#: count of blocked ones -- a number too format-sensitive to mean anything. V4
#: already asks the affordance question one action at a time. V7 (CPU goal,
#: asked from frames where the CPU sub-plan is under way) joins the set.
TASKS = tuple(t for t in ALL_TASKS if t != "V5")

#: The redesign of 2026-11-03 (owner): the action tasks show one moment from all four views and live in the
#: `multi` export unit; the per-view units keep single-view perception and V15.
ACTION_TASKS = ("V3", "V4", "V6", "V7", "V10", "V12", "V16")
PERCEPTION_TASKS_SINGLE = ("V1", "V2", "V8", "V15")
#: The export unit whose records show every view of a moment.
MULTI_VIEW = "multi"
#: The action tasks whose parts are outlined in red on the scanner image (owner 2026-11-03):
#: their records show that image alone, no side camera (:meth:`TaskCtx.moment` ``marked``).
#: V6's listed form (multiple choice) does the same; its open form, V3, V10 and V12 carry
#: no mark and show every camera.
MARKED_ACTION_TASKS = ("V4", "V7", "V16")

#: Tasks whose answer is read off the compiled truth table of one view.
PERCEPTION_TASKS = frozenset({"V1", "V2", "V8", "V12", "V15"})
#: Tasks whose answer is the step log, the state machine and the graph.
PLANNING_TASKS = frozenset({"V3", "V4", "V5", "V6", "V7", "V10", "V16"})
#: Tasks that cannot be derived without a trustworthy constraint graph.
GRAPH_TASKS = frozenset({"V4", "V5", "V6", "V7", "V16"})

#: Spec 8.2's capability layers. V3/V12 are single- and two-frame perception, so
#: L1; V10 is history and progress, which the spec files under planning, so L3.
LAYERS = {
    "V1": "L1", "V2": "L1", "V3": "L1", "V12": "L1", "V8": "L1",
    "V4": "L2", "V5": "L2",
    "V6": "L3", "V7": "L3", "V10": "L3", "V16": "L3",
    "V15": "L4",
}

#: The large parts (W3b): a state change of one of these is decidable at
#: RealSense / OAK2 resolution, a screw or a plug is not. V15's "not the same
#: moment" negatives and V3 / V12 on a view without confirmed rows are only
#: asked when one of them changed between the two frames.
BIG_PARTS = frozenset({"psu", "motherboard", "drive_cage", "optical_drive", "cover",
                       "cpu_cooler", "expansion_card", "ram_module"})

#: V7's goal class: the one removal target the paper keeps (minutes 2026-10-31
#: §3 -- the motherboard plan crosses parts the graph does not know are on it).
V7_GOAL_CLASS = "cpu"
#: The cooler class whose ``covered_by(cpu <- cooler)`` edge says the desktop's
#: cooler set is resolved; without one the CPU plan lifts the CPU through it.
V7_COOLER_CLASS = "cpu_cooler"
#: Edges whose certainty rests on confirmed-frame pixels: imported by W3a
#: (``source = 'pixels:W1'``) or written from the user's covering rulings
#: (``source = 'review:covers'``). A "no" resting only on such edges is
#: ``graph_blocked_pixels`` (:func:`is_pixel_edge`).
PIXEL_SOURCE_PREFIXES = ("pixels:", "review:")

#: The closed set of ``answer_check.type`` values. A record carrying anything
#: else is a record no checker can execute, and :func:`record` refuses it.
CHECK_TYPES = frozenset({
    "boxes",         # a set of instances, each localised: set F1 + box IoU
    "exact",         # named fields must match exactly
    "feasibility",   # V4: a yes/no plus the blocker set
    "bounded_set",   # V5: must-include, must-not-include, permitted upper bound
    "next_action",   # V6: the demonstrated next action, plus the blocked set
    "history",       # V10: the done set, the remaining count and the bin
    "plan",          # V16: validity, the first bad index and the violated edge
    "goal_plan",     # V7: violations, required-action recall, goal reached
    "frontier",      # V6 (final round): every action possible now, grouped -- set F1 / one option
    "components",    # V8 on a class with several instances: V1's grade on that class
})

#: Where a V4 label's certainty comes from (the governing principle, above).
#: ``graph_blocked_pixels`` is a graph "no" that rests **only** on edges
#: imported from confirmed-frame pixels (W3a, ``source = 'pixels:W*'``): the
#: paper writes those as "class rule x overlap in the annotation", not as an
#: observed constraint, so a sample has to say which kind it is.
TRUTH_SOURCES = ("demonstrated", "graph_blocked", "state_inapplicable",
                 "failed_attempt", "graph_blocked_pixels", "occlusion_blocked",
                 "loose_first", "simulated")
(DEMONSTRATED, GRAPH_BLOCKED, STATE_INAPPLICABLE, FAILED_ATTEMPT,
 GRAPH_BLOCKED_PIXELS, OCCLUSION_BLOCKED, LOOSE_FIRST, SIMULATED) = TRUTH_SOURCES
#: The kinds of rule-certain "no": the stored graph, and the two moment rules of the
#: owner's 2026-11-04 rulings -- dynamic occlusion (a part above covers it) and loose parts
#: first (R16) -- which :func:`graph_kind` names when they alone say no.
GRAPH_KINDS = frozenset({GRAPH_BLOCKED, GRAPH_BLOCKED_PIXELS, OCCLUSION_BLOCKED, LOOSE_FIRST})

#: ``Edge.source`` of the moment edges (never stored): occlusion and loose parts first.
MOMENT_SOURCES = frozenset({OCCLUSION_SOURCE, LOOSE_SOURCE})
#: ``Edge.source`` of an edge a constrained question's own constraint adds (V6, V7).
CONSTRAINT_SOURCE = "constraint"


@dataclass
class Moment:
    """The rules of one moment (the frame after ``step``): the stored graph plus the moment
    edges -- dynamic occlusion and loose parts first (:func:`tda.core.graph.moment_edges`)."""

    step: int
    occluded: dict
    extra: list
    edges: list
    by_target: dict
    loose: list


#: Tasks whose records never carry a mark: the box is the answer (V1, V8) or would say the
#: part is visible (a V2 question on a hidden part, which names it in words) -- spec 3.2.
UNMARKED_TASKS = frozenset({"V1", "V8"})
#: V3 (spec 3.6): the acted-on part (or its group) must show this many visible pixels on
#: the scanner frame before or after the action (1600 x 1600 frames), unless it is large.
V3_MIN_PX = 400

#: The three corruptions V16 builds a certainly-wrong plan with.
CORRUPTIONS = ("swap", "move", "substitute")

#: V10's progress bins, quarters of the desktop's own action count.
PROGRESS_BINS = ("0-25", "25-50", "50-75", "75-100")

#: The salt the opaque image ids are hashed with. It is a constant, not a
#: secret: the point is that a *prompt* carries no desktop, view or step, while
#: two exports of the same database still name the same frame the same way. The
#: manifest records it, so a reader can recompute the mapping.
IMAGE_ID_SALT = "tda-vlm-images-v1"

CABLE_CLASS = "cable"
CHASSIS_CLASS = "chassis"
REMOVED = "removed"
STATE_ATTR = "state"
#: A tool value that is not a fact about the machine and is never graded (I3).
UNGRADED_TOOLS = frozenset({None, "", "unknown"})


# --------------------------------------------------------------------------- #
# wording
# --------------------------------------------------------------------------- #
#: The present participle of every verb of spec 6.3, so a template can say
#: "Would removing the PSU work?" instead of "would remove the PSU work".
GERUNDS = {
    "unscrew": "unscrewing", "disconnect": "disconnecting", "open": "opening",
    "release": "releasing", "remove": "removing", "displace": "displacing",
    "reorient": "reorienting",
}


def instance_label(instance: str, rec: Optional[InstanceRec],
                   tax: Optional[Taxonomy] = None) -> str:
    """Human wording of an instance key: ``screw.motherboard.03`` -> "Motherboard screw 3"."""
    if rec is None:
        return instance
    disc = rec.attrs.get("role") or rec.attrs.get("kind") or ""
    if rec.cls == "drive_cage" and rec.attrs.get("of") == "optical":
        # an optical drive that never leaves its bracket (owner 2026-11-03): "Optical drive cage 1",
        # not the HDD / SSD cage's "Drive cage 1"
        disc = "optical"
    words = _words(disc) if disc else []
    words.extend(_words(class_label(str(rec.cls), tax)))
    ordinal = instance.rsplit(".", 1)[-1]
    if ordinal.isdigit():
        words.append(str(int(ordinal)))
    words = [w for w in words if w]
    if words and not words[0].isupper():
        words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)


def verb_phrase(verb: str, label: str) -> str:
    """``("unscrew", "PSU screw 1")`` -> ``"unscrew PSU screw 1"`` (imperative)."""
    return f"{verb} {label}"


def verb_gerund(verb: str, label: str) -> str:
    """``("unscrew", "PSU screw 1")`` -> ``"unscrewing PSU screw 1"``."""
    return f"{GERUNDS.get(verb, verb + 'ing')} {label}"


def pick(templates: Sequence[str], seed: str) -> tuple[int, str]:
    """Choose one wording deterministically: same record id, same question."""
    index = zlib.crc32(seed.encode("utf-8")) % len(templates)
    return index, templates[index]


def stable_order(items: Sequence[Any], seed: str,
                 key: Callable[[Any], str] = str) -> list[Any]:
    """A deterministic pseudo-random permutation of ``items``.

    Seeded by the caller's ``seed`` (desktop, step and task), so the hard
    negatives of one frame are the same in every export of the same database and
    are not the same ones on the next frame -- which is what keeps a sampled
    task from always asking about the first screw in the table. CRC32 rather
    than ``hash()``: the built-in is salted per process.
    """
    return sorted(items, key=lambda item: (zlib.crc32(f"{seed}|{key(item)}"
                                                      .encode("utf-8")), key(item)))


def opaque_image_id(desktop: int, view: str, step: int,
                    salt: str = IMAGE_ID_SALT) -> str:
    """The prompt-side name of one frame: no desktop, no view, no step.

    A prompt that says ``scan/D13/s017.png`` has answered "how far along is this
    teardown?" before the model has looked at the picture, and half of the P0
    set asks exactly that. The label side keeps the path, through the manifest
    :mod:`tda.core.export.vlm` writes beside the JSONL.
    """
    body = f"{salt}|{int(desktop)}|{view}|{int(step)}".encode("utf-8")
    return "img_" + hashlib.sha256(body).hexdigest()[:16]


def opaque_record_id(readable: str, salt: str = IMAGE_ID_SALT) -> str:
    """The record's own id, with the teardown hashed out of it.

    ``V4-D07-scan-s001-demonstrated-remove-psu.01`` names the desktop, the view,
    the step and the answer's own target. It is not on the prompt side, but a
    harness that prints record ids -- and they all do, in logs and in error
    messages -- would hand a model everything the opaque image ids were there to
    withhold. The readable form stays on the label side as ``readable_id``.
    """
    return "rec_" + hashlib.sha256(f"{salt}|{readable}".encode("utf-8")
                                   ).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# per-frame and per-desktop context
# --------------------------------------------------------------------------- #
@dataclass
class FrameData:
    """What one frame of one view offers a task generator.

    ``rows`` are the compiled rows with their masks dropped: every task here
    needs a box, a visibility, a placement and a status, and keeping several
    hundred 12 MP RLEs alive for the length of a desktop is the one thing that
    would make this export expensive.
    """

    step: int
    image: str
    image_id: str
    verified: bool
    rows: dict[str, dict] = field(default_factory=dict)
    pointable: dict[str, tuple[dict, list]] = field(default_factory=dict)
    #: Is this image from a view with confirmed rows on this desktop (W3b)? An
    #: image from a view without one -- every OAK1 / OAK2 / RealSense frame
    #: today -- is a picture and nothing more: its ``rows`` and ``pointable``
    #: are empty, so no mask and no visibility of that view is ever used or
    #: implied by a record about it, and V3 / V12 ask it about large parts only.
    confirmed: bool = True
    #: ``frame.ts`` in seconds, when the frame row has one (V15-moment's check).
    ts: Optional[float] = None
    #: ``(H, W)`` of the picture (V3's ``roi`` hint region is clipped to it); ``None`` when
    #: the export did not say, and then the view's nominal size is used.
    hw: Optional[tuple[int, int]] = None


def slim_row(row: dict) -> dict:
    """The part of a compiled row a question can be asked about."""
    return {
        "visibility": row.get("visibility"),
        "placement": row.get("placement"),
        "occlusion_ratio": row.get("occlusion_ratio"),
        "status": row.get("status"),
        "geom_type": row.get("geom_type"),
        "box": row_bbox_xywh(row),
        # visible pixels (V3_MIN_PX, the redesign's V3 rule): the mask's area, or a bench box's
        "area": _row_area(row),
    }


def _row_area(row: dict) -> Optional[int]:
    rle = row.get("visible_rle")
    if rle:
        from tda.core.masks import rle_area

        return rle_area(rle)
    box = row_bbox_xywh(row)
    return None if box is None else int(round(float(box[2]) * float(box[3])))


def pointable_rows(ctx: DesktopCtx, rows: dict[str, dict],
                   only_verified: bool) -> dict[str, tuple[dict, list]]:
    """``instance -> (row, bbox)`` for the rows this view can be asked about.

    Both geometries qualify -- a visible mask and a bench box -- as long as the
    row's visibility is answerable and its instance carries a taxonomy class. An
    instance whose visibility is ``occluded_full`` or ``out_of_view`` is exactly
    the one V14 must abstain on, so it is deliberately not here.
    """
    out: dict[str, tuple[dict, list]] = {}
    for instance in sorted(rows):
        row = rows[instance]
        if row.get("visibility") not in ANSWERABLE:
            continue
        if only_verified and row.get("status") != VERIFIED:
            continue
        if ctx.cls_of(instance) is None:
            continue
        box = row.get("box") if "box" in row else row_bbox_xywh(row)
        if box is not None:
            out[instance] = (row, list(box))
    return out


@dataclass(frozen=True)
class Candidate:
    """One action that is **certainly not possible** right now, and why.

    ``graph_blocked`` carries the unmet edges; ``state_inapplicable`` carries a
    reason read off the state -- the plug is already out, the part is already
    gone. Both are certainties; only the second needs the picture to settle.
    """

    verb: str
    target: str
    kind: str
    edges: tuple = ()
    reason: Optional[str] = None


def inapplicable_reason(state: FrameState, key: str) -> str:
    """Why this instance cannot be acted on: it is elsewhere, or already there."""
    inst = state.get(key)
    if inst is None:
        return "not_present"
    if inst.state == REMOVED or gone_with_parent(state, key):
        return "not_present"
    return f"already_{inst.state}"


@dataclass
class TaskCtx:
    """One desktop in one view, with everything the twelve generators read."""

    ctx: DesktopCtx
    view: str
    tier: Optional[str]
    graph_version: Optional[str]
    meta: dict
    edges: list[Edge]
    frames: dict[int, FrameData]
    steps: list[int]
    #: view -> {step -> FrameData}; only filled when V15 is wanted.
    others: dict[str, dict[int, FrameData]] = field(default_factory=dict)
    #: side view -> {step -> FrameData} for the ``multi`` unit: the frames shown beside the scanner's (oak1, oak2,
    #: rs). Empty in a per-view unit.
    side_frames: dict[str, dict[int, FrameData]] = field(default_factory=dict)
    #: step -> the edges a logged action of that step broke (spec 7.4).
    illegal: dict[int, list[str]] = field(default_factory=dict)
    #: why this desktop's graph-derived tasks are refused, or ``None`` (C2).
    excluded: Optional[str] = None
    #: how many hard negatives one frame may contribute per task.
    budget: int = 2
    #: ``task -> reason -> count`` of what a generator declined to ask and why,
    #: so the summary can say it (W3b: the paper cites these).
    skips: dict[str, dict[str, int]] = field(default_factory=dict)
    _legal: dict[tuple[int, bool], list] = field(default_factory=dict, repr=False)
    _blocked: dict[int, list] = field(default_factory=dict, repr=False)
    _inapplicable: dict[int, list] = field(default_factory=dict, repr=False)
    _by_target: Optional[dict[str, list[Edge]]] = field(default=None, repr=False)
    _v12: Optional[set[int]] = field(default=None, repr=False)
    _first_app: Optional[dict] = field(default=None, repr=False)
    _verbs: Optional[frozenset] = field(default=None, repr=False)
    _v7_plan0: dict = field(default_factory=dict, repr=False)

    _groups: Optional[dict] = field(default=None, repr=False)
    #: ``step -> {X: [Y, ...]}``: dynamic occlusion in this unit's scanner frame of ``step``
    #: (:func:`tda.core.occlusion.occlusion_blockers`, R17); ``None`` -- a unit read without
    #: a database, a test -- covers nothing.
    occlusion: Optional[Callable[[int], dict]] = field(default=None, repr=False)
    _moments: dict = field(default_factory=dict, repr=False)
    _active: Optional[Moment] = field(default=None, repr=False)

    # -- the rules of one moment (owner 2026-11-04) -------------------------- #
    @property
    def stepwise(self) -> bool:
        """May step-ordered questions be asked here? (D66: several parts at once -- no.)"""
        return is_stepwise(self.desktop, self.meta)

    def moment_at(self, step: int) -> Moment:
        """The rules of the frame after ``step`` (cached): stored edges + moment edges."""
        hit = self._moments.get(step)
        if hit is None:
            occluded = {k: list(v) for k, v in
                        ((self.occlusion(step) if self.occlusion is not None else {}) or {}).items()
                        if v}
            state = self.ctx.state_at(step)
            extra = moment_edges(self.ctx.instances, self.edges, state, self.tax,
                                 occluded=occluded, loose_first=True)
            edges = [*self.edges, *extra]
            index: dict[str, list[Edge]] = {}
            for edge in active_edges(edges):
                index.setdefault(edge.target, []).append(edge)
            hit = Moment(step, occluded, extra, edges, index,
                         loose_parts(self.ctx.instances, state))
            self._moments[step] = hit
        return hit

    @contextmanager
    def at(self, step: int, extra: Sequence[Edge] = ()):
        """Generate under the rules of ``step``'s moment (and a question's own ``extra``
        constraint edges): :attr:`by_target` and :attr:`cur_edges` answer with them."""
        before = self._active
        moment = self.moment_at(step)
        if extra:
            edges = [*moment.edges, *extra]
            index = {k: list(v) for k, v in moment.by_target.items()}
            for edge in active_edges(extra):
                index.setdefault(edge.target, []).append(edge)
            moment = Moment(step, moment.occluded, [*moment.extra, *extra], edges, index,
                            moment.loose)
        self._active = moment
        try:
            yield moment
        finally:
            self._active = before

    @property
    def cur_edges(self) -> list[Edge]:
        """The edges in force: the active moment's, else the stored graph."""
        return self._active.edges if self._active is not None else self.edges

    @property
    def groups(self) -> dict[str, PartGroup]:
        """``instance key -> its group`` (:func:`~tda.core.export.vlm_groups.part_groups`), once."""
        if self._groups is None:
            self._groups = part_groups(self.ctx)
        return self._groups

    def group_of(self, key: str) -> Optional[PartGroup]:
        return self.groups.get(key)

    def group_by_id(self, gid: str) -> Optional[PartGroup]:
        """The group a group id names, or ``None`` for an instance key or a cable node."""
        if "@" not in str(gid):
            return None
        return next((g for g in self.groups.values() if g.gid == gid), None)

    @property
    def own_view(self) -> str:
        """The camera this unit's own frames come from: the scanner for the ``multi`` unit."""
        return VIEWS[0] if self.view == MULTI_VIEW else self.view

    def legend(self, *moments: tuple[int, str],
               marked: bool = False) -> list[tuple[str, str, int]]:
        """``(view, moment, step)`` per image of :meth:`moment` calls, in image order."""
        out: list[tuple[str, str, int]] = []
        for step, name in moments:
            for frame in self.moment(step, marked=marked):
                out.append((self.view_of(frame), name, step))
        return out

    def view_of(self, frame: FrameData) -> str:
        if self.frames.get(frame.step) is frame:
            return self.own_view
        for view, frames in self.side_frames.items():
            if frames.get(frame.step) is frame:
                return view
        return self.own_view

    def moment(self, step: int, *, marked: bool = False) -> list[FrameData]:
        """The frames of one step to show together: this unit's own frame first, then the side views' frames
        in VIEWS order, each only where that camera has a picture of the step.

        ``marked`` (owner 2026-11-03): a question whose parts are outlined in red on the
        scanner image (V4, V6 multiple choice, V7, V16) shows that image alone -- the
        outline says where to look, so the side cameras are left out."""
        own = self.frames.get(step)
        out = [own] if own is not None else []
        if marked:
            return out
        for view in VIEWS:
            frame = (self.side_frames.get(view) or {}).get(step)
            if frame is not None:
                out.append(frame)
        return out

    # -- shorthand ---------------------------------------------------------- #
    @property
    def tax(self) -> Taxonomy:
        return self.ctx.tax

    @property
    def desktop(self) -> int:
        return self.ctx.desktop

    @property
    def log_steps(self) -> list[int]:
        """The desktop's own annotatable steps, frame or no frame.

        Not the same list as :attr:`steps`, which is the frames *this view* has.
        A scanner frame may be missing (spec 4.2, 缺帧处理) and the action still
        happened, so "what is the next step" and "how much is left" are counted
        on the log; only the questions that need a picture are counted on frames.
        """
        return sorted(s for s in self.ctx.steps if self.ctx.exportable(s))

    @property
    def by_target(self) -> dict[str, list[Edge]]:
        """``target -> its active edges``, built once per desktop -- the active moment's
        (:meth:`at`: stored + occlusion + loose-first edges) while one is set.

        :func:`~tda.core.graph.applicable_preconditions` scans the whole edge
        list, which is nothing for one call and 76 million comparisons for a
        66-desktop export: every frame asks for the permitted set and the
        blocked set, and every candidate action asks for its own preconditions.
        """
        if self._active is not None:
            return self._active.by_target
        if self._by_target is None:
            index: dict[str, list[Edge]] = {}
            for edge in active_edges(self.edges):
                index.setdefault(edge.target, []).append(edge)
            self._by_target = index
        return self._by_target

    def frame(self, step: int) -> Optional[FrameData]:
        return self.frames.get(step)

    def state(self, step: int) -> FrameState:
        return self.ctx.state_at(step)

    def label(self, instance: str) -> str:
        return instance_label(instance, self.ctx.instances.get(instance), self.tax)

    def key(self, step: int) -> FrameKey:
        return FrameKey(self.desktop, step, self.view)

    def seed(self, step: int, task: str) -> str:
        return f"D{self.desktop}|s{step}|{self.view}|{task}"

    def next_action_step(self, step: int) -> Optional[int]:
        """The next step after ``step`` that carries a named successful action."""
        for candidate in self.log_steps:
            if candidate > step and self.named_actions(candidate):
                return candidate
        return None

    def demonstrated(self, step: int) -> Optional[tuple[int, ActionRec]]:
        """``(step, action)`` the log shows being carried out from this state.

        The *first* action of the next step that has one. A compound step's
        later actions happen after its first one and are not demonstrated at
        *this* state, so they are not claimed here.

        ``None`` for three reasons, and all three mean "no certainty here":
        nothing follows, the log and the graph contradict each other at the next
        step, or the next action changes no state at all. The last is
        ``reorient``: spec 6.3 gives it no effect and nothing gates it, so
        "could you do it?" has no answer the constraint graph can grade -- and
        answering with the step *after* it would be answering a different
        question.
        """
        nxt = self.next_action_step(step)
        if nxt is None or nxt in self.illegal:
            return None
        action = self.named_actions(nxt)[0]
        return (nxt, action) if self.changes_state(action) else None

    def changes_state(self, action: ActionRec) -> bool:
        """Does this action move the state machine at all? (``reorient`` does not.)"""
        rec = self.ctx.instances.get(action.target)
        cls = rec.cls if rec is not None else (
            CABLE_CLASS if action.target.startswith(CABLE_PREFIX) else None)
        if cls is None:
            return False
        effect = self.tax.apply_verb(cls, rec.attrs if rec is not None else {},
                                     action.verb)
        return effect is not None and effect[0] == STATE_ATTR

    def named_actions(self, step: int) -> list[ActionRec]:
        """Successful actions of ``step`` whose target this export can name."""
        return [a for a in self.ctx.actions_at(step)
                if target_class(self.ctx, a.target) is not None]

    def failed_actions(self, step: int) -> list[ActionRec]:
        return sorted(
            (a for a in self.ctx.actions
             if a.step == step and a.result != "success"
             and target_class(self.ctx, a.target) is not None),
            key=lambda a: a.idx,
        )

    def permitted(self, step: int) -> list[tuple[str, str]]:
        """The graph-permitted **upper bound** at the state after ``step``.

        Required edges only, which is the standard
        :func:`~tda.core.graph.validate_sequence` judges the log by. This is not
        "what is possible": it is what no recorded constraint forbids.
        """
        return self._permitted(step, False)

    def permitted_strict(self, step: int) -> list[tuple[str, str]]:
        return self._permitted(step, True)

    def _permitted(self, step: int, strict: bool) -> list[tuple[str, str]]:
        hit = self._legal.get((step, strict))
        if hit is None:
            # the owner's rules of 2026-11-04: loose parts first, and nothing a part above
            # still covers (dynamic occlusion of this frame)
            hit = legal_actions(self.ctx.instances, self.edges, self.state(step),
                                self.tax, strict=strict, loose_first=True,
                                occluded=self.moment_at(step).occluded)
            self._legal[(step, strict)] = hit
        return hit

    def blocked(self, step: int) -> list[tuple[str, str, list[Edge]]]:
        """``(verb, target, unmet edges)`` -- what the graph *certainly* forbids."""
        hit = self._blocked.get(step)
        if hit is None:
            with self.at(step):
                hit = blocked_actions(self, step)
            self._blocked[step] = hit
        return hit

    @property
    def first_applicable(self) -> dict[tuple[str, str], int]:
        """``(verb, instance) -> the earliest step the verb could be performed``.

        What makes "can you disconnect this connector?" a fair question at step
        40 is that it *was* a fair question at step 12: the plug was in, and the
        model has to see that it is not any more. An action that was never
        applicable at any point -- ``open`` on a PSU, ``unscrew`` on a cable --
        is not a question about this machine at all.
        """
        if self._first_app is None:
            out: dict[tuple[str, str], int] = {}
            for j in [0, *self.log_steps]:
                state = self.state(j)
                for key, rec in self.ctx.instances.items():
                    inst = state.get(key)
                    if inst is None:
                        continue
                    for verb in self.tax.verbs:
                        if (verb, key) in out:
                            continue
                        if verb_applies(self.tax, rec.cls, rec.attrs, verb,
                                        inst.state):
                            out[(verb, key)] = j
            self._first_app = out
        return self._first_app

    def askable(self, key: str) -> bool:
        """Is this instance one a question may be asked about at all?

        Real, not a Label Studio draft, and not an identity
        :mod:`tda.core.implied` invented because the log referenced a class
        nobody operated -- an implied motherboard has no history of its own, so
        "is it still there?" is a question about the inference, not the machine.
        """
        rec = self.ctx.instances.get(key)
        return (rec is not None and not is_provisional(key)
                and not is_implied(rec))

    def inapplicable(self, step: int) -> list[Candidate]:
        """Actions the *state* rules out right now (cached per frame)."""
        hit = self._inapplicable.get(step)
        if hit is None:
            hit = inapplicable_in(self, self.state(step), step)
            self._inapplicable[step] = hit
        return hit

    def certainly_wrong(self, step: int) -> list[Candidate]:
        """Everything that certainly cannot be done at ``step``, both kinds.

        A graph "no" that rests only on pixel-imported edges is kind
        ``graph_blocked_pixels`` (:func:`graph_kind`), so the label says so.
        """
        return [*(Candidate(v, t, graph_kind(e), tuple(e))
                  for v, t, e in self.blocked(step) if self.askable(t)),
                *self.inapplicable(step)]

    def skip(self, task: str, reason: str) -> None:
        """Count one question a generator declined to ask, by reason."""
        bucket = self.skips.setdefault(task, {})
        bucket[reason] = bucket.get(reason, 0) + 1

    def matched(self, candidates: Sequence[Candidate], verb: str,
                cls: Optional[str], seed: str) -> list[Candidate]:
        """``candidates`` ordered so the closest match to ``(verb, cls)`` is first.

        Same verb and same class beats same verb, which beats same class, which
        beats anything -- and inside a tier a graph-blocked action beats a
        state-inapplicable one, because the graph's "no" needs no picture to be
        certain. The tail is a seeded permutation, so the choice is stable and
        is not always the first instance in the table.
        """
        used = self.demonstrated_verbs

        def rank(c: Candidate) -> tuple:
            same_verb = c.verb == verb
            same_cls = self.ctx.cls_of(c.target) == cls
            tier = (0 if same_verb and same_cls else
                    1 if same_verb else 2 if same_cls else 3)
            # when the verb cannot be matched, at least prefer one this teardown
            # actually uses: a verb that is only ever a negative -- `displace`,
            # which no sheet records -- is a giveaway of its own
            spoken = 0 if c.verb in used else 1
            kind = 0 if c.kind in GRAPH_KINDS else 1
            return (tier, spoken, kind,
                    zlib.crc32(f"{seed}|{c.verb}|{c.target}".encode("utf-8")),
                    c.verb, c.target)
        return sorted(candidates, key=rank)

    @property
    def displaced_in_log(self) -> frozenset[str]:
        """The parts this desktop's log swings aside (R14: only those are hinged)."""
        return frozenset(a.target for a in self.ctx.actions if a.verb == "displace")

    @property
    def demonstrated_verbs(self) -> frozenset[str]:
        """The verbs this desktop's log actually uses."""
        if self._verbs is None:
            self._verbs = frozenset(a.verb for s in self.log_steps
                                    for a in self.named_actions(s))
        return self._verbs

    @property
    def v12_steps(self) -> set[int]:
        """The steps V12 may ask about, balanced across this desktop.

        Its negatives can only come from ``dupli`` rows (spec 8.2), and there
        are sixteen of those in the whole dataset against two and a half
        thousand ordinary steps. Publishing every change would make "yes,
        something changed" right 99 % of the time, so the positives are
        subsampled to the number of negatives the desktop actually has, by a
        seed of the desktop. A desktop with no ``dupli`` row asks nothing.
        """
        if self._v12 is None:
            negative, positive = [], []
            for step in sorted(self.frames):
                before = self.frame(step - 1)
                if before is None:
                    continue
                frame = self.frames[step]
                verified = before.verified and frame.verified
                # a view without confirmed rows (W3b) asks too, but its "yes"
                # must be a large part: a lone screw is undecidable at RS / OAK2
                # resolution. "No change" (a dupli pair) is decidable anywhere.
                unlabelled = not (before.confirmed and frame.confirmed)
                if not (verified or unlabelled):
                    continue
                rec = self.ctx.steps.get(step)
                kind = self.ctx.step_type(step)
                if kind == StepType.DUPLI.value or (rec is not None and rec.dupli):
                    negative.append(step)
                elif captive_only_step(self, step):
                    # the log moved, the picture did not: a "no visible change"
                    # negative of its own kind, never a "yes" (W3b)
                    negative.append(step)
                elif kind not in NO_CHANGE_STEP_TYPES and kind != StepType.FAILED.value:
                    if self.state(step - 1) != self.state(step):
                        if unlabelled and not big_changes(self, step - 1, step):
                            continue
                        positive.append(step)
            chosen = stable_order(positive, f"D{self.desktop}|{self.view}|V12",
                                  str)[:len(negative)]
            self._v12 = set(negative) | set(chosen)
        return self._v12


def captive_only_step(tc: "TaskCtx", step: int) -> bool:
    """Is every state change of ``step`` the loosening of a captive screw? (W3b)

    281 of the 2 740 logged steps are: the cooler's screws are turned and stay
    where they were, so frames ``step - 1`` and ``step`` show the same machine.
    """
    actions = tc.ctx.actions_at(step)
    if not actions or not all(invisible_action(tc.ctx, a) for a in actions):
        return False
    old, new = tc.state(step - 1), tc.state(step)
    moved = [k for k in old.keys() & new.keys() if old[k].state != new[k].state]
    return bool(moved) and all(is_captive_screw(tc.ctx, k) for k in moved)


def graded_tool(tool: Optional[str]) -> Optional[str]:
    """The tool if it is a fact, else ``None`` -- ``unknown`` is never graded (I3)."""
    return None if tool in UNGRADED_TOOLS else tool


#: What a 2D image can say about a screw (W3b, the user's data facts): whether
#: it is still there. Tightness was never annotated -- a drawn screw is a
#: fastened one, an unscrewed one is gone from the frame -- so ``fastened`` and
#: ``loosened`` are one visible state.
SCREW_PRESENCE = {"fastened": "present", "loosened": "present", "removed": "removed"}
#: The state space a screw question is asked in.
PRESENCE_STATES = ("present", "removed")
#: V12's second "no visible change" source: a step whose every action is the
#: unscrewing of a captive screw, which stays on its fan and looks the same.
CAPTIVE_LOOSENED = "captive_screw_loosened"


def is_captive_screw(ctx: DesktopCtx, key: str) -> bool:
    """A screw that stays on its part when unscrewed (``attrs.captive``)."""
    rec = ctx.instances.get(key)
    return bool(rec is not None and rec.cls == "screw" and rec.attrs.get("captive"))


def visible_state(ctx: DesktopCtx, key: str, state: Optional[str]) -> Optional[str]:
    """The state as a picture can show it: a screw's presence, else the state."""
    rec = ctx.instances.get(key)
    if rec is not None and rec.cls == "screw" and state is not None:
        return SCREW_PRESENCE.get(state, state)
    return state


def invisible_action(ctx: DesktopCtx, action: ActionRec) -> bool:
    """``unscrew`` on a captive screw: the log moved, the picture did not."""
    return action.verb == "unscrew" and is_captive_screw(ctx, action.target)


#: Tasks whose answer can hinge on which captive screws are already loose.
TIGHTNESS_TASKS = frozenset({"V4", "V5", "V6", "V7", "V10", "V16"})


def image_undecidable(tc: "TaskCtx", rec: dict) -> bool:
    """Does this record's truth rest on a captive screw's tightness? (W3b)

    True when, at the record's step, a captive screw is ``loosened`` and the
    answer or its check names a captive screw: the picture of a loose captive
    screw is the picture of a tight one, so which of them are left is in the
    log and nowhere in the image. The state-oracle reference may show it; the
    oracle-vs-VLM gap analysis leaves these out.
    """
    if rec["prompt"]["task"] not in TIGHTNESS_TASKS:
        return False
    state = tc.state(int(rec["label"]["step"]))
    captive = [k for k in tc.ctx.instances if is_captive_screw(tc.ctx, k)]
    if not any(state.get(k) is not None and state[k].state == "loosened"
               for k in captive):
        return False
    check = {k: v for k, v in rec["label"]["answer_check"].items()
             if k != "start_state"}
    text = json.dumps([rec["label"]["answer"], check], ensure_ascii=False)
    return any(f'"{k}"' in text for k in captive)


def is_pixel_edge(edge: Edge) -> bool:
    """Does this edge rest on pixels -- ``pixels:*`` (W3a) or ``review:*`` (rulings)?

    The one predicate V4, V6, V16 and V7 ask (through :func:`graph_kind` and
    :func:`required_actions`), so the four cannot disagree about a source.
    """
    return str(getattr(edge, "source", "") or "").startswith(PIXEL_SOURCE_PREFIXES)


def graph_kind(edges: Sequence[Edge]) -> str:
    """``graph_blocked_pixels`` when every edge saying "no" is pixel-imported;
    ``occlusion_blocked`` / ``loose_first`` when every one is a moment edge of that kind
    (:func:`tda.core.graph.moment_edges`).

    One ordinary edge among them is enough for the plain ``graph_blocked``: the
    "no" is then certain without the pixels.
    """
    edges = list(edges)
    if edges and all(e.source == OCCLUSION_SOURCE for e in edges):
        return OCCLUSION_BLOCKED
    if edges and all(e.source == LOOSE_SOURCE for e in edges):
        return LOOSE_FIRST
    if edges and all(is_pixel_edge(e) or e.source in MOMENT_SOURCES for e in edges) \
            and any(is_pixel_edge(e) for e in edges):
        return GRAPH_BLOCKED_PIXELS
    return GRAPH_BLOCKED


def unordered_group(ctx: DesktopCtx, target: str) -> Optional[str]:
    """The instance's ``group_id`` when its group is ``unordered``, else ``None``."""
    rec = ctx.instances.get(target)
    if rec is None or not rec.group_id or rec.group_order != "unordered":
        return None
    return str(rec.group_id)


def big_changes(tc: "TaskCtx", before: int, after: int) -> list[str]:
    """The large parts (:data:`BIG_PARTS`) whose state differs between two steps."""
    old, new = tc.state(before), tc.state(after)
    out = []
    for key in sorted(old.keys() & new.keys()):
        if old[key].state == new[key].state:
            continue
        if tc.ctx.cls_of(key) in BIG_PARTS:
            out.append(key)
    return out


# --------------------------------------------------------------------------- #
# the graph, asked the three ways the tasks need
# --------------------------------------------------------------------------- #
def blocked_actions(tc: TaskCtx, step: int) -> list[tuple[str, str, list[Edge]]]:
    """``(verb, target, unmet edges)`` for everything the graph forbids right now.

    The verb applies to the class in this state (spec 6.3), so "can you do this?"
    is a sensible question, and at least one hard precondition is not satisfied,
    so the answer is certainly no and the graph says why. This is the half of
    the graph that is a fact rather than an upper bound.
    """
    state = tc.state(step)
    out: list[tuple[str, str, list[Edge]]] = []
    for key, rec in sorted(tc.ctx.instances.items()):
        inst = state.get(key)
        if inst is None or inst.state == REMOVED or key not in tc.by_target:
            continue  # nothing points at it: nothing can be blocking it
        for verb in tc.tax.verbs:
            if not verb_applies(tc.tax, rec.cls, rec.attrs, verb, inst.state):
                continue
            bad = unmet_for(tc, verb, key, state)
            if bad:
                out.append((verb, key, bad))
    for key, current in sorted(cable_nodes(active_edges(tc.cur_edges), state).items()):
        if not verb_applies(tc.tax, CABLE_CLASS, {}, "release", current):
            continue
        bad = unmet_for(tc, "release", key, state)
        if bad:
            out.append(("release", key, bad))
    return out


def blocked_in(tc: TaskCtx, state: FrameState) -> list[Candidate]:
    """:func:`blocked_actions`, asked of an arbitrary state (a plan's mid-point)."""
    out: list[Candidate] = []
    for key, rec in sorted(tc.ctx.instances.items()):
        inst = state.get(key)
        if inst is None or inst.state == REMOVED or key not in tc.by_target:
            continue
        for verb in tc.tax.verbs:
            if not verb_applies(tc.tax, rec.cls, rec.attrs, verb, inst.state):
                continue
            bad = unmet_for(tc, verb, key, state)
            if bad:
                out.append(Candidate(verb, key, graph_kind(bad), tuple(bad)))
    return out


def inapplicable_in(tc: TaskCtx, state: FrameState, step: int) -> list[Candidate]:
    """Actions the state rules out: the verb no longer starts from where it is.

    Only verbs that belong to the class (spec 6.3) and that **were** applicable
    to this very instance at some earlier step: the question is "has this
    already happened?", which is answerable from the picture, and not "does this
    verb exist?", which is answerable from the vocabulary.
    """
    first = tc.first_applicable
    out: list[Candidate] = []
    for key, rec in sorted(tc.ctx.instances.items()):
        if not tc.askable(key):
            continue
        inst = state.get(key)
        if inst is None:
            continue
        reason = inapplicable_reason(state, key)
        if reason == "already_loosened" and is_captive_screw(tc.ctx, key):
            # "it is already loosened" is the one state a picture of a captive
            # screw cannot show (W3b): no certain negative of it
            continue
        for verb in tc.tax.verbs:
            if verb_applies(tc.tax, rec.cls, rec.attrs, verb, inst.state):
                continue
            began = first.get((verb, key))
            if began is None or began >= step:
                continue
            out.append(Candidate(verb, key, STATE_INAPPLICABLE, (), reason))
    return out


def unmet_for(tc: TaskCtx, verb: str, target: str, state: FrameState) -> list[Edge]:
    """Which required preconditions of ``(verb, target)`` are not satisfied.

    The **one** way anything in this module asks that question -- V4, V5, V6 and
    V16 all come through here, so a change to
    :func:`~tda.core.graph.applicable_preconditions` (B5 narrows the gate by
    ``blocked_by.mode``) reaches all four at once. A private re-implementation
    in the plan simulator once made V16 disagree with V4 about a ``cable:*``
    blocker that is not in the snapshot.
    """
    return unmet(applicable_preconditions(tc.by_target.get(target, []), (verb, target)),
                 state, "required")


# --------------------------------------------------------------------------- #
# reasons in what the picture shows (owner 2026-11-04)
# --------------------------------------------------------------------------- #
#: How an edge's blocker stands to its target, as a plain verb (``answer.why``): the model never
#: sees the graph, so no reason a person or a model reads says ``locked_by`` or ``violates_edge``.
RELATION_WORDS = {"fastened_by": "fastens", "connected_to": "is plugged into",
                  "locked_by": "locks", "covered_by": "covers", "blocked_by": "is in the way of"}
#: ... and a ``blocked_by`` edge by its mode.
BLOCKED_RELATION_WORDS = {"physical_path": "is in the way of", "mounted": "is mounted on",
                          "contained": "sits in", "tool_access": "blocks access to",
                          "cable_tension": "holds back by its cable",
                          LOOSE_MODE: "lies loose in the case and comes out before"}
#: What a blocker looks like once it no longer holds the part.
MET_WORDS = {REMOVED: "out", "open": "open", "loosened": "undone", "unplugged": "unplugged",
             "displaced": "moved aside", "released": "released"}
#: "the CPU can be <taken out>".
PARTICIPLES = {"remove": "taken out", "displace": "moved aside", "open": "opened",
               "unscrew": "unscrewed", "disconnect": "unplugged", "release": "released",
               "reorient": "turned"}
#: The order the clauses of one reason are told in: what is over it, in it, holding it, on it.
_CLAUSE_ORDER = ("covered_by", "blocked_by", "locked_by", "fastened_by", "connected_to")


def relation_word(edge: Edge) -> str:
    """The plain verb of ``edge`` (:data:`RELATION_WORDS`): "locks", "is mounted on"."""
    if edge.type == "blocked_by":
        return BLOCKED_RELATION_WORDS.get(edge.mode or "", RELATION_WORDS["blocked_by"])
    return RELATION_WORDS.get(edge.type, "holds")


def _met_word(edge: Edge, current: str) -> str:
    if edge.type == "covered_by" and current == REMOVED:
        return "off"
    if edge.type == "connected_to" and current == REMOVED:
        return "unplugged"
    return MET_WORDS.get(current, current)


def _blocker_words(tc: "TaskCtx", key: str, many: bool) -> str:
    if key.startswith(CABLE_PREFIX):
        words = cable_words(key)[len("the "):]
        return plural(words) if many else words
    return class_phrase(tc.ctx, key, many=many)


def _anchored(tc: "TaskCtx", blocker: str, targets: Sequence[str]) -> bool:
    """Does ``blocker`` belong to one of ``targets`` (its latch, its screw, its plug)?"""
    rec = tc.ctx.instances.get(blocker)
    if rec is None:
        return False
    owners = {rec.attrs.get("of"), rec.fastens, rec.socket_host, rec.parent}
    return bool(owners & set(targets))


def _and(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def asked_words(tc: "TaskCtx", ident: str) -> tuple[str, bool]:
    """``("the CPU", False)`` / ``("the RAM modules", True)``: the part a reason is about."""
    group = tc.group_by_id(ident)
    if group is not None:
        return f"the {group.label}", True
    if ident.startswith(CABLE_PREFIX):
        return cable_words(ident), False
    return f"the {class_phrase(tc.ctx, ident)}", False


def reason_clauses(tc: "TaskCtx", edges: Sequence[Edge], state: FrameState, *, met: bool,
                   asked: Optional[tuple[str, Sequence[str]]] = None,
                   every: Optional[Sequence[Edge]] = None) -> list[str]:
    """One clause per kind of thing that still holds the part (``met = False``: "the CPU socket
    lever still locks the CPU") or that no longer does (``met = True``: "both its RAM latches
    are open"), in words of what the picture shows -- classes and counts, never an ordinal,
    never an edge type.

    ``asked`` is ``(ident, members)``: edges on its members speak of "it" / "them"; ``every``
    is all the edges of that kind (met and unmet), so a count says "2 of its 4 PSU screws".
    """
    ident, members = asked if asked is not None else (None, ())
    members = list(members)
    own_words, own_many = asked_words(tc, ident) if ident is not None else ("", False)
    every = list(every if every is not None else edges)

    def target_key(edge: Edge) -> str:
        return "asked" if edge.target in members else edge.target

    def blocker_key(edge: Edge) -> str:
        return _blocker_words(tc, edge.blocker, False)

    totals: dict[tuple, set] = {}
    for edge in every:
        totals.setdefault((target_key(edge), edge.type, edge.mode or "", blocker_key(edge)),
                          set()).add(edge.blocker)
    groups: dict[tuple, dict[str, set]] = {}
    for edge in edges:
        current = blocker_state(state, edge.blocker)
        if current is None:
            continue
        word = _met_word(edge, current) if met else ""
        head = (target_key(edge), edge.type, edge.mode or "", word)
        groups.setdefault(head, {}).setdefault(blocker_key(edge), set()).add(edge.blocker)

    def order(head):
        kind = _CLAUSE_ORDER.index(head[1]) if head[1] in _CLAUSE_ORDER else len(_CLAUSE_ORDER)
        return (head[0] != "asked", head[0], kind, head[2], head[3])

    out: list[str] = []
    for head in sorted(groups, key=order):
        tkey, etype, mode, word = head
        if tkey == "asked":
            targets, tgt = members, (f"all {own_words}" if own_many else own_words)
            pron, poss = ("them", "their") if own_many else ("it", "its")
        else:
            targets, tgt = [tkey], asked_words(tc, tkey)[0]
            pron, poss = tgt, f"{tgt}'s"
        subjects, count = [], 0
        for words, blockers in sorted(groups[head].items()):
            n = len(blockers)
            total = len(totals.get((tkey, etype, mode, words), blockers))
            count += n
            first = sorted(blockers)[0]
            mine = _anchored(tc, first, targets)
            phrase = _blocker_words(tc, first, n > 1)
            many = _blocker_words(tc, first, True)
            who = f"{poss} " if mine and tkey == "asked" else "the "
            if n == 1 and total == 1:
                subjects.append(f"the {phrase}")
            elif n == total == 2:
                subjects.append(f"both {who}{phrase}")
            elif n == total:
                subjects.append(f"all {n} of {who}{phrase}" if who != "the "
                                else f"all {n} {phrase}")
            else:
                subjects.append(f"{n} of {who}{total} {many}")
        subject = _and(subjects)
        plural_subject = count > 1
        be = "are" if plural_subject else "is"
        s = "" if plural_subject else "s"
        if met:
            where = {"covered_by": f" over {pron}",
                     "blocked_by": {"mounted": f" mounted on {pron}", "contained": f" in {pron}",
                                    "cable_tension": ""}.get(mode, f" in {poss} way")}.get(etype, "")
            out.append(f"{subject}{where} {be} {word}")
            continue
        if etype == "fastened_by":
            out.append(f"{subject} {be} still in" + ("" if tkey == "asked" else f", holding {tgt}"))
        elif etype == "connected_to":
            out.append(f"{subject} {be} still plugged in" + ("" if tkey == "asked" else f" to {tgt}"))
        elif etype == "locked_by":
            out.append(f"{subject} still lock{s} {tgt}")
        elif etype == "covered_by":
            out.append(f"{subject} still cover{s} {tgt}")
        elif mode == "mounted":
            out.append(f"{subject} {be} still mounted on {tgt}")
        elif mode == "contained":
            out.append(f"{subject} {be} still in {tgt}")
        elif mode == "tool_access":
            out.append(f"{subject} still block{s} access to {tgt}")
        elif mode == "cable_tension":
            out.append(f"{subject} still hold{s} {tgt} back by its cable")
        elif mode == LOOSE_MODE:
            # R16: a part lying loose in the case comes out before anything else is done
            out.append(f"{subject} lying loose in the case {'have' if plural_subject else 'has'}"
                       f" to come out first")
        else:
            out.append(f"{subject} {be} still in the way of {tgt}")
    return out


def gated_edges(tc: "TaskCtx", verb: str, members: Sequence[str]) -> list[Edge]:
    """The required edges that gate ``verb`` on any of ``members`` (met or not)."""
    out = []
    for m in members:
        out.extend(e for e in applicable_preconditions(tc.by_target.get(m, []), (verb, m))
                   if e.necessity == "required")
    return sorted(set(out), key=lambda e: (e.type, e.target, e.blocker))


def why_list(tc: "TaskCtx", edges: Sequence[Edge], state: FrameState) -> list[dict]:
    """``answer.why``: per blocker, its plain name, its state as a picture shows it, and how it
    stands to the part (a plain verb) -- the machine-readable twin of the reason text."""
    out, seen = [], set()
    for edge in sorted(edges, key=lambda e: (e.blocker, e.type, e.target)):
        current = blocker_state(state, edge.blocker)
        if current is None or (edge.blocker, relation_word(edge)) in seen:
            continue
        seen.add((edge.blocker, relation_word(edge)))
        out.append({"part": plain_name(tc, edge.blocker),
                    "state": visible_state(tc.ctx, edge.blocker, current),
                    "relation": relation_word(edge)})
    return out


def feasibility_reason(tc: "TaskCtx", verb: str, ident: str, members: Sequence[str],
                       state: FrameState, bad: Sequence[Edge]) -> tuple[str, list[dict]]:
    """``(reason text, why list)`` of "can <verb> <ident> be done now?" (owner 2026-11-04).

    Feasible (no unmet edge): the prerequisites now met -- "both its RAM latches are open, so
    the RAM module can be taken out"; infeasible: what still holds it -- "the CPU socket lever
    still locks the CPU". Words of what the picture shows, never the graph's.
    """
    every = gated_edges(tc, verb, members)
    asked = (ident, list(members))
    if bad:
        clauses = reason_clauses(tc, bad, state, met=False, asked=asked, every=every)
        return _and(clauses) or f"something still holds {asked_words(tc, ident)[0]}", \
            why_list(tc, bad, state)
    met = [e for e in every if e not in set(bad) and blocker_state(state, e.blocker) is not None]
    words, many = asked_words(tc, ident)
    done = f"{words} can be {PARTICIPLES.get(verb, verb + 'ed')}"
    clauses = reason_clauses(tc, met, state, met=True, asked=asked, every=every)
    if not clauses:
        return f"nothing is in the way, so {done}", []
    return f"{_and(clauses)}, so {done}", why_list(tc, met, state)


def _copy_state(state: FrameState) -> FrameState:
    return {key: InstState(state=inst.state, placement=inst.placement)
            for key, inst in state.items()}


def _apply_effect(tax: Taxonomy, instances: dict[str, InstanceRec],
                  sim: FrameState, target: str, verb: str) -> None:
    """Apply one verb to a simulated state, with the spec-3.3 attached cascade."""
    rec = instances.get(target)
    cls = rec.cls if rec is not None else (
        CABLE_CLASS if target.startswith(CABLE_PREFIX) else None)
    if cls is None:
        return
    effect = tax.apply_verb(cls, rec.attrs if rec is not None else {}, verb)
    if effect is None or effect[0] != STATE_ATTR:
        return
    inst = sim.get(target)
    if inst is None:
        inst = sim[target] = InstState(state=tax.default_state(cls),
                                       placement="in_chassis")
    inst.state = effect[1]
    if effect[1] != REMOVED or (rec is not None and rec.cls == "connector"):
        return
    for key, child in instances.items():
        if child.attached and child.parent == target and key in sim:
            sim[key].state = REMOVED


def illegal_steps(ctx: DesktopCtx, edges: list[Edge],
                  tax: Taxonomy) -> dict[int, list[str]]:
    """``step -> the edges its logged action broke`` (spec 7.4, first check).

    The same replay ``constraints --validate`` prints, kept as data rather than
    as prose so the export can act on it: a step listed here is one where the
    log and the graph contradict each other, and V4/V5/V6/V16 refuse to speak
    about the moment before it.

    The state it folds is :meth:`~tda.core.export.coco.DesktopCtx.state_at` --
    **the same one every task reads**, hand-written events included. Folding the
    derived events only, as an independent replay would, is how a desktop whose
    annotator corrected the log by hand came to be judged against a history
    nobody had.
    """
    by_target: dict[str, list[Edge]] = {}
    for edge in active_edges(edges):
        by_target.setdefault(edge.target, []).append(edge)
    out: dict[int, list[str]] = {}
    for step in sorted({a.step for a in ctx.actions}):
        sim = _copy_state(ctx.state_at(step - 1))
        for action in sorted((a for a in ctx.actions if a.step == step),
                             key=lambda a: a.idx):
            if action.result != "success":
                continue
            bad = unmet(
                applicable_preconditions(by_target.get(action.target, []), action),
                sim, "required")
            if bad:
                out.setdefault(step, []).extend(e.label() for e in bad)
            _apply_effect(tax, ctx.instances, sim, action.target, action.verb)
    return out


def unresolved_action_targets(ctx: DesktopCtx) -> list[str]:
    """Targets of successful actions this export cannot name (C3).

    ``?``, ``connector.?`` and friends: a step whose sheet did not say which part
    was operated. They are not instances, so the history V10 reports is
    incomplete by exactly those actions -- and V10's metric is exhaustive, so an
    incomplete history is a wrong answer rather than a partial one.
    """
    return sorted({a.target for a in ctx.actions
                   if a.result == "success" and ctx.exportable(a.step)
                   and target_class(ctx, a.target) is None})


def exclusion_reason(ctx: DesktopCtx, edges: list[Edge],
                     graph_version: Optional[str]) -> Optional[str]:
    """Why this desktop's graph-derived tasks are refused, or ``None`` (C2).

    Three reasons, and all three are the same reason: the blocked and permitted
    sets would not be trustworthy.

    * ``no_graph_version`` / ``no_constraint_edges`` -- a desktop with no graph
      is not one whose affordance questions are easy, it is one whose every
      answer would be "nothing is blocked", the most confidently wrong ground
      truth this export could ship;
    * ``deadlock`` -- :func:`~tda.core.graph_plan.find_deadlocks` finds a ring of
      actions each of which waits on the next (spec 7.4). Nothing in that ring
      can ever be done, so the graph contradicts a teardown that demonstrably
      happened, and "what can be done now" has no honest answer while it stands.
      Required edges only: a ``recommended`` edge is a preference and never
      makes anything impossible.

    A desktop carrying more Label Studio drafts than resolved instances used to
    be refused here too. It is not a reason: ``ls:*`` rows are reference
    geometry somebody traced in a parallel table, they are already out of
    :attr:`~tda.core.export.coco.DesktopCtx.instances`, and the five desktops it
    caught (D13, D18, D19, D24, D33) carry 41-49 hard edges each and a complete
    instance table.
    """
    if graph_version is None:
        return "no_graph_version"
    if not active_edges(edges):
        return "no_constraint_edges"
    if find_deadlocks(edges, ctx.instances, ctx.tax, necessity="required",
                      initial=initial_overrides(ctx.events)):
        return "deadlock"
    return None


# --------------------------------------------------------------------------- #
# record assembly
# --------------------------------------------------------------------------- #
def evidence(bboxes: dict[str, list], tc: Optional[TaskCtx] = None,
             attrs: Optional[dict] = None, **extra) -> dict:
    """The instance keys, boxes and per-instance attributes an answer rests on.

    Boxes are the **visible**-mask box of the compiled row (spec 3.4); the COCO
    export additionally ships the amodal shape, which is where a consumer that
    wants the whole part should look.

    ``attributes.implied`` mirrors the COCO export's field (spec 8.1): the
    instance was created by :mod:`tda.core.implied` because the desktop plainly
    has one and its log never touched it, not because a step named it.
    """
    names = sorted(set(bboxes) | set(attrs or {}))
    table: dict[str, dict] = {}
    for name in names:
        entry = dict((attrs or {}).get(name) or {})
        if tc is not None:
            rec = tc.ctx.instances.get(name)
            entry.setdefault("implied", bool(rec is not None and is_implied(rec)))
        table[name] = entry
    out = {"instances": names, "bboxes": bboxes}
    if table:
        out["attributes"] = table
    out.update(extra)
    return out


def record(tc: TaskCtx, task: str, rec_id: str, step: int,
           frames: Sequence[FrameData], question: str, template_index: int,
           answer: dict, check: dict, evid: dict, rationale: dict, verified: bool,
           *, views: Optional[Sequence[str]] = None,
           options: Optional[Sequence[dict]] = None,
           negative: Optional[str] = None,
           truth_source: Optional[str] = None,
           view_confirmed: Optional[bool] = None,
           extra: Optional[dict] = None,
           marks: Optional[Sequence[dict]] = None,
           mark_names: Optional[dict] = None,
           terms: Optional[Sequence[str]] = None,
           state_terms: Optional[Sequence[str]] = None,
           moment: Optional[Sequence[tuple[str, str, int]]] = None,
           hints: Optional[Sequence[dict]] = None) -> dict:
    """One JSONL record, split into what the model sees and what grades it.

    ``prompt`` is everything a model may be shown: the task id, the **opaque**
    image ids, the question and, where the task has them, the options. It
    carries no step number, no view, no desktop, no instance key and no field of
    the answer. ``label`` is everything else.

    ``answer_check`` is not decoration: it is the promise that an independent
    checker can re-derive this answer, and a type outside :data:`CHECK_TYPES`
    raises here rather than shipping a question nobody can grade.

    ``model_family`` travels with every record so a platform-disjoint split can
    be made (spec 8.1), and ``split`` (W3b) is the desktop's own
    ``desktop.split`` as written by ``cli set-split`` -- copied, never invented,
    ``null`` while no split is cut. Nothing else in a record depends on it.
    """
    if check.get("type") not in CHECK_TYPES:
        raise ValueError(f"{rec_id}: answer_check type {check.get('type')!r} "
                         f"is not one a checker can execute")
    if not rationale.get("text"):
        # every record carries a reasoning chain in words (owner 2026-11-04): what is seen,
        # which rule applies, so the answer -- the training side product
        rationale = dict(rationale, text=default_reasoning(tc, task, step, answer, check))
    if truth_source is not None and truth_source not in TRUTH_SOURCES:
        raise ValueError(f"{rec_id}: truth_source {truth_source!r} is not one of "
                         f"{', '.join(TRUTH_SOURCES)}")
    if marks and task in UNMARKED_TASKS:
        # the box would be the answer (V1, V8)
        raise ValueError(f"{rec_id}: {task} never carries marks")
    prompt: dict[str, Any] = {
        "task": task,
        "images": [f.image_id for f in frames],
        # "the PSU cannot be judged ..." starts a sentence
        "question": question[:1].upper() + question[1:],
    }
    if options is not None:
        # V6's options name the whole action in words ("unplug" the connectors): `say_verb`
        prompt["options"] = [{"verb": o.get("say_verb", o["verb"]),
                              "target_label": o["target_label"]} for o in options]
    if marks:
        prompt["marks"] = [dict(m) for m in marks]
    if hints:
        # V3's roi form (owner 2026-11-04): a dashed box, another colour than the marks, no
        # tag -- "the change happened inside it"; drawn by the converter like a mark
        prompt["hint_regions"] = [{"image": int(h["image"]), "bbox": list(h["bbox"])}
                                  for h in hints]
    if terms:
        prompt["terms"] = sorted(set(terms))
    if state_terms:
        prompt["state_terms"] = sorted(set(state_terms))
    if moment:
        # one entry per image: which camera, and which moment of the question
        prompt["image_legend"] = [{"view": v, "moment": m} for v, m, _ in moment]
    label: dict[str, Any] = {
        "readable_id": rec_id,
        "answer": answer,
        "answer_check": check,
        "evidence": evid,
        "rationale": rationale,
        "desktop": tc.desktop,
        "step": int(step),
        "view": tc.view,
        "layer": LAYERS[task],
        "template_id": f"{task}.{template_index + 1}",
        "tier": tc.tier,
        "verified": bool(verified),
        "graph_version": tc.graph_version,
        "model_family": tc.meta.get("model_family"),
        "chassis_type": tc.meta.get("chassis_type"),
        # the desktop's split (W3b; W3a's `cli set-split` writes it): public
        # metadata, label side only, so the eval set filters to test desktops
        # without another lookup. Null until a split is cut.
        "split": tc.meta.get("split") or None,
    }
    if views:
        label["views"] = list(views)
    if marks and mark_names:
        # label side: what each mark tag outlines, in words and by key -- the text-only
        # references say "part A (a RAM module)" and tag the state table's rows
        label["mark_names"] = {tag: dict(entry) for tag, entry in sorted(mark_names.items())}
    if moment:
        label["views"] = [v for v, _, _ in moment]
        label["image_steps"] = [int(s) for _, _, s in moment]
    if options is not None:
        label["options"] = list(options)
    if negative:
        label["negative"] = negative
    if truth_source:
        label["truth_source"] = truth_source
    # does every image of this record carry confirmed rows (W3b)? A record about
    # an OAK / RealSense frame says `false`: its truth is the state machine, and
    # nothing of that view's masks or visibility went into it
    label["view_confirmed"] = (all(f.confirmed for f in frames)
                               if view_confirmed is None else bool(view_confirmed))
    if extra:
        label.update(extra)
    return {"id": opaque_record_id(rec_id), "prompt": prompt, "label": label}


def _say(tc: TaskCtx, key: str) -> str:
    """A part, a group or a cable in words, no ordinal ("the PSU", "the RAM latches")."""
    if tc.group_by_id(key) is not None:
        return f"the {tc.group_by_id(key).label}"
    if key.startswith(CABLE_PREFIX):
        return cable_words(key)
    return asked_words(tc, key)[0]


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def default_reasoning(tc: TaskCtx, task: str, step: int, answer: dict, check: dict) -> str:
    """The reasoning chain of a record whose generator wrote none: what the picture shows,
    the rule, so the answer -- in visible terms, never an ordinal or an edge."""
    if task == "V1":
        counts: dict[str, int] = {}
        for c in answer.get("components") or []:
            counts[class_label(c["class"], tc.tax)] = counts.get(class_label(c["class"], tc.tax),
                                                                 0) + 1
        seen = _and([f"{n} {plural(label)}" if n > 1 else f"a {label}"
                     for label, n in sorted(counts.items())])
        return (f"The picture shows {seen}. Each visible part is listed with the box around "
                f"what can be seen of it; the chassis holds everything and is not listed.")
    if task == "V2" and "count" in answer:
        role = str(check.get("role") or "").replace("_", " ")
        n = int(answer["count"])
        return (f"The {role} screws still in place are counted on the machine as it stands "
                f"(a screw taken out leaves an empty hole): {n}. So the count is {n}.")
    if task == "V3":
        target = answer.get("target_instance") or answer.get("target")
        words = _say(tc, target) if target else "a part"
        old, new = tc.state(step - 1).get(target), tc.state(step).get(target)
        how = (f": it was {old.state.replace('_', ' ')} and is now {new.state.replace('_', ' ')}"
               if old is not None and new is not None and old.state != new.state else "")
        return (f"Comparing the two moments, {words} changed{how}. So the action carried out "
                f"was: {VERB_WORDS.get(answer.get('verb'), answer.get('verb'))} {words}.")
    if task == "V8":
        if "components" in answer:
            n = len(answer["components"])
            return (f"The picture shows {n} of the parts asked for; each is given by the box "
                    f"around what can be seen of it.")
        target = answer.get("instance")
        words = _say(tc, target) if target in tc.ctx.instances else "the part"
        return f"{_cap(words)} is the part that matches the description; its box is the answer."
    if task == "V10":
        done = _and([f"{VERB_WORDS.get(a['verb'], a['verb'])} {_say(tc, a['target'])}"
                     for a in answer.get("done") or []][:8])
        started = _and([f"{VERB_WORDS.get(a['verb'], a['verb'])} {_say(tc, a['target'])}"
                        for a in answer.get("started") or []])
        text = (f"Compared with the machine at the start, these are done: {done}."
                if done else "Compared with the machine at the start, nothing is done yet.")
        if started:
            text += f" Begun but not finished: {started}."
        return text + (f" That puts the teardown in the {answer.get('progress_bin')} % band of "
                       f"its actions.")
    if task == "V12":
        if answer.get("changed"):
            what = _and([_say(tc, e["target"]) for e in answer.get("events") or []][:4])
            return (f"Comparing the two pictures, {what} changed between the two moments. "
                    f"So yes, the machine changed.")
        return ("Comparing the two pictures, every part is where and how it was; nothing a "
                "picture can show changed. So no, the machine did not change.")
    if task == "V15":
        if "same_moment" in answer:
            if answer["same_moment"]:
                return ("Both pictures show the machine with the same parts in and out, in the "
                        "same states. So they are the same moment.")
            return ("The two pictures disagree about a large part: it is in one and gone or "
                    "moved in the other. So they are not the same moment.")
        target = check.get("instance")
        words = _say(tc, target) if target else "the part"
        return (f"The first view cannot see {words}; the second view shows it "
                f"{str(answer.get('state')).replace('_', ' ')}. So its state is "
                f"{answer.get('state')}, and the second view answers better.")
    if task == "V5":
        return "The actions listed are the ones the log shows can be done from here."
    return "The answer follows from what the picture shows."


def v2_state_text(label: str, many: bool, value: str, screw: bool) -> str:
    """V2's reasoning in words: what the picture shows of the part, so its state."""
    be = "are" if many else "is"
    if screw:
        seen = "still in its hole" if value == "present" else "gone, its hole empty"
        if many:
            seen = "still in their holes" if value == "present" else "gone, their holes empty"
        return f"In the picture {label} {be} {seen}. So the answer is {value}."
    return (f"In the picture {label} {be} {value.replace('_', ' ')}. "
            f"So the state is {value}.")


def frame_id(desktop: int, view: str, step: int) -> str:
    return f"D{desktop:02d}-{view}-s{step:03d}"


def row_verified(rows: Sequence[dict], frame_verified: bool) -> bool:
    """Has a human confirmed the frame *and* every row this answer was read off?"""
    return bool(frame_verified) and all(r.get("status") == VERIFIED for r in rows)


def observed(instance: str, value: Any, tc: TaskCtx,
             seen: dict[str, tuple[dict, list]], step: Optional[int] = None) -> dict:
    """``observe`` when the frame can point at the instance, else ``propagate_state``."""
    if instance not in seen:
        return R.propagate_state(instance, value)
    row, box = seen[instance]
    return R.observe(instance, value, tc.view, box, row.get("visibility"), step=step,
                     occluder=row.get("occluder"))


def seen_as(row: dict) -> dict:
    """The per-instance evidence attributes of a row: its visibility, and its occluder.

    ``occluder`` (task U15, 「被线挡住」) only when the frame names one: the part
    is in its place and something nobody labels -- a cable -- covers it.
    """
    out = {"visibility": row.get("visibility")}
    if row.get("occluder"):
        out["occluder"] = row["occluder"]
    return out


#: How V14 abstains on a part 被线挡住（完全）: in place, and why it is not seen.
HIDDEN_BY = {"cable": "a cable", "other_unlabeled": "something not annotated"}
#: ... and on a part 角度看不到 (task U24): nothing hides it, the camera cannot
#: see it from where it stands (从这个视角看不到).
VIEWPOINT = "viewpoint"
UNSEEN_FROM_HERE = "not visible from this viewpoint"


def unseen_reason(instance: str, row: dict, view: str) -> str:
    """Why this view cannot tell: the label, or -- for a part under a cable -- the cable.

    ``"X is occluded_full in view scan"`` for a bare label; a part the frame
    says is 被线挡住 is ``"X is in place, hidden by a cable, in view scan"``
    (X 在原位，被线缆挡住); one 角度看不到 (task U24) is ``"X is in place, not
    visible from this viewpoint, in view oak2"`` (X 在原位，从这个视角看不到).
    """
    occluder = row.get("occluder")
    if occluder == VIEWPOINT:
        return f"{instance} is in place, {UNSEEN_FROM_HERE}, in view {view}"
    if occluder:
        return (f"{instance} is in place, hidden by "
                f"{HIDDEN_BY.get(str(occluder), str(occluder))}, in view {view}")
    return f"{instance} is {row.get('visibility')} in view {view}"


def question_label(tc: TaskCtx, instance: str,
                   frame: Optional[FrameData] = None) -> Optional[str]:
    """How a question names one part: "the RAM module outlined in red" (spec 3.2).

    Never an ordinal: "RAM module 3" names nothing a viewer can find. The part is outlined
    on ``frame`` (:class:`~tda.core.export.vlm_groups.Namer`); ``None`` when it cannot be
    -- the question is then not asked (``unmarkable_part``). A generator that names
    several parts, or wants the marks, uses a :class:`Namer` itself.
    """
    namer = Namer(tc, frame)
    return namer.say(instance) if namer.part(instance) is not None else None


def unique_on_desktop(tc: TaskCtx, key: str) -> bool:
    """Is ``key`` the one part of its match class on this desktop (so its class names it)?"""
    mc = match_class(tc.ctx, key)
    return mc is not None and sum(
        1 for k in tc.ctx.instances if match_class(tc.ctx, k) == mc) == 1


def class_words(tc: TaskCtx, keys: Sequence[str]) -> str:
    """"the PSU" / "the motherboard screws": a class named in words, no mark, no ordinal."""
    return f"the {class_phrase(tc.ctx, keys[0], many=len(keys) > 1)}"


def plain_name(tc: TaskCtx, key: str) -> str:
    """How anything a person or a model reads names a part: **never an ordinal** (owner
    2026-11-04, "CPU 1", "connector 3" name nothing a viewer can find).

    A group id is its group label ("CPU cooler screws"); a part alone of its match class on
    the desktop is its class ("CPU socket lever"); a part an unordered locator singles out
    (:func:`_referring`: "screw that fastens the PSU") is that; a member of a group is "one of
    the <group label>"; any other part "one of the N <class>" -- it is one of several and no
    word tells which. A ``cable:*`` node is its cable ("PSU cable"). The oracle carries it as
    ``instances[*].name``; a mark tag ("part A") is the reader's business, not this one's.
    """
    group = tc.group_by_id(key)
    if group is not None:
        return group.label
    if key.startswith(CABLE_PREFIX):
        return cable_words(key)[len("the "):]
    if key not in tc.ctx.instances:
        return key
    referring = _referring(tc, key)
    if referring is not None:
        phrase = referring[0]
        return phrase[len("the "):] if phrase.startswith("the ") else phrase
    group = tc.group_of(key)
    if group is not None:
        return f"one of the {group.label}"
    mc = match_class(tc.ctx, key)
    peers = sum(1 for k in tc.ctx.instances if match_class(tc.ctx, k) == mc)
    return f"one of the {peers} {class_phrase(tc.ctx, key, many=True)}"


def state_is_askable(tax: Taxonomy, cls: str) -> bool:
    """Is "what state is it in?" a real question for this class?

    Only when the class has at least two states the annotation actually tracks
    (spec 6.2's ``needs_mask`` table). A ``connector`` has one -- ``plugged`` --
    because an unplugged connector is not tracked at all, so every frame that
    carries one answers "plugged" and the question teaches nothing; a
    ``chassis`` is always ``present``.
    """
    try:
        states = tax.states_of(cls)
    except KeyError:
        return False
    return sum(1 for s in states if tax.needs_mask(cls, s, "in_chassis")) >= 2


# --------------------------------------------------------------------------- #
# V1 -- what is visible, and where
# --------------------------------------------------------------------------- #
V1_QUESTIONS = (
    "List every component visible in this image, with a bounding box for each.",
    "Which components can you see in this image? Give each one a bounding box.",
    "Name the parts visible in this photo of the desktop PC and localise them.",
    "Enumerate the visible components and give their boxes.",
)


def gen_v1(tc: TaskCtx, step: int) -> Iterator[dict]:
    """Every pointable instance of a verified frame, except the chassis.

    The chassis is in every frame, fills most of it, and is named by the task --
    a model that answers "a chassis, somewhere near the middle" scores on it
    without looking. It is excluded from the exhaustive set rather than left in
    to inflate the F1 (the COCO export still carries it).
    """
    frame = tc.frame(step)
    if frame is None or not frame.verified:
        return
    components, bboxes, steps, used, attrs = [], {}, [], [], {}
    for instance, (row, box) in frame.pointable.items():
        cls = tc.ctx.cls_of(instance)
        if cls is None or cls == CHASSIS_CLASS:
            continue
        components.append({"class": cls, "instance": instance, "bbox": box,
                           "placement": row.get("placement")})
        bboxes[instance] = box
        attrs[instance] = seen_as(row)
        used.append(row)
        steps.append(R.observe(instance, cls, tc.view, box, row.get("visibility"),
                               occluder=row.get("occluder")))
    if not components:
        return
    rec_id = f"V1-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V1_QUESTIONS, rec_id)
    yield record(
        tc, "V1", rec_id, step, [frame], question, index,
        {"components": components},
        {"type": "boxes", "field": "components", "id_field": "instance",
         "box_field": "bbox", "exhaustive": True, "exclude_classes": [CHASSIS_CLASS],
         "geometry": "visible_mask_bbox", "metric": "set_f1+box_iou",
         "iou_threshold": 0.5},
        evidence(bboxes, tc, attrs), R.rationale(steps),
        row_verified(used, frame.verified),
    )


# --------------------------------------------------------------------------- #
# V2 -- per-instance state and fastener counts
# --------------------------------------------------------------------------- #
V2_STATE_QUESTIONS = (
    "What is the state of {label}?",
    "In this image, what state is {label} in?",
    "Report the current state of {label}.",
)
#: A screw is asked whether it is still there, never how tight it is (W3b).
V2_PRESENCE_QUESTIONS = (
    "Is {label} still in place, or has it been removed? Answer present or removed.",
    "In this image, is {label} still there (present) or already taken out (removed)?",
    "Report whether {label} is present or removed.",
)
V2_COUNT_QUESTIONS = (
    "How many {role} screws are still in place?",
    "Count the {role} screws that are still in place in this image.",
)


def gen_v2(tc: TaskCtx, step: int) -> Iterator[dict]:
    frame = tc.frame(step)
    if frame is None or not frame.verified:
        return
    yield from _v2_states(tc, frame)
    yield from _v2_hidden(tc, frame)
    yield from _v2_counts(tc, frame)


#: V2's extra answer (V14 merged into V2, owner 2026-11-04): this picture cannot tell.
CANNOT_TELL = "cannot_tell"
#: The visibilities of a part in its place that this image cannot judge.
HIDDEN_VISIBILITY = ("occluded_full", "too_small", "out_of_view")
#: Why a picture cannot tell, in words.
HIDDEN_WORDS = {"occluded_full": "something covers it completely",
                "too_small": "it is too small to make out", "out_of_view": "it is outside the picture"}
#: A hidden-part question on about one frame in this many (a modest share of V2's states).
V2_HIDDEN_EVERY = 3


def v2_allowed(tc: TaskCtx, cls: str) -> list[str]:
    """The answer space of a V2 state question: the class's states (present / removed for a
    screw) and ``cannot_tell``."""
    return ([*PRESENCE_STATES] if cls == "screw" else sorted(tc.tax.states_of(cls))) \
        + [CANNOT_TELL]


def hidden_why(row: dict) -> str:
    if row.get("occluder") == "cable":
        return "a cable hides it completely"
    if row.get("occluder") == VIEWPOINT:
        return UNSEEN_FROM_HERE
    return HIDDEN_WORDS.get(str(row.get("visibility")), "it cannot be seen")


def _v2_hidden(tc: TaskCtx, frame: FrameData) -> Iterator[dict]:
    """"What state is <a part this view cannot see> in?" -- gold ``cannot_tell`` (V14 merged
    into V2). The part is in its place and named in words (no mark: a box would say where
    it is): a part alone of its class, or a part group every present member of which is
    hidden. At most one per frame, on about one frame in :data:`V2_HIDDEN_EVERY`."""
    seed = tc.seed(frame.step, "V2hidden")
    if zlib.crc32(seed.encode("utf-8")) % V2_HIDDEN_EVERY:
        return
    state = tc.state(frame.step)
    units: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for key in sorted(frame.rows):
        row = frame.rows[key]
        cls = tc.ctx.cls_of(key)
        if key in seen or cls is None or not tc.askable(key) or not state_is_askable(tc.tax, cls):
            continue
        if row.get("visibility") not in HIDDEN_VISIBILITY or not present(state, key):
            continue
        if is_captive_screw(tc.ctx, key) or row.get("placement") not in (None, "in_chassis"):
            continue
        group = tc.group_of(key)
        if group is not None:
            members = [m for m in group.members if present(state, m)]
            if all((frame.rows.get(m) or {}).get("visibility") in HIDDEN_VISIBILITY
                   for m in members) and len(members) >= 2:
                units.append((group.gid, members))
                seen.update(members)
            continue
        if unique_on_desktop(tc, key):
            units.append((key, [key]))
            seen.add(key)
    if not units:
        return
    ident, members = stable_order(units, seed, lambda u: u[0])[0]
    first = members[0]
    cls = tc.ctx.cls_of(first)
    screw = cls == "screw"
    # outlined where its whole shape is known (a part wholly covered in its place: its
    # amodal box), as every V2 question is -- else named in words
    namer = Namer(tc, frame)
    ref = namer.group(ident, members) if ident != first else namer.part(first)
    words = (namer.say(ref) if ref is not None else
             f"the {tc.group_by_id(ident).label}" if tc.group_by_id(ident) is not None
             else asked_words(tc, first)[0])
    many = len(members) > 1
    rec_id = f"V2-{frame_id(tc.desktop, tc.view, frame.step)}-hidden-{ident}"
    templates = ((V2_PRESENCE_QUESTIONS_MANY if many else V2_PRESENCE_QUESTIONS) if screw
                 else (V2_STATE_QUESTIONS_MANY if many else V2_STATE_QUESTIONS))
    index, template = pick(templates, rec_id)
    check = {"type": "exact", "derive": "instance_state", "instance": ident, "class": cls,
             "fields": ["state"], "allowed": v2_allowed(tc, cls), "hidden": True,
             "visibility": {k: frame.rows[k].get("visibility") for k in members}}
    if many:
        check["members"] = list(members)
    if screw:
        check["state_space"] = "presence"
    start = tc.state(0).get(first)
    prior = visible_state(tc.ctx, first, start.state if start is not None
                          else tc.tax.default_state(cls))
    why = hidden_why(frame.rows[first])
    if many:
        why = why.replace(" it ", " them ").replace(" it.", " them.").replace(
            "covers it", "covers them").replace("it is", "they are")
    be = "are" if many else "is"
    text = (f"{words[:1].upper() + words[1:]} {be} in place but not visible in this picture: "
            f"{why}. A state has to be seen to be told, so it cannot be told from this image.")
    yield record(
        tc, "V2", rec_id, frame.step, [frame], template.format(label=words), index,
        {"state": CANNOT_TELL}, check,
        evidence({}, tc, {k: seen_as(frame.rows[k]) for k in members}),
        dict(R.rationale([R.propagate_state(k, None) for k in members],
                         R.abstain("; ".join(unseen_reason(k, frame.rows[k], tc.view)
                                             for k in members))), text=text),
        row_verified([frame.rows[k] for k in members], frame.verified),
        negative="cannot_tell", extra={"class_prior_answer": prior},
        terms=[cls], state_terms=[cls],
        marks=namer.marks() if ref is not None else None,
        mark_names=namer.names() if ref is not None else None,
    )


V2_STATE_QUESTIONS_MANY = (
    "What state are {label} in?",
    "In this image, what state are {label} in?",
    "Report the current state of {label}.",
)
V2_PRESENCE_QUESTIONS_MANY = (
    "Are {label} still in place, or have they been removed? Answer present or removed.",
    "In this image, are {label} still there (present) or already taken out (removed)?",
    "Report whether {label} are present or removed.",
)


def _v2_units(tc: TaskCtx, frame: FrameData) -> list[tuple[str, list[str]]]:
    """What V2 asks about on this frame: ``(id, members)`` -- a part, or a group (spec 6.4).

    A group whose visible members all show the same state is asked once, by its group id,
    every member outlined; a group whose members differ is asked member by member, each
    with its own mark. A part that cannot be outlined -- removed, lying on the bench -- is
    not asked (``unmarkable_part``).
    """
    state = tc.state(frame.step)
    askable: list[str] = []
    for instance in frame.pointable:
        cls = tc.ctx.cls_of(instance)
        if cls is None or state.get(instance) is None or not state_is_askable(tc.tax, cls):
            continue
        if is_captive_screw(tc.ctx, instance):
            # it stays on its fan, tight or not, until the cooler leaves: there
            # is no state a picture of it can show (W3b, user data fact b)
            tc.skip("V2", "captive_screw")
            continue
        if mark_box(tc, frame, instance) is None:
            tc.skip("V2", "unmarkable_part")
            continue
        askable.append(instance)
    units: list[tuple[str, list[str]]] = []
    done: set[str] = set()
    for instance in askable:
        if instance in done:
            continue
        group = tc.group_of(instance)
        members = [k for k in askable if group is not None and tc.group_of(k) is group]
        if len(members) < 2:
            units.append((instance, [instance]))
            done.add(instance)
            continue
        done.update(members)
        shown = {visible_state(tc.ctx, k, state[k].state) for k in members}
        if len(shown) == 1:
            units.append((group.gid, members))
        else:
            units.extend((k, [k]) for k in members)
    return units


def _v2_states(tc: TaskCtx, frame: FrameData) -> Iterator[dict]:
    state = tc.state(frame.step)
    for ident, members in _v2_units(tc, frame):
        first = members[0]
        cls = tc.ctx.cls_of(first)
        screw = cls == "screw"
        value = visible_state(tc.ctx, first, state[first].state)
        namer = Namer(tc, frame)
        ref = namer.group(ident, members) if ident != first else namer.part(first)
        many = namer.many(ref)
        rec_id = f"V2-{frame_id(tc.desktop, tc.view, frame.step)}-state-{ident}"
        templates = ((V2_PRESENCE_QUESTIONS_MANY if many else V2_PRESENCE_QUESTIONS) if screw
                     else (V2_STATE_QUESTIONS_MANY if many else V2_STATE_QUESTIONS))
        index, template = pick(templates, rec_id)
        # the answer space has a "cannot tell from this image" (V14 merged into V2)
        check = {"type": "exact", "derive": "instance_state", "instance": ident,
                 "class": cls, "fields": ["state"], "allowed": v2_allowed(tc, cls)}
        if ident != first:
            check["members"] = list(members)
        if screw:
            # tightness was never annotated: a screw is asked present / removed
            check["state_space"] = "presence"
        # the class prior: where this part starts, as a picture shows it --
        # what "the PSU screw" alone would answer (W3b, per desktop)
        start = tc.state(0).get(first)
        prior = visible_state(tc.ctx, first,
                              start.state if start is not None
                              else tc.tax.default_state(cls))
        rows = [frame.pointable[k][0] for k in members]
        yield record(
            tc, "V2", rec_id, frame.step, [frame],
            template.format(label=namer.say(ref)), index,
            {"state": value}, check,
            evidence({k: frame.pointable[k][1] for k in members}, tc,
                     {k: seen_as(frame.pointable[k][0]) for k in members}),
            dict(R.rationale([R.observe(k, value, tc.view, frame.pointable[k][1],
                                        frame.pointable[k][0].get("visibility"),
                                        occluder=frame.pointable[k][0].get("occluder"))
                              for k in members]),
                 text=v2_state_text(namer.say(ref), many, value, screw)),
            row_verified(rows, frame.verified),
            extra={"class_prior_answer": prior}, marks=namer.marks(), mark_names=namer.names(),
            terms=[cls], state_terms=[cls],
        )


def counted_roles(tc: TaskCtx, frame: FrameData) -> list[str]:
    """Screw roles this frame asks a V2 counting question about.

    Captive screws count for nothing (W3b): they stay on their part, so "how
    many are still in place" about them is a question about the cooler.
    """
    return sorted({
        str(rec.attrs.get("role") or "other")
        for key, rec in tc.ctx.instances.items()
        if rec.cls == "screw" and key in frame.pointable and not rec.attrs.get("captive")
    })


def _v2_counts(tc: TaskCtx, frame: FrameData) -> Iterator[dict]:
    """"How many <role> screws are still in place?" -- counted on the state machine.

    The count covers every screw of that role on the machine, not only the ones
    this view can see, because that is what the question asks; the evidence
    lists the boxes of those that are visible here, and the ones that are not
    enter the chain as ``propagate_state``.
    """
    state = tc.state(frame.step)
    initial = tc.state(0)
    for role in counted_roles(tc, frame):
        # "still in place" = present (W3b: tightness was never annotated, a
        # drawn screw is a fastened one); captive screws are not counted
        fastened = [
            key for key, rec in sorted(tc.ctx.instances.items())
            if rec.cls == "screw" and str(rec.attrs.get("role") or "other") == role
            and not rec.attrs.get("captive")
            and visible_state(tc.ctx, key, state.get(key).state if state.get(key)
                              else None) == "present"
        ]
        bboxes = {k: frame.pointable[k][1] for k in fastened if k in frame.pointable}
        rec_id = f"V2-{frame_id(tc.desktop, tc.view, frame.step)}-count-{role}"
        index, template = pick(V2_COUNT_QUESTIONS, rec_id)
        yield record(
            tc, "V2", rec_id, frame.step, [frame],
            template.format(role=role.replace("_", " ")), index,
            {"count": len(fastened)},
            {"type": "exact", "derive": "screw_count", "role": role,
             "state": "present", "state_space": "presence", "captive": False,
             "fields": ["count"],
             # owner 2026-11-04: exact 1.0, off by one 0.8, else 0 (vlm_check.COUNT_SCORES);
             # the exact match is the scorer's extra row
             "credit": {"exact": 1.0, "off_by_one": 0.8}},
            evidence(bboxes, tc),
            R.rationale([observed(k, "present", tc, frame.pointable) for k in fastened]),
            row_verified([frame.pointable[k][0] for k in fastened
                          if k in frame.pointable], frame.verified),
            # the prior: how many of them the machine starts with
            extra={"class_prior_answer": sum(
                1 for key, rec in tc.ctx.instances.items()
                if rec.cls == "screw" and str(rec.attrs.get("role") or "other") == role
                and not rec.attrs.get("captive")
                and visible_state(tc.ctx, key, initial[key].state
                                  if key in initial else None) == "present")},
        )


#: The most one answer may make up of a class's V2 questions on one desktop
#: after balancing (W3b). The day-1 export had 59 884 V2 records, 64 % of the
#: file, and the class alone answered 80.6 % of them.
V2_PRIOR_CAP = 0.5


def v2_prior_key(record: dict) -> Optional[tuple[str, str]]:
    """``("state", class)`` or ``("count", role)``: what a text-only prior keys on."""
    check = record["label"]["answer_check"]
    if check.get("derive") == "instance_state":
        return ("state", str(check.get("class")))
    if check.get("derive") == "screw_count":
        return ("count", str(check.get("role")))
    return None


def _v2_value(record: dict) -> Any:
    answer = record["label"]["answer"]
    return answer.get("state") if "state" in answer else answer.get("count")


def _v2_draw_key(record: dict) -> str:
    """What a V2 record's place in the thinning order is seeded by.

    Desktop, the instance (or the screw role a count is about) and the step --
    nothing about the rest of the export, so a desktop exported alone and the
    same desktop inside 1-66 keep exactly the same questions.
    """
    label, check = record["label"], record["label"]["answer_check"]
    what = check.get("instance") or f"role:{check.get('role')}"
    return f"V2thin|D{label['desktop']}|{what}|s{label['step']}"


def balance_v2(records: list[dict], cap: float = V2_PRIOR_CAP) -> tuple[list[dict], dict]:
    """Thin V2 so no answer makes up more than ``cap`` of a class's questions.

    Per desktop and per key (:func:`v2_prior_key`: a class, or a screw role for
    the counts): every answer value is kept at most as often as all the other
    values together (``cap = 0.5``), the survivors chosen by a hash of
    :func:`_v2_draw_key` -- per desktop, so the result never depends on which
    other desktops are exported (review M). A key whose every answer is the
    same keeps none: those questions would only measure the prior.

    The **class prior** a record is labelled with is set by the generator, not
    here: the instance's state at the start of the teardown as a picture shows
    it (a screw present, a latch closed, a part installed) or, for a count, the
    number of non-captive screws of that role the machine starts with. It is
    what a model that reads only "the PSU screw" would answer, and it depends
    on the desktop alone. ``prior_is_correct`` is whether the answer is that;
    the scorer's V2 headline is accuracy on ``prior_is_correct = false``.
    Records of other tasks pass through untouched, in order.
    """
    pools: dict[tuple, list[dict]] = {}
    for rec in records:
        if rec["prompt"]["task"] == "V2":
            key = v2_prior_key(rec)
            if key is not None:
                pools.setdefault((int(rec["label"]["desktop"]), key), []).append(rec)
    keep: set[str] = set()
    stats: dict[str, dict] = {}
    share = cap / (1.0 - cap) if cap < 1.0 else float("inf")
    for (desktop, key), pool in sorted(pools.items()):
        by_value: dict[str, list[dict]] = {}
        for rec in pool:
            by_value.setdefault(json_key(_v2_value(rec)), []).append(rec)
        kept = 0
        for value, rows in sorted(by_value.items()):
            others = len(pool) - len(rows)
            allowed = len(rows) if share == float("inf") else int(others * share)
            chosen = sorted(rows, key=lambda r: hashlib.sha256(
                _v2_draw_key(r).encode("utf-8")).hexdigest())[:allowed]
            keep.update(r["id"] for r in chosen)
            kept += len(chosen)
        for rec in pool:
            prior = rec["label"].get("class_prior_answer")
            rec["label"]["prior_is_correct"] = (json_key(_v2_value(rec)) == json_key(prior))
        row = stats.setdefault(f"{key[0]}:{key[1]}", {"before": 0, "after": 0,
                                                       "prior_correct_before": 0})
        row["before"] += len(pool)
        row["after"] += kept
        row["prior_correct_before"] += sum(1 for r in pool if r["label"]["prior_is_correct"])
    out = [r for r in records
           if r["prompt"]["task"] != "V2" or v2_prior_key(r) is None or r["id"] in keep]
    return out, stats


def json_key(value: Any) -> str:
    """A hashable, order-stable spelling of an answer value."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# V3 -- what happened between two frames
# --------------------------------------------------------------------------- #
V3_QUESTIONS = (
    "What action was carried out between the pictures before and the pictures after?",
    "Compare the machine before and after the action: which action was carried out?",
    "What did the operator just do between the moment before and the moment after?",
)


def _slots(tc: TaskCtx, action: ActionRec) -> dict:
    return {"verb": action.verb, "target_class": target_class(tc.ctx, action.target),
            "target_instance": action.target, "tool": graded_tool(action.tool)}


def _seen_px(frame: FrameData, keys: Sequence[str]) -> int:
    """Visible pixels of ``keys`` on one frame, summed (0 for a part it cannot point at)."""
    total = 0
    for key in keys:
        if key in frame.pointable:
            total += int(frame.pointable[key][0].get("area") or 0)
    return total


def gen_v3(tc: TaskCtx, step: int) -> Iterator[dict]:
    frame, before = tc.frame(step), tc.frame(step - 1)
    if frame is None or before is None:
        return
    if tc.ctx.step_type(step) in NO_CHANGE_STEP_TYPES:
        return
    named = tc.named_actions(step)
    if not named:
        return
    # unscrewing a captive screw leaves it on its fan, looking exactly as it
    # did (W3b, user data fact b): not an action a pair of pictures can show
    named = [a for a in named if not invisible_action(tc.ctx, a)]
    if not named:
        tc.skip("V3", CAPTIVE_LOOSENED)
        return
    # a view without confirmed rows (W3b): a screw or a plug may be invisible
    # at that resolution, and "you cannot tell from here" is then the honest
    # answer -- so only a large part's action is asked, and the record says so
    confirmed = before.confirmed and frame.confirmed
    big = big_changes(tc, step - 1, step)
    target = named[0].target
    if not confirmed and tc.ctx.cls_of(target) not in BIG_PARTS:
        tc.skip("V3", "unconfirmed_view_small_change")
        return
    if confirmed and tc.ctx.cls_of(target) not in BIG_PARTS:
        # an action nobody can see is no V3 question (spec 3.6): the part -- or its group
        # taken together -- shows at least V3_MIN_PX pixels before or after
        group = tc.group_of(target)
        keys = list(group.members) if group is not None else [target]
        seen = max(_seen_px(before, [target]), _seen_px(frame, [target]),
                   _seen_px(before, keys), _seen_px(frame, keys))
        if seen < V3_MIN_PX:
            tc.skip("V3", "invisible_action")
            return
    answer = _slots(tc, named[0])
    if len(named) > 1:  # a compound step: keep every slot set, first one on top
        answer["actions"] = [_slots(tc, a) for a in named]
    # `unknown` is not a fact about the machine and is never graded (I3)
    fields = ["verb", "target_class", "target_instance"]
    if answer["tool"] is not None:
        fields.append("tool")

    source_step, source = step - 1, before.pointable
    if target not in before.pointable and target in frame.pointable:
        source_step, source = step, frame.pointable
    bboxes = {target: source[target][1]} if target in source else {}
    # the part acted on is the answer and nothing else (owner 2026-11-03): a part
    # attached to it (a RAM latch on the motherboard, a screw on the cooler) leaves
    # with it but is not what was taken out; only a member of the target's own group
    # of interchangeable parts stands for it (`target_group`)
    group = tc.group_of(target)

    stem = f"V3-{frame_id(tc.desktop, tc.view, step)}"
    chain = [R.compare_frames([step - 1, step], tc.view),
             observed(target, answer["verb"], tc, source, step=source_step)]
    frames = [*tc.moment(step - 1), *tc.moment(step)]
    moment = tc.legend((step - 1, "before"), (step, "after"))
    # the two forms of one step (owner 2026-11-04, "两个都要"), paired like V6's: `free` (the
    # question as it was, now also asking where the machine changed) keeps the record id the
    # eval subset knows; `roi` draws a larger hint box around that place and asks the action
    # (the region is a box on the top-down scanner image: a unit of another camera asks none)
    region = (v3_region(tc, before, frame, [a.target for a in named], group)
              if tc.own_view == VIEWS[0] else None)
    scanner = [i for i, f in enumerate(frames) if f is before or f is frame]
    pair = opaque_record_id(stem)
    check = {"type": "exact", "derive": "action_slots", "fields": fields,
             "visible_only": True, "accept_targets": [target],
             "target_group": group.gid if group is not None else None}
    verified = row_verified([source[target][0]] if target in source else [],
                            (before if source_step == step - 1 else frame).verified)
    free_answer = dict(answer)
    free_check = dict(check, form="free")
    if region is not None:
        # where the machine changed: the changed part's scanner box before and after, as one
        # box; graded beside the action (region IoU, region hit at IoU >= 0.5), never in it
        free_answer["region"] = region
        free_check.update(region=region, region_iou_threshold=V3_REGION_IOU,
                          region_images=scanner)
    index, question = pick(V3_QUESTIONS, stem)
    if region is not None:
        question = f"{question} {V3_REGION_ASK}"
    yield record(
        tc, "V3", stem, step, frames, question, index, free_answer, free_check,
        evidence(bboxes, tc, from_step=source_step), R.rationale(chain),
        # the grade follows the frame the answer was actually read off, which
        # for a V3 pair is usually the earlier one: the part is gone from the later
        verified, view_confirmed=confirmed,
        extra={"big_change": big, "v3_form": "free", "v3_pair": pair},
        moment=moment,
    )
    if region is None:
        tc.skip("V3", "roi_without_region")
        return
    hint = v3_hint(region, before.hw or frame.hw or _nominal_hw(tc.own_view))
    rec_id = f"{stem}-roi"
    index, question = pick(V3_ROI_QUESTIONS, rec_id)
    yield record(
        tc, "V3", rec_id, step, frames, question, index, dict(answer),
        dict(check, form="roi", hint_region=hint),
        evidence(bboxes, tc, from_step=source_step), R.rationale(chain), verified,
        view_confirmed=confirmed,
        extra={"big_change": big, "v3_form": "roi", "v3_pair": pair,
               "template_id": f"V3-roi.{index + 1}"},
        moment=moment, hints=[{"image": i, "bbox": list(hint)} for i in scanner],
    )


#: The region a V3 ``free`` answer gives is a hit at this IoU with the gold region.
V3_REGION_IOU = 0.5
#: What the ``free`` form adds to its question (owner 2026-11-04).
V3_REGION_ASK = ("Also say where the machine changed: give that region as a box on the "
                 "top-down scanner image.")
V3_ROI_QUESTIONS = (
    "The change happened inside the yellow dashed box drawn on the scanner images. "
    "What action was carried out between the pictures before and the pictures after?",
    "Look inside the yellow dashed box on the scanner images: something there changed "
    "between before and after. Which action was carried out?",
    "Between the moment before and the moment after, the operator acted on something inside "
    "the yellow dashed box drawn on the scanner images. What did the operator do?",
)
#: The ``roi`` hint: the gold region grown this many times around its centre ...
V3_HINT_SCALE = 2.5
#: ... and never smaller than this many pixels a side (clipped to the image).
V3_HINT_MIN = 300


def _box_union(boxes: Sequence[Sequence[float]]) -> Optional[list]:
    boxes = [b for b in boxes if b is not None and len(b) == 4]
    if not boxes:
        return None
    x0 = min(float(b[0]) for b in boxes)
    y0 = min(float(b[1]) for b in boxes)
    x1 = max(float(b[0]) + float(b[2]) for b in boxes)
    y1 = max(float(b[1]) + float(b[3]) for b in boxes)
    return [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]


def v3_region(tc: TaskCtx, before: FrameData, after: FrameData, targets: Sequence[str],
              group: Optional[PartGroup]) -> Optional[list]:
    """V3's gold region: where the machine changed, ``[x, y, w, h]`` on the scanner image.

    The union of the changed parts' scanner boxes before and after (a part gone from the
    later frame keeps its earlier box); where no part acted on has a box on either frame, its
    group's (the screw too small to point at, its fan's other screws are where it was).
    ``None`` on frames without confirmed rows -- no box, no region question.
    """
    def boxes(keys):
        return [f.pointable[k][1] for f in (before, after) for k in keys if k in f.pointable]

    found = boxes(list(dict.fromkeys(targets)))
    if not found and group is not None:
        found = boxes(group.members)
    return _box_union(found)


def v3_hint(region: Sequence[float], hw: tuple[int, int]) -> list:
    """V3's ``roi`` hint box: ``region`` grown :data:`V3_HINT_SCALE` times about its centre, at
    least :data:`V3_HINT_MIN` px a side, moved inside the image where it would cross an edge
    and clipped to it (an image smaller than the box)."""
    H, W = int(hw[0]), int(hw[1])
    x, y, w, h = (float(v) for v in region)
    cx, cy = x + w / 2, y + h / 2
    bw = min(float(W), max(V3_HINT_SCALE * w, float(V3_HINT_MIN)))
    bh = min(float(H), max(V3_HINT_SCALE * h, float(V3_HINT_MIN)))
    x0 = min(max(0.0, cx - bw / 2), W - bw)
    y0 = min(max(0.0, cy - bh / 2), H - bh)
    return [round(x0, 1), round(y0, 1), round(bw, 1), round(bh, 1)]


def _nominal_hw(view: str) -> tuple[int, int]:
    from tda.core.truth_inputs import infer_hw

    try:
        return infer_hw(view)
    except ValueError:
        return (1600, 1600)


# --------------------------------------------------------------------------- #
# V4 -- can this be done now, and what is in the way
# --------------------------------------------------------------------------- #
V4_QUESTIONS = (
    "Can you {phrase} right now? If not, what has to be dealt with first?",
    "Is it possible to {phrase} in this state? Name anything that blocks it.",
    "Would {gerund} work right now? If not, list what is in the way.",
)
V4_FAILED_QUESTIONS = (
    "The operator tried to {phrase} here and could not. Why not?",
    "This attempt at {gerund} failed. What was in the way?",
)


def blocker_ids(tc: TaskCtx, blockers: Sequence[str]) -> list[str]:
    """Blockers as **blocker groups** (spec 3.3): a group's members are one requirement --
    "loosen the CPU cooler screws" -- whatever the log says about each of them."""
    return sorted({(tc.group_of(b).gid if tc.group_of(b) is not None else b)
                   for b in blockers})


def captive_dependent(tc: TaskCtx, verb: str, targets: Sequence[str],
                      state: FrameState) -> bool:
    """Does a V4 answer about ``verb`` on ``targets`` rest on a captive screw's tightness?

    The owner's rule (spec 6.6): a question whose answer depends on whether captive /
    cooler screws are loosened is not asked, neither "yes, they are" nor "no, loosen them
    first" -- the picture of a loose captive screw is the picture of a tight one.
    """
    for target in targets:
        if is_captive_screw(tc.ctx, target):
            return True
        for edge in applicable_preconditions(tc.by_target.get(target, []), (verb, target)):
            if is_captive_screw(tc.ctx, edge.blocker) and present(state, edge.blocker):
                return True
    return False


def _v4_target(tc: TaskCtx, state: FrameState, verb: str, target: str
               ) -> Optional[tuple[str, list[str], list[Edge]]]:
    """``(id, members, unmet edges)``: the part, or its group when every member still to do
    gives the same answer (same feasibility, same blocker groups -- spec 3.1).

    ``None`` for a screw of a group whose members differ: the screws of one group are
    never told apart (spec 6.4); another part of a mixed group is asked on its own.
    """
    own = unmet_for(tc, verb, target, state)
    group = tc.group_of(target)
    if group is None:
        return target, [target], own
    members = to_do(tc.ctx, state, verb, group.members)
    if target not in members or len(members) < 2:
        return target, [target], own
    answers = {m: unmet_for(tc, verb, m, state) for m in members}
    shapes = {(bool(bad), tuple(blocker_ids(tc, [e.blocker for e in bad])))
              for bad in answers.values()}
    if len(shapes) == 1:
        edges = sorted({e for bad in answers.values() for e in bad},
                       key=lambda e: (e.type, e.target, e.blocker))
        return group.gid, members, edges
    if tc.ctx.cls_of(target) == "screw":
        return None
    return target, [target], own


def _v4_record(tc: TaskCtx, frame: FrameData, verb: str, target: str,
               templates: Sequence[str], truth_source: str,
               reason: Optional[str] = None) -> Optional[dict]:
    """One V4 record, or ``None`` when it is not to be asked.

    Not asked: a target that cannot be outlined (``unmarkable_part``), a member of a
    screw group whose members differ (``mixed_group``), an answer resting on captive screw
    tightness (``captive_tightness``).
    """
    state = tc.state(frame.step)
    named = _v4_target(tc, state, verb, target)
    if named is None:
        tc.skip("V4", "mixed_group")
        return None
    ident, members, bad = named
    if captive_dependent(tc, verb, members, state):
        tc.skip("V4", "captive_tightness")
        return None
    if truth_source == DEMONSTRATED and bad:
        return None  # the graph forbids what the log did: say nothing
    if truth_source != DEMONSTRATED and not bad:
        return None
    if truth_source in GRAPH_KINDS:
        # what says "no" to the asked part or group as a whole: the graph, a part above, a
        # loose part -- or several of them
        truth_source = graph_kind(bad)
    blockers = blocker_ids(tc, [e.blocker for e in bad])
    rec_id = (f"V4-{frame_id(tc.desktop, tc.view, frame.step)}"
              f"-{truth_source}-{verb}-{ident}")
    index, template = pick(templates, rec_id)
    namer = Namer(tc, frame)
    if (namer.group(ident, members) if ident != target else namer.part(target)) is None:
        tc.skip("V4", "unmarkable_part")
        return None
    label = namer.say(ident)
    chain: list[dict] = [observed(m, state[m].state, tc, frame.pointable) for m in members]
    for edge in sorted(bad, key=lambda e: (e.type, e.target, e.blocker)):
        current = state[edge.blocker].state if edge.blocker in state else None
        chain.append(R.recall_relation(edge.type, edge.target, edge.blocker,
                                       edge.necessity, edge.mode))
        chain.append(observed(edge.blocker, current, tc, frame.pointable))
        chain.append(R.check_precondition(edge.blocker,
                                          required_states(edge.type, edge.mode),
                                          current, False))
    feasible = truth_source == DEMONSTRATED
    # why this answer is what it is, in what the picture shows (owner 2026-11-04): the
    # prerequisites now met, or what still holds it -- never the graph's words. The internal
    # code (an edge said no, or nothing did) stays in answer_check.reason_code
    reason_code = None if feasible else (reason or REASON_CODES.get(truth_source,
                                                                     "violates_edge"))
    reason, why = feasibility_reason(tc, verb, ident, members, state, bad)
    raw_blockers = sorted({e.blocker for e in bad})
    bboxes = {k: frame.pointable[k][1]
              for k in [*members, *raw_blockers] if k in frame.pointable}
    check = {"type": "feasibility", "verb": verb, "target": ident,
             "target_class": tc.ctx.cls_of(target),
             "match_class": match_class(tc.ctx, target),
             "state_after": frame.step, "truth_source": truth_source,
             "reason_code": reason_code,
             "metric": "per_verb_accuracy_macro", "secondary_metric": "accuracy",
             "edges": [{"type": e.type, "target": e.target, "blocker": e.blocker,
                        "necessity": e.necessity, "mode": e.mode,
                        "moment": e.source in MOMENT_SOURCES,
                        "required": sorted(required_states(e.type, e.mode))}
                       for e in sorted(bad, key=lambda e: (e.type, e.target, e.blocker))],
             # the moment's own rules on the asked parts (never stored edges): what covers
             # them in this frame, and the loose parts that come out first
             "moment": moment_json(tc, members),
             # the blocker groups' members, for a grader that wants them
             "blocker_members": {b: sorted({e.blocker for e in bad
                                            if (tc.group_of(e.blocker).gid
                                                if tc.group_of(e.blocker) else e.blocker) == b})
                                 for b in blockers}}
    if ident != target:
        check["members"] = list(members)
    return record(
        tc, "V4", rec_id, frame.step, tc.moment(frame.step, marked=True),
        template.format(phrase=verb_phrase(verb, label),
                        gerund=verb_gerund(verb, label)), index,
        {"feasible": feasible, "blockers": blockers, "reason": reason, "why": why},
        check, evidence(bboxes, tc),
        dict(R.rationale(chain), text=("Yes: " if feasible else "No: ") + reason + "."),
        row_verified([frame.pointable[k][0] for k in bboxes], frame.verified),
        negative=None if feasible else truth_source,
        truth_source=truth_source, view_confirmed=frame.confirmed,
        marks=namer.marks(), mark_names=namer.names(), terms=[target_class(tc.ctx, target)],
        moment=tc.legend((frame.step, "now"), marked=True),
    )


#: The internal code of a V4 "no" by what said it (label side only).
REASON_CODES = {OCCLUSION_BLOCKED: "covered_now", LOOSE_FIRST: "loose_part_first"}


def gen_v4(tc: TaskCtx, step: int) -> Iterator[dict]:
    """Demonstrated positives against graph-blocked negatives, one for one.

    A "yes" here is never merely permitted. The graph is incomplete -- there is
    not one ``blocked_by`` edge in the database -- so "no recorded constraint
    forbids it" is not "it works", and a benchmark whose positive class is that
    guess teaches a model to trust an upper bound. The only certain yes is the
    one the operator went on to do, so that is the only yes this task ships.

    The negatives are the certain noes: a graph-blocked action, and the failed
    attempts the graph explains. The three attempts of
    ``reports/constraints_report.md`` that break no constraint are exactly the
    case where the dataset does not yet know the answer; the constraint editor
    (task B5) is where that gap is filled.

    The redesign (spec 3.3): no state-inapplicable negative ("can you remove the PSU?"
    when it is long gone is V2's question, not an affordance), so a positive without a
    graph negative to pair with goes too; blockers are blocker groups; nothing resting on
    captive screw tightness is asked.
    """
    frame = tc.frame(step)
    if frame is None or tc.excluded or not tc.stepwise:
        return  # D66 (several parts at once) asks no step-ordered question
    # the rules of this moment: the stored graph, what a part above still covers, and
    # loose parts first (owner 2026-11-04)
    with tc.at(step):
        made = list(_gen_v4(tc, frame, step))
    yield from made


def _gen_v4(tc: TaskCtx, frame: FrameData, step: int) -> Iterator[dict]:
    shown = tc.demonstrated(step)
    if shown is None:
        return  # no demonstrated action here: nothing certain to be positive about
    _, action = shown
    state = tc.state(step)
    if action.target not in state:
        return
    # one certain "no" for each certain "yes", matched to the positive's verb
    # and class so the pair cannot be told apart without looking at the picture
    # ... and of the **same verb**, always (the controller's ruling, 2026-11-03): with only
    # the graph's "no"s left, a verb no edge gates (`unscrew`, `disconnect`) would
    # otherwise only ever be a "yes", and the verb would answer the question
    candidates = [c for c in tc.certainly_wrong(step)
                  if c.kind in GRAPH_KINDS and c.verb == action.verb]
    yield from _v4_pair(tc, frame, step, action, candidates)
    yield from _v4_failed(tc, frame, step, state)


def _v4_pair(tc: TaskCtx, frame: FrameData, step: int, action: ActionRec,
             candidates: Sequence[Candidate]) -> Iterator[dict]:
    """The demonstrated "yes" and its same-verb graph "no", both or neither."""
    if not candidates:
        tc.skip("V4", "no_same_verb_negative")
        return
    positive = _v4_record(tc, frame, action.verb, action.target, V4_QUESTIONS,
                          DEMONSTRATED)
    if positive is None:
        return  # the demonstrated part cannot be asked about: no pair here
    matched = tc.matched(candidates, action.verb,
                         target_class(tc.ctx, action.target), tc.seed(step, "V4"))
    # the negative names its part the way the positive does -- a group (the outlines of
    # one group) against a group, a single part against a single part -- or "is it a
    # group?" answers the question (final review I1: a group target was "yes" 133 of 178
    # times); and inside that the same match class first (motherboard: 51 of 56 "no")
    mine = match_class(tc.ctx, action.target)
    matched = sorted(matched, key=lambda c: match_class(tc.ctx, c.target) != mine)
    pos_check = positive["label"]["answer_check"]
    negative = None
    for candidate in matched:
        negative = _v4_record(tc, frame, candidate.verb, candidate.target, V4_QUESTIONS,
                              candidate.kind)
        if negative is not None:
            check = negative["label"]["answer_check"]
            if check["target"] != pos_check["target"] and \
                    ("members" in check) == ("members" in pos_check):
                break
        negative = None
    if negative is None:
        tc.skip("V4", "no_same_verb_negative")
    else:
        yield positive
        yield negative


def _v4_failed(tc: TaskCtx, frame: FrameData, step: int,
               state: FrameState) -> Iterator[dict]:
    """The attempt the operator actually made and could not finish."""
    for attempt in tc.failed_actions(step + 1):
        rec = tc.ctx.instances.get(attempt.target)
        if rec is None or attempt.target not in state:
            continue
        if not verb_applies(tc.tax, rec.cls, rec.attrs, attempt.verb,
                            state[attempt.target].state):
            continue
        if not unmet_for(tc, attempt.verb, attempt.target, state):
            continue  # the graph cannot explain it: not a ground truth yet
        failed = _v4_record(tc, frame, attempt.verb, attempt.target,
                            V4_FAILED_QUESTIONS, FAILED_ATTEMPT, "violates_edge")
        if failed is not None:
            yield failed


# --------------------------------------------------------------------------- #
# V5 -- the bounded set
# --------------------------------------------------------------------------- #
V5_QUESTIONS = (
    "What can be done to this machine right now? List the actions you are "
    "confident are possible.",
    "Given this state, which disassembly actions could be carried out next? "
    "List them.",
    "Name the actions that could be performed on this machine as it stands.",
)


def _actions_json(pairs: Sequence[tuple[str, str]]) -> list[dict]:
    return [{"verb": verb, "target": target} for verb, target in sorted(pairs)]


def gen_v5(tc: TaskCtx, step: int) -> Iterator[dict]:
    """A bounded set, because the exact set is not knowable from this data.

    ``must_include`` is what the log went on to do, so it is certainly possible;
    ``must_not_include`` is what the graph forbids, so it is certainly
    impossible; ``permitted_upper_bound`` is everything no recorded constraint
    rules out, and is labelled as an upper bound because that is all it is. An
    answer is right when it contains every certainty and none of the
    impossibilities -- not when it reproduces the upper bound.
    """
    frame = tc.frame(step)
    if frame is None or tc.excluded:
        return
    shown = tc.demonstrated(step)
    if shown is None:
        return
    nxt, action = shown
    permitted = tc.permitted(step)
    must_include = [(action.verb, action.target)]
    if must_include[0] not in set(permitted):
        return  # the graph would forbid what the log did: say nothing
    must_not = [(v, t) for v, t, _ in tc.blocked(step)]

    rec_id = f"V5-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V5_QUESTIONS, rec_id)
    state = tc.state(step)
    chain = [observed(target, state[target].state if target in state else None,
                      tc, frame.pointable)
             for _, target in sorted(permitted)[:tc.budget]]
    yield record(
        tc, "V5", rec_id, step, [frame], question, index,
        {"must_include": _actions_json(must_include),
         "must_not_include": _actions_json(must_not),
         "permitted_upper_bound": _actions_json(permitted)},
        {"type": "bounded_set", "reference_step": nxt,
         "metric": "demonstrated_recall+blocked_rate", "necessity": "required",
         "strict_upper_bound": _actions_json(tc.permitted_strict(step))},
        evidence({k: v[1] for k, v in frame.pointable.items()
                  if k in {t for _, t in permitted}}, tc),
        R.rationale(chain), frame.verified,
    )


# --------------------------------------------------------------------------- #
# V6 -- the next step
# --------------------------------------------------------------------------- #
#: The open form (owner 2026-11-04): every action possible now, not the one demonstrated.
V6_QUESTIONS = (
    "List every disassembly action that can be carried out now on this machine.",
    "Looking at the machine in these pictures, which actions could be carried out right now? "
    "List all of them.",
    "Name every action that is possible on this machine as it stands, not only the one you "
    "would pick.",
)
#: The multiple-choice form: exactly one option is possible now, the others are blocked.
V6_MC_QUESTIONS = (
    "Which of these actions can be carried out now?",
    "Looking at the machine in these pictures, which one of the following actions is possible "
    "right now?",
    "Only one of these actions can be done on the machine as it stands. Which one?",
)
#: The constrained form: the same list, under one extra rule given in words.
V6_CONSTRAINED_QUESTIONS = (
    "List every disassembly action that can be carried out now on this machine, under this "
    "rule: {rule}",
    "Rule for this teardown: {rule} Under that rule, which actions could be carried out right "
    "now? List all of them.",
    "Name every action that is possible on this machine as it stands, keeping to this rule: "
    "{rule}",
)
#: How many blocked distractors a multiple choice carries (at most), and how few it needs.
V6_WRONG_OPTIONS = 3
V6_MIN_WRONG = 2

#: ``(class, state) -> the verbs the 66 logs use from there`` (the graph audit of
#: 2026-11-04): the frontier only ever offers an action the teardowns actually do --
#: no ``displace`` of a RAM stick, no ``remove`` of a plugged connector, no virtual
#: ``cable:*`` node. ``displace`` is further limited to the parts this desktop's own log
#: swings aside (R14: only a part that can be swung is hinged).
LOG_VOCAB: dict[tuple[str, str], frozenset[str]] = {
    ("cable_clip", "closed"): frozenset({"release"}),
    ("card_latch", "closed"): frozenset({"open", "remove"}),
    ("case_fan", "installed"): frozenset({"remove"}),
    ("connector", "plugged"): frozenset({"disconnect"}),
    ("cooler_bracket", "installed"): frozenset({"remove"}),
    ("cooler_latch", "closed"): frozenset({"remove"}),
    ("cover", "closed"): frozenset({"open", "remove"}),
    ("cover", "open"): frozenset({"remove"}),
    ("cpu", "installed"): frozenset({"remove"}),
    ("cpu_cooler", "installed"): frozenset({"remove", "displace"}),
    ("cpu_cooler", "displaced"): frozenset({"remove"}),
    ("cpu_socket_lever", "closed"): frozenset({"open"}),
    ("drive_cage", "installed"): frozenset({"remove", "displace"}),
    ("drive_cage", "displaced"): frozenset({"remove"}),
    ("drive_latch", "closed"): frozenset({"open"}),
    ("expansion_card", "installed"): frozenset({"remove"}),
    ("misc_part", "installed"): frozenset({"remove"}),
    ("motherboard", "installed"): frozenset({"remove"}),
    ("optical_drive", "installed"): frozenset({"remove"}),
    ("psu", "installed"): frozenset({"remove", "displace"}),
    ("psu", "displaced"): frozenset({"remove"}),
    ("psu_latch", "closed"): frozenset({"remove"}),
    ("ram_latch", "closed"): frozenset({"open"}),
    ("ram_module", "installed"): frozenset({"remove"}),
    ("screw", "fastened"): frozenset({"unscrew"}),
    ("storage_drive", "installed"): frozenset({"remove", "displace"}),
    ("storage_drive", "displaced"): frozenset({"remove"}),
}
#: Classes whose every possible action is **one** group, interchangeable and unordered
#: (the rules' preamble: 插头和插头之间、内存卡扣和内存卡扣之间没有先后).
ONE_GROUP = {"connector": "connectors", "ram_latch": "RAM latches", "ram_module": "RAM modules"}


@dataclass
class FrontierGroup:
    """One group of the frontier: interchangeable actions possible now, no order among them."""

    key: str
    actions: list            # [(verb, target)], sorted
    phrase: str = ""         # "unplug the connectors": how a person says the group
    hidden: list = field(default_factory=list)  # targets a cable hides (named in words)

    def as_json(self) -> dict:
        return {"key": self.key, "phrase": self.phrase,
                "actions": [{"verb": v, "target": t} for v, t in self.actions],
                **({"hidden_by_cable": list(self.hidden)} if self.hidden else {})}


def frontier_key(tc: TaskCtx, verb: str, key: str) -> str:
    """The group an action belongs to: every plug one group, every RAM latch, every RAM
    module; screws by the part they fasten; any other part by its part group (alternative
    verbs on one part -- open or remove a cover -- are one group: the part is what is
    "possible now")."""
    rec = tc.ctx.instances.get(key)
    if rec is None:
        return f"part:{key}"
    if rec.cls in ONE_GROUP:
        return f"all:{rec.cls}"
    if rec.cls == "screw":
        return f"screws:{rec.fastens or rec.attrs.get('role') or key}"
    group = tc.group_of(key)
    return f"part:{group.gid}" if group is not None else f"part:{key}"


def in_log_vocab(tc: TaskCtx, verb: str, key: str, state: FrameState) -> bool:
    """Is ``verb`` on ``key`` an action the teardowns do from its present state (:data:`LOG_VOCAB`)?"""
    rec = tc.ctx.instances.get(key)
    inst = state.get(key)
    if rec is None or inst is None:
        return False
    if verb not in LOG_VOCAB.get((rec.cls, inst.state), frozenset()):
        return False
    if verb == "displace" and key not in tc.displaced_in_log:
        return False  # only a part this machine's log swings aside is hinged (R14)
    if rec.cls == "ram_latch":
        module = rec.attrs.get("of")
        if module and not present(state, module):
            return False  # a latch of an empty slot: opening it means nothing
    return True


def frontier_visible(frame: FrameData, key: str) -> Optional[str]:
    """``"seen"`` when the scanner frame can point at ``key``, ``"cable"`` when only a cable
    hides it (it stays, named in words), else ``None``."""
    if key in frame.pointable:
        return "seen"
    row = frame.rows.get(key) or {}
    if row.get("occluder") == "cable":
        return "cable"
    return None


def frontier_parts(tc: TaskCtx, actions: Sequence[tuple[str, str]]) -> tuple[str, str]:
    """``("unplug", "the connectors")``: the verb words and the part words of one group."""
    verbs = sorted({v for v, _ in actions}, key=lambda v: (v != "open", v))
    targets = sorted({t for _, t in actions})
    return " or ".join(VERB_WORDS.get(v, v) for v in verbs), target_words(tc, targets)


def frontier_phrase(tc: TaskCtx, actions: Sequence[tuple[str, str]]) -> str:
    """How a person says one group of actions, no ordinal: "unplug the connectors",
    "unscrew the PSU screws", "open or take out the side cover"."""
    head, words = frontier_parts(tc, actions)
    return f"{head} {words}"


#: How a reason says a verb (the log's ``disconnect`` is "unplug" to a person).
VERB_WORDS = {"disconnect": "unplug", "displace": "move aside", "remove": "take out"}


def target_words(tc: TaskCtx, targets: Sequence[str]) -> str:
    """"the PSU", "the connectors", "the PSU screws", "the RAM modules": parts in words."""
    targets = sorted(set(targets))
    if len(targets) == 1:
        key = targets[0]
        if tc.group_by_id(key) is not None:
            return f"the {tc.group_by_id(key).label}"
        return asked_words(tc, key)[0]
    groups = {tc.group_of(t).gid if tc.group_of(t) else None for t in targets}
    if len(groups) == 1 and None not in groups:
        return f"the {tc.group_of(targets[0]).label}"
    classes = {tc.ctx.cls_of(t) for t in targets}
    if classes == {"screw"}:
        held = {tc.ctx.instances[t].fastens for t in targets}
        if len(held) == 1 and None not in held:
            return f"the screws that fasten {asked_words(tc, next(iter(held)))[0]}"
        return "the screws"
    if len(classes) == 1:
        cls = next(iter(classes))
        if cls in ONE_GROUP:
            return f"the {ONE_GROUP[cls]}"
        return f"the {class_phrase(tc.ctx, targets[0], many=True)}"
    return _and([asked_words(tc, t)[0] for t in targets])


def captive_pending(tc: TaskCtx, state: FrameState) -> bool:
    """Is some captive screw loosened while the part it holds is still in? (spec 6.6)

    Its tightness is in no picture, and whether the cooler can come off rests on it: the
    frontier of such a moment is not asked (``captive_tightness``)."""
    for key, rec in tc.ctx.instances.items():
        if not is_captive_screw(tc.ctx, key):
            continue
        inst = state.get(key)
        if inst is not None and inst.state == "loosened" and rec.fastens \
                and present(state, rec.fastens):
            return True
    return False


def frontier(tc: TaskCtx, frame: FrameData, step: int,
             extra: Sequence[Edge] = ()) -> list[FrontierGroup]:
    """The V6 gold (owner 2026-11-04): every action possible now, grouped.

    :func:`~tda.core.graph.legal_actions` with ``loose_first`` and this frame's dynamic
    occlusion (and a constrained question's ``extra`` edges), restricted to the log's
    vocabulary (:func:`in_log_vocab`), to parts a question may name, and to targets the
    scanner frame can point at -- a target only a cable hides stays, named in words.
    """
    state = tc.state(step)
    moment = tc.moment_at(step)
    legal = legal_actions(tc.ctx.instances, [*tc.edges, *extra], state, tc.tax, strict=False,
                          loose_first=True, occluded=moment.occluded)
    buckets: dict[str, FrontierGroup] = {}
    for verb, key in legal:
        if key.startswith(CABLE_PREFIX) or not tc.askable(key):
            continue
        if not in_log_vocab(tc, verb, key, state):
            continue
        seen = frontier_visible(frame, key)
        if seen is None:
            continue
        gk = frontier_key(tc, verb, key)
        group = buckets.setdefault(gk, FrontierGroup(gk, []))
        group.actions.append((verb, key))
        if seen == "cable" and key not in group.hidden:
            group.hidden.append(key)
    out = []
    for gk in sorted(buckets):
        group = buckets[gk]
        group.actions = sorted(set(group.actions))
        group.hidden = sorted(group.hidden)
        group.phrase = frontier_phrase(tc, group.actions)
        out.append(group)
    return out


def blocked_frontier(tc: TaskCtx, frame: FrameData, step: int,
                     groups: Sequence[FrontierGroup]) -> list[FrontierGroup]:
    """Groups of actions certainly **not** possible now: rule-blocked (graph, occlusion, loose
    parts), in the log's vocabulary, on a part present and visible, and no member of the
    group possible at all (a member hidden but legal would make "it cannot be done" false).
    Never an action already done or meaningless (a state-inapplicable one, an empty slot)."""
    state = tc.state(step)
    legal = set(tc.permitted(step))
    keys = {g.key for g in groups}
    buckets: dict[str, FrontierGroup] = {}
    for c in tc.certainly_wrong(step):
        if c.kind not in GRAPH_KINDS or not tc.askable(c.target):
            continue
        if not in_log_vocab(tc, c.verb, c.target, state) or c.target not in frame.pointable:
            continue
        gk = frontier_key(tc, c.verb, c.target)
        if gk in keys:
            continue
        buckets.setdefault(gk, FrontierGroup(gk, [])).actions.append((c.verb, c.target))
    out = []
    for gk in sorted(buckets):
        group = buckets[gk]
        # every action of this kind on the group's parts must be blocked
        # a legal action of the same group anywhere -- hidden from this camera, say -- would
        # make "it cannot be done" false
        if any(frontier_key(tc, v, t) == gk and in_log_vocab(tc, v, t, state)
               for v, t in legal if not t.startswith(CABLE_PREFIX)):
            continue
        group.actions = sorted(set(group.actions))
        group.phrase = frontier_phrase(tc, group.actions)
        out.append(group)
    return out


def group_reason(tc: TaskCtx, state: FrameState, group: FrontierGroup) -> str:
    """Why one frontier group is possible now, in what the picture shows."""
    verb, key = group.actions[0]
    members = sorted({t for v, t in group.actions if v == verb})
    ident = key if len(members) == 1 else (tc.group_of(key).gid if tc.group_of(key) is not None
                                           and set(tc.group_of(key).members) >= set(members)
                                           else key)
    text, _ = feasibility_reason(tc, verb, ident, members, state, [])
    if len(members) > 1 and tc.group_by_id(ident) is None:
        text = re.sub(r", so .* can be ([a-z ]+)$", r", so they can be \1", text)
        text = re.sub(r"^nothing is in the way, so .*$", "nothing is in the way of any of them",
                      text)
    return text


def blocked_reason(tc: TaskCtx, state: FrameState, group: FrontierGroup) -> str:
    """Why a blocked group is not possible yet: what still holds it, in visible terms."""
    verb, key = group.actions[0]
    members = sorted({t for v, t in group.actions if v == verb})
    bad = sorted({e for m in members for e in unmet_for(tc, verb, m, state)},
                 key=lambda e: (e.type, e.target, e.blocker))
    ident = key if len(members) == 1 else members[0]
    text, _ = feasibility_reason(tc, verb, ident, members, state, bad)
    return text


def loose_sentence(tc: TaskCtx, step: int) -> str:
    loose = tc.moment_at(step).loose
    if not loose:
        return ""
    words = _and([asked_words(tc, k)[0] for k in loose])
    be = "lie" if len(loose) > 1 else "lies"
    return (f"{words[:1].upper() + words[1:]} {be} loose in the case, and a loose part comes out "
            f"before anything else is done. ")


def frontier_text(tc: TaskCtx, step: int, groups: Sequence[FrontierGroup],
                  blocked: Sequence[FrontierGroup], rule: Optional[str] = None,
                  forbidden: Sequence[FrontierGroup] = ()) -> str:
    """The V6 reasoning chain: what the picture shows, which rule applies, so the answer."""
    state = tc.state(step)
    parts = [loose_sentence(tc, step)]
    for group in groups:
        reason = group_reason(tc, state, group)
        hidden = (" (a cable hides some of them, but they are there)" if group.hidden else "")
        parts.append(f"{group.phrase[:1].upper() + group.phrase[1:]}: {reason}{hidden}. ")
    for group in blocked[:3]:
        parts.append(f"Not yet {group.phrase}: {blocked_reason(tc, state, group)}. ")
    if rule:
        parts.append(f"The rule ({rule.rstrip('.')}) rules out "
                     + _and([g.phrase for g in forbidden]) + " for now. ")
    if groups:
        parts.append("So the actions possible now are: " + "; ".join(g.phrase for g in groups)
                     + ".")
    else:
        parts.append("So nothing can be done now.")
    return "".join(parts).strip()


def _frontier_answer(groups: Sequence[FrontierGroup], why: str) -> dict:
    """One action per group (its first): the gold reply lists every group once."""
    return {"actions": [{"verb": g.actions[0][0], "target": g.actions[0][1]} for g in groups],
            "why": why}


def v6_constraints(tc: TaskCtx, frame: FrameData, step: int,
                   groups: Sequence[FrontierGroup]) -> list[dict]:
    """The constrained form's candidate rules, each removing at least one frontier group.

    * **until every connector is unplugged**: a part the frontier would take out must stay
      in until every plug still in is pulled -- ``connected_to(part <- plug)`` per plug;
    * **A before any B**: a large part still in (A) has to come out before a part the
      frontier would take out (B) -- ``blocked_by(B <- A, contained)``: only A out frees B.
      Only where A's own way out needs nothing of B.

    No ordinal, no number: the rule names classes and parts in words.
    """
    state = tc.state(step)
    if tc.moment_at(step).loose:
        return []  # a loose part comes out first: a rule on top of that is not asked
    plugged = sorted(k for k, rec in tc.ctx.instances.items()
                     if rec.cls == "connector" and tc.askable(k)
                     and (state.get(k) is not None and state[k].state == "plugged"))
    out: list[dict] = []
    # a rule holds back a part the frontier would take out -- a part, not a latch or lever
    removable = [g for g in groups if g.key != "all:connector"
                 and any(v == "remove" for v, _ in g.actions)
                 and not any(str(tc.ctx.cls_of(t)).endswith(("latch", "lever"))
                             for _, t in g.actions)]
    for g in removable:
        targets = sorted({t for v, t in g.actions if v == "remove"})
        if not plugged or set(targets) & set(plugged):
            continue
        words = target_words(tc, targets)
        many = len(targets) > 1
        rule = (f"{words[:1].upper() + words[1:]} must not be taken out until every connector "
                f"is unplugged.")
        edges = [Edge(type="connected_to", target=t, blocker=c, source=CONSTRAINT_SOURCE,
                      status="accepted", reason="the question's rule")
                 for t in targets for c in plugged]
        out.append({"kind": "until_connectors", "rule": rule, "held": targets,
                    "until": plugged, "edges": edges, "many": many})
    # A: a large part that comes out on its own way (never the board everything sits on)
    bigs = sorted(k for k, rec in tc.ctx.instances.items()
                  if rec.cls in RULE_FIRST_CLASSES
                  and tc.askable(k) and present(state, k) and unique_on_desktop(tc, k))
    for a in bigs:
        plan = remaining_plan(tc.ctx.instances, tc.moment_at(step).edges, state, a, tc.tax)
        if not plan:
            continue
        touched = {t for _, t in plan}
        for g in removable:
            targets = sorted({t for v, t in g.actions if v == "remove"})
            if a in targets or touched & set(targets):
                continue
            if tc.ctx.cls_of(a) == tc.ctx.cls_of(targets[0]):
                continue
            words = target_words(tc, targets)
            single = (f"before {words} is taken out" if len(targets) == 1 else
                      f"before any {class_phrase(tc.ctx, targets[0])} is taken out")
            rule = f"{asked_words(tc, a)[0][:1].upper() + asked_words(tc, a)[0][1:]} has to " \
                   f"come out {single}."
            edges = [Edge(type="blocked_by", target=t, blocker=a, mode="contained",
                          source=CONSTRAINT_SOURCE, status="accepted",
                          reason="the question's rule") for t in targets]
            out.append({"kind": "before", "rule": rule, "held": targets, "until": [a],
                        "edges": edges})
    return out


#: The parts a "before" rule may ask to come out first: large parts with a way out of their own.
RULE_FIRST_CLASSES = frozenset({"psu", "drive_cage", "optical_drive", "cpu_cooler",
                                "expansion_card", "storage_drive"})


def constraint_json(c: dict) -> dict:
    """A constraint as the label carries it: kind, words, the parts held, the condition, and
    the edges a checker adds to the graph to replay it."""
    return {"kind": c["kind"], "rule": c["rule"], "held": list(c["held"]),
            "until": list(c["until"]),
            "edges": [{"type": e.type, "target": e.target, "blocker": e.blocker,
                       "mode": e.mode, "necessity": e.necessity, "source": e.source}
                      for e in c["edges"]]}


def gen_v6(tc: TaskCtx, step: int) -> Iterator[dict]:
    """Every action possible now (owner 2026-11-04), in three forms.

    * **open**: "list every action that can be carried out now" -- the gold is the
      frontier (:func:`frontier`), graded by precision / recall / F1 over its groups;
    * **mc**: options named directly (no mark, no ordinal), exactly one inside the frontier,
      the others certainly blocked (:func:`blocked_frontier`) -- never one already done or
      meaningless;
    * **constrained**: the open question plus one rule in words that removes at least one
      group (:func:`v6_constraints`); the gold is the frontier under graph + rule.

    Not asked: D66 (not stepwise), a desktop the graph cannot speak for, a moment whose
    frontier rests on a captive screw's tightness, a frame with nothing possible.
    """
    frame = tc.frame(step)
    if frame is None or tc.excluded or not tc.stepwise:
        return
    with tc.at(step):
        made = list(_gen_v6(tc, frame, step))
    yield from made


def _gen_v6(tc: TaskCtx, frame: FrameData, step: int) -> Iterator[dict]:
    state = tc.state(step)
    if captive_pending(tc, state):
        tc.skip("V6", "captive_tightness")
        return
    groups = frontier(tc, frame, step)
    if not groups:
        tc.skip("V6", "empty_frontier")
        return
    seed = tc.seed(step, "V6")
    blocked = blocked_frontier(tc, frame, step, groups)
    shown = tc.demonstrated(step)
    demo = (shown[1].verb, shown[1].target) if shown is not None else None
    in_front = {a: g.key for g in groups for a in g.actions}
    demo_group = in_front.get(demo) if demo is not None else None
    stem = f"V6-{frame_id(tc.desktop, tc.view, step)}"
    pair = opaque_record_id(stem)
    common = {"type": "frontier", "metric": "group_f1",
              "groups": [g.as_json() for g in groups],
              "blocked_groups": [g.as_json() for g in blocked],
              "demonstrated": list(demo) if demo else None,
              "demonstrated_group": demo_group,
              "loose_parts": list(tc.moment_at(step).loose),
              "moment": moment_json(tc),
              "previous_action": previous_action(tc, step)}
    text = frontier_text(tc, step, groups, blocked)
    answer = _frontier_answer(groups, text)
    targets = sorted({t for g in groups for _, t in g.actions})
    evid = evidence({t: frame.pointable[t][1] for t in targets if t in frame.pointable}, tc)
    chain = [observed(t, state[t].state, tc, frame.pointable, step=step) for t in targets[:4]]
    rationale = dict(R.rationale(chain), text=text)
    index, question = pick(V6_QUESTIONS, stem)
    yield record(
        tc, "V6", stem, step, tc.moment(step), question, index, answer,
        dict(common, form="open", gold_groups=[g.key for g in groups]),
        evid, rationale, frame.verified, extra={"v6_form": "open", "v6_pair": pair},
        view_confirmed=frame.confirmed, moment=tc.legend((step, "now")),
    )
    # the multiple choice: one option inside the frontier, the rest certainly blocked
    if len(blocked) >= V6_MIN_WRONG:
        right = next((g for g in groups if g.key == demo_group), None)
        if right is None:
            right = stable_order(list(groups), seed + "|right", lambda g: g.key)[0]
        wrong = stable_order(list(blocked), seed + "|wrong", lambda g: g.key)[:V6_WRONG_OPTIONS]
        def option(g: FrontierGroup, kind: str) -> dict:
            say_verb, words = frontier_parts(tc, g.actions)
            # named directly, no mark, no tag, no ordinal: "unplug the connectors"
            return {"verb": g.actions[0][0], "target": g.actions[0][1], "kind": kind,
                    "group": g.key, "members": sorted({t for _, t in g.actions}),
                    "actions": [list(a) for a in g.actions],
                    "say_verb": say_verb, "target_label": words}

        listed = [option(right, "frontier")]
        listed += [option(g, graph_kind([e for v, t in g.actions
                                          for e in unmet_for(tc, v, t, state)]))
                   for g in wrong]
        if len({(o["say_verb"], o["target_label"]) for o in listed}) == len(listed):
            options = stable_order(listed, seed + "|order", lambda o: o["group"])
            rec_id = stem + "-mc"
            index, question = pick(V6_MC_QUESTIONS, rec_id)
            mc_text = (frontier_text(tc, step, [right], wrong)
                       .rsplit("So the actions possible now are:", 1)[0].strip()
                       + f" So the one possible now is: {right.phrase}.")
            yield record(
                tc, "V6", rec_id, step, tc.moment(step), question, index,
                {"verb": right.actions[0][0], "target": right.actions[0][1],
                 "group": right.key},
                dict(common, form="mc", options_kind="listed",
                     reference=[right.actions[0][0], right.actions[0][1]],
                     gold_groups=[right.key]),
                evid, dict(R.rationale(chain), text=mc_text), frame.verified,
                options=options, extra={"v6_form": "mc", "v6_pair": pair},
                view_confirmed=frame.confirmed, moment=tc.legend((step, "now")),
                terms=[tc.ctx.cls_of(o["target"]) for o in options],
            )
    else:
        tc.skip("V6", "mc_too_few_blocked")
    # the constrained form: one rule in words that removes at least one group
    candidates = v6_constraints(tc, frame, step, groups)
    for c in stable_order(candidates, seed + "|rule", lambda c: c["rule"]):
        with tc.at(step, c["edges"]):
            ruled = frontier(tc, frame, step, c["edges"])
            keys = {g.key for g in ruled}
            forbidden = [g for g in groups if g.key not in keys]
            if not forbidden or not ruled:
                continue
            ctext = frontier_text(tc, step, ruled, blocked, rule=c["rule"], forbidden=forbidden)
            rec_id = stem + "-constrained"
            index, template = pick(V6_CONSTRAINED_QUESTIONS, rec_id)
            yield record(
                tc, "V6", rec_id, step, tc.moment(step), template.format(rule=c["rule"]), index,
                _frontier_answer(ruled, ctext),
                dict(common, form="constrained", gold_groups=[g.key for g in ruled],
                     forbidden_groups=[g.key for g in forbidden],
                     constraint=constraint_json(c),
                     ruled_groups=[g.as_json() for g in ruled]),
                evid, dict(R.rationale(chain), text=ctext), frame.verified,
                extra={"v6_form": "constrained"}, view_confirmed=frame.confirmed,
                moment=tc.legend((step, "now")),
            )
        break
    else:
        tc.skip("V6", "no_constraint")


def moment_json(tc: TaskCtx, members: Optional[Sequence[str]] = None) -> dict:
    """The active moment's own rules, compact, for a checker to rebuild its edges
    (``vlm_check.moment_edges_of``): ``occluded`` ``{X: [Y, ...]}`` (``covered_by``), the
    loose parts still in, and ``loose_scope`` -- the parts a loose part does not hold back
    (its own way out). With ``members``, only what concerns those parts."""
    m = tc._active if tc._active is not None else None
    if m is None:
        return {"occluded": {}, "loose": []}
    # the occlusion edges the moment adds (a pair the graph already stores is not repeated)
    occ: dict[str, list[str]] = {}
    for e in m.extra:
        if e.source == OCCLUSION_SOURCE and (members is None or e.target in set(members)):
            occ.setdefault(e.target, []).append(e.blocker)
    occ = {x: sorted(ys) for x, ys in sorted(occ.items())}
    out: dict = {"occluded": occ, "loose": list(m.loose)}
    if m.loose:
        held = {e.target for e in m.extra if e.source == LOOSE_SOURCE}
        state = tc.ctx.state_at(m.step)
        nodes = (set(members) if members is not None else
                 {k for k, inst in state.items() if inst.state != REMOVED})
        out["loose_scope"] = sorted(k for k in nodes if k not in held)
    return out


def _verb_draw(tc: TaskCtx, seed: str) -> Callable[[str], float]:
    """A seeded weighted order of verbs, weighted by this log's use of each.

    The Efraimidis-Spirakis key ``-ln(u) / w``: sorting by it draws verbs
    without replacement with probability proportional to ``w``, so the verbs of
    stand-in distractors follow the verbs of the answers.
    """
    import math

    counts: dict[str, int] = {}
    for step in tc.log_steps:
        for action in tc.named_actions(step):
            counts[action.verb] = counts.get(action.verb, 0) + 1

    def key(verb: str) -> float:
        u = (zlib.crc32(f"{seed}|verbdraw|{verb}".encode("utf-8")) + 1) / (2 ** 32 + 2)
        return -math.log(u) / max(counts.get(verb, 0), 0.01)
    return key


def v6_siblings(tc: TaskCtx, verb: str, target: str,
                permitted: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """The demonstrated action and every permitted action interchangeable with it.

    Interchangeable = same verb and either the same unordered ``group_id`` or
    the same :func:`match_class`. Sorted, the demonstrated one included.
    """
    group = unordered_group(tc.ctx, target)
    cls = match_class(tc.ctx, target)
    out = {(verb, target)}
    for v, t in permitted:
        if v != verb:
            continue
        if (group is not None and unordered_group(tc.ctx, t) == group) or \
                (cls is not None and match_class(tc.ctx, t) == cls):
            out.add((v, t))
    return sorted(out)


def prefer_graph(candidates: Sequence[Candidate]) -> list[Candidate]:
    """The graph's "no"s alone where there is one (spec 3.3).

    "Remove the PSU" when the PSU is out is no distractor, it is a giveaway: V6's wrong
    options and V16's substitutes fall back on a state-inapplicable action only when no
    graph-blocked one exists.
    """
    graph = [c for c in candidates if c.kind in GRAPH_KINDS]
    return graph if graph else list(candidates)


def group_action(tc: TaskCtx, state: FrameState, verb: str, key: str,
                 ok: Callable[[str], bool]) -> tuple[str, list[str]]:
    """``(target, members)``: ``key``'s group when the action is one for the whole group.

    The group (spec 3.1, 6.1) when ``key`` belongs to one, at least two members are still
    in the asked situation (:func:`~tda.core.export.vlm_groups.to_do`) and every one of
    them passes ``ok`` (the same answer for each); else the part alone.
    """
    group = tc.group_of(key)
    if group is None:
        return key, [key]
    members = to_do(tc.ctx, state, verb, group.members)
    if key not in members or len(members) < 2 or not all(ok(m) for m in members):
        return key, [key]
    return group.gid, members


def previous_action(tc: TaskCtx, step: int) -> Optional[dict]:
    """The last named, state-changing action at or before ``step`` (or ``None``).

    The one input the protocol-prior reference (``scripts/vlm_references.py``)
    is allowed: "what did the operator do last".
    """
    for candidate in reversed([s for s in tc.log_steps if s <= step]):
        for action in reversed(tc.named_actions(candidate)):
            if tc.changes_state(action):
                return {"verb": action.verb, "target": action.target,
                        "class": target_class(tc.ctx, action.target),
                        "match_class": match_class(tc.ctx, action.target),
                        "step": int(candidate)}
    return None


# --------------------------------------------------------------------------- #
# V8 -- referring-expression grounding
# --------------------------------------------------------------------------- #
V8_QUESTIONS = (
    "Where is {phrase}? Give its bounding box.",
    "Localise {phrase} in this image.",
    "Point at {phrase}: answer with one bounding box.",
)
#: The set form (redesign, spec 6.3): a class with several instances is asked as a whole.
V8_SET_QUESTIONS = (
    "Localise all visible {phrase} in this image: one bounding box each.",
    "Find all the {phrase} you can see and give a bounding box for each.",
    "Where are the {phrase}? Give a bounding box for every one you can see.",
)
#: How many instances one frame is asked to ground.
V8_BUDGET = 3


def _referring(tc: TaskCtx, instance: str) -> Optional[tuple[str, str]]:
    """``(phrase, family)`` naming exactly one part in words, or ``None``.

    A relational expression is the interesting half of V8 -- it forces the model
    to use the assembly structure rather than a class name -- but only while it
    names exactly one part. ``the screw that fastens the PSU`` is a referring
    expression; with a second such screw it is a trap. The part it is relative to
    must itself be the one of its class (no ordinal, ever: spec 3.2). Failing a
    relational phrase, a part alone of its match class on the desktop is named by its
    class ("the PSU"); any other part is asked in the set form (:func:`gen_v8`).
    """
    rec = tc.ctx.instances.get(instance)
    if rec is None:
        return None
    if rec.cls == "screw" and rec.fastens in tc.ctx.instances \
            and unique_on_desktop(tc, rec.fastens):
        peers = [k for k, r in tc.ctx.instances.items()
                 if r.cls == "screw" and r.fastens == rec.fastens]
        if len(peers) == 1:
            return (f"the screw that fastens {class_words(tc, [rec.fastens])}",
                    "fastens")
    of = rec.attrs.get("of")
    if of in tc.ctx.instances and unique_on_desktop(tc, of):
        peers = [k for k, r in tc.ctx.instances.items()
                 if r.cls == rec.cls and r.attrs.get("of") == of]
        if len(peers) == 1:
            verb = "covers" if rec.cls == "cover" else "locks"
            return (f"the {class_label(rec.cls, tc.tax)} that {verb} "
                    f"{class_words(tc, [of])}"), "of"
    if unique_on_desktop(tc, instance):
        return class_words(tc, [instance]), "class"
    return None


def gen_v8(tc: TaskCtx, step: int) -> Iterator[dict]:
    frame = tc.frame(step)
    if frame is None or not frame.verified:
        return
    seed = tc.seed(step, "V8")
    candidates = [k for k in sorted(frame.pointable)
                  if tc.ctx.cls_of(k) != CHASSIS_CLASS]
    asked_sets: set[str] = set()
    for instance in stable_order(candidates, seed)[:V8_BUDGET]:
        phrase_family = _referring(tc, instance)
        if phrase_family is None:
            mc = match_class(tc.ctx, instance)
            if mc is None or mc in asked_sets:
                continue
            asked_sets.add(mc)
            yield _v8_set(tc, frame, mc, instance)
            continue
        phrase, family = phrase_family
        row, box = frame.pointable[instance]
        rec_id = f"V8-{frame_id(tc.desktop, tc.view, step)}-{instance}"
        index, template = pick(V8_QUESTIONS, rec_id)
        tiny = row.get("visibility") == "visible_tiny"
        yield record(
            tc, "V8", rec_id, step, [frame],
            template.format(phrase=phrase), index,
            {"instance": instance, "bbox": box},
            {"type": "boxes", "field": None, "id_field": "instance",
             "box_field": "bbox", "iou_threshold": 0.5,
             "geometry": "visible_mask_bbox",
             "metric": "point_in_box" if tiny else "box_iou",
             "expression": family},
            evidence({instance: box}, tc, {instance: seen_as(row)}),
            R.rationale([R.observe(instance, tc.ctx.cls_of(instance), tc.view, box,
                                   row.get("visibility"), occluder=row.get("occluder"))]),
            row_verified([row], frame.verified),
            terms=[tc.ctx.cls_of(instance)],
        )


def _v8_set(tc: TaskCtx, frame: FrameData, mc: str, instance: str) -> dict:
    """"Localise every visible <class>": V1's shape restricted to one match class (spec 6.3)."""
    keys = [k for k in sorted(frame.pointable) if match_class(tc.ctx, k) == mc]
    components, bboxes, attrs, steps, used = [], {}, {}, [], []
    for key in keys:
        row, box = frame.pointable[key]
        components.append({"class": tc.ctx.cls_of(key), "instance": key, "bbox": box,
                           "placement": row.get("placement")})
        bboxes[key] = box
        attrs[key] = seen_as(row)
        used.append(row)
        steps.append(R.observe(key, tc.ctx.cls_of(key), tc.view, box, row.get("visibility"),
                               occluder=row.get("occluder")))
    rec_id = f"V8-{frame_id(tc.desktop, tc.view, frame.step)}-set-{mc}"
    index, template = pick(V8_SET_QUESTIONS, rec_id)
    return record(
        tc, "V8", rec_id, frame.step, [frame],
        template.format(phrase=class_phrase(tc.ctx, instance, many=True)), index,
        {"components": components},
        {"type": "components", "field": "components", "id_field": "instance",
         "box_field": "bbox", "class_filter": mc, "exhaustive": True,
         "geometry": "visible_mask_bbox", "metric": "set_f1+box_iou",
         "iou_threshold": 0.5, "expression": "class_set"},
        evidence(bboxes, tc, attrs), R.rationale(steps),
        row_verified(used, frame.verified), terms=[tc.ctx.cls_of(instance)],
    )


# --------------------------------------------------------------------------- #
# V10 -- what has already been done, and how far along this is
# --------------------------------------------------------------------------- #
V10_QUESTIONS = (
    "Which disassembly actions have already been carried out on this machine?",
    "From these pictures of one moment, list the actions that have happened so far and say "
    "how far the teardown has got.",
    "What has been done to this machine up to now, and what fraction of the "
    "teardown is finished?",
)


def grouped_history(tc: TaskCtx, state: FrameState, pairs: set[tuple[str, str]]
                    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict]:
    """``(done, started, groups)``: the done set in group actions (spec 6.1, 6.6).

    A group action counts as done when no member is left to do it to; a group partly done
    is a separate "started" item. The unscrewing of captive (cooler) screws is in neither
    while the part they hold is in: which of them are loose is in no picture, so "loosen
    the cooler screws" stays to do until the cooler comes off.
    """
    done: set[tuple[str, str]] = set()
    started: set[tuple[str, str]] = set()
    groups: dict[str, list[str]] = {}
    for verb, target in pairs:
        group = tc.group_of(target)
        if group is None:
            done.add((verb, target))
            continue
        if verb == "unscrew" and is_captive_screw(tc.ctx, target):
            held = tc.ctx.instances[target].fastens
            if held is not None and present(state, held):
                continue
            done.add((verb, group.gid))
            groups[group.gid] = list(group.members)
            continue
        groups[group.gid] = list(group.members)
        if to_do(tc.ctx, state, verb, group.members):
            started.add((verb, group.gid))
        else:
            done.add((verb, group.gid))
    return sorted(done), sorted(started - done), groups


def gen_v10(tc: TaskCtx, step: int) -> Iterator[dict]:
    frame = tc.frame(step)
    if frame is None:
        return
    if unresolved_action_targets(tc.ctx):
        return  # an exhaustive metric cannot be asked of an incomplete log (C3)
    # counted on the *log*, not on this view's frames: a missing scanner frame
    # (spec 4.2) does not mean the action never happened, and "how much is
    # left" must be the same number in all four views
    done = [a for s in tc.log_steps if s <= step for a in tc.named_actions(s)]
    total = sum(len(tc.named_actions(s)) for s in tc.log_steps)
    if not total:
        return
    # `done` is a *set* (spec 8.2's metric for V10 is set F1), so a chassis
    # reoriented seven times is one thing that has happened -- while the
    # remainder and the progress bin count the log's actions, because "how many
    # steps are left" is not a question about distinct pairs
    state = tc.state(step)
    pairs, started, groups = grouped_history(tc, state, {(a.verb, a.target) for a in done})
    share = len(done) / total
    bin_name = PROGRESS_BINS[min(int(share * len(PROGRESS_BINS)), len(PROGRESS_BINS) - 1)]
    rec_id = f"V10-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V10_QUESTIONS, rec_id)
    chain = [observed(a.target,
                      state[a.target].state if a.target in state else None,
                      tc, frame.pointable, step=a.step)
             for a in done[-tc.budget:]]
    yield record(
        tc, "V10", rec_id, step, tc.moment(step), question, index,
        {"done": [{"verb": verb, "target": target} for verb, target in pairs],
         "started": [{"verb": verb, "target": target} for verb, target in started],
         "remaining_actions": total - len(done), "progress_bin": bin_name},
        {"type": "history", "bins": list(PROGRESS_BINS), "total_actions": total,
         "done_actions": len(done), "metric": "set_f1+bin_accuracy", "groups": groups},
        evidence({a.target: frame.pointable[a.target][1] for a in done
                  if a.target in frame.pointable}, tc),
        R.rationale(chain), frame.verified, view_confirmed=frame.confirmed,
        moment=tc.legend((step, "now")),
    )


# --------------------------------------------------------------------------- #
# V12 -- did anything change between these two frames
# --------------------------------------------------------------------------- #
V12_QUESTIONS = (
    "Did anything change between the moment before and the moment after? If so, what?",
    "Compare the machine before and after: is it in a different state?",
    "Do the pictures before and after show the same machine state, or did something change?",
)


def gen_v12(tc: TaskCtx, step: int) -> Iterator[dict]:
    """Change / no-change over a consecutive pair, both frames confirmed.

    The "no change" half comes from ``dupli`` rows -- a genuine near-duplicate
    the operator shot twice -- which is spec 8.2's own answer to how to build
    that negative. A ``failed`` step is deliberately **not** used: no state
    changed, but a hand or a tool usually did, and a question whose answer is
    "nothing changed" about a picture where something visibly moved is a
    question about the annotation, not about the machine.
    """
    frame, before = tc.frame(step), tc.frame(step - 1)
    if frame is None or before is None:
        return
    confirmed = frame.confirmed and before.confirmed
    if confirmed and not (frame.verified and before.verified):
        return
    if step not in tc.v12_steps:
        return
    kind = tc.ctx.step_type(step)
    rec = tc.ctx.steps.get(step)
    hidden: list[dict] = []
    reason = None
    if kind == StepType.DUPLI.value or (rec is not None and rec.dupli):
        changed, events = False, []
        reason = "dupli"
    elif captive_only_step(tc, step):
        # the log loosened the cooler's captive screws; the pictures cannot
        # show it, so the honest answer is "no visible change" (W3b)
        old, new = tc.state(step - 1), tc.state(step)
        hidden = [{"target": key, "old": old[key].state, "new": new[key].state}
                  for key in sorted(old.keys() & new.keys())
                  if old[key].state != new[key].state]
        changed, events = False, []
        reason = CAPTIVE_LOOSENED
    elif kind in NO_CHANGE_STEP_TYPES or kind == StepType.FAILED.value:
        return
    else:
        old, new = tc.state(step - 1), tc.state(step)
        events = [{"target": key, "old": old[key].state, "new": new[key].state}
                  for key in sorted(old.keys() & new.keys())
                  if old[key].state != new[key].state]
        if not events:
            # a `reorient` or an `auxiliary` step: the picture changed a great
            # deal and the machine did not, which is neither answer
            return
        changed = True
    rec_id = f"V12-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V12_QUESTIONS, rec_id)
    touched = [e["target"] for e in events]
    big = big_changes(tc, step - 1, step) if changed else []
    check = {"type": "exact", "derive": "changed",
             # on a view without confirmed rows only "did it change" is graded:
             # the small events riding along with the large one may not show
             "fields": ["changed", "events"] if confirmed else ["changed"],
             "step_type": kind}
    if reason:
        check["reason"] = reason
    if hidden:
        check["hidden_events"] = hidden  # what the log did that no picture shows
    yield record(
        tc, "V12", rec_id, step, [*tc.moment(step - 1), *tc.moment(step)], question, index,
        {"changed": changed, "events": events}, check,
        evidence({k: before.pointable[k][1] for k in touched if k in before.pointable},
                 tc),
        R.rationale([R.compare_frames([step - 1, step], tc.view),
                     *(observed(k, "changed", tc, before.pointable, step=step - 1)
                       for k in touched[:tc.budget])]),
        frame.verified and before.verified,
        negative=(None if changed else
                  CAPTIVE_LOOSENED if reason == CAPTIVE_LOOSENED else "dupli_no_change"),
        view_confirmed=confirmed,
        extra={"big_change": big, "change_visible": changed,
               **({"reason": reason} if reason else {})},
        moment=tc.legend((step - 1, "before"), (step, "after")),
    )


# --------------------------------------------------------------------------- #
# V15 -- two views of the same moment
# --------------------------------------------------------------------------- #
V15_CROSS_QUESTIONS = (
    "{label} cannot be judged in the first image. Using the second view, what "
    "state is it in, and which view answers better?",
    "The first view cannot see {label}. Read its state off the second view and "
    "say which view is the right one to ask.",
)
V15_MOMENT_QUESTIONS = (
    "Are these two images the same machine at the same moment of the teardown?",
    "Do these two views show the same disassembly state of the same machine?",
    "Two cameras, two pictures: were they taken at the same point of the "
    "disassembly?",
    "Is the machine in the second image in exactly the state shown in the first?",
)
#: How far away the negative half of the same-moment question may be taken from,
#: nearest first: a neighbouring step is the hardest honest negative there is.
#: Out to six steps (W3b): the negative needs a large part to have changed, and
#: three steps of screws often change none.
V15_MOMENT_OFFSETS = (2, -2, 1, -1, 3, -3, 4, -4, 5, -5, 6, -6)
#: The views on one capture clock: OAK1, OAK2 and the RealSense are triggered
#: together, so one step's frames are seconds apart. The scanner's ``ts`` is
#: not a capture time (consecutive scanner frames are 2 s apart and days off
#: the cameras' clock), so no pair with the scanner is checked by time.
SYNCED_VIEWS = frozenset({"oak1", "oak2", "rs"})
#: The most two synced frames of one step may be apart and still be "the same
#: moment" (review M: D42 RS steps 16-17 are a known raw-data shift of ~36 s).
V15_MAX_DT_S = 20.0


def gen_v15(tc: TaskCtx, step: int) -> Iterator[dict]:
    """Cross-view consistency, built from per-view ``visibility`` and nothing else.

    Spec v1.5 removed cross-view geometry from the design: there is no
    homography and no projection, so this task may not use one. What is left is
    exactly what the annotation does carry -- the *same instance*, at the *same
    logical step*, with a visibility of its own in each view -- and that is
    enough for both halves of the spec's V15: which view can answer, and whether
    two frames are the same moment at all.
    """
    frame = tc.frame(step)
    if frame is None:
        return
    state = tc.state(step)
    for other, frames in sorted(tc.others.items()):
        mate = frames.get(step)
        if mate is None:
            continue  # the other camera has no picture of this step (missing)
        # cross needs both views confirmed: it reads a visibility in each
        if frame.verified and mate.verified:
            yield from _v15_cross(tc, step, frame, other, mate, state)
        # moment needs nothing of either view's annotation (W3b): the four
        # cameras fire on one key press, so the same step *is* the same moment
        yield from _v15_moment(tc, step, frame, other, frames)


def _v15_cross(tc: TaskCtx, step: int, frame: FrameData, other: str,
               mate: FrameData, state: FrameState) -> Iterator[dict]:
    seed = tc.seed(step, f"V15|{other}")
    blind = [key for key, row in sorted(frame.rows.items())
             if row.get("visibility") not in ANSWERABLE
             and key in mate.pointable and key in state
             and tc.ctx.cls_of(key) is not None
             and state_is_askable(tc.tax, tc.ctx.cls_of(key))
             and not is_captive_screw(tc.ctx, key)]
    # named in words, never by ordinal (spec 3.2): only a part alone of its class
    named = [k for k in blind if unique_on_desktop(tc, k)]
    if len(named) < len(blind):
        tc.skip("V15", "ambiguous_name")
    for instance in stable_order(named, seed)[:tc.budget]:
        row, box = mate.pointable[instance]
        here = frame.rows[instance]
        rec_id = f"V15-{frame_id(tc.desktop, tc.view, step)}-{other}-{instance}"
        index, template = pick(V15_CROSS_QUESTIONS, rec_id)
        yield record(
            tc, "V15", rec_id, step, [frame, mate],
            template.format(label=class_words(tc, [instance])), index,
            {"state": visible_state(tc.ctx, instance, state[instance].state),
             "best_view": other},
            {"type": "exact", "derive": "cross_view", "instance": instance,
             "blind_view": tc.view, "answer_view": other,
             "fields": ["state", "best_view"],
             **({"state_space": "presence"}
                if tc.ctx.cls_of(instance) == "screw" else {})},
            evidence({instance: box}, tc,
                     {instance: seen_as(here) | {
                         f"visibility_{other}": row.get("visibility")}}),
            R.rationale([
                R.propagate_state(instance, None),
                R.observe(instance, state[instance].state, other, box,
                          row.get("visibility"), step=step,
                          occluder=row.get("occluder")),
            ]),
            row_verified([row], frame.verified and mate.verified),
            views=[tc.view, other], extra={"v15_variant": "cross"},
        )


def _v15_moment(tc: TaskCtx, step: int, frame: FrameData, other: str,
                frames: dict[int, FrameData]) -> Iterator[dict]:
    """Same moment or not, one of each.

    The negative is a nearby frame of the other view whose folded state really
    **differs** from this one. Two steps apart is not enough on its own: a
    ``dupli`` pair or a pair of ``reorient`` steps leaves the machine in exactly
    the same state, and "no" would then be the wrong answer to "do these show
    the same disassembly state?".

    W3b: no ``verified`` gate on either frame -- the answer is the step
    alignment and the state machine, never a mask -- and the negative must
    differ by a **large part** (:data:`BIG_PARTS`): a one-screw difference is
    undecidable at RealSense / OAK2 resolution, so "not the same moment" would
    be a question about the log, not the pictures. Nearest offset first, out to
    :data:`V15_MOMENT_OFFSETS`.
    """
    here = tc.state(step)
    mate_now = frames[step]
    if (tc.view in SYNCED_VIEWS and other in SYNCED_VIEWS
            and frame.ts is not None and mate_now.ts is not None
            and abs(frame.ts - mate_now.ts) > V15_MAX_DT_S):
        # the two cameras did not fire together here: "same moment" is not
        # certain, so neither half of the pair is asked
        tc.skip("V15", f"moment_ts_mismatch_{other}")
        return
    candidates: list[tuple[bool, FrameData, list[str]]] = [(True, mate_now, [])]
    for delta in V15_MOMENT_OFFSETS:
        mate = frames.get(step + delta)
        if mate is None or mate.step == step:
            continue
        if tc.state(mate.step) == here:
            continue  # the same state at another step: "no" would be a lie
        big = big_changes(tc, min(step, mate.step), max(step, mate.step))
        if not big:
            continue  # only small parts differ: not decidable from these views
        candidates.append((False, mate, big))
        break
    if len(candidates) == 1:
        tc.skip("V15", f"moment_no_big_negative_{other}")
        return  # no honest negative near this frame: skip the pair entirely
    shared = sorted(set(frame.rows) & set(frames[step].rows))
    for same, mate, big in candidates:
        rec_id = (f"V15m-{frame_id(tc.desktop, tc.view, step)}-{other}"
                  f"-s{mate.step:03d}")
        index, question = pick(V15_MOMENT_QUESTIONS, rec_id)
        yield record(
            tc, "V15", rec_id, step, [frame, mate], question, index,
            {"same_moment": same},
            {"type": "exact", "derive": "same_moment", "same_moment": same,
             "fields": ["same_moment"], "other_view": other,
             "other_step": mate.step, "variant": "moment"},
            evidence({}, tc, {k: {} for k in shared[:tc.budget]}),
            R.rationale([R.compare_frames([step, mate.step], other)]),
            frame.verified and mate.verified, views=[tc.view, other],
            negative=None if same else "different_moment",
            extra={"big_change": big, "v15_variant": "moment"},
        )


# --------------------------------------------------------------------------- #
# V16 -- is this plan legal, and where does it first go wrong
# --------------------------------------------------------------------------- #
V16_QUESTIONS = (
    "Here is a proposed continuation of the teardown. Can it be carried out as "
    "written? If not, which step is the first that cannot be?",
    "Check this plan against the machine's current state: is every step "
    "possible in this order, and where does it first break?",
    "Verify the following sequence. Say whether it can be carried out and, if "
    "not, the index of the first impossible action.",
)
#: How many logged actions a candidate plan is made of.
#:
#: Five, so that the first error can land on index 0, 1, 2 or 3. It can never
#: land on the **last** index of a rearranged plan, and that is a fact about
#: disassembly rather than a gap in the corruption search: taking a part out
#: only ever *clears* preconditions, so an action moved later is never more
#: blocked than it was, and the step that breaks is always one that has been
#: pulled forward. The reachable range is therefore ``0 .. len(plan) - 2``.
V16_PLAN_LEN = 5


def _group_members(tc: TaskCtx, target: str) -> Optional[tuple[str, ...]]:
    """The members of a group id, or ``None`` for an instance key / cable node."""
    group = tc.group_by_id(target)
    return None if group is None else group.members


def _simulate(tc: TaskCtx, step: int, plan: Sequence[tuple[str, str]],
              start: Optional[FrameState] = None
              ) -> tuple[Optional[int], list[Edge], Optional[str]]:
    """Replay ``plan`` from the state after ``step``; ``(first bad index, edge)``.

    Through :func:`unmet_for`, so this asks the constraint graph exactly the
    question V4, V5 and V6 ask it. A private re-implementation used to let V16
    disagree with them about a ``cable:*`` blocker that is not in the snapshot
    (satisfied here, unmet there) and about an edge type nobody recognises
    (gating nothing here, everything there).

    A **group action** (a group id as the target, spec 3.1) is the verb on every member
    still in the asked situation at that point (:func:`~tda.core.export.vlm_groups.to_do`):
    it fails on the union of their unmet edges, and with no member left to do it is a
    state-inapplicable step. A captive screw already loosened is left as it is (spec 6.6).
    """
    sim = _copy_state(tc.state(step) if start is None else start)
    for index, (verb, target) in enumerate(plan):
        members = _group_members(tc, target)
        if members is not None:
            todo = to_do(tc.ctx, sim, verb, members)
            if not todo:
                return index, [], "not_present"
            bad = sorted({e for m in todo for e in unmet_for(tc, verb, m, sim)},
                         key=lambda e: (e.type, e.target, e.blocker))
            if bad:
                return index, bad, "violates_edge"
            for m in todo:
                rec = tc.ctx.instances[m]
                if verb_applies(tc.tax, rec.cls, rec.attrs, verb, sim[m].state):
                    _apply_effect(tc.tax, tc.ctx.instances, sim, m, verb)
            continue
        rec = tc.ctx.instances.get(target)
        # a `cable:*` node is a real step of the teardown with no instance row:
        # releasing a loom is what makes the part behind it reachable, and a
        # plan that leaves it out is a plan that cannot be carried out
        cls = rec.cls if rec is not None else (
            CABLE_CLASS if target.startswith(CABLE_PREFIX) else None)
        if cls is None:
            return index, [], "not_present"
        attrs = rec.attrs if rec is not None else {}
        bad = unmet_for(tc, verb, target, sim)
        if bad:
            # every edge this step breaks, not the first one the list happened
            # to hold: "which constraint stopped it" has no single answer when
            # a part is both screwed down and still plugged in
            return (index, sorted(bad, key=lambda e: (e.type, e.target, e.blocker)),
                    "violates_edge")
        current = (sim[target].state if target in sim
                   else tc.tax.default_state(cls))
        if not verb_applies(tc.tax, cls, attrs, verb, current):
            return index, [], inapplicable_reason(sim, target)
        _apply_effect(tc.tax, tc.ctx.instances, sim, target, verb)
    return None, [], None


def _state_before(tc: TaskCtx, step: int, plan: Sequence[tuple[str, str]],
                  index: int) -> FrameState:
    """The simulated state a plan reaches just before its ``index``-th action."""
    sim = _copy_state(tc.state(step))
    for verb, target in plan[:index]:
        members = _group_members(tc, target)
        for m in (to_do(tc.ctx, sim, verb, members) if members is not None else [target]):
            _apply_effect(tc.tax, tc.ctx.instances, sim, m, verb)
    return sim


def grouped_plan(tc: TaskCtx, state: FrameState,
                 pairs: Sequence[tuple[str, str]],
                 keep: frozenset = frozenset()) -> list[tuple[str, str]]:
    """Member actions folded into **group actions** (spec 3.1, 6.1).

    One ``(verb, gid)`` at the place of the first member action, for a group with at
    least two members still to do in ``state``; the other members' actions go. The graph
    orders interchangeable parts "unordered" already. Parts of a class in ``keep`` stay one
    step each (V7's constrained form counts the plugs one by one).
    """
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for verb, key in pairs:
        group = tc.group_of(key)
        ident = key
        if group is not None and tc.ctx.cls_of(key) not in keep \
                and len(to_do(tc.ctx, state, verb, group.members)) >= 2:
            ident = group.gid
        if (verb, ident) in seen:
            continue
        seen.add((verb, ident))
        out.append((verb, ident))
    return out


def pending_captive(tc: TaskCtx, state: FrameState) -> list[PartGroup]:
    """Groups of captive screws whose held part is still in (spec 6.6).

    "Loosen the cooler screws" stays one required group action until the part they hold
    comes off: whether some of them are already loose is not in any picture.
    """
    out: dict[str, PartGroup] = {}
    for key, group in tc.groups.items():
        rec = tc.ctx.instances.get(key)
        if not is_captive_screw(tc.ctx, key) or rec.fastens is None:
            continue
        if present(state, rec.fastens) and to_do(tc.ctx, state, "unscrew", group.members):
            out[group.gid] = group
    return [out[g] for g in sorted(out)]


def with_captive(tc: TaskCtx, state: FrameState,
                 plan: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """``plan`` with "unscrew the <captive group>" put first where a plan touches the held
    part and lacks it (spec 6.6)."""
    for group in reversed(pending_captive(tc, state)):
        held = tc.ctx.instances[group.members[0]].fastens
        if ("unscrew", group.gid) in plan:
            continue
        if any(t == held for _, t in plan):
            plan = [("unscrew", group.gid), *plan]
    return plan


def plan_groups(tc: TaskCtx, plan: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    """``gid -> members`` of every group action in ``plan`` (label side)."""
    out = {}
    for _, target in plan:
        members = _group_members(tc, target)
        if members is not None:
            out[target] = list(members)
    return out


def _corruptions(plan: Sequence[tuple[str, str]], seed: str) -> list[tuple[str, str]]:
    """Every corruption of ``plan`` worth trying, as ``(family, tag)`` descriptors.

    Three families. Adjacent **swaps** put the first error at ``i``; **moves**
    put it where the moved action lands. Neither can ever break the *last* step
    of a valid plan -- disassembly only clears preconditions, so an action
    pushed later is never more blocked -- which is why the third family exists:
    a **substitute** replaces one step with an action that is certainly wrong at
    exactly that point, so the error can land anywhere, the last index included.

    The descriptors are materialised one at a time (a substitute has to look at
    the simulated state), in a seeded order, so the three families are mixed.
    """
    out: list[tuple[str, str]] = []
    for i in range(len(plan) - 1):
        out.append(("swap", f"swap{i}"))
    for j in range(len(plan)):
        for i in range(len(plan)):
            if abs(i - j) < 2:
                continue  # that is the adjacent swap, already listed
            out.append(("move", f"move{j}to{i}"))
    for i in range(len(plan)):
        out.append(("substitute", f"substitute{i}"))
    return stable_order(out, seed, lambda it: it[1])


def _rearranged(plan: Sequence[tuple[str, str]], tag: str) -> list[tuple[str, str]]:
    """Apply a ``swap``/``move`` tag to a plan."""
    out = list(plan)
    if tag.startswith("swap"):
        i = int(tag[len("swap"):])
        out[i], out[i + 1] = out[i + 1], out[i]
        return out
    j, i = (int(part) for part in tag[len("move"):].split("to"))
    out.insert(i, out.pop(j))
    return out


def _substituted(tc: TaskCtx, frame: FrameData, plan: Sequence[tuple[str, str]],
                 index: int, seed: str
                 ) -> Optional[list[tuple[str, str]]]:
    """Replace ``plan[index]`` with an action that cannot be done at that point.

    Same verb wherever one is available, so the corrupted plan still reads like
    a plan and the difference is a fact about the machine rather than about the
    wording. A graph-blocked action wherever one exists (spec 3.3), a part the frame can
    outline, and never one member of a group whose other members are still to do: the
    members of a group are never told apart (spec 6.4).
    """
    step = frame.step
    state = _state_before(tc, step, plan, index)
    verb, target = plan[index]
    candidates = prefer_graph([c for c in [*blocked_in(tc, state),
                                           *inapplicable_in(tc, state, step)]
                               if (c.verb, c.target) not in set(plan)])
    blocked = {(c.verb, c.target) for c in candidates if c.kind in GRAPH_KINDS}
    cls = target_class(tc.ctx, (_group_members(tc, target) or [target])[0])
    ordered = tc.matched(candidates, verb, cls, f"{seed}|substitute{index}")
    for chosen in ordered:
        if mark_box(tc, frame, chosen.target) is None:
            continue
        ident, members = group_action(tc, state, chosen.verb, chosen.target,
                                      lambda m, v=chosen.verb: (v, m) in blocked)
        group = tc.group_of(chosen.target)
        if ident == chosen.target and group is not None and \
                len(to_do(tc.ctx, state, chosen.verb, group.members)) >= 2:
            continue
        if (chosen.verb, ident) in set(plan):
            continue
        out = list(plan)
        out[index] = (chosen.verb, ident)
        return out
    return None


def _name_plan(tc: TaskCtx, frame: FrameData, plan: Sequence[tuple[str, str]]
               ) -> Optional[tuple[Namer, list[str]]]:
    """Each step's words, every part outlined; ``None`` when a part cannot be.

    A ``cable:*`` node has no outline; it is named in words ("the PSU cable").
    """
    namer = Namer(tc, frame)
    state = tc.state(frame.step)
    for verb, target in plan:
        if target.startswith(CABLE_PREFIX):
            continue
        members = _group_members(tc, target)
        if members is not None:
            got = namer.group(target, to_do(tc.ctx, state, verb, members) or members)
        else:
            got = namer.part(target)
        if got is None:
            return None
    words = [cable_words(t) if t.startswith(CABLE_PREFIX) else namer.say(t)
             for _, t in plan]
    return namer, words


def cable_words(node: str) -> str:
    """``cable:psu`` -> "the PSU cable": a virtual node has no outline, only a name."""
    owner = node[len(CABLE_PREFIX):]
    return f"the {' '.join(_words(owner))} cable" if owner else "the cable"


def plan_reason(tc: TaskCtx, step: int, plan: Sequence[tuple[str, str]], words: Sequence[str],
                first_bad: Optional[int], edges: Sequence[Edge],
                code: Optional[str]) -> Optional[str]:
    """Why a V16 plan breaks, in what the picture shows (owner 2026-11-04), or ``None``.

    "step 3 (remove part B) cannot be done yet: the CPU socket lever still locks the CPU";
    a step on a part that has gone or is already where the verb would take it says so.
    """
    if first_bad is None:
        return None
    verb, target = plan[first_bad]
    head = f"step {first_bad + 1} ({verb_phrase(verb, words[first_bad])})"
    sim = _state_before(tc, step, plan, first_bad)
    if edges:
        members = list(_group_members(tc, target) or [target])
        clauses = reason_clauses(tc, edges, sim, met=False, asked=(target, members),
                                 every=gated_edges(tc, verb, members))
        return f"{head} cannot be done yet: {_and(clauses)}"
    what, many = asked_words(tc, target)
    be = "are" if many else "is"
    if code and code.startswith("already_"):
        return f"{head} cannot be done: {what} {be} already {code[len('already_'):]}"
    return f"{head} cannot be done: {what} {be} no longer there"


def plan_steps_text(tc: TaskCtx, step: int, plan: Sequence[tuple[str, str]],
                    words: Sequence[str], first_bad: Optional[int], bad_text: Optional[str]) -> str:
    """V16's reasoning chain, step by step in visible terms: why each step can be done at
    that point of the replay, up to the first that cannot (owner 2026-11-04)."""
    parts = []
    last = len(plan) if first_bad is None else first_bad
    for i in range(last):
        verb, target = plan[i]
        sim = _state_before(tc, step, plan, i)
        members = _group_members(tc, target)
        members = list(to_do(tc.ctx, sim, verb, members) or members) if members else [target]
        why, _ = feasibility_reason(tc, verb, target, members, sim, [])
        parts.append(f"Step {i + 1} ({verb_phrase(verb, words[i])}): {why}.")
    if first_bad is None:
        parts.append("So every step can be carried out in this order.")
    else:
        parts.append(f"{bad_text[:1].upper() + bad_text[1:]}.")
        parts.append(f"So the plan cannot be carried out: step {first_bad + 1} is the first "
                     f"that breaks.")
    return " ".join(parts)


def _plan_record(tc: TaskCtx, frame: FrameData, plan: Sequence[tuple[str, str]],
                 first_bad: Optional[int], edges: Sequence[Edge], tag: str,
                 family: Optional[str] = None,
                 reason: Optional[str] = None) -> Optional[dict]:
    step = frame.step
    named = _name_plan(tc, frame, plan)
    if named is None:
        tc.skip("V16", "unmarkable_part")
        return None
    namer, words = named
    rec_id = f"V16-{frame_id(tc.desktop, tc.view, step)}-{tag}"
    index, question = pick(V16_QUESTIONS, rec_id)
    listed = [{"verb": v, "target": t} for v, t in plan]
    state = tc.state(step)
    chain: list[dict] = []
    for edge in edges:
        current = state[edge.blocker].state if edge.blocker in state else None
        chain.append(R.recall_relation(edge.type, edge.target, edge.blocker,
                                       edge.necessity, edge.mode))
        chain.append(observed(edge.blocker, current, tc, frame.pointable))
        chain.append(R.check_precondition(edge.blocker,
                                          required_states(edge.type, edge.mode),
                                          current, False))
    if not edges:
        for _, t in plan[:tc.budget]:
            for m in (_group_members(tc, t) or [t])[:1]:
                chain.append(observed(m, state[m].state if m in state else None,
                                      tc, frame.pointable))
    labels = [e.label() for e in edges]
    # validity is what a replay under the rules says -- the stored graph, this frame's
    # occlusion, loose parts first -- never "it was demonstrated" (owner 2026-11-04)
    source = (SIMULATED if first_bad is None
              else graph_kind(edges) if edges else STATE_INAPPLICABLE)
    # the reason in what the picture shows (owner 2026-11-04); the code stays label side
    text = plan_reason(tc, step, plan, words, first_bad, edges, reason)
    members = sorted({m for _, t in plan for m in (_group_members(tc, t) or [t])})
    check = {"type": "plan", "plan": listed, "metric": "accuracy+localisation",
             "truth_source": source, "groups": plan_groups(tc, plan),
             "reason_code": reason, "validity": "simulated",
             "moment": moment_json(tc, members)}
    if family is not None:
        check["corruption"] = family
        check["corruption_tag"] = tag
    shown = [m for _, t in plan for m in (_group_members(tc, t) or [t])]
    return record(
        tc, "V16", rec_id, step, tc.moment(step, marked=True),
        question + "\n" + "\n".join(
            f"{i + 1}. {verb_phrase(v, words[i])}" for i, (v, _) in enumerate(plan)),
        index,
        {"valid": first_bad is None, "first_error_index": first_bad,
         "violated_edge": labels[0] if labels else None,
         "violated_edges": labels, "reason": text},
        check,
        evidence({t: frame.pointable[t][1] for t in shown if t in frame.pointable},
                 tc, plan_length=len(listed)),
        dict(R.rationale(chain), text=plan_steps_text(tc, step, plan, words, first_bad, text)),
        frame.verified,
        negative=None if first_bad is None else family,
        truth_source=source, view_confirmed=frame.confirmed, marks=namer.marks(), mark_names=namer.names(),
        terms=[target_class(tc.ctx, (_group_members(tc, t) or [t])[0]) for _, t in plan],
        moment=tc.legend((step, "now"), marked=True),
    )


def gen_v16(tc: TaskCtx, step: int) -> Iterator[dict]:
    """A demonstrated continuation and, where one exists, a corruption of it.

    The valid plan is the logged suffix and nothing else: a rearrangement the
    graph happens to permit is not known to work, because the graph is an upper
    bound. The invalid one is a rearrangement that violates a required edge --
    certainly impossible, and the edge says why. A corruption that violates
    nothing is **not** emitted: it is a different plan of unknown validity, and
    calling it an error would teach exactly the wrong lesson.

    The redesign (2026-11-03): the plan is told in **group actions** (one "unscrew the
    motherboard screws" for the members still to do, :func:`grouped_plan`), and "unscrew
    the CPU cooler screws" stays in it until the cooler comes off (:func:`with_captive`).
    """
    frame = tc.frame(step)
    if frame is None or tc.excluded or not tc.stepwise:
        return  # D66 (several parts at once) asks no step-ordered question
    with tc.at(step):
        made = list(_gen_v16(tc, frame, step))
    yield from made


def _gen_v16(tc: TaskCtx, frame: FrameData, step: int) -> Iterator[dict]:
    state = tc.state(step)
    logged: list[tuple[str, str]] = []
    for candidate in tc.log_steps:
        if candidate <= step:
            continue
        if candidate in tc.illegal:
            return  # a suffix built on a contradiction is not a plan
        for action in tc.named_actions(candidate):
            # a plan is a sequence of disassembly actions; `reorient` is a
            # capture action with no state effect and nothing gating it. A
            # `cable:*` node has no instance row and belongs in the plan all the
            # same -- releasing a loom is what frees the part behind it.
            known = (action.target in tc.ctx.instances
                     or action.target.startswith(CABLE_PREFIX))
            if known and tc.changes_state(action):
                logged.append((action.verb, action.target))
        if len(grouped_plan(tc, state, logged)) >= V16_PLAN_LEN:
            break
    plan = with_captive(tc, state, grouped_plan(tc, state, logged))[:V16_PLAN_LEN]
    if len(plan) < 2:
        return
    first_bad, _, _ = _simulate(tc, step, plan)
    if first_bad is not None:
        # the logged suffix breaks a rule of this moment (an occlusion, a loose part): the
        # rule, not the teardown, is what is in doubt there -- say nothing
        tc.skip("V16", "log_breaks_rules")
        return

    seed = tc.seed(step, "V16")
    corruption = None
    for family, tag in _corruptions(plan, seed):
        if family == "substitute":
            corrupted = _substituted(tc, frame, plan, int(tag[len("substitute"):]), seed)
            if corrupted is None:
                continue
        else:
            corrupted = _rearranged(plan, tag)
        bad_at, broken, reason = _simulate(tc, step, corrupted)
        if bad_at is None:
            continue
        if family != "substitute" and not broken:
            # a rearrangement that violates nothing is a different plan of
            # unknown validity, not a wrong one (spec 8.2 principle 6)
            continue
        corruption = (family, tag, corrupted, bad_at, broken, reason)
        break
    if corruption is None:
        return  # nothing about this suffix is certainly wrong: say nothing
    # the plan the log suggests, valid because its replay under the rules is clean (the
    # record kind says so: `valid`, not `true`)
    true = _plan_record(tc, frame, plan, None, (), "valid")
    if true is None:
        return
    family, tag, corrupted, bad_at, broken, reason = corruption
    wrong = _plan_record(tc, frame, corrupted, bad_at, broken, tag, family, reason)
    if wrong is None:
        return
    yield true
    yield wrong


# --------------------------------------------------------------------------- #
# V7 -- the rest of the CPU plan, asked half-way through it
# --------------------------------------------------------------------------- #
V7_QUESTIONS = (
    "The goal is to take {goal} out of this machine, and the work towards it has "
    "already started. List, in order, the actions still needed to remove it.",
    "Some of the steps towards removing {goal} have been done. What remains? "
    "Give the remaining actions in order.",
    "Plan the rest of the job: starting from the machine as it is now, which "
    "actions, in which order, get {goal} out?",
    "Removal of {goal} is under way. Write the sequence of actions that "
    "finishes it from this state.",
    "Looking at the machine as it stands, give the ordered list of disassembly "
    "actions that still have to happen before {goal} can be lifted out.",
    "Continue the teardown towards {goal}: list every remaining action, one per "
    "step, ending with its removal.",
)


@dataclass(frozen=True)
class Requirement:
    """One state change every legal plan from here must contain (V7)."""

    target: str
    states: frozenset
    via: tuple = ()          # the edges that make it necessary, goal-side first
    pixel_only: bool = False  # an edge on its path is pixel-imported

    def as_json(self) -> dict:
        return {"target": self.target, "states": sorted(self.states),
                "via": [e.label() for e in self.via], "pixel_only": self.pixel_only}


def _candidate_verbs(tc: TaskCtx, key: str, current: str,
                     wanted: frozenset) -> list[str]:
    """Every verb that takes ``key`` from ``current`` into ``wanted`` in one step.

    Removing always satisfies an edge (``unmet`` counts a removed blocker as
    clear), so ``remove`` is a candidate whenever it applies.
    """
    rec = tc.ctx.instances.get(key)
    cls = rec.cls if rec is not None else (
        CABLE_CLASS if key.startswith(CABLE_PREFIX) else None)
    if cls is None:
        return []
    attrs = rec.attrs if rec is not None else {}
    out = []
    for verb in tc.tax.verbs:
        if not verb_applies(tc.tax, cls, attrs, verb, current):
            continue
        new = verb_effect(tc.tax, cls, attrs, verb)
        if new in wanted or new == REMOVED:
            out.append(verb)
    return out


def required_actions(tc: TaskCtx, state: FrameState, goal: str) -> list[Requirement]:
    """The state changes **every** legal plan from ``state`` to ``goal`` removed contains.

    The goal itself has to reach ``removed``. Then, recursively: an unmet edge
    on a required node is itself required when it gates **every** verb that
    could bring that node where it has to be -- a cooler can be swung aside or
    taken out, both are gated by its screws (``fastened_by``), only the second
    by its fan lead (``connected_to``), so the screws are required and the lead
    is not. That is the strongest claim the graph supports: anything else a plan
    does may be needed or may be redundant, and V7 does not penalise it.
    """
    out: dict[tuple[str, frozenset], Requirement] = {}
    seen: set[tuple[str, frozenset]] = set()

    def need(key: str, wanted: frozenset, via: tuple) -> None:
        current = blocker_state(state, key)
        if current is None or current == REMOVED or current in wanted:
            return
        mark = (key, wanted)
        if mark in seen:
            return
        seen.add(mark)
        # a requirement exists only because of every edge on its path, so one
        # pixel-imported edge on it makes it a pixel-certain requirement
        pixel_only = any(is_pixel_edge(e) for e in via)
        out[mark] = Requirement(key, wanted, via, pixel_only)
        verbs = _candidate_verbs(tc, key, current, wanted)
        if not verbs:
            return
        for edge in tc.by_target.get(key, ()):
            if not unmet([edge], state, "required"):
                continue
            if not all(v in gated_verbs(edge.type, edge.mode) for v in verbs):
                continue
            need(edge.blocker, required_states(edge.type, edge.mode),
                 via + (edge,))

    need(goal, frozenset({REMOVED}), ())
    return sorted(out.values(), key=lambda r: (len(r.via), r.target,
                                               sorted(r.states)))


def _v7_goal_ok(tc: TaskCtx, goal: str) -> Optional[str]:
    """Why ``goal`` cannot be a V7 goal on this desktop, or ``None``."""
    if not tc.askable(goal):
        return "goal_not_askable"
    # the stored graph's word, never a moment's occlusion edge
    covered = [e for e in active_edges(tc.edges) if e.target == goal
               if e.type == "covered_by"
               and tc.ctx.cls_of(e.blocker) == V7_COOLER_CLASS]
    if not covered:
        # W3a's S1 report: a desktop whose cooler set is unresolved has no
        # `covered_by(cpu <- cooler)` edge, and its plan lifts the CPU through
        # the cooler. Until W3a's rule lands those desktops ask no V7.
        return "cooler_unresolved"
    return None


def gen_v7(tc: TaskCtx, step: int) -> Iterator[dict]:
    """The plan to take the CPU out (owner 2026-11-04), in two forms.

    * **free**: "plan the steps to remove the CPU" (outlined in red only where it is
      visible). Valid = every step legal under the stored graph, this frame's occlusion
      and loose parts first, and the goal reached; required = the minimal prerequisite
      closure from those rules (:func:`required_actions`), never the demonstrated
      sequence -- on a bare board: open the socket lever, take the CPU out. The gold is
      the rules' shortest plan, same-kind actions folded into one step.
    * **constrained**: the same, under one rule in words, no number ("every connector must
      be unplugged before the CPU is taken out"): the plan must keep to it; the label
      counts the plugs it has to unplug, so the scorer reports how many the answer unplugs.

    Asked only where the plan is visibly under way (a frame where one action of the plan
    from the first frame is done and the CPU is still in: from the first frame it is a
    recipe a model writes without looking). No redundancy penalty.
    """
    frame = tc.frame(step)
    if frame is None or tc.excluded or not tc.stepwise:
        return  # D66 (several parts at once) asks no step-ordered question
    with tc.at(step):
        made = list(_gen_v7(tc, frame, step))
    yield from made


def _gen_v7(tc: TaskCtx, frame: FrameData, step: int) -> Iterator[dict]:
    goals = sorted(k for k, rec in tc.ctx.instances.items()
                   if rec.cls == V7_GOAL_CLASS)
    for goal in goals:
        why = _v7_goal_ok(tc, goal)
        if why:
            tc.skip("V7", why)
            continue
        here = tc.state(step)
        inst = here.get(goal)
        if inst is None or inst.state == REMOVED:
            continue  # nothing left to plan
        removal = next((s for s in tc.log_steps if s > step
                        and tc.state(s).get(goal) is not None
                        and tc.state(s)[goal].state == REMOVED), None)
        if removal is None:
            tc.skip("V7", "goal_never_removed")
            continue
        initial = tc.state(0)
        if goal not in tc._v7_plan0:
            tc._v7_plan0[goal] = remaining_plan(tc.ctx.instances, tc.edges, initial,
                                                goal, tc.tax)
        plan0 = tc._v7_plan0[goal]
        if not plan0:
            tc.skip("V7", "no_initial_plan")
            continue
        started = [t for _, t in plan0
                   if blocker_state(here, t) != blocker_state(initial, t)]
        # the progress has to be one a picture shows (W3b): a loosened captive
        # cooler screw looks like a tight one
        if not [t for t in started if not is_captive_screw(tc.ctx, t)]:
            tc.skip("V7", "not_started_visibly" if started else "not_started")
            continue
        if any(s in tc.illegal for s in tc.log_steps if step < s <= removal):
            tc.skip("V7", "illegal_step_ahead")
            continue
        demonstrated = [(a.verb, a.target) for s in tc.log_steps
                        if step < s <= removal for a in tc.named_actions(s)
                        if (a.target in tc.ctx.instances
                            or a.target.startswith(CABLE_PREFIX))
                        and tc.changes_state(a)]
        made = _v7_form(tc, frame, goal, demonstrated, started, removal, None)
        if made is None:
            continue
        yield made
        for rule in v7_constraints(tc, step, goal, made["label"]["answer"]["plan"]):
            with tc.at(step, rule["edges"]):
                ruled = _v7_form(tc, frame, goal, demonstrated, started, removal, rule)
            if ruled is not None:
                yield ruled
                break
        else:
            tc.skip("V7", "no_constraint")


V7_CONSTRAINED_QUESTIONS = (
    "Plan the steps to take {goal} out of this machine, starting from the machine as it is "
    "now, under this rule: {rule} List the actions in order.{each}",
    "Rule for this teardown: {rule} Keeping to it, which actions, in which order, get {goal} "
    "out of this machine from this state?{each}",
    "Write the sequence of disassembly actions that takes {goal} out of this machine from "
    "this state. The plan must keep to this rule: {rule}{each}",
)
#: The constrained question asks the plugs one by one: how many the answer unplugs is
#: what the scorer counts.
EACH_CONNECTOR = " Give each connector you unplug a step of its own."


def v7_constraints(tc: TaskCtx, step: int, goal: str,
                   free_plan: Sequence[dict]) -> list[dict]:
    """V7's candidate rules, in words and without a number, each changing the plan.

    "Every connector must be unplugged before the CPU is taken out" where at least two
    plugs are still in and the free plan leaves one of them in; else "the <A> has to come
    out before the CPU is taken out" for a large part A still in whose own way out needs
    nothing of the CPU and that the free plan does not touch.
    """
    state = tc.state(step)
    if tc.moment_at(step).loose:
        return []  # a loose part comes out first: a rule on top of that is not asked
    goal_words = asked_words(tc, goal)[0]
    plugged = sorted(k for k, rec in tc.ctx.instances.items()
                     if rec.cls == "connector" and tc.askable(k)
                     and state.get(k) is not None and state[k].state == "plugged")
    planned = set()
    for item in free_plan:
        members = _group_members(tc, item["target"])
        planned.update(members or [item["target"]])
    out: list[dict] = []
    if len(plugged) >= 2 and not set(plugged) <= planned:
        edges = [Edge(type="connected_to", target=goal, blocker=c, source=CONSTRAINT_SOURCE,
                      status="accepted", reason="the question's rule") for c in plugged]
        out.append({"kind": "until_connectors",
                    "rule": f"Every connector must be unplugged before {goal_words} is taken out.",
                    "held": [goal], "until": plugged, "edges": edges, "count": len(plugged)})
    bigs = sorted(k for k, rec in tc.ctx.instances.items()
                  if rec.cls in RULE_FIRST_CLASSES
                  and tc.askable(k) and present(state, k) and unique_on_desktop(tc, k)
                  and k not in planned)
    for a in bigs:
        plan = remaining_plan(tc.ctx.instances, tc.cur_edges, state, a, tc.tax)
        if not plan or any(t == goal for _, t in plan):
            continue
        words = asked_words(tc, a)[0]
        edges = [Edge(type="blocked_by", target=goal, blocker=a, mode="contained",
                      source=CONSTRAINT_SOURCE, status="accepted", reason="the question's rule")]
        out.append({"kind": "before",
                    "rule": f"{words[:1].upper() + words[1:]} has to come out before "
                            f"{goal_words} is taken out.",
                    "held": [goal], "until": [a], "edges": edges})
    return out


def rules_plan(tc: TaskCtx, state: FrameState, goal: str) -> Optional[list[tuple[str, str]]]:
    """The rules' shortest plan to take ``goal`` out (the active moment's edges, loose parts
    first), where no part is swung aside that this machine's log never swings (R14: only a
    hinged part is): such a step is replaced by taking the part out, its own way out first,
    and the rest re-planned from there. ``None`` when there is no such plan."""
    sim = _copy_state(state)
    out: list[tuple[str, str]] = []
    for _ in range(12):
        plan = remaining_plan(tc.ctx.instances, tc.cur_edges, sim, goal, tc.tax)
        if plan is None:
            return None
        bad = next((i for i, (v, t) in enumerate(plan)
                    if v == "displace" and t not in tc.displaced_in_log), None)
        if bad is None:
            return out + list(plan)
        head = list(plan[:bad])
        for verb, target in head:
            _apply_effect(tc.tax, tc.ctx.instances, sim, target, verb)
        sub = remaining_plan(tc.ctx.instances, tc.cur_edges, sim, plan[bad][1], tc.tax)
        if not sub or any(v == "displace" and t not in tc.displaced_in_log for v, t in sub):
            return None
        for verb, target in sub:
            _apply_effect(tc.tax, tc.ctx.instances, sim, target, verb)
        out += head + [a for a in sub if a not in head]
    return None


def plan_words(tc: TaskCtx, state: FrameState, plan: Sequence[tuple[str, str]]) -> list[str]:
    """Each step of a plan said in words, no ordinal: "unplug the connectors"."""
    sim = _copy_state(state)
    out = []
    for verb, target in plan:
        members = _group_members(tc, target)
        targets = (to_do(tc.ctx, sim, verb, members) or list(members)) if members else [target]
        if target.startswith(CABLE_PREFIX):
            words = cable_words(target)
        elif members:
            words = f"the {tc.group_by_id(target).label}"
        else:
            words = target_words(tc, targets)
        out.append(f"{VERB_WORDS.get(verb, verb)} {words}")
        for m in targets:
            _apply_effect(tc.tax, tc.ctx.instances, sim, m, verb)
    return out


def _v7_form(tc: TaskCtx, frame: FrameData, goal: str,
             demonstrated: Sequence[tuple[str, str]], started: Sequence[str], removal: int,
             rule: Optional[dict]) -> Optional[dict]:
    """One V7 record (free when ``rule`` is ``None``), under the active moment's rules."""
    step = frame.step
    state = tc.state(step)
    plan = rules_plan(tc, state, goal)
    if not plan:
        tc.skip("V7", "no_plan")
        return None
    keep = frozenset({"connector"}) if rule and rule["kind"] == "until_connectors" \
        else frozenset()
    gold = with_captive(tc, state, grouped_plan(tc, state, plan, keep=keep))
    bad_at, _, _ = _simulate(tc, step, gold)
    if bad_at is not None:
        tc.skip("V7", "gold_plan_fails")
        return None
    # the goal is outlined only where it can be seen (owner 2026-11-04); a CPU under its
    # cooler is named in words, and the question then shows every camera
    namer = Namer(tc, frame)
    marked = goal in frame.pointable and namer.part(goal) is not None
    goal_words = namer.say(goal) if marked else asked_words(tc, goal)[0]
    required = required_actions(tc, state, goal)
    form = "free" if rule is None else "constrained"
    rec_id = f"V7-{frame_id(tc.desktop, tc.view, step)}-{goal}" + ("" if rule is None
                                                                  else "-constrained")
    if rule is None:
        index, template = pick(V7_QUESTIONS, rec_id)
        question = template.format(goal=goal_words)
    else:
        index, template = pick(V7_CONSTRAINED_QUESTIONS, rec_id)
        question = template.format(goal=goal_words, rule=rule["rule"],
                                   each=EACH_CONNECTOR if rule["kind"] == "until_connectors"
                                   else "")
    chain: list[dict] = [observed(goal, state[goal].state, tc, frame.pointable)]
    for req in required[1:1 + tc.budget]:
        edge = req.via[-1]
        current = blocker_state(state, req.target)
        chain.append(R.recall_relation(edge.type, edge.target, edge.blocker,
                                       edge.necessity, edge.mode))
        chain.append(observed(req.target, current, tc, frame.pointable))
        chain.append(R.check_precondition(req.target, sorted(req.states), current, False))
    pixel = any(r.pixel_only for r in required)
    text = v7_text(tc, state, goal, required, gold, rule)
    groups = plan_groups(tc, gold)
    check = {"type": "goal_plan", "form": form, "goal": goal, "goal_state": REMOVED,
             "start_step": step, "goal_removed_at": removal,
             "metric": "violation_free+required_recall+goal_reached",
             "any_legal_plan_scores": True, "validity": "simulated",
             "redundancy_penalty": False, "necessity": "required",
             "required_actions": required_json(tc, state, required),
             "reference_shortest": _actions_seq(plan),
             "demonstrated_remainder": _actions_seq(demonstrated),
             "plan_groups": groups,
             "plan_length": len(plan), "gold_length": len(gold),
             "demonstrated_length": len(demonstrated),
             "started": sorted(set(started)),
             "start_state": {k: v.state for k, v in sorted(state.items())},
             # the moment's own rules (occlusion, loose parts first), for the scorer's replay
             "moment": moment_json(tc),
             "previous_action": previous_action(tc, step)}
    if rule is not None:
        check["constraint"] = constraint_json(rule)
        if rule["kind"] == "until_connectors":
            # how many plugs the answer has to unplug: the count the scorer compares
            check["count"] = {"class": "connector", "verb": "disconnect",
                              "present": int(rule["count"]), "members": list(rule["until"])}
    return record(
        tc, "V7", rec_id, step, tc.moment(step, marked=marked), question, index,
        {"plan": _actions_seq(gold), "goal": goal}, check,
        evidence({r.target: frame.pointable[r.target][1] for r in required
                  if r.target in frame.pointable}, tc, plan_length=len(gold)),
        dict(R.rationale(chain), text=text), frame.verified,
        truth_source=GRAPH_BLOCKED_PIXELS if pixel else GRAPH_BLOCKED,
        view_confirmed=frame.confirmed,
        marks=namer.marks() if marked else None,
        mark_names=namer.names() if marked else None, terms=[V7_GOAL_CLASS],
        extra={"v7_form": form},
        moment=tc.legend((step, "now"), marked=marked),
    )


def v7_text(tc: TaskCtx, state: FrameState, goal: str, required: Sequence[Requirement],
            gold: Sequence[tuple[str, str]], rule: Optional[dict]) -> str:
    """V7's reasoning chain: what still holds the goal (and what holds those), the rule if
    any, then the plan step by step -- in what the picture shows."""
    pending = {g.gid: g for g in pending_captive(tc, state)}
    held_by_captive = {tc.ctx.instances[g.members[0]].fastens: g for g in pending.values()}
    via = [r.via[-1] for r in required
           if r.via and not is_captive_screw(tc.ctx, r.target)
           and r.via[-1].source != CONSTRAINT_SOURCE]
    clauses = reason_clauses(tc, via, state, met=False, asked=(goal, [goal]))
    for held, group in sorted(held_by_captive.items(), key=lambda kv: str(kv[0])):
        if any(r.target == held for r in required):
            clauses.append(f"the {group.label} still hold {asked_words(tc, held)[0]}")
    goal_words = asked_words(tc, goal)[0]
    parts = [loose_sentence(tc, tc._active.step if tc._active is not None else 0).strip()]
    parts.append(f"To take {goal_words} out: {_and(clauses)}." if clauses
                 else f"To take {goal_words} out: nothing else holds it.")
    if rule is not None:
        parts.append(f"The rule says: {rule['rule']}")
        if rule["kind"] == "until_connectors":
            n = int(rule["count"])
            parts.append(f"{n} connectors are still plugged in, so the plan unplugs each of "
                         f"them before {goal_words} comes out.")
    steps = plan_words(tc, state, gold)
    parts.append("So the plan: " + "; ".join(f"{i + 1}. {w}" for i, w in enumerate(steps))
                 + ".")
    return " ".join(p for p in parts if p)


def required_json(tc: TaskCtx, state: FrameState,
                  required: Sequence[Requirement]) -> list[dict]:
    """V7's required state changes, captive screws as **one** group item (spec 6.6).

    :func:`required_actions` stops requiring a cooler screw once the log loosened it --
    a fact no picture shows, and a list that shrinks screw by screw would say on the label
    side which of them are loose. While the part they hold is in, the group's members are
    one requirement, every member, whatever the log says: target the group id, ``members``
    all of them, states ``loosened`` / ``removed``.
    """
    pending = {g.gid: g for g in pending_captive(tc, state)}
    out: list[dict] = []
    placed: set[str] = set()
    for req in required:
        group = tc.group_of(req.target)
        if group is not None and group.gid in pending:
            continue   # folded into the group item below
        out.append(dict(req.as_json(), group=_gid_of(tc, req.target)))
    by_target = {r.target: r for r in required}
    for gid, group in sorted(pending.items()):
        held = tc.ctx.instances[group.members[0]].fastens
        if held not in by_target or gid in placed:
            continue   # the plan never has to move the part they hold
        placed.add(gid)
        edges = [e for e in tc.by_target.get(held, ()) if e.blocker in group.members]
        via = tuple(by_target[held].via) + tuple(edges[:1])
        out.append({"target": gid, "states": ["loosened", REMOVED],
                    "via": [e.label() for e in via],
                    "pixel_only": any(is_pixel_edge(e) for e in via),
                    "group": gid, "members": list(group.members)})
    return out


def _gid_of(tc: TaskCtx, key: str) -> Optional[str]:
    group = tc.group_of(key)
    return None if group is None else group.gid


def _actions_seq(pairs: Sequence[tuple[str, str]]) -> list[dict]:
    """``(verb, target)`` pairs as JSON, **in order** (``_actions_json`` sorts)."""
    return [{"verb": verb, "target": target} for verb, target in pairs]


#: ``task id -> generator``. The order is the order records are written.
GENERATORS: dict[str, Callable[[TaskCtx, int], Iterator[dict]]] = {
    "V1": gen_v1, "V2": gen_v2, "V3": gen_v3, "V4": gen_v4, "V5": gen_v5,
    "V6": gen_v6, "V7": gen_v7, "V8": gen_v8, "V10": gen_v10, "V12": gen_v12,
    "V15": gen_v15, "V16": gen_v16,
}
