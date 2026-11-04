"""One generator per VLM task of spec 8.2, P0 set (task B6).

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
import zlib
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
    Edge,
    applicable_preconditions,
    find_deadlocks,
    gated_verbs,
    legal_actions,
    remaining_plan,
    unmet,
)
from tda.core.graph_rules import (
    CABLE_PREFIX,
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
    "record",
    "required_actions",
    "row_verified",
    "unresolved_action_targets",
]

#: Every task this module can generate, in the order records are written per
#: frame. ``--tasks V5`` still reaches V5 through this tuple.
ALL_TASKS = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V10", "V12", "V14",
             "V15", "V16")

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
PERCEPTION_TASKS_SINGLE = ("V1", "V2", "V8", "V14", "V15")
#: The export unit whose records show every view of a moment.
MULTI_VIEW = "multi"

#: Tasks whose answer is read off the compiled truth table of one view.
PERCEPTION_TASKS = frozenset({"V1", "V2", "V8", "V12", "V14", "V15"})
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
    "V14": "L4", "V15": "L4",
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
    "components",    # V8 on a class with several instances: V1's grade on that class
})

#: Where a V4 label's certainty comes from (the governing principle, above).
#: ``graph_blocked_pixels`` is a graph "no" that rests **only** on edges
#: imported from confirmed-frame pixels (W3a, ``source = 'pixels:W*'``): the
#: paper writes those as "class rule x overlap in the annotation", not as an
#: observed constraint, so a sample has to say which kind it is.
TRUTH_SOURCES = ("demonstrated", "graph_blocked", "state_inapplicable",
                 "failed_attempt", "graph_blocked_pixels")
(DEMONSTRATED, GRAPH_BLOCKED, STATE_INAPPLICABLE, FAILED_ATTEMPT,
 GRAPH_BLOCKED_PIXELS) = TRUTH_SOURCES
#: The two kinds of graph-certain "no".
GRAPH_KINDS = frozenset({GRAPH_BLOCKED, GRAPH_BLOCKED_PIXELS})

#: Tasks whose records never carry a mark: the box is the answer (V1, V8) or would say the
#: part is visible (V14) -- spec 3.2.
UNMARKED_TASKS = frozenset({"V1", "V8", "V14"})
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

    def legend(self, *moments: tuple[int, str]) -> list[tuple[str, str, int]]:
        """``(view, moment, step)`` per image of :meth:`moment` calls, in image order."""
        out: list[tuple[str, str, int]] = []
        for step, name in moments:
            for frame in self.moment(step):
                out.append((self.view_of(frame), name, step))
        return out

    def view_of(self, frame: FrameData) -> str:
        if self.frames.get(frame.step) is frame:
            return self.own_view
        for view, frames in self.side_frames.items():
            if frames.get(frame.step) is frame:
                return view
        return self.own_view

    def moment(self, step: int) -> list[FrameData]:
        """The frames of one step to show together: this unit's own frame first, then the side views' frames
        in VIEWS order, each only where that camera has a picture of the step."""
        own = self.frames.get(step)
        out = [own] if own is not None else []
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
        """``target -> its active edges``, built once per desktop.

        :func:`~tda.core.graph.applicable_preconditions` scans the whole edge
        list, which is nothing for one call and 76 million comparisons for a
        66-desktop export: every frame asks for the permitted set and the
        blocked set, and every candidate action asks for its own preconditions.
        """
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
            hit = legal_actions(self.ctx.instances, self.edges, self.state(step),
                                self.tax, strict=strict)
            self._legal[(step, strict)] = hit
        return hit

    def blocked(self, step: int) -> list[tuple[str, str, list[Edge]]]:
        """``(verb, target, unmet edges)`` -- what the graph *certainly* forbids."""
        hit = self._blocked.get(step)
        if hit is None:
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
    """``graph_blocked_pixels`` when every edge saying "no" is pixel-imported.

    One ordinary edge among them is enough for the plain ``graph_blocked``: the
    "no" is then certain without the pixels.
    """
    edges = list(edges)
    if edges and all(is_pixel_edge(e) for e in edges):
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
    for key, current in sorted(cable_nodes(active_edges(tc.edges), state).items()):
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
           terms: Optional[Sequence[str]] = None,
           state_terms: Optional[Sequence[str]] = None,
           moment: Optional[Sequence[tuple[str, str, int]]] = None) -> dict:
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
    if truth_source is not None and truth_source not in TRUTH_SOURCES:
        raise ValueError(f"{rec_id}: truth_source {truth_source!r} is not one of "
                         f"{', '.join(TRUTH_SOURCES)}")
    if marks and task in UNMARKED_TASKS:
        # the box would be the answer (V1, V8) or say the part is visible (V14)
        raise ValueError(f"{rec_id}: {task} never carries marks")
    prompt: dict[str, Any] = {
        "task": task,
        "images": [f.image_id for f in frames],
        # "the PSU cannot be judged ..." starts a sentence
        "question": question[:1].upper() + question[1:],
    }
    if options is not None:
        prompt["options"] = [{"verb": o["verb"], "target_label": o["target_label"]}
                             for o in options]
    if marks:
        prompt["marks"] = [dict(m) for m in marks]
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
    yield from _v2_counts(tc, frame)


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
        check = {"type": "exact", "derive": "instance_state", "instance": ident,
                 "class": cls, "fields": ["state"],
                 "allowed": (list(PRESENCE_STATES) if screw
                             else sorted(tc.tax.states_of(cls)))}
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
            R.rationale([R.observe(k, value, tc.view, frame.pointable[k][1],
                                   frame.pointable[k][0].get("visibility"),
                                   occluder=frame.pointable[k][0].get("occluder"))
                         for k in members]),
            row_verified(rows, frame.verified),
            extra={"class_prior_answer": prior}, marks=namer.marks(),
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
             "fields": ["count"]},
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
    # two parts moving as one (spec 3.6: the shroud on its fan, captive screws): the
    # target or any part attached to it is the right answer
    attached = sorted(k for k, r in tc.ctx.instances.items()
                      if r.attached and r.parent == target and tc.askable(k))
    group = tc.group_of(target)

    rec_id = f"V3-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V3_QUESTIONS, rec_id)
    chain = [R.compare_frames([step - 1, step], tc.view),
             observed(target, answer["verb"], tc, source, step=source_step)]
    yield record(
        tc, "V3", rec_id, step, [*tc.moment(step - 1), *tc.moment(step)], question, index,
        answer,
        {"type": "exact", "derive": "action_slots", "fields": fields,
         "visible_only": True, "accept_targets": [target, *attached],
         "target_group": group.gid if group is not None else None},
        evidence(bboxes, tc, from_step=source_step), R.rationale(chain),
        # the grade follows the frame the answer was actually read off, which
        # for a V3 pair is usually the earlier one: the part is gone from the later
        row_verified([source[target][0]] if target in source else [],
                     (before if source_step == step - 1 else frame).verified),
        view_confirmed=confirmed, extra={"big_change": big},
        moment=tc.legend((step - 1, "before"), (step, "after")),
    )


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
    # one field, one meaning: why this answer is what it is. An edge said no, or
    # nothing did and the operator went on to do it.
    reason = None if feasible else (reason or "violates_edge")
    raw_blockers = sorted({e.blocker for e in bad})
    bboxes = {k: frame.pointable[k][1]
              for k in [*members, *raw_blockers] if k in frame.pointable}
    check = {"type": "feasibility", "verb": verb, "target": ident,
             "target_class": tc.ctx.cls_of(target),
             "match_class": match_class(tc.ctx, target),
             "state_after": frame.step, "truth_source": truth_source,
             "metric": "per_verb_accuracy_macro", "secondary_metric": "accuracy",
             "edges": [{"type": e.type, "target": e.target, "blocker": e.blocker,
                        "necessity": e.necessity, "mode": e.mode,
                        "required": sorted(required_states(e.type, e.mode))}
                       for e in sorted(bad, key=lambda e: (e.type, e.target, e.blocker))],
             # the blocker groups' members, for a grader that wants them
             "blocker_members": {b: sorted({e.blocker for e in bad
                                            if (tc.group_of(e.blocker).gid
                                                if tc.group_of(e.blocker) else e.blocker) == b})
                                 for b in blockers}}
    if ident != target:
        check["members"] = list(members)
    return record(
        tc, "V4", rec_id, frame.step, tc.moment(frame.step),
        template.format(phrase=verb_phrase(verb, label),
                        gerund=verb_gerund(verb, label)), index,
        {"feasible": feasible, "blockers": blockers, "reason": reason},
        check, evidence(bboxes, tc), R.rationale(chain),
        row_verified([frame.pointable[k][0] for k in bboxes], frame.verified),
        negative=None if feasible else truth_source,
        truth_source=truth_source, view_confirmed=frame.confirmed,
        marks=namer.marks(), terms=[target_class(tc.ctx, target)],
        moment=tc.legend((frame.step, "now")),
    )


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
    if frame is None or tc.excluded:
        return
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
V6_QUESTIONS = (
    "What is the next disassembly step? Answer with one verb and one target.",
    "Looking at the machine in these pictures, which action would you perform next?",
    "Name the single action to carry out next on this machine.",
)
#: The multiple-choice form (W3b): a secondary record beside the open one.
V6_MC_QUESTIONS = (
    "Which of these is the next disassembly step?",
    "Looking at the machine in these pictures, which of the following actions would you "
    "perform next?",
    "Pick the single action to carry out next on this machine.",
)
#: How many graph-permitted "unknown" distractors ride along, and how many
#: certainly-wrong ones the options list needs before it is offered at all.
V6_UNKNOWN_OPTIONS = 2
V6_WRONG_OPTIONS = 2


def gen_v6(tc: TaskCtx, step: int) -> Iterator[dict]:
    """The demonstrated next action, graded on match and on blocked rate.

    "Legal rate" was the wrong metric: the permitted set is an upper bound, so
    an answer inside it has only failed to be *provably* wrong. What can be
    measured exactly is the opposite -- how often an answer is one the graph
    certainly forbids -- and that is what ``blocked_rate`` is.

    The options, when they are offered, are one demonstrated action, at least
    two certainly-blocked ones and a couple of merely-permitted distractors. The
    prompt does not say which is which; the label does, so an evaluator can
    score the blocked ones as wrong and, if it chooses, ignore the unknowns.
    """
    frame = tc.frame(step)
    if frame is None or tc.excluded:
        return
    shown = tc.demonstrated(step)
    if shown is None:
        return
    nxt, action = shown
    permitted = set(tc.permitted(step))
    if (action.verb, action.target) not in permitted:
        return  # an action the graph says was impossible is nobody's reference

    seed = tc.seed(step, "V6")
    blocked = [(v, t) for v, t, _ in tc.blocked(step)]
    # group-equivalent matching (W3b): 60.7 % of steps have a second legal
    # action with the demonstrated verb on an interchangeable part -- four
    # unordered cooler screws -- and an answer naming .03 when the operator
    # took .04 is right. The equivalent set is the demonstrated action plus
    # every *permitted* sibling: same unordered group, or same verb and
    # match class (:func:`match_class`)
    equivalent = v6_siblings(tc, action.verb, action.target, permitted)
    # the wrong options match the demonstrated verb wherever they can: "pick the
    # option whose verb is never blocked" scored 45 % against 20 % random while
    # the only certainly-wrong options were graph-blocked ones, and no edge
    # gates a connector or a latch
    every_wrong = tc.matched(prefer_graph(tc.certainly_wrong(step)), action.verb, None,
                             seed + "|wrong")
    # ... and, inside the verb, a part of *another* match class first: the
    # certainly-wrong same-class ones are mostly the siblings already done
    # (screws 1 and 2 out, 3 demonstrated), and "pick the highest ordinal"
    # then answers without the picture
    mine = match_class(tc.ctx, action.target)
    # where another verb has to stand in, draw it in proportion to how often
    # this teardown uses it (a seeded weighted order): `remove` applies to
    # nearly everything, and filling every gap with it made "never pick remove"
    # a shortcut of its own
    verb_key = _verb_draw(tc, seed)
    every_wrong = sorted(every_wrong, key=lambda c: (
        c.verb != action.verb,
        verb_key(c.verb) if c.verb != action.verb else 0.0,
        match_class(tc.ctx, c.target) == mine))
    # the distractors match the demonstrated verb too, or "which option has the
    # verb that is usually right?" answers the question without the picture --
    # but never a sibling of the demonstrated action: that one is also right.
    # And **no two options are the same kind of action** (verb + match class):
    # with the siblings gone, the demonstrated one was otherwise the odd one
    # out among four motherboard screws, and "pick the option whose kind
    # appears once" scored 49 % against 20 % random (W3b probe, D1-D12)
    others = tc.matched([Candidate(v, t, "permitted_unknown")
                         for v, t in sorted(permitted - set(equivalent)
                                            - {(action.verb, action.target)})],
                        action.verb, None, seed + "|unknown")
    kinds_used = {(action.verb, mine)}

    def take(pool, limit, verb_rule) -> list[Candidate]:
        got: list[Candidate] = []
        for c in pool:
            if len(got) >= limit:
                break
            if not verb_rule(c):
                continue
            if mark_box(tc, frame, c.target) is None:
                continue  # an option names its part by a mark (spec 3.2)
            kind = (c.verb, match_class(tc.ctx, c.target))
            if kind in kinds_used:
                continue
            kinds_used.add(kind)
            got.append(c)
        return got

    wrong = take(every_wrong, V6_WRONG_OPTIONS, lambda c: True)
    unknown = take(others, V6_UNKNOWN_OPTIONS, lambda c: c.verb == action.verb)
    # a slot no same-verb permitted action can fill takes another certainly
    # wrong one of that verb first, a permitted one of another verb last
    room = V6_UNKNOWN_OPTIONS - len(unknown)
    extra = take(every_wrong, room, lambda c: c.verb == action.verb)
    wrong += extra
    unknown += take(sorted(others, key=lambda c: verb_key(c.verb)),
                    room - len(extra), lambda c: c.verb != action.verb)
    # a frame with only one certainly-wrong action left is no multiple choice --
    # so the question is asked open-ended instead of being dropped, and says so
    state = tc.state(step)
    blocked_set = set(blocked)
    equivalent_set = set(equivalent)
    # the demonstrated action as a **group action** where its group's remaining
    # members are all equivalent to it (spec 3.1: "unscrew the motherboard screws")
    shown_id, shown_members = group_action(
        tc, state, action.verb, action.target, lambda m: (action.verb, m) in equivalent_set)
    options: Optional[list[dict]] = None
    namer = Namer(tc, frame)
    if len(wrong) >= V6_WRONG_OPTIONS and mark_box(tc, frame, action.target) is not None:
        listed = [{"verb": action.verb, "target": shown_id, "kind": DEMONSTRATED,
                   "members": shown_members}]
        for c in wrong:
            ok = ((lambda m, c=c: (c.verb, m) in blocked_set) if c.kind in GRAPH_KINDS
                  else (lambda m: False))
            ident, members = group_action(tc, state, c.verb, c.target, ok)
            listed.append({"verb": c.verb, "target": ident, "kind": c.kind,
                           "members": members})
        for c in unknown:
            ident, members = group_action(
                tc, state, c.verb, c.target,
                lambda m, c=c: (c.verb, m) in permitted and (c.verb, m) not in equivalent_set)
            listed.append({"verb": c.verb, "target": ident, "kind": "permitted_unknown",
                           "members": members})
        options = stable_order(listed, seed + "|order",
                               lambda o: f"{o['verb']}|{o['target']}")
        for option in options:
            if option["target"] == option["members"][0]:
                namer.part(option["target"])
            else:
                namer.group(option["target"], option["members"])
        for option in options:
            option["target_label"] = namer.say(option["target"])
            if option["target"] == option["members"][0]:
                del option["members"]

    # two record types per frame sharing `v6_pair` (W3b ruling 2026-10-31):
    # the **open** question is V6's headline, graded group-equivalently; the
    # multiple choice is a secondary record, because a bigram over the
    # previous action alone picks its demonstrated option 79.9 % of the time
    stem = f"V6-{frame_id(tc.desktop, tc.view, step)}"
    chain = [observed(action.target,
                      state[action.target].state if action.target in state else None,
                      tc, frame.pointable, step=step)]
    answer = {"verb": action.verb, "target": shown_id,
              "target_class": target_class(tc.ctx, action.target),
              "tool": graded_tool(action.tool)}
    fields = ["verb", "target"] + (["tool"] if answer["tool"] is not None else [])
    # the reference is the answer: the group action where the answer is one (the
    # multiple choice lists the group, not its member, so the grader and the gold
    # reply must look for the group); the instance the log names stays in
    # `demonstrated`, and the group action is equivalent to itself
    grouped = shown_id != action.target
    check = {"type": "next_action", "reference": [action.verb, shown_id],
             "demonstrated": [action.verb, action.target],
             "reference_step": nxt, "fields": fields,
             "metric": "group_match_rate+blocked_rate",
             # primary: the answer is the demonstrated action or one of its
             # permitted siblings (same unordered group, or same verb + match
             # class); secondary: the reference exactly (a group reference: the
             # group or any member of it still to do)
             "match": {"primary": "group_equivalent", "secondary": "instance_exact"},
             "equivalent_actions": _actions_json(
                 [*equivalent, *([(action.verb, shown_id)] if grouped else [])]),
             "match_class": [action.verb, match_class(tc.ctx, action.target)],
             "group_id": unordered_group(tc.ctx, action.target),
             # the part group of the redesign (gid, or None) and its members still to do
             "part_group": shown_id if grouped else None,
             "part_group_members": shown_members,
             # the asked group's members, as every group record says them
             **({"members": list(shown_members)} if grouped else {}),
             # `blocked_rate` is the share of answers the graph certainly
             # forbids; on a frame where it forbids nothing the rate is
             # undefined, not zero, and the scorer leaves the frame out of the
             # blocked denominator
             "blocked_rate_defined": bool(blocked),
             "blocked_actions": _actions_json(blocked),
             "permitted_upper_bound": _actions_json(sorted(permitted)),
             # the protocol-prior reference reads this and nothing else (W3b)
             "previous_action": previous_action(tc, step)}
    evid = evidence({action.target: frame.pointable[action.target][1]}
                    if action.target in frame.pointable else {}, tc)
    pair = opaque_record_id(stem)
    forms: list[tuple[str, str, Optional[list[dict]]]] = [("open", stem, None)]
    if options is not None:
        forms.append(("mc", stem + "-mc", options))
    for form, rec_id, listed_options in forms:
        index, question = pick(V6_QUESTIONS if form == "open" else V6_MC_QUESTIONS,
                               rec_id)
        mc = form == "mc"
        yield record(
            tc, "V6", rec_id, step, tc.moment(step), question, index, answer,
            dict(check, options_kind="listed" if mc else "open", form=form),
            evid, R.rationale(chain), frame.verified, options=listed_options,
            extra={"v6_form": form, "v6_pair": pair}, view_confirmed=frame.confirmed,
            marks=namer.marks() if mc else None,
            terms=([target_class(tc.ctx, (o.get("members") or [o["target"]])[0])
                    for o in listed_options] if mc else None),
            moment=tc.legend((step, "now")),
        )


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
# V14 -- can this view decide at all
# --------------------------------------------------------------------------- #
V14_QUESTIONS = (
    "From this view alone, can you tell what state {label} is in? "
    "Answer with the state, or say you cannot tell.",
    "Can this image decide the state of {label}? If it can, give the state.",
    "Does this single view settle the state of {label}? Say so, or give it.",
)


V14_QUESTIONS_MANY = (
    "From this view alone, can you tell what state {label} are in? "
    "Answer with the state, or say you cannot tell.",
    "Can this image decide the state of {label}? If it can, give the state.",
    "Does this single view settle the state of {label}? Say so, or give it.",
)


def _v14_units(tc: TaskCtx, frame: FrameData, state: FrameState
               ) -> list[tuple[str, list[str], bool, Optional[str]]]:
    """``(id, members, answerable, state)`` per match class the frame can be asked about.

    No ordinal and no mark (a box would say the part is visible): a class is named in
    words, so it is asked as a whole -- every part of it still on the machine -- and only
    when every one of them gives the same answer (spec 6.3); else ``mixed_group``.
    """
    by_class: dict[str, list[str]] = {}
    for key in sorted(tc.ctx.instances):
        cls = tc.ctx.cls_of(key)
        if cls is None or not tc.askable(key) or not state_is_askable(tc.tax, cls):
            continue
        if is_captive_screw(tc.ctx, key) or not present(state, key):
            continue
        by_class.setdefault(match_class(tc.ctx, key), []).append(key)
    out = []
    for mc, members in sorted(by_class.items()):
        if not any(k in frame.rows for k in members):
            continue
        if not all(k in frame.rows for k in members):
            tc.skip("V14", "mixed_group")
            continue
        answers = set()
        for key in members:
            ok = frame.rows[key].get("visibility") in ANSWERABLE
            answers.add((ok, visible_state(tc.ctx, key, state[key].state) if ok else None))
        if len(answers) != 1:
            tc.skip("V14", "mixed_group")
            continue
        ok, shown = next(iter(answers))
        # one part: its key; one part group: its id; several groups of one class: the
        # class as a whole (`<match class>@*`)
        groups = {tc.group_of(k).gid if tc.group_of(k) else k for k in members}
        ident = (members[0] if len(members) == 1
                 else next(iter(groups)) if len(groups) == 1 else f"{mc}@*")
        out.append((ident, members, ok, shown))
    return out


def gen_v14(tc: TaskCtx, step: int) -> Iterator[dict]:
    """The abstain task: an instance this view cannot see has no state to report.

    Spec 8.2 principle 5. The truth is the compiled ``visibility`` and nothing
    else: ``occluded_full`` and ``out_of_view`` -- and ``too_small`` /
    ``motion_blur``, which are the same admission in other words -- mean the
    honest answer is "not from here".

    Both halves come off the same frame in equal numbers, drawn by the frame's
    own seed, so the majority-class baseline is 50 % by construction and a frame
    with nothing unanswerable contributes nothing rather than a run of easy
    yeses. The redesign asks a class as a whole (:func:`_v14_units`).
    """
    frame = tc.frame(step)
    if frame is None or not frame.verified:
        return
    state = tc.state(step)
    seed = tc.seed(step, "V14")
    # tightness is undecidable from any 2D view by construction (W3b): a captive
    # screw has no visible state at all, so it is no V14 item either way, and
    # a screw's state is asked as presence only
    units = _v14_units(tc, frame, state)
    blind = [u for u in units if not u[2]]
    seen = [u for u in units if u[2]]
    take = min(len(blind), len(seen), tc.budget)
    if not take:
        return
    chosen = (stable_order(blind, seed, lambda it: it[0])[:take]
              + stable_order(seen, seed + "|pos", lambda it: it[0])[:take])
    for ident, members, answerable, shown in chosen:
        rec_id = f"V14-{frame_id(tc.desktop, tc.view, step)}-{ident}"
        many = len(members) > 1
        index, template = pick(V14_QUESTIONS_MANY if many else V14_QUESTIONS, rec_id)
        rows = [frame.rows[k] for k in members]
        if answerable:
            bboxes = {k: frame.pointable[k][1] for k in members if k in frame.pointable}
            chain = R.rationale([R.observe(k, shown, tc.view, frame.pointable[k][1],
                                           frame.rows[k].get("visibility"),
                                           occluder=frame.rows[k].get("occluder"))
                                 for k in members if k in frame.pointable])
        else:
            chain = R.rationale(
                [R.propagate_state(k, None) for k in members],
                R.abstain("; ".join(unseen_reason(k, frame.rows[k], tc.view)
                                    for k in members)),
            )
            bboxes = {}
        cls = tc.ctx.cls_of(members[0])
        check = {"type": "exact", "derive": "answerable", "instance": ident,
                 "fields": ["answerable", "state"], "metric": "risk_coverage",
                 **({"state_space": "presence"} if cls == "screw" else {})}
        if many:
            check["members"] = list(members)
        yield record(
            tc, "V14", rec_id, step, [frame],
            template.format(label=class_words(tc, members)), index,
            {"answerable": answerable, "state": shown if answerable else None},
            check,
            evidence(bboxes, tc, {k: seen_as(frame.rows[k]) for k in members}),
            chain, row_verified(rows, frame.verified),
            negative=None if answerable else "unanswerable",
            terms=[cls], state_terms=[cls],
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
                 pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Member actions folded into **group actions** (spec 3.1, 6.1).

    One ``(verb, gid)`` at the place of the first member action, for a group with at
    least two members still to do in ``state``; the other members' actions go. The graph
    orders interchangeable parts "unordered" already.
    """
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for verb, key in pairs:
        group = tc.group_of(key)
        ident = key
        if group is not None and len(to_do(tc.ctx, state, verb, group.members)) >= 2:
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
    source = (DEMONSTRATED if first_bad is None
              else graph_kind(edges) if edges else STATE_INAPPLICABLE)
    check = {"type": "plan", "plan": listed, "metric": "accuracy+localisation",
             "truth_source": source, "groups": plan_groups(tc, plan)}
    if family is not None:
        check["corruption"] = family
        check["corruption_tag"] = tag
    shown = [m for _, t in plan for m in (_group_members(tc, t) or [t])]
    return record(
        tc, "V16", rec_id, step, tc.moment(step),
        question + "\n" + "\n".join(
            f"{i + 1}. {verb_phrase(v, words[i])}" for i, (v, _) in enumerate(plan)),
        index,
        {"valid": first_bad is None, "first_error_index": first_bad,
         "violated_edge": labels[0] if labels else None,
         "violated_edges": labels, "reason": reason},
        check,
        evidence({t: frame.pointable[t][1] for t in shown if t in frame.pointable},
                 tc, plan_length=len(listed)),
        R.rationale(chain), frame.verified,
        negative=None if first_bad is None else family,
        truth_source=source, view_confirmed=frame.confirmed, marks=namer.marks(),
        terms=[target_class(tc.ctx, (_group_members(tc, t) or [t])[0]) for _, t in plan],
        moment=tc.legend((step, "now")),
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
    if frame is None or tc.excluded:
        return
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
        return  # the logged suffix does not replay cleanly: say nothing

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
    true = _plan_record(tc, frame, plan, None, (), "true")
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
    covered = [e for e in tc.by_target.get(goal, ())
               if e.type == "covered_by"
               and tc.ctx.cls_of(e.blocker) == V7_COOLER_CLASS]
    if not covered:
        # W3a's S1 report: a desktop whose cooler set is unresolved has no
        # `covered_by(cpu <- cooler)` edge, and its plan lifts the CPU through
        # the cooler. Until W3a's rule lands those desktops ask no V7.
        return "cooler_unresolved"
    return None


def gen_v7(tc: TaskCtx, step: int) -> Iterator[dict]:
    """The remaining CPU plan, from a frame where it is **in progress**.

    From the first frame this is a generic recipe -- [open the shroud] ->
    unscrew the cooler x4 -> lift the cooler -> open the lever -> lift the CPU
    -- that a model writes without looking. Half-way it is not: two screws out
    and the shroud open is something only the picture says. So a frame is asked
    only when at least one action of the plan from the initial state has been
    done and the CPU is still in.

    The answer is graded on three things the incomplete graph makes certain
    (``answer_check``): (a) no step violates an existing required edge, (b)
    every :func:`required_actions` state change is in the plan, (c) the goal is
    reached. **No redundancy penalty**: a plan that also takes out the drive
    cage on its way may be exactly right and the graph would not know.
    """
    frame = tc.frame(step)
    if frame is None or tc.excluded:
        return
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
        plan = remaining_plan(tc.ctx.instances, tc.edges, here, goal, tc.tax)
        if not plan:
            tc.skip("V7", "no_plan")
            continue
        demonstrated = [(a.verb, a.target) for s in tc.log_steps
                        if step < s <= removal for a in tc.named_actions(s)
                        if (a.target in tc.ctx.instances
                            or a.target.startswith(CABLE_PREFIX))
                        and tc.changes_state(a)]
        bad_at, _, _ = _simulate(tc, step, demonstrated)
        if bad_at is not None:
            tc.skip("V7", "demonstrated_does_not_replay")
            continue
        made = _v7_record(tc, frame, goal, plan, demonstrated, started, removal)
        if made is not None:
            yield made


def _v7_record(tc: TaskCtx, frame: FrameData, goal: str,
               plan: Sequence[tuple[str, str]],
               demonstrated: Sequence[tuple[str, str]], started: Sequence[str],
               removal: int) -> Optional[dict]:
    step = frame.step
    state = tc.state(step)
    # the goal is outlined (spec 3.2): "the CPU outlined in red", under its cooler by
    # its amodal box; a CPU that cannot be outlined is not asked about
    namer = Namer(tc, frame)
    if namer.part(goal) is None:
        tc.skip("V7", "unmarkable_part")
        return None
    required = required_actions(tc, state, goal)
    rec_id = f"V7-{frame_id(tc.desktop, tc.view, step)}-{goal}"
    index, template = pick(V7_QUESTIONS, rec_id)
    chain: list[dict] = [observed(goal, state[goal].state, tc, frame.pointable)]
    for req in required[1:1 + tc.budget]:
        edge = req.via[-1]
        current = blocker_state(state, req.target)
        chain.append(R.recall_relation(edge.type, edge.target, edge.blocker,
                                       edge.necessity, edge.mode))
        chain.append(observed(req.target, current, tc, frame.pointable))
        chain.append(R.check_precondition(req.target, sorted(req.states), current,
                                          False))
    pixel = any(r.pixel_only for r in required)
    shortest = _actions_seq(plan)
    shown = _actions_seq(demonstrated)
    # the answer is the demonstrated remainder told in group actions (spec 3.1): one
    # "unscrew the CPU cooler screws" for the members still to do -- and, while the
    # cooler is in, whatever the log says about each of them (spec 6.6)
    grouped = with_captive(tc, state, grouped_plan(tc, state, demonstrated))
    groups = plan_groups(tc, grouped)
    # the answer is the **demonstrated** remainder: the one plan known to work.
    # The graph's shortest plan is only an upper-bound construction (the graph
    # is incomplete), so it is a reference, not the answer; and the check
    # accepts any plan at all that breaks no edge, contains every required
    # action and reaches the goal (review M, 2026-10-31)
    return record(
        tc, "V7", rec_id, step, tc.moment(step), template.format(goal=namer.say(goal)),
        index,
        {"plan": _actions_seq(grouped), "goal": goal},
        {"type": "goal_plan", "goal": goal, "goal_state": REMOVED,
         "start_step": step, "goal_removed_at": removal,
         "metric": "violation_free+required_recall+goal_reached",
         "any_legal_plan_scores": True,
         "redundancy_penalty": False, "necessity": "required",
         "required_actions": required_json(tc, state, required),
         "reference_shortest": shortest, "demonstrated_remainder": shown,
         "plan_groups": groups,
         "plan_length": len(shortest), "demonstrated_length": len(demonstrated),
         "started": sorted(set(started)),
         "start_state": {k: v.state for k, v in sorted(state.items())},
         "previous_action": previous_action(tc, step)},
        evidence({r.target: frame.pointable[r.target][1] for r in required
                  if r.target in frame.pointable}, tc, plan_length=len(grouped)),
        R.rationale(chain), frame.verified,
        truth_source=GRAPH_BLOCKED_PIXELS if pixel else GRAPH_BLOCKED,
        view_confirmed=frame.confirmed, marks=namer.marks(), terms=[V7_GOAL_CLASS],
        moment=tc.legend((step, "now")),
    )


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
    "V14": gen_v14, "V15": gen_v15, "V16": gen_v16,
}
