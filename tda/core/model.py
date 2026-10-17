"""Shared dataclasses and enums for the Teardown Annotator (TDA).

Everything in ``tda.core`` and ``tda.ui`` exchanges data through these types.
Coordinates are always in the *original image* frame of the view; masks are
COCO RLE dicts ``{"size": [h, w], "counts": str}``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

#: The four views of the teardown **sequence**: one frame per logical step.
VIEWS = ("scan", "oak1", "oak2", "rs")

# --------------------------------------------------------------------------- #
# the components gallery (task U10, spec 3.2b 零件合照)
# --------------------------------------------------------------------------- #
#: After each teardown the operators laid every removed part out on the table
#: and photographed the layout several times (re-arranged between shots) with
#: both OAK cameras and the RealSense. Those photos are **not** steps of the
#: teardown: each camera's shots are a view of their own, a shot ``k = 1..N`` is
#: an ordinary frame ``(desktop, k, view)``, and keyframes, the tools, the
#: compiler, the truth table, Review and the COCO export work on it unchanged.
COMPONENT_VIEWS = ("cmp_oak1", "cmp_oak2", "cmp_rs")
#: Every view a frame row can carry: the sequence first, then the gallery.
ALL_VIEWS = VIEWS + COMPONENT_VIEWS
#: The camera each gallery view was shot with.
COMPONENT_CAMERA = {"cmp_oak1": "oak1", "cmp_oak2": "oak2", "cmp_rs": "rs"}
#: Classes that are not drawn in a layout photo: a plug, a cable and a screw
#: cannot be told apart from the next one lying beside it.
GALLERY_EXCLUDED_CLASSES = frozenset({"connector", "cable", "screw"})


def is_component_view(view: Optional[str]) -> bool:
    """Is ``view`` one of the components gallery's (:data:`COMPONENT_VIEWS`)?"""
    return str(view or "") in COMPONENT_VIEWS


class Visibility(str, Enum):
    VISIBLE = "visible"
    OCCLUDED_PARTIAL = "occluded_partial"
    OCCLUDED_FULL = "occluded_full"
    OUT_OF_VIEW = "out_of_view"
    TOO_SMALL = "too_small"
    VISIBLE_TINY = "visible_tiny"
    MOTION_BLUR = "motion_blur"


class Placement(str, Enum):
    IN_CHASSIS = "in_chassis"
    ON_BENCH = "on_bench"
    ELSEWHERE = "elsewhere"


class StepType(str, Enum):
    INITIAL = "initial"
    NORMAL = "normal"
    DUPLI = "dupli"
    COMPOUND = "compound"
    FAILED = "failed"
    AUXILIARY = "auxiliary"
    REORIENT = "reorient"
    IGNORE = "ignore"


# --------------------------------------------------------------------------- #
# Label Studio drafts
# --------------------------------------------------------------------------- #
#: Prefix of the provisional keys :mod:`tda.core.ls_import` writes
#: (``ls:Motherboard#1``).
LS_PREFIX = "ls:"


def is_provisional(key: str) -> bool:
    """Is this a Label Studio draft key rather than an instance of the machine?

    The importer wrote 414 of these, each with a *real* taxonomy class and the
    keyframes the team traced before this tool existed. They are draft material
    an annotator resolves onto a real instance in S1 (spec 3.2), and until then
    they are not parts of anything: they carry no actions, so the state machine
    would have them installed and in the chassis for ever, and every one of them
    would be a missing chassis shape on every frame of its desktop.

    So a draft never needs geometry, is never compiled, never reaches a task
    card or a queue, and is never exported. Two choke points enforce that --
    :func:`tda.core.states.needs_geom` for geometry and
    :meth:`tda.core.export.coco.DesktopCtx.cls_of` for the exports -- and both
    ask this one question. The draft *keyframes* stay in the database untouched:
    they are what a later "adopt draft" tool will read.
    """
    return str(key).startswith(LS_PREFIX)


# --------------------------------------------------------------------------- #
# parts the picture shows but the log never names (task U5b, spec 3.2)
# --------------------------------------------------------------------------- #
#: ``attrs`` key naming the annotator who added an instance by hand in S1
#: (:mod:`tda.core.extra`). Its presence is what makes a row an *extra*: the
#: log never named it, so it has no actions and no raw names.
EXTRA_ATTR = "added_by"


def is_extra(rec: Optional["InstanceRec"]) -> bool:
    """Did an annotator add this instance in S1 rather than the log name it?

    Here rather than in :mod:`tda.core.extra` for the reason
    :func:`is_provisional` is: the graph rules, the relational heuristic and the
    log importer all have to ask it, and none of them may import the module
    that writes extras. An extra is an ordinary part of the machine -- it needs
    geometry, it is compiled, it is exported -- but nothing *about* it may be
    guessed: no heuristic fills a relation on it, no template edge starts at it
    and no re-import drops it, because the sheet never knew it existed.

    A 框外零件 (:func:`is_context`, task U18) carries ``added_by`` too, and is
    **not** an extra: it is no part of the machine at all.
    """
    return (rec is not None and bool((rec.attrs or {}).get(EXTRA_ATTR))
            and not is_context(rec))


# --------------------------------------------------------------------------- #
# 框外零件: objects outside the machine (task U18, spec 3.2c)
# --------------------------------------------------------------------------- #
#: Every key of a context object starts so: ``ctx.<class>.NN``. No taxonomy
#: class is called ``ctx``, so a machine part's key never does.
CONTEXT_PREFIX = "ctx."
#: ``attrs`` flag of a context object.
CONTEXT_ATTR = "context"
#: ``attrs`` key of its free-text name (required for :data:`CONTEXT_OTHER`).
CONTEXT_LABEL = "label"
#: The class of a context object no taxonomy class describes.
CONTEXT_OTHER = "other"
#: The placement chain -- and the compiler's placement group -- of its shapes:
#: they are layered among themselves only, never against the machine's parts.
CONTEXT_PLACEMENT = "context"


def is_context_key(key: Optional[str]) -> bool:
    """Is ``key`` the key of a context object (:data:`CONTEXT_PREFIX`)?"""
    return str(key or "").startswith(CONTEXT_PREFIX)


def is_context(rec: Optional["InstanceRec"]) -> bool:
    """Is this instance a 框外零件 -- something in the picture, not in the machine?

    Something the frames show outside the machine nobody listed (another part
    lying beside it, an unrelated object), drawn on its own (task U18). It has
    no state, no action, no relation; it is never implied, never in the log,
    never on a task card, never a reason ``Space`` refuses, and the machine's
    parts neither cover it nor are covered by it. Everything that walks the
    machine's parts asks this (or :func:`machine_only`); the key alone answers
    where only a key is at hand (:func:`is_context_key`).
    """
    if rec is None:
        return False
    return bool((rec.attrs or {}).get(CONTEXT_ATTR)) or is_context_key(rec.key)


def machine_only(instances: dict) -> dict:
    """``instances`` without the context objects: the machine's own table."""
    return {key: rec for key, rec in instances.items()
            if not is_context_key(key) and not is_context(rec)}


# --------------------------------------------------------------------------- #
# what a step type means (spec 6.6)
# --------------------------------------------------------------------------- #
#: Step types that describe no annotatable moment of the teardown.
#:
#: An ``ignore`` step -- a calibration shot, a re-take -- gets no task card, is
#: never compiled into the truth table, never carries a review status of its own
#: and is never exported. The timeline still shows its frame, in the neutral
#: colour of a step nobody has annotated, because the frame exists and the state
#: machine and the shape anchors still run through it (spec 4.2, 缺帧处理).
SKIP_STEP_TYPES = frozenset({StepType.IGNORE.value})

#: Step types that are never the "after" frame of a change question (spec 8.2
#: V3): nothing happened before an ``initial`` frame, a ``dupli`` step is a
#: repeat of one already asked about, and an ``ignore`` step is not a moment.
NO_CHANGE_STEP_TYPES = frozenset(
    {StepType.IGNORE.value, StepType.INITIAL.value, StepType.DUPLI.value}
)


def step_is_annotatable(step_type: Optional[str]) -> bool:
    """Is there a moment of the teardown here to compile, confirm and export?

    The single answer both halves of the tool read: the truth table gates on it
    (:func:`tda.core.truth_inputs.annotatable_steps`) and so do the exports. They
    used to disagree -- an ``ignore`` step was compiled, stamped and
    confirmable, and then dropped on the way out -- which is annotator work with
    nowhere to go. A step the table has no row for counts as ``normal``.
    """
    return str(step_type or StepType.NORMAL.value) not in SKIP_STEP_TYPES


@dataclass(frozen=True, order=True)
class FrameKey:
    """Identifies one image: desktop id, 1-based logical step, view name."""

    desktop: int
    step: int
    view: str


@dataclass
class InstanceRec:
    key: str  # e.g. "screw.cpu_cooler.03"
    desktop: int
    cls: str  # taxonomy class
    attrs: dict = field(default_factory=dict)  # role, head, head_source, kind, captive, of, ...
    parent: Optional[str] = None  # leaves the chassis together with this instance when attached
    attached: bool = False
    mounted_on: Optional[str] = None  # physical support
    fastens: Optional[str] = None  # screws only: the part this screw fastens
    socket_host: Optional[str] = None  # connectors only
    cable: Optional[str] = None  # connectors only: virtual cable node id
    slot_id: Optional[str] = None
    group_id: Optional[str] = None
    group_order: str = "unordered"  # unordered | sequential | opposite_pairs
    removal_direction: Optional[str] = None
    raw_names: list[str] = field(default_factory=list)


@dataclass
class StepRec:
    desktop: int
    step: int
    step_type: str
    raw_name: str
    dupli: bool = False
    notes: str = ""
    duration_s: Optional[float] = None


@dataclass
class ActionRec:
    desktop: int
    step: int
    idx: int  # order within the step
    target: str  # instance key or virtual node id ("cable:psu_harness")
    verb: str  # unscrew|disconnect|open|release|remove|displace|reorient
    tool: str = "none"
    direction: str = "none"
    result: str = "success"  # success|failed
    failure_reason: Optional[str] = None
    difficulty: Optional[int] = None


@dataclass
class StateEvent:
    desktop: int
    step: int
    target: str
    attr: str  # "state" | "placement"
    old: str
    new: str
    evidence_view: Optional[str] = None
    auto: bool = True


@dataclass
class ShapePart:
    name: str  # "main" or a named part, e.g. "floor", "wall"
    rle: Optional[dict] = None  # COCO RLE in reference-frame coordinates
    box: Optional[tuple[float, float, float, float]] = None  # x0, y0, x1, y1


@dataclass
class ShapeKeyframe:
    id: Optional[int]
    instance: str
    desktop: int
    view: str
    pose_segment: int
    anchor_step: int  # latest logical step this shape applies to
    placement: str = Placement.IN_CHASSIS.value
    geom_type: str = "mask"  # mask | box
    parts: list[ShapePart] = field(default_factory=list)
    amodal_complete: bool = True
    # manual | sam | model:<name>@<ver> | labelstudio | ls_adopted | carried | cut
    # ``labelstudio`` is the importer's own untouched draft row; ``ls_adopted``
    # is an annotator's shape built from one (``draft_id`` names which).
    # ``cut`` is the pre-edit shape copied to the far side of a state boundary
    # (tda.core.cut_follow), until somebody redraws its pixels.
    source: str = "manual"
    draft_id: Optional[int] = None
    version: int = 1
    edit_count: int = 0
    edit_time_ms: int = 0


@dataclass
class ZOrderRec:
    desktop: int
    view: str
    pose_segment: int
    order: list[tuple[str, str]]  # (instance_key, part_name), bottom -> top
    version: int = 1


@dataclass(frozen=True)
class PairOverride:
    desktop: int
    view: str
    pose_segment: int
    above: str
    below: str


@dataclass
class OccluderMask:
    frame: FrameKey
    occluder_type: str  # hand | arm | body | tool | cable | other
    rle: dict


#: What a frame's ``visibility`` label may name as the reason a part in its
#: place is not seen (task U15, 「被线挡住」): something nobody labels. The
#: annotator's ``I`` writes ``cable``; ``other_unlabeled`` is for anything else
#: that is not annotated (a later dialog), so far store-level only.
OCCLUDER_CABLE = "cable"
OCCLUDER_OTHER = "other_unlabeled"
OCCLUDERS = (OCCLUDER_CABLE, OCCLUDER_OTHER)


@dataclass
class FrameOverride:
    frame: FrameKey
    instance: str
    visible_rle: Optional[dict] = None
    visibility: Optional[str] = None
    #: Why the part is (partly) not seen although it is in its place: one of
    #: :data:`OCCLUDERS`, or ``None`` for a bare label (task U15).
    occluder: Optional[str] = None


@dataclass
class Similarity:
    """2-D similarity transform: x' = s * R(theta) * x + t."""

    scale: float = 1.0
    theta: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def is_identity(self, eps: float = 1e-9) -> bool:
        return (
            abs(self.scale - 1.0) < eps
            and abs(self.theta) < eps
            and abs(self.tx) < eps
            and abs(self.ty) < eps
        )
