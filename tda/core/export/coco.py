"""COCO instance-segmentation export of the compiled truth table (spec 8.1).

:func:`export_coco` turns the ``compiled_mask`` rows of one view into a COCO
file: one image per frame that has a compiled row, one annotation per row that
carries geometry, and the 23 taxonomy classes as categories 1..23 in
``Taxonomy.classes`` order, so category ids are stable across exports as long as
``configs/taxonomy.yaml`` keeps its order.

Beyond the COCO fields every annotation carries an ``attributes`` block (state,
placement, visibility, the ``occluder`` nobody labels that hides it -- or
``viewpoint``, nothing hides it and this camera cannot see it --, whether a
human audited that label in Review's 可见性审查 (``visibility_reviewed``, task
U24), occlusion ratio, amodal completeness, provenance, the view's ``tier`` and
whether it is ``verified``), because the spec's downstream users -- and the VLM export next
door -- need the truth table's semantics, not only its pixels. A part
被线挡住 (task U15) hidden completely is written with an empty segmentation,
its amodal box and ``iscrowd: 1`` (:func:`_annotation`).

Amodal shapes (task U26)
------------------------
The dataset's point is layered amodal masks, so **every mask annotation**
carries the part's whole shape on that frame beside its visible one:

* ``amodal_segmentation`` -- compressed RLE (``{"size", "counts"}``) in the
  same coordinates as ``segmentation``: the full frame, or the ROI under
  ``roi_crop`` (cropped like the visible mask; a shape with no pixel left in
  the ROI is an empty RLE of the ROI's size);
* ``amodal_bbox`` -- its ``[x, y, w, h]`` (``null`` when it is empty);
* ``amodal_area`` -- its pixel count.

The shape is the compiler's: the union of the parts of the keyframe that
applies, through the frame's registration -- what the visible mask was cut out
of, so the visible mask lies inside it and ``attributes.occlusion_ratio`` is
``1 - area / amodal_area``. **``visible ⊆ amodal`` holds in the file**: where a
row's visible mask reaches outside the compiled shape -- a 「只改这一帧」 mask
drawn away from it, or a frozen row that kept the mask it was confirmed with
after the shape moved (D2-D14 scan: 19 of 16 497) -- the annotation carries
``amodal ∪ visible`` instead, and ``info.amodal_union_rows`` counts them
(their ``occlusion_ratio`` stays the compiler's). An annotation with no shape
on the frame at all -- its pixels come from a 「只改这一帧」 mask alone -- is the
same rule with nothing to unite: ``amodal = visible``, counted too. All three
are ``null`` only for a row with neither a shape nor a visible pixel, which is
never an annotation. A ``box`` row (``include_boxes``) is not a mask
annotation and carries none of them. ``amodal=False`` (``export-coco
--no-amodal``) leaves out what U26 added, for size -- ``amodal_area``, and both
other fields on a row with no ``occluder`` -- and keeps what U15 / U24 already
wrote: a row 被线挡住 / 角度看不到 still carries ``amodal_segmentation`` and
``amodal_bbox``, so ``--no-amodal --no-hidden`` writes the pre-U26 annotations
byte for byte. ``info.amodal`` says which it is; ``info.version`` is "2".

A part the labelled parts above it cover completely -- ``occluded_full``,
nothing visible, no ``occluder`` -- is exported too, the way a part
被线挡住（完全） is: an empty ``segmentation``, ``area`` 0, ``bbox`` = its amodal
box, ``iscrowd: 1``, the three amodal fields, ``attributes.occluder`` null
(task U26; until then it was dropped, nothing of it being seen).
``hidden=False`` (``--no-hidden``) drops them again; ``info.hidden`` says
which it is.

``tier`` and ``verified`` are two fields on purpose. The tier is the **view's**
(spec 8.1: scanner and OAK1 gold, OAK2 silver, RealSense bronze) and is read
from ``configs/taxonomy.yaml``; ``verified`` says whether a human confirmed this
row. One field spelling both -- "gold" for confirmed, "auto" for not -- answered
the second question in the first question's vocabulary.

Coordinates
-----------
``roi_crop=False`` (the default) writes full-frame coordinates. ``roi_crop=True``
writes them relative to the frame's ROI (``pose_segment.roi_json``): masks are
cropped, boxes shifted and clipped, the image entry records ``roi`` and its
width/height are the ROI's. A row falling entirely outside the ROI is dropped.

框外零件 (task U18)
-------------------
A context object -- something in the picture outside the machine, drawn on its
own -- is exported with the rest of the frame: its class as the category (one
no taxonomy class describes goes under ``other``, id ``len(classes) + 1``,
listed only in a file that has one), ``attributes.context = true``, its
``label``, ``instance_key``, and ``state`` / ``placement`` ``null``. Under
``roi_crop`` it is kept **only when it intersects the ROI** (cropped to it like
any row); one lying wholly outside is dropped.

This module also owns the identity helpers shared with
:mod:`tda.core.export.vlm` -- image ids, cache-relative file names, frame size,
the per-desktop state context and the keyframe lookup -- so that both exports
name the same frame the same way.

The export **writes**: it brings the truth table up to date first
(:meth:`~tda.core.truth.TruthService.ensure_fresh`), so a caller needs the
single-user lock of spec 3.5. What it never writes is annotation *content* --
it only reads out what the database already implies -- and nothing here imports
Qt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from tda.core import masks
from tda.core.compiler import keyframe_amodal, select_keyframe
from tda.core.db import Db
from tda.core.implied import is_implied
from tda.core.model import (
    ALL_VIEWS,
    CONTEXT_LABEL,
    CONTEXT_OTHER,
    NO_CHANGE_STEP_TYPES,
    SKIP_STEP_TYPES,
    VIEWS,  # noqa: F401 - read off this module by tda.core.export.vlm
    ActionRec,
    FrameKey,
    InstanceRec,
    ShapeKeyframe,
    StateEvent,
    StepRec,
    StepType,
    Visibility,
    is_component_view,
    is_context,
    is_provisional,
    step_is_annotatable,
)
from tda.core.states import GALLERY_EXPORT_STATE, FrameState, absent_keys, state_at
from tda.core.taxonomy import Taxonomy
from tda.core.truth import TruthService
from tda.core.truth_fresh import selected_keyframes
from tda.core.truth_inputs import InputCache, events_of, gather, infer_hw, pose_segment_of
from tda.core.vis_review import stamp_covers

__all__ = [
    "ANSWERABLE",
    "NO_CHANGE_STEP_TYPES",
    "SKIP_STEP_TYPES",
    "DesktopCtx",
    "bbox_xywh",
    "cache_rel_path",
    "categories",
    "category_ids",
    "conflicted_steps",
    "export_coco",
    "frame_file_name",
    "frame_hw",
    "frame_is_verified",
    "image_id",
    "load_ctx",
    "mask_bbox_xywh",
    "roi_of",
    "row_bbox_xywh",
    "view_index",
    "view_tier",
]

#: ``desktop * 100000 + step * 10 + view_index`` keeps ids unique and readable.
DESKTOP_STRIDE = 100_000
STEP_STRIDE = 10

#: The visibility labels whose row can be pointed at in this view.
ANSWERABLE = (
    Visibility.VISIBLE.value,
    Visibility.OCCLUDED_PARTIAL.value,
    Visibility.VISIBLE_TINY.value,
)

VERIFIED = "verified"
GEOM_BOX = "box"
DEFAULT_EXT = ".png"
#: ``info.version``. "2" (task U26): every mask annotation carries its amodal
#: shape, and the parts the labelled parts cover completely are annotations
#: of their own -- the annotation set changed, so the ids after them shift.
EXPORT_VERSION = "2"
#: ``info.split`` of an export: the teardown, or the components gallery (U10).
SPLIT_SEQUENCE = "sequence"
SPLIT_COMPONENTS = "components"


def burst_manifest(cache_dir: Optional[str], desktop: int, view: str) -> dict:
    """``{shot: record}`` of the cache's burst choices for one burst view (task U42).

    ``<cache>/<view>/D<nn>/manifest.json``, as :func:`tda.core.cache.build_cache`
    wrote it; ``{}`` for a view without a burst, without a cache or without one.
    """
    from tda.core.cache import BURST_VIEWS, MANIFEST_NAME, _load_manifest

    if not cache_dir or view not in BURST_VIEWS:
        return {}
    root = str(cache_dir).replace("\\", "/").rstrip("/")
    return _load_manifest(f"{root}/{view}/D{int(desktop):02d}/{MANIFEST_NAME}")


def _shot_source(frame: dict, key: FrameKey, manifest: Optional[dict] = None) -> dict:
    """What a components-gallery image says about the photo it is (task U10).

    The shot number among its camera's shots, the source file, and -- for a
    sheet row of the laid-out parts moved out of the sequence -- the step it
    was filed under.

    ``source`` is the picture the shapes were drawn on. For the scanner's
    gallery (``cmp_scan``, task U42) a shot is a burst and the cache chose one
    of it (a dark ``P_0`` is replaced, ``p0_dark``): the cache manifest's
    ``src`` is that picture, and ``burst_choice`` its reason (``p0``,
    ``p0_dark`` ...). Without a manifest record -- the OAK / RealSense shots,
    no cache, or a record from another burst than the frame's (re-pointed and
    not cached again) -- it is the frame's own path and there is no
    ``burst_choice``.
    """
    frame = frame or {}
    out = {"split": SPLIT_COMPONENTS, "shot": int(key.step), "source": frame.get("path")}
    record = (manifest or {}).get(str(key.step)) or {}
    src = record.get("src")
    burst = {str(p).replace("\\", "/") for p in (frame.get("aux") or {}).get("burst") or []}
    if src and str(src).replace("\\", "/") in burst:
        out["source"] = src
        if record.get("reason"):
            out["burst_choice"] = record["reason"]
    sheet_step = (frame.get("aux") or {}).get("sheet_step")
    if sheet_step is not None:
        out["sheet_step"] = int(sheet_step)
    return out

# What a step type means is :mod:`tda.core.model`'s table, re-exported here
# because :mod:`tda.core.export.vlm` reads it off this module. The truth table
# gates on the same table, so a step nobody may export is a step nobody was
# asked to annotate in the first place.


# --------------------------------------------------------------------------- #
# frame identity (shared with the VLM export)
# --------------------------------------------------------------------------- #
def view_index(view: str) -> int:
    """Position of ``view`` in :data:`tda.core.model.ALL_VIEWS`.

    scan=0 .. rs=3 for the sequence, then the components gallery
    (cmp_scan=4, cmp_oak1=5, cmp_oak2=6, cmp_rs=7): still below
    :data:`STEP_STRIDE`, so a gallery shot's image id never collides with a
    sequence frame's (task U10). Task U42 put the scanner's gallery first and
    so moved the three others up by one; no gallery image id had been
    released, so :data:`EXPORT_VERSION` stays.
    """
    try:
        return ALL_VIEWS.index(view)
    except ValueError:
        raise ValueError(f"unknown view: {view!r}") from None


def image_id(desktop: int, step: int, view: str) -> int:
    """Stable COCO image id of one frame."""
    return int(desktop) * DESKTOP_STRIDE + int(step) * STEP_STRIDE + view_index(view)


def cache_rel_path(desktop: int, step: int, view: str, ext: str = DEFAULT_EXT) -> str:
    """The frame's cached image path relative to ``cache_dir``.

    ``<view>/D<nn>/s<kkk>.<ext>`` -- the layout the image cache writes and the
    one every export references, so a COCO/JSONL file stays valid wherever the
    cache is mounted.
    """
    if not ext.startswith("."):
        ext = "." + ext
    return f"{view}/D{int(desktop):02d}/s{int(step):03d}{ext}"


def frame_file_name(frame: Optional[dict], key: FrameKey) -> str:
    """Cache-relative name of a frame, keeping the source file's extension."""
    source = (frame or {}).get("path") or ""
    ext = Path(source).suffix or DEFAULT_EXT
    return cache_rel_path(key.desktop, key.step, key.view, ext)


def frame_hw(frame: Optional[dict], compiled: dict[str, dict], view: str) -> tuple[int, int]:
    """``(H, W)`` of a frame: recorded size, else the masks', else the view's native one.

    The compiled RLEs are produced at the frame's own size, so they are the
    authority whenever the frame row does not carry an explicit ``aux["hw"]``.
    """
    recorded = ((frame or {}).get("aux") or {}).get("hw")
    if isinstance(recorded, (list, tuple)) and len(recorded) == 2:
        return (int(recorded[0]), int(recorded[1]))
    for row in compiled.values():
        size = (row.get("visible_rle") or {}).get("size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return (int(size[0]), int(size[1]))
    return infer_hw(view)


def bbox_xywh(box: Optional[Sequence[float]]) -> Optional[list]:
    """``(x0, y0, x1, y1)`` (exclusive) -> COCO ``[x, y, w, h]``."""
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return [x0, y0, x1 - x0, y1 - y0]


def mask_bbox_xywh(rle: Optional[dict]) -> Optional[list]:
    """COCO ``[x, y, w, h]`` of an RLE, or ``None`` when it is empty/absent.

    Measured off the run lengths rather than by decoding: an export writes one
    box and one area per annotation, and building two ``H x W`` arrays to read
    four numbers out of them is the single most expensive thing it used to do.
    """
    if not rle:
        return None
    box = masks.rle_bbox_xywh(rle)
    return None if box[2] <= 0 or box[3] <= 0 else [int(round(v)) for v in box]


def row_bbox_xywh(row: dict) -> Optional[list]:
    """COCO ``[x, y, w, h]`` of a compiled row, whichever geometry it carries.

    A ``mask`` row is measured off its visible RLE, a ``box`` row (a part lying
    on the bench, spec 3.4) reads its stored rectangle. ``None`` means the row
    has no geometry at all -- a missing shape, or one put out of view.
    """
    box = bbox_xywh(row.get("box")) if row.get("geom_type") == GEOM_BOX else None
    return box if box is not None else mask_bbox_xywh(row.get("visible_rle"))


# --------------------------------------------------------------------------- #
# per-desktop context (shared with the VLM export)
# --------------------------------------------------------------------------- #
@dataclass
class DesktopCtx:
    """Everything both exports need about one desktop, loaded once.

    ``events`` come from :func:`tda.core.truth_inputs.events_of`, so an export
    reads exactly the log the truth service compiled the frames from: always
    derived from the recorded actions, with the hand-written (``auto=False``)
    events merged on top.

    **``instances`` holds only real instances.** A Label Studio draft
    (:func:`tda.core.model.is_provisional`) carries a real taxonomy class, so a
    loop that asks "every screw of this role" finds it and counts it -- which is
    how ``ls:Screw#7`` came to answer a V2 counting question and to appear in
    its rationale, on a frame the compiler never put it in. Filtering here
    rather than at each call site is the point: the drafts are not in the
    mapping, so a loop cannot walk past the rule. What was dropped is kept in
    :attr:`drafts` for a caller that has to say so.

    The 框外零件 (task U18) are taken out the same way, into :attr:`context`:
    no VLM question may count, name or reason about something that is not part
    of the machine, and the COCO export writes them from there, on their own.

    So are the parts this machine does not have (「这台没有这个零件」, task
    U34, :func:`tda.core.states.absent_keys`), into :attr:`absent`, and the
    sheet's actions on them with them: D34's failed "remove the optical
    drive" is no history of anything in the pictures.
    """

    desktop: int
    tax: Taxonomy
    instances: dict[str, InstanceRec]
    events: list[StateEvent]
    actions: list[ActionRec]
    steps: dict[int, StepRec]
    keyframes: dict[tuple[str, str, int], list[ShapeKeyframe]] = field(default_factory=dict)
    #: The provisional keys :meth:`__post_init__` took out of ``instances``.
    drafts: dict[str, InstanceRec] = field(default_factory=dict, repr=False)
    #: The context objects :meth:`__post_init__` took out of ``instances`` (U18).
    context: dict[str, InstanceRec] = field(default_factory=dict, repr=False)
    #: The parts the machine does not have, taken out of ``instances`` (U34).
    absent: dict[str, InstanceRec] = field(default_factory=dict, repr=False)
    _states: dict[int, FrameState] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        drafts = {k: v for k, v in self.instances.items() if is_provisional(k)}
        if drafts:
            self.drafts = drafts
            self.instances = {k: v for k, v in self.instances.items()
                              if k not in drafts}
        context = {k: v for k, v in self.instances.items() if is_context(v)}
        if context:
            self.context = context
            self.instances = {k: v for k, v in self.instances.items()
                              if k not in context}
        gone = absent_keys(self.events)
        absent = {k: v for k, v in self.instances.items() if k in gone}
        if absent:
            self.absent = absent
            self.instances = {k: v for k, v in self.instances.items() if k not in absent}
            self.actions = [a for a in self.actions if a.target not in absent]

    def context_cls(self, instance: str) -> Optional[str]:
        """The class a context object is exported under (a taxonomy class or ``other``)."""
        rec = self.context.get(instance)
        if rec is None:
            return None
        if rec.cls == CONTEXT_OTHER or rec.cls in self.tax.classes:
            return rec.cls
        return None

    def state_at(self, step: int) -> FrameState:
        """State of every instance at ``step`` (memoised); drafts are not in it."""
        if step not in self._states:
            self._states[step] = state_at(self.instances, self.events, step, self.tax)
        return self._states[step]

    def cls_of(self, instance: str) -> Optional[str]:
        """Taxonomy class of an exportable instance key, else ``None``.

        ``None`` covers three things, and all three are skipped rather than
        allowed to crash or leak into a release: an instance that is not in the
        table at all, one carrying a class the taxonomy does not know, and a
        Label Studio draft -- which is not in ``instances`` to begin with, so
        this answers ``None`` for it without a rule of its own.
        """
        rec = self.instances.get(instance)
        if rec is None or rec.cls not in self.tax.classes:
            return None
        return rec.cls

    def keyframe_at(self, instance: str, step: int, placement: str,
                    pose_segment: int) -> Optional[ShapeKeyframe]:
        """The keyframe that applies to one instance at one step, if any.

        Narrowed to the frame's placement chain *and* pose segment first, the
        way :func:`tda.core.compiler.compile_frame` does it: shapes of two pose
        segments are drawn in different reference frames and are not comparable.
        """
        chain = self.keyframes.get((instance, placement, pose_segment), [])
        return select_keyframe(chain, step)

    def step_type(self, step: int) -> str:
        """Recorded step type, or ``"normal"`` when the step table has no row."""
        rec = self.steps.get(step)
        return StepType.NORMAL.value if rec is None else rec.step_type

    def exportable(self, step: int) -> bool:
        """Is this step a moment worth exporting at all? (``ignore`` is not.)"""
        return step_is_annotatable(self.step_type(step))

    def actions_at(self, step: int, successful_only: bool = True) -> list[ActionRec]:
        """Actions recorded at ``step``, in ``idx`` order."""
        rows = [a for a in self.actions if a.step == step]
        if successful_only:
            rows = [a for a in rows if a.result == "success"]
        return sorted(rows, key=lambda a: a.idx)


def load_ctx(db: Db, tax: Taxonomy, desktop: int, view: str) -> DesktopCtx:
    """Read instances, steps, actions, events and keyframes of one desktop/view."""
    chains: dict[tuple[str, str, int], list[ShapeKeyframe]] = {}
    for kf in db.keyframes(desktop, view):
        chains.setdefault((kf.instance, kf.placement, kf.pose_segment), []).append(kf)
    return DesktopCtx(
        desktop=desktop, tax=tax, instances=db.instances(desktop),
        events=events_of(db, tax, desktop), actions=db.actions(desktop),
        steps={s.step: s for s in db.steps(desktop)}, keyframes=chains,
    )


# --------------------------------------------------------------------------- #
# categories
# --------------------------------------------------------------------------- #
def categories(tax: Taxonomy) -> list[dict]:
    """The 23 taxonomy classes as COCO categories, ids 1..23 in config order."""
    return [
        {"id": i, "name": name, "supercategory": defn.get("group") or "other"}
        for i, (name, defn) in enumerate(tax.classes.items(), start=1)
    ]


def category_ids(tax: Taxonomy) -> dict[str, int]:
    """``class name -> category id``."""
    return {c["name"]: c["id"] for c in categories(tax)}


#: ``supercategory`` of the ``other`` category of the 框外零件 (task U18).
CONTEXT_SUPERCATEGORY = "context"


def other_category(tax: Taxonomy) -> dict:
    """The category of a context object no taxonomy class describes (task U18).

    One past the taxonomy's last id, so it never moves a class's id; it is
    listed only in a file that has such an annotation.
    """
    return {"id": len(tax.classes) + 1, "name": CONTEXT_OTHER,
            "supercategory": CONTEXT_SUPERCATEGORY}


# --------------------------------------------------------------------------- #
# ROI
# --------------------------------------------------------------------------- #
def roi_of(db: Db, key: FrameKey) -> Optional[tuple[int, int, int, int]]:
    """The frame's ROI as ``(x0, y0, x1, y1)``, or ``None`` for the full frame.

    Accepts the two shapes the ROI is stored in: ``[x0, y0, x1, y1]`` and a
    mapping with either ``x0/y0/x1/y1`` or ``x/y/w/h``.
    """
    raw = db.pose_segment_for(key).get("roi")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        x0, y0, x1, y1 = (int(round(float(v))) for v in raw)
    elif isinstance(raw, dict):
        if {"x0", "y0", "x1", "y1"} <= set(raw):
            x0, y0 = int(raw["x0"]), int(raw["y0"])
            x1, y1 = int(raw["x1"]), int(raw["y1"])
        elif {"x", "y", "w", "h"} <= set(raw):
            x0, y0 = int(raw["x"]), int(raw["y"])
            x1, y1 = x0 + int(raw["w"]), y0 + int(raw["h"])
        else:
            return None
    else:
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _crop_rle(rle: dict, roi: tuple[int, int, int, int]) -> Optional[dict]:
    """Re-encode an RLE in ROI coordinates; ``None`` when nothing is left."""
    cropped = masks.crop(masks.decode_rle(rle), roi)
    if not cropped.any():
        return None
    return masks.encode_rle(cropped)


def _clip_box(box: list, roi: tuple[int, int, int, int]) -> Optional[list]:
    """Shift a ``[x, y, w, h]`` box into ROI coordinates and clip it; ``None`` if empty."""
    x0, y0, x1, y1 = roi
    bx0, by0 = max(float(box[0]), float(x0)), max(float(box[1]), float(y0))
    bx1 = min(float(box[0]) + float(box[2]), float(x1))
    by1 = min(float(box[1]) + float(box[3]), float(y1))
    if bx1 <= bx0 or by1 <= by0:
        return None
    return [bx0 - x0, by0 - y0, bx1 - bx0, by1 - by0]


# --------------------------------------------------------------------------- #
# annotations
# --------------------------------------------------------------------------- #
def frame_is_verified(db: Db, key: FrameKey, rows: dict[str, dict]) -> bool:
    """Has a human signed this whole *frame* off? (spec 3.4)

    The frame-level notion :meth:`tda.core.truth.TruthService._frame_is_verified`
    works with, narrowed the safe way: the frame's own ``review_status``, or
    **every** compiled row of it frozen. The truth service can settle for *any*
    frozen row because it only needs to know whether a demotion is due; a
    tier is not a grade of the review, and one confirmed screw must not make the
    whole image count as confirmed.
    """
    frame = db.get_frame(key)
    if frame is not None and frame.get("review_status") == VERIFIED:
        return True
    return bool(rows) and all(row.get("status") == VERIFIED for row in rows.values())


def conflicted_steps(service: TruthService, desktop: int, view: str,
                     allow_conflicts: bool) -> set[int]:
    """Steps of one view with an open disagreement; refuses unless allowed.

    A *standing* conflict is a frozen row a human was told is disputed and has
    not settled: the compiler may not overwrite it, so every export of that view
    republishes the disputed value, marked as confirmed, for as long as the
    queue entry sits there. Both exports therefore stop and say so.

    ``allow_conflicts=True`` is the deliberate "publish it anyway" -- a
    rehearsal export, a mid-annotation snapshot -- and then the frames involved
    go out with ``verified: false``, because that is what they are.
    """
    rows = service.open_conflicts(desktop, view)
    if rows and not allow_conflicts:
        steps = sorted({int(r["step"]) for r in rows})
        raise RuntimeError(
            f"desktop {desktop} view {view}: {len(rows)} open conflict(s) on step(s) "
            f"{steps[:5]}{'...' if len(steps) > 5 else ''} are still unsettled; "
            f"resolve them in the review panel, or pass allow_conflicts=True to "
            f"export those frames as unverified"
        )
    return {int(r["step"]) for r in rows}


def view_tier(view: str, tax: Taxonomy) -> str:
    """The annotation tier of one view (spec 8.1); raises for an unknown one.

    Read from ``configs/taxonomy.yaml``'s ``view_tiers`` and never written down
    in code: which camera is annotated to which standard is a decision about the
    dataset, and it has already been changed once.

    A view the configuration does not describe is a *configuration* problem, and
    a loud one: returning ``None`` instead put ``"tier": null`` on every record
    of that whole export and said nothing, which is the shape of a mistake
    nobody notices until the data is somewhere else.
    """
    try:
        return tax.view_tiers[str(view)]
    except KeyError:
        raise KeyError(
            f"no annotation tier for view {view!r}: add it to the view_tiers "
            f"table in configs/taxonomy.yaml (known views: "
            f"{', '.join(sorted(tax.view_tiers)) or 'none'})"
        ) from None


def _attributes(ctx: DesktopCtx, instance: str, row: dict, step: int,
                keyframe: Optional[ShapeKeyframe], tier: Optional[str],
                disputed: bool = False, gallery: bool = False,
                occluder: Optional[str] = None, reviewed: bool = False) -> dict:
    """The truth-table semantics carried alongside every annotation.

    ``tier`` and ``verified`` are deliberately two fields. The tier is the
    *view's* (spec 8.1: scanner and OAK1 gold, OAK2 silver, RealSense bronze)
    and says how carefully these frames are annotated at all; ``verified`` says
    whether a human confirmed this particular row. The single ``quality`` field
    that used to hold ``gold``/``auto`` answered the second question with the
    first question's vocabulary.

    ``disputed`` says the frame carries an open conflict and is being exported
    anyway (``allow_conflicts``). No row of such a frame is confirmed, whatever
    its own ``status`` column still says: the image entry already said so, and
    a consumer filtering on the annotations rather than on the images would
    otherwise have taken the disputed rows as signed off.

    ``implied`` is provenance: the instance was created by
    :mod:`tda.core.implied` because the desktop plainly has one and its log
    never touched it, not because a step named it.

    ``gallery`` is a components-gallery shot (task U10): every part on it is
    ``removed`` and ``on_bench`` (:data:`~tda.core.states.GALLERY_EXPORT_STATE`).

    ``occluder`` is why a part in its place is (partly) not seen on this frame
    -- ``"cable"`` / ``"other_unlabeled"``, nothing annotated covers it, or
    ``"viewpoint"`` (task U24, 「角度看不到」), nothing covers it and this camera
    cannot see it from where it stands -- or ``None`` (task U15, 「被线挡住」):
    the frame's own statement, read off its
    :class:`~tda.core.model.FrameOverride`.

    ``visibility_reviewed`` (task U24) says a human audited this row's label in
    Review's 「可见性审查」 -- stamped for the very label and reason it has
    (:func:`tda.core.vis_review.stamp_covers`); ``false`` for a row nobody
    audited, a plain ``visible`` one included.
    """
    inst_state = ctx.state_at(step).get(instance)
    rec = ctx.instances.get(instance)
    state = None if inst_state is None else inst_state.state
    placement = row.get("placement")
    if gallery:
        # a part laid out for the components gallery (task U10): taken out of
        # the machine and lying on the table, whatever chain it is drawn in
        state, placement = GALLERY_EXPORT_STATE
    return {
        "instance_key": instance,
        "state": state,
        "placement": placement,
        "visibility": row.get("visibility"),
        "occluder": occluder,
        "visibility_reviewed": bool(reviewed),
        "occlusion_ratio": row.get("occlusion_ratio"),
        "amodal_complete": None if keyframe is None else bool(keyframe.amodal_complete),
        "implied": bool(rec is not None and is_implied(rec)),
        "tier": tier,
        "verified": not disputed and row.get("status") == VERIFIED,
    }


def _context_attributes(ctx: DesktopCtx, instance: str, row: dict,
                        keyframe: Optional[ShapeKeyframe], tier: Optional[str],
                        disputed: bool = False, occluder: Optional[str] = None) -> dict:
    """The ``attributes`` of a 框外零件 (task U18): no state, no placement.

    ``context: true`` says what it is, ``label`` its free-text name (``None``
    when it has none); everything else as for any row, so a consumer filtering
    on ``verified`` or ``visibility`` treats it alike.
    """
    rec = ctx.context.get(instance)
    label = str(((rec.attrs if rec else None) or {}).get(CONTEXT_LABEL) or "").strip()
    return {
        "instance_key": instance,
        "state": None,
        "placement": None,
        "visibility": row.get("visibility"),
        "occluder": occluder,
        # the visibility review (task U24) audits the machine's parts only
        "visibility_reviewed": False,
        "occlusion_ratio": row.get("occlusion_ratio"),
        "amodal_complete": None if keyframe is None else bool(keyframe.amodal_complete),
        "implied": False,
        "tier": tier,
        "verified": not disputed and row.get("status") == VERIFIED,
        "context": True,
        "label": label or None,
    }


def _annotation(ann_id: int, img_id: int, category: int, row: dict, attributes: dict,
                roi: Optional[tuple[int, int, int, int]], include_boxes: bool,
                amodal: Optional[dict] = None, *,
                with_amodal: bool = True, with_hidden: bool = True) -> Optional[dict]:
    """One COCO annotation, or ``None`` when the row carries nothing to export.

    A ``mask`` row becomes a full segmentation; a ``box`` row -- a bench part
    whose truth is a rectangle -- only becomes an annotation under
    ``include_boxes``, with ``segmentation: []``. Both geometries come from the
    compiled row itself, so nothing here re-derives what the truth service has
    already resolved.

    ``amodal`` is the part's whole shape on this frame, full-frame RLE, or
    ``None`` when it has none (the caller has already made it hold the visible
    mask, :func:`_with_visible`). A mask annotation carries it as
    ``amodal_segmentation`` / ``amodal_bbox`` / ``amodal_area`` (task U26, spec
    8.1's extension field; all three ``null`` without one), cropped to ``roi``
    like the visible mask. ``with_amodal=False`` (``--no-amodal``) writes only
    what U15 / U24 wrote before U26: ``amodal_segmentation`` and
    ``amodal_bbox`` on a row with an ``occluder``, nothing on any other.

    A part hidden completely -- ``occluded_full`` with an empty visible mask,
    被线挡住 / 角度看不到 (tasks U15, U24) or covered by the labelled parts
    above it (task U26) -- is exported anyway: an empty ``segmentation``,
    ``area`` 0, the amodal box as ``bbox`` and **``iscrowd: 1``**; its
    segmentation is the compressed RLE (``{"size", "counts"}``) that
    pycocotools reads for a crowd region. What COCOeval makes of it depends on
    the ``iouType``. **bbox** evaluation ignores a crowd region: a detection
    on the hidden part matches its (amodal) box and is neither a hit nor a
    false positive, and a visible-mask detector that finds nothing there loses
    nothing (a perfect one scored 1.000 against this, 0.972 against
    ``iscrowd: 0``). **segm** evaluation does not ignore it: the crowd
    region's segmentation is empty, so its IoU with any detection is 0 and a
    detection on the hidden part is a false positive there (U26 review, N2).
    ``with_hidden=False`` (``--no-hidden``) drops the covered ones -- those
    with no ``occluder`` -- as before U26. A part with nothing visible and no
    shape is always dropped.
    """
    known = amodal is not None
    if amodal is not None and roi is not None:
        amodal = _crop_rle(amodal, roi)
    hidden = (amodal is not None
              and attributes.get("visibility") == Visibility.OCCLUDED_FULL.value
              and (bool(attributes.get("occluder")) or with_hidden))
    crowd = 0
    rle = row.get("visible_rle")
    if rle:
        if roi is not None:
            cropped = _crop_rle(rle, roi)
            if cropped is None and hidden:
                cropped = masks.encode_rle(np.zeros((roi[3] - roi[1], roi[2] - roi[0]),
                                                    dtype=bool))
            rle = cropped
            if rle is None:
                return None
        box = mask_bbox_xywh(rle)
        if box is None and hidden:
            box = mask_bbox_xywh(amodal)       # nothing seen: where the part is
            # ... and a crowd region (ruled, U15 review M5): bbox evaluation
            # ignores detections there, segm evaluation does not (U26 N2)
            crowd = 1
        if box is None:
            return None
        area = float(masks.rle_area(rle))
        segmentation: Any = rle
    else:
        if not include_boxes:
            return None
        box = bbox_xywh(row.get("box"))
        if box is None:
            return None
        if roi is not None:
            box = _clip_box(box, roi)
            if box is None:
                return None
        area = float(box[2]) * float(box[3])
        segmentation = []
    ann = {
        "id": ann_id, "image_id": img_id, "category_id": category,
        "segmentation": segmentation, "bbox": box, "area": _round_int(area),
        "iscrowd": crowd, "attributes": attributes,
    }
    if rle and with_amodal:
        ann.update(_amodal_fields(amodal, known, segmentation))
    elif rle and attributes.get("occluder") and amodal is not None:
        # --no-amodal governs only what U26 added: a row with a reason keeps
        # the two fields U15 / U24 wrote, exactly as they wrote them (U26 N1)
        ann["amodal_segmentation"] = amodal
        ann["amodal_bbox"] = mask_bbox_xywh(amodal)
    return ann


def _amodal_fields(amodal: Optional[dict], known: bool, segmentation: dict) -> dict:
    """``amodal_segmentation`` / ``amodal_bbox`` / ``amodal_area`` of one mask annotation.

    ``amodal`` is already in the annotation's coordinates (cropped under
    ``roi_crop``); ``known`` says whether the part had a shape at all. A shape
    that had pixels and lost them all to the crop is an empty RLE of the
    segmentation's size -- known to be empty here -- and an unknown one is
    three ``null``. The export never hands in an unknown one with visible
    pixels (:func:`_with_visible` makes it the visible mask), so the ``null``
    triple is for a row with neither, which is not an annotation.
    """
    if not known:
        return {"amodal_segmentation": None, "amodal_bbox": None, "amodal_area": None}
    if amodal is None:
        height, width = (int(v) for v in segmentation["size"])
        amodal = masks.encode_rle(np.zeros((height, width), dtype=bool))
    return {"amodal_segmentation": amodal, "amodal_bbox": mask_bbox_xywh(amodal),
            "amodal_area": int(masks.rle_area(amodal))}


def _occluders_of(db: Db, key: FrameKey) -> dict[str, str]:
    """``instance -> occluder`` of the frame's 「被线挡住」 statements (task U15)."""
    return {name: fo.occluder for name, fo in db.frame_overrides(key).items()
            if fo.occluder}


def _amodal_rles(service: TruthService, key: FrameKey, wanted,
                 cache: Optional[InputCache] = None,
                 shapes: Optional[dict] = None) -> dict[str, dict]:
    """The whole shape of each ``wanted`` instance on this frame, as full-frame RLE.

    The truth table stores the visible mask alone (spec 3.4). The whole shape
    of a part is its keyframe's parts on the frame and nothing else
    (:func:`tda.core.compiler.keyframe_amodal`), so it is derived from the
    keyframe the compiler would select here (:func:`selected_keyframes`, read
    off the frame's gathered inputs) -- not by compiling the frame, which is
    what U26 did, once per exported frame: 1.6 s of D3/scan's 1.7, 34 s of
    D2-D21's 36 (task U29, U26 N6). One keyframe usually shapes a part over
    dozens of frames, so the encoded shape is kept in ``shapes`` by keyframe
    (id, version), transform and frame size, for the export's whole run.
    Byte for byte what ``compile(...).instances[name].amodal`` encodes to: the
    same walk over the parts, the same window, the same encoder. An instance
    the frame does not shape -- not needed here, no keyframe, a box -- is left
    out, as before.
    """
    if not wanted:
        return {}
    inputs = gather(service.db, service.tax, key, cache, service.cache_dir)
    chosen = selected_keyframes(inputs)
    shapes = shapes if shapes is not None else {}
    transform = inputs.transform
    turn = (float(transform.scale), float(transform.theta), float(transform.tx),
            float(transform.ty))
    out: dict[str, dict] = {}
    for name in wanted:
        kf = chosen.get(name)
        if kf is None or kf.geom_type == GEOM_BOX:
            continue
        memo = (kf.id, kf.version, id(kf) if kf.id is None else None, turn, inputs.hw)
        rle = shapes.get(memo)
        if rle is None:
            amodal, window = keyframe_amodal(kf, transform, inputs.hw)
            rle = shapes[memo] = masks.encode_rle(amodal, window)
        out[name] = dict(rle)            # an annotation's own object, as before
    return out


def _with_visible(amodal: Optional[dict], visible: Optional[dict]
                  ) -> tuple[Optional[dict], bool]:
    """``(amodal ∪ visible, whether it grew)``: what is seen is part of the part.

    The compiled shape holds the visible mask everywhere but where a frame
    override drew pixels away from it, or a frozen row kept the mask it was
    confirmed with after the shape moved (task U26: 19 of 16 497 on D2-D14
    scan). The file promises ``visible ⊆ amodal``, so those rows get the
    union; the export counts them (``info.amodal_union_rows``). A row with
    visible pixels and **no** shape on the frame (a 「只改这一帧」 mask alone) is
    the same rule with nothing to unite: ``amodal = visible``, counted too
    (U26 N3a). No visible pixel leaves the shape as it is, ``None`` included.

    Off the run lengths, no decoding. Two sizes are two frames, which a row
    and its own compilation never are: ``ValueError`` rather than a file
    that silently breaks its promise (U26 N3b).
    """
    if not visible:
        return amodal, False
    seen = masks.rle_area(visible)
    if seen == 0:
        return amodal, False
    if amodal is None:
        return visible, True
    if [int(v) for v in visible["size"]] != [int(v) for v in amodal["size"]]:
        raise ValueError(f"a {list(visible['size'])} visible mask against a "
                         f"{list(amodal['size'])} amodal shape: two frames, not one")
    if masks.rle_overlap(visible, amodal) == seen:
        return amodal, False
    return masks.rle_union(amodal, visible), True


def _round_int(value: float):
    """Keep whole numbers integral so mask areas read as pixel counts."""
    return int(value) if float(value).is_integer() else float(value)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export_coco(
    db: Db,
    tax: Taxonomy,
    desktops: list[int],
    view: str,
    out_json: str,
    only_verified: bool = True,
    roi_crop: bool = False,
    *,
    include_boxes: bool = False,
    truth: Optional[TruthService] = None,
    allow_conflicts: bool = False,
    amodal: bool = True,
    hidden: bool = True,
    cache_dir: Optional[str] = None,
) -> dict:
    """Write the compiled truth of ``desktops`` in ``view`` as one COCO file.

    Parameters
    ----------
    desktops, view:
        What to export; frames without a compiled row are skipped entirely.
    out_json:
        Destination path; parent directories are created.
    only_verified:
        Keep only rows a human confirmed (``status == "verified"``). A frame
        whose rows are all dropped gets no image entry either.
    roi_crop:
        Write ROI-relative coordinates (see the module docstring). A 框外零件
        (task U18) is kept only when it intersects the ROI.
    include_boxes:
        Also emit the compiled ``box`` rows -- a part lying on the bench, whose
        truth is a rectangle rather than a mask -- as bbox-only annotations with
        ``segmentation: []``.
    allow_conflicts:
        Export a view that still has open conflicts. Without it the export
        **refuses** (:func:`conflicted_steps`); with it the frames involved are
        written with ``verified: false``, and ``info`` records how many there
        were.
    amodal:
        Give every mask annotation its whole shape -- ``amodal_segmentation``,
        ``amodal_bbox``, ``amodal_area`` (task U26, the module docstring). On
        by default; ``False`` (``--no-amodal``) leaves out what U26 added --
        40-50 % of the file (D3 scan: 2.51 MB against 1.68 MB) -- keeping the
        two fields U15 / U24 wrote on a row with an ``occluder``, and spares
        the one compilation per exported frame they cost on every export, the
        digests notwithstanding: the shapes are not stored (a frame with a
        part hidden completely, or with a reason, is still compiled -- a
        covered part's only unless ``hidden`` is off). ``info.amodal`` records
        it. Where a row's visible mask reaches outside the compiled shape (or
        there is no shape) the file carries their union, and
        ``info.amodal_union_rows`` counts those annotations.
    hidden:
        Export a part the labelled parts above it cover completely
        (``occluded_full``, nothing visible, no ``occluder``) as a crowd
        region with its amodal shape, like a part 被线挡住（完全） (task U26):
        what a layered amodal dataset is for. On by default; ``False``
        (``--no-hidden``) drops them as before. ``info.hidden`` records it.
    cache_dir:
        The image cache, read for the scanner gallery's burst choices (task
        U42, :func:`_shot_source`); ``None`` takes the truth service's.

    Steps typed ``ignore`` are skipped: they describe no moment of the teardown.

    A components-gallery view (``cmp_scan``/``cmp_oak1``/``cmp_oak2``/``cmp_rs``,
    tasks U10 and U42)
    is its own split, ``info.split == "components"``: one image per shot, every
    annotation ``removed`` / ``on_bench``, and each image's ``extra`` naming the
    source photo (``shot``, ``source`` -- the picture drawn on, the cache's
    burst choice for ``cmp_scan`` with ``burst_choice`` its reason --,
    ``sheet_step`` when it came from a sheet row). The step table's types do
    not apply to shots.

    Returns the COCO document that was written (``images``, ``annotations``,
    ``categories``, plus a deterministic ``info`` block: nothing in the file
    depends on the wall clock, so two exports of the same truth are identical).
    """
    gallery = is_component_view(view)
    doc: dict[str, Any] = {
        "info": {
            "description": "TDA rehearsal export (spec 8.1)",
            "version": EXPORT_VERSION,
            "view": view,
            # the teardown sequence, or the components gallery's layout photos
            # of the removed parts (task U10, spec 3.2b): two datasets
            "split": SPLIT_COMPONENTS if gallery else SPLIT_SEQUENCE,
            "desktops": [int(d) for d in desktops],
            "only_verified": bool(only_verified),
            "roi_crop": bool(roi_crop),
            "include_boxes": bool(include_boxes),
            "allow_conflicts": bool(allow_conflicts),
            "open_conflicts": 0,
            # every mask annotation carries its whole shape (task U26) ...
            "amodal": bool(amodal),
            # ... and ``visible ⊆ amodal`` holds: the rows that needed a union
            "amodal_union_rows": 0,
            # a part the labelled parts cover completely is a crowd region
            "hidden": bool(hidden),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories(tax),
    }
    cat_of = category_ids(tax)
    ann_id = 1
    service = truth or TruthService(db, tax)
    cache_dir = cache_dir if cache_dir is not None else getattr(service, "cache_dir", None)

    tier = view_tier(view, tax)  # the view's, so it is read once for the file
    #: the encoded whole shapes by keyframe, transform and size (U29)
    shapes: dict = {}
    for desktop in desktops:
        ctx = load_ctx(db, tax, desktop, view)
        # which picture of each scanner burst was drawn on (cmp_scan, U42)
        burst_choices = burst_manifest(cache_dir, desktop, view) if gallery else {}
        # the amodal shapes read each frame's inputs once more; the
        # step-independent reads are shared by the desktop's frames
        inputs = InputCache()
        # the compiled rows of an unverified frame are a cache the
        # annotator's commits leave stale (spec 3.4): fill it before
        # reading the view out, or a frame nobody visited is exported
        # as it was several edits ago -- or silently not at all
        service.ensure_fresh(desktop, view, only_verified)
        disputed = conflicted_steps(service, desktop, view, allow_conflicts)
        doc["info"]["open_conflicts"] += len(disputed)
        for frame in db.frames_for(desktop, view):
            key = FrameKey(desktop, frame["step"], view)
            # a gallery shot is not a step: the step table says nothing about it
            if not gallery and not ctx.exportable(key.step):
                continue
            # a step this view never photographed (frame.missing, task U38) has no
            # picture to export anything over, whatever rows a cache still holds
            if frame.get("missing"):
                continue
            rows = db.compiled(key)
            if only_verified:
                rows = {k: r for k, r in rows.items() if r.get("status") == VERIFIED}
            if not rows:
                continue
            roi = roi_of(db, key) if roi_crop else None
            hw = frame_hw(frame, rows, view)
            img_id = image_id(desktop, key.step, view)
            image = {
                "id": img_id,
                "file_name": frame_file_name(frame, key),
                "width": (roi[2] - roi[0]) if roi else hw[1],
                "height": (roi[3] - roi[1]) if roi else hw[0],
                # the frame-level answer to the same question the annotations
                # answer per row, so a consumer picking whole frames does not
                # have to read all of them; `review_status` is the raw column
                # underneath and stays for whoever already reads it
                "tier": tier,
                # a frame somebody is still arguing about is not a confirmed
                # one, whatever its rows say (spec 3.4)
                "verified": (key.step not in disputed
                             and frame_is_verified(db, key, db.compiled(key))),
                "extra": {"desktop": desktop, "step": key.step, "view": view,
                          "review_status": frame.get("review_status")},
            }
            if gallery:
                image["extra"].update(_shot_source(frame, key, burst_choices))
            if roi is not None:
                image["roi"] = list(roi)
            doc["images"].append(image)

            segment = pose_segment_of(db, key)
            # 「被线挡住」 (task U15): the reason, and the amodal it keeps
            occluders = _occluders_of(db, key)
            # every mask row's whole shape (task U26) -- or, with --no-amodal,
            # only the ones a hidden (完全) annotation takes its box from
            shaped = sorted(
                name for name, row in rows.items() if row.get("visible_rle")
                and (amodal or name in occluders
                     or (hidden and row.get("visibility") == Visibility.OCCLUDED_FULL.value)))
            amodals = _amodal_rles(service, key, shaped, inputs, shapes)
            # 「可见性审查」 (task U24): the rows stamped for the label they have
            flags = db.instance_flags_at(key)
            for instance in sorted(rows):
                row = rows[instance]
                keyframe = ctx.keyframe_at(instance, key.step, row.get("placement") or "",
                                           segment)
                if instance in ctx.context:
                    # a 框外零件 (task U18): its class -- `other` its own category --
                    # and no state or placement; under --roi-crop kept only where
                    # it reaches into the ROI (cropped like any row)
                    cls = ctx.context_cls(instance)
                    if cls is None:
                        continue
                    if cls == CONTEXT_OTHER and cls not in cat_of:
                        cat_of[cls] = other_category(tax)["id"]
                        doc["categories"].append(other_category(tax))
                    attributes = _context_attributes(
                        ctx, instance, row, keyframe, tier, key.step in disputed,
                        occluders.get(instance))
                else:
                    cls = ctx.cls_of(instance)
                    if cls is None:
                        continue  # an unresolved draft key: no category to export it under
                    attributes = _attributes(
                        ctx, instance, row, key.step, keyframe, tier, key.step in disputed,
                        gallery, occluders.get(instance),
                        stamp_covers(flags.get(instance), row.get("visibility"),
                                     occluders.get(instance)))
                shape, grew = amodals.get(instance), False
                if amodal or instance in occluders:
                    # wherever a shape is written, what is seen is part of the
                    # part: visible ⊆ amodal, a shapeless row's being its
                    # visible mask (U26, N3a)
                    shape, grew = _with_visible(shape, row.get("visible_rle"))
                ann = _annotation(ann_id, img_id, cat_of[cls], row, attributes,
                                  roi, include_boxes, shape,
                                  with_amodal=amodal, with_hidden=hidden)
                if ann is not None:
                    doc["annotations"].append(ann)
                    doc["info"]["amodal_union_rows"] += int(grew)
                    ann_id += 1

    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ASCII on purpose: a 框外零件's ``label`` is the annotator's own words (task
    # U18), and pycocotools opens the file in the locale's encoding -- cp1252
    # on this machine, where a raw Chinese label made ``COCO(path)`` raise
    out.write_text(json.dumps(doc, ensure_ascii=True, indent=1), encoding="utf-8")
    return doc
