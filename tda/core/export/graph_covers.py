"""Who covers whom on the confirmed frames: the pixel evidence of ``export-graph`` (task U46).

A precedence edge says *B waits for A*. A picture can only say *A takes pixels
of B*: on a confirmed frame, part of B's whole shape (its amodal mask) lies
under A's visible mask. That is evidence for an edge, not the edge itself --
a screw on a board covers a few pixels of the board and is a ``fastened_by``,
a cable lying across a fan covers it and gates nothing -- so this module only
measures, and says which stored edge (if any) the covering agrees with. A
covering no edge explains is a **candidate**: the triage list for whoever
decides whether a ``covered_by`` / ``blocked_by`` is missing.

Most coverings need no edge and say so by construction, so a candidate is a
covering that **nothing** accounts for: no edge from ``below`` to ``above``
(``explained_by``), none the other way round (``reverse_edges``: the two are
already ordered, the pixels are one outline against the other) and no relation
in the instance table (``structural``: one is attached to, mounted on, plugged
into or screwing down the other, the upper one rides out with a parent, or the
lower part is the chassis, which never moves). ``no_edge`` keeps the brief's
own definition beside it (no edge either way, whatever the instance table says).

What is read, and nothing else:

* the stored ``compiled_mask`` rows of the view's **confirmed** frames
  (:func:`tda.core.export.coco.frame_is_verified`; a frame with an open
  conflict is not confirmed), and of those only the ``verified`` mask rows of
  machine parts in the chassis -- no Label Studio draft, no 框外零件, no part
  this machine does not have, nothing on the bench (a part lying on another one
  on the table says nothing about the order they came out in);
* each part's whole shape on that frame: the keyframe the compiler would select
  there (:meth:`tda.core.export.coco.DesktopCtx.keyframe_at`, the same
  narrowing by placement chain and pose segment), turned onto the frame by its
  stored transform (:func:`tda.core.compiler.keyframe_amodal`) -- what the COCO
  export's ``amodal_segmentation`` is. **No frame is compiled**: the visible
  masks are the confirmed rows exactly as stored.

The ``occluder_mask`` table the brief pointed at holds the unlabelled occluder
layers of a frame (cable, hand) by type, not part-against-part pixels, and is
empty on the real database; the scanner package (``scripts/export_scanpack.py``)
measures its ``occluders`` the same way this does, off amodal and visible
masks.

For a frame and an ordered pair ``(above, below)`` the count is::

    px = | amodal(below) ∩ visible(above) ∖ visible(below) |

computed off the run lengths (:mod:`pycocotools`), never decoding a frame:
``| amodal(below) ∩ visible(above) |`` minus the part of it ``below`` itself
still shows (two confirmed masks may claim one pixel). Only pairs whose boxes
meet are measured, and a pair is listed when ``px > 0`` on at least one frame.

``order_source`` says what put those pixels on ``above``'s side on that frame:
``per_frame_mask`` when either part carries a 「只改这一帧」 mask there (the
annotator drew the visible pixels by hand), else ``pair_override`` when the
pose segment has a pairwise layering exception for the two, else ``zorder``
(the segment's stored layer order, or the compiler's default one where none
was saved). A pair seen under several gets the one of most frames, and
``order_sources`` counts them all.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from pycocotools import mask as coco_mask

from tda.core import masks
from tda.core.compiler import keyframe_amodal
from tda.core.db import Db
from tda.core.export.coco import DesktopCtx, frame_is_verified
from tda.core.graph import GATED_VERBS, Edge
from tda.core.graph_rules import active_edges, verb_effect
from tda.core.model import FrameKey, is_context_key
from tda.core.states import IN_CHASSIS
from tda.core.truth_inputs import pose_segment_of

__all__ = [
    "EXPLAINING_TYPES",
    "ORDER_SOURCES",
    "PAIR_OVERRIDE",
    "PER_FRAME_MASK",
    "ZORDER",
    "CoverScan",
    "class_key",
    "explaining_edges",
    "reverse_edges",
    "scan_covers",
    "structural_reasons",
]

ZORDER = "zorder"
PAIR_OVERRIDE = "pair_override"
PER_FRAME_MASK = "per_frame_mask"
#: Most specific first: the tie-break when a pair is seen as often under two.
ORDER_SOURCES = (PER_FRAME_MASK, PAIR_OVERRIDE, ZORDER)
#: Edge types that explain a covering whichever way round they point.
EXPLAINING_TYPES = frozenset({"covered_by", "blocked_by"})
VERIFIED = "verified"
MASK = "mask"


def class_key(rec) -> str:
    """The class a covering is grouped under: ``cpu_cooler.fan``, ``screw.motherboard``.

    The taxonomy class, refined by the instance's ``kind`` (or a screw's
    ``role``) when it has one -- "a cover over a CPU fan" is the triage
    question, "a cover over a CPU cooler" is two different ones.
    """
    if rec is None:
        return "?"
    attrs = rec.attrs or {}
    refine = attrs.get("kind") or attrs.get("role")
    return f"{rec.cls}.{refine}" if refine else str(rec.cls)


def explaining_edges(edges: list[Edge], above: str, below: str) -> list[str]:
    """Labels of the active edges a covering of ``below`` by ``above`` agrees with.

    ``below`` waits for ``above`` (``target = below``, ``blocker = above``) of
    any hard type, or a ``covered_by`` / ``blocked_by`` between the two in
    either direction. Sorted, so the list is the same on every run.
    """
    out = set()
    for edge in active_edges(edges):
        if edge.target == below and edge.blocker == above:
            out.add(edge.label())
        elif edge.type in EXPLAINING_TYPES and {edge.target, edge.blocker} == {above, below}:
            out.add(edge.label())
    return sorted(out)


def reverse_edges(edges: list[Edge], above: str, below: str) -> list[str]:
    """Labels of the active edges that tie the two the *other* way: ``above`` waits for ``below``.

    A CPU covers a few pixels of the socket lever that locks it: the pair is
    already related (``locked_by(cpu, lever)``), the covering is the outline of
    one lying against the other, and no edge is missing. Not in
    ``explained_by`` -- the edge does not say what the picture says -- but no
    candidate either. ``covered_by`` / ``blocked_by`` are in ``explained_by``
    whichever way round they point, so they never show up here.
    """
    return sorted({e.label() for e in active_edges(edges)
                   if e.target == above and e.blocker == below
                   and e.type not in EXPLAINING_TYPES})


#: The relational fields that make a covering a matter of build, not of order.
_STRUCTURE_FIELDS = ("mounted_on", "socket_host", "fastens")


def structural_reasons(instances: dict, tax, above: str, below: str) -> list[str]:
    """Why a covering of ``below`` by ``above`` needs no precedence edge, from the instance table.

    * ``above.attached_to`` / ``below.attached_to`` -- one rides out of the
      chassis inside the other (``parent`` + ``attached``, any depth): a RAM
      latch on its board, a captive screw on its fan;
    * ``above.rides_with_parent`` -- the upper part is attached to a third
      one and leaves with it (a cooler's captive screw over the board): the
      covering that can order anything is the parent's, which is its own pair;
    * ``above.mounted_on`` / ``above.socket_host`` / ``above.fastens`` and the
      ``below.*`` mirror -- one sits on, plugs into or screws down the other;
    * ``below_fixed`` -- the lower part has no verb that changes its state
      (the chassis): covering it can hold nothing up.
    Sorted; empty when the instance table says nothing about the pair.
    """
    out: set[str] = set()
    for name, one, other in (("above", above, below), ("below", below, above)):
        rec = instances.get(one)
        if rec is None:
            continue
        seen: set[str] = set()
        cur = rec
        while cur is not None and cur.attached and cur.parent and cur.parent not in seen:
            if cur.parent == other:
                out.add(f"{name}.attached_to")
                break
            seen.add(cur.parent)
            cur = instances.get(cur.parent)
        if name == "above" and seen and f"{name}.attached_to" not in out:
            # a captive screw over the board rides out with its cooler: the
            # covering that orders anything is the cooler's, listed on its own
            out.add("above.rides_with_parent")
        for field_name in _STRUCTURE_FIELDS:
            if getattr(rec, field_name, None) == other:
                out.add(f"{name}.{field_name}")
    low = instances.get(below)
    if low is not None and tax is not None and not any(
            verb_effect(tax, low.cls, low.attrs or {}, verb) is not None
            for verb in sorted(GATED_VERBS)):
        out.add("below_fixed")
    return sorted(out)


@dataclass
class _Pair:
    """What one ordered pair accumulates over the frames."""

    px: dict[int, int] = field(default_factory=dict)        # step -> pixels taken
    sources: Counter = field(default_factory=Counter)        # order source -> frames
    segments: set[int] = field(default_factory=set)


@dataclass
class CoverScan:
    """The covers of one desktop and view, plus what they were measured on."""

    covers: list[dict]
    confirmed_frames: int = 0
    frames_with_covers: int = 0


def _boxes_meet(a, b) -> bool:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    return not (ax >= bx + bw or bx >= ax + aw or ay >= by + bh or by >= ay + ah)


def _turn(transform) -> tuple[float, float, float, float]:
    return (float(transform.scale), float(transform.theta), float(transform.tx),
            float(transform.ty))


def _order_source(above: str, below: str, hand_drawn: set[str],
                  pairs: set[tuple[str, str]]) -> str:
    if above in hand_drawn or below in hand_drawn:
        return PER_FRAME_MASK
    if (above, below) in pairs or (below, above) in pairs:
        return PAIR_OVERRIDE
    return ZORDER


def _frame_pairs(vis: dict[str, dict], amodal: dict[str, dict]
                 ) -> list[tuple[str, str, int]]:
    """``(above, below, px)`` of one frame, ``px > 0`` only, in key order."""
    vis_box = {k: coco_mask.toBbox(v) for k, v in vis.items()}
    out: list[tuple[str, str, int]] = []
    for below in sorted(amodal):
        whole = amodal[below]
        whole_box = coco_mask.toBbox(whole)
        own = vis.get(below)
        for above in sorted(vis):
            if above == below or not _boxes_meet(vis_box[above], whole_box):
                continue
            px = int(coco_mask.area(coco_mask.merge([whole, vis[above]], intersect=1)))
            if px and own is not None:
                px -= int(coco_mask.area(coco_mask.merge([whole, vis[above], own],
                                                         intersect=1)))
            if px > 0:
                out.append((above, below, px))
    return out


def scan_covers(db: Db, ctx: DesktopCtx, view: str, edges: list[Edge],
                removed_at: dict[str, Optional[int]]) -> CoverScan:
    """Every ordered covering pair of one desktop's confirmed ``view`` frames.

    ``removed_at`` is the step each instance first reads ``removed`` in the
    state timeline (``None``: the log never takes it out); it is copied onto
    each pair so a reader sees at once whether the covering part did come out
    first. Read-only: SELECTs and run-length arithmetic, nothing written.
    """
    desktop = ctx.desktop
    disputed = {int(r["step"]) for r in db.conflicts(desktop, view, open_only=True)}
    shapes: dict[tuple, dict] = {}
    seg_pairs: dict[int, set[tuple[str, str]]] = {}
    found: dict[tuple[str, str], _Pair] = {}
    confirmed = with_covers = 0

    for frame in db.frames_for(desktop, view):
        step = int(frame["step"])
        if not ctx.exportable(step) or step in disputed:
            continue
        key = FrameKey(desktop, step, view)
        compiled = {inst: row for inst, row in db.compiled(key).items()
                    if not is_context_key(inst)}
        if not frame_is_verified(db, key, compiled):
            continue
        confirmed += 1
        rows = {inst: row for inst, row in compiled.items()
                if inst in ctx.instances and row.get("status") == VERIFIED
                and (row.get("geom_type") or MASK) == MASK and row.get("visible_rle")
                and (row.get("placement") or IN_CHASSIS) == IN_CHASSIS}
        if len(rows) < 2:
            continue
        seg = pose_segment_of(db, key)
        transform = db.transform(key)
        turn = _turn(transform)
        if seg not in seg_pairs:
            seg_pairs[seg] = {(po.above, po.below)
                              for po in db.pair_overrides(desktop, view, seg)}
        hand_drawn = {inst for inst, fo in db.frame_overrides(key).items() if fo.visible_rle}

        vis: dict[str, dict] = {}
        amodal: dict[str, dict] = {}
        for inst, row in sorted(rows.items()):
            rle = masks._coco_rle(row["visible_rle"])
            hw = (int(rle["size"][0]), int(rle["size"][1]))
            if int(coco_mask.area(rle)) > 0:
                vis[inst] = rle
            kf = ctx.keyframe_at(inst, step, row.get("placement") or IN_CHASSIS, seg)
            if kf is None or kf.geom_type != MASK:
                continue
            memo = (kf.id, kf.version, id(kf) if kf.id is None else None, turn, hw)
            whole = shapes.get(memo)
            if whole is None:
                mask, window = keyframe_amodal(kf, transform, hw)
                whole = shapes[memo] = masks._coco_rle(masks.encode_rle(mask, window))
            amodal[inst] = whole

        pairs = _frame_pairs(vis, amodal)
        if pairs:
            with_covers += 1
        for above, below, px in pairs:
            acc = found.setdefault((above, below), _Pair())
            acc.px[step] = px
            acc.sources[_order_source(above, below, hand_drawn, seg_pairs[seg])] += 1
            acc.segments.add(int(seg))

    covers = []
    for (above, below), acc in sorted(found.items()):
        steps = sorted(acc.px)
        explained = explaining_edges(edges, above, below)
        reverse = reverse_edges(edges, above, below)
        structure = structural_reasons(ctx.instances, ctx.tax, above, below)
        top = max(ORDER_SOURCES, key=lambda s: (acc.sources.get(s, 0),
                                                -ORDER_SOURCES.index(s)))
        covers.append({
            "above": above,
            "below": below,
            "above_class": class_key(ctx.instances.get(above)),
            "below_class": class_key(ctx.instances.get(below)),
            "frames": steps,
            "n_frames": len(steps),
            "max_px": max(acc.px.values()),
            "first": steps[0],
            "last": steps[-1],
            "explained_by": explained,
            "reverse_edges": reverse,
            "structural": structure,
            "no_edge": not explained and not reverse,
            "candidate": not explained and not reverse and not structure,
            "order_source": top,
            "order_sources": {s: int(acc.sources[s]) for s in ORDER_SOURCES
                              if acc.sources.get(s)},
            "pose_segments": sorted(acc.segments),
            "above_removed_at": removed_at.get(above),
            "below_removed_at": removed_at.get(below),
        })
    return CoverScan(covers=covers, confirmed_frames=confirmed,
                     frames_with_covers=with_covers)
