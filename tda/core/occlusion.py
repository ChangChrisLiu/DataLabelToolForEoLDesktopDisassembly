"""Dynamic occlusion: which part has to come out of the way before another (R17).

The owner's ruling of 2026-11-04 (「动态遮挡」): at a step, a part Y blocks a
part X when

* Y is still in the machine (not ``removed``; nor ``open`` / ``displaced``,
  the states that clear a ``covered_by`` -- Y has been taken away or opened),
  and is not a cable, a loose cable or a virtual cable node -- a cable can be
  pushed aside;
* Y is **above** X in the frame's layer order (the ``(view, pose segment)``
  z-order and its pair overrides, :func:`tda.core.compiler.above`);
* Y's amodal mask covers at least :data:`OCCLUSION_THRESHOLD` (20 %) of X's
  amodal mask area, in the scanner frame of that step -- the picture *after*
  the step, the one a "what next" question is asked on.

Y then has to be removed, displaced or opened first; once its geometry no
longer covers X (or it is out of the way) the block is gone by itself. It is
a property of the moment, so it is **never stored** as a ``relation`` row: the
VLM export asks :func:`occlusion_blockers` per step and hands the answer to
:func:`tda.core.graph.legal_actions` (``occluded=``).

Six kinds of pair are left out on purpose, each a ruling or a contradiction:

* a case fan that is not loose (R15: fixed to the chassis, it constrains
  nothing but through its own lead);
* a drive cage the log never opens, moves or takes out (R14: only a cage that
  can be opened has anything to do with the teardown; D54 D59 D60 D61 D65
  keep their optical-drive cage in place from the first frame to the last,
  and the board still came out from under it);
* a drive cage (bracket) over a connector, whatever the stored layer order
  says (the owner, 2026-11-04: the plug lies on top of the cage -- D27 and
  the five optical brackets were all drawn the wrong way round);
* the chassis and the fan screw holes, which are not parts that move (nor
  are they ever X: nothing is done to them);
* a plug over a plug, a latch over a latch, a stick over a stick, a screw
  over a screw (:data:`SAME_CLASS_FREE`): interchangeable actions have no
  order between them;
* a Y that waits, through the stored constraint graph, for X itself (its own
  screws, its own lead, the latch that locks it): X under Y would then be a
  loop nobody can start -- the screws of a fan are under the fan's outline
  and come out first, which is what the graph already says.

The amodal masks are the keyframes' shapes transformed onto the frame
(:func:`tda.core.compiler.keyframe_amodal`, the COCO export's
``amodal_segmentation``); presence and placement are the compiled rows'.
Results are cached per ``(database, desktop, view, threshold)`` and dropped
whenever the database's op log has moved on.

**Owner exceptions** (:data:`EXCEPTIONS_PATH`, ``configs/occlusion_exceptions.yaml``):
one logged action the owner ruled is not blocked -- ``(desktop, step, target,
reason)``. The part is left out of the answer on the frame that action is
judged on (the last frame before its step), and nowhere else.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

import yaml
from pycocotools import mask as coco_mask

from tda.core import masks
from tda.core.graph_rules import (
    CABLE_PREFIX,
    REQUIRED_STATES,
    Edge,
    active_edges,
    is_loose,
)
from tda.core.model import FrameKey

__all__ = [
    "NEVER_OCCLUDES",
    "EXCEPTIONS_PATH",
    "NEVER_OVER",
    "OCCLUSION_THRESHOLD",
    "SAME_CLASS_FREE",
    "Occlusion",
    "OcclusionException",
    "occlusion_exceptions",
    "clear_cache",
    "frame_occlusions",
    "occlusion_blockers",
    "occlusion_details",
    "waits_for",
]

#: The owner's threshold: Y covers at least this share of X's amodal area.
OCCLUSION_THRESHOLD = 0.20
#: Classes that never block by covering: cables can be pushed aside, the
#: chassis and the fan screw holes are not parts that move.
NEVER_OCCLUDES = frozenset({"cable", "loose_cable", "chassis", "screw_hole"})
#: Classes whose actions are interchangeable among themselves: one plug over
#: another, one latch, stick or screw over the next, puts no order between
#: them (the rules' preamble: 「插头和插头之间、内存卡扣和内存卡扣之间」没有先后).
SAME_CLASS_FREE = frozenset({"connector", "ram_latch", "ram_module", "screw"})
#: A case fan fixed to the chassis constrains only through its lead (R15).
CASE_FAN_CLASS = "case_fan"
#: A drive cage blocks by covering only if the log ever moves it (R14).
CAGE_CLASS = "drive_cage"
#: ``(Y class, X class)`` pairs that never occlude whatever the layer order
#: says: a plug lies on top of the drive cage (the owner, 2026-11-04).
NEVER_OVER = frozenset({(CAGE_CLASS, "connector")})
#: The owner's per-action exceptions.
EXCEPTIONS_PATH = Path(__file__).resolve().parents[2] / "configs" / "occlusion_exceptions.yaml"


@dataclass(frozen=True)
class OcclusionException:
    """One logged action the owner ruled is not blocked by covering."""

    desktop: int
    step: int
    target: str
    reason: str


@lru_cache(maxsize=4)
def _exceptions_cached(path: str) -> tuple[OcclusionException, ...]:
    p = Path(path)
    if not p.exists():
        return ()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return tuple(OcclusionException(int(e["desktop"]), int(e["step"]), str(e["target"]),
                                    str(e.get("reason") or ""))
                 for e in raw.get("exceptions") or [])


def occlusion_exceptions(path: Optional[str | Path] = None) -> tuple[OcclusionException, ...]:
    """The owner's exceptions (``configs/occlusion_exceptions.yaml``)."""
    return _exceptions_cached(str(Path(path or EXCEPTIONS_PATH).resolve()))
#: Y in one of these states is out of the way (the ``covered_by`` states).
CLEARED_STATES = REQUIRED_STATES["covered_by"]
REMOVED = "removed"
IN_CHASSIS = "in_chassis"
MAIN_PART = "main"


@dataclass(frozen=True)
class Occlusion:
    """Y (``blocker``) covers ``frac`` of X's (``target``) amodal area, from above."""

    target: str
    blocker: str
    px: int
    target_px: int

    @property
    def frac(self) -> float:
        return self.px / self.target_px if self.target_px else 0.0


def waits_for(edges: Iterable[Edge]) -> dict[str, set[str]]:
    """``part -> every node it waits for``, transitively, through the active edges."""
    direct: dict[str, set[str]] = {}
    for edge in active_edges(edges):
        direct.setdefault(edge.target, set()).add(edge.blocker)
    out: dict[str, set[str]] = {}
    for start in direct:
        seen: set[str] = set()
        stack = list(direct[start])
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(direct.get(node, ()))
        out[start] = seen
    return out


def _boxes_meet(a, b) -> bool:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    return not (ax >= bx + bw or bx >= ax + aw or ay >= by + bh or by >= ay + ah)


def frame_occlusions(
    amodal: Mapping[str, dict],
    targets: Iterable[str],
    blockers: Iterable[str],
    is_above: Callable[[str, str], bool],
    threshold: float = OCCLUSION_THRESHOLD,
    exempt: Optional[Callable[[str, str], bool]] = None,
) -> dict[str, list[Occlusion]]:
    """The pure core: ``X -> [Occlusion]`` of one frame, largest cover first.

    ``amodal`` maps instance -> COCO RLE of its whole shape on the frame;
    ``targets`` / ``blockers`` say who may be X / Y; ``is_above(Y, X)`` is the
    layer order; ``exempt(Y, X)`` drops a pair outright.
    """
    boxes = {k: coco_mask.toBbox(v) for k, v in amodal.items()}
    areas = {k: int(coco_mask.area(v)) for k, v in amodal.items()}
    ys = [y for y in sorted(set(blockers)) if y in amodal and areas[y] > 0]
    out: dict[str, list[Occlusion]] = {}
    for x in sorted(set(targets)):
        if x not in amodal or areas[x] <= 0:
            continue
        found: list[Occlusion] = []
        for y in ys:
            if y == x or not _boxes_meet(boxes[y], boxes[x]):
                continue
            if exempt is not None and exempt(y, x):
                continue
            px = int(coco_mask.area(coco_mask.merge([amodal[x], amodal[y]], intersect=1)))
            if px <= 0 or px < threshold * areas[x]:
                continue
            if not is_above(y, x):
                continue
            found.append(Occlusion(target=x, blocker=y, px=px, target_px=areas[x]))
        if found:
            out[x] = sorted(found, key=lambda o: (-o.frac, o.blocker))
    return out


# --------------------------------------------------------------------------- #
# the database half
# --------------------------------------------------------------------------- #
_CACHE: dict[tuple, dict[int, dict[str, list[Occlusion]]]] = {}


def clear_cache() -> None:
    """Forget every cached answer (the cache also drops itself on any new op)."""
    _CACHE.clear()


def _fingerprint(db) -> tuple:
    row = db.conn.execute("SELECT max(id), count(*) FROM op_log").fetchone()
    return (str(getattr(db, "path", id(db))), row[0], row[1], db.conn.total_changes)


def _above_fn(db, desktop: int, view: str, seg: int) -> Callable[[str, str], bool]:
    from tda.core.compiler_layers import above

    zorder = db.zorder(desktop, view, seg)
    overrides = db.pair_overrides(desktop, view, seg)
    layers: dict[str, list[tuple[str, str]]] = {}
    for inst, part in zorder.order:
        layers.setdefault(str(inst), []).append((str(inst), str(part)))

    def is_above(y: str, x: str) -> bool:
        ly = layers.get(y) or [(y, MAIN_PART)]
        lx = layers.get(x) or [(x, MAIN_PART)]
        return any(above(a, b, zorder, overrides) for a in ly for b in lx)

    return is_above


def occlusion_details(
    db,
    desktop: int,
    step: int,
    *,
    view: str = "scan",
    threshold: float = OCCLUSION_THRESHOLD,
    ctx=None,
    edges: Optional[list[Edge]] = None,
) -> dict[str, list[Occlusion]]:
    """``X -> [Occlusion]`` at ``step``: who covers X from above in that step's frame.

    ``ctx`` is the desktop's :class:`~tda.core.export.coco.DesktopCtx` of
    ``view`` (read when not given); ``edges`` its stored constraint edges
    (read when not given). A step without a frame, or a frame without
    compiled rows, blocks nothing.
    """
    from tda.core.compiler import keyframe_amodal
    from tda.core.graph import constraint_edges, edges_from_db
    from tda.core.truth_inputs import pose_segment_of

    cache_key = (desktop, view, float(threshold), _fingerprint(db))
    per_step = _CACHE.setdefault(cache_key, {})
    if step in per_step:
        return per_step[step]
    if ctx is None:
        from tda.core.export.coco import load_ctx
        from tda.core.taxonomy import load_taxonomy

        ctx = load_ctx(db, load_taxonomy(), desktop, view)
    if edges is None:
        edges = constraint_edges(edges_from_db(db, desktop))
    waiting = waits_for(edges)

    key = FrameKey(desktop, step, view)
    compiled = db.compiled(key)
    rows = {inst: row for inst, row in compiled.items()
            if inst in ctx.instances and row.get("visible_rle")
            and (row.get("geom_type") or "mask") == "mask"
            and (row.get("placement") or IN_CHASSIS) == IN_CHASSIS}
    if not rows:
        per_step[step] = {}
        return {}
    state = ctx.state_at(step)
    seg = pose_segment_of(db, key)
    transform = db.transform(key)
    amodal: dict[str, dict] = {}
    for inst, row in sorted(rows.items()):
        current = state.get(inst)
        if current is None or current.state == REMOVED:
            continue
        rle = masks._coco_rle(row["visible_rle"])
        hw = (int(rle["size"][0]), int(rle["size"][1]))
        kf = ctx.keyframe_at(inst, step, row.get("placement") or IN_CHASSIS, seg)
        if kf is None or kf.geom_type != "mask":
            continue
        mask, window = keyframe_amodal(kf, transform, hw)
        amodal[inst] = masks._coco_rle(masks.encode_rle(mask, window))

    moved = {a.target for a in ctx.actions if a.result == "success"}
    moved |= {e.target for e in ctx.events if e.attr == "state" and e.step >= 1}

    def can_block(y: str) -> bool:
        rec = ctx.instances[y]
        if rec.cls in NEVER_OCCLUDES or y.startswith(CABLE_PREFIX):
            return False
        if rec.cls == CASE_FAN_CLASS and not is_loose(rec):
            return False  # R15
        if rec.cls == CAGE_CLASS and y not in moved:
            return False  # R14: a cage nobody can open is part of the chassis
        return state[y].state not in CLEARED_STATES

    def exempt(y: str, x: str) -> bool:
        cls, xcls = ctx.instances[y].cls, ctx.instances[x].cls
        if cls == xcls and cls in SAME_CLASS_FREE:
            return True
        if (cls, xcls) in NEVER_OVER:
            return True
        return x in waiting.get(y, ())

    frames = sorted(int(f["step"]) for f in db.frames_for(desktop, view))
    excused = set()
    for exc in occlusion_exceptions():
        before = [f for f in frames if f < exc.step]
        if exc.desktop == desktop and before and before[-1] == step:
            excused.add(exc.target)
    targets = [x for x in amodal
               if ctx.instances[x].cls not in NEVER_OCCLUDES and x not in excused]
    found = frame_occlusions(
        amodal, targets=targets, blockers=[y for y in amodal if can_block(y)],
        is_above=_above_fn(db, desktop, view, seg), threshold=threshold, exempt=exempt)
    per_step[step] = found
    return found


def occlusion_blockers(db, desktop: int, step: int, **kw) -> dict[str, list[str]]:
    """``{X: [Y, ...]}``: the parts that must come out of X's way first, at ``step``.

    The answer :func:`tda.core.graph.legal_actions` takes as ``occluded=``.
    Keyword arguments as :func:`occlusion_details`.
    """
    return {x: [o.blocker for o in found]
            for x, found in occlusion_details(db, desktop, step, **kw).items()}
