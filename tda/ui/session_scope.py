"""Which scope an edit means, read off the pixels that changed (spec 4.3).

The annotator paints; what they *meant* is a separate question, and spec 4.3
answers it from where the changed pixels landed.  Adding pixels under an
instance that is painted over this one is a statement about layering, not about
the silhouette; erasing exactly where another shape lies under this one is the
same statement the other way round; anything else is the shape itself.

**Never between a part and what it sits on** (task U6).  The D13 trial drew
the chassis and the screws on its motherboard: painting the chassis's
silhouette runs under every screw, erasing a screw to redraw it exposes the
chassis, and both used to come out as "change the layering" -- accepted with
the ``Enter`` pressed a hundred times a day, it put the chassis over the
screw.  A part lying (almost) wholly inside the other one's shape, or the other
one being its host -- its ``parent``, ``mounted_on``, ``fastens`` or
``socket_host``, the instance of its class's ``host_class``, any of those
transitively, or the chassis -- is a part sitting on it, and the edit is a
silhouette edit: :data:`~tda.ui.session_api.SCOPE_KEYFRAME`.
"""
from __future__ import annotations

from typing import Collection, Mapping, Optional

import numpy as np

from tda.core import masks as _masks
from tda.core.compiler import CompiledFrame
from tda.core.model import InstanceRec
from tda.ui import session_api as api

Box = tuple[int, int, int, int]

__all__ = ["CHASSIS_CLASS", "INSIDE_FRAC", "SCOPE_SPLIT_PREFIX", "SCOPE_ZORDER_ABOVE",
           "SCOPE_ZORDER_BELOW", "ZORDER_HINT_FRAC", "hosts_of", "split_zorder_scope",
           "splits_the_shape", "suggest_scope"]

#: How much of the *changed* pixels has to land inside another instance before
#: the default scope becomes "change the layering" (spec 4.3).
ZORDER_HINT_FRAC = 0.6

#: How much of a part's shape has to lie inside another's for the part to count
#: as sitting on it (task U6): "almost wholly" -- a screw on the board, the
#: board in the chassis -- with room for a re-trace that strays over the edge.
INSIDE_FRAC = 0.9

#: The class every other part sits in.
CHASSIS_CLASS = "chassis"

#: The pointers of :class:`~tda.core.model.InstanceRec` that say "this part is
#: on that one".
_HOST_POINTERS = ("parent", "mounted_on", "fastens", "socket_host")

#: ``suggest_scope`` answers, and what ``commit_edit`` accepts besides the three
#: :data:`~tda.ui.session_api.COMMIT_SCOPES`.
SCOPE_ZORDER_ABOVE = "zorder:above:"
SCOPE_ZORDER_BELOW = "zorder:below:"

#: ``split+zorder:above:<B>`` -- the same layering answer, but cutting a new
#: version of the shape instead of re-tracing the keyframe in force.  ``Ctrl+K``
#: on an open suggestion means this: without the pair it wrote the pixels and
#: left them hidden under ``B``, so the screen did not change at all.
SCOPE_SPLIT_PREFIX = "split+"


def splits_the_shape(scope: str) -> bool:
    """Does this layering scope ask for a new shape version rather than a re-trace?"""
    return str(scope).startswith(SCOPE_SPLIT_PREFIX)


def split_zorder_scope(scope: str) -> Optional[tuple[str, bool]]:
    """``(other instance, this one goes above)`` for a layering scope, else ``None``."""
    scope = str(scope)
    if splits_the_shape(scope):
        scope = scope[len(SCOPE_SPLIT_PREFIX):]
    for prefix, above in ((SCOPE_ZORDER_ABOVE, True), (SCOPE_ZORDER_BELOW, False)):
        if scope.startswith(prefix):
            return scope[len(prefix):], above
    return None


def hosts_of(instances: Mapping[str, InstanceRec],
             tax=None) -> dict[str, frozenset[str]]:
    """``instance -> every instance it sits on``, transitively, chassis included.

    Read off the relation pointers (:data:`_HOST_POINTERS`) and, for a part
    whose ``parent`` is blank, the instances of its class's ``host_class``
    (``tax`` may be ``None``: then only the pointers count).  The chassis is
    every other part's host.  A pointer to an instance this desktop does not
    have, or to the part itself, is ignored; a loop in the pointers ends where
    it closes.
    """
    chassis = {key for key, rec in instances.items() if rec.cls == CHASSIS_CLASS}
    direct: dict[str, set[str]] = {}
    for key, rec in instances.items():
        found = {str(target) for name in _HOST_POINTERS
                 if (target := getattr(rec, name, None)) and str(target) in instances}
        if tax is not None and not rec.parent:
            host_class = tax.host_class(rec.cls)
            if host_class:
                found |= {other for other, o in instances.items() if o.cls == host_class}
        found.discard(key)
        direct[key] = found
    out: dict[str, frozenset[str]] = {}
    for key in instances:
        seen: set[str] = set()
        stack = list(direct.get(key, ()))
        while stack:
            host = stack.pop()
            if host in seen or host == key:
                continue
            seen.add(host)
            stack.extend(direct.get(host, ()))
        if key not in chassis:
            seen |= chassis
        out[key] = frozenset(seen)
    return out


def suggest_scope(compiled: CompiledFrame, instance: str, before: np.ndarray,
                  edited: np.ndarray,
                  hosts: Optional[Mapping[str, Collection[str]]] = None) -> str:
    """The scope an edit defaults to (spec 4.3, 默认触发).

    The question is about the pixels that actually *changed*, never about the
    whole shape -- loading an instance into the editing layer and touching
    nothing is not a statement about anything:

    * pixels **added** where another instance ``B`` currently paints *over* this
      one say "I want to see this one there instead" -- i.e. put it above ``B``;
    * pixels **erased** exactly where ``B``'s shape lies under this one say the
      opposite: ``B`` should have been on top all along;
    * anything else is a change to the silhouette, so it edits the keyframe.

    Both answers name the pair and the direction, ``zorder:above:<B>`` and
    ``zorder:below:<B>``, because "change the layering" alone does not say which
    way round.  A hint is only given when at least
    :data:`ZORDER_HINT_FRAC` of the changed pixels fall inside ``B``.

    **And never one that hides a part sitting on another** (task U6).  "Above
    ``B``" is not offered when ``B`` sits on the edited instance -- ``B``'s
    shape lies at least :data:`INSIDE_FRAC` inside the edited one's (its shape
    before or after the edit), or the edited instance is one of ``B``'s
    ``hosts`` (:func:`hosts_of`) -- and "below ``B``" is not offered when the
    edited instance sits on ``B`` the same way.  The chassis drawn over the
    screws on its board, and a screw erased to be redrawn over the chassis,
    are silhouette edits.

    **Where it looks.**  The answer is about the pixels that changed, so the
    box those pixels live in is the only part of the canvas that can carry it,
    and each instance's own window is the only part *of that box* the instance
    can be in.  The comparison runs inside the two, which is the same answer
    for a great deal less reading: at 4032x3040 with forty instances the
    whole-canvas version was 1.09 s of a 1.9 s ``Enter`` -- eighty-four 12 MP
    ``&`` passes to explain a brush stroke.
    """
    before = np.asarray(before, dtype=bool)
    edited = np.asarray(edited, dtype=bool)
    _refuse_wrong_size(compiled, before, edited)
    window = _masks.bbox(before ^ edited)
    if window is None:      # nothing changed: not a statement about anything
        return api.SCOPE_KEYFRAME
    here, there = _crop(before, window), _crop(edited, window)
    added, erased = there & ~here, here & ~there

    shape = _EditedShape(before, edited, window)
    covering = _partner(compiled, instance, added, window, above=True)
    if covering is not None and not _sits_on(compiled, covering, instance,
                                             instance, shape, hosts):
        return f"zorder:above:{covering}"
    covered = _partner(compiled, instance, erased, window, above=False)
    if covered is not None and not _sits_on(compiled, instance, covered,
                                            instance, shape, hosts):
        return f"zorder:below:{covered}"
    return api.SCOPE_KEYFRAME


class _EditedShape:
    """The edited instance's shape for :func:`_sits_on`: ``before | edited``.

    What it was and what it is being made -- either can be the one the other
    part sits in.  Its extent is the box of ``before`` joined with the box the
    edit changed, measured only when a layering partner was found, and once.
    """

    def __init__(self, before: np.ndarray, edited: np.ndarray, changed: Box) -> None:
        self.before, self.edited, self.changed = before, edited, changed
        self._box: Optional[Box] = None

    def box(self) -> Box:
        if self._box is None:
            had = _masks.bbox(self.before)
            self._box = self.changed if had is None else (
                min(had[0], self.changed[0]), min(had[1], self.changed[1]),
                max(had[2], self.changed[2]), max(had[3], self.changed[3]))
        return self._box

    def inside(self, box: Box) -> np.ndarray:
        return _crop(self.before, box) | _crop(self.edited, box)


def _sits_on(compiled: CompiledFrame, part: str, base: str, editing: str,
             shape: _EditedShape,
             hosts: Optional[Mapping[str, Collection[str]]]) -> bool:
    """Does ``part`` sit on ``base``?  One of the two is ``editing``.

    The relations decide first: ``base`` among ``part``'s ``hosts`` is yes,
    ``part`` among ``base``'s is no -- a part never sits on what sits on it,
    and the chassis, everybody's host, is never read pixel by pixel.  Without
    either, the shapes do: ``part``'s lies at least :data:`INSIDE_FRAC` inside
    ``base``'s.  The edited instance's shape is ``shape``, the other one's its
    compiled amodal mask, and both are read only inside ``part``'s own box --
    the screw's, not the board's.
    """
    if hosts is not None:
        if base in (hosts.get(part) or ()):
            return True
        if part in (hosts.get(base) or ()):
            return False
    other = base if part == editing else part
    found = compiled.instances.get(other)
    if found is None or found.amodal is None:
        return False
    theirs_box = found.amodal_window
    if theirs_box is None:
        theirs_box = (0, 0, found.amodal.shape[1], found.amodal.shape[0])
    if part == editing:                 # the edited part, inside the other one?
        box = shape.box()
        whole = int(np.count_nonzero(shape.inside(box)))
        meet = _intersect(box, theirs_box)
        overlap = 0 if meet is None else int(np.count_nonzero(
            shape.inside(meet) & _crop(found.amodal, meet)))
    else:                               # the other part, inside the edited one?
        whole = int(np.count_nonzero(_crop(found.amodal, theirs_box)))
        overlap = int(np.count_nonzero(_crop(found.amodal, theirs_box)
                                       & shape.inside(theirs_box)))
    return whole > 0 and overlap >= INSIDE_FRAC * whole


def _canvas_of(compiled: CompiledFrame) -> Optional[tuple]:
    """The frame's ``(H, W)``, off the first mask it has, or ``None``."""
    for inst in compiled.instances.values():
        for mask in (inst.visible, inst.amodal):
            if mask is not None:
                return tuple(mask.shape)
    return None


def _refuse_wrong_size(compiled: CompiledFrame, before: np.ndarray,
                       edited: np.ndarray) -> None:
    """Raise unless both masks are this frame's canvas.

    The whole-canvas version of this function raised out of numpy the moment
    it tried to ``&`` the changed pixels with an instance's mask.  Working
    inside boxes means the slices always line up whatever the masks' real
    size, so the mistake would be made silently -- against the wrong pixels --
    instead of reported.
    """
    if before.shape != edited.shape:
        raise ValueError(
            f"the edit's two states do not match: the shape before is "
            f"{before.shape!r} and after {edited.shape!r}"
        )
    canvas = _canvas_of(compiled)
    if canvas is not None and tuple(before.shape) != canvas:
        raise ValueError(
            f"the edited mask is {tuple(before.shape)!r}, which does not match "
            f"this frame's canvas {canvas!r}"
        )


def _crop(mask: np.ndarray, box: Box) -> np.ndarray:
    """``mask`` inside ``box`` -- a view, so nothing is copied."""
    x0, y0, x1, y1 = box
    return mask[y0:y1, x0:x1]


def _intersect(a: Optional[Box], b: Optional[Box]) -> Optional[Box]:
    """The box both cover, or ``None`` when they do not meet.

    ``None`` on the way *in* means "no bound known", which is not the same as
    "empty": a caller that cannot say where its pixels are must be compared
    everywhere the other one allows.
    """
    if a is None:
        return b
    if b is None:
        return a
    box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return None if box[2] <= box[0] or box[3] <= box[1] else box


def _partner(compiled: CompiledFrame, instance: str, changed: np.ndarray,
             window: Box, above: bool) -> Optional[str]:
    """The instance the changed pixels are a layering statement about, if any.

    ``above=True`` looks for an instance painting *over* ``instance`` whose
    visible pixels the edit reached into; ``above=False`` for one painting
    *under* it, compared on the amodal shapes, since what lies under is by
    definition not visible.

    ``changed`` is the changed pixels **cropped to** ``window``; each instance
    is then read only where its own window and ``window`` overlap.  An instance
    with no window is read across the whole of ``window``, so a compiled frame
    that cannot say where an instance is still gets the same answer.
    """
    total = int(changed.sum())
    if total == 0:
        return None
    best, best_overlap = None, 0
    for other, inst in compiled.instances.items():
        if other == instance or _paints_above(compiled, other, instance) is not above:
            continue
        region = inst.visible if above else inst.amodal
        if region is None:
            continue
        box = _intersect(window, inst.window if above else inst.amodal_window)
        if box is None:
            continue
        local = (box[0] - window[0], box[1] - window[1],
                 box[2] - window[0], box[3] - window[1])
        overlap = int(np.count_nonzero(_crop(changed, local) & _crop(region, box)))
        if overlap > best_overlap:
            best, best_overlap = other, overlap
    return best if best_overlap >= ZORDER_HINT_FRAC * total else None


def _paints_above(compiled: CompiledFrame, other: str, instance: str) -> Optional[bool]:
    """Is ``other`` painted over ``instance``? ``None`` when they never meet.

    A pair override that held between the two decides (task U6: it is no
    longer a position in ``painted``); the painted order otherwise.
    """
    for group, order in compiled.painted.items():
        if other in order and instance in order:
            pairs = compiled.pairs.get(group, ())
            if (other, instance) in pairs:
                return True
            if (instance, other) in pairs:
                return False
            return order.index(other) > order.index(instance)
    return None
