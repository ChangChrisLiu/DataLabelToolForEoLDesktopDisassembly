"""「这一帧的遮挡顺序」: who is painted over whom on the frame on screen (task U44).

The annotator explains a frame's occlusion in the note (「挡板拉下来之后遮住了
psu 的一小部分，遮挡关系变了」), and the controller reads the notes and fixes
the layers.  To write only what is *wrong*, the annotator has to see what the
compiler did: this module turns the compiled frame the canvas already holds
into plain lines, one per part, top to bottom in painting order, each with the
parts it takes pixels from on this frame and how many.

Read off the compiled rows and nothing else -- no compile, no database: a part
``X`` takes ``n`` pixels from ``Y`` when ``n`` of ``X``'s visible pixels lie on
``Y``'s shape (its amodal mask; its visible mask when it has none, as a shape
drawn for this frame only may not).  Only parts of one placement group compete
(spec §3.3 step 5: bench and chassis are composited apart), and only pairs that
overlap here (``n > 0``) are listed, largest first.  A pair the compiler took
from a ``PairOverride`` (:attr:`~tda.core.compiler.CompiledFrame.pairs`) is
marked 「（手动）」; a part with a per-frame mask (「只改这一帧」, the caller
says which) 「（这一帧单独画）」; a part whose visible mask is empty 「被完全盖住」.
``out_of_view`` parts are left out: they take no pixels (spec §4.2 画框外).

Pairs are counted inside the windows the compiler already knows each mask is
empty outside, so a frame of forty parts costs a few milliseconds, not forty
full-canvas intersections.  No Qt here: the dock (:mod:`tda.ui.panels.notes_dock`)
shows the lines and the window feeds it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional

import numpy as np

from tda.core.compiler import CompiledFrame
from tda.core.model import CONTEXT_PLACEMENT, Placement, Visibility
from tda.ui.class_names import instance_label

__all__ = ["COVERED", "EMPTY", "MANUAL", "OWN_FRAME", "OrderLine", "Taken", "header_text",
           "labeller", "line_text", "occlusion_order", "overlap_pairs"]

#: Marks of a line (spec §4.4, v1.43).
MANUAL = "（手动）"
OWN_FRAME = "（这一帧单独画）"
COVERED = "被完全盖住"
#: The header of a frame with nothing compiled on it.
EMPTY = "这一帧没有编译出的零件"
#: Painting order of the groups, bottom-up -- :func:`tda.ui.session_rows.painted`'s.
GROUPS = (Placement.IN_CHASSIS.value, Placement.ON_BENCH.value, CONTEXT_PLACEMENT)
OUT_OF_VIEW = Visibility.OUT_OF_VIEW.value

Window = tuple[int, int, int, int]


@dataclass(frozen=True)
class Taken:
    """``pixels`` of ``below``'s shape that the line's part paints over on this frame."""

    below: str
    pixels: int
    #: The order of the pair comes from a ``PairOverride`` (spec §4.3 改层级).
    manual: bool = False


@dataclass(frozen=True)
class OrderLine:
    """One part of the frame, as the list shows it."""

    instance: str
    #: The parts it takes pixels from, largest first.
    takes: tuple[Taken, ...] = ()
    #: It has a per-frame mask (「只改这一帧」).
    own_frame: bool = False
    #: Nothing of it is visible: the parts above cover it completely.
    covered: bool = False


def _bbox(mask: np.ndarray) -> Optional[Window]:
    """``(x0, y0, x1, y1)`` of a mask's pixels, upper bounds exclusive; ``None``: empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask[rows[0]:rows[-1] + 1].any(axis=0))
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def _clip(window: Optional[Window], hw: tuple[int, int]) -> Optional[Window]:
    if window is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in window)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(int(hw[1]), x1), min(int(hw[0]), y1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _meet(a: Optional[Window], b: Optional[Window]) -> Optional[Window]:
    if a is None or b is None:
        return None
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


class _Part:
    """A listed part's visible mask and shape, each with the window it lies in."""

    __slots__ = ("instance", "visible", "vis_window", "shape", "shape_window")

    def __init__(self, instance: str, inst) -> None:
        self.instance = instance
        self.visible = np.asarray(inst.visible, dtype=bool)
        hw = self.visible.shape[:2]
        known = _clip(inst.window, hw)
        # a window is a promise of emptiness outside it; inside, measure once
        self.vis_window = self._tight(self.visible, known)
        amodal = inst.amodal
        if amodal is not None:
            self.shape = np.asarray(amodal, dtype=bool)
            self.shape_window = self._tight(self.shape, _clip(inst.amodal_window, hw))
        else:
            self.shape, self.shape_window = self.visible, self.vis_window

    @staticmethod
    def _tight(mask: np.ndarray, window: Optional[Window]) -> Optional[Window]:
        if window is None:
            return _bbox(mask)
        x0, y0, x1, y1 = window
        inner = _bbox(mask[y0:y1, x0:x1])
        if inner is None:
            return None
        return (x0 + inner[0], y0 + inner[1], x0 + inner[2], y0 + inner[3])


def _listed(compiled: CompiledFrame, instance: str) -> bool:
    inst = compiled.instances.get(instance)
    return (inst is not None and inst.visible is not None
            and str(inst.visibility) != OUT_OF_VIEW)


def _groups(compiled: CompiledFrame) -> list[list[_Part]]:
    """The listed parts of each placement group, bottom-up, groups in painting order."""
    return [[_Part(name, compiled.instances[name])
             for name in compiled.painted.get(group, []) if _listed(compiled, name)]
            for group in GROUPS]


def overlap_pairs(compiled: CompiledFrame,
                  groups: Optional[list[list[_Part]]] = None) -> dict[tuple[str, str], int]:
    """``{(above, below): pixels}`` of every pair that overlaps on this frame.

    ``above`` paints ``pixels`` of ``below``'s shape; parts of different
    placement groups are never paired, and ``out_of_view`` parts not at all.
    """
    found: dict[tuple[str, str], int] = {}
    for parts in (_groups(compiled) if groups is None else groups):
        for above in parts:
            if above.vis_window is None:
                continue
            for below in parts:
                if below is above:
                    continue
                w = _meet(above.vis_window, below.shape_window)
                if w is None:
                    continue
                x0, y0, x1, y1 = w
                n = int(np.count_nonzero(above.visible[y0:y1, x0:x1]
                                         & below.shape[y0:y1, x0:x1]))
                if n > 0:
                    found[(above.instance, below.instance)] = n
    return found


def occlusion_order(compiled: Optional[CompiledFrame],
                    drawn: Iterable[str] = ()) -> list[OrderLine]:
    """The frame's parts top to bottom in painting order, with what each takes.

    ``drawn``: the instances with a per-frame mask on this frame (the frame
    overrides with pixels; the compiled row does not tell them apart).
    """
    if compiled is None:
        return []
    drawn = set(drawn)
    manual = {tuple(pair) for pairs in (compiled.pairs or {}).values() for pair in pairs}
    groups = _groups(compiled)
    taken: dict[str, list[Taken]] = {}
    for (above, below), n in overlap_pairs(compiled, groups).items():
        taken.setdefault(above, []).append(Taken(below, n, (above, below) in manual))
    lines = []
    for part in reversed([part for parts in groups for part in parts]):
        name = part.instance
        takes = sorted(taken.get(name, ()), key=lambda t: (-t.pixels, t.below))
        # no visible pixel inside its window is no visible pixel at all
        lines.append(OrderLine(name, tuple(takes), name in drawn, part.vis_window is None))
    return lines


def _px(n: int) -> str:
    """``12 345 px``: thousands set apart the way the spec writes them."""
    return f"{int(n):,} px".replace(",", " ")


def line_text(line: OrderLine, label: Optional[Callable[[str], str]] = None) -> str:
    """「导风罩 cover.01　压着 SSD ssd.01 (1 234 px)（手动）、…」."""
    name = label or instance_label
    text = name(line.instance) + (OWN_FRAME if line.own_frame else "")
    if line.takes:
        text += "　压着 " + "、".join(
            f"{name(t.below)} ({_px(t.pixels)}){MANUAL if t.manual else ''}"
            for t in line.takes)
    if line.covered:
        text += "　" + COVERED
    return text


def header_text(lines: list[OrderLine]) -> str:
    """「这一帧 n 个零件，m 对重叠」, or 「这一帧没有编译出的零件」."""
    if not lines:
        return EMPTY
    pairs = {frozenset((line.instance, t.below)) for line in lines for t in line.takes}
    return f"这一帧 {len(lines)} 个零件，{len(pairs)} 对重叠"


def labeller(records: Mapping[str, object]) -> Callable[[str], str]:
    """``instance -> 「<名称> <key>」`` from the instance records (``cls``, ``attrs``)."""
    def name(instance: str) -> str:
        rec = records.get(instance)
        return instance_label(instance, getattr(rec, "cls", None), getattr(rec, "attrs", None))
    return name
