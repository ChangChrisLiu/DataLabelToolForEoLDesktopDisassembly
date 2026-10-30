"""「这一步的变化」: this frame's task card painted on the scanner frame (task U43).

While annotating an OAK / RealSense frame the annotator wants the scanner
view's annotation of the same step beside it, as a reference -- not to check
it, just to see which parts the step is about and where they are.  The user,
2026-10-30: 「变化的部分，给一个大图，然后把变化的都标出来」, and on a fuller
design: 「我只是需要一个主视角的标注作为 reference……reference 我可以鼠标过去然后放大什么的」.

So the 「主视角」 dock (task U23) gets a third kind of scene, built here as a
:class:`ChangeScene`: the *whole* scanner frame of the same step (chosen as
U23 chooses it, :meth:`~tda.ui.locator_source.LocatorSource.scan_step_for`),
with every part of this frame's task card -- the parts that change here,
cascades and state-only changes included -- painted in its canvas colour from
its compiled scanner *visible* mask (the locator's
:meth:`~tda.ui.locator_source.LocatorSource.geometry`: the stored row, or the
frame compiled once).  A part with no scanner geometry on that frame is not
painted.  No labels on the picture: the mouse names a part
(:meth:`ChangeScene.part_at`, a lookup in a small label image at the
thumbnail's size, each part's id painted into it once), and the panel
(:mod:`tda.ui.panels.locator_change`) draws it.

Costs: the picture and the outlines are the locator's caches; a scene is
remembered by ``(machine, step, the card's parts, the scanner digest)`` so
walking back to a frame, or a selection change on it, builds nothing.
Nothing here writes to the database.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import cv2
import numpy as np
from PySide6.QtGui import QImage

from tda.core.model import is_context_key, is_provisional
from tda.ui.canvas.overlay import palette_color
from tda.ui.locator_source import (
    NO_SCAN_FRAME,
    NOT_ANNOTATED,
    UNREADABLE,
    Geometry,
    caption_for,
)

__all__ = ["CHANGE_CACHE", "CHANGE_NAME", "COUNT_LINE", "ChangeMaker", "ChangePart",
           "ChangeScene", "EMPTY_CARD", "card_parts"]

#: The scene's name: the caption's tail and the dock's toggle.
CHANGE_NAME = "这一步的变化"
#: The caption's second line, off any painted part.
COUNT_LINE = "卡片上 {n} 个零件，扫描仪上画了 {m} 个"
#: ... and on a frame whose card names no part (a dupli frame's one row).
EMPTY_CARD = "这一帧的卡片上没有零件"
#: Scenes remembered (a few frames walked back and forth).
CHANGE_CACHE = 8


@dataclass(frozen=True)
class ChangePart:
    """One painted part: what it is called, its colour and where it is."""

    instance: str
    label: str
    state: str
    colour: tuple[int, int, int]
    geometry: Geometry

    @property
    def text(self) -> str:
        """「<名称> <key>（<状态>）」, what the caption says under the mouse."""
        return f"{self.label}（{self.state}）" if self.state else self.label


@dataclass
class ChangeScene:
    """Everything the panel draws for 「这一步的变化」 on one step."""

    desktop: Optional[int] = None
    asked_step: Optional[int] = None
    scan_step: Optional[int] = None
    #: The scanner frame at the locator's thumbnail side, ``None`` when there is none.
    image: Optional[QImage] = None
    #: The scanner frame's own size, ``(h, w)``: what the geometry is in.
    frame_hw: tuple[int, int] = (0, 0)
    #: The card's parts, in card order (a part named twice once).
    card: tuple[str, ...] = ()
    #: The ones painted: a scanner geometry on that frame.
    parts: tuple[ChangePart, ...] = ()
    #: ``parts`` index + 1 per pixel, at ``labels.shape`` (0: no part).
    labels: Optional[np.ndarray] = None
    #: The parts' colours at full alpha, ``labels``' size; the panel fades it.
    overlay: Optional[QImage] = None
    #: The union bounding box of the painted parts, in frame pixels.
    box: Optional[tuple[float, float, float, float]] = None
    #: 「主视角 第 k 帧 · 这一步的变化」 (with U23's earlier-frame clause).
    caption: str = ""
    #: The sentence under the picture (U23's reasons), or ``""``.
    note: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def line(self) -> str:
        """The caption's second line off any part: the counts, or the empty card."""
        if not self.card:
            return EMPTY_CARD
        return COUNT_LINE.format(n=len(self.card), m=len(self.parts))

    def part_at(self, x: float, y: float) -> Optional[ChangePart]:
        """The painted part at frame pixel ``(x, y)``, or ``None``."""
        labels = self.labels
        h_img, w_img = self.frame_hw
        if labels is None or h_img <= 0 or w_img <= 0:
            return None
        lh, lw = labels.shape[:2]
        col = int(float(x) * lw / float(w_img))
        row = int(float(y) * lh / float(h_img))
        if not (0 <= col < lw and 0 <= row < lh):
            return None
        index = int(labels[row, col])
        return self.parts[index - 1] if 0 < index <= len(self.parts) else None


def card_parts(instances: Iterable, is_part) -> tuple[str, ...]:
    """The card's rows that are parts of the machine, in order, each once.

    A dupli frame's row names a raw step, a 框外零件 or a Label Studio draft
    key is no part (U28's rule); ``is_part`` answers for the machine.
    """
    seen: dict[str, None] = {}
    for name in instances:
        if not name:
            continue
        name = str(name)
        if name in seen or is_context_key(name) or is_provisional(name):
            continue
        if is_part(name):
            seen[name] = None
    return tuple(seen)


def _label_size(image: Optional[QImage], frame_hw: tuple[int, int],
                side: int) -> tuple[int, int]:
    """``(h, w)`` of the label image: the thumbnail's, else the frame's scaled to ``side``."""
    if image is not None and not image.isNull():
        return image.height(), image.width()
    h_img, w_img = frame_hw
    if h_img <= 0 or w_img <= 0:
        return 0, 0
    k = min(1.0, float(side) / float(max(h_img, w_img)))
    return max(1, int(round(h_img * k))), max(1, int(round(w_img * k)))


def paint_labels(parts: tuple[ChangePart, ...], frame_hw: tuple[int, int],
                 size: tuple[int, int]) -> tuple[np.ndarray, QImage]:
    """``(label image, colour overlay)`` at ``size``: each part's id painted once.

    The larger parts go first, so a small part lying on a large one's outer
    outline (a fan in a bracket's ring) stays on top, as it is on the frame.
    """
    lh, lw = size
    h_img, w_img = frame_hw
    labels = np.zeros((lh, lw), dtype=np.uint16)
    sx, sy = lw / float(w_img), lh / float(h_img)

    def area(item) -> float:
        box = item[1].geometry.box
        return (box[2] - box[0]) * (box[3] - box[1])

    for index, part in sorted(enumerate(parts, start=1), key=area, reverse=True):
        rings = []
        for ring in part.geometry.outlines:
            if len(ring) < 3:
                continue
            pts = np.empty((len(ring), 2), dtype=np.int32)
            pts[:, 0] = np.floor(np.asarray(ring[:, 0], dtype=float) * sx)
            pts[:, 1] = np.floor(np.asarray(ring[:, 1], dtype=float) * sy)
            rings.append(pts)
        if rings:
            cv2.fillPoly(labels, rings, int(index))
    lut = np.zeros(len(parts) + 1, dtype=np.uint32)
    for index, part in enumerate(parts, start=1):
        r, g, b = (int(c) & 0xFF for c in part.colour)
        lut[index] = np.uint32((0xFF << 24) | (r << 16) | (g << 8) | b)
    argb = np.ascontiguousarray(lut[labels])
    overlay = QImage(argb.data, lw, lh, 4 * lw, QImage.Format.Format_ARGB32).copy()
    return labels, overlay


class ChangeMaker:
    """Builds and remembers :class:`ChangeScene` for a :class:`LocatorSource`."""

    def __init__(self, source) -> None:
        self.source = source
        self._scenes: "OrderedDict[tuple, ChangeScene]" = OrderedDict()

    def change(self, desktop: int, step: int, instances: Iterable) -> ChangeScene:
        """The scene for ``step`` and the card's ``instances`` (facts pinned by the caller)."""
        source = self.source
        desktop, step = int(desktop), int(step)
        facts = source._desktop_facts(desktop)
        card = card_parts(instances, lambda name: source.is_instance(desktop, name))
        key = (desktop, step, card, facts["digest"])
        hit = self._scenes.get(key)
        if hit is not None:
            self._scenes.move_to_end(key)
            return hit
        scene = self._build(desktop, step, card, facts)
        source.stats["changes"] = source.stats.get("changes", 0) + 1
        self._scenes[key] = scene
        while len(self._scenes) > CHANGE_CACHE:
            self._scenes.popitem(last=False)
        return scene

    def _build(self, desktop: int, step: int, card: tuple[str, ...], facts: dict) -> ChangeScene:
        source = self.source
        scene = ChangeScene(desktop=desktop, asked_step=step, card=card)
        scan_step, why = source.scan_step_for(desktop, step)
        if scan_step is None:
            scene.caption = f"主视角 · {CHANGE_NAME}"
            scene.note = NO_SCAN_FRAME.format(asked=step)
            return scene
        scene.scan_step = scan_step
        scene.caption = f"{caption_for(scan_step, step, why, '', '')} · {CHANGE_NAME}"
        scene.image, scene.frame_hw = source.thumbnail(desktop, scan_step)
        if scene.frame_hw == (0, 0):
            # no picture: the size the frame row recorded, so the parts still
            # land where they are (U23's fallback)
            row = facts["frames"].get(scan_step) or {}
            hw = (row.get("aux") or {}).get("hw")
            if hw and len(hw) == 2:
                scene.frame_hw = (int(hw[0]), int(hw[1]))
        if not facts["annotated"]:
            scene.note = NOT_ANNOTATED
        else:
            found = []
            for instance in card:
                geometry, _why, _src = source.geometry(desktop, scan_step, instance)
                if geometry is not None:
                    found.append((instance, geometry))
            if found:
                names = source.labels_and_states(desktop, scan_step,
                                                 [name for name, _ in found])
                scene.parts = tuple(
                    ChangePart(name, names[name][0], names[name][1],
                               tuple(int(c) for c in palette_color(name)), geometry)
                    for name, geometry in found)
                boxes = np.array([p.geometry.box for p in scene.parts], dtype=float)
                scene.box = (float(boxes[:, 0].min()), float(boxes[:, 1].min()),
                             float(boxes[:, 2].max()), float(boxes[:, 3].max()))
        if scene.image is None:
            unreadable = UNREADABLE.format(step=scan_step)
            scene.note = f"{unreadable}；{scene.note}" if scene.note else unreadable
        size = _label_size(scene.image, scene.frame_hw, source.thumb_side)
        if scene.parts and size[0] > 0:
            scene.labels, scene.overlay = paint_labels(scene.parts, scene.frame_hw, size)
        return scene
