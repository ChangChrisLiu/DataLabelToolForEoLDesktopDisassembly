"""「这一步的变化」 in the 「主视角」 dock: the card's parts on the whole scanner frame (U43).

:class:`ChangeView` draws a :class:`~tda.ui.locator_change.ChangeScene`: the
scanner frame, every painted part filled in its canvas colour at the canvas's
fill alpha with a thin outline of the same colour, nothing written on it.

* **The mouse** names the part under it: :attr:`ChangeView.sigHover` carries
  its 「<名称> <key>（<状态>）」 (``""`` off any part) and the panel writes it in
  the caption's second line.  No tooltip, no selection.
* **A click** switches between the whole frame and a crop around the painted
  parts (never below U23's tenth of the long side); **the wheel** zooms about
  the point under the mouse, in either.  The view never takes the keyboard.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QSizePolicy, QWidget

from tda.ui.locator_change import ChangePart, ChangeScene
from tda.ui.panels.locator import (
    BACKGROUND,
    CROP,
    MIN_CROP_PX,
    WHOLE,
    ZOOM_STEP,
    crop_rect,
)

__all__ = ["CHANGE_FILL", "ChangeView", "FILL_ALPHA", "MIN_ZOOM"]

#: The canvas's own default fill alpha (``CanvasView.overlay_alpha``); the
#: window hands the live value over (:attr:`ChangeView.alpha`).
FILL_ALPHA = 110
#: The painted parts span this much of the crop: they are several, the
#: frame around them is context enough.
CHANGE_FILL = 0.8
#: The wheel zooms in down to this share of the starting window.
MIN_ZOOM = 0.05
LINE_PX = 1.2
HOVER_PX = 2.6


class ChangeView(QWidget):
    """The picture of 「这一步的变化」: whole frame or crop, the parts painted."""

    #: The part under the mouse, as the caption says it; ``""`` off any part.
    sigHover = Signal(str)
    #: The mode after a click (:data:`CROP` / :data:`WHOLE`).
    sigModeChanged = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._change: Optional[ChangeScene] = None
        self._pixmap = QPixmap()
        self._overlay = QPixmap()
        self._paths: list[QPainterPath] = []
        self._mode = WHOLE
        #: The window the wheel left, in frame pixels; ``None``: the mode's own.
        self._window: Optional[tuple[float, float, float, float]] = None
        self._hover: Optional[str] = None
        #: The fill's alpha, 0-255: the canvas's (the window keeps it in step).
        self.alpha = FILL_ALPHA
        self.paints = 0
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(120, 80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:  # noqa: D102 - Qt override
        return QSize(300, 210)

    # -- state ----------------------------------------------------------------
    def set_change(self, change: Optional[ChangeScene], *, reset_zoom: bool = False) -> None:
        """Show ``change``; pictures and outlines are converted once per scene."""
        if change is not self._change:
            old = self._change
            if old is None or change is None or change.image is not old.image:
                self._pixmap = (QPixmap.fromImage(change.image)
                                if change is not None and change.image is not None
                                else QPixmap())
            self._overlay = (QPixmap.fromImage(change.overlay)
                             if change is not None and change.overlay is not None
                             else QPixmap())
            self._paths = [] if change is None else [_path(p) for p in change.parts]
            self._change = change
            if self._hover is not None:
                self._hover = None
                self.sigHover.emit("")
        if reset_zoom:
            self._window = None
        self.update()

    def change(self) -> Optional[ChangeScene]:
        return self._change

    def mode(self) -> str:
        """:data:`WHOLE` (the default) or :data:`CROP`."""
        return self._mode

    def showing(self) -> str:
        """What is on screen: the crop only when there is something painted to crop to."""
        return CROP if self._mode == CROP and self._has_box() else WHOLE

    def toggle_mode(self) -> str:
        """A click: whole frame <-> crop.  Nothing painted, nothing changes."""
        if not self._has_box():
            return self._mode
        self._mode = WHOLE if self._mode == CROP else CROP
        self._window = None
        self.update()
        self.sigModeChanged.emit(self._mode)
        return self._mode

    def _has_box(self) -> bool:
        change = self._change
        return change is not None and change.box is not None and change.frame_hw[1] > 0

    def _aspect(self) -> float:
        return max(1, self.width()) / float(max(1, self.height()))

    def base_rect(self) -> Optional[tuple[float, float, float, float]]:
        """The mode's own window, before the wheel; ``None`` with no frame."""
        change = self._change
        if change is None or change.frame_hw[1] <= 0 or change.frame_hw[0] <= 0:
            return None
        h_img, w_img = change.frame_hw
        if self.showing() == CROP:
            return crop_rect(change.box, change.frame_hw, self._aspect(), 1.0,
                             fill=CHANGE_FILL)
        return (0.0, 0.0, float(w_img), float(h_img))

    def source_rect(self) -> Optional[tuple[float, float, float, float]]:
        """The part of the frame on screen, in frame pixels."""
        base = self.base_rect()
        if base is None:
            return None
        return self._window if self._window is not None else base

    def zoom(self) -> float:
        """The window's width over the mode's own: below 1 is zoomed in."""
        base, src = self.base_rect(), self.source_rect()
        if base is None or src is None:
            return 1.0
        return (src[2] - src[0]) / max(1e-6, base[2] - base[0])

    def zoom_by(self, notches: float, at: Optional[QPointF] = None) -> float:
        """The wheel: positive notches zoom in, about the frame point under ``at``."""
        src = self.source_rect()
        base = self.base_rect()
        if src is None or base is None or not notches:
            return self.zoom()
        h_img, w_img = self._change.frame_hw
        factor = ZOOM_STEP ** (-float(notches))
        old_w, old_h = src[2] - src[0], src[3] - src[1]
        floor = max(MIN_CROP_PX, MIN_ZOOM * (base[2] - base[0]))
        w = min(max(old_w * factor, floor), float(w_img))
        h = min(max(old_h * factor, floor * old_h / max(1e-6, old_w)), float(h_img))
        anchor = self.frame_point(at) if at is not None else None
        ax, ay = anchor if anchor is not None else ((src[0] + src[2]) / 2.0,
                                                    (src[1] + src[3]) / 2.0)
        left = ax - (ax - src[0]) * (w / max(1e-6, old_w))
        top = ay - (ay - src[1]) * (h / max(1e-6, old_h))
        left = min(max(left, 0.0), max(0.0, w_img - w))
        top = min(max(top, 0.0), max(0.0, h_img - h))
        self._window = (left, top, left + w, top + h)
        self.update()
        return self.zoom()

    def _target(self, src) -> QRectF:
        """Where ``src`` lands in the widget: fitted, centred, its shape kept."""
        sw, sh = max(1e-6, src[2] - src[0]), max(1e-6, src[3] - src[1])
        k = min(self.width() / sw, self.height() / sh)
        w, h = sw * k, sh * k
        return QRectF((self.width() - w) / 2.0, (self.height() - h) / 2.0, w, h)

    def frame_point(self, pos: QPointF) -> Optional[tuple[float, float]]:
        """The frame pixel under widget point ``pos``, ``None`` off the picture."""
        src = self.source_rect()
        if src is None:
            return None
        target = self._target(src)
        if not target.contains(QPointF(pos)):
            return None
        kx = (src[2] - src[0]) / max(1e-6, target.width())
        ky = (src[3] - src[1]) / max(1e-6, target.height())
        return (src[0] + (pos.x() - target.x()) * kx, src[1] + (pos.y() - target.y()) * ky)

    def part_at(self, pos: QPointF) -> Optional[ChangePart]:
        """The painted part under widget point ``pos``, or ``None``."""
        point = self.frame_point(pos)
        if point is None or self._change is None:
            return None
        return self._change.part_at(*point)

    def hover_at(self, pos: Optional[QPointF]) -> str:
        """Say which part is under ``pos`` (``None``: the mouse left); the caption's line."""
        part = None if pos is None else self.part_at(pos)
        name = None if part is None else part.instance
        if name != self._hover:
            self._hover = name
            self.sigHover.emit("" if part is None else part.text)
            self.update()
        return "" if part is None else part.text

    def hovered(self) -> Optional[str]:
        """The instance under the mouse, or ``None``."""
        return self._hover

    def hover_text(self) -> str:
        """What the caption says for the part under the mouse, ``""`` for none."""
        change = self._change
        if change is None or self._hover is None:
            return ""
        return next((p.text for p in change.parts if p.instance == self._hover), "")

    # -- mouse ------------------------------------------------------------------
    def mouseMoveEvent(self, event) -> None:  # noqa: D102 - Qt override
        self.hover_at(event.position())
        event.accept()

    def leaveEvent(self, event) -> None:  # noqa: D102 - Qt override
        self.hover_at(None)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: D102 - Qt override
        event.accept()       # never a click on whatever is underneath

    def mouseReleaseEvent(self, event) -> None:  # noqa: D102 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_mode()
            # the picture moved under the mouse: name what is there now,
            # not at the next move (round 2, N2)
            self.hover_at(event.position())
        event.accept()

    def wheelEvent(self, event) -> None:  # noqa: D102 - Qt override
        notches = event.angleDelta().y() / 120.0
        if notches:
            self.zoom_by(notches, event.position())
        event.accept()

    # -- paint ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: D102 - Qt override
        self.paints += 1
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        change = self._change
        src = self.source_rect()
        if change is None or src is None:
            painter.end()
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        target = self._target(src)
        h_img, w_img = change.frame_hw
        for pixmap, opacity in ((self._pixmap, 1.0), (self._overlay, self.alpha / 255.0)):
            if pixmap.isNull():
                continue
            sx = pixmap.width() / float(w_img)
            sy = pixmap.height() / float(h_img)
            painter.setOpacity(max(0.0, min(1.0, opacity)))
            painter.drawPixmap(target, pixmap,
                               QRectF(src[0] * sx, src[1] * sy,
                                      (src[2] - src[0]) * sx, (src[3] - src[1]) * sy))
        painter.setOpacity(1.0)
        kx = target.width() / max(1e-6, src[2] - src[0])
        ky = target.height() / max(1e-6, src[3] - src[1])
        painter.setClipRect(target)
        painter.setTransform(QTransform(kx, 0.0, 0.0, ky, target.x() - src[0] * kx,
                                        target.y() - src[1] * ky))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for part, path in zip(change.parts, self._paths):
            pen = QPen(QColor(*part.colour),
                       HOVER_PX if part.instance == self._hover else LINE_PX)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawPath(path)
        painter.end()


def _path(part: ChangePart) -> QPainterPath:
    """The part's outline in frame pixels (the painter maps it to the widget)."""
    path = QPainterPath()
    for ring in part.geometry.outlines:
        if len(ring) < 2:
            continue
        path.moveTo(float(ring[0][0]), float(ring[0][1]))
        for x, y in ring[1:]:
            path.lineTo(float(x), float(y))
        path.closeSubpath()
    return path
