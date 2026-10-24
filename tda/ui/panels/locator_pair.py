"""「拆前 / 拆后」: the two scanner frames around the step a part leaves at (task U28).

:class:`PairView` draws a :class:`~tda.ui.locator_pair.PairScene` -- the frame
before and the frame after, side by side, each under its caption
(「拆前 第 7 帧」 / 「拆后 第 8 帧」), both cropped to the **same** window around
what changed (padded 1.6 times, at least a sixth of the frame, in the shape of
the panel's cells, :func:`~tda.ui.panels.locator.crop_rect`).  The window is
drawn on both, and the part's own outline on the "before" frame when the
window is that outline.  :class:`InsetView` is the small whole frame that
shows where the crop sits.

As on the location scene (:class:`~tda.ui.panels.locator.LocatorView`): a
click shows both frames whole and back, the wheel zooms both crops together,
and nothing here takes the keyboard.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from tda.ui.locator_pair import MIN_SIDE_FRAC, PAD, PairScene
from tda.ui.panels.locator import (
    BACKGROUND,
    CROP,
    MIN_ZOOM,
    ROI_PEN,
    WHOLE,
    ZOOM_STEP,
    crop_rect,
)

__all__ = ["CELL_GAP", "InsetView", "PAIR_HINT", "PAIR_TIP", "PairView", "TOGGLE_TEXT"]

#: The toggle next to the dock's caption, and what it says it does.
TOGGLE_TEXT = "拆前/拆后"
PAIR_TIP = ("拆前/拆后：这个零件被拆走（或第一次改变状态）前后两步的扫描仪画面。"
            "第一次标某个零件时自动显示；点一下切换。"
            " / the scanner frames before and after this part leaves")
PAIR_HINT = ("拆走前后两帧，裁到同一处。单击：裁剪 ↔ 整帧；滚轮：放大 / 缩小。"
             "before and after the part leaves; click: crop or whole; wheel: zoom")
#: Between the two pictures.
CELL_GAP = 4.0
NO_PICTURE = "（没有图）"
CAPTION_PEN = QColor(230, 230, 230)


def _fit(src, cell: QRectF) -> QRectF:
    """Where ``src`` (frame pixels) lands in ``cell``: fitted, centred, shape kept."""
    sw, sh = max(1e-6, src[2] - src[0]), max(1e-6, src[3] - src[1])
    k = min(cell.width() / sw, cell.height() / sh)
    w, h = sw * k, sh * k
    return QRectF(cell.x() + (cell.width() - w) / 2.0, cell.y() + (cell.height() - h) / 2.0,
                  w, h)


class PairView(QWidget):
    """Two pictures, one window: what the part looked like before and after it left."""

    #: The mode or the zoom changed (the inset follows).
    sigChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pair: Optional[PairScene] = None
        self._pixmaps: list[QPixmap] = [QPixmap(), QPixmap()]
        self._mode = CROP
        self._zoom = 1.0
        self.paints = 0
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(PAIR_HINT)
        self.setMinimumSize(160, 90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:  # noqa: D102 - Qt override
        return QSize(300, 170)

    # -- state ----------------------------------------------------------------
    def set_pair(self, pair: Optional[PairScene], *, reset_zoom: bool = False) -> None:
        """Show ``pair``; each picture is converted once here, not per paint."""
        before = self._pair
        for i in (0, 1):
            new = None if pair is None else pair.images[i]
            old = None if before is None else before.images[i]
            if new is not old:
                self._pixmaps[i] = QPixmap.fromImage(new) if new is not None else QPixmap()
        self._pair = pair
        if reset_zoom:
            self._zoom = 1.0
        self.update()
        self.sigChanged.emit()

    def pair(self) -> Optional[PairScene]:
        return self._pair

    def pixmap(self, i: int) -> QPixmap:
        return self._pixmaps[int(i)]

    def mode(self) -> str:
        return self._mode

    def showing(self) -> str:
        """:data:`CROP` only when there is a window to crop to."""
        return CROP if self._mode == CROP and self._has_window() else WHOLE

    def set_mode(self, mode: str) -> None:
        self._mode = CROP if mode == CROP else WHOLE
        self.update()
        self.sigChanged.emit()

    def toggle_mode(self) -> str:
        """A click: both crops <-> both whole frames.  No window, nothing changes."""
        if not self._has_window():
            return self._mode
        self.set_mode(WHOLE if self._mode == CROP else CROP)
        return self._mode

    def zoom(self) -> float:
        return self._zoom

    def zoom_by(self, notches: float) -> float:
        """The wheel: positive notches zoom in (both crops smaller)."""
        if self.showing() != CROP:
            return self._zoom
        factor = ZOOM_STEP ** (-float(notches))
        self._zoom = min(max(self._zoom * factor, MIN_ZOOM), self._max_zoom())
        self.update()
        self.sigChanged.emit()
        return self._zoom

    def _has_window(self) -> bool:
        pair = self._pair
        return (pair is not None and pair.window is not None
                and pair.frame_hw[0][1] > 0)

    def _crop(self, zoom: float) -> tuple[float, float, float, float]:
        pair = self._pair
        return crop_rect(pair.window, pair.frame_hw[0], self._aspect(), zoom,
                         fill=1.0 / PAD, min_frac=MIN_SIDE_FRAC)

    def _max_zoom(self) -> float:
        if not self._has_window():
            return 1.0
        base = self._crop(1.0)
        h_img, w_img = self._pair.frame_hw[0]
        return max(1.0, w_img / max(1.0, base[2] - base[0]),
                   h_img / max(1.0, base[3] - base[1]))

    # -- layout -----------------------------------------------------------------
    def caption_height(self) -> float:
        return float(self.fontMetrics().height() + 2)

    def cells(self) -> tuple[QRectF, QRectF]:
        """The two picture areas, under their captions."""
        top = self.caption_height()
        width = max(1.0, (self.width() - CELL_GAP) / 2.0)
        height = max(1.0, self.height() - top)
        return (QRectF(0.0, top, width, height),
                QRectF(width + CELL_GAP, top, width, height))

    def _aspect(self) -> float:
        cell = self.cells()[0]
        return max(1.0, cell.width()) / max(1.0, cell.height())

    def _scale(self, i: int) -> tuple[float, float]:
        """Frame ``i``'s pixels per pixel of the before frame (both are one camera)."""
        pair = self._pair
        (h0, w0), (h1, w1) = pair.frame_hw
        if i == 0 or not h0 or not w0 or not h1 or not w1:
            return 1.0, 1.0
        return w1 / float(w0), h1 / float(h0)

    def window_in(self, i: int) -> Optional[tuple[float, float, float, float]]:
        """The window in frame ``i``'s pixels, ``None`` with no window."""
        pair = self._pair
        if pair is None or pair.window is None:
            return None
        sx, sy = self._scale(i)
        x0, y0, x1, y1 = pair.window
        return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)

    def source_rect(self, i: int) -> Optional[tuple[float, float, float, float]]:
        """The part of frame ``i`` on screen, in its pixels; ``None`` with no frame."""
        pair = self._pair
        if pair is None:
            return None
        h_img, w_img = pair.frame_hw[i]
        if w_img <= 0 or h_img <= 0:
            return None
        if self.showing() == CROP:
            sx, sy = self._scale(i)
            x0, y0, x1, y1 = self._crop(self._zoom)
            return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
        return (0.0, 0.0, float(w_img), float(h_img))

    # -- mouse ------------------------------------------------------------------
    def mouseReleaseEvent(self, event) -> None:  # noqa: D102 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_mode()
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: D102 - Qt override
        event.accept()

    def wheelEvent(self, event) -> None:  # noqa: D102 - Qt override
        notches = event.angleDelta().y() / 120.0
        if notches:
            self.zoom_by(notches)
        event.accept()

    # -- paint ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: D102 - Qt override
        self.paints += 1
        painter = QPainter(self)
        pair = self._pair
        if pair is None:
            painter.end()
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        metrics = self.fontMetrics()
        for i, cell in enumerate(self.cells()):
            caption = metrics.elidedText(pair.captions[i], Qt.TextElideMode.ElideRight,
                                         int(cell.width()))
            painter.setClipping(False)
            painter.setPen(self.palette().windowText().color())
            painter.drawText(QRectF(cell.x(), 0.0, cell.width(), cell.y()),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             caption)
            painter.fillRect(cell, BACKGROUND)
            self._paint_cell(painter, i, cell)
        painter.end()

    def _paint_cell(self, painter: QPainter, i: int, cell: QRectF) -> None:
        pair = self._pair
        src = self.source_rect(i)
        pixmap = self._pixmaps[i]
        if src is None or pixmap.isNull():
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(cell, int(Qt.AlignmentFlag.AlignCenter), NO_PICTURE)
            if src is None:
                return
        target = _fit(src, cell)
        h_img, w_img = pair.frame_hw[i]
        if not pixmap.isNull():
            sx, sy = pixmap.width() / float(w_img), pixmap.height() / float(h_img)
            painter.drawPixmap(target, pixmap,
                               QRectF(src[0] * sx, src[1] * sy,
                                      (src[2] - src[0]) * sx, (src[3] - src[1]) * sy))
        kx = target.width() / max(1e-6, src[2] - src[0])
        ky = target.height() / max(1e-6, src[3] - src[1])

        def at(x: float, y: float) -> QPointF:
            return QPointF(target.x() + (x - src[0]) * kx, target.y() + (y - src[1]) * ky)

        painter.setClipRect(target)
        sx, sy = self._scale(i)
        if self.showing() == WHOLE and pair.roi is not None:
            painter.setPen(QPen(ROI_PEN, 1.0, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            r = pair.roi
            painter.drawRect(QRectF(at(r[0] * sx, r[1] * sy), at(r[2] * sx, r[3] * sy)))
        colour = QColor(*pair.colour)
        geometry = pair.geometry
        if i == 0 and geometry is not None:
            path = QPainterPath()
            for ring in geometry.outlines:
                if len(ring) < 2:
                    continue
                path.moveTo(at(float(ring[0][0]), float(ring[0][1])))
                for x, y in ring[1:]:
                    path.lineTo(at(float(x), float(y)))
                path.closeSubpath()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 170), 3.0))
            painter.drawPath(path)
            painter.setPen(QPen(colour, 1.5))
            painter.drawPath(path)
        window = self.window_in(i)
        if window is not None:
            rect = QRectF(at(window[0], window[1]), at(window[2], window[3])).adjusted(
                -2, -2, 2, 2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 160), 3.0))
            painter.drawRect(rect)
            painter.setPen(QPen(colour, 1.5, Qt.PenStyle.DashLine))
            painter.drawRect(rect)
        painter.setClipping(False)


class InsetView(QWidget):
    """The before frame, small and whole: where the two crops are cut from."""

    SIDE = 84

    def __init__(self, view: PairView, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._view = view
        self.paints = 0
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(self.SIDE, self.SIDE * 3 // 4)
        self.setToolTip("整帧：白框是上面裁出的部分 / the whole frame: the white box is the crop")
        view.sigChanged.connect(self.update)

    def paintEvent(self, event) -> None:  # noqa: D102 - Qt override
        self.paints += 1
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        view = self._view
        pair = view.pair()
        which = 0 if pair is not None and not view.pixmap(0).isNull() else 1
        if pair is None or pair.frame_hw[which][1] <= 0:
            painter.end()
            return
        h_img, w_img = pair.frame_hw[which]
        whole = (0.0, 0.0, float(w_img), float(h_img))
        target = _fit(whole, QRectF(self.rect()))
        pixmap = view.pixmap(which)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        if not pixmap.isNull():
            painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
        k = target.width() / float(w_img)

        def box(rect) -> QRectF:
            return QRectF(target.x() + rect[0] * k, target.y() + rect[1] * k,
                          max(1.0, (rect[2] - rect[0]) * k), max(1.0, (rect[3] - rect[1]) * k))

        crop = view.source_rect(which)
        if crop is not None and view.showing() == CROP:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 200), 3.0))     # seen on a white table too
            painter.drawRect(box(crop))
            painter.setPen(QPen(QColor(255, 255, 255), 1.0))
            painter.drawRect(box(crop))
        window = view.window_in(which)
        if window is not None:
            rect = box(window)
            if rect.width() < 4 or rect.height() < 4:     # a screw: still a mark
                centre = rect.center()
                rect = QRectF(centre.x() - 2, centre.y() - 2, 4, 4)
            painter.setPen(QPen(QColor(*pair.colour), 1.5))
            painter.setBrush(QBrush(QColor(*pair.colour, 90)))
            painter.drawRect(rect)
        painter.end()
