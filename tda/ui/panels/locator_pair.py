"""「拆前 / 拆后」: the two scanner frames around the step a part leaves at (task U28).

:class:`PairView` draws a :class:`~tda.ui.locator_pair.PairScene` -- the frame
before and the frame after, side by side, each under its caption
(「拆前 第 7 帧」 / 「拆后 第 8 帧」), both cropped to the **same** window around
what changed (padded 1.6 times, at least a sixth of the frame, in the shape of
the panel's cells, :func:`~tda.ui.panels.locator.crop_rect`).  The window is
drawn on both.  Two kinds of window, drawn apart (round 2, B2):

* the part's own scanner outline -- the outline in the part's colour on the
  "before" frame and a dashed box in that colour on both;
* a guess from the difference of the two pictures -- a grey dashed box, no
  part colour, no outline, and the panel's note says it is a guess.  Up to
  three such candidates: :class:`CandidateBar` shows the ones not on screen
  as small crops of the "before" frame, a click puts one on screen, and it
  says which one is up (「候选 1/3」).

:class:`InsetView` is the small whole frame that shows where the crop sits.
As on the location scene (:class:`~tda.ui.panels.locator.LocatorView`): a
click shows both frames whole and back, the wheel zooms both crops together,
and nothing here takes the keyboard.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from tda.ui.locator_pair import CANDIDATE, MIN_SIDE_FRAC, PAD, PairScene
from tda.ui.panels.locator import (
    BACKGROUND,
    CROP,
    MIN_ZOOM,
    ROI_PEN,
    WHOLE,
    ZOOM_STEP,
    crop_rect,
)

__all__ = ["CELL_GAP", "CandidateBar", "GUESS_COLOUR", "InsetView", "PAIR_HINT", "PAIR_TIP",
           "PairView", "TOGGLE_TEXT"]

#: The toggle next to the dock's caption, and what it says it does.
TOGGLE_TEXT = "拆前/拆后"
PAIR_TIP = ("拆前/拆后：这个零件被拆走（或第一次改变状态）前后两步的扫描仪画面。"
            "第一次标某个零件时自动显示；点一下切换。"
            " / the scanner frames before and after this part leaves")
PAIR_HINT = ("拆走前后两帧，裁到同一处。单击：裁剪 ↔ 整帧；滚轮：放大 / 缩小。"
             "before and after the part leaves; click: crop or whole; wheel: zoom")
CHIP_TIP = ("另一处变化：点一下换成它 / another place that changed: click to show it")
#: Between the two pictures.
CELL_GAP = 4.0
NO_PICTURE = "（没有图）"
CAPTION_PEN = QColor(230, 230, 230)
#: A window guessed from the difference: grey, never the part's colour.
GUESS_COLOUR = QColor(200, 200, 200)


def _fit(src, cell: QRectF) -> QRectF:
    """Where ``src`` (frame pixels) lands in ``cell``: fitted, centred, shape kept."""
    sw, sh = max(1e-6, src[2] - src[0]), max(1e-6, src[3] - src[1])
    k = min(cell.width() / sw, cell.height() / sh)
    w, h = sw * k, sh * k
    return QRectF(cell.x() + (cell.width() - w) / 2.0, cell.y() + (cell.height() - h) / 2.0,
                  w, h)


def _guess(pair: Optional[PairScene]) -> bool:
    return pair is not None and pair.window_source == "diff"


class PairView(QWidget):
    """Two pictures, one window: what the part looked like before and after it left."""

    #: The mode, the zoom or the candidate changed (the inset and the chips follow).
    sigChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pair: Optional[PairScene] = None
        self._pixmaps: list[QPixmap] = [QPixmap(), QPixmap()]
        self._mode = CROP
        self._zoom = 1.0
        #: Which of the pair's candidate windows is on screen.
        self._index = 0
        self.paints = 0
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(PAIR_HINT)
        self.setMinimumSize(160, 80)
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
        same_windows = (before is not None and pair is not None
                        and before.candidates == pair.candidates)
        self._pair = pair
        if reset_zoom or not same_windows:
            self._index = 0
        if reset_zoom:
            self._zoom = 1.0
        self.update()
        self.sigChanged.emit()

    def pair(self) -> Optional[PairScene]:
        return self._pair

    def pixmap(self, i: int) -> QPixmap:
        return self._pixmaps[int(i)]

    # -- the candidates (a window guessed from the difference) -------------------
    def candidate_count(self) -> int:
        pair = self._pair
        return len(pair.candidates) if pair is not None else 0

    def candidate_index(self) -> int:
        return self._index

    def set_candidate(self, index: int) -> None:
        """Put the difference's candidate ``index`` on screen (the zoom starts over)."""
        index = int(index)
        if not 0 <= index < self.candidate_count() or index == self._index:
            return
        self._index = index
        self._zoom = 1.0
        self.update()
        self.sigChanged.emit()

    def candidate_text(self) -> str:
        """「候选 k/n」, or ``""`` when there is nothing to choose from."""
        n = self.candidate_count()
        return CANDIDATE.format(k=self._index + 1, n=n) if n > 1 else ""

    def window(self) -> Optional[tuple]:
        """The window on screen, in the before frame's pixels, or ``None``."""
        pair = self._pair
        if pair is None:
            return None
        if pair.candidates:
            return tuple(pair.candidates[min(self._index, len(pair.candidates) - 1)])
        return pair.window

    def is_guess(self) -> bool:
        """Is the window a guess from the difference (not the part's outline)?"""
        return _guess(self._pair)

    def window_colour(self) -> QColor:
        """The window's pen: the part's colour for its outline, grey for a guess."""
        pair = self._pair
        if pair is None or self.is_guess():
            return QColor(GUESS_COLOUR)
        return QColor(*pair.colour)

    # -- mode and zoom -----------------------------------------------------------
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
        return (pair is not None and self.window() is not None
                and pair.frame_hw[0][1] > 0)

    def crop_of(self, window, zoom: float = 1.0, aspect: Optional[float] = None
                ) -> tuple[float, float, float, float]:
        """The before frame's crop around ``window`` (padded 1.6x, a sixth at least)."""
        return crop_rect(window, self._pair.frame_hw[0],
                         self._aspect() if aspect is None else aspect, zoom,
                         fill=1.0 / PAD, min_frac=MIN_SIDE_FRAC)

    def _max_zoom(self) -> float:
        if not self._has_window():
            return 1.0
        base = self.crop_of(self.window())
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
        """The window on screen in frame ``i``'s pixels, ``None`` with no window."""
        window = self.window()
        if window is None:
            return None
        sx, sy = self._scale(i)
        x0, y0, x1, y1 = window
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
            x0, y0, x1, y1 = self.crop_of(self.window(), self._zoom)
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
        colour = self.window_colour()
        geometry = pair.geometry
        if i == 0 and geometry is not None and not self.is_guess():
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

    SIDE = 72

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
            colour = view.window_colour()
            painter.setPen(QPen(colour, 1.5))
            fill = QColor(colour)
            fill.setAlpha(90)
            painter.setBrush(QBrush(fill))
            painter.drawRect(rect)
        painter.end()


class CandidateBar(QWidget):
    """「候选 k/n」 and the other candidates as small crops of the before frame.

    Shown only for a window guessed from the difference with more than one
    candidate; a click on a chip puts that candidate on screen.
    """

    CHIP = 48
    GAP = 4

    def __init__(self, view: PairView, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._view = view
        self.paints = 0
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedHeight(self.CHIP)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(CHIP_TIP)
        view.sigChanged.connect(self._follow)

    def sizeHint(self) -> QSize:  # noqa: D102 - Qt override
        return QSize(self._label_width() + 2 * (self.CHIP + self.GAP), self.CHIP)

    def _follow(self) -> None:
        self.setVisible(self._view.candidate_count() > 1)
        self.update()

    def _label_width(self) -> int:
        return self.fontMetrics().horizontalAdvance(CANDIDATE.format(k=3, n=3)) + 8

    def text(self) -> str:
        return self._view.candidate_text()

    def chips(self) -> list[tuple[int, QRectF]]:
        """``(candidate index, where its chip is)`` for every candidate not on screen."""
        view = self._view
        x = float(self._label_width())
        out = []
        for index in range(view.candidate_count()):
            if index == view.candidate_index():
                continue
            out.append((index, QRectF(x, 0.0, self.CHIP, self.CHIP)))
            x += self.CHIP + self.GAP
        return out

    def mousePressEvent(self, event) -> None:  # noqa: D102 - Qt override
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: D102 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            pos = QPointF(event.position())
            for index, rect in self.chips():
                if rect.contains(pos):
                    self._view.set_candidate(index)
                    break
        event.accept()

    def paintEvent(self, event) -> None:  # noqa: D102 - Qt override
        self.paints += 1
        painter = QPainter(self)
        view = self._view
        pair = view.pair()
        painter.setPen(self.palette().windowText().color())
        painter.drawText(QRectF(0.0, 0.0, self._label_width(), self.CHIP),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                         self.text())
        pixmap = view.pixmap(0)
        if pair is None or pixmap.isNull() or pair.frame_hw[0][1] <= 0:
            painter.end()
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        h_img, w_img = pair.frame_hw[0]
        sx, sy = pixmap.width() / float(w_img), pixmap.height() / float(h_img)
        for index, rect in self.chips():
            window = pair.candidates[index]
            src = view.crop_of(window, 1.0, aspect=1.0)
            painter.fillRect(rect, BACKGROUND)
            target = _fit(src, rect)
            painter.drawPixmap(target, pixmap,
                               QRectF(src[0] * sx, src[1] * sy,
                                      (src[2] - src[0]) * sx, (src[3] - src[1]) * sy))
            painter.setPen(QPen(GUESS_COLOUR, 1.0, Qt.PenStyle.DotLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
        painter.end()
