"""「主视角」: where this frame's part is in the scanner view (task U23).

On an OAK1 / OAK2 / RealSense frame the annotator used to switch to the
scanner view to see what the task card's part is and where it sits, and back.
This panel shows it next to the frame they are drawing on: the scanner frame of
the same step, cropped around that part's compiled scanner mask so the part
spans about half the panel, its outline in the part's own colour (the one the
canvas paints it with) and a dot inside it.  The caption says which scanner
frame and which part: 「主视角 第 k 帧 · <label>（<state>）」.

* **A click** switches between the crop and the whole frame (with the part's
  bounding box, and the scanner ROI drawn faintly); **the wheel** zooms the
  crop.  The panel never takes the keyboard.
* **No outline**: the sentence under the picture says why -- the part is not
  on that scanner frame, was never drawn in the scanner view, or the scanner
  view of this machine is not annotated at all -- and the whole frame is shown.

The panel only draws a :class:`~tda.ui.locator_source.LocatorScene`; the
window (:mod:`tda.ui.app_locator`) decides when it is shown and what for, and
:class:`~tda.ui.locator_source.LocatorSource` reads the data.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QDockWidget, QLabel, QSizePolicy, QVBoxLayout, QWidget

from tda.ui.locator_source import LocatorScene

__all__ = ["CROP", "HINT", "LocatorDock", "LocatorPanel", "LocatorView", "TITLE", "WHOLE",
           "crop_rect"]

#: The dock's title and the top bar's toggle.
TITLE = "主视角"
#: The two ways the picture is shown.
CROP = "crop"
WHOLE = "whole"
#: The picture's tooltip: what the mouse does on it.
HINT = ("扫描仪视角里这个零件在哪。单击：裁剪 ↔ 整帧；滚轮：放大 / 缩小。"
        "where this part is in the scanner view; click: crop or whole frame; "
        "wheel: zoom")
#: The part spans this much of the crop, across and down.
FILL = 0.5
#: ... but a crop is never smaller than this share of the frame's long side:
#: a screw filling half the panel would be a blur with nothing around it.
MIN_CROP_FRAC = 0.1
#: The wheel: one notch in or out, and how far.
ZOOM_STEP = 1.25
MIN_ZOOM = 0.2
#: A crop never narrower than this many frame pixels.
MIN_CROP_PX = 16.0
BACKGROUND = QColor(32, 33, 36)
ROI_PEN = QColor(255, 255, 255, 110)


def crop_rect(box, frame_hw: tuple[int, int], aspect: float,
              zoom: float = 1.0) -> tuple[float, float, float, float]:
    """The part of the frame to show around ``box``, ``(x0, y0, x1, y1)``.

    The part spans :data:`FILL` of it, it is at least :data:`MIN_CROP_FRAC` of
    the frame's long side, it has the panel's shape (``aspect`` = width /
    height) where the frame allows, it is ``zoom`` times that, and it stays
    inside the frame -- moved, not cut, when the part is near an edge.
    """
    h_img, w_img = float(frame_hw[0]), float(frame_hw[1])
    x0, y0, x1, y1 = (float(v) for v in box)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    floor = MIN_CROP_FRAC * max(h_img, w_img)
    w = max((x1 - x0) / FILL, floor)
    h = max((y1 - y0) / FILL, floor)
    aspect = aspect if aspect > 0 else 1.0
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    w = max(MIN_CROP_PX, w * zoom)
    h = max(MIN_CROP_PX, h * zoom)
    if w_img > 0:
        w = min(w, w_img)
    if h_img > 0:
        h = min(h, h_img)
    left = min(max(cx - w / 2.0, 0.0), max(0.0, w_img - w))
    top = min(max(cy - h / 2.0, 0.0), max(0.0, h_img - h))
    return (left, top, left + w, top + h)


class LocatorView(QWidget):
    """The picture: a crop or the whole scanner frame, with the part marked."""

    #: The mode after a click (:data:`CROP` / :data:`WHOLE`).
    sigModeChanged = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene: Optional[LocatorScene] = None
        self._pixmap = QPixmap()
        self._mode = CROP
        self._zoom = 1.0
        self.paints = 0
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(HINT)
        self.setMinimumSize(120, 80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:  # noqa: D102 - Qt override
        return QSize(300, 210)

    # -- state ----------------------------------------------------------------
    def set_scene(self, scene: Optional[LocatorScene], *, reset_zoom: bool = False) -> None:
        """Show ``scene``; the picture is converted once here, not per paint."""
        same_image = (self._scene is not None and scene is not None
                      and scene.image is self._scene.image)
        self._scene = scene
        if not same_image:
            self._pixmap = (QPixmap.fromImage(scene.image)
                            if scene is not None and scene.image is not None else QPixmap())
        if reset_zoom:
            self._zoom = 1.0
        self.update()

    def scene(self) -> Optional[LocatorScene]:
        return self._scene

    def mode(self) -> str:
        """:data:`CROP` or :data:`WHOLE` -- what a part with an outline is shown as."""
        return self._mode

    def showing(self) -> str:
        """What is on screen: the crop only when there is something to crop to."""
        return CROP if self._mode == CROP and self._has_geometry() else WHOLE

    def set_mode(self, mode: str) -> None:
        self._mode = CROP if mode == CROP else WHOLE
        self.update()

    def toggle_mode(self) -> str:
        """A click: crop <-> whole frame.  Nothing to crop to, nothing changes."""
        if not self._has_geometry():
            return self._mode
        self.set_mode(WHOLE if self._mode == CROP else CROP)
        self.sigModeChanged.emit(self._mode)
        return self._mode

    def zoom(self) -> float:
        return self._zoom

    def zoom_by(self, notches: float) -> float:
        """The wheel: positive notches zoom in (a smaller crop)."""
        if self.showing() != CROP:
            return self._zoom
        factor = ZOOM_STEP ** (-float(notches))
        self._zoom = min(max(self._zoom * factor, MIN_ZOOM), self._max_zoom())
        self.update()
        return self._zoom

    def _has_geometry(self) -> bool:
        scene = self._scene
        return scene is not None and scene.geometry is not None and scene.frame_hw[1] > 0

    def _max_zoom(self) -> float:
        """The zoom at which the crop is the whole frame (no point going further)."""
        scene = self._scene
        if not self._has_geometry():
            return 1.0
        base = crop_rect(scene.geometry.box, scene.frame_hw, self._aspect(), 1.0)
        h_img, w_img = scene.frame_hw
        return max(1.0, w_img / max(1.0, base[2] - base[0]),
                   h_img / max(1.0, base[3] - base[1]))

    def _aspect(self) -> float:
        return max(1, self.width()) / float(max(1, self.height()))

    def source_rect(self) -> Optional[tuple[float, float, float, float]]:
        """The part of the frame on screen, in frame pixels; ``None`` with no frame."""
        scene = self._scene
        if scene is None or scene.frame_hw[1] <= 0:
            return None
        h_img, w_img = scene.frame_hw
        if self.showing() == CROP:
            return crop_rect(scene.geometry.box, scene.frame_hw, self._aspect(), self._zoom)
        return (0.0, 0.0, float(w_img), float(h_img))

    def _target(self, src) -> QRectF:
        """Where ``src`` lands in the widget: fitted, centred, its shape kept."""
        sw, sh = max(1e-6, src[2] - src[0]), max(1e-6, src[3] - src[1])
        k = min(self.width() / sw, self.height() / sh)
        w, h = sw * k, sh * k
        return QRectF((self.width() - w) / 2.0, (self.height() - h) / 2.0, w, h)

    # -- mouse ------------------------------------------------------------------
    def mouseReleaseEvent(self, event) -> None:  # noqa: D102 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_mode()
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: D102 - Qt override
        event.accept()       # never a click on whatever is underneath

    def wheelEvent(self, event) -> None:  # noqa: D102 - Qt override
        notches = event.angleDelta().y() / 120.0
        if notches:
            self.zoom_by(notches)
        event.accept()

    # -- paint ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: D102 - Qt override
        self.paints += 1
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        scene = self._scene
        src = self.source_rect()
        if scene is None or src is None:
            painter.end()
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        target = self._target(src)
        h_img, w_img = scene.frame_hw
        if not self._pixmap.isNull():
            sx = self._pixmap.width() / float(w_img)
            sy = self._pixmap.height() / float(h_img)
            painter.drawPixmap(target, self._pixmap,
                               QRectF(src[0] * sx, src[1] * sy,
                                      (src[2] - src[0]) * sx, (src[3] - src[1]) * sy))
        kx = target.width() / max(1e-6, src[2] - src[0])
        ky = target.height() / max(1e-6, src[3] - src[1])

        def at(x: float, y: float) -> QPointF:
            return QPointF(target.x() + (x - src[0]) * kx, target.y() + (y - src[1]) * ky)

        painter.setClipRect(target)
        whole = self.showing() == WHOLE
        if whole and scene.roi is not None:
            pen = QPen(ROI_PEN, 1.0, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(at(scene.roi[0], scene.roi[1]),
                                    at(scene.roi[2], scene.roi[3])))
        geometry = scene.geometry
        if geometry is not None:
            colour = QColor(*scene.colour)
            path = QPainterPath()
            for ring in geometry.outlines:
                if len(ring) < 2:
                    continue
                path.moveTo(at(float(ring[0][0]), float(ring[0][1])))
                for x, y in ring[1:]:
                    path.lineTo(at(float(x), float(y)))
                path.closeSubpath()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 170), 4.0))
            painter.drawPath(path)
            painter.setPen(QPen(colour, 2.0))
            painter.drawPath(path)
            if whole:
                box = geometry.box
                painter.setPen(QPen(colour, 1.5, Qt.PenStyle.DashLine))
                painter.drawRect(QRectF(at(box[0], box[1]), at(box[2], box[3])).adjusted(
                    -3, -3, 3, 3))
            centre = at(*geometry.centre)
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            painter.setBrush(QBrush(colour))
            painter.drawEllipse(centre, 4.5, 4.5)
        painter.end()


class LocatorPanel(QWidget):
    """Caption, picture, sentence: what the 「主视角」 dock holds."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("font-weight: bold;")
        self.view = LocatorView()
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #8a5a00;")
        self.note.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(2)
        layout.addWidget(self.caption)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.note)
        #: How many scenes were shown (the window's refresh count).
        self.shown = 0

    def sizeHint(self) -> QSize:  # noqa: D102 - Qt override
        return QSize(300, 260)

    def minimumSizeHint(self) -> QSize:  # noqa: D102 - Qt override
        return QSize(140, 120)

    def show_scene(self, scene: Optional[LocatorScene]) -> None:
        """Draw ``scene``; the zoom starts over when the part changes."""
        before = self.view.scene()
        new_part = (before is None or scene is None or before.instance != scene.instance)
        self.view.set_scene(scene, reset_zoom=new_part)
        self.caption.setText("" if scene is None else scene.caption)
        note = "" if scene is None else scene.note
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self.shown += 1

    def scene(self) -> Optional[LocatorScene]:
        return self.view.scene()

    def caption_text(self) -> str:
        return self.caption.text()

    def note_text(self) -> str:
        return self.note.text()


class LocatorDock(QDockWidget):
    """The dock, telling a close by the annotator apart from the window hiding it."""

    #: The title bar's close button: the annotator does not want the panel.
    sigClosedByUser = Signal()

    def closeEvent(self, event) -> None:  # noqa: D102 - Qt override
        super().closeEvent(event)
        if event.isAccepted():
            self.sigClosedByUser.emit()
