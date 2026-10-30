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

Since task U28 the panel has a second kind of scene, 「拆前 / 拆后」 -- the
scanner frames before and after the part leaves
(:mod:`tda.ui.panels.locator_pair`) -- and a toggle next to its caption that
asks for it; :class:`LocatorPanel` shows one kind or the other.

Task U43 adds a third, 「这一步的变化」 -- the whole scanner frame of the step
with every part of the frame's task card painted on it
(:mod:`tda.ui.panels.locator_change`) -- and the toggle becomes one button
that cycles the three kinds, its text naming the kind a click switches to.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

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
#: Which picture :class:`LocatorPanel` has up.
SHOW_SCENE = "scene"
SHOW_PAIR = "pair"
SHOW_CHANGE = "change"
BACKGROUND = QColor(32, 33, 36)
ROI_PEN = QColor(255, 255, 255, 110)


def crop_rect(box, frame_hw: tuple[int, int], aspect: float,
              zoom: float = 1.0, *, fill: float = FILL,
              min_frac: float = MIN_CROP_FRAC) -> tuple[float, float, float, float]:
    """The part of the frame to show around ``box``, ``(x0, y0, x1, y1)``.

    The part spans ``fill`` (:data:`FILL`) of it, it is at least ``min_frac``
    (:data:`MIN_CROP_FRAC`) of the frame's long side, it has the panel's shape
    (``aspect`` = width / height) where the frame allows, it is ``zoom`` times
    that, and it stays inside the frame -- moved, not cut, when the part is
    near an edge.  The 「拆前 / 拆后」 pair (task U28) pads its window by 1.6
    (``fill = 1 / 1.6``) with a sixth of the frame as the floor.
    """
    h_img, w_img = float(frame_hw[0]), float(frame_hw[1])
    x0, y0, x1, y1 = (float(v) for v in box)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    floor = min_frac * max(h_img, w_img)
    w = max((x1 - x0) / fill, floor)
    h = max((y1 - y0) / fill, floor)
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
    """Caption, picture, sentence: what the 「主视角」 dock holds.

    Three kinds of scene: where the part is on this step's scanner frame
    (:class:`LocatorView`); 「拆前 / 拆后」 (task U28) -- the scanner frames
    before and after it leaves (:class:`~tda.ui.panels.locator_pair.PairView`,
    with a small whole frame and the log's words under it); 「这一步的变化」
    (task U43) -- the card's parts painted on the whole scanner frame
    (:class:`~tda.ui.panels.locator_change.ChangeView`, the part under the
    mouse named in the caption's second line).  The button next to the
    caption asks for the next kind; the window decides which is up.
    """

    #: The kind button was clicked (task U43: it cycles the three kinds).
    sigKindClicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        from tda.ui.panels.locator_change import ChangeView
        from tda.ui.panels.locator_pair import (
            PAIR_TIP,
            TOGGLE_TEXT,
            CandidateBar,
            InsetView,
            PairView,
        )

        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("font-weight: bold;")
        self.pair_button = QToolButton()
        self.pair_button.setText(TOGGLE_TEXT)
        self.pair_button.setCheckable(True)
        self.pair_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.pair_button.setToolTip(PAIR_TIP)
        self.pair_button.setObjectName("locator_pair_toggle")
        self.pair_button.clicked.connect(lambda checked=False: self.sigKindClicked.emit())
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        head.addWidget(self.caption, 1)
        head.addWidget(self.pair_button, 0, Qt.AlignmentFlag.AlignTop)
        self.view = LocatorView()
        self.pair_view = PairView()
        self.pair_view.setVisible(False)
        self.change_view = ChangeView()
        self.change_view.setVisible(False)
        self.change_view.sigHover.connect(self._change_line)
        #: 「这一步的变化」's first caption line, while it is up.
        self._change_head = ""
        self.inset = InsetView(self.pair_view)
        self.candidates = CandidateBar(self.pair_view)
        self.candidates.setVisible(False)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        # under the pair: the other candidates (a guessed window) on a row of
        # their own, then the small whole frame with the log's words beside it
        self.below = QWidget()
        below = QVBoxLayout(self.below)
        below.setContentsMargins(0, 0, 0, 0)
        below.setSpacing(2)
        below.addWidget(self.candidates)
        self._row = QHBoxLayout()
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(4)
        self._row.addWidget(self.inset, 0, Qt.AlignmentFlag.AlignTop)
        self._row.addWidget(self.info, 1)
        below.addLayout(self._row)
        self.below.setVisible(False)
        # the note follows the candidate on screen: a vanished detection has its own
        self.pair_view.sigChanged.connect(self._pair_note)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #8a5a00;")
        self.note.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(2)
        layout.addLayout(head)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.pair_view, 1)
        layout.addWidget(self.change_view, 1)
        layout.addWidget(self.below)
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
        self._show_kind(SHOW_SCENE)
        self.caption.setText("" if scene is None else scene.caption)
        self._set_note("" if scene is None else scene.note)
        # A wrapped caption may have gained or lost a line: lay out now, not
        # at the next pass of the event loop, so the picture is never drawn
        # under a caption cut in half.
        self.layout().activate()
        self.shown += 1

    def show_pair(self, pair, head: str = "") -> None:
        """Draw a 「拆前 / 拆后」 (:class:`~tda.ui.locator_pair.PairScene`) under ``head``."""
        before = self.pair_view.pair()
        new_part = (before is None or pair is None or before.instance != pair.instance
                    or not self.showing_pair())
        self.pair_view.set_pair(pair, reset_zoom=new_part)
        self._show_kind(SHOW_PAIR)
        self.caption.setText(head)
        self.info.setText("" if pair is None else "\n".join(pair.lines()))
        self.candidates.setVisible(self.pair_view.has_choice())
        self._pair_note()
        self.layout().activate()
        self.inset.update()
        self.shown += 1

    def show_change(self, change) -> None:
        """Draw 「这一步的变化」 (:class:`~tda.ui.locator_change.ChangeScene`), task U43.

        The zoom starts over on another scanner frame; the same frame rebuilt
        (an edit, a selection) keeps where the annotator was looking.
        """
        before = self.change_view.change()
        new_frame = (before is None or change is None or not self.showing_change()
                     or (before.desktop, before.scan_step) != (change.desktop,
                                                               change.scan_step))
        self.change_view.set_change(change, reset_zoom=new_frame)
        self._show_kind(SHOW_CHANGE)
        self._change_head = "" if change is None else change.caption
        self._change_line(self.change_view.hover_text())
        self._set_note("" if change is None else change.note)
        self.layout().activate()
        self.shown += 1

    def _change_line(self, text: str = "") -> None:
        """The caption's second line: the part under the mouse, else the counts."""
        if not self.showing_change():
            return
        change = self.change_view.change()
        line = text or ("" if change is None else change.line)
        self.caption.setText(f"{self._change_head}\n{line}" if line else self._change_head)

    def _pair_note(self) -> None:
        """The pair's note -- or, with a vanished detection on screen, what it is."""
        from tda.ui.locator_dets import DET_NOTE

        if not self.showing_pair():
            return
        pair = self.pair_view.pair()
        if pair is None:
            self._set_note("")
            return
        label = self.pair_view.candidate_label()
        if label:
            name = label.split(" ", 1)[-1]
            self._set_note(DET_NOTE.format(cls=name, step=pair.asked[1]))
        else:
            self._set_note(pair.note)

    def candidate_text(self) -> str:
        """「候选 k/n」 while a guessed window has others to choose from, else ``""``."""
        return self.pair_view.candidate_text() if self.showing_pair() else ""

    def _show_kind(self, kind: str) -> None:
        """One of the three pictures on screen (:data:`SHOW_SCENE` / ``PAIR`` / ``CHANGE``)."""
        for widget, on in ((self.view, kind == SHOW_SCENE),
                           (self.pair_view, kind == SHOW_PAIR),
                           (self.below, kind == SHOW_PAIR),
                           (self.change_view, kind == SHOW_CHANGE)):
            if widget.isHidden() == on:
                widget.setVisible(on)

    def _set_note(self, note: str) -> None:
        self.note.setText(note)
        self.note.setVisible(bool(note))

    def set_pair_button(self, checked: bool, enabled: bool, tip: str = "",
                        text: str = "") -> None:
        """The kind button: checked while the pair is up, live when it has somewhere to go.

        ``text`` names the kind a click switches to (task U43).
        """
        button = self.pair_button
        if text and button.text() != text:
            button.setText(text)
        if button.isChecked() != bool(checked):
            blocked = button.blockSignals(True)
            button.setChecked(bool(checked))
            button.blockSignals(blocked)
        if button.isEnabled() != bool(enabled):
            button.setEnabled(bool(enabled))
        if tip and button.toolTip() != tip:
            button.setToolTip(tip)

    def showing_pair(self) -> bool:
        """Is the 「拆前 / 拆后」 up (rather than the location)?"""
        return not self.pair_view.isHidden()

    def showing_change(self) -> bool:
        """Is 「这一步的变化」 up (task U43)?"""
        return not self.change_view.isHidden()

    def scene(self) -> Optional[LocatorScene]:
        """The location scene on screen, ``None`` while another kind is up."""
        return None if self.showing_pair() or self.showing_change() else self.view.scene()

    def change(self):
        """「这一步的变化」 on screen, ``None`` while another kind is up."""
        return self.change_view.change() if self.showing_change() else None

    def pair(self):
        """The 「拆前 / 拆后」 on screen, ``None`` while the location is up."""
        return self.pair_view.pair() if self.showing_pair() else None

    def caption_text(self) -> str:
        return self.caption.text()

    def info_text(self) -> str:
        return self.info.text()

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
