"""Video canvas and a source-time range bar; no decoding or project writes."""
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QSlider, QDoubleSpinBox, QCheckBox, QStyle, QStyleOptionSlider)


def tc(seconds):
    sign = '-' if seconds < 0 else ''
    milliseconds = round(abs(seconds) * 1000)
    return f'{sign}{milliseconds // 60000:02d}:{milliseconds // 1000 % 60:02d}.{milliseconds % 1000:03d}'


def button(text, callback, layout, tooltip=''):
    widget = QPushButton(text)
    widget.setToolTip(tooltip)
    widget.clicked.connect(callback)
    layout.addWidget(widget)
    return widget


def time_box():
    widget = QDoubleSpinBox()
    widget.setDecimals(4)
    widget.setRange(0, 1000000)
    widget.setSingleStep(.1)
    widget.setSuffix(' s')
    widget.setKeyboardTracking(False)
    widget.setMinimumWidth(120)
    return widget


class VideoCanvas(QWidget):
    roi_selected = Signal(list)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source
        self.image = None
        self.full_view = False
        self.selecting = False
        self.drag_start = self.drag_end = None
        self.message = '正在读取原视频…'
        self.setMinimumSize(240, 180)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_frame(self, frame):
        arr = frame.rgb
        self.image = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format.Format_RGB888).copy()
        self.message = ''
        self.update()

    def bounds(self):
        if self.full_view or self.selecting:
            return [0, 0, self.source['width'], self.source['height']]
        return self.source['roi']

    def image_rect(self):
        _, _, width, height = self.bounds()
        scale = min(self.width() / width, self.height() / height)
        w, h = width * scale, height * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def source_point(self, point):
        rect = self.image_rect()
        x, y, w, h = self.bounds()
        px = max(0., min(1., (point.x() - rect.x()) / max(1, rect.width())))
        py = max(0., min(1., (point.y() - rect.y()) / max(1, rect.height())))
        return QPointF(x + px * w, y + py * h)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#080e17'))
        target = self.image_rect()
        if self.image:
            x, y, w, h = self.bounds()
            sx, sy = self.image.width() / self.source['width'], self.image.height() / self.source['height']
            p.drawImage(target, self.image, QRectF(x*sx, y*sy, w*sx, h*sy))
            if self.full_view or self.selecting:
                x, y, w, h = self.source['roi']
                scale = target.width() / self.source['width']
                p.setPen(QPen(QColor('#43dbc5'), 2))
                p.drawRect(QRectF(target.x()+x*scale, target.y()+y*scale, w*scale, h*scale))
        if self.drag_start is not None:
            p.setPen(QPen(QColor('#ffce70'), 2))
            p.setBrush(QColor(255, 206, 112, 35))
            p.drawRect(QRectF(self.drag_start, self.drag_end).normalized())
        if self.message:
            p.fillRect(self.rect(), QColor(8, 14, 23, 210))
            p.setPen(QColor('#cad8e8'))
            p.drawText(self.rect().adjusted(15, 10, -15, -10), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.message)

    def mousePressEvent(self, event):
        self.setFocus()
        if self.selecting and event.button() == Qt.MouseButton.LeftButton:
            self.drag_start = self.drag_end = event.position()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            self.drag_end = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_start is None:
            return
        a, b = self.source_point(self.drag_start), self.source_point(event.position())
        x, y = int(min(a.x(), b.x())), int(min(a.y(), b.y()))
        w, h = int(abs(a.x()-b.x())), int(abs(a.y()-b.y()))
        self.drag_start = self.drag_end = None
        if w >= 2 and h >= 2:
            self.roi_selected.emit([x, y, w, h])
        self.update()


class SeekSlider(QSlider):
    """A video seek bar: clicking the groove seeks, and dragging previews live."""
    scrub_started = Signal()
    scrub_moved = Signal(float)
    scrub_finished = Signal(float)

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setSingleStep(1000)
        self.setPageStep(10000)
        self.setToolTip('点击或拖动定位原视频；播放中拖动，松开后继续播放')

    def move_to(self, point):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                            QStyle.SubControl.SC_SliderHandle, self)
        value = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(),
            round(point.x()-handle.width()/2), max(1, self.width()-handle.width()), option.upsideDown)
        self.setValue(value)
        self.scrub_moved.emit(value/1000)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self.setFocus()
        self.setSliderDown(True)
        self.scrub_started.emit()
        self.move_to(event.position())
        event.accept()

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            self.move_to(event.position())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isSliderDown():
            self.move_to(event.position())
            self.setSliderDown(False)
            self.scrub_finished.emit(self.value()/1000)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        previous = self.value()
        super().keyPressEvent(event)
        if self.value() != previous:
            self.scrub_finished.emit(self.value()/1000)

    def wheelEvent(self, event):
        previous = self.value()
        super().wheelEvent(event)
        if self.value() != previous:
            self.scrub_finished.emit(self.value()/1000)


class CameraPane(QWidget):
    seek_requested = Signal(float)
    scrub_started = Signal()
    scrub_moved = Signal(float)
    scrub_finished = Signal(float)
    step_requested = Signal(int)
    play_requested = Signal()
    roi_requested = Signal(list)
    roi_numbers_requested = Signal()
    source_requested = Signal()

    def __init__(self, name, source, parent=None):
        super().__init__(parent)
        self.source = source
        self.frame = None
        self.busy = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel(f'<b>{name}</b>　原视频'))
        self.clock = QLabel('--:--.---')
        self.clock.setObjectName('clock')
        row.addStretch()
        row.addWidget(self.clock)
        layout.addLayout(row)
        self.canvas = VideoCanvas(source)
        layout.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.full_box = QCheckBox('全画面')
        self.full_box.toggled.connect(self.full_view)
        row.addWidget(self.full_box)
        self.roi_button = button('框选 ROI', self.select_roi, row)
        button('ROI 数值', self.roi_numbers_requested.emit, row)
        row.addStretch()
        button('定位原片', self.source_requested.emit, row)
        layout.addLayout(row)
        self.slider = SeekSlider()
        self.slider.setRange(0, round(source['duration'] * 1000))
        self.slider.scrub_started.connect(self.scrub_started.emit)
        self.slider.scrub_moved.connect(self.scrub_moved.emit)
        self.slider.scrub_finished.connect(self.scrub_finished.emit)
        layout.addWidget(self.slider)
        row = QHBoxLayout()
        button('◀ 帧', lambda: self.step_requested.emit(-1), row)
        self.play_button = button('播放', self.play_requested.emit, row)
        button('帧 ▶', lambda: self.step_requested.emit(1), row)
        self.position = time_box()
        self.position.setRange(0, source['duration'])
        row.addWidget(self.position)
        button('跳转', lambda: self.seek_requested.emit(self.position.value()), row)
        layout.addLayout(row)
        self.canvas.roi_selected.connect(self.finish_roi)

    def full_view(self, enabled):
        self.canvas.full_view = enabled
        self.canvas.update()

    def select_roi(self):
        self.canvas.selecting = not self.canvas.selecting
        self.roi_button.setText('取消框选' if self.canvas.selecting else '框选 ROI')
        self.canvas.update()

    def finish_roi(self, roi):
        self.canvas.selecting = False
        self.roi_button.setText('框选 ROI')
        self.roi_requested.emit(roi)

    def show_frame(self, frame):
        self.frame, self.busy = frame, False
        self.canvas.set_frame(frame)
        self.clock.setText(tc(frame.pts))
        if not self.position.hasFocus():
            self.position.setValue(frame.pts)
        if not self.slider.isSliderDown():
            self.slider.setValue(round(frame.pts * 1000))


class RangeBar(QWidget):
    scrub_started = Signal()
    scrub_moved = Signal(float)
    scrub_finished = Signal(float)
    boundary_changed = Signal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.low, self.high = 0., 10.
        self.start, self.end, self.position = 2., 8., 2.
        self.baseline = [2., 8.]
        self.dragging = None
        self.setMinimumHeight(72)
        self.setMouseTracking(True)

    def to_x(self, value):
        return 18 + (value-self.low) / max(.001, self.high-self.low) * (self.width()-36)

    def to_time(self, x):
        return self.low + max(0, min(1, (x-18)/max(1, self.width()-36))) * (self.high-self.low)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#101c2c'))
        p.fillRect(QRectF(18, 26, self.width()-36, 22), QColor('#23354b'))
        p.fillRect(QRectF(self.to_x(self.start), 26, self.to_x(self.end)-self.to_x(self.start), 22), QColor('#24796f'))
        p.setPen(QPen(QColor('#6e849e'), 1, Qt.PenStyle.DashLine))
        for v in self.baseline:
            p.drawLine(QPointF(self.to_x(v), 16), QPointF(self.to_x(v), 56))
        p.setPen(QPen(QColor('#67ead5'), 4))
        for v in [self.start, self.end]:
            p.drawLine(QPointF(self.to_x(v), 24), QPointF(self.to_x(v), 50))
        p.setPen(QPen(QColor('#ffcd78'), 2))
        px = max(18., min(self.width()-18., self.to_x(self.position)))
        p.drawLine(QPointF(px, 14), QPointF(px, 56))
        p.setPen(QColor('#b1c4da'))
        p.drawText(18, 68, tc(self.low))
        p.drawText(self.width()-90, 68, tc(self.high))
        p.drawText(18, 13, '原片时间轴 · 拖动绿色边界调整 · 点击定位')

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        self.dragging = ('start' if abs(x-self.to_x(self.start)) < 10 else
                         'end' if abs(x-self.to_x(self.end)) < 10 else 'seek')
        if self.dragging == 'seek':
            self.scrub_started.emit()
            self.position = self.to_time(x)
            self.scrub_moved.emit(self.position)
            self.update()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            if self.dragging == 'seek':
                self.position = self.to_time(event.position().x())
                self.scrub_moved.emit(self.position)
                self.update()

    def mouseReleaseEvent(self, event):
        value = self.to_time(event.position().x())
        if self.dragging in ('start', 'end'):
            self.boundary_changed.emit(self.dragging, value)
        elif self.dragging:
            self.scrub_finished.emit(value)
        self.dragging = None
        self.unsetCursor()
