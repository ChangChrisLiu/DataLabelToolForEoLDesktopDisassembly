"""Independent dual-camera source-video refinement window."""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
import time

from PySide6.QtCore import Qt, QTimer, QLockFile, QEvent
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QLineEdit, QCheckBox,
    QComboBox, QSplitter, QTabWidget, QFileDialog, QDialog, QDialogButtonBox,
    QSpinBox, QProgressBar, QAbstractSpinBox,
)

from .project import Project, History, output_path
from .media import probe
from .workers import DecodeWorker, ExportWorker
from .widgets import CameraPane, RangeBar, button, time_box, tc


STYLE = '''
QMainWindow,QDialog { background:#101927; color:#e0e9f4; }
QWidget { font-family:"Microsoft YaHei UI"; font-size:12px; color:#dce7f3; }
QLabel#title { font-size:20px; font-weight:700; }
QLabel#clock { font-family:Consolas; font-size:18px; color:#72ead7; }
QPushButton { background:#25364c; border:1px solid #3b506b; border-radius:5px; padding:6px 9px; }
QPushButton:hover { background:#344d68; }
QPushButton:disabled { color:#738398; background:#182536; }
QPushButton#primary { background:#20776b; border-color:#3cae99; color:white; }
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox { background:#182638; border:1px solid #425773; padding:5px; border-radius:4px; selection-background-color:#2d7d75; }
QListWidget { background:#142031; border:1px solid #2b405a; border-radius:6px; }
QListWidget::item { padding:9px 7px; border-bottom:1px solid #26354a; }
QListWidget::item:selected { background:#254e54; color:#e5fff8; }
QTabWidget::pane { border:1px solid #30455e; border-radius:5px; }
QTabBar::tab { background:#19293c; padding:8px 22px; }
QTabBar::tab:selected { background:#2a4854; color:#8ff3df; }
QSlider::groove:horizontal { height:5px; background:#2c405b; }
QSlider::handle:horizontal { width:12px; margin:-4px 0; border-radius:5px; background:#62cdbb; }
QCheckBox { spacing:5px; }
QProgressBar { background:#17283c; border:1px solid #344b64; text-align:center; }
QProgressBar::chunk { background:#237d70; }
QToolTip { background:#203047; color:white; border:1px solid #44607b; }
'''


class VideoRefiner(QMainWindow):
    def __init__(self, project: Project, project_path, parent=None):
        super().__init__(parent)
        self.project_path = output_path(project_path)
        self.project_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.acquire_lock(self.project_path)
        self.project = project
        self.history = History()
        self.workers = {}
        self.panes = {}
        self.generation = {'oak': 0, 'rs': 0}
        self.positions = {'oak': 0., 'rs': 0.}
        self.playing = None
        self.export_worker = None
        self.loading = False
        self.selected_uid = None
        self.closed = False
        self.setWindowTitle('双机位视频精修器 · 原片同步')
        self.resize(1550, 960)
        self.setStyleSheet(STYLE)
        self.autosave = QTimer(self)
        self.autosave.setSingleShot(True)
        self.autosave.setInterval(600)
        self.autosave.timeout.connect(self.save)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.tick)
        self.build_ui()
        self.refresh_list()
        self.start_readers()
        self.list.setCurrentRow(min(project.data.get('last_selected', 0), len(project.segments)-1))
        self.select_current()
        self.update_sync()
        QApplication.instance().installEventFilter(self)

    @staticmethod
    def acquire_lock(path):
        lock = QLockFile(str(path) + '.lock')
        lock.setStaleLockTime(0)
        if not lock.tryLock(0):
            raise ValueError('这个工程已在另一个窗口打开，请在已有窗口中继续。')
        return lock

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        top = QHBoxLayout()
        title = QLabel('视频精修 / 双机位')
        title.setObjectName('title')
        top.addWidget(title)
        top.addStretch()
        button('导入初筛', self.import_manifest, top)
        button('打开工程', self.open_project, top)
        button('保存', self.save, top, 'Ctrl+S；编辑后也会自动保存')
        button('另存为', self.save_as, top)
        button('撤销', self.undo, top, 'Ctrl+Z')
        button('重做', self.redo, top, 'Ctrl+Y')
        self.export_button = button('导出视频', self.export_dialog, top)
        self.export_button.setObjectName('primary')
        button('使用说明', self.help, top)
        layout.addLayout(top)
        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        self.summary = QLabel()
        left_layout.addWidget(self.summary)
        self.list = QListWidget()
        self.list.setMinimumWidth(235)
        self.list.currentRowChanged.connect(self.select_current)
        left_layout.addWidget(self.list, 1)
        left_layout.addWidget(QLabel('✓ 已确认　○ 待精修　× 已排除\n编号对应初筛；时间均来自原片'))
        splitter.addWidget(left)
        right = QWidget()
        body = QVBoxLayout(right)
        body.setContentsMargins(6, 0, 0, 0)
        self.player_layout = QHBoxLayout()
        body.addLayout(self.player_layout, 1)
        for camera, name in [('oak', 'OAK · 主机位'), ('rs', 'RealSense · 对应机位')]:
            source = self.project.sources.get(camera)
            if source is None:
                source = dict(width=1920, height=1080, roi=[0, 0, 1920, 1080], duration=1, path='')
            pane = CameraPane(name, source)
            pane.seek_requested.connect(lambda value, c=camera: self.seek(c, value))
            pane.step_requested.connect(lambda direction, c=camera: self.step_frame(c, direction))
            pane.play_requested.connect(lambda c=camera: self.toggle_play(c))
            pane.roi_requested.connect(lambda roi, c=camera: self.set_roi(c, roi))
            pane.roi_numbers_requested.connect(lambda c=camera: self.roi_dialog(c))
            pane.source_requested.connect(lambda c=camera: self.locate_source(c))
            self.panes[camera] = pane
            self.player_layout.addWidget(pane, 1)
        row = QHBoxLayout()
        self.link_box = QCheckBox('联动预览（按原片同步关系）')
        self.link_box.setChecked(bool(self.project.data['anchors']))
        self.link_box.toggled.connect(self.link_changed)
        row.addWidget(self.link_box)
        self.loop_box = QCheckBox('循环当前片段')
        self.loop_box.setChecked(True)
        row.addWidget(self.loop_box)
        row.addStretch()
        row.addWidget(QLabel('速度'))
        self.speed = QComboBox()
        self.speed.addItems(['0.25×', '0.5×', '1×', '2×'])
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self.pause)
        row.addWidget(self.speed)
        body.addLayout(row)
        self.tabs = QTabWidget()
        body.addWidget(self.tabs)
        self.build_edit_tab()
        self.build_sync_tab()
        self.tabs.currentChanged.connect(self.tab_changed)
        splitter.addWidget(right)
        splitter.setSizes([265, 1235])
        footer = QHBoxLayout()
        self.status = QLabel('原片只读 · 修改自动保存到精修工程')
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(210)
        self.progress.hide()
        footer.addWidget(self.progress)
        self.cancel_button = button('取消导出', self.cancel_export, footer)
        self.cancel_button.hide()
        layout.addLayout(footer)

    def build_edit_tab(self):
        tab = QWidget()
        box = QVBoxLayout(tab)
        self.range_bar = RangeBar()
        self.range_bar.seek_requested.connect(lambda t: self.seek('oak', t))
        self.range_bar.boundary_changed.connect(self.drag_boundary)
        box.addWidget(self.range_bar)
        row = QHBoxLayout()
        row.addWidget(QLabel('入点'))
        self.start_box = time_box()
        row.addWidget(self.start_box)
        button('用当前帧 [I]', lambda: self.mark_boundary('start'), row)
        row.addWidget(QLabel('出点'))
        self.end_box = time_box()
        row.addWidget(self.end_box)
        button('用当前帧后沿 [O]', lambda: self.mark_boundary('end'), row)
        row.addStretch()
        row.addWidget(QLabel('边界前后'))
        self.context = QComboBox()
        self.context.addItems(['2 秒', '5 秒', '15 秒', '60 秒', '全片'])
        self.context.setCurrentIndex(1)
        self.context.currentIndexChanged.connect(self.update_range)
        row.addWidget(self.context)
        box.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('动作'))
        self.label_box = QLineEdit()
        row.addWidget(self.label_box, 2)
        row.addWidget(QLabel('备注'))
        self.note_box = QLineEdit()
        row.addWidget(self.note_box, 3)
        self.enabled_box = QCheckBox('保留')
        row.addWidget(self.enabled_box)
        button('应用修改', self.apply_form, row)
        box.addLayout(row)
        self.mapping_label = QLabel()
        box.addWidget(self.mapping_label)
        row = QHBoxLayout()
        button('上一段', lambda: self.move_selection(-1), row)
        button('下一段', lambda: self.move_selection(1), row)
        button('拆分 [S]', self.split_current, row)
        button('新增片段', self.add_current, row)
        button('恢复初筛切点', self.reset_current, row)
        row.addStretch()
        self.confirm_button = button('确认并下一段 [Enter]', self.confirm_current, row)
        self.confirm_button.setObjectName('primary')
        box.addLayout(row)
        self.tabs.addTab(tab, '1  逐段精修')
        for widget in [self.start_box, self.end_box, self.label_box, self.note_box]:
            widget.editingFinished.connect(self.apply_form)
        self.enabled_box.toggled.connect(self.apply_form)

    def build_sync_tab(self):
        tab = QWidget()
        box = QVBoxLayout(tab)
        text = QLabel('先在两段原视频中分别定位同一瞬间（例如工具接触螺丝、零件离开机箱），暂停后记录。\n'
                      '推荐第二对点选在片尾，检查长录像漂移；录像中断或非线性跳时不能用两个点完全修正。')
        text.setWordWrap(True)
        box.addWidget(text)
        self.anchor_labels = []
        for i, title in enumerate(['基准同步点', '片尾校验点']):
            row = QHBoxLayout()
            row.addWidget(QLabel(title))
            label = QLabel('尚未设置')
            self.anchor_labels.append(label)
            row.addWidget(label, 1)
            button('记录左右当前帧', lambda checked=False, n=i: self.capture_anchor(n), row)
            button('跳到这对点', lambda checked=False, n=i: self.goto_anchor(n), row)
            box.addLayout(row)
        row = QHBoxLayout()
        self.drift_box = QCheckBox('启用两点线性校正（导出 RealSense 时匹配主机位时长）')
        self.drift_box.toggled.connect(self.drift_changed)
        row.addWidget(self.drift_box)
        row.addStretch()
        button('清除同步点', self.clear_sync, row)
        box.addLayout(row)
        self.sync_label = QLabel()
        self.sync_label.setWordWrap(True)
        box.addWidget(self.sync_label)
        self.tabs.addTab(tab, '2  原片同步')

    def index(self):
        return next((i for i, s in enumerate(self.project.segments) if s['uid'] == self.selected_uid), -1)

    def current(self):
        i = self.index()
        return self.project.segments[i] if i >= 0 else None

    def refresh_list(self):
        keep = self.selected_uid
        self.list.blockSignals(True)
        self.list.clear()
        for i, s in enumerate(self.project.segments):
            marker = '×' if not s['enabled'] else '✓' if s['confirmed'] else '○'
            item = QListWidgetItem(f'{marker}  {i+1:02d}  {s["label_zh"]}\n     {tc(s["start"])} → {tc(s["end"])}')
            item.setData(Qt.ItemDataRole.UserRole, s['uid'])
            item.setToolTip(f'日志步 {s.get("step", "—")} · 原始片段 {s.get("clip_id", "新增")}\n{s.get("note", "")}')
            self.list.addItem(item)
            if s['uid'] == keep:
                self.list.setCurrentRow(i)
        self.list.blockSignals(False)
        self.summary.setText(f'{len(self.project.segments)} 段 · {sum(s["confirmed"] and s["enabled"] for s in self.project.segments)} 已确认')

    def select_current(self, *args):
        item = self.list.currentItem()
        if item is None:
            return
        self.pause()
        self.selected_uid = item.data(Qt.ItemDataRole.UserRole)
        self.refresh_form()
        self.seek('oak', self.current()['start'])

    def refresh_form(self):
        s = self.current()
        if not s:
            return
        self.loading = True
        self.start_box.setValue(s['start'])
        self.end_box.setValue(s['end'])
        self.label_box.setText(s['label_zh'])
        self.note_box.setText(s.get('note', ''))
        self.enabled_box.setChecked(s['enabled'])
        self.confirm_button.setText('已确认 · 下一段 [Enter]' if s['confirmed'] else '确认并下一段 [Enter]')
        self.loading = False
        self.update_range()
        try:
            rstart, rend = self.project.mapped_range(s)
            mapped = f'RealSense 原片 {tc(rstart)} → {tc(rend)}'
        except ValueError as exc:
            mapped = str(exc)
        self.mapping_label.setText(f'时长 {s["end"]-s["start"]:.3f} s　|　{mapped}')

    def update_range(self, *args):
        s = self.current()
        if not s:
            return
        context = [2, 5, 15, 60, self.project.sources['oak']['duration']][self.context.currentIndex()]
        self.range_bar.low = max(0, s['start']-context)
        self.range_bar.high = min(self.project.sources['oak']['duration'], s['end']+context)
        self.range_bar.start, self.range_bar.end = s['start'], s['end']
        self.range_bar.baseline = [s['baseline']['start'], s['baseline']['end']]
        self.range_bar.update()

    def mutation(self, callback):
        before = copy.deepcopy(self.project.data)
        try:
            callback()
            self.project.validate()
        except Exception as exc:
            self.project.data = before
            self.error(str(exc))
            self.refresh_form()
            return False
        if before != self.project.data:
            self.history.push(before)
            self.autosave.start()
            self.refresh_list()
            self.refresh_form()
            self.update_sync()
            for c, p in self.panes.items():
                if self.project.sources.get(c):
                    p.source = p.canvas.source = self.project.sources[c]
                    p.canvas.update()
        return True

    def apply_form(self, *args):
        if self.loading or self.index() < 0:
            return True
        s = self.current()
        # The display has four decimal places; merely saving or confirming must
        # not replace the precise source PTS with its rounded UI representation.
        start = s['start'] if self.start_box.value() == round(s['start'], 4) else self.start_box.value()
        end = s['end'] if self.end_box.value() == round(s['end'], 4) else self.end_box.value()
        return self.mutation(lambda: self.project.edit(self.index(), start=start,
            end=end, label_zh=self.label_box.text().strip() or '未命名动作',
            note=self.note_box.text(), enabled=self.enabled_box.isChecked()))

    def undo(self):
        self.pause()
        self.project.data = self.history.undo(self.project.data)
        self.restore_state()

    def redo(self):
        self.pause()
        self.project.data = self.history.redo(self.project.data)
        self.restore_state()

    def restore_state(self):
        self.refresh_list()
        if self.index() < 0 and self.list.count():
            self.list.setCurrentRow(0)
        self.refresh_form()
        self.update_sync()
        for c, p in self.panes.items():
            if self.project.sources.get(c):
                p.source = p.canvas.source = self.project.sources[c]
                p.canvas.update()
        self.autosave.start()

    def start_readers(self):
        for c, p in self.panes.items():
            source = self.project.sources.get(c)
            if not source or not Path(source['path']).is_file():
                p.canvas.message = '原视频未找到\n请点击「定位原片」'
                p.busy = False
                p.canvas.update()
                continue
            worker = DecodeWorker(c, source['path'], self)
            worker.frame_ready.connect(self.frame_ready)
            worker.failed.connect(lambda camera, text: self.decoder_error(camera, text))
            self.workers[c] = worker
            worker.start()
            self.request_frame(c, self.positions[c], precise=True)

    def decoder_error(self, camera, text):
        pane = self.panes[camera]
        pane.busy, pane.frame = False, None
        pane.canvas.message = text
        pane.canvas.update()
        self.error(f'{camera}：{text}')

    def request_frame(self, camera, seconds, precise=False, mode='seek'):
        source = self.project.sources.get(camera)
        if not source or camera not in self.workers:
            return
        if precise:
            self.generation[camera] += 1
            self.panes[camera].busy = True
        self.positions[camera] = seconds
        if camera == 'oak':
            self.range_bar.position = seconds
            self.range_bar.update()
        if seconds < 0 or seconds >= source['duration']:
            self.generation[camera] += 1
            pane = self.panes[camera]
            pane.frame, pane.busy = None, False
            pane.canvas.message = '此时刻没有对应原视频画面'
            pane.canvas.update()
            return
        self.workers[camera].request(seconds, self.generation[camera], mode)

    def seek(self, camera, seconds):
        self.pause()
        if self.link_box.isChecked() and self.project.data['anchors']:
            a, b = self.project.mapping()
            master = seconds if camera == 'oak' else (seconds-b)/a
            self.request_frame('oak', master, precise=True)
            self.request_frame('rs', a*master+b, precise=True)
        else:
            self.request_frame(camera, seconds, precise=True)

    def frame_ready(self, camera, frame, generation, mode):
        if self.closed or generation != self.generation[camera]:
            return
        self.panes[camera].show_frame(frame)
        if mode in ('previous', 'next'):
            self.positions[camera] = frame.pts
            if camera == 'oak':
                self.range_bar.position = frame.pts
                self.range_bar.update()
            if self.link_box.isChecked() and self.project.data['anchors']:
                a, b = self.project.mapping()
                other = 'rs' if camera == 'oak' else 'oak'
                target = a*frame.pts+b if camera == 'oak' else (frame.pts-b)/a
                self.request_frame(other, target, precise=True)

    def ready_frame(self, camera='oak'):
        pane = self.panes[camera]
        if pane.frame is None or pane.busy:
            self.error('请等待原片画面定位完成，再设置切点或同步点。')
            return None
        return pane.frame

    def step_frame(self, camera, direction):
        self.pause()
        frame = self.ready_frame(camera)
        if frame:
            self.request_frame(camera, frame.pts, precise=True, mode='previous' if direction < 0 else 'next')

    def toggle_play(self, camera='oak'):
        if self.playing:
            self.pause()
            return
        if camera not in self.workers or not self.workers[camera].isRunning():
            return
        if self.link_box.isChecked():
            camera = 'oak'
        self.playing = camera
        start = self.positions[camera]
        s = self.current()
        if camera == 'oak' and self.loop_box.isChecked() and s and not s['start'] <= start < s['end']:
            start = s['start']
        self.play_start, self.play_wall = start, time.monotonic()
        self.timer.start()
        for p in self.panes.values():
            p.play_button.setText('暂停')

    def pause(self, *args):
        if not self.playing:
            return
        self.timer.stop()
        camera, self.playing = self.playing, None
        for p in self.panes.values():
            p.play_button.setText('播放')
        self.request_frame(camera, self.positions[camera], precise=True)
        if camera == 'oak' and self.link_box.isChecked() and self.project.data['anchors']:
            a, b = self.project.mapping()
            self.request_frame('rs', a*self.positions['oak']+b, precise=True)

    def tick(self):
        camera = self.playing
        if camera is None:
            return
        speed = [.25, .5, 1, 2][self.speed.currentIndex()]
        t = self.play_start + (time.monotonic()-self.play_wall)*speed
        s = self.current()
        if camera == 'oak' and self.loop_box.isChecked() and s and t >= s['end']:
            t = self.play_start = s['start']
            self.play_wall = time.monotonic()
            self.generation['oak'] += 1
            self.generation['rs'] += 1
        if t >= self.project.sources[camera]['duration']:
            self.pause()
            return
        self.request_frame(camera, t)
        if camera == 'oak' and self.link_box.isChecked() and self.project.data['anchors']:
            a, b = self.project.mapping()
            self.request_frame('rs', a*t+b)

    def mark_boundary(self, boundary):
        frame = self.ready_frame()
        if frame is None:
            return
        value = frame.pts if boundary == 'start' else frame.end
        self.pause()
        self.mutation(lambda: self.project.edit(self.index(), **{boundary: value}))

    def drag_boundary(self, boundary, value):
        if self.mutation(lambda: self.project.edit(self.index(), **{boundary: value})):
            self.seek('oak', value if boundary == 'start' else max(0, value-1e-5))

    def confirm_current(self, checked=False, advance=True):
        if not self.apply_form() or self.index() < 0:
            return
        self.mutation(lambda: self.project.edit(self.index(), confirmed=True))
        if advance:
            self.move_selection(1)

    def move_selection(self, delta):
        if self.apply_form():
            self.list.setCurrentRow(max(0, min(self.list.count()-1, self.list.currentRow()+delta)))

    def split_current(self):
        frame = self.ready_frame()
        if frame and self.index() >= 0:
            self.mutation(lambda: self.project.split(self.index(), frame.pts))

    def add_current(self):
        at = max(0, self.positions['oak'])
        end = min(self.project.sources['oak']['duration'], at+2)
        uid = []
        if self.mutation(lambda: uid.append(self.project.add(at, end))):
            self.selected_uid = uid[0]
            self.refresh_list()
            self.refresh_form()

    def reset_current(self):
        if self.index() >= 0:
            self.mutation(lambda: self.project.reset(self.index()))

    def link_changed(self, enabled):
        self.pause()
        if enabled and not self.project.data['anchors']:
            self.link_box.setChecked(False)
            self.error('请先在「原片同步」中记录一对同步点。')
        elif enabled:
            self.seek('oak', self.positions['oak'])

    def tab_changed(self, index):
        self.pause()
        if index == 1:
            self.link_box.setChecked(False)
            self.loop_box.setChecked(False)
            self.status.setText('同步定位模式：左右分别定位同一瞬间，再记录同步点。')

    def capture_anchor(self, index):
        self.pause()
        oak, rs = self.ready_frame('oak'), self.ready_frame('rs')
        if oak is None or rs is None:
            return
        anchors = copy.deepcopy(self.project.data['anchors'])
        if index > len(anchors):
            self.error('请先记录基准同步点。')
            return
        if index == len(anchors):
            anchors.append([oak.pts, rs.pts])
        else:
            anchors[index] = [oak.pts, rs.pts]
        self.mutation(lambda: self.project.set_sync(anchors, self.project.data['drift_correction']))

    def goto_anchor(self, index):
        if index >= len(self.project.data['anchors']):
            return
        self.link_box.setChecked(False)
        oak, rs = self.project.data['anchors'][index]
        self.seek('oak', oak)
        self.seek('rs', rs)

    def clear_sync(self):
        self.link_box.setChecked(False)
        self.mutation(lambda: self.project.set_sync([], False))

    def drift_changed(self, checked):
        if not self.loading:
            self.mutation(lambda: self.project.set_sync(self.project.data['anchors'], checked))

    def update_sync(self):
        self.loading = True
        anchors = self.project.data['anchors']
        for i, label in enumerate(self.anchor_labels):
            label.setText(f'OAK {tc(anchors[i][0])}  ↔  RS {tc(anchors[i][1])}' if i < len(anchors) else '尚未设置')
        self.drift_box.setEnabled(len(anchors) == 2)
        self.drift_box.setChecked(self.project.data['drift_correction'])
        if anchors:
            a, b = self.project.mapping()
            error = self.project.drift_error()
            tail = '' if error is None else f'　|　第二点相对固定偏移误差：{error:+.3f} 秒'
            self.sync_label.setText(f'原片映射：RealSense = {a:.9f} × OAK {b:+.6f} 秒{tail}')
        else:
            self.sync_label.setText('尚未对齐。同步关系只由你选择的原片画面决定，不使用剪辑拼接时间或文件长度差。')
            self.link_box.setChecked(False)
        self.loading = False

    def set_roi(self, camera, roi):
        if not self.project.sources.get(camera):
            return
        self.mutation(lambda: self.project.set_roi(camera, roi))

    def roi_dialog(self, camera):
        source = self.project.sources.get(camera)
        if not source:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f'{camera} · 原片像素 ROI')
        form = QFormLayout(dialog)
        values = []
        for name, value, maximum in zip(['X', 'Y', '宽', '高'], source['roi'],
                                        [source['width'], source['height'], source['width'], source['height']]):
            box = QSpinBox()
            box.setRange(0, maximum)
            box.setValue(value)
            values.append(box)
            form.addRow(name, box)
        full = QPushButton('恢复全画幅')
        full.clicked.connect(lambda: [w.setValue(v) for w, v in zip(values, [0, 0, source['width'], source['height']])])
        form.addRow(full)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec():
            self.set_roi(camera, [w.value() for w in values])

    def save(self, *args):
        try:
            self.project.data['last_selected'] = max(0, self.list.currentRow())
            self.project.save(self.project_path)
            self.status.setText(f'已保存 · {self.project_path}')
            return True
        except Exception as exc:
            self.error(f'保存失败：{exc}')
            return False

    def save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, '精修工程另存为', str(self.project_path.parent / 'review_copy.json'), '精修工程 (*.json)')
        if not path:
            return
        try:
            target = output_path(path)
            if target == self.project_path:
                self.save()
                return
            lock = self.acquire_lock(target)
            try:
                self.project.save(target)
            except Exception:
                lock.unlock()
                raise
            self.lock.unlock()
            self.lock, self.project_path = lock, target
            self.status.setText(f'已另存为 · {target}')
        except Exception as exc:
            self.error(str(exc))

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开精修工程', str(self.project_path.parent), '精修工程 (*.json)')
        if path:
            try:
                self.replace_project(Project.load(path), Path(path))
            except Exception as exc:
                self.error(str(exc))

    def import_manifest(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择初筛 segments.json', 'D:/DataSet/exports', '初筛清单 (*.json)')
        if not path:
            return
        rs, _ = QFileDialog.getOpenFileName(self, '选择对应 RealSense 原视频（可取消，稍后定位）', 'G:/PHD Data Backup/Desktop Dataset', '视频 (*.mp4 *.mkv *.avi *.mov)')
        try:
            project = Project.from_manifest(path, probe(rs) if rs else None)
            target = Path('D:/DataSet/exports/video_refiner') / f'{Path(path).parent.name}_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
            self.replace_project(project, target)
            self.save()
        except Exception as exc:
            self.error(str(exc))

    def stop_readers(self):
        for worker in self.workers.values():
            worker.stop()
        for worker in self.workers.values():
            worker.wait()
            worker.deleteLater()
        self.workers.clear()

    def replace_project(self, project, path):
        if self.export_worker and self.export_worker.isRunning():
            raise ValueError('请等待当前导出结束，再切换工程。')
        path = output_path(path)
        if path == self.project_path:
            raise ValueError('此工程已经打开。')
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.acquire_lock(path)
        if not self.save():
            lock.unlock()
            return
        self.pause()
        self.stop_readers()
        self.lock.unlock()
        self.lock, self.project_path, self.project = lock, path, project
        self.history = History()
        self.selected_uid = None
        self.positions = {'oak': 0., 'rs': 0.}
        for c, pane in self.panes.items():
            self.generation[c] += 1
            pane.frame, pane.canvas.image = None, None
            source = project.sources.get(c)
            if source:
                pane.source = pane.canvas.source = source
                pane.slider.setRange(0, round(source['duration']*1000))
                pane.position.setRange(0, source['duration'])
        self.refresh_list()
        self.update_sync()
        self.start_readers()
        self.list.setCurrentRow(0)

    def locate_source(self, camera):
        old = self.project.sources.get(camera)
        path, _ = QFileDialog.getOpenFileName(self, f'定位 {camera} 原视频', str(Path(old['path']).parent) if old else 'G:/', '视频 (*.mp4 *.mkv *.avi *.mov)')
        if not path:
            return
        try:
            info = probe(path)
            same = old and all(info[k] == old[k] for k in ['duration', 'width', 'height'])
            if camera == 'oak' and any(s['end'] > info['duration'] for s in self.project.segments):
                raise ValueError('新原片太短，无法覆盖已有切段。')
            if same:
                info['roi'] = old['roi']
            self.pause()
            self.stop_readers()
            self.project.sources[camera] = info
            # A new file, even with identical duration, requires explicit re-alignment.
            if not old or Path(path).resolve() != Path(old['path']).resolve():
                self.project.set_sync([], False)
                for s in self.project.segments:
                    s['confirmed'] = False
            self.history = History()
            pane = self.panes[camera]
            pane.source = pane.canvas.source = info
            pane.frame, pane.canvas.image = None, None
            pane.slider.setRange(0, round(info['duration']*1000))
            pane.position.setRange(0, info['duration'])
            self.generation['oak'] += 1
            self.generation['rs'] += 1
            self.start_readers()
            self.restore_state()
        except Exception as exc:
            self.error(str(exc))

    def export_dialog(self):
        self.pause()
        if not self.apply_form() or not self.save():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('导出原片对应区间')
        dialog.resize(640, 250)
        form = QFormLayout(dialog)
        scope = QComboBox()
        scope.addItems(['全部保留片段', '仅已确认片段', '当前片段'])
        form.addRow('范围', scope)
        paired = QCheckBox('同时导出 OAK + RealSense（同编号文件）')
        paired.setChecked(bool(self.project.data['anchors']))
        form.addRow(paired)
        destination = QLineEdit(str(Path('D:/DataSet/exports/video_refined') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        form.addRow('新目录', destination)
        note = QLabel('30 fps · H.264 · 无音频 · 每段独立文件\n每次导出使用当前工程的快照；目录必须不存在。未同步时可仅导出 OAK。')
        note.setWordWrap(True)
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec():
            ids = (None if scope.currentIndex() == 0 else
                   [s['uid'] for s in self.project.segments if s['confirmed']] if scope.currentIndex() == 1 else
                   [self.current()['uid']] if self.current() else [])
            self.export_worker = ExportWorker(Project(self.project.data), destination.text(), ids, paired.isChecked(), self)
            self.export_worker.progress.connect(self.export_progress)
            self.export_worker.done.connect(self.export_done)
            self.export_worker.failed.connect(self.export_failed)
            self.export_button.setEnabled(False)
            self.cancel_button.show()
            self.progress.setValue(0)
            self.progress.show()
            self.export_worker.start()

    def export_progress(self, value, total, message):
        self.progress.setRange(0, total)
        self.progress.setValue(value)
        self.status.setText(f'导出 {value}/{total} · {message}')

    def export_done(self, path):
        self.export_button.setEnabled(True)
        self.cancel_button.hide()
        self.status.setText(f'导出完成 · {path}')

    def export_failed(self, text):
        self.export_button.setEnabled(True)
        self.cancel_button.hide()
        self.error(text)

    def cancel_export(self):
        if self.export_worker:
            self.export_worker.cancel.set()

    def error(self, text):
        self.status.setText(f'⚠ {text}')

    def help(self):
        path = Path(__file__).resolve().parents[2] / 'docs' / 'video_refiner_guide.md'
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def eventFilter(self, obj, event):
        if self.closed or not self.isActiveWindow() or event.type() != QEvent.Type.KeyPress:
            return False
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QAbstractSpinBox)):
            return False
        key, modifier = event.key(), event.modifiers()
        if modifier & Qt.KeyboardModifier.ControlModifier:
            action = {Qt.Key.Key_S: self.save, Qt.Key.Key_Z: self.undo, Qt.Key.Key_Y: self.redo}.get(key)
        else:
            action = {Qt.Key.Key_Space: lambda: self.toggle_play('oak'),
                      Qt.Key.Key_Left: lambda: self.step_frame('oak', -1),
                      Qt.Key.Key_Right: lambda: self.step_frame('oak', 1),
                      Qt.Key.Key_I: lambda: self.mark_boundary('start'),
                      Qt.Key.Key_O: lambda: self.mark_boundary('end'),
                      Qt.Key.Key_S: self.split_current,
                      Qt.Key.Key_Return: self.confirm_current,
                      Qt.Key.Key_Enter: self.confirm_current}.get(key)
        if action:
            action()
            return True
        return False

    def closeEvent(self, event):
        if self.closed:
            event.accept()
            return
        self.pause()
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.cancel.set()
            self.status.setText('正在结束导出，然后关闭窗口…')
            QTimer.singleShot(500, self.close)
            event.ignore()
            return
        self.autosave.stop()
        if not self.apply_form() or not self.save():
            event.ignore()
            return
        self.closed = True
        self.stop_readers()
        QApplication.instance().removeEventFilter(self)
        self.lock.unlock()
        event.accept()
