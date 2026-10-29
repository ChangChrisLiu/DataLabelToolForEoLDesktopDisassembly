"""Background decoding and rendering, with latest-request seeking."""
import threading
from PySide6.QtCore import QThread, Signal

from .media import FrameReader
from .export import export_project


class DecodeWorker(QThread):
    frame_ready = Signal(str, object, int, str)
    failed = Signal(str, str)

    def __init__(self, camera, path, parent=None):
        super().__init__(parent)
        self.camera, self.path = camera, path
        self.condition = threading.Condition()
        self.pending = None
        self.stopping = False

    def request(self, seconds, generation, mode='seek'):
        with self.condition:
            self.pending = seconds, generation, mode
            self.condition.notify()

    def stop(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()

    def run(self):
        try:
            with FrameReader(self.path, max_width=1280) as reader:
                while True:
                    with self.condition:
                        while self.pending is None and not self.stopping:
                            self.condition.wait()
                        if self.stopping:
                            return
                        seconds, generation, mode = self.pending
                        self.pending = None
                    frame = (reader.previous(seconds) if mode == 'previous' else
                             reader.next(seconds) if mode == 'next' else reader.frame_at(seconds))
                    self.frame_ready.emit(self.camera, frame, generation, mode)
        except Exception as exc:
            self.failed.emit(self.camera, str(exc))


class ExportWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, project, directory, ids, paired, parent=None):
        super().__init__(parent)
        self.args = project, directory, ids, paired
        self.cancel = threading.Event()

    def run(self):
        try:
            path = export_project(*self.args, cancel=self.cancel, progress=self.progress.emit)
            self.done.emit(str(path))
        except Exception as exc:
            self.failed.emit(str(exc))
