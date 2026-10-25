"""Task U29 (E): the 「拆前 / 拆后」 detection chips' worker, U28 review N14 / N18.

* N14 -- the dock's engine read the detector's disk cache once per desktop, so
  every frame the window's own pass wrote afterwards was computed a second
  time, by a second model; and it never let go of its decoded frames (two
  12 MP pictures, ~73 MB) or the model's GPU cache. Now the file is read again
  when it changed, before anything is computed, and the worker releases both
  whenever nothing is queued.
* N18 -- the dock built its second RF-DETR under the ``heavy`` lock whenever it
  first needed it, and a heavy load at that moment (SAM's import, the window's
  detector) queued behind it (2.7 s in the round-4 probe). Now its model is
  built only once the window's own detector has gone idle for the first time
  (its first pass over the view on screen done); until then the dock answers
  from the cache, which that pass is filling.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from pathlib import Path

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from tda.models.det_engine import DetectionEngine, PlanItem
from tda.models.detector import Det, DetectorConfig
from tda.models.load_gate import HEAVY
from tda.ui.locator_dets import LocatorDetections, ReadOnlyEngine

GONE = (5.0, 5.0, 9.0, 9.0)
DESKTOP = 13


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def det(box, cls="screw", conf=0.8) -> Det:
    return Det(tuple(float(v) for v in box), cls, conf)


class StubModel:
    names = {0: "screw"}
    identity = "u29-stub"

    def __init__(self, table) -> None:
        self.table = dict(table)
        self.calls: list[int] = []
        self.released = 0

    def detect(self, img, crop, view, step=None):
        self.calls.append(int(step))
        return list(self.table.get(int(step), []))

    def release(self) -> None:
        self.released += 1


def config(tmp_path: Path) -> DetectorConfig:
    return DetectorConfig(source=tmp_path / "detector.yaml", model=tmp_path / "stub.pt",
                          classes=("screw",), views=("scan",), conf=0.10, roi_crop=False,
                          work_scale=(("scan", 1.0),), tile=640, stride=512, yield_ms=0.0,
                          cache_root=tmp_path / "det")


def frames(tmp_path: Path, steps) -> dict[int, PlanItem]:
    out = {}
    for step in steps:
        path = tmp_path / f"scan_{step}.png"
        image = QImage(64, 48, QImage.Format.Format_RGB888)
        image.fill(QColor(10 * step, 10 * step, 10 * step))
        image.save(str(path))
        out[step] = PlanItem(step=step, candidates=(str(path),), crop=None)
    return out


def window_pass(tmp_path: Path, model: StubModel, items) -> None:
    """What the window's own detector does: compute the frames and write the file."""
    engine = DetectionEngine(config(tmp_path), tmp_path / "det", factory=lambda _c: model,
                             identity=model.identity)
    engine.prepare()
    engine.open_view(DESKTOP, "scan")
    engine.load_model()
    for item in items:
        engine.process(item)
    assert engine.flush() is not None
    time.sleep(0.02)                     # a new mtime even on a coarse clock


def dock(tmp_path: Path, model: StubModel, ready=None) -> tuple[LocatorDetections, list]:
    built: list[int] = []

    def factory(_c):
        built.append(1)
        return model

    engine = ReadOnlyEngine(config(tmp_path), tmp_path / "det", factory=factory,
                            identity=model.identity)
    worker = LocatorDetections(engine, 0.10, ready=ready)
    answers: list = []
    worker.sigDone.connect(lambda token, found: answers.append((token, found)))
    worker.built = built
    return worker, answers


def drain(worker: LocatorDetections) -> None:
    assert worker.wait(20.0)
    for _ in range(3):
        QApplication.processEvents()


TABLE = {1: [det(GONE)], 2: [], 3: [det(GONE)], 4: []}


# --------------------------------------------------------------------------- #
# N14: read the file again; let go of the frames
# --------------------------------------------------------------------------- #
def test_frames_the_window_wrote_later_are_read_not_computed(qapp, tmp_path):
    items = frames(tmp_path, (1, 2, 3, 4))
    window_model, dock_model = StubModel(TABLE), StubModel(TABLE)
    window_pass(tmp_path, window_model, [items[1], items[2]])
    worker, answers = dock(tmp_path, dock_model)
    try:
        worker.request("a", DESKTOP, items[1], items[2], "screw")
        drain(worker)
        assert worker.stats["cached"] == 2 and worker.stats["reread"] == 0
        # the window's pass goes on and writes 3 and 4 after the dock read the file
        window_pass(tmp_path, window_model, [items[3], items[4]])
        worker.request("b", DESKTOP, items[3], items[4], "screw")
        drain(worker)
        assert worker.stats["detected"] == 0 and worker.stats["cached"] == 4
        assert worker.stats["reread"] == 1
        assert dock_model.calls == [] and worker.built == []      # no second model
        assert [t for t, _f in answers] == ["a", "b"]
        assert [d.box for d in answers[1][1]] == [GONE]
    finally:
        worker.shutdown()


def test_what_the_dock_computed_survives_a_reread(qapp, tmp_path):
    items = frames(tmp_path, (1, 2, 3, 4))
    window_model, dock_model = StubModel(TABLE), StubModel(TABLE)
    worker, answers = dock(tmp_path, dock_model)
    try:
        worker.request("a", DESKTOP, items[1], items[2], "screw")     # nothing on disk
        drain(worker)
        assert dock_model.calls == [1, 2]
        window_pass(tmp_path, window_model, [items[3], items[4]])
        worker.request("b", DESKTOP, items[3], items[4], "screw")
        drain(worker)
        worker.request("c", DESKTOP, items[1], items[2], "screw")     # its own, kept
        drain(worker)
        assert dock_model.calls == [1, 2] and worker.stats["reread"] == 1
        # and it still wrote nothing
        assert sorted(p.name for p in (tmp_path / "det").rglob("*.json")) == ["13_scan.json"]
    finally:
        worker.shutdown()


def test_the_frames_and_the_gpu_cache_go_when_nothing_is_queued(qapp, tmp_path):
    items = frames(tmp_path, (1, 2))
    model = StubModel(TABLE)
    worker, _answers = dock(tmp_path, model)
    try:
        worker.request("a", DESKTOP, items[1], items[2], "screw")
        drain(worker)
        assert model.calls == [1, 2]
        assert worker.engine._decoded == {} and model.released == 1
        assert worker.stats["released"] == 1
    finally:
        worker.shutdown()


# --------------------------------------------------------------------------- #
# N18: the dock's model waits for the window's first pass
# --------------------------------------------------------------------------- #
def test_the_model_waits_for_the_window_and_is_not_built_when_the_pass_fills_the_cache(
        qapp, tmp_path):
    items = frames(tmp_path, (1, 2))
    window_model, dock_model = StubModel(TABLE), StubModel(TABLE)
    idle = threading.Event()
    worker, answers = dock(tmp_path, dock_model, ready=idle.is_set)
    try:
        worker.request("a", DESKTOP, items[1], items[2], "screw")
        time.sleep(0.4)
        assert worker.pending() and answers == [] and worker.built == []
        # meanwhile nobody holds the heavy lock on the dock's account
        assert HEAVY.acquire(timeout=0.5)
        HEAVY.release()
        window_pass(tmp_path, window_model, [items[1], items[2]])  # the window's first pass
        idle.set()
        drain(worker)
        assert worker.built == [] and dock_model.calls == []       # answered from the file
        assert worker.stats["cached"] == 2 and worker.stats["waited"] > 0.3
        assert [d.box for d in answers[0][1]] == [GONE]
        # the turn is taken once: later frames the file lacks are computed at once
        idle.clear()
        more = frames(tmp_path, (3, 4))
        worker.request("b", DESKTOP, more[3], more[4], "screw")
        drain(worker)
        assert worker.built == [1] and dock_model.calls == [3, 4]
    finally:
        worker.shutdown()


def test_a_job_waiting_for_its_turn_gives_way_to_a_newer_request(qapp, tmp_path):
    items = frames(tmp_path, (1, 2, 3, 4))
    model = StubModel(TABLE)
    idle = threading.Event()
    worker, answers = dock(tmp_path, model, ready=idle.is_set)
    try:
        worker.request("a", DESKTOP, items[1], items[2], "screw")
        time.sleep(0.3)
        assert worker.request("b", DESKTOP, items[3], items[4], "screw") is None
        time.sleep(0.3)
        idle.set()
        drain(worker)
        assert answers[0] == ("a", None)                    # no answer: asked again later
        assert answers[1][0] == "b" and [d.box for d in answers[1][1]] == [GONE]
        assert worker.stats["abandoned"] == 1 and worker.stats["failed"] == 0
        assert model.calls == [3, 4]
    finally:
        worker.shutdown()


def test_in_the_window_the_dock_waits_for_the_first_pass_and_reads_it(qapp, tmp_path):
    """The reviewer's set-up (U28 round 4, N18): the window's detector still on its
    first pass when the dock asks. The dock builds no model and takes no heavy
    lock; once the pass is done its frames come from the file it wrote."""
    import test_u28_round3 as R3
    from app_scene import COOLER, close_window
    from test_u23_locator import STEP, goto, make_scene, open_window, select

    session, paths = make_scene(tmp_path, own_skip=(COOLER,))
    session.open(13, "scan", force=True)
    win = open_window(session, paths)
    try:
        goto(win, STEP)
        stub = R3.StubModel({12: [R3.det(R3.GONE)], 13: []})
        stub.gate = threading.Event()             # the window's pass is held
        win.enable_detector(R3.config(tmp_path), factory=lambda _c: stub,
                            cache_root=str(tmp_path / "det"))
        QApplication.processEvents()
        select(win, R3.SCREW)
        dock_worker = win._locator_det
        assert dock_worker is not None and win.locator_det_requests == 1
        time.sleep(0.3)
        assert dock_worker.pending() and dock_worker.engine.model is None
        assert not win.window_detector_idle()
        assert HEAVY.acquire(timeout=0.5)         # nobody holds it on the dock's account
        HEAVY.release()
        stub.gate.set()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and win.det_worker.pending():
            QApplication.processEvents()
            time.sleep(0.01)
        R3.settle(win)
        assert dock_worker.engine.model is None   # never built: the pass filled the file
        assert dock_worker.stats["detected"] == 0 and dock_worker.stats["cached"] == 2
        assert win.locator.pair().candidates[1] == R3.GONE
    finally:
        close_window(win)


def test_the_window_says_when_its_detector_is_idle(qapp):
    from tda.ui.app_locator import LocatorMixin

    class Busy:
        def __init__(self, busy):
            self.busy = busy

        def pending(self):
            if self.busy is None:
                raise RuntimeError("gone")
            return self.busy

    class Win(LocatorMixin):
        det_worker = None

    win = Win()
    assert win.window_detector_idle()
    win.det_worker = Busy(True)
    assert not win.window_detector_idle()
    win.det_worker = Busy(False)
    assert win.window_detector_idle()
    win.det_worker = Busy(None)
    assert win.window_detector_idle()
