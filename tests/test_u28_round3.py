"""Task U28 round 3: vanished detections as extra 「拆前 / 拆后」 candidates.

* the rule: a detection of the part's class in frame r-1 with no detection of
  that class overlapping it in r; best first, at most two; a class the
  detector does not know gets none;
* the order: diff #1, the detections, diff #2 / #3 -- never a detection first
  (with no region of the difference the frames stay whole and the detections
  are only offered);
* where they come from: the detector's disk cache when it has the two frames,
  read and never written, else the model on a worker thread; an answer for a
  pair that is no longer up is dropped; nothing while the dock is down; a
  failure is no chips and no message.

The window scene is U23's on the scanner view (D13: the four captive cooler
screws leave with the fan at 13, so a screw's pair is frames 12 / 13 and its
window a guess from the difference), with a stub detector whose answers per
step the test sets.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_scene import COOLER, SCREWS, close_window
from test_u23_locator import STEP, goto, make_scene, open_window, select
from tda.models.det_engine import PlanItem
from tda.models.detector import Det, DetectorConfig
from tda.ui.locator_dets import (
    DET_LABEL,
    DET_NOTE,
    MAX_CANDIDATES,
    ReadOnlyEngine,
    merge_candidates,
    vanished,
)
from tda.ui.locator_pair import NO_DIFF, PairScene
from tda.ui.panels.locator import LocatorPanel

SCREW = SCREWS[2]
GONE = (5.0, 5.0, 9.0, 9.0)              # a screw in 12, not in 13
STAYS = (50.0, 50.0, 54.0, 54.0)         # a screw in both


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def det(box, cls="screw", conf=0.8) -> Det:
    return Det(tuple(float(v) for v in box), cls, conf)


# --------------------------------------------------------------------------- #
# the rule and the order
# --------------------------------------------------------------------------- #
def test_a_detection_that_went_and_one_that_stayed():
    before = [det(GONE, conf=0.6), det(STAYS, conf=0.9), det((20, 20, 24, 24), conf=0.7),
              det((30, 30, 34, 34), cls="connector", conf=0.95),     # another class
              det((40, 40, 44, 44), conf=0.04)]                       # under the floor
    after = [det((51, 50, 55, 54), conf=0.9)]                         # STAYS, moved a pixel
    found = vanished(before, after, "screw", 0.05)
    assert [d.box for d in found] == [(20.0, 20.0, 24.0, 24.0), GONE]      # best first
    assert vanished(before, after, "connector", 0.05)[0].cls == "connector"
    assert vanished(before, after, "cpu", 0.05) == []
    many = [det((i * 10, 0, i * 10 + 4, 4), conf=0.5 + i / 100) for i in range(5)]
    assert len(vanished(many, [], "screw", 0.05)) == 2


def test_the_order_is_diff_first_then_the_detections():
    diffs = [(0, 0, 10, 10), (20, 20, 30, 30), (40, 40, 50, 50)]
    windows, labels = merge_candidates(diffs, [det(GONE), det(STAYS)], "检测到消失的 螺丝")
    assert windows == (diffs[0], GONE, STAYS, diffs[1], diffs[2])
    assert labels == ("", "检测到消失的 螺丝", "检测到消失的 螺丝", "", "")
    assert len(windows) == MAX_CANDIDATES
    windows, labels = merge_candidates([], [det(GONE)], "x")
    assert windows == (GONE,) and labels == ("x",)


def _picture(value: int) -> QImage:
    image = QImage(400, 300, QImage.Format.Format_RGB888)
    image.fill(QColor(value, value, value))
    return image


def test_with_no_region_of_the_difference_the_detections_are_only_offered(qapp):
    pair = PairScene(desktop=13, instance=SCREW, steps=(12, 13), asked=(12, 13),
                     images=(_picture(90), _picture(120)), frame_hw=((300, 400), (300, 400)),
                     captions=("拆前 第 12 帧", "拆后 第 13 帧"), note=NO_DIFF, compared=True,
                     cls="screw")
    pair = pair.with_detections([det((100, 100, 120, 120)), det((300, 50, 320, 70))],
                                DET_LABEL.format(cls="螺丝"))
    assert pair.window is None and len(pair.candidates) == 2
    panel = LocatorPanel()
    panel.resize(320, 300)
    panel.show()
    try:
        panel.show_pair(pair)
        QApplication.processEvents()
        view = panel.pair_view
        assert view.window() is None and view.candidate_index() == -1     # never first
        assert panel.candidates.isVisible() and panel.candidate_text() == "候选 2 处"
        assert panel.note_text() == NO_DIFF
        index, rect = panel.candidates.chips()[0]
        assert panel.candidates.label_at(rect.center().toPoint()) == "检测到消失的 螺丝"
        QTest.mouseClick(panel.candidates, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, rect.center().toPoint())
        assert view.candidate_index() == index == 0
        assert view.window() == (100.0, 100.0, 120.0, 120.0) and view.is_detection()
        assert panel.note_text() == DET_NOTE.format(cls="螺丝", step=13)
        assert panel.note_text() == "检测到一个 螺丝 消失了"      # a finding, not a step
        assert panel.candidate_text() == "候选 1/2"
        assert not panel.grab().isNull()
    finally:
        panel.close()
        panel.deleteLater()


# --------------------------------------------------------------------------- #
# the engine never writes the shared cache
# --------------------------------------------------------------------------- #
class StubModel:
    names = {0: "screw", 1: "connector"}
    identity = "u28-stub"

    def __init__(self, table=None) -> None:
        #: ``step -> [Det]`` for the scanner view.
        self.table = dict(table or {})
        self.calls: list[int] = []
        self.gate: threading.Event | None = None
        self.fail = False

    def detect(self, img, crop, view, step=None):
        if self.gate is not None:
            self.gate.wait(10.0)
        if self.fail:
            raise RuntimeError("the stub's GPU is on fire")
        self.calls.append(int(step))
        return list(self.table.get(int(step), []))


def config(tmp_path: Path, classes=("screw",)) -> DetectorConfig:
    return DetectorConfig(source=tmp_path / "detector.yaml", model=tmp_path / "stub.pt",
                          classes=tuple(classes), views=("scan",), conf=0.10, roi_crop=True,
                          work_scale=(("scan", 1.0),), tile=640, stride=512, yield_ms=0.0,
                          cache_root=tmp_path / "det")


def test_the_locator_engine_writes_nothing(qapp, tmp_path):
    stub = StubModel({1: [det(GONE)]})
    engine = ReadOnlyEngine(config(tmp_path), tmp_path / "det", factory=lambda _c: stub)
    engine.prepare()
    engine.open_view(13, "scan")
    image = tmp_path / "frame.png"
    _picture(80).save(str(image))
    frame = engine.process(PlanItem(step=1, candidates=(str(image),), crop=None))
    assert frame.dets and stub.calls == [1]
    assert engine.flush() is None
    assert not (tmp_path / "det").exists() or not any((tmp_path / "det").rglob("*.json"))


# --------------------------------------------------------------------------- #
# in the window
# --------------------------------------------------------------------------- #
@pytest.fixture
def window(qapp, tmp_path: Path):
    session, paths = make_scene(tmp_path, own_skip=(COOLER,))
    session.open(13, "scan", force=True)
    win = open_window(session, paths)
    goto(win, STEP)
    yield win
    close_window(win)


def arm(win, tmp_path: Path, table, classes=("screw",)) -> StubModel:
    stub = StubModel(table)
    win.enable_detector(config(tmp_path, classes), factory=lambda _c: stub,
                        cache_root=str(tmp_path / "det"))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and win.det_worker.pending():
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    return stub


def settle(win) -> None:
    detector = win._locator_det
    if detector is not None:
        assert detector.wait(30.0)
    for _ in range(3):
        QApplication.processEvents()


def test_a_vanished_detection_becomes_chip_2(window, tmp_path):
    arm(window, tmp_path, {12: [det(GONE, conf=0.7), det(STAYS, conf=0.9)],
                           13: [det((50.5, 50.0, 54.5, 54.0), conf=0.9)]})
    select(window, SCREW)                       # never drawn in the scanner view
    pair = window.locator.pair()
    assert pair is not None and pair.window_source == "diff"
    first = pair.window
    settle(window)
    pair = window.locator.pair()
    assert window.locator_det_requests == 1
    assert pair.candidates[0] == first                  # diff #1 stays first, on screen
    assert pair.candidates[1] == GONE                   # the vanished screw is chip #2
    assert pair.candidate_labels[1] == "检测到消失的 螺丝"
    assert STAYS not in pair.candidates                 # present in both frames: no chip
    view, bar = window.locator.pair_view, window.locator.candidates
    assert view.candidate_index() == 0 and view.window() == first
    index, rect = bar.chips()[0]
    assert index == 1
    bar.resize(max(bar.width(), 240), bar.height())
    index, rect = bar.chips()[0]
    QTest.mouseClick(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     rect.center().toPoint())
    assert view.window() == GONE and view.is_detection()
    assert window.locator.note_text() == "检测到一个 螺丝 消失了"
    # the same pair again: from the answers kept, no second request
    select(window, "chassis")
    select(window, SCREW)
    assert window.locator.pair().candidates[1] == GONE
    assert window.locator_det_requests == 1


def test_the_disk_cache_answers_when_it_has_the_two_frames(window, tmp_path):
    stub = arm(window, tmp_path, {12: [det(GONE)], 13: []})
    seen = list(stub.calls)                     # the window's own pass, written to disk
    assert 12 in seen and 13 in seen
    calls = len(stub.calls)
    select(window, SCREW)
    settle(window)
    stats = window._locator_det.stats
    assert stats["cached"] == 2 and stats["detected"] == 0      # no model run for it
    assert len(stub.calls) == calls
    assert window.locator.pair().candidates[1] == GONE


def test_a_class_the_detector_does_not_know_gets_no_chips(window, tmp_path):
    # the detector knows connectors only: a screw asks for nothing
    arm(window, tmp_path, {12: [det(GONE, cls="connector")], 13: []}, classes=("connector",))
    select(window, SCREW)
    settle(window)
    assert window.locator_det_requests == 0
    assert all(label == "" for label in window.locator.pair().candidate_labels)


def test_detections_of_another_class_are_not_the_parts(window, tmp_path):
    arm(window, tmp_path, {12: [det(GONE, cls="connector")], 13: []},
        classes=("screw", "connector"))
    select(window, SCREW)
    settle(window)
    assert window.locator_det_requests == 1
    assert GONE not in window.locator.pair().candidates


def _no_disk_cache(tmp_path: Path) -> None:
    for path in (tmp_path / "det").rglob("*.json"):
        path.unlink()


def test_an_answer_for_a_pair_no_longer_up_is_dropped(window, tmp_path):
    stub = arm(window, tmp_path, {12: [det(GONE)], 13: []})
    _no_disk_cache(tmp_path)                     # the locator's model has to run ...
    stub.gate = threading.Event()                # ... and is held there
    select(window, SCREW)
    assert window.locator_det_requests == 1
    select(window, COOLER)                       # the dock moves on to the fan's pair
    stub.gate.set()
    settle(window)
    assert window.locator.pair().instance == COOLER
    assert window.locator_det_dropped == 1
    assert all(label == "" for label in window.locator.pair().candidate_labels)
    # the answer is kept: back to the screw, its chips come from it at once
    select(window, SCREW)
    assert window.locator.pair().candidates[1] == GONE
    assert window.locator_det_requests == 1


def test_another_screw_of_the_same_two_frames_takes_the_answer(window, tmp_path):
    stub = arm(window, tmp_path, {12: [det(GONE)], 13: []})
    _no_disk_cache(tmp_path)
    stub.gate = threading.Event()
    select(window, SCREW)
    select(window, SCREWS[3])                    # leaves on 13 too: same frames, same class
    stub.gate.set()
    settle(window)
    pair = window.locator.pair()
    assert pair.instance == SCREWS[3] and pair.candidates[1] == GONE
    assert window.locator_det_dropped == 0


def test_nothing_is_asked_while_the_dock_is_down(window, tmp_path):
    arm(window, tmp_path, {12: [det(GONE)], 13: []})
    window.locator_dock.close()
    QApplication.processEvents()
    select(window, SCREW)
    goto(window, STEP - 1)
    assert window.locator_det_requests == 0 and window._locator_det is None


def test_a_detector_failure_is_silent(window, tmp_path):
    stub = arm(window, tmp_path, {12: [det(GONE)], 13: []})
    _no_disk_cache(tmp_path)
    stub.fail = True
    select(window, SCREW)
    settle(window)
    assert window._locator_det.stats["failed"] == 1
    assert all(label == "" for label in window.locator.pair().candidate_labels)
    assert window.locator.note_text() == window.locator.pair().note    # nothing said
