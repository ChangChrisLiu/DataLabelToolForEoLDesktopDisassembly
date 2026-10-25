"""Task U28 round 4: the detection requests' bookkeeping, a stored ROI, a reconfigured detector.

* B3: a request still queued when a newer one replaces it gets no answer --
  its pair is no longer marked as asked, and is asked again when it is back;
* N13: an ROI stored while a pair is up (no edit) rebuilds the pair, and the
  detections are asked with the new crops;
* N17: reconfiguring the detector never waits on the GUI thread for the old
  worker; the old worker's answer is not delivered.

The window scene of round 3 (U23's, on the scanner view, a stub detector); the
three connectors that come out on 8, 9 and 12 make three pairs of different
frames: (7, 8), (8, 9), (11, 12).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

from PySide6.QtWidgets import QApplication

from app_scene import COOLER, DESKTOP
from test_u23_locator import STEP, goto, select
from test_u28_round3 import _no_disk_cache, arm, det, qapp, settle, window  # noqa: F401

CONNECTORS = ("connector.01", "connector.02", "connector.fan.01")
#: Per step, a connector detection that is gone a step later.
TABLE = {7: [det((5, 5, 9, 9), cls="connector")], 8: [det((20, 5, 24, 9), cls="connector")],
         9: [], 11: [det((40, 5, 44, 9), cls="connector")], 12: []}


def _busy(detector, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not detector._busy and time.monotonic() < deadline:
        time.sleep(0.005)
    assert detector._busy


def test_a_request_replaced_in_the_queue_is_asked_again(window, tmp_path):
    stub = arm(window, tmp_path, TABLE, classes=("connector",))
    _no_disk_cache(tmp_path)                      # the model has to run ...
    stub.gate = threading.Event()                 # ... and the first job is held there
    pairs = [window.locator_source.pair(DESKTOP, k) for k in CONNECTORS]
    keys = [window._locator_det_key(p) for p in pairs]
    assert None not in keys and len(set(keys)) == 3
    assert [p.steps for p in pairs] == [(7, 8), (8, 9), (11, 12)]
    window._locator_detected(pairs[0])
    _busy(window._locator_det)                    # the first one is running
    window._locator_detected(pairs[1])            # queued ...
    window._locator_detected(pairs[2])            # ... and replaced by this one
    assert window.locator_det_requests == 3
    assert keys[1] not in window._locator_det_asked          # no answer will come
    assert {keys[0], keys[2]} <= window._locator_det_asked
    stub.gate.set()
    settle(window)
    assert keys[0] in window._locator_det_answers and keys[2] in window._locator_det_answers
    assert keys[1] not in window._locator_det_answers
    # back to the second pair: asked again, and it gets its chip
    window._locator_detected(pairs[1])
    assert window.locator_det_requests == 4
    settle(window)
    again = window._locator_detected(pairs[1])
    assert (20.0, 5.0, 24.0, 9.0) in again.candidates
    assert not window._locator_det_asked


def test_a_stored_roi_rebuilds_the_pair_on_the_next_frame(window, tmp_path):
    window.act_set_view("scan")
    QApplication.processEvents()
    select(window, COOLER)                        # never drawn in the scanner view
    assert window.locator.pair() is not None
    count = window.locator_refreshes
    goto(window, STEP - 1)                        # same part, nothing stored: no rebuild
    assert window.locator_refreshes == count
    window.db.set_pose_segment_roi(DESKTOP, "scan", 1, [4, 4, 50, 50], annotator="tester",
                                   hw=(64, 64))   # an ROI stored is no edit
    goto(window, STEP - 2)
    assert window.locator_target() == COOLER
    assert window.locator_refreshes == count + 1
    assert window.locator.pair().roi == (4.0, 4.0, 50.0, 50.0)


def test_a_new_roi_asks_the_detections_with_the_new_crops(window, tmp_path):
    arm(window, tmp_path, TABLE, classes=("connector",))
    pair = window.locator_source.pair(DESKTOP, "connector.01")
    before = window._locator_det_key(pair)
    window.db.set_pose_segment_roi(DESKTOP, "scan", 1, [2, 2, 60, 60], annotator="tester",
                                   hw=(64, 64))
    after = window._locator_det_key(window.locator_source.pair(DESKTOP, "connector.01"))
    assert before != after and after[4] == ((2, 2, 60, 60), (2, 2, 60, 60))


def test_reconfiguring_the_detector_never_waits_for_the_old_worker(window, tmp_path):
    stub = arm(window, tmp_path, TABLE, classes=("connector",))
    _no_disk_cache(tmp_path)
    stub.gate = threading.Event()
    pair = window.locator_source.pair(DESKTOP, "connector.01")
    window._locator_detected(pair)
    old = window._locator_det
    _busy(old)                                    # the old worker is in the model
    arm(window, tmp_path / "again", TABLE, classes=("connector",))   # a new detector
    started = time.perf_counter()
    new = window.locator_detector()
    took = time.perf_counter() - started
    assert new is not old and took < 0.5, took
    assert not window._locator_det_asked          # forgotten: asked again when wanted
    stub.gate.set()
    old._thread.join(10.0)
    for _ in range(3):
        QApplication.processEvents()
    assert not old._thread.is_alive()
    key = window._locator_det_key(pair)
    assert key not in window._locator_det_answers  # the old worker's answer never came
