"""Vanished detections as 「拆前 / 拆后」 candidates (task U28, round 3).

When the window of a pair is a guess from the difference of the two scanner
frames, the shipped small-part detector can offer more places to look: a
detection of the part's class in frame ``r - 1`` with no detection of that
class overlapping it in frame ``r`` -- a screw that was there and is not.
Measured on the annotated desktops (round 2-3), such detections are never
good enough to be the first window, but as extra candidates they lift
"one of the candidates holds the part" for single-change screw and
connector steps from 54 % to 82 % (91 % counting the steps where the
difference finds nothing).  So:

* :func:`vanished` -- which detections went (class, confidence, no
  overlapping detection of the class in ``r``), best first, at most
  :data:`MAX_DET_CHIPS`;
* :func:`merge_candidates` -- the order the panel offers them in: the
  difference's largest region first, then the detections, then the
  difference's second and third; never a detection first;
* :class:`LocatorDetections` -- the detections of the two frames, off the GUI
  thread: the detector's on-disk cache when it has them (read, never
  written), else the model, through the same
  :class:`~tda.models.det_engine.DetectionEngine` the reverse walk's
  suggestions use.  One job at a time, the newest request replacing a queued
  one; the answer comes back on the GUI thread with the request's token, and
  the window drops it if the dock has moved on.  Any failure is an empty
  answer, said nowhere but the log.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional, Sequence

from PySide6.QtCore import QObject, Qt, Signal

from tda.models.det_engine import DetectionEngine, PlanItem
from tda.models.load_gate import Turn, heavy

__all__ = ["DET_LABEL", "DET_NOTE", "LocatorDetections", "MAX_CANDIDATES", "MAX_DET_CHIPS",
           "ReadOnlyEngine", "merge_candidates", "overlaps", "vanished"]

log = logging.getLogger(__name__)

#: At most this many vanished detections are offered ...
MAX_DET_CHIPS = 2
#: ... and at most this many candidates in all (one on screen, four chips).
MAX_CANDIDATES = 5
#: A detection chip's label, and the note while it is the window on screen --
#: the detector's finding, not a fact about the step (round 4, N15: a captive
#: screw it stops finding at r may well still be in its hole until r + 2).
DET_LABEL = "检测到消失的 {cls}"
DET_NOTE = "检测到一个 {cls} 消失了"
VIEW = "scan"
#: How often an idle worker asks whether its window has closed.
IDLE_CHECK_S = 1.0


def overlaps(p, q) -> bool:
    """Do two boxes ``(x0, y0, x1, y1)`` share any area?"""
    return min(p[2], q[2]) > max(p[0], q[0]) and min(p[3], q[3]) > max(p[1], q[1])


def vanished(before: Sequence[Any], after: Sequence[Any], cls: str, conf: float,
             limit: int = MAX_DET_CHIPS) -> list:
    """The detections of ``cls`` in ``before`` that ``after`` no longer has.

    Only detections at ``conf`` or above count, on both sides; one in
    ``before`` has gone when no detection of the class in ``after`` overlaps
    it.  Best first (by confidence), at most ``limit``.
    """
    mine = [d for d in before if d.cls == cls and d.conf >= conf]
    theirs = [d for d in after if d.cls == cls and d.conf >= conf]
    gone = [d for d in mine if not any(overlaps(d.box, t.box) for t in theirs)]
    return sorted(gone, key=lambda d: -d.conf)[:max(0, int(limit))]


def merge_candidates(diffs: Sequence[tuple], dets: Sequence[Any], label: str
                     ) -> tuple[tuple, tuple]:
    """``(windows, labels)``: diff #1, the detections, diff #2 / #3 -- capped.

    ``labels`` holds ``""`` for a region of the difference and ``label`` (the
    chip's 「检测到消失的 …」) for a detection.  With no region of the difference
    the detections are the only candidates -- and the panel then shows the
    two frames whole and offers them as chips: a detection is never first.
    """
    windows = list(diffs[:1]) + [tuple(float(v) for v in d.box) for d in dets] + list(diffs[1:3])
    labels = [""] * len(diffs[:1]) + [label] * len(dets) + [""] * len(diffs[1:3])
    return tuple(windows[:MAX_CANDIDATES]), tuple(labels[:MAX_CANDIDATES])


class ReadOnlyEngine(DetectionEngine):
    """The suggestions' engine that never writes the shared on-disk cache.

    What it computes stays in its memory for the session (keyed by frame and
    file stamp, like the file); the background pass of the window owns the file.
    """

    def flush(self):  # noqa: D102 - nothing is written
        return None


class _Bridge(QObject):
    """Carries the worker's answer onto the GUI thread (a queued connection)."""

    sigDone = Signal(object, object)


class LocatorDetections(QObject):
    """The two frames' vanished detections, one request at a time, off the GUI thread.

    Signals:
        sigDone: ``(token, detections)`` on the GUI thread -- a list of
            :class:`~tda.models.detector.Det` (maybe empty), or ``None`` when
            the detector failed.
    """

    sigDone = Signal(object, object)

    def __init__(self, engine: DetectionEngine, conf: float, parent: Optional[QObject] = None,
                 after: Optional[Turn] = None,
                 stop_when: Optional[Callable[[], bool]] = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.conf = float(conf)
        self._after = after
        #: Asked about once a second while idle: the window closed, so stop.
        self._stop_when = stop_when
        self._lock = threading.Condition()
        self._job: Optional[tuple] = None
        self._busy = False
        self._stopped = False
        self._prepared = False
        self._opened: Optional[int] = None
        #: Jobs run, frames the model was run on (tests, the report).
        self.stats = {"jobs": 0, "detected": 0, "cached": 0, "failed": 0}
        self._bridge = _Bridge(self)
        self._bridge.sigDone.connect(self.sigDone, Qt.ConnectionType.QueuedConnection)
        self._thread = threading.Thread(target=self._run, name="tda-locator-detect",
                                        daemon=True)
        self._thread.start()

    # -- requests (GUI thread) ------------------------------------------------
    def request(self, token: Any, desktop: int, before: PlanItem, after: PlanItem,
                cls: str) -> Optional[Any]:
        """Ask for the vanished ``cls`` detections between two scanner frames.

        Returns the token of a request that will get **no** answer: the one
        still queued that this one replaces, or this one when the worker has
        stopped -- the caller forgets it was asked, so it is asked again the
        next time it is wanted (round 4, B3).
        """
        with self._lock:
            if self._stopped:
                return token
            displaced = self._job[0] if self._job is not None else None
            self._job = (token, int(desktop), before, after, str(cls))
            self._lock.notify_all()
            return displaced

    def pending(self) -> bool:
        with self._lock:
            return self._busy or self._job is not None

    def wait(self, timeout: float = 30.0) -> bool:
        """Block until nothing is queued or running (tests, probes)."""
        import time

        deadline = time.perf_counter() + float(timeout)
        with self._lock:
            while self._busy or self._job is not None:
                left = deadline - time.perf_counter()
                if left <= 0:
                    return False
                self._lock.wait(left)
        return True

    def shutdown(self, timeout: float = 10.0, wait: bool = True) -> None:
        """Stop; ``wait=False`` lets a job in hand finish on its own (no answer is sent)."""
        with self._lock:
            self._stopped = True
            self._job = None
            self._lock.notify_all()
        if wait and self._thread.is_alive():
            self._thread.join(timeout)

    # -- the worker -------------------------------------------------------------
    def _closing(self) -> bool:
        try:
            return bool(self._stop_when is not None and self._stop_when())
        except Exception:  # noqa: BLE001 - a window half torn down is a closed one
            return True

    def _run(self) -> None:
        while True:
            with self._lock:
                while self._job is None and not self._stopped:
                    self._lock.wait(IDLE_CHECK_S)
                    if self._job is None and self._closing():
                        self._stopped = True
                if self._stopped:
                    return
                job, self._job = self._job, None
                self._busy = True
            token = job[0]
            try:
                found = self._answer(*job[1:])
            except BaseException as exc:  # noqa: BLE001 - silent: no chips
                log.info("locator detector: no answer for %s: %s: %s", token,
                         type(exc).__name__, exc)
                self.stats["failed"] += 1
                found = None
            with self._lock:
                stopped = self._stopped or self._closing()
            if not stopped:
                try:
                    # posted before the job counts as done: a wait() that returns
                    # has the answer already queued for the GUI thread
                    self._bridge.sigDone.emit(token, found)
                except RuntimeError:        # the window (and the bridge) went away
                    stopped = True
            with self._lock:
                self._busy = False
                self._lock.notify_all()
            if stopped:
                return

    def _answer(self, desktop: int, before: PlanItem, after: PlanItem, cls: str) -> list:
        self.stats["jobs"] += 1
        engine = self.engine
        if not self._prepared:
            engine.prepare()
            self._prepared = True
        if self._opened != desktop:
            engine.open_view(desktop, VIEW)
            self._opened = desktop
        frames = []
        for item in (before, after):
            frame = engine.answer_from_cache(item)
            if frame is None:
                self._load()
                frame = engine.process(item)
                if frame is None:
                    raise RuntimeError(f"scanner frame {item.step} has no pixels")
                self.stats["detected"] += 1
            else:
                self.stats["cached"] += 1
            frames.append(frame)
        return vanished(frames[0].dets, frames[1].dets, cls, self.conf)

    def _load(self) -> None:
        """The model, built once -- after SAM's import phase, under the heavy lock."""
        if self.engine.model is not None:
            return
        if self._after is not None and not self._after.released:
            self._after.wait()
        with heavy("locator detector"):
            self.engine.load_model()
