"""One process-wide lock around the heavy imports and model builds (task U6).

The window loads two models on background threads: SAM (``tda-sam-load``,
:meth:`tda.ui.app_assist.AssistMixin.start_sam`) and the small-part detector
(``tda-detect-load``, :class:`tda.ui.app_detect.DetectionWorker`).  Both import
torch and torchvision, and the detector imports rfdetr (or ultralytics) on top.
Python's import system takes one lock *per module*; two threads importing the
same package tree in different orders can each hold a module the other one
needs.  On 2026-10-12 that is what happened: the import machinery detected the
cycle and raised ``_DeadlockError: deadlock detected by
_ModuleLock('torchvision.transforms.functional')`` into SAM's import, and SAM
was off for the whole session.

So both loaders do their importing and their model construction inside
:func:`heavy`, which holds :data:`HEAVY` -- one re-entrant lock for the
process.  Only one thread at a time is inside the import system for these
packages, so no cycle can form.

**SAM goes first.**  The window makes a :class:`Turn` before it starts the
detector, hands it to the detector's worker, and the SAM loader
:meth:`~Turn.release` s it the moment its import phase has finished or failed;
the detector's model build waits for it (:meth:`~Turn.wait`).  A window that
loads no SAM at all releases the turn at once.  The wait has a ceiling
(:data:`TURN_TIMEOUT_S`) because the ordering is a preference -- the lock is
what makes it safe -- and a turn nobody ever releases must not keep the
detector off for the session.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Iterator

__all__ = ["HEAVY", "TURN_TIMEOUT_S", "Turn", "heavy"]

log = logging.getLogger(__name__)

#: The lock every heavy import and model construction in the process holds.
HEAVY = threading.RLock()

#: How long the detector waits for SAM's import phase before it goes ahead
#: anyway (still under :data:`HEAVY`).  SAM's cold import on the RTX 5090
#: machine took 35 s on the day of the deadlock (log, 15:17:26 -> 15:18:01).
TURN_TIMEOUT_S = 180.0


@contextlib.contextmanager
def heavy(who: str) -> Iterator[None]:
    """Hold :data:`HEAVY` for one loader's imports and construction.

    ``who`` names the loader in the log line written when it had to wait for
    the other one, which is the only trace of the ordering there is.
    """
    started = time.perf_counter()
    with HEAVY:
        waited = time.perf_counter() - started
        if waited > 0.05:
            log.info("load gate: %s waited %.1f s for the other model's import", who, waited)
        yield


class Turn:
    """"Do not start before SAM's import phase is over" -- a one-shot gate.

    :meth:`release` is idempotent and :meth:`wait` returns at once afterwards,
    so the window can release it from every path that ends SAM's import phase:
    loaded, failed, never started, window closed.
    """

    def __init__(self, released: bool = False) -> None:
        self._event = threading.Event()
        if released:
            self._event.set()

    def release(self) -> None:
        self._event.set()

    @property
    def released(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float = TURN_TIMEOUT_S) -> bool:
        """Block until released; ``False`` when the ceiling was reached first."""
        done = self._event.wait(timeout)
        if not done:
            log.warning("load gate: SAM's import did not finish within %.0f s; the "
                        "detector loads now (the lock still serialises them)", timeout)
        return done
