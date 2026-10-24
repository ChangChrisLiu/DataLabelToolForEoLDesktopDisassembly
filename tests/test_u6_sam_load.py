"""Task U6, ruling A: no import deadlock between the two loaders, and SAM can be retried.

On 2026-10-12 SAM's loader (``tda-sam-load``) and the detector's
(``tda-detect-load``) imported torch / torchvision / rfdetr at the same time;
Python's import locks deadlocked, ``_DeadlockError`` went into SAM's import,
and S and X stayed grey all session with no way back short of a restart.
"""
from __future__ import annotations

import importlib
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import StubSamQueue, close_window, make_paths, make_session
from tda.models import load_gate
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.app_assist import DET_CHIP, DET_CHIP_NO_SAM, short_reason
from tda.ui.guide import DRAW_NO_SAM, PHASE_DRAW_EMPTY
from test_app_u3 import (
    A_BOX,
    B_BOX,
    C_BOX,
    PLAIN_STEPS,
    SCREW_STEP,
    StubModel,
    _window,
    det,
    det_config,
    enable,
    fbox,
    go,
    pump,
)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------- #
# the race itself, with two fake libraries
# --------------------------------------------------------------------------- #
def _two_libraries(root: Path) -> tuple[str, str, list[str]]:
    """Two packages whose ``__init__`` imports a submodule of the other, after a pause.

    The shape of the day: one thread imports ``tv.functional`` (holding that
    submodule's lock while ``tv`` itself initialises), the other ``rf.models``;
    each package needs the submodule the other thread holds.  A plain module
    cycle would not do -- Python accepts a half-initialised module there -- but
    a submodule that is not in ``sys.modules`` yet is waited for, and that
    wait is where ``_DeadlockError: deadlock detected by
    _ModuleLock('torchvision.transforms.functional')`` came from.
    """
    tag = uuid.uuid4().hex[:8]
    tv, rf = f"u6tv_{tag}", f"u6rf_{tag}"
    for package, other in ((tv, f"{rf}.models"), (rf, f"{tv}.functional")):
        (root / package).mkdir()
        (root / package / "__init__.py").write_text(
            f"import time\ntime.sleep(0.4)\nimport {other}\n", encoding="utf-8")
    (root / tv / "functional.py").write_text("X = 1\n", encoding="utf-8")
    (root / rf / "models.py").write_text("Y = 2\n", encoding="utf-8")
    names = [tv, rf, f"{tv}.functional", f"{rf}.models"]
    return f"{tv}.functional", f"{rf}.models", names


def _import_on_two_threads(root: Path, gated: bool) -> list[BaseException]:
    tv, rf, names = _two_libraries(root)
    errors: list[BaseException] = []
    start = threading.Barrier(2)

    def load(name: str, who: str) -> None:
        start.wait()
        try:
            if gated:
                with load_gate.heavy(who):
                    importlib.import_module(name)
            else:
                importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001 - the deadlock is the point
            errors.append(exc)

    sys.path.insert(0, str(root))
    try:
        threads = [threading.Thread(target=load, args=(tv, "sam")),
                   threading.Thread(target=load, args=(rf, "detector"))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20.0)
        assert not any(t.is_alive() for t in threads), "an import hung"
    finally:
        sys.path.remove(str(root))
        for name in names:
            sys.modules.pop(name, None)
    return errors


def test_the_two_imports_deadlock_without_the_gate(tmp_path):
    """The trial's failure, reproduced: Python detects the cycle and one import dies."""
    errors = _import_on_two_threads(tmp_path, gated=False)
    assert errors, "the race did not happen"
    assert any("deadlock detected" in str(e) for e in errors), errors
    assert type(errors[0]).__name__ == "_DeadlockError"


def test_the_gate_serialises_them_and_both_import(tmp_path):
    assert _import_on_two_threads(tmp_path, gated=True) == []


def test_a_turn_is_released_once_and_waits_no_more():
    turn = load_gate.Turn()
    assert turn.wait(0.01) is False
    turn.release()
    turn.release()
    assert turn.released and turn.wait(0.01) is True
    assert load_gate.Turn(released=True).wait(0) is True


# --------------------------------------------------------------------------- #
# the window: SAM first, the detector's build after it, both under the lock
# --------------------------------------------------------------------------- #
def _bare_window(tmp_path) -> MainWindow:
    """A window that loads its own SAM, as ``tda.cli app`` opens it."""
    session = make_session(tmp_path)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=None)
    win.resize(900, 700)
    return win


def _fake_sam(monkeypatch, events: list, fail: int = 0, pause: float = 0.4,
              queues: list | None = None) -> list:
    """Replace the SAM service: records when it ran and whether it held the lock.

    ``queues`` collects the stub queues the loader builds, so a test can see
    whether one was stopped.
    """
    import tda.models.sam_service as sam_service

    attempts: list[int] = []
    built = queues if queues is not None else []

    def make_queue(service):
        built.append(StubSamQueue())
        return built[-1]

    class _DeadlockError(ImportError):
        pass

    class FakeService:
        def __init__(self, checkpoint=None, **kwargs):
            attempts.append(1)
            events.append(("sam_start", time.perf_counter(),
                           load_gate.HEAVY._is_owned()))
            time.sleep(pause)
            events.append(("sam_end", time.perf_counter(), load_gate.HEAVY._is_owned()))
            if len(attempts) <= fail:
                raise _DeadlockError(
                    "deadlock detected by _ModuleLock('torchvision.transforms."
                    "functional') at 2226949304144")

    monkeypatch.setattr(sam_service, "SamService", FakeService)
    monkeypatch.setattr(sam_service, "SamQueue", make_queue)
    monkeypatch.setattr(sam_service, "checkpoint_in", lambda paths: None)
    return attempts


def _settle_sam(win, timeout: float = 20.0) -> None:
    thread = getattr(win, "_sam_loader_thread", None)
    if thread is not None:
        thread.join(timeout)
    for _ in range(5):
        QApplication.processEvents()


def test_the_detector_builds_only_after_sams_import_phase(qapp, tmp_path, monkeypatch):
    events: list = []
    _fake_sam(monkeypatch, events)
    win = _bare_window(tmp_path)
    try:
        def factory(_config):
            events.append(("det_build", time.perf_counter(), load_gate.HEAVY._is_owned()))
            return StubModel({}, identity="stub-u6")

        # the detector first, as the window does it (its worker starts in the
        # constructor); its build must wait for SAM, which has not started yet
        win.enable_detector(det_config(tmp_path), factory=factory, identity="stub-u6",
                            cache_root=str(tmp_path / "det"))
        time.sleep(0.5)
        assert [e for e in events if e[0] == "det_build"] == [], \
            "the detector built before SAM's import phase"
        win.start_sam()
        _settle_sam(win)
        assert win.det_worker.wait(20.0)
        names = [e[0] for e in events]
        assert names == ["sam_start", "sam_end", "det_build"], names
        sam_end = next(e[1] for e in events if e[0] == "sam_end")
        det_build = next(e[1] for e in events if e[0] == "det_build")
        assert det_build >= sam_end
        assert all(held for _name, _t, held in events), "a load ran outside the lock"
        assert win.det_state == "on" and win.sam_available
    finally:
        close_window(win)


def test_a_window_with_a_queue_handed_in_never_holds_the_detector_back(qapp, tmp_path):
    win = _window(tmp_path)
    try:
        assert win._sam_turn.released
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# a failed load: the badge, the guide, S / X, and F5
# --------------------------------------------------------------------------- #
def test_a_failed_load_says_so_and_f5_retries_once_per_press(qapp, tmp_path, monkeypatch):
    events: list = []
    attempts = _fake_sam(monkeypatch, events, fail=1, pause=0.3)
    win = _bare_window(tmp_path)
    try:
        win.show()
        QApplication.processEvents()
        win.set_mode(A.MODE_ANNOTATE)
        win.start_sam()
        _settle_sam(win)

        badge = win.sam_label.text()
        assert badge.startswith("SAM 加载失败：_DeadlockError: deadlock detected"), badge
        assert badge.endswith("（F5 重试）") and "2226949304144" not in badge

        # S and X say what works instead, and the guide's draw step too
        win.act_tool("sam_point")
        assert win.active_tool is not win.sam_point
        assert DRAW_NO_SAM in win.status_message() and "F5" in win.status_message()
        instance = next(r["instance"] for r in win.task_card.rows()
                        if r.get("kind") == "add_shape")
        win.task_card.sigRequestEdit.emit(instance)
        QApplication.processEvents()
        plan = win.guide_plan()
        assert plan.phase == PHASE_DRAW_EMPTY
        assert DRAW_NO_SAM in plan.now and "按 S" not in plan.now
        assert plan.steps[2][1] == DRAW_NO_SAM
        win.act_clear_edit()

        # F5: one load per press, none while one is running
        win.act_refresh_all()
        win.act_refresh_all()
        assert len(attempts) == 2
        _settle_sam(win)
        assert win.sam_available and win.sam_label.text().startswith("SAM ready")
        win.act_refresh_all()
        assert len(attempts) == 2            # a SAM that loaded is not reloaded
    finally:
        close_window(win)


def test_the_short_reason_drops_the_address_and_keeps_the_name():
    text = ("_DeadlockError: deadlock detected by _ModuleLock("
            "'torchvision.transforms.functional') at 2226949304144")
    short = short_reason(text)
    assert short.startswith("_DeadlockError: deadlock detected") and len(short) <= 48
    assert "2226949304144" not in short
    assert short_reason("FileNotFoundError: x") == "FileNotFoundError: x"


# --------------------------------------------------------------------------- #
# the detector's chip says what the box is for while SAM is not there
# --------------------------------------------------------------------------- #
def test_the_detector_chip_follows_sam(qapp, tmp_path):
    win = _window(tmp_path)
    try:
        win.set_sam_unavailable("loading")
        enable(win, tmp_path, {SCREW_STEP: [det(A_BOX), det(B_BOX)]})
        go(win, SCREW_STEP)
        assert win._prompt_box == fbox(A_BOX)        # the box is still shown
        assert win.canvas.prompt_band() == (fbox(A_BOX), DET_CHIP_NO_SAM.format(name="螺丝"))
        assert DET_CHIP_NO_SAM.format(name="螺丝") == \
            "检测器找到的螺丝（SAM 未加载：R 拖一下就画好）"      # Y until U22

        win.set_sam_queue(StubSamQueue(), owns=False)      # SAM arrives
        assert win.canvas.prompt_band() == (fbox(A_BOX), DET_CHIP.format(name="螺丝"))
        pump(win)
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# round 2
# --------------------------------------------------------------------------- #
def test_a_window_closed_during_sams_load_never_builds_the_detector(
        qapp, tmp_path, monkeypatch):
    """Item 1: the detector is stopped before the SAM loader is waited for.

    SAM takes 3 s; the window closes 1.5 s in.  The detector's build was
    waiting for SAM's turn; it must wake, see the stop and build nothing --
    neither during the close nor after it -- and the queue SAM finishes
    building after the close is stopped by the loader itself.
    """
    events: list = []
    queues: list = []
    _fake_sam(monkeypatch, events, pause=3.0, queues=queues)
    builds: list[float] = []

    def factory(_config):
        builds.append(time.perf_counter())
        time.sleep(2.0)                 # what a real RF-DETR build costs, at least
        return StubModel({}, identity="stub-u6-close")

    win = _bare_window(tmp_path)
    loader = None
    try:
        win.enable_detector(det_config(tmp_path), factory=factory,
                            identity="stub-u6-close", cache_root=str(tmp_path / "det"))
        win.start_sam()
        loader = win._sam_loader_thread
        time.sleep(1.5)
        started = time.perf_counter()
        close_window(win)
        closed_in = time.perf_counter() - started
    finally:
        if not win.closed:
            close_window(win)
    assert closed_in < 2.5, f"the close took {closed_in:.1f} s"    # SAM's rest, no build
    assert builds == [], "the detector was built during the close"
    time.sleep(2.5)
    QApplication.processEvents()
    assert builds == [], "the detector was built after the close"
    assert loader is not None and not loader.is_alive()
    assert queues and all(q.stopped for q in queues), "SAM's late queue was left running"


def test_a_window_that_never_starts_sam_has_its_detector_within_seconds(qapp, tmp_path):
    """Item 2: no ``start_sam`` -> the turn is released by the grace timer, not after 180 s."""
    win = _bare_window(tmp_path)
    try:
        builds: list[float] = []

        def factory(_config):
            builds.append(time.perf_counter())
            return StubModel({}, identity="stub-u6-grace")

        started = time.perf_counter()
        win.enable_detector(det_config(tmp_path), factory=factory, identity="stub-u6-grace",
                            cache_root=str(tmp_path / "det"))
        deadline = started + 8.0
        while win.det_state != "on" and time.perf_counter() < deadline:
            QApplication.processEvents()
            time.sleep(0.02)
        assert win.det_state == "on", "the detector waited for a SAM nobody starts"
        assert builds and builds[0] - started < 5.0
        assert win._sam_turn.released and not win.sam_available
    finally:
        close_window(win)


def test_declining_sam_or_handing_a_queue_in_releases_the_turn_at_once(qapp, tmp_path):
    """Item 2: ``--no-sam`` (``decline_sam``) and ``set_sam_queue`` say "no SAM of our own"."""
    win = _bare_window(tmp_path)
    try:
        assert not win._sam_turn.released
        win.decline_sam("--no-sam")
        assert win._sam_turn.released and "--no-sam" in win.sam_label.text()
    finally:
        close_window(win)
    win = _bare_window(tmp_path / "b")
    try:
        assert not win._sam_turn.released
        win.set_sam_queue(StubSamQueue(), owns=False)
        assert win._sam_turn.released
    finally:
        close_window(win)


def test_the_smoke_script_releases_the_turn_in_both_modes():
    """Item 2: ``scripts/mvp_smoke.py`` declines SAM with ``--no-sam``, else loads it under the lock."""
    text = (Path(__file__).resolve().parents[1] / "scripts" / "mvp_smoke.py").read_text(
        encoding="utf-8")
    assert 'window.decline_sam("--no-sam")' in text
    assert 'with heavy("sam"):' in text and "window.set_sam_queue(SamQueue(service)" in text


def test_every_box_says_sam_is_down_and_comes_back_when_it_loads(qapp, tmp_path):
    """Item 3: the difference map's chip, its status line, and the Shift+C chips."""
    from tda.ui.app_assist import (
        DET_CHIP_RANK,
        DET_CHIP_RANK_NO_SAM,
        PROMPT_ARMED,
        PROMPT_ARMED_NO_SAM,
        PROMPT_CHIP,
        PROMPT_CHIP_NO_SAM,
        PROMPT_CHIP_RANK,
        PROMPT_CHIP_RANK_NO_SAM,
    )

    win = _window(tmp_path)
    try:
        # the difference map's own box, armed while SAM is down
        win.set_sam_unavailable("loading")
        enable(win, tmp_path, {s: [det(A_BOX), det(B_BOX)] for s in range(1, 15)})
        go(win, PLAIN_STEPS[0])
        box = win._prompt_box
        assert box is not None
        assert win.canvas.prompt_band() == (box, PROMPT_CHIP_NO_SAM)
        assert win.status_message() == PROMPT_ARMED_NO_SAM and "点 S" not in PROMPT_ARMED_NO_SAM
        assert "SAM 提示框" not in PROMPT_CHIP_NO_SAM
        win.set_sam_queue(StubSamQueue(), owns=False)
        assert win.canvas.prompt_band() == (box, PROMPT_CHIP)
        go(win, PLAIN_STEPS[1])                 # armed with SAM: today's line
        assert win.status_message() == PROMPT_ARMED

        # Shift+C's chips: walked with SAM, then SAM goes and comes back
        from test_app_u3 import _with_diff_offers

        enable(win, tmp_path, {SCREW_STEP: [det(A_BOX), det(B_BOX), det(C_BOX)]})
        go(win, SCREW_STEP)
        _with_diff_offers(win)
        alternates = win.prompt_alternates()
        total = len(alternates) + 1
        assert type(alternates[0]).__name__ == "DetCandidate"
        assert type(alternates[2]).__name__ != "DetCandidate"    # rank 4 is the map's
        win.act_cycle_prompt_box()
        rank_two = DET_CHIP_RANK.format(rank=2, total=total, name="螺丝")
        assert win.canvas.prompt_band() == (fbox(B_BOX), rank_two)
        win.set_sam_unavailable("the GPU went away")
        assert win.canvas.prompt_band() == (
            fbox(B_BOX), DET_CHIP_RANK_NO_SAM.format(rank=2, total=total, name="螺丝"))
        assert "SAM 提示框" not in win.canvas.prompt_band()[1]
        win.set_sam_queue(StubSamQueue(), owns=False)
        assert win.canvas.prompt_band() == (fbox(B_BOX), rank_two)
        # rank 4 comes from the difference map: the same both ways
        win.act_cycle_prompt_box()
        win.act_cycle_prompt_box()
        here = win._prompt_box
        rank_four = PROMPT_CHIP_RANK.format(rank=4, total=total)
        assert here == tuple(float(v) for v in alternates[2].box)
        assert win.canvas.prompt_band() == (here, rank_four)
        win.set_sam_unavailable("the GPU went away")
        assert win.canvas.prompt_band() == (
            here, PROMPT_CHIP_RANK_NO_SAM.format(rank=4, total=total))
        win.set_sam_queue(StubSamQueue(), owns=False)
        assert win.canvas.prompt_band() == (here, rank_four)
        pump(win)
    finally:
        close_window(win)
