"""Task U19: the window tells SAM which part is being drawn, and which head answered last.

``SamRequest.instance_cls`` picks the head of a prompt without a box;
``SamRequest.mask_input_head`` sends a correction back to the head whose answer
it corrects.  The status bar says nothing new unless the fine-tuned head is off,
and SAM still loads before the detector -- the head's configuration is read
inside SAM's turn.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import StubSamQueue, close_window
from tda.models import load_gate
from tda.models.sam_heads import STOCK, TUNED, SamHeadConfig
from tda.ui.app_assist import HEAD_OFF_NOTE
from test_app_u3 import StubModel, _window, det_config
from test_u6_sam_load import _bare_window, _settle_sam


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    win = _window(tmp_path, loose=False)
    yield win
    close_window(win)


def _card(win) -> list[dict]:
    return [r for r in win.session.task_card() if r.get("instance")]


def _answer_as(queue: StubSamQueue, head: str) -> None:
    """The stub answers as if ``head`` had: what ``SamService`` puts in ``SamResult.head``."""
    plain = StubSamQueue.result_for

    def result_for(req, multimask=None):
        result = plain(queue, req, multimask)
        result.head = head
        return result

    queue.result_for = result_for


def _click(win, x: float = 32.0, y: float = 32.0):
    win.sam_point.on_press(x, y, None)
    return win.sam_queue.requests[-1]


def _answer(win) -> None:
    win.sam_queue.flush()
    QApplication.processEvents()


def test_a_prompt_carries_the_class_of_the_part_being_drawn(window):
    row = _card(window)[0]
    instance, cls = str(row["instance"]), str(row["cls"])
    window.task_card.sigRequestEdit.emit(instance)
    assert (window.sam_point.instance_cls, window.sam_box.instance_cls) == (cls, cls)
    window.act_tool("sam_point")
    req = _click(window)
    assert req.instance_cls == cls and req.mask_input_head is None


def test_a_correction_goes_back_to_the_head_that_answered(window):
    instance = str(_card(window)[0]["instance"])
    window.task_card.sigRequestEdit.emit(instance)
    window.act_tool("sam_point")

    _answer_as(window.sam_queue, TUNED)
    _click(window)
    _answer(window)
    again = _click(window, 30.0, 30.0)
    assert again.mask_input is not None, "the second click is a refinement"
    assert again.mask_input_head == TUNED

    # the layer replaced from outside the tool: whatever is in it is no answer
    # of this prompt any more, and the next click is routed by the rule
    window.reset_sam_prompt()
    fresh = _click(window, 28.0, 28.0)
    assert fresh.mask_input is not None and fresh.mask_input_head is None

    _answer_as(window.sam_queue, STOCK)
    _answer(window)
    stock = _click(window, 26.0, 26.0)
    assert stock.mask_input_head == STOCK


def test_the_class_comes_from_the_card_the_table_or_the_key(window):
    card = _card(window)
    first = str(card[0]["instance"])
    assert window.sam_class_of(first) == str(card[0]["cls"])
    assert window.sam_class_of(None) is None
    on_card = {str(r["instance"]) for r in card}
    rec = next(r for key, r in window.db.instances(int(window.session.desktop)).items()
               if key not in on_card)
    assert window.sam_class_of(rec.key) == rec.cls
    # a 框外零件 nobody has stored yet says its class after ``ctx.`` (U18) ...
    assert window.sam_class_of("ctx.psu.01") == "psu"
    # ... and one of no class is ``other``: a small part as far as SAM goes
    assert window.sam_class_of("ctx.other.03") == "other"
    window.set_sam_instance("ctx.other.03")
    assert window.sam_point.instance_cls == window.sam_box.instance_cls == "other"


def test_the_class_is_looked_up_when_the_instance_changes_not_per_frame(window, monkeypatch):
    asked: list = []
    real = window.sam_class_of
    monkeypatch.setattr(window, "sam_class_of", lambda inst: (asked.append(inst), real(inst))[1])
    instance = str(_card(window)[0]["instance"])
    window.set_sam_instance(instance)
    window.set_sam_instance(instance)          # what every frame change does
    window.set_sam_instance(instance)
    assert asked == [instance]
    assert window.sam_point.instance_cls == real(instance)
    window.set_sam_instance(None)
    assert window.sam_point.instance_cls is None


class _QueueWithService(StubSamQueue):
    def __init__(self, reason: str) -> None:
        super().__init__()
        self.service = SimpleNamespace(head_off_reason=reason, heads=(STOCK,))


def test_the_status_says_nothing_new_unless_the_head_is_off(window):
    assert window.sam_status_text() == "SAM ready"          # a stub that never asked
    window.set_sam_queue(_QueueWithService(""), owns=False)
    assert window.sam_status_text() == "SAM ready"
    window.set_sam_queue(_QueueWithService("TDA_SAM_HEAD=off"), owns=False)
    assert window.sam_status_text() == f"SAM ready{HEAD_OFF_NOTE}"
    window.task_card.sigRequestEdit.emit(str(_card(window)[0]["instance"]))
    window.act_tool("sam_point")
    _click(window)
    _answer(window)
    assert window.sam_status_text() == f"SAM ready · 1/3{HEAD_OFF_NOTE}"
    # SAM not loaded at all: the existing wording, nothing about heads
    window.set_sam_unavailable("not loaded yet")
    assert window.sam_status_text() == "SAM unavailable: not loaded yet"


def test_sam_still_loads_first_and_reads_its_heads_inside_its_turn(qapp, tmp_path, monkeypatch):
    """U6's order with the head: config, SAM, then the detector -- all under the lock."""
    import tda.models.sam_heads as sam_heads
    import tda.models.sam_service as sam_service

    events: list = []
    handed: list = []
    config = SamHeadConfig(off="a test")

    def load_config(path=None):
        events.append(("heads", load_gate.HEAVY._is_owned()))
        return config

    class FakeService:
        def __init__(self, checkpoint=None, **kwargs):
            handed.append(kwargs.get("heads"))
            events.append(("sam", load_gate.HEAVY._is_owned()))
            time.sleep(0.3)

    monkeypatch.setattr(sam_heads, "load_sam_head_config", load_config)
    monkeypatch.setattr(sam_service, "SamService", FakeService)
    monkeypatch.setattr(sam_service, "SamQueue", lambda service: StubSamQueue())
    monkeypatch.setattr(sam_service, "checkpoint_in", lambda paths: None)
    win = _bare_window(tmp_path)
    try:
        def factory(_config):
            events.append(("det_build", load_gate.HEAVY._is_owned()))
            return StubModel({}, identity="stub-u19")

        win.enable_detector(det_config(tmp_path), factory=factory, identity="stub-u19",
                            cache_root=str(tmp_path / "det"))
        time.sleep(0.4)
        assert not any(name == "det_build" for name, _held in events)
        win.start_sam()
        _settle_sam(win)
        assert win.det_worker.wait(20.0)
        assert [name for name, _held in events] == ["heads", "sam", "det_build"], events
        assert all(held for _name, held in events), "a load ran outside the lock"
        assert handed == [config]
        assert win.sam_available and win.det_state == "on"
    finally:
        close_window(win)
