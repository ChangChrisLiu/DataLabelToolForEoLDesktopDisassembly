"""Task U19: the window tells SAM which part is being drawn, and which head answered last.

``SamRequest.instance_cls`` picks the head of a prompt without a box;
``SamRequest.mask_input_head`` sends a correction back to the head whose answer
it corrects.  The status bar says nothing new unless the fine-tuned head is off,
and SAM still loads before the detector -- the head's configuration is read
inside SAM's turn.
"""
from __future__ import annotations

import time
from pathlib import Path
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
    asked: list = []
    config = SamHeadConfig(off="a test")

    def load_config(path=None, weights_dir=None):
        events.append(("heads", load_gate.HEAVY._is_owned()))
        asked.append(weights_dir)
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
        # the head is looked for where the app's own paths put the weights (N2)
        assert asked == [win.paths["weights_dir"]]
        assert win.sam_available and win.det_state == "on"
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# the U19 review
# --------------------------------------------------------------------------- #
def test_box_drag_then_point_correction_goes_back_to_the_box_head(window):
    """M1: ``X`` (the tuned head answers the drag), then ``S`` to correct it.

    The switch detaches the box tool and its reset dropped the tag, so the
    correcting click went out with ``mask_input`` and no head, and for a class
    in ``stock_point_classes`` the stock head refined the tuned head's mask.
    The tag goes over with the switch now; every other reset still drops it.
    """
    from tda.models.sam_heads import Routing, choose_head
    from test_u19_heads import STOCK_POINT

    row = _card(window)[0]
    window.task_card.sigRequestEdit.emit(str(row["instance"]))
    for tool in (window.sam_point, window.sam_box):     # a large part, as on a real desk
        tool.instance_cls = "motherboard"
    window.act_tool("sam_box")
    _answer_as(window.sam_queue, TUNED)
    window.sam_box.on_press(10.0, 10.0, None)
    window.sam_box.on_move(50.0, 50.0, None)
    window.sam_box.on_release(50.0, 50.0, None)
    _answer(window)
    assert window.sam_box.answer_head == TUNED and window.overlay.editing.any()

    window.act_tool("sam_point")
    req = _click(window, 30.0, 30.0)
    assert req.box is None and req.mask_input is not None
    assert req.mask_input_head == TUNED
    assert choose_head(Routing(stock_point_classes=STOCK_POINT), has_box=False,
                       instance_cls=req.instance_cls,
                       mask_input_head=req.mask_input_head) == TUNED

    # ... and back: S's own answer (stock this time) goes over to X
    _answer_as(window.sam_queue, STOCK)
    _answer(window)
    window.act_tool("sam_box")
    assert window.sam_box.answer_head == STOCK
    window.act_tool("sam_point")
    assert window.sam_point.answer_head == STOCK

    # a detour through the brush is not a SAM-to-SAM switch: the tag goes
    window.act_tool("brush")
    window.act_tool("sam_point")
    assert window.sam_point.answer_head is None


@pytest.mark.parametrize("action", ["commit", "esc", "undo"])
def test_the_tag_carried_over_a_switch_still_goes_with_the_layer(window, action):
    window.task_card.sigRequestEdit.emit(str(_card(window)[0]["instance"]))
    window.act_tool("sam_box")
    _answer_as(window.sam_queue, TUNED)
    window.sam_box.on_press(10.0, 10.0, None)
    window.sam_box.on_release(50.0, 50.0, None)
    _answer(window)
    window.act_tool("sam_point")
    assert window.sam_point.answer_head == TUNED
    {"commit": window.act_commit, "esc": window.act_clear_edit, "undo": window.act_undo}[action]()
    QApplication.processEvents()
    assert window.sam_point.answer_head is None and window.sam_box.answer_head is None


def test_start_sam_reads_the_real_head_config_and_says_it_is_off(qapp, tmp_path, monkeypatch):
    """M2: no GPU, the real ``load_sam_head_config`` -- the conftest's ``off`` replaced."""
    import yaml

    import tda.models.sam_service as sam_service
    from tda.models.sam_heads import ENV_VAR
    from tda.ui.app_support import log_path

    config_file = tmp_path / "sam_test.yaml"
    config_file.write_text(yaml.safe_dump({
        "enabled": True, "head": "gone_head.pt",
        "route": {"box_prompts": "tuned",
                  "point_only": {"stock_point_classes": "stock", "other": "tuned"}},
        "stock_point_classes": ["motherboard", "ram_module"]}), encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(config_file))
    handed: list = []

    class FakeService:
        """What the window reads of a ``SamService``: its heads and why one is off."""

        def __init__(self, checkpoint=None, heads=None, **kwargs):
            handed.append(heads)
            self.heads = (STOCK,)
            self.head_off_reason = heads.off if heads is not None else ""

    class QueueWithService(StubSamQueue):
        def __init__(self, service):
            super().__init__()
            self.service = service

    monkeypatch.setattr(sam_service, "SamService", FakeService)
    monkeypatch.setattr(sam_service, "SamQueue", QueueWithService)
    monkeypatch.setattr(sam_service, "checkpoint_in", lambda paths: None)
    win = _bare_window(tmp_path)
    try:
        win.start_sam()
        _settle_sam(win)
        [config] = handed
        assert isinstance(config, SamHeadConfig) and config.source == config_file
        assert not config.wanted
        # a bare name is looked for in the app's own weights_dir (N2)
        missing = Path(win.paths["weights_dir"]) / "gone_head.pt"
        assert f"the head file {missing} does not exist" in config.off
        lines = open(log_path(win.paths), encoding="utf-8").read().splitlines()
        warned = [line for line in lines
                  if " WARNING " in line and "SAM fine-tuned head OFF" in line]
        assert len(warned) == 1, warned
        assert any("SAM: ready (heads: stock; fine-tuned head off:" in line for line in lines)
        assert win.sam_status_text() == f"SAM ready{HEAD_OFF_NOTE}"
    finally:
        close_window(win)
