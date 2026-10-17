"""Task U15: 「被线挡住」 in the main window -- ``I``, its button, ``Ctrl+I``, the docs.

The 14-step D13 scene of the window tests, every part of step 14 drawn, the
window on step 12 (where the cooler, lifted out at 13, has no shape). ``I`` on a
sequence frame cycles the selected part 完全 -> 部分 -> 撤回; the instance table
tags it, the card row says it, and the canvas draws a 完全 part's whole outline
dashed. The palette's 「看不见」 button is the same slot -- live here, and its
tooltip names both meanings -- and the palette does not grow (U13's height
tests run unchanged). ``Ctrl+I`` asks for the label and the step and writes one
op. A components-gallery shot keeps ``I`` = 看不见 and refuses ``Ctrl+I``.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_scene import (
    COOLER,
    DESKTOP,
    LAST_STEP,
    VIEW,
    StubSamQueue,
    close_window,
    make_paths,
    make_session,
    seed_shapes,
)
from test_app_actions import key_event
from tda.core import masks
from tda.core.model import FrameKey
from tda.core.occluder import FULL, PARTIAL, cable_label
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.app_cable import PICK_FIRST
from tda.ui.app_state import REMOVED_ROW
from tda.ui.canvas.overlay import palette_color
from tda.ui.gallery import GALLERY_REFUSALS
from tda.ui.panels.taskcard import row_view
from tda.ui.session_cable import CABLE_EDITING, CABLE_NEEDS_SHAPE, OP_RANGE

import gallery_scene as GS

CHASSIS = "chassis"
STEP = 12


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def answer_roi(win: MainWindow) -> None:
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()


@pytest.fixture
def window(qapp, tmp_path: Path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(1400, 900)
    answer_roi(win)
    seed_shapes(win.session, LAST_STEP)
    win.session.goto(STEP, force=True)
    QApplication.processEvents()
    yield win
    close_window(win)


@pytest.fixture
def gallery_window(qapp, tmp_path: Path):
    from tda.core.truth import TruthService
    from tda.ui.session import AnnotationSession

    cfg = make_paths(tmp_path)
    Path(cfg["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    db, tax, cache_dir = GS.make_db(tmp_path, db_path=cfg["db_path"],
                                    cache_dir=Path(cfg["cache_dir"]))
    session = AnnotationSession(db, tax, TruthService(db, tax), str(cache_dir), "tester")
    session.sweeper_enabled = False
    session.open(GS.DESKTOP, "cmp_oak1")
    win = MainWindow(session, cfg, "tester", sam_queue=StubSamQueue())
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    yield win
    close_window(win)


def state(win: MainWindow, name: str = "mark_absent") -> tuple[bool, str]:
    win.refresh_guidance()
    button = win.palette.button(name)
    return button.isEnabled(), button.reason()


def click(button) -> None:
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     button.rect().center())
    QApplication.processEvents()


def label_at(win: MainWindow, step: int = STEP, instance: str = CHASSIS):
    fo = win.db.frame_overrides(FrameKey(DESKTOP, step, VIEW)).get(instance)
    return cable_label(fo)


def vis_cell(win: MainWindow, instance: str = CHASSIS) -> str:
    table = win.instances.table()
    for row in range(table.rowCount()):
        item = table.item(row, 1)
        if item is not None and item.text() == instance:
            return table.item(row, 3).text()
    raise AssertionError(f"no row for {instance}")


def card_row(win: MainWindow, instance: str = CHASSIS):
    return next((r for r in win.task_card.rows() if r.get("instance") == instance), None)


# --------------------------------------------------------------------------- #
# I on a sequence frame
# --------------------------------------------------------------------------- #
def test_i_cycles_the_selected_part_and_every_panel_says_it(window):
    assert window.instances.select_instance(CHASSIS)
    window.handle_key(key_event("I"))
    assert label_at(window) == FULL
    assert "被线挡住（完全）" in window.status_message()
    assert window.instances.row_of(CHASSIS)["cable"] == FULL
    assert vis_cell(window) == "线挡"
    row = card_row(window)
    if row is not None:                              # the card lists the drawn part
        assert row["cable"] == FULL and row["done"] and not row.get("invisible")
        assert "⚠ 被线挡住（完全）" in row_view(row)["sentence"]
    # nothing visible, so the overlay paints nothing: a dashed whole outline
    [(polygons, rgb)] = window.canvas.cable_outlines()
    assert rgb == palette_color(CHASSIS) and polygons and len(polygons[0]) >= 3

    window.instances.select_instance(CHASSIS)
    window.handle_key(key_event("I"))
    assert label_at(window) == PARTIAL
    assert "被线挡住（部分）" in window.status_message()
    assert vis_cell(window) == "线半挡"
    assert window.canvas.cable_outlines() == []      # its pixels are painted again

    window.instances.select_instance(CHASSIS)
    window.handle_key(key_event("I"))
    assert label_at(window) is None
    assert "撤回被线挡住" in window.status_message()
    assert window.instances.row_of(CHASSIS)["cable"] is None
    assert vis_cell(window) != "线挡"

    window.act_undo()
    assert label_at(window) == PARTIAL
    window.act_undo()
    assert label_at(window) == FULL and window.canvas.cable_outlines()


def test_a_hidden_part_hides_its_outline_too(window):
    assert window.instances.select_instance(CHASSIS)
    window.act_mark_absent()
    assert window.canvas.cable_outlines()
    window.session.set_hidden(CHASSIS, True)
    window.refresh_overlay()
    assert window.canvas.cable_outlines() == []


# --------------------------------------------------------------------------- #
# the palette's 「看不见」: the same slot, live here
# --------------------------------------------------------------------------- #
def test_the_button_is_live_and_its_tooltip_says_both_meanings(window):
    action = A.action_named("mark_absent")
    assert "被线挡住" in action.label_zh and "零件合照" in action.label_zh
    assert A.short_parts(action) == ("看不见", "")             # the caption is unchanged
    tip = window.palette.button("mark_absent").toolTip()
    assert tip.startswith(f"{action.label_zh}（I）") and "cable" in tip
    assert window.instances.select_instance(CHASSIS)
    assert state(window) == (True, "")
    window.show()
    QApplication.processEvents()
    click(window.palette.button("mark_absent"))
    assert label_at(window) == FULL


def test_ctrl_i_is_a_key_not_a_button(window):
    """The palette must not grow (U13's height tests): Ctrl+I has no button."""
    action = A.action_named("cable_range")
    assert action.keys == ("Ctrl+I",) and action.modes == (A.MODE_ANNOTATE,)
    assert not action.short
    for group in (A.PALETTE_TOOLS, A.PALETTE_ACTIONS, A.PALETTE_STATE, A.PALETTE_OTHER,
                  A.PALETTE_REVIEW):
        assert "cable_range" not in group
    assert len(A.PALETTE_STATE) == 5


def test_the_refusals_grey_the_button_and_the_key_says_them(window):
    # nothing chosen: "select a part", naming the gallery's meaning
    window.instances.table().setCurrentCell(-1, -1)
    QApplication.processEvents()
    enabled, why = state(window)
    assert not enabled and why.startswith(PICK_FIRST)
    # the cooler has no shape at 12
    assert window.instances.select_instance(COOLER)
    enabled, why = state(window)
    assert not enabled and why.endswith(CABLE_NEEDS_SHAPE)
    window.handle_key(key_event("I"))
    assert CABLE_NEEDS_SHAPE in window.status_message()
    assert label_at(window, instance=COOLER) is None


def test_pixels_being_drawn_grey_it(window):
    assert window.instances.select_instance(CHASSIS)
    window.on_request_edit(CHASSIS)
    window.act_tool("brush")
    viewport = window.canvas.viewport()
    centre = viewport.rect().center()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, centre)
    QTest.mouseMove(viewport, centre + QPoint(8, 0))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, centre + QPoint(8, 0))
    QApplication.processEvents()
    assert window.has_uncommitted_edit()
    enabled, why = state(window)
    assert not enabled and why.endswith(CABLE_EDITING)
    window.handle_key(key_event("I"))
    assert CABLE_EDITING in window.status_message() and label_at(window) is None
    window.act_clear_edit()
    assert state(window) == (True, "")


def test_a_removed_row_greys_it(window):
    window.instances.show_removed.setChecked(True)
    QApplication.processEvents()
    rows = len(window.instances.rows())
    if window.instances.table().rowCount() <= rows:
        pytest.skip("nothing is removed by this frame of the scene")
    window.instances.table().setCurrentCell(rows, 1)
    QApplication.processEvents()
    assert window.instances.selected_is_removed()
    assert state(window) == (False, REMOVED_ROW)


# --------------------------------------------------------------------------- #
# Ctrl+I
# --------------------------------------------------------------------------- #
def test_ctrl_i_writes_the_run_in_one_op(window, monkeypatch):
    asked: dict = {}

    def answer(instance, value, low, high):
        asked.update(instance=instance, value=value, low=low, high=high)
        return FULL, 10

    monkeypatch.setattr(window, "ask_cable_range", answer)
    assert window.instances.select_instance(CHASSIS)
    window.handle_key(key_event("Ctrl+I"))
    assert asked == {"instance": CHASSIS, "value": FULL, "low": 1, "high": LAST_STEP}
    assert [label_at(window, s) for s in (9, 10, 11, 12, 13)] == [None, FULL, FULL, FULL, None]
    assert "第 10–12 步都标为被线挡住（完全）" in window.status_message()
    ops = [op for op in window.db.ops(DESKTOP, VIEW, limit=100) if op["kind"] == OP_RANGE]
    assert len(ops) == 1
    window.act_undo()
    assert all(label_at(window, s) is None for s in (10, 11, 12))


def test_ctrl_i_names_the_frames_the_part_has_no_shape_on(window, monkeypatch):
    monkeypatch.setattr(window, "ask_cable_range", lambda *a: (PARTIAL, 10))
    # the cooler has no shape on 10-12: the frames are refused by name, nothing written
    assert window.instances.select_instance(COOLER)
    window.handle_key(key_event("Ctrl+I"))
    assert "第 10–12 步" in window.status_message()
    assert CABLE_NEEDS_SHAPE in window.status_message()
    assert all(label_at(window, s, COOLER) is None for s in (10, 11, 12))


def test_ctrl_i_on_a_gallery_shot_is_refused_with_the_gallerys_words(gallery_window):
    win = gallery_window
    part = win.task_card.current_instance()
    assert part
    win.handle_key(key_event("Ctrl+I"))
    assert GALLERY_REFUSALS["cable_range"] in win.status_message()
    # ... and I there is still 看不见 (task U10)
    win.handle_key(key_event("I"))
    row = next(r for r in win.task_card.rows() if r["instance"] == part)
    assert row["done"] and row["absent"]


# --------------------------------------------------------------------------- #
# the documents
# --------------------------------------------------------------------------- #
def _squash(text: str) -> str:
    return "".join(text.split())


def test_the_documents_say_it():
    repo = Path(__file__).resolve().parents[1]
    guide = _squash((repo / "docs" / "annotation_guide.md").read_text(encoding="utf-8"))
    assert _squash("在原位被线挡住：选中按 `I`（完全→部分→撤回；`Ctrl+I` 一段帧），"
                   "形状照画、导出写明。") in guide
    for name in ("mark_absent", "cable_range"):                     # the key table
        assert _squash(A.action_named(name).label_zh) in guide
    quick = _squash((repo / "docs" / "trial_quickstart.md").read_text(encoding="utf-8"))
    assert _squash("**零件在原位，只是被线挡住了**") in quick
    assert _squash("`connector.front_panel.01` 插在原位，被风扇线盖住") in quick
    assert _squash("整个看不见按一次 `I`，露出一半按两次") in quick
    spec = _squash((repo / "docs" / "superpowers" / "specs"
                    / "2026-09-17-teardown-annotator-design.md").read_text(encoding="utf-8"))
    assert _squash("**v1.15（2026-10-16）**：**被线挡住**（U15，§6.2）") in spec
    assert _squash("**遮挡原因 `occluder`**（v1.15，U15「被线挡住」）") in spec
