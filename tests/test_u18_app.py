"""Task U18 in the main window: the 「框外零件…」 entry, drawing one, ``I`` / ``Ctrl+I``.

The 14-step D13 scene of the window tests, every chassis part of step 14
drawn, the window on step 12. The canvas' right-click menu makes a context
object (the dialog answered by the test), selects it and starts its edit layer
at once; ``Enter`` commits it in its own chain; SAM gets no prompt box for it.
``I`` on one is 「不在这张里」 and the instance table says 「不在」; ``Ctrl+I`` marks
a span. The 「＋ 添加零件」 dialog of Steps mode has the same entry, and the
Steps Instances table lists the context objects in a block of their own. No
palette button: the U13 height tests run unchanged.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app_scene import (
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
from tda.core import context as C
from tda.core.compiler import marked_absent
from tda.core.model import CONTEXT_PLACEMENT, FrameKey
from tda.ui import app_actions as A
from tda.ui import context_text as T
from tda.ui.app import MainWindow

import gallery_scene as GS

STEP = 12
BRACKET = "ctx.cooler_bracket.01"
ANSWER = {"cls": "cooler_bracket", "label": "台面上的支架", "note": ""}


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


def make_one(win: MainWindow, answer: dict = ANSWER) -> str:
    win.ask_context_object = lambda: dict(answer)
    return win.act_context_object()


def square(y0: int = 44, x0: int = 44, side: int = 12) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=bool)
    mask[y0:y0 + side, x0:x0 + side] = True
    return mask


def commit_layer(win: MainWindow, mask: np.ndarray) -> None:
    win.set_editing_mask(mask, undoable=True)
    QApplication.processEvents()
    win.act_commit()
    for _ in range(2):                      # a size warning waits for a second Enter
        if win._pending_warning is None:
            break
        win.act_commit()
    QApplication.processEvents()


def vis_text(win: MainWindow, instance: str) -> str:
    table = win.instances.table()
    for row in range(table.rowCount()):
        item = table.item(row, 1)
        if item is not None and item.text().split()[0] == instance:
            return table.item(row, 3).text()
    raise AssertionError(f"no row for {instance}")


# --------------------------------------------------------------------------- #
# the right-click entry
# --------------------------------------------------------------------------- #
def test_the_canvas_menu_has_the_entry_and_it_starts_an_edit_at_once(window):
    menu = window.canvas_menu()
    texts = [a.text() for a in menu.actions()]
    assert T.MENU_TEXT in texts
    assert window.canvas.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    window.ask_context_object = lambda: dict(ANSWER)
    next(a for a in menu.actions() if a.text() == T.MENU_TEXT).trigger()
    QApplication.processEvents()
    assert BRACKET in window.db.instances(DESKTOP)
    assert window.session.editing_instance == BRACKET
    assert window.instances.selected_instance() == BRACKET
    assert "框外" in window.status_message()


def test_the_menu_stays_shut_while_sam_takes_the_right_button(window):
    assert window.canvas_menu_wanted()
    key = make_one(window)
    window.act_tool("sam_point")
    assert window.session.editing_instance == key
    assert not window.canvas_menu_wanted()          # a negative point, not a menu
    window.act_clear_edit()
    window.set_mode(A.MODE_REVIEW)
    assert not window.canvas_menu_wanted()


def test_refused_outside_annotate_and_cancel_makes_nothing(window):
    window.ask_context_object = lambda: None
    assert window.act_context_object() is None
    assert not C.context_keys(window.db.instances(DESKTOP))
    window.set_mode(A.MODE_REVIEW)
    assert window.act_context_object() is None
    assert "标注模式" in window.status_message()


def test_drawn_and_committed_in_its_own_chain_with_enter(window):
    key = make_one(window)
    commit_layer(window, square())
    kfs = window.db.keyframes(DESKTOP, VIEW, key)
    assert len(kfs) == 1 and kfs[0].placement == CONTEXT_PLACEMENT
    assert kfs[0].anchor_step == STEP
    compiled = window.session.compiled()
    assert compiled.painted.get(CONTEXT_PLACEMENT) == [key]
    # the card is the one it was; nothing about it refuses Space
    assert all(r.get("instance") != key for r in window.task_card.rows())
    assert not [p for p in window.session.current_problems() if key in p]


def test_sam_gets_no_prompt_box_for_it_and_shift_c_says_why(window):
    window._arm_prompt_box((10.0, 10.0, 30.0, 30.0))
    assert window.sam_point.prompt_box is not None
    key = make_one(window)
    assert window.session.editing_instance == key
    assert window.sam_point.prompt_box is None and window.sam_box.prompt_box is None
    window.act_cycle_prompt_box()
    assert T.NO_PROMPT_BOX in window.status_message()
    window.act_adopt_draft(at=(40.0, 40.0))
    assert T.NO_ADOPT in window.status_message()


# --------------------------------------------------------------------------- #
# I / Ctrl+I
# --------------------------------------------------------------------------- #
def test_i_on_it_is_not_in_this_frame_and_the_table_says_so(window):
    key = make_one(window)
    commit_layer(window, square())
    assert window.instances.select_instance(key)
    window.handle_key(key_event("I"))
    frame = FrameKey(DESKTOP, STEP, VIEW)
    assert marked_absent(window.db.frame_overrides(frame)[key])
    assert vis_text(window, key) == T.TAG_ABSENT
    assert "不在这张里" in window.status_message()
    assert window.instances.select_instance(key)
    window.handle_key(key_event("I"))
    assert key not in window.db.frame_overrides(frame)
    assert vis_text(window, key) != T.TAG_ABSENT


def test_i_where_it_has_no_shape_says_so_and_the_button_greys(window):
    key = make_one(window)
    window.act_clear_edit()
    assert window.instances.select_instance(key)
    why = window.absent_refusal()
    assert "没有形状" in why
    window.handle_key(key_event("I"))
    assert not window.db.frame_overrides(FrameKey(DESKTOP, STEP, VIEW))


def test_ctrl_i_marks_a_span_in_one_op(window):
    key = make_one(window)
    commit_layer(window, square())
    window.instances.select_instance(key)
    window.ask_context_range = lambda instance, absent, low, high: (True, 9)
    window.handle_key(key_event("Ctrl+I"))
    for step in range(9, STEP + 1):
        assert marked_absent(window.db.frame_overrides(FrameKey(DESKTOP, step, VIEW))[key])
    assert "不在" in window.status_message()
    window.act_undo()
    assert all(key not in window.db.frame_overrides(FrameKey(DESKTOP, s, VIEW))
               for s in range(9, STEP + 1))


def test_ctrl_z_takes_a_fresh_one_back(window):
    key = make_one(window)
    window.act_clear_edit()
    window.act_undo()
    assert key not in window.db.instances(DESKTOP)
    assert window.session.editing_instance is None


def test_ctrl_z_while_its_untouched_layer_is_open_drops_the_layer_too(window):
    key = make_one(window)
    assert window.session.editing_instance == key
    window.handle_key(key_event("Ctrl+Z"))
    QApplication.processEvents()
    assert key not in window.db.instances(DESKTOP)
    assert window.session.editing_instance is None
    assert all(r.get("key") != key for r in window.instances.rows())


# --------------------------------------------------------------------------- #
# Steps mode
# --------------------------------------------------------------------------- #
def test_the_add_parts_dialog_has_the_entry_and_it_goes_to_annotate(window):
    window.set_mode(A.MODE_STEPS)
    panel = window.steps_panel
    dialog = panel.open_add_parts()
    assert dialog.context_button is not None and dialog.context_button.text() == "框外零件…"
    window.ask_context_object = lambda: dict(ANSWER)
    dialog.context_button.click()
    QApplication.processEvents()
    assert window.mode == A.MODE_ANNOTATE
    assert BRACKET in window.db.instances(DESKTOP)
    assert window.session.editing_instance == BRACKET


def test_no_palette_button_and_the_add_parts_dialog_is_as_it_was(window):
    from tda.ui.panels.add_parts import AddPartsDialog

    palette = A.PALETTE_TOOLS + A.PALETTE_ACTIONS + A.PALETTE_STATE + A.PALETTE_OTHER
    assert all("context" not in name for name in palette)
    window.set_mode(A.MODE_STEPS)
    plain = AddPartsDialog(window.steps_panel.data)
    assert plain.context_button is None                 # only where a caller wires it
    assert set(plain.values()) == {"cls", "count", "parent", "state", "note", "attrs",
                                   "removed_at"}
    plain.deleteLater()


def test_the_steps_instance_table_lists_them_in_their_own_block(window):
    key = make_one(window)
    commit_layer(window, square())
    window.set_mode(A.MODE_STEPS)
    panel = window.steps_panel
    panel.set_desktop(DESKTOP)
    model = panel.instances_model
    assert key not in panel.data.instances and key in panel.data.context
    heading = model.keys.index(None)
    assert model.keys[heading + 1:] == [key]
    assert model.data(model.index(heading, 0)) == T.BLOCK_TITLE
    row = heading + 1
    last = len(model.columns) - 1
    assert model.data(model.index(row, 0)) == key
    assert model.data(model.index(row, last)) == "框外：台面上的支架"
    assert not model.flags(model.index(row, 2)) & Qt.ItemIsEditable      # Parent
    assert model.flags(model.index(row, last)) & Qt.ItemIsEditable
    # rename: staged like a cell edit, written by Apply
    assert model.setData(model.index(row, last), "旁边的支架", Qt.EditRole)
    assert panel.has_unapplied_edits()
    panel.apply()
    assert window.db.instances(DESKTOP)[key].attrs["label"] == "旁边的支架"
    assert window.db.ops_of_kind(DESKTOP, C.OP_LABEL)
    # its menu deletes, never merges; the delete takes the shapes with it
    menu = panel.instances_menu(row)
    texts = [a.text() for a in menu.actions()]
    assert len(texts) == 1 and "删除框外零件" in texts[0]
    panel.delete_instance(key)
    assert key not in window.db.instances(DESKTOP)
    assert window.db.keyframes(DESKTOP, VIEW, key) == []
    assert window.db.ops_of_kind(DESKTOP, C.OP_DELETE)
    assert None not in panel.instances_model.keys


def test_no_step_editor_and_no_question_names_it(window):
    key = make_one(window)
    window.act_clear_edit()
    window.set_mode(A.MODE_STEPS)
    panel = window.steps_panel
    panel.set_desktop(DESKTOP)
    assert key not in panel.data.instance_keys()
    assert not [issue for issue in panel.data.issues if key in issue]
    assert key not in {e.target for e in panel.data.relations.view().edges}
    assert key not in {e.blocker for e in panel.data.relations.view().edges}


# --------------------------------------------------------------------------- #
# the components gallery
# --------------------------------------------------------------------------- #
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


def test_a_gallery_shot_can_have_one_too(gallery_window):
    win = gallery_window
    card = [r.get("instance") for r in win.task_card.rows()]
    win.ask_context_object = lambda: {"cls": "other", "label": "茶杯", "note": ""}
    key = win.act_context_object()
    assert key == "ctx.other.01"
    hw = win.session.compiled().key and win.session.image().shape[:2]
    mask = np.zeros(hw, dtype=bool)
    mask[2:8, 2:8] = True
    commit_layer(win, mask)
    shot = win.session.current()
    compiled = win.session.compiled()
    assert compiled.painted.get(CONTEXT_PLACEMENT) == [key]
    assert [r.get("instance") for r in win.task_card.rows()] == card
    # I there: 「不在这张里」 for it too, not the card's 看不见
    win.instances.select_instance(key)
    win.act_mark_absent()
    assert marked_absent(win.db.frame_overrides(shot)[key])
