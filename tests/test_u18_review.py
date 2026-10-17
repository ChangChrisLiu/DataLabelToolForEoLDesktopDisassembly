"""Task U18, review round: the Steps delete and the undo history, the Steps block
kept current, keys never reused, the delete question, ``I`` round trips, the docs.

B1: a context object deleted in Steps mode (written through) left its ops on the
annotate undo stack; ``Ctrl+Z`` / ``Ctrl+Y`` then wrote an orphan keyframe and a
ghost layer entry, and the next ``ctx.other.01`` inherited a shape nobody drew.
B2: the Steps panel's 「框外零件」 block did not follow Annotate mode.
M1-M4 and the nits as the review lists them.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
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
from tda.core.model import FrameKey
from tda.ui import app_actions as A
from tda.ui import context_text as T
from tda.ui.app import MainWindow
from tda.ui.app_context import CONTEXT_UNDO_CLEARED, RANGE_ABSENT_NOTE, RANGE_BACK_NOTE
from tda.ui.session_api import SCOPE_FRAME_OVERRIDE, SCOPE_KEYFRAME

STEP = 12


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(1400, 900)
    win.show()
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()
    seed_shapes(win.session, LAST_STEP)
    win.session.goto(STEP, force=True)
    QApplication.processEvents()
    yield win
    close_window(win)


def square(y0: int = 44, x0: int = 44, side: int = 12) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=bool)
    mask[y0:y0 + side, x0:x0 + side] = True
    return mask


def make_one(win: MainWindow, cls: str, label: str) -> str:
    win.ask_context_object = lambda: {"cls": cls, "label": label, "note": ""}
    return win.act_context_object()


def enter(win: MainWindow, mask: np.ndarray) -> None:
    """Paint, then the real ``Return`` key (and again past a size warning)."""
    win.set_editing_mask(mask, undoable=True)
    QApplication.processEvents()
    win.handle_key(key_event("Return"))
    for _ in range(2):
        if win._pending_warning is None:
            break
        win.handle_key(key_event("Return"))
    QApplication.processEvents()


def orphans(win: MainWindow) -> list[tuple]:
    return [tuple(r) for r in win.db.conn.execute(
        "SELECT k.instance, k.anchor_step FROM shape_keyframe k LEFT JOIN instance i "
        "ON i.desktop = k.desktop AND i.key = k.instance WHERE i.key IS NULL").fetchall()]


def ghosts(win: MainWindow, key: str) -> list:
    return [e for e in win.db.zorder(DESKTOP, VIEW, 1).order if e[0] == key]


def steps_panel(win: MainWindow):
    win.set_mode(A.MODE_STEPS)
    QApplication.processEvents()
    assert win.mode == A.MODE_STEPS
    return win.steps_panel


# --------------------------------------------------------------------------- #
# B1: a Steps delete clears the annotate undo history
# --------------------------------------------------------------------------- #
def test_b1_ctrl_z_ctrl_y_after_a_steps_delete_write_nothing(window):
    key = make_one(window, "other", "茶杯")
    enter(window, square())
    assert window.db.keyframes(DESKTOP, VIEW, key)
    panel = steps_panel(window)
    panel.delete_instance(key)
    assert CONTEXT_UNDO_CLEARED in window.status_message()
    window.set_mode(A.MODE_ANNOTATE)
    QApplication.processEvents()
    stack = window.session.undo_stack
    assert not stack.can_undo and not stack.can_redo
    for spec in ("Ctrl+Z", "Ctrl+Y", "Ctrl+Z"):
        window.handle_key(key_event(spec))
        QApplication.processEvents()
        assert key not in window.db.instances(DESKTOP), spec
        assert orphans(window) == [], spec
        assert ghosts(window, key) == [], spec
    # the next one of its class gets a key of its own, with no shape it never had
    new = make_one(window, "other", "另一个杯子（新）")
    window.act_clear_edit()
    QApplication.processEvents()
    assert new == "ctx.other.02"
    assert window.db.keyframes(DESKTOP, VIEW, new) == []
    assert new not in window.session.compiled().instances


# --------------------------------------------------------------------------- #
# B2: the Steps block follows Annotate mode
# --------------------------------------------------------------------------- #
def test_b2_one_made_in_annotate_shows_in_steps_without_revert(window):
    panel = steps_panel(window)                      # the panel is built and loaded
    assert panel.data.context_keys() == []
    window.set_mode(A.MODE_ANNOTATE)
    key = make_one(window, "cooler_bracket", "支架")
    enter(window, square())
    panel = steps_panel(window)
    assert panel.data.context_keys() == [key]
    assert key in panel.instances_model.keys
    assert not panel.database_moved() and not panel.has_unapplied_edits()


def test_b2_one_undone_in_annotate_leaves_the_steps_block(window):
    key = make_one(window, "cooler_bracket", "支架")
    window.act_clear_edit()
    panel = steps_panel(window)
    assert panel.data.context_keys() == [key]
    window.set_mode(A.MODE_ANNOTATE)
    window.handle_key(key_event("Ctrl+Z"))            # takes the creation back
    QApplication.processEvents()
    assert key not in window.db.instances(DESKTOP)
    panel = steps_panel(window)
    assert panel.data.context_keys() == []
    assert None not in panel.instances_model.keys     # no empty block heading
    assert "ContextError" not in panel.status.text()


def test_b2_a_rename_in_the_signature_and_the_moved_test(window):
    key = make_one(window, "cooler_bracket", "支架")
    window.act_clear_edit()
    panel = steps_panel(window)
    assert not panel.has_unapplied_edits()
    panel.data.relabel_context(key, "旁边的支架")
    assert panel.has_unapplied_edits()
    # the database moves under the panel: the block alone is re-read, the
    # staged rename stays on top
    C.relabel_context(window.db, DESKTOP, key, "库里改的")
    assert panel.context_moved() and panel.database_moved()
    panel.refresh_context()
    assert not panel.database_moved()
    assert panel.data.context[key].attrs["label"] == "旁边的支架"
    assert panel.has_unapplied_edits()
    panel.apply()
    assert window.db.instances(DESKTOP)[key].attrs["label"] == "旁边的支架"
    assert not panel.has_unapplied_edits() and not panel.database_moved()


# --------------------------------------------------------------------------- #
# M1: a key is never handed out twice
# --------------------------------------------------------------------------- #
def confirm(session, *steps) -> None:
    for step in steps:
        session.goto(step, force=True)
        assert session.confirm_frame()


def test_m1_a_deleted_objects_rows_and_conflicts_stay_its_own(window):
    session = window.session
    cup = session.add_context_object("other", "茶杯")
    session.begin_edit(cup)
    session.set_editing_mask(square(40, 40, 20))
    session.commit_edit(SCOPE_KEYFRAME)
    session.clear_edit()
    session.goto(LAST_STEP, force=True)
    session.begin_edit(cup)                         # on the start frame too: all drawn
    session.set_editing_mask(square(40, 40, 20))
    session.commit_edit(SCOPE_KEYFRAME)
    session.clear_edit()
    confirm(session, LAST_STEP)
    assert C.confirmed_frames(session.db, DESKTOP, cup) == {VIEW: [LAST_STEP]}
    panel = steps_panel(window)
    panel.delete_instance(cup)
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    opened = [r["instance"] for r in session.db.conflicts(DESKTOP, VIEW)]
    assert opened == [cup]
    window.set_mode(A.MODE_ANNOTATE)
    pen = session.add_context_object("other", "笔")
    assert pen == "ctx.other.02" and pen != cup
    assert not session.db.compiled(FrameKey(DESKTOP, LAST_STEP, VIEW)).get(pen)
    assert [r["instance"] for r in session.db.conflicts(DESKTOP, VIEW)] == [cup]
    assert cup in C.keys_in_use(session.db, DESKTOP)


def test_m1_keys_in_use_counts_every_table_and_every_creation(window):
    db = window.db
    rec = C.plan_context(db.instances(DESKTOP), window.session.tax, DESKTOP, "other", "a")
    C.write_context(db, rec, "t")
    with db.transaction():
        C.remove_context(db, DESKTOP, rec.key)
    assert rec.key not in db.instances(DESKTOP)
    assert rec.key in C.keys_in_use(db, DESKTOP)       # the op_log remembers it
    nxt = C.plan_context(db.instances(DESKTOP), window.session.tax, DESKTOP, "other", "b",
                         taken=C.keys_in_use(db, DESKTOP))
    assert nxt.key == "ctx.other.02"


# --------------------------------------------------------------------------- #
# M2: the delete asks first when confirmed rows exist
# --------------------------------------------------------------------------- #
def test_m2_the_delete_asks_and_says_what_it_re_checks(window):
    session = window.session
    session.goto(LAST_STEP, force=True)
    cup = session.add_context_object("other", "茶杯")
    session.begin_edit(cup)
    session.set_editing_mask(square(40, 40, 20))
    session.commit_edit(SCOPE_KEYFRAME)
    session.clear_edit()
    confirm(session, LAST_STEP)
    panel = steps_panel(window)
    asked: list = []
    panel.ask_delete_context = lambda key, frames: asked.append((key, frames)) or False
    panel.prompt_delete_context(cup)
    assert asked == [(cup, {VIEW: [LAST_STEP]})]
    assert cup in window.db.instances(DESKTOP) and "没有删除" in panel.status.text()
    panel.ask_delete_context = lambda key, frames: True
    panel.prompt_delete_context(cup)
    assert cup not in window.db.instances(DESKTOP)
    assert "1 个已确认帧" in panel.status.text() and f"{VIEW} 第 {LAST_STEP} 帧" in panel.status.text()
    question = __import__("tda.ui.panels.steptable", fromlist=["x"]).CONTEXT_DELETE_ASK
    assert "Review" in question and "冲突" in question


def test_m2_no_question_without_confirmed_rows(window):
    key = make_one(window, "other", "茶杯")
    enter(window, square())
    panel = steps_panel(window)
    panel.ask_delete_context = lambda key, frames: pytest.fail("asked")
    panel.prompt_delete_context(key)
    assert key not in window.db.instances(DESKTOP)
    assert "已删除框外零件" in panel.status.text() and "已确认帧" not in panel.status.text()


# --------------------------------------------------------------------------- #
# M3: I / Ctrl+I keep a single-frame mask under the label
# --------------------------------------------------------------------------- #
def test_m3_i_i_round_trips_an_alt_enter_mask(window):
    session = window.session
    key = session.add_context_object("other", "茶杯")
    session.begin_edit(key)
    session.set_editing_mask(square(40, 40, 20))
    session.commit_edit(SCOPE_KEYFRAME)
    session.goto(11, force=True)
    session.begin_edit(key)
    session.set_editing_mask(square(44, 44, 6))
    session.commit_edit(SCOPE_FRAME_OVERRIDE)
    session.clear_edit()
    frame = FrameKey(DESKTOP, 11, VIEW)
    hand = session.db.frame_overrides(frame)[key].visible_rle
    assert int(session.compiled().instances[key].visible.sum()) == 36
    session.toggle_context_absent(key)
    held = session.db.frame_overrides(frame)[key]
    assert held.visibility == "out_of_view" and held.visible_rle == hand
    assert session.compiled().instances[key].visible is None
    assert key in session.context_absent_here()
    session.toggle_context_absent(key)
    back = session.db.frame_overrides(frame)[key]
    assert back.visibility is None and back.visible_rle == hand
    assert int(session.compiled().instances[key].visible.sum()) == 36
    # Ctrl+I over the span, set and taken back: the same round trip
    session.set_context_absent_range(key, True, 9)
    assert session.db.frame_overrides(frame)[key].visible_rle == hand
    session.set_context_absent_range(key, False, 9)
    assert session.db.frame_overrides(frame)[key].visible_rle == hand
    assert session.db.frame_overrides(frame)[key].visibility is None
    assert key not in session.db.frame_overrides(FrameKey(DESKTOP, 10, VIEW))
    assert int(session.compiled().instances[key].visible.sum()) == 36


# --------------------------------------------------------------------------- #
# nits
# --------------------------------------------------------------------------- #
def test_n1_the_i_button_tooltip_names_the_context_meaning(window):
    key = make_one(window, "cooler_bracket", "支架")
    enter(window, square())
    window.instances.select_instance(key)
    window.refresh_guidance()
    tip = window.palette.button("mark_absent").toolTip()
    assert "框外零件 = 不在这张里" in tip
    assert "框外零件" in A.action_named("cable_range").label_zh


def test_n2_the_ctrl_i_dialog_line_follows_the_option(window):
    from tda.ui.app_context import ContextRangeDialog

    dialog = ContextRangeDialog(window, "框外·支架", STEP, True, 1, LAST_STEP)
    assert dialog.explain.text() == RANGE_ABSENT_NOTE
    dialog.what.setCurrentIndex(1)
    assert dialog.explain.text() == RANGE_BACK_NOTE and "撤回" in RANGE_BACK_NOTE
    dialog.deleteLater()


def test_n3_the_area_priors_leave_context_objects_out(window):
    from experiments.area_priors_from_db import collect

    before, _ = collect(window.db.path)
    key = window.session.add_context_object("screw", "地上的螺丝")
    window.session.begin_edit(key)
    window.session.set_editing_mask(square(2, 2, 30))
    window.session.commit_edit(SCOPE_KEYFRAME)
    after, _ = collect(window.db.path)
    assert {c: len(v) for c, v in after.items()} == {c: len(v) for c, v in before.items()}


def test_m4_the_quickstart_example_is_a_hypothetical(window):
    body = (Path(__file__).resolve().parents[1] / "docs" / "trial_quickstart.md").read_text(
        encoding="utf-8")
    section = body.split("### 机器外面的东西（「框外零件」）", 1)[1].split("\n#", 1)[0]
    assert "假如 D5 第 20 帧" in section and "D5 的画面里并没有" in section
    assert "例子：D5 第 20 帧，机箱旁边" not in section
    assert "已经确认过的帧复检时会回到待复核" in section
    assert T.MENU_TEXT in section
