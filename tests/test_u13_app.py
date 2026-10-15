"""Task U13: the state actions as buttons -- the palette's 「状态」 group.

The annotator asked whether the state actions have visible controls or only
keys. ``Ctrl+T`` and ``Ctrl+Shift+K`` had keys only, and none of the five was on
the tool palette. These tests pin the group:

* five buttons under their own heading, in Annotate mode only, each running its
  key's slot -- the very ``dispatch`` the key goes through;
* greyed with the key's own sentence where the key would refuse: nothing
  selected, a removed row, a class with no state to change, a part that is not
  a two-way cover, a shape that is not a cut copy, pixels not committed, a
  components-gallery shot (and, for 「看不见」, anywhere else);
* tooltips with the Chinese label and the key, captions short enough for the
  strip, and the session asked for a reason once per frame, not per refresh.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from app_scene import (
    DESKTOP,
    LAST_STEP,
    StubSamQueue,
    close_window,
    make_paths,
    make_session,
    seed_shapes,
)
from test_app_actions import key_event
from tda.core import masks
from tda.core.model import ShapeKeyframe, ShapePart
from tda.ui import app_actions as A
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.app_gallery import NOT_A_GALLERY
from tda.ui.app_state import CONFIRM_DIRTY, NO_TARGET_FOR, REMOVED_ROW
from tda.ui.app_verb import NO_VERB_TARGET
from tda.ui.gallery import GALLERY_REFUSALS
from tda.ui.panels.palette import PALETTE_WIDTH
from tda.ui.session_commits import NOT_A_CUT_COPY

import gallery_scene as GS

GROUP = ("cycle_state", "state_range", "correct_verb", "confirm_cut_copy", "mark_absent")
CORRECTED = "ram_latch.02"     # closed throughout the 14-step scene
COVER = "cover.01"             # the log opens it at 2: offered 「其实拆下」
LATCH = "ram_latch.01"         # grows on the board: open only, never taken out alone
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


def goto(win: MainWindow, step: int) -> None:
    win.session.goto(step, force=True)
    QApplication.processEvents()


@pytest.fixture
def window(qapp, tmp_path: Path):
    """The 14-step D13 scene of the window tests, shapes seeded, at step 12."""
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(1400, 900)
    answer_roi(win)
    seed_shapes(win.session, LAST_STEP)
    goto(win, STEP)
    yield win
    close_window(win)


@pytest.fixture
def cut_window(qapp, tmp_path: Path):
    """The D13 sheet to step 42, walked as in U11: a cut copy of ram_latch.01 at 40."""
    session = make_session(tmp_path, last_step=42)
    session.sweeper_enabled = False
    db = session.db
    for key, rec in db.instances(DESKTOP).items():
        if key.startswith("ram_latch."):
            rec.parent, rec.attached = "motherboard.01", True
            db.upsert_instance(rec)
    shape = np.zeros((64, 64), dtype=bool)
    shape[40:50, 40:50] = True                    # the open clip, 1-42
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=LATCH, desktop=DESKTOP, view="scan", pose_segment=1,
        anchor_step=42, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(shape))]))
    session._invalidate()
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    answer_roi(win)
    goto(win, 41)
    assert win.instances.select_instance(LATCH)
    win.act_cycle_state()                         # closed at 41
    win.session.begin_edit(LATCH)
    closed = np.zeros((64, 64), dtype=bool)
    closed[40:44, 40:56] = True
    win.session.set_editing_mask(closed)
    win.act_commit()
    if win._pending_warning is not None:          # "too big" for a clip
        win.act_commit()
    assert win.status_message().startswith("已先在第 40 步切开旧版本")
    goto(win, 40)
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


def state(win: MainWindow, name: str) -> tuple[bool, str]:
    """``(enabled, reason)`` of a 「状态」 button after a refresh."""
    win.refresh_guidance()
    button = win.palette.button(name)
    return button.isEnabled(), button.reason()


def select_nothing(win: MainWindow) -> None:
    win.instances.table().setCurrentCell(-1, -1)
    QApplication.processEvents()
    assert win.session.editing_instance is None


def manual(win: MainWindow, target: str) -> list[tuple]:
    return [(e.step, e.old, e.new) for e in win.db.events(DESKTOP)
            if not e.auto and e.target == target]


def click(button) -> None:
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     button.rect().center())
    QApplication.processEvents()


# --------------------------------------------------------------------------- #
# the group
# --------------------------------------------------------------------------- #
def test_the_group_is_the_five_state_keys_in_order():
    assert A.PALETTE_STATE == GROUP
    assert [A.action_named(n).keys for n in GROUP] == [
        ("T",), ("Ctrl+T",), ("Shift+T",), ("Ctrl+Shift+K",), ("I",)]
    others = set(A.PALETTE_TOOLS) | set(A.PALETTE_ACTIONS) | set(A.PALETTE_REVIEW)
    assert not others & set(GROUP)
    for name in GROUP:
        action = A.action_named(name)
        head, note = A.short_parts(action)
        assert action.modes == (A.MODE_ANNOTATE,)
        # short enough for the strip: a name of at most four characters, no note
        assert 0 < len(head) <= 4 and not note, (name, action.short)
    assert [A.short_parts(A.action_named(n))[0] for n in GROUP] == [
        "换状态", "一段状态", "其实拆下", "形状没变", "看不见"]


def test_the_group_sits_under_its_own_heading_after_the_actions(window):
    inner = window.palette.scroll.widget().layout()
    widgets = [inner.itemAt(i).widget() for i in range(inner.count())]
    widgets = [w for w in widgets if w is not None]
    header = window.palette.state_header
    assert isinstance(header, QLabel) and header.text() == "状态"
    at = widgets.index(header)
    buttons = window.palette.buttons()
    assert widgets[at + 1:at + 6] == [buttons[n] for n in GROUP]
    # below the 操作 group: the commit keys stay where the hand knows them
    assert max(widgets.index(buttons[n]) for n in A.PALETTE_ACTIONS) < at
    # a heading like the others
    operations = next(w for w in widgets if isinstance(w, QLabel) and w.text() == "操作")
    assert header.styleSheet() == operations.styleSheet()
    assert header.font().bold()


def test_annotate_shows_the_group_review_and_steps_hide_it(window):
    palette = window.palette
    assert not palette.state_header.isHidden()
    assert all(not palette.button(n).isHidden() for n in GROUP)
    window.set_mode(A.MODE_REVIEW)
    assert palette.state_header.isHidden()
    assert all(palette.button(n).isHidden() for n in GROUP)
    window.set_mode(A.MODE_STEPS)
    assert palette.isHidden()
    window.set_mode(A.MODE_ANNOTATE)
    assert not palette.isHidden() and not palette.state_header.isHidden()
    assert all(not palette.button(n).isHidden() for n in GROUP)


@pytest.mark.parametrize("name", GROUP)
def test_each_button_emits_its_action_and_reaches_its_slot(window, name):
    """The click is the key: sigAction, run_palette_action, dispatch, the slot."""
    action = A.action_named(name)
    emitted: list[tuple] = []
    calls: list[tuple] = []
    window.palette.sigAction.connect(lambda n, pressed: emitted.append((n, pressed)))
    setattr(window, action.slot, lambda *args: calls.append(args))
    button = window.palette.button(name)
    button.setEnabled(True)            # the gate is the slot's; here only the wiring
    click(button)
    assert emitted == [(name, True)]
    assert calls == [tuple(action.args)]


def test_the_tooltips_carry_the_chinese_label_and_the_key(window):
    for name in GROUP:
        action = A.action_named(name)
        button = window.palette.button(name)
        assert button.key == action.keys[0]
        tip = button.toolTip()
        assert tip.startswith(f"{action.label_zh}（{action.keys[0]}）"), tip
        assert action.label in tip


def test_the_palette_keeps_its_width(window):
    """Short captions: the strip stays 124 px and no state button is taller than
    the tallest button already on it."""
    window.show()
    QApplication.processEvents()
    assert window.palette.width() == PALETTE_WIDTH
    tallest = max(window.palette.button(n).sizeHint().height() for n in A.PALETTE_ACTIONS)
    for name in GROUP:
        button = window.palette.button(name)
        assert button.sizeHint().width() <= PALETTE_WIDTH - 12
        assert button.width() <= PALETTE_WIDTH
        assert button.sizeHint().height() <= tallest, name


# --------------------------------------------------------------------------- #
# greyed, with the key's own sentence
# --------------------------------------------------------------------------- #
def test_nothing_selected_greys_them_with_the_keys_own_sentences(window):
    select_nothing(window)
    assert state(window, "cycle_state") == (False, NO_TARGET_FOR["cycle_state"])
    assert state(window, "state_range") == (False, NO_TARGET_FOR["state_range"])
    assert state(window, "correct_verb") == (False, NO_VERB_TARGET)
    assert state(window, "confirm_cut_copy") == (False, NO_TARGET_FOR["confirm_cut_copy"])
    assert state(window, "mark_absent") == (False, NOT_A_GALLERY)
    for name in GROUP:
        button = window.palette.button(name)
        assert f"现在不能用：{button.reason()}" in button.toolTip()
        # the key says the same thing
        window.handle_key(key_event(A.action_named(name).keys[0]))
        assert button.reason() in window.status_message(), name


def test_a_click_on_a_greyed_one_says_why(window):
    select_nothing(window)
    window.show()
    QApplication.processEvents()
    button = window.palette.button("state_range")
    assert not button.isEnabled()
    asked: list = []
    window.ask_state_range = lambda *a: asked.append(a)
    click(button)
    assert asked == []
    assert window.status_message() == f"一段状态：{NO_TARGET_FOR['state_range']}"


def test_the_sessions_refusals_grey_them_too(window):
    """A class with no state, a latch that only opens, a shape that is no cut copy."""
    assert window.instances.select_instance("chassis")
    enabled, why = state(window, "cycle_state")
    assert not enabled and "只有一种" in why
    assert state(window, "state_range") == (False, why)
    window.handle_key(key_event("T"))
    assert window.status_message() == f"改不了状态：{why}"
    assert window.instances.select_instance(LATCH)
    enabled, why = state(window, "correct_verb")
    assert not enabled and "只能打开" in why
    window.handle_key(key_event("Shift+T"))
    assert window.status_message() == f"改不了：{why}"
    assert state(window, "confirm_cut_copy") == (
        False, NOT_A_CUT_COPY.format(instance=LATCH))
    window.handle_key(key_event("Ctrl+Shift+K"))
    assert NOT_A_CUT_COPY.format(instance=LATCH) in window.status_message()
    # the latch's state can be set here, so T is live
    assert state(window, "cycle_state") == (True, "")
    assert window.instances.select_instance(COVER)
    assert state(window, "correct_verb") == (True, "")


def test_a_removed_row_greys_the_state_keys(window):
    window.instances.show_removed.setChecked(True)
    QApplication.processEvents()
    rows = len(window.instances.rows())
    if window.instances.table().rowCount() <= rows:
        pytest.skip("nothing is removed by this frame of the scene")
    window.instances.table().setCurrentCell(rows, 1)
    QApplication.processEvents()
    assert window.instances.selected_is_removed()
    for name in ("cycle_state", "state_range", "confirm_cut_copy"):
        assert state(window, name) == (False, REMOVED_ROW), name
    window.handle_key(key_event("T"))
    assert window.status_message() == REMOVED_ROW


def test_pixels_not_committed_grey_ctrl_shift_k_and_shift_t(window):
    window.instances.select_instance(COVER)
    window.on_request_edit(COVER)
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
    assert state(window, "confirm_cut_copy") == (False, CONFIRM_DIRTY)
    assert state(window, "correct_verb") == (
        False, api.VERB_NEEDS_CLEAN_LAYER.format(instance=COVER))
    window.handle_key(key_event("Shift+T"))
    assert api.VERB_NEEDS_CLEAN_LAYER.format(instance=COVER) in window.status_message()
    window.act_clear_edit()
    assert state(window, "correct_verb") == (True, "")


def test_on_a_gallery_shot_only_the_absent_button_is_live(gallery_window):
    win = gallery_window
    for name in ("cycle_state", "state_range", "correct_verb", "confirm_cut_copy"):
        assert state(win, name) == (False, GALLERY_REFUSALS[name]), name
    part = win.task_card.current_instance()
    assert part and state(win, "mark_absent") == (True, "")
    click(win.palette.button("mark_absent"))
    row = next(r for r in win.task_card.rows() if r["instance"] == part)
    assert row["done"] and row["absent"]
    # again takes it back: the card's part is the next open row now, so ask for it
    assert win.absent_refusal(part) == ""


# --------------------------------------------------------------------------- #
# a live click is the key
# --------------------------------------------------------------------------- #
def test_the_state_buttons_do_what_t_and_ctrl_t_do(window, monkeypatch):
    window.show()
    QApplication.processEvents()
    assert window.instances.select_instance(CORRECTED)
    assert state(window, "cycle_state") == (True, "")
    click(window.palette.button("cycle_state"))
    assert manual(window, CORRECTED) == [(STEP, "closed", "open"),
                                         (STEP + 1, "open", "closed")]
    assert "ram_latch.02 在第 12 步：打开（之前为 关上）" in window.status_message()
    assert QApplication.focusWidget() is window.canvas      # the keyboard stays home
    click(window.palette.button("cycle_state"))
    assert manual(window, CORRECTED) == []
    asked: dict = {}

    def answer(instance, choices, held, to_step):
        asked.update(instance=instance, choices=choices)
        return "open", 10

    monkeypatch.setattr(window, "ask_state_range", answer)
    window.instances.select_instance(CORRECTED)
    click(window.palette.button("state_range"))
    assert asked == {"instance": CORRECTED, "choices": ["closed", "open"]}
    assert manual(window, CORRECTED) == [(10, "closed", "open"),
                                         (STEP + 1, "open", "closed")]
    assert "第 10–12 步都是 打开" in window.status_message()


def test_the_verb_button_does_what_shift_t_does(window):
    window.show()
    QApplication.processEvents()
    assert window.instances.select_instance(COVER)
    assert state(window, "correct_verb") == (True, "")
    click(window.palette.button("correct_verb"))
    assert [(a.step, a.verb) for a in window.db.actions(DESKTOP)
            if a.target == COVER] == [(2, "remove")]
    assert "cover.01 在第 2 步是拆下的（不是打开）" in window.status_message()


def test_a_cut_copy_frame_lights_the_button_and_a_click_confirms_it(cut_window):
    win = cut_window
    win.show()
    QApplication.processEvents()
    assert win.instances.select_instance(LATCH)
    assert state(win, "confirm_cut_copy") == (True, "")
    click(win.palette.button("confirm_cut_copy"))
    assert win.status_message().startswith(f"内存卡扣 {LATCH}：切开时留下的形状就是这几帧的样子")
    row = next(r for r in win.task_card.rows() if r["instance"] == LATCH)
    assert row["done"] and not row.get("cut_copy")
    # confirmed, it is the annotator's shape: nothing left to confirm
    assert state(win, "confirm_cut_copy") == (False, NOT_A_CUT_COPY.format(instance=LATCH))
    win.act_undo()
    assert state(win, "confirm_cut_copy") == (True, "")


# --------------------------------------------------------------------------- #
# cost: the session is asked once per frame, part and re-read of the list
# --------------------------------------------------------------------------- #
def test_the_session_is_asked_for_a_reason_once_per_frame(window, monkeypatch):
    asked: list[str] = []
    real = window.session.state_gate
    monkeypatch.setattr(window.session, "state_gate",
                        lambda part: asked.append(part) or real(part))
    assert window.instances.select_instance("chassis")
    for _ in range(5):
        window.refresh_guidance()
    assert asked == ["chassis"]
    goto(window, STEP - 1)
    window.instances.select_instance("chassis")
    window.refresh_guidance()
    assert asked[-1] == "chassis" and len(asked) > 1    # another frame: asked again
    before = len(asked)
    for _ in range(5):
        window.refresh_guidance()
    assert len(asked) == before
    # a part whose row offers its states costs no question at all
    asked.clear()
    window.instances.select_instance(CORRECTED)
    window.refresh_guidance()
    assert asked == [] and window.palette.button("cycle_state").isEnabled()


# --------------------------------------------------------------------------- #
# the documents
# --------------------------------------------------------------------------- #
def _squash(text: str) -> str:
    return "".join(text.split())


def test_the_documents_say_it():
    repo = Path(__file__).resolve().parents[1]
    guide = _squash((repo / "docs" / "annotation_guide.md").read_text(encoding="utf-8"))
    assert _squash("每个工具和提交/确认/撤销/状态键各一个按钮") in guide
    quick = _squash((repo / "docs" / "trial_quickstart.md").read_text(encoding="utf-8"))
    assert _squash("下面是操作，最下面是**状态**") in quick
    spec = _squash((repo / "docs" / "superpowers" / "specs"
                    / "2026-09-17-teardown-annotator-design.md").read_text(encoding="utf-8"))
    assert _squash("**v1.13（2026-10-14）**：**状态键有了按钮**（U13）") in spec
