"""Task U43: 「这一步的变化」 -- the card's parts painted on the whole scanner frame.

U23's scene (D13, 14 steps, every chassis part of step 14 drawn in the scanner
view, a picture per step on ``oak1`` / ``oak2`` / ``rs``), the window on
``oak1``.  What the cards ask for there: step 13 the RAM latch (drawn in the
scanner view), step 12 the CPU fan and its four screws (never drawn there --
they are out at 14), step 1 the cover (drawn) and a cooler screw's state
(not drawn), step 14 the start card (32 parts, all drawn).

* the scene: the card's parts that have scanner geometry are painted, the
  others counted and not painted, a row naming no part is no part;
* an empty card says so; U23's earlier-frame clause is in the caption;
* the dock's button cycles 定位 → 拆前/拆后 → 这一步的变化, names the next
  kind, and the choice sticks -- across parts and in the INI;
* the first-time pair is 定位's rule only; the scanner view never shows it;
* the mouse names the part in the caption's second line; a click is whole ↔
  crop, the wheel zooms about the mouse;
* nothing written, no new key, a scene built once per frame and card, and a
  40-part frame built well under 100 ms with warm caches.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QShortcut, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_scene import COOLER, DESKTOP, close_window
from test_u23_locator import (
    CHASSIS,
    STEP,
    goto,
    make_scene,
    open_window,
    select,
    select_nothing,
)
from tda.core.model import FrameKey
from tda.core.truth import TruthService
from tda.ui.app_locator import (
    KIND_CHANGE,
    KIND_PAIR,
    KIND_SCENE,
    KIND_SETTING,
    KIND_TEXT,
    next_kind,
)
from tda.ui.canvas.overlay import palette_color
from tda.ui.locator_change import CHANGE_NAME, COUNT_LINE, EMPTY_CARD, ChangeMaker
from tda.ui.locator_source import NOT_ANNOTATED
from tda.ui.panels.locator import CROP, MIN_CROP_FRAC, WHOLE
from tda.ui.session import AnnotationSession

#: Drawn in the scanner view; the only part of step 13's card.
LATCH = "ram_latch.01"
SCREWS = tuple(f"screw.cpu_cooler.0{i}" for i in range(1, 5))


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    """On ``oak1``, step 12; the CPU fan has no shape on the OAK / RealSense views."""
    session, paths = make_scene(tmp_path, own_skip=(COOLER,))
    win = open_window(session, paths)
    goto(win, STEP)
    yield win
    close_window(win)


def kind(win) -> str:
    if not win.locator_shown():
        return "down"
    if win.locator.showing_change():
        return "change"
    return "pair" if win.locator.showing_pair() else "location"


def click_kind(win) -> None:
    win.locator.pair_button.click()
    QApplication.processEvents()


def to_change(win) -> None:
    win.set_locator_mode(KIND_CHANGE)
    QApplication.processEvents()
    assert kind(win) == "change"


def card(win) -> list:
    return [row.get("instance") for row in win.task_card.rows()]


def widget_point(view, x: float, y: float) -> QPointF:
    """Where frame pixel ``(x, y)`` is on the change view."""
    src = view.source_rect()
    target = view._target(src)
    return QPointF(target.x() + (x - src[0]) * target.width() / (src[2] - src[0]),
                   target.y() + (y - src[1]) * target.height() / (src[3] - src[1]))


def move(view, pos: QPointF) -> None:
    event = QMouseEvent(QEvent.Type.MouseMove, pos, view.mapToGlobal(pos),
                        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(view, event)


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #
def test_the_cards_parts_with_scanner_geometry_are_painted(window):
    source = window.locator_source
    goto(window, 13)
    change = source.change(DESKTOP, 13, card(window))
    assert change.card == (LATCH,)
    assert [p.instance for p in change.parts] == [LATCH]
    found = change.parts[0]
    assert found.colour == tuple(palette_color(LATCH))
    assert found.geometry.kind == "mask" and change.box == found.geometry.box
    assert change.caption == f"主视角 第 13 帧 · {CHANGE_NAME}"
    assert change.line == COUNT_LINE.format(n=1, m=1) == "卡片上 1 个零件，扫描仪上画了 1 个"
    assert change.note == ""
    # the label image is at the thumbnail's size and names the part at its centre
    assert change.labels.shape == (change.image.height(), change.image.width())
    assert change.part_at(*found.geometry.centre) is found
    assert change.part_at(0.5, 63.5) is None
    label, state = source.label_and_state(DESKTOP, 13, LATCH)
    assert found.text == f"{label}（{state}）" and LATCH in label and state


def test_parts_never_drawn_in_the_scanner_view_are_counted_not_painted(window):
    source = window.locator_source
    # step 12: the fan and its screws are out at 14, so never drawn there
    change = source.change(DESKTOP, STEP, card(window))
    assert set(change.card) == {COOLER, *SCREWS}
    assert change.parts == () and change.box is None and change.labels is None
    assert change.line == "卡片上 5 个零件，扫描仪上画了 0 个"
    # step 1: the cover drawn, a cooler screw's state change not
    goto(window, 1)
    change = source.change(DESKTOP, 1, card(window))
    assert change.card == ("cover.01", "screw.cpu_cooler.01")
    assert [p.instance for p in change.parts] == ["cover.01"]
    assert change.line == "卡片上 2 个零件，扫描仪上画了 1 个"


def test_rows_naming_no_part_are_no_parts_and_a_part_named_twice_counts_once(window):
    source = window.locator_source
    change = source.change(DESKTOP, 13, ["Remove the cover", LATCH, LATCH, None, ""])
    assert change.card == (LATCH,) and len(change.parts) == 1
    empty = source.change(DESKTOP, 13, ["Remove the cover"])     # a dupli frame's row
    assert empty.card == () and empty.parts == ()
    assert empty.line == EMPTY_CARD == "这一帧的卡片上没有零件"
    assert empty.image is not None and empty.frame_hw == (64, 64)


def test_the_start_card_paints_every_part_and_small_ones_stay_on_top(window):
    goto(window, 14)
    change = window.locator_source.change(DESKTOP, 14, card(window))
    assert len(change.card) == len(change.parts) == 32
    for part in change.parts:
        # each part's own id is under its centre: none hidden under another
        assert change.part_at(*part.geometry.centre) is part, part.instance
    assert set(np.unique(change.labels)) == set(range(len(change.parts) + 1))


def test_the_earlier_frame_is_said_in_the_caption(window):
    window.db.set_frame_flags(FrameKey(DESKTOP, STEP, "scan"), missing=True)
    change = window.locator_source.change(DESKTOP, STEP, card(window))
    assert change.scan_step == 11
    assert change.caption == f"主视角 第 11 帧（第 {STEP} 帧扫描仪没有图） · {CHANGE_NAME}"


def test_an_unannotated_scanner_view_shows_the_frame_and_says_why(qapp, tmp_path):
    session, paths = make_scene(tmp_path, drawn=False)
    win = open_window(session, paths)
    try:
        goto(win, 13)
        change = win.locator_source.change(DESKTOP, 13, card(win))
        assert change.parts == () and change.note == NOT_ANNOTATED
        assert change.line == "卡片上 1 个零件，扫描仪上画了 0 个"
    finally:
        close_window(win)


def test_the_scene_is_built_once_per_frame_and_card(window):
    source = window.locator_source
    first = source.change(DESKTOP, 13, [LATCH])
    built = source.stats["changes"]
    assert source.change(DESKTOP, 13, [LATCH]) is first
    assert source.stats["changes"] == built
    assert source.change(DESKTOP, 13, [LATCH, COOLER]) is not first   # another card
    assert source.stats["changes"] == built + 1


# --------------------------------------------------------------------------- #
# the button: three kinds, remembered
# --------------------------------------------------------------------------- #
def test_next_kind_cycles_and_skips_what_cannot_be_shown(qapp):
    assert next_kind(KIND_SCENE, True) == KIND_PAIR
    assert next_kind(KIND_PAIR, True) == KIND_CHANGE
    assert next_kind(KIND_CHANGE, True) == KIND_SCENE
    assert next_kind(KIND_SCENE, False) == KIND_CHANGE          # no part: no pair
    assert next_kind(KIND_PAIR, True, beside=False) == KIND_SCENE   # the scanner view
    assert KIND_TEXT == {KIND_SCENE: "定位", KIND_PAIR: "拆前/拆后", KIND_CHANGE: CHANGE_NAME}


def test_the_button_cycles_the_three_kinds_and_names_the_next(window):
    button = window.locator.pair_button
    select(window, CHASSIS)                      # drawn on oak1: the location
    assert kind(window) == "location" and button.text() == "拆前/拆后"
    assert button.isEnabled()
    click_kind(window)
    assert kind(window) == "pair" and button.text() == CHANGE_NAME
    click_kind(window)
    assert kind(window) == "change" and button.text() == "定位"
    assert not button.isChecked()
    assert window.locator_mode == KIND_CHANGE
    assert window.settings.value(KIND_SETTING) == KIND_CHANGE
    lines = window.locator.caption_text().split("\n")
    assert lines == [f"主视角 第 {STEP} 帧 · {CHANGE_NAME}", "卡片上 5 个零件，扫描仪上画了 0 个"]
    click_kind(window)
    assert kind(window) == "location" and button.text() == "拆前/拆后"
    assert window.settings.value(KIND_SETTING) == KIND_SCENE


def test_without_a_part_the_button_goes_straight_to_the_change(window):
    window.locator_target = lambda: "Remove the cover"     # a card row of a dupli frame
    window.follow_locator(force=True)
    button = window.locator.pair_button
    assert kind(window) == "location" and button.isEnabled()
    assert button.text() == CHANGE_NAME
    click_kind(window)
    assert kind(window) == "change"


def test_the_choice_sticks_across_parts_and_frames(window):
    to_change(window)
    select(window, CHASSIS)
    assert kind(window) == "change"
    select_nothing(window)                       # the fan, met for the first time on oak1
    assert window.locator_first_time()
    assert kind(window) == "change"              # no first-time pair outside 定位
    goto(window, 13)
    assert kind(window) == "change"
    assert window.locator.change().asked_step == 13
    assert [p.instance for p in window.locator.change().parts] == [LATCH]
    window.act_set_view("rs")
    QApplication.processEvents()
    assert kind(window) == "change"


def test_in_pair_mode_every_part_gets_its_pair_and_in_scene_mode_only_a_first_time(window):
    window.set_locator_mode(KIND_PAIR)
    select(window, CHASSIS)
    assert kind(window) == "pair"
    window.set_locator_mode(KIND_SCENE)
    assert kind(window) == "location"
    select_nothing(window)
    assert kind(window) == "pair"                # U28's rule, in 定位


def test_the_scanner_view_never_shows_the_change(window):
    to_change(window)
    window.act_set_view("scan")
    QApplication.processEvents()
    select(window, CHASSIS)
    assert kind(window) == "down"
    select(window, COOLER)                       # never drawn there: no auto pair either
    assert kind(window) == "down"
    window.locator_button.click()                # the top bar asks for this part's pair
    QApplication.processEvents()
    assert kind(window) == "pair" and window.locator.pair().instance == COOLER
    assert window.locator.pair_button.text() == "定位"     # the change is skipped here
    assert window.locator_mode == KIND_CHANGE


def test_the_choice_is_there_in_the_next_session(qapp, tmp_path):
    session, paths = make_scene(tmp_path)
    win = open_window(session, paths)
    goto(win, 13)
    to_change(win)
    win.save_window_state()
    close_window(win)

    session = AnnotationSession(session.db, session.tax, TruthService(session.db, session.tax),
                                paths["cache_dir"], "tester")
    session.sweeper_enabled = False
    session.open(DESKTOP, "oak1")
    again = open_window(session, paths)
    try:
        goto(again, 13)
        assert again.locator_mode == KIND_CHANGE
        assert kind(again) == "change"
        assert [p.instance for p in again.locator.change().parts] == [LATCH]
    finally:
        close_window(again)


def test_a_bad_ini_value_is_the_default(qapp, tmp_path):
    session, paths = make_scene(tmp_path)
    win = open_window(session, paths)
    win.settings.setValue(KIND_SETTING, "nonsense")
    win.save_window_state()
    close_window(win)
    session = AnnotationSession(session.db, session.tax, TruthService(session.db, session.tax),
                                paths["cache_dir"], "tester")
    session.sweeper_enabled = False
    session.open(DESKTOP, "oak1")
    again = open_window(session, paths)
    try:
        assert again.locator_mode == KIND_SCENE
    finally:
        close_window(again)


# --------------------------------------------------------------------------- #
# the picture: mouse-over, click, wheel
# --------------------------------------------------------------------------- #
def test_the_mouse_names_the_part_in_the_captions_second_line(window):
    window.locator_dock.resize(320, 320)
    goto(window, 13)
    to_change(window)
    view = window.locator.change_view
    change = window.locator.change()
    part = change.parts[0]
    move(view, widget_point(view, *part.geometry.centre))
    assert view.hovered() == LATCH
    lines = window.locator.caption_text().split("\n")
    assert lines == [f"主视角 第 13 帧 · {CHANGE_NAME}", part.text]
    move(view, widget_point(view, 1.0, 62.0))    # off every part
    assert view.hovered() is None
    assert window.locator.caption_text().split("\n")[1] == "卡片上 1 个零件，扫描仪上画了 1 个"
    # leaving the picture says the counts again
    move(view, widget_point(view, *part.geometry.centre))
    QApplication.sendEvent(view, QEvent(QEvent.Type.Leave))
    assert window.locator.caption_text().split("\n")[1] == "卡片上 1 个零件，扫描仪上画了 1 个"
    assert view.toolTip() == ""                  # no tooltip on the picture


def test_an_empty_card_shows_the_whole_frame_and_says_so(window):
    to_change(window)
    window.task_card.rows = lambda: [{"instance": "Remove the cover", "kind": "confirm"}]
    window.follow_locator(force=True)
    change = window.locator.change()
    assert change.card == () and change.image is not None
    assert window.locator.caption_text().split("\n")[1] == EMPTY_CARD
    view = window.locator.change_view
    assert view.showing() == WHOLE
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     view.rect().center())
    assert view.showing() == WHOLE               # nothing painted: nothing to crop to


def test_a_click_toggles_whole_and_a_crop_around_the_painted_parts(window):
    window.locator_dock.resize(320, 300)
    goto(window, 1)
    to_change(window)
    view = window.locator.change_view
    change = window.locator.change()
    assert view.showing() == WHOLE and view.source_rect() == (0.0, 0.0, 64.0, 64.0)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     view.rect().center())
    assert view.showing() == CROP
    src = view.source_rect()
    x0, y0, x1, y1 = change.box
    assert src[0] <= x0 and src[1] <= y0 and src[2] >= x1 and src[3] >= y1
    assert max(src[2] - src[0], src[3] - src[1]) >= MIN_CROP_FRAC * 64 - 1e-6
    assert (src[2] - src[0]) < 64 or (src[3] - src[1]) < 64
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     view.rect().center())
    assert view.showing() == WHOLE
    assert window.session.current().step == 1     # nothing else moved


def test_the_wheel_zooms_about_the_mouse_and_another_frame_starts_over(window):
    window.locator_dock.resize(320, 320)
    goto(window, 13)
    to_change(window)
    view = window.locator.change_view
    part = window.locator.change().parts[0]
    cx, cy = part.geometry.centre
    pos = widget_point(view, cx, cy)
    event = QWheelEvent(pos, view.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 240),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(view, event)
    src = view.source_rect()
    assert view.zoom() < 1.0 and (src[2] - src[0]) < 64
    assert src[0] <= cx <= src[2] and src[1] <= cy <= src[3]   # the part stays under it
    assert view.part_at(widget_point(view, cx, cy)) is part
    assert not window.locator.grab().isNull()                  # it paints
    assert view.paints > 0
    select(window, CHASSIS)                      # same frame: where they were looking
    assert view.zoom() < 1.0
    goto(window, 12)                             # another frame: whole again
    assert view.zoom() == 1.0


# --------------------------------------------------------------------------- #
# costs and rules
# --------------------------------------------------------------------------- #
def test_nothing_is_written_and_no_key_is_added(window):
    goto(window, 14)                 # a frame change writes what the session writes
    before = window.db.conn.total_changes
    to_change(window)
    window.follow_locator(force=True)
    window.locator_source.change(DESKTOP, 14, card(window))
    window.locator_source.change(DESKTOP, 13, [LATCH])
    assert window.db.conn.total_changes == before
    assert window.locator.findChildren(QShortcut) == []
    assert window.locator.pair_button.shortcut().isEmpty()


def test_one_refresh_per_settled_change_and_a_selection_builds_nothing(window):
    to_change(window)
    count = window.locator_refreshes
    for _ in range(3):
        window.refresh_guidance()
    QApplication.processEvents()
    assert window.locator_refreshes == count
    built = window.locator_source.stats["changes"]
    select(window, CHASSIS)                      # the stamp moved, the scene did not
    assert window.locator_source.stats["changes"] == built
    goto(window, 13)
    goto(window, STEP)                           # back: from the cache
    assert window.locator_source.stats["changes"] == built + 1


def test_a_forty_part_frame_builds_well_under_100_ms_with_warm_caches(window):
    goto(window, 14)
    source = window.locator_source
    # the start card's 32 parts and 8 more of the machine: 40 rows, all parts
    machine = sorted(k for k in window.db.instances(DESKTOP) if source.is_instance(DESKTOP, k))
    names = list(dict.fromkeys(card(window) + machine))[:40]
    source.change(DESKTOP, 14, names)            # warm: picture, rows, outlines
    times = []
    for _ in range(5):
        source._change_maker = ChangeMaker(source)
        start = time.perf_counter()
        change = source.change(DESKTOP, 14, names)
        times.append((time.perf_counter() - start) * 1000.0)
    assert len(change.card) == 40 and len(change.parts) >= 32
    assert min(times) < 100.0, times
