"""Task U28: 「拆前 / 拆后」 in the window -- when the dock shows the pair, and for what.

U23's scene (D13, 14 steps, the scanner view drawn, a picture per step on
``oak1`` / ``oak2`` / ``rs``) with every part drawn on those three views but
the CPU fan, which the card asks for on step 12: the fan is met there for the
first time.

* a part with no shape of its own in the view on screen gets the pair, on any
  view of the teardown, the scanner's own included; one already drawn there
  gets the location (OAK / RealSense) or no dock (scanner view);
* the first commit in a view flips it, and a cut copy counts as a shape;
* the toggle next to the caption asks for the pair or takes it away, for the
  part on screen only;
* nothing is read while the dock is down, a settled frame change is one
  refresh, a repeated one reads nothing new.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_scene import COOLER, DESKTOP, LAST_STEP, close_window
from test_u23_locator import (
    CHASSIS,
    OTHER_HW,
    STEP,
    goto,
    make_scene,
    open_window,
    select,
    select_nothing,
)
from tda.core import masks
from tda.core.model import ShapeKeyframe, ShapePart
from tda.ui.locator_pair import NO_CHANGE, PAIR_HEAD, PAIR_HEAD_FIRST, MIN_SIDE_FRAC
from tda.ui.panels.locator import CROP, WHOLE

#: In the machine on step 12, drawn on every view, opened at 14 (its pair: 13 / 14).
LATCH = "ram_latch.01"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    """On ``oak1``, step 12; the CPU fan has no shape on ``oak1``/``oak2``/``rs``."""
    session, paths = make_scene(tmp_path, own_skip=(COOLER,))
    win = open_window(session, paths)
    goto(win, STEP)
    yield win
    close_window(win)


def pair(win):
    return win.locator.pair()


def kind(win) -> str:
    if not win.locator_shown():
        return "down"
    return "pair" if win.locator.showing_pair() else "location"


def oak1_shape(win, instance: str, source: str = "manual") -> None:
    mask = np.zeros(OTHER_HW, dtype=bool)
    mask[30:40, 30:40] = True
    win.db.add_keyframe(ShapeKeyframe(
        id=None, instance=instance, desktop=DESKTOP, view="oak1", pose_segment=1,
        anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(mask))], source=source))


# --------------------------------------------------------------------------- #
# which scene
# --------------------------------------------------------------------------- #
def test_a_part_met_for_the_first_time_on_oak1_gets_its_pair(window):
    select_nothing(window)
    assert window.locator_target() == COOLER and window.locator_first_time()
    assert kind(window) == "pair"
    found = pair(window)
    assert found.instance == COOLER and found.steps == (12, 13)
    assert window.locator.caption_text() == PAIR_HEAD_FIRST
    assert found.captions == ("拆前 第 12 帧", "拆后 第 13 帧")
    assert window.locator.info_text() == ("CPU 风扇 cpu_cooler.fan.01（装着 → 拆下）\n"
                                          "第 13 帧：CPU fan · remove")
    button = window.locator.pair_button
    assert button.isChecked() and button.isEnabled()


def test_a_part_already_drawn_on_oak1_gets_the_location(window):
    select(window, CHASSIS)
    assert not window.locator_first_time()
    assert kind(window) == "location"
    assert window.locator.scene().instance == CHASSIS
    assert window.locator.caption_text().startswith(f"主视角 第 {STEP} 帧 · 机箱 chassis")
    assert not window.locator.pair_button.isChecked()
    assert window.locator.pair_button.isEnabled()


@pytest.mark.parametrize("view", ["oak2", "rs"])
def test_the_other_cameras_follow_the_same_rule(window, view):
    window.act_set_view(view)
    QApplication.processEvents()
    select(window, COOLER)
    assert kind(window) == "pair" and pair(window).instance == COOLER
    select(window, CHASSIS)
    assert kind(window) == "location"


def test_on_the_scanner_view_the_dock_is_up_only_for_a_pair(window):
    window.act_set_view("scan")
    QApplication.processEvents()
    select(window, CHASSIS)                     # drawn in the scanner view
    assert kind(window) == "down"
    assert window.locator_button.isEnabled()    # the top bar's toggle is live here
    select(window, COOLER)                      # never drawn there (out at 13)
    assert kind(window) == "pair"
    assert pair(window).instance == COOLER and pair(window).steps == (12, 13)
    assert pair(window).window_source == "diff"
    # taking the pair away leaves the scanner view nothing to show
    window.locator.pair_button.click()
    QApplication.processEvents()
    assert kind(window) == "down"
    select(window, CHASSIS)
    select(window, COOLER)                      # another part, then back: asked afresh
    assert kind(window) == "pair"


def test_a_part_that_never_changes_says_so(window):
    window.act_set_view("scan")
    QApplication.processEvents()
    window.db.conn.execute("DELETE FROM shape_keyframe WHERE desktop=? AND view='scan' "
                           "AND instance=?", (DESKTOP, CHASSIS))
    window.db.conn.commit()
    select(window, CHASSIS)
    window.follow_locator(force=True)
    assert kind(window) == "pair"
    assert window.locator.note_text() == NO_CHANGE
    assert pair(window).images == (None, None)


def test_the_first_commit_in_the_view_flips_it(window):
    select_nothing(window)
    assert kind(window) == "pair"
    window.on_request_edit(COOLER)
    QApplication.processEvents()
    assert kind(window) == "pair"                # drawing it: still the first time
    mask = np.zeros(OTHER_HW, dtype=bool)
    mask[30:40, 30:44] = True
    window.set_editing_mask(mask, undoable=True)
    window.act_commit()
    for _ in range(2):
        if not (window.warn_bar.isVisible() or window.scope_bar.isVisible()):
            break
        window.act_commit()
    QApplication.processEvents()
    assert window.session.editing_instance is None
    assert window.db.keyframes(DESKTOP, "oak1", COOLER)
    select(window, COOLER)
    assert not window.locator_first_time()
    assert kind(window) == "location"
    # ... in that view only: on oak2 the fan is still met for the first time
    window.act_set_view("oak2")
    QApplication.processEvents()
    select(window, COOLER)
    assert kind(window) == "pair"


def test_a_cut_copy_counts_as_a_shape(window):
    select_nothing(window)
    assert kind(window) == "pair"
    oak1_shape(window, COOLER, source="cut")
    window.follow_locator(force=True)
    assert kind(window) == "location"


def test_the_toggle_asks_for_the_pair_and_takes_it_back(window):
    select(window, LATCH)                        # drawn on oak1: the location
    assert kind(window) == "location"
    window.locator.pair_button.click()
    QApplication.processEvents()
    assert kind(window) == "pair"
    assert pair(window).instance == LATCH and pair(window).steps == (13, 14)
    assert pair(window).log_line == "第 14 帧：RAM clip 1 · open"
    assert window.locator.caption_text() == PAIR_HEAD     # asked for, not a first time
    assert window.locator.pair_button.isChecked()
    window.locator.pair_button.click()
    QApplication.processEvents()
    assert kind(window) == "location"
    # the choice is for that part only: the fan is still its first time
    window.locator.pair_button.click()
    QApplication.processEvents()
    assert kind(window) == "pair"
    select(window, CHASSIS)
    assert kind(window) == "location"
    select(window, LATCH)
    assert kind(window) == "location"            # forgotten once the part changed
    # and the first-time pair can be put away too
    select_nothing(window)
    assert kind(window) == "pair"
    window.locator.pair_button.click()
    QApplication.processEvents()
    assert kind(window) == "location" and window.locator.scene().instance == COOLER


def test_nothing_to_look_for_has_no_pair(window):
    window.act_set_view("scan")
    QApplication.processEvents()
    window.locator_target = lambda: "Remove the cover"    # a card row of a dupli frame
    window.follow_locator(force=True)
    assert kind(window) == "down"
    assert not window.locator.pair_button.isEnabled()


# --------------------------------------------------------------------------- #
# the picture
# --------------------------------------------------------------------------- #
def test_both_crops_are_the_same_window_a_click_shows_both_whole(window):
    window.locator_dock.resize(320, 300)
    select(window, LATCH)
    window.locator.pair_button.click()          # the latch is drawn on the scanner view
    QApplication.processEvents()
    view = window.locator.pair_view
    found = pair(window)
    assert found.window_source == "part" and view.showing() == CROP
    before, after = view.source_rect(0), view.source_rect(1)
    assert before == after
    x0, y0, x1, y1 = found.window
    assert before[0] <= x0 and before[1] <= y0 and before[2] >= x1 and before[3] >= y1
    assert max(before[2] - before[0], before[3] - before[1]) >= 64 * MIN_SIDE_FRAC - 1e-6
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     view.rect().center())
    assert view.showing() == WHOLE
    assert view.source_rect(0) == view.source_rect(1) == (0.0, 0.0, 64.0, 64.0)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     view.rect().center())
    assert view.showing() == CROP


def test_the_wheel_zooms_both_crops_and_the_inset_follows(window):
    select(window, LATCH)
    window.locator.pair_button.click()
    QApplication.processEvents()
    view = window.locator.pair_view
    wide = view.source_rect(0)
    pos = QPointF(view.rect().center())
    event = QWheelEvent(pos, view.mapToGlobal(pos), QPoint(0, 0), QPoint(0, -240),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(view, event)
    out = view.source_rect(0)
    assert (out[2] - out[0]) >= (wide[2] - wide[0]) and view.zoom() > 1.0
    assert view.source_rect(1) == out
    painted = window.locator.inset.paints
    window.locator.inset.grab()
    assert window.locator.inset.paints > painted
    assert not window.locator.grab().isNull()     # it paints, both cells and the inset


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
def test_nothing_is_read_while_it_is_closed(window):
    window.locator_dock.close()
    QApplication.processEvents()
    assert not window.locator_shown()
    count, checks = window.locator_refreshes, window.locator_checks
    stats = dict(window.locator_source.stats)
    goto(window, STEP - 1)
    select_nothing(window)
    window.act_set_view("scan")
    QApplication.processEvents()
    goto(window, STEP - 2)
    assert not window.locator_shown()
    assert window.locator_refreshes == count and window.locator_checks == checks
    assert window.locator_source.stats == stats


def test_one_refresh_per_settled_change_and_nothing_new_for_the_same_pair(window):
    select_nothing(window)
    assert kind(window) == "pair"
    count = window.locator_refreshes
    stats = dict(window.locator_source.stats)
    for _ in range(3):
        window.refresh_guidance()
    QApplication.processEvents()
    assert window.locator_refreshes == count
    window.follow_locator(force=True)           # a refresh of the same pair
    assert window.locator_refreshes == count + 1
    assert window.locator_source.stats == stats  # from the caches: nothing read
    # the fan stays selected on 11: one refresh, the same pair, no reads
    select(window, COOLER)
    count = window.locator_refreshes
    goto(window, STEP - 1)
    assert window.locator_target() == COOLER
    assert window.locator_refreshes == count + 1
    assert window.locator.pair().instance == COOLER
    assert window.locator_source.stats == stats
