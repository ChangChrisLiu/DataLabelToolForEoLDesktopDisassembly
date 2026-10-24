"""Task U28 round 2: the guessed window says it is one, its candidates, the top button.

* B2: a window guessed from the difference is drawn grey (no part colour, no
  outline) and the note says it is a guess; up to three candidates, the ones
  not on screen as chips under the pair, a click swaps one in 「候选 k/n」;
* N2: on the scanner view, with the dock down because the part is drawn
  there, the top bar's 「主视角」 asks for that part's pair and leaves
  ``locator/open`` alone;
* N3: the toggle's choice follows the part to the other views.

The window scene is U23's (D13, 14 steps, the scanner view drawn, a picture per
step on ``oak1`` / ``oak2`` / ``rs``) with the CPU fan undrawn on the other three
cameras; the panel tests draw a synthetic scene.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_scene import COOLER, close_window
from test_u23_locator import CHASSIS, STEP, goto, make_scene, open_window, select
from tda.ui.app_locator import SETTING, TOGGLE_TIP_DOWN
from tda.ui.locator_pair import CANDIDATE, DIFF_GUESS, PairScene
from tda.ui.panels.locator import CROP, LocatorPanel
from tda.ui.panels.locator_pair import GUESS_COLOUR

#: In the machine on step 12, drawn on every view (the scanner's too), opened at 14.
LATCH = "ram_latch.01"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    session, paths = make_scene(tmp_path, own_skip=(COOLER,))
    win = open_window(session, paths)
    goto(win, STEP)
    yield win
    close_window(win)


def kind(win) -> str:
    if not win.locator_shown():
        return "down"
    return "pair" if win.locator.showing_pair() else "location"


# --------------------------------------------------------------------------- #
# the panel: a guess, and its candidates
# --------------------------------------------------------------------------- #
def _picture(value: int) -> QImage:
    image = QImage(400, 300, QImage.Format.Format_RGB888)
    image.fill(QColor(value, value, value))
    return image


def _pair(candidates, source="diff") -> PairScene:
    return PairScene(
        desktop=13, instance="screw.cpu_cooler.03", label="散热器螺丝 screw.cpu_cooler.03",
        colour=(255, 40, 40), steps=(7, 8), images=(_picture(90), _picture(120)),
        frame_hw=((300, 400), (300, 400)), captions=("拆前 第 7 帧", "拆后 第 8 帧"),
        window=candidates[0] if candidates else None, window_source=source,
        candidates=tuple(candidates) if source == "diff" else (),
        label_line="散热器螺丝 screw.cpu_cooler.03（拧松 → 拆下）",
        log_line="第 8 帧：Heatsink · remove", note=DIFF_GUESS if source == "diff" else "")


@pytest.fixture
def panel(qapp):
    found = LocatorPanel()
    found.resize(320, 300)
    found.show()
    QApplication.processEvents()
    yield found
    found.close()
    found.deleteLater()


def test_a_guessed_window_is_grey_and_the_note_says_it_is_a_guess(panel):
    panel.show_pair(_pair([(50.0, 60.0, 90.0, 100.0)]))
    view = panel.pair_view
    assert view.is_guess()
    assert view.window_colour() == GUESS_COLOUR
    assert panel.note_text() == DIFF_GUESS
    assert not panel.candidates.isVisible()        # one candidate: nothing to choose
    assert panel.candidate_text() == ""
    # the part's own outline keeps the part's colour
    panel.show_pair(_pair([(50.0, 60.0, 90.0, 100.0)], source="part"))
    assert not view.is_guess() and view.window_colour() == QColor(255, 40, 40)
    assert panel.note_text() == ""


def test_the_other_candidates_are_chips_and_a_click_swaps_one_in(panel):
    candidates = [(300.0, 40.0, 360.0, 90.0), (40.0, 200.0, 80.0, 240.0),
                  (180.0, 120.0, 200.0, 140.0)]
    panel.show_pair(_pair(candidates))
    QApplication.processEvents()
    view, bar = panel.pair_view, panel.candidates
    assert bar.isVisible()
    assert panel.candidate_text() == CANDIDATE.format(k=1, n=3) == "候选 1/3"
    assert [index for index, _rect in bar.chips()] == [1, 2]
    assert view.window() == candidates[0]
    first = view.source_rect(0)
    assert view.source_rect(1) == first             # both cells: one crop
    # the log's words go under the row of chips (the room they need)
    assert panel._row.indexOf(panel.info) < 0
    index, rect = bar.chips()[1]
    QTest.mouseClick(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     rect.center().toPoint())
    assert view.candidate_index() == index == 2
    assert view.window() == candidates[2]
    assert panel.candidate_text() == "候选 3/3"
    assert [i for i, _r in bar.chips()] == [0, 1]
    crop = view.source_rect(0)
    assert crop != first and view.source_rect(1) == crop and view.showing() == CROP
    x0, y0, x1, y1 = candidates[2]
    assert crop[0] <= x0 and crop[1] <= y0 and crop[2] >= x1 and crop[3] >= y1
    assert not bar.grab().isNull() and not panel.inset.grab().isNull()
    # the same pair again (a refresh) keeps the choice; another part starts over
    panel.show_pair(_pair(candidates))
    assert view.candidate_index() == 2
    other = _pair(candidates)
    other.instance = "screw.cpu_cooler.04"
    panel.show_pair(other)
    assert view.candidate_index() == 0
    # back to a part's outline: no chips, the words beside the small frame
    panel.show_pair(_pair([(50.0, 60.0, 90.0, 100.0)], source="part"))
    assert not bar.isVisible() and panel._row.indexOf(panel.info) >= 0


# --------------------------------------------------------------------------- #
# in the window
# --------------------------------------------------------------------------- #
def test_on_the_scanner_view_a_guess_says_so(window):
    window.act_set_view("scan")
    QApplication.processEvents()
    select(window, COOLER)                       # never drawn in the scanner view
    pair = window.locator.pair()
    assert pair.window_source == "diff" and window.locator.pair_view.is_guess()
    assert DIFF_GUESS in window.locator.note_text()


def test_the_top_button_asks_for_the_pair_of_a_part_drawn_on_the_scanner_view(window):
    window.act_set_view("scan")
    QApplication.processEvents()
    select(window, LATCH)                        # drawn there: the dock is down
    assert kind(window) == "down" and window.locator_down_for_part()
    button = window.locator_button
    assert button.isEnabled() and button.isChecked()
    assert button.toolTip() == TOGGLE_TIP_DOWN
    before = window.settings.value(SETTING)
    button.click()
    QApplication.processEvents()
    assert kind(window) == "pair" and window.locator.pair().instance == LATCH
    assert window.locator_wanted is True and button.isChecked()
    assert window.settings.value(SETTING) == before       # the wish is untouched
    select(window, CHASSIS)                      # another part: the choice is gone
    assert kind(window) == "down"
    selected = window.locator_target
    window.locator_target = lambda: None         # nothing to show it for
    window.follow_locator(force=True)
    button.click()
    QApplication.processEvents()
    assert kind(window) == "down" and window.locator_wanted is True
    assert button.isChecked()
    # with the dock up, the button closes it, remembered, as it always did
    window.locator_target = selected
    select(window, COOLER)
    assert kind(window) == "pair"
    button.click()
    QApplication.processEvents()
    assert kind(window) == "down" and window.locator_wanted is False
    assert str(window.settings.value(SETTING)).lower() == "false"


def test_the_toggles_choice_follows_the_part_to_the_other_views(window):
    select(window, LATCH)                        # drawn on oak1: the location
    window.locator.pair_button.click()
    QApplication.processEvents()
    assert kind(window) == "pair"
    window.act_set_view("oak2")
    QApplication.processEvents()
    select(window, LATCH)
    assert kind(window) == "pair"                # drawn there too: still asked for
    window.act_set_view("scan")
    QApplication.processEvents()
    select(window, LATCH)
    assert kind(window) == "pair"                # and on the scanner view
