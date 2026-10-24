"""Task U10: the components gallery (零件合照) in the main window.

A window opened on ``cmp_oak1`` of the D13 gallery scene (:mod:`gallery_scene`):
the title, the card, the status bar and the guide say which shot it is; ``I``
and the card's right-click mark a part 「看不见」; the sequence's keys that have
nothing to act on in a layout photo refuse with their reason.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import StubSamQueue, close_window, make_paths
from test_app_actions import key_event
from tda.core.db import Db
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import InputCache, instances_of, state_of
from tda.core.verb_fix import offers
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.gallery import GALLERY_REFUSALS, NO_VERB
from tda.ui.panels.taskcard import MENU_ABSENT, MENU_PRESENT
from tda.ui.session import AnnotationSession
from tda.ui.session_api import SessionRefusal

import gallery_scene as GS

TITLE = "零件合照 OAK1 第 1/3 张"


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
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


def test_the_window_says_which_shot_it_is(window):
    assert TITLE in window.windowTitle()
    assert window.task_card.header_text().startswith(
        f"{TITLE}：把这张里看得到的零件都画出来；不在这张里的点「看不见」")
    assert "零件合照·OAK1 · 第 1/3 张" in window._frame_text()
    assert window.guide_facts().gallery == TITLE
    assert "零件合照·OAK1" in window.view_buttons["cmp_oak1"].text()
    assert window.view_buttons["cmp_oak1"].isChecked()
    # no chassis ROI question: the whole image is the working area
    assert not window.roi_editing
    assert window.roi() == (0, 0, GS.HW[1], GS.HW[0])


def test_the_grave_key_marks_the_cards_part_absent_and_again_takes_it_back(window):
    part = window.task_card.current_instance()
    assert part
    window.handle_key(key_event("`"))            # I until task U22
    QApplication.processEvents()
    row = next(r for r in window.task_card.rows() if r["instance"] == part)
    assert row["done"] and row["absent"]
    assert window.task_card.absent_menu_text(part) == MENU_PRESENT
    assert window.task_card.current_instance() != part     # the next open row
    window.act_mark_absent(part)
    row = next(r for r in window.task_card.rows() if r["instance"] == part)
    assert not row["done"] and not row.get("absent")
    assert window.task_card.absent_menu_text(part) == MENU_ABSENT


def test_the_cards_right_click_reaches_the_same_slot(window):
    part = window.task_card.current_instance()
    window.task_card.sigMarkAbsent.emit(part)
    QApplication.processEvents()
    assert next(r for r in window.task_card.rows() if r["instance"] == part)["absent"]


# U16: Ctrl+K is the commit now -- a plain commit on a shot -- and 改全部 is
# what a shot refuses in its place (tests/test_u16_enter_default.py).  U22:
# the left-hand keys (Ctrl+Q, G, V), with the Enter alias still refused alike.
@pytest.mark.parametrize("key,name", [("T", "cycle_state"), ("Ctrl+Q", "commit_all"),
                                      ("Ctrl+Return", "commit_all"),
                                      ("G", "toggle_heat"), ("Shift+C", "cycle_prompt_box"),
                                      ("V", "tool_bench_box"), ("Shift+R", "edit_roi")])
def test_sequence_keys_refuse_with_their_reason(window, key, name):
    reported: list[str] = []
    window.report = reported.append
    window.handle_key(key_event(key))
    assert reported and GALLERY_REFUSALS[name] in reported[-1]


def test_the_grave_key_outside_the_gallery_says_so(window):
    window.act_set_view("scan")
    QApplication.processEvents()
    assert window.session.view == "scan"
    assert "零件合照" not in window.windowTitle()
    reported: list[str] = []
    window.report = reported.append
    window.handle_key(key_event("`"))
    assert reported and "只用在零件合照里" in reported[-1]


# --------------------------------------------------------------------------- #
# review round
# --------------------------------------------------------------------------- #
def test_switching_family_lands_where_that_family_was_left(window):
    """M5: a step number means nothing in the gallery and vice versa."""
    window.session.goto(2)                       # shot 2 of cmp_oak1
    window.act_set_view("scan")
    QApplication.processEvents()
    assert window.session.view == "scan"
    assert window.session.current().step == max(window.session.available_steps())
    window.session.goto(5)
    window.act_set_view("cmp_oak2")              # back in the gallery: the shot left
    QApplication.processEvents()
    assert window.session.view == "cmp_oak2" and window.session.current().step == 2
    window.act_set_view("scan")
    QApplication.processEvents()
    assert window.session.current().step == 5


def test_the_timeline_lists_shots_forward_by_position(window):
    """M7: ascending (Space walks forward), labelled 第 k 张."""
    timeline = window.timeline.list_widget() if hasattr(window.timeline, "list_widget") \
        else window.timeline._list
    texts = [timeline.item(i).text() for i in range(timeline.count())]
    assert [t.split(" · ")[0] for t in texts] == ["第 1 张", "第 2 张", "第 3 张"]


def test_status_words_count_shots(window):
    """M4: 「第 k 张」, not "step k"."""
    assert window.frame_word(2) == "第 2 张"
    window.act_set_view("scan")
    QApplication.processEvents()
    assert window.frame_word(7) == "step 7"


# --------------------------------------------------------------------------- #
# review round 3
# --------------------------------------------------------------------------- #
def actions_of(win: MainWindow) -> list[tuple]:
    return [(a.step, a.idx, a.verb, a.target) for a in win.db.actions(GS.DESKTOP)]


def test_shift_t_and_the_verb_button_never_touch_the_step_table_from_a_shot(window):
    """Blocker: shot 2 is not step 2 -- yet step 2 of D13 is where cover.01 opens.

    Read as a step, shot 2 offered 「其实拆下来了」 for cover.01 and the button's
    signal flipped step 2's ``open`` to ``remove``. On a shot nothing is
    offered, no row lights the button, and the key, the slot, the signal and
    the session all refuse: the step table and the op log stay as they were.
    """
    window.session.goto(2)
    QApplication.processEvents()
    assert window.in_gallery() and window.session.current().step == 2
    tax = window.session.tax
    cache = InputCache()
    as_a_step = offers(tax, instances_of(window.db, GS.DESKTOP, cache),
                       state_of(window.db, tax, GS.DESKTOP, 2, cache),
                       window.db.actions(GS.DESKTOP), 2)
    assert as_a_step == {"cover.01": "remove"}         # what the shot used to offer
    before = actions_of(window)
    assert window.session.verb_offers() == {}          # nothing to offer on a shot
    window.instances.show_removed.setChecked(True)
    QApplication.processEvents()
    table = window.instances.table()
    button = window.instances.verb_button
    assert table.rowCount()
    for row in range(table.rowCount()):                # kept rows and removed rows
        table.setCurrentCell(row, 1)
        QApplication.processEvents()
        assert window.instances.verb_offer() == (None, None) and not button.isEnabled()
    table.setCurrentCell(0, 1)
    reported: list[str] = []
    window.report = reported.append
    window.handle_key(key_event("Shift+T"))             # the key: refused by dispatch
    assert reported and GALLERY_REFUSALS["correct_verb"] in reported[-1]
    window.act_correct_verb()                           # the slot itself
    assert NO_VERB in reported[-1]
    window.instances.sigCorrectVerb.emit("cover.01", "remove")   # the button's signal
    assert NO_VERB in reported[-1]
    with pytest.raises(SessionRefusal, match="零件合照不改步骤表"):
        window.session.correct_verb("cover.01", "remove")
    assert actions_of(window) == before
    assert not window.db.ops_of_kind(GS.DESKTOP, "correct_verb")
    # the card asks nothing about verbs either
    assert all(not r.get("verb_fix") for r in window.task_card.rows())
