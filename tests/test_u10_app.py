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
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.gallery import GALLERY_REFUSALS
from tda.ui.panels.taskcard import MENU_ABSENT, MENU_PRESENT
from tda.ui.session import AnnotationSession

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


def test_i_marks_the_cards_part_absent_and_again_takes_it_back(window):
    part = window.task_card.current_instance()
    assert part
    window.handle_key(key_event("I"))
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


@pytest.mark.parametrize("key,name", [("T", "cycle_state"), ("Ctrl+K", "commit_split"),
                                      ("D", "toggle_heat"), ("Shift+C", "cycle_prompt_box"),
                                      ("R", "tool_bench_box"), ("Shift+R", "edit_roi")])
def test_sequence_keys_refuse_with_their_reason(window, key, name):
    reported: list[str] = []
    window.report = reported.append
    window.handle_key(key_event(key))
    assert reported and GALLERY_REFUSALS[name] in reported[-1]


def test_i_outside_the_gallery_says_so(window):
    window.act_set_view("scan")
    QApplication.processEvents()
    assert window.session.view == "scan"
    assert "零件合照" not in window.windowTitle()
    reported: list[str] = []
    window.report = reported.append
    window.handle_key(key_event("I"))
    assert reported and "只用在零件合照里" in reported[-1]
