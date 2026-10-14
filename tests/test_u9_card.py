"""Task U9 (card): the row of an open cover / latch says which key takes it out.

On D2/48 the card asked for six parts that were on the components pile; the
annotator had no way to know the card could be told so. Each row of a part
that :meth:`~tda.ui.session_verb.VerbMixin.verb_offers` marks carries one
short line naming the key.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import LAST_STEP, StubSamQueue, close_window, make_paths, make_session, \
    seed_shapes
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.panels.taskcard import row_view, taken_out_hint

COVER = "cover.01"         # the log opens it at 2
LATCH = "drive_latch.01"   # ... and this one at 7
HINT = "画面里没有它了？Shift+T：其实拆下来了"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()
    seed_shapes(win.session, LAST_STEP)
    win.session.goto(LAST_STEP, force=True)
    QApplication.processEvents()
    yield win
    close_window(win)


def test_the_hint_names_the_bound_key():
    assert taken_out_hint() == HINT


def test_the_session_marks_the_rows_of_the_open_covers_and_latches(window):
    marked = {r["instance"] for r in window.session.task_card() if r.get("verb_fix")}
    assert marked == {COVER, LATCH}
    window.session.goto(1, force=True)
    assert not [r for r in window.session.task_card() if r.get("verb_fix")]


def test_the_card_row_of_an_open_cover_says_which_key(window):
    texts = {r["instance"]: t for r, t in zip(window.task_card.rows(),
                                              window.task_card.row_texts())}
    assert HINT in texts[COVER] and HINT in texts[LATCH]
    assert HINT not in texts["ram_latch.01"] and HINT not in texts["cover.02"]
    window.instances.select_instance(COVER)
    window.act_correct_verb()
    assert COVER not in {r["instance"] for r in window.task_card.rows()}


def test_the_pure_view_only_says_it_on_a_marked_row():
    row = {"instance": COVER, "kind": api.KIND_ADD_SHAPE, "cls": "cover"}
    assert HINT not in row_view(row)["sentence"]
    assert row_view(row | {"verb_fix": "remove"})["sentence"].endswith(f"（{HINT}）")
    assert HINT not in row_view({"instance": "", "kind": api.KIND_CONFIRM,
                                 "verb_fix": "remove"})["sentence"]
