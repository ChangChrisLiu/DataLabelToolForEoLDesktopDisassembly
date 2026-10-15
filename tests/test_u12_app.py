"""Task U12: a merge in Steps mode reopens the session on the same frame."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import DESKTOP, LAST_STEP, StubSamQueue, close_window, make_paths, make_session
from tda.core.model import InstanceRec
from tda.ui import app_actions as A
from tda.ui.app import MainWindow

COVER = "cover.01"      # the log opens it at 2
TWIN = "cover.09"       # the importer's second name for it, in this test


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
    win.session.goto(LAST_STEP, force=True)
    QApplication.processEvents()
    yield win
    close_window(win)


def test_a_merge_in_steps_mode_reopens_the_session(window):
    window.db.upsert_instance(InstanceRec(TWIN, DESKTOP, "cover", attrs={"of": "other"},
                                          raw_names=["Side Cover"]))
    window.set_mode(A.MODE_STEPS)
    panel = window.steps_panel
    panel.set_desktop(DESKTOP)
    assert TWIN in panel.data.instances
    step = window.session.current().step

    report = panel.merge_instance(TWIN, COVER)
    QApplication.processEvents()

    assert report is not None
    assert TWIN not in window.db.instances(DESKTOP)
    assert TWIN not in panel.data.instances
    assert window.session.current().step == step
    assert TWIN not in {str(r["key"]) for r in window.session.instance_rows()}
    assert COVER in {str(r["key"]) for r in window.session.instance_rows()}
    assert window._steps_dirty is False
    assert panel.status.text().startswith(f"已把 {TWIN} 合并到 {COVER}")
