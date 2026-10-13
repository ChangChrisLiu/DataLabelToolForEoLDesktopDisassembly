"""Task U7, D: 添加零件 says how many confirmed frames it will demote.

On D13/scan the annotator added the four clips of the empty RAM slots after
confirming frames 42 -> 17.  Each frame gained four instances and went back to
需复核 -- by design -- and nothing said so beforehand.  The dialog now counts
the confirmed frames of the open view the new parts will appear in: present
from step 0 until a state event takes them out (their host leaving).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox

import session_scene
from session_scene import BOARD, COOLER, DESKTOP, VIEW, seed_shapes
from tda.core import extra as X
from tda.core.model import FrameKey
from tda.ui.panels.add_parts import demotion_text
from tda.ui.steps_model import StepTableData

#: The cooler comes out at step 13 of the D13 sheet.
COOLER_OUT = 13


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(qapp, tmp_path: Path):
    made = session_scene.make_session(tmp_path)
    made.sweeper_enabled = False
    # frames 13 and 12 confirmed, 11 left as it is
    made.goto(12)
    seed_shapes(made, 12)
    made.goto(13)
    assert made.confirm_frame() is True and made.confirm_frame() is True
    yield made
    made.close(force=True)


def _status(session, step: int) -> str:
    return session.db.get_frame(FrameKey(DESKTOP, step, VIEW))["review_status"]


def test_the_count_is_the_confirmed_frames_the_part_is_present_in(session):
    data = StepTableData.load(session.db, DESKTOP, session.tax)

    assert data.frames_demoted_by_extras(session.db, VIEW, "ram_latch", 4) == [12, 13]
    # on a part that leaves with the cooler at 13 it is present on 12 only
    assert data.frames_demoted_by_extras(session.db, VIEW, "ram_latch", 1,
                                         parent=COOLER) == [12]
    # a frame that is not confirmed is not counted
    session.truth.demote_frame(FrameKey(DESKTOP, 13, VIEW), "test")
    assert data.frames_demoted_by_extras(session.db, VIEW, "ram_latch", 1) == [12]
    # nothing for what add_extras would refuse
    assert data.frames_demoted_by_extras(session.db, VIEW, "chassis", 1) == []
    assert data.frames_demoted_by_extras(session.db, VIEW, "no_such_class", 1) == []


def test_the_count_is_what_actually_happens(session):
    """Predicted, then written and refreshed: the same frames go back to 需复核."""
    data = StepTableData.load(session.db, DESKTOP, session.tax)
    predicted = data.frames_demoted_by_extras(session.db, VIEW, "ram_latch", 2,
                                              parent=COOLER)
    assert predicted == [12]

    data.add_extras("ram_latch", 2, parent=COOLER)
    data.save(session.db)
    session.truth.refresh_range(DESKTOP, VIEW, session.available_steps())

    demoted = [s for s in (12, 13) if _status(session, s) == "needs_review"]
    assert demoted == predicted
    assert _status(session, 13) == "verified"


def test_the_sentence():
    assert demotion_text([], 4) == ""
    assert demotion_text(list(range(17, 42)), 4) == (
        "这 4 个零件出现在 25 个已确认帧（17–41）里；添加后这些帧会回到“需复核”，"
        "补画后逐帧按 Space 即可。")
    assert demotion_text([40, 41], 1).startswith("这个零件出现在 2 个已确认帧（40、41）里")


def test_the_dialog_says_it_for_the_open_view(session, tmp_path):
    from tda.ui.panels.steptable import StepTablePanel

    panel = StepTablePanel(session.db, DESKTOP, taxonomy=session.tax,
                           cache_dir=tmp_path / "cache", view_of=lambda: VIEW)
    dialog = panel.open_add_parts()
    try:
        index = dialog.class_combo.findData("ram_latch")
        dialog.class_combo.setCurrentIndex(index)
        dialog.count_spin.setValue(4)
        assert dialog.parent_combo.currentData() == BOARD
        assert not dialog.warning.isHidden()
        assert dialog.warning.text().startswith("这 4 个零件出现在 2 个已确认帧（12、13）里")
        assert dialog.demoted_frames() == [12, 13]
        # not a confirmation: the button stays live
        assert dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
        dialog.parent_combo.setCurrentIndex(dialog.parent_combo.findData(COOLER))
        assert dialog.demoted_frames() == [12]
    finally:
        dialog.reject()
        panel.deleteLater()

    blind = StepTablePanel(session.db, DESKTOP, taxonomy=session.tax,
                           cache_dir=tmp_path / "cache")
    quiet = blind.open_add_parts()
    try:
        quiet.class_combo.setCurrentIndex(quiet.class_combo.findData("ram_latch"))
        assert quiet.warning.isHidden() and quiet.demoted_frames() == []
    finally:
        quiet.reject()
        blind.deleteLater()


def test_the_core_function_reads_the_database_when_no_table_is_given(session):
    batch = X.plan_extras(session.db.instances(DESKTOP), session.tax, DESKTOP,
                          "ram_latch", 1, parent=BOARD)
    assert X.verified_frames_gaining(session.db, session.tax, DESKTOP, VIEW,
                                     batch.records) == [12, 13]
    assert X.verified_frames_gaining(session.db, session.tax, DESKTOP, VIEW, []) == []
