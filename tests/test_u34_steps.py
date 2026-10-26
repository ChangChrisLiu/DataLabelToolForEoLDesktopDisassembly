"""Task U34 (Steps mode): 「这台没有这个零件…」 / 「其实有…」 on an Instances row.

Deleting D34's ``optical_drive.01`` in Steps is refused -- the sheet's step 37
targets it, and that action is the record of what the operator tried. The row's
right-click now says the machine does not have it: written through (the panel
reloads, the window reopens its session), the row reads 「一开始就不在」, and
the same entry on it is 「其实有…」, which takes it back.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from absent_scene import DRIVE, add_drive
from session_scene import DESKTOP, seed_db
from tda.core import absent as AB
from tda.core.db import Db
from tda.core.taxonomy import load_taxonomy
from tda.ui import steps_absent as SA
from tda.ui.panels.steptable import StepTablePanel
from tda.ui.panels.steptable_models import INSTANCE_COLUMNS


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qapp, tmp_path: Path):
    tax = load_taxonomy()
    db = Db(str(tmp_path / "steps.sqlite"))
    seed_db(db, tax, tmp_path / "cache", last_step=14)
    add_drive(db, tax, DESKTOP)
    widget = StepTablePanel(db, DESKTOP, taxonomy=tax, cache_dir=tmp_path / "cache")
    widget.asked = []

    def ask(key, lines):
        widget.asked.append((key, list(lines)))
        return getattr(widget, "answer", True)

    widget.ask_absent = ask
    widget.saved = []
    widget.sigSaved.connect(widget.saved.append)
    yield widget
    widget.deleteLater()
    db.close()


def _entries(panel, key) -> dict:
    menu = panel.instances_menu(panel.instances_model.keys.index(key))
    return {a.text(): a for a in menu.actions()}


def _cls_cell(panel, key, role=Qt.DisplayRole):
    model = panel.instances_model
    column = [c.field for c in INSTANCE_COLUMNS].index("cls")
    return model.data(model.index(model.keys.index(key), column), role)


def test_delete_is_refused_and_the_new_entry_is_right_below_it(panel):
    entries = list(_entries(panel, DRIVE))
    assert entries[:2] == [f"Delete {DRIVE}", SA.MENU_ABSENT]
    _entries(panel, DRIVE)[f"Delete {DRIVE}"].trigger()
    assert panel.status.text().startswith("Rejected:") and DRIVE in panel.db.instances(DESKTOP)
    assert SA.MENU_ABSENT not in _entries(panel, "chassis")


def test_the_entry_writes_through_and_the_row_says_so(panel):
    assert _cls_cell(panel, DRIVE) == "optical_drive"
    _entries(panel, DRIVE)[SA.MENU_ABSENT].trigger()
    [(key, lines)] = panel.asked
    assert key == DRIVE and lines[0] == f"{DRIVE}：这台机器根本没有它？"
    assert "螺丝、接头不跟着变" in "".join(lines)
    assert AB.is_absent(panel.db, DESKTOP, DRIVE)
    assert panel.saved == [DESKTOP] and not panel.has_unapplied_edits()
    assert panel.status.text().startswith(f"{DRIVE}：一开始就不在（这台没有它）")
    assert _cls_cell(panel, DRIVE) == f"optical_drive（{AB.SHORT}）"
    assert "其实有" in _cls_cell(panel, DRIVE, Qt.ToolTipRole)
    # the sheet's failed remove is still there: the step table did not move
    assert [a.verb for a in panel.db.actions(DESKTOP) if a.target == DRIVE] == [
        "remove", "displace"]
    # no state warning about it in the open questions
    assert not [m for m in panel.data.messages if DRIVE in m]


def test_its_way_back_is_the_same_entry(panel):
    _entries(panel, DRIVE)[SA.MENU_ABSENT].trigger()
    entries = _entries(panel, DRIVE)
    assert SA.MENU_PRESENT in entries and SA.MENU_ABSENT not in entries
    entries[SA.MENU_PRESENT].trigger()
    assert panel.asked[-1][1][0] == f"{DRIVE}：其实有这个零件？"
    assert not AB.is_absent(panel.db, DESKTOP, DRIVE)
    assert _cls_cell(panel, DRIVE) == "optical_drive"
    assert panel.status.text().startswith(f"{DRIVE}：其实有")
    assert panel.saved == [DESKTOP, DESKTOP]


def test_no_writes_nothing(panel):
    panel.answer = False
    _entries(panel, DRIVE)[SA.MENU_ABSENT].trigger()
    assert not AB.is_absent(panel.db, DESKTOP, DRIVE) and panel.saved == []
    assert "没有改" in panel.status.text()


def test_it_is_refused_with_edits_staged(panel):
    panel.data.apply_edit(2, "notes", "a note")
    _entries(panel, DRIVE)[SA.MENU_ABSENT].trigger()
    assert not panel.asked and not AB.is_absent(panel.db, DESKTOP, DRIVE)
    assert panel.status.text().startswith("Rejected: 先 Apply 或 Revert")


def test_a_drawn_part_is_refused_in_words(panel):
    import numpy as np

    from tda.core import masks
    from tda.core.model import ShapeKeyframe, ShapePart

    mask = np.zeros((64, 64), dtype=bool)
    mask[1:4, 1:4] = True
    panel.db.add_keyframe(ShapeKeyframe(
        id=None, instance=DRIVE, desktop=DESKTOP, view="scan", pose_segment=1,
        anchor_step=14, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(mask))]))
    _entries(panel, DRIVE)[SA.MENU_ABSENT].trigger()
    assert not panel.asked
    assert "画过的零件就是在这台机器里" in panel.status.text()


def test_staged_edits_do_not_count_a_frame_object_as_one(panel):
    """A desktop with a 框外零件 used to read as an unapplied instance edit (U34 fix)."""
    from tda.core.model import InstanceRec

    panel.db.upsert_instance(InstanceRec(key="ctx.misc_part.01", desktop=DESKTOP,
                                         cls="misc_part",
                                         attrs={"context": True, "label": "cup"}))
    panel.set_desktop(DESKTOP)
    from tda.ui.steps_merge import staged_edits

    assert staged_edits(panel.data, panel.db) == []
