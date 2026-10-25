"""Task U27, the user's scene: D23's cable clip leaves with the CPU fan shroud.

D23 (HP Compaq 8100 Elite SFF, the real sheet in ``tests/fixtures/logs``):
step 2 "remove the cable from the case fan clip" releases ``cable_clip.01`` --
it only lifts the clip to pull the cable out -- and step 3 "CPU fan cover"
takes ``cover.01`` out (the controller said so with ``Shift+T``: the sheet's
verb is ``open``). The clip is part of the shroud. In Steps mode's instance
table: ``cable_clip.01`` -> Parent ``cover.01``, Attached ✓ -> Apply. The clip
is then closed on frame 1, open on 2 and removed on 3 with the shroud -- and
stays so through two forced re-imports, which leave the instance table, the
actions and the states exactly as they were.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from relation_scenes import CLIP, D23, SHROUD, fixes, imported, reimport, replayed
from tda.core.model import FrameKey
from tda.core.states import validate_events
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import events_of, state_of
from tda.ui.panels.steptable import StepTablePanel
from tda.ui.session_verb import commit_verb_fix


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def states(db, tax) -> list[tuple]:
    out = []
    for step in (1, 2, 3, 4):
        held = state_of(db, tax, D23, step)
        out.append((step, held[CLIP].state, held[CLIP].left_with, held[SHROUD].state))
    return out


def column(model, title: str) -> int:
    return [c.title for c in model.columns].index(title)


@pytest.fixture
def d23(qapp, tmp_path, tax):
    db = imported(tmp_path, tax, D23)
    assert [(a.step, a.target, a.verb) for a in db.actions(D23) if a.step in (2, 3)] == \
        [(2, CLIP, "release"), (3, SHROUD, "open")]
    commit_verb_fix(db, TruthService(db, tax), FrameKey(D23, 3, "scan"), SHROUD, "remove",
                    "chang")
    # before: the clip stays in the chassis, open, for ever
    assert states(db, tax)[2:] == [(3, "open", None, "removed"), (4, "open", None, "removed")]
    panel = StepTablePanel(db, D23, taxonomy=tax, cache_dir=tmp_path / "cache",
                           annotator="chang")
    model = panel.instances_model
    row = model.keys.index(CLIP)
    assert model.setData(model.index(row, column(model, "Parent")), SHROUD, Qt.EditRole)
    assert model.setData(model.index(row, column(model, "Attached")), Qt.Checked,
                         Qt.CheckStateRole)
    panel.apply()
    assert panel.status.text().startswith("Saved D23")
    panel.deleteLater()
    yield db
    db.close()


def test_the_clip_leaves_with_the_shroud(d23, tax):
    assert states(d23, tax) == [(1, "closed", None, "closed"), (2, "open", None, "closed"),
                                (3, "removed", SHROUD, "removed"),
                                (4, "removed", SHROUD, "removed")]
    assert fixes(d23, D23) == [(CLIP, "parent", None, SHROUD, "chang"),
                               (CLIP, "attached", False, True, "chang")]
    assert not validate_events(d23.instances(D23), events_of(d23, tax, D23), tax)


def test_two_forced_re_imports_change_nothing(d23, tmp_path, tax):
    table, actions, before = d23.instances(D23), d23.actions(D23), states(d23, tax)
    printed: list[str] = []
    first = reimport(d23, tmp_path, tax, D23, log=printed)
    assert d23.instances(D23) == table and d23.actions(D23) == actions
    assert states(d23, tax) == before
    once = replayed(d23, D23)
    second = reimport(d23, tmp_path, tax, D23)
    assert d23.instances(D23) == table and d23.actions(D23) == actions
    assert states(d23, tax) == before
    assert replayed(d23, D23) == once and second.corrections == first.corrections
    text = "\n".join(printed)
    # the warning says it is replayed now; the import says it was
    assert "the relational fields set in Steps' instance table" in text
    assert "relation fixes kept: cable_clip.01.parent = cover.01 (sheet: none), " \
           "cable_clip.01.attached = True (sheet: False)" in text
    assert "Shift+T corrections: kept the annotator's verb on 1 action(s): step 3 cover.01 " \
           "remove (sheet: open)" in text
