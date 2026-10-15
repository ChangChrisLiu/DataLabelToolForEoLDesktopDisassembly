"""Task U12: 「合并到…」 in the Steps panel's Instances table (offscreen Qt)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from merge_scenes import DESKTOP, GONE, KEEP, PSU, blob, build
from tda.core.db import Db
from tda.core.merge_instances import OP_MERGE
from tda.core.model import InstanceRec
from tda.ui import steps_merge as M
from tda.ui.panels.steptable import StepTablePanel
from tda.ui.steps_model import StepTableData


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def db(tmp_db_path):
    d = Db(tmp_db_path)
    yield d
    d.close()


@pytest.fixture
def scene(db):
    return build(db)


@pytest.fixture
def panel(qapp, scene, tmp_path):
    widget = StepTablePanel(scene.db, DESKTOP, taxonomy=scene.tax, cache_dir=tmp_path / "cache")
    yield widget
    widget.deleteLater()


def _menu_entry(panel, key):
    menu = panel.instances_menu(panel.instances_model.keys.index(key))
    entries = {a.text(): a for a in menu.actions()}
    return menu, entries.get(M.MENU_TEXT)


def _answer(monkeypatch, pick: str, confirm=QMessageBox.Yes) -> dict:
    seen: dict = {}

    def get_item(_parent, _title, label, items, *_a, **_k):
        seen["items"], seen["label"] = list(items), label
        return next(i for i in items if i.startswith(pick)), True

    def question(_parent, _title, text, *_a, **_k):
        seen["question"] = text
        return confirm

    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(get_item))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return seen


def test_the_menu_offers_the_merge_only_when_there_is_a_same_class_part(panel):
    menu, entry = _menu_entry(panel, GONE)
    assert entry is not None and entry.isEnabled()
    assert menu.actions()[0].text() == f"Delete {GONE}"     # delete stays first
    menu.deleteLater()
    menu, entry = _menu_entry(panel, PSU)
    assert entry is not None and not entry.isEnabled()
    menu.deleteLater()


def test_choices_are_the_same_class_without_drafts(panel, scene):
    scene.db.upsert_instance(InstanceRec("ls:Cover#1", DESKTOP, "cover"))
    panel.set_desktop(DESKTOP)
    assert M.merge_choices(panel.data, GONE) == [KEEP]
    assert M.merge_choices(panel.data, "ls:Cover#1") == []
    assert M.choice_label(panel.data, KEEP) == \
        "cover.01 — Screw Cover（第 3 步 open，第 5 步 remove）"


def test_pick_confirm_and_merge(panel, scene, monkeypatch):
    seen = _answer(monkeypatch, KEEP)
    saved: list[int] = []
    panel.sigSaved.connect(saved.append)
    menu, entry = _menu_entry(panel, GONE)
    entry.trigger()
    menu.deleteLater()

    assert seen["items"] == [M.choice_label(panel.data, KEEP)]
    assert "把 cover.02（Front Cover） 合并到 cover.01（Screw Cover）" in seen["question"]
    assert "已确认的行 2 行（scan 第 3–4 帧）" in seen["question"]
    assert GONE not in scene.db.instances(DESKTOP)
    assert GONE not in panel.data.instances                  # the panel read it back
    assert panel.status.text().startswith("已把 cover.02 合并到 cover.01（形状 2")
    assert saved == [DESKTOP]                                # the window reopens the session
    assert len([op for op in scene.db.ops(DESKTOP, "-") if op["kind"] == OP_MERGE]) == 1
    assert not M.staged_edits(panel.data, scene.db)


def test_no_writes_nothing(panel, scene, monkeypatch):
    _answer(monkeypatch, KEEP, confirm=QMessageBox.No)
    before = blob(scene.db)
    panel.prompt_merge(GONE)
    assert blob(scene.db) == before
    assert "没有合并" in panel.status.text()


@pytest.mark.parametrize("edit", ["step", "instance", "extra"])
def test_staged_edits_are_refused_first(panel, scene, monkeypatch, edit):
    seen = _answer(monkeypatch, KEEP)
    if edit == "step":
        panel.data.apply_edit(3, "notes", "looked again")
        what = "步骤/动作"
    elif edit == "instance":
        panel.data.apply_instance_edit(GONE, "removal_direction", "+Z")
        what = "实例表"
    else:
        panel.add_parts("ram_latch", 1)
        what = "添加的零件"
    before = blob(scene.db)
    panel.prompt_merge(GONE)
    assert blob(scene.db) == before
    assert "question" not in seen                       # refused before the confirmation
    assert panel.status.text().startswith("Rejected: 先 Apply 或 Revert")
    assert what in panel.status.text()


def test_a_core_refusal_is_shown(qapp, db, tmp_path, monkeypatch):
    """Merging before S1 is corrected contradicts the confirmed frames 3-4."""
    scene = build(db, s1=False)
    panel = StepTablePanel(db, DESKTOP, taxonomy=scene.tax, cache_dir=tmp_path / "cache")
    seen = _answer(monkeypatch, KEEP)
    panel.prompt_merge(GONE)
    assert "question" not in seen
    assert "和这些已确认帧对不上" in panel.status.text()
    assert GONE in db.instances(DESKTOP)
    panel.deleteLater()


def test_the_view_model_half_alone(scene):
    data = StepTableData.load(scene.db, DESKTOP, scene.tax)
    report = M.merge(data, scene.db, GONE, KEEP)
    assert report.plan.keep == KEEP and GONE not in scene.db.instances(DESKTOP)
    with pytest.raises(M.EditError, match="没有实例"):
        M.plan(StepTableData.load(scene.db, DESKTOP, scene.tax), scene.db, GONE, KEEP)
