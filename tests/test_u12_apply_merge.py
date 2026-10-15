"""Task U12 round 2: 「Apply 并合并」 on D2 -- one transaction, nothing demoted.

The scene is the real D2 sheet (``tests/fixtures/logs/desktop_02.csv``) as the
backup before the controller's merge had it: both covers taken out (38
``cover.01``, 46 ``cover.02``), the annotator's ``T`` "open at 38" on
``cover.02``, the cover drawn under ``cover.02`` (37, 41, 45) and scan frames
38-48 confirmed as they compiled. The fix needs two S1 edits -- step 38 back
to ``open``, step 46 onto ``cover.01`` -- and the merge.

Applied on their own, the two edits make ``cover.01`` open without a shape on
38-45 (and ``cover.02`` never taken out on 46-48), so the window's background
re-check after ``Apply`` demotes all eleven confirmed frames -- what the
annotator would have met. In one transaction with the merge nobody sees that
state: 38-48 stay confirmed, no conflict, no demotion.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from merge_scenes import everything, mask_kf
from tda.core.db import Db
from tda.core.extra import recompile_events
from tda.core.merge_instances import OP_MERGE
from tda.core.model import FrameKey, StateEvent
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.pipeline_logs import import_logs_into_db
from tda.ui import steps_merge as M
from tda.ui.panels.steptable import StepTablePanel
from tda.ui.steps_model import StepTableData

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
D2 = 2
VIEW = "scan"
CONFIRMED = list(range(38, 49))
ALL = list(range(1, 49))


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def _kf(anchor: int):
    kf = mask_kf("cover.02", (4, 4, 20, 20), anchor)
    kf.desktop = D2
    return kf


@pytest.fixture
def scene(tmp_path, tax):
    folder = tmp_path / "drive"
    folder.mkdir()
    for name in ("desktop_02.csv", "desktop_02_meta.csv"):
        shutil.copy(FIXTURES / name, folder / name)
    db = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(db, str(folder), tax, desktops={D2})
    # the backup: both covers taken out, the T at 38 on cover.02
    data = StepTableData.load(db, D2, tax)
    data.apply_edit(38, "verb", "remove")
    data.apply_edit(46, "verb", "remove")
    data.save(db)
    db.add_events([StateEvent(D2, 38, "cover.02", "state", "closed", "open", VIEW, auto=False)])
    recompile_events(db, D2, tax)
    db.set_pose_segment(D2, VIEW, 1, 1, 48, 48, None, None)
    for step in ALL:
        db.upsert_frame(FrameKey(D2, step, VIEW), f"s{step:03d}.jpg", {"hw": [64, 64]}, None)
    for anchor in (37, 41, 45):
        db.add_keyframe(_kf(anchor))
    svc = TruthService(db, tax, cache_dir=None)
    svc.refresh_range(D2, VIEW, ALL)
    # frames 38-48 confirmed as they compile (the parts nobody drew in this
    # miniature are frozen shape-less; the truth table compares them the same)
    with db.transaction():
        db.conn.execute(
            "UPDATE compiled_mask SET status='verified', verified_by='tester', "
            "verified_at='2026-10-14' WHERE desktop=? AND view=? AND step BETWEEN 38 AND 48",
            (D2, VIEW))
        for step in CONFIRMED:
            db.set_frame_flags(FrameKey(D2, step, VIEW), review_status="verified")
    assert svc.refresh_range(D2, VIEW, ALL)["conflicts"] == 0
    assert db.compiled(FrameKey(D2, 40, VIEW))["cover.02"]["status"] == "verified"
    yield db, svc
    db.close()


@pytest.fixture
def panel(qapp, scene, tmp_path, tax):
    db, _svc = scene
    widget = StepTablePanel(db, D2, taxonomy=tax, cache_dir=tmp_path / "cache")
    # the two S1 corrections, staged
    widget.data.change_target(46, target="cover.01", verb="remove")
    widget.data.apply_edit(38, "verb", "open")
    yield widget
    widget.deleteLater()


def _answer(monkeypatch, confirm=QMessageBox.Yes) -> dict:
    seen: dict = {}

    def get_item(_parent, _title, _label, items, *_a, **_k):
        return next(i for i in items if i.startswith("cover.01")), True

    def question(_parent, _title, text, *_a, **_k):
        seen["question"] = text
        return confirm

    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(get_item))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return seen


def _status(db: Db) -> dict[int, str]:
    return {r["step"]: r["review_status"] for r in db.frames_for(D2, VIEW)
            if r["step"] in CONFIRMED}


def _demotions(db: Db) -> int:
    return len([op for op in db.ops(D2, VIEW, limit=10_000) if op["kind"] == "demote_frame"])


def test_apply_and_merge_keeps_the_confirmed_frames(panel, scene, monkeypatch):
    db, svc = scene
    seen = _answer(monkeypatch)
    saved: list[int] = []
    panel.sigSaved.connect(saved.append)
    demoted = _demotions(db)

    panel.prompt_merge("cover.02")

    assert seen["question"].startswith("「Apply 并合并」：Steps 里还有没 Apply 的修改（步骤/动作）")
    assert seen["question"].endswith("和 Apply 一起立即写入（Revert 撤不回）。")
    assert "动作 1（第 46 步改成 cover.01）" not in seen["question"]   # S1 retargeted it
    assert "cover.02" not in db.instances(D2)
    covers = [(a.step, a.target, a.verb) for a in db.actions(D2) if a.target.startswith("cover")]
    assert covers == [(38, "cover.01", "open"), (46, "cover.01", "remove")]
    assert [kf.anchor_step for kf in db.keyframes(D2, VIEW, "cover.01")] == [37, 41, 45]
    assert len([op for op in db.ops(D2, "-", limit=1000) if op["kind"] == OP_MERGE]) == 1
    assert panel.status.text().startswith(
        "已 Apply（步骤/动作）并合并：已把 cover.02 合并到 cover.01")
    assert saved == [D2] and M.staged_edits(panel.data, db) == []

    # what the window does next: reopen the session, re-check every frozen frame
    out = svc.refresh_range(D2, VIEW, ALL)
    assert out["conflicts"] == 0 and out["standing"] == 0
    assert db.conflicts(D2, VIEW, open_only=True) == []
    assert _status(db) == {step: "verified" for step in CONFIRMED}
    assert _demotions(db) == demoted
    rows = db.compiled(FrameKey(D2, 40, VIEW))
    assert rows["cover.01"]["status"] == "verified" and "cover.02" not in rows


def test_why_not_apply_first(panel, scene):
    """The cost round 2 removes: Apply alone, then the re-check, demotes 38-48."""
    db, svc = scene
    panel.apply()
    svc.refresh_range(D2, VIEW, ALL)
    assert set(_status(db).values()) == {"needs_review"}


def test_a_refusal_after_a_good_save_rolls_the_save_back(panel, scene, monkeypatch):
    """Step 46 keeps an action on cover.02 too: the save is fine, the merge is not."""
    db, _svc = scene
    seen = _answer(monkeypatch)
    panel.data.add_action(46, target="cover.02", verb="remove")
    before = everything(db)

    panel.prompt_merge("cover.02")                     # the rehearsal refuses
    assert "question" not in seen
    assert panel.status.text().startswith("Rejected: 第 46 步 cover.02 和 cover.01 都有动作")
    assert everything(db) == before

    with pytest.raises(M.EditError, match="都有动作"):   # and the real thing too
        M.merge(panel.data, db, "cover.02", "cover.01", with_staged=True)
    assert everything(db) == before                   # nothing written, op_log included
    assert M.staged_edits(panel.data, db) == ["步骤/动作"]
    assert [a.target for a in panel.data.row(46).actions] == ["cover.01", "cover.02"]
    assert panel.data.row(38).actions[0].verb == "open"


def test_not_inside_another_transaction(panel, scene):
    db, _svc = scene
    with db.transaction():
        with pytest.raises(M.EditError, match="另一个事务"):
            M.merge(panel.data, db, "cover.02", "cover.01", with_staged=True)
