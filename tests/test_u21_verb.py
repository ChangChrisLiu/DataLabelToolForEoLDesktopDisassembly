"""Task U21, scope (a): 「其实只是打开了」 hangs a drive locker on its cage.

The user's clarification: the blue drive-cage locker is never taken off the
cage; it only opens and closes by pivoting on the cage's big screw and leaves
with the cage. Every annotated desktop's locker verb is ``remove`` only because
the annotator pressed ``Shift+T``. So the reverse ``Shift+T`` on a logged
``drive_latch`` whose desktop has a logged ``drive_cage`` runs the cage rule in
the same op: the latch is adopted (``parent`` = cage, ``attached``) and the
cage's untouched implied latch goes; Ctrl+Z puts both back.

Scenes: D5's sheet as the live D6/D7 hold it (locker out at 10 by Shift+T, a
forced U20 re-import gave the cage its implied ``drive_latch.02``), and as the
live D5 holds it (the cage's latch added by hand in S1 instead).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tda.core.db import Db
from tda.core.implied import is_cage_latch, is_extra
from tda.core.merge_instances import merge_instances
from tda.core.model import FrameKey
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import state_of
from tda.pipeline_logs import import_logs_into_db
from tda.ui.app_verb import verb_note
from tda.ui.session_verb import CAGE_PATCH, apply_correct_verb, commit_verb_fix
from tda.ui.steps_model import StepTableData

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
D5 = 5
CAGE, LOCKER, LATCH = "drive_cage.01", "drive_latch.01", "drive_latch.02"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def sheets(tmp_path: Path) -> str:
    folder = tmp_path / "drive"
    folder.mkdir(exist_ok=True)
    for suffix in ("", "_meta"):
        shutil.copy(FIXTURES / f"desktop_05{suffix}.csv", folder / f"desktop_05{suffix}.csv")
    return str(folder)


def flip(db, tax, step, to):
    return commit_verb_fix(db, TruthService(db, tax), FrameKey(D5, step, "scan"), LOCKER, to,
                           "tester")


def states(db, tax, key, steps):
    return [state_of(db, tax, D5, s)[key].state for s in steps]


def hung(db) -> tuple:
    rec = db.instances(D5)[LOCKER]
    return rec.parent, rec.attached


@pytest.fixture
def d6(tmp_path, tax):
    """The live D6/D7: the locker out at 10 by Shift+T, the cage's implied latch .02."""
    db = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(db, sheets(tmp_path), tax, desktops={D5})
    flip(db, tax, 12, "remove")
    import_logs_into_db(db, sheets(tmp_path), tax, desktops={D5}, force=True)
    assert hung(db) == (None, False) and is_cage_latch(db.instances(D5)[LATCH])
    yield db
    db.close()


@pytest.fixture
def d5(tmp_path, tax):
    """The live D5: the locker out at 10 by Shift+T, the cage's latch added by hand."""
    db = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(db, sheets(tmp_path), tax, desktops={D5})
    flip(db, tax, 12, "remove")
    db.delete_auto_events(D5, LATCH)              # the live D5 predates U20's .02
    db.delete_instance(D5, LATCH)
    data = StepTableData.load(db, D5, tax)
    assert data.add_extras("drive_latch", parent=CAGE, attrs={"of": "hdd"}) == [LATCH]
    data.save(db)
    import_logs_into_db(db, sheets(tmp_path), tax, desktops={D5}, force=True)
    assert hung(db) == (None, False) and is_extra(db.instances(D5)[LATCH])
    yield db
    db.close()


def test_only_opened_hangs_the_locker_on_the_cage_and_drops_the_implied_latch(d6, tax):
    result = flip(d6, tax, 12, "open")
    assert [(a.step, a.verb) for a in d6.actions(D5) if a.target == LOCKER] == [(10, "open")]
    assert hung(d6) == (CAGE, True) and LATCH not in d6.instances(D5)
    assert states(d6, tax, LOCKER, (9, 10, 13, 14)) == ["closed", "open", "open", "removed"]
    assert result["cage"] == {"cage": CAGE, "added": [], "dropped": [LATCH], "kept": [],
                              "opens": None, "adopted": [LOCKER], "also": []}
    note = verb_note("硬盘架卡扣 drive_latch.01", result)
    assert note == ("硬盘架卡扣 drive_latch.01 在第 10 步只是打开了（不是拆下）：从第 10 步起它还在"
                    "机箱里；drive_latch.01 已挂到 drive_cage.01 上，随机架一起拆下；"
                    "机架 drive_cage.01 的隐含卡扣 drive_latch.02 用不着了，已删掉（Ctrl+Z 撤销）")
    op = result["op"]
    assert op.payload[CAGE_PATCH]["repoint"] == [
        {"key": LOCKER, "before": [None, False], "after": [CAGE, True]}]
    assert op.inverse[CAGE_PATCH]["repoint"] == [
        {"key": LOCKER, "before": [CAGE, True], "after": [None, False]}]


def test_ctrl_z_unhangs_it_and_brings_the_implied_latch_back(d6, tax):
    latch = d6.instances(D5)[LATCH]
    locker = d6.instances(D5)[LOCKER]
    op = flip(d6, tax, 12, "open")["op"]
    truth = TruthService(d6, tax)
    apply_correct_verb(d6, truth, op.inverse)                    # Ctrl+Z
    assert d6.instances(D5)[LOCKER] == locker and d6.instances(D5)[LATCH] == latch
    assert [a.verb for a in d6.actions(D5) if a.target == LOCKER] == ["remove"]
    apply_correct_verb(d6, truth, op.payload)                    # redo
    assert hung(d6) == (CAGE, True) and LATCH not in d6.instances(D5)


def test_the_next_forced_import_agrees(d6, tmp_path, tax):
    flip(d6, tax, 12, "open")
    table = d6.instances(D5)
    run = import_logs_into_db(d6, sheets(tmp_path), tax, desktops={D5}, force=True).runs[0]
    assert d6.instances(D5) == table
    assert run.cage_latches == [
        "cage latch adopted: drive_latch.01 is drive_cage.01's own latch (opened at step 10, "
        "never removed on its own) - parent drive_cage.01, attached"]
    assert not run.implied and not run.dropped


def test_d5_the_hand_added_latch_stays_and_is_named(d5, tax):
    result = flip(d5, tax, 12, "open")
    assert hung(d5) == (CAGE, True) and LATCH in d5.instances(D5)
    assert result["cage"]["adopted"] == [LOCKER] and result["cage"]["also"] == [LATCH]
    assert result["cage"]["dropped"] == []
    assert ("机架 drive_cage.01 上还有 drive_latch.02：是同一个卡扣就到 Steps 右键「合并到…」"
            in verb_note("x", result))
    # the controller's last step on D5: the duplicate merged into the locker
    merge_instances(d5, tax, D5, LATCH, LOCKER, "controller")
    assert LATCH not in d5.instances(D5) and hung(d5) == (CAGE, True)


def test_a_latch_with_no_cage_on_its_desktop_is_not_hung(tmp_path, tax):
    """D2 before the shield is reclassed: no logged cage, nothing to hang it on."""
    from reclass_scenes import DESKTOP, build

    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        scene = build(db, locker_verb="remove", verify=False)
        result = commit_verb_fix(db, scene.svc, FrameKey(DESKTOP, 3, "scan"), "drive_latch.01",
                                 "open", "tester")
        assert result["cage"] is None
        assert db.instances(DESKTOP)["drive_latch.01"].parent is None
    finally:
        db.close()
