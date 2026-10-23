"""Task U20: ``import-logs`` gives every drive cage its own latch.

Two real sheets:

* ``desktop_13.csv`` (D13): "SSD shield locker" opened at 7 and never removed,
  "SSD shield" at 11 -- the log names the cage's latch, and the rule adopts it
  (the annotator had set that parent by hand in Steps);
* ``desktop_05.csv`` (D5; D6 is the same machine and the same rows): a
  separate "HDD shield locker" at 10, the drive at 13, the cage at 14. The
  annotators said with ``Shift+T`` that the locker was taken out at 10; after
  that, a forced re-import has no latch for the cage in the sheet: D5's was
  added by hand in Steps (kept), D6's is implied -- closed, open at 13, out
  with the cage at 14.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
from pathlib import Path

import pytest

from tda.core.db import Db
from tda.core.implied import OP_CAGE_LATCH, OP_DELETE_IMPLIED, is_cage_latch
from tda.core.model import FrameKey
from tda.core.state_assert import assert_state
from tda.core.states import gone_with_parent, validate_events
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import events_of, state_of
from tda.pipeline_logs import DROP_OP_KIND, import_logs_into_db, inferred_section
from tda.ui.session_verb import commit_verb_fix
from tda.ui.steps_delete import delete_instance
from tda.ui.steps_model import StepTableData

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
D5, D13 = 5, 13
CAGE = "drive_cage.01"
LOCKER = "drive_latch.01"
LATCH = "drive_latch.02"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def sheets(tmp_path: Path, desktop: int) -> str:
    folder = tmp_path / "drive"
    folder.mkdir(exist_ok=True)
    for suffix in ("", "_meta"):
        name = f"desktop_{desktop:02d}{suffix}.csv"
        shutil.copy(FIXTURES / name, folder / name)
    return str(folder)


def states(db: Db, tax, desktop: int, key: str, steps) -> list[str]:
    return [state_of(db, tax, desktop, s)[key].state for s in steps]


def reimport(db: Db, tmp_path: Path, tax, desktop: int = D5):
    printed: list[str] = []
    run = import_logs_into_db(db, sheets(tmp_path, desktop), tax, desktops={desktop},
                              force=True, log=printed.append)
    assert [r.status for r in run.runs] == ["imported"]
    return run.runs[0], "\n".join(printed)


@pytest.fixture
def d5(tmp_path, tax):
    """D5 imported from its sheet; the locker opened at 10."""
    db = Db(str(tmp_path / "tda.sqlite"))
    run = import_logs_into_db(db, sheets(tmp_path, D5), tax, desktops={D5})
    db.first_run = run.runs[0]
    yield db
    db.close()


@pytest.fixture
def d6(d5, tax):
    """D5's sheet with the locker taken out at 10 (Shift+T), as on D5 and D6."""
    commit_verb_fix(d5, TruthService(d5, tax), FrameKey(D5, 12, "scan"), LOCKER, "remove",
                    "tester")
    assert [a.verb for a in d5.actions(D5) if a.target == LOCKER] == ["remove"]
    return d5


# --------------------------------------------------------------------------- #
# adopt: D13
# --------------------------------------------------------------------------- #
def test_d13s_ssd_shield_locker_is_the_cages_latch(tmp_path, tax):
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        printed: list[str] = []
        run = import_logs_into_db(db, sheets(tmp_path, D13), tax, desktops={D13},
                                  log=printed.append)
        latch = db.instances(D13)[LOCKER]
        assert (latch.parent, latch.attached) == (CAGE, True)
        assert not [k for k, r in db.instances(D13).items() if is_cage_latch(r)]
        assert states(db, tax, D13, LOCKER, (6, 7, 10, 11)) == ["closed", "open", "open",
                                                                 "removed"]
        assert gone_with_parent(state_of(db, tax, D13, 11), LOCKER)
        line = ("cage latch adopted: drive_latch.01 is drive_cage.01's own latch (opened at "
                "step 7, never removed on its own) - parent drive_cage.01, attached")
        assert run.runs[0].cage_latches == [line] and line in "\n".join(printed)
        assert f"- {line}" in inferred_section(run.runs[0])
        [op] = db.ops_of_kind(D13, OP_CAGE_LATCH)
        assert op["payload"]["adopted"] == [{"latch": LOCKER, "cage": CAGE, "opened": 7}]
        assert op["inverse"] == {"adopted": [{"latch": LOCKER, "parent": None,
                                              "attached": False}], "created": []}
        # a forced re-import adopts it again: same parent, one more row
        reimport(db, tmp_path, tax, D13)
        assert db.instances(D13)[LOCKER].parent == CAGE
        assert len(db.ops_of_kind(D13, OP_CAGE_LATCH)) == 2
    finally:
        db.close()


def test_d5s_sheet_as_drafted_adopts_the_hdd_shield_locker(d5, tax):
    """Before Shift+T the sheet opens the locker at 10 and never removes it."""
    assert d5.first_run.cage_latches[0].startswith("cage latch adopted: drive_latch.01")
    assert d5.instances(D5)[LOCKER].parent == CAGE
    assert states(d5, tax, D5, LOCKER, (9, 10, 13, 14)) == ["closed", "open", "open",
                                                            "removed"]


# --------------------------------------------------------------------------- #
# create: D6
# --------------------------------------------------------------------------- #
def test_d6_gets_its_cages_latch_on_a_forced_re_import(d6, tmp_path, tax):
    run, printed = reimport(d6, tmp_path, tax)
    latch = d6.instances(D5)[LATCH]
    assert is_cage_latch(latch) and (latch.parent, latch.attached) == (CAGE, True)
    assert latch.attrs["of"] == "hdd" and latch.raw_names == []
    assert states(d6, tax, D5, LATCH, (1, 10, 12, 13, 14, 46)) == [
        "closed", "closed", "closed", "open", "removed", "removed"]
    assert gone_with_parent(state_of(d6, tax, D5, 14), LATCH)
    # the locker is out at 10 on its own, as Shift+T said
    assert states(d6, tax, D5, LOCKER, (9, 10)) == ["closed", "removed"]
    assert not validate_events(d6.instances(D5), events_of(d6, tax, D5), tax)
    line = ("implied instance drive_latch.02: drive_cage.01 自带的卡扣（日志里没有） - closed, "
            "open at step 13, out with drive_cage.01 at 14")
    assert line in run.implied and line in printed and "1 implied instance(s)" in printed
    assert not run.cage_latches and not run.dropped
    op = d6.ops_of_kind(D5, OP_CAGE_LATCH)[-1]
    assert [(c["instance"], c["cage"], c["opens"]) for c in op["payload"]["created"]] == [
        (LATCH, CAGE, 13)]
    assert op["inverse"]["created"] == [LATCH]


def test_a_second_forced_import_gives_back_the_same_latch(d6, tmp_path, tax):
    reimport(d6, tmp_path, tax)
    first = d6.instances(D5)[LATCH]
    run, _printed = reimport(d6, tmp_path, tax)
    assert d6.instances(D5)[LATCH] == first
    assert not run.dropped and not d6.ops_of_kind(D5, DROP_OP_KIND)
    assert sorted(k for k, r in d6.instances(D5).items() if is_cage_latch(r)) == [LATCH]


def test_d5s_latch_added_by_hand_is_kept(d6, tmp_path, tax):
    """D5: 「添加零件」 drive_latch.02 on the cage, drawn open at 13 with T."""
    data = StepTableData.load(d6, D5, tax)
    assert data.add_extras("drive_latch", parent=CAGE, attrs={"of": "hdd"}) == [LATCH]
    data.save(d6)
    assert_state(d6, tax, D5, LATCH, 13, "open")
    run, printed = reimport(d6, tmp_path, tax)
    line = "cage latch kept: drive_latch.02 (added in S1) is drive_cage.01's own latch"
    assert run.cage_latches == [line] and line in printed
    assert not [k for k, r in d6.instances(D5).items() if is_cage_latch(r)]
    assert d6.instances(D5)[LATCH].attrs.get("added_by")
    assert states(d6, tax, D5, LATCH, (12, 13, 14)) == ["closed", "open", "removed"]


# --------------------------------------------------------------------------- #
# deleted in Steps: stays deleted
# --------------------------------------------------------------------------- #
def test_a_latch_deleted_in_steps_stays_deleted(d6, tmp_path, tax):
    reimport(d6, tmp_path, tax)
    stored = d6.instances(D5)[LATCH]
    data = StepTableData.load(d6, D5, tax)
    delete_instance(data, d6, LATCH)
    assert LATCH not in d6.instances(D5)
    assert d6.declined_implied(D5) == set()           # per cage, not for the class
    [op] = d6.ops_of_kind(D5, OP_DELETE_IMPLIED)
    assert (op["payload"]["instance"], op["payload"]["cage"], op["payload"]["rule"]) == (
        LATCH, CAGE, "drive_cage_latch")
    assert op["inverse"]["record"]["attrs"] == stored.attrs
    assert op["inverse"]["record"]["parent"] == CAGE

    for _ in range(2):
        run, printed = reimport(d6, tmp_path, tax)
        line = ("cage latch kept deleted: drive_latch.02 of drive_cage.01 was deleted in "
                "Steps - not made again")
        assert run.cage_latches == [line] and line in printed
        assert LATCH not in d6.instances(D5) and not run.implied
        assert states(d6, tax, D5, CAGE, (13, 14)) == ["installed", "removed"]

    # the way back: add it by hand; the deletion then fits nothing any more
    data = StepTableData.load(d6, D5, tax)
    data.add_extras("drive_latch", parent=CAGE)
    data.save(d6)
    run, _printed = reimport(d6, tmp_path, tax)
    assert run.cage_latches == [
        "cage latch kept: drive_latch.02 (added in S1) is drive_cage.01's own latch",
        "cage latch delete not matched: drive_latch.02 of drive_cage.01 (drive_cage.01 has "
        "its latch drive_latch.02 now)"]
