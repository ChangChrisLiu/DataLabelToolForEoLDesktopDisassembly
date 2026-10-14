"""Task U9 round 2: a forced re-import replays the ``Shift+T`` verb corrections.

``import-logs --force`` writes the freshly parsed actions over the whole step
table. The corrections made with ``Shift+T`` are ``correct_verb`` rows in the
op log, so the import puts the ones still in effect back, in op order, and
says which it kept and which the new sheet no longer fits.

The scene is the real D2 sheet (``tests/fixtures/logs/desktop_02.csv``) as it
was imported before the importer read "remove ..." names (step 17 ``open``),
with the controller's six corrections made through the session's path.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
from pathlib import Path

import pytest

from tda.core.db import Db
from tda.core.model import ActionRec, FrameKey, InstanceRec
from tda.core.states import validate_events
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import events_of
from tda.core.verb_fix import OP_CORRECT_VERB
from tda.core.verb_replay import OP_REPLAY, replay_corrections
from tda.pipeline_logs import import_logs_into_db
from tda.ui.session_verb import apply_correct_verb, commit_verb_fix
from tda.ui.steps_model import StepTableData

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
D2 = 2
SIX = {"cooler_latch.01": 2, "drive_latch.01": 12, "drive_latch.02": 17, "cover.01": 38,
       "cover.02": 46, "psu_latch.01": 47}


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def sheets(tmp_path: Path, edit=None) -> str:
    """The D2 sheet in a folder of its own, optionally with one row renamed."""
    folder = tmp_path / "drive"
    folder.mkdir(exist_ok=True)
    for name in ("desktop_02.csv", "desktop_02_meta.csv"):
        shutil.copy(FIXTURES / name, folder / name)
    if edit is not None:
        path = folder / "desktop_02.csv"
        old, new = edit
        text = path.read_text(encoding="utf-8-sig")
        assert old in text
        path.write_text(text.replace(old, new), encoding="utf-8")
    return str(folder)


def verbs(db: Db) -> dict[str, tuple[int, str]]:
    return {a.target: (a.step, a.verb) for a in db.actions(D2)}


@pytest.fixture
def corrected(tmp_path, tax):
    """D2 imported as the old importer did, then the six corrections (Shift+T)."""
    db = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(db, sheets(tmp_path), tax, desktops={D2})
    data = StepTableData.load(db, D2, tax)
    data.apply_edit(17, "verb", "open")          # before C, the sheet drafted open
    data.save(db)
    truth = TruthService(db, tax)
    key = FrameKey(D2, 48, "scan")
    for target in SIX:
        commit_verb_fix(db, truth, key, target, "remove", "tester")
    assert all(verbs(db)[t] == (s, "remove") for t, s in SIX.items())
    yield db, truth
    db.close()


def test_the_six_corrections_survive_a_forced_re_import(corrected, tmp_path, tax):
    db, _truth = corrected
    printed: list[str] = []
    run = import_logs_into_db(db, sheets(tmp_path), tax, desktops={D2}, force=True,
                              log=printed.append)
    assert [r.status for r in run.runs] == ["imported"]
    assert {t: verbs(db)[t] for t in SIX} == {t: (s, "remove") for t, s in SIX.items()}
    assert not validate_events(db.instances(D2), events_of(db, tax, D2), tax)
    op = db.ops_of_kind(D2, OP_REPLAY)[-1]
    replayed = {(e["step"], e["target"]) for e in op["payload"]["replayed"]}
    assert replayed == {(s, t) for t, s in SIX.items() if t != "drive_latch.02"}
    # the sheet itself says "remove SSD locker shield" now (C): nothing to replay
    assert [(e["step"], e["target"]) for e in op["payload"]["already"]] == \
        [(17, "drive_latch.02")]
    assert op["payload"]["unmatched"] == []
    assert sorted(a["verb"] for a in op["inverse"]["actions"]) == ["open"] * 5
    text = "\n".join(printed)
    assert "Shift+T are replayed" in text                       # the --force warning
    assert "kept 6 verb correction(s) made with Shift+T (5 replayed)" in text
    assert "step 17 drive_latch.02 open->remove (already in the sheet)" in text
    assert run.runs[0].corrections and run.runs[0].corrections[0].startswith("kept 6")


def test_a_second_forced_import_changes_nothing(corrected, tmp_path, tax):
    db, _truth = corrected
    folder = sheets(tmp_path)
    import_logs_into_db(db, folder, tax, desktops={D2}, force=True)
    first = (verbs(db), db.ops_of_kind(D2, OP_REPLAY)[-1]["payload"])
    import_logs_into_db(db, folder, tax, desktops={D2}, force=True)
    second = (verbs(db), db.ops_of_kind(D2, OP_REPLAY)[-1]["payload"])
    assert first == second
    assert len(db.ops_of_kind(D2, OP_REPLAY)) == 2


def test_a_correction_the_new_sheet_no_longer_fits_is_reported_not_applied(
        corrected, tmp_path, tax):
    """Step 47 now names a GPU locker: the PSU latch's correction has no row to go on."""
    db, _truth = corrected
    printed: list[str] = []
    folder = sheets(tmp_path, ("47,Power Module Blocker", "47,GPU locker"))
    import_logs_into_db(db, folder, tax, desktops={D2}, force=True, log=printed.append)
    assert verbs(db)["card_latch.01"] == (47, "open")          # the sheet's own verb
    assert "psu_latch.01" not in verbs(db)
    op = db.ops_of_kind(D2, OP_REPLAY)[-1]
    [missed] = op["payload"]["unmatched"]
    assert (missed["step"], missed["target"]) == (47, "psu_latch.01")
    assert "no longer has psu_latch.01 at step 47" in missed["why"]
    assert "card_latch.01" in missed["why"]
    assert any("verb correction not replayed: step 47 psu_latch.01 open->remove" in line
               for line in printed)
    assert len(op["payload"]["replayed"]) == 4                  # the other five stay


def test_a_correction_taken_back_with_ctrl_z_is_not_replayed(corrected, tmp_path, tax):
    db, truth = corrected
    op = [o for o in db.ops_of_kind(D2, OP_CORRECT_VERB)
          if o["payload"]["target"] == "cover.01"][-1]
    apply_correct_verb(db, truth, op["inverse"])               # what Ctrl+Z runs
    assert verbs(db)["cover.01"] == (38, "open")
    import_logs_into_db(db, sheets(tmp_path), tax, desktops={D2}, force=True)
    assert verbs(db)["cover.01"] == (38, "open")
    replay = db.ops_of_kind(D2, OP_REPLAY)[-1]["payload"]
    [back] = replay["unmatched"]
    assert back["target"] == "cover.01" and "taken back" in back["why"]


def test_ops_replay_in_order_and_a_verb_must_fit_the_class(tax):
    """Pure: open -> remove -> open on one action nets open; a third verb is reported."""
    inst = {"cover.01": InstanceRec(key="cover.01", desktop=1, cls="cover"),
            "psu_latch.01": InstanceRec(key="psu_latch.01", desktop=1, cls="psu_latch")}

    def op(i, step, target, old, new):
        return {"id": i, "payload": {"step": step, "target": target, "verb": new},
                "inverse": {"step": step, "target": target, "verb": old}}

    fresh = [ActionRec(1, 3, 0, "cover.01", "open"), ActionRec(1, 5, 0, "psu_latch.01", "open")]
    stored = [ActionRec(1, 3, 0, "cover.01", "open"), ActionRec(1, 5, 0, "psu_latch.01", "remove")]
    ops = [op(1, 3, "cover.01", "open", "remove"), op(2, 3, "cover.01", "remove", "open"),
           op(3, 5, "psu_latch.01", "open", "remove")]
    out, report = replay_corrections(ops, stored, fresh, inst, tax)
    assert [a.verb for a in out] == ["open", "remove"]
    assert [e["op"] for e in report.replayed] == [1, 2, 3]
    assert [a.verb for a in fresh] == ["open", "open"]          # the input is left alone
    # the sheet now has a third verb on the latch: reported, not applied
    fresh2 = [fresh[0], ActionRec(1, 5, 0, "psu_latch.01", "displace")]
    out2, report2 = replay_corrections(ops[2:], stored, fresh2, inst, tax)
    assert out2[1].verb == "displace"
    assert "the sheet has displace there, not open" in report2.unmatched[0]["why"]
