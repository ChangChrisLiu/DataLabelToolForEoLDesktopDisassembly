"""Task U9: a forced re-import keeps what ``Shift+T`` left in the step table.

``import-logs --force`` writes the freshly parsed actions over the whole step
table. The ``correct_verb`` rows of the op log say which actions an annotator
corrected; for each, the verb the stored table carries -- the annotator's last
word, whatever undo/redo led there -- is copied onto the parsed action.

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
KEY = FrameKey(D2, 48, "scan")
#: The parts said with Shift+T to be taken out. Task U21: D2's "remove SSD
#: locker shield" (step 17) is the cage now -- the sheet's own ``remove``, no
#: correction -- and its "SSD Locker" (step 12) is the cage's latch, which the
#: old drafts had taken out and the annotator says was only opened.
SIX = {"cooler_latch.01": 2, "cover.01": 38, "cover.02": 46, "psu_latch.01": 47}
LOCKER, LOCKER_STEP = "drive_latch.01", 12


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


def old_import(tmp_path: Path, tax) -> Db:
    """D2 imported, the SSD locker at 12 ``remove`` as the live D2 had it before U21."""
    db = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(db, sheets(tmp_path), tax, desktops={D2})
    data = StepTableData.load(db, D2, tax)
    data.apply_edit(LOCKER_STEP, "verb", "remove")
    data.save(db)
    return db


@pytest.fixture
def corrected(tmp_path, tax):
    """The four corrections made with Shift+T, and the locker said only opened."""
    db = old_import(tmp_path, tax)
    truth = TruthService(db, tax)
    for target in SIX:
        commit_verb_fix(db, truth, KEY, target, "remove", "tester")
    commit_verb_fix(db, truth, FrameKey(D2, LOCKER_STEP, "scan"), LOCKER, "open", "tester")
    assert all(verbs(db)[t] == (s, "remove") for t, s in SIX.items())
    assert verbs(db)[LOCKER] == (LOCKER_STEP, "open")
    yield db, truth
    db.close()


def test_the_six_corrections_survive_a_forced_re_import(corrected, tmp_path, tax):
    db, _truth = corrected
    printed: list[str] = []
    run = import_logs_into_db(db, sheets(tmp_path), tax, desktops={D2}, force=True,
                              log=printed.append)
    assert [r.status for r in run.runs] == ["imported"]
    assert {t: verbs(db)[t] for t in SIX} == {t: (s, "remove") for t, s in SIX.items()}
    assert verbs(db)[LOCKER] == (LOCKER_STEP, "open")
    assert not validate_events(db.instances(D2), events_of(db, tax, D2), tax)
    op = db.ops_of_kind(D2, OP_REPLAY)[-1]
    kept = {(e["step"], e["target"], e["verb"], e["sheet"]) for e in op["payload"]["kept"]}
    assert kept == {(s, t, "remove", "open") for t, s in SIX.items()}
    # the sheet itself says the locker was opened: it already agrees
    assert [(e["step"], e["target"], e["verb"]) for e in op["payload"]["agrees"]] == \
        [(LOCKER_STEP, LOCKER, "open")]
    assert op["payload"]["unmatched"] == []
    assert sorted(a["verb"] for a in op["inverse"]["actions"]) == ["open"] * 4
    text = "\n".join(printed)
    assert "Shift+T are replayed" in text                       # the --force warning
    assert "kept the annotator's verb on 4 action(s): step 2 cooler_latch.01 remove " \
           "(sheet: open)" in text
    assert "the sheet already agrees on 1: step 12 drive_latch.01 open" in text
    assert run.runs[0].corrections[0].startswith("Shift+T corrections: kept")


def test_a_second_forced_import_changes_nothing(corrected, tmp_path, tax):
    db, _truth = corrected
    folder = sheets(tmp_path)
    import_logs_into_db(db, folder, tax, desktops={D2}, force=True)
    first = (verbs(db), db.ops_of_kind(D2, OP_REPLAY)[-1]["payload"])
    import_logs_into_db(db, folder, tax, desktops={D2}, force=True)
    second = (verbs(db), db.ops_of_kind(D2, OP_REPLAY)[-1]["payload"])
    assert first == second
    assert len(db.ops_of_kind(D2, OP_REPLAY)) == 2


def test_an_action_the_new_sheet_no_longer_has_is_reported_not_applied(
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
    assert any("Shift+T correction not matched: step 47 psu_latch.01" in line
               for line in printed)
    assert len(op["payload"]["kept"]) == 3                      # the other four stay
    assert len(op["payload"]["agrees"]) == 1


#: The reviewer's table: what the step table holds after each sequence on
#: cover.01 (T = Shift+T, back = 「其实只是打开了」, Z = Ctrl+Z, Y = Ctrl+Y),
#: and so what a forced re-import must leave there.
SEQUENCES = {
    "T,Z,Y": "remove",
    "T,Z": "open",
    "T,Z,T": "remove",
    "T,back,T": "remove",
    "T,back": "open",
    "T,back,Z": "remove",
}


@pytest.mark.parametrize("sequence,expected", sorted(SEQUENCES.items()),
                         ids=sorted(SEQUENCES))
def test_undo_and_redo_paths_keep_the_last_word(tmp_path, tax, sequence, expected):
    db = old_import(tmp_path, tax)
    truth = TruthService(db, tax)
    done: list[dict] = []
    undone: list[dict] = []
    for move in sequence.split(","):
        if move in ("T", "back"):
            result = commit_verb_fix(db, truth, KEY, "cover.01",
                                     "remove" if move == "T" else "open", "tester")
            done.append({"payload": result["op"].payload, "inverse": result["op"].inverse})
            undone.clear()
        elif move == "Z":
            op = done.pop()
            apply_correct_verb(db, truth, op["inverse"])        # what Ctrl+Z runs
            undone.append(op)
        else:
            op = undone.pop()
            apply_correct_verb(db, truth, op["payload"])        # what Ctrl+Y runs
            done.append(op)
    assert verbs(db)["cover.01"] == (38, expected)              # the table's last word
    folder = sheets(tmp_path)
    import_logs_into_db(db, folder, tax, desktops={D2}, force=True)
    assert verbs(db)["cover.01"] == (38, expected)
    import_logs_into_db(db, folder, tax, desktops={D2}, force=True)
    assert verbs(db)["cover.01"] == (38, expected)
    payload = db.ops_of_kind(D2, OP_REPLAY)[-1]["payload"]
    assert payload["unmatched"] == []
    listed = payload["kept"] if expected == "remove" else payload["agrees"]
    assert [(e["target"], e["verb"]) for e in listed] == [("cover.01", expected)]
    db.close()


def test_the_rule_is_pure_and_order_free(tax):
    """Which actions were corrected comes from the ops; what they say, from the table."""
    inst = {"cover.01": InstanceRec(key="cover.01", desktop=1, cls="cover"),
            "psu_latch.01": InstanceRec(key="psu_latch.01", desktop=1, cls="psu_latch"),
            "screw.01": InstanceRec(key="screw.01", desktop=1, cls="screw")}

    def op(i, step, target):
        return {"id": i, "payload": {"step": step, "target": target, "verb": "remove"},
                "inverse": {"step": step, "target": target, "verb": "open"}}

    fresh = [ActionRec(1, 3, 0, "cover.01", "open"), ActionRec(1, 5, 0, "psu_latch.01", "open"),
             ActionRec(1, 6, 0, "screw.01", "unscrew")]
    stored = [ActionRec(1, 3, 0, "cover.01", "remove"),
              ActionRec(1, 5, 0, "psu_latch.01", "open"),
              ActionRec(1, 6, 0, "screw.01", "unscrew")]
    ops = [op(1, 3, "cover.01"), op(2, 5, "psu_latch.01"), op(3, 3, "cover.01"),
           op(4, 6, "screw.01"), op(5, 9, "cover.02")]
    out, report = replay_corrections(ops, stored, fresh, inst, tax)
    again, _ = replay_corrections(list(reversed(ops)), stored, fresh, inst, tax)
    assert [a.verb for a in out] == [a.verb for a in again] == ["remove", "open", "unscrew"]
    assert [a.verb for a in fresh] == ["open", "open", "unscrew"]   # the input is left alone
    assert [(e["target"], e["ops"]) for e in report.kept] == [("cover.01", [1, 3])]
    assert [e["target"] for e in report.agrees] == ["psu_latch.01"]
    whys = {e["target"]: e["why"] for e in report.unmatched}
    assert "no cover.02 at step 9" in whys["cover.02"]
    assert "unscrew" in whys["screw.01"]
