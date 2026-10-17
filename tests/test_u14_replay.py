"""Task U14: a forced re-import keeps the captive flags ``Shift+T`` flipped.

``import-logs --force`` re-creates every instance from the sheet and the
importer guesses ``captive`` again (every CPU-cooler screw of a Dell). The
``captive_fix`` rows of the op log say which screws an annotator flipped; the
value the stored table carries -- the annotator's last word -- overrides the
guess on the parsed record, matched by key, before the relational inference
fills a captive screw's ``parent``.

The scene is the real D2 sheet (``tests/fixtures/logs/desktop_02.csv``, a Dell
OptiPlex 7020): four captive CPU-fan screws unscrewed at 3-6, six plain
motherboard screws at 39-44.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
from pathlib import Path

import pytest

from tda.core.captive_fix import OP_CAPTIVE_FIX
from tda.core.captive_replay import OP_REPLAY_CAPTIVE, replay_captive
from tda.core.db import Db
from tda.core.model import FrameKey, InstanceRec
from tda.core.states import validate_events
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import events_of, state_of
from tda.pipeline_logs import import_logs_into_db
from tda.ui.session_captive import apply_captive_fix, commit_captive_fix

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
D2 = 2
KEY = FrameKey(D2, 48, "scan")
FAN = "screw.cpu_cooler.01"        # captive by the guess, unscrewed at 3
FAN4 = "screw.cpu_cooler.04"       # ... at 6
BOARD = "screw.motherboard.01"     # plain, unscrewed at 39


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


@pytest.fixture
def imported(tmp_path, tax):
    db = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(db, sheets(tmp_path), tax, desktops={D2})
    yield db, TruthService(db, tax)
    db.close()


def rec(db: Db, key: str) -> InstanceRec:
    return db.instances(D2)[key]


def flip(db, truth, key: str) -> dict:
    return commit_captive_fix(db, truth, KEY, key, None, "tester")


def test_the_guess_is_what_the_fixture_says(imported):
    db, _truth = imported
    assert rec(db, FAN).attrs["captive"] is True and rec(db, FAN).parent == "cpu_cooler.fan.01"
    assert rec(db, BOARD).attrs["captive"] is False and rec(db, BOARD).parent is None


def test_a_flip_survives_a_forced_re_import(imported, tmp_path, tax):
    db, truth = imported
    flip(db, truth, FAN)
    printed: list[str] = []
    run = import_logs_into_db(db, sheets(tmp_path), tax, desktops={D2}, force=True,
                              log=printed.append)
    assert [r.status for r in run.runs] == ["imported"]
    fan = rec(db, FAN)
    assert (fan.attrs["captive"], fan.attrs["captive_source"]) == (False, "manual")
    assert fan.parent is None and not fan.attached          # the inference gave it none
    held = state_of(db, tax, D2, 3)[FAN]
    assert (held.state, held.placement) == ("removed", "on_bench")
    assert not validate_events(db.instances(D2), events_of(db, tax, D2), tax)
    op = db.ops_of_kind(D2, OP_REPLAY_CAPTIVE)[-1]
    assert [(e["target"], e["captive"], e["sheet"]) for e in op["payload"]["kept"]] == \
        [(FAN, False, True)]
    assert op["payload"]["agrees"] == op["payload"]["unmatched"] == []
    assert op["inverse"]["instances"] == [{"target": FAN, "captive": True,
                                           "captive_source": "heuristic"}]
    text = "\n".join(printed)
    assert "the captive flags flipped with Shift+T" in text           # the --force warning
    line = "captive fixes (Shift+T) kept 1: screw.cpu_cooler.01 not captive (sheet: captive)"
    assert line in text and line in run.runs[0].corrections
    # the other fan screws keep the guess
    assert rec(db, "screw.cpu_cooler.02").attrs["captive"] is True


def test_stays_in_survives_with_the_part_it_fastens(imported, tmp_path, tax):
    db, truth = imported
    result = flip(db, truth, BOARD)
    assert result["fix"].parent == "motherboard.01"
    import_logs_into_db(db, sheets(tmp_path), tax, desktops={D2}, force=True)
    board = rec(db, BOARD)
    assert board.attrs["captive"] is True and board.attrs["captive_source"] == "manual"
    assert (board.parent, board.attached) == ("motherboard.01", True)
    assert state_of(db, tax, D2, 39)[BOARD].state == "loosened"


def test_a_second_forced_import_changes_nothing(imported, tmp_path, tax):
    db, truth = imported
    flip(db, truth, FAN)
    folder = sheets(tmp_path)
    import_logs_into_db(db, folder, tax, desktops={D2}, force=True)
    first = (rec(db, FAN), db.ops_of_kind(D2, OP_REPLAY_CAPTIVE)[-1]["payload"])
    import_logs_into_db(db, folder, tax, desktops={D2}, force=True)
    second = (rec(db, FAN), db.ops_of_kind(D2, OP_REPLAY_CAPTIVE)[-1]["payload"])
    assert first == second
    assert len(db.ops_of_kind(D2, OP_REPLAY_CAPTIVE)) == 2


def test_a_screw_the_new_sheet_no_longer_has_is_reported_not_applied(imported, tmp_path, tax):
    db, truth = imported
    flip(db, truth, FAN)
    flip(db, truth, FAN4)
    printed: list[str] = []
    folder = sheets(tmp_path, ("6,CPU fan screw 4", "6,Optical drive screw 1"))
    import_logs_into_db(db, folder, tax, desktops={D2}, force=True, log=printed.append)
    assert FAN4 not in db.instances(D2)
    op = db.ops_of_kind(D2, OP_REPLAY_CAPTIVE)[-1]
    [missed] = op["payload"]["unmatched"]
    assert missed["target"] == FAN4 and "no longer has the screw" in missed["why"]
    assert [e["target"] for e in op["payload"]["kept"]] == [FAN]
    assert any(line.endswith(f"captive fix not matched: {FAN4} (the sheet no longer has the "
                             f"screw {FAN4})") for line in printed)


def test_a_manual_value_the_sheet_agrees_with_stays_manual(imported, tmp_path, tax):
    """Flipped twice (out, back in): the guess agrees, the annotator's word stands (N1)."""
    db, truth = imported
    flip(db, truth, FAN)
    flip(db, truth, FAN)
    assert (rec(db, FAN).attrs["captive"], rec(db, FAN).attrs["captive_source"]) == \
        (True, "manual")
    folder = sheets(tmp_path)
    for _ in range(2):                                         # and on the next one too
        import_logs_into_db(db, folder, tax, desktops={D2}, force=True)
        fan = rec(db, FAN)
        assert (fan.attrs["captive"], fan.attrs["captive_source"]) == (True, "manual")
        op = db.ops_of_kind(D2, OP_REPLAY_CAPTIVE)[-1]
        assert [(e["target"], e["captive"], e["sheet"]) for e in op["payload"]["agrees"]] == \
            [(FAN, True, True)]
        assert op["payload"]["kept"] == op["payload"]["unmatched"] == []


def test_a_flip_taken_back_agrees_with_the_sheet(imported, tmp_path, tax):
    db, truth = imported
    result = flip(db, truth, FAN)
    apply_captive_fix(db, truth, result["op"].inverse)          # Ctrl+Z
    assert rec(db, FAN).attrs["captive_source"] == "heuristic"
    import_logs_into_db(db, sheets(tmp_path), tax, desktops={D2}, force=True)
    assert rec(db, FAN).attrs["captive"] is True
    op = db.ops_of_kind(D2, OP_REPLAY_CAPTIVE)[-1]
    assert [(e["target"], e["captive"]) for e in op["payload"]["agrees"]] == [(FAN, True)]
    assert op["payload"]["kept"] == []


def test_the_pure_replay_follows_merges_and_keeps_added_screws():
    def screw(key, captive, source="heuristic", **kw):
        return InstanceRec(key, 1, "screw", attrs={"captive": captive,
                                                   "captive_source": source}, **kw)

    ops = [{"id": 1, "payload": {"target": "s.02"}}, {"id": 2, "payload": {"target": "s.01"}},
           {"id": 3, "payload": {"target": "s.09"}}, {"id": 4, "payload": {"target": "s.05"}},
           {"id": 5, "payload": {"target": "s.06"}}]
    added = InstanceRec("s.09", 1, "screw", attrs={"captive": True, "captive_source": "manual",
                                                   "added_by": "tester"})   # added in S1
    stored = {"s.01": screw("s.01", False, "manual"), "s.09": added,
              "s.05": screw("s.05", True, "heuristic"), "s.06": screw("s.06", True, "manual")}
    fresh = {"s.01": screw("s.01", True), "s.02": screw("s.02", True),
             "s.05": screw("s.05", False), "s.06": screw("s.06", True)}
    out, report = replay_captive(ops, stored, fresh, rename={"s.02": "s.01"})
    assert out["s.01"].attrs == {"captive": False, "captive_source": "manual"}
    assert fresh["s.01"].attrs["captive"] is True                  # the input is left alone
    assert [(e["target"], e["ops"]) for e in report.kept] == [("s.01", [1, 2])]
    # added in S1: kept as is; the guess agrees with a manual value: still manual
    assert [e["target"] for e in report.agrees] == ["s.06", "s.09"]
    assert out["s.06"].attrs == {"captive": True, "captive_source": "manual"}
    assert fresh["s.06"].attrs["captive_source"] == "heuristic"
    [taken_back] = report.unmatched
    assert taken_back["target"] == "s.05" and "taken back" in taken_back["why"]
    assert out["s.05"].attrs["captive"] is False                   # the guess stands


def test_the_op_kinds_are_one(tax):
    from tda.core.captive_replay import OP_CAPTIVE_FIX as REPLAYED
    from tda.ui.commands import KINDS

    assert OP_CAPTIVE_FIX == REPLAYED == "captive_fix" and OP_CAPTIVE_FIX in KINDS
