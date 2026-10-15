"""Task U12: a forced re-import keeps the merges made with 「合并到…」.

The D2 sheet (``tests/fixtures/logs/desktop_02.csv``) names one cover twice:
row 38 "Screw Cover" -> ``cover.01`` and row 46 "Front Cover" -> ``cover.02``.
After ``cover.02`` is merged into ``cover.01``, ``import-logs --force`` must not
bring ``cover.02`` back nor point step 46 at it again.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tda.core.db import Db
from tda.core.merge_instances import merge_instances, undo_merge
from tda.core.merge_replay import OP_REPLAY_MERGES, replay_merges
from tda.core.model import ActionRec, FrameKey, InstanceRec
from tda.core.states import validate_events
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import events_of
from tda.pipeline_logs import import_logs_into_db
from tda.ui.session_verb import commit_verb_fix

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
D2 = 2


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def sheets(tmp_path: Path, edit=None) -> str:
    folder = tmp_path / "drive"
    folder.mkdir(exist_ok=True)
    for name in ("desktop_02.csv", "desktop_02_meta.csv"):
        shutil.copy(FIXTURES / name, folder / name)
    if edit is not None:
        path = folder / "desktop_02.csv"
        text = path.read_text(encoding="utf-8-sig")
        assert edit[0] in text
        path.write_text(text.replace(*edit), encoding="utf-8")
    return str(folder)


@pytest.fixture
def db(tmp_path, tax):
    d = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(d, sheets(tmp_path), tax, desktops={D2})
    yield d
    d.close()


def targets(db: Db) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for a in db.actions(D2):
        out.setdefault(a.step, []).append(a.target)
    return out


def table(db: Db) -> tuple:
    return ([(a.step, a.idx, a.target, a.verb) for a in db.actions(D2)],
            sorted(db.instances(D2)))


def test_the_d2_merge_survives_two_forced_imports(db, tmp_path, tax):
    assert targets(db)[46] == ["cover.02"]
    merge_instances(db, tax, D2, "cover.02", "cover.01", "tester")
    assert targets(db)[46] == ["cover.01"]
    merged = table(db)

    printed: list[str] = []
    folder = sheets(tmp_path)
    run = import_logs_into_db(db, folder, tax, desktops={D2}, force=True, log=printed.append)
    assert [r.status for r in run.runs] == ["imported"]
    assert "cover.02" not in db.instances(D2)
    assert targets(db)[46] == ["cover.01"] and targets(db)[38] == ["cover.01"]
    assert table(db) == merged
    assert not validate_events(db.instances(D2), events_of(db, tax, D2), tax)
    [op] = db.ops_of_kind(D2, OP_REPLAY_MERGES)
    assert [(e["gone"], e["keep"], e["steps"]) for e in op["payload"]["replayed"]] == [
        ("cover.02", "cover.01", [46])]
    assert op["payload"]["already"] == [] and op["payload"]["unmatched"] == []
    assert op["inverse"]["actions"] == [{"step": 46, "idx": 0, "target": "cover.02"}]
    assert [r["key"] for r in op["inverse"]["instances"]] == ["cover.02"]
    assert "merges kept: cover.02 -> cover.01 (steps 46)" in run.runs[0].corrections
    text = "\n".join(printed)
    assert "the instances merged in S1" in text            # the --force warning
    assert "merges kept: cover.02 -> cover.01" in text

    import_logs_into_db(db, folder, tax, desktops={D2}, force=True)     # idempotent
    assert table(db) == merged
    ops = db.ops_of_kind(D2, OP_REPLAY_MERGES)
    assert len(ops) == 2 and ops[0]["payload"] == ops[1]["payload"]


def test_an_undone_merge_is_not_replayed(db, tmp_path, tax):
    report = merge_instances(db, tax, D2, "cover.02", "cover.01", "tester")
    undo_merge(db, tax, db.get_op(report.op_id))
    import_logs_into_db(db, sheets(tmp_path), tax, desktops={D2}, force=True)
    assert "cover.02" in db.instances(D2)
    assert targets(db)[46] == ["cover.02"]
    assert db.ops_of_kind(D2, OP_REPLAY_MERGES) == []


def test_a_sheet_that_no_longer_names_gone_needs_nothing(db, tmp_path, tax):
    merge_instances(db, tax, D2, "cover.02", "cover.01", "tester")
    folder = sheets(tmp_path, ("46,Front Cover", "46,Power Module Blocker"))
    run = import_logs_into_db(db, folder, tax, desktops={D2}, force=True)
    assert "cover.02" not in db.instances(D2)
    [op] = db.ops_of_kind(D2, OP_REPLAY_MERGES)
    assert [(e["gone"], e["keep"]) for e in op["payload"]["already"]] == [
        ("cover.02", "cover.01")]
    assert "merges the sheet no longer needs: cover.02 -> cover.01" in run.runs[0].corrections


def test_shift_t_corrections_follow_the_merged_key(db, tmp_path, tax):
    """Step 46 was corrected to `remove` on cover.02, then cover.02 was merged."""
    truth = TruthService(db, tax, cache_dir=None)
    key = FrameKey(D2, 48, "scan")
    db.upsert_frame(key, "x.jpg", {"hw": [64, 64]}, None)
    for target in ("cover.01", "cover.02"):
        commit_verb_fix(db, truth, key, target, "remove", "tester")
    merge_instances(db, tax, D2, "cover.02", "cover.01", "tester")
    run = import_logs_into_db(db, sheets(tmp_path), tax, desktops={D2}, force=True)
    verbs = {(a.step, a.target): a.verb for a in db.actions(D2)}
    assert verbs[(38, "cover.01")] == "remove" and verbs[(46, "cover.01")] == "remove"
    replay = run.runs[0].corrections
    assert not any("not matched" in line for line in replay), replay


# --------------------------------------------------------------------------- #
# the pure rule
# --------------------------------------------------------------------------- #
def _inst(key: str, cls: str, **kw) -> InstanceRec:
    return InstanceRec(key=key, desktop=1, cls=cls, **kw)


def test_the_rule_follows_chains_and_reports_what_does_not_fit():
    instances = {k: _inst(k, "cover") for k in ("cover.01", "cover.02", "cover.03")}
    instances["screw.01"] = _inst("screw.01", "screw", parent="cover.01")
    instances["psu.01"] = _inst("psu.01", "psu")
    actions = [ActionRec(1, 3, 0, "cover.01", "open"), ActionRec(1, 5, 0, "cover.02", "remove"),
               ActionRec(1, 7, 0, "screw.01", "unscrew")]
    merges = [
        {"ops": [1], "gone": "cover.01", "keep": "cover.02", "cls": "cover"},
        {"ops": [2], "gone": "cover.02", "keep": "cover.03", "cls": "cover"},
        {"ops": [3], "gone": "psu.01", "keep": "cover.03", "cls": "psu"},
        {"ops": [4], "gone": "cover.09", "keep": "cover.08", "cls": "cover"},
    ]
    table, out, report = replay_merges(merges, instances, actions)
    assert sorted(table) == ["cover.03", "psu.01", "screw.01"]
    assert [(a.step, a.target) for a in out] == [(3, "cover.03"), (5, "cover.03"),
                                                 (7, "screw.01")]
    assert table["screw.01"].parent == "cover.03"
    assert instances["screw.01"].parent == "cover.01"            # inputs untouched
    assert [e["gone"] for e in report.replayed] == ["cover.01", "cover.02"]
    whys = {e["gone"]: e["why"] for e in report.unmatched}
    assert "psu" in whys["psu.01"] and "neither" in whys["cover.09"]
    assert report.rename["cover.01"] == "cover.02"


def test_the_rule_refuses_two_actions_at_one_step():
    instances = {k: _inst(k, "cover") for k in ("cover.01", "cover.02")}
    actions = [ActionRec(1, 3, 0, "cover.01", "open"), ActionRec(1, 3, 1, "cover.02", "remove")]
    table, out, report = replay_merges(
        [{"ops": [1], "gone": "cover.02", "keep": "cover.01", "cls": "cover"}],
        instances, actions)
    assert sorted(table) == ["cover.01", "cover.02"] and out == actions
    assert "both an action at step 3" in report.unmatched[0]["why"]
