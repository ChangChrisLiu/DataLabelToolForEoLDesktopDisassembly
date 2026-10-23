"""Task U21: a forced re-import keeps 「其实是硬盘架」, on the real D2/D3/D4 sheets.

The live D2/D3/D4 were imported under the old name patterns, which made the
cage a latch ("remove SSD locker shield", "HDD Locker Shield", "HDD Shield
Locker"). ``old_patterns`` reproduces that: ``tda.core.logs`` parses with a
copy of ``taxonomy_map.yaml`` without the two "... locker shield" rules. Each
scene then does what the annotators did (``Shift+T`` the locker out) and what
the controller will do (the reverse ``Shift+T``, then the reclass), and is
re-imported twice -- under the new patterns, where D2/D3's sheets already say
cage, and under the old ones, where the replay makes the cage.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from tda.core import logs as L
from tda.core.db import Db
from tda.core.implied import is_implied
from tda.core.model import FrameKey
from tda.core.reclass import reclass_instance
from tda.core.reclass_replay import OP_REPLAY_RECLASS, replay_reclasses
from tda.core.taxonomy import TAXONOMY_MAP_PATH, load_taxonomy, parse_raw_name
from tda.core.truth import TruthService
from tda.core.verb_replay import OP_REPLAY
from tda.pipeline_logs import import_logs_into_db
from tda.ui.session_verb import commit_verb_fix

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
LOCKER, SHIELD, CAGE = "drive_latch.01", "drive_latch.02", "drive_cage.01"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def sheets(tmp_path: Path, desktop: int, edit=None) -> str:
    folder = tmp_path / "drive"
    folder.mkdir(exist_ok=True)
    for suffix in ("", "_meta"):
        name = f"desktop_{desktop:02d}{suffix}.csv"
        shutil.copy(FIXTURES / name, folder / name)
    if edit is not None:
        path = folder / f"desktop_{desktop:02d}.csv"
        old, new = edit
        text = path.read_text(encoding="utf-8-sig")
        assert old in text
        path.write_text(text.replace(old, new), encoding="utf-8")
    return str(folder)


@pytest.fixture
def old_patterns(monkeypatch, tmp_path):
    """The name patterns the live D2/D3/D4 were imported under (before U21)."""
    cfg = yaml.safe_load(Path(TAXONOMY_MAP_PATH).read_text(encoding="utf-8"))
    cfg["rules"] = [r for r in cfg["rules"] if not r["name"].endswith("_locker_shield")]
    path = tmp_path / "taxonomy_map_old.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(L, "parse_raw_name",
                        lambda raw, nest="": parse_raw_name(raw, nest, str(path)))
    assert L.parse_raw_name("HDD Locker Shield").cls == "drive_latch"
    return monkeypatch


def verbs(db: Db, desktop: int) -> dict[str, list]:
    out: dict[str, list] = {}
    for a in db.actions(desktop):
        out.setdefault(a.target, []).append((a.step, a.verb))
    return out


def bay(db: Db, desktop: int) -> dict[str, tuple]:
    """The drive bay's parts: class, parent, attached, implied."""
    return {k: (r.cls, r.parent, r.attached, is_implied(r))
            for k, r in db.instances(desktop).items()
            if r.cls in ("drive_latch", "drive_cage")}


def flip(db, tax, desktop, step, key, to):
    return commit_verb_fix(db, TruthService(db, tax), FrameKey(desktop, step, "scan"), key, to,
                           "tester")


def live(tmp_path, tax, desktop: int, shield_step: int, locker_step: int,
         locker: str = LOCKER, shield: str = SHIELD) -> Db:
    """The desktop as the live DB holds it: imported (old patterns), lockers out by Shift+T."""
    db = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(db, sheets(tmp_path, desktop), tax, desktops={desktop})
    for key, step in ((locker, locker_step), (shield, shield_step)):
        if key and verbs(db, desktop)[key] == [(step, "open")]:
            flip(db, tax, desktop, step, key, "remove")
    return db


def keyframe_ids(db: Db, desktop: int, key: str) -> list[int]:
    return sorted(int(r["id"]) for r in db.conn.execute(
        "SELECT id FROM shape_keyframe WHERE desktop=? AND instance=?", (desktop, key)))


def draw(db: Db, desktop: int, key: str, step: int) -> None:
    """A shape on ``key`` (the replay must keep it on the key)."""
    from reclass_scenes import mask_kf
    kf = mask_kf(key, (4, 4, 20, 20), step, "scan")
    kf.desktop = desktop
    db.add_keyframe(kf)


def reimport(db, tmp_path, tax, desktop, edit=None) -> tuple:
    printed: list[str] = []
    run = import_logs_into_db(db, sheets(tmp_path, desktop, edit), tax, desktops={desktop},
                              force=True, force_verified=True, log=printed.append)
    assert [r.status for r in run.runs] == ["imported"]
    return run.runs[0], printed


# --------------------------------------------------------------------------- #
# D2: "SSD Locker" 12, "remove SSD locker shield" 17
# --------------------------------------------------------------------------- #
@pytest.fixture
def d2(tmp_path, tax, old_patterns):
    db = live(tmp_path, tax, 2, 17, 12)
    assert bay(db, 2) == {LOCKER: ("drive_latch", None, False, False),
                          SHIELD: ("drive_latch", None, False, False)}
    assert verbs(db, 2)[LOCKER] == [(12, "remove")] and verbs(db, 2)[SHIELD] == [(17, "remove")]
    draw(db, 2, SHIELD, 16)
    # the controller: the reverse Shift+T (no cage yet: nothing to hang it on) ...
    back = flip(db, tax, 2, 12, LOCKER, "open")
    assert back["cage"] is None and verbs(db, 2)[LOCKER] == [(12, "open")]
    # ... then the reclass, whose cage rule hangs the open locker on the cage
    report = reclass_instance(db, tax, 2, SHIELD, annotator="controller")
    assert report.plan.cage["adopted"] == [LOCKER]
    assert bay(db, 2) == {LOCKER: ("drive_latch", CAGE, True, False),
                          CAGE: ("drive_cage", None, False, False)}
    yield db
    db.close()


def test_d2_new_patterns_the_sheet_already_says_cage(d2, tmp_path, tax, old_patterns):
    old_patterns.undo()                         # the importer as U21 ships it
    shapes = keyframe_ids(d2, 2, CAGE)
    first, printed = reimport(d2, tmp_path, tax, 2)
    assert bay(d2, 2) == {LOCKER: ("drive_latch", CAGE, True, False),
                          CAGE: ("drive_cage", None, False, False)}
    assert verbs(d2, 2)[CAGE] == [(17, "remove")] and verbs(d2, 2)[LOCKER] == [(12, "open")]
    assert keyframe_ids(d2, 2, CAGE) == shapes
    assert ('reclass kept: drive_cage.01 (drive_cage; step 17 "remove SSD locker shield"; '
            'the sheet already says so)') in first.corrections
    assert "cage latch adopted: drive_latch.01 is drive_cage.01's own latch (opened at step " \
           "12, never removed on its own) - parent drive_cage.01, attached" in first.cage_latches
    assert not first.dropped and not first.kept_for_review and not first.implied
    replay = d2.ops_of_kind(2, OP_REPLAY)[-1]["payload"]
    assert [(e["step"], e["target"], e["verb"]) for e in replay["agrees"]] == \
        [(12, LOCKER, "open")]
    assert replay["unmatched"] == []
    table = d2.instances(2)
    second, _printed = reimport(d2, tmp_path, tax, 2)
    assert d2.instances(2) == table
    assert second.corrections == first.corrections
    assert len(d2.ops_of_kind(2, OP_REPLAY_RECLASS)) == 2


def test_d2_old_patterns_the_replay_makes_the_cage(d2, tmp_path, tax):
    first, printed = reimport(d2, tmp_path, tax, 2)
    assert bay(d2, 2) == {LOCKER: ("drive_latch", CAGE, True, False),
                          CAGE: ("drive_cage", None, False, False)}
    assert ('reclass kept: drive_latch.02 -> drive_cage.01 (drive_cage; step 17 '
            '"remove SSD locker shield")') in first.corrections
    assert any(line.startswith("[import-logs]   reclass kept: drive_latch.02 -> drive_cage.01")
               for line in printed)
    assert not first.dropped and not first.kept_for_review
    table = d2.instances(2)
    second, _printed = reimport(d2, tmp_path, tax, 2)
    assert d2.instances(2) == table and second.corrections == first.corrections


# --------------------------------------------------------------------------- #
# D3: "HDD Locker" 10, "HDD Locker Shield" 11 (both "open" in the sheet)
# --------------------------------------------------------------------------- #
@pytest.fixture
def d3(tmp_path, tax, old_patterns):
    db = live(tmp_path, tax, 3, 11, 10)
    assert verbs(db, 3)[LOCKER] == [(10, "remove")] and verbs(db, 3)[SHIELD] == [(11, "remove")]
    flip(db, tax, 3, 10, LOCKER, "open")
    reclass_instance(db, tax, 3, SHIELD, annotator="controller")
    assert bay(db, 3) == {LOCKER: ("drive_latch", CAGE, True, False),
                          CAGE: ("drive_cage", None, False, False)}
    yield db
    db.close()


def test_d3_new_patterns_both_corrections_agree(d3, tmp_path, tax, old_patterns):
    old_patterns.undo()
    first, _printed = reimport(d3, tmp_path, tax, 3)
    assert bay(d3, 3) == {LOCKER: ("drive_latch", CAGE, True, False),
                          CAGE: ("drive_cage", None, False, False)}
    replay = d3.ops_of_kind(3, OP_REPLAY)[-1]["payload"]
    # the Shift+T made on drive_latch.02 at 11 follows it to the cage
    assert [(e["step"], e["target"], e["verb"]) for e in replay["agrees"]] == \
        [(10, LOCKER, "open"), (11, CAGE, "remove")]
    assert replay["unmatched"] == [] and replay["kept"] == []
    assert ('reclass kept: drive_cage.01 (drive_cage; step 11 "HDD Locker Shield"; '
            'the sheet already says so)') in first.corrections
    table = d3.instances(3)
    reimport(d3, tmp_path, tax, 3)
    assert d3.instances(3) == table


def test_d3_old_patterns_the_cage_takes_the_verb_it_had(d3, tmp_path, tax):
    first, _printed = reimport(d3, tmp_path, tax, 3)
    assert verbs(d3, 3)[CAGE] == [(11, "remove")]
    assert ('reclass kept: drive_latch.02 -> drive_cage.01 (drive_cage; step 11 '
            '"HDD Locker Shield"; step 11 open -> remove)') in first.corrections
    replay = d3.ops_of_kind(3, OP_REPLAY)[-1]["payload"]
    assert (11, CAGE, "remove") in [(e["step"], e["target"], e["verb"])
                                    for e in replay["agrees"]]
    table = d3.instances(3)
    reimport(d3, tmp_path, tax, 3)
    assert d3.instances(3) == table


# --------------------------------------------------------------------------- #
# D4: "HDD Shield Locker" 13 -- the cage and its latch drawn as one; a latch by name
# --------------------------------------------------------------------------- #
@pytest.fixture
def d4(tmp_path, tax):
    db = live(tmp_path, tax, 4, shield_step=0, locker_step=13, shield="")
    assert verbs(db, 4)[LOCKER] == [(13, "remove")]
    draw(db, 4, LOCKER, 12)
    report = reclass_instance(db, tax, 4, LOCKER, annotator="controller")
    assert report.plan.new == CAGE
    assert bay(db, 4) == {CAGE: ("drive_cage", None, False, False),
                          "drive_latch.02": ("drive_latch", CAGE, True, True)}
    assert report.plan.cage["opens"] == 12
    yield db
    db.close()


def test_d4_the_replay_makes_the_cage_and_keeps_its_implied_latch(d4, tmp_path, tax):
    shapes = keyframe_ids(d4, 4, CAGE)
    latch = d4.instances(4)["drive_latch.02"]
    first, _printed = reimport(d4, tmp_path, tax, 4)
    assert bay(d4, 4) == {CAGE: ("drive_cage", None, False, False),
                          "drive_latch.02": ("drive_latch", CAGE, True, True)}
    assert d4.instances(4)["drive_latch.02"] == latch
    assert verbs(d4, 4)[CAGE] == [(13, "remove")] and keyframe_ids(d4, 4, CAGE) == shapes
    assert ('reclass kept: drive_latch.01 -> drive_cage.01 (drive_cage; step 13 '
            '"HDD Shield Locker"; step 13 open -> remove)') in first.corrections
    assert "cage latch kept: drive_latch.02 (implied) is drive_cage.01's own latch" in \
        first.cage_latches
    replay = d4.ops_of_kind(4, OP_REPLAY)[-1]["payload"]
    assert [(e["step"], e["target"]) for e in replay["agrees"]] == [(13, CAGE)]
    assert not first.dropped and not first.kept_for_review and not first.implied
    table = d4.instances(4)
    second, _printed = reimport(d4, tmp_path, tax, 4)
    assert d4.instances(4) == table and second.corrections == first.corrections


def test_d4_a_sheet_that_no_longer_names_it_is_reported_not_applied(d4, tmp_path, tax):
    run, printed = reimport(d4, tmp_path, tax, 4, edit=("13,HDD Shield Locker", "13,HDD bracket"))
    [line] = [x for x in run.corrections if x.startswith("reclass")]
    assert line == ('reclass not matched: drive_latch.01 -> drive_cage.01 (the sheet names no '
                    'part step 13 "HDD Shield Locker")')
    assert any("reclass not matched" in x for x in printed)
    assert CAGE in d4.instances(4)                      # held: it carries shapes


# --------------------------------------------------------------------------- #
# the pure replay
# --------------------------------------------------------------------------- #
def test_the_replay_swaps_keys_the_importer_gave_the_other_way(tax):
    from tda.core.model import ActionRec, InstanceRec

    parsed = {"drive_cage.01": InstanceRec("drive_cage.01", 9, "drive_cage",
                                           raw_names=["SSD locker shield"]),
              "drive_cage.02": InstanceRec("drive_cage.02", 9, "drive_cage",
                                           raw_names=["SSD shield"]),
              "screw.drive.01": InstanceRec("screw.drive.01", 9, "screw",
                                            fastens="drive_cage.02")}
    actions = [ActionRec(9, 3, 0, "drive_cage.01", "remove"),
               ActionRec(9, 8, 0, "drive_cage.02", "remove")]
    entry = {"op": 5, "old": "drive_latch.02", "new": "drive_cage.02",
             "old_cls": "drive_latch", "new_cls": "drive_cage", "steps": [3],
             "raw_names": ["SSD locker shield"], "verbs": [[3, "remove"]], "extra": False}
    table, out, report = replay_reclasses([entry], parsed, actions, tax)
    assert table["drive_cage.02"].raw_names == ["SSD locker shield"]
    assert table["drive_cage.01"].raw_names == ["SSD shield"]
    assert table["screw.drive.01"].fastens == "drive_cage.01"     # the pointer followed
    assert [(a.step, a.target) for a in out] == [(3, "drive_cage.02"), (8, "drive_cage.01")]
    assert [e["parsed"] for e in report.kept] == ["drive_cage.01"]
    # pure: the inputs are left alone
    assert parsed["drive_cage.01"].raw_names == ["SSD locker shield"]
    # idempotent on its own result
    again, out2, report2 = replay_reclasses([entry], table, out, tax)
    assert again == table and out2 == out and [e["new"] for e in report2.already] == \
        ["drive_cage.02"]
