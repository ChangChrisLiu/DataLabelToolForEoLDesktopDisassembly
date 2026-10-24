"""Task U27 round 2: a screw's ``captive`` and its ``parent`` / ``attached`` come back together.

The review's blocker (B1): the captive replay (U14) found its screws by key,
the relation replay by key or raw name, and the ``Captive`` column of Steps
wrote no ``captive_fix`` row at all -- so a forced re-import could leave a
captive screw unattached, loosened in the chassis for ever, while both replays
said "kept". Now (A) a screw's ``parent`` / ``attached`` fix records its
``captive`` and is refused when the screw comes back the other kind, (B) the
captive replay finds the screw with :func:`locate_part` as well, and (C) the
``Captive`` cell writes the ``captive_fix`` row ``Shift+T`` writes. The two
triggers the reviewer built are here on the real D2 sheet, with the smaller
rulings: N4 (an implied cage latch made again under a new key), N5
(``attached`` without its ``parent``), N6 / N7 (``adopt-relations --keys`` and
the backup).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
from pathlib import Path

import pytest
import yaml

from relation_scenes import FIXTURES, D2, apply_steps, fixes, imported, reimport, replayed
from tda.cli import EXIT_ERROR, EXIT_OK, main
from tda.core.captive_replay import OP_CAPTIVE_FIX, OP_REPLAY_CAPTIVE
from tda.core.db import Db
from tda.core.model import FrameKey, InstanceRec
from tda.core.relation_replay import replay_relations
from tda.core.states import validate_events
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import events_of, state_of
from tda.pipeline_logs import import_logs_into_db
from tda.ui.session_captive import commit_captive_fix

KEY = FrameKey(D2, 48, "scan")
FLIPPED = "screw.cpu_cooler.02"              # "CPU fan screw 2", captive by the guess
RENUMBER = [("3,CPU fan screw 1,", "3,Optical drive screw 9,")]
LAST = 48


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def d2(tmp_path, tax):
    db = imported(tmp_path, tax, D2)
    yield db
    db.close()


def screws_by_name(db: Db, tax) -> dict[tuple[str, int], tuple]:
    """``(raw name, step) -> (state, placement, left_with)`` of every CPU fan screw."""
    table = db.instances(D2)
    names = {key: rec.raw_names[0] for key, rec in table.items()
             if rec.cls == "screw" and rec.raw_names
             and rec.raw_names[0].startswith("CPU fan screw")}
    out = {}
    for step in range(1, LAST + 1):
        held = state_of(db, tax, D2, step)
        for key, name in names.items():
            inst = held[key]
            out[(name, step)] = (inst.state, inst.placement, inst.left_with)
    return out


def loosened_at_the_end(db: Db, tax) -> list[str]:
    held = state_of(db, tax, D2, LAST)
    return sorted(key for key, rec in db.instances(D2).items()
                  if rec.cls == "screw" and held[key].state == "loosened")


# --------------------------------------------------------------------------- #
# B1 (ii): Shift+T, then the sheet renumbers the screws
# --------------------------------------------------------------------------- #
def test_a_flip_and_its_attached_follow_a_renumbered_sheet_onto_one_screw(d2, tmp_path, tax):
    commit_captive_fix(d2, TruthService(d2, tax), KEY, FLIPPED, None, "tester")
    flipped = d2.instances(D2)[FLIPPED]
    assert (flipped.attrs["captive"], flipped.attached) == (False, False)
    before = screws_by_name(d2, tax)
    run = reimport(d2, tmp_path, tax, D2, edits=RENUMBER)
    table = d2.instances(D2)
    # "CPU fan screw 2" is screw.cpu_cooler.01 now: both replays found it there
    moved = table["screw.cpu_cooler.01"]
    assert moved.raw_names == ["CPU fan screw 2"]
    assert (moved.attrs["captive"], moved.attrs["captive_source"]) == (False, "manual")
    assert (moved.parent, moved.attached) == (None, False)
    # and the screw now under the old key keeps the sheet's guess
    other = table[FLIPPED]
    assert other.raw_names == ["CPU fan screw 3"]
    assert (other.attrs["captive"], other.parent, other.attached) == \
        (True, "cpu_cooler.fan.01", True)
    # every fan screw the sheet still names does exactly what it did before
    after = screws_by_name(d2, tax)
    assert {k: v for k, v in after.items() if k[0] != "CPU fan screw 1"} == \
        {k: v for k, v in before.items() if k[0] != "CPU fan screw 1"}
    assert loosened_at_the_end(d2, tax) == []
    assert ("captive fixes (Shift+T) kept 1: screw.cpu_cooler.01 (was screw.cpu_cooler.02, "
            "found by its name) not captive (sheet: captive)") in run.corrections
    done = replayed(d2, D2)
    assert done["unmatched"] == []
    assert [(e.get("parsed"), e["field"]) for e in done["agrees"]] == \
        [("screw.cpu_cooler.01", "attached")]
    assert not validate_events(table, events_of(d2, tax, D2), tax)
    # and once more: nothing moves
    reimport(d2, tmp_path, tax, D2, edits=RENUMBER)
    assert d2.instances(D2) == table


# --------------------------------------------------------------------------- #
# B1 (i): Captive and Attached unticked in Steps' instance table
# --------------------------------------------------------------------------- #
def test_the_captive_cell_is_a_captive_fix_and_survives_with_its_attached(d2, tmp_path, tax):
    apply_steps(d2, tax, D2, (FLIPPED, "captive", False), (FLIPPED, "attached", False))
    [op] = d2.ops_of_kind(D2, OP_CAPTIVE_FIX)
    p, inv = op["payload"], op["inverse"]
    assert (p["target"], p["captive"], p["captive_source"], p["attached"], p["source"]) == \
        (FLIPPED, False, "manual", False, "steps")
    assert (p["step"], p["view"], op["annotator"]) == (4, "-", "tester")
    assert (inv["captive"], inv["captive_source"], inv["attached"]) == (True, "heuristic", True)
    [(key, name, old, new, by)] = fixes(d2, D2)
    assert (key, name, old, new) == (FLIPPED, "attached", True, False)
    assert d2.ops_of_kind(D2, "relation_fix")[0]["payload"]["captive"] is False
    before = screws_by_name(d2, tax)
    assert before[("CPU fan screw 2", 4)][0] == "removed"
    for _ in range(2):
        run = reimport(d2, tmp_path, tax, D2)
        assert screws_by_name(d2, tax) == before
        rec = d2.instances(D2)[FLIPPED]
        assert (rec.attrs["captive"], rec.attrs["captive_source"], rec.attached) == \
            (False, "manual", False)
        assert "captive fixes (Shift+T) kept 1: screw.cpu_cooler.02 not captive " \
               "(sheet: captive)" in run.corrections
    assert loosened_at_the_end(d2, tax) == []


def test_without_its_captive_row_the_attached_fix_is_refused_not_half_applied(
        d2, tmp_path, tax):
    """The pre-round-2 stack: the Captive cell wrote no row. A keeps it whole (as main did)."""
    fresh = screws_by_name(d2, tax)                     # the import, before any edit
    apply_steps(d2, tax, D2, (FLIPPED, "captive", False), (FLIPPED, "attached", False))
    with d2.transaction():
        d2.conn.execute("DELETE FROM op_log WHERE desktop=? AND kind=?", (D2, OP_CAPTIVE_FIX))
    run = reimport(d2, tmp_path, tax, D2)
    rec = d2.instances(D2)[FLIPPED]
    assert (rec.attrs["captive"], rec.parent, rec.attached) == (True, "cpu_cooler.fan.01", True)
    assert screws_by_name(d2, tax) == fresh             # leaves with the fan, as on main
    assert loosened_at_the_end(d2, tax) == []
    [miss] = replayed(d2, D2)["unmatched"]
    assert miss["why"] == ("captive differs: screw.cpu_cooler.02 is captive now, the fix was "
                           "set while it was not captive")
    assert any(line.startswith("relation fix not matched: screw.cpu_cooler.02.attached = False "
                               "(captive differs") for line in run.corrections)


def test_shift_t_writes_one_captive_row_not_two(d2, tax):
    commit_captive_fix(d2, TruthService(d2, tax), KEY, FLIPPED, None, "tester")
    assert len(d2.ops_of_kind(D2, OP_CAPTIVE_FIX)) == 1


# --------------------------------------------------------------------------- #
# the pure checks (A, N5, N4)
# --------------------------------------------------------------------------- #
def screw(key, captive, **kw):
    return InstanceRec(key, 1, "screw", attrs={"captive": captive}, raw_names=[key], **kw)


def fix(op_id, key, name, new, **extra):
    return {"id": op_id, "payload": {"key": key, "field": name, "new": new, "raw_name": key,
                                     **extra}}


def test_a_screw_fix_needs_the_captive_it_was_made_with():
    fresh = {"s.01": screw("s.01", True, parent="fan.01", attached=True),
             "s.02": screw("s.02", False), "fan.01": InstanceRec("fan.01", 1, "cpu_cooler")}
    ops = [fix(1, "s.01", "attached", False, captive=False),     # made on a loose screw
           fix(2, "s.02", "attached", False, captive=False),     # still loose: fine
           fix(3, "s.01", "group_id", "g", captive=False)]       # no captive meaning
    out, report = replay_relations(ops, fresh)
    assert out["s.01"].attached is True and out["s.01"].group_id == "g"
    [miss] = report.unmatched
    assert miss["why"] == "captive differs: s.01 is captive now, the fix was set while it was " \
                          "not captive"
    assert [e["key"] for e in report.agrees] == ["s.02"]
    # a fix recorded before round 2 carries no captive: applied as it is
    out, report = replay_relations([fix(1, "s.01", "attached", False)], fresh)
    assert out["s.01"].attached is False


def test_attached_without_its_parent_is_not_applied():
    fresh = {"clip.01": InstanceRec("clip.01", 1, "cable_clip", raw_names=["clip"])}
    ops = [fix(1, "clip.01", "parent", "cover.09", new_raw_name="gone"),
           fix(2, "clip.01", "attached", True)]
    out, report = replay_relations(ops, fresh)
    assert (out["clip.01"].parent, out["clip.01"].attached) == (None, False)
    assert [e["field"] for e in report.unmatched] == ["parent", "attached"]
    assert report.unmatched[1]["why"].startswith("not applied (no parent)")
    # unticking needs no parent
    out, report = replay_relations([ops[0], fix(2, "clip.01", "attached", False)],
                                   {"clip.01": InstanceRec("clip.01", 1, "cable_clip",
                                                           raw_names=["clip"], attached=True)})
    assert out["clip.01"].attached is False


def test_an_implied_cage_latch_made_again_is_said_to_be_rebuilt(tmp_path, tax):
    """D5 with its locker row renamed: the cage's latch is implied; re-parented in Steps (N4)."""
    edits = [("10,HDD shield locker", "10,Front fan")]
    db = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(db, _sheets(tmp_path, 5, edits), tax, desktops={5})
    apply_steps(db, tax, 5, ("drive_latch.01", "parent", ""),
                ("drive_latch.01", "removal_direction", "+Z"))
    run = reimport(db, tmp_path, tax, 5, edits=edits)
    assert "drive_latch.02" in db.instances(5) and "drive_latch.01" not in db.instances(5)
    lines = [x for x in run.corrections if x.startswith("relation fix not matched")]
    assert lines and all("(rebuilt as drive_latch.02: the import made the drive cage's "
                         "implied latch again, under a new key)" in x for x in lines)
    db.close()


def _sheets(tmp_path: Path, desktop: int, edits) -> str:
    from relation_scenes import sheets
    return sheets(tmp_path, desktop, edits)


# --------------------------------------------------------------------------- #
# adopt-relations: --keys and the backup (N6, N7)
# --------------------------------------------------------------------------- #
@pytest.fixture
def env(tmp_path):
    drive = tmp_path / "raw_logs" / "drive"
    drive.mkdir(parents=True)
    for suffix in ("", "_meta"):
        shutil.copy(FIXTURES / f"desktop_23{suffix}.csv", drive / f"desktop_23{suffix}.csv")
    (tmp_path / "backups").mkdir()
    cfg = {"cache_dir": str(tmp_path / "cache"), "db_path": str(tmp_path / "tda.sqlite"),
           "backup_dir": str(tmp_path / "backups"), "raw_logs_dir": str(tmp_path / "raw_logs"),
           "oak_root": str(tmp_path / "src"), "scanner_root": str(tmp_path / "src"),
           "rs_root": str(tmp_path / "src"), "weights_dir": str(tmp_path / "weights")}
    paths = tmp_path / "paths.yaml"
    paths.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return {"paths": str(paths), "backups": tmp_path / "backups"}


def cli(env, *argv) -> int:
    return main(["--paths", env["paths"], *argv])


def test_adopt_refuses_unknown_keys_and_backs_up_before_writing(env, capsys):
    assert cli(env, "import-logs", "--desktops", "23") == EXIT_OK
    capsys.readouterr()
    assert cli(env, "adopt-relations", "--desktop", "23",
               "--keys", "cable_clip.01,cable_clip.09,cover.07") == EXIT_ERROR
    assert "D23 has no instance cable_clip.09, cover.07; nothing was compared" in \
        capsys.readouterr().out
    assert cli(env, "adopt-relations", "--desktop", "23", "--dry-run") == EXIT_OK
    assert list(env["backups"].glob("tda_*.sqlite")) == []          # a dry run writes nothing
    assert cli(env, "adopt-relations", "--desktop", "23", "--keys", "cable_clip.01") == EXIT_OK
    out = capsys.readouterr().out
    assert "backed the database up first" in out
    assert len(list(env["backups"].glob("tda_*.sqlite"))) == 1


def test_the_force_warning_no_longer_loses_the_captive_column(d2, tmp_path, tax):
    printed: list[str] = []
    reimport(d2, tmp_path, tax, D2, log=printed)
    text = "\n".join(printed)
    assert "the captive flags flipped with Shift+T or in Steps' Captive column where it still " \
           "has that screw" in text
    assert "a screw head set by hand" in text
    assert len(d2.ops_of_kind(D2, OP_REPLAY_CAPTIVE)) == 0
