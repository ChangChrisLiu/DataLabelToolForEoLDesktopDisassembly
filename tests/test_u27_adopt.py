"""Task U27: ``adopt-relations`` makes the fields set by hand before U27 durable.

A relational field written before there was a ``relation_fix`` row -- in Steps
before U27, or by hand in the database -- is overwritten by ``import-logs
--force``. ``adopt-relations --desktop N`` rehearses that re-import (rolled
back), compares, and records one ``relation_fix`` row per difference;
``--dry-run`` only lists them. The scene is the real D23 sheet.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
from pathlib import Path

import pytest
import yaml

from merge_scenes import everything
from relation_scenes import CLIP, D23, FIXTURES, SHROUD, fixes, imported, reimport
from tda.cli import EXIT_ERROR, EXIT_OK, main
from tda.cli_relations import ADOPT_ANNOTATOR, adopt_relations, sheet_of
from tda.core.db import Db
from tda.core.model import InstanceRec
from tda.core.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def hand_edit(db: Db, key: str, **fields) -> None:
    """A field written with no op, as Steps did before U27."""
    rec = db.instances(D23)[key]
    for name, value in fields.items():
        setattr(rec, name, value)
    db.upsert_instance(rec)


@pytest.fixture
def d23(tmp_path, tax):
    db = imported(tmp_path, tax, D23)
    hand_edit(db, CLIP, parent=SHROUD, attached=True)
    yield db
    db.close()


def test_a_dry_run_lists_the_differences_and_writes_nothing(d23, tmp_path, tax):
    before = everything(d23)
    found = adopt_relations(d23, tax, D23, sheet_of(tmp_path / "drive", D23), dry_run=True)
    assert found.differences == [(CLIP, "parent", None, SHROUD), (CLIP, "attached", False, True)]
    assert found.recorded == 0 and found.vanished == []
    assert everything(d23) == before                  # the rehearsal rolled back, op_log too


def test_adopting_records_one_fix_per_difference_and_the_re_import_keeps_them(
        d23, tmp_path, tax):
    path = sheet_of(tmp_path / "drive", D23)
    found = adopt_relations(d23, tax, D23, path)
    assert found.recorded == 2 and found.remaining == []
    assert fixes(d23, D23) == [(CLIP, "parent", None, SHROUD, ADOPT_ANNOTATOR),
                               (CLIP, "attached", False, True, ADOPT_ANNOTATOR)]
    op = d23.ops_of_kind(D23, "relation_fix")[0]["payload"]
    assert (op["raw_name"], op["new_raw_name"]) == \
        ("remove the cable from the case fan clip", "CPU fan cover")
    # a second run finds nothing: the rehearsal already replays the fixes
    again = adopt_relations(d23, tax, D23, path)
    assert again.differences == [] and again.recorded == 0
    reimport(d23, tmp_path, tax, D23)
    clip = d23.instances(D23)[CLIP]
    assert (clip.parent, clip.attached) == (SHROUD, True)


def test_without_adopting_the_hand_edit_is_lost(d23, tmp_path, tax):
    reimport(d23, tmp_path, tax, D23)
    assert d23.instances(D23)[CLIP].parent is None


def test_keys_narrow_it_and_a_fix_the_replay_refuses_is_said_to_be_still_lost(
        d23, tmp_path, tax):
    path = sheet_of(tmp_path / "drive", D23)
    d23.upsert_instance(InstanceRec("cover.07", D23, "cover", raw_names=["Old shroud"]))
    hand_edit(d23, "cpu_cooler.heatsink.01", removal_direction="+Z")
    hand_edit(d23, CLIP, parent="cover.07")
    found = adopt_relations(d23, tax, D23, path, keys={CLIP})
    assert [(k, f) for k, f, _old, _new in found.differences] == [(CLIP, "parent"),
                                                                 (CLIP, "attached")]
    assert found.vanished == []
    assert found.remaining == [
        'cable_clip.01.parent = cover.07 (it names cover.07: the sheet no longer has '
        'cover.07 nor a part named "Old shroud")',
        'cable_clip.01.attached = True (not applied (no parent): the parent fix of this '
        'part was not matched)']
    # the stray cover itself is no part of the sheet: listed, nothing to adopt on it
    rest = adopt_relations(d23, tax, D23, path, dry_run=True)
    assert rest.vanished == ["cover.07"]
    # the clip's two fixes are recorded but refused, so they still differ (round 2, N5)
    assert [(k, f) for k, f, _old, _new in rest.differences] == [
        (CLIP, "parent"), (CLIP, "attached"), ("cpu_cooler.heatsink.01", "removal_direction")]


# --------------------------------------------------------------------------- #
# the command
# --------------------------------------------------------------------------- #
@pytest.fixture
def env(tmp_path, tax):
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
    return {"paths": str(paths), "db": cfg["db_path"]}


def cli(env, *argv) -> int:
    return main(["--paths", env["paths"], *argv])


def test_the_command_dry_run_then_write(env, capsys):
    assert cli(env, "import-logs", "--desktops", "23") == EXIT_OK
    db = Db(env["db"])
    hand_edit(db, CLIP, parent=SHROUD, attached=True)
    db.close()
    capsys.readouterr()
    assert cli(env, "adopt-relations", "--desktop", "23", "--dry-run") == EXIT_OK
    out = capsys.readouterr().out
    assert ("[adopt-relations] (dry run) D23: 2 relational field(s) differ from what a forced "
            "re-import of desktop_23.csv would make:") in out
    assert "cable_clip.01.parent = cover.01 (a forced re-import would make it none)" in out
    assert "cable_clip.01.attached = True (a forced re-import would make it False)" in out
    assert "nothing recorded; run without --dry-run" in out
    db = Db(env["db"])
    assert fixes(db, D23) == []
    db.close()
    assert cli(env, "adopt-relations", "--desktop", "23") == EXIT_OK
    out = capsys.readouterr().out
    assert ("[adopt-relations] D23: recorded 2 relation_fix op(s); a forced re-import now "
            "keeps 2 of them") in out
    assert cli(env, "adopt-relations", "--desktop", "23") == EXIT_OK
    assert "nothing to adopt" in capsys.readouterr().out
    assert cli(env, "import-logs", "--desktops", "23", "--force") == EXIT_OK
    out = capsys.readouterr().out
    assert "relation fixes kept: cable_clip.01.parent = cover.01 (sheet: none)" in out
    db = Db(env["db"])
    assert db.instances(D23)[CLIP].parent == SHROUD
    db.close()


def test_the_command_refuses_a_desktop_without_steps_or_sheet(env, capsys, tmp_path):
    assert cli(env, "adopt-relations", "--desktop", "23") == EXIT_ERROR
    assert "D23 has no steps" in capsys.readouterr().out
    assert cli(env, "import-logs", "--desktops", "23") == EXIT_OK
    for name in Path(tmp_path / "raw_logs" / "drive").glob("desktop_23*"):
        name.unlink()
    capsys.readouterr()
    assert cli(env, "adopt-relations", "--desktop", "23") == EXIT_ERROR
    assert "no sheet for D23" in capsys.readouterr().out
