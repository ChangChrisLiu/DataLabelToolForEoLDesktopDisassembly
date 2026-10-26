"""Task U34: ``scripts/absent_from_start.py`` -- the same op, through the same path.

A dry run (the default) works on a backup-API copy and leaves ``--db`` byte for
byte as it was; ``--apply`` writes ``--db`` itself; ``--present`` is the inverse.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from absent_scene import DRIVE, add_drive
from session_scene import DESKTOP, seed_db
from tda.core import absent as AB
from tda.core.db import Db
from tda.core.taxonomy import load_taxonomy

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "absent_from_start.py"


def _script():
    spec = importlib.util.spec_from_file_location("absent_from_start", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path: Path) -> str:
    tax = load_taxonomy()
    path = tmp_path / "src.sqlite"
    db = Db(str(path))
    seed_db(db, tax, tmp_path / "cache", last_step=14)
    add_drive(db, tax, DESKTOP)
    db.close()
    return str(path)


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_a_dry_run_leaves_the_database_alone(source, tmp_path, capsys):
    before = _sha(source)
    out = tmp_path / "out"
    code = _script().main(["--desktop", str(DESKTOP), "--instance", DRIVE, "--db", source,
                           "--out", str(out)])
    assert code == 0 and _sha(source) == before
    text = capsys.readouterr().out
    assert "DRY RUN" in text and "wrote op" in text
    assert "14/14 frames ask for its geometry" in text
    assert "0/14 frames ask for its geometry" in text
    [summary] = out.glob("absent_summary_dryrun_*.json")
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["before"]["absent"] is False and data["after"]["absent"] is True
    assert data["after"]["state_first"] == "step 1: removed / elsewhere"
    assert data["after"]["derived_events"] == 0 and data["after"]["warnings"] == []
    db = Db(source)
    try:
        assert not AB.is_absent(db, DESKTOP, DRIVE)
    finally:
        db.close()


def test_apply_writes_and_present_takes_it_back(source, tmp_path, monkeypatch):
    script = _script()
    args = ["--desktop", str(DESKTOP), "--instance", DRIVE, "--db", source,
            "--out", str(tmp_path / "out"), "--apply"]
    live = script.LIVE
    monkeypatch.setattr(script, "LIVE", source)       # as if it were the live database
    monkeypatch.setattr(script, "app_running", lambda: True)
    before = _sha(source)
    assert script.main(args) == 2 and _sha(source) == before   # the app is open: no
    monkeypatch.setattr(script, "LIVE", live)         # a copy: the app does not matter
    assert script.main(args) == 0
    db = Db(source)
    try:
        assert AB.is_absent(db, DESKTOP, DRIVE)
        assert db.ops_of_kind(DESKTOP, AB.OP_KIND)[-1]["annotator"] == "controller"
    finally:
        db.close()
    assert script.main(args) == 1             # refused: it already is
    assert script.main([*args, "--present"]) == 0
    db = Db(source)
    try:
        assert not AB.is_absent(db, DESKTOP, DRIVE)
    finally:
        db.close()
