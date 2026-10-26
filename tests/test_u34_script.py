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


# --------------------------------------------------------------------------- #
# round 2
# --------------------------------------------------------------------------- #
def test_a_cp1252_pipe_does_not_stop_it(source, tmp_path):
    """N1: piped or redirected on Windows, stdout is cp1252 -- the Chinese lines must print."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--desktop", str(DESKTOP), "--instance", DRIVE,
         "--db", source, "--out", str(tmp_path / "out")],
        capture_output=True, env=env, timeout=600)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    text = done.stdout.decode("utf-8")
    assert f"D{DESKTOP:02d} {DRIVE} (这台没有这个零件)" in text and "wrote op" in text


def test_residues_are_listed_and_a_cable_label_refuses(source, tmp_path, capsys):
    """N5: listed before writing; a 「被线挡住」 reason is the app's refusal too."""
    from tda.core.model import FrameKey, FrameOverride

    db = Db(source)
    try:
        db.set_frame_override(FrameOverride(FrameKey(DESKTOP, 2, "scan"), DRIVE,
                                            visibility="occluded_partial"))
    finally:
        db.close()
    script = _script()
    args = ["--desktop", str(DESKTOP), "--instance", DRIVE, "--db", source,
            "--out", str(tmp_path / "out")]
    assert script.main(args) == 0
    text = capsys.readouterr().out
    assert "still naming it (kept as they are):" in text
    assert "    - 单帧可见性标签 1 处：scan 第 2 帧" in text
    db = Db(source)
    try:
        db.set_frame_override(FrameOverride(FrameKey(DESKTOP, 3, "scan"), DRIVE,
                                            visibility="occluded_full", occluder="cable"))
    finally:
        db.close()
    assert script.main(args) == 1
    text = capsys.readouterr().out
    assert "refused:" in text and "标着「被线挡住」" in text and "nothing written" in text


def test_apply_says_to_run_check(source, tmp_path, capsys):
    """N7: the part's old compiled rows go with the next refresh: say which command."""
    script = _script()
    assert script.main(["--desktop", str(DESKTOP), "--instance", DRIVE, "--db", source,
                        "--out", str(tmp_path / "out"), "--apply"]) == 0
    assert (f"run: python -m tda.cli check --desktop {DESKTOP} --views scan,oak1,oak2,rs"
            in capsys.readouterr().out)


def test_without_psutil_the_live_database_is_refused(source, tmp_path, monkeypatch):
    """N6: fail-closed -- nobody can tell whether the app is open."""
    import sys

    script = _script()
    monkeypatch.setitem(sys.modules, "psutil", None)      # import psutil -> ImportError
    assert script.app_running() is True
    monkeypatch.setattr(script, "LIVE", source)
    before = _sha(source)
    assert script.main(["--desktop", str(DESKTOP), "--instance", DRIVE, "--db", source,
                        "--out", str(tmp_path / "out"), "--apply"]) == 2
    assert _sha(source) == before


def test_an_abandoned_lock_taken_over_is_put_back(source, tmp_path):
    """N6: only a lock this run made is removed; one it took over is left as it was."""
    from tda.core.db_backup import lock_path_for

    script = _script()
    args = ["--desktop", str(DESKTOP), "--instance", DRIVE, "--db", source,
            "--out", str(tmp_path / "out"), "--apply"]
    lock = lock_path_for(source)
    stale = '{"annotator": "chang", "ts": "2026-01-01T00:00:00+00:00"}'.encode("utf-8")
    lock.write_bytes(stale)
    assert script.main(args) == 0
    assert lock.read_bytes() == stale                    # as it was, not deleted
    lock.unlink()
    assert script.main([*args, "--present"]) == 0
    assert not lock.exists()                             # its own: removed


def test_a_lock_in_its_own_name_is_removed_and_a_fresh_foreign_one_refuses(source, tmp_path):
    """R2-N1: a killed run's leftover is never put back (it would lock the annotator
    out for the TTL); somebody else's fresh lock refuses and is not touched."""
    import json as _json

    from tda.core.dbrows import now_iso
    from tda.core.db_backup import lock_path_for

    script = _script()
    args = ["--desktop", str(DESKTOP), "--instance", DRIVE, "--db", source,
            "--out", str(tmp_path / "out"), "--apply"]
    lock = lock_path_for(source)
    # a fresh leftover in the script's own name: taken, then removed
    lock.write_text(_json.dumps({"annotator": script.ANNOTATOR, "ts": now_iso()}),
                    encoding="utf-8")
    assert script.main(args) == 0
    assert not lock.exists()
    # the annotator's fresh lock: refused, nothing written, the lock as it was
    fresh = _json.dumps({"annotator": "chang", "ts": now_iso()}).encode("utf-8")
    lock.write_bytes(fresh)
    before = _sha(source)
    assert script.main([*args, "--present"]) == 2
    assert lock.read_bytes() == fresh and _sha(source) == before
    db = Db(source)
    try:
        assert AB.is_absent(db, DESKTOP, DRIVE)          # the first run's write stands
    finally:
        db.close()
