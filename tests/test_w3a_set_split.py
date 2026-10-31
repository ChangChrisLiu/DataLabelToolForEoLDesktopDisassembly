"""``cli set-split``: the platform-disjoint train / test split into ``desktop.split`` (task W3a, 2.4)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml

from tda.cli import EXIT_ERROR, EXIT_OK, main
from tda.core.db import Db
from tda.core.model import StepRec


@pytest.fixture
def env(tmp_path: Path) -> dict:
    """Five desktops: 1-4 with steps, 5 without (a machine nobody has imported yet)."""
    db_path = tmp_path / "tda.sqlite"
    cfg = {"cache_dir": str(tmp_path / "cache"), "db_path": str(db_path),
           "backup_dir": str(tmp_path / "backups"), "raw_logs_dir": str(tmp_path / "raw")}
    (tmp_path / "backups").mkdir()
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    db = Db(str(db_path))
    try:
        for d in range(1, 6):
            db.upsert_desktop(d, {"brand": "Dell", "model_family": f"f{d}"})
            if d < 5:
                db.replace_steps(d, [StepRec(d, 1, "initial", "Initial conditions")], [])
    finally:
        db.close()
    return {"paths": str(tmp_path / "paths.yaml"), "db": db_path, "tmp": tmp_path}


@pytest.fixture(autouse=True)
def no_app(monkeypatch):
    monkeypatch.setattr("tda.cli_split.app_running", lambda: False)


def splits(path: Path) -> dict[int, object]:
    conn = sqlite3.connect(str(path))
    try:
        return dict(conn.execute("SELECT id, split FROM desktop ORDER BY id").fetchall())
    finally:
        conn.close()


def ops(path: Path) -> list[tuple]:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("SELECT desktop, kind, payload_json, inverse_json FROM op_log "
                            "WHERE kind='set_split' ORDER BY id").fetchall()
    finally:
        conn.close()


def run(env, *argv) -> int:
    return main(["--paths", env["paths"], "set-split", *argv])


def test_dry_run_writes_nothing(env, capsys):
    assert run(env, "--test", "2,4") == EXIT_OK
    out = capsys.readouterr().out
    assert "test 2, train 2, untouched 1 (D05)" in out and "D01: - -> train" in out
    assert set(splits(env["db"]).values()) == {None}
    assert ops(env["db"]) == [] and not list((env["tmp"] / "backups").iterdir())


def test_apply_writes_test_train_and_leaves_the_rest(env):
    assert run(env, "--test", "2,4", "--apply") == EXIT_OK
    assert splits(env["db"]) == {1: "train", 2: "test", 3: "train", 4: "test", 5: None}
    assert [(d, k) for d, k, _, _ in ops(env["db"])] == [
        (1, "set_split"), (2, "set_split"), (3, "set_split"), (4, "set_split")]
    assert ops(env["db"])[1][2:] == ('{"split": "test"}', '{"split": null}')
    [backup] = list((env["tmp"] / "backups").iterdir())
    assert backup.name.startswith("tda_pre_set_split_")
    # the rest of the desktop row is kept
    db = Db(str(env["db"]))
    try:
        assert db.get_desktop(2)["model_family"] == "f2"
    finally:
        db.close()
    # a second run changes nothing and logs nothing
    assert run(env, "--test", "2,4", "--apply") == EXIT_OK
    assert len(ops(env["db"])) == 4


def test_an_explicit_train_list_and_its_refusals(env):
    assert run(env, "--test", "2", "--train", "1", "--apply") == EXIT_OK
    assert splits(env["db"]) == {1: "train", 2: "test", 3: None, 4: None, 5: None}
    assert run(env, "--test", "2", "--train", "2,3") == EXIT_ERROR      # on both sides
    assert run(env, "--test", "9") == EXIT_ERROR                         # not in the database


def test_the_coco_export_carries_the_split_once_it_is_set(tmp_path):
    """Spec 8.1: the exporters take the split from Desktop.split (`images[].extra.split`)."""
    from graph_scenes import DESKTOP
    from test_u46_export_graph import build

    from tda.core.export.coco import export_coco
    from tda.core.taxonomy import load_taxonomy

    path = build(tmp_path / "scene.sqlite")
    db = Db(str(path))
    try:
        before = export_coco(db, load_taxonomy(), [DESKTOP], "scan", str(tmp_path / "a.json"))
        assert before["images"] and all("split" not in i["extra"] for i in before["images"])
        db.upsert_desktop(DESKTOP, {**{k: v for k, v in db.get_desktop(DESKTOP).items()
                                       if k != "id"}, "split": "test"})
        after = export_coco(db, load_taxonomy(), [DESKTOP], "scan", str(tmp_path / "b.json"))
    finally:
        db.close()
    assert {i["extra"]["split"] for i in after["images"]} == {"test"}


def test_a_running_app_refuses_the_write(env, monkeypatch):
    monkeypatch.setattr("tda.cli_split.app_running", lambda: True)
    assert run(env, "--test", "2", "--apply") == EXIT_ERROR
    assert set(splits(env["db"]).values()) == {None}
