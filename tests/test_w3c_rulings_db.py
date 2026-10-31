"""Task W3c: the ``cover_ruling`` table (schema version 7) and its one-op writes.

* migration: a v6 file opens unchanged but for the new, empty table and the
  stamp; a v6 build refuses a v7 file; the DDL's columns are exactly the ones
  W3a's ``pixel-edges`` reads;
* ``set_cover_ruling``: upsert, one ``op_log`` row of kind ``cover_ruling``
  whose inverse is the previous row (or its deletion); the same ruling again
  writes and logs nothing; a batch is one op over several rows;
* ``clear_cover_ruling``: one op whose inverse is the row; the op's payload and
  inverse put the rows back byte for byte both ways.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tda.core.db import SCHEMA_VERSION, Db
from tda.core.db_rulings import OP_KIND, RULING_TABLE
from tda.core.dbconn import ADDED_TABLES
from tda.ui.commands import KINDS

SCHEMA = Path(__file__).resolve().parents[1] / "tda" / "core" / "schema.sql"
D = 13
COLUMNS = ["desktop", "above", "below", "ruling", "by", "ts", "view", "step", "note"]


@pytest.fixture
def db(tmp_path):
    d = Db(str(tmp_path / "t.sqlite"))
    yield d
    d.close()


def ops(db: Db) -> list[dict]:
    return db.ops_of_kind(D, OP_KIND)


def rows(db: Db) -> list[tuple]:
    return [tuple(r) for r in db.conn.execute(
        "SELECT * FROM cover_ruling ORDER BY desktop, above, below")]


# --------------------------------------------------------------------------- #
# the schema
# --------------------------------------------------------------------------- #
def test_the_table_is_the_briefs_ddl(db):
    assert SCHEMA_VERSION == 7 and ADDED_TABLES[7] == (RULING_TABLE,) == ("cover_ruling",)
    info = db.conn.execute("PRAGMA table_info(cover_ruling)").fetchall()
    assert [r["name"] for r in info] == COLUMNS
    assert [r["pk"] for r in info] == [1, 2, 3, 0, 0, 0, 0, 0, 0]
    assert {r["name"]: r["type"] for r in info}["step"] == "INTEGER"
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("INSERT INTO cover_ruling(desktop, above, below, ruling) "
                        "VALUES(1, 'a', 'b', 'maybe')")
    assert OP_KIND in KINDS


def test_a_v6_file_gains_the_table_and_a_v6_build_refuses_a_v7_file(tmp_path):
    path = str(tmp_path / "old.sqlite")
    text = SCHEMA.read_text(encoding="utf-8")
    start = text.index("-- schema_version 7 (task W3c")
    end = text.index("));", text.index("CREATE TABLE IF NOT EXISTS cover_ruling")) + 3
    old = text[:start] + text[end:]
    assert "cover_ruling" not in old
    conn = sqlite3.connect(path)
    with conn:
        conn.executescript(old)
        conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', '6')")
        conn.execute("INSERT INTO desktop(id) VALUES(?)", (D,))
        conn.execute("INSERT INTO frame_note(desktop, view, step, text) VALUES(?, 'scan', 3, 'n')",
                     (D,))
    tables_before = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()

    db = Db(path)
    try:
        assert db.conn.execute("SELECT value FROM meta WHERE key='schema_version'"
                               ).fetchone()[0] == "7"
        tables = {r[0] for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables - tables_before == {"cover_ruling"}
        assert db.cover_rulings(D) == {}
        assert db.frame_notes(D)[0]["text"] == "n"       # the old rows are untouched
        assert db.set_cover_ruling(D, "a.01", "b.01", "blocks", view="scan", step=3,
                                   by="alice") is not None
    finally:
        db.close()
    Db(path).close()                     # opening it again changes nothing

    older = Db.__new__(Db)               # a v6 build, spelled as the version it knows
    older.path, older._tx_depth = path, 0
    older.conn = sqlite3.connect(path)
    older.conn.row_factory = sqlite3.Row
    try:
        with pytest.raises(RuntimeError, match="schema version 7"):
            older.init_schema(SCHEMA, 6)
    finally:
        older.conn.close()


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #
def test_set_upserts_one_op_with_the_previous_row_as_inverse(db):
    written = db.set_cover_ruling(D, "optical_drive.01", "motherboard.01", "blocks",
                                  view="scan", step=7, by="alice")
    payload, inverse = written
    got = db.cover_rulings(D)
    assert list(got) == [("optical_drive.01", "motherboard.01")]
    row = got[("optical_drive.01", "motherboard.01")]
    assert (row["ruling"], row["by"], row["view"], row["step"], row["note"]) == \
        ("blocks", "alice", "scan", 7, None)
    assert row["ts"]
    [op] = ops(db)
    assert (op["view"], op["annotator"]) == ("scan", "alice")
    assert op["payload"] == payload and op["inverse"] == inverse
    assert payload["rows"] == [{"above": "optical_drive.01", "below": "motherboard.01",
                                "exists": True, "ruling": "blocks", "by": "alice",
                                "ts": row["ts"], "view": "scan", "step": 7, "note": None}]
    assert inverse["rows"][0]["exists"] is False and inverse["desktop"] == D

    # the same again: nothing written, nothing logged
    assert db.set_cover_ruling(D, "optical_drive.01", "motherboard.01", "blocks",
                               view="oak1", step=9, by="bob") is None
    assert len(ops(db)) == 1 and db.cover_rulings(D) == got

    # changed: upsert; the inverse is the previous row exactly
    payload2, inverse2 = db.set_cover_ruling(D, "optical_drive.01", "motherboard.01",
                                             "not_blocks", view="scan", step=5, by="bob",
                                             note="悬空")
    assert len(ops(db)) == 2
    assert inverse2["rows"][0] == payload["rows"][0]
    now = db.cover_ruling(D, "optical_drive.01", "motherboard.01")
    assert (now["ruling"], now["by"], now["step"], now["note"]) == ("not_blocks", "bob", 5, "悬空")

    with pytest.raises(ValueError):
        db.set_cover_ruling(D, "a", "b", "maybe", view="scan", step=1, by="x")


def test_the_op_replays_both_ways_byte_for_byte(db):
    db.set_cover_ruling(D, "a.01", "b.01", "blocks", view="scan", step=1, by="alice")
    first = rows(db)
    payload, inverse = db.set_cover_ruling(D, "a.01", "b.01", "not_blocks", view="scan",
                                           step=2, by="bob")
    second = rows(db)
    assert db.apply_cover_ruling_op(inverse) == D
    assert rows(db) == first
    db.apply_cover_ruling_op(payload)
    assert rows(db) == second
    # a first ruling undone is no row at all
    p3, i3 = db.set_cover_ruling(D, "c.01", "b.01", "blocks", view="scan", step=1, by="a")
    db.apply_cover_ruling_op(i3)
    assert db.cover_ruling(D, "c.01", "b.01") is None
    db.apply_cover_ruling_op(p3)
    assert db.cover_ruling(D, "c.01", "b.01")["ruling"] == "blocks"


def test_a_batch_is_one_op_and_skips_what_already_says_so(db):
    db.set_cover_ruling(D, "a.01", "m.01", "blocks", view="scan", step=1, by="alice")
    before = len(ops(db))
    payload, inverse = db.set_cover_rulings(
        D, [("a.01", "m.01"), ("a.02", "m.01"), ("a.02", "m.01"), ("a.03", "m.01")], "blocks",
        view="scan", step=4, by="alice")
    assert len(ops(db)) == before + 1
    assert [(r["above"], r["below"]) for r in payload["rows"]] == [("a.02", "m.01"),
                                                                    ("a.03", "m.01")]
    assert all(not r["exists"] for r in inverse["rows"])
    assert set(db.cover_rulings(D)) == {("a.01", "m.01"), ("a.02", "m.01"), ("a.03", "m.01")}
    db.apply_cover_ruling_op(inverse)
    assert set(db.cover_rulings(D)) == {("a.01", "m.01")}
    assert db.set_cover_rulings(D, [("a.01", "m.01")], "blocks", view="scan", step=4,
                                by="alice") is None


def test_clear_is_one_op_whose_inverse_is_the_row(db):
    assert db.clear_cover_ruling(D, "a.01", "b.01") is None
    db.set_cover_ruling(D, "a.01", "b.01", "not_blocks", view="oak1", step=3, by="alice")
    kept = rows(db)
    payload, inverse = db.clear_cover_ruling(D, "a.01", "b.01", view="scan", by="bob")
    assert db.cover_rulings(D) == {}
    [_first, op] = ops(db)
    assert (op["view"], op["annotator"]) == ("scan", "bob")
    assert payload["rows"][0]["exists"] is False
    assert inverse["rows"][0]["ruling"] == "not_blocks" and inverse["rows"][0]["view"] == "oak1"
    db.apply_cover_ruling_op(inverse)
    assert rows(db) == kept
    # the op row's JSON is the payload as written
    raw = db.conn.execute("SELECT payload_json FROM op_log WHERE id=?", (op["id"],)).fetchone()[0]
    assert json.loads(raw) == payload


def test_rulings_are_per_desktop_and_ordered(db):
    db.set_cover_ruling(5, "z.01", "b.01", "blocks", view="scan", step=1, by="a")
    db.set_cover_ruling(5, "a.01", "b.01", "not_blocks", view="scan", step=1, by="a")
    db.set_cover_ruling(24, "a.01", "b.01", "blocks", view="scan", step=1, by="a")
    assert list(db.cover_rulings(5)) == [("a.01", "b.01"), ("z.01", "b.01")]
    assert list(db.cover_rulings(24)) == [("a.01", "b.01")]
    assert db.ruling_counts() == {5: {"blocks": 1, "not_blocks": 1}, 24: {"blocks": 1}}
