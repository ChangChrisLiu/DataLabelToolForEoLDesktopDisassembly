"""Task U35, the data side of 「这一帧的说明」: the table, the op, how it is read back.

* schema version 6: ``frame_note``; a v5 file gains the table on open, a v5
  build refuses a v6 file;
* one ``set_frame_note`` op per change, with its inverse (the old row byte for
  byte, or none); saving the same text writes and logs nothing; the session
  puts it on the window's undo stack (undo / redo);
* ``cli notes`` (text, Markdown, one view, a file) and the count in ``cli
  status``;
* ``load-index`` moves a note with its photo -- two old steps landing on one
  keep the main photo's note first and append the other after a marker,
  counted in the log; a note on a photo the new index no longer has refuses
  the desktop like any other work; ``import-logs --force`` never touches one;
* the scanner package writes ``notes.json`` for the packed frames that have one.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest
from test_cli import d13_steps, env, open_db, run  # noqa: F401
from test_index import build_tree, fixes  # noqa: F401
from test_u25_pipeline import _index_file
from test_u25_round2 import _old_and_new

from tda.core.db import SCHEMA_VERSION, Db
from tda.core.db_moves import NOTE_JOIN
from tda.core.frame_notes import (
    OP_KIND,
    apply_note,
    clean_text,
    first_line,
    notes_markdown,
    notes_text,
    set_note,
    when_text,
)
from tda.core.model import FrameKey

DESKTOP, VIEW = 13, "scan"
SCHEMA = Path(__file__).resolve().parents[1] / "tda" / "core" / "schema.sql"


def key(step: int, view: str = VIEW, desktop: int = DESKTOP) -> FrameKey:
    return FrameKey(desktop, step, view)


@pytest.fixture
def db(tmp_path):
    d = Db(str(tmp_path / "notes.sqlite"))
    d.upsert_desktop(DESKTOP, {})
    yield d
    d.close()


def _ops(db: Db) -> list[dict]:
    return db.ops_of_kind(DESKTOP, OP_KIND)


# --------------------------------------------------------------------------- #
# the schema: version 6
# --------------------------------------------------------------------------- #
def test_a_v5_file_gains_the_note_table_and_a_v5_build_refuses_a_v6_file(tmp_path):
    path = str(tmp_path / "old.sqlite")
    text = SCHEMA.read_text(encoding="utf-8")
    start = text.index("-- schema_version 6 (task U35")
    end = text.index(");", text.index("CREATE TABLE IF NOT EXISTS frame_note")) + 2
    old = text[:start] + text[end:]
    assert "frame_note" not in old
    conn = sqlite3.connect(path)
    with conn:
        conn.executescript(old)
        conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', '5')")
        conn.execute("INSERT INTO desktop(id) VALUES(?)", (DESKTOP,))
    conn.close()

    assert SCHEMA_VERSION == 6
    db = Db(path)
    try:
        assert db.conn.execute("SELECT value FROM meta WHERE key='schema_version'"
                               ).fetchone()[0] == "6"
        assert db.frame_notes(DESKTOP) == []
        db.put_frame_note(key(3), "a note", "alice", "2026-10-26T06:03:00+00:00")
        assert db.frame_note(key(3))["text"] == "a note"
    finally:
        db.close()
    Db(path).close()                     # opening it again changes nothing

    older = Db.__new__(Db)               # a v5 build, spelled as the version it knows
    older.path, older._tx_depth = path, 0
    older.conn = sqlite3.connect(path)
    older.conn.row_factory = sqlite3.Row
    try:
        with pytest.raises(RuntimeError, match="schema version 6"):
            older.init_schema(SCHEMA, 5)
        assert older.conn.execute("SELECT value FROM meta WHERE key='schema_version'"
                                  ).fetchone()[0] == "6"
    finally:
        older.conn.close()


def test_the_table_is_one_note_per_frame_and_an_empty_one_is_no_row(db):
    db.put_frame_note(key(3), "first", "alice", "t1")
    db.put_frame_note(key(3), "second", "bob", "t2")
    assert db.frame_notes(DESKTOP) == [{"desktop": DESKTOP, "view": VIEW, "step": 3,
                                        "text": "second", "author": "bob", "updated_at": "t2"}]
    with pytest.raises(ValueError):
        db.put_frame_note(key(4), "  \n ", "alice", "t3")
    db.put_frame_note(key(2, "oak1"), "oak", "alice", "t4")
    assert [(n["view"], n["step"]) for n in db.frame_notes(DESKTOP)] == [("oak1", 2),
                                                                        ("scan", 3)]
    assert db.noted_steps(DESKTOP, VIEW) == {3}
    assert db.note_counts() == {DESKTOP: {"oak1": 1, "scan": 1}}
    db.delete_frame_note(key(3))
    assert db.frame_note(key(3)) is None
    # a note is somebody's work on the frame
    db.put_frame_note(key(5), "kept", "alice", "t5")
    assert "1 frame_note row(s)" in db.frame_work(key(5))


# --------------------------------------------------------------------------- #
# the op
# --------------------------------------------------------------------------- #
def test_set_note_writes_the_row_and_one_op_with_its_inverse(db):
    written = set_note(db, key(3), "  螺丝 screw.psu.03 被胶带挡住\r\n第二行  \n", "alice")
    assert written is not None
    payload, inverse = written
    note = db.frame_note(key(3))
    assert note["text"] == "螺丝 screw.psu.03 被胶带挡住\n第二行"
    assert note["author"] == "alice" and note["updated_at"]
    assert payload == {"desktop": DESKTOP, "view": VIEW, "step": 3, "exists": True,
                       "text": note["text"], "author": "alice",
                       "updated_at": note["updated_at"]}
    assert inverse == {"desktop": DESKTOP, "view": VIEW, "step": 3, "exists": False,
                       "text": None, "author": None, "updated_at": None}
    [logged] = _ops(db)
    assert logged["payload"] == payload and logged["inverse"] == inverse
    assert logged["annotator"] == "alice" and logged["view"] == VIEW

    # the same text again (whitespace aside): nothing written, nothing logged
    assert set_note(db, key(3), note["text"] + "  ", "bob") is None
    assert len(_ops(db)) == 1 and db.frame_note(key(3))["author"] == "alice"

    # changed by somebody else: the inverse is the old row, byte for byte
    _payload, inverse2 = set_note(db, key(3), "改了", "bob")
    assert inverse2 == {"desktop": DESKTOP, "view": VIEW, "step": 3, "exists": True,
                        "text": note["text"], "author": "alice",
                        "updated_at": note["updated_at"]}
    apply_note(db, inverse2)
    assert db.frame_note(key(3)) == note

    # cleared: no row, and the inverse puts it back
    _payload, inverse3 = set_note(db, key(3), "   ", "alice")
    assert db.frame_note(key(3)) is None
    assert inverse3["exists"] and inverse3["text"] == note["text"]
    apply_note(db, inverse3)
    assert db.frame_note(key(3)) == note
    assert len(_ops(db)) == 3


def test_text_helpers():
    assert clean_text(" a\r\nb c ") == "a\nb\nc"
    assert first_line("\n  first \nsecond") == "first…"
    assert first_line("only") == "only"
    assert first_line("x" * 10, limit=4) == "xxxx…"
    assert when_text("2026-10-26T06:03:00+00:00") == "2026-10-26 06:03 UTC"
    assert when_text(None) == "-"


def test_the_session_puts_the_op_on_the_undo_stack(tmp_path, qapp):
    import app_scene

    session = app_scene.make_session(tmp_path)
    try:
        seen: list = []
        session.sigNoteChanged.connect(seen.append)
        here = session.current()
        assert session.frame_note() is None and session.noted_steps() == set()
        assert session.set_frame_note("画面左上角有手") == here
        assert session.frame_note()["text"] == "画面左上角有手"
        assert session.frame_note()["author"] == "tester"
        assert session.noted_steps() == {here.step}
        assert session.undo_stack.ops[-1].kind == OP_KIND
        assert session.dirty
        assert session.set_frame_note("画面左上角有手") is None      # unchanged: no op
        assert len([o for o in session.undo_stack.ops if o.kind == OP_KIND]) == 1

        assert session.undo()
        assert session.frame_note() is None and session.noted_steps() == set()
        assert session.redo()
        assert session.frame_note()["text"] == "画面左上角有手"
        assert seen == [here, here, here]

        # another step of the view, by number; then a frame of another view by key
        other = session.available_steps()[0]
        session.set_frame_note("那一帧", step=other)
        assert session.frame_note(other)["text"] == "那一帧"
        elsewhere = FrameKey(here.desktop, 1, "oak1")
        depth = len(session.undo_stack.ops)
        assert session.set_frame_note("别的视角", key=elsewhere) == elsewhere
        assert session.db.frame_note(elsewhere)["text"] == "别的视角"
        assert len(session.undo_stack.ops) == depth      # not this view's stack
    finally:
        session.close(force=True)
        session.db.close()


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------- #
# reading them back: cli notes, cli status
# --------------------------------------------------------------------------- #
def _notes_in(env) -> None:
    assert run(env, "load-index") == 0
    db = open_db(env)
    try:
        db.put_frame_note(key(4), "第 4 帧：硬盘架挡住了 screw.psu.03\n决定：按看不见处理",
                          "alice", "2026-10-26T06:03:00+00:00")
        db.put_frame_note(key(7), "反光", "bob", "2026-10-26T07:00:00+00:00")
        db.put_frame_note(key(2, "oak1"), "oak1 上手挡住", "alice",
                          "2026-10-26T08:00:00+00:00")
    finally:
        db.close()


def test_cli_notes_prints_step_author_time_and_text(env, capsys):
    _notes_in(env)
    capsys.readouterr()
    assert run(env, "notes", "--desktop", "13") == 0
    out = capsys.readouterr().out
    assert "D13: 3 条说明 / frame note(s)" in out
    assert out.index("[oak1] 1") < out.index("[scan] 2")
    assert "step 4  alice  2026-10-26 06:03 UTC" in out
    assert "    第 4 帧：硬盘架挡住了 screw.psu.03\n    决定：按看不见处理" in out
    assert "step 7  bob  2026-10-26 07:00 UTC\n    反光" in out

    assert run(env, "notes", "--desktop", "13", "--view", "scan", "--format", "md") == 0
    md = capsys.readouterr().out
    assert md.startswith("# D13 这一帧的说明 / frame notes (2)")
    assert "## scan" in md and "oak1" not in md
    assert "### 第 4 帧 / step 4 · alice · 2026-10-26 06:03 UTC" in md
    assert "> 第 4 帧：硬盘架挡住了 screw.psu.03\n> 决定：按看不见处理" in md

    out_file = env["cache"] / "notes_d13.md"
    assert run(env, "notes", "--desktop", "13", "--format", "md", "--out", str(out_file)) == 0
    assert "3 note(s) ->" in capsys.readouterr().out
    assert out_file.read_text(encoding="utf-8").startswith("# D13 这一帧的说明 / frame notes (3)")

    assert run(env, "notes", "--desktop", "77") == 0
    assert "D77: 没有说明 / no frame notes" in capsys.readouterr().out
    assert run(env, "notes", "--desktop", "99") == 1
    assert "not in the database" in capsys.readouterr().out


def test_cli_status_counts_the_notes_per_desktop(env, capsys):
    _notes_in(env)
    capsys.readouterr()
    assert run(env, "status") == 0
    out = capsys.readouterr().out
    assert "这一帧的说明 / frame notes (python -m tda.cli notes --desktop N)" in out
    assert re.search(r"^D13\s+3 \(oak1 1, scan 2\)$", out, re.M)
    assert "D77" not in out.split("这一帧的说明 / frame notes", 1)[1]
    assert run(env, "status", "--desktop", "13") == 0
    assert ("这一帧的说明 / frame notes: 3 (oak1 1, scan 2) -- python -m tda.cli notes "
            "--desktop 13") in capsys.readouterr().out


def test_status_says_nothing_about_notes_when_there_are_none(env, capsys):
    assert run(env, "load-index") == 0
    capsys.readouterr()
    assert run(env, "status") == 0
    assert "frame notes" not in capsys.readouterr().out


def test_notes_text_and_markdown_of_nothing():
    assert notes_text(2, []) == "D02: 没有说明 / no frame notes"
    assert "没有说明" in notes_markdown(2, [])


def test_an_empty_line_of_a_note_prints_empty():
    """Round 2, N7: no four spaces on a blank line."""
    note = {"desktop": 2, "view": "scan", "step": 4, "text": "第一段\n\n第二段",
            "author": "a", "updated_at": None}
    assert "\n    第一段\n\n    第二段" in notes_text(2, [note])


def test_cli_notes_opens_the_file_read_only_and_never_upgrades_it(env, capsys):
    """Round 2, N2: a v5 database stays v5 (no table made); the answer says why."""
    assert run(env, "load-index") == 0
    path = env["db_path"]
    conn = sqlite3.connect(path)
    with conn:
        conn.execute("DROP TABLE frame_note")
        conn.execute("UPDATE meta SET value='5' WHERE key='schema_version'")
    conn.close()
    capsys.readouterr()
    assert run(env, "notes", "--desktop", "13") == 0
    out = capsys.readouterr().out
    assert "D13: 没有说明（库结构 v5，应用升级后第一次打开会建表）" in out
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert conn.execute("SELECT value FROM meta WHERE key='schema_version'"
                            ).fetchone()[0] == "5"
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='frame_note'"
                            ).fetchone()[0] == 0
    finally:
        conn.close()
    # the same holds for --out and --format md
    assert run(env, "notes", "--desktop", "13", "--format", "md",
               "--out", str(env["cache"] / "n.md")) == 0
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert conn.execute("SELECT value FROM meta WHERE key='schema_version'"
                            ).fetchone()[0] == "5"
    finally:
        conn.close()


def test_cli_notes_help_says_how_to_keep_the_chinese(capsys):
    from tda.cli import main
    from tda.cli_notes import ENCODING_HINT

    with pytest.raises(SystemExit):
        main(["notes", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert " ".join(ENCODING_HINT.split()) in out
    assert "--out FILE" in ENCODING_HINT and "chcp 65001" in ENCODING_HINT


def test_a_gallery_row_held_only_by_a_note_says_so(tmp_path):
    """Round 2, N9: the layout photo stays in the sequence because of a note."""
    from test_u10_gallery import _d66_like
    from tda import pipeline as P

    db = _d66_like(tmp_path)
    try:
        db.put_frame_note(FrameKey(66, 3, "oak1"), "零件摆好了", "alice", "t")
        out = P.settle_gallery(db, 66)
        assert ("step 3 (oak1) is a photo of the laid-out parts but has a frame note; "
                "left in the sequence") in out["moved"]
        assert db.get_frame(FrameKey(66, 3, "oak1")) is not None
        assert db.frame_note(FrameKey(66, 3, "oak1"))["text"] == "零件摆好了"
    finally:
        db.close()


def test_import_logs_force_never_touches_a_note(env):
    _notes_in(env)
    assert run(env, "import-logs") == 0
    assert run(env, "import-logs", "--desktops", "13", "--force") == 0
    db = open_db(env)
    try:
        assert [(n["view"], n["step"], n["author"], n["text"])
                for n in db.frame_notes(DESKTOP)] == [
            ("oak1", 2, "alice", "oak1 上手挡住"),
            ("scan", 4, "alice", "第 4 帧：硬盘架挡住了 screw.psu.03\n决定：按看不见处理"),
            ("scan", 7, "bob", "反光")]
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# load-index: the note moves with its photo
# --------------------------------------------------------------------------- #
def test_load_index_moves_the_notes_and_joins_two_that_land_on_one_step(env, tmp_path,
                                                                        capsys):
    """Rows 3 and 6 are dupli rows: their photos become extra shots of steps 2 and 4
    (rows 2 and 5); rows 4 and 7 become steps 3 and 5."""
    old, new = _old_and_new(tmp_path)
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        for step, text, who in ((2, "two", "alice"), (3, "three", "bob"),
                                (4, "four", "alice"), (5, "five", "alice"),
                                (6, "six", "carol"), (7, "seven", "alice")):
            db.put_frame_note(key(step, desktop=95), text, who, f"2026-10-2{step}T01:00:00")
        # an extra shot's note with nothing on the main photo: it just moves
        db.put_frame_note(key(6, "oak1", 95), "oak six", "bob", "2026-10-26T02:00:00")
    finally:
        db.close()
    capsys.readouterr()
    assert run(env, "load-index", "--index", new) == 0
    out = capsys.readouterr().out
    scan_line = next(line for line in out.splitlines() if "D95: scan:" in line)
    assert ("2 frame note(s) landed on a step that has one: appended to it with a "
            "[load-index] marker") in scan_line
    oak_line = next(line for line in out.splitlines() if "D95: oak1:" in line)
    assert "frame note" not in oak_line
    db = open_db(env)
    try:
        got = {n["step"]: n for n in db.frame_notes(95, "scan")}
        assert sorted(got) == [2, 3, 4, 5]
        assert got[2]["text"] == "two" + NOTE_JOIN.format(
            old=3, author="bob", when="2026-10-23 01:00") + "three"
        assert (got[2]["author"], got[2]["updated_at"]) == ("alice", "2026-10-22T01:00:00")
        assert got[3]["text"] == "four" and got[3]["updated_at"] == "2026-10-24T01:00:00"
        assert got[4]["text"] == "five" + NOTE_JOIN.format(
            old=6, author="carol", when="2026-10-26 01:00") + "six"
        assert got[5]["text"] == "seven"
        # the time as cli notes writes it (round 2, N7)
        assert "[load-index：以下是原第 3 步的说明（bob，2026-10-23 01:00）]" in got[2]["text"]
        [oak] = db.frame_notes(95, "oak1")
        assert (oak["step"], oak["text"], oak["author"]) == (4, "oak six", "bob")
    finally:
        db.close()
    capsys.readouterr()
    assert run(env, "load-index", "--index", new) == 0      # idempotent
    assert "frame note" not in capsys.readouterr().out


def test_the_main_photos_note_leads_even_from_a_higher_old_step(env, tmp_path, capsys):
    """Round 2, N10: row 3 is step 2's own photo, row 2 its extra shot. Old step 2
    keeps its number (its photo is now the extra shot), old step 3 moves onto it:
    the main photo's note (old 3) leads, old 2's follows after the marker."""
    old = _index_file(tmp_path / "old.json", 95, {r: r for r in range(1, 6)})
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        db.put_frame_note(key(2, desktop=95), "extra two", "bob", "2026-10-22T01:00:00")
        db.put_frame_note(key(3, desktop=95), "main three", "alice", "2026-10-23T01:00:00")
    finally:
        db.close()
    new = _index_file(tmp_path / "new.json", 95, {1: 1, 3: 2, 4: 3, 5: 4}, extra={2: 2})
    capsys.readouterr()
    assert run(env, "load-index", "--index", new) == 0
    assert "1 frame note(s) landed on a step that has one" in capsys.readouterr().out
    db = open_db(env)
    try:
        two = db.frame_note(key(2, desktop=95))
        assert two["text"] == "main three" + NOTE_JOIN.format(
            old=2, author="bob", when="2026-10-22 01:00") + "extra two"
        assert (two["author"], two["updated_at"]) == ("alice", "2026-10-23T01:00:00")
        assert db.frame_note(key(3, desktop=95)) is None
    finally:
        db.close()


def test_a_note_on_a_photo_the_new_index_drops_refuses_the_desktop(env, tmp_path, capsys):
    old = _index_file(tmp_path / "old.json", 95, {r: r for r in range(1, 7)})
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        db.put_frame_note(key(6, desktop=95), "这张照片之后不在了", "alice", "t")
    finally:
        db.close()
    new = _index_file(tmp_path / "new.json", 95, {r: r for r in range(1, 6)})
    capsys.readouterr()
    assert run(env, "load-index", "--index", new) == 1
    out = capsys.readouterr().out
    assert "scan step 6" in out and "1 frame_note row(s)" in out
    db = open_db(env)
    try:
        assert db.frame_note(key(6, desktop=95))["text"] == "这张照片之后不在了"
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# the scanner package: notes.json
# --------------------------------------------------------------------------- #
def test_the_scanpack_writes_notes_json_for_the_packed_frames(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import test_u30_scanpack as T

    desktop = 2
    db_path = T.build_scene(tmp_path, desktop)
    db = Db(str(db_path))
    try:
        db.put_frame_note(key(3, desktop=2), "螺丝 screw.psu.01 反光，按可见处理",
                          "alice", "2026-10-26T06:03:00+00:00")
        db.put_frame_note(key(T.EXCLUDED[desktop], desktop=2), "excluded frame", "a", "t")
        db.put_frame_note(key(T.LAST[desktop], desktop=2), "not confirmed", "a", "t")
        db.put_frame_note(key(3, "oak1", 2), "another view", "a", "t")
    finally:
        db.close()
    summary = T.make_pack(tmp_path, db_path, desktop)
    ddir = tmp_path / "pack" / "D02"
    notes = json.loads((ddir / "notes.json").read_text(encoding="utf-8"))
    assert notes["desktop"] == 2 and notes["view"] == "scan"
    assert notes["notes"] == [{"step": 3, "frame": "S03",
                               "text": "螺丝 screw.psu.01 反光，按可见处理",
                               "updated_at": "2026-10-26T06:03:00+00:00"}]
    assert "author" not in notes["notes"][0]
    (desk,) = summary["per_desktop"]
    assert desk["frame_notes"] == 1
    manifest = json.loads((tmp_path / "pack" / "manifest.json").read_text(encoding="utf-8"))
    assert "D02/notes.json" in manifest["files"]
    readme = (tmp_path / "pack" / "README.md").read_text(encoding="utf-8")
    assert "  notes.json         the annotator's notes on some frames" in readme
    assert "* `notes.json` (D02) -- what the annotator wrote about a frame" in readme


def test_the_scanpack_readme_says_nothing_of_notes_without_any():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import test_u30_scanpack as T
    import export_scanpack as S

    summary = T._summary(
        pack_version="1", desktops=[30],
        per_desktop=[{"desktop": 30, "model": "m", "frames": 1, "instances_annotated": 1,
                      "annotations": 1, "crowd": 0, "clean": 1, "clean_instances": 1,
                      "pose_segments": 1, "rois": 1, "frame_notes": 0}],
        totals={"frames": 1, "annotations": 1, "crowd": 0, "clean": 1, "overlap_frames": 0,
                "overlap_px": 0, "amodal_widened_rows": 0},
        checks={"amodal_box_outside_roi": 0, "clean_but_for_roi": 0,
                "occlusion_ratio_differs_from_confirmed": 0, "occluded_full_with_pixels": 0},
        visibility_counts={"visible": 1}, burst_sizes={"10": 1}, burst_reasons={"p0": 1},
        frame_size={"width": 1600, "height": 1600}, tier="gold", state_values=["installed"],
        placement_values=["in_chassis"], frame_record_names_other_image=[], exclusions=[],
        notes=[], amodal_box_outside_roi_by_class={})
    assert "notes.json" not in S.readme_text(summary)
