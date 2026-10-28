"""Task U25 round 2: the review's blockers and non-blockers.

B2  ``import-logs`` refuses a desktop whose frames are not loaded under the
    step numbering its sheet collapses to (``load-index`` refused, failed, not run).
B3  ``load-index`` renumbers the hand-written state events with the step table.
N3  work on a photo that left the index refuses the desktop (test_u25_pipeline).
N4  an extra shot's human frame columns are carried or counted; colliding pose
    breaks are counted.
N5  a Label Studio draft on an extra shot does not double the step's own one --
    neither when ``load-index`` moves it nor when ``import-ls`` reads it.
N7  ``build-index`` keeps the previous index and report.
plus the index rules of round 2: ``scan.<d>.skipped_steps``, ``oak.<d>.cam1_order``,
``oak.<d>.gallery_captures``.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import pytest
from test_cli import LOG_FIXTURES, d13_steps, env, open_db, run  # noqa: F401
from test_index import build_tree, fixes  # noqa: F401
from test_u25_pipeline import SHEET, _index_file, _keyframe

from tda.core.db_moves import loaded_collapse_refusal
from tda.core.index import keep_previous, load_index, save_index, scan_desktop, scan_step
from tda.core.index_fixes import declared_missing, verify_scan_rule
from tda.core.model import VIEWS, FrameKey, StepRec, StepType

#: D95's sheet: 7 rows, dupli 3 and 6 -> 5 steps (rows 1 2 4 5 7).
ROWS_TO_STEPS = {1: 1, 2: 2, 4: 3, 5: 4, 7: 5}
EXTRA = {3: 2, 6: 4}


def _sheet_dir(tmp_path: Path) -> str:
    drive = tmp_path / "drive95"
    drive.mkdir()
    with open(drive / "desktop_95.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Sequence Number", "Sequence Name", "Target Nest Group",
                         "Tool Utility", "Notes"])
        for i, name in enumerate(SHEET, start=1):
            writer.writerow([i, name, "", "", ""])
    return str(drive)


def _old_and_new(tmp_path: Path) -> tuple[str, str]:
    old = _index_file(tmp_path / "old.json", 95, {r: r for r in range(1, 8)})
    new = _index_file(tmp_path / "new.json", 95, ROWS_TO_STEPS, extra=EXTRA)
    return old, new


# --------------------------------------------------------------------------- #
# B2
# --------------------------------------------------------------------------- #
def test_loaded_collapse_refusal_rules():
    assert loaded_collapse_refusal(None, False, [3]) is None          # no frames yet
    assert loaded_collapse_refusal({}, True, []) is None              # nothing to collapse
    assert "before the sheet's dupli rows [3]" in loaded_collapse_refusal({}, True, [3])
    meta = {"index_dupli_rows": [3, 6]}
    assert loaded_collapse_refusal(meta, True, [6, 3]) is None
    assert "with dupli rows [3, 6]" in loaded_collapse_refusal(meta, True, [3])
    assert "with dupli rows none" in loaded_collapse_refusal({"index_dupli_rows": []},
                                                             True, [3])


def test_import_logs_waits_for_load_index(env, tmp_path, capsys):
    """A verified frame makes load-index refuse D95; import-logs --force then
    refuses too and writes nothing; once load-index went through, it imports."""
    drive = _sheet_dir(tmp_path)
    old, new = _old_and_new(tmp_path)
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        db.set_frame_flags(FrameKey(95, 4, "oak1"), review_status="verified")
    finally:
        db.close()
    assert run(env, "load-index", "--index", new) == 1
    capsys.readouterr()
    assert run(env, "import-logs", "--force", "--desktops", "95", "--dir", drive,
               "--index", new) == 1
    out = capsys.readouterr().out
    assert "0 desktops imported" in out and "1 refused" in out
    assert "refused: D95 have frames that are not loaded" in out
    assert "dupli rows none collapsed, the sheet has [3, 6]" in out
    db = open_db(env)
    try:
        assert db.steps(95) == []                          # nothing written
        db.set_frame_flags(FrameKey(95, 4, "oak1"), review_status="unlabeled")
    finally:
        db.close()
    assert run(env, "load-index", "--index", new) == 0
    assert run(env, "import-logs", "--force", "--desktops", "95", "--dir", drive,
               "--index", new) == 0
    db = open_db(env)
    try:
        assert [s.step for s in db.steps(95)] == [1, 2, 3, 4, 5]
        assert db.get_desktop(95)["index_dupli_rows"] == [3, 6]
    finally:
        db.close()


def test_import_logs_of_a_desktop_without_frames_or_dupli_rows_is_not_refused(env):
    """D13 has no dupli row: frames loaded from any index are fine (and the env's
    index is a test index without the marker's history)."""
    assert run(env, "load-index") == 0
    db = open_db(env)
    try:
        meta = db.get_desktop(13)
        meta.pop("id", None)
        meta.pop("index_dupli_rows", None)     # as loaded before U25
        db.upsert_desktop(13, meta)
    finally:
        db.close()
    assert run(env, "import-logs", "--desktops", "13") == 0


# --------------------------------------------------------------------------- #
# B3, N4, N5 (load-index)
# --------------------------------------------------------------------------- #
def test_hand_written_events_follow_the_step_table(env, tmp_path, capsys):
    old, new = _old_and_new(tmp_path)
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        # row 5 (old step 5) is step 4 after the collapse; row 3 is dupli -> 2
        for step, auto in ((5, 0), (3, 0), (1, 0), (5, 1)):
            db.conn.execute(
                "INSERT INTO state_event(desktop, step, target, attr, old, new, auto) "
                "VALUES(95, ?, 'screw.01', 'state', 'fastened', 'removed', ?)", (step, auto))
        db.conn.commit()
    finally:
        db.close()
    capsys.readouterr()
    assert run(env, "load-index", "--index", new) == 0
    assert "2 hand-written state event(s) renumbered with the step table" in \
        capsys.readouterr().out
    db = open_db(env)
    try:
        rows = db.conn.execute("SELECT step, auto FROM state_event WHERE desktop=95 "
                               "ORDER BY id").fetchall()
        # hand-written ones moved; the derived one waits for import-logs
        assert [(r[0], r[1]) for r in rows] == [(4, 0), (2, 0), (1, 0), (5, 1)]
    finally:
        db.close()
    capsys.readouterr()
    assert run(env, "load-index", "--index", new) == 0       # idempotent
    assert "state event" not in capsys.readouterr().out


def test_extra_shot_review_status_is_carried_photo_columns_stay(env, tmp_path, capsys):
    """N4 / N16: of an old step whose photo became an extra shot, only the review
    status moves to the new step (where that has none); the photo columns stay
    with the photo and are reported. Colliding pose breaks are counted."""
    old, new = _old_and_new(tmp_path)
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        # row 3 (dupli of row 2): a review status step 2's own row lacks -> carried;
        # a photo column -> left behind
        db.set_frame_flags(FrameKey(95, 3, "oak2"), review_status="ls_draft", in_progress=1)
        # row 6 (dupli of row 5, which moves to 4): its photo column stays with it
        db.set_frame_flags(FrameKey(95, 5, "oak2"), image_quality="good")
        db.set_frame_flags(FrameKey(95, 6, "oak2"), image_quality="blurred")
        for step in (5, 6):
            db.conn.execute("INSERT INTO pose_break(desktop, view, step, source) "
                            "VALUES(95, 'oak2', ?, 'manual:test')", (step,))
        db.conn.commit()
    finally:
        db.close()
    capsys.readouterr()
    assert run(env, "load-index", "--index", new) == 0
    out = capsys.readouterr().out
    assert "1 review status(es) carried from an extra shot's old step" in out
    assert ("2 photo column(s) left behind on an extra shot: step 3->2 in_progress=1, "
            "step 6->4 image_quality=blurred") in out
    assert "1 pose break(s) landed on one already there" in out
    db = open_db(env)
    try:
        two = db.get_frame(FrameKey(95, 2, "oak2"))
        assert two["review_status"] == "ls_draft" and two["in_progress"] is None
        assert db.get_frame(FrameKey(95, 4, "oak2"))["image_quality"] == "good"
    finally:
        db.close()


def test_a_draft_on_an_extra_shot_does_not_double_the_steps_own(env, tmp_path, capsys):
    old, new = _old_and_new(tmp_path)
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        own = _keyframe(db, 95, "oak1", "ls:PSU#1", 5, source="labelstudio")
        twice = _keyframe(db, 95, "oak1", "ls:PSU#2", 6, source="labelstudio")
        only = _keyframe(db, 95, "oak1", "ls:RAM#1", 6, source="labelstudio")
        kept = _keyframe(db, 95, "oak1", "ls:PSU#1", 3, source="labelstudio")   # step 2 has none
    finally:
        db.close()
    capsys.readouterr()
    assert run(env, "load-index", "--index", new) == 0
    assert "dropped 1 Label Studio draft(s) of an extra shot" in capsys.readouterr().out
    db = open_db(env)
    try:
        anchors = {kf.id: kf.anchor_step for kf in db.keyframes(95, "oak1")}
        assert anchors[own] == 4 and anchors[only] == 4 and anchors[kept] == 2
        assert twice not in anchors
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# N5 (import-ls)
# --------------------------------------------------------------------------- #
def _polygon(rid: str, label: str, x: float) -> list[dict]:
    pts = [[x, 40.0], [x + 6, 40.0], [x + 6, 50.0], [x, 50.0]]
    common = {"id": rid, "origin": "manual", "to_name": "image1", "image_rotation": 0,
              "original_width": 1280, "original_height": 720}
    return [dict(common, type="polygon", from_name="polygon",
                 value={"closed": True, "points": pts}),
            dict(common, type="labels", from_name="valuable_component_labels",
                 value={"closed": True, "points": pts, "labels": [label]})]


def _task(tid: int, folder: int, shapes: list[list[dict]]) -> dict:
    return {"id": tid, "data": {"image": f"/storage-data/uploaded/?filepath=upload/9/"
                                          f"{tid:08x}-Rs_95_{folder}.png"},
            "annotations": [{"id": tid * 10, "result": [r for s in shapes for r in s]}]}


def test_import_ls_skips_an_extra_shots_draft_the_steps_photo_has(env, tmp_path):
    _old, new = _old_and_new(tmp_path)
    export = {"exported_at": "2026-10-24", "source": "test", "projects": [{
        "id": 9, "title": "Realsense_camera", "workspace": "Desktop_95",
        # the extra shot (row 3) first: the order of the export must not matter
        "tasks": [_task(2, 3, [_polygon("a1", "Power Supply Unit (PSU)", 10),
                               _polygon("a2", "RAM Module", 30)]),
                  _task(1, 2, [_polygon("b1", "Power Supply Unit (PSU)", 12)])]}]}
    path = tmp_path / "ls95.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    db = open_db(env)
    try:
        db.replace_steps(95, [StepRec(95, k, StepType.NORMAL.value, f"row {k}")
                              for k in range(1, 6)], [])
    finally:
        db.close()
    assert run(env, "import-ls", "--export", str(path), "--index", new,
               "--desktops", "95") == 0
    summary = json.loads((env["cache"] / "ls_import_summary.json").read_text(encoding="utf-8"))
    assert summary["extra_shot_drafts_skipped"] == 1
    assert summary["extra_shot_drafts_kept"] == 1
    db = open_db(env)
    try:
        drafts = sorted((kf.anchor_step, kf.instance) for kf in db.keyframes(95, "rs"))
        assert [s for s, _ in drafts] == [2, 2]
        labels = sorted(i.split("#")[0] for _s, i in drafts)
        assert len(labels) == 2 and labels[0] != labels[1]     # one PSU, one RAM
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# N7
# --------------------------------------------------------------------------- #
def test_build_index_keeps_the_previous_index_and_report(tmp_path):
    out, report = tmp_path / "index.json", tmp_path / "index_report.md"
    assert keep_previous(str(out), str(report)) == []
    out.write_text('{\n "version": 2,\n "desktops": {}}', encoding="utf-8")
    report.write_text("# old report", encoding="utf-8")
    when = datetime(2026, 10, 24, 9, 30, 0)
    kept = keep_previous(str(out), str(report), when)
    assert [os.path.basename(k) for k in kept] == ["index_v2_20261024.json",
                                                    "index_report_v2_20261024.md"]
    assert keep_previous(str(out), str(report), when) == []        # same bytes: once
    out.write_text('{\n "version": 2,\n "desktops": {"1": {}}}', encoding="utf-8")
    again = keep_previous(str(out), None, when)
    assert [os.path.basename(k) for k in again] == ["index_v2_20261024_093000.json"]


# --------------------------------------------------------------------------- #
# the index rules
# --------------------------------------------------------------------------- #
def test_scan_step_with_skipped_steps():
    assert [scan_step(n, [], [32]) for n in (30, 31, 32, 33, 34)] == [30, 31, 33, 34, 35]
    assert [scan_step(n, [], [5, 7]) for n in (4, 5, 6, 7)] == [4, 6, 8, 9]
    assert [scan_step(n, [6], [3]) for n in (2, 3, 5, 6, 7)] == [2, 4, 6, 6, 7]
    assert declared_missing({"scan": {41: {"skipped_steps": [32]}}}, "scan", 41) == {32}
    assert declared_missing({"rs": {41: {"skipped_steps": [32]}}}, "rs", 41) == set()


def test_a_skipped_step_has_no_scanner_frame_and_the_rest_shift(tmp_path, fixes):
    fx = dict(fixes)
    fx["scan"] = dict(fixes["scan"])
    fx["scan"][96] = {"skipped_steps": [3], "reason": "test"}
    ts = [f"20250101_10{m:02d}00_000" for m in range(5)]
    roots = build_tree(tmp_path, 96, oak1={f"{i + 1:03d}": [ts[i]] for i in range(5)},
                       oak2={}, scan_steps={1: 10, 2: 10, 3: 10, 4: 10})
    idx = scan_desktop(96, roots, fx)
    scan = {k.step: ff.src_step_dir for k, ff in idx.frames.items() if k.view == "scan"}
    assert scan == {1: "RGB11", 2: "RGB21", 4: "RGB31", 5: "RGB41"}
    assert FrameKey(96, 3, "scan") in idx.missing
    line = [i for i in idx.issues if i.startswith("fix:scan[96] duplicate_folders")]
    assert line and "skipped steps [3]" in line[0] and "MISMATCH" not in line[0]
    # an undeclared gap would be a MISMATCH
    assert "MISMATCH" in verify_scan_rule(96, {"duplicate_folders": []}, [], 5, idx.frames,
                                          idx.missing)[0]


def test_gallery_captures_leave_the_sequence_for_shot_1000_plus_folder(tmp_path, fixes):
    fx = dict(fixes)
    fx["oak"] = dict(fixes["oak"])
    fx["oak"][96] = {"gallery_captures": ["003"], "reason": "test"}
    ts = ["20250101_100000_000", "20250101_100100_000", "20250101_100200_000"]
    roots = build_tree(tmp_path, 96, oak1={f"{i + 1:03d}": [ts[i]] for i in range(3)},
                       oak2={f"{i + 1:03d}": [ts[i]] for i in range(3)}, rs_steps=[1, 2, 3])
    idx = scan_desktop(96, roots, fx)
    assert idx.n_steps == 2 and idx.n_rows == 2
    for view, gallery in (("oak1", "cmp_oak1"), ("oak2", "cmp_oak2"), ("rs", "cmp_rs")):
        assert FrameKey(96, 3, view) not in idx.frames and FrameKey(96, 3, view) \
            not in idx.missing
        ff = idx.frames[FrameKey(96, 1003, gallery)]
        assert ff.aux["sequence_capture"].endswith("003")
    assert any("gallery_captures: Camera_1 003" in i for i in idx.issues)


def test_the_real_round2_declarations_hold(fixes):
    """The yaml's round-2 entries are the coordinator's rulings (U25 round 2)."""
    assert fixes["oak"][1]["gallery_captures"] == ["040"]
    assert fixes["oak"][60]["cam1_order"] == "folder"
    assert fixes["scan"][41]["skipped_steps"] == [32]
    assert fixes["scan"][61]["skipped_steps"] == [26]


def test_scan_align_declares_a_skipped_real_step():
    from tda.core.scan_align import declaration

    # 4 folders for 5 rows: folder 3 already shows row 4 -> step 3 was skipped
    p = declaration([1, 2, 4, 5], [1, 2, 3, 4], 5, [])
    assert p.skipped_steps == [3] and p.duplicate_folders == [] and p.expressible
    assert p.missing_steps == [3]
    # a dupli row the scanner skipped is not a skipped *step* (the collapse has it)
    p = declaration([1, 2, 4, 5], [1, 2, 3, 4], 5, [3])
    assert p.skipped_steps == [] and p.skipped_dupli == [3]


# --------------------------------------------------------------------------- #
# round 3: N17 and the pre-check script
# --------------------------------------------------------------------------- #
def test_the_force_line_comes_only_after_the_frames_check(env, tmp_path, capsys):
    """N17: a desktop refused for its frames never says '--force, replacing ...'."""
    drive = _sheet_dir(tmp_path)
    old, new = _old_and_new(tmp_path)
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        db.replace_steps(95, [StepRec(95, k, StepType.NORMAL.value, f"row {k}")
                              for k in range(1, 8)], [])
    finally:
        db.close()
    capsys.readouterr()
    assert run(env, "import-logs", "--force", "--desktops", "95", "--dir", drive,
               "--index", new) == 1
    out = capsys.readouterr().out
    assert "refused: D95" in out and "--force, replacing" not in out
    assert run(env, "load-index", "--index", new) == 0
    capsys.readouterr()
    assert run(env, "import-logs", "--force", "--desktops", "95", "--dir", drive,
               "--index", new) == 0
    out = capsys.readouterr().out
    assert "--force, replacing 7 steps" in out


def test_precheck_index_moves(env, tmp_path, capsys):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import precheck_index_moves as PC

    _old, new = _old_and_new(tmp_path)
    assert run(env, "load-index", "--index", new) == 0
    capsys.readouterr()
    assert PC.main(["--db", env["db_path"], "--desktops", "95"]) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "OK" and "[blocks] verified frames: none" in out
    db = open_db(env)
    try:
        db.set_frame_flags(FrameKey(95, 2, "scan"), review_status="verified")
        db.conn.execute("INSERT INTO state_event(desktop, step, target, attr, old, new, auto) "
                        "VALUES(95, 2, 'x', 'state', 'a', 'b', 0)")
        db.conn.commit()
    finally:
        db.close()
    assert PC.main(["--db", env["db_path"], "--desktops", "95"]) == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "NOT OK: verified frames on D95"
    assert any("[note] hand-written state events" in ln and "D95: 1" in ln for ln in lines)
    assert PC.main(["--db", str(tmp_path / "nowhere.sqlite")]) == 2
