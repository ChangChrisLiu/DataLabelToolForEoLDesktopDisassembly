"""Task U25 through the pipeline: the sheet import drops ``dupli`` rows, the step
table and the index agree, ``load-index`` moves the work of a renumbered photo
with it, and Label Studio drafts land on the step their photo is now.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from test_cli import LOG_FIXTURES, LS_EXPORT, d13_steps, env, open_db, run  # noqa: F401
from test_logs import row, tax  # noqa: F401

from tda.core.index import DesktopIndex, FrameFile, save_index
from tda.core.logs import (dupli_rows_of, import_log, read_desktop_csv,
                           sheet_rows_by_desktop)
from tda.core.masks import encode_rle
from tda.core.model import VIEWS, FrameKey, ShapeKeyframe, ShapePart, StepRec, StepType
from tda.pipeline_logs import _apply_index

import numpy as np


# --------------------------------------------------------------------------- #
# the sheet
# --------------------------------------------------------------------------- #
SHEET = ["Initial Conditions", "CPU Fan Screw 1", "dupli", "CPU Fan Screw 2", "CPU fan",
         "replacement of 5 (dupli)", "Motherboard"]


def _rows(names, notes=None):
    notes = notes or {}
    return [row(n, i + 1, Notes=notes.get(i + 1, "")) for i, n in enumerate(names)]


def test_two_dupli_rows_are_dropped_and_the_rest_renumbered(tax):
    li = import_log(90, _rows(SHEET, {3: "Scanner rescan", 4: "glued"}), {}, tax)
    assert li.dupli_rows == [3, 6] and li.sheet_rows == 7
    assert [(s.step, s.raw_name) for s in li.steps] == [
        (1, "Initial Conditions"), (2, "CPU Fan Screw 1"), (3, "CPU Fan Screw 2"),
        (4, "CPU fan"), (5, "Motherboard")]
    assert not any(s.step_type == "dupli" or s.dupli for s in li.steps)
    # a dupli row's note stays, on the step it repeats; the others keep theirs
    assert li.steps[1].notes == "(dupli row 3) Scanner rescan"
    assert li.steps[2].notes == "glued"
    # actions follow the new numbers
    assert sorted({a.step for a in li.actions}) == [2, 3, 4, 5]
    # the dupli rows' sequence numbers still count: nothing reported "skipped"
    assert not [i for i in li.issues if "skipped" in i]


def test_dupli_rows_of_and_sheet_rows(tmp_path):
    path = tmp_path / "desktop_90.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Sequence Number", "Sequence Name", "Target Nest Group",
                         "Tool Utility", "Notes"])
        for i, name in enumerate(SHEET, start=1):
            writer.writerow([i, name, "", "", "Scanner rescan" if i == 3 else ""])
    rows, _meta = read_desktop_csv(path)
    assert dupli_rows_of(rows) == [3, 6]
    sheets = sheet_rows_by_desktop(tmp_path)
    assert sheets[90].n_rows == 7 and sheets[90].dupli_rows == (3, 6)
    assert sheets[90].notes[2] == "Scanner rescan" and sheets[90].names[5].startswith("repl")
    assert sheet_rows_by_desktop(tmp_path / "nowhere") == {}


@pytest.mark.parametrize("desktop", [2, 3, 4, 5, 13, 63])
def test_a_sheet_without_dupli_rows_keeps_one_step_per_row(tax, desktop):
    """D2 and the other fixtures have no dupli row: nothing is renumbered."""
    rows, meta = read_desktop_csv(LOG_FIXTURES / f"desktop_{desktop:02d}.csv")
    li = import_log(desktop, rows, meta, tax)
    assert li.dupli_rows == []
    assert [s.step for s in li.steps] == list(range(1, len(rows) + 1))
    assert [s.raw_name for s in li.steps] == [r["Sequence Name"].strip() for r in rows]


def test_reimporting_a_desktop_without_dupli_rows_leaves_its_steps_identical(env):
    """D13 (no dupli row): import-logs, then import-logs --force -- same table."""
    assert run(env, "load-index") == 0
    assert run(env, "import-logs", "--desktops", "13") == 0
    db = open_db(env)
    try:
        before = [(s.step, s.step_type, s.raw_name, s.notes, s.duration_s) for s in db.steps(13)]
    finally:
        db.close()
    assert run(env, "import-logs", "--desktops", "13", "--force") == 0
    db = open_db(env)
    try:
        after = [(s.step, s.step_type, s.raw_name, s.notes, s.duration_s) for s in db.steps(13)]
    finally:
        db.close()
    assert after == before and before


# --------------------------------------------------------------------------- #
# the index's side of import-logs
# --------------------------------------------------------------------------- #
def _collapsed_index(desktop: int, n_rows: int, dupli: list[int]) -> DesktopIndex:
    """oak1 only: one capture per row, a minute apart; dupli captures as extra shots."""
    base = datetime(2025, 1, 1, 10, 0, 0)
    frames, step = {}, 0
    for r in range(1, n_rows + 1):
        ts = (base + timedelta(minutes=r)).isoformat()
        if r in dupli:
            held = frames[FrameKey(desktop, step, "oak1")]
            held.aux.setdefault("extra_shots", []).append(
                {"path": f"F:/oak/{r}.jpg", "src_step_dir": f"{r:03d}", "ts": ts, "row": r})
            continue
        step += 1
        key = FrameKey(desktop, step, "oak1")
        frames[key] = FrameFile(key, f"F:/oak/{r}.jpg", {}, ts, f"{r:03d}")
    return DesktopIndex(desktop, step, frames, [], [], n_rows=n_rows,
                        rows=[f"{r:03d}" for r in range(1, n_rows + 1)], dupli_rows=dupli)


def test_durations_start_at_the_dupli_capture_of_the_step_before():
    di = _collapsed_index(90, 7, [3, 6])
    steps = [StepRec(90, k, "normal", f"s{k}") for k in range(1, 6)]
    filled, issues = _apply_index(steps, di, [3, 6])
    # step 3 is row 4: from the dupli capture (row 3, 10:03) to row 4 (10:04)
    assert steps[2].duration_s == 60.0
    assert steps[1].duration_s == 60.0     # row 2 - row 1
    assert not issues and filled == 4


def test_index_built_without_the_sheet_is_reported():
    di = _collapsed_index(90, 7, [])
    steps = [StepRec(90, k, "normal", f"s{k}") for k in range(1, 6)]
    _filled, issues = _apply_index(steps, di, [3, 6])
    assert any("dupli rows [3, 6] are not the ones the index was collapsed by (none)" in i
               for i in issues)


# --------------------------------------------------------------------------- #
# load-index: work follows its photo
# --------------------------------------------------------------------------- #
def _index_file(path: Path, desktop: int, rows_to_steps: dict[int, int],
                extra: dict[int, int] | None = None) -> str:
    """Every view photographs row r at step rows_to_steps[r]; ``extra`` rows are
    extra shots of the step they map to."""
    extra = extra or {}
    frames = {}
    n_steps = max(rows_to_steps.values())
    for view in VIEWS:
        for r, step in sorted(rows_to_steps.items()):
            key = FrameKey(desktop, step, view)
            frames[key] = FrameFile(key, f"F:/{view}/{desktop}/r{r}.png", {}, None, f"{r:03d}")
        for r, step in sorted(extra.items()):
            frames[FrameKey(desktop, step, view)].aux.setdefault("extra_shots", []).append(
                {"path": f"F:/{view}/{desktop}/r{r}.png", "src_step_dir": f"{r:03d}",
                 "ts": None, "row": r})
    di = DesktopIndex(desktop, n_steps, frames, [], [],
                      n_rows=len(rows_to_steps) + len(extra),
                      rows=[f"{r:03d}" for r in sorted(set(rows_to_steps) | set(extra))],
                      dupli_rows=sorted(extra))
    save_index({desktop: di}, str(path))
    return str(path)


def _keyframe(db, desktop, view, instance, anchor, source="manual"):
    mask = np.zeros((8, 8), bool)
    mask[2:4, 2:4] = True
    return db.add_keyframe(ShapeKeyframe(
        None, instance, desktop, view, 1, anchor, source=source,
        parts=[ShapePart("main", encode_rle(mask))]))


def test_load_index_moves_keyframes_and_flags_with_their_photos(env, tmp_path):
    old = _index_file(tmp_path / "old.json", 95, {r: r for r in range(1, 7)})
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        moved = _keyframe(db, 95, "scan", "chassis", 4)
        on_dupli = _keyframe(db, 95, "oak1", "ls:RAM#1", 3, source="labelstudio")
        stays = _keyframe(db, 95, "oak1", "psu.01", 2)
        beyond = _keyframe(db, 95, "scan", "motherboard.01", 6)
        db.set_frame_flags(FrameKey(95, 4, "scan"), review_status="ls_draft")
        db.set_frame_flags(FrameKey(95, 2, "oak1"), review_status="ls_draft")
    finally:
        db.close()
    # the collapse: row 3 is a dupli of row 2; rows 4..6 become steps 3..5
    new = _index_file(tmp_path / "new.json", 95, {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}, extra={3: 2})
    assert run(env, "load-index", "--index", new) == 0
    db = open_db(env)
    try:
        anchors = {kf.id: kf.anchor_step for view in ("scan", "oak1")
                   for kf in db.keyframes(95, view)}
        assert anchors[moved] == 3 and anchors[on_dupli] == 2 and anchors[stays] == 2
        assert anchors[beyond] == 5
        assert db.get_frame(FrameKey(95, 3, "scan"))["review_status"] == "ls_draft"
        assert db.get_frame(FrameKey(95, 4, "scan"))["review_status"] in (None, "unlabeled")
        # the step that only gained an extra shot keeps its own row and flags
        assert db.get_frame(FrameKey(95, 2, "oak1"))["review_status"] == "ls_draft"
        assert db.get_frame(FrameKey(95, 3, "scan"))["path"] == "F:/scan/95/r4.png"
        assert db.get_frame(FrameKey(95, 6, "scan")) is None   # past the new last step
        assert {r["end_step"] for r in db.pose_segments(95, "scan")} == {5}
    finally:
        db.close()


def test_load_index_refuses_to_renumber_a_verified_frame(env, tmp_path, capsys):
    old = _index_file(tmp_path / "old.json", 95, {r: r for r in range(1, 7)})
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        db.set_frame_flags(FrameKey(95, 5, "oak1"), review_status="verified")
    finally:
        db.close()
    new = _index_file(tmp_path / "new.json", 95, {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}, extra={3: 2})
    assert run(env, "load-index", "--index", new) == 1
    assert "verified frames" in capsys.readouterr().out
    db = open_db(env)
    try:   # nothing was written for the desktop
        assert db.get_frame(FrameKey(95, 6, "oak1"))["path"] == "F:/oak1/95/r6.png"
    finally:
        db.close()


def test_a_view_whose_photos_stay_still_loses_its_rows_past_the_new_last_step(env, tmp_path):
    """D24's scanner: no photo moves, but its old last step was a missing frame of a
    step the collapse removed -- that row (and anything keyed to it) must not stay."""
    old = _index_file(tmp_path / "old.json", 95, {r: r for r in range(1, 7)})
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        # the old last scanner step had no photo (a missing frame), as on D24
        db.upsert_frame(FrameKey(95, 6, "scan"), None, {}, None, {"missing": True})
        kf = _keyframe(db, 95, "scan", "chassis", 6)
    finally:
        db.close()
    new = _index_file(tmp_path / "new.json", 95, {r: r for r in range(1, 6)})
    assert run(env, "load-index", "--index", new) == 0
    db = open_db(env)
    try:
        for view in VIEWS:
            assert db.get_frame(FrameKey(95, 6, view)) is None, view
            assert db.get_frame(FrameKey(95, 5, view))["path"] == f"F:/{view}/95/r5.png"
        assert [k.anchor_step for k in db.keyframes(95, "scan") if k.id == kf] == [5]
    finally:
        db.close()


def test_work_on_a_photo_that_left_the_index_is_refused_not_moved(env, tmp_path, capsys):
    """N3: the new index no longer has photo r6 at all; a shape drawn on it would
    land on another photo, so the desktop is refused and nothing is written. The
    same photo with no work on it is simply dropped."""
    old = _index_file(tmp_path / "old.json", 95, {r: r for r in range(1, 7)})
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        kf = _keyframe(db, 95, "scan", "chassis", 6)
        before = db.conn.execute("SELECT COUNT(*) FROM frame WHERE desktop=95").fetchone()[0]
    finally:
        db.close()
    new = _index_file(tmp_path / "new.json", 95, {r: r for r in range(1, 6)})
    capsys.readouterr()
    assert run(env, "load-index", "--index", new) == 1
    out = capsys.readouterr().out
    assert "no longer has in their view" in out and "scan step 6" in out
    assert "re-run build-index" not in out.split("refused because work")[1]
    db = open_db(env)
    try:
        assert db.conn.execute("SELECT COUNT(*) FROM frame WHERE desktop=95").fetchone()[0] \
            == before
        assert [k.anchor_step for k in db.keyframes(95, "scan") if k.id == kf] == [6]
        db.conn.execute("DELETE FROM shape_keyframe WHERE id=?", (kf,))
        db.conn.commit()
    finally:
        db.close()
    assert run(env, "load-index", "--index", new) == 0
    db = open_db(env)
    try:
        for view in VIEWS:
            assert db.get_frame(FrameKey(95, 6, view)) is None, view
    finally:
        db.close()


def test_load_index_of_an_unchanged_index_moves_nothing(env, tmp_path, capsys):
    old = _index_file(tmp_path / "old.json", 95, {r: r for r in range(1, 5)})
    assert run(env, "load-index", "--index", old) == 0
    db = open_db(env)
    try:
        kf = _keyframe(db, 95, "scan", "chassis", 4)
    finally:
        db.close()
    capsys.readouterr()
    assert run(env, "load-index", "--index", old) == 0
    assert "photo(s) moved step" not in capsys.readouterr().out
    db = open_db(env)
    try:
        assert [k.anchor_step for k in db.keyframes(95, "scan") if k.id == kf] == [4]
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# the local image cache follows a renumbered step
# --------------------------------------------------------------------------- #
def test_build_cache_recopies_a_scanner_step_whose_burst_changed(tmp_path):
    """D18's step 42 was RGB421 and is RGB431: same cache name, other burst. The
    size check alone kept the old copy (equal-sized PNGs); the burst check does not."""
    from test_cache import _scan_index

    from tda.core.cache import build_cache, cache_path

    index = _scan_index(tmp_path, desktop=7, steps=(1, 2, 3))
    cache = str(tmp_path / "cache")
    build_cache(index, cache, views=("scan",))
    frames = index[7].frames
    # the collapse: step 2 now shows folder RGB31's burst, step 3 is gone
    moved = frames[FrameKey(7, 3, "scan")]
    frames[FrameKey(7, 2, "scan")] = FrameFile(FrameKey(7, 2, "scan"), moved.path,
                                               dict(moved.aux), None, "RGB31")
    del frames[FrameKey(7, 3, "scan")]
    index[7].n_steps = 2
    stats = build_cache(index, cache, views=("scan",))
    assert stats["copied"] == 1 and stats["steps"] == 2
    manifest = stats["manifests"]["scan"][7]
    assert "RGB31" in manifest["2"]["src"] and "RGB11" in manifest["1"]["src"]
    want = open(manifest["2"]["src"], "rb").read()
    assert open(cache_path(cache, FrameKey(7, 2, "scan"), "png"), "rb").read() == want


# --------------------------------------------------------------------------- #
# Label Studio drafts land on the step their photo is now
# --------------------------------------------------------------------------- #
def test_import_ls_puts_a_draft_on_its_folders_collapsed_step(env, tmp_path):
    """ls_small.json has Rs_24_35.png: RealSense folder 035. With a dupli row 10
    the folder is row 35, step 34; ``--desktops 24`` leaves D19 alone."""
    index = _index_file(tmp_path / "idx.json", 24,
                        {r: (r if r < 10 else r - 1) for r in range(1, 44) if r != 10},
                        extra={10: 9})
    db = open_db(env)
    try:
        for desktop in (19, 24):
            db.replace_steps(desktop, [StepRec(desktop, k, StepType.NORMAL.value, f"row {k}")
                                       for k in range(1, 60)], [])
    finally:
        db.close()
    assert run(env, "import-ls", "--export", str(LS_EXPORT), "--index", index,
               "--desktops", "24") == 0
    summary = json.loads((env["cache"] / "ls_import_summary.json").read_text(encoding="utf-8"))
    assert summary["frames_renumbered"] == 1
    db = open_db(env)
    try:
        assert {kf.anchor_step for kf in db.keyframes(24, "rs")} == {34}
        assert not db.keyframes(19, "scan")
    finally:
        db.close()
