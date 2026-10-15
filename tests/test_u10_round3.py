"""Task U10, review round 3: what the second review found.

* a measured size is of a photo, not of a frame key: it goes when the row
  names another file or turns missing (an inferred one stays);
* ``import-logs --adopt-extras`` takes the removal 「拆下于第 k 步」 wrote along
  with the initial state, and its undo puts both back;
* ``cli build-cache --db`` is the database the gallery is read from;
* a version-1 index is refused by ``load-index``;
* 「拆下于第 k 步」 refuses the initial frame and a ``dupli`` frame;
* ``cli check`` reports a sheet shot whose sheet row was retyped.

(The gallery's refusal of ``Shift+T`` is a window test, in ``test_u10_app``.)
"""
from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from tda import pipeline as P
from tda.core import extra as X
from tda.core.cache import VIEW_EXT, cache_path
from tda.core.db import Db
from tda.core.index import load_fixes, load_index, save_index, scan_desktop
from tda.core.model import FrameKey, StateEvent, StepRec
from tda.core.taxonomy import load_taxonomy
from tda.core.truth_inputs import TRUTH_AUX_KEYS, frame_hw, infer_hw
from tda.pipeline_logs import import_logs_into_db
from tda.ui.steps_model import EditError, StepTableData

import gallery_scene as GS
from steps_fixtures import seeded_db
from test_extra_instances_r2 import EXTRAS, _sheet
from test_index import OAK_ROLES, _touch
from test_u10_gallery import TS, _run, sequence_tree

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATOR = "chang"


@pytest.fixture
def fixes() -> dict:
    return load_fixes(str(REPO_ROOT / "configs" / "index_fixes.yaml"))


# --------------------------------------------------------------------------- #
# 1. a measured size goes with its photo
# --------------------------------------------------------------------------- #
def _cam2(tmp_path: Path, folder: str) -> Path:
    return tmp_path / "oak" / "Desktop 99" / "Disassemble" / "Camera_2" / folder


def _photo(tmp_path: Path, folder: str, token: str) -> None:
    for role in OAK_ROLES:
        _touch(_cam2(tmp_path, folder) / f"{token}_camera_2_{role}")


def _measure(db: Db, key: FrameKey) -> None:
    """What ``frame_hw`` leaves on a row it measured."""
    row = db.get_frame(key)
    db.upsert_frame(key, row["path"], dict(row["aux"] or {}, hw=[9, 9], hw_source="measured",
                                           cache_path="D:/cache/old.jpg"),
                    row["ts"], {"missing": bool(row.get("missing"))})


def _truth_keys(db: Db, key: FrameKey) -> set:
    return set(TRUTH_AUX_KEYS) & set(db.get_frame(key)["aux"] or {})


def test_present_missing_present_with_another_photo_measures_again(tmp_path, fixes):
    """The reviewer's scene: the old photo's size used to come back with the new one."""
    roots = sequence_tree(tmp_path)
    db = Db(str(tmp_path / "tda.sqlite"))

    def load() -> None:
        assert not P.load_index_into_db(db, {99: scan_desktop(99, roots, fixes)})["failed"]

    load()
    key = FrameKey(99, 2, "oak2")
    first = db.get_frame(key)["path"]
    _measure(db, key)
    load()                                       # same photo: the measurement stays
    assert db.get_frame(key)["aux"]["hw_source"] == "measured"
    shutil.rmtree(_cam2(tmp_path, "002"))
    load()                                       # the photo is gone
    row = db.get_frame(key)
    assert row["missing"] and not _truth_keys(db, key)
    _photo(tmp_path, "012", TS[1])               # Camera_2 pairs by capture time
    load()                                       # another photo at that step
    row = db.get_frame(key)
    assert not row["missing"] and row["path"] != first
    assert not _truth_keys(db, key)
    # so the next compile decides the size afresh (nothing to measure here: inferred)
    assert frame_hw(db, key) == infer_hw("oak2")
    assert db.get_frame(key)["aux"]["hw_source"] == "inferred"


def test_another_photo_at_a_present_step_drops_the_old_measurement(tmp_path, fixes):
    """All views, not only the gallery: the key stays, the file changes."""
    roots = sequence_tree(tmp_path)
    db = Db(str(tmp_path / "tda.sqlite"))
    P.load_index_into_db(db, {99: scan_desktop(99, roots, fixes)})
    key = FrameKey(99, 2, "oak2")
    _measure(db, key)
    shutil.rmtree(_cam2(tmp_path, "002"))
    _photo(tmp_path, "012", TS[1])               # Camera_2 pairs by capture time
    P.load_index_into_db(db, {99: scan_desktop(99, roots, fixes)})
    assert not db.get_frame(key)["missing"] and not _truth_keys(db, key)


def test_a_row_turning_missing_keeps_only_an_inferred_size(tmp_path, fixes):
    roots = sequence_tree(tmp_path)
    db = Db(str(tmp_path / "tda.sqlite"))
    P.load_index_into_db(db, {99: scan_desktop(99, roots, fixes)})
    measured, inferred = FrameKey(99, 2, "oak2"), FrameKey(99, 3, "oak2")
    _measure(db, measured)
    row = db.get_frame(inferred)
    db.upsert_frame(inferred, row["path"], dict(row["aux"], hw=[3040, 4032],
                                                hw_source="inferred"), row["ts"])
    shutil.rmtree(_cam2(tmp_path, "002"))
    shutil.rmtree(_cam2(tmp_path, "003"))
    P.load_index_into_db(db, {99: scan_desktop(99, roots, fixes)})
    assert db.get_frame(measured)["missing"] and db.get_frame(measured)["aux"] == {}
    assert db.get_frame(inferred)["aux"] == {"hw": [3040, 4032], "hw_source": "inferred"}


# --------------------------------------------------------------------------- #
# 2. --adopt-extras takes the removal step too
# --------------------------------------------------------------------------- #
def test_adopt_extras_takes_the_removal_step_with_the_addition(tmp_db_path, tmp_path):
    tax = load_taxonomy()
    db = seeded_db(tmp_db_path, tax, desktops=(13,))
    try:
        data = StepTableData.load(db, 13, tax, ANNOTATOR)
        assert data.add_extras("ram_latch", 4, state="open", removed_at=30) == list(EXTRAS)
        data.save(db)
        added = [e for e in db.events(13) if e.evidence_view == X.ADDED_EVIDENCE]
        assert len(added) == 8 and {e.target for e in added} == set(EXTRAS)
        run = import_logs_into_db(db, str(_sheet(tmp_path, clips=4)), tax, desktops={13},
                                  force=True, adopt_extras=True)
        assert run.runs[0].status == "imported", run.runs[0].issues
        # the sheet's clips leave with the board, by the sheet's own steps
        assert not [e for e in db.events(13) if e.target in EXTRAS and not e.auto]
        op = [o for o in db.ops(13, X.OP_VIEW) if o["kind"] == X.OP_ADOPT][-1]
        assert len(op["inverse"]["events"]) == 4 + 8
        X.undo_op(db, op, ANNOTATOR, tax)
        back = [e for e in db.events(13) if e.evidence_view == X.ADDED_EVIDENCE]
        assert sorted((e.step, e.target, e.attr) for e in back) == sorted(
            (e.step, e.target, e.attr) for e in added)
    finally:
        db.close()


def test_the_adoption_delete_keeps_every_other_hand_set_state(tmp_path):
    db = Db(str(tmp_path / "tda.sqlite"))
    events = [StateEvent(5, 0, "clip.01", "state", None, "open", "s1:initial", False),
              StateEvent(5, 30, "clip.01", "state", "open", "removed", X.ADDED_EVIDENCE, False),
              StateEvent(5, 12, "clip.01", "state", "closed", "open", "assert:oak1", False),
              StateEvent(5, 30, "clip.01", "state", "open", "removed", None, True)]
    db.add_events(events)
    assert db.delete_manual_events(5, "clip.01", up_to_step=0,
                                   evidence_view=X.ADDED_EVIDENCE) == 2
    assert [(e.step, e.evidence_view, e.auto) for e in db.events(5)] == [
        (12, "assert:oak1", False), (30, None, True)]


# --------------------------------------------------------------------------- #
# 3. build-cache --db
# --------------------------------------------------------------------------- #
def test_cli_build_cache_reads_the_gallery_from_the_db_it_is_given(tmp_path, monkeypatch,
                                                                   capsys):
    env = GS.make_env(tmp_path)
    other = tmp_path / "elsewhere" / "other.sqlite"
    other.parent.mkdir()
    Db(str(other)).close()
    seen: list[list[str]] = []
    monkeypatch.setattr("tda.core.cache.main", lambda argv: seen.append(list(argv)) or 0)
    assert _run(env, "build-cache", "--views", "components", "--desktops", "13",
                "--db", str(other)) == 0
    assert seen[-1][seen[-1].index("--db") + 1] == str(other)
    # a --db that is not there is said, not silently replaced by paths.yaml's
    assert _run(env, "build-cache", "--views", "components", "--desktops", "13",
                "--db", str(tmp_path / "nope.sqlite")) == 0
    assert "--db" not in seen[-1]
    assert "no database at" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 4. a version-1 index is refused by the command too
# --------------------------------------------------------------------------- #
def test_cli_load_index_refuses_a_version_1_index(tmp_path, fixes, capsys):
    env = GS.make_env(tmp_path)
    old = {13: scan_desktop(13, sequence_tree(tmp_path / "bare", desktop=13), fixes)}
    path = tmp_path / "old_index.json"
    save_index(old, str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_index(str(path))[13].gallery is False
    db = Db(env["cfg"]["db_path"])
    before = {v: db.frames_for(13, v) for v in ("scan", "cmp_oak1")}
    db.close()
    capsys.readouterr()
    assert _run(env, "load-index", "--index", str(path)) == 1
    out = capsys.readouterr().out
    assert "version-1 index" in out and "Run build-index first" in out
    db = Db(env["cfg"]["db_path"])
    try:
        assert {v: db.frames_for(13, v) for v in before} == before
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 5. 「拆下于第 k 步」: not the initial frame, not a dupli
# --------------------------------------------------------------------------- #
STEPS = [StepRec(1, 1, "initial", "Initial"), StepRec(1, 2, "normal", "Cover"),
         StepRec(1, 3, "dupli", "dupli", dupli=True), StepRec(1, 4, "normal", "Fan"),
         StepRec(1, 5, "dupli", "dupli", dupli=True)]


def test_the_removal_step_refusals_say_where_to_go():
    assert X.removal_steps(STEPS) == [2, 4]
    assert X.removal_step_refusal(2, STEPS) == ""
    initial = X.removal_step_refusal(1, STEPS)
    assert "起始帧" in initial and "最早填第 2 步" in initial
    dupli = X.removal_step_refusal(3, STEPS)
    assert "重复帧（dupli）" in dupli and "和第 2 步是同一个样子" in dupli
    assert "还在就填第 4 步" in dupli and "step 3 duplicates step 2" in dupli
    assert "还在就填" not in X.removal_step_refusal(5, STEPS)      # nothing after it
    assert "没有第 9 步" in X.removal_step_refusal(9, STEPS)


def _d13_data(tmp_path):
    db, tax, _cache = GS.make_db(tmp_path)
    return db, StepTableData.load(db, GS.DESKTOP, tax)


def test_taken_out_at_the_initial_frame_is_refused(tmp_path):
    _db, data = _d13_data(tmp_path)
    assert data.row(1).step_type == "initial"
    with pytest.raises(EditError, match="起始帧"):
        data.add_extras("misc_part", 1, parent=None, removed_at=1)
    assert data.add_extras("misc_part", 1, parent=None, removed_at=2)


def test_taken_out_at_a_dupli_frame_is_refused_and_never_the_default(tmp_path):
    _db, data = _d13_data(tmp_path)
    last = data.last_step()
    row = data.row(last)
    row.step = dataclasses.replace(row.step, step_type="dupli", dupli=True)
    with pytest.raises(EditError, match=f"和第 {last - 1} 步是同一个样子"):
        data.add_extras("misc_part", 1, parent=None, removed_at=last)
    defaults = data.extra_defaults("misc_part")
    assert defaults["removed_at"] == last - 1 and defaults["last_step"] == last - 1
    assert data.frames_demoted_by_extras(_db, "scan", "misc_part", 1, parent=None,
                                         removed_at=last) == []


# --------------------------------------------------------------------------- #
# 6. a sheet shot whose sheet row was retyped
# --------------------------------------------------------------------------- #
def test_a_sheet_shot_without_its_row_is_reported(tmp_path):
    db = Db(str(tmp_path / "tda.sqlite"))
    db.replace_steps(66, [StepRec(66, 3, "ignore", "All components")], [])
    shot = FrameKey(66, P.SHEET_SHOT_BASE + 3, "cmp_oak1")
    db.upsert_frame(shot, "F:/D66/003.jpg", {"sheet_step": 3}, None)
    assert P.orphan_sheet_shots(db, 66, "cmp_oak1") == []
    db.replace_steps(66, [StepRec(66, 3, "normal", "All components")], [])
    [text] = P.orphan_sheet_shots(db, 66, "cmp_oak1")
    assert text.startswith(f"{P.ORPHAN_SHEET_SHOT}:cmp_oak1 shot 1003 (sheet step 3")


def test_cli_check_lists_it_as_a_problem(tmp_path, capsys):
    env = GS.make_env(tmp_path)
    db = Db(env["cfg"]["db_path"])
    key = FrameKey(GS.DESKTOP, P.SHEET_SHOT_BASE + 13, "cmp_oak1")
    path = cache_path(env["cfg"]["cache_dir"], key, VIEW_EXT["cmp_oak1"])
    cv2.imwrite(path, np.full((*GS.HW, 3), 90, dtype=np.uint8))
    # step 13 of D13 is an ordinary step: as if its row had been retyped
    db.upsert_frame(key, path, {"cache_path": path, "hw": list(GS.HW), "sheet_step": 13}, None)
    P.settle_gallery(db, GS.DESKTOP)
    db.close()
    capsys.readouterr()
    assert _run(env, "check", "--desktop", str(GS.DESKTOP), "--view", "cmp_oak1") == 1
    out = capsys.readouterr().out
    assert f"{P.ORPHAN_SHEET_SHOT}:cmp_oak1 shot 1013 (sheet step 13" in out
