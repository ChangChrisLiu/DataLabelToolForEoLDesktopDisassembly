"""Task U10 review round: the fixes that must be in before the first live load-index.

B1 RealSense shots in capture-time order; B2 a frame with shapes or LS drafts is
never dropped; B3 load-index never moves work on a gallery shot, and a sheet
row's photo keeps a number no folder can reach; M2 a shape drawn on a part
marked 「看不见」 takes the label back; M6 build-cache reads the gallery from the
database; M7 a re-run says nothing moved; M9 the RealSense frames of D1/D24 filed
under Components fill their steps; and parts the log never names that were
taken out (「拆下于第 k 步」) join the gallery.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from tda import pipeline as P
from tda.core import extra as X
from tda.core.cache import _source_moved, gallery_frames_from_db
from tda.core.db import Db
from tda.core.index import FrameFile, load_fixes, load_index, save_index, scan_desktop
from tda.core.model import (
    VIEWS,
    FrameKey,
    FrameOverride,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    StepRec,
)
from tda.core.states import needs_geom
from tda.core.taxonomy import load_taxonomy
from tda.core.truth_inputs import gallery_state_of, state_of
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui.steps_model import StepTableData

import gallery_scene as GS
from test_index import OAK_ROLES, _epoch, _touch, build_tree
from test_u10_gallery import SHOT_TS, TS, add_components, sequence_tree

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixes() -> dict:
    return load_fixes(str(REPO_ROOT / "configs" / "index_fixes.yaml"))


def _box_keyframe(desktop: int, view: str, seg: int, anchor: int,
                  instance: str = "psu.01", source: str = "manual") -> ShapeKeyframe:
    return ShapeKeyframe(None, instance, desktop, view, seg, anchor,
                         parts=[ShapePart("main", box=(1, 2, 3, 4))], source=source)


# --------------------------------------------------------------------------- #
# B1: RealSense shots by capture time
# --------------------------------------------------------------------------- #
def test_realsense_shots_follow_capture_time_not_folder_name(tmp_path, fixes):
    """D6/D7/D10/D12/D18: folder 005 holds the first layout photo."""
    roots = sequence_tree(tmp_path)
    times = {"001": "20250101_103100_000", "002": "20250101_103200_000",
             "003": "20250101_103300_000", "005": "20250101_103000_000"}
    add_components(tmp_path, 99, {}, rs=times)
    idx = scan_desktop(99, roots, fixes)
    order = [idx.frames[FrameKey(99, k, "cmp_rs")].src_step_dir for k in (1, 2, 3, 4)]
    assert order == ["Components/005", "Components/001", "Components/002", "Components/003"]


def test_equal_times_fall_back_to_the_folder_name(tmp_path, fixes):
    roots = sequence_tree(tmp_path)
    add_components(tmp_path, 99, {}, rs={"002": SHOT_TS[0], "001": SHOT_TS[0]})
    idx = scan_desktop(99, roots, fixes)
    assert idx.frames[FrameKey(99, 1, "cmp_rs")].src_step_dir == "Components/001"


# --------------------------------------------------------------------------- #
# M9: rs_extra_from fills the sequence's last RealSense frame
# --------------------------------------------------------------------------- #
def test_rs_extra_from_fills_the_missing_step(tmp_path, fixes):
    roots = build_tree(tmp_path, 99,
                       oak1={f"{i + 1:03d}": [TS[i]] for i in range(3)},
                       oak2={f"{i + 1:03d}": [TS[i]] for i in range(3)},
                       rs_steps=[1, 2], rs_mtimes={1: TS[0], 2: TS[1]})
    add_components(tmp_path, 99, {}, rs={"001": "20250101_100204_000",
                                         "002": SHOT_TS[0]})
    fixes = dict(fixes)
    fixes["rs"] = {**fixes["rs"], 99: {"rs_extra_from": "Components/001",
                                       "as_step": 3, "reason": "test"}}
    idx = scan_desktop(99, roots, fixes)
    filled = idx.frames[FrameKey(99, 3, "rs")]
    assert filled.src_step_dir == "Components/001"
    assert FrameKey(99, 3, "rs") not in idx.missing
    assert any("fix:rs[99] rs_extra_from Components/001 -> logical step 3" in i
               for i in idx.issues)
    # the folder is the sequence's, so the gallery starts at the next one
    assert [idx.frames[FrameKey(99, k, "cmp_rs")].src_step_dir
            for k in (1,)] == ["Components/002"]
    assert FrameKey(99, 2, "cmp_rs") not in idx.frames


def test_rs_extra_from_on_a_filled_step_is_a_mismatch(tmp_path, fixes):
    roots = sequence_tree(tmp_path)
    add_components(tmp_path, 99, {}, rs={"001": SHOT_TS[0]})
    fixes = dict(fixes)
    fixes["rs"] = {**fixes["rs"], 99: {"rs_extra_from": "Components/001",
                                       "as_step": 3, "reason": "test"}}
    idx = scan_desktop(99, roots, fixes)
    assert idx.frames[FrameKey(99, 3, "rs")].src_step_dir == "003"
    assert any("MISMATCH" in i and "rs_extra_from" in i for i in idx.issues)


def test_the_d1_and_d24_fixes_are_declared(fixes):
    assert fixes["rs"][1] == {"rs_extra_from": "Components/001", "as_step": 40,
                              "reason": fixes["rs"][1]["reason"]}
    assert fixes["rs"][24]["rs_extra_from"] == "Components/043"
    assert fixes["rs"][24]["as_step"] == 43


# --------------------------------------------------------------------------- #
# B2: what counts as somebody's work on a frame
# --------------------------------------------------------------------------- #
def _frame_db(tmp_path) -> tuple[Db, FrameKey]:
    db = Db(str(tmp_path / "tda.sqlite"))
    key = FrameKey(66, 3, "oak1")
    db.upsert_frame(key, "F:/x.png", {}, None)
    return db, key


@pytest.mark.parametrize("source", ["manual", "labelstudio"])
def test_a_shape_anchored_on_the_frame_is_work(tmp_path, source):
    db, key = _frame_db(tmp_path)
    db.add_keyframe(_box_keyframe(66, "oak1", 1, 3, source=source))
    assert db.frame_is_annotated(key)
    assert "1 shape keyframe(s) anchored here" in db.frame_work(key)
    assert not db.delete_unannotated_frame(key) and db.get_frame(key) is not None
    # a shape anchored on another step of the view is not on this frame
    other = FrameKey(66, 2, "oak1")
    db.upsert_frame(other, "F:/y.png", {}, None)
    assert not db.frame_is_annotated(other)


def test_an_ls_draft_frame_is_work(tmp_path):
    db, key = _frame_db(tmp_path)
    db.set_frame_flags(key, review_status="ls_draft")
    assert db.frame_is_annotated(key)
    assert not db.delete_unannotated_frame(key)


def test_an_unconfirmed_compiled_row_is_a_cache_and_goes_with_the_frame(tmp_path):
    from tda.core import masks
    import numpy as np

    db, key = _frame_db(tmp_path)
    rle = masks.encode_rle(np.ones((4, 4), dtype=bool))
    db.put_compiled(key, "psu.01", rle, 0.0, "visible", "in_chassis", "auto", "h")
    assert not db.frame_is_annotated(key)
    assert db.delete_unannotated_frame(key)
    assert db.get_frame(key) is None and not db.compiled(key)


def test_settle_gallery_never_orphans_a_shape(tmp_path):
    db = Db(str(tmp_path / "tda.sqlite"))
    for view in VIEWS:
        db.upsert_frame(FrameKey(66, 3, view), f"F:/x/{view}.png", {}, None)
    db.replace_steps(66, [StepRec(66, 3, "ignore", "All components")], [])
    db.add_keyframe(_box_keyframe(66, "oak1", 1, 3, source="labelstudio"))
    out = P.settle_gallery(db, 66)
    assert db.get_frame(FrameKey(66, 3, "oak1")) is not None      # kept, reported
    assert any("oak1" in line and "has annotations" in line for line in out["moved"])
    assert db.get_frame(FrameKey(66, 3, "oak2")) is None           # the others moved


# --------------------------------------------------------------------------- #
# B3: load-index never moves work on a gallery shot
# --------------------------------------------------------------------------- #
def _gallery_tree(tmp_path, n: int = 3) -> dict:
    roots = sequence_tree(tmp_path)
    tokens = [f"20250101_10{30 + i}00_000" for i in range(n)]
    add_components(tmp_path, 99, {"Camera_1": tokens})
    return roots


def _load(db, roots, fixes, log=None) -> dict:
    return P.load_index_into_db(db, {99: scan_desktop(99, roots, fixes)}, log=log)


def _drop_folder(tmp_path, name: str) -> None:
    shutil.rmtree(tmp_path / "oak" / "Desktop 99" / "Components" / "Camera_1" / name)


def _paths(db) -> dict[int, str]:
    return {int(r["step"]): r["path"].rsplit("/", 1)[-1][:19]
            for r in db.frames_for(99, "cmp_oak1")}


def test_a_shot_removed_under_work_fails_the_desktop(tmp_path, fixes):
    db = Db(str(tmp_path / "tda.sqlite"))
    roots = _gallery_tree(tmp_path)
    _load(db, roots, fixes)
    before = _paths(db)
    db.add_keyframe(_box_keyframe(99, "cmp_oak1", 2, 2))      # work on shot 2
    _drop_folder(tmp_path, "001")                             # shots 2, 3 move to 1, 2
    counts = _load(db, roots, fixes)
    assert len(counts["failed"]) == 1
    text = counts["failed"][0]
    assert "GalleryConflict" in text and "cmp_oak1 shot 2" in text
    assert "shape keyframe" in text
    assert _paths(db) == before                               # nothing was written


def test_unannotated_shots_are_re_pointed_and_a_stale_tail_dropped(tmp_path, fixes):
    db = Db(str(tmp_path / "tda.sqlite"))
    roots = _gallery_tree(tmp_path)
    _load(db, roots, fixes)
    before = _paths(db)
    key = FrameKey(99, 1, "cmp_oak1")
    row = db.get_frame(key)
    db.upsert_frame(key, row["path"], dict(row["aux"], hw=[9, 9], hw_source="measured"),
                    row["ts"])
    _drop_folder(tmp_path, "001")
    lines: list[str] = []
    counts = _load(db, roots, fixes, log=lines.append)
    assert not counts["failed"]
    assert _paths(db) == {1: before[2], 2: before[3]}
    assert [s["seg"] for s in db.pose_segments(99, "cmp_oak1")] == [1, 2]
    # the size measured on the old photo of shot 1 is not the new photo's
    assert "hw" not in db.get_frame(key)["aux"]
    assert any("now on another photo" in l for l in lines)
    assert any("dropped: cmp_oak1 shot 3" in l for l in lines)


def test_a_stale_tail_with_work_fails_the_desktop(tmp_path, fixes):
    db = Db(str(tmp_path / "tda.sqlite"))
    roots = _gallery_tree(tmp_path)
    _load(db, roots, fixes)
    db.set_frame_override(FrameOverride(FrameKey(99, 3, "cmp_oak1"), "psu.01", None,
                                        "out_of_view"))
    _drop_folder(tmp_path, "003")
    counts = _load(db, roots, fixes)
    assert counts["failed"] and "cmp_oak1 shot 3" in counts["failed"][0]
    assert "no longer in the index" in counts["failed"][0]
    assert len(db.frames_for(99, "cmp_oak1")) == 3


def test_an_appended_shot_changes_nothing_else(tmp_path, fixes):
    db = Db(str(tmp_path / "tda.sqlite"))
    roots = _gallery_tree(tmp_path)
    _load(db, roots, fixes)
    db.add_keyframe(_box_keyframe(99, "cmp_oak1", 1, 1))
    before = _paths(db)
    for role in OAK_ROLES:
        _touch(tmp_path / "oak" / "Desktop 99" / "Components" / "Camera_1" / "004"
               / f"20250101_104500_000_camera_1_{role}")
    counts = _load(db, roots, fixes)
    assert not counts["failed"]
    after = _paths(db)
    assert {k: after[k] for k in before} == before and len(after) == 4


def test_an_index_from_before_the_gallery_is_refused(tmp_path, fixes):
    """Round 3: a version-1 file is refused outright (it predates rs_extra_from too)."""
    db = Db(str(tmp_path / "tda.sqlite"))
    roots = _gallery_tree(tmp_path)
    _load(db, roots, fixes)
    before = {view: db.frames_for(99, view) for view in ("oak1", "rs", "cmp_oak1")}
    old = {99: scan_desktop(99, sequence_tree(tmp_path / "bare"), fixes)}
    path = tmp_path / "old_index.json"
    save_index(old, str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_index(str(path))
    assert loaded[99].gallery is False and P.stale_index(loaded) == [99]
    with pytest.raises(P.StaleIndex, match="Run build-index first"):
        P.load_index_into_db(db, loaded)
    assert {view: db.frames_for(99, view) for view in before} == before


def _d66_tree(tmp_path, components: bool) -> dict:
    roots = build_tree(tmp_path, 66,
                       oak1={f"{i + 1:03d}": [TS[i]] for i in range(3)},
                       oak2={f"{i + 1:03d}": [TS[i]] for i in range(3)},
                       rs_steps=[1, 2, 3], rs_mtimes={i + 1: TS[i] for i in range(3)})
    if components:
        for i, ts in enumerate(SHOT_TS[:2], start=1):
            for role in OAK_ROLES:
                _touch(tmp_path / "oak" / "Desktop 66" / "Components" / "Camera_1"
                       / f"{i:03d}" / f"{ts}_camera_1_{role}")
    return roots


def test_a_sheet_shot_stays_put_when_folders_appear_later(tmp_path, fixes):
    db = Db(str(tmp_path / "tda.sqlite"))
    db.replace_steps(66, [StepRec(66, 1, "initial", "Initial"),
                          StepRec(66, 2, "normal", "Remove PSU"),
                          StepRec(66, 3, "ignore", "All components")], [])
    roots = _d66_tree(tmp_path, components=False)
    P.load_index_into_db(db, {66: scan_desktop(66, roots, fixes)})
    sheet = P.SHEET_SHOT_BASE + 3
    assert [r["step"] for r in db.frames_for(66, "cmp_oak1")] == [sheet]
    db.add_keyframe(_box_keyframe(66, "cmp_oak1", sheet, sheet))     # work on it
    roots = _d66_tree(tmp_path, components=True)                    # folders appear
    lines: list[str] = []
    counts = P.load_index_into_db(db, {66: scan_desktop(66, roots, fixes)},
                                  log=lines.append)
    assert not counts["failed"]
    rows = {int(r["step"]): r for r in db.frames_for(66, "cmp_oak1")}
    assert sorted(rows) == [1, 2, sheet]
    assert rows[sheet]["aux"]["sheet_step"] == 3
    assert rows[sheet]["path"].endswith(f"{TS[2]}_camera_1_rgb_12mp.jpg")
    assert len(db.keyframes(66, "cmp_oak1")) == 1
    # M7: nothing moved this time, and the log does not say it did
    assert not any("photo of the laid-out parts" in l for l in lines)
    assert db.get_frame(FrameKey(66, 3, "oak1")) is None


def test_a_changed_sheet_photo_under_work_fails_the_desktop(tmp_path, fixes):
    db = Db(str(tmp_path / "tda.sqlite"))
    db.replace_steps(66, [StepRec(66, 3, "ignore", "All components")], [])
    roots = _d66_tree(tmp_path, components=False)
    index = {66: scan_desktop(66, roots, fixes)}
    P.load_index_into_db(db, index)
    sheet = P.SHEET_SHOT_BASE + 3
    db.add_keyframe(_box_keyframe(66, "cmp_oak1", sheet, sheet))
    ff = index[66].frames[FrameKey(66, 3, "oak1")]
    index[66].frames[ff.key] = FrameFile(ff.key, ff.path.replace("003", "009"), ff.aux,
                                         ff.ts, ff.src_step_dir)
    counts = P.load_index_into_db(db, index)
    assert counts["failed"] and f"cmp_oak1 shot {sheet} (sheet step 3)" in counts["failed"][0]


# --------------------------------------------------------------------------- #
# M2: drawing a part marked 看不见
# --------------------------------------------------------------------------- #
def test_a_shape_on_a_part_marked_absent_takes_the_label_back(tmp_path):
    session = GS.make_session(tmp_path)
    part = session.task_card()[0]["instance"]
    session.set_visibility(part, "out_of_view")
    session.begin_edit(part)
    session.set_editing_mask(GS.rect(4, 4, 20, 20))
    result = session.commit_edit(api.SCOPE_KEYFRAME)
    assert result["cleared_absent"] is True
    key = session.current()
    assert part not in session.db.frame_overrides(key)
    row = next(r for r in session.task_card() if r["instance"] == part)
    assert row["done"] and not row.get("absent") and not row.get("invisible")
    # one op: Ctrl+Z takes the shape back and puts the label back
    assert session.undo()
    assert session.db.frame_overrides(key)[part].visibility == "out_of_view"
    assert not session.db.keyframes(GS.DESKTOP, "cmp_oak1", part)
    assert session.redo()
    assert part not in session.db.frame_overrides(key)


def test_a_sequence_label_is_left_alone_by_a_commit(tmp_path):
    """Only a gallery shot's 看不见 is taken back; the scanner's labels are labels."""
    from session_scene import make_session as seq_session

    session = seq_session(tmp_path)
    key = session.current()
    part = "chassis"
    session.db.set_frame_override(FrameOverride(key, part, None, "out_of_view"))
    session.begin_edit(part)
    session.set_editing_mask(GS.rect(4, 4, 20, 20))
    result = session.commit_edit(api.SCOPE_KEYFRAME)
    assert not result.get("cleared_absent")
    assert session.db.frame_overrides(key)[part].visibility == "out_of_view"


# --------------------------------------------------------------------------- #
# M3: the guide on a shot
# --------------------------------------------------------------------------- #
def test_the_guide_walks_forward_on_a_shot():
    title = "零件合照 OAK1 第 1/3 张"
    drawn = (G.CardItem("psu.01", api.KIND_ADD_SHAPE, True, "电源 psu.01"),)
    plan = G.plan_for(G.GuideFacts(step=1, gallery=title, roi_stored=True, items=drawn))
    assert plan.title == title
    assert "下一张" in plan.now and "上一帧" not in plan.now
    lines = [text for _state, text in plan.steps]
    assert G.COMMIT_GALLERY in lines and G.CONFIRM_GALLERY in lines
    assert not any("Ctrl+K" in text for text in lines)
    done = G.plan_for(G.GuideFacts(step=1, gallery=title, frame_confirmed=True))
    assert "PgUp 去下一张" in done.now
    seq = G.plan_for(G.GuideFacts(step=5, neighbour=6))
    assert "自动退到上一帧" in seq.now


# --------------------------------------------------------------------------- #
# M6: build-cache takes the gallery from the database
# --------------------------------------------------------------------------- #
def test_build_cache_reads_the_gallery_from_the_database(tmp_path, fixes):
    db = Db(str(tmp_path / "tda.sqlite"))
    db.replace_steps(66, [StepRec(66, 3, "ignore", "All components")], [])
    roots = _d66_tree(tmp_path, components=False)
    index = {66: scan_desktop(66, roots, fixes)}
    P.load_index_into_db(db, index)
    db.close()
    taken = gallery_frames_from_db(index, str(tmp_path / "tda.sqlite"),
                                   ["scan", "cmp_oak1"], [66])
    sheet = P.SHEET_SHOT_BASE + 3
    assert taken == 1
    assert [k.step for k in index[66].frames if k.view == "cmp_oak1"] == [sheet]
    assert gallery_frames_from_db(index, "", ["cmp_oak1"], [66]) == 0


def test_a_re_pointed_shot_is_copied_again():
    frame = FrameFile(FrameKey(1, 1, "cmp_oak1"), "F:/new.jpg")
    assert _source_moved({"src": "F:/old.jpg"}, frame, "cmp_oak1")
    assert not _source_moved({"src": "F:/new.jpg"}, frame, "cmp_oak1")
    scan = FrameFile(FrameKey(1, 1, "scan"), "F:/RGB11/P_0.png",
                     {"burst": ["F:/RGB11/P_0.png", "F:/RGB11/P_3.png"]})
    assert not _source_moved({"src": "F:/RGB11/P_3.png"}, scan, "scan")   # a burst choice
    # U25: a renumbered scanner step keeps its name but not its burst
    assert _source_moved({"src": "F:/RGB21/P_0.png"}, scan, "scan")
    assert not _source_moved(None, frame, "cmp_oak1")


def test_cli_build_cache_passes_the_database(tmp_path, monkeypatch):
    from tda.cli import main

    env = GS.make_env(tmp_path)
    seen: list[list[str]] = []
    monkeypatch.setattr("tda.core.cache.main", lambda argv: seen.append(list(argv)) or 0)
    assert main(["--paths", env["paths"], "build-cache", "--views", "components",
                 "--desktops", "13"]) == 0
    argv = seen[0]
    assert argv[argv.index("--db") + 1] == env["cfg"]["db_path"]
    assert "cmp_scan,cmp_oak1,cmp_oak2,cmp_rs" in argv


# --------------------------------------------------------------------------- #
# parts the log never names that were taken out (「拆下于第 k 步」)
# --------------------------------------------------------------------------- #
def test_plan_extras_writes_the_removal_as_two_hand_made_events():
    tax = load_taxonomy()
    batch = X.plan_extras({}, tax, 2, "misc_part", 1, removed_at=48)
    removal = [e for e in batch.events if e.evidence_view == X.ADDED_EVIDENCE]
    assert [(e.step, e.attr, e.old, e.new, e.auto) for e in removal] == [
        (48, "state", "installed", "removed", False),
        (48, "placement", "in_chassis", "on_bench", False)]
    with pytest.raises(X.ExtraError, match="从 1 开始"):
        X.plan_extras({}, tax, 2, "misc_part", 1, removed_at=0)
    assert "removed" not in X.state_choices(tax, "misc_part")   # never an initial state


def _gallery_data(tmp_path):
    db, tax, _cache = GS.make_db(tmp_path)
    return db, tax, StepTableData.load(db, GS.DESKTOP, tax)


def test_a_part_taken_out_is_drawn_until_then_and_listed_in_the_gallery(tmp_path):
    db, tax, data = _gallery_data(tmp_path)
    [key] = data.add_extras("misc_part", 1, parent=None, note="standoff", removed_at=40)
    data.save(db)
    assert key in gallery_state_of(db, tax, GS.DESKTOP)
    assert gallery_state_of(db, tax, GS.DESKTOP)[key].gallery
    instances = db.instances(GS.DESKTOP)
    # the scanner has no staging area: once taken out the part is not its work
    asked = [step for step in range(1, 43)
             if key in needs_geom(instances, state_of(db, tax, GS.DESKTOP, step), tax,
                                  bench_roi=None)]
    assert asked == list(range(1, 40))
    from tda.ui.session_tasks import gallery_card, task_card_for

    assert key in [r["instance"] for r in gallery_card(db, tax, GS.DESKTOP, "cmp_oak1", 1)]
    card = task_card_for(db, tax, GS.DESKTOP, "scan", 39, neighbour=40)
    assert key in [r["instance"] for r in card]
    # the delete takes the removal with the part, and its undo brings both back
    from tda.ui.steps_delete import delete_instance

    data = StepTableData.load(db, GS.DESKTOP, tax)
    delete_instance(data, db, key)
    assert not [e for e in db.events(GS.DESKTOP) if e.target == key]
    op = db.ops(GS.DESKTOP, "-")[0]          # newest first: the delete
    assert op["kind"] == X.OP_DELETE
    X.undo_op(db, op, tax=tax)
    assert len([e for e in db.events(GS.DESKTOP) if e.target == key]) == 2


def test_the_undo_of_an_addition_takes_its_removal_too(tmp_path):
    db, tax, data = _gallery_data(tmp_path)
    [key] = data.add_extras("misc_part", 1, parent=None, removed_at=40)
    data.save(db)
    op = next(o for o in db.ops(GS.DESKTOP, "-") if o["kind"] == X.OP_ADD)
    X.undo_op(db, op, tax=tax)
    assert key not in db.instances(GS.DESKTOP)
    assert not [e for e in db.events(GS.DESKTOP) if e.target == key]


def test_a_removal_after_the_parent_left_is_refused(tmp_path):
    from tda.ui.steps_values import EditError

    db, tax, data = _gallery_data(tmp_path)
    with pytest.raises(EditError, match="motherboard.01 在第 42 步就离开机箱了"):
        data.add_extras("ram_latch", 1, parent="motherboard.01", removed_at=42)
    with pytest.raises(EditError, match="只有 42 步"):
        data.add_extras("misc_part", 1, parent=None, removed_at=50)
    # before the board leaves it is a clip taken out on its own: fine, and it
    # does not leave with the board (it is out already) -- no validator message
    [clip] = data.add_extras("ram_latch", 1, parent="motherboard.01", removed_at=30)
    assert data.instances[clip].attached is False
    assert data.instances[clip].parent == "motherboard.01"
    assert not [m for m in data.save(db) if clip in m]
    assert clip in gallery_state_of(db, tax, GS.DESKTOP)
    assert gallery_state_of(db, tax, GS.DESKTOP)[clip].gallery


def test_the_gallery_counts_what_an_addition_demotes(tmp_path):
    session = GS.make_session(tmp_path)
    from test_u10_gallery import _confirm_shot

    _confirm_shot(session, 1)
    data = StepTableData.load(session.db, GS.DESKTOP, session.tax)
    assert data.frames_demoted_by_extras(session.db, "cmp_oak1", "misc_part", 1,
                                         parent=None, removed_at=42) == [1]
    assert data.frames_demoted_by_extras(session.db, "cmp_oak1", "misc_part", 1,
                                         parent=None) == []
    assert data.frames_demoted_by_extras(session.db, "cmp_oak2", "misc_part", 1,
                                         parent=None, removed_at=42) == []


def test_the_dialog_pre_fills_the_removal_step(tmp_path):
    from PySide6.QtWidgets import QApplication

    from tda.ui.panels.add_parts import NEVER_REMOVED, AddPartsDialog, elsewhere_text

    QApplication.instance() or QApplication([])
    db, tax, data = _gallery_data(tmp_path)
    dialog = AddPartsDialog(data, cls="misc_part")
    try:
        assert dialog.removed_spin.value() == 42
        assert dialog.values()["removed_at"] == 42
        assert "第 42 步拆下" in dialog.preview.text()
        dialog.class_combo.setCurrentIndex(dialog.class_combo.findData("ram_latch"))
        assert dialog.parent_combo.currentData() == "motherboard.01"
        assert dialog.values()["removed_at"] is None
        assert dialog.removed_spin.specialValueText() == NEVER_REMOVED
    finally:
        dialog.reject()
    assert elsewhere_text({"cmp_oak1": [1, 2], "oak1": [5], "scan": []}) == (
        "另外：oak1 1 帧、零件合照·OAK1 2 张 的已确认帧也会回到“需复核”。")
    assert elsewhere_text({}) == ""


def test_a_reload_keeps_what_the_truth_table_wrote_on_a_missing_frame(tmp_path, fixes):
    """The probe's one sequence difference: a missing row's inferred size was wiped."""
    roots = build_tree(tmp_path, 97, oak1={"001": [TS[0]]}, oak2={})
    db = Db(str(tmp_path / "tda.sqlite"))
    index = {97: scan_desktop(97, roots, fixes)}
    P.load_index_into_db(db, index)
    key = FrameKey(97, 1, "oak2")
    assert db.get_frame(key)["missing"]
    db.upsert_frame(key, None, {"hw": [3040, 4032], "hw_source": "inferred"}, None,
                    {"missing": True})
    P.load_index_into_db(db, index)
    assert db.get_frame(key)["aux"] == {"hw": [3040, 4032], "hw_source": "inferred"}
