"""Task U10: the components gallery (零件合照, spec 3.2b) -- core, session, exports.

After each teardown the parts were laid out on the table and photographed; those
shots are views of their own (``cmp_oak1``/``cmp_oak2``/``cmp_rs``; the scanner's
``cmp_scan`` since task U42, tests/test_u42_scan_gallery.py), one
frame per shot, and everything that works on a frame works on them. What these
tests pin down is what is *different*: the index, the D66-style sheet row that
leaves the sequence, what a shot asks to be drawn, one pose segment per shot,
「看不见」, no sequence semantics, and the exports.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from tda import pipeline as P
from tda.core import masks
from tda.core.compiler import compile_frame, marked_absent
from tda.core.export.coco import export_coco, image_id, view_index
from tda.core.export.vlm import export_vlm
from tda.core.index import load_fixes, load_index, save_index, scan_desktop
from tda.core.model import (
    ALL_VIEWS,
    COMPONENT_VIEWS,
    GALLERY_EXCLUDED_CLASSES,
    VIEWS,
    FrameKey,
    FrameOverride,
    InstanceRec,
    StateEvent,
    StepRec,
    is_component_view,
)
from tda.core.split_rows import split_rows_at, stale_split_rows
from tda.core.states import (
    GALLERY_PLACEMENT,
    gallery_state,
    needs_geom,
    state_at,
)
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import (
    annotatable_steps,
    gallery_state_of,
    gather,
    state_of,
    view_steps,
)
from tda.ui import session_api as api
from tda.ui.gallery import NO_STATE, gallery_header, gallery_refusal, view_label
from tda.ui.panels.taskcard import card_header, row_view
from tda.ui.session_api import SessionRefusal
from tda.ui.session_tasks import gallery_card, task_card_for

import gallery_scene as GS
from test_index import OAK_ROLES, _epoch, _touch, build_tree

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixes() -> dict:
    return load_fixes(str(REPO_ROOT / "configs" / "index_fixes.yaml"))


# --------------------------------------------------------------------------- #
# the views
# --------------------------------------------------------------------------- #
def test_four_gallery_views_after_the_four_sequence_views():
    # the scanner's first since task U42, as in VIEWS
    assert COMPONENT_VIEWS == ("cmp_scan", "cmp_oak1", "cmp_oak2", "cmp_rs")
    assert ALL_VIEWS == VIEWS + COMPONENT_VIEWS
    assert all(is_component_view(v) for v in COMPONENT_VIEWS)
    assert not any(is_component_view(v) for v in VIEWS)
    assert not is_component_view(None) and not is_component_view("")


def test_gallery_image_ids_never_collide_with_the_sequence():
    """``desktop*100000 + step*10 + view index``: the gallery takes 4..7 (U42)."""
    assert [view_index(v) for v in ALL_VIEWS] == list(range(8))
    ids = {image_id(2, s, v) for s in range(1, 60) for v in ALL_VIEWS}
    assert len(ids) == 59 * 8


# --------------------------------------------------------------------------- #
# the index (synthetic Components folders)
# --------------------------------------------------------------------------- #
TS = ["20250101_100000_000", "20250101_100100_000", "20250101_100200_000"]
#: the layout photos, minutes after the last step
SHOT_TS = ["20250101_103000_000", "20250101_103100_000", "20250101_103200_000"]


def add_components(root: Path, desktop: int, cams: dict[str, list[str]],
                   rs: dict[str, str] | None = None) -> None:
    """``Components/<cam>/<NNN>/<ts>_camera_<c>_<role>`` and the RealSense folders."""
    for cam, tokens in cams.items():
        number = 1 if cam in ("Camera_1", "C1") else 2
        for i, ts in enumerate(tokens, start=1):
            for role in OAK_ROLES:
                _touch(root / "oak" / f"Desktop {desktop}" / "Components" / cam
                       / f"{i:03d}" / f"{ts}_camera_{number}_{role}")
    for folder, ts in (rs or {}).items():
        d = root / "rs" / f"Desktop {desktop}" / "Components" / folder
        _touch(d / "original_color.png")
        _touch(d / "depth_raw.npy")
        stamp = _epoch(ts)
        os.utime(d / "original_color.png", (stamp, stamp))


def sequence_tree(tmp_path: Path, desktop: int = 99) -> dict:
    return build_tree(
        tmp_path, desktop,
        oak1={f"{i + 1:03d}": [TS[i]] for i in range(3)},
        oak2={f"{i + 1:03d}": [TS[i]] for i in range(3)},
        scan_steps={1: 10, 2: 10, 3: 10},
        rs_steps=[1, 2, 3],
        rs_mtimes={i + 1: TS[i] for i in range(3)},
    )


def test_index_reads_two_cameras_and_the_realsense(tmp_path, fixes):
    roots = sequence_tree(tmp_path)
    before = scan_desktop(99, roots, fixes)
    add_components(tmp_path, 99, {"Camera_1": SHOT_TS, "Camera_2": SHOT_TS},
                   rs={f"{i:03d}": SHOT_TS[i - 1] for i in (1, 2, 3)})
    idx = scan_desktop(99, roots, fixes)
    for view in ("cmp_oak1", "cmp_oak2", "cmp_rs"):     # the scanner's: test_u42
        shots = sorted(k.step for k in idx.frames if k.view == view)
        assert shots == [1, 2, 3], view
    first = idx.frames[FrameKey(99, 1, "cmp_oak1")]
    assert first.path.endswith(f"{SHOT_TS[0]}_camera_1_rgb_12mp.jpg")
    assert first.src_step_dir == "Components/Camera_1/001"
    assert first.ts == datetime(2025, 1, 1, 10, 30).isoformat()
    assert "depth_npy" in first.aux
    assert idx.frames[FrameKey(99, 3, "cmp_rs")].src_step_dir == "Components/003"
    # the gallery misses nothing, and the sequence is exactly what it was
    assert not any(is_component_view(k.view) for k in idx.missing)
    sequence = {k: v for k, v in idx.frames.items() if not is_component_view(k.view)}
    assert sequence.keys() == before.frames.keys()
    assert idx.n_steps == before.n_steps and idx.missing == before.missing


def test_index_reads_the_c1_c2_spelling_of_desktops_1_to_24(tmp_path, fixes):
    roots = sequence_tree(tmp_path)
    add_components(tmp_path, 99, {"C1": SHOT_TS[:2], "C2": SHOT_TS})
    idx = scan_desktop(99, roots, fixes)
    assert sorted(k.step for k in idx.frames if k.view == "cmp_oak1") == [1, 2]
    assert sorted(k.step for k in idx.frames if k.view == "cmp_oak2") == [1, 2, 3]
    assert idx.frames[FrameKey(99, 2, "cmp_oak2")].src_step_dir == "Components/C2/002"


def test_shots_are_numbered_by_capture_time_not_folder(tmp_path, fixes):
    """D22-D28 number their Components folders on from the teardown (047...)."""
    roots = sequence_tree(tmp_path)
    for name, ts in (("051", SHOT_TS[0]), ("047", SHOT_TS[1])):
        for role in OAK_ROLES:
            _touch(tmp_path / "oak" / "Desktop 99" / "Components" / "Camera_1" / name
                   / f"{ts}_camera_1_{role}")
    idx = scan_desktop(99, roots, fixes)
    assert idx.frames[FrameKey(99, 1, "cmp_oak1")].src_step_dir == "Components/Camera_1/051"
    assert idx.frames[FrameKey(99, 2, "cmp_oak1")].src_step_dir == "Components/Camera_1/047"


def test_a_teardown_capture_filed_under_components_is_not_a_shot(tmp_path, fixes):
    """D24: Camera_2's step 43 and the RealSense's last frame sit in Components."""
    roots = sequence_tree(tmp_path)
    # the capture of step 3 (same token as cam1's), filed as Components/C2/001
    add_components(tmp_path, 99, {"C2": [TS[2], SHOT_TS[0]]},
                   rs={"003": "20250101_100205_000", "004": SHOT_TS[1]})
    idx = scan_desktop(99, roots, fixes)
    assert sorted(k.step for k in idx.frames if k.view == "cmp_oak2") == [1]
    assert idx.frames[FrameKey(99, 1, "cmp_oak2")].src_step_dir == "Components/C2/002"
    assert any("Components/C2/001" in i and "teardown" in i for i in idx.issues)
    rs = sorted((k.step, f.src_step_dir) for k, f in idx.frames.items() if k.view == "cmp_rs")
    assert rs == [(1, "Components/004")]
    assert any("Components/003" in i and "teardown" in i for i in idx.issues)


def test_the_d24_fix_still_puts_cam2_step_43_in_the_sequence(tmp_path, fixes):
    oak1 = {f"{k:03d}": [f"20250604_1713{k:02d}_514"] for k in range(1, 44)}
    oak2 = {k: v for k, v in oak1.items() if k != "043"}
    roots = build_tree(tmp_path, 24, oak1=oak1, oak2=oak2,
                       components_c2={"043": ["20250604_171343_514"]})
    idx = scan_desktop(24, roots, fixes)
    assert idx.frames[FrameKey(24, 43, "oak2")].src_step_dir == "Components/C2/043"
    assert not any(k.view == "cmp_oak2" for k in idx.frames)


def test_gallery_frames_survive_save_and_load(tmp_path, fixes):
    roots = sequence_tree(tmp_path)
    add_components(tmp_path, 99, {"Camera_1": SHOT_TS})
    idx = {99: scan_desktop(99, roots, fixes)}
    save_index(idx, str(tmp_path / "index.json"))
    back = load_index(str(tmp_path / "index.json"))
    assert back[99].frames.keys() == idx[99].frames.keys()


# --------------------------------------------------------------------------- #
# load-index: frames, one segment per shot, idempotent, sequence untouched
# --------------------------------------------------------------------------- #
def _sequence_digest(db, desktop: int) -> str:
    """Every sequence frame row, keyframe and pose segment of one desktop, hashed."""
    rows = []
    for view in VIEWS:
        rows += [json.dumps(r, sort_keys=True, default=str)
                 for r in db.frames_for(desktop, view)]
        rows += [json.dumps(s, sort_keys=True, default=str)
                 for s in db.pose_segments(desktop, view)]
        rows += [repr(kf) for kf in db.keyframes(desktop, view)]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def test_load_index_writes_the_gallery_and_is_idempotent(tmp_path, fixes):
    from tda.core.db import Db

    roots = sequence_tree(tmp_path)
    add_components(tmp_path, 99, {"Camera_1": SHOT_TS, "C2": SHOT_TS[:2]},
                   rs={"001": SHOT_TS[0]})
    index = {99: scan_desktop(99, roots, fixes)}
    db = Db(str(tmp_path / "tda.sqlite"))
    counts = P.load_index_into_db(db, index)
    assert counts["gallery"] == 6 and not counts["failed"]
    for view, n in (("cmp_oak1", 3), ("cmp_oak2", 2), ("cmp_rs", 1)):
        segs = db.pose_segments(99, view)
        assert [(s["seg"], s["start_step"], s["end_step"]) for s in segs] == [
            (k, k, k) for k in range(1, n + 1)]
        hw = (720, 1280) if view == "cmp_rs" else (3040, 4032)
        assert all(s["roi"] == [0, 0, hw[1], hw[0]] for s in segs)
    seq = _sequence_digest(db, 99)
    ops = db.conn.execute("SELECT COUNT(*) FROM op_log").fetchone()[0]
    frames = db.conn.execute("SELECT COUNT(*) FROM frame").fetchone()[0]
    P.load_index_into_db(db, index)
    assert _sequence_digest(db, 99) == seq
    assert db.conn.execute("SELECT COUNT(*) FROM op_log").fetchone()[0] == ops
    assert db.conn.execute("SELECT COUNT(*) FROM frame").fetchone()[0] == frames


def test_reindexing_with_components_leaves_sequence_annotations_alone(tmp_path, fixes):
    """What the probe checks on D13/scan: verified frames and keyframes byte-identical."""
    from tda.core.db import Db
    from tda.core.model import ShapeKeyframe, ShapePart

    roots = sequence_tree(tmp_path)
    db = Db(str(tmp_path / "tda.sqlite"))
    P.load_index_into_db(db, {99: scan_desktop(99, roots, fixes)})
    db.add_keyframe(ShapeKeyframe(None, "chassis", 99, "scan", 1, 3,
                                  parts=[ShapePart("main", box=(1, 2, 3, 4))]))
    db.set_frame_flags(FrameKey(99, 2, "scan"), review_status="verified")
    seq = _sequence_digest(db, 99)
    add_components(tmp_path, 99, {"Camera_1": SHOT_TS})
    P.load_index_into_db(db, {99: scan_desktop(99, roots, fixes)})
    assert _sequence_digest(db, 99) == seq
    assert len(db.frames_for(99, "cmp_oak1")) == 3


# --------------------------------------------------------------------------- #
# a sheet row of the laid-out parts (D66 step 13)
# --------------------------------------------------------------------------- #
def _d66_like(tmp_path):
    from tda.core.db import Db

    db = Db(str(tmp_path / "tda.sqlite"))
    for step in (1, 2, 3):
        for view in VIEWS:
            missing = view == "scan" and step == 3
            db.upsert_frame(FrameKey(66, step, view),
                            None if missing else f"F:/x/{view}/{step:03d}.png",
                            {} if missing else {"depth_npy": "d.npy"}, None,
                            {"missing": missing})
    db.replace_steps(66, [StepRec(66, 1, "initial", "Initial"),
                          StepRec(66, 2, "normal", "Remove PSU"),
                          StepRec(66, 3, "ignore", "All components")], [])
    return db


SHEET_SHOT = P.SHEET_SHOT_BASE + 3      # D66-like: the sheet row is step 3


def test_an_all_components_row_moves_into_the_gallery(tmp_path):
    db = _d66_like(tmp_path)
    assert P.gallery_rows(db, 66) == [3]
    out = P.settle_gallery(db, 66)
    for view in VIEWS:
        assert db.get_frame(FrameKey(66, 3, view)) is None, view
        assert db.get_frame(FrameKey(66, 2, view)) is not None
    for view, camera in (("cmp_oak1", "oak1"), ("cmp_oak2", "oak2"), ("cmp_rs", "rs")):
        row = db.get_frame(FrameKey(66, SHEET_SHOT, view))
        assert row["path"] == f"F:/x/{camera}/003.png"
        assert row["aux"]["sheet_step"] == 3 and row["aux"]["depth_npy"] == "d.npy"
        assert [(s["seg"], s["start_step"]) for s in db.pose_segments(66, view)] == [
            (SHEET_SHOT, SHEET_SHOT)]
    # the scanner's row of step 3 has no photo here (the U42 case with one is
    # tests/test_u42_scan_gallery.py): it goes nowhere, and says why
    assert out["shots"] == {"cmp_scan": 0, "cmp_oak1": 1, "cmp_oak2": 1, "cmp_rs": 1}
    assert len(out["moved"]) == 4      # three photos and the scanner's empty row
    assert any("(scan)" in line and "nowhere (it has no photo)" in line
               for line in out["moved"])
    # the step keeps its number and its type
    assert [(s.step, s.step_type) for s in db.steps(66)][-1] == (3, "ignore")
    # idempotent: a re-load puts the sequence rows back, the rule moves them again
    db.upsert_frame(FrameKey(66, 3, "oak1"), "F:/x/oak1/003.png", {}, None)
    P.settle_gallery(db, 66)
    assert db.get_frame(FrameKey(66, 3, "oak1")) is None
    assert [r["step"] for r in db.frames_for(66, "cmp_oak1")] == [SHEET_SHOT]


def test_a_sheet_shot_never_collides_with_folder_shots(tmp_path):
    """Folders before or after the relocation: 1..N for them, 1000+step for it."""
    db = _d66_like(tmp_path)
    for shot in (1, 2):
        db.upsert_frame(FrameKey(66, shot, "cmp_oak1"), f"F:/c/{shot}.jpg", {}, None)
    P.settle_gallery(db, 66)
    assert [r["step"] for r in db.frames_for(66, "cmp_oak1")] == [1, 2, SHEET_SHOT]
    assert db.get_frame(FrameKey(66, SHEET_SHOT, "cmp_oak1"))["aux"]["sheet_step"] == 3
    assert db.get_frame(FrameKey(66, 1, "cmp_oak1"))["path"] == "F:/c/1.jpg"


def test_a_sheet_shot_from_an_older_build_moves_to_its_key(tmp_path):
    db = _d66_like(tmp_path)
    db.upsert_frame(FrameKey(66, 1, "cmp_oak1"), "F:/x/oak1/003.png", {"sheet_step": 3},
                    None)
    P.settle_gallery(db, 66)
    assert [r["step"] for r in db.frames_for(66, "cmp_oak1")] == [SHEET_SHOT]


def test_the_measured_size_of_a_sheet_shot_survives_a_reload(tmp_path):
    """M1: re-loading must not throw away hw / hw_source / cache_path."""
    db = _d66_like(tmp_path)
    P.settle_gallery(db, 66)
    key = FrameKey(66, SHEET_SHOT, "cmp_oak1")
    row = db.get_frame(key)
    db.upsert_frame(key, row["path"], dict(row["aux"], hw=[3040, 4032], hw_source="measured",
                                           cache_path="D:/c/s1003.jpg"), row["ts"])
    db.upsert_frame(FrameKey(66, 3, "oak1"), "F:/x/oak1/003.png", {"depth_npy": "d.npy"},
                    None)
    P.settle_gallery(db, 66)
    aux = db.get_frame(key)["aux"]
    assert (aux["hw"], aux["hw_source"], aux["cache_path"]) == (
        [3040, 4032], "measured", "D:/c/s1003.jpg")


def test_an_annotated_row_is_never_moved(tmp_path):
    db = _d66_like(tmp_path)
    db.set_frame_override(FrameOverride(FrameKey(66, 3, "oak1"), "chassis", None,
                                        "out_of_view"))
    out = P.settle_gallery(db, 66)
    assert db.get_frame(FrameKey(66, 3, "oak1")) is not None
    assert not db.frames_for(66, "cmp_oak1")
    assert any("has annotations" in line for line in out["moved"])


def test_only_an_ignore_row_named_for_the_components_moves(tmp_path):
    db = _d66_like(tmp_path)
    db.replace_steps(66, [StepRec(66, 1, "initial", "Initial"),
                          StepRec(66, 2, "normal", "Remove all components"),
                          StepRec(66, 3, "ignore", "Calibration shot")], [])
    assert P.gallery_rows(db, 66) == []


# --------------------------------------------------------------------------- #
# what a shot asks to be drawn
# --------------------------------------------------------------------------- #
def _rec(key, cls, **kw) -> InstanceRec:
    return InstanceRec(key=key, desktop=1, cls=cls, **kw)


def _removed(target: str, step: int) -> list[StateEvent]:
    return [StateEvent(1, step, target, "state", "installed", "removed"),
            StateEvent(1, step, target, "placement", "in_chassis", "on_bench")]


def test_gallery_state_is_what_is_removed_at_the_end_minus_the_excluded():
    tax = load_taxonomy()
    instances = {
        "chassis": _rec("chassis", "chassis"),
        "psu.01": _rec("psu.01", "psu"),
        "motherboard.01": _rec("motherboard.01", "motherboard"),
        "ram_latch.01": _rec("ram_latch.01", "ram_latch", parent="motherboard.01",
                             attached=True),
        "cpu.01": _rec("cpu.01", "cpu"),
        "screw.psu.01": _rec("screw.psu.01", "screw"),
        "connector.fan.01": _rec("connector.fan.01", "connector"),
        "case_fan.01": _rec("case_fan.01", "case_fan"),
        "ls:Fan#1": _rec("ls:Fan#1", "case_fan"),
    }
    events = (_removed("psu.01", 3) + _removed("cpu.01", 5)
              + _removed("motherboard.01", 9)
              + [StateEvent(1, 2, "screw.psu.01", "state", "fastened", "removed"),
                 StateEvent(1, 4, "connector.fan.01", "state", "plugged", "removed")])
    fs = gallery_state(instances, events, tax, GALLERY_EXCLUDED_CLASSES)
    parts = sorted(k for k, v in fs.items() if v.gallery)
    # the latch went out inside the board; the fan is still in; the chassis stays
    assert parts == ["cpu.01", "motherboard.01", "psu.01"]
    assert all(fs[k].state == "removed" and fs[k].placement == GALLERY_PLACEMENT
               for k in parts)
    assert fs["case_fan.01"].placement == "elsewhere"
    assert fs["screw.psu.01"].placement == "elsewhere"
    # masks, with no bench ROI
    assert needs_geom(instances, fs, tax, bench_roi=None) == {k: "mask" for k in parts}
    # the snapshot's own state is the same at any "step": the gallery has none
    assert state_at(instances, events, 2, tax)["cpu.01"].state != "removed"


def test_the_d13_gallery(tmp_path):
    db, tax, _cache = GS.make_db(tmp_path)
    fs = gallery_state_of(db, tax, GS.DESKTOP)
    parts = sorted(k for k, v in fs.items() if v.gallery)
    assert "motherboard.01" in parts and "psu.01" in parts and "cpu.01" in parts
    assert not any(k.startswith(("screw.", "connector.", "ram_latch.")) for k in parts)
    # every shot of every gallery view asks the same, whatever its number
    for view in COMPONENT_VIEWS:
        for shot in (1, 2, 3):
            assert state_of(db, tax, GS.DESKTOP, shot, view=view) == fs
    # ... and a gallery shot number is not a step: the scanner's step 1 is not
    assert state_of(db, tax, GS.DESKTOP, 1, view="scan") != fs


def test_gather_and_compile_a_shot(tmp_path):
    db, tax, _cache = GS.make_db(tmp_path)
    key = FrameKey(GS.DESKTOP, 2, "cmp_oak1")
    inputs = gather(db, tax, key)
    assert inputs.bench_roi is None and inputs.pose_segment == 2
    assert inputs.needs and set(inputs.needs.values()) == {"mask"}
    frame = compile_frame(key, inputs.hw, inputs.needs, inputs.keyframes, inputs.zorder,
                          inputs.overrides, inputs.occluders, inputs.frame_overrides,
                          inputs.transform, placements=inputs.placements,
                          bench_roi=inputs.bench_roi, pose_segment=inputs.pose_segment)
    missing = sorted(p.split(":", 1)[1] for p in frame.problems
                     if p.startswith("missing_shape:"))
    assert missing == sorted(inputs.needs)
    assert not any(p.startswith("bench_missing:") for p in frame.problems)


def test_a_part_marked_absent_settles_its_missing_shape(tmp_path):
    db, tax, _cache = GS.make_db(tmp_path)
    key = FrameKey(GS.DESKTOP, 1, "cmp_oak1")
    part = sorted(gather(db, tax, key).needs)[0]
    absent = FrameOverride(key, part, None, "out_of_view")
    assert marked_absent(absent)
    assert not marked_absent(FrameOverride(key, part, {"size": [1, 1], "counts": "1"},
                                           "out_of_view"))
    assert not marked_absent(FrameOverride(key, part, None, "visible"))
    db.set_frame_override(absent)
    problems = TruthService(db, tax).refresh(key)["problems"]
    assert f"missing_shape:{part}" not in problems
    assert any(p.startswith("missing_shape:") for p in problems)
    # on the sequence the same label is 画框外 since task U39 -- the part is in
    # the machine, outside this view's picture -- and settles it there too
    seq = FrameKey(GS.DESKTOP, 42, "scan")
    assert "missing_shape:chassis" in TruthService(db, tax).refresh(seq)["problems"]
    db.set_frame_override(FrameOverride(seq, "chassis", None, "out_of_view"))
    assert "missing_shape:chassis" not in TruthService(db, tax).refresh(seq)["problems"]


def test_shots_are_their_own_steps_and_segments(tmp_path):
    db, tax, _cache = GS.make_db(tmp_path)
    assert view_steps(db, GS.DESKTOP, "cmp_rs") == [1, 2, 3]
    assert view_steps(db, GS.DESKTOP, "scan") == list(range(1, 43))
    # D13's step types do not gate a shot number
    assert annotatable_steps(db, GS.DESKTOP, "cmp_oak2", [1, 2, 3]) == [1, 2, 3]


# --------------------------------------------------------------------------- #
# the card
# --------------------------------------------------------------------------- #
def test_gallery_card_rows_and_header(tmp_path):
    db, tax, _cache = GS.make_db(tmp_path)
    card = task_card_for(db, tax, GS.DESKTOP, "cmp_oak1", 2, neighbour=3)
    assert card == gallery_card(db, tax, GS.DESKTOP, "cmp_oak1", 2)
    parts = sorted(gather(db, tax, FrameKey(GS.DESKTOP, 2, "cmp_oak1")).needs)
    assert sorted(r["instance"] for r in card) == parts
    assert all(r["kind"] == api.KIND_ADD_SHAPE and r["gallery"] and not r["done"]
               for r in card)
    db.set_frame_override(FrameOverride(FrameKey(GS.DESKTOP, 2, "cmp_oak1"), parts[0],
                                        None, "out_of_view"))
    card = gallery_card(db, tax, GS.DESKTOP, "cmp_oak1", 2)
    row = next(r for r in card if r["instance"] == parts[0])
    assert row["done"] and row["absent"]
    view = row_view(row)
    assert view["chip"] == "看不见" and "看不见" in view["sentence"]
    open_row = next(r for r in card if not r["done"])
    assert "看不见" in row_view(open_row)["sentence"]
    text = gallery_header("cmp_oak1", 2, [1, 2, 3])
    assert text == ("零件合照 OAK1 第 2/3 张：把这张里看得到的零件都画出来；"
                    "不在这张里的点「看不见」")
    header = card_header(2, None, [r["kind"] for r in card], 1, 3,
                         [r["done"] for r in card], gallery=text)
    assert header.startswith(text) and f"还剩 {len(card) - 1} 个" in header
    done = card_header(2, None, [r["kind"] for r in card], 1, 3, [True] * len(card),
                       gallery=text)
    assert "直接 Space 确认" in done


def test_no_split_rows_on_a_gallery_shot(tmp_path):
    db, tax, _cache = GS.make_db(tmp_path)
    assert split_rows_at(db, tax, GS.DESKTOP, "cmp_oak1", 1, 2) == []
    assert stale_split_rows(db, tax, GS.DESKTOP, "cmp_oak1") == []


def test_words():
    assert view_label("cmp_rs") == "零件合照·RealSense" and view_label("scan") == "scan"
    assert gallery_refusal("cmp_oak2", "cycle_state") == NO_STATE
    assert gallery_refusal("oak2", "cycle_state") == ""
    assert gallery_refusal("cmp_oak2", "commit") == ""


# --------------------------------------------------------------------------- #
# the session
# --------------------------------------------------------------------------- #
def test_session_opens_on_shot_one_and_has_no_neighbour(tmp_path):
    session = GS.make_session(tmp_path)
    assert session.gallery and session.current().step == 1
    assert session.task_neighbour() is None and session.task_span() == []
    assert session.flash_compare() is None
    assert all(r.get("gallery") for r in session.task_card())


def test_t_and_ctrl_k_are_refused_on_a_shot(tmp_path):
    session = GS.make_session(tmp_path)
    part = session.task_card()[0]["instance"]
    assert session.state_choices(part) == []
    assert session.next_state(part) is None
    with pytest.raises(SessionRefusal, match="没有状态"):
        session.assert_state(part, "removed")
    session.begin_edit(part)
    session.set_editing_mask(GS.rect(4, 4, 20, 20))
    with pytest.raises(SessionRefusal, match="用 Q 提交"):       # Enter until U22
        session.commit_edit(api.SCOPE_SPLIT)
    assert all(r["state_choices"] == [] for r in session.instance_rows())
    assert session.scope_hosts() == {}


def test_a_shape_drawn_on_one_shot_reaches_no_other(tmp_path):
    session = GS.make_session(tmp_path)
    part = session.task_card()[0]["instance"]
    session.begin_edit(part)
    session.set_editing_mask(GS.rect(4, 4, 20, 20))
    result = session.commit_edit(api.SCOPE_KEYFRAME)
    assert result["affected"] == [1]
    kf = session.db.keyframes(GS.DESKTOP, "cmp_oak1", part)
    assert [(k.pose_segment, k.anchor_step, k.placement, k.geom_type) for k in kf] == [
        (1, 1, "in_chassis", "mask")]
    assert next(r for r in session.task_card() if r["instance"] == part)["done"]
    session.goto(2)
    assert not next(r for r in session.task_card() if r["instance"] == part)["done"]


def test_absent_then_space_goes_to_the_next_shot(tmp_path):
    session = GS.make_session(tmp_path)
    parts = [r["instance"] for r in session.task_card()]
    session.begin_edit(parts[0])
    session.set_editing_mask(GS.rect(4, 4, 20, 20))
    session.commit_edit(api.SCOPE_KEYFRAME)
    assert not session.confirm_frame()          # the rest are neither drawn nor absent
    for part in parts[1:]:
        session.set_visibility(part, "out_of_view")
    assert all(r["done"] for r in session.task_card())
    assert session.confirm_frame()
    assert session.current().step == 2         # forward: the next shot
    assert session.frame_status(1) == api.STATUS_VERIFIED
    # 看不见 taken back: the part is to be drawn again, with no label
    session.goto(1)
    session.clear_visibility(parts[1])
    row = next(r for r in session.task_card() if r["instance"] == parts[1])
    assert not row["done"] and not row.get("absent")
    assert parts[1] not in session.db.frame_overrides(session.current())
    assert session.undo()
    assert next(r for r in session.task_card() if r["instance"] == parts[1])["absent"]


# --------------------------------------------------------------------------- #
# exports, check, status
# --------------------------------------------------------------------------- #
def _confirm_shot(session, shot: int) -> list[str]:
    session.goto(shot)
    parts = [r["instance"] for r in session.task_card()]
    session.begin_edit(parts[0])
    session.set_editing_mask(GS.rect(4, 4, 20, 20))
    session.commit_edit(api.SCOPE_KEYFRAME)
    for part in parts[1:]:
        session.set_visibility(part, "out_of_view")
    assert session.confirm_frame()
    return parts


def test_export_coco_writes_the_components_split(tmp_path):
    session = GS.make_session(tmp_path)
    parts = _confirm_shot(session, 1)
    doc = export_coco(session.db, session.tax, [GS.DESKTOP], "cmp_oak1",
                      str(tmp_path / "coco_cmp_oak1.json"))
    assert doc["info"]["split"] == "components" and doc["info"]["view"] == "cmp_oak1"
    assert len(doc["images"]) == 1 and len(doc["annotations"]) == 1
    image = doc["images"][0]
    assert image["id"] == image_id(GS.DESKTOP, 1, "cmp_oak1")
    assert image["extra"]["shot"] == 1 and image["extra"]["split"] == "components"
    assert image["extra"]["source"].endswith(".jpg") and image["tier"] == "gold"
    attrs = doc["annotations"][0]["attributes"]
    assert attrs["instance_key"] == parts[0]
    assert (attrs["state"], attrs["placement"]) == ("removed", "on_bench")
    assert attrs["verified"] is True
    # the sequence's export is still the sequence's
    seq = export_coco(session.db, session.tax, [GS.DESKTOP], "scan",
                      str(tmp_path / "coco_scan.json"), only_verified=False)
    assert seq["info"]["split"] == "sequence"


def test_export_vlm_refuses_a_gallery_view(tmp_path):
    db, tax, _cache = GS.make_db(tmp_path)
    with pytest.raises(ValueError, match="components-gallery"):
        export_vlm(db, tax, [GS.DESKTOP], "cmp_oak1", str(tmp_path / "x.jsonl"))


def test_an_added_part_never_demotes_a_gallery_shot(tmp_path):
    """An S1 extra has no action of its own: no gallery shot ever asks for it."""
    from tda.core.extra import verified_frames_gaining

    session = GS.make_session(tmp_path)
    _confirm_shot(session, 1)
    clip = InstanceRec(key="ram_latch.09", desktop=GS.DESKTOP, cls="ram_latch",
                       attrs={"added_by": "tester"})
    assert verified_frames_gaining(session.db, session.tax, GS.DESKTOP, "cmp_oak1",
                                   [clip]) == []


def test_the_removed_rows_of_a_shot_read_the_gallery_state(tmp_path):
    """The instance table's ``Removed`` list on a shot: never a part to draw there."""
    from tda.ui import app_compat as compat

    session = GS.make_session(tmp_path)
    removed = {str(r.get("key")) for r in compat.removed_rows(session)}
    drawn = {r["instance"] for r in session.task_card()}
    assert removed and not removed & drawn


def test_status_rows_count_the_gallery(tmp_path):
    from tda.cli import format_desktop, format_status

    session = GS.make_session(tmp_path)
    _confirm_shot(session, 1)
    rows = P.status_rows(session.db)
    views = rows[0]["views"]
    assert views["cmp_oak1"]["frames"] == 3 and views["cmp_oak1"]["work"] == 3
    assert views["cmp_oak1"]["done"] == 1 and views["cmp_oak1"]["keyframes"] == 1
    table = format_status(rows)
    assert "零件合照 / components gallery" in table
    # four gallery views since U42, like the sequence: the line after the heading
    gallery = table.partition("零件合照 / components gallery")[2]
    gallery_line = next(l for l in gallery.splitlines() if l.startswith("D13") and "|" in l
                        and l.count("|") == len(COMPONENT_VIEWS))
    assert "1/3" in gallery_line
    detail = format_desktop(session.db, rows[0])
    assert "cmp_oak1" in detail and "shots" in detail


# --------------------------------------------------------------------------- #
# the command line
# --------------------------------------------------------------------------- #
def _run(env: dict, *argv: str) -> int:
    from tda.cli import main

    return main(["--paths", env["paths"], *argv])


def _confirm_in_env(env: dict) -> None:
    """Confirm shot 1 of cmp_oak1 in the workspace's own database."""
    from tda.core.db import Db
    from tda.ui.session import AnnotationSession

    db = Db(env["cfg"]["db_path"])
    tax = load_taxonomy()
    session = AnnotationSession(db, tax, TruthService(db, tax), env["cfg"]["cache_dir"],
                                "tester")
    session.sweeper_enabled = False
    session.open(GS.DESKTOP, "cmp_oak1")
    _confirm_shot(session, 1)
    session.close(force=True)
    db.close()


def test_view_list_knows_the_gallery_and_its_alias():
    from tda.cli_app import _view_list

    assert _view_list("components") == list(COMPONENT_VIEWS)
    assert _view_list("scan,cmp_rs,components") == ["scan", "cmp_rs", "cmp_scan", "cmp_oak1",
                                                    "cmp_oak2"]
    with pytest.raises(ValueError, match="cmp_oak1"):
        _view_list("cmp_oak3")


def test_cli_check_and_export_coco_of_the_gallery(tmp_path, capsys):
    env = GS.make_env(tmp_path)
    _confirm_in_env(env)
    capsys.readouterr()
    code = _run(env, "check", "--desktop", str(GS.DESKTOP), "--view", "cmp_oak1")
    out = capsys.readouterr().out
    # shots 2 and 3 are neither drawn nor marked absent: missing shapes
    assert code == 1 and "D13 cmp_oak1: 3 shots" in out and "missing_shape:" in out
    assert "shape_predates_state" not in out
    out_json = tmp_path / "coco_gallery.json"
    code = _run(env, "export-coco", "--desktops", str(GS.DESKTOP), "--view", "cmp_oak1",
                "--out", str(out_json))
    assert code == 0, capsys.readouterr().out
    doc = json.loads(out_json.read_text(encoding="utf-8"))
    assert doc["info"]["split"] == "components" and len(doc["annotations"]) == 1
    # the alias writes one file per gallery view; two of them have nothing confirmed
    capsys.readouterr()
    code = _run(env, "export-coco", "--desktops", str(GS.DESKTOP), "--view", "components")
    out = capsys.readouterr().out
    assert code == 1 and "1 images, 1 annotations" in out
    assert (Path(env["cfg"]["cache_dir"]) / "coco_cmp_oak1.json").exists()


def test_cli_export_vlm_skips_the_gallery(tmp_path, capsys):
    env = GS.make_env(tmp_path)
    code = _run(env, "export-vlm", "--desktops", str(GS.DESKTOP), "--view", "cmp_oak1")
    out = capsys.readouterr().out
    assert code == 1 and "no VLM questions" in out
    assert not list(Path(env["cfg"]["cache_dir"]).glob("vlm_cmp_*"))


def test_cli_status_shows_the_gallery_apart(tmp_path, capsys):
    env = GS.make_env(tmp_path)
    assert _run(env, "status") == 0
    out = capsys.readouterr().out
    head, _sep, gallery = out.partition("零件合照 / components gallery")
    assert "cmp_oak1" not in head and "cmp_oak1" in gallery
    assert _run(env, "status", "--desktop", str(GS.DESKTOP)) == 0
    assert "cmp_rs" in capsys.readouterr().out
