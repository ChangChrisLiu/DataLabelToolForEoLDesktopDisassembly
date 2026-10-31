"""Task U42: the scanner's all-components scans are the gallery view ``cmp_scan``.

After every teardown the parts were also laid out on the scanner's bed and
scanned, one to five times per desktop, into a folder beside the step scans'
payload (``TAMU_B2.0_<a>-<b>_RGB_allcompoentsrgb/<payload>/<n>/<shot>/P_k.png``).
They are a fourth components-gallery view, first among them as the scanner is
first among the sequence's: indexed (index file v4), loaded with one pose segment
and a whole-image ROI per shot, cached with the scanner's burst choice, shown
behind a toolbar button with no key, exported and counted like the others.

The synthetic scanner tree below has every spelling the real drive has
(surveyed 2026-10-30): ``allRGB1_color`` (D1), ``allcomponentsrgb1_color`` (D2),
``...rgb11_color`` (D3), ``allcompoentsrgb11`` (D7), ``allcomponenetsrgb11``
(D51), D52's ``rgb1`` + ``rgb21``, D64's fringe sequence ``seq11`` and two
broken folders, both payload spellings, the group holding two all-components
folders (50-59 / 60-66) and D20 filed in the 10-19 batch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import pytest

from tda import pipeline as P
from tda.core.cache import BURST_VIEWS, VIEW_EXT, build_cache
from tda.core.db import Db
from tda.core.export.coco import STEP_STRIDE, export_coco, image_id, view_index
from tda.core.index import (
    DesktopIndex,
    FrameFile,
    _add_scanner_components,
    load_fixes,
    load_index,
    save_index,
    scan_desktop,
    scan_shot_k,
)
from tda.core.model import (
    ALL_VIEWS,
    COMPONENT_CAMERA,
    COMPONENT_VIEWS,
    FrameKey,
    StepRec,
    is_component_view,
)
from tda.core.sources import scanner_components_dir, scanner_desktop_dir
from tda.core.truth_inputs import infer_hw
from tda.ui import app_actions as A
from tda.ui.gallery import gallery_title, view_label

import gallery_scene as GS
from test_cache import _texture, _write_burst
from test_index import _touch, build_tree
from test_u10_gallery import _confirm_shot

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixes() -> dict:
    return load_fixes(str(REPO_ROOT / "configs" / "index_fixes.yaml"))


# --------------------------------------------------------------------------- #
# a scanner tree with every spelling of the real drive
# --------------------------------------------------------------------------- #
def _burst(folder: Path, n: int = 10, start: int = 0) -> None:
    for k in range(start, start + n):
        _touch(folder / f"P_{k}.png")


#: ``(group, all-components folder, payload, {desktop: {shot folder: (n, start)}})``
LAYOUT = [
    ("TAMU_B2.3_1-9_RGB", "TAMU_B2.0_1-9_RGB_allcompoentsrgb",
     "TAMU_B2.0_1-9_RGB_allcompoentsrgb", {
         1: {"allRGB1_color": (10, 0)},
         2: {"allcomponentsrgb1_color": (10, 0)},
         3: {"allcomponentsrgb21_color": (10, 0), "allcomponentsrgb11_color": (10, 0)},
         7: {"allcompoentsrgb11": (10, 0), "allcompoentsrgb21": (10, 0)},
     }),
    ("TAMU_B2.3_10-19_RGB", "TAMU_B2.0_10-19_RGB_allcompoentsrgb",
     "TAMU_color_10-19_B2.0", {
         12: {"allcomponentsrgb11": (10, 0)},
         20: {"allcomponentsrgb11": (10, 0), "allcomponentsrgb21": (10, 0)},  # D20 here
     }),
    ("TAMU_B2.3_20-29_RGB", "TAMU_B2.0_20-29_RGB_allcompoentsrgb",
     "TAMU_color_20-29_B2.0", {
         21: {"allcompoentsrgb11": (10, 0)},
     }),
    ("TAMU_B2.3_50-66_RGB", "TAMU_B2.0_50-59_RGB_allcompoentsrgb",
     "TAMU_color_50-59_B2.0", {
         51: {"allcomponenetsrgb11": (10, 0), "allcomponenetsrgb21": (10, 0)},
         52: {"allcomponentsrgb1": (10, 0), "allcomponentsrgb21": (10, 0)},
     }),
    ("TAMU_B2.3_50-66_RGB", "TAMU_B2.0_60-66_RGB_allcompoentsrgb",
     "TAMU_color_60-66_B2.0", {
         64: {"allcomponentsrgb11": (10, 0), "allcomponentsrgb21": (10, 0),
              "allcomponentsseq11": (30, 0),       # a fringe-projection sequence
              "allcomponentsrgb31": (25, 0),       # an "rgb" folder far too long
              "allcomponentsrgb41": (9, 1)},       # no P_0
         66: {"allcomponentsrgb11": (10, 0)},
     }),
]


def scanner_tree(root: Path) -> str:
    """The all-components folders of :data:`LAYOUT`, plus the step scans' payload
    and ``new`` of the 50-66 group; returns the scanner root."""
    scan = root / "scan"
    for group, folder, payload, desktops in LAYOUT:
        for desktop, shots in desktops.items():
            for name, (n, start) in shots.items():
                _burst(scan / group / folder / payload / str(desktop) / name, n, start)
    _burst(scan / "TAMU_B2.3_50-66_RGB" / "tamu_color_50-66_Bright2.3" / "52" / "RGB11")
    _touch(scan / "TAMU_B2.3_50-66_RGB" / "new" / "52_1.png")
    _burst(scan / "TAMU_B2.3_20-29_RGB" / "TAMU_B2.3_20-29_RGB" / "20" / "RGB11")
    return str(scan)


def _shots(root: str, desktop: int) -> tuple[dict[int, FrameFile], list[str]]:
    frames: dict[FrameKey, FrameFile] = {}
    issues = _add_scanner_components(desktop, {"scanner_root": root}, frames)
    assert all(k.view == "cmp_scan" for k in frames)
    return {k.step: ff for k, ff in frames.items()}, issues


# --------------------------------------------------------------------------- #
# the view
# --------------------------------------------------------------------------- #
def test_cmp_scan_is_the_first_gallery_view_and_the_scanners():
    assert COMPONENT_VIEWS[0] == "cmp_scan" and is_component_view("cmp_scan")
    assert COMPONENT_CAMERA["cmp_scan"] == "scan"
    assert P.GALLERY_OF["scan"] == "cmp_scan"
    assert VIEW_EXT["cmp_scan"] == "png" and "cmp_scan" in BURST_VIEWS
    assert infer_hw("cmp_scan") == (1600, 1600)
    assert view_label("cmp_scan") == "零件合照·扫描仪"
    assert gallery_title("cmp_scan", 2, [1, 2, 3]) == "零件合照 扫描仪 第 2/3 张"


def test_view_index_shifts_the_other_gallery_views_by_one():
    """No gallery image id was ever released, so the shift is accepted (U42)."""
    assert {v: view_index(v) for v in ALL_VIEWS} == {
        "scan": 0, "oak1": 1, "oak2": 2, "rs": 3,
        "cmp_scan": 4, "cmp_oak1": 5, "cmp_oak2": 6, "cmp_rs": 7}
    assert max(view_index(v) for v in ALL_VIEWS) < STEP_STRIDE
    assert image_id(66, P.SHEET_SHOT_BASE + 13, "cmp_scan") == 66 * 100_000 + 10_130 + 4


# --------------------------------------------------------------------------- #
# the index
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name, k", [
    ("allRGB1_color", 1), ("allcomponentsrgb1_color", 1), ("allcomponentsrgb11_color", 1),
    ("allcomponentsrgb21_color", 2), ("allcompoentsrgb31", 3), ("allcomponenetsrgb41", 4),
    ("allcomponentsrgb1", 1), ("allcompoentsrgb51", 5),
    ("allcomponentsseq11", None), ("RGB11", None), ("notes", None),
])
def test_shot_folder_names(name, k):
    assert scan_shot_k(name) == k


def test_the_locator_finds_every_batch(tmp_path):
    root = scanner_tree(tmp_path)
    for group, folder, payload, desktops in LAYOUT:
        for desktop in desktops:
            found = scanner_components_dir(root, desktop)
            assert found is not None, desktop
            assert Path(found) == Path(root) / group / folder / payload / str(desktop)
    # D20 is filed under 10-19 although its range says 20-29
    assert "10-19" in scanner_components_dir(root, 20)
    assert scanner_components_dir(root, 40) is None
    # the step scans' locator does not mistake an all-components folder for one
    assert Path(scanner_desktop_dir(root, 52)).parent.name == "tamu_color_50-66_Bright2.3"
    assert Path(scanner_desktop_dir(root, 20)).parent.name == "TAMU_B2.3_20-29_RGB"


def test_every_spelling_is_a_shot_numbered_by_k(tmp_path):
    root = scanner_tree(tmp_path)
    for desktop, n in ((1, 1), (2, 1), (3, 2), (7, 2), (12, 1), (20, 2), (21, 1), (51, 2),
                       (52, 2), (66, 1)):
        shots, issues = _shots(root, desktop)
        assert sorted(shots) == list(range(1, n + 1)), desktop
        assert not issues, (desktop, issues)
    shots, _ = _shots(root, 3)
    assert shots[1].src_step_dir.endswith("/3/allcomponentsrgb11_color")
    assert shots[2].src_step_dir.endswith("/3/allcomponentsrgb21_color")
    # D52: a bare "rgb1" is k=1, before "rgb21"
    shots, _ = _shots(root, 52)
    assert [ff.src_step_dir.rsplit("/", 1)[1] for _s, ff in sorted(shots.items())] == [
        "allcomponentsrgb1", "allcomponentsrgb21"]


def test_a_shot_is_a_burst_like_a_step_scan(tmp_path):
    root = scanner_tree(tmp_path)
    shots, _ = _shots(root, 1)
    ff = shots[1]
    assert ff.key == FrameKey(1, 1, "cmp_scan")
    assert ff.path.endswith("/1/allRGB1_color/P_0.png") and "\\" not in ff.path
    assert [Path(p).name for p in ff.aux["burst"]] == [f"P_{k}.png" for k in range(10)]
    assert ff.src_step_dir == ("TAMU_B2.3_1-9_RGB/TAMU_B2.0_1-9_RGB_allcompoentsrgb/"
                               "TAMU_B2.0_1-9_RGB_allcompoentsrgb/1/allRGB1_color")
    assert ff.ts


def test_sequences_and_broken_folders_are_left_out_and_said(tmp_path):
    root = scanner_tree(tmp_path)
    shots, issues = _shots(root, 64)
    names = {s: ff.src_step_dir.rsplit("/", 1)[1] for s, ff in shots.items()}
    # rgb11, rgb21 and the one without P_0; shots stay contiguous
    assert names == {1: "allcomponentsrgb11", 2: "allcomponentsrgb21",
                     3: "allcomponentsrgb41"}
    assert shots[3].path.endswith("allcomponentsrgb41/P_1.png")
    text = "\n".join(issues)
    assert ("cmp_scan: allcomponentsseq11 is not an RGB scan (its name is not an "
            "all-components RGB scan's, 30 images); left out") in text
    assert "cmp_scan: allcomponentsrgb31 is not an RGB scan (more than 20 images, 25 " \
           "images); left out" in text
    assert "cmp_scan: allcomponentsrgb41 has no P_0.png; using P_1.png" in text
    assert "cmp_scan: allcomponentsrgb41 (k=4) is shot 3" in text


# --------------------------------------------------------------------------- #
# the whole path: build-index -> load-index (a D66-style desktop)
# --------------------------------------------------------------------------- #
N_STEPS = 13
TS = [f"20250101_10{m:02d}00_000" for m in range(N_STEPS)]


def d66_tree(tmp_path: Path, shots: int = 2) -> dict:
    """13 steps in every sequence view, and ``shots`` all-components scans."""
    roots = build_tree(
        tmp_path, 66,
        oak1={f"{i + 1:03d}": [TS[i]] for i in range(N_STEPS)},
        oak2={f"{i + 1:03d}": [TS[i]] for i in range(N_STEPS)},
        scan_steps={s: 10 for s in range(1, N_STEPS + 1)},
        rs_steps=list(range(1, N_STEPS + 1)),
    )
    group = Path(roots["scanner_root"]) / "TAMU_B2.3_66-66_RGB"
    for k in range(1, shots + 1):
        _burst(group / "TAMU_B2.0_60-66_RGB_allcompoentsrgb" / "TAMU_color_60-66_B2.0"
               / "66" / f"allcomponentsrgb{k}1")
    return roots


def d66_db(tmp_path: Path) -> Db:
    db = Db(str(tmp_path / "tda.sqlite"))
    steps = [StepRec(66, s, "normal", f"Remove part {s}") for s in range(1, N_STEPS)]
    db.replace_steps(66, [StepRec(66, 1, "initial", "Initial")] + steps[1:]
                     + [StepRec(66, N_STEPS, "ignore", "All components")], [])
    return db


def test_scan_desktop_indexes_the_scanners_gallery(tmp_path, fixes):
    idx = scan_desktop(66, d66_tree(tmp_path), fixes)
    assert idx.n_steps == N_STEPS
    assert sorted(k.step for k in idx.frames if k.view == "cmp_scan") == [1, 2]
    # a gallery has no missing frames, and the sequence is what it was
    assert not [k for k in idx.missing if is_component_view(k.view)]
    assert len([k for k in idx.frames if k.view == "scan"]) == N_STEPS


def test_the_index_report_shows_the_shots_per_camera(tmp_path, fixes):
    """A desktop without its scanner gallery shows it (review round 2)."""
    from tda.core.index_report import write_report

    idx = {66: scan_desktop(66, d66_tree(tmp_path), fixes),
           67: scan_desktop(67, d66_tree(tmp_path / "b", shots=0), fixes)}
    idx[67].desktop = 67
    path = tmp_path / "report.md"
    write_report(idx, str(path))
    text = path.read_text(encoding="utf-8")
    assert "| 2 (scan 2 / oak1 0 / oak2 0 / rs 0) |" in text
    assert "| 0 (scan 0 / oak1 0 / oak2 0 / rs 0) |" in text


def test_a_version_3_index_is_refused_and_a_version_4_one_loaded(tmp_path, fixes):
    idx = {66: scan_desktop(66, d66_tree(tmp_path), fixes)}
    path = tmp_path / "index.json"
    save_index(idx, str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 4
    back = load_index(str(path))
    assert back[66].scan_gallery and not P.stale_index(back)
    assert back[66].frames[FrameKey(66, 1, "cmp_scan")].aux["burst"]
    payload["version"] = 3
    path.write_text(json.dumps(payload), encoding="utf-8")
    old = load_index(str(path))
    assert old[66].scan_gallery is False and P.stale_index(old) == [66]
    assert "version-3" in P.STALE_INDEX and "cmp_scan" in P.STALE_INDEX
    db = Db(str(tmp_path / "tda.sqlite"))
    with pytest.raises(P.StaleIndex):
        P.load_index_into_db(db, old)
    assert not db.frames_for(66, "cmp_scan")
    assert P.load_index_into_db(db, back)["gallery"] == 2


def test_load_index_gives_every_shot_a_segment_and_the_whole_image(tmp_path, fixes):
    db = d66_db(tmp_path)
    lines: list[str] = []
    P.load_index_into_db(db, {66: scan_desktop(66, d66_tree(tmp_path), fixes)},
                         log=lines.append)
    rows = {int(r["step"]): r for r in db.frames_for(66, "cmp_scan")}
    sheet = P.SHEET_SHOT_BASE + N_STEPS
    assert sorted(rows) == [1, 2, sheet]
    assert len(rows[1]["aux"]["burst"]) == 10
    segments = {int(s["seg"]): s for s in db.pose_segments(66, "cmp_scan")}
    assert sorted(segments) == [1, 2, sheet]
    for shot, seg in segments.items():
        assert (seg["start_step"], seg["end_step"], seg["ref_step"]) == (shot, shot, shot)
        assert list(seg["roi"]) == [0, 0, 1600, 1600]
    assert "gallery cmp_scan 3" in lines[0]


def test_the_all_components_row_moves_its_scanner_frame_to_shot_1013(tmp_path, fixes):
    """D66 step 13: the scanner's frame went nowhere before U42; now cmp_scan 1013."""
    db = d66_db(tmp_path)
    roots = d66_tree(tmp_path)
    index = {66: scan_desktop(66, roots, fixes)}
    scan13 = index[66].frames[FrameKey(66, N_STEPS, "scan")]
    lines: list[str] = []
    P.load_index_into_db(db, index, log=lines.append)
    sheet = P.SHEET_SHOT_BASE + N_STEPS
    assert sheet == 1013
    assert db.get_frame(FrameKey(66, N_STEPS, "scan")) is None
    row = db.get_frame(FrameKey(66, sheet, "cmp_scan"))
    assert row["path"] == scan13.path and row["aux"]["sheet_step"] == N_STEPS
    assert row["aux"]["burst"] == scan13.aux["burst"]        # build-cache chooses on it
    assert f"step {N_STEPS} (scan) is a photo of the laid-out parts: moved to cmp_scan " \
           f"shot {sheet}" in "\n".join(lines)
    for camera in ("oak1", "oak2", "rs"):
        assert db.get_frame(FrameKey(66, sheet, f"cmp_{camera}")) is not None
    # a second load-index has nothing to move and says nothing
    again: list[str] = []
    P.load_index_into_db(db, {66: scan_desktop(66, roots, fixes)}, log=again.append)
    assert not any("photo of the laid-out parts" in line for line in again)
    assert db.get_frame(FrameKey(66, N_STEPS, "scan")) is None
    assert sorted(int(r["step"]) for r in db.frames_for(66, "cmp_scan")) == [1, 2, sheet]


def test_a_shot_that_disappears_is_dropped_unless_worked_on(tmp_path, fixes):
    db = d66_db(tmp_path)
    P.load_index_into_db(db, {66: scan_desktop(66, d66_tree(tmp_path / "a", shots=2), fixes)})
    lines: list[str] = []
    P.load_index_into_db(db, {66: scan_desktop(66, d66_tree(tmp_path / "b", shots=1), fixes)},
                         log=lines.append)
    assert sorted(int(r["step"]) for r in db.frames_for(66, "cmp_scan")) == [
        1, P.SHEET_SHOT_BASE + N_STEPS]
    assert any("cmp_scan shot 2" in line and "dropped" in line for line in lines)


# --------------------------------------------------------------------------- #
# build-cache: the scanner's burst choice
# --------------------------------------------------------------------------- #
def test_build_cache_chooses_a_burst_image_for_cmp_scan(tmp_path):
    folder = tmp_path / "src" / "allcomponentsrgb11"
    burst = _write_burst(folder, [_texture(seed=s) for s in range(10)])
    assert cv2.imwrite(burst[0], _texture(seed=1) * 0 + 9)       # P_0 unexposed
    key = FrameKey(7, 1, "cmp_scan")
    index = {7: DesktopIndex(7, 0, {key: FrameFile(key, burst[0], {"burst": burst})})}
    stats = build_cache(index, str(tmp_path / "cache"), views=("cmp_scan",))
    assert not stats["failures"]
    manifest = json.loads((tmp_path / "cache" / "cmp_scan" / "D07" / "manifest.json")
                          .read_text("utf-8"))
    rec = manifest["1"]
    assert rec["reason"] == "p0_dark" and rec["chosen"] != 0 and len(rec["metrics"]) == 10
    dest = tmp_path / "cache" / "cmp_scan" / "D07" / "s001.png"
    assert dest.read_bytes() == Path(rec["src"]).read_bytes()
    assert stats["non_p0"][0]["view"] == "cmp_scan"
    # a re-run re-decides from the stored metrics and copies nothing
    again = build_cache(index, str(tmp_path / "cache"), views=("cmp_scan",))
    assert again["skipped"] == 1 and again["copied"] == 0


# --------------------------------------------------------------------------- #
# the exports, status and check
# --------------------------------------------------------------------------- #
def test_export_coco_of_cmp_scan(tmp_path):
    session = GS.make_session(tmp_path, view="cmp_scan")
    _confirm_shot(session, 1)
    doc = export_coco(session.db, session.tax, [GS.DESKTOP], "cmp_scan",
                      str(tmp_path / "coco_cmp_scan.json"))
    assert doc["info"]["split"] == "components" and doc["info"]["view"] == "cmp_scan"
    assert len(doc["images"]) == 1 and len(doc["annotations"]) == 1
    image = doc["images"][0]
    assert image["id"] == image_id(GS.DESKTOP, 1, "cmp_scan")
    assert image["extra"]["shot"] == 1 and image["extra"]["source"].endswith(".png")
    assert image["tier"] == "gold"


def _p0_dark_shot(db, src_dir: Path, cache_dir: Path, shot: int = 1) -> list[str]:
    """Give ``cmp_scan`` shot ``shot`` a burst whose ``P_0`` is dark, and a cache
    manifest that chose ``P_9`` for it (as build-cache does, ``p0_dark``)."""
    burst = _write_burst(src_dir, [_texture(seed=s) for s in range(10)])
    assert cv2.imwrite(burst[0], _texture(seed=1) * 0 + 9)
    key = FrameKey(GS.DESKTOP, shot, "cmp_scan")
    row = db.get_frame(key)
    aux = dict(row.get("aux") or {})
    aux["burst"] = burst
    db.upsert_frame(key, row["path"], aux, row.get("ts"), {"missing": False})
    manifest = cache_dir / "cmp_scan" / f"D{GS.DESKTOP:02d}" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({str(shot): {
        "chosen": 9, "reason": "p0_dark", "src": burst[9],
        "metrics": [{"path": p, "p_index": i} for i, p in enumerate(burst)]}}),
        encoding="utf-8")
    return burst


def test_export_coco_names_the_burst_image_the_cache_chose(tmp_path):
    """``extra.source`` is the picture drawn on: P_9 when P_0 was dark (review S1)."""
    session = GS.make_session(tmp_path, view="cmp_scan")
    burst = _p0_dark_shot(session.db, tmp_path / "src" / "allcomponentsrgb11",
                          tmp_path / "cache2")
    _confirm_shot(session, 1)
    doc = export_coco(session.db, session.tax, [GS.DESKTOP], "cmp_scan",
                      str(tmp_path / "coco.json"), cache_dir=str(tmp_path / "cache2"))
    extra = doc["images"][0]["extra"]
    assert extra["source"] == burst[9] and extra["source"].endswith("P_9.png")
    assert extra["burst_choice"] == "p0_dark"
    # no cache: the frame's own path, and no choice to name
    bare = export_coco(session.db, session.tax, [GS.DESKTOP], "cmp_scan",
                       str(tmp_path / "coco_bare.json"), cache_dir=str(tmp_path / "none"))
    extra = bare["images"][0]["extra"]
    assert extra["source"] == session.db.get_frame(FrameKey(GS.DESKTOP, 1, "cmp_scan"))["path"]
    assert "burst_choice" not in extra


def test_a_manifest_of_another_burst_is_not_believed(tmp_path):
    from tda.core.export.coco import _shot_source

    frame = {"path": "F:/a/P_0.png", "aux": {"burst": ["F:/a/P_0.png", "F:/a/P_9.png"]}}
    key = FrameKey(1, 2, "cmp_scan")
    good = {"2": {"src": "F:/a/P_9.png", "reason": "p0_dark"}}
    stale = {"2": {"src": "F:/old/P_9.png", "reason": "p0_dark"}}
    assert _shot_source(frame, key, good)["source"] == "F:/a/P_9.png"
    out = _shot_source(frame, key, stale)
    assert out["source"] == "F:/a/P_0.png" and "burst_choice" not in out
    assert "burst_choice" not in _shot_source({"path": "F:/x.jpg"}, FrameKey(1, 2, "cmp_oak1"))


def test_cli_export_coco_reads_the_workspaces_cache(tmp_path, capsys):
    from tda.core.db import Db as _Db
    from tda.core.taxonomy import load_taxonomy
    from tda.core.truth import TruthService
    from tda.ui.session import AnnotationSession

    env = GS.make_env(tmp_path)
    db = _Db(env["cfg"]["db_path"])
    burst = _p0_dark_shot(db, tmp_path / "src" / "allcomponentsrgb11",
                          Path(env["cfg"]["cache_dir"]))
    tax = load_taxonomy()
    session = AnnotationSession(db, tax, TruthService(db, tax), env["cfg"]["cache_dir"],
                                "tester")
    session.sweeper_enabled = False
    session.open(GS.DESKTOP, "cmp_scan")
    _confirm_shot(session, 1)
    session.close(force=True)
    db.close()
    out = tmp_path / "coco_cli.json"
    assert _run(env, "export-coco", "--desktops", str(GS.DESKTOP), "--view", "cmp_scan",
                "--out", str(out)) == 0, capsys.readouterr().out
    image = json.loads(out.read_text(encoding="utf-8"))["images"][0]
    assert image["extra"]["source"] == burst[9]
    assert image["extra"]["burst_choice"] == "p0_dark"


def _run(env: dict, *argv: str) -> int:
    from tda.cli import main

    return main(["--paths", env["paths"], *argv])


def test_status_and_check_know_four_gallery_views(tmp_path, capsys):
    env = GS.make_env(tmp_path)
    assert _run(env, "status") == 0
    out = capsys.readouterr().out
    head, _sep, gallery = out.partition("零件合照 / components gallery")
    for view in COMPONENT_VIEWS:
        assert view in gallery and view not in head, view
    assert _run(env, "status", "--desktop", str(GS.DESKTOP)) == 0
    assert "cmp_scan" in capsys.readouterr().out
    code = _run(env, "check", "--desktop", str(GS.DESKTOP), "--view", "cmp_scan")
    out = capsys.readouterr().out
    assert code == 1 and f"D{GS.DESKTOP} cmp_scan: {GS.SHOTS} shots" in out


def test_cli_app_accepts_cmp_scan(tmp_path, monkeypatch):
    env = GS.make_env(tmp_path)
    seen: dict = {}
    monkeypatch.setattr("tda.ui.app.main", lambda **kw: seen.update(kw) or 0)
    assert _run(env, "app", "--desktop", str(GS.DESKTOP), "--view", "cmp_scan",
                "--annotator", "chang") == 0
    assert seen["view"] == "cmp_scan"


# --------------------------------------------------------------------------- #
# the window: a toolbar button and no key
# --------------------------------------------------------------------------- #
def test_the_scanners_gallery_has_no_key():
    action = A.action_named("view_cmp_scan")
    assert action.keys == () and not A.combos_of(action)
    assert action.slot == "act_set_view" and action.args == ("cmp_scan",)
    # a key row of the guide's table like any other ("| `" -- the prose budget
    # of tests/test_u*_docs.py does not count it)
    assert f"| `{A.NO_KEY_MARK}` {A.NO_KEY_TEXT} | {action.label_zh} |" in A.shortcut_markdown()
    used = {key for a in A.ACTIONS for key in a.keys}
    for key in ("F1", "F2", "F3", "F4", "F6", "F7", "F8"):
        assert key in used
    assert "F9" not in used          # no key was taken for it


SPEC = REPO_ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
GUIDE = REPO_ROOT / "docs" / "annotation_guide.md"


def test_the_docs_say_it():
    spec = SPEC.read_text(encoding="utf-8")
    gallery = spec.split("### 3.2b 零件合照", 1)[1].split("### 3.2c", 1)[0]
    for needed in ("**扫描仪的零件合照 `cmp_scan`**（v1.41，U42）", "allcompoentsrgb",
                   "`allcomponentsseq11`", "**索引版本 4**", "**没有快捷键**"):
        assert needed in gallery, needed
    changes = spec.split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # on top when it landed; U45's v1.44 has gone above it since
    assert any(e.startswith("- **v1.41（2026-10-30）**：**扫描仪的零件合照 `cmp_scan`**（U42")
               for e in entries)
    guide = GUIDE.read_text(encoding="utf-8")
    assert "**零件合照（`F6`–`F8`；扫描仪的点工具栏按钮）。** 零件拆完摆开拍/扫的图" in guide
    quick = (REPO_ROOT / "docs" / "trial_quickstart.md").read_text(encoding="utf-8")
    section = quick.split("## 4.5 零件合照", 1)[1].split("\n## ", 1)[0]
    for needed in ("单独的四个视图", "`零件合照·扫描仪`（**没有快捷键**", "`--view cmp_scan`",
                   "`coco_cmp_scan.json`"):
        assert needed in section, needed
    assert "`extra` 写 `shot`（帧号）、`source`（标注时画在上面的那张原图" in gallery
    assert A.shortcut_markdown() in guide


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_toolbar_button_opens_the_scanners_gallery(qapp, tmp_path):
    from PySide6.QtWidgets import QApplication

    from app_scene import StubSamQueue, close_window, make_paths
    from tda.core.truth import TruthService
    from tda.ui.app import MainWindow
    from tda.ui.session import AnnotationSession

    cfg = make_paths(tmp_path)
    Path(cfg["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    db, tax, cache_dir = GS.make_db(tmp_path, db_path=cfg["db_path"],
                                    cache_dir=Path(cfg["cache_dir"]))
    session = AnnotationSession(db, tax, TruthService(db, tax), str(cache_dir), "tester")
    session.sweeper_enabled = False
    session.open(GS.DESKTOP, "cmp_oak1")
    win = MainWindow(session, cfg, "tester", sam_queue=StubSamQueue())
    try:
        win.show()
        QApplication.processEvents()
        button = win.view_buttons["cmp_scan"]
        assert button.text() == "零件合照·扫描仪"            # no "()" for no key
        assert win.view_buttons["cmp_oak1"].text() == "零件合照·OAK1 (F6)"
        order = list(win.view_buttons)
        assert order.index("cmp_scan") == order.index("rs") + 1
        button.click()
        QApplication.processEvents()
        assert win.session.view == "cmp_scan" and button.isChecked()
        assert f"零件合照 扫描仪 第 1/{GS.SHOTS} 张" in win.windowTitle()
    finally:
        close_window(win)
