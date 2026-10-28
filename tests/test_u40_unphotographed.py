"""Task U40: steps no camera photographed (``unphotographed.<d>.steps``).

D63's sheet has 37 rows and every camera 34 captures: the operator did three
steps nobody photographed. Declared once per desktop, the rule of the scanner's
``skipped_steps`` then holds in all four views -- the ``i``-th capture is on the
``i``-th step that is not declared -- and the declared steps are missing frames
everywhere. Synthetic source trees (``test_index.build_tree``); nothing touches
the raw drive.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_cli import LOG_FIXTURES, d13_steps, env, open_db, run  # noqa: F401
from test_index import build_tree
from test_u25_pipeline import _keyframe

from tda.core.index import (
    EXTRA_SHOTS,
    DesktopIndex,
    capture_rows,
    capture_steps,
    collapse_rows,
    folder_steps,
    load_index,
    save_index,
    scan_desktop,
)
from tda.core.index_fixes import (
    declared_missing,
    unphotographed_steps,
    verify_declared_missing,
    verify_unphotographed,
)
from tda.core.index_report import write_report
from tda.core.model import VIEWS, FrameKey, StepRec, StepType

REPO_ROOT = Path(__file__).resolve().parents[1]
D = 96


def _ts(k: int) -> str:
    return f"20250101_10{k:02d}00_000"


def _tree(tmp_path: Path, n_captures: int, scan: list[int] | None = None,
          rs: list[int] | None = None, oak2_extra: list[int] = ()) -> dict:
    """``n_captures`` Camera_1/Camera_2 captures and RealSense folders (one a
    minute), scanner folders ``scan`` (default: one per capture)."""
    oak = {f"{k:03d}": [_ts(k)] for k in range(1, n_captures + 1)}
    oak2 = {k: list(v) for k, v in oak.items()}
    for k in oak2_extra:                      # a Camera_2 capture without a partner
        oak2.setdefault(f"{k:03d}", []).append(f"20250101_11{k:02d}00_000")
    rs = list(range(1, n_captures + 1)) if rs is None else rs
    return build_tree(tmp_path, D, oak1=oak, oak2=oak2,
                      scan_steps={n: 9 for n in (scan if scan is not None
                                                 else range(1, n_captures + 1))},
                      rs_steps=rs, rs_mtimes={k: _ts(k) for k in rs})


def _fixes(steps: list[int] | None, scan: dict | None = None, rs: dict | None = None,
           oak: dict | None = None) -> dict:
    out: dict = {"scan": {}, "oak": {}, "rs": {}, "unphotographed": {}}
    if steps is not None:
        out["unphotographed"][D] = {"steps": steps, "reason": "test"}
    for view, entry in (("scan", scan), ("rs", rs), ("oak", oak)):
        if entry is not None:
            out[view][D] = {**entry, "reason": "test"}
    return out


def _dirs(idx, view: str) -> dict[int, str]:
    return {k.step: ff.src_step_dir for k, ff in idx.frames.items() if k.view == view}


def _line(idx, head: str) -> str:
    found = [i for i in idx.issues if i.startswith(head)]
    assert len(found) == 1, (head, idx.issues)
    return found[0]


# --------------------------------------------------------------------------- #
# the declaration
# --------------------------------------------------------------------------- #
def test_unphotographed_steps_parse():
    fixes = {"unphotographed": {63: {"steps": [33, 31, 32, 31], "reason": "r"},
                                64: "not a dict", 65: {"reason": "no steps"}}}
    assert unphotographed_steps(fixes, 63) == [31, 32, 33]
    assert unphotographed_steps(fixes, 64) == []
    assert unphotographed_steps(fixes, 65) == []
    assert unphotographed_steps(fixes, 66) == []
    assert unphotographed_steps({}, 63) == []


def test_declared_missing_adds_them_to_every_view():
    fixes = {"unphotographed": {63: {"steps": [31, 32]}},
             "scan": {63: {"skipped_steps": [5]}}, "rs": {63: {"missing_steps": [7]}}}
    assert declared_missing(fixes, "scan", 63) == {5, 31, 32}
    assert declared_missing(fixes, "rs", 63) == {7, 31, 32}
    assert declared_missing(fixes, "oak1", 63) == {31, 32}
    assert declared_missing(fixes, "oak2", 63) == {31, 32}
    assert declared_missing(fixes, "oak1", 62) == set()


# --------------------------------------------------------------------------- #
# captures -> steps
# --------------------------------------------------------------------------- #
def test_capture_steps_without_a_declaration_is_collapse_rows():
    assert capture_steps(10, [4, 8]) == collapse_rows(10, [4, 8])
    assert capture_rows(4, [3]) == [0, 1, 2, 3, 4]


def test_capture_steps_seven_rows_five_captures():
    """7 sheet rows, dupli row 6, steps 2 and 3 unphotographed: 6 steps, and the
    5 captures are rows 1 4 5 6 7 -> steps 1 4 5 5(extra) 6."""
    assert capture_rows(5, [6], [2, 3]) == [0, 1, 4, 5, 6, 7]
    step_of, extra = capture_steps(5, [6], [2, 3])
    assert step_of[1:] == [1, 4, 5, 5, 6] and extra == {4}
    # D63-like: 37 rows, no dupli, 34 captures
    step_of, extra = capture_steps(34, [], [31, 32, 33])
    assert step_of[30:] == [30, 34, 35, 36, 37] and not extra
    # a dupli row after the unphotographed one is numbered in the sheet's rows
    assert capture_rows(5, [5], [2]) == [0, 1, 3, 4, 5, 6]
    assert capture_steps(5, [5], [2]) == ([0, 1, 3, 4, 4, 5], {4})


def test_every_view_skips_the_declared_steps(tmp_path):
    """The example above on disk: OAK1/OAK2/RealSense and the scanner (which
    also photographed the dupli row: its duplicate folder 4)."""
    roots = _tree(tmp_path, 5)
    fixes = _fixes([2, 3], scan={"duplicate_folders": [4]})
    idx = scan_desktop(D, roots, fixes, dupli_rows=[6], sheet_rows=7)
    assert (idx.n_steps, idx.n_rows, idx.dupli_rows, idx.unphotographed) == (6, 5, [6], [2, 3])
    for view in ("oak1", "oak2", "rs"):
        assert _dirs(idx, view) == {1: "001", 4: "002", 5: "003", 6: "005"}, view
        shots = idx.frames[FrameKey(D, 5, view)].aux[EXTRA_SHOTS]
        # the extra shot names its sheet row (6), not its capture number (4)
        assert [(s["src_step_dir"], s["row"]) for s in shots] == [("004", 6)], view
    assert _dirs(idx, "scan") == {1: "RGB11", 4: "RGB21", 5: "RGB31", 6: "RGB51"}
    assert [s["src_step_dir"] for s in idx.frames[FrameKey(D, 5, "scan")].aux[EXTRA_SHOTS]] \
        == ["RGB41"]
    assert sorted((k.step, k.view) for k in idx.missing) == sorted(
        (s, v) for s in (2, 3) for v in VIEWS)
    assert "fix:unphotographed[96] steps [2, 3] applied" in _line(
        idx, "fix:unphotographed[96] steps [2, 3] applied")
    ok = _line(idx, "fix:unphotographed[96] steps [2, 3]:")
    assert "MISMATCH" not in ok and "6 steps (as the sheet)" in ok
    assert "oak1 4 + 1 extra" in ok and "scan 4 + 1 extra" in ok
    assert "MISMATCH" not in " ".join(idx.issues)
    # the dupli line names the capture's own folder
    assert any("row 6 (cam1 004) -> extra shot of step 5" in i for i in idx.issues)
    # the sheet's 7 rows are the 5 captures and the 2 declared steps: no warning
    assert not [i for i in idx.issues if i.startswith("rows: the sheet has")]


def test_with_the_scanners_own_skipped_steps_the_union_applies(tmp_path):
    """No dupli; step 3 unphotographed, and the scanner also skipped step 5."""
    roots = _tree(tmp_path, 6, scan=[1, 2, 3, 4, 5])
    fixes = _fixes([3], scan={"skipped_steps": [5]})
    idx = scan_desktop(D, roots, fixes, dupli_rows=[], sheet_rows=7)
    assert idx.n_steps == 7
    assert _dirs(idx, "oak1") == {1: "001", 2: "002", 4: "003", 5: "004", 6: "005",
                                  7: "006"}
    assert _dirs(idx, "scan") == {1: "RGB11", 2: "RGB21", 4: "RGB31", 6: "RGB41",
                                  7: "RGB51"}
    scan_line = _line(idx, "fix:scan[96] duplicate_folders")
    assert ("skipped steps [3, 5] (scan.96.skipped_steps [5]; unphotographed.96.steps [3])"
            in scan_line) and "MISMATCH" not in scan_line
    assert "declared missing [3, 5]" in scan_line
    assert _line(idx, "fix:scan[96] declared missing").endswith(
        "confirmed absent (of which unphotographed.96.steps: [3])")
    assert _line(idx, "fix:oak2[96] declared missing").endswith(
        "confirmed absent (unphotographed.96.steps)")
    assert "MISMATCH" not in _line(idx, "fix:unphotographed[96] steps [3]:")


def test_a_desktop_without_the_declaration_is_unchanged(tmp_path):
    roots = _tree(tmp_path, 5)
    idx = scan_desktop(D, roots, _fixes(None), dupli_rows=[], sheet_rows=7)
    assert idx.n_steps == 5 and idx.unphotographed == []
    assert _dirs(idx, "oak1") == {s: f"{s:03d}" for s in range(1, 6)}
    assert not [i for i in idx.issues if "unphotographed" in i]
    assert not [i for i in idx.issues if i.startswith("fix:oak1")]
    assert any("the sheet has 7 rows but Camera_1 5 captures" in i for i in idx.issues)


# --------------------------------------------------------------------------- #
# the assertion: MISMATCH
# --------------------------------------------------------------------------- #
def test_mismatch_a_declared_step_has_a_frame(tmp_path):
    """Row 4 is a dupli of row 3 -- the step declared unphotographed: its
    capture stands in for step 3, so the step was photographed after all."""
    roots = _tree(tmp_path, 5, scan=[1, 2, 3, 4])
    idx = scan_desktop(D, roots, _fixes([3], scan={"duplicate_folders": []}),
                       dupli_rows=[4], sheet_rows=6)
    line = _line(idx, "fix:unphotographed[96] steps [3] MISMATCH")
    assert "oak1 has a frame on declared step(s) 3 (003)" in line
    assert "rs has a frame on declared step(s) 3 (003)" in line
    assert "scan has" not in line
    assert any(i.startswith("fix:oak1[96] MISMATCH: declared-missing logical steps present "
                            "in index: [3] (unphotographed.96.steps)") for i in idx.issues)


def test_mismatch_a_step_without_a_capture_that_nothing_declares(tmp_path):
    """The RealSense has no folder 004: step 5 has no frame and no declaration."""
    roots = _tree(tmp_path, 5, rs=[1, 2, 3, 5])
    idx = scan_desktop(D, roots, _fixes([2]), dupli_rows=[], sheet_rows=6)
    line = _line(idx, "fix:unphotographed[96] steps [2] MISMATCH")
    assert "rs: steps [5] have no capture and are not declared missing" in line
    # declared, it holds
    idx = scan_desktop(D, roots, _fixes([2], rs={"missing_steps": [5]}), dupli_rows=[],
                       sheet_rows=6)
    assert "rs 4)" in _line(idx, "fix:unphotographed[96] steps [2]:")
    assert _line(idx, "fix:rs[96] declared missing").endswith(
        "(of which unphotographed.96.steps: [2])")


def test_mismatch_a_capture_without_a_step(tmp_path):
    """A Camera_2 capture with no Camera_1 partner, a RealSense folder past the
    captures and a scanner folder past the steps all found no step."""
    roots = _tree(tmp_path, 5, scan=[1, 2, 3, 4, 5, 6], rs=[1, 2, 3, 4, 5, 6],
                  oak2_extra=[5])
    idx = scan_desktop(D, roots, _fixes([2]), dupli_rows=[], sheet_rows=6)
    line = _line(idx, "fix:unphotographed[96] steps [2] MISMATCH")
    assert "oak2: captures ['005 (20250101_110500_000)'] have no step" in line
    assert "rs: captures ['006'] have no step" in line
    assert "scan: captures ['RGB61'] have no step" in line


def test_mismatch_the_sheet_counts_other_steps(tmp_path):
    roots = _tree(tmp_path, 5)
    idx = scan_desktop(D, roots, _fixes([2, 3]), dupli_rows=[], sheet_rows=8)
    line = _line(idx, "fix:unphotographed[96] steps [2, 3] MISMATCH")
    assert "the sheet has 8 steps, the captures and the declared steps make 7" in line
    assert any("the sheet has 8 rows but Camera_1 5 captures and 2 unphotographed step(s) "
               "make 7" in i for i in idx.issues)


def test_a_step_beyond_the_steps_is_dropped(tmp_path):
    """5 captures and 2 declared steps make 7: step 9 is no step; 8 then neither."""
    roots = _tree(tmp_path, 5)
    idx = scan_desktop(D, roots, _fixes([3, 9]), dupli_rows=[], sheet_rows=6)
    assert idx.unphotographed == [3] and idx.n_steps == 6
    assert any(i.startswith("fix:unphotographed[96] MISMATCH: step(s) [9] are not steps of "
                            "the 7") for i in idx.issues)
    roots2 = _tree(tmp_path / "b", 5)
    idx = scan_desktop(D, roots2, _fixes([7, 9]), dupli_rows=[], sheet_rows=6)
    assert idx.unphotographed == [] and idx.n_steps == 5
    bad = [i for i in idx.issues if "MISMATCH: step(s)" in i]
    assert len(bad) == 2 and "[9]" in bad[0] and "[7]" in bad[1]


def test_verify_unphotographed_on_its_own():
    fixes = {"unphotographed": {5: {"steps": [2]}}}
    frames = {FrameKey(5, s, v): type("F", (), {"src_step_dir": f"{s:03d}", "aux": {}})()
              for s in (1, 3) for v in VIEWS}
    missing = [FrameKey(5, 2, v) for v in VIEWS]
    ok = verify_unphotographed(5, fixes, [2], 3, frames, missing, {}, 3)
    assert ok == ["fix:unphotographed[5] steps [2]: no view photographed them; 3 steps (as "
                  "the sheet), every view's captures on the other 2 (scan 2, oak1 2, oak2 2, "
                  "rs 2)"]
    assert verify_unphotographed(5, {}, [], 3, frames, missing) == []
    frames[FrameKey(5, 2, "oak2")] = frames[FrameKey(5, 1, "oak2")]
    bad = verify_unphotographed(5, fixes, [2], 3, frames, missing, {"rs": ["009"]}, 4)
    assert len(bad) == 1 and "MISMATCH" in bad[0]
    for part in ("oak2 has a frame on declared step(s) 2 (001)",
                 "rs: captures ['009'] have no step",
                 "the sheet has 4 steps, the captures and the declared steps make 3"):
        assert part in bad[0]


def test_verify_declared_missing_counts_them_in_all_four_views():
    fixes = {"unphotographed": {5: {"steps": [2]}}, "scan": {5: {"missing_steps": [4]}}}
    missing = [FrameKey(5, 2, v) for v in VIEWS] + [FrameKey(5, 4, "scan")]
    assert verify_declared_missing(5, fixes, 4, missing) == [
        "fix:scan[5] declared missing steps [2, 4] confirmed absent "
        "(of which unphotographed.5.steps: [2])",
        "fix:oak1[5] declared missing steps [2] confirmed absent (unphotographed.5.steps)",
        "fix:oak2[5] declared missing steps [2] confirmed absent (unphotographed.5.steps)",
        "fix:rs[5] declared missing steps [2] confirmed absent (unphotographed.5.steps)",
    ]
    present = verify_declared_missing(5, fixes, 4, [FrameKey(5, 4, "scan")])
    assert present[1] == ("fix:oak1[5] MISMATCH: declared-missing logical steps present in "
                          "index: [2] (unphotographed.5.steps)")


# --------------------------------------------------------------------------- #
# report, JSON, folder numbers, scan-align
# --------------------------------------------------------------------------- #
def test_report_says_which_steps_no_camera_photographed(tmp_path):
    roots = _tree(tmp_path, 5)
    idx = scan_desktop(D, roots, _fixes([2, 3], scan={"duplicate_folders": [4]}),
                       dupli_rows=[6], sheet_rows=7)
    out = tmp_path / "report.md"
    write_report({D: idx}, str(out))
    text = out.read_text(encoding="utf-8")
    assert "- steps no camera photographed (`unphotographed`): 2 on D96 (2-3 (n=2))" in text
    assert "| 96 | 6 | 5 (6) +2 unphotographed |" in text
    assert "  - `oak1`: 2-3 (n=2) -- of which unphotographed (no view): 2-3 (n=2)" in text
    assert "## Steps no camera photographed" in text
    assert "- **Desktop 96**: steps 2-3 (n=2)" in text
    # a report without any keeps its old shape (and without the declaration
    # the 5 captures hold no dupli row 6)
    plain = scan_desktop(D, roots, _fixes(None), dupli_rows=[6], sheet_rows=7)
    write_report({D: plain}, str(out))
    text = out.read_text(encoding="utf-8")
    assert "unphotographed" not in text and "| 96 | 5 | 5 | 20 |" in text


def test_save_load_and_folder_steps(tmp_path):
    roots = _tree(tmp_path, 5)
    idx = scan_desktop(D, roots, _fixes([2, 3], scan={"duplicate_folders": [4]}),
                       dupli_rows=[6], sheet_rows=7)
    path = tmp_path / "index.json"
    save_index({D: idx}, str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 3 and payload["desktops"]["96"]["unphotographed"] == [2, 3]
    back = load_index(str(path))[D]
    assert back == idx and back.unphotographed == [2, 3]
    assert folder_steps(back, "oak1") == {1: 1, 2: 4, 3: 5, 4: 5, 5: 6}
    assert folder_steps(back, "rs") == folder_steps(back, "oak1")
    assert folder_steps(back, "scan") == {1: 1, 2: 4, 3: 5, 4: 5, 5: 6}
    # an index written before U40 has no such key: nothing was declared
    del payload["desktops"]["96"]["unphotographed"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_index(str(path))[D].unphotographed == []


def test_scan_align_rows_are_captures(tmp_path):
    from tda.core.scan_align import declared_rows, rows_of_index

    roots = _tree(tmp_path, 5)
    idx = scan_desktop(D, roots, _fixes([2, 3], scan={"duplicate_folders": [4]}),
                       dupli_rows=[6], sheet_rows=7)
    paths, main_row, extras = rows_of_index(idx, "oak1")
    assert [p.replace("\\", "/").split("/")[-2] for p in paths] == [
        "001", "002", "003", "004", "005"]
    assert main_row == {1: 1, 4: 2, 5: 3, 6: 5} and extras == {5: [4]}
    assert declared_rows(idx, main_row, extras) == {"RGB11": 1, "RGB21": 2, "RGB31": 3,
                                                     "RGB41": 4, "RGB51": 5}


# --------------------------------------------------------------------------- #
# load-index: the D63 case on a small scale
# --------------------------------------------------------------------------- #
def test_load_index_moves_the_last_photos_past_the_unphotographed_steps(env, tmp_path):
    """7 steps, the last two turn the chassis; every camera has 5 captures. The
    old index put them on steps 1-5; declared, steps 2-3 are unphotographed and
    the captures are steps 1, 4, 5, 6, 7. A keyframe the annotator drew on the
    last frame was clamped to the step table's last step (7): it stays there --
    the last capture's step -- in the third pose segment, and each reorient
    step opens a segment of its own in all four views."""
    roots = _tree(tmp_path, 5)
    old = scan_desktop(D, roots, _fixes(None), dupli_rows=[], sheet_rows=7)
    new = scan_desktop(D, roots, _fixes([2, 3]), dupli_rows=[], sheet_rows=7)
    assert old.n_steps == 5 and new.n_steps == 7
    old_path, new_path = tmp_path / "old.json", tmp_path / "new.json"
    save_index({D: old}, str(old_path))
    save_index({D: new}, str(new_path))
    assert run(env, "load-index", "--desktops", str(D), "--index", str(old_path)) == 0
    db = open_db(env)
    try:
        steps = [StepRec(D, k, StepType.REORIENT.value if k >= 6 else StepType.NORMAL.value,
                         f"row {k}") for k in range(1, 8)]
        db.replace_steps(D, steps, [])
        last = _keyframe(db, D, "scan", "chassis", 7)
        on_four = _keyframe(db, D, "oak1", "psu.01", 4)
    finally:
        db.close()
    assert run(env, "load-index", "--desktops", str(D), "--index", str(new_path)) == 0
    db = open_db(env)
    try:
        for view in VIEWS:
            rows = {r["step"]: r for r in db.frames_for(D, view)}
            assert sorted(rows) == list(range(1, 8)), view
            assert sorted(s for s, r in rows.items() if r["missing"]) == [2, 3], view
            # photo 5 is step 7 now, photo 2 step 4
            assert rows[7]["path"] == new.frames[FrameKey(D, 7, view)].path
            assert "005" in rows[7]["path"].replace("\\", "/") or "RGB51" in rows[7]["path"]
            segs = sorted((s["start_step"], s["end_step"]) for s in db.pose_segments(D, view))
            assert segs == [(1, 5), (6, 6), (7, 7)], view
        kfs = {kf.id: kf for view in ("scan", "oak1") for kf in db.keyframes(D, view)}
        assert kfs[last].anchor_step == 7 and kfs[last].pose_segment == 3
        # the capture it was drawn on (capture 4) is step 6 now
        assert kfs[on_four].anchor_step == 6
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# the real yaml
# --------------------------------------------------------------------------- #
def test_the_real_d63_declaration():
    import yaml

    with open(REPO_ROOT / "configs" / "index_fixes.yaml", encoding="utf-8") as fh:
        fixes = yaml.safe_load(fh)
    entry = fixes["unphotographed"][63]
    assert entry["steps"] == [14, 15, 16]
    reason = " ".join(entry["reason"].split())
    assert "UNCERTAIN" not in reason and "RGB321" in reason
    for needed in ("The capture number is the sheet's Sequence Number",
                   "the CPU at capture 26 (number 26", "Quadro card at capture 28 (number 28",
                   "repeat 11-13", "already off", "actions that got no photo of their own"):
        assert needed in reason, needed
    assert unphotographed_steps(fixes, 63) == [14, 15, 16]
    for desktop, value in fixes["unphotographed"].items():
        assert isinstance(desktop, int) and value.get("reason"), desktop
    # D63 declares nothing for the scanner alone
    assert 63 not in (fixes.get("scan") or {})


@pytest.mark.parametrize("view", VIEWS)
def test_the_real_yaml_declares_d63s_steps_missing_in_every_view(view):
    import yaml

    with open(REPO_ROOT / "configs" / "index_fixes.yaml", encoding="utf-8") as fh:
        fixes = yaml.safe_load(fh)
    assert {14, 15, 16} <= declared_missing(fixes, view, 63)


# --------------------------------------------------------------------------- #
# the data audit
# --------------------------------------------------------------------------- #
def test_the_audit_checks_such_a_desktop_by_step_number():
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import data_audit as DA

    di = SimpleNamespace(desktop=63, n_rows=5, n_steps=7, unphotographed=[2, 3])
    rows = [{"Sequence Number": str(q)} for q in (1, 2, 3, 2, 3, 4, 5)]
    li = SimpleNamespace(sheet_rows=7, dupli_rows=[])
    align, note = DA.capture_alignment(di, li, rows)
    # step 4 is the first frame after the unphotographed steps: it answers for them
    assert align == {1: [1], 4: [2, 3, 4], 5: [5], 6: [6], 7: [7]}
    assert "steps [2, 3] no camera photographed (`unphotographed.63.steps`, U40)" in note
