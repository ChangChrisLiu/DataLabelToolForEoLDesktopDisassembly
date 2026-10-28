"""Task U25: a ``dupli`` row is not a step -- the index collapses it.

Synthetic source trees (``test_index.build_tree``) with the sheet's ``dupli``
rows handed to :func:`scan_desktop`; nothing touches the raw drive.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from test_index import build_tree

from tda.core.index import (
    EXTRA_SHOTS,
    collapse_rows,
    folder_steps,
    load_index,
    save_index,
    scan_desktop,
    scan_step,
)
from tda.core.index_fixes import verify_declared_missing
from tda.core.index_report import write_report
from tda.core.model import FrameKey

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ts(k: int) -> str:
    return f"20250101_10{k:02d}00_000"


def _tree(tmp_path: Path, desktop: int, n_rows: int, scan: list[int], rs: bool = True) -> dict:
    """``n_rows`` Camera_1/2 captures (and RealSense folders), scanner folders ``scan``."""
    oak = {f"{k:03d}": [_ts(k)] for k in range(1, n_rows + 1)}
    return build_tree(tmp_path, desktop, oak1=oak, oak2=dict(oak),
                      scan_steps={n: 9 for n in scan},
                      rs_steps=list(range(1, n_rows + 1)) if rs else [],
                      rs_mtimes={k: _ts(k) for k in range(1, n_rows + 1)} if rs else None)


def _fixes(desktop: int, scan: dict | None = None, oak: dict | None = None) -> dict:
    out: dict = {"scan": {}, "oak": {}, "rs": {}}
    if scan is not None:
        out["scan"][desktop] = {**scan, "reason": "test"}
    if oak is not None:
        out["oak"][desktop] = {**oak, "reason": "test"}
    return out


def _scan_dirs(idx) -> dict[int, str]:
    return {k.step: ff.src_step_dir for k, ff in idx.frames.items() if k.view == "scan"}


# --------------------------------------------------------------------------- #
# the mapping functions
# --------------------------------------------------------------------------- #
def test_collapse_rows_two_dupli():
    step_of_row, extra = collapse_rows(10, [4, 8])
    assert extra == {4, 8}
    assert step_of_row[1:] == [1, 2, 3, 3, 4, 5, 6, 6, 7, 8]


def test_collapse_rows_ignores_row_one_and_rows_past_the_end():
    step_of_row, extra = collapse_rows(5, [1, 9])
    assert extra == set() and step_of_row[1:] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("dups, expected", [
    ([], {1: 1, 5: 5, 49: 49}),                         # skipped every dupli row
    ([42], {41: 41, 42: 41, 43: 42, 49: 48}),           # D18
    ([15, 33], {14: 14, 15: 14, 16: 15, 33: 31, 36: 34}),  # D1: folders named by row
])
def test_scan_step(dups, expected):
    for folder, step in expected.items():
        assert scan_step(folder, dups) == step


# --------------------------------------------------------------------------- #
# OAK / RealSense: rows collapse, dupli captures become extra shots
# --------------------------------------------------------------------------- #
def test_oak_and_rs_rows_collapse_with_extra_shots(tmp_path):
    roots = _tree(tmp_path, 90, 10, scan=list(range(1, 10)))
    idx = scan_desktop(90, roots, _fixes(90, scan={"duplicate_folders": [7]}),
                       dupli_rows=[4, 8], sheet_rows=10)
    assert (idx.n_rows, idx.n_steps, idx.dupli_rows) == (10, 8, [4, 8])
    assert idx.rows == [f"{k:03d}" for k in range(1, 11)]
    for view in ("oak1", "oak2", "rs"):
        dirs = {k.step: ff.src_step_dir for k, ff in idx.frames.items() if k.view == view}
        assert dirs == {1: "001", 2: "002", 3: "003", 4: "005", 5: "006", 6: "007",
                        7: "009", 8: "010"}, view
        shots = idx.frames[FrameKey(90, 3, view)].aux[EXTRA_SHOTS]
        assert [s["src_step_dir"] for s in shots] == ["004"] and shots[0]["row"] == 4
        assert idx.frames[FrameKey(90, 6, view)].aux[EXTRA_SHOTS][0]["src_step_dir"] == "008"
    assert not idx.missing
    assert any("rows: sheet dupli rows [4, 8] collapsed: 10 Camera_1 captures are 8 steps"
               in i for i in idx.issues)


def test_scanner_one_dupli_captured_one_skipped(tmp_path):
    """9 folders for 10 rows: the scanner skipped row 4 and photographed row 8
    (its folder 7 re-shoots folder 6) -- every step complete, nothing missing."""
    roots = _tree(tmp_path, 90, 10, scan=list(range(1, 10)))
    idx = scan_desktop(90, roots, _fixes(90, scan={"duplicate_folders": [7]}),
                       dupli_rows=[4, 8], sheet_rows=10)
    assert _scan_dirs(idx) == {1: "RGB11", 2: "RGB21", 3: "RGB31", 4: "RGB41", 5: "RGB51",
                               6: "RGB61", 7: "RGB81", 8: "RGB91"}
    shots = idx.frames[FrameKey(90, 6, "scan")].aux[EXTRA_SHOTS]
    assert [s["src_step_dir"] for s in shots] == ["RGB71"]
    assert shots[0]["path"].endswith("RGB71/P_0.png")
    assert not [k for k in idx.missing if k.view == "scan"]
    assert "fix:scan[90] duplicate_folders [7]: 9 folders -> 8 of 8 steps + 1 extra shot(s), " \
           "every step photographed" in idx.issues


def test_scanner_rule_mismatch_when_a_folder_is_left_over(tmp_path):
    """Two dupli rows, 9 folders, nothing declared: one folder has no step."""
    roots = _tree(tmp_path, 90, 10, scan=list(range(1, 10)))
    idx = scan_desktop(90, roots, _fixes(90), dupli_rows=[4, 8], sheet_rows=10)
    lines = [i for i in idx.issues if "MISMATCH" in i and i.startswith("scan[90]")]
    assert len(lines) == 1 and "folders ['RGB91'] have no step" in lines[0]
    assert "no scan.90.duplicate_folders declared" in lines[0]


def test_scanner_rule_mismatch_when_a_step_is_left_without_a_frame(tmp_path):
    roots = _tree(tmp_path, 90, 10, scan=list(range(1, 8)))
    idx = scan_desktop(90, roots, _fixes(90, scan={"duplicate_folders": []}),
                       dupli_rows=[4, 8], sheet_rows=10)
    line = next(i for i in idx.issues if i.startswith("fix:scan[90]"))
    assert "MISMATCH" in line and "steps [8] have no scanner frame" in line


def test_scanner_without_dupli_rows_or_rule_is_unchanged(tmp_path):
    """A desktop the sheet gives no dupli row keeps folder n = step n, no line."""
    roots = _tree(tmp_path, 91, 5, scan=[1, 2, 3, 4])
    idx = scan_desktop(91, roots, _fixes(91), dupli_rows=[], sheet_rows=5)
    assert idx.n_steps == 5 and _scan_dirs(idx) == {s: f"RGB{s}1" for s in range(1, 5)}
    assert [k for k in idx.missing if k.view == "scan"] == [FrameKey(91, 5, "scan")]
    assert not [i for i in idx.issues if "scan" in i]


def test_d1_style_gaps_in_collapsed_numbering(tmp_path):
    """Folders named by row with gaps; both dupli rows photographed; a duplicate
    folder whose step has no other photo stands in for it."""
    roots = _tree(tmp_path, 1, 12, scan=[1, 5, 6, 7, 9, 12])   # dupli rows 6 and 9
    fixes = _fixes(1, scan={"duplicate_folders": [6, 9], "missing_steps": [2, 3, 4, 7, 9]})
    idx = scan_desktop(1, roots, fixes, dupli_rows=[6, 9], sheet_rows=12)
    assert idx.n_steps == 10
    # rows 1..12 -> steps 1,2,3,4,5,5,6,7,7,8,9,10
    assert _scan_dirs(idx) == {1: "RGB11", 5: "RGB51", 6: "RGB71", 7: "RGB91", 10: "RGB121"}
    assert [s["src_step_dir"] for s in idx.frames[FrameKey(1, 5, "scan")].aux[EXTRA_SHOTS]] \
        == ["RGB61"]
    assert any("duplicate folder RGB91 stands in for it" in i for i in idx.issues)
    scan_missing = sorted(k.step for k in idx.missing if k.view == "scan")
    assert scan_missing == [2, 3, 4, 8, 9]
    # step 7 was declared missing but RGB91 photographs it: the yaml is stale
    assert any(i.startswith("fix:scan[1] MISMATCH: declared-missing logical steps present "
                            "in index: [7]") for i in idx.issues)
    # ... and step 8 has no frame and is not declared
    assert any("steps [8] have no scanner frame" in i for i in idx.issues)


def test_dupli_replaces_makes_the_retake_the_frame(tmp_path):
    roots = _tree(tmp_path, 90, 6, scan=[1, 2, 3, 4, 5, 6])
    fixes = _fixes(90, scan={"duplicate_folders": [4], "replacing_folders": [4]},
                   oak={"dupli_replaces": [4]})
    idx = scan_desktop(90, roots, fixes, dupli_rows=[4], sheet_rows=6)
    for view in ("oak1", "oak2", "rs"):
        ff = idx.frames[FrameKey(90, 3, view)]
        assert ff.src_step_dir == "004", view
        assert [(s["src_step_dir"], s["row"]) for s in ff.aux[EXTRA_SHOTS]] == [("003", 3)]
    scan3 = idx.frames[FrameKey(90, 3, "scan")]
    assert scan3.src_step_dir == "RGB41"
    assert [s["src_step_dir"] for s in scan3.aux[EXTRA_SHOTS]] == ["RGB31"]
    assert any("row 4 (cam1 004) -> frame of step 3, replacing row 3" in i for i in idx.issues)


def test_dupli_replaces_of_a_row_that_is_not_dupli_is_a_mismatch(tmp_path):
    roots = _tree(tmp_path, 90, 5, scan=[1, 2, 3, 4])
    idx = scan_desktop(90, roots, _fixes(90, oak={"dupli_replaces": [2]}),
                       dupli_rows=[4], sheet_rows=5)
    assert any("dupli_replaces MISMATCH: row 2" in i for i in idx.issues)


def test_rows_before_the_collapse_are_restepped_in_issue_lines(tmp_path):
    """An rs time line written in rows names the collapsed step too."""
    oak = {f"{k:03d}": [_ts(k)] for k in range(1, 6)}
    roots = build_tree(tmp_path, 90, oak1=oak, rs_steps=[1, 2, 3, 4, 5],
                       rs_mtimes={1: _ts(1), 2: _ts(2), 3: _ts(3), 4: _ts(4),
                                  5: "20250101_120000_000"})
    idx = scan_desktop(90, roots, _fixes(90), dupli_rows=[3], sheet_rows=5)
    line = next(i for i in idx.issues if "rs time mismatch" in i)
    assert "logical step 4 (row 5)" in line


def test_sheet_with_more_rows_than_captures_is_reported(tmp_path):
    roots = _tree(tmp_path, 90, 5, scan=[1, 2, 3, 4], rs=False)
    idx = scan_desktop(90, roots, _fixes(90), dupli_rows=[3, 7], sheet_rows=7)
    assert any("the sheet has 7 rows but Camera_1 5 captures" in i for i in idx.issues)
    assert any("dupli row 7 has no Camera_1 capture" in i for i in idx.issues)
    assert idx.n_steps == 4


# --------------------------------------------------------------------------- #
# declared missing steps, the report, JSON
# --------------------------------------------------------------------------- #
def test_verify_declared_missing_counts_collapsed_steps():
    fixes = {"scan": {5: {"missing_steps": [2, 9], "reason": "t"}}}
    missing = [FrameKey(5, 2, "scan")]
    assert verify_declared_missing(5, fixes, 8, missing) == [
        "fix:scan[5] declared missing steps [2] confirmed absent "
        "(1 declared step(s) are beyond the 8 steps: [9])"]
    missing = []
    assert "MISMATCH" in verify_declared_missing(5, fixes, 8, missing)[0]


def test_report_counts_rows_dupli_and_extra_shots(tmp_path):
    roots = _tree(tmp_path, 90, 10, scan=list(range(1, 10)))
    idx = scan_desktop(90, roots, _fixes(90, scan={"duplicate_folders": [7]}),
                       dupli_rows=[4, 8], sheet_rows=10)
    out = tmp_path / "report.md"
    write_report({90: idx}, str(out))
    text = out.read_text(encoding="utf-8")
    assert "- logical steps: 8 (the sheet's dupli rows collapsed: 2)" in text
    # 2 dupli captures x 3 views + 1 scanner duplicate folder
    assert "- extra shots (dupli captures, scanner duplicate folders): 7" in text
    assert "| 90 | 8 | 10 (4, 8) |" in text
    assert "| 90 | 8 | 10 (4, 8) | 32 | 7 | 0 | 0 | 0 | 0 | 0 |" in text


def test_save_load_keeps_rows_and_extra_shots(tmp_path):
    roots = _tree(tmp_path, 90, 10, scan=list(range(1, 10)))
    idx = scan_desktop(90, roots, _fixes(90, scan={"duplicate_folders": [7]}),
                       dupli_rows=[4, 8], sheet_rows=10)
    path = tmp_path / "index.json"
    save_index({90: idx}, str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 3
    assert payload["desktops"]["90"]["dupli_rows"] == [4, 8]
    back = load_index(str(path))[90]
    assert back == idx and back.collapsed


def test_a_version_2_index_is_stale(tmp_path):
    from tda import pipeline as P

    roots = _tree(tmp_path, 90, 4, scan=[1, 2, 3, 4])
    path = tmp_path / "index.json"
    save_index({90: scan_desktop(90, roots, _fixes(90))}, str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_index(str(path))
    assert loaded[90].collapsed is False and P.stale_index(loaded) == [90]


def test_respell_reaches_extra_shot_paths(tmp_path):
    from tda.core.index import respell_paths

    roots = _tree(tmp_path, 90, 6, scan=[1, 2, 3, 4, 5])
    idx = {90: scan_desktop(90, roots, _fixes(90, scan={"duplicate_folders": [4]}),
                            dupli_rows=[4], sheet_rows=6)}
    respell_paths(idx, lambda p: "X:" + p if "/" in p or "\\" in p else p)
    shot = idx[90].frames[FrameKey(90, 3, "oak1")].aux[EXTRA_SHOTS][0]
    assert shot["path"].startswith("X:") and shot["src_step_dir"] == "004"


# --------------------------------------------------------------------------- #
# folder numbers -> steps (Label Studio image names)
# --------------------------------------------------------------------------- #
def test_folder_steps_scan_and_oak(tmp_path):
    roots = _tree(tmp_path, 90, 10, scan=list(range(1, 10)))
    idx = scan_desktop(90, roots, _fixes(90, scan={"duplicate_folders": [7]}),
                       dupli_rows=[4, 8], sheet_rows=10)
    scan = folder_steps(idx, "scan")
    assert scan[6] == 6 and scan[7] == 6 and scan[8] == 7 and scan[9] == 8
    oak = folder_steps(idx, "oak1")
    assert oak[3] == 3 and oak[4] == 3 and oak[5] == 4 and oak[10] == 8
    assert folder_steps(idx, "rs") == oak


# --------------------------------------------------------------------------- #
# the yaml
# --------------------------------------------------------------------------- #
def test_u25_yaml_entries_parse():
    with open(REPO_ROOT / "configs" / "index_fixes.yaml", encoding="utf-8") as fh:
        fixes = yaml.safe_load(fh)
    scan = fixes["scan"]
    assert scan[18]["duplicate_folders"] == [42]
    assert scan[52]["duplicate_folders"] == [6]
    assert scan[31] == {**scan[31], "duplicate_folders": [13], "replacing_folders": [13]}
    for d in (24, 28, 36, 40, 48, 55, 62):
        assert scan[d]["duplicate_folders"] == [], d
    assert fixes["oak"][31]["dupli_replaces"] == [13]
    assert fixes["oak"][62]["dupli_replaces"] == [28]
    # round 2 (the coordinator's rulings on the reviewer's readings)
    assert fixes["oak"][24]["dupli_replaces"] == [37]
    assert fixes["oak"][22]["dupli_replaces"] == [27, 32]
    assert scan[22] == {**scan[22], "duplicate_folders": [27], "replacing_folders": [27]}
    assert scan[60]["duplicate_folders"] == [29]
    assert "cam1_reorder_by_ts" not in fixes["oak"][60]
    assert fixes["oak"][60]["cam1_order"] == "folder"
    assert "dupli_replaces" not in fixes["oak"][60]
    assert fixes["oak"][1]["gallery_captures"] == ["040"]
    assert 38 not in scan[1]["missing_steps"]
    assert scan[41]["skipped_steps"] == [32] and scan[61]["skipped_steps"] == [26]
    assert "UNCERTAIN" in scan[61]["reason"]
    for view in ("oak", "scan", "rs"):
        for desktop, entry in fixes[view].items():
            if isinstance(desktop, int):
                assert entry.get("reason"), (view, desktop)
