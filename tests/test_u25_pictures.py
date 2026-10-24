"""Task U25 audit: the per-step scanner picture check (scripts/step_changes.py and
the contradiction rules in scripts/data_audit.py), on synthetic frames."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import data_audit as DA  # noqa: E402
import step_changes as SC  # noqa: E402


def _board(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((256, 256), 140, np.uint8)
    img[40:220, 30:230] = 70                                  # the board
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def test_the_same_scene_twice_does_not_change_even_brighter():
    a = _board(1)
    b = np.clip(_board(2).astype(np.int16) + 30, 0, 255).astype(np.uint8)   # exposure only
    m = SC.measure(a, b, None, (1600, 1600))
    assert m["frac"] < 0.002 and m["blob_frac"] < 0.002


def test_a_removed_part_is_a_blob_in_full_resolution_pixels():
    a = _board(1)
    b = a.copy()
    b[100:150, 60:120] = 200                                  # a drive leaves: bright floor
    m = SC.measure(a, b, [100, 200, 1380, 1480], (1600, 1600))   # a 1280-px ROI -> 5 px/px
    assert 0.03 < m["frac"] < 0.07
    x0, y0, x1, y1 = m["blob"]
    assert abs(x0 - (100 + 60 * 5)) <= 15 and abs(y1 - (200 + 150 * 5)) <= 15


def _li(steps: dict[int, list[tuple[str, str]]]):
    actions, instances = [], {}
    for s, acts in steps.items():
        for k, (verb, cls) in enumerate(acts):
            key = f"{cls}.{s}{k}"
            instances[key] = SimpleNamespace(cls=cls)
            actions.append(SimpleNamespace(step=s, verb=verb, target=key))
    return SimpleNamespace(actions=actions, instances=instances,
                           steps=[SimpleNamespace(step=s, raw_name=f"row {s}") for s in steps])


def test_the_three_contradictions():
    changes = {"steps": {
        "2": {"frac": 0.001, "blob": None, "blob_frac": 0.0},      # PSU out, nothing changed
        "3": {"frac": 0.30, "blob": [1, 2, 3, 4], "blob_frac": 0.2},  # two screws, 30 % changed
        "4": {"frac": 0.30, "blob": None, "blob_frac": 0.2},       # a cover opened: fine
        "5": {"frac": 0.0, "blob": None, "blob_frac": 0.0},        # RAM out: not held to (a)
        "6": {"frac": 0.20, "blob": None, "blob_frac": 0.1},       # disconnect under its limit
    }, "first_last": {"frac": 0.05, "blob": None, "blob_frac": 0.0}, "last": 6}
    li = _li({2: [("remove", "psu")], 3: [("unscrew", "screw"), ("unscrew", "screw")],
              4: [("open", "cover")], 5: [("remove", "ram_module")],
              6: [("disconnect", "connector")]})
    flags = DA.picture_flags({"d": 90}, changes, li)
    assert [(f["kind"], f["step"]) for f in flags] == [("a", 2), ("b", 3), ("c", 6)]
    assert "psu" in flags[0]["why"] and "limit 15%" in flags[1]["why"]


def test_the_shot_the_cache_shows_stands_in_for_an_outlier_p0(tmp_path):
    import json

    import cv2

    burst = tmp_path / "RGB131"
    burst.mkdir()
    p0, p2 = (burst / "P_0.png").as_posix(), (burst / "P_2.png").as_posix()
    cv2.imwrite(p0, np.full((64, 64), 250, np.uint8))         # a flash of glare
    cv2.imwrite(p2, _board(3)[:64, :64])
    manifest = tmp_path / "cache" / "scan" / "D90" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"13": {"metrics": [
        {"path": p0, "p_index": 0, "mean": 212.0, "sat_frac": 0.40, "lap_var": 600.0,
         "dist_to_median": 44.8},
        {"path": p2, "p_index": 2, "mean": 160.0, "sat_frac": 0.18, "lap_var": 700.0,
         "dist_to_median": 3.0}]}}), encoding="utf-8")
    choices = SC.cache_choices(str(tmp_path / "cache"), 90)
    assert choices[p0] == (p2, "p0_outlier")
    ff = SimpleNamespace(path=p0, aux={"burst": [p0, p2]})
    small = SC.A.load_frames([p0], "scan", None)[0]
    frame, why = SC._exposed(small, ff, None, choices)
    assert why == "RGB131/P_2.png (P_0 outlier)" and abs(float(frame.mean()) - 140) < 60
    assert SC._exposed(small, ff, None, {})[1] is None         # no manifest: P_0 is exposed


# --------------------------------------------------------------------------- #
# U25 round 2 (N13, N15)
# --------------------------------------------------------------------------- #
def test_a_dark_p0_without_a_manifest_falls_back_to_the_first_exposed_shot(tmp_path):
    import cv2

    burst = tmp_path / "RGB71"
    burst.mkdir()
    p0, p1, p2 = ((burst / f"P_{k}.png").as_posix() for k in (0, 1, 2))
    cv2.imwrite(p0, np.full((64, 64), 10, np.uint8))           # the lamp did not fire
    cv2.imwrite(p1, np.full((64, 64), 20, np.uint8))           # nor here
    cv2.imwrite(p2, _board(4)[:64, :64])
    ff = SimpleNamespace(path=p0, aux={"burst": [p0, p1, p2]})
    small = SC.A.load_frames([p0], "scan", None)[0]
    frame, why = SC._exposed(small, ff, None, None)
    assert why == "RGB71/P_2.png (P_0 dark)" and float(frame.mean()) > SC.DARK_MEAN


def test_a_contrast_change_alone_is_aligned_away():
    a = _board(1)
    b = np.clip((_board(2).astype(np.float32) - 105) * 1.5 + 105, 0, 255).astype(np.uint8)
    m = SC.measure(a, b, None, (1600, 1600))
    assert m["frac"] < 0.002


def test_a_step_that_hits_a_is_not_also_reported_as_b():
    changes = {"steps": {"2": {"frac": 0.0, "blob": None, "blob_frac": 0.0},
                         "3": {"frac": 0.5, "blob": None, "blob_frac": 0.4}},
               "first_last": {"frac": 0.5, "blob": None, "blob_frac": 0.0}, "last": 3}
    li = _li({2: [("remove", "psu"), ("unscrew", "screw")],
              3: [("remove", "psu"), ("unscrew", "screw")]})
    flags = DA.picture_flags({"d": 90}, changes, li)
    # step 2: (a) only; step 3: a large change with a removal is no contradiction
    assert [(f["kind"], f["step"]) for f in flags] == [("a", 2)]


def _sheet(seqs):
    rows = [{"Sequence Number": str(q)} for q in seqs]
    return rows, SimpleNamespace(sheet_rows=len(rows), dupli_rows=[])


def test_rows_are_aligned_to_captures_by_sequence_number():
    di = SimpleNamespace(n_rows=5, n_steps=5)
    rows, li = _sheet([1, 2, 3, 4, 5])
    assert DA.capture_alignment(di, li, rows) == (None, "")          # rows = captures
    # D56: number 3 skipped -> capture 3 has no row; 4 rows for 5 captures
    rows, li = _sheet([1, 2, 4, 5])
    align, note = DA.capture_alignment(di, li, rows)
    assert align == {1: [1], 2: [2], 4: [3], 5: [4]} and "captures [3] have no row" in note
    # D63: numbers 2-3 repeated -> their rows checked together; 7 rows for 5 captures
    rows, li = _sheet([1, 2, 3, 2, 3, 4, 5])
    align, note = DA.capture_alignment(di, li, rows)
    assert align[2] == [2, 4] and align[5] == [7] and "[2, 3] used twice" in note
    # D43: 4 rows numbered 1-4 for 5 captures -- which one has no row?
    rows, li = _sheet([1, 2, 3, 4])
    align, note = DA.capture_alignment(di, li, rows)
    assert align == {} and "not checked" in note


def test_an_aligned_capture_takes_the_actions_of_its_rows():
    """D63: capture 4 is the row numbered 4 (sheet step 6), a reorient -- not the
    unscrew of sheet step 4."""
    changes = {"steps": {"4": {"frac": 0.7, "blob": None, "blob_frac": 0.6}},
               "first_last": {"frac": 0.5, "blob": None, "blob_frac": 0.0}, "last": 5}
    li = _li({4: [("unscrew", "screw")], 6: [("reorient", "chassis")]})
    assert [f["step"] for f in DA.picture_flags({"d": 63}, changes, li)] == [4]
    assert DA.picture_flags({"d": 63}, changes, li, {4: [6]}) == []
