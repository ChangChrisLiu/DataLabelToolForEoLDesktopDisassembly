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
