"""``scripts/export_components.py``: the components-gallery package (2026-11-02).

The pure helpers: where a gallery picture's raw file sits on the Hugging Face
repository, and how a shot is named. The build itself runs on the real
database only (the scanner package's fixtures carry no gallery).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_components as EC  # noqa: E402


def test_scanner_gallery_maps_below_raw_scanner():
    below = ("UGA DATA/TAMU_B2.3_1-9_RGB/TAMU_B2.0_1-9_RGB_allcompoentsrgb/"
             "TAMU_B2.0_1-9_RGB_allcompoentsrgb/1/allRGB1_color/P_0.png")
    assert EC.hf_candidates(below) == [
        "raw/scanner/TAMU_B2.3_1-9_RGB/TAMU_B2.0_1-9_RGB_allcompoentsrgb/"
        "TAMU_B2.0_1-9_RGB_allcompoentsrgb/1/allRGB1_color/P_0.png"]


def test_oak_gallery_maps_below_raw_oak():
    below = ("OAKD Capture/Desktop_Datacollection/DesktopData/Desktop 1/Components/C1/001/"
             "x_camera_1_rgb_12mp.jpg")
    assert EC.hf_candidates(below) == ["raw/oak/Desktop 1/Components/C1/001/x_camera_1_rgb_12mp.jpg"]


def test_realsense_gallery_tries_both_layouts():
    below = "Realsense Capture/Dataset Information/Exp/Desktop 1/Components/001/original_color.png"
    got = EC.hf_candidates(below)
    assert got[0] == "raw/realsense/Desktop 1/Components/001/original_color.png"
    assert "raw/realsense/Components/Desktop 1/001/original_color.png" in got


def test_unknown_root_has_no_candidate():
    assert EC.hf_candidates("Somewhere/else.png") == []


def test_shot_names():
    assert EC.shot_name(1) == "C01"
    assert EC.shot_name(12) == "C12"
    assert EC.shot_name(1040) == "C1040"
