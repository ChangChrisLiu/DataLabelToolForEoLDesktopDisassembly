"""Layout of the four raw capture trees on the read-only F: drive.

This module knows *where the files are* and nothing about logical steps: it is
the only place that encodes folder and filename conventions.  Everything is
enumerated with :func:`os.scandir` at the known depths - never a recursive walk,
because the source drive is a slow external disk - and nothing here writes.

Layout (spec 2.1)::

    oak_root/Desktop <n>/<Disassemble>/Camera_<1|2>/<NNN>/<ts>_camera_<c>_<role>
    scanner_root/TAMU_B2.3_<a>-<b>_RGB/<subdir>/<n>/RGB<step>1/P_<k>.png
    rs_root/Desktop <n>/<Disassemble|Disassembly>/<NNN>/original_color.png

and the components gallery (task U10)::

    oak_root/Desktop <n>/Components/<Camera_1|C1|Camera_2|C2>/<NNN>/<ts>_camera_<c>_<role>
    rs_root/Desktop <n>/Components/<NNN>/original_color.png
    scanner_root/TAMU_B2.3_<a>-<b>_RGB/TAMU_B2.0_<c>-<d>_RGB_allcompoentsrgb/<payload>/<n>/
        <all...rgb<k>1[_color]>/P_<k>.png                                  (task U42)
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Iterable, Optional

OAK_FILE_RE = re.compile(r"^(\d{8}_\d{6}_\d{3})_camera_(\d)_(.+)$")
STEP_DIR_RE = re.compile(r"^\d{3}$")
SCAN_GROUP_RE = re.compile(r"^TAMU_B2\.3_(\d+)-(\d+)_RGB$")
BURST_RE = re.compile(r"^P_(\d+)\.png$", re.IGNORECASE)
#: The scanner's all-components folder inside a group (task U42), e.g.
#: ``TAMU_B2.0_1-9_RGB_allcompoentsrgb`` (sic): matched on ``allcompo`` without
#: case, its ``<a>-<b>`` the desktops it was meant to hold.
SCAN_COMPONENTS_RE = re.compile(r"allcompo", re.IGNORECASE)
SCAN_RANGE_RE = re.compile(r"(\d+)-(\d+)")
#: A shot folder of the scanner's all-components scans, lower-cased (U42):
#: ``allrgb1_color``, ``allcomponentsrgb11``, ``allcompoentsrgb21_color``,
#: ``allcomponenetsrgb31`` ... The digits are the step folders' ``<k>1``
#: (``11`` -> 1, ``21`` -> 2); a bare ``1`` is 1.
SCAN_SHOT_RE = re.compile(r"^all.*rgb(\d*)(?:_color)?$")

#: OAK filename suffix -> ``FrameFile.aux`` key ("main" goes to ``FrameFile.path``).
OAK_ROLES = {
    "rgb_12mp.jpg": "main",
    "rgb_aligned.png": "aligned",
    "depth_raw.npy": "depth_npy",
    "depth_raw.png": "depth_png",
    "pointcloud.ply": "pointcloud",
}
#: The disassembly section folder is spelled both ways across the dataset.
DEFAULT_FOLDER_ALIASES = ("Disassemble", "Disassembly")

#: The components gallery (task U10): after the teardown every removed part was
#: laid out and photographed, into a ``Components`` folder next to the
#: disassembly one -- ``Desktop <n>/Components/<camera>/<NNN>/...`` for the two
#: OAK cameras, ``Desktop <n>/Components/<NNN>/original_color.png`` for the
#: RealSense.
COMPONENTS_DIR = "Components"
#: The OAK camera folders inside ``Components``, spelled two ways: ``Camera_1``
#: from desktop 25 on, ``C1`` before it (and on desktops 1-24 only).
COMPONENT_CAMERA_DIRS: dict[str, tuple[str, ...]] = {
    "cmp_oak1": ("Camera_1", "C1"),
    "cmp_oak2": ("Camera_2", "C2"),
}


def iso_from_oak_token(token: str) -> str:
    """``20250531_104736_629`` -> ``2025-05-31T10:47:36.629000``."""
    base = datetime.strptime(token[:15], "%Y%m%d_%H%M%S")
    return base.replace(microsecond=int(token[16:19]) * 1000).isoformat()


def iso_from_mtime(mtime: float) -> str:
    """File modification time as an ISO string (used where no capture time exists)."""
    return datetime.fromtimestamp(mtime).isoformat(timespec="milliseconds")


def norm(path: str) -> str:
    """Store every path with forward slashes so the JSON index is platform-neutral."""
    return path.replace("\\", "/")


def _scan(path: str, keep, errors: Optional[list[str]]) -> list[os.DirEntry]:
    """``os.scandir`` one level, keeping the entries ``keep(entry)`` accepts.

    A failed listing must never look like an empty folder: F: is a slow external
    drive, so a transient read error would otherwise turn into a silent "missing
    frame".  The exception text is appended to ``errors`` (when a list is given)
    so the caller can surface it in ``DesktopIndex.issues``.
    """
    try:
        return [e for e in os.scandir(path) if keep(e)]
    except OSError as exc:
        if errors is not None:
            errors.append(f"scandir failed: {norm(path)}: {type(exc).__name__}: {exc}")
        return []


def dirs(path: str, errors: Optional[list[str]] = None) -> list[os.DirEntry]:
    """``os.scandir`` one level, directories only; see :func:`_scan` for errors."""
    return _scan(path, os.DirEntry.is_dir, errors)


def files(path: str, errors: Optional[list[str]] = None) -> list[os.DirEntry]:
    """``os.scandir`` one level, files only; see :func:`_scan` for errors."""
    return _scan(path, os.DirEntry.is_file, errors)


def first_existing(base: str, names: Iterable[str]) -> Optional[str]:
    """First of ``names`` that exists as a sub-directory of ``base``."""
    for name in names:
        candidate = os.path.join(base, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def scan_oak_step_dir(
    path: str, errors: Optional[list[str]] = None
) -> dict[str, dict[str, str]]:
    """Group one OAK ``NNN`` folder's files by capture timestamp token.

    Returns ``{ts_token: {role: abs_path}}``; a folder holding two captures
    (D10 step 1, D60 cam2 folder 005) yields two entries.
    """
    out: dict[str, dict[str, str]] = {}
    for entry in files(path, errors):
        m = OAK_FILE_RE.match(entry.name)
        if not m:
            continue
        role = OAK_ROLES.get(m.group(3))
        if role is None:
            continue
        out.setdefault(m.group(1), {})[role] = norm(entry.path)
    return out


def scan_oak_camera(
    cam_dir: str, errors: Optional[list[str]] = None
) -> dict[str, dict[str, dict[str, str]]]:
    """``{step_dir: {ts_token: {role: path}}}`` for one ``Camera_<n>`` folder."""
    return {
        e.name: scan_oak_step_dir(e.path, errors)
        for e in dirs(cam_dir, errors)
        if STEP_DIR_RE.match(e.name)
    }


def scanner_desktop_dir(
    scanner_root: str, desktop: int, errors: Optional[list[str]] = None
) -> Optional[str]:
    """Locate ``<group>/<subdir>/<desktop>`` for one desktop.

    The group folder is ``TAMU_B2.3_<a>-<b>_RGB`` whose range contains the
    desktop.  Inside it the payload sub-folder usually repeats the group name,
    but for 50-66 it is ``tamu_color_50-66_Bright2.3``; ``new``/``New`` staging
    folders and ``ext.py`` are ignored.
    """
    groups = []
    for entry in dirs(scanner_root, errors):
        m = SCAN_GROUP_RE.match(entry.name)
        if m and int(m.group(1)) <= desktop <= int(m.group(2)):
            groups.append(entry)
    for group in groups:
        candidates = [e for e in dirs(group.path, errors) if e.name.lower() != "new"]
        same_name = [e for e in candidates if e.name == group.name]
        for pool in (same_name, candidates):
            for cand in pool:
                target = os.path.join(cand.path, str(desktop))
                if os.path.isdir(target):
                    return target
    return None


def scanner_components_dir(
    scanner_root: str, desktop: int, errors: Optional[list[str]] = None
) -> Optional[str]:
    """Locate the scanner's all-components scans of one desktop (task U42).

    They sit beside the step scans' payload in the same group folder:
    ``<group>/TAMU_B2.0_<a>-<b>_RGB_allcompoentsrgb/<payload>/<desktop>``, the
    payload being the one sub-folder inside (spelled as its parent, or
    ``TAMU_color_<a>-<b>_B2.0``). The group ``TAMU_B2.3_50-66_RGB`` holds two
    of them (50-59, 60-66), so the folder's own range picks the one -- but
    desktop 20 lies in the 10-19 batch (``10..20``): when no folder whose range
    holds the desktop has it, every other all-components folder is looked in.
    """
    groups = [e for e in sorted(dirs(scanner_root, errors), key=lambda e: e.name)
              if SCAN_GROUP_RE.match(e.name)]
    ranged: list[os.DirEntry] = []
    others: list[os.DirEntry] = []
    for group in groups:
        for entry in sorted(dirs(group.path, errors), key=lambda e: e.name):
            if not SCAN_COMPONENTS_RE.search(entry.name):
                continue
            m = SCAN_RANGE_RE.search(entry.name)
            if m and int(m.group(1)) <= desktop <= int(m.group(2)):
                ranged.append(entry)
            else:
                others.append(entry)
    for folder in ranged + others:
        for payload in sorted(dirs(folder.path, errors), key=lambda e: e.name):
            target = os.path.join(payload.path, str(desktop))
            if os.path.isdir(target):
                return target
    return None
