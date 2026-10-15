"""The components gallery's test scene (task U10, not a test module itself).

The real D13 step sheet (``tests/fixtures/logs/desktop_13.csv``) imported in full
-- 42 steps, so the board and everything on it are out by the end -- with a
synthetic 64x64 image per scanner step and per gallery shot in a temporary
cache directory. :func:`seed_gallery` adds ``shots`` frames to each of the
three gallery views and seeds their pose segments the way ``load-index`` does
(:func:`tda.pipeline.settle_gallery`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from tda import pipeline as P
from tda.core.cache import VIEW_EXT, cache_path
from tda.core.db import Db
from tda.core.model import COMPONENT_VIEWS, FrameKey
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.ui.session import AnnotationSession

import session_scene as S

DESKTOP = S.DESKTOP
HW = (64, 64)
#: The D13 sheet's last step: the board goes, and with it everything on it.
LAST_STEP = 42
SHOTS = 3
VIEW = "cmp_oak1"


def write_shot(db: Db, cache_dir: Path, view: str, shot: int,
               hw: tuple[int, int] = HW) -> str:
    """One synthetic gallery image plus its ``frame`` row; returns the path."""
    key = FrameKey(DESKTOP, shot, view)
    path = cache_path(str(cache_dir), key, VIEW_EXT[view])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = np.full((*hw, 3), 30 + 10 * shot, dtype=np.uint8)
    img[4 + shot:20 + shot, 6:30] = 210        # a "part" that moves between shots
    cv2.imwrite(path, img)
    db.upsert_frame(key, path, {"cache_path": path, "hw": [hw[0], hw[1]]}, None)
    return path


def seed_gallery(db: Db, cache_dir: Path, shots: int = SHOTS,
                 views=COMPONENT_VIEWS) -> dict:
    """``shots`` frames in each gallery view, then their segments and ROIs."""
    for view in views:
        for shot in range(1, shots + 1):
            write_shot(db, cache_dir, view, shot)
    return P.settle_gallery(db, DESKTOP)


def make_db(tmp_path: Path, shots: int = SHOTS, last_step: int = LAST_STEP,
            db_path: Optional[str] = None, cache_dir: Optional[Path] = None):
    """``(db, tax, cache_dir)`` of the D13 sheet with a seeded gallery."""
    tax = load_taxonomy()
    db = Db(str(db_path or tmp_path / "tda.sqlite"))
    cache_dir = Path(cache_dir or tmp_path / "cache")
    S.seed_db(db, tax, cache_dir, last_step=last_step)
    seed_gallery(db, cache_dir, shots=shots)
    return db, tax, cache_dir


def make_env(tmp_path: Path, shots: int = SHOTS) -> dict:
    """A ``paths.yaml`` workspace (:mod:`app_scene`) holding the gallery scene."""
    from app_scene import make_paths, write_paths_yaml

    cfg = make_paths(tmp_path)
    Path(cfg["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    db, _tax, _cache = make_db(tmp_path, shots=shots, db_path=cfg["db_path"],
                               cache_dir=Path(cfg["cache_dir"]))
    db.close()
    return {"paths": write_paths_yaml(tmp_path), "cfg": cfg, "tmp": tmp_path}


def make_session(tmp_path: Path, view: str = VIEW, shots: int = SHOTS) -> AnnotationSession:
    """A session opened on one gallery view of the scene."""
    db, tax, cache_dir = make_db(tmp_path, shots=shots)
    session = AnnotationSession(db, tax, TruthService(db, tax), str(cache_dir), "tester")
    session.sweeper_enabled = False
    session.open(DESKTOP, view)
    return session


def rect(x0: int, y0: int, x1: int, y1: int, hw: tuple[int, int] = HW) -> np.ndarray:
    """A filled ``[x0, x1) x [y0, y1)`` rectangle as a bool mask."""
    mask = np.zeros(hw, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask
