"""The scene of the task-U36 tests: the U24 scene seen from two cameras.

The scanner view is :mod:`vis_review_scene`'s (frames 14-10 confirmed: a part
wholly under a screw, one half under another, a 4 px part, a cable label on
13, ``out_of_view`` by hand on 12) plus two more lines:

* ``connector.05`` (:data:`ANGLE`) set 完全遮挡 by hand on 10-14, no reason;
* ``connector.04`` (:data:`MARKED`) 被线挡住（完全） on 10-14 (``Ctrl+```).

``oak1`` (:data:`OTHER`) has its own frames on the same steps, every part in
its cell, frames 14-10 confirmed, and in it:

* ``ram_latch.03`` wholly under its screw as on the scanner (a part over it);
* ``connector.05`` plainly seen -- the scanner's 完全 is the camera angle;
* ``connector.04`` 被线挡住（完全） on 10-13, answered 角度看不到 on 13 in
  oak1's own review, and seen on 14;
* ``ram_module.01`` seen whole (nothing over it in this view).

``rs`` (:data:`DRAFT`) has a shape of one part and no confirmed frame (a
draft); ``oak2`` has nothing (「—」).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import vis_review_scene as V
from session_scene import DESKTOP, VIEW, cell, chassis_instances, make_session, rect, seed_shapes
from tda.core import masks
from tda.core.cache import VIEW_EXT, cache_path
from tda.core.model import FrameKey, OccluderMask, ShapeKeyframe, ShapePart, ZOrderRec
from tda.core.occluder import FULL
from tda.core.vis_review import VERDICT_VIEWPOINT

HW = V.HW
STEPS = [10, 11, 12, 13, 14]
OTHER = "oak1"
DRAFT = "rs"
ANGLE = "connector.05"
MARKED = "connector.04"


def write_frames(db, cache_dir: Path, view: str, steps, hw=HW) -> None:
    """One synthetic image per step of ``view``, and its ``frame`` row."""
    for step in steps:
        key = FrameKey(DESKTOP, int(step), view)
        path = cache_path(str(cache_dir), key, VIEW_EXT[view])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img = np.full((*hw, 3), 40 + step, dtype=np.uint8)
        img[8:24, 8:24] = 220
        cv2.imwrite(path, img)
        db.upsert_frame(key, path, {"cache_path": path, "hw": [hw[0], hw[1]]}, None, None)


def keyframe(db, view: str, instance: str, mask: np.ndarray, anchor: int = 14) -> None:
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=instance, desktop=DESKTOP, view=view, pose_segment=1,
        anchor_step=anchor, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(mask))]))


def seed_view(session, view: str, keys: list[str], cover_hidden: bool = True) -> None:
    """Every chassis part of ``view`` in its own cell; the screw over ram_latch.03."""
    index = {key: i for i, key in enumerate(keys)}
    for key, i in index.items():
        mask = cell(i, HW)
        if cover_hidden and key == V.COVER:
            mask = mask | cell(index[V.HIDDEN], HW)
        keyframe(session.db, view, key, mask)
    session.db.set_zorder(ZOrderRec(DESKTOP, view, 1, [(k, "main") for k in keys]))


def build(tmp_path: Path, hand_over: Optional[str] = None):
    """The two-camera session, the sweeper off, standing on the scanner's frame 10.

    ``hand_over`` (a part of the scene; round 2, B2): set 完全遮挡 by hand on the
    scanner's 10-14, and in oak1 covered wholly by a hand occluder layer on
    10-14 -- a computed 完全 there with no part over it, only the layer.
    """
    session = make_session(tmp_path, hw=HW)
    session.sweeper_enabled = False
    keys = seed_shapes(session, 1, grid=8, refresh=False)
    index = {key: i for i, key in enumerate(keys)}
    # the U24 scene's covers (vis_review_scene.build)
    V.keyframe(session, V.COVER, cell(index[V.COVER], HW) | cell(index[V.HIDDEN], HW))
    half = cell(index[V.HALF], HW)
    ys, xs = np.nonzero(half)
    left = np.zeros(HW, dtype=bool)
    left[ys.min():ys.max() + 1, xs.min():xs.min() + 15] = True
    V.keyframe(session, V.HALF_COVER, cell(index[V.HALF_COVER], HW) | left)
    ys, xs = np.nonzero(cell(index[V.TINY], HW))
    V.keyframe(session, V.TINY, rect(int(xs.min()), int(ys.min()), int(xs.min()) + 4,
                                     int(ys.min()) + 4, HW))
    session.refresh_all()
    session.goto(V.CABLE_STEP, force=True)
    session.set_cable(V.CABLED, FULL)
    session.goto(V.OUT_STEP, force=True)
    session.set_visibility(V.OUT, "out_of_view")
    for step in STEPS:
        session.goto(step, force=True)
        session.set_visibility(ANGLE, "occluded_full")
        if hand_over:
            session.set_visibility(hand_over, "occluded_full")
    session.goto(14, force=True)
    session.set_cable_range(MARKED, FULL, 10)
    for step in reversed(STEPS):
        session.goto(step, force=True)
        assert session.confirm_frame(), step

    # the second camera, on the same steps
    cache_dir = Path(session.cache_dir)
    write_frames(session.db, cache_dir, OTHER, range(1, 15))
    others = chassis_instances(session, 1)
    seed_view(session, OTHER, others)
    if hand_over:
        hand = masks.encode_rle(cell(others.index(hand_over), HW))
        for step in STEPS:
            session.db.set_occluder(OccluderMask(key(step, OTHER), "hand", hand))
    session.open(DESKTOP, OTHER, force=True)
    session.refresh_all()
    session.goto(13, force=True)
    session.set_cable_range(MARKED, FULL, 10)
    for step in reversed(STEPS):
        session.goto(step, force=True)
        assert session.confirm_frame(), (OTHER, step)
    session.goto(13, force=True)
    session.review_visibility(MARKED, VERDICT_VIEWPOINT, FULL, steps=[13])
    # a draft on the third: shapes, nothing confirmed
    keyframe(session.db, DRAFT, ANGLE, cell(0, HW))

    session.open(DESKTOP, VIEW, force=True)
    session.goto(10, force=True)
    session.undo_stack.clear()
    return session


def key(step: int, view: str = VIEW) -> FrameKey:
    return FrameKey(DESKTOP, int(step), view)
