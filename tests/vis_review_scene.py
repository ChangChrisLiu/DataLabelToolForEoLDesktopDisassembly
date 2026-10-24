"""The scene of the task-U24 tests: confirmed frames with labels to audit.

The D13 sheet's first 14 steps (:mod:`session_scene`) on a 256x256 canvas, every
part of the scene drawn in a cell of its own (30 px squares: plainly
``visible``), then:

* ``ram_latch.03`` wholly under ``screw.motherboard.01`` (whose shape is
  re-drawn over both cells) -- ``occluded_full``, computed;
* ``ram_module.01`` half under ``screw.motherboard.02`` -- ``occluded_partial``;
* ``connector.03`` drawn 4 px wide -- ``too_small``;
* ``cpu.01`` 被线挡住（完全） on frame 13 -- a hand label with a reason;
* ``psu.01`` set ``out_of_view`` by hand on frame 12;

and frames 14 down to 10 confirmed. Steps 1-9 are not confirmed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from session_scene import DESKTOP, VIEW, cell, make_session, rect, seed_shapes
from tda.core import masks
from tda.core.model import FrameKey, ShapeKeyframe, ShapePart
from tda.core.occluder import FULL

HW = (256, 256)
CONFIRMED = (14, 13, 12, 11, 10)
HIDDEN, COVER = "ram_latch.03", "screw.motherboard.01"
HALF, HALF_COVER = "ram_module.01", "screw.motherboard.02"
TINY = "connector.03"
CABLED, CABLE_STEP = "cpu.01", 13
OUT, OUT_STEP = "psu.01", 12


def keyframe(session, instance: str, mask: np.ndarray, anchor: int = 14) -> None:
    session.db.add_keyframe(ShapeKeyframe(
        id=None, instance=instance, desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=anchor, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(mask))]))


def build(tmp_path: Path, confirm=CONFIRMED):
    """The session of the scene, the sweeper off, standing on the last confirmed frame."""
    session = make_session(tmp_path, hw=HW)
    session.sweeper_enabled = False
    keys = seed_shapes(session, 1, grid=8, refresh=False)
    index = {key: i for i, key in enumerate(keys)}
    keyframe(session, COVER, cell(index[COVER], HW) | cell(index[HIDDEN], HW))
    half = cell(index[HALF], HW)
    ys, xs = np.nonzero(half)
    left = np.zeros(HW, dtype=bool)
    left[ys.min():ys.max() + 1, xs.min():xs.min() + 15] = True
    keyframe(session, HALF_COVER, cell(index[HALF_COVER], HW) | left)
    ys, xs = np.nonzero(cell(index[TINY], HW))
    keyframe(session, TINY, rect(int(xs.min()), int(ys.min()), int(xs.min()) + 4,
                                 int(ys.min()) + 4, HW))
    session.refresh_all()
    session.goto(CABLE_STEP, force=True)
    session.set_cable(CABLED, FULL)
    session.goto(OUT_STEP, force=True)
    session.set_visibility(OUT, "out_of_view")
    for step in confirm:
        session.goto(step, force=True)
        assert session.confirm_frame(), step
    return session


def key(step: int) -> FrameKey:
    return FrameKey(DESKTOP, int(step), VIEW)
