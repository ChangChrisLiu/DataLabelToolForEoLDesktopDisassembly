"""The scene behind the task U21 tests (not a test module itself).

D2's drive bay in miniature: desktop 1, eight steps, 64x64 frames in ``scan``
and ``oak1``. The importer made the SSD cage a latch: ``drive_latch.02``
("remove SSD locker shield", taken out at step 5) is a metal cage, drawn and
confirmed; ``drive_latch.01`` ("SSD Locker", step 3) is the blue locker, which
an annotator said with ``Shift+T`` was taken out (``locker_verb="remove"``) or
which is only opened (``"open"``, after the reverse ``Shift+T``). The locker's
``of`` names the shield, the drive is mounted on it, a manual ``locked_by``
edge ties the two, and an annotator pressed ``T`` on the shield at step 3
(closed -> open: a state a cage does not have). Every frame of ``scan`` and
frames 4-8 of ``oak1`` are confirmed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tda.core import masks
from tda.core.db import Db
from tda.core.extra import recompile_events, restamp_graph
from tda.core.model import (
    ActionRec,
    FrameKey,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    StateEvent,
    StepRec,
    ZOrderRec,
)
from tda.core.taxonomy import Taxonomy, load_taxonomy
from tda.core.truth import TruthService

DESKTOP = 1
VIEWS = ("scan", "oak1")
HW = (64, 64)
STEPS = tuple(range(1, 9))
LOCKER = "drive_latch.01"
SHIELD = "drive_latch.02"
DRIVE = "storage_drive.ssd.01"
CAGE = "drive_cage.01"          # what the shield becomes
IMPLIED = "drive_latch.03"      # the cage's own latch, when the locker is not it
SHIELD_A = (4, 4, 40, 40)
SHIELD_B = (5, 5, 41, 41)
LOCKER_RECT = (44, 4, 52, 12)
DRIVE_RECT = (10, 44, 30, 60)
VERIFIED = {"scan": STEPS, "oak1": (4, 5, 6, 7, 8)}


def rect(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros(HW, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def mask_kf(instance: str, box, anchor: int, view: str) -> ShapeKeyframe:
    return ShapeKeyframe(id=None, instance=instance, desktop=DESKTOP, view=view,
                         pose_segment=1, anchor_step=anchor, placement="in_chassis",
                         geom_type="mask",
                         parts=[ShapePart("main", masks.encode_rle(rect(*box)))])


@dataclass
class Scene:
    db: Db
    tax: Taxonomy
    svc: TruthService

    def key(self, step: int, view: str = "scan") -> FrameKey:
        return FrameKey(DESKTOP, step, view)

    def refresh_all(self) -> dict:
        totals: dict = {}
        for view in VIEWS:
            out = self.svc.refresh_range(DESKTOP, view, list(STEPS))
            for name in ("updated", "conflicts", "standing"):
                totals[name] = totals.get(name, 0) + int(out.get(name) or 0)
        return totals

    def status(self, step: int, view: str = "scan") -> str:
        return self.db.get_frame(self.key(step, view))["review_status"]

    def statuses(self, view: str = "scan") -> dict[int, str]:
        return {step: self.status(step, view) for step in STEPS}


def instances(locker_of: str = SHIELD) -> list[InstanceRec]:
    return [
        InstanceRec(LOCKER, DESKTOP, "drive_latch", attrs={"of": locker_of},
                    raw_names=["SSD Locker"]),
        InstanceRec(SHIELD, DESKTOP, "drive_latch", attrs={"of": "ssd", "qualifier": "x"},
                    raw_names=["remove SSD locker shield"]),
        InstanceRec(DRIVE, DESKTOP, "storage_drive", attrs={"kind": "ssd"},
                    mounted_on=SHIELD, raw_names=["take SSD out"]),
    ]


def build(db: Db, *, locker_verb: str = "remove", verify: bool = True,
          locker_of: str = SHIELD) -> Scene:
    tax = load_taxonomy()
    db.upsert_desktop(DESKTOP, {"brand": "Dell"})
    for rec in instances(locker_of):
        db.upsert_instance(rec)
    db.replace_steps(DESKTOP, [StepRec(DESKTOP, s, "normal", f"step {s}") for s in STEPS],
                     [ActionRec(DESKTOP, 3, 0, LOCKER, locker_verb),
                      ActionRec(DESKTOP, 4, 0, DRIVE, "remove"),
                      ActionRec(DESKTOP, 5, 0, SHIELD, "remove")])
    db.add_events([StateEvent(DESKTOP, 3, SHIELD, "state", "closed", "open", "scan",
                              auto=False)])
    recompile_events(db, DESKTOP, tax)
    for view in VIEWS:
        db.set_pose_segment(DESKTOP, view, 1, 1, 8, 8, None, None)
        for step in STEPS:
            db.upsert_frame(FrameKey(DESKTOP, step, view), f"{view}_{step:03d}.jpg",
                            {"hw": list(HW)}, None)
        # a shape holds on its frame and the ones before it (annotated backwards)
        db.add_keyframe(mask_kf(SHIELD, SHIELD_A, 2, view))
        db.add_keyframe(mask_kf(SHIELD, SHIELD_B, 4, view))
        db.add_keyframe(mask_kf(LOCKER, LOCKER_RECT, 8, view))
        db.add_keyframe(mask_kf(DRIVE, DRIVE_RECT, 3, view))
        db.set_zorder(ZOrderRec(DESKTOP, view, 1, [(SHIELD, "main"), (DRIVE, "main"),
                                                   (LOCKER, "main")]))
    db.add_relation(DESKTOP, "locked_by", SHIELD, LOCKER, source="manual", reason="seen")
    scene = Scene(db=db, tax=tax, svc=TruthService(db, tax, cache_dir=None))
    scene.refresh_all()
    if verify:
        for view, steps in VERIFIED.items():
            for step in steps:
                scene.svc.verify_frame(scene.key(step, view), "tester")
    restamp_graph(db, DESKTOP, tax)
    return scene


__all__ = ["CAGE", "DESKTOP", "DRIVE", "HW", "IMPLIED", "LOCKER", "SHIELD", "STEPS", "Scene",
           "VERIFIED", "VIEWS", "build", "instances", "mask_kf", "rect"]
