"""The scene behind the task U12 tests (not a test module itself).

D2's cover in miniature: desktop 1, six steps, 64x64 frames in ``scan`` and
``oak1``. The importer made two covers of one: ``cover.01`` ("Screw Cover",
the log takes it out at step 3) and ``cover.02`` ("Front Cover", taken out at
step 5). The annotator drew the cover under ``cover.02`` (keyframes at 2 and
4), said "it is open at 3" with ``T`` (a hand-written event), and confirmed
frames 3-6, on which ``cover.01`` was already gone.

``s1=True`` then makes the two S1 corrections the controller made on D2 before
merging: step 3 ``cover.01`` -> ``open``, step 5 retargeted to ``cover.01``.
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
KEEP = "cover.01"
GONE = "cover.02"
PSU = "psu.01"
STEPS = (1, 2, 3, 4, 5, 6)
COVER_A = (4, 4, 20, 20)
COVER_B = (6, 6, 22, 22)
PSU_RECT = (30, 30, 60, 60)


def rect(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros(HW, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def mask_kf(instance: str, box, anchor: int, view: str = "scan") -> ShapeKeyframe:
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

    def set_actions(self, actions: list[ActionRec]) -> None:
        self.db.replace_steps(DESKTOP, self.db.steps(DESKTOP), actions)
        recompile_events(self.db, DESKTOP, self.tax)


def build(db: Db, *, s1: bool = True, verify=(3, 4, 5, 6)) -> Scene:
    tax = load_taxonomy()
    db.upsert_desktop(DESKTOP, {"brand": "test"})
    db.upsert_instance(InstanceRec(KEEP, DESKTOP, "cover", attrs={"of": "motherboard_screws"},
                                   raw_names=["Screw Cover"]))
    db.upsert_instance(InstanceRec(GONE, DESKTOP, "cover", attrs={"of": "front_bezel"},
                                   raw_names=["Front Cover"]))
    db.upsert_instance(InstanceRec(PSU, DESKTOP, "psu", raw_names=["PSU"]))
    db.replace_steps(DESKTOP, [StepRec(DESKTOP, s, "normal", f"step {s}") for s in STEPS],
                     [ActionRec(DESKTOP, 3, 0, KEEP, "remove"),
                      ActionRec(DESKTOP, 5, 0, GONE, "remove")])
    db.add_events([StateEvent(DESKTOP, 3, GONE, "state", "closed", "open", "scan", auto=False)])
    recompile_events(db, DESKTOP, tax)
    for view in VIEWS:
        db.set_pose_segment(DESKTOP, view, 1, 1, 6, 6, None, None)
        for step in STEPS:
            db.upsert_frame(FrameKey(DESKTOP, step, view), f"{view}_{step:03d}.jpg",
                            {"hw": list(HW)}, None)
        db.add_keyframe(mask_kf(PSU, PSU_RECT, 6, view))
    db.add_keyframe(mask_kf(GONE, COVER_A, 2))
    db.add_keyframe(mask_kf(GONE, COVER_B, 4))
    db.set_zorder(ZOrderRec(DESKTOP, "scan", 1, [(PSU, "main"), (GONE, "main")]))
    db.add_relation(DESKTOP, "covered_by", PSU, KEEP, source="rule")
    db.add_relation(DESKTOP, "covered_by", PSU, GONE, source="manual", reason="seen")
    scene = Scene(db=db, tax=tax, svc=TruthService(db, tax, cache_dir=None))
    scene.refresh_all()
    for step in verify:
        scene.svc.verify_frame(scene.key(step), "tester")
    if s1:
        scene.set_actions([ActionRec(DESKTOP, 3, 0, KEEP, "open"),
                           ActionRec(DESKTOP, 5, 0, KEEP, "remove")])
    restamp_graph(db, DESKTOP, tax)
    return scene


#: Tables that are bookkeeping rather than annotation state: an undo writes
#: its own op row, re-queues frames and drops digests by design.
VOLATILE = {"op_log", "recheck_queue", "frame_digest", "sqlite_sequence"}


def dump(db: Db) -> dict[str, list]:
    """Every annotation table as a sorted list of rows (see :data:`VOLATILE`).

    ``state_event``'s derived rows (``auto=1``) are recompiled after every
    merge and undo, so they are compared without their ``id``; the
    hand-written ones keep theirs.
    """
    out: dict[str, list] = {}
    names = [r[0] for r in db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    for name in names:
        if name in VOLATILE:
            continue
        rows = [tuple(r) for r in db.conn.execute(f"SELECT * FROM {name}")]
        if name == "state_event":
            rows = [(None,) + row[1:] if row[-1] else row for row in rows]
        out[name] = sorted(rows, key=repr)
    return out


def blob(db: Db) -> bytes:
    """``dump`` as bytes, for an exact comparison message."""
    return repr(dump(db)).encode("utf-8")


__all__ = ["DESKTOP", "GONE", "KEEP", "PSU", "STEPS", "Scene", "VIEWS", "blob", "build",
           "dump", "mask_kf", "rect"]
