"""The W3b fixture: a CPU teardown with a cooler, its screws, a lever and a shroud (not a test).

One desktop, eleven logical steps, two views -- the scanner confirmed frame by
frame, OAK1 with frames and nothing else (the state every OAK / RealSense view
of the real database is in):

* the CPU is under a heatsink-fan (``covered_by``) and locked by its lever
  (``locked_by``); the cooler is held by four **captive** screws
  (``fastened_by``), plugged in by its fan lead (``connected_to``) and covered
  by a shroud (``covered_by``);
* the lever is covered by the optical drive through an edge imported from
  pixels (``source = 'pixels:W1'``), which is the only thing standing between
  "open the lever" and its truth -- the ``graph_blocked_pixels`` case;
* steps 4-7 unscrew the captive screws one by one: the log moves, the picture
  does not (the V3 / V12 captive rule).

The graph is written by hand (``status = accepted``) so the tests know exactly
which edge says what.
"""
from __future__ import annotations

import numpy as np

from tda.core import masks
from tda.core.db import Db
from tda.core.model import (
    ActionRec,
    FrameKey,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    StepRec,
    ZOrderRec,
)
from tda.core.taxonomy import Taxonomy, load_taxonomy
from tda.core.truth import TruthService

DESKTOP = 41
VIEW = "scan"
OTHER = "oak1"
HW = (64, 64)
LAST_STEP = 11
STEPS = tuple(range(1, LAST_STEP + 1))

CHASSIS = "chassis.01"
BOARD = "motherboard.01"
CPU = "cpu.01"
LEVER = "cpu_socket_lever.01"
COOLER = "cpu_cooler.01"
SHROUD = "cover.01"
FAN = "connector.fan.01"
OPTICAL = "optical_drive.01"
SCREWS = tuple(f"screw.cpu_cooler.{i:02d}" for i in range(1, 5))
BOARD_SCREW = "screw.motherboard.01"

RECTS = {
    CHASSIS: (0, 0, 64, 64),
    BOARD: (2, 2, 50, 50),
    CPU: (16, 16, 28, 28),
    LEVER: (28, 16, 32, 30),
    COOLER: (12, 12, 32, 32),
    SHROUD: (10, 10, 36, 36),
    FAN: (36, 4, 42, 10),
    OPTICAL: (40, 30, 62, 50),
    SCREWS[0]: (12, 12, 15, 15),
    SCREWS[1]: (29, 12, 32, 15),
    SCREWS[2]: (12, 29, 15, 32),
    SCREWS[3]: (29, 29, 32, 32),
    BOARD_SCREW: (4, 38, 14, 48),
}

#: step -> (verb, target)
LOG = {
    2: ("open", SHROUD),
    3: ("disconnect", FAN),
    4: ("unscrew", SCREWS[0]),
    5: ("unscrew", SCREWS[1]),
    6: ("unscrew", SCREWS[2]),
    7: ("unscrew", SCREWS[3]),
    8: ("remove", OPTICAL),
    9: ("remove", COOLER),
    10: ("open", LEVER),
    11: ("remove", CPU),
}

#: The edges, one per fact the tests read. ``source`` defaults to ``manual``.
EDGES = (
    *({"type": "fastened_by", "target": COOLER, "blocker": s} for s in SCREWS),
    {"type": "covered_by", "target": COOLER, "blocker": SHROUD},
    {"type": "connected_to", "target": COOLER, "blocker": FAN},
    {"type": "covered_by", "target": CPU, "blocker": COOLER},
    {"type": "locked_by", "target": CPU, "blocker": LEVER},
    {"type": "covered_by", "target": LEVER, "blocker": OPTICAL, "source": "pixels:W1"},
    {"type": "fastened_by", "target": BOARD, "blocker": BOARD_SCREW},
)


def rect(box) -> np.ndarray:
    x0, y0, x1, y1 = box
    m = np.zeros(HW, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _instances() -> list[InstanceRec]:
    out = [
        InstanceRec(key=CHASSIS, desktop=DESKTOP, cls="chassis"),
        InstanceRec(key=BOARD, desktop=DESKTOP, cls="motherboard"),
        InstanceRec(key=CPU, desktop=DESKTOP, cls="cpu", mounted_on=BOARD),
        InstanceRec(key=LEVER, desktop=DESKTOP, cls="cpu_socket_lever",
                    parent=BOARD, attached=True),
        InstanceRec(key=COOLER, desktop=DESKTOP, cls="cpu_cooler",
                    attrs={"kind": "heatsink_fan"}),
        InstanceRec(key=SHROUD, desktop=DESKTOP, cls="cover",
                    attrs={"of": "cpu_cooler"}),
        InstanceRec(key=FAN, desktop=DESKTOP, cls="connector",
                    attrs={"kind": "fan"}, socket_host=BOARD),
        InstanceRec(key=OPTICAL, desktop=DESKTOP, cls="optical_drive"),
        InstanceRec(key=BOARD_SCREW, desktop=DESKTOP, cls="screw",
                    attrs={"role": "motherboard", "head": "PH2", "captive": False},
                    fastens=BOARD),
    ]
    out += [InstanceRec(key=s, desktop=DESKTOP, cls="screw",
                        attrs={"role": "cpu_cooler", "head": "PH2", "captive": True},
                        fastens=COOLER, parent=COOLER, attached=True)
            for s in SCREWS]
    return out


def build(db: Db, *, views=(VIEW, OTHER), confirm=(VIEW,), edges=EDGES,
          group: bool = False, cooler_kind: str = "heatsink_fan") -> Taxonomy:
    """Seed ``db``; ``confirm`` lists the views whose frames are signed off.

    ``group=True`` puts the four cooler screws in one unordered ``group_id``;
    ``cooler_kind`` is the cooler's ``kind`` (``heatsink`` is what W3a's rule
    makes the CPU's direct cover on a two-cooler desktop).
    """
    tax = load_taxonomy()
    db.upsert_desktop(DESKTOP, {"brand": "Dell", "model_family": "OptiPlex 7020 MT",
                                "chassis_type": "mt"})
    for rec in _instances():
        if group and rec.key in SCREWS:
            rec.group_id, rec.group_order = "cooler_screws", "unordered"
        if rec.key == COOLER:
            rec.attrs = {"kind": cooler_kind}
        db.upsert_instance(rec)
    steps = [StepRec(DESKTOP, 1, "initial", "initial state")]
    actions = []
    for step, (verb, target) in sorted(LOG.items()):
        steps.append(StepRec(DESKTOP, step, "normal", f"{verb} {target}"))
        actions.append(ActionRec(DESKTOP, step, 0, target, verb, tool="hand"))
    db.replace_steps(DESKTOP, steps, actions)
    for view in views:
        for step in STEPS:
            db.upsert_frame(
                FrameKey(DESKTOP, step, view), f"F:/{view}/041/{step:03d}/P_0.png",
                {"hw": [HW[0], HW[1]]}, "2025-06-01T09:00:00",
                flags={"review_status": "unlabeled"},
            )
        db.set_pose_segment(DESKTOP, view, 1, 1, LAST_STEP, 1, None, None)
        if view not in confirm:
            continue  # a view nobody drew: frames and nothing else
        for key, box in RECTS.items():
            db.add_keyframe(ShapeKeyframe(
                id=None, instance=key, desktop=DESKTOP, view=view, pose_segment=1,
                anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
                parts=[ShapePart("main", rle=masks.encode_rle(rect(box)))],
                amodal_complete=True,
            ))
        order = [(CHASSIS, "main"), (BOARD, "main"), (CPU, "main"), (LEVER, "main"),
                 (COOLER, "main"), (SHROUD, "main"), (FAN, "main"), (OPTICAL, "main"),
                 *((s, "main") for s in SCREWS), (BOARD_SCREW, "main")]
        db.set_zorder(ZOrderRec(DESKTOP, view, 1, order))
    for spec in edges:
        db.add_relation(DESKTOP, spec["type"], spec["target"], spec["blocker"],
                        necessity=spec.get("necessity", "required"),
                        mode=spec.get("mode"), source=spec.get("source", "manual"),
                        status=spec.get("status", "accepted"))
    service = TruthService(db, tax)
    for view in confirm:
        service.refresh_range(DESKTOP, view, STEPS)
        for step in STEPS:
            service.verify_frame(FrameKey(DESKTOP, step, view), "tester")
    return tax
