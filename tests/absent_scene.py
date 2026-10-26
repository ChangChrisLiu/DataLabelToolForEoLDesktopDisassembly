"""The D34 situation on the D13 test scene (task U34; not a test module).

D34 has no optical drive, yet its sheet logs one: a failed "Try to remove the
optical drive", two "optical drive screw" rows. D37's sheet *pushes* one out
(``displace``). This adds both to the session scene's truncated D13 sheet:

* ``optical_drive.01`` -- named by the sheet (``raw_names``), never drawn:
  step 9 tries to remove it (``failed``), step 11 pushes it out (``displace``);
* ``screw.optical_drive.01`` / ``.02`` fasten it and are unscrewed at 7 / 8;
* ``screw.optical_drive.03`` rides on it (``parent`` + ``attached``) and is
  never operated -- the case of a child of an absent part, which must stay
  as the sheet says.

The actions go in at ``idx 1`` of existing steps, so every other part of the
scene is exactly the session scene's.
"""
from __future__ import annotations

from tda.core.db import Db
from tda.core.extra import recompile_events
from tda.core.model import ActionRec, InstanceRec

DRIVE = "optical_drive.01"
SCREWS = ("screw.optical_drive.01", "screw.optical_drive.02")
RIDER = "screw.optical_drive.03"
FAILED_AT = 9
DISPLACED_AT = 11


def add_drive(db: Db, tax, desktop: int, displace: bool = True) -> None:
    """Put the logged-but-absent optical drive into ``desktop``'s tables."""
    db.upsert_instance(InstanceRec(key=DRIVE, desktop=desktop, cls="optical_drive",
                                   raw_names=["Try to remove the optical drive"]))
    for n, key in enumerate(SCREWS, start=1):
        db.upsert_instance(InstanceRec(
            key=key, desktop=desktop, cls="screw", fastens=DRIVE,
            attrs={"role": "optical_drive", "captive": False},
            raw_names=[f"Optical drive Screw {n}"]))
    db.upsert_instance(InstanceRec(
        key=RIDER, desktop=desktop, cls="screw", parent=DRIVE, attached=True,
        attrs={"role": "optical_drive", "captive": True},
        raw_names=["Optical drive Screw 3"]))
    steps = db.steps(desktop)
    actions = list(db.actions(desktop))
    actions += [
        ActionRec(desktop, 7, 1, SCREWS[0], "unscrew", tool="T15"),
        ActionRec(desktop, 8, 1, SCREWS[1], "unscrew", tool="T15"),
        ActionRec(desktop, FAILED_AT, 1, DRIVE, "remove", result="failed"),
    ]
    if displace:
        actions.append(ActionRec(desktop, DISPLACED_AT, 1, DRIVE, "displace", tool="hand"))
    db.replace_steps(desktop, steps, actions)
    recompile_events(db, desktop, tax)
