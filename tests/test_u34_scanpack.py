"""Task U34 round 2 (N4): the scanner-view package never names a part that was never there.

``scripts/export_scanpack.py`` numbered every real instance and listed every
action row; a part said absent from the start (「这台没有这个零件」) kept an id,
an ``instances.json`` entry and its sheet's failed "remove" in ``frames.json``.
Like the COCO and VLM exports it now skips ``DesktopCtx.absent`` and those
parts' action rows. D34 itself cannot be packed yet, so the U30 fixture scene
(D13) gets the optical drive the sheet logs and the machine does not have.
"""
from __future__ import annotations

import json
from pathlib import Path

from test_u30_scanpack import VIEW, build_scene, make_pack  # noqa: F401

from tda.core import absent as AB
from tda.core.db import Db
from tda.core.model import ActionRec, InstanceRec
from tda.core.taxonomy import load_taxonomy

DESKTOP = 13
DRIVE = "optical_drive.01"
TRIED_AT = 9


def test_an_absent_part_has_no_id_no_entry_and_no_action_row(tmp_path: Path):
    db_path = build_scene(tmp_path, DESKTOP)
    tax = load_taxonomy()
    db = Db(str(db_path))
    try:
        db.upsert_instance(InstanceRec(key=DRIVE, desktop=DESKTOP, cls="optical_drive",
                                       raw_names=["Try to remove the optical drive"]))
        db.replace_steps(DESKTOP, db.steps(DESKTOP),
                         [*db.actions(DESKTOP),
                          ActionRec(DESKTOP, TRIED_AT, 1, DRIVE, "remove", result="failed")])
        AB.mark_absent(db, tax, DESKTOP, DRIVE, "tester")
    finally:
        db.close()
    make_pack(tmp_path, db_path, DESKTOP)
    ddir = tmp_path / "pack" / f"D{DESKTOP:02d}"
    instances = json.loads((ddir / "instances.json").read_text(encoding="utf-8"))
    assert DRIVE not in {v["key"] for v in instances["instances"].values()}
    frames = json.loads((ddir / "frames.json").read_text(encoding="utf-8"))["frames"]
    rows = frames[str(TRIED_AT)]["actions"]
    assert rows and all(r["target"] != DRIVE for r in rows)          # the SSD's still there
    assert all(r["target_id"] is not None for r in rows)
    seq = json.loads((ddir / "sequence.json").read_text(encoding="utf-8"))
    assert DRIVE not in json.dumps(seq)
    coco = json.loads((ddir / "annotations.json").read_text(encoding="utf-8"))
    assert all(a["attributes"]["instance_key"] != DRIVE for a in coco["annotations"])
