"""Task U39 in the exports: a part 画框外 on a frame is not annotated on it.

画框外 (``out_of_view``, no pixels) leaves no visible mask and no box, so the
COCO export writes no row for the part on that frame -- as for any
``out_of_view`` -- even though its shape (the amodal estimate) is still there
and the frame is confirmed; frames before the label are untouched. The
scanner package (``scripts/export_scanpack.py``) is built on COCO: no
annotation, no pixel in the label map, nothing in ``clean``. The VLM export
reads the compiled label: ``out_of_view``, never pointable, and the abstain
reason says so.
"""
from __future__ import annotations

import csv
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import cv2
import numpy as np
import pytest

import vis_review_scene as V
from session_scene import DESKTOP, VIEW, chassis_instances
from tda.core.export import export_coco
from tda.core.export.coco import load_ctx
from tda.core.export.vlm import _read_view
from tda.core.export.vlm_tasks import seen_as, unseen_reason
from tda.core.model import FrameKey
from tda.core.occluder import OFF_FRAME

OUT = "out_of_view"
MARKED = [12, 13, 14]            # 画框外 here; 10 and 11 keep their rows
_TAKEN = {V.HIDDEN, V.COVER, V.HALF, V.HALF_COVER, V.TINY, V.CABLED, V.OUT, "chassis"}


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    session = V.build(tmp_path_factory.mktemp("u39_export"))
    part = next(k for k in chassis_instances(session, 14)
                if k not in _TAKEN and k in chassis_instances(session, 10))
    session.goto(14, force=True)
    assert session.set_cable_range(part, OFF_FRAME, 12)["frames"] == MARKED
    session.part = part
    yield session
    session.close(force=True)


@pytest.fixture(scope="module")
def coco(scene, tmp_path_factory):
    out = tmp_path_factory.mktemp("u39_coco") / "c.json"
    return export_coco(scene.db, scene.tax, [DESKTOP], VIEW, str(out), truth=scene.truth)


def anns_of(doc: dict, step: int, instance: str) -> list[dict]:
    image = next(i["id"] for i in doc["images"] if i["extra"]["step"] == step)
    return [a for a in doc["annotations"]
            if a["image_id"] == image and a["attributes"]["instance_key"] == instance]


def test_coco_has_no_row_for_it_where_it_is_off_frame(scene, coco):
    part = scene.part
    for step in MARKED:
        key = FrameKey(DESKTOP, step, VIEW)
        assert scene.db.get_frame(key)["review_status"] == "verified"
        assert scene.db.compiled_label(key, part)["visibility"] == OUT
        assert anns_of(coco, step, part) == []
    for step in (10, 11):
        [row] = anns_of(coco, step, part)
        assert row["attributes"]["visibility"] != OUT and row["area"] > 0
    # the frames themselves are exported, with their other parts
    for step in MARKED:
        assert anns_of(coco, step, "chassis")
    # psu.01's out_of_view on 12 (a digit key's, the same statement): no row either
    assert anns_of(coco, V.OUT_STEP, V.OUT) == []


def test_coco_without_amodal_or_hidden_agrees(scene, tmp_path):
    doc = export_coco(scene.db, scene.tax, [DESKTOP], VIEW, str(tmp_path / "n.json"),
                      truth=scene.truth, amodal=False, hidden=False)
    assert all(anns_of(doc, step, scene.part) == [] for step in MARKED)


def test_vlm_reads_out_of_view_and_never_points_at_it(scene):
    ctx = load_ctx(scene.db, scene.tax, DESKTOP, VIEW)
    frames, _order = _read_view(scene.db, ctx, DESKTOP, VIEW, set(), True)
    part = scene.part
    for step in MARKED:
        row = frames[step].rows[part]
        assert row["visibility"] == OUT and "occluder" not in row
        assert part not in frames[step].pointable
        assert seen_as(row) == {"visibility": OUT}
        assert unseen_reason(part, row, VIEW) == f"{part} is {OUT} in view {VIEW}"
    assert part in frames[11].pointable


# --------------------------------------------------------------------------- #
# the scanner package
# --------------------------------------------------------------------------- #
def test_the_scanpack_has_no_annotation_and_no_clean_row_for_it(tmp_path):
    from test_u30_scanpack import (  # noqa: F401 - the U30 fixture scene
        CABLED,
        CHASSIS,
        COVER,
        HALF,
        HALF_COVER,
        HIDDEN,
        NEIGHBOR,
        OUTSIDE,
        PATCHED,
        STALE_LOWER,
        STALE_UPPER,
        build_scene,
        make_pack,
    )
    from tda.core.db import Db
    from tda.core.states import needs_geom
    from tda.core.taxonomy import load_taxonomy
    from tda.core.truth import TruthService
    from tda.core.truth_inputs import instances_of, state_of
    from tda.ui.session_cable import cable_range

    desktop, marked = 13, (11, 12)
    special = {CABLED, CHASSIS, COVER, HALF, HALF_COVER, HIDDEN, NEIGHBOR, OUTSIDE, PATCHED,
               STALE_LOWER, STALE_UPPER}
    db_path = build_scene(tmp_path, desktop)
    tax = load_taxonomy()
    db = Db(str(db_path))
    try:
        insts = instances_of(db, desktop)

        def here(step: int) -> set[str]:
            state = state_of(db, tax, desktop, step)
            return {k for k, kind in needs_geom(insts, state, tax).items()
                    if kind == "mask" and state[k].placement == "in_chassis"}

        part = sorted((here(11) & here(12) & here(13)) - special)[0]
        truth = TruthService(db, tax, cache_dir=str(tmp_path / "cache"))
        result = cable_range(db, truth, FrameKey(desktop, 12, VIEW), part, OFF_FRAME, 11,
                             "tester")
        assert result["frames"] == list(marked)
        for step in marked:
            assert db.get_frame(FrameKey(desktop, step, VIEW))["review_status"] == "verified"
    finally:
        db.close()
    make_pack(tmp_path, db_path, desktop)
    ddir = tmp_path / "pack" / f"D{desktop:02d}"
    coco = json.loads((ddir / "annotations.json").read_text(encoding="utf-8"))
    instances = json.loads((ddir / "instances.json").read_text(encoding="utf-8"))
    ident = next(int(i) for i, v in instances["instances"].items() if v["key"] == part)
    step_of = {i["id"]: i["extra"]["step"] for i in coco["images"]}
    mine = {step_of[a["image_id"]]: a for a in coco["annotations"]
            if a["attributes"]["instance_key"] == part}
    assert not set(marked) & set(mine)                # no annotation where it is off frame
    assert 13 in mine and 10 in mine                  # ... and its rows elsewhere
    for img in coco["images"]:
        label = cv2.imread(str(ddir / img["label_file"]), cv2.IMREAD_UNCHANGED)
        assert label is not None
        if img["extra"]["step"] in marked:
            assert not np.any(label == ident)          # no pixel of it in the map
        elif img["extra"]["step"] in mine:
            assert np.any(label == ident)
    with open(tmp_path / "pack" / "clean_parts.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert not [r for r in rows if int(r["instance_id"]) == ident
                and int(r["step"]) in marked]
    manifest = json.loads((tmp_path / "pack" / "manifest.json").read_text(encoding="utf-8"))
    (desk,) = manifest["per_desktop"]
    assert desk["annotations"] == len(coco["annotations"])
    assert desk["clean"] == sum(a["clean"] for a in coco["annotations"]) == len(rows)
    # still a part of the machine: it keeps its id and its entry
    assert instances["instances"][str(ident)]["key"] == part
