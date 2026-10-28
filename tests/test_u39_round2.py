"""Task U39 round 2: a part 画框外 takes no pixel and hides nothing.

The controller's ruling: a part said to be outside this view's picture is not
in it, so in the compiler's layer painting it can neither hold a pixel nor
cover a part under it -- whatever shape carries over from the segment's other
frames. The shape stays in the database (taking 画框外 back brings it into
force again).

The scene is :mod:`vis_review_scene`'s: ``screw.motherboard.01`` (A) is drawn
over its own cell and the whole of ``ram_latch.03``'s (B), so B is
``occluded_full`` under A. Pinned: with A 画框外, B's visible mask is its whole
shape and A has no pixel, no box and no layer; ``Ctrl+Z`` restores both; on a
confirmed frame B's new pixels go to Review as a conflict (nobody confirmed
them), A's label is signed; and the exports and ``cli check`` agree -- A is not
there, B is whole -- in COCO, the VLM rows, the scanner package and the check.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import vis_review_scene as V
from session_scene import DESKTOP, VIEW
from tda.core.export import export_coco
from tda.core.export.coco import load_ctx
from tda.core.export.vlm import _read_view
from tda.core.model import FrameKey
from tda.core.occluder import OFF_FRAME

A, B = V.COVER, V.HIDDEN
OUT = "out_of_view"
MARKED = [12, 13, 14]


def area(mask) -> int:
    return 0 if mask is None else int(np.count_nonzero(mask))


def painted(compiled) -> set[str]:
    return {inst for layers in compiled.layers.values() for inst, _part in layers}


@pytest.fixture
def scene(tmp_path):
    session = V.build(tmp_path, confirm=())
    session.goto(14, force=True)
    yield session
    session.close(force=True)


# --------------------------------------------------------------------------- #
# the compiler
# --------------------------------------------------------------------------- #
def test_an_off_frame_part_hides_nothing_and_takes_no_pixel(scene):
    before = scene.compiled()
    assert before.instances[B].visibility == "occluded_full" and area(before.instances[B].visible) == 0
    assert area(before.instances[A].visible) > 0 and A in painted(before)
    shapes = [kf.id for kf in scene.db.keyframes(DESKTOP, VIEW, A)]

    scene.set_cable_range(A, OFF_FRAME, 14)
    after = scene.compiled()
    a, b = after.instances[A], after.instances[B]
    assert (a.visibility, a.visible, a.box) == (OUT, None, None)
    assert A not in painted(after) and A not in after.painted.get("in_chassis", [])
    assert a.amodal is not None and area(a.amodal) > 0        # the shape is still there
    # B is whole: every pixel of its shape is seen
    assert b.visibility == "visible" and b.occlusion_ratio == 0.0
    assert area(b.visible) == area(b.amodal) > 0 and b.box is not None
    assert not [p for p in after.problems if A in p or B in p]
    # the stored shape is not touched
    assert [kf.id for kf in scene.db.keyframes(DESKTOP, VIEW, A)] == shapes

    scene.undo()
    back = scene.compiled()
    assert back.instances[B].visibility == "occluded_full" and area(back.instances[B].visible) == 0
    assert area(back.instances[A].visible) == area(before.instances[A].visible)
    assert A in painted(back)


def test_other_frames_of_the_segment_are_untouched(scene):
    scene.set_cable_range(A, OFF_FRAME, 14)
    scene.goto(13, force=True)
    here = scene.compiled()
    assert here.instances[B].visibility == "occluded_full" and A in painted(here)


def test_on_a_confirmed_frame_the_revealed_part_goes_to_review(tmp_path):
    session = V.build(tmp_path)                       # 14-10 confirmed
    try:
        session.goto(14, force=True)
        session.set_cable_range(A, OFF_FRAME, 14)
        key = FrameKey(DESKTOP, 14, VIEW)
        open_ = [(int(c["step"]), c["instance"])
                 for c in session.db.conflicts(DESKTOP, VIEW, open_only=True)]
        # A's own label is the annotator's word, signed; B's pixels nobody
        # confirmed -- a conflict for Review, never signed in passing
        assert open_ == [(14, B)]
        assert session.db.compiled(key)[A]["visibility"] == OUT
        assert session.db.compiled(key)[B]["visibility"] == "occluded_full"
    finally:
        session.close(force=True)


# --------------------------------------------------------------------------- #
# the exports and the check, on frames where A is 画框外 over B
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def confirmed(tmp_path_factory):
    root = tmp_path_factory.mktemp("u39_r2")
    session = V.build(root, confirm=())
    session.goto(14, force=True)
    assert session.set_cable_range(A, OFF_FRAME, 12)["frames"] == MARKED
    for step in V.CONFIRMED:                          # 14 down to 10
        session.goto(step, force=True)
        assert session.confirm_frame(), step
    session.root = root
    yield session
    session.close(force=True)


def anns_of(doc: dict, step: int, instance: str) -> list[dict]:
    image = next(i["id"] for i in doc["images"] if i["extra"]["step"] == step)
    return [a for a in doc["annotations"]
            if a["image_id"] == image and a["attributes"]["instance_key"] == instance]


def test_coco_has_no_a_and_a_whole_b(confirmed, tmp_path):
    doc = export_coco(confirmed.db, confirmed.tax, [DESKTOP], VIEW, str(tmp_path / "c.json"),
                      truth=confirmed.truth)
    for step in MARKED:
        assert anns_of(doc, step, A) == []
        [b] = anns_of(doc, step, B)
        assert b["attributes"]["visibility"] == "visible" and b["iscrowd"] == 0
        assert b["area"] == b["amodal_area"] > 0
    for step in (10, 11):
        assert len(anns_of(doc, step, A)) == 1
        [b] = anns_of(doc, step, B)
        assert b["iscrowd"] == 1 and b["area"] == 0     # covered by A there


def test_the_vlm_rows_say_the_same(confirmed):
    ctx = load_ctx(confirmed.db, confirmed.tax, DESKTOP, VIEW)
    frames, _order = _read_view(confirmed.db, ctx, DESKTOP, VIEW, set(), True)
    for step in MARKED:
        assert frames[step].rows[A]["visibility"] == OUT and A not in frames[step].pointable
        assert frames[step].rows[B]["visibility"] == "visible" and B in frames[step].pointable
    assert B not in frames[11].pointable


def test_cli_check_has_nothing_to_say_about_either(confirmed, capsys):
    from app_scene import make_paths
    from tda.cli import main

    root = Path(confirmed.root)
    cfg = make_paths(root)
    cfg.update(db_path=str(root / "tda.sqlite"), cache_dir=str(root / "cache"))
    paths = root / "paths_u39.yaml"
    paths.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    main(["--paths", str(paths), "check", "--desktop", str(DESKTOP), "--view", VIEW,
          "--limit", "500"])
    out = capsys.readouterr().out
    listed = [line.strip() for line in out.splitlines() if line.strip().startswith("- ")]
    assert not [line for line in listed if A in line or B in line]
    assert "open conflicts: 0" in out
    # what check counts as covered (normal) is B on 10-11 only, never on 12-14
    stats = confirmed.truth.refresh_range(DESKTOP, VIEW, [10, 11, 12, 13, 14],
                                          ignore_digest=True)
    covered = sorted((row.step, row.instance) for row in stats["empty_visible"]
                     if row.instance in (A, B))
    assert covered == [(10, B), (11, B)]


def test_the_scanner_package_has_no_a_and_a_whole_b(tmp_path):
    from test_u30_scanpack import COVER, HIDDEN, build_scene, make_pack
    from tda.core.db import Db
    from tda.core.taxonomy import load_taxonomy
    from tda.core.truth import TruthService
    from tda.ui.session_cable import cable_range

    desktop, marked = 13, (11, 12)
    db_path = build_scene(tmp_path, desktop)
    tax = load_taxonomy()
    db = Db(str(db_path))
    try:
        truth = TruthService(db, tax, cache_dir=str(tmp_path / "cache"))
        cable_range(db, truth, FrameKey(desktop, 12, VIEW), COVER, OFF_FRAME, 11, "tester")
        # what the label revealed of HIDDEN nobody confirmed: Review accepts it
        conflicts = db.conflicts(desktop, VIEW, open_only=True)
        assert sorted((int(c["step"]), c["instance"]) for c in conflicts) == [
            (11, HIDDEN), (12, HIDDEN)]
        for c in conflicts:
            truth.resolve_conflict(int(c["id"]), "accept_new", "tester")
        assert not db.conflicts(desktop, VIEW, open_only=True)
    finally:
        db.close()
    make_pack(tmp_path, db_path, desktop)
    ddir = tmp_path / "pack" / f"D{desktop:02d}"
    coco = json.loads((ddir / "annotations.json").read_text(encoding="utf-8"))
    ids = {v["key"]: int(k) for k, v in json.loads(
        (ddir / "instances.json").read_text(encoding="utf-8"))["instances"].items()}
    step_of = {i["id"]: i["extra"]["step"] for i in coco["images"]}
    by = {(step_of[a["image_id"]], a["attributes"]["instance_key"]): a
          for a in coco["annotations"]}
    for step in marked:
        assert (step, COVER) not in by
        hidden = by[(step, HIDDEN)]
        assert hidden["iscrowd"] == 0 and hidden["area"] == hidden["amodal_area"] > 0
        assert hidden["occluders"] == []
        img = next(i for i in coco["images"] if i["extra"]["step"] == step)
        label = cv2.imread(str(ddir / img["label_file"]), cv2.IMREAD_UNCHANGED)
        assert not np.any(label == ids[COVER])
        assert int(np.count_nonzero(label == ids[HIDDEN])) == hidden["area"]
    # elsewhere HIDDEN is still the crowd row under COVER
    assert by[(13, HIDDEN)]["iscrowd"] == 1 and (13, COVER) in by
