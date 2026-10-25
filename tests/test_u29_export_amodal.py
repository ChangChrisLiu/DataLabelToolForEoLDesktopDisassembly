"""Task U29 (F, U26 N6): the COCO export's whole shapes without compiling every frame.

U26 gave every mask annotation its ``amodal_segmentation`` and paid for it with
one compilation per exported frame (D3/scan 0.1 -> 1.6 s, D2-D21 2.6 -> 36 s):
the truth table stores the visible mask only. But a part's whole shape is its
own keyframe's parts on the frame and nothing else -- painting, occluders,
pairwise overrides and labels never touch it -- so the export now takes it
from the keyframe the compiler would select (:func:`keyframe_amodal`), encoded
once per keyframe, transform and frame size for the whole run.

Pinned here: the file is **byte for byte** the one the compiling exporter
wrote, on the U24/U26 scene (a part wholly covered, a cable, a 4 px part, a
「看不见」) and on the same scene with a registration transform, a two-part
keyframe and a part of the wrong size; the shape equals the compiler's array
and window; and an export over an untouched view compiles nothing at all now,
whatever the flags (U2's property, which U26 had to give up for the default).
"""
from __future__ import annotations

import hashlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import vis_review_scene as V
from session_scene import rect
from tda.core import masks
from tda.core.compiler import keyframe_amodal
from tda.core.export import coco as C
from tda.core.export import export_coco
from tda.core.model import FrameKey, ShapePart, Similarity

FLAGS = {
    "default": {},
    "roi_crop": {"roi_crop": True},
    "no_amodal": {"amodal": False},
    "no_hidden": {"hidden": False},
    "boxes": {"include_boxes": True},
}


def compiled_rles(service, key, wanted, cache=None, shapes=None):
    """The exporter as U26 wrote it: compile the frame, encode each amodal."""
    if not wanted:
        return {}
    compiled = service.compile(key, cache)
    out = {}
    for name in wanted:
        found = compiled.instances.get(name)
        if found is not None and found.amodal is not None:
            out[name] = masks.encode_rle(found.amodal, getattr(found, "amodal_window", None))
    return out


def sha(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def both(session, tmp_path, monkeypatch, name: str, **kw) -> tuple[str, str]:
    """The sha256 of the export written now and of the one U26's exporter writes."""
    common = dict(only_verified=False, truth=session.truth, allow_conflicts=True, **kw)
    now = tmp_path / f"{name}_now.json"
    export_coco(session.db, session.tax, [V.DESKTOP], V.VIEW, str(now), **common)
    with monkeypatch.context() as patch:
        patch.setattr(C, "_amodal_rles", compiled_rles)
        then = tmp_path / f"{name}_u26.json"
        export_coco(session.db, session.tax, [V.DESKTOP], V.VIEW, str(then), **common)
    return sha(now), sha(then)


@pytest.fixture
def scene(tmp_path):
    session = V.build(tmp_path / "scene")
    yield session
    session.close(force=True)


@pytest.mark.parametrize("flags", sorted(FLAGS))
def test_the_file_is_byte_for_byte_the_compiling_exporters(scene, tmp_path, monkeypatch,
                                                           flags):
    now, then = both(scene, tmp_path, monkeypatch, flags, **FLAGS[flags])
    assert now == then


def _twist(session) -> None:
    """A registration transform on some frames, a two-part shape, a part of the wrong size."""
    db = session.db
    for step in (11, 13):
        db.set_transform(FrameKey(V.DESKTOP, step, V.VIEW),
                         Similarity(scale=1.03, theta=0.04, tx=3.5, ty=-2.25))
    kf = next(k for k in db.keyframes(V.DESKTOP, V.VIEW, V.HALF))
    kf.parts = list(kf.parts) + [
        ShapePart("tab", masks.encode_rle(rect(200, 200, 230, 214, V.HW))),
        ShapePart("stray", masks.encode_rle(rect(1, 1, 5, 5, (32, 32))))]
    db.update_keyframe(kf)
    session.refresh_all()


@pytest.mark.parametrize("flags", ["default", "roi_crop", "no_amodal"])
def test_with_a_transform_two_parts_and_a_wrong_size_part_too(scene, tmp_path, monkeypatch,
                                                              flags):
    _twist(scene)
    now, then = both(scene, tmp_path, monkeypatch, flags, **FLAGS[flags])
    assert now == then


def test_the_shape_is_the_compilers_array_and_window(scene):
    _twist(scene)
    truth = scene.truth
    for step in (10, 11, 13, 14):
        key = FrameKey(V.DESKTOP, step, V.VIEW)
        compiled = truth.compile(key)
        from tda.core.truth_fresh import selected_keyframes
        from tda.core.truth_inputs import gather

        inputs = gather(scene.db, scene.tax, key, None, truth.cache_dir)
        for name, kf in selected_keyframes(inputs).items():
            found = compiled.instances[name]
            if kf is None or kf.geom_type == "box":
                assert found.amodal is None, name
                continue
            amodal, window = keyframe_amodal(kf, inputs.transform, inputs.hw)
            assert np.array_equal(amodal, found.amodal), (step, name)
            assert window == found.amodal_window, (step, name)


def test_an_untouched_view_compiles_nothing_whatever_the_flags(scene, tmp_path, monkeypatch):
    """U2's property is back for the default export; each shape is encoded once."""
    import tda.core.export.coco as coco_mod
    import tda.core.truth as truth_mod

    export_coco(scene.db, scene.tax, [V.DESKTOP], V.VIEW, str(tmp_path / "warm.json"),
                only_verified=False, truth=scene.truth)
    compiled: list[int] = []
    original = truth_mod.compile_frame
    monkeypatch.setattr(truth_mod, "compile_frame",
                        lambda key, *a, **k: (compiled.append(key.step),
                                              original(key, *a, **k))[1])
    shaped: list[int] = []
    shape = coco_mod.keyframe_amodal
    monkeypatch.setattr(coco_mod, "keyframe_amodal",
                        lambda kf, *a: (shaped.append(kf.id), shape(kf, *a))[1])
    for name, flags in sorted(FLAGS.items()):
        compiled.clear()
        shaped.clear()
        doc = export_coco(scene.db, scene.tax, [V.DESKTOP], V.VIEW,
                          str(tmp_path / f"{name}.json"), only_verified=False,
                          truth=scene.truth, **flags)
        assert compiled == [], name
        assert len(shaped) == len(set(shaped)), name           # once per keyframe
        assert len(shaped) <= len(scene.db.keyframes(V.DESKTOP, V.VIEW))
        assert len(shaped) < len(doc["annotations"]) / 2, name  # not once per annotation
