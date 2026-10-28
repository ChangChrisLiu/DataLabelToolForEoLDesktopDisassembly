"""Task U38: a step with no photo has no pixels, cannot be confirmed, and loses its copy.

D61's scanner step 26 is a missed shot the index declares (``scan.61.skipped_steps``,
U25 round 2), so its frame row is ``missing`` with no path. The cache still held
``s026.png`` -- the copy of ``RGB261`` from when that folder was step 26 -- and
three things let it through:

* **reading**: :func:`tda.ui.session_images.frame_candidates` tried the cache
  name first and never asked whether the row had a photo, so the canvas showed
  step 27's picture under step 26's number;
* **measuring**: compiling the missing frame (visiting it) measured that file
  and wrote ``hw_source = measured`` and its ``cache_path`` onto the row that
  ``load-index`` had just cleared;
* **confirming**: ``verify_frame`` had no rule about missing frames, and 23
  rows were signed on a step nobody photographed.

Pinned here: the three gates (a, b, c), the cache's report and ``--prune-missing``
(d), and that a photographed frame reads and confirms exactly as before (e).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import cv2
import numpy as np
import pytest

from tda import pipeline as P
from tda.core import cache as C
from tda.core import masks
from tda.core.cache import VIEW_EXT, cache_path, has_no_photo
from tda.core.db import Db
from tda.core.index import DesktopIndex, FrameFile, save_index
from tda.core.model import FrameKey, ShapeKeyframe, ShapePart, ZOrderRec
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import HW_INFERRED, HW_MEASURED, frame_hw, infer_hw
from tda.core.truth_refusals import (
    FRAME_MISSING,
    NO_PHOTO,
    VerifyRefused,
    blocking_reasons,
    is_blocking,
    no_photo,
)
from tda.ui.panels.taskcard import NO_PHOTO_SENTENCE, explain_code, pair_problems
from tda.ui.session_images import ImageCache, frame_candidates, thumb_path

import app_scene as A

DESKTOP = 96
VIEW = "scan"
#: The scanner's nominal size: what a step with no photo is compiled at.
NOMINAL = list(infer_hw(VIEW))
NO_PHOTO_TEXT = ("this view has no photo of this step (a missed shot): there is "
                 "nothing to confirm")


def _png(path: str, value: int, hw=(48, 64)) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = np.full((*hw, 3), value, dtype=np.uint8)
    img[4:12, 4:20] = 255 - value          # not a flat image: a burst's metrics need texture
    assert cv2.imwrite(str(path), img)
    return str(path).replace("\\", "/")


def _stale_copy(cache_dir: Path, key: FrameKey, value: int = 77) -> tuple[str, str]:
    """What an earlier ``build-cache`` left under a step that has no photo now."""
    image = _png(cache_path(str(cache_dir), key, VIEW_EXT[key.view]), value)
    thumb = thumb_path(str(cache_dir), key)
    _png(thumb, value, hw=(12, 16))
    return image, thumb


# --------------------------------------------------------------------------- #
# (a) reading: no pixels for a frame with no photo
# --------------------------------------------------------------------------- #
def test_a_missing_frame_reads_nothing_whatever_its_aux_and_its_cache_name_hold(tmp_path):
    """The live D61 row: missing, no path, aux naming the stale copy, which exists."""
    cache = tmp_path / "cache"
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        key = FrameKey(61, 26, VIEW)
        stale, thumb = _stale_copy(cache, key)
        db.upsert_frame(key, None, {"hw": [48, 64], "hw_source": HW_MEASURED,
                                    "cache_path": stale}, None, {"missing": True})
        row = db.get_frame(key)
        assert has_no_photo(row)
        assert frame_candidates(str(cache), key, row) == []
        images = ImageCache(db, str(cache))
        assert images.image_path(key) is None
        assert images.thumb_path(key) is None
        assert images.get(key) is None
        # nothing to read is not "unreadable"
        assert images.why_unreadable(key) is None
        assert Path(stale).is_file() and Path(thumb).is_file()     # nothing deleted here
    finally:
        db.close()


def test_a_row_without_a_path_has_no_photo_either(tmp_path):
    """Missing flag or no stored path: the same answer, the one ``why_unreadable`` gives."""
    cache = tmp_path / "cache"
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        key = FrameKey(13, 5, VIEW)
        stale, _thumb = _stale_copy(cache, key)
        db.upsert_frame(key, "", {"cache_path": stale}, None)
        assert not db.get_frame(key)["missing"] and has_no_photo(db.get_frame(key))
        images = ImageCache(db, str(cache))
        assert images.image_path(key) is None and images.thumb_path(key) is None
        assert images.why_unreadable(key) is None
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# (e) the regression: a photographed frame reads exactly as before
# --------------------------------------------------------------------------- #
def test_a_photographed_frame_still_reads_its_copy_and_its_thumbnail(tmp_path):
    cache = tmp_path / "cache"
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        key = FrameKey(61, 27, VIEW)
        copy, thumb = _stale_copy(cache, key, value=90)
        db.upsert_frame(key, "F:/nowhere/RGB261/P_0.png", {"hw": [48, 64],
                                                           "hw_source": HW_MEASURED,
                                                           "cache_path": copy}, None)
        assert not has_no_photo(db.get_frame(key))
        images = ImageCache(db, str(cache))
        assert images.image_path(key) == copy
        assert images.thumb_path(key) == thumb
        assert images.get(key).shape == (48, 64, 3)
        assert frame_candidates(str(cache), key, db.get_frame(key))[0] == copy
        # a frame with no row at all is no missing frame: the cache is read as before
        other = FrameKey(61, 28, VIEW)
        copy28, _ = _stale_copy(cache, other, value=60)
        assert not has_no_photo(None)
        assert frame_candidates(str(cache), other, None)[0] == copy28
        assert images.image_path(other) == copy28
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# (b) confirming: refused, before anything is compiled
# --------------------------------------------------------------------------- #
@pytest.fixture
def scene(tmp_path):
    """The window scene with step LAST_STEP - 1 missing and a stale copy at its number."""
    db, paths, tax = A.make_db(tmp_path, missing=(A.LAST_STEP - 1,))
    key = FrameKey(A.DESKTOP, A.LAST_STEP - 1, A.VIEW)
    stale, _thumb = _stale_copy(Path(paths["cache_dir"]), key)
    db.upsert_frame(key, "", {"hw": [A.HW[0], A.HW[1]], "hw_source": HW_MEASURED,
                              "cache_path": stale}, None, {"missing": True})
    truth = TruthService(db, tax, cache_dir=paths["cache_dir"])
    yield db, truth, key
    db.close()


def test_verify_frame_refuses_a_missing_frame_in_these_words(scene):
    db, truth, key = scene
    before = db.get_frame(key)["aux"]
    with pytest.raises(VerifyRefused) as refused:
        truth.verify_frame(key, "tester")
    assert str(refused.value) == (f"frame {key.desktop}/{key.view}/step {key.step} cannot be "
                                  f"verified: {NO_PHOTO_TEXT}")
    assert refused.value.blockers == [no_photo()]
    assert refused.value.blockers[0].kind == NO_PHOTO
    assert refused.value.blockers[0].code == FRAME_MISSING
    # nothing written: no row, no flag, no op -- and nothing compiled, so the
    # row's aux was not even measured
    assert db.compiled(key) == {}
    assert db.get_frame(key)["review_status"] in (None, "unlabeled")
    assert not [op for op in db.ops(key.desktop, key.view) if op["kind"] == "verify_frame"]
    assert db.get_frame(key)["aux"] == before


def test_the_refusal_is_one_reason_in_either_mode_and_nothing_is_compiled(scene):
    db, _truth, key = scene

    def frame():
        raise AssertionError("a frame with no photo is not compiled")

    for thorough in (True, False):
        assert blocking_reasons(db, key, frame, thorough=thorough) == [no_photo()]
    assert is_blocking(FRAME_MISSING)
    # the window's words for it: Chinese, and no part to click
    assert explain_code(FRAME_MISSING) == NO_PHOTO_SENTENCE
    rows = pair_problems(["frame 13/scan/step 13 cannot be verified: ...", FRAME_MISSING])
    assert rows[1] == {"text": NO_PHOTO_SENTENCE, "code": FRAME_MISSING, "instance": ""}


def test_a_photographed_frame_confirms_exactly_as_before(scene):
    """(e) The neighbour of the missing step, every shape drawn: Space goes through."""
    db, truth, _key = scene
    session = A.AnnotationSession(db, truth.tax, truth, str(truth.cache_dir), "tester")
    session.open(A.DESKTOP, A.VIEW)
    try:
        A.seed_shapes(session, A.LAST_STEP)
        session.goto(A.LAST_STEP, force=True)
        good = FrameKey(A.DESKTOP, A.LAST_STEP, A.VIEW)
        truth.verify_frame(good, "tester")
        assert db.get_frame(good)["review_status"] == "verified"
        assert db.compiled(good)
        assert all(row["status"] == "verified" for row in db.compiled(good).values())
    finally:
        session.close(force=True)


# --------------------------------------------------------------------------- #
# cli check / the re-check drain: no row written, nothing reported
# --------------------------------------------------------------------------- #
def test_the_recheck_drain_retires_a_missing_frame_without_writing(scene):
    db, truth, key = scene
    # the live D61 state: a confirmed row on the missing step, its re-check owed
    mask = np.zeros(A.HW, dtype=bool)
    mask[8:40, 8:40] = True
    db.put_compiled(key, "chassis", masks.encode_rle(mask), 0.0, "visible", "in_chassis",
                    "verified", "ghost", verified_by="tester")
    db.add_rechecks(key.desktop, key.view, [key.step])
    rows_before = db.compiled(key)
    aux_before = db.get_frame(key)["aux"]
    stats = truth.run_pending_rechecks(key.desktop, key.view)
    assert stats["problems"] == [] and stats["updated"] == 0 and stats["conflicts"] == 0
    assert db.rechecks(key.desktop, key.view) == []
    assert db.compiled(key) == rows_before            # not one row written or dropped
    assert db.get_frame(key)["aux"] == aux_before     # nothing compiled, nothing measured


def test_cli_check_writes_no_row_on_a_missing_frame_and_does_not_report_it(tmp_path, capsys):
    from tda.cli import main

    db, paths, _tax = A.make_db(tmp_path, missing=(A.LAST_STEP - 1,))
    key = FrameKey(A.DESKTOP, A.LAST_STEP - 1, A.VIEW)
    try:
        # a chassis shape over the whole segment: compiling the missing step
        # would give it a row to write
        db.add_keyframe(ShapeKeyframe(
            id=None, instance="chassis", desktop=A.DESKTOP, view=A.VIEW, pose_segment=1,
            anchor_step=A.LAST_STEP, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(A.cell(0)))]))
        db.set_zorder(ZOrderRec(A.DESKTOP, A.VIEW, 1, [("chassis", "main")]))
        db.add_rechecks(key.desktop, key.view, [key.step])
        assert TruthService(db, _tax, cache_dir=None).compile(key).instances  # it would
        db.upsert_frame(key, "", {"hw": [A.HW[0], A.HW[1]]}, None, {"missing": True})
    finally:
        db.close()
    path = A.write_paths_yaml(tmp_path)
    main(["--paths", path, "check", "--desktop", str(A.DESKTOP), "--view", A.VIEW])
    out = capsys.readouterr().out
    assert "still queued" not in out, out
    db = Db(paths["db_path"])
    try:
        assert db.compiled(key) == {}
        assert db.get_frame(key)["aux"] == {"hw": [A.HW[0], A.HW[1]]}   # not even measured
        assert db.rechecks(key.desktop, key.view) == []
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# (c) renumbering: load-index clears the copy, compiling never measures it again
# --------------------------------------------------------------------------- #
def _photos(tmp_path: Path, n: int) -> dict[int, str]:
    """Scanner folders RGB11..RGBn1 on a fake source drive, one shot each."""
    return {i: _png(str(tmp_path / "src" / f"RGB{i}1" / "P_0.png"), 30 + 20 * i)
            for i in range(1, n + 1)}


def _index(photos: dict[int, str], where: dict[int, int], n_steps: int,
           missing=()) -> dict[int, DesktopIndex]:
    """``where`` maps a folder number to its step; ``missing`` are steps with none."""
    frames = {}
    for folder, step in where.items():
        key = FrameKey(DESKTOP, step, VIEW)
        frames[key] = FrameFile(key, photos[folder], {"burst": [photos[folder]]}, None,
                                f"RGB{folder}1")
    return {DESKTOP: DesktopIndex(DESKTOP, n_steps, frames,
                                  [FrameKey(DESKTOP, s, VIEW) for s in missing], [])}


def _copy_in(cache: Path, photos: dict[int, str], where: dict[int, int]) -> None:
    """What ``build-cache`` copies for an index: each step's photo under its number."""
    C.build_cache(_index(photos, where, max(where.values())), str(cache), views=(VIEW,))


def test_a_skipped_step_loses_its_measurement_and_is_never_measured_again(tmp_path):
    """The D61 sequence: measured as RGB31, declared a missed shot, RGB31 moves to 4."""
    photos = _photos(tmp_path, 4)
    cache = tmp_path / "cache"
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        before = {1: 1, 2: 2, 3: 3, 4: 4}
        P.load_index_into_db(db, _index(photos, before, 5, missing=(5,)))
        _copy_in(cache, photos, before)
        three = FrameKey(DESKTOP, 3, VIEW)
        assert frame_hw(db, three, str(cache)) == (48, 64)
        aux = db.get_frame(three)["aux"]
        assert aux["hw_source"] == HW_MEASURED and aux["cache_path"].endswith("s003.png")

        # U25 round 2: step 3 is a missed shot, every folder from RGB31 on moves one up
        after = {1: 1, 2: 2, 3: 4, 4: 5}
        counts = P.load_index_into_db(db, _index(photos, after, 5, missing=(3,)))
        assert not counts["failed"]
        row = db.get_frame(three)
        assert row["missing"] and not row["path"]
        assert "cache_path" not in row["aux"] and row["aux"].get("hw_source") != HW_MEASURED
        # the copy of RGB31 is still at s003.png -- and is nobody's picture now
        assert (cache / VIEW / f"D{DESKTOP}" / "s003.png").is_file()
        assert ImageCache(db, str(cache)).image_path(three) is None

        # visiting the step compiles it: the nominal size, never the copy's
        assert frame_hw(db, three, str(cache)) == tuple(NOMINAL)
        assert db.get_frame(three)["aux"] == {"hw": NOMINAL, "hw_source": HW_INFERRED}
        TruthService(db, load_taxonomy(), cache_dir=str(cache)).compile(three)
        assert db.get_frame(three)["aux"] == {"hw": NOMINAL, "hw_source": HW_INFERRED}
        assert ImageCache(db, str(cache)).image_path(three) is None

        # RGB31 is step 4 now: a new measurement, of its own copy once build-cache ran
        _copy_in(cache, photos, after)
        four = FrameKey(DESKTOP, 4, VIEW)
        assert db.get_frame(four)["path"] == photos[3]
        assert frame_hw(db, four, str(cache)) == (48, 64)
        assert db.get_frame(four)["aux"]["cache_path"].endswith("s004.png")
    finally:
        db.close()


def test_a_step_that_gets_its_photo_back_is_measured_again(tmp_path):
    photos = _photos(tmp_path, 4)
    cache = tmp_path / "cache"
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        three = FrameKey(DESKTOP, 3, VIEW)
        skipped = {1: 1, 2: 2, 3: 4, 4: 5}
        P.load_index_into_db(db, _index(photos, skipped, 5, missing=(3,)))
        assert frame_hw(db, three, str(cache)) == tuple(NOMINAL)     # inferred while missing
        back = {1: 1, 2: 2, 3: 3, 4: 4}
        assert not P.load_index_into_db(db, _index(photos, back, 5, missing=(5,)))["failed"]
        row = db.get_frame(three)
        assert not row["missing"] and row["path"] == photos[3]
        assert "hw_source" not in row["aux"]          # the inferred size went with the gap
        _copy_in(cache, photos, back)
        assert frame_hw(db, three, str(cache)) == (48, 64)
        aux = db.get_frame(three)["aux"]
        assert aux["hw_source"] == HW_MEASURED and aux["cache_path"].endswith("s003.png")
        assert ImageCache(db, str(cache)).image_path(three).endswith("s003.png")
    finally:
        db.close()


def test_a_leftover_measurement_on_a_missing_row_is_replaced_and_reloaded_away(tmp_path):
    """The live D61 row before any fix: missing, measured, ``cache_path`` to the copy."""
    photos = _photos(tmp_path, 4)
    cache = tmp_path / "cache"
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        index = _index(photos, {1: 1, 2: 2, 3: 4, 4: 5}, 5, missing=(3,))
        P.load_index_into_db(db, index)
        three = FrameKey(DESKTOP, 3, VIEW)
        stale, _thumb = _stale_copy(cache, three)
        leftover = {"hw": [48, 64], "hw_source": HW_MEASURED, "cache_path": stale}
        db.upsert_frame(three, None, leftover, None, {"missing": True})
        # a re-load clears it ...
        P.load_index_into_db(db, index)
        assert db.get_frame(three)["aux"] == {}
        # ... and so does the next compile, without a re-load
        db.upsert_frame(three, None, leftover, None, {"missing": True})
        assert frame_hw(db, three, str(cache)) == tuple(NOMINAL)
        assert db.get_frame(three)["aux"] == {"hw": NOMINAL, "hw_source": HW_INFERRED}
        assert db.get_frame(three)["missing"]
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# (d) the cache: never copied, reported, pruned on request
# --------------------------------------------------------------------------- #
def test_build_cache_never_copies_a_step_the_index_calls_missing(tmp_path):
    photos = _photos(tmp_path, 3)
    index = _index(photos, {1: 1, 2: 2, 3: 3}, 3)
    # an index that lists a step both ways is not to be trusted with that step
    index[DESKTOP].missing.append(FrameKey(DESKTOP, 2, VIEW))
    stats = C.build_cache(index, str(tmp_path / "cache"), views=(VIEW,))
    assert stats["steps"] == 2 and not stats["failures"]
    folder = tmp_path / "cache" / VIEW / f"D{DESKTOP}"
    assert sorted(p.name for p in folder.glob("s*.png")) == ["s001.png", "s003.png"]


def _stale_cache(tmp_path: Path) -> tuple[Path, dict, dict]:
    """The D61 cache: built when RGB31 was step 3, then step 3 became a missed shot."""
    photos = _photos(tmp_path, 4)
    cache = tmp_path / "cache"
    _copy_in(cache, photos, {1: 1, 2: 2, 3: 3, 4: 4})
    C.build_thumbs(str(cache), views=(VIEW,), auto_roi=False)
    after = _index(photos, {1: 1, 2: 2, 3: 4, 4: 5}, 5, missing=(3,))
    C.build_cache(after, str(cache), views=(VIEW,))
    return cache, photos, after


def test_missing_copies_finds_the_copy_the_thumbnail_and_the_manifest_record(tmp_path):
    cache, _photos_, index = _stale_cache(tmp_path)
    found = C.missing_copies(index, str(cache), views=(VIEW,))
    image = cache_path(str(cache), FrameKey(DESKTOP, 3, VIEW), "png")
    thumb = thumb_path(str(cache), FrameKey(DESKTOP, 3, VIEW))
    assert found == [{"desktop": DESKTOP, "step": 3, "view": VIEW,
                      "files": [image, thumb], "manifest": True}]
    # the database's missing rows count too, a desktop the index does not list as well
    extra = FrameKey(DESKTOP + 1, 2, VIEW)
    _stale_copy(cache, extra)
    found = C.missing_copies(index, str(cache), views=(VIEW,), db_missing=[extra])
    assert [(f["desktop"], f["step"]) for f in found] == [(DESKTOP, 3), (DESKTOP + 1, 2)]
    # a step the index has a photo of is not one, whatever the database says
    assert C.missing_copies(index, str(cache), views=(VIEW,),
                            db_missing=[FrameKey(DESKTOP, 4, VIEW)])[0]["step"] == 3


def test_prune_missing_deletes_them_and_leaves_every_other_step(tmp_path):
    cache, _photos_, index = _stale_cache(tmp_path)
    folder = cache / VIEW / f"D{DESKTOP}"
    keep = {p.name: p.read_bytes() for p in folder.glob("s*.png") if p.name != "s003.png"}
    stats = C.prune_missing(C.missing_copies(index, str(cache), views=(VIEW,)), str(cache))
    assert stats == {"files": 2, "records": 1, "failures": []}
    assert not (folder / "s003.png").exists()
    assert not Path(thumb_path(str(cache), FrameKey(DESKTOP, 3, VIEW))).exists()
    manifest = C._load_manifest(str(folder / C.MANIFEST_NAME))
    assert "3" not in manifest and {"1", "2", "4", "5"} <= set(manifest)
    assert {p.name: p.read_bytes() for p in folder.glob("s*.png")} == keep
    assert C.missing_copies(index, str(cache), views=(VIEW,)) == []
    # the thumbnails pass does not bring it back
    C.build_thumbs(str(cache), views=(VIEW,), auto_roi=False)
    assert not Path(thumb_path(str(cache), FrameKey(DESKTOP, 3, VIEW))).exists()


def test_the_cli_reports_on_every_run_and_deletes_only_when_asked(tmp_path, capsys):
    cache, _photos_, index = _stale_cache(tmp_path)
    index_path = str(tmp_path / "index.json")
    save_index(index, index_path)
    span = ["--first", str(DESKTOP), "--last", str(DESKTOP)]
    stale = cache / VIEW / f"D{DESKTOP}" / "s003.png"

    assert C.main(["--cache", str(cache), "--index", index_path, *span]) == 0
    out = capsys.readouterr().out
    assert "1 step(s) with no photo still have cached copies" in out, out
    assert f"stale D{DESKTOP} s003 scan: s003.png, thumbs/s003.jpg, manifest record" in out
    assert "--prune-missing deletes them" in out
    assert stale.is_file()

    assert C.main(["--cache", str(cache), "--index", index_path, "--prune-missing", *span]) == 0
    out = capsys.readouterr().out
    assert f"prune D{DESKTOP} s003 scan: s003.png, thumbs/s003.jpg, manifest record" in out
    assert "deleted 2 file(s) and 1 manifest record(s), failures=0" in out
    assert not stale.exists()

    assert C.main(["--cache", str(cache), "--index", index_path, *span]) == 0
    assert "no photo" not in capsys.readouterr().out


def test_the_prune_reads_the_databases_missing_rows_read_only(tmp_path, capsys):
    cache, _photos_, index = _stale_cache(tmp_path)
    index_path = str(tmp_path / "index.json")
    save_index(index, index_path)
    db_path = tmp_path / "tda.sqlite"
    db = Db(str(db_path))
    try:
        other = FrameKey(DESKTOP + 1, 2, VIEW)
        db.upsert_frame(other, None, {}, None, {"missing": True})
    finally:
        db.close()
    _stale_copy(cache, other)
    stamp = db_path.stat().st_mtime_ns
    assert C.main(["--cache", str(cache), "--index", index_path, "--db", str(db_path),
                   "--prune-missing", "--first", str(DESKTOP), "--last", str(DESKTOP + 1)]) == 0
    out = capsys.readouterr().out
    assert f"prune D{DESKTOP + 1} s002 scan" in out and f"prune D{DESKTOP} s003 scan" in out
    assert not Path(cache_path(str(cache), other, "png")).exists()
    assert db_path.stat().st_mtime_ns == stamp
