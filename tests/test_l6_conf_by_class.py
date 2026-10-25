"""L6 round 3: a class's own confidence floor (``conf_by_class`` in the detector file).

L6's v3 detector finds more screws than v2 and also more low-score boxes on
cable bundles, which compete in the dE ranking; the controller ships it with
screws at 0.15 and every other class at ``conf`` (0.05)
(``experiments/l6_models/REPORT.md``, rounds 2-3).  The floor is a reading of
the detections, never a property of what is stored:

* the file: ``conf_by_class`` maps armed classes to floors at or above
  ``merge_floor``; anything else is the usual WARNING and a detector that is
  off; it is not part of the cache's recipe;
* the engine: the candidates (and their dE) honour the floor, the other
  classes are untouched, the cache still holds the box under the floor -- so a
  file without the floor is served the same cache and gets it back;
* the manifest: one WARNING when its ``conf_by_class`` and the file's differ;
* the window: rank 1 of the guess, and the 「拆前 / 拆后」 chips of U28.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import COOLER, close_window
from test_app_u3 import A_BOX, C_BOX, SCREW_STEP, _window, det, fbox, go, pump
from test_detector import CONN, StubModel, good_yaml, item, write_frame
from test_detector import FR, _manifest, _rf_config, _rf_ready
from test_detector import guard_process, guard_rfdetr  # noqa: F401 (fixtures)
from test_u23_locator import STEP, goto, make_scene, open_window, select
from test_u28_round3 import GONE, SCREW
from test_u28_round3 import StubModel as LocatorStub
from tda.models import detector as DM
from tda.models.det_engine import DetectionEngine
from tda.models.detector import Det, DetectorConfig

FLOORS = (("screw", 0.15),)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def frames(tmp_path):
    j = write_frame(tmp_path / "img" / "s012.png", square_at=10)
    k = write_frame(tmp_path / "img" / "s013.png", square_at=30)
    return j, k


@pytest.fixture
def warnings_logged():
    records: list = []

    class Keep(logging.Handler):
        def emit(self, record):
            records.append(record)

    keep = Keep(logging.DEBUG)
    logger = logging.getLogger("tda.models.detector")
    logger.addHandler(keep)
    yield records
    logger.removeHandler(keep)


def cfg(tmp_path: Path, classes=("screw", "connector"), conf=0.05, floors=FLOORS,
        **over) -> DetectorConfig:
    base = dict(source=tmp_path / "detector.yaml", model=tmp_path / "stub.pt",
                classes=tuple(classes), views=("scan",), conf=conf, roi_crop=True,
                work_scale=(("scan", 1.0),), tile=640, stride=512, yield_ms=0.0,
                cache_root=tmp_path / "det", merge_floor=min(conf, 0.05),
                conf_by_class=tuple(floors))
    base.update(over)
    return DetectorConfig(**base)


# --------------------------------------------------------------------------- #
# the file
# --------------------------------------------------------------------------- #
def test_the_file_reads_conf_by_class_and_conf_for_falls_back_to_conf(tmp_path):
    got = DM.load_detector_config(str(good_yaml(
        tmp_path, classes=["screw", "connector"], conf=0.05, merge_floor=0.05,
        conf_by_class={"screw": 0.15})))
    assert got is not None and got.conf_by_class == (("screw", 0.15),)
    assert got.conf_for("screw") == pytest.approx(0.15)
    assert got.conf_for("connector") == pytest.approx(0.05)
    assert got.conf_for("cpu") == pytest.approx(0.05)          # not armed: still conf
    none = DM.load_detector_config(str(good_yaml(tmp_path, conf=0.05)))
    assert none.conf_by_class == () and none.conf_for("screw") == pytest.approx(0.05)


def test_the_floor_is_not_part_of_the_caches_recipe(tmp_path):
    """A new floor reads the same cache: it is applied to what is read."""
    assert cfg(tmp_path).recipe() == cfg(tmp_path, floors=()).recipe()
    assert cfg(tmp_path).store_conf == cfg(tmp_path, floors=()).store_conf


@pytest.mark.parametrize("bad", [
    {"conf_by_class": ["screw", 0.15]},                 # not a mapping
    {"conf_by_class": {"bolt": 0.15}},                  # not an armed class
    {"conf_by_class": {"screw": 0.01}},                 # under merge_floor
    {"conf_by_class": {"screw": 1.5}},                  # above one
    {"conf_by_class": {"screw": "high"}},
    {"conf_by_class": {"screw": True}},
], ids=["list", "unknown class", "under the merge floor", "above one", "a word", "a bool"])
def test_a_malformed_conf_by_class_is_a_warning_and_off(tmp_path, warnings_logged, bad):
    path = good_yaml(tmp_path, conf=0.05, merge_floor=0.05, **bad)
    assert DM.load_detector_config(str(path)) is None
    assert [r.levelname for r in warnings_logged] == ["WARNING"]
    assert "conf_by_class" in warnings_logged[0].getMessage()


# --------------------------------------------------------------------------- #
# the engine: candidates filtered, the cache untouched
# --------------------------------------------------------------------------- #
LOW_SCREW = Det((10.0, 20.0, 20.0, 32.0), "screw", 0.10)       # the square: it moves
GOOD_SCREW = Det((40.0, 40.0, 50.0, 50.0), "screw", 0.80)
LOW_CONN = Det((30.0, 40.0, 38.0, 48.0), "connector", 0.06)    # under 0.15, over conf


def _engine(tmp_path, model, **over) -> DetectionEngine:
    eng = DetectionEngine(cfg(tmp_path, **over), tmp_path / "det", factory=lambda _c: model)
    eng.load()
    eng.open_view(13, "scan")
    return eng


def test_the_floor_filters_one_class_and_the_cache_keeps_its_boxes(tmp_path, frames):
    model = StubModel({12: [LOW_SCREW, GOOD_SCREW, LOW_CONN]})
    eng = _engine(tmp_path, model)
    first = eng.process(item(frames))
    # stored: every box above the store floor, the 0.10 screw included
    assert [d.conf for d in first.dets] == [0.10, 0.80, 0.06]
    # candidates: the screw under its floor is out, the 0.06 connector stays
    assert eng.candidates_of(first.dets) == [1, 2]
    assert sorted(first.change) == [1, 2]
    path = eng.flush()
    stored = json.loads(path.read_text(encoding="utf-8"))["steps"]["12"]["dets"]
    assert [round(row[5], 2) for row in stored] == [0.10, 0.80, 0.06]
    # the same cache without the floor: no model call, the 0.10 screw is back
    again = StubModel({})
    eng2 = _engine(tmp_path, again, floors=())
    second = eng2.process(item(frames))
    assert again.calls == [] and not second.detected
    assert second.dets == first.dets
    assert sorted(second.change) == [0, 1, 2]
    # ... and served from the cache alone, the floor applies again
    eng3 = DetectionEngine(cfg(tmp_path), tmp_path / "det", factory=lambda _c: again,
                           identity=model.identity)
    eng3.prepare()
    eng3.open_view(13, "scan")
    third = eng3.answer_from_cache(item(frames))
    assert third is not None and sorted(third.change) == [1, 2]


def test_other_classes_are_untouched_by_a_screws_floor(tmp_path, frames):
    model = StubModel({12: [CONN, LOW_CONN, GOOD_SCREW]})
    eng = _engine(tmp_path, model)
    got = eng.process(item(frames))
    assert eng.candidates_of(got.dets) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #
def test_the_manifests_floors_and_the_files_agree_or_one_warning(tmp_path, warnings_logged):
    manifest = {"conf_by_class": {"screw": 0.15}, "__path__": str(tmp_path / "v3.json")}
    assert DM.check_manifest_floors(manifest, cfg(tmp_path)) == ""
    assert warnings_logged == []
    # a file without the floor, or with another one: one WARNING naming both; on
    text = DM.check_manifest_floors(manifest, cfg(tmp_path, floors=()))
    assert "conf_by_class differs from the manifest v3.json for ['screw']" in text
    assert "the configuration says none" in text and "{'screw': 0.15}" in text
    other = DM.check_manifest_floors(manifest, cfg(tmp_path, floors=(("screw", 0.25),)))
    assert "{'screw': 0.25}" in other and "{'screw': 0.15}" in other
    assert [r.levelname for r in warnings_logged] == ["WARNING", "WARNING"]
    # a class the file does not arm is not compared
    assert DM.check_manifest_floors(manifest, cfg(tmp_path, classes=("connector",),
                                                  floors=())) == ""
    # no manifest, or one without the key (v1, v2): nothing to compare
    assert DM.check_manifest_floors(None, cfg(tmp_path)) == ""
    assert DM.check_manifest_floors({"classes": ["screw"]}, cfg(tmp_path, floors=())) == ""
    # an unreadable one is said, and the file's floors are used
    assert "cannot be read" in DM.check_manifest_floors(
        {"conf_by_class": "screw", "__path__": "m.json"}, cfg(tmp_path))


@pytest.mark.parametrize("floors, warned", [(FLOORS, False), ((), True)],
                         ids=["the file agrees", "the file has no floor"])
def test_the_model_loads_either_way_and_a_disagreeing_floor_is_one_warning(
        tmp_path, monkeypatch, guard_rfdetr, warnings_logged, floors, warned):
    """The check runs where the model is built (RF-DETR over the fake rfdetr of U4)."""
    _rf_ready(monkeypatch)
    FR.install(monkeypatch)
    config = _rf_config(tmp_path, conf_by_class=tuple(floors))
    _manifest(tmp_path, conf_by_class={"screw": 0.15})
    det = DM.RFDetrTileDetector(config)
    assert "manifest rf.json agrees" in det.describe()          # the file and its classes
    said = [r.getMessage() for r in warnings_logged
            if r.levelname == "WARNING" and "conf_by_class differs" in r.getMessage()]
    assert len(said) == int(warned)


def test_the_v3_manifest_carries_the_ruled_floor():
    path = Path("D:/DataSet/models/weights/tda_det_rfdetr_small_all_v3.json")
    if not path.is_file():
        pytest.skip(f"{path} is not on this machine")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["conf_by_class"] == {"screw": 0.15}
    assert manifest["inference"]["conf"] == 0.05 and manifest["inference"]["merge_floor"] == 0.05


# --------------------------------------------------------------------------- #
# the window: rank 1 of the guess
# --------------------------------------------------------------------------- #
def _enable(win, tmp_path, table, floors):
    model = StubModel(table, identity="l6-floor-stub")
    change = {A_BOX: 9.0, C_BOX: 2.0}
    config = DetectorConfig(source=tmp_path / "detector.yaml", model=tmp_path / "stub.pt",
                            classes=("screw",), views=("scan",), conf=0.05, roi_crop=True,
                            work_scale=(("scan", 1.0),), tile=640, stride=512, yield_ms=0.0,
                            cache_root=tmp_path / "det", merge_floor=0.05,
                            conf_by_class=tuple(floors))
    win.enable_detector(config, factory=lambda _c: model, cache_root=str(tmp_path / "det"),
                        change_fn=lambda j, k, box: float(
                            change.get(tuple(int(v) for v in box), 0.5)))
    return model


@pytest.mark.parametrize("floors, rank_one", [((), A_BOX), (FLOORS, C_BOX)],
                         ids=["without the floor", "screws at 0.15"])
def test_a_low_score_screw_that_changes_most_is_rank_one_only_without_the_floor(
        qapp, tmp_path, floors, rank_one):
    """The fixture of round 2's D19 failure: a 0.10 screw box with the largest dE."""
    win = _window(tmp_path)
    try:
        _enable(win, tmp_path, {SCREW_STEP: [det(A_BOX, 0.10), det(C_BOX, 0.5)]}, floors)
        go(win, SCREW_STEP)
        pump(win)
        ranking = win.detector_ranking()
        assert ranking is not None and ranking.chosen[0].box == fbox(rank_one)
        assert win._prompt_box == fbox(rank_one)
        if floors:
            assert [c.box for c in ranking.chosen] == [fbox(C_BOX)]
            # the box under the floor is still in the frame's answer and on disk
            frame = win._det_frames[SCREW_STEP]
            assert any(d.conf == pytest.approx(0.10) for d in frame.dets)
            stored = [json.loads(p.read_text(encoding="utf-8"))
                      for p in (tmp_path / "det").rglob("*.json")]
            assert any(round(row[5], 2) == 0.10 for body in stored
                       for row in body["steps"][str(SCREW_STEP)]["dets"])
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# U28: the 「拆前 / 拆后」 chips
# --------------------------------------------------------------------------- #
@pytest.fixture
def locator(qapp, tmp_path: Path):
    session, paths = make_scene(tmp_path, own_skip=(COOLER,))
    session.open(13, "scan", force=True)
    win = open_window(session, paths)
    goto(win, STEP)
    yield win
    close_window(win)


def _arm_locator(win, tmp_path, table, floors):
    stub = LocatorStub(table)
    win.enable_detector(cfg(tmp_path, classes=("screw",), floors=floors),
                        factory=lambda _c: stub, cache_root=str(tmp_path / "det"))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and win.det_worker.pending():
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    return stub


def _settle(win) -> None:
    detector = win._locator_det
    if detector is not None:
        assert detector.wait(30.0)
    for _ in range(3):
        QApplication.processEvents()


@pytest.mark.parametrize("conf, offered", [(0.10, False), (0.20, True)],
                         ids=["under the screw floor", "over it"])
def test_a_vanished_screw_is_a_chip_only_at_its_floor(locator, tmp_path, conf, offered):
    _arm_locator(locator, tmp_path, {12: [Det(GONE, "screw", conf)], 13: []}, FLOORS)
    select(locator, SCREW)
    _settle(locator)
    assert locator._locator_det.conf_for("screw") == pytest.approx(0.15)
    assert locator.locator_det_requests == 1
    assert (GONE in locator.locator.pair().candidates) is offered
