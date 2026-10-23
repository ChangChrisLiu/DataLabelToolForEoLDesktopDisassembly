"""Task U20: what the annotator sees of a drive cage's implied latch.

* the task card: on the frame before the cage comes out the latch is drawn back
  with the cage (「跟 drive_cage.01 一起装回来的」) and says it is open there by
  default; the start frame does not ask for it;
* Steps: 「隐含」 in the Raw names cell, one S1 question;
* ``status`` names it; COCO marks it ``implied: true``.

The scene is D5's sheet with the locker taken out at 10 (``Shift+T``) and a
forced re-import -- D6's state of affairs.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt

from tda import pipeline as P
from tda.cli import format_desktop
from tda.core.db import Db
from tda.core.export import export_coco
from tda.core.implied import cage_latches
from tda.core.masks import encode_rle
from tda.core.model import ActionRec, FrameKey, InstanceRec, ShapeKeyframe, ShapePart, StepRec
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.pipeline_logs import import_logs_into_db
from tda.ui import session_api as api
from tda.ui.panels.steptable_models import INSTANCE_COLUMNS, InstanceTableModel
from tda.ui.panels.taskcard import CAGE_LATCH_HINT, cage_latch_hint, row_view
from tda.ui.session_tasks import task_card_for
from tda.ui.session_verb import commit_verb_fix
from tda.ui.steps_issues import orphan_issues, unresolved_issues
from tda.ui.steps_model import StepTableData

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
D5 = 5
CAGE = "drive_cage.01"
LOCKER = "drive_latch.01"
LATCH = "drive_latch.02"
NOTE = "drive_cage.01 自带的卡扣（日志里没有）"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def d6(tmp_path, tax):
    folder = tmp_path / "drive"
    folder.mkdir()
    for suffix in ("", "_meta"):
        shutil.copy(FIXTURES / f"desktop_05{suffix}.csv", folder / f"desktop_05{suffix}.csv")
    db = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(db, str(folder), tax, desktops={D5})
    commit_verb_fix(db, TruthService(db, tax), FrameKey(D5, 12, "scan"), LOCKER, "remove",
                    "tester")
    import_logs_into_db(db, str(folder), tax, desktops={D5}, force=True)
    assert LATCH in db.instances(D5)
    yield db
    db.close()


def card(db, tax, step, neighbour):
    return {row["instance"]: row for row in task_card_for(db, tax, D5, "scan", step, neighbour)}


# --------------------------------------------------------------------------- #
# the task card
# --------------------------------------------------------------------------- #
def test_the_hint_reads_the_span_key_off_the_key_map():
    assert cage_latch_hint() == "机架卡扣：默认这里是打开的；更早就开着的话到那一帧 Ctrl+T"
    assert CAGE_LATCH_HINT.format(key="Ctrl+T") == cage_latch_hint()


def test_frame_13_draws_the_latch_back_in_with_the_cage(d6, tax):
    rows = task_card_for(d6, tax, D5, "scan", 13, 14)
    keys = [row["instance"] for row in rows]
    assert keys.index(CAGE) + 1 == keys.index(LATCH)          # grouped under the cage
    latch = rows[keys.index(LATCH)]
    assert latch["kind"] == api.KIND_ADD_SHAPE and latch["parent"] == CAGE
    assert latch["transition"] == ["removed", "open"]
    view = row_view(latch)
    assert view["sentence"] == ("在这一帧画出它的完整形状（跟 drive_cage.01 一起装回来的）"
                                f"（{cage_latch_hint()}）")
    assert view["raw"] == f"隐含：{NOTE}"
    assert "机架卡扣" not in row_view(rows[keys.index(CAGE)])["sentence"]


def test_frame_12_splits_it_to_closed_without_the_hint(d6, tax):
    latch = card(d6, tax, 12, 13)[LATCH]
    assert latch["kind"] == api.KIND_SPLIT_KEYFRAME
    assert latch["transition"] == ["open", "closed"]
    assert "机架卡扣" not in row_view(latch)["sentence"]


def test_the_start_frame_does_not_ask_for_it(d6, tax):
    last = max(s.step for s in d6.steps(D5))
    rows = task_card_for(d6, tax, D5, "scan", last, None)
    assert LATCH not in {row["instance"] for row in rows}
    assert CAGE not in {row["instance"] for row in rows}


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def test_steps_shows_it_as_implied_and_asks_once(d6, tax):
    data = StepTableData.load(d6, D5, tax)
    asked = [line for line in unresolved_issues(data.instances, tax, data.declined)
             if LATCH in line]
    assert asked == [
        "implied instance drive_latch.02: the own latch of drive_cage.01 (机架自带卡扣, "
        "日志里没有) - closed, open on the frame before the cage comes out, then out with "
        "it; keep it, or delete it in the Instances tab if this cage has no latch; deleting "
        "it is remembered, so no re-import brings it back"]
    assert sum(LATCH in line for line in data.issues) == 1
    assert not [line for line in orphan_issues(data.instances, d6.actions(D5)) if LATCH in line]
    model = InstanceTableModel(data)
    row = model.keys.index(LATCH)
    raw = [c.title for c in INSTANCE_COLUMNS].index("Raw names")
    assert model.index(row, raw).data() == f"隐含：{NOTE} / implied"
    tip = model.index(row, 0).data(Qt.ItemDataRole.ToolTipRole)
    assert "隐含" in tip and NOTE in tip and "右键删除" in tip
    # the sheet's own latch keeps its name
    assert model.index(model.keys.index(LOCKER), raw).data() == "HDD shield locker"


def test_status_names_it(d6):
    [row] = P.status_rows(d6, {D5})
    text = format_desktop(d6, row)
    assert "  implied (not in the log): 1 - drive_latch.02 (drive_cage.01's own latch)" in text


# --------------------------------------------------------------------------- #
# COCO
# --------------------------------------------------------------------------- #
HW = (64, 64)


def _rect(y0, y1, x0, x1):
    mask = np.zeros(HW, bool)
    mask[y0:y1, x0:x1] = True
    return encode_rle(mask)


def test_coco_marks_the_latch_implied_with_its_state(tmp_path, tax):
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        steps = [StepRec(7, 1, "initial", "Initial Conditions"),
                 StepRec(7, 2, "normal", "HDD (ssd)"),
                 StepRec(7, 3, "normal", "HDD shield")]
        actions = [ActionRec(7, 3, 0, CAGE, "remove")]
        instances = {CAGE: InstanceRec(CAGE, 7, "drive_cage", attrs={"of": "hdd"})}
        [latch] = cage_latches(instances, actions, tax).created
        db.replace_steps(7, steps, actions)
        for rec in (instances[CAGE], latch):
            db.upsert_instance(rec)
        for step in (1, 2, 3):
            db.upsert_frame(FrameKey(7, step, "scan"), f"F:/scan/{step}.png",
                            {"hw": list(HW)}, None)
        assert latch.key == LOCKER          # the desktop has no other drive latch
        for key, rle in ((CAGE, _rect(10, 40, 10, 40)), (latch.key, _rect(44, 50, 44, 52))):
            db.add_keyframe(ShapeKeyframe(
                id=None, instance=key, desktop=7, view="scan", pose_segment=1, anchor_step=2,
                placement="in_chassis", geom_type="mask", parts=[ShapePart("main", rle=rle)]))
        truth = TruthService(db, tax)
        for step in (1, 2, 3):
            truth.refresh(FrameKey(7, step, "scan"))
        out = tmp_path / "coco.json"
        export_coco(db, tax, [7], "scan", str(out), only_verified=False)
        rows = json.loads(out.read_text(encoding="utf-8"))["annotations"]
        mine = sorted((a["image_id"], a["attributes"]["state"], a["attributes"]["implied"])
                      for a in rows if a["attributes"]["instance_key"] == latch.key)
        assert [(state, implied) for _img, state, implied in mine] == [
            ("closed", True), ("open", True)]
        assert all(not a["attributes"]["implied"] for a in rows
                   if a["attributes"]["instance_key"] == CAGE)
    finally:
        db.close()
