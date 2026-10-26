"""Task U33: the app never asks for the loose cable D36 step 2 takes out.

The step targets the virtual node ``cable:isolated.01`` (``remove``), which has
no instance row and no geometry. Pinned here, on the real D36 sheet:

* **the task card** of every frame around it -- and the start card -- has no
  row for it (before U33 the start card asked for ``cable_clip.01``);
* **the Steps table** shows it as a step with no instance, like a reorient:
  the target cell names the node, the verb is ``remove`` (its combo offers the
  node's verbs), no issue, and the instance table has no row for it;
* **``cli check``** has nothing to say about it;
* **the exports**: COCO has nothing for it; the VLM tasks name it only as a
  step of the log (as they already did a released ``cable:*`` loom), and the
  independent checker agrees with every record.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import numpy as np
import pytest

from tda.core import masks
from tda.core.db import Db
from tda.core.model import ActionRec, FrameKey, ShapeKeyframe, ShapePart, ZOrderRec
from tda.core.taxonomy import load_taxonomy
from tda.pipeline_logs import import_logs_into_db
from tda.ui import session_api as api
from tda.ui.session_tasks import task_card_for
from tda.ui.steps_model import StepTableData

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
D36 = 36
NODE = "cable:isolated.01"
VIEW = "scan"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def db(tmp_path, tax):
    conn = Db(str(tmp_path / "d36.sqlite"))
    [run] = import_logs_into_db(conn, str(FIXTURES), tax, desktops={D36}).runs
    assert run.status == "imported"
    yield conn
    conn.close()


def _rows(card: list[dict]) -> list[str]:
    return [item["instance"] for item in card]


# --------------------------------------------------------------------------- #
# the task card
# --------------------------------------------------------------------------- #
def test_no_card_ever_asks_for_the_cable(db, tax):
    last = max(rec.step for rec in db.steps(D36))
    for step in range(1, last + 1):
        neighbour = step + 1 if step < last else None
        card = task_card_for(db, tax, D36, VIEW, step, neighbour)
        assert not [r for r in _rows(card) if r.startswith("cable")], (step, card)


def test_the_frame_before_the_cable_step_has_nothing_to_draw(db, tax):
    """Frame 1 against frame 2: the only change is the cable, so: confirm."""
    [item] = task_card_for(db, tax, D36, VIEW, 1, 2)
    assert item["kind"] == api.KIND_CONFIRM
    assert item["instance"] == "step 1"


def test_before_u33_the_start_card_asked_for_the_clip(tmp_path, tax):
    """The complaint, reproduced: the same sheet with the old step-2 target."""
    from test_u33_reimport import _as_before_u33

    conn = Db(str(tmp_path / "old.sqlite"))
    try:
        import_logs_into_db(conn, str(FIXTURES), tax, desktops={D36})
        last = max(rec.step for rec in conn.steps(D36))
        assert "cable_clip.01" not in _rows(task_card_for(conn, tax, D36, VIEW, last))
        _as_before_u33(conn, tax, asserted=False)
        assert "cable_clip.01" in _rows(task_card_for(conn, tax, D36, VIEW, last))
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# the Steps table
# --------------------------------------------------------------------------- #
def test_the_steps_data_shows_a_step_without_an_instance(db, tax):
    data = StepTableData.load(db, D36, tax)
    row = data.row(2)
    assert row.raw_name == "Remove 1 isolated cable from motherboard"
    assert row.step_type == "normal"
    assert [(a.target, a.verb) for a in row.actions] == [(NODE, "remove")]
    assert data.class_of(NODE) == "cable"
    assert NODE not in data.instances
    assert not row.issues
    assert not [issue for issue in data.issues if "cable" in issue]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def test_the_steps_table_draws_it_like_a_reorient(qapp, db, tax, tmp_path):
    from PySide6.QtCore import Qt

    from tda.ui.panels.steptable import InstanceTableModel, StepTableModel

    data = StepTableData.load(db, D36, tax)
    steps = StepTableModel(data, tmp_path / "cache")
    view_row = steps.first_row_of(2)
    titles = [c.title for c in steps.columns]

    def cell(title: str):
        return steps.data(steps.index(view_row, titles.index(title)), Qt.DisplayRole)

    assert cell("Raw name") == "Remove 1 isolated cable from motherboard"
    assert (cell("Type"), cell("Target"), cell("Verb")) == ("normal", NODE, "remove")
    assert cell("Issues") == ""
    verb = steps.index(view_row, titles.index("Verb"))
    assert steps.choices_for(verb) == ["release", "remove"]
    instances = InstanceTableModel(data)
    keys = [instances.key_at(r) for r in range(instances.rowCount())]
    assert NODE not in keys and not [k for k in keys if k and k.startswith("cable")]


# --------------------------------------------------------------------------- #
# cli check
# --------------------------------------------------------------------------- #
def test_check_has_nothing_to_say_about_it(tmp_path, tax, capsys):
    """D36 up to the cable step, the chassis drawn: zero problems, no cable."""
    from app_scene import write_paths_yaml
    from tda.cli import EXIT_OK, main

    paths = write_paths_yaml(tmp_path)
    path = tmp_path / "annotations" / "tda.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = Db(str(path))
    try:
        import_logs_into_db(conn, str(FIXTURES), tax, desktops={D36})
        steps = [s for s in conn.steps(D36) if s.step <= 2]
        conn.replace_steps(D36, steps, [a for a in conn.actions(D36) if a.step <= 2])
        for key in list(conn.instances(D36)):
            if key != "chassis":
                conn.delete_instance(D36, key)
        conn.replace_events(D36, [], auto_only=False)
        mask = np.zeros((64, 64), dtype=bool)
        mask[4:60, 4:60] = True
        for step in (1, 2):
            conn.upsert_frame(FrameKey(D36, step, VIEW), "", {"hw": [64, 64]}, None)
        conn.set_pose_segment(D36, VIEW, 1, 1, 2, 2, None, None)
        conn.add_keyframe(ShapeKeyframe(
            id=None, instance="chassis", desktop=D36, view=VIEW, pose_segment=1,
            anchor_step=2, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(mask))]))
        conn.set_zorder(ZOrderRec(D36, VIEW, 1, [("chassis", "main")]))
        assert [(a.step, a.target) for a in conn.actions(D36)] == [(2, NODE)]
    finally:
        conn.close()
    code = main(["--paths", paths, "check", "--desktop", str(D36), "--view", VIEW])
    out = capsys.readouterr().out
    assert "problems: 0" in out, out
    assert "cable" not in out.lower(), out
    assert code == EXIT_OK


# --------------------------------------------------------------------------- #
# the exports
# --------------------------------------------------------------------------- #
LOOSE = ActionRec(0, 4, 1, NODE, "remove", tool="hand")


@pytest.fixture
def scene(tmp_path, monkeypatch):
    """The VLM scene with a loose cable taken out at step 4, after the plug."""
    import vlm_scene as S

    actions, steps = S._actions, S._steps

    def with_cable():
        return [*actions(), ActionRec(S.DESKTOP, LOOSE.step, LOOSE.idx, LOOSE.target,
                                      LOOSE.verb, tool=LOOSE.tool)]

    def compound():
        out = steps()
        for rec in out:
            if rec.step == LOOSE.step:
                rec.step_type = "compound"
        return out

    monkeypatch.setattr(S, "_actions", with_cable)
    monkeypatch.setattr(S, "_steps", compound)
    conn = Db(str(tmp_path / "vlm.sqlite"))
    tax = S.build(conn)
    yield S, conn, tax
    conn.close()


def test_coco_has_nothing_for_it(scene, tmp_path):
    from tda.core.export.coco import export_coco

    S, conn, tax = scene
    assert NODE in {a.target for a in conn.actions(S.DESKTOP)}
    out = tmp_path / "coco.json"
    export_coco(conn, tax, [S.DESKTOP], S.VIEW, str(out), only_verified=False)
    text = out.read_text(encoding="utf-8")
    assert "cable:" not in text
    doc = json.loads(text)
    assert doc["annotations"]
    assert "cable" not in {c["name"] for c in doc["categories"]}


def test_the_vlm_tasks_take_it_as_a_step_and_the_checker_agrees(scene, tmp_path):
    from vlm_checker import Checker

    from tda.core.export.vlm import export_vlm

    S, conn, tax = scene
    out = tmp_path / "vlm.jsonl"
    export_vlm(conn, tax, [S.DESKTOP], S.VIEW, str(out))
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
               if line]
    assert records
    assert Checker(conn, tax, S.DESKTOP).check_all(records) == len(records)
    # it never stands as a part: no record points at it in a picture
    for record in records:
        evidence = record["label"].get("evidence") or {}
        assert NODE not in (evidence.get("instances") or [])
        assert NODE not in (evidence.get("bboxes") or {})
    # ... while the history and the plans past step 4 carry it where the log put it
    done = [r for r in records if r["prompt"]["task"] == "V10"
            and r["label"]["step"] >= LOOSE.step]
    assert done and all({"verb": "remove", "target": NODE} in r["label"]["answer"]["done"]
                        for r in done)
    plans = [r for r in records if r["prompt"]["task"] == "V16"
             and r["label"]["answer"]["valid"] and r["label"]["step"] < LOOSE.step]
    assert plans and all(
        ("remove", NODE) in [(i["verb"], i["target"])
                             for i in r["label"]["answer_check"]["plan"]] for r in plans)
