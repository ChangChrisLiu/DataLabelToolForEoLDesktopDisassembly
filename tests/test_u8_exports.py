"""Task U8: a state set at a frame reaches the exports, and only that frame.

The VLM scene's ``ram_latch.01`` is closed until the log opens it at step 7.
"Open at 4" is a correction with no action behind it (4 and 5 are ordinary
steps; 3 is a ``dupli`` one, which takes the state of 2). COCO's ``state``
attribute and V2's state question must both carry it at step 4 -- and the
log's own history everywhere else -- and the independent re-derivation in
:mod:`tests.vlm_checker` must agree with every record.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import vlm_scene as S
from tda.core import state_assert as SA
from tda.core.db import Db
from tda.core.export.coco import export_coco
from tda.core.export.vlm import export_vlm
from vlm_checker import Checker

STEP = 4


@pytest.fixture
def scene(tmp_path: Path):
    db = Db(str(tmp_path / "vlm.sqlite"))
    tax = S.build(db, views=(S.VIEW,))
    yield db, tax
    db.close()


def test_a_duplicate_frame_carries_the_state_into_v12(scene, tmp_path: Path):
    """Round 2 (B2): open at 2, dupli 3 shows the same frame, the restore is at 4.

    V12 exports the ``dupli`` step 3 as "nothing changed" -- true again, since
    3 takes 2's state -- and the independent checker agrees with every record.
    """
    db, tax = scene
    types = {r.step: r.step_type for r in db.steps(S.DESKTOP)}
    assert types[2] == types[STEP] == "normal" and types[3] == "dupli"
    with pytest.raises(SA.StateAssertError, match="和第 2 步是同一个样子"):
        SA.assert_state(db, tax, S.DESKTOP, S.LATCH, 3, "open")
    patch = SA.assert_state(db, tax, S.DESKTOP, S.LATCH, 2, "open")
    assert patch.inherits == [3] and patch.next_step == STEP
    assert [(e.step, e.old, e.new) for e in patch.after] == [
        (2, "closed", "open"), (STEP, "open", "closed")]
    out = tmp_path / "v.jsonl"
    export_vlm(db, tax, [S.DESKTOP], S.VIEW, str(out))
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
               if line]
    v12 = [{**r["prompt"], **r["label"]} for r in records if r["prompt"]["task"] == "V12"]
    assert any(r["step"] == 3 and r["answer"]["changed"] is False for r in v12)
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)


def test_coco_and_vlm_carry_the_corrected_state(scene, tmp_path: Path):
    db, tax = scene
    patch = SA.assert_state(db, tax, S.DESKTOP, S.LATCH, STEP, "open",
                            evidence_view=S.VIEW)
    assert [(e.step, e.old, e.new) for e in patch.after] == [
        (STEP, "closed", "open"), (STEP + 1, "open", "closed")]

    out = tmp_path / "v.jsonl"
    export_vlm(db, tax, [S.DESKTOP], S.VIEW, str(out))
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
               if line]
    asked = {}
    for record in records:
        flat = {**record["prompt"], **record["label"]}
        if flat["task"] == "V2" and flat["answer_check"].get("instance") == S.LATCH:
            asked[flat["step"]] = flat["answer"]["state"]
    assert asked[STEP] == "open"
    assert asked[1] == "closed" and asked[STEP + 1] == "closed"
    assert asked[S.LAST_STEP] == "open"                  # the log's own open at 7
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)

    doc = export_coco(db, tax, [S.DESKTOP], S.VIEW, str(tmp_path / "c.json"),
                      only_verified=False)
    step_of = {img["id"]: img["extra"]["step"] for img in doc["images"]}
    states = {step_of[a["image_id"]]: a["attributes"]["state"]
              for a in doc["annotations"]
              if a["attributes"]["instance_key"] == S.LATCH}
    assert states[1] == "closed" and states[STEP] == "open"
    assert states[STEP + 1] == "closed" and states[S.LAST_STEP] == "open"
