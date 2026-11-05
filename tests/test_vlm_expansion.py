"""The expansion round (owner 2026-11-04 late: "make many more questions, but none may be wrong").

V12 over every pair and across 2-3 steps, V3's multi-step form, V4 over every part, V7 over
every major part, several V16 plans per moment, plain names in every answer (no ordinal,
graded 1.0), the checker's gallery-only fix, and the correctness invariants on a scene.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import vlm_scene as S
import w3b_scene as W
from tda.core.db import Db
from tda.core.export import vlm_tasks as T
from tda.core.export.vlm import export_vlm, oracle_path
from tda.core.export.vlm_check import World, gold_reply, grade
from tda.core.model import InstanceRec
from vlm_checker import Checker

KEY = re.compile(r"\b[a-z_]+(?:\.[a-z_0-9]+)*\.\d{2}\b")


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def _world(out: Path) -> World:
    for line in oracle_path(str(out)).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "desktop":
            return World.from_oracle(row)
    raise AssertionError("no desktop line")


@pytest.fixture
def scene(tmp_path: Path):
    db = Db(str(tmp_path / "s.sqlite"))
    tax = S.build(db)
    yield db, tax, tmp_path
    db.close()


@pytest.fixture
def cpu_scene(tmp_path: Path):
    db = Db(str(tmp_path / "w.sqlite"))
    tax = W.build(db)
    yield db, tax, tmp_path
    db.close()


def _export(db, tax, tmp, desktop, tasks, name="v"):
    out = tmp / f"{name}.jsonl"
    summary = export_vlm(db, tax, [desktop], "scan", str(out), tasks=tasks)
    return summary, _read(out), out


def _keys(r):
    return r["label"]["answer_check"].get("answer_keys") or r["label"]["answer"]


def test_v12_asks_every_pair_and_spans_of_two_and_three(scene):
    db, tax, tmp = scene
    _, records, _ = _export(db, tax, tmp, S.DESKTOP, ["V12"])
    assert {r["label"]["v12_span"] for r in records} == {1, 2, 3}
    for r in records:
        assert _keys(r)["changed"] == bool(_keys(r)["events"])
        assert r["label"]["answer_check"]["from_step"] == \
            r["label"]["step"] - r["label"]["v12_span"]
    quiet = [r for r in records if not _keys(r)["changed"]]
    # the duplicate capture (step 3) is "no change" against step 2 alone
    assert quiet and all(r["label"]["answer_check"]["reason"] == "dupli" for r in quiet)
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)


def test_v3_span_lists_the_actions_between_two_moments(scene, monkeypatch):
    # the scene's parts are a few pixels on a 64 x 64 frame: every action counts as seen
    monkeypatch.setattr(T, "V3_MIN_PX", 1)
    db, tax, tmp = scene
    _, records, out = _export(db, tax, tmp, S.DESKTOP, ["V3"])
    world = _world(out)
    spans = [r for r in records if r["label"].get("v3_form") == "span"]
    assert spans and Checker(db, tax, S.DESKTOP).check_all(records) == len(records)
    for r in spans:
        check = r["label"]["answer_check"]
        assert check["type"] == "action_set" and len(check["gold_groups"]) >= 2
        assert grade(r, gold_reply(r, world), world, tax).score == 1.0
        one = {"actions": json.loads(gold_reply(r, world))["actions"][:1]}
        assert grade(r, json.dumps(one), world, tax).metrics["recall"] < 1.0


def test_v4_asks_every_part_and_verb_of_the_moment(scene):
    db, tax, tmp = scene
    _, records, _ = _export(db, tax, tmp, S.DESKTOP, ["V4"])
    assert {"permitted", "demonstrated"} <= {r["label"]["truth_source"] for r in records}
    first = {(r["label"]["answer_check"]["verb"], r["label"]["answer_check"]["target"])
             for r in records if r["label"]["step"] == 1}
    # at step 1: the screw and the latch can be done, the PSU cannot
    assert {("unscrew", S.SCREW), ("open", S.LATCH), ("remove", S.PSU)} <= first
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)


def test_v7_plans_every_major_part(cpu_scene):
    db, tax, tmp = cpu_scene
    _, records, out = _export(db, tax, tmp, W.DESKTOP, ["V7"])
    world = _world(out)
    assert {W.CPU, W.COOLER} <= {r["label"]["answer_check"]["goal"] for r in records}
    for r in records:
        if r["label"]["v7_form"] == "free":
            assert len(r["label"]["answer_check"]["reference_shortest"]) >= 2
        assert grade(r, gold_reply(r, world), world, tax).score == 1.0
    assert Checker(db, tax, W.DESKTOP).check_all(records) == len(records)


def test_v16_asks_several_plans_per_moment(scene):
    db, tax, tmp = scene
    _, records, _ = _export(db, tax, tmp, S.DESKTOP, ["V16"])
    ids = [r["label"]["readable_id"] for r in records]
    assert any(i.endswith("-valid3") for i in ids)
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)
    for r in records:
        # the graph's own notation stays label side
        assert "violated_edges" not in r["label"]["answer"]


@pytest.mark.parametrize("which", ["scene", "cpu_scene"])
def test_every_answer_names_parts_plainly_and_grades_one(which, request):
    from vlm_invariants import ordinals

    db, tax, tmp = request.getfixturevalue(which)
    desk = S.DESKTOP if which == "scene" else W.DESKTOP
    _, records, out = _export(db, tax, tmp, desk, T.TASKS, name="all")
    world = _world(out)
    for r in records:
        for s in re.findall(r'"([^"]*)"', json.dumps(r["label"]["answer"], ensure_ascii=False)):
            assert not KEY.search(s) and not ordinals(s), (r["label"]["readable_id"], s)
        assert grade(r, gold_reply(r, world), world, tax).score == pytest.approx(1.0), \
            r["label"]["readable_id"]


def test_the_checker_leaves_out_a_gallery_only_part(scene):
    db, tax, _ = scene
    db.upsert_instance(InstanceRec(key="psu_latch.09", desktop=S.DESKTOP, cls="psu_latch",
                                   attrs={"gallery_only": True}))
    assert "psu_latch.09" not in Checker(db, tax, S.DESKTOP).instances


def test_the_invariants_hold_on_the_scene(scene):
    import vlm_invariants

    db, tax, tmp = scene
    data = tmp / "pack" / "data"
    data.mkdir(parents=True)
    export_vlm(db, tax, [S.DESKTOP], "scan", str(data / "vlm_scan.jsonl"), tasks=T.TASKS)
    report = vlm_invariants.check_unit(data / "vlm_scan.jsonl", tax)
    assert report["violations"] == {}, report["examples"]
    # and they fail when a record is made wrong
    rows = _read(data / "vlm_scan.jsonl")
    bad = next(r for r in rows if r["prompt"]["task"] == "V12")
    bad["label"]["answer"]["events"] = [{"target": "psu.01", "old": "x", "new": "y"}]
    (data / "vlm_scan.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    assert vlm_invariants.check_unit(data / "vlm_scan.jsonl", tax)["violations"]


def test_v2_never_asks_an_implied_part(scene):
    """An implied part (the board nobody operated in this scene) moves by inference only: no
    V2 state question is about it (the checker found one on D04's implied cage latch)."""
    db, tax, tmp = scene
    _, records, _ = _export(db, tax, tmp, S.DESKTOP, ["V2"])
    for r in records:
        check = r["label"]["answer_check"]
        assert S.BOARD not in (check.get("members") or [check.get("instance")])
