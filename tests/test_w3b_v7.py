"""V7-cpu (W3b): the rest of the CPU plan, asked from frames where it is under way.

Fixture :mod:`tests.w3b_scene`: a shroud over a heatsink-fan held by four
captive screws, a fan lead, a lever under an optical drive (a pixel-imported
edge) and the CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import w3b_scene as W
from tda.core.db import Db
from tda.core.export.vlm import export_vlm, oracle_path
from tda.core.export.vlm_check import World, gold_reply, grade
from tda.core.export.vlm_tasks import TASKS, V7_QUESTIONS
from vlm_checker import Checker, scan_prompt_leaks


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


@pytest.fixture
def scene(tmp_path: Path):
    db = Db(str(tmp_path / "w3b.sqlite"))
    tax = W.build(db)
    yield db, tax, tmp_path
    db.close()


def _export(db, tax, tmp_path, view=W.VIEW, **kw):
    out = tmp_path / f"v_{view}.jsonl"
    summary = export_vlm(db, tax, [W.DESKTOP], view, str(out), **kw)
    return summary, _read(out), out


def _world(out: Path) -> World:
    for line in oracle_path(str(out)).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "desktop":
            return World.from_oracle(row)
    raise AssertionError("no desktop line in the oracle")


def _v7(records):
    return [r for r in records if r["prompt"]["task"] == "V7"]


def test_v7_is_in_the_default_set_and_has_five_to_ten_templates():
    assert "V7" in TASKS and "V5" not in TASKS
    assert 5 <= len(V7_QUESTIONS) <= 10


def test_v7_is_asked_only_while_the_cpu_plan_is_visibly_under_way(scene):
    db, tax, tmp = scene
    _, records, _ = _export(db, tax, tmp)
    steps = sorted(r["label"]["step"] for r in _v7(records))
    # step 1: nothing done; step 11: the CPU is out. Steps 2-10 have the
    # shroud open (a visible change) and the CPU still in.
    assert steps == list(range(2, 11))
    assert Checker(db, tax, W.DESKTOP).check_all(records) == len(records)
    assert not scan_prompt_leaks(records)
    for r in _v7(records):
        assert "cpu.01" not in r["prompt"]["question"]
        assert "the CPU" in r["prompt"]["question"]


def test_the_answer_is_the_remaining_legal_plan_and_its_requirements(scene):
    db, tax, tmp = scene
    _, records, _ = _export(db, tax, tmp)
    first = next(r for r in _v7(records) if r["label"]["step"] == 2)
    check = first["label"]["answer_check"]
    assert check["type"] == "goal_plan" and check["redundancy_penalty"] is False
    assert check["goal"] == W.CPU and check["goal_removed_at"] == 11
    plan = [(a["verb"], a["target"]) for a in check["reference_plan"]]
    assert plan[-1] == ("remove", W.CPU)
    assert check["plan_length"] == len(plan)
    required = {r["target"]: r for r in check["required_actions"]}
    # the cooler's screws (fastened_by gates both ways of moving it), the
    # cooler, the lever, and what covers the lever -- but not the fan lead:
    # connected_to only gates *removing* the cooler, and swinging it aside works
    assert set(required) == {W.CPU, W.COOLER, W.LEVER, W.OPTICAL, *W.SCREWS}
    assert W.FAN not in required
    assert required[W.OPTICAL]["pixel_only"] is True
    assert first["label"]["truth_source"] == "graph_blocked_pixels"
    demo = [(a["verb"], a["target"]) for a in check["demonstrated_remainder"]]
    assert demo == [W.LOG[s] for s in range(3, 12)]
    assert first["label"]["answer_check"]["previous_action"]["target"] == W.SHROUD


def test_a_desktop_whose_cooler_set_is_unresolved_asks_no_v7(tmp_path: Path):
    db = Db(str(tmp_path / "nocover.sqlite"))
    edges = [e for e in W.EDGES if not (e["type"] == "covered_by" and e["target"] == W.CPU)]
    tax = W.build(db, edges=edges)
    summary, records, _ = _export(db, tax, tmp_path)
    assert not _v7(records)
    assert summary["skip_reasons"][str(W.DESKTOP)]["V7"]["cooler_unresolved"] >= 1
    db.close()


def test_the_gold_plan_scores_one_and_extra_steps_cost_nothing(scene):
    db, tax, tmp = scene
    _, records, out = _export(db, tax, tmp)
    world = _world(out)
    for r in _v7(records):
        g = grade(r, gold_reply(r, world), world, tax)
        assert g.score == 1.0, (r["label"]["step"], g.metrics)
    r = next(r for r in _v7(records) if r["label"]["step"] == 2)
    plan = json.loads(gold_reply(r, world))["plan"]
    # an extra, harmless action and a repeat of one already done: no penalty
    padded = ([{"verb": "unscrew", "target": "Motherboard screw 1"},
               {"verb": "open", "target": "Cover 1"}] + plan)
    g = grade(r, json.dumps({"plan": padded}), world, tax)
    assert g.score == 1.0 and g.metrics["noop_steps"] == 1


def test_a_plan_through_the_lever_or_without_the_cooler_is_wrong(scene):
    db, tax, tmp = scene
    _, records, out = _export(db, tax, tmp)
    world = _world(out)
    r = next(r for r in _v7(records) if r["label"]["step"] == 8)
    # straight for the CPU: the cooler covers it and the lever locks it
    g = grade(r, json.dumps({"plan": [{"verb": "remove", "target": "CPU 1"}]}), world, tax)
    assert g.score == 0.0
    assert g.metrics["violations"] == 1 and g.metrics["first_violation"] == 0
    assert g.metrics["goal_reached"] is True and g.metrics["required_recall"] < 1.0
    # the class phrase is grounded on an eligible part: "a CPU cooler" is the cooler
    ok = {"plan": [{"verb": "displace", "target": "the heatsink fan CPU cooler"},
                   {"verb": "open", "target": "CPU socket lever"},
                   {"verb": "remove", "target": "CPU"}]}
    g = grade(r, json.dumps(ok), world, tax)
    assert g.metrics["violation_free"] and g.metrics["goal_reached"], g.metrics
    assert g.score == 1.0


def test_unnamed_screws_ground_to_the_screws_still_in(scene):
    """"Unscrew a CPU cooler screw" four times is four screws, group-equivalently."""
    db, tax, tmp = scene
    _, records, out = _export(db, tax, tmp)
    world = _world(out)
    r = next(r for r in _v7(records) if r["label"]["step"] == 3)
    plan = [{"verb": "unscrew", "target": "CPU cooler screw"}] * 4 + [
        {"verb": "remove", "target": "optical drive"},
        {"verb": "remove", "target": "CPU cooler"},
        {"verb": "open", "target": "CPU socket lever"},
        {"verb": "remove", "target": "CPU"}]
    g = grade(r, json.dumps({"plan": plan}), world, tax)
    assert g.score == 1.0, g.metrics
    assert r["label"]["image_undecidable"] is False  # no captive screw is loose yet
    later = next(r for r in _v7(records) if r["label"]["step"] == 5)
    assert later["label"]["image_undecidable"] is True  # two of four are loose
