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
    plan = [(a["verb"], a["target"]) for a in check["reference_shortest"]]
    assert plan[-1] == ("remove", W.CPU)
    assert check["plan_length"] == len(plan)
    # the answer is the rules' shortest plan (owner 2026-11-04: the required steps are the
    # rules', not the demonstrated sequence), told in group actions -- one "unscrew the CPU
    # cooler screws"; any legal plan scores
    gid = f"screw.cpu_cooler@{W.COOLER}"
    answer = [(a["verb"], a["target"]) for a in first["label"]["answer"]["plan"]]
    assert ("unscrew", gid) in answer and answer[-1] == ("remove", W.CPU)
    assert ("remove", W.COOLER) in answer and ("open", W.LEVER) in answer
    assert check["validity"] == "simulated"
    assert check["plan_groups"] == {gid: list(W.SCREWS)}
    assert check["any_legal_plan_scores"] is True
    required = {r["target"]: r for r in check["required_actions"]}
    # the cooler's screws (fastened_by gates both ways of moving it), the
    # cooler, the lever, and what covers the lever -- but not the fan lead:
    # connected_to only gates *removing* the cooler, and swinging it aside works
    # ... the cooler screws as one group item (redesign 2026-11-03, spec 6.6)
    assert set(required) == {W.CPU, W.COOLER, W.LEVER, W.OPTICAL, gid}
    assert required[gid]["members"] == list(W.SCREWS)
    assert W.FAN not in required
    assert required[W.OPTICAL]["pixel_only"] is True
    assert first["label"]["truth_source"] == "graph_blocked_pixels"
    demo = [(a["verb"], a["target"]) for a in check["demonstrated_remainder"]]
    assert demo == [W.LOG[s] for s in range(3, 12)]
    assert first["label"]["answer_check"]["previous_action"]["target"] == W.SHROUD


@pytest.mark.parametrize("source", ["pixels:W1", "review:covers"])
def test_both_pixel_sources_are_graph_blocked_pixels(tmp_path: Path, source):
    """Review S1: W3a's imports (`pixels:*`) and the user's rulings (`review:*`)."""
    from tda.core.export.vlm_tasks import is_pixel_edge
    from tda.core.graph import Edge

    assert is_pixel_edge(Edge("covered_by", "a", "b", source=source))
    assert not is_pixel_edge(Edge("covered_by", "a", "b", source="rule"))
    edges = [dict(e, source=source) if e.get("source") else e for e in W.EDGES]
    db = Db(str(tmp_path / "src.sqlite"))
    tax = W.build(db, edges=edges)
    _, records, _ = _export(db, tax, tmp_path)
    v4 = [r for r in records if r["prompt"]["task"] == "V4"
          and r["label"]["answer_check"]["target"] == W.LEVER and r["label"].get("negative")]
    assert v4 and all(r["label"]["truth_source"] == "graph_blocked_pixels" for r in v4)
    assert {r["label"]["truth_source"] for r in _v7(records)
            if r["label"]["step"] < 8} == {"graph_blocked_pixels"}
    db.close()


def test_a_desktop_whose_cooler_set_is_unresolved_asks_no_v7(tmp_path: Path):
    db = Db(str(tmp_path / "nocover.sqlite"))
    edges = [e for e in W.EDGES if not (e["type"] == "covered_by" and e["target"] == W.CPU)]
    tax = W.build(db, edges=edges)
    summary, records, _ = _export(db, tax, tmp_path)
    assert not _v7(records)
    assert summary["skip_reasons"][str(W.DESKTOP)]["V7"]["cooler_unresolved"] >= 1
    db.close()


def test_any_cooler_kind_over_the_cpu_resolves_the_set(tmp_path: Path):
    """W3a's cooler rule writes ``covered_by(cpu <- heatsink)`` on the eight
    two-cooler desktops; the skip condition reads the class, not the kind, so
    those desktops ask V7 as soon as the edge is in the database."""
    db = Db(str(tmp_path / "heatsink.sqlite"))
    tax = W.build(db, cooler_kind="heatsink")
    summary, records, _ = _export(db, tax, tmp_path)
    assert _v7(records)
    assert "cooler_unresolved" not in summary["skip_reasons"][str(W.DESKTOP)].get("V7", {})
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
    # the answer is the demonstrated remainder, but any legal plan scores --
    # the graph's shortest one included
    names = {k: v["label"] for k, v in world.instances.items()}
    shortest = [{"verb": a["verb"], "target": names[a["target"]]}
                for a in r["label"]["answer_check"]["reference_shortest"]]
    assert grade(r, json.dumps({"plan": shortest}), world, tax).score == 1.0


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
    # ... and "loosen the cooler screws" stays required while the cooler is in, whatever
    # the log says about each of them (redesign 2026-11-03, spec 6.6)
    ok = {"plan": [{"verb": "unscrew", "target": "the CPU cooler screws"},
                   {"verb": "displace", "target": "the heatsink fan CPU cooler"},
                   {"verb": "open", "target": "CPU socket lever"},
                   {"verb": "remove", "target": "CPU"}]}
    g = grade(r, json.dumps(ok), world, tax)
    assert g.metrics["violation_free"] and g.metrics["goal_reached"], g.metrics
    assert g.score == 1.0


GROUP = f"screw.cpu_cooler@{W.COOLER}"


def _grouped(world: World, record: dict) -> tuple[World, dict]:
    """The scene in the redesign's contract: the four cooler screws are one
    group, and the plan unscrews them in one group action (spec 3.1)."""
    import copy

    instances = {k: dict(v, group=GROUP if k in W.SCREWS else None)
                 for k, v in world.instances.items()}
    groups = {GROUP: {"members": list(W.SCREWS), "match_class": "screw.cpu_cooler",
                      "anchor": W.COOLER, "label": "CPU cooler screws"}}
    out = copy.deepcopy(record)

    def collapse(items):
        rows = []
        for item in items:
            item = dict(item, target=GROUP if item["target"] in W.SCREWS else item["target"])
            if not rows or rows[-1] != item:
                rows.append(item)
        return rows

    check = out["label"]["answer_check"]
    for name in ("reference_shortest", "demonstrated_remainder"):
        check[name] = collapse(check[name])
    out["label"]["answer"]["plan"] = collapse(out["label"]["answer"]["plan"])
    check["required_actions"] = [r for r in check["required_actions"]
                                 if r["target"] not in W.SCREWS] + [
        {"target": GROUP, "states": ["loosened", "removed"], "via": [], "pixel_only": False}]
    return World(world.desktop, instances, world.edges, world.log, groups), out


def test_v7_group_plan_on_the_scene(scene):
    db, tax, tmp = scene
    _, records, out = _export(db, tax, tmp)
    world = _world(out)
    for step in (2, 5):
        grouped, r = _grouped(world, next(r for r in _v7(records) if r["label"]["step"] == step))
        assert sum(a["target"] == GROUP for a in r["label"]["answer"]["plan"]) == 1
        g = grade(r, gold_reply(r, grouped), grouped, tax)
        assert g.score == 1.0 and g.metrics["violation_free"], (step, g.metrics)
        assert "CPU cooler screws" in gold_reply(r, grouped)
        # in words, the screws once: every remaining member is unscrewed
        plan = [{"verb": "unscrew", "target": "the cooler screws"},
                {"verb": "remove", "target": "the optical drive"},
                {"verb": "remove", "target": "the CPU cooler"},
                {"verb": "open", "target": "the CPU socket lever"},
                {"verb": "remove", "target": "the CPU"}]
        if step == 2:
            plan.insert(0, {"verb": "disconnect", "target": "the fan connector"})
        g = grade(r, json.dumps({"plan": plan}), grouped, tax)
        assert g.score == 1.0 and g.metrics["noop_steps"] == 0, (step, g.metrics)
        # the cooler before its screws: a violation
        g = grade(r, json.dumps({"plan": plan[-3:]}), grouped, tax)
        assert g.metrics["violations"] >= 1 and g.score == 0.0


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
