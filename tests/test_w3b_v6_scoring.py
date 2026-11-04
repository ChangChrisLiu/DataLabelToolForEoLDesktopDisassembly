"""V6 graded over the frontier's groups (W3b, rewritten for the owner's 2026-11-04 rulings).

The gold is every action possible now, grouped (all plugs one group, all RAM latches,
the RAM modules, screws by the part they fasten, otherwise one group per part). A reply
is grounded action by action: a member of a group, a class phrase or the group itself
hits the group, repeats cost nothing, an action on a blocked or absent part is a wrong
answer. Precision, recall and F1 over the groups; F1 is the score.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import w3b_scene as W
from tda.core.db import Db
from tda.core.export.vlm import export_vlm, oracle_path
from tda.core.export.vlm_check import World, gold_reply, grade
from vlm_checker import Checker


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def _world(out: Path) -> World:
    for line in oracle_path(str(out)).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "desktop":
            return World.from_oracle(row)
    raise AssertionError


@pytest.fixture(params=[False, True], ids=["match-class", "group-id"])
def exported(request, tmp_path: Path):
    db = Db(str(tmp_path / "v6.sqlite"))
    tax = W.build(db, group=request.param, big_screws=True)
    out = tmp_path / "v.jsonl"
    export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out), tasks=["V6"])
    records = _read(out)
    yield db, tax, records, out, request.param
    db.close()


def _v6(records, step=None, form="open"):
    rows = [r for r in records if r["prompt"]["task"] == "V6"
            and (form is None or r["label"]["v6_form"] == form)]
    return [r for r in rows if step is None or r["label"]["step"] == step]


def _groups(record) -> dict:
    return {g["key"]: g for g in record["label"]["answer_check"]["groups"]}


def test_every_frame_has_an_open_record_and_the_checker_agrees(exported):
    db, tax, records, out, _ = exported
    opens, mcs = _v6(records), _v6(records, form="mc")
    assert opens
    assert len({r["label"]["v6_pair"] for r in opens}) == len(opens)
    for r in opens:
        assert "options" not in r["prompt"]
        assert r["label"]["v6_pair"] == r["id"]
    by_pair = {r["label"]["v6_pair"]: r for r in opens}
    for r in mcs:
        twin = by_pair[r["label"]["v6_pair"]]
        assert r["label"]["answer"]["group"] in _groups(twin)
    assert Checker(db, tax, W.DESKTOP).check_all(records) == len(records)


def test_a_member_of_a_group_hits_the_group_and_repeats_cost_nothing(exported):
    db, tax, records, out, _ = exported
    world = _world(out)
    # step 3: the shroud open, the fan lead out; the four cooler screws can be undone
    [r] = _v6(records, step=3)
    groups = _groups(r)
    screws = next(k for k in groups if k.startswith("screws:"))
    assert {a["target"] for a in groups[screws]["actions"]} == set(W.SCREWS)
    gold = json.loads(gold_reply(r, world))
    assert grade(r, json.dumps(gold), world, tax).score == 1.0
    # every screw by name instead of one: the same group, hit once, nothing wrong
    rest = [a for a in gold["actions"] if not a["target"].lower().startswith("cpu cooler screw")]
    each = rest + [{"verb": "unscrew", "target": world.label_of(s)} for s in W.SCREWS]
    g = grade(r, json.dumps({"actions": each}), world, tax)
    assert g.score == 1.0 and g.metrics["wrong_actions"] == 0
    # the class phrase alone
    phrase = rest + [{"verb": "unscrew", "target": "the CPU cooler screws"}]
    assert grade(r, json.dumps({"actions": phrase}), world, tax).score == 1.0


def test_a_blocked_or_absent_action_costs_precision(exported):
    db, tax, records, out, _ = exported
    world = _world(out)
    [r] = _v6(records, step=3)
    gold = json.loads(gold_reply(r, world))
    blocked = gold["actions"] + [{"verb": "remove", "target": "the CPU"}]
    g = grade(r, json.dumps({"actions": blocked}), world, tax)
    assert g.metrics["recall"] == 1.0 and g.metrics["precision"] < 1.0
    gone = gold["actions"] + [{"verb": "open", "target": "the cover"}]  # already open
    assert grade(r, json.dumps({"actions": gone}), world, tax).metrics["wrong_actions"] == 1
    assert grade(r, json.dumps({"actions": []}), world, tax).score == 0.0


def test_listed_options_are_scored_by_letter(exported):
    db, tax, records, out, _ = exported
    world = _world(out)
    for r in _v6(records, form="mc"):
        options = r["label"]["options"]
        for i, option in enumerate(options):
            want = 1.0 if option["kind"] == "frontier" else 0.0
            assert grade(r, json.dumps({"choice": "ABCD"[i]}), world, tax).score == want
        assert grade(r, gold_reply(r, world), world, tax).score == 1.0
