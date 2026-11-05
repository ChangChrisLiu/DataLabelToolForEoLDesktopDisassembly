"""The final round of the VLM question redesign (owner decisions of 2026-11-04).

One test (or a few) per item of ``exports/vlm_audit/final_round_decisions.md``:

1. the graph-derived truths use loose parts first and dynamic occlusion (``moment_edges``,
   ``remaining_plan``), and D66 asks no step-ordered question;
2. V6 is the frontier -- open (F1 over groups), multiple choice (one option possible),
   constrained (a rule in words);
3. V7 free / constrained (the plug count);
4. V16's validity is a replay, its record kinds say so, its reasoning is step by step;
5. V4 negatives from occlusion and from a loose part, in words;
6. V14 is merged into V2 (``cannot_tell``);
7. every record reasons in words, and the pack writes ``reasoning/<unit>.jsonl``;
8. ``eval_v3`` keeps the old ids, the audit gives every form its share, ``chat-v5``.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

import vlm_scene as S
import w3b_scene as W
from graph_scenes import act, inst, state_after
from tda.core.db import Db
from tda.core.export import vlm as V
from tda.core.export import vlm_tasks as T
from tda.core.export.vlm import export_vlm, oracle_path
from tda.core.export.vlm_check import World, gold_reply, grade, moment_edges_of
from tda.core.graph import (
    LOOSE_SOURCE,
    OCCLUSION_SOURCE,
    legal_actions,
    moment_edges,
    propose_edges,
    remaining_plan,
)
from tda.core.graph_rules import LOOSE_ATTR
from tda.core.model import InstanceRec
from tda.core.states import initial_state
from tda.core.taxonomy import load_taxonomy
from vlm_checker import Checker


def _keys(record: dict) -> dict:
    """The answer with its keys (``answer_check.answer_keys``; ``label.answer`` names parts
    plainly since the owner's 2026-11-04 round)."""
    label = record["label"]
    return (label.get("answer_check") or {}).get("answer_keys") or label["answer"]


def _set_keys(record: dict, answer: dict) -> None:
    label = record["label"]
    if (label.get("answer_check") or {}).get("answer_keys"):
        label["answer_check"]["answer_keys"] = answer
    else:
        label["answer"] = answer


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

KEY = re.compile(r"\b[a-z_]+(?:\.[a-z_0-9]+)*\.\d{2}\b")


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def _world(out: Path) -> World:
    for line in oracle_path(str(out)).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "desktop":
            return World.from_oracle(row)
    raise AssertionError("no desktop line")


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


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


def _export(db, tax, tmp, desktop, tasks, name="v", **kw):
    out = tmp / f"{name}.jsonl"
    summary = export_vlm(db, tax, [desktop], "scan", str(out), tasks=tasks, **kw)
    return summary, _read(out), out


def _form(records, task, form):
    field = {"V6": "v6_form", "V7": "v7_form", "V3": "v3_form"}[task]
    return [r for r in records if r["prompt"]["task"] == task and r["label"].get(field) == form]


# --------------------------------------------------------------------------- #
# 1. the corrected graph
# --------------------------------------------------------------------------- #
def _loose_fan_desk():
    recs = [inst("motherboard.01", "motherboard"),
            inst("case_fan.01", "case_fan", attrs={LOOSE_ATTR: True}),
            inst("connector.fan.01", "connector", attrs={"kind": "fan"},
                 socket_host="motherboard.01", cable="cable:case_fan"),
            inst("ram_latch.01", "ram_latch"),
            inst("screw.motherboard.01", "screw",
                 attrs={"role": "motherboard", "captive": False}, fastens="motherboard.01")]
    return {r.key: r for r in recs}


def test_moment_edges_say_what_legal_actions_filters(tax):
    table = _loose_fan_desk()
    edges = propose_edges(table, tax)
    state = initial_state(table, tax)
    occluded = {"screw.motherboard.01": ["ram_latch.01"]}
    extra = moment_edges(table, edges, state, tax, occluded=occluded)
    assert {e.source for e in extra} == {OCCLUSION_SOURCE, LOOSE_SOURCE}
    filtered = legal_actions(table, edges, state, tax, loose_first=True, occluded=occluded)
    as_edges = legal_actions(table, [*edges, *extra], state, tax, loose_first=False)
    # the same set, but for the loose part's own verbs other than `remove` (the filter keeps
    # only its removal; an edge cannot gate a part by itself)
    assert set(filtered) == {a for a in as_edges if a[1] != "case_fan.01" or a[0] == "remove"}
    after = state_after(table, [act(1, "connector.fan.01", "disconnect"),
                                act(2, "case_fan.01", "remove")], tax)
    gone = moment_edges(table, edges, after, tax, occluded={})
    assert gone == []


def test_remaining_plan_takes_the_loose_part_out_first(tax):
    table = _loose_fan_desk()
    edges = propose_edges(table, tax)
    state = initial_state(table, tax)
    plan = remaining_plan(table, edges, state, "motherboard.01", tax)
    assert plan[:2] == [("disconnect", "connector.fan.01"), ("remove", "case_fan.01")]
    assert plan[-1] == ("remove", "motherboard.01")
    old = remaining_plan(table, edges, state, "motherboard.01", tax, loose_first=False)
    assert ("remove", "case_fan.01") not in old


def test_d66_asks_no_step_ordered_question(scene):
    db, tax, tmp = scene
    meta = dict(db.get_desktop(S.DESKTOP) or {})
    db.upsert_desktop(S.DESKTOP, {**{k: v for k, v in meta.items() if k != "id"},
                                  "stepwise": False})
    summary, records, _ = _export(db, tax, tmp, S.DESKTOP, ["V2", "V4", "V6", "V16"])
    tasks = {r["prompt"]["task"] for r in records}
    assert not tasks & {"V4", "V6", "V7", "V16"} and "V2" in tasks
    assert summary["skip_reasons"][str(S.DESKTOP)]["graph_tasks"] == "not_stepwise"
    assert summary["non_stepwise"] == [S.DESKTOP]


# --------------------------------------------------------------------------- #
# 2. V6: the frontier
# --------------------------------------------------------------------------- #
def test_v6_open_is_every_action_possible_now_graded_by_group_f1(scene):
    db, tax, tmp = scene
    _, records, out = _export(db, tax, tmp, S.DESKTOP, ["V6"])
    world = _world(out)
    opens = _form(records, "V6", "open")
    assert opens and Checker(db, tax, S.DESKTOP).check_all(records) == len(records)
    first = next(r for r in opens if r["label"]["step"] == 1)
    check = first["label"]["answer_check"]
    assert check["type"] == "frontier" and check["metric"] == "group_f1"
    acts = {(a["verb"], a["target"]) for g in check["groups"] for a in g["actions"]}
    # the screw, the plug and the latch can be done now; the PSU and the RAM cannot
    assert {("unscrew", S.SCREW), ("disconnect", S.PLUG), ("open", S.LATCH)} <= acts
    assert not {a for a in acts if a[1] in (S.PSU, S.RAM)}
    gold = gold_reply(first, world)
    assert grade(first, gold, world, tax).score == 1.0
    # one blocked action more: precision drops, recall stays
    reply = json.loads(gold)
    reply["actions"].append({"verb": "remove", "target": "the PSU"})
    g = grade(first, json.dumps(reply), world, tax)
    assert g.metrics["recall"] == 1.0 and g.metrics["precision"] < 1.0 and g.score < 1.0
    # a group left out: recall drops
    short = {"actions": json.loads(gold)["actions"][:1]}
    assert grade(first, json.dumps(short), world, tax).metrics["recall"] < 1.0
    assert "why" in _keys(first)
    # the question names no part and no ordinal
    assert not KEY.search(first["prompt"]["question"])


def test_v6_multiple_choice_has_one_possible_option_of_the_same_verb():
    """The choice: one option inside the frontier, the others blocked, every one of the
    possible option's verb (the option-verb shortcut was 82 % against 26 % random)."""
    G = T.FrontierGroup
    groups = [G("all:connector", [("disconnect", "connector.01")]),
              G("part:psu.01", [("remove", "psu.01")])]
    blocked = [G("part:cpu.01", [("remove", "cpu.01")]),
               G("part:motherboard.01", [("remove", "motherboard.01")]),
               G("all:ram_latch", [("open", "ram_latch.01")])]
    right, wrong = T.pick_mc(groups, blocked, "all:connector", "seed")
    # the demonstrated group (a plug) has no blocked twin of its verb: the PSU stands in
    assert right.key == "part:psu.01"
    assert {g.key for g in wrong} == {"part:cpu.01", "part:motherboard.01"}
    assert T.pick_mc(groups, blocked[2:], None, "seed") == (None, [])


def test_v6_multiple_choice_is_graded_by_the_frontier_option():
    from tda.core.export.vlm_check import World

    world = World(1, {"psu.01": {"class": "psu", "label": "PSU"},
                      "cpu.01": {"class": "cpu", "label": "CPU"}}, [])
    options = [{"verb": "remove", "target": "cpu.01", "kind": "graph_blocked",
                "say_verb": "take out", "target_label": "the CPU"},
               {"verb": "remove", "target": "psu.01", "kind": "frontier",
                "say_verb": "take out", "target_label": "the PSU"}]
    record = {"id": "rec_0000000000000001",
              "prompt": {"task": "V6", "images": [], "question": "Which?",
                         "options": [{"verb": o["say_verb"], "target_label": o["target_label"]}
                                     for o in options]},
              "label": {"answer": {"verb": "remove", "target": "psu.01"}, "options": options,
                        "v6_form": "mc",
                        "answer_check": {"type": "frontier", "form": "mc",
                                         "reference": ["remove", "psu.01"]}}}
    assert grade(record, gold_reply(record, world), world, None).score == 1.0
    assert grade(record, json.dumps({"choice": "A"}), world, None).score == 0.0
    for shown in record["prompt"]["options"]:
        assert set(shown) == {"verb", "target_label"} and not KEY.search(shown["target_label"])


def test_v6_constrained_removes_a_group_and_counts_violations(cpu_scene):
    db, tax, tmp = cpu_scene
    _, records, out = _export(db, tax, tmp, W.DESKTOP, ["V6"])
    world = _world(out)
    ruled = _form(records, "V6", "constrained")
    assert ruled and Checker(db, tax, W.DESKTOP).check_all(records) == len(records)
    r = ruled[0]
    check = r["label"]["answer_check"]
    assert check["forbidden_groups"] and set(check["gold_groups"]).isdisjoint(
        check["forbidden_groups"])
    rule = check["constraint"]["rule"]
    assert rule in r["prompt"]["question"] and not re.search(r"\d", rule)
    gold = gold_reply(r, world)
    g = grade(r, gold, world, tax)
    assert g.score == 1.0 and g.metrics["constraint_violated"] is False
    forbidden = next(gr for gr in check["groups"] if gr["key"] == check["forbidden_groups"][0])
    bad = json.loads(gold)
    a = forbidden["actions"][0]
    bad["actions"].append({"verb": a["verb"], "target": world.label_of(a["target"])})
    g = grade(r, json.dumps(bad), world, tax)
    assert g.metrics["constraint_violated"] is True and g.score < 1.0


def test_v6_a_part_under_another_leaves_the_frontier(scene, monkeypatch):
    db, tax, tmp = scene
    monkeypatch.setattr(V, "_occlusion_of",
                        lambda db, d, ctx, edges: (lambda step: {S.PLUG: [S.PSU]}))
    _, records, _ = _export(db, tax, tmp, S.DESKTOP, ["V6"])
    first = next(r for r in _form(records, "V6", "open") if r["label"]["step"] == 1)
    acts = {(a["verb"], a["target"]) for g in first["label"]["answer_check"]["groups"]
            for a in g["actions"]}
    assert ("disconnect", S.PLUG) not in acts and ("unscrew", S.SCREW) in acts
    assert first["label"]["answer_check"]["moment"]["occluded"] == {S.PLUG: [S.PSU]}
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)


# --------------------------------------------------------------------------- #
# 3. V7 free / constrained
# --------------------------------------------------------------------------- #
def _with_second_plug(db):
    db.upsert_instance(InstanceRec(key="connector.sata_power.01", desktop=W.DESKTOP,
                                   cls="connector", attrs={"kind": "sata_power"},
                                   socket_host=W.BOARD))


def test_v7_free_is_the_rules_minimal_plan(cpu_scene):
    db, tax, tmp = cpu_scene
    _, records, out = _export(db, tax, tmp, W.DESKTOP, ["V7"])
    world = _world(out)
    free = _form(records, "V7", "free")
    assert free and Checker(db, tax, W.DESKTOP).check_all(records) == len(records)
    late = next(r for r in free if r["label"]["step"] == 9)  # the cooler is off
    assert [(a["verb"], a["target"]) for a in _keys(late)["plan"]] == \
        [("open", W.LEVER), ("remove", W.CPU)]
    for r in free:
        assert grade(r, gold_reply(r, world), world, tax).score == 1.0
        assert r["label"]["answer_check"]["validity"] == "simulated"


def test_v7_constrained_keeps_the_rule_and_counts_the_plugs(cpu_scene):
    db, tax, tmp = cpu_scene
    _with_second_plug(db)
    _, records, out = _export(db, tax, tmp, W.DESKTOP, ["V7"])
    world = _world(out)
    ruled = _form(records, "V7", "constrained")
    assert ruled and Checker(db, tax, W.DESKTOP).check_all(records) == len(records)
    r = next(x for x in ruled if x["label"]["answer_check"].get("count"))
    check = r["label"]["answer_check"]
    assert check["count"]["present"] == len(check["count"]["members"]) >= 2
    assert "Give each connector you unplug a step of its own" in r["prompt"]["question"]
    g = grade(r, gold_reply(r, world), world, tax)
    assert g.score == 1.0 and g.metrics["count_exact"] and g.metrics["constraint_kept"]
    # the free plan (no plug pulled) breaks the rule, and counts too few plugs
    plain = [a for a in _keys(r)["plan"] if a["verb"] != "disconnect"]
    reply = json.dumps({"plan": [{"verb": a["verb"], "target": world.label_of(a["target"])}
                                 for a in plain]})
    g = grade(r, reply, world, tax)
    assert g.score == 0.0 and g.metrics["constraint_kept"] is False
    assert g.metrics["count_planned"] < g.metrics["count_present"]


# --------------------------------------------------------------------------- #
# 4. V16 by replay
# --------------------------------------------------------------------------- #
def test_v16_validity_is_a_replay_and_its_reasoning_goes_step_by_step(scene):
    db, tax, tmp = scene
    _, records, _ = _export(db, tax, tmp, S.DESKTOP, ["V16"])
    v16 = [r for r in records if r["prompt"]["task"] == "V16"]
    assert v16 and Checker(db, tax, S.DESKTOP).check_all(records) == len(records)
    valid = [r for r in v16 if _keys(r)["valid"]]
    assert valid and all(re.search(r"-valid\d?$", r["label"]["readable_id"]) for r in valid)
    assert not any(r["label"]["readable_id"].endswith("-true") for r in v16)
    assert {r["label"]["truth_source"] for r in valid} == {"simulated"}
    for r in v16:
        text = r["label"]["rationale"]["text"]
        assert text.startswith("Step 1 (")
        if not _keys(r)["valid"]:
            k = _keys(r)["first_error_index"] + 1
            assert f"step {k} is the first that breaks" in text


# --------------------------------------------------------------------------- #
# 5. V4: occlusion and loose parts
# --------------------------------------------------------------------------- #
def test_v4_reasons_speak_of_the_cover_and_of_the_loose_part(scene, monkeypatch):
    db, tax, tmp = scene
    rec = db.instances(S.DESKTOP)[S.PSU]
    db.upsert_instance(InstanceRec(**{**rec.__dict__, "attrs": {**rec.attrs, LOOSE_ATTR: True}}))
    monkeypatch.setattr(V, "_occlusion_of",
                        lambda db, d, ctx, edges: (lambda step: {S.LATCH: [S.RAM]}))
    tc_records = []
    _, records, _ = _export(db, tax, tmp, S.DESKTOP, ["V4", "V6"])
    tc_records.extend(records)
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)
    first = next(r for r in _form(records, "V6", "open") if r["label"]["step"] == 1)
    acts = {(a["verb"], a["target"]) for g in first["label"]["answer_check"]["groups"]
            for a in g["actions"]}
    # the loose PSU's way out alone: its screw and its plug; the latch waits
    assert acts == {("unscrew", S.SCREW), ("disconnect", S.PLUG)}
    assert "loose in the case" in first["label"]["rationale"]["text"]
    # the reason clause a loose part and an occluder produce, in words
    from tda.core.export.vlm_tasks import reason_clauses  # noqa: F401
    sources = {r["label"].get("truth_source") for r in records if r["prompt"]["task"] == "V4"}
    assert sources <= {"demonstrated", "permitted", *T.GRAPH_KINDS, "failed_attempt"}


def test_reason_clauses_word_the_moment_edges(scene):
    from tda.core.export.coco import load_ctx
    from tda.core.graph import Edge
    from tda.core.export.vlm_tasks import TaskCtx, reason_clauses

    db, tax, _ = scene
    ctx = load_ctx(db, tax, S.DESKTOP, "scan")
    tc = TaskCtx(ctx=ctx, view="scan", tier=None, graph_version="x", meta={}, edges=[],
                 frames={}, steps=[])
    state = ctx.state_at(1)
    loose = Edge(type="blocked_by", target=S.LATCH, blocker=S.PSU, mode="loose",
                 source=LOOSE_SOURCE, status="accepted")
    cover = Edge(type="covered_by", target=S.LATCH, blocker=S.RAM, source=OCCLUSION_SOURCE,
                 status="accepted")
    text = " ".join(reason_clauses(tc, [loose, cover], state, met=False,
                                   asked=(S.LATCH, [S.LATCH])))
    assert "lying loose in the case has to come out first" in text
    assert "still covers" in text and not KEY.search(text)


# --------------------------------------------------------------------------- #
# 6. V14 merged into V2
# --------------------------------------------------------------------------- #
def test_v2_answers_cannot_tell_on_a_hidden_part(scene, monkeypatch):
    db, tax, tmp = scene
    monkeypatch.setattr(T, "V2_HIDDEN_EVERY", 1)
    summary, records, out = _export(db, tax, tmp, S.DESKTOP, ["V2"])
    world = _world(out)
    assert "V14" not in T.ALL_TASKS and "V14" not in T.GENERATORS
    states = [r for r in records if r["label"]["answer_check"].get("derive") == "instance_state"]
    assert states and all("cannot_tell" in r["label"]["answer_check"]["allowed"] for r in states)
    hidden = [r for r in states if r["label"]["answer_check"].get("hidden")]
    assert hidden and {_keys(r)["state"] for r in hidden} == {"cannot_tell"}
    assert Checker(db, tax, S.DESKTOP).check_all(records) == len(records)
    for r in hidden:
        assert grade(r, json.dumps({"state": "cannot tell"}), world, tax).score == 1.0
        assert grade(r, json.dumps({"state": "installed"}), world, tax).score == 0.0
        assert "cannot be told" in r["label"]["rationale"]["text"]
    assert summary["by_form"].get("V2.hidden") == len(hidden)


# --------------------------------------------------------------------------- #
# 7. reasoning everywhere, and the side product
# --------------------------------------------------------------------------- #
def test_every_record_reasons_in_words(scene):
    db, tax, tmp = scene
    _, records, _ = _export(db, tax, tmp, S.DESKTOP, T.TASKS)
    assert {r["prompt"]["task"] for r in records} >= {"V1", "V2", "V3", "V4", "V6", "V8",
                                                      "V12", "V16"}
    for r in records:
        text = r["label"]["rationale"].get("text")
        assert isinstance(text, str) and text.strip(), r["label"]["readable_id"]
        assert not KEY.search(text), (r["label"]["readable_id"], text)
        assert "violates" not in text and "edge" not in text


def test_the_pack_writes_the_reasoning_rows(scene):
    import export_vlmpack as P

    db, tax, tmp = scene
    data = tmp / "data"
    data.mkdir()
    export_vlm(db, tax, [S.DESKTOP], "scan", str(data / "vlm_scan.jsonl"), tasks=T.TASKS)
    counts = P.write_reasoning(data, tmp / "reasoning", ["scan"])
    rows = _read(tmp / "reasoning" / "scan.jsonl")
    assert counts == {"scan": len(rows)} and rows
    for row in rows:
        assert set(row) == {"id", "task", "question", "answer", "reasoning"}
        assert row["reasoning"] and not KEY.search(json.dumps(row["answer"]))


# --------------------------------------------------------------------------- #
# 8. eval_v3, the audit, chat-v5
# --------------------------------------------------------------------------- #
def test_eval_v3_refuses_the_old_subsets_and_keeps_old_ids():
    import vlm_eval_set as E

    assert E.EVAL_VERSION == "eval_v3" and set(E.OLD_EVAL_VERSIONS) == {"eval_v1", "eval_v2"}
    with pytest.raises(SystemExit):
        E.refuse_old_eval("D:/x/eval_v2/multi.jsonl")
    E.refuse_old_eval("D:/x/eval_v3/multi.jsonl")
    assert "V14" not in E.TASK_ORDER and "V14" not in E.BINARY_TASKS

    def rec(i, task="V12", form=None, desk=1):
        label = {"desktop": desk, "step": i, "answer": {"changed": True}}
        if form:
            label["v6_form"] = form
        return {"id": f"rec_{i:016x}", "prompt": {"task": task}, "label": label}

    units = E.units_of([rec(i) for i in range(40)], "multi")
    for u in units:
        u.kept = u.key in {"rec_0000000000000021", "rec_0000000000000007"}
    chosen = E.draw(units, targets={"V12": 2}, max_share=1.0)["V12"]
    assert {u.key for u in chosen} == {"rec_0000000000000021", "rec_0000000000000007"}
    assert E.bucket_of(rec(1, "V6", "constrained")) == "V6.constrained"
    assert E.bucket_of(rec(1, "V6", "open")) == "V6"


def test_the_audit_gives_every_form_its_share_and_keeps_old_items():
    import vlm_sample

    recs = []
    for form in ("open", "mc", "constrained"):
        for i in range(30):
            recs.append({"id": f"rec_{form}{i:03d}", "prompt": {"task": "V6"},
                         "label": {"desktop": i % 5, "v6_form": form, "answer": {}}})
    keep = {"rec_mc007", "rec_constrained011"}
    rows = vlm_sample.draw(recs, size=30, per_task=25, seed=0, by_form=True, keep=keep,
                           max_share=1.0, test_desktops={0, 1, 2, 3, 4})
    forms = [r["label"]["v6_form"] for r in rows]
    assert forms.count("open") == forms.count("mc") == forms.count("constrained") == 10
    assert keep <= {r["id"] for r in rows}


def test_chat_format_is_v5_and_v14_has_no_format():
    import vlm_to_chat
    from tda.core.export.vlm_check import CAPABILITY_LAYERS, RESPONSE_FORMATS

    assert vlm_to_chat.CHAT_FORMAT_ID == "chat-v5"
    assert "V14" not in RESPONSE_FORMATS and "V14" not in CAPABILITY_LAYERS
    assert '"actions"' in RESPONSE_FORMATS["V6.open"]


def test_moment_edges_of_rebuilds_the_record_rules():
    check = {"moment": {"occluded": {"x.01": ["y.01"]}, "loose": ["l.01"],
                        "loose_scope": ["a.01"]}}
    got = moment_edges_of(check, ["a.01", "b.01", "l.01", "x.01"])
    pairs = {(e.type, e.target, e.blocker, e.mode) for e in got}
    assert ("covered_by", "x.01", "y.01", None) in pairs
    assert ("blocked_by", "b.01", "l.01", "loose") in pairs
    assert not any(e.target in ("a.01", "l.01") for e in got if e.mode == "loose")
    assert copy.deepcopy(check) == check
