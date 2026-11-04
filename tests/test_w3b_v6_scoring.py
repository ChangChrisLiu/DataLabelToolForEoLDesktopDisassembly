"""V6 group-equivalent matching and sibling-free distractors (W3b).

60.7 % of the real steps have a second legal action interchangeable with the
demonstrated one -- four unordered cooler screws -- so an answer naming screw
3 when the operator took screw 4 is right. The primary match is "same unordered
group, or same verb + match class"; the exact instance is the secondary.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import w3b_scene as W
from tda.core.db import Db
from tda.core.export.vlm import export_vlm, oracle_path
from tda.core.export.vlm_check import World, gold_reply, grade
from tda.core.export.vlm_tasks import match_class
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
    tax = W.build(db, group=request.param)
    out = tmp_path / "v.jsonl"
    export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out))
    records = _read(out)
    yield db, tax, records, out, request.param
    db.close()


def _v6(records, step=None, form="open"):
    rows = [r for r in records if r["prompt"]["task"] == "V6"
            and (form is None or r["label"]["v6_form"] == form)]
    return [r for r in rows if step is None or r["label"]["step"] == step]


def test_every_frame_has_an_open_record_and_its_multiple_choice_twin(exported):
    """W3b ruling: the open question is the headline, the choice its secondary twin."""
    db, tax, records, out, _ = exported
    opens, mcs = _v6(records), _v6(records, form="mc")
    assert opens and mcs
    by_pair = {r["label"]["v6_pair"]: r for r in opens}
    assert len(by_pair) == len(opens)  # one open record per frame
    for r in opens:
        assert "options" not in r["prompt"]
        assert r["label"]["answer_check"]["options_kind"] == "open"
        assert r["label"]["v6_pair"] == r["id"]  # the open record's own id
    for r in mcs:
        twin = by_pair[r["label"]["v6_pair"]]
        assert r["prompt"]["options"] and r["label"]["answer_check"]["options_kind"] == "listed"
        assert r["label"]["answer"] == twin["label"]["answer"]
        assert r["label"]["step"] == twin["label"]["step"] and r["id"] != twin["id"]


def test_the_equivalent_set_is_the_demonstrated_action_and_its_permitted_siblings(exported):
    db, tax, records, out, grouped = exported
    # before step 4 the next action is "unscrew cooler screw 1"; all four are
    # fastened and permitted, so all four are equivalent
    [r] = _v6(records, step=3)
    check = r["label"]["answer_check"]
    # all four still to do and equivalent: the answer is the group action, and so is the
    # reference (redesign 2026-11-03); the screw the log names is `demonstrated`
    gid = r["label"]["answer"]["target"]
    assert gid in _world(out).groups
    assert check["reference"] == ["unscrew", gid]
    assert check["demonstrated"] == ["unscrew", W.SCREWS[0]]
    assert {a["target"] for a in check["equivalent_actions"]} == set(W.SCREWS) | {gid}
    assert check["match"] == {"primary": "group_equivalent", "secondary": "instance_exact"}
    assert check["match_class"] == ["unscrew", "screw.cpu_cooler"]
    assert check["group_id"] == ("cooler_screws" if grouped else None)
    assert Checker(db, tax, W.DESKTOP).check_all(records) == len(records)


def test_no_distractor_is_a_sibling_and_no_two_options_are_the_same_kind(exported):
    db, tax, records, out, _ = exported
    from tda.core.export.coco import load_ctx

    ctx = load_ctx(db, tax, W.DESKTOP, W.VIEW)
    listed = _v6(records, form="mc")
    assert listed
    for r in listed:
        check = r["label"]["answer_check"]
        equivalent = {(a["verb"], a["target"]) for a in check["equivalent_actions"]}
        kinds = [(o["verb"], match_class(ctx, o["target"])) for o in r["label"]["options"]]
        assert len(kinds) == len(set(kinds)), kinds
        for option in r["label"]["options"]:
            if option["kind"] == "permitted_unknown":
                assert (option["verb"], option["target"]) not in equivalent


def test_a_sibling_answer_matches_and_the_exact_instance_is_secondary(exported):
    db, tax, records, out, _ = exported
    world = _world(out)
    # step 4: screw 2 is next, screws 2-4 are its permitted siblings, no group action
    [r] = _v6(records, step=4)
    assert r["label"]["answer_check"]["reference"] == ["unscrew", W.SCREWS[1]]
    g = grade(r, json.dumps({"verb": "unscrew", "target": "CPU cooler screw 3"}), world, tax)
    assert g.score == 1.0 and g.metrics["group_match"] and not g.metrics["exact_match"]
    g = grade(r, json.dumps({"verb": "unscrew", "target": "a CPU cooler screw"}),
              world, tax)
    assert g.metrics["group_match"]
    g = grade(r, json.dumps({"verb": "unscrew", "target": "CPU cooler screw 2"}), world, tax)
    assert g.metrics["group_match"] and g.metrics["exact_match"]
    # step 3: the reference is the group action, which any member named exactly is
    [r] = _v6(records, step=3)
    g = grade(r, json.dumps({"verb": "unscrew", "target": "CPU cooler screw 3"}), world, tax)
    assert g.score == 1.0 and g.metrics["group_match"] and g.metrics["exact_match"]
    # a different verb on the same part is no match
    g = grade(r, json.dumps({"verb": "remove", "target": "CPU cooler screw 3"}), world, tax)
    assert g.score == 0.0


def test_a_certainly_blocked_answer_counts_as_blocked(exported):
    db, tax, records, out, _ = exported
    world = _world(out)
    [r] = _v6(records, step=3)
    # the CPU is under the cooler and locked by the lever
    g = grade(r, json.dumps({"verb": "remove", "target": "CPU 1"}), world, tax)
    assert g.score == 0.0 and g.metrics["blocked"] is True
    assert g.metrics["blocked_rate_defined"] is True


def test_listed_options_are_scored_by_letter(exported):
    db, tax, records, out, _ = exported
    world = _world(out)
    for r in _v6(records, form="mc"):
        assert grade(r, gold_reply(r, world), world, tax).score == 1.0
        wrong = next(i for i, o in enumerate(r["label"]["options"])
                     if o["kind"] in ("graph_blocked", "graph_blocked_pixels",
                                      "state_inapplicable"))
        g = grade(r, json.dumps({"choice": "ABCDE"[wrong]}), world, tax)
        assert g.score == 0.0 and g.metrics["certainly_wrong"]


def test_the_scorer_leaves_undefined_frames_out_of_the_blocked_rate(exported, tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import vlm_score

    db, tax, records, out, _ = exported
    v6 = _v6(records)
    undefined = [r for r in v6 if not r["label"]["answer_check"]["blocked_rate_defined"]]
    defined = [r for r in v6 if r["label"]["answer_check"]["blocked_rate_defined"]]
    assert defined
    preds = tmp_path / "p.jsonl"
    preds.write_text("".join(json.dumps({"id": r["id"], "raw_text": "nonsense"}) + "\n"
                             for r in records), encoding="utf-8")
    result = vlm_score.score(out, preds, resamples=50)
    # an unparsed reply counts nowhere in the blocked denominator
    assert "blocked_rate" not in result["strict"]["extra"].get("V6", {})
    world = _world(out)
    preds.write_text("".join(json.dumps({"id": r["id"], "raw_text": gold_reply(r, world)})
                             + "\n" for r in records), encoding="utf-8")
    result = vlm_score.score(out, preds, resamples=50)
    cell = result["strict"]["extra"]["V6"]["blocked_rate"]
    assert cell["n"] == len(defined) and cell["mean"] == 0.0
    assert len(undefined) + len(defined) == len(v6)
    # the headline is the open form; the multiple choice is its own row
    assert result["headline"]["v6_open"] == result["strict"]["by_task"]["V6"]
    assert result["headline"]["v6_open"]["n"] == len(v6)
    assert result["headline"]["v6_mc"]["n"] == len(_v6(records, form="mc"))
    assert result["strict"]["by_layer"]["L3"]["n"] == sum(
        1 for r in records if r["prompt"]["task"] in ("V7", "V16")) + len(v6)


def test_a_world_with_groups_grades_the_group_and_its_members(exported):
    """The redesign's contract (spec 3.1, 6.1): the screws are one group in the
    oracle; a group answer, a remaining member and the class are all the same
    answer, and a blocked action needs no ordinal to count as blocked."""
    db, tax, records, out, _ = exported
    plain = _world(out)
    gid = f"screw.cpu_cooler@{W.COOLER}"
    world = World(plain.desktop, {k: dict(v, group=gid if k in W.SCREWS else None)
                                  for k, v in plain.instances.items()}, plain.edges, plain.log,
                  {gid: {"members": list(W.SCREWS), "match_class": "screw.cpu_cooler",
                         "anchor": W.COOLER, "label": "CPU cooler screws"}})
    [r] = _v6(records, step=3)
    for name in ("the CPU cooler screws", "a cooler screw", "CPU cooler screw 3"):
        g = grade(r, json.dumps({"verb": "unscrew", "target": name}), world, tax)
        assert g.score == 1.0 and g.metrics["group_match"], name
    g = grade(r, json.dumps({"verb": "remove", "target": "the CPU"}), world, tax)
    assert g.score == 0.0 and g.metrics["blocked"] is True
    for r in _v6(records, form=None):
        assert grade(r, gold_reply(r, world), world, tax).score == 1.0


def test_a_group_demonstrated_option_is_the_right_letter(tmp_path: Path):
    """C1 (final review): with the cooler screws large enough to outline, step 3's
    multiple choice lists "the CPU cooler screws" as one option; the reference is that
    group action, the gold letter is its letter and scores 1.0, every other letter 0."""
    from tda.core.export.vlm_check import LETTERS, reference_option
    from tda.core.taxonomy import load_taxonomy

    db = Db(str(tmp_path / "v6.sqlite"))
    tax = W.build(db, big_screws=True)
    out = tmp_path / "v.jsonl"
    export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out))
    records = _read(out)
    world = _world(out)
    [mc] = _v6(records, step=3, form="mc")
    check, options = mc["label"]["answer_check"], mc["label"]["options"]
    [shown] = [o for o in options if o["kind"] == "demonstrated"]
    assert world.is_group(shown["target"]) and sorted(shown["members"]) == sorted(W.SCREWS)
    assert tuple(check["reference"]) == (shown["verb"], shown["target"])
    assert check["demonstrated"] == ["unscrew", W.SCREWS[0]]
    assert {"verb": "unscrew", "target": shown["target"]} in check["equivalent_actions"]
    right = options.index(shown)
    assert reference_option(mc["label"]) == right
    assert json.loads(gold_reply(mc, world)) == {"choice": LETTERS[right]}
    tax = load_taxonomy()
    for i in range(len(options)):
        g = grade(mc, json.dumps({"choice": LETTERS[i]}), world, tax)
        assert g.score == (1.0 if i == right else 0.0), (i, options[i], g.metrics)
    assert grade(mc, json.dumps({"choice": LETTERS[right]}), world, tax).metrics["exact_match"]
    # a record of the old shape (reference = the member, the option = its group) is
    # graded the same: the option whose members hold the reference
    old = json.loads(json.dumps(mc))
    old["label"]["answer_check"]["reference"] = ["unscrew", W.SCREWS[0]]
    old["label"]["answer_check"]["equivalent_actions"] = [
        a for a in check["equivalent_actions"] if a["target"] != shown["target"]]
    assert json.loads(gold_reply(old, world)) == {"choice": LETTERS[right]}
    assert grade(old, gold_reply(old, world), world, tax).score == 1.0
    # every gold reply of the export scores 1, and the independent checker agrees
    for r in records:
        assert grade(r, gold_reply(r, world), world, tax).score == 1.0, r["label"]["readable_id"]
    assert Checker(db, tax, W.DESKTOP).check_all(records) == len(records)
    db.close()
