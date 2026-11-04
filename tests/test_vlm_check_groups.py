"""The grader on part groups (VLM question redesign, lane B).

A hand-built oracle world in the redesign's record contract -- no database,
no export: the generator of the contract (lane A) is tested on its own, and
this file only needs the shapes. One desktop with

* a heatsink-fan CPU cooler held by four captive screws (one group,
  ``screw.cpu_cooler@cpu_cooler.01``), plugged in by a fan lead (a single part,
  named by its key) and carrying a shroud attached to it;
* the CPU under the cooler, locked by its socket lever;
* two RAM modules (one anchor-less group ``ram_module@-``), each locked by its
  own two latches -- two groups of one class, ``ram_latch@ram_module.01`` and
  ``ram_latch@ram_module.02``, both called "RAM latches".

Every gold target below is a group id or an instance key; every model answer
is words.
"""
from __future__ import annotations

import json

import pytest

from tda.core.export.vlm_check import World, gold_reply, grade, grounder
from tda.core.taxonomy import load_taxonomy

DESKTOP = 7
COOLER, SHROUD, FAN, CPU, LEVER, BOARD = ("cpu_cooler.01", "cover.01", "connector.fan.01",
                                          "cpu.01", "cpu_socket_lever.01", "motherboard.01")
SCREWS = tuple(f"screw.cpu_cooler.{i:02d}" for i in range(1, 5))
RAMS = ("ram_module.01", "ram_module.02")
LATCHES = {RAMS[0]: ("ram_latch.01", "ram_latch.02"), RAMS[1]: ("ram_latch.03", "ram_latch.04")}
G_SCREWS = f"screw.cpu_cooler@{COOLER}"
G_RAMS = "ram_module@-"
G_LATCH = {ram: f"ram_latch@{ram}" for ram in RAMS}


def _inst(cls, label, mclass=None, attrs=None, parent=None, attached=False, group=None):
    return {"class": cls, "label": label, "match_class": mclass or cls, "attrs": attrs or {},
            "parent": parent, "attached": attached, "group_id": None,
            "group_order": "unordered", "group": group}


def oracle_line(*, groups: bool = True) -> dict:
    """The desktop line of the oracle sidecar; ``groups=False`` is the old shape."""
    inst = {
        BOARD: _inst("motherboard", "Motherboard 1"),
        CPU: _inst("cpu", "CPU 1"),
        LEVER: _inst("cpu_socket_lever", "CPU socket lever 1", parent=BOARD, attached=True),
        COOLER: _inst("cpu_cooler", "Heatsink fan CPU cooler 1", "cpu_cooler.heatsink_fan",
                      {"kind": "heatsink_fan"}),
        SHROUD: _inst("cover", "Cover 1", attrs={"of": "cpu_cooler"}, parent=COOLER,
                      attached=True),
        FAN: _inst("connector", "Fan connector 1", "connector.fan", {"kind": "fan"}),
    }
    for i, s in enumerate(SCREWS, 1):
        inst[s] = _inst("screw", f"CPU cooler screw {i}", "screw.cpu_cooler",
                        {"role": "cpu_cooler", "captive": True}, parent=COOLER, attached=True,
                        group=G_SCREWS)
    for i, ram in enumerate(RAMS, 1):
        inst[ram] = _inst("ram_module", f"RAM module {i}", group=G_RAMS)
        for latch in LATCHES[ram]:
            inst[latch] = _inst("ram_latch", f"RAM latch {latch[-1]}", attrs={"of": ram},
                                parent=BOARD, attached=True, group=G_LATCH[ram])
    edges = [*({"type": "fastened_by", "target": COOLER, "blocker": s} for s in SCREWS),
             {"type": "connected_to", "target": COOLER, "blocker": FAN},
             {"type": "covered_by", "target": CPU, "blocker": COOLER},
             {"type": "locked_by", "target": CPU, "blocker": LEVER},
             *({"type": "locked_by", "target": ram, "blocker": l}
               for ram in RAMS for l in LATCHES[ram])]
    line = {"type": "desktop", "desktop": DESKTOP, "instances": inst, "edges": edges, "log": []}
    if groups:
        line["groups"] = {
            G_SCREWS: {"members": list(SCREWS), "match_class": "screw.cpu_cooler",
                       "anchor": COOLER, "label": "CPU cooler screws"},
            G_RAMS: {"members": list(RAMS), "match_class": "ram_module", "anchor": "-",
                     "label": "RAM modules"},
            **{G_LATCH[ram]: {"members": list(LATCHES[ram]), "match_class": "ram_latch",
                              "anchor": ram, "label": "RAM latches"} for ram in RAMS},
        }
    else:
        for row in inst.values():
            row.pop("group")
    return line


@pytest.fixture(scope="module")
def world() -> World:
    return World.from_oracle(oracle_line())


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def _record(task: str, answer: dict, check: dict, **label) -> dict:
    return {"id": f"rec_{task}_{len(json.dumps(answer))}",
            "prompt": {"task": task, "images": ["img_0"], "question": "?"},
            "label": {"desktop": DESKTOP, "view": "multi", "step": 3, "answer": answer,
                      "answer_check": check, "template_id": f"{task}.t0", **label}}


def _v10(done, started=(), progress="0-25") -> dict:
    return _record("V10", {"done": [{"verb": v, "target": t} for v, t in done],
                           "started": [{"verb": v, "target": t} for v, t in started],
                           "remaining_actions": 9, "progress_bin": progress},
                   {"type": "history", "bins": ["0-25", "25-50", "50-75", "75-100"],
                    "metric": "set_f1+bin_accuracy"})


def _reply(**fields) -> str:
    return json.dumps(fields)


# --------------------------------------------------------------------------- #
# B1: grounding to groups
# --------------------------------------------------------------------------- #
def test_world_reads_the_groups(world):
    assert set(world.groups) == {G_SCREWS, G_RAMS, *G_LATCH.values()}
    assert world.group_of(SCREWS[2]) == G_SCREWS
    assert world.group_of(LATCHES[RAMS[1]][0]) == G_LATCH[RAMS[1]]
    assert world.group_of(FAN) is None and world.group_of("nothing.01") is None
    # a group id answers like a part of its class
    assert world.match_class(G_SCREWS) == "screw.cpu_cooler"
    assert world.cls_of(G_SCREWS) == "screw" and world.cls_of(G_RAMS) == "ram_module"
    assert world.members(G_SCREWS) == list(SCREWS) and world.members(FAN) == [FAN]
    # the old shape: no groups, every lookup answers "none"
    old = World.from_oracle(oracle_line(groups=False))
    assert old.groups == {} and old.group_of(SCREWS[0]) is None


@pytest.mark.parametrize("text", ["the cooler screws", "the CPU cooler screws",
                                  "CPU cooler screws", "a CPU cooler screw", "cooler screw",
                                  "CPU cooler screw 3", SCREWS[1]])
def test_plural_and_singular_ground_to_group(world, text):
    g = grounder(world)
    assert g.ground_group(text, [FAN, G_SCREWS]) == G_SCREWS
    # the class is found whether or not the group is eligible
    assert g.ground(text)[1] == "screw.cpu_cooler"


def test_instance_key_still_grounds(world):
    g = grounder(world)
    for text in (FAN, "Fan connector 1", "the fan connector", "fan connectors"):
        assert g.ground_group(text, [G_SCREWS, FAN]) == FAN, text
    # a member named exactly stays itself where its group is not asked for
    assert g.ground_group("CPU cooler screw 2", [FAN]) == SCREWS[1]
    assert g.ground_group(SCREWS[1], []) == SCREWS[1]
    # a class phrase nothing eligible has grounds to nothing
    assert g.ground_group("the CPU cooler screws", [FAN]) is None
    assert g.ground_group("a flux capacitor", [FAN, G_SCREWS]) is None
    # the old world: exactly the old reading (first eligible member of the class)
    old = grounder(World.from_oracle(oracle_line(groups=False)))
    assert old.ground_group("a CPU cooler screw", list(SCREWS[2:])) == SCREWS[2]
    assert old.ground_group("CPU cooler screw 2", list(SCREWS[2:])) == SCREWS[1]
    assert old.ground_group("the CPU cooler screws", list(SCREWS)) is None  # base: class cpu


def test_old_world_keeps_the_old_plural_reading():
    """No groups in the oracle: words are not put in the singular, so a plural
    wording grounds as it always did (review fix, lane B round 1)."""
    old = grounder(World.from_oracle(oracle_line(groups=False)))
    assert old.ground("the CPU cooler screws") == (None, "cpu")
    for text in ("the RAM latches", "RAM modules", "covers", "the fan connectors",
                 "socket levers"):
        assert old.ground(text) == (None, None), text
    # the singular wording is unchanged too
    assert old.ground("a RAM latch") == (None, "ram_latch")
    assert old.ground("CPU cooler screw 2") == (SCREWS[1], "screw.cpu_cooler")
    # and graded the old way: a plural blocker names nothing on an old world
    keyed = _v4("remove", RAMS[0], list(LATCHES[RAMS[0]]))
    g = grade(keyed, _reply(feasible=False, blockers=["the RAM latches"]),
              World.from_oracle(oracle_line(groups=False)))
    assert g.metrics["blockers_f1"] == 0.0
    # the same words on a world with groups are the latch group
    assert grounder(World.from_oracle(oracle_line())).ground("the RAM latches") == \
        (None, "ram_latch")


def test_two_groups_one_class_graded_by_count(world):
    """Spec 6.2: answers name the class, not the anchor -- the count of group
    actions per class is what is compared."""
    g = grounder(world)
    both = [G_LATCH[RAMS[0]], G_LATCH[RAMS[1]]]
    assert g.ground_group("the RAM latches", both) == both[0]
    assert g.ground_group("the RAM latches", both[1:]) == both[1]
    assert g.ground_group("a RAM latch", both[1:]) == both[1]
    record = _v10([("open", both[0]), ("open", both[1])])
    once = grade(record, _reply(done=[{"verb": "open", "target": "the RAM latches"}],
                                progress="0-25"), world)
    assert once.score == pytest.approx(2 / 3)
    twice = grade(record, _reply(done=[{"verb": "open", "target": "the RAM latches"}] * 2,
                                 progress="0-25"), world)
    assert twice.score == 1.0
    thrice = grade(record, _reply(done=[{"verb": "open", "target": "RAM latches"}] * 3,
                                  progress="0-25"), world)
    # a third names no third group of the class: a repeat, not a new action
    assert thrice.score == 1.0
    assert twice.metrics["bin_correct"] is True


# --------------------------------------------------------------------------- #
# B2: the task grades
# --------------------------------------------------------------------------- #
def _v4(verb, target, blockers) -> dict:
    return _record("V4", {"feasible": not blockers, "blockers": list(blockers),
                          "reason": "violates_edge" if blockers else None},
                   {"type": "feasibility", "verb": verb, "target": target,
                    "metric": "per_verb_accuracy_macro"},
                   truth_source="graph_blocked" if blockers else "demonstrated",
                   **({"negative": "graph_blocked"} if blockers else {}))


def _blockers(record, world, *names, feasible=False):
    g = grade(record, _reply(feasible=feasible, blockers=list(names)), world)
    return g.metrics.get("blockers_f1"), g.score


def test_v4_group_blockers_f1(world):
    fan = _v4("remove", COOLER, [G_SCREWS, FAN])
    assert _blockers(fan, world, "the CPU cooler screws", "the fan connector") == (1.0, 1.0)
    assert _blockers(fan, world, "fan connector", "a cooler screw") == (1.0, 1.0)
    # a member named exactly is its group's requirement
    assert _blockers(fan, world, "CPU cooler screw 2", "Fan connector 1") == (1.0, 1.0)
    assert _blockers(fan, world, "the cooler screws")[0] == pytest.approx(2 / 3)
    # a part that is no blocker costs precision, whether named by class or exactly
    assert _blockers(fan, world, "the cooler screws", "the fan connector",
                     "the CPU")[0] == pytest.approx(0.8)
    assert _blockers(fan, world, "the cooler screws", "the fan connector",
                     "CPU 1")[0] == pytest.approx(0.8)
    # words that name no part are no credit and no cost
    assert _blockers(fan, world, "the cooler screws", "the fan connector",
                     "the flux capacitor")[0] == 1.0
    assert _blockers(fan, world, feasible=True) == (0.0, 0.0)
    # two latch groups of one class: the count is what is compared
    rams = _v4("remove", G_RAMS, list(G_LATCH.values()))
    assert _blockers(rams, world, "the RAM latches")[0] == pytest.approx(2 / 3)
    assert _blockers(rams, world, "the RAM latches", "the RAM latches")[0] == 1.0
    # the old shape grades as it always did: one screw of four named
    old = World.from_oracle(oracle_line(groups=False))
    keyed = _v4("remove", COOLER, [*SCREWS, FAN])
    assert _blockers(keyed, old, "a CPU cooler screw", "a CPU cooler screw")[0] == \
        pytest.approx(2 * 0.2 / 1.2)
    assert _blockers(keyed, old, "the CPU", *[f"CPU cooler screw {i}" for i in range(1, 5)],
                     "fan connector")[0] == 1.0


def _start_state(world, tax) -> dict:
    return {k: tax.default_state(inst["class"]) for k, inst in world.instances.items()}


PLAN = [("disconnect", FAN), ("unscrew", G_SCREWS), ("remove", COOLER), ("open", LEVER),
        ("remove", CPU)]


def _v7(world, tax) -> dict:
    plan = [{"verb": v, "target": t} for v, t in PLAN]
    return _record("V7", {"plan": plan, "goal": CPU},
                   {"type": "goal_plan", "goal": CPU, "goal_state": "removed",
                    "start_state": _start_state(world, tax), "plan_length": len(plan),
                    "reference_shortest": plan, "demonstrated_remainder": plan,
                    "required_actions": [
                        {"target": CPU, "states": ["removed"]},
                        {"target": COOLER, "states": ["displaced", "open", "removed", "unplugged"]},
                        {"target": LEVER, "states": ["open"]},
                        {"target": G_SCREWS, "states": ["loosened", "removed"]}],
                    "metric": "violation_free+required_recall+goal_reached",
                    "any_legal_plan_scores": True, "redundancy_penalty": False})


def _plan(*steps) -> str:
    return _reply(plan=[{"verb": v, "target": t} for v, t in steps])


def test_v7_group_plan_violation_free(world, tax):
    r = _v7(world, tax)
    words = [("disconnect", "the fan connector"), ("unscrew", "the CPU cooler screws"),
             ("remove", "the CPU cooler"), ("open", "the CPU socket lever"), ("remove", "the CPU")]
    g = grade(r, _plan(*words), world, tax)
    assert g.score == 1.0 and g.metrics["violation_free"] and g.metrics["required_recall"] == 1.0
    # one group action unscrews every member, however it is worded
    for name in ("a CPU cooler screw", "cooler screws", "CPU cooler screw 4"):
        steps = list(words)
        steps[1] = ("unscrew", name)
        assert grade(r, _plan(*steps), world, tax).score == 1.0, name
    # said again, the group action changes nothing: a no-op, no penalty
    g = grade(r, _plan(words[0], words[1], words[1], *words[2:]), world, tax)
    assert g.score == 1.0 and g.metrics["noop_steps"] == 1
    # without the screws the cooler cannot come off: a violation at step 2
    g = grade(r, _plan(words[0], *words[2:]), world, tax)
    assert g.score == 0.0 and g.metrics["violations"] == 1
    assert g.metrics["first_violation"] == 1 and g.metrics["goal_reached"]
    # a required group is met only when every member is
    g = grade(r, _plan(*words[:2]), world, tax)
    assert g.metrics["required_recall"] == pytest.approx(0.25)
    assert not g.metrics["goal_reached"] and g.score == 0.0


def test_v10_started_set(world):
    r = _v10([("open", G_LATCH[RAMS[0]]), ("disconnect", FAN)],
             started=[("unscrew", G_SCREWS)])
    done = [{"verb": "open", "target": "the RAM latches"},
            {"verb": "disconnect", "target": "the fan connector"}]
    started = [{"verb": "unscrew", "target": "the CPU cooler screws"}]
    g = grade(r, _reply(done=done, started=started, progress="0-25"), world)
    assert g.score == 1.0 and g.metrics["started_f1"] == 1.0
    # a group only begun is not done: in the done list it is a false positive
    g = grade(r, _reply(done=done + started, started=[], progress="0-25"), world)
    assert g.score == pytest.approx(0.8) and g.metrics["started_f1"] == 0.0
    # started is its own set: a wrong one leaves the done score alone
    g = grade(r, _reply(done=done, started=[{"verb": "open", "target": "the RAM latches"}],
                        progress="0-25"), world)
    assert g.score == 1.0 and g.metrics["started_f1"] == 0.0
    # nothing started, nothing said: right
    none = _v10([("disconnect", FAN)])
    g = grade(none, _reply(done=done[1:], started=[], progress="0-25"), world)
    assert g.score == 1.0 and g.metrics["started_f1"] == 1.0
    # an old record (no `started`) has no started metric
    old = _v10([("disconnect", FAN)])
    del old["label"]["answer"]["started"]
    assert "started_f1" not in grade(old, _reply(done=done[1:], progress="0-25"),
                                     world).metrics


def _v3(verb, target, cls, accept=None) -> dict:
    check = {"type": "exact", "derive": "action_slots",
             "fields": ["verb", "target_class", "target_instance", "tool"], "visible_only": True}
    if accept is not None:
        check["accept_targets"] = list(accept)
    return _record("V3", {"verb": verb, "target_class": cls, "target_instance": target,
                          "tool": None}, check)


def test_v3_accept_targets(world):
    # owner 2026-11-03: the part acted on is the answer, never a part attached to it --
    # even where an older record lists it in accept_targets
    for accept in ([COOLER], [COOLER, SHROUD], None):
        r = _v3("remove", COOLER, "cpu_cooler", accept=accept)
        for name in ("the CPU cooler", COOLER):
            g = grade(r, _reply(verb="remove", target=name), world)
            assert g.score == 1.0 and g.metrics["target_instance"], (accept, name)
        for name in ("the cover", "Cover 1", SHROUD):
            g = grade(r, _reply(verb="remove", target=name), world)
            assert g.score == 0.0 and not g.metrics["target_instance"], (accept, name)
        assert grade(r, _reply(verb="remove", target="the CPU"), world).score == 0.0
        assert grade(r, _reply(verb="open", target="the CPU cooler"), world).score == 0.0
    # a group target, named in words
    screws = _v3("unscrew", G_SCREWS, "screw", accept=[G_SCREWS])
    for name in ("the CPU cooler screws", "a cooler screw", "CPU cooler screw 1"):
        g = grade(screws, _reply(verb="unscrew", target=name), world)
        assert g.score == 1.0 and g.metrics["target_instance"], name


SCREW_BOXES = [[10, 10, 4, 4], [30, 10, 4, 4], [10, 30, 4, 4], [30, 30, 4, 4]]


def _v8_components() -> dict:
    return _record("V8", {"components": [
        {"class": "screw", "instance": s, "bbox": b} for s, b in zip(SCREWS, SCREW_BOXES)]},
        {"type": "components", "class_filter": "screw.cpu_cooler", "field": "components",
         "box_field": "bbox", "iou_threshold": 0.5, "metric": "set_f1+box_iou",
         "geometry": "visible_mask_bbox"})


def test_v8_components_grade(world):
    from tda.core.export.vlm_check import format_instruction, format_key

    r = _v8_components()
    assert format_key(r) == "V8.components"
    assert '"components"' in format_instruction(r)

    def boxes(*bs, cls=None):
        return _reply(components=[dict({"bbox": b}, **({"class": cls} if cls else {}))
                                  for b in bs])

    g = grade(r, boxes(*SCREW_BOXES), world)
    assert g.score == 1.0 and g.metrics["tp"] == 4
    # the question names the class: a class word in the answer is not needed, nor read
    assert grade(r, boxes(*SCREW_BOXES, cls="screw"), world).score == 1.0
    assert grade(r, boxes(*SCREW_BOXES[:2]), world).score == pytest.approx(2 / 3)
    assert grade(r, boxes(*SCREW_BOXES, [50, 50, 8, 8]), world).score == pytest.approx(8 / 9)
    assert grade(r, boxes(), world).score == 0.0
    # a box answered on a resized image is mapped back
    half = [[x / 2, y / 2, w / 2, h / 2] for x, y, w, h in SCREW_BOXES]
    assert grade(r, boxes(*half), world, transform={"scale": 0.5}).score == 1.0
    # truth outside the class filter is not asked for
    extra = _v8_components()
    extra["label"]["answer"]["components"].append(
        {"class": "connector", "instance": FAN, "bbox": [40, 4, 6, 6]})
    assert grade(extra, boxes(*SCREW_BOXES), world).score == 1.0
    # the single-box V8 is unchanged
    single = _record("V8", {"instance": FAN, "bbox": [40, 4, 6, 6]},
                     {"type": "boxes", "iou_threshold": 0.5, "metric": "box_iou"})
    assert format_key(single) == "V8"
    assert grade(single, _reply(bbox=[40, 4, 6, 6]), world).score == 1.0


def test_v6_group_target(world):
    r = _record("V6", {"verb": "unscrew", "target": G_SCREWS, "target_class": "screw"},
                {"type": "next_action", "reference": ["unscrew", G_SCREWS],
                 "equivalent_actions": [{"verb": "unscrew", "target": G_SCREWS}],
                 "match_class": ["unscrew", "screw.cpu_cooler"],
                 "blocked_actions": [{"verb": "remove", "target": CPU},
                                     {"verb": "remove", "target": COOLER}],
                 "blocked_rate_defined": True}, v6_form="open")
    for name in ("the CPU cooler screws", "a cooler screw", "CPU cooler screw 3"):
        g = grade(r, _reply(verb="unscrew", target=name), world)
        assert g.score == 1.0 and g.metrics["group_match"] and g.metrics["exact_match"], name
    # the words of a blocked action are blocked, without an ordinal
    g = grade(r, _reply(verb="remove", target="the CPU"), world)
    assert g.score == 0.0 and g.metrics["blocked"] is True
    g = grade(r, _reply(verb="disconnect", target="the fan connector"), world)
    assert g.score == 0.0 and g.metrics["blocked"] is False
    # equivalent members, no group id in the gold: any remaining member, or the group
    members = _record("V6", {"verb": "unscrew", "target": SCREWS[1]},
                      {"type": "next_action", "reference": ["unscrew", SCREWS[1]],
                       "equivalent_actions": [{"verb": "unscrew", "target": s}
                                              for s in SCREWS[1:]],
                       "match_class": ["unscrew", "screw.cpu_cooler"],
                       "blocked_rate_defined": False}, v6_form="open")
    for name in ("the CPU cooler screws", "CPU cooler screw 4", "a cooler screw"):
        assert grade(members, _reply(verb="unscrew", target=name), world).score == 1.0, name


def _fixture_records(world, tax) -> list[dict]:
    v6_mc = _record("V6", {"verb": "unscrew", "target": G_SCREWS},
                    {"type": "next_action", "reference": ["unscrew", G_SCREWS],
                     "equivalent_actions": [{"verb": "unscrew", "target": G_SCREWS}],
                     "match_class": ["unscrew", "screw.cpu_cooler"],
                     "blocked_rate_defined": True},
                    v6_form="mc", options=[
                        {"verb": "remove", "target": CPU, "kind": "graph_blocked"},
                        {"verb": "unscrew", "target": G_SCREWS, "kind": "demonstrated"}])
    v6_mc["prompt"]["options"] = [{"verb": o["verb"], "target_label": world.label_of(o["target"])}
                                  for o in v6_mc["label"]["options"]]
    v16 = _record("V16", {"valid": False, "first_error_index": 1, "violated_edges": [],
                          "reason": "violates_edge"},
                  {"type": "plan", "plan": [{"verb": "unscrew", "target": G_SCREWS},
                                            {"verb": "remove", "target": CPU}]})
    v16_ok = _record("V16", {"valid": True, "first_error_index": None},
                     {"type": "plan", "plan": [{"verb": "open", "target": G_LATCH[RAMS[1]]}]})
    records = [
        _v3("remove", COOLER, "cpu_cooler", accept=[COOLER, SHROUD]),
        _v3("unscrew", G_SCREWS, "screw", accept=[G_SCREWS]),
        _v4("remove", COOLER, [G_SCREWS, FAN]),
        _v4("remove", G_RAMS, list(G_LATCH.values())),
        _v4("disconnect", FAN, []),
        _record("V6", {"verb": "unscrew", "target": G_SCREWS},
                {"type": "next_action", "reference": ["unscrew", G_SCREWS],
                 "equivalent_actions": [{"verb": "unscrew", "target": G_SCREWS}],
                 "match_class": ["unscrew", "screw.cpu_cooler"], "blocked_rate_defined": True},
                v6_form="open"),
        v6_mc,
        _v7(world, tax),
        _v8_components(),
        _v10([("open", G_LATCH[RAMS[0]]), ("disconnect", FAN)], started=[("unscrew", G_SCREWS)]),
        _v10([("open", G_LATCH[RAMS[0]]), ("open", G_LATCH[RAMS[1]])], progress="25-50"),
        v16, v16_ok,
    ]
    for i, r in enumerate(records):
        r["id"] = f"rec_{i:02d}_{r['prompt']['task']}"
    return records


def test_gold_scores_one(world, tax):
    for r in _fixture_records(world, tax):
        for mode in ("strict", "lenient"):
            g = grade(r, gold_reply(r, world), world, tax, mode=mode)
            assert g.score == 1.0, (r["id"], mode, g.metrics)
            for key in ("blockers_f1", "started_f1", "required_recall"):
                assert g.metrics.get(key, 1.0) == 1.0, (r["id"], key, g.metrics)
        # the gold reply names parts in words: no key, no group id
        text = gold_reply(r, world)
        assert "@" not in text and "screw.cpu_cooler." not in text, text


def test_the_scorer_on_group_records(world, tax, tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import vlm_io as IO
    import vlm_score

    records = _fixture_records(world, tax)
    truth = tmp_path / "vlm_multi.jsonl"
    IO.write_jsonl(truth, records)
    IO.write_jsonl(IO.sidecar(truth, "oracle.jsonl"), [oracle_line()])
    preds = tmp_path / "gold.jsonl"
    IO.write_jsonl(preds, [{"id": r["id"], "raw_text": gold_reply(r, world)} for r in records])
    result = vlm_score.score(truth, preds, resamples=20)
    for task, cell in result["strict"]["by_task"].items():
        assert cell["mean"] == 1.0, task
    assert result["strict"]["extra"]["V10"]["started_f1"]["mean"] == 1.0
    assert result["strict"]["extra"]["V10"]["started_f1"]["n"] == 2
    assert result["strict"]["extra"]["V4"]["blockers_f1"]["mean"] == 1.0


# --------------------------------------------------------------------------- #
# final review C2: an exact name spends its target; class phrases in gold order
# --------------------------------------------------------------------------- #
PSU_PLUGS = ("connector.01", "connector.02")
SSD_PLUGS = ("connector.03", "connector.04")
LONE = "connector.05"
G_PSU, G_SSD = "connector@motherboard.01|cable:psu", "connector@storage_drive.01|cable:storage_drive"


def _connector_world() -> World:
    """Two connector groups of one class told apart by their anchors' words, a lone
    connector of the same class, and the two RAM latch groups of the base world."""
    line = oracle_line()
    inst = line["instances"]
    inst["storage_drive.01"] = _inst("storage_drive", "SSD storage drive 1")
    for i, key in enumerate([*PSU_PLUGS, *SSD_PLUGS, LONE], 1):
        group = G_PSU if key in PSU_PLUGS else G_SSD if key in SSD_PLUGS else None
        inst[key] = _inst("connector", f"Connector {i}", group=group)
    line["groups"][G_PSU] = {"members": list(PSU_PLUGS), "match_class": "connector",
                             "anchor": "motherboard.01|cable:psu",
                             "label": "PSU cable connectors"}
    line["groups"][G_SSD] = {"members": list(SSD_PLUGS), "match_class": "connector",
                             "anchor": "storage_drive.01|cable:storage_drive",
                             "label": "storage drive cable connectors"}
    return World.from_oracle(line)


def test_an_exact_name_spends_its_target_before_a_class_phrase():
    world = _connector_world()
    # the gold: the lone connector by its name, then the two groups
    r = _v10([("disconnect", LONE), ("disconnect", G_PSU), ("disconnect", G_SSD),
              ("open", G_LATCH[RAMS[0]]), ("open", G_LATCH[RAMS[1]])])
    g = grade(r, gold_reply(r, world), world)
    assert g.score == 1.0, g.metrics
    # "connectors" twice after "Connector 5": the two groups, never the lone one again
    said = [{"verb": "disconnect", "target": t} for t in ("Connector 5", "connectors",
                                                          "connectors")]
    said += [{"verb": "open", "target": "RAM latches"}] * 2
    assert grade(r, _reply(done=said, progress="0-25"), world).score == 1.0
    # a member named exactly stands for its group and spends it
    said[1] = {"verb": "disconnect", "target": "Connector 3"}
    assert grade(r, _reply(done=said, progress="0-25"), world).score == 1.0


def test_a_class_phrase_takes_the_group_its_words_fit():
    world = _connector_world()
    g = grounder(world)
    assert g.ground_group("the storage drive cable connectors", [G_PSU, G_SSD]) == G_SSD
    assert g.ground_group("PSU cable connectors", [G_SSD, G_PSU]) == G_PSU
    assert g.ground_group("connectors", [G_SSD, G_PSU]) == G_SSD  # no word tells: first
    # V4 blockers: an exact name claims its gold target, a phrase after it the other one
    r = _v4("remove", "storage_drive.01", [G_SSD, G_PSU])
    assert _blockers(r, world, "Connector 3", "connectors")[0] == 1.0
    assert _blockers(r, world, "PSU cable connectors", "storage drive cable connectors")[0] == 1.0
    assert _blockers(r, world, *[world.label_of(b) for b in (G_SSD, G_PSU)])[0] == 1.0


def test_v10_grounds_in_the_gold_order():
    world = _connector_world()
    # the same label done and begun: the done list's phrase is the done group
    r = _v10([("open", G_LATCH[RAMS[1]])], started=[("open", G_LATCH[RAMS[0]])])
    g = grade(r, gold_reply(r, world), world)
    assert g.score == 1.0 and g.metrics["started_f1"] == 1.0, g.metrics
