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
    assert old.ground_group("the CPU cooler screws", list(SCREWS)) == SCREWS[0]


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
