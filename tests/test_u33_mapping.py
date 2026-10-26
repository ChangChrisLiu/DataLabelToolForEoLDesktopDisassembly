"""Task U33: a log row that removes a loose cable is not a cable clip.

D36 step 2, "Remove 1 isolated cable from motherboard", was imported as
``cable_clip.01`` + ``release`` -- a clip the card then asked the annotator to
draw. A cable is no annotated part (spec 6.2: the virtual node ``cable``, no
geometry), so the row now targets ``cable:isolated.01`` with the verb
``remove`` and makes no instance. A row that says what held the cable -- "remove
the cable from the case fan clip", "... from cable locker 1", D63's "Remove
cable from heatsink cover" -- still opens a clip.

Pinned here: the rule on positive and negative phrasings, the virtual node's
new state ``removed`` (and ``release`` unchanged), the D36 fixture (a verbatim
copy of the real sheet), and the survey of every cable row of the 66 sheets --
re-run against the real sheets when they are on this machine.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tda.core.graph import legal_actions
from tda.core.graph_rules import cable_owner, verb_applies
from tda.core.logs import (
    LOOSE_CABLE_OWNER,
    import_log,
    is_cable_event,
    iter_desktop_csvs,
    loose_cable_key,
    read_desktop_csv,
)
from tda.core.states import (
    event_problems,
    events_from_actions,
    needs_geom,
    state_at,
)
from tda.core.taxonomy import load_taxonomy, parse_raw_name

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "logs"
COLUMNS = ("Sequence Number", "Sequence Name", "Target Class Label", "Target Nest Group",
           "Tool Utility", "Notes")
D36_ROW = "Remove 1 isolated cable from motherboard"
NODE = "cable:isolated.01"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def _sheet(names: list[str]) -> list[dict]:
    rows = []
    for i, name in enumerate(["Initial Conditions", *names], start=1):
        row = {c: "" for c in COLUMNS}
        row["Sequence Number"], row["Sequence Name"] = str(i), name
        rows.append(row)
    return rows


def _one(name: str, tax):
    """The single action a one-row sheet (after the initial row) drafts."""
    li = import_log(99, _sheet([name]), {}, tax)
    [action] = li.actions
    return li, action


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
CABLE_EVENTS = [
    D36_ROW,
    "Remove isolated cable",
    "remove the cable",
    "Remove cables",
    "Take out the SATA cable",
    "take the cable off the motherboard",
    "Pull the cable out",
    "pull out cable from chassis",
]


@pytest.mark.parametrize("name", CABLE_EVENTS)
def test_a_cable_taken_out_is_a_cable_event(name, tax):
    parsed = parse_raw_name(name)
    assert parsed.matched_rule == "cable_removed"
    assert parsed.cls == "" and parsed.virtual == "cable" and parsed.verb == "remove"
    assert is_cable_event(parsed)
    li, action = _one(name, tax)
    assert (action.target, action.verb, action.result) == (NODE, "remove", "success")
    assert li.steps[1].step_type == "normal"
    # no part: only the implicit chassis, and nothing for S1 to answer
    assert set(li.instances) == {"chassis"}
    assert not li.issues


#: name -> (rule, class of the target) -- every one of them says what held the
#: cable, opens something else, or is no removal at all
NOT_CABLE_EVENTS = {
    "remove the cable from the case fan clip": ("cable_routing", "cable_clip"),
    "remove cable from cable locker 1": ("cable_routing", "cable_clip"),
    "Remove cables from cable locker": ("cable_routing", "cable_clip"),
    "Remove cable from two cable locker": ("cable_routing", "cable_clip"),
    "Cable locker on heatsink": ("cable_routing", "cable_clip"),
    "Cable locker for power module cables on CPU fan": ("cable_routing", "cable_clip"),
    "Remove cable from heatsink cover": ("cable_routing", "cable_clip"),
    "Remove cable from fan shroud": ("cable_routing", "cable_clip"),
    "Remove the cable (from clip)": ("cable_routing", "cable_clip"),
    "Moving cables to show power module - optical drive connector":
        ("action_auxiliary_reveal", None),
    "Try to remove power module (failed) due to the cable on the motherboard":
        ("psu_attempt", "psu"),
}


@pytest.mark.parametrize("name", sorted(NOT_CABLE_EVENTS))
def test_a_cable_freed_from_what_held_it_is_not_a_cable_event(name, tax):
    rule, cls = NOT_CABLE_EVENTS[name]
    parsed = parse_raw_name(name)
    assert parsed.matched_rule == rule
    assert not is_cable_event(parsed)
    li, action = _one(name, tax)
    assert not action.target.startswith("cable:")
    assert (li.instances[action.target].cls if action.target in li.instances else None) == cls
    if cls == "cable_clip":
        assert action.verb == "release"


@pytest.mark.parametrize("name", ["Remove cable tie", "pull the cable out of the holder",
                                  "Remove cable connector", "release the cable"])
def test_a_holder_a_tie_a_plug_or_another_verb_is_never_a_cable_event(name):
    parsed = parse_raw_name(name)
    assert parsed.matched_rule != "cable_removed" and not is_cable_event(parsed)


def test_two_loose_cables_are_two_nodes(tax):
    li = import_log(99, _sheet(["Remove isolated cable", "Heatsink screw 1",
                                "Pull the cable out"]), {}, tax)
    targets = [a.target for a in li.actions]
    assert targets == [loose_cable_key(1), "screw.cpu_cooler.01", loose_cable_key(2)]
    assert loose_cable_key(2) == "cable:isolated.02"


def test_the_node_belongs_to_no_part(tax):
    """Spec 3.1's 「独立」 owner: no class, so no instance, behind the token."""
    assert LOOSE_CABLE_OWNER == "isolated"
    owner = cable_owner(NODE)
    assert owner not in tax.classes and owner not in tax.virtual_nodes
    rows, meta = read_desktop_csv(FIXTURES / "desktop_36.csv")
    assert cable_owner(NODE, import_log(36, rows, meta, tax).instances) is None


def test_the_canon_groups_stay_53():
    """The new rule reuses the csv's group for these rows (test_taxonomy pins 53)."""
    cfg = yaml.safe_load((ROOT / "configs" / "taxonomy_map.yaml").read_text(encoding="utf-8"))
    rules = {r["name"]: r for r in cfg["rules"]}
    assert rules["cable_removed"]["canon_group"] == rules["cable_routing"]["canon_group"]
    names = [r["name"] for r in cfg["rules"]]
    assert names.index("cable_removed") == names.index("cable_routing") - 1


# --------------------------------------------------------------------------- #
# the virtual node's state
# --------------------------------------------------------------------------- #
def test_the_cable_node_has_a_removed_state(tax):
    assert tax.states_of("cable") == ["routed", "released", "removed"]
    assert tax.default_state("cable") == "routed"
    assert tax.apply_verb("cable", {}, "remove") == ("state", "removed")
    assert tax.apply_verb("cable", {}, "release") == ("state", "released")
    assert tax.apply_verb("cable_clip", {}, "release") == ("state", "open")
    for state in tax.states_of("cable"):
        for placement in ("in_chassis", "on_bench", "elsewhere"):
            assert tax.needs_mask("cable", state, placement) is False
    # still no class that is opened *or* taken out because of it (U9)
    assert "cable" not in tax.opened_or_taken_out()


def test_remove_starts_from_routed_or_released_and_ends_there(tax):
    assert verb_applies(tax, "cable", {}, "remove", "routed")
    assert verb_applies(tax, "cable", {}, "remove", "released")
    assert not verb_applies(tax, "cable", {}, "remove", "removed")
    assert not verb_applies(tax, "cable", {}, "release", "removed")


def test_the_d36_fixture_imports_step_2_as_a_cable_event(tax):
    rows, meta = read_desktop_csv(FIXTURES / "desktop_36.csv")
    li = import_log(36, rows, meta, tax)
    step2 = [a for a in li.actions if a.step == 2]
    assert li.steps[1].raw_name == D36_ROW and li.steps[1].step_type == "normal"
    assert [(a.target, a.verb, a.tool) for a in step2] == [(NODE, "remove", "none")]
    assert not [k for k, rec in li.instances.items() if rec.cls == "cable_clip"]
    assert NODE not in li.instances
    assert not [i for i in li.issues if "cable" in i]
    # the state log: the node goes routed -> removed at step 2, and that is all
    events = events_from_actions(li.instances, li.actions, tax)
    cable = [(e.step, e.target, e.attr, e.old, e.new) for e in events
             if e.target.startswith("cable:")]
    assert cable == [(2, NODE, "state", "routed", "removed")]
    assert not event_problems(li.instances, events, tax)
    for step in (1, 2, 3, len(li.steps)):
        state = state_at(li.instances, events, step, tax)
        geom = needs_geom(li.instances, state, tax)
        assert not [k for k in geom if k.startswith("cable")]
    assert state_at(li.instances, events, 2, tax)[NODE].state == "removed"
    assert NODE not in state_at(li.instances, events, 1, tax)
    # the graph never offers it as a thing to do (it is no blocker of anything)
    final = state_at(li.instances, events, len(li.steps), tax)
    assert not [pair for pair in legal_actions(li.instances, [], final, tax)
                if pair[1].startswith("cable")]


# --------------------------------------------------------------------------- #
# the survey (U33 report section 2): every cable row of the 66 sheets
# --------------------------------------------------------------------------- #
#: (desktop, step, sheet name, rule, target, verb, result) -- the importer's
#: answer for every row whose name says "cable", after U33. Only D36 step 2
#: changed (before: cable_routing / cable_clip.01 / release).
SURVEY = [
    (22, 2, 'remove the cable from the case fan clip', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (23, 2, 'remove the cable from the case fan clip', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (24, 2, 'remove the cable from the case fan clip', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (25, 2, 'remove the cable from the case fan clip', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (26, 2, 'remove the cable from the case fan clip', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (27, 23, 'remove cable from cable locker 1', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (27, 24, 'remove cable from cable locker 2', 'cable_routing', 'cable_clip.02', 'release', 'success'),
    (27, 25, 'remove cable from cable locker 3', 'cable_routing', 'cable_clip.03', 'release', 'success'),
    (27, 26, 'remove cable from cable locker 4', 'cable_routing', 'cable_clip.04', 'release', 'success'),
    (28, 3, 'Cable locker for power module cables on CPU fan', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (28, 23, 'remove cable from cable locker 1', 'cable_routing', 'cable_clip.02', 'release', 'success'),
    (28, 24, 'remove cable from cable locker 2', 'cable_routing', 'cable_clip.03', 'release', 'success'),
    (28, 25, 'remove cable from cable locker 3', 'cable_routing', 'cable_clip.04', 'release', 'success'),
    (28, 26, 'remove cable from cable locker 4', 'cable_routing', 'cable_clip.05', 'release', 'success'),
    (29, 2, 'remove the cable from the case fan clip', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (30, 2, 'Cable locker for power module on heatsink', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (31, 2, 'Cable locker on heatsink', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (32, 2, 'remove the cable from the case fan clip', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (33, 2, 'remove the cable from the case fan clip', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (34, 18, 'Moving cables to show power module - optical drive connector', 'action_auxiliary_reveal', '?', 'displace', 'success'),
    (35, 21, 'Try to remove power module (failed) due to the cable on the motherboard', 'psu_attempt', 'psu.01', 'remove', 'failed'),
    (36, 2, 'Remove 1 isolated cable from motherboard', 'cable_removed', 'cable:isolated.01', 'remove', 'success'),
    (50, 26, 'Remove cable from cable locker', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (51, 26, 'Remove cable from cable locker', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (52, 24, 'Remove cable from cable locker', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (53, 26, 'Remove cable from cable locker', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (54, 25, 'Remove cable from cable locker', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (55, 26, 'Remove cable from cable locker', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (56, 24, 'Remove cable from cable locker', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (57, 25, 'Remove cable from cable locker', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (58, 26, 'Remove cable from cable locker', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (59, 24, 'Remove cable from cable locker', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (60, 24, 'Remove cable from two cable locker', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (63, 3, 'Remove cable from heatsink cover', 'cable_routing', 'cable_clip.01', 'release', 'success'),
    (65, 25, 'Remove cables from cable locker', 'cable_routing', 'cable_clip.01', 'release', 'success'),
]


@pytest.mark.parametrize("desktop,step,name,rule,target,verb,result", SURVEY)
def test_every_surveyed_cable_row_maps_as_pinned(desktop, step, name, rule, target, verb,
                                                 result, tax):
    """Each pinned name, on its own: same rule, same kind of target, same verb."""
    assert parse_raw_name(name).matched_rule == rule
    li, action = _one(name, tax)
    assert (action.verb, action.result) == (verb, result)
    if target.startswith("cable:"):
        assert action.target == loose_cable_key(1)
    elif target == "?":
        assert action.target == "?"
    else:
        assert li.instances[action.target].cls == target.split(".")[0]


def test_only_d36_step_2_is_a_cable_event_in_the_survey():
    events = [(d, s) for d, s, _n, _r, target, _v, _x in SURVEY if target.startswith("cable:")]
    assert events == [(36, 2)]


def _real_drive() -> Path:
    with open(ROOT / "configs" / "paths.yaml", "r", encoding="utf-8") as f:
        return Path(yaml.safe_load(f)["raw_logs_dir"]) / "drive"


def test_the_survey_is_what_the_real_sheets_give(tax):
    """All 66 sheets, every row: the cable rows are the pinned table, keys and all.

    Skipped where the exports are not on disk (they are not in the repository).
    """
    drive = _real_drive()
    if not (drive / "desktop_36.csv").exists():
        pytest.skip(f"the Drive exports are not at {drive}")
    found, events, desktops = [], [], 0
    for desktop, path in iter_desktop_csvs(drive):
        desktops += 1
        rows, meta = read_desktop_csv(path)
        li = import_log(desktop, rows, meta, tax)
        for rec in li.steps:
            acts = [a for a in li.actions if a.step == rec.step]
            events += [(desktop, rec.step) for a in acts if a.target.startswith("cable:")]
            if re.search(r"cable", rec.raw_name, re.IGNORECASE):
                [a] = acts
                found.append((desktop, rec.step, rec.raw_name,
                              parse_raw_name(rec.raw_name).matched_rule,
                              a.target, a.verb, a.result))
        assert not [k for k in li.instances if k.startswith("cable:")]
    assert desktops == 66
    assert found == SURVEY
    assert events == [(36, 2)]
