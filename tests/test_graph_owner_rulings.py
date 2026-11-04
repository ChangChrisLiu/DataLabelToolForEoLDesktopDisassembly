"""The owner's graph rulings of 2026-11-04 afternoon (R1 R6 R12 R14 R15 R16, D66, occlusion).

Each rule on a tiny hand-made desktop, plus the pure core of the dynamic
occlusion (:func:`tda.core.occlusion.frame_occlusions`).
"""
from __future__ import annotations

import numpy as np
import pytest
from graph_scenes import DESKTOP, act, inst, state_after, triples
from pycocotools import mask as coco_mask

from tda.core import masks
from tda.core.graph import (
    Edge,
    legal_actions,
    loose_parts,
    propose_edges,
    validate_sequence,
)
from tda.core.graph_rules import (
    LOOSE_ATTR,
    NON_STEPWISE_DESKTOPS,
    STEPWISE_TASKS,
    fan_leads_by_timing,
    is_stepwise,
)
from tda.core.model import EXTRA_ATTR, StateEvent
from tda.core.occlusion import OCCLUSION_THRESHOLD, frame_occlusions, waits_for
from tda.core.states import initial_state
from tda.core.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def _table(*recs):
    return {r.key: r for r in recs}


# --------------------------------------------------------------------------- #
# R12: a card latch holds every expansion card
# --------------------------------------------------------------------------- #
def test_r12_card_latch_with_a_class_word_locks_every_card(tax):
    table = _table(
        inst("card_latch.01", "card_latch", attrs={"of": "gpu"}),
        inst("expansion_card.gpu.01", "expansion_card", attrs={"kind": "gpu"}),
        inst("expansion_card.wlan.01", "expansion_card", attrs={"kind": "wlan"}),
    )
    got = triples(propose_edges(table, tax))
    assert ("locked_by", "expansion_card.gpu.01", "card_latch.01") in got
    assert ("locked_by", "expansion_card.wlan.01", "card_latch.01") in got


def test_r12_a_latch_naming_one_card_still_holds_them_all(tax):
    table = _table(
        inst("card_latch.01", "card_latch", attrs={"of": "expansion_card.gpu.01"}),
        inst("expansion_card.gpu.01", "expansion_card"),
        inst("expansion_card.other.01", "expansion_card"),
    )
    got = [e for e in propose_edges(table, tax) if e.type == "locked_by"]
    assert {(e.target, e.blocker) for e in got} == {
        ("expansion_card.gpu.01", "card_latch.01"),
        ("expansion_card.other.01", "card_latch.01"),
    }


# --------------------------------------------------------------------------- #
# R6: a RAM stick added in S1 also comes off before the board
# --------------------------------------------------------------------------- #
def _board_with_extra_ram():
    return _table(
        inst("motherboard.01", "motherboard"),
        inst("ram_module.01", "ram_module"),
        inst("ram_module.02", "ram_module", attrs={EXTRA_ATTR: "controller"}),
    )


def test_r6_an_added_ram_stick_blocks_the_board(tax):
    got = triples(propose_edges(_board_with_extra_ram(), tax))
    assert ("blocked_by", "motherboard.01", "ram_module.01") in got
    assert ("blocked_by", "motherboard.01", "ram_module.02") in got


def test_r6_an_added_board_still_derives_nothing(tax):
    table = _board_with_extra_ram()
    table["motherboard.01"].attrs[EXTRA_ATTR] = "controller"
    assert not [e for e in propose_edges(table, tax) if e.type == "blocked_by"]


def test_validate_sequence_folds_the_hand_written_removal_of_an_added_stick(tax):
    table = _board_with_extra_ram()
    edges = propose_edges(table, tax)
    actions = [act(1, "ram_module.01", "remove"), act(3, "motherboard.01", "remove")]
    blind = validate_sequence(table, edges, actions, tax)
    assert any("ram_module.02" in line for line in blind)
    hand = [StateEvent(DESKTOP, 2, "ram_module.02", "state", "installed", "removed",
                       auto=False)]
    assert validate_sequence(table, edges, actions, tax, manual=hand) == []


def test_validate_sequence_does_not_fold_a_hand_event_of_a_later_step(tax):
    table = _board_with_extra_ram()
    edges = propose_edges(table, tax)
    actions = [act(1, "ram_module.01", "remove"), act(3, "motherboard.01", "remove")]
    late = [StateEvent(DESKTOP, 5, "ram_module.02", "state", "installed", "removed",
                       auto=False)]
    assert validate_sequence(table, edges, actions, tax, manual=late)


# --------------------------------------------------------------------------- #
# R1: a fan lead belongs to the CPU fan when it is pulled right before the fan
# --------------------------------------------------------------------------- #
def _two_part_cooler():
    return _table(
        inst("motherboard.01", "motherboard"),
        inst("cpu_cooler.fan.01", "cpu_cooler", attrs={"kind": "fan"}),
        inst("cpu_cooler.heatsink.01", "cpu_cooler", attrs={"kind": "heatsink"}),
        inst("connector.fan.01", "connector", attrs={"kind": "fan"},
             socket_host="motherboard.01", cable="cable:cpu_fan"),
        inst("connector.fan.02", "connector", attrs={"kind": "fan"},
             socket_host="motherboard.01", cable="cable:case_fan"),
    )


def _two_part_log():
    return [act(15, "connector.fan.01", "disconnect"), act(16, "cpu_cooler.fan.01", "remove"),
            act(21, "cpu_cooler.heatsink.01", "remove"),
            act(29, "connector.fan.02", "disconnect")]


def test_r1_timing_gives_the_two_part_fan_its_lead_and_only_that_one(tax):
    table = _two_part_cooler()
    got = triples(propose_edges(table, tax, _two_part_log()))
    assert ("connected_to", "cpu_cooler.fan.01", "connector.fan.01") in got
    assert not any(t == "connected_to" and b == "connector.fan.02" and tg.startswith("cpu_cooler")
                   for t, tg, b in got)
    assert ("connected_to", "motherboard.01", "connector.fan.02") in got
    assert fan_leads_by_timing(table, _two_part_log()) == {
        "cpu_cooler.fan.01": {"connector.fan.01"}}


def test_r1_without_the_log_the_cable_tag_alone_decides(tax):
    got = triples(propose_edges(_two_part_cooler(), tax))
    assert not any(tg.startswith("cpu_cooler") and t == "connected_to" for t, tg, _b in got)


def test_r1_a_fan_moved_aside_first_keeps_its_window(tax):
    table = _two_part_cooler()
    log = [act(6, "cpu_cooler.fan.01", "displace"), act(7, "connector.fan.01", "disconnect"),
           act(8, "cpu_cooler.fan.01", "remove"), act(9, "connector.fan.02", "disconnect")]
    assert fan_leads_by_timing(table, log) == {"cpu_cooler.fan.01": {"connector.fan.01"}}


def test_r1_a_cpu_fan_tagged_lead_pulled_after_the_fan_is_a_case_fan_lead(tax):
    table = _table(
        inst("motherboard.01", "motherboard"),
        inst("cpu_cooler.fan.01", "cpu_cooler", attrs={"kind": "fan"}),
        inst("connector.fan.01", "connector", attrs={"kind": "fan"},
             socket_host="motherboard.01", cable="cable:case_fan"),
        inst("connector.fan.02", "connector", attrs={"kind": "fan"},
             socket_host="motherboard.01", cable="cable:cpu_fan"),
    )
    log = [act(2, "connector.fan.01", "disconnect"), act(3, "cpu_cooler.fan.01", "remove"),
           act(9, "connector.fan.02", "disconnect")]
    got = triples(propose_edges(table, tax, log))
    assert ("connected_to", "cpu_cooler.fan.01", "connector.fan.01") in got
    assert ("connected_to", "cpu_cooler.fan.01", "connector.fan.02") not in got


def test_r1_a_heatsink_only_machine_owns_no_fan_lead(tax):
    table = _table(
        inst("motherboard.01", "motherboard"),
        inst("cpu_cooler.heatsink.01", "cpu_cooler", attrs={"kind": "heatsink"}),
        inst("connector.fan.01", "connector", attrs={"kind": "fan"},
             socket_host="motherboard.01", cable="cable:cpu_fan"),
    )
    log = [act(7, "connector.fan.01", "disconnect"), act(8, "cpu_cooler.heatsink.01", "remove")]
    got = triples(propose_edges(table, tax, log))
    assert ("connected_to", "cpu_cooler.heatsink.01", "connector.fan.01") not in got
    assert fan_leads_by_timing(table, log) == {}


# --------------------------------------------------------------------------- #
# R16: loose parts come out before anything else
# --------------------------------------------------------------------------- #
def _loose_fan_desk():
    return _table(
        inst("motherboard.01", "motherboard"),
        inst("case_fan.01", "case_fan", attrs={LOOSE_ATTR: True}),
        inst("connector.fan.01", "connector", attrs={"kind": "fan"},
             socket_host="motherboard.01", cable="cable:case_fan"),
        inst("ram_latch.01", "ram_latch"),
        inst("screw.motherboard.01", "screw", attrs={"role": "motherboard", "captive": False},
             fastens="motherboard.01"),
    )


def test_r16_while_a_loose_part_is_in_only_its_way_out_is_legal(tax):
    table = _loose_fan_desk()
    edges = propose_edges(table, tax)
    state = initial_state(table, tax)
    assert loose_parts(table, state) == ["case_fan.01"]
    got = legal_actions(table, edges, state, tax)
    # its lead, pulled or taken out with the cable; the fan itself waits for it
    assert got == [("disconnect", "connector.fan.01"), ("remove", "connector.fan.01")]
    pulled = state_after(table, [act(1, "connector.fan.01", "disconnect")], tax)
    assert legal_actions(table, edges, pulled, tax) == [("remove", "case_fan.01")]
    wide = legal_actions(table, edges, state, tax, loose_first=False)
    assert ("unscrew", "screw.motherboard.01") in wide and ("open", "ram_latch.01") in wide


def test_r16_once_the_loose_part_is_out_everything_is_back(tax):
    table = _loose_fan_desk()
    edges = propose_edges(table, tax)
    after = state_after(table, [act(1, "connector.fan.01", "disconnect"),
                                act(2, "case_fan.01", "remove")], tax)
    assert loose_parts(table, after) == []
    assert legal_actions(table, edges, after, tax) == \
        legal_actions(table, edges, after, tax, loose_first=False)


# --------------------------------------------------------------------------- #
# dynamic occlusion
# --------------------------------------------------------------------------- #
def _rle(x0, y0, x1, y1, hw=(100, 100)):
    m = np.zeros(hw, dtype=bool, order="F")
    m[y0:y1, x0:x1] = True
    return masks._coco_rle(masks.encode_rle(m))


def test_occlusion_threshold_is_the_owners_twenty_percent():
    assert OCCLUSION_THRESHOLD == pytest.approx(0.20)


def test_frame_occlusions_needs_twenty_percent_and_the_upper_layer():
    amodal = {"x": _rle(0, 0, 10, 10), "big": _rle(0, 0, 10, 3),     # 30 % of x
              "small": _rle(0, 0, 10, 1),                             # 10 % of x
              "under": _rle(0, 0, 10, 10)}
    above = {("big", "x"), ("small", "x"), ("x", "under")}
    got = frame_occlusions(amodal, targets=["x", "under"], blockers=["big", "small", "x", "under"],
                           is_above=lambda y, x: (y, x) in above)
    assert [o.blocker for o in got["x"]] == ["big"]
    assert got["x"][0].frac == pytest.approx(0.3)
    assert [o.blocker for o in got["under"]] == ["x"]


def test_frame_occlusions_exempt_and_non_blockers():
    amodal = {"x": _rle(0, 0, 10, 10), "y": _rle(0, 0, 10, 10)}
    up = lambda y, x: y == "y"  # noqa: E731
    assert frame_occlusions(amodal, ["x"], ["y"], up)
    assert not frame_occlusions(amodal, ["x"], [], up)
    assert not frame_occlusions(amodal, ["x"], ["y"], up, exempt=lambda y, x: True)


def test_waits_for_is_transitive_and_skips_rejected_edges():
    edges = [Edge("connected_to", "fan", "lead"), Edge("fastened_by", "lead", "clip"),
             Edge("covered_by", "fan", "shroud", status="rejected")]
    got = waits_for(edges)
    assert got["fan"] == {"lead", "clip"}


def test_legal_actions_leaves_out_an_occluded_part(tax):
    table = _table(inst("motherboard.01", "motherboard"), inst("ram_latch.01", "ram_latch"),
                   inst("drive_cage.01", "drive_cage"))
    state = initial_state(table, tax)
    free = legal_actions(table, [], state, tax)
    assert ("open", "ram_latch.01") in free
    hidden = legal_actions(table, [], state, tax, occluded={"ram_latch.01": ["drive_cage.01"]})
    assert ("open", "ram_latch.01") not in hidden
    assert set(free) - set(hidden) == {("open", "ram_latch.01")}


# --------------------------------------------------------------------------- #
# D66: not a stepwise teardown
# --------------------------------------------------------------------------- #
def test_d66_is_not_stepwise():
    assert 66 in NON_STEPWISE_DESKTOPS
    assert not is_stepwise(66)
    assert is_stepwise(65)
    assert not is_stepwise(5, {"stepwise": False})
    assert is_stepwise(66, {"stepwise": True})
    assert set(STEPWISE_TASKS) == {"V4", "V6", "V7", "V16"}


def test_a_drive_cage_never_occludes_a_connector():
    from tda.core.occlusion import NEVER_OVER

    assert ("drive_cage", "connector") in NEVER_OVER


def test_the_owner_exception_list_holds_d63_cover(tmp_path):
    from tda.core.occlusion import occlusion_exceptions

    got = occlusion_exceptions()
    assert [(e.desktop, e.step, e.target) for e in got] == [(63, 2, "cover.05")]
    assert all(e.reason for e in got)
    empty = tmp_path / "none.yaml"
    assert occlusion_exceptions(empty) == ()
