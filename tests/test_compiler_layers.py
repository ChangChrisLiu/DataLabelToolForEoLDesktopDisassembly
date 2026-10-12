"""Tests for the compiler's layer ordering (spec 3.3 step 5).

The global z-order, the pairwise overrides that win over it, and the cycles
they can form. :func:`tda.core.compiler_layers.above` is tested directly; the
ordering is tested through :func:`~tda.core.compiler.compile_frame`, because
the visible masks are what the order is *for*.
"""
from __future__ import annotations

import numpy as np
import pytest

from compiler_scenes import (
    A_BOX,
    B_BOX,
    BENCH,
    CHASSIS,
    kf,
    po,
    rect,
    run,
    three_squares,
    two_squares,
    zo,
)
from tda.core.compiler import above


# --- above ---
def test_above_uses_the_global_order_and_treats_unlisted_layers_as_topmost():
    zorder = zo([("A", "main"), ("B", "main")])
    assert above(("B", "main"), ("A", "main"), zorder, []) is True
    assert above(("A", "main"), ("B", "main"), zorder, []) is False
    assert above(("A", "main"), ("A", "main"), zorder, []) is False
    # unlisted layers float to the top, two unlisted ones are unordered
    assert above(("C", "main"), ("B", "main"), zorder, []) is True
    assert above(("B", "main"), ("C", "main"), zorder, []) is False
    assert above(("C", "main"), ("D", "main"), zorder, []) is False


def test_above_override_matches_on_instance_keys_regardless_of_part():
    zorder = zo([("A", "main"), ("B", "main")])
    overrides = [po("A", "B")]
    assert above(("A", "main"), ("B", "main"), zorder, overrides) is True
    assert above(("B", "main"), ("A", "main"), zorder, overrides) is False
    # the override also decides for parts that are not in the global order
    assert above(("A", "wall"), ("B", "floor"), zorder, overrides) is True


# --- the global order and the overrides on top of it ---
def test_an_override_against_the_global_order_moves_only_the_pairs_pixels():
    """Task U6: "A above C" is about A and C; B, painted between them, keeps its pixels.

    The topological sort this replaced lifted A over B as well -- "A above C"
    became "A above everything up to C" -- and B lost ``a & b`` to a statement
    nobody made about it.
    """
    a, b, c = rect(8, 8, 40, 40), rect(16, 16, 48, 48), rect(24, 24, 56, 56)
    plain = run(**three_squares())
    assert np.array_equal(plain.instances["C"].visible, c)
    assert np.array_equal(plain.instances["B"].visible, b & ~c)
    assert np.array_equal(plain.instances["A"].visible, a & ~b & ~c)

    out = run(**three_squares(), overrides=[po("A", "C")])
    assert out.problems == []
    # B is byte-identical; C gives A exactly what both cover
    assert np.array_equal(out.instances["B"].visible, plain.instances["B"].visible)
    assert np.array_equal(out.instances["C"].visible, c & ~a)
    assert np.array_equal(out.instances["A"].visible, (a & ~b & ~c) | (a & c))
    # painting stays in the global order; the pair is reported apart
    assert out.painted == plain.painted
    assert out.pairs == {"in_chassis": [("A", "C")]}


def test_a_third_layer_above_both_keeps_the_pixels_it_covers():
    """A above B, with C painted over both: A gets ``a & b`` minus what C holds."""
    a, b, c = rect(8, 8, 40, 40), rect(16, 16, 48, 48), rect(24, 24, 56, 56)
    plain = run(**three_squares())
    out = run(**three_squares(), overrides=[po("A", "B")])
    assert out.problems == []
    assert np.array_equal(out.instances["C"].visible, plain.instances["C"].visible)
    assert np.array_equal(out.instances["A"].visible, a & ~c)
    assert np.array_equal(out.instances["B"].visible, b & ~c & ~a)
    # nothing was created or lost: the three visible masks still tile a | b | c
    union = (out.instances["A"].visible | out.instances["B"].visible
             | out.instances["C"].visible)
    assert np.array_equal(union, a | b | c)
    total = sum(int(out.instances[k].visible.sum()) for k in "ABC")
    assert total == int((a | b | c).sum())


def test_a_chain_of_overrides_follows_through_and_rivals_go_by_the_global_order():
    """A > B > C hands a pixel of all three to A; two rivals over C: the higher wins."""
    a, b, c = rect(8, 8, 40, 40), rect(16, 16, 48, 48), rect(24, 24, 56, 56)
    chain = run(**three_squares(), overrides=[po("B", "C"), po("A", "B")])
    assert chain.problems == []
    assert np.array_equal(chain.instances["A"].visible, a)
    assert np.array_equal(chain.instances["B"].visible, b & ~a)
    assert np.array_equal(chain.instances["C"].visible, c & ~a & ~b)
    # the same rows in the other order: the same pixels
    again = run(**three_squares(), overrides=[po("A", "B"), po("B", "C")])
    for key in "ABC":
        assert np.array_equal(again.instances[key].visible, chain.instances[key].visible)

    # A and B both above C, nothing between them: B is above A globally, so a
    # pixel all three cover goes to B, and the rest of a & c to A
    rivals = run(**three_squares(), overrides=[po("A", "C"), po("B", "C")])
    assert np.array_equal(rivals.instances["B"].visible, b)
    assert np.array_equal(rivals.instances["A"].visible, a & ~b)
    assert np.array_equal(rivals.instances["C"].visible, c & ~a & ~b)


def test_d13_chassis_above_a_screw_moves_only_the_screw_pixels_inside_the_chassis():
    """The trial's override, on a synthetic D13: nobody between them may change.

    Bottom-up: the chassis, the cover, the motherboard, the lever, four RAM
    latches, then screw 06 on the board.  "chassis above screw.motherboard.06"
    -- op 2165 of the live log -- emptied the board, the lever and the latches
    and left 152 px of the cover.  Now the screw's pixels inside the chassis go
    to the chassis and every other mask is byte-identical.
    """
    boxes = {
        "chassis": (0, 0, 64, 64),
        "cover.02": (2, 2, 20, 12),
        "motherboard.01": (6, 10, 58, 58),
        "cpu_socket_lever.01": (30, 14, 34, 30),
        "ram_latch.01": (40, 12, 43, 16),
        "ram_latch.02": (40, 50, 43, 54),
        "ram_latch.03": (46, 12, 49, 16),
        "ram_latch.04": (46, 50, 49, 54),
        "screw.motherboard.06": (12, 40, 17, 45),
    }
    keyframes = {name: [kf(name, {"main": box}, kf_id=i + 1)]
                 for i, (name, box) in enumerate(boxes.items())}
    zorder = zo([(name, "main") for name in boxes])
    plain = run(keyframes, zorder)
    out = run(keyframes, zorder, overrides=[po("chassis", "screw.motherboard.06")])
    assert plain.problems == []
    # the screw lies wholly inside the chassis, so it is the one that empties
    assert out.problems == ["empty_visible:screw.motherboard.06"]
    for name in boxes:
        if name in ("chassis", "screw.motherboard.06"):
            continue
        assert np.array_equal(out.instances[name].visible, plain.instances[name].visible), name
        assert out.instances[name].visibility == plain.instances[name].visibility
    screw = rect(*boxes["screw.motherboard.06"])
    assert not out.instances["screw.motherboard.06"].visible.any()
    assert np.array_equal(out.instances["chassis"].visible,
                          plain.instances["chassis"].visible | screw)


def test_a_bench_override_never_reaches_into_the_chassis_group():
    """A pair holds only where both instances are painted in the same group."""
    a, b = rect(*A_BOX), rect(*B_BOX)
    keyframes = {
        "A": [kf("A", {"main": A_BOX}, kf_id=1)],
        "B": [kf("B", {"main": B_BOX}, kf_id=2, placement=BENCH)],
    }
    zorder = zo([("A", "main"), ("B", "main")])
    out = run(keyframes, zorder, placements={"A": CHASSIS, "B": BENCH},
              overrides=[po("A", "B")])
    assert np.array_equal(out.instances["A"].visible, a)
    assert np.array_equal(out.instances["B"].visible, b)
    assert out.pairs == {}


def test_a_part_missing_from_the_zorder_floats_to_the_top():
    scene = two_squares()
    scene["zorder"] = zo([("A", "main")])
    out = run(**scene)
    assert "zorder_missing:B/main" in out.problems
    assert np.array_equal(out.instances["A"].visible, rect(*A_BOX) & ~rect(*B_BOX))
    assert np.array_equal(out.instances["B"].visible, rect(*B_BOX))


def test_multi_part_shapes_are_layered_part_by_part():
    floor, wall = (8, 40, 24, 56), (8, 8, 24, 24)
    keyframes = {
        "cage": [kf("cage", {"floor": floor, "wall": wall}, kf_id=1)],
        "board": [kf("board", {"main": (8, 20, 56, 44)}, kf_id=2)],
    }
    zorder = zo([("cage", "floor"), ("board", "main"), ("cage", "wall")])
    out = run(keyframes, zorder)
    floor_m, wall_m, board_m = rect(*floor), rect(*wall), rect(8, 20, 56, 44)

    cage = out.instances["cage"]
    assert np.array_equal(cage.amodal, floor_m | wall_m)
    assert np.array_equal(cage.visible, wall_m | (floor_m & ~board_m))
    assert cage.occlusion_ratio == pytest.approx(64 / 512)

    board = out.instances["board"]
    assert np.array_equal(board.visible, board_m & ~wall_m)
    assert board.occlusion_ratio == pytest.approx(64 / 1152)
    assert out.problems == []


# --- cycles ---
def test_a_cycle_of_overrides_is_reported_and_falls_back_to_the_global_order():
    out = run(**two_squares(), overrides=[po("A", "B"), po("B", "A")])
    assert "zorder_cycle:A,B" in out.problems
    a_mask, b_mask = rect(*A_BOX), rect(*B_BOX)
    assert np.array_equal(out.instances["A"].visible, a_mask & ~b_mask)
    assert np.array_equal(out.instances["B"].visible, b_mask)


def test_a_cycle_with_an_edge_reaching_into_it_is_still_reported():
    # b <-> c is the cycle; d only points into it, so it is not part of it.
    # Walking d first used to leave the traversal's stack dirty.
    scene = three_squares()
    scene["keyframes"]["d"] = [kf("d", {"main": (0, 0, 20, 20)}, kf_id=4)]
    scene["zorder"] = zo(
        [("d", "main"), ("A", "main"), ("B", "main"), ("C", "main")]
    )
    needs = {"A": "mask", "B": "mask", "C": "mask", "d": "mask"}
    out = run(
        scene["keyframes"],
        scene["zorder"],
        needs=needs,
        overrides=[po("C", "B"), po("B", "C"), po("B", "d")],
    )
    assert [p for p in out.problems if p.startswith("zorder_cycle")] == [
        "zorder_cycle:B,C"
    ]
    # the whole group fell back to the global order d < A < B < C
    a, b, c = rect(8, 8, 40, 40), rect(16, 16, 48, 48), rect(24, 24, 56, 56)
    d = rect(0, 0, 20, 20)
    assert np.array_equal(out.instances["C"].visible, c)
    assert np.array_equal(out.instances["B"].visible, b & ~c)
    assert np.array_equal(out.instances["A"].visible, a & ~b & ~c)
    assert np.array_equal(out.instances["d"].visible, d & ~a & ~b & ~c)


def test_two_separate_cycles_are_both_reported():
    keyframes = {
        "A": [kf("A", {"main": (0, 0, 16, 16)}, kf_id=1)],
        "B": [kf("B", {"main": (8, 8, 24, 24)}, kf_id=2)],
        "C": [kf("C", {"main": (32, 32, 48, 48)}, kf_id=3)],
        "D": [kf("D", {"main": (40, 40, 56, 56)}, kf_id=4)],
    }
    zorder = zo([("A", "main"), ("B", "main"), ("C", "main"), ("D", "main")])
    out = run(
        keyframes,
        zorder,
        overrides=[po("A", "B"), po("B", "A"), po("C", "D"), po("D", "C")],
    )
    assert [p for p in out.problems if p.startswith("zorder_cycle")] == [
        "zorder_cycle:A,B",
        "zorder_cycle:C,D",
    ]
    # both groups fell back to the global order
    assert np.array_equal(out.instances["B"].visible, rect(8, 8, 24, 24))
    assert np.array_equal(
        out.instances["A"].visible, rect(0, 0, 16, 16) & ~rect(8, 8, 24, 24)
    )
    assert np.array_equal(out.instances["D"].visible, rect(40, 40, 56, 56))
    assert np.array_equal(
        out.instances["C"].visible, rect(32, 32, 48, 48) & ~rect(40, 40, 56, 56)
    )


def test_a_three_instance_cycle_is_reported_once():
    scene = three_squares()
    out = run(**scene, overrides=[po("A", "B"), po("B", "C"), po("C", "A")])
    assert [p for p in out.problems if p.startswith("zorder_cycle")] == [
        "zorder_cycle:A,B,C"
    ]
    a, b, c = rect(8, 8, 40, 40), rect(16, 16, 48, 48), rect(24, 24, 56, 56)
    assert np.array_equal(out.instances["C"].visible, c)
    assert np.array_equal(out.instances["A"].visible, a & ~b & ~c)
