"""Task W3c: which overlapping pairs of a frame still want the user's 「谁压着谁」 ruling (Qt-free).

* the rule: pixels >= 1000; no Label Studio draft / 框外零件 / gallery-only /
  implied part / chassis / unknown key; no active edge of any type either way
  (a rejected edge does not count); no structural reason (parent, attached
  chain, mounted_on / fastens / socket_host / cable, the support list); no
  ruling yet for the ordered pair;
* the tags -- since task W3g the user's class rulings of the shipped whitelist
  (用户已定：挡 / 其他; ``class`` or ``class.kind``, the most exact entry wins)
  with the user's words; ``same_pair_pending``; the order
  (tag, then pixels, then keys); the batch groups; the texts.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tda.core.graph import Edge
from tda.core.model import InstanceRec
from tda.ui import cover_rulings as CR
from tda.ui.cover_rulings import (
    BLOCKS,
    MIN_PX,
    NOT_BLOCKS,
    SUPPORT,
    TAG_OTHER,
    TAG_USER_BLOCKS,
    TAG_USER_NOT_BLOCKS,
    Candidate,
    ClassRulings,
    candidate_text,
    chassis_pairs,
    class_key,
    class_pairs,
    excluded_part,
    pair_tag,
    pending_pairs,
    ruled_text,
    status_text,
    structural_reasons,
)

D = 13


def rec(key: str, cls: str, **kw) -> InstanceRec:
    attrs = kw.pop("attrs", {})
    return InstanceRec(key=key, desktop=D, cls=cls, attrs=attrs, **kw)


def table(*recs: InstanceRec) -> dict:
    return {r.key: r for r in recs}


BASE = table(
    rec("optical_drive.01", "optical_drive"),
    rec("motherboard.01", "motherboard"),
    rec("drive_cage.01", "drive_cage"),
    rec("cpu.01", "cpu"),
    rec("psu.01", "psu"),
    rec("cover.01", "cover"),
    rec("case_fan.01", "case_fan"),
    rec("chassis", "chassis"),
    rec("expansion_card.gpu.01", "expansion_card", attrs={"kind": "gpu"}),
    rec("cpu_cooler.heatsink.01", "cpu_cooler", attrs={"kind": "heatsink"}),
    rec("cpu_cooler.fan.01", "cpu_cooler", attrs={"kind": "fan"}),
    rec("ram_module.01", "ram_module"),
)


def keys(cands) -> list[tuple[str, str]]:
    return [(c.above, c.below) for c in cands]


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def test_the_threshold_is_1000_pixels():
    assert MIN_PX == 1000
    pairs = [("optical_drive.01", "motherboard.01", 1000),
             ("drive_cage.01", "motherboard.01", 999)]
    assert keys(pending_pairs(pairs, BASE, [], {})) == [("optical_drive.01", "motherboard.01")]
    # a mapping is as good as triples
    assert keys(pending_pairs({("optical_drive.01", "motherboard.01"): 5000}, BASE, [], {})) \
        == [("optical_drive.01", "motherboard.01")]


def test_an_edge_either_way_excludes_and_a_rejected_one_does_not():
    pair = [("optical_drive.01", "drive_cage.01", 50_000)]
    for edge in (Edge("covered_by", "drive_cage.01", "optical_drive.01"),        # below waits
                 Edge("fastened_by", "optical_drive.01", "drive_cage.01"),       # the other way
                 Edge("partner_of", "drive_cage.01", "optical_drive.01", source="ls"),  # any type
                 Edge("covered_by", "drive_cage.01", "optical_drive.01", status="accepted")):
        assert pending_pairs(pair, BASE, [edge], {}) == [], edge
    for status in ("rejected", "rejected_orphan", "accepted_orphan"):
        dead = Edge("covered_by", "drive_cage.01", "optical_drive.01", status=status)
        assert keys(pending_pairs(pair, BASE, [dead], {})) == [pair[0][:2]], status
    # an edge between other parts says nothing about this pair
    other = Edge("covered_by", "motherboard.01", "optical_drive.01")
    assert keys(pending_pairs(pair, BASE, [other], {})) == [pair[0][:2]]


def test_structural_relations_exclude():
    pair = [("psu.01", "drive_cage.01", 9000)]
    for field_name in ("mounted_on", "fastens", "socket_host", "cable", "parent"):
        for side in ("above", "below"):
            one, other = ("psu.01", "drive_cage.01") if side == "above" else \
                ("drive_cage.01", "psu.01")
            inst = dict(BASE)
            inst[one] = rec(one, BASE[one].cls, **{field_name: other})
            assert pending_pairs(pair, inst, [], {}) == [], (field_name, side)
            assert structural_reasons(inst, "psu.01", "drive_cage.01") == [
                f"{side}.{field_name}"]
    # parent further up an attached chain: a latch on a bracket on the cage
    inst = dict(BASE)
    inst["bracket.01"] = rec("bracket.01", "cover", parent="drive_cage.01", attached=True)
    inst["psu.01"] = rec("psu.01", "psu", parent="bracket.01", attached=True)
    assert structural_reasons(inst, "psu.01", "drive_cage.01") == ["above.parent"]
    # ... but not through a link that is not attached
    inst["psu.01"] = rec("psu.01", "psu", parent="bracket.01", attached=False)
    assert structural_reasons(inst, "psu.01", "drive_cage.01") == []
    # a cycle in the table does not hang
    inst["bracket.01"] = rec("bracket.01", "cover", parent="psu.01", attached=True)
    inst["psu.01"] = rec("psu.01", "psu", parent="bracket.01", attached=True)
    assert structural_reasons(inst, "psu.01", "drive_cage.01") == []


def test_the_support_list_excludes_the_parts_that_sit_on_the_motherboard():
    assert SUPPORT == (("cpu", "motherboard"), ("ram_module", "motherboard"),
                       ("cpu_cooler", "motherboard"), ("expansion_card", "motherboard"))
    pairs = [("cpu.01", "motherboard.01", 6600), ("ram_module.01", "motherboard.01", 2700),
             ("cpu_cooler.fan.01", "motherboard.01", 48000),
             ("expansion_card.gpu.01", "motherboard.01", 17000),
             ("case_fan.01", "motherboard.01", 42000)]
    assert keys(pending_pairs(pairs, BASE, [], {})) == [("case_fan.01", "motherboard.01")]
    assert structural_reasons(BASE, "cpu.01", "motherboard.01") == ["supported_by"]
    # the other way round is no support
    assert structural_reasons(BASE, "motherboard.01", "cpu.01") == []


def test_a_ruling_on_the_ordered_pair_excludes_it_and_only_it():
    pairs = [("optical_drive.01", "drive_cage.01", 50_000),
             ("drive_cage.01", "optical_drive.01", 2000)]
    ruled = {("optical_drive.01", "drive_cage.01"): {"ruling": BLOCKS}}
    assert keys(pending_pairs(pairs, BASE, [], ruled)) == [("drive_cage.01", "optical_drive.01")]


def test_drafts_context_gallery_only_implied_chassis_and_unknown_parts_are_left_out():
    inst = dict(BASE)
    inst["ls:Motherboard#1"] = rec("ls:Motherboard#1", "motherboard")
    inst["ctx.psu.01"] = rec("ctx.psu.01", "psu", attrs={"context": True})
    inst["loose_cable.01"] = rec("loose_cable.01", "loose_cable",
                                 attrs={"gallery_only": True, "added_by": "a"})
    inst["drive_latch.01"] = rec("drive_latch.01", "drive_latch",
                                 attrs={"implied": True, "implied_rule": "drive_cage_latch"})
    assert excluded_part("ls:Motherboard#1", inst["ls:Motherboard#1"]) == "ls"
    assert excluded_part("ctx.psu.01", inst["ctx.psu.01"]) == "context"
    assert excluded_part("loose_cable.01", inst["loose_cable.01"]) == "gallery_only"
    assert excluded_part("drive_latch.01", inst["drive_latch.01"]) == "implied"
    assert excluded_part("chassis", inst["chassis"]) == "fixed"
    assert excluded_part("ghost.01", None) == "unknown"
    assert excluded_part("optical_drive.01", inst["optical_drive.01"]) is None
    # an extra the annotator added in S1 is an ordinary part
    extra = rec("hdd.09", "storage_drive", attrs={"added_by": "a", "kind": "hdd"})
    assert excluded_part("hdd.09", extra) is None
    for odd in ("ls:Motherboard#1", "ctx.psu.01", "loose_cable.01", "drive_latch.01",
                "chassis", "ghost.01"):
        pairs = [(odd, "drive_cage.01", 9000), ("drive_cage.01", odd, 9000)]
        assert pending_pairs(pairs, inst, [], {}) == [], odd


def test_small_and_latch_parts_on_top_and_screws_connectors_cables_underneath_are_not_asked():
    """Round 2: the sheet's 「本版不做」 rule, both sides."""
    assert CR.SMALL_ABOVE == {"screw", "connector", "cable_clip", "cable", "ram_latch",
                              "drive_latch", "card_latch", "psu_latch", "cooler_latch",
                              "cpu_socket_lever"}
    assert CR.SMALL_BELOW == {"screw", "connector", "cable_clip", "cable"}
    inst = dict(BASE)
    for cls in sorted(CR.SMALL_ABOVE | CR.SMALL_BELOW):
        inst[f"{cls}.01"] = rec(f"{cls}.01", cls)
    for cls in sorted(CR.SMALL_ABOVE):                       # on top: never asked
        assert pending_pairs([(f"{cls}.01", "motherboard.01", 50_000),
                              (f"{cls}.01", "drive_cage.01", 50_000)], inst, [], {}) == [], cls
    for cls in sorted(CR.SMALL_BELOW):                       # underneath: never asked
        assert pending_pairs([("drive_cage.01", f"{cls}.01", 50_000),
                              ("optical_drive.01", f"{cls}.01", 50_000)], inst, [], {}) == [], cls
    # a lever or a RAM latch underneath stays a real question
    kept = pending_pairs([("cover.01", "cpu_socket_lever.01", 6000),
                          ("drive_cage.01", "ram_latch.01", 3000),
                          ("drive_cage.01", "connector.01", 3000),
                          ("ram_latch.01", "drive_cage.01", 3000)], inst, [], {})
    assert [(c.above, c.below, c.tag) for c in kept] == [
        ("cover.01", "cpu_socket_lever.01", TAG_OTHER),
        ("drive_cage.01", "ram_latch.01", TAG_OTHER)]
    # by class, whatever the kind or role
    inst["screw.motherboard.01"] = rec("screw.motherboard.01", "screw",
                                       attrs={"role": "motherboard"})
    inst["connector.fan.01"] = rec("connector.fan.01", "connector", attrs={"kind": "fan"})
    assert pending_pairs([("drive_cage.01", "screw.motherboard.01", 9000),
                          ("drive_cage.01", "connector.fan.01", 9000)], inst, [], {}) == []


# --------------------------------------------------------------------------- #
# tags, order, counts, texts
# --------------------------------------------------------------------------- #
def test_the_tags_are_the_users_class_rulings_of_the_shipped_whitelist():
    """Task W3g: the W2 whitelist replaced the sheet's 建议收 / 存疑 tables."""
    classes = CR.class_rulings()
    assert len(classes.blocks) == 30 and len(classes.not_blocks) == 12   # W3: 19 + the 11 card rulings
    assert ("optical_drive", "motherboard", "A2 挡") in classes.blocks
    # the support table wins: a part on the board is a rule edge, never a tag
    for above, below, _note in classes.blocks + classes.not_blocks:
        assert (above.split(".")[0], below.split(".")[0]) not in SUPPORT, (above, below)
    assert pair_tag("expansion_card.gpu", "motherboard")[0] == TAG_OTHER
    assert all(note for *_pair, note in classes.blocks + classes.not_blocks)
    assert not ({(a, b) for a, b, _ in classes.blocks}
                & {(a, b) for a, b, _ in classes.not_blocks})
    assert not hasattr(CR, "RECOMMENDED") and not hasattr(CR, "UNCERTAIN")


def test_tags_name_a_class_or_a_class_kind_and_the_most_exact_wins():
    assert pair_tag("optical_drive", "motherboard") == (TAG_USER_BLOCKS, "A2 挡")
    assert pair_tag("drive_cage", "motherboard")[0] == TAG_USER_BLOCKS
    assert pair_tag("storage_drive.ssd", "drive_cage")[0] == TAG_USER_BLOCKS
    assert pair_tag("storage_drive.unknown", "motherboard")[0] == TAG_OTHER   # the kind decides
    assert pair_tag("case_fan", "cpu_cooler.fan")[0] == TAG_USER_BLOCKS
    assert pair_tag("case_fan", "cpu_cooler.heatsink")[0] == TAG_OTHER
    # 不挡 by class: any kind of either side
    tag, reason = pair_tag("psu", "motherboard")
    assert tag == TAG_USER_NOT_BLOCKS and reason.startswith("A4 不挡")
    assert pair_tag("storage_drive.unknown", "cpu_cooler.heatsink")[0] == TAG_USER_NOT_BLOCKS
    assert pair_tag("cover.side", "cpu") == (TAG_OTHER, "用户还没有按类别裁定这一类对：看图判断")
    # a refined entry beats the bare class; no rulings at all: every pair is 其他
    mine = ClassRulings(blocks=(("x", "y", "class"), ("x", "y.k", "kind")))
    assert pair_tag("x.a", "y.k", classes=mine) == (TAG_USER_BLOCKS, "kind")
    assert pair_tag("x", "y.z", classes=mine) == (TAG_USER_BLOCKS, "class")
    assert pair_tag("optical_drive", "motherboard", classes=ClassRulings())[0] == TAG_OTHER


def test_candidates_carry_classes_tag_reason_and_the_order_is_tag_then_pixels():
    pairs = [("case_fan.01", "motherboard.01", 60_000),            # 用户已定：挡
             ("drive_cage.01", "motherboard.01", 30_000),          # 用户已定：挡
             ("optical_drive.01", "motherboard.01", 20_000),       # 用户已定：挡
             ("optical_drive.01", "drive_cage.01", 90_000),        # 用户已定：挡
             ("psu.01", "drive_cage.01", 30_000),                  # 其他
             ("cover.01", "drive_cage.01", 30_000),                # 其他, ties broken by key
             ("psu.01", "motherboard.01", 30_000)]                 # 不挡 by class: not asked
    cands = pending_pairs(pairs, BASE, [], {})
    assert keys(cands) == [("optical_drive.01", "drive_cage.01"),
                           ("case_fan.01", "motherboard.01"),
                           ("drive_cage.01", "motherboard.01"),
                           ("optical_drive.01", "motherboard.01"),
                           ("cover.01", "drive_cage.01"),
                           ("psu.01", "drive_cage.01")]
    assert [c.tag for c in cands] == [TAG_USER_BLOCKS] * 4 + [TAG_OTHER] * 2
    first = cands[0]
    assert (first.above_class, first.below_class) == first.classes == ("optical_drive",
                                                                       "drive_cage")
    assert first.px == 90_000 and first.reason.startswith("B1 挡")
    gpu = pending_pairs([("expansion_card.gpu.01", "cover.01", 4000)], BASE, [], {})[0]
    assert gpu.classes == ("expansion_card.gpu", "cover")
    assert pending_pairs(list(reversed(pairs)), BASE, [], {}) == cands      # deterministic


def test_same_pair_pending_counts_the_other_rows_of_the_class_pair_on_this_frame():
    inst = dict(BASE)
    inst["optical_drive.02"] = rec("optical_drive.02", "optical_drive")
    inst["motherboard.02"] = rec("motherboard.02", "motherboard")
    pairs = [("optical_drive.01", "motherboard.01", 9000),
             ("optical_drive.02", "motherboard.01", 8000),
             ("optical_drive.02", "motherboard.02", 7000),
             ("psu.01", "drive_cage.01", 6000)]
    cands = pending_pairs(pairs, inst, [], {})
    assert [c.same_pair_pending for c in cands] == [2, 2, 2, 0]
    groups = class_pairs(cands)
    assert [(classes, [c.above for c in group]) for classes, group in groups] == [
        (("optical_drive", "motherboard"), ["optical_drive.01", "optical_drive.02",
                                            "optical_drive.02"])]
    # a ruled pair is no longer pending, so it no longer counts
    ruled = {("optical_drive.01", "motherboard.01"): {"ruling": NOT_BLOCKS}}
    assert [c.same_pair_pending for c in pending_pairs(pairs, inst, [], ruled)] == [1, 1, 0]


def test_only_pairs_inside_the_chassis_are_offered():
    def inst(placement):
        return SimpleNamespace(placement=placement)
    compiled = SimpleNamespace(instances={"a": inst("in_chassis"), "b": inst("in_chassis"),
                                          "c": inst("on_bench"), "d": inst("context")})
    pairs = {("a", "b"): 5, ("c", "a"): 7, ("a", "d"): 9, ("b", "a"): 3}
    assert chassis_pairs(compiled, pairs) == [("a", "b", 5), ("b", "a", 3)]


def test_the_texts():
    cand = Candidate("optical_drive.01", "motherboard.01", "optical_drive", "motherboard",
                     68797, TAG_USER_BLOCKS, "r")
    assert candidate_text(cand) == "optical_drive.01 → motherboard.01 · 像素 68 797 · 用户已定：挡"
    assert status_text(13, 7, [("optical_drive.01", "motherboard.01")], BLOCKS) \
        == "D13 第 7 帧：optical_drive.01 压着 motherboard.01 → 挡着"
    assert status_text(5, 12, [("a", "b"), ("c", "b")], NOT_BLOCKS,
                       ("optical_drive", "motherboard")) \
        == "D05 第 12 帧：2 对 optical_drive 压着 motherboard → 不挡"
    row = {"ruling": BLOCKS, "view": "scan", "step": 7}
    assert ruled_text("optical_drive.01", "motherboard.01", row) \
        == "✓ optical_drive.01 → motherboard.01 挡着（scan 第 7 帧）"
    assert ruled_text("a", "b", {"ruling": NOT_BLOCKS}) == "✗ a → b 不挡"


def test_class_key_is_the_covers_spelling():
    assert class_key(BASE["cpu_cooler.heatsink.01"]) == "cpu_cooler.heatsink"
    assert class_key(rec("screw.motherboard.01", "screw", attrs={"role": "motherboard"})) \
        == "screw.motherboard"
    assert class_key(None) == "?"
    from tda.core.export.graph_covers import class_key as covers_key
    for r in BASE.values():
        assert class_key(r) == covers_key(r)


def test_no_qt_in_the_rule():
    import sys
    import subprocess
    code = ("import sys, tda.ui.cover_rulings; "
            "print(any(m.startswith('PySide6') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "False"
    assert np is not None and CR.__doc__ and "W3c" in CR.__doc__
