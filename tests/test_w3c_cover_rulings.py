"""Task W3c: which overlapping pairs of a frame still want the user's 「谁压着谁」 ruling (Qt-free).

* the rule: pixels >= 1000; no Label Studio draft / 框外零件 / gallery-only /
  implied part / chassis / unknown key; no active edge of any type either way
  (a rejected edge does not count); no structural reason (parent, attached
  chain, mounted_on / fastens / socket_host / cable, the support list); no
  ruling yet for the ordered pair;
* the sheet's tags (建议收 / 存疑 / 其他, ``class`` or ``class.kind``, the most
  exact entry wins) with the reason; ``same_pair_pending``; the order
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
    RECOMMENDED,
    SUPPORT,
    TAG_OTHER,
    TAG_RECOMMENDED,
    TAG_UNCERTAIN,
    UNCERTAIN,
    Candidate,
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


# --------------------------------------------------------------------------- #
# tags, order, counts, texts
# --------------------------------------------------------------------------- #
def test_the_sheets_class_pairs_are_encoded_with_their_reasons():
    assert len(RECOMMENDED) == 19 and len(UNCERTAIN) == 21
    assert ("optical_drive", "motherboard") in {(a, b) for a, b, _ in RECOMMENDED}
    assert ("drive_cage", "motherboard") in {(a, b) for a, b, _ in UNCERTAIN}
    assert all(reason for *_pair, reason in RECOMMENDED + UNCERTAIN)
    assert not ({(a, b) for a, b, _ in RECOMMENDED} & {(a, b) for a, b, _ in UNCERTAIN})


def test_tags_name_a_class_or_a_class_kind_and_the_most_exact_wins():
    assert pair_tag("optical_drive", "motherboard")[0] == TAG_RECOMMENDED
    assert pair_tag("drive_cage", "motherboard") == (TAG_UNCERTAIN, "日志里上面那件先拆的只有 65%")
    assert pair_tag("optical_drive", "cpu_cooler.heatsink")[0] == TAG_RECOMMENDED
    assert pair_tag("optical_drive", "cpu_cooler.fan")[0] == TAG_UNCERTAIN       # the kind decides
    assert pair_tag("optical_drive", "cpu_cooler")[0] == TAG_OTHER                # no kind: no row
    assert pair_tag("expansion_card.gpu", "cover")[0] == TAG_UNCERTAIN
    assert pair_tag("expansion_card.other", "cover")[0] == TAG_OTHER
    assert pair_tag("cover.side", "cpu")[0] == TAG_RECOMMENDED                    # class: any kind
    assert pair_tag("case_fan", "motherboard") == (TAG_OTHER, "裁定单表 A 里没有建议：看图判断")


def test_candidates_carry_classes_tag_reason_and_the_order_is_tag_then_pixels():
    pairs = [("case_fan.01", "motherboard.01", 60_000),            # 其他
             ("drive_cage.01", "motherboard.01", 30_000),          # 存疑
             ("optical_drive.01", "motherboard.01", 20_000),       # 建议收
             ("optical_drive.01", "drive_cage.01", 90_000),        # 建议收
             ("psu.01", "motherboard.01", 30_000)]                 # 存疑, ties broken by key
    cands = pending_pairs(pairs, BASE, [], {})
    assert keys(cands) == [("optical_drive.01", "drive_cage.01"),
                           ("optical_drive.01", "motherboard.01"),
                           ("drive_cage.01", "motherboard.01"),
                           ("psu.01", "motherboard.01"),
                           ("case_fan.01", "motherboard.01")]
    assert [c.tag for c in cands] == [TAG_RECOMMENDED, TAG_RECOMMENDED, TAG_UNCERTAIN,
                                      TAG_UNCERTAIN, TAG_OTHER]
    first = cands[0]
    assert (first.above_class, first.below_class) == first.classes == ("optical_drive",
                                                                       "drive_cage")
    assert first.px == 90_000 and "刚性大件" in first.reason
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
             ("psu.01", "motherboard.01", 6000)]
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
                     68797, TAG_RECOMMENDED, "r")
    assert candidate_text(cand) == "optical_drive.01 → motherboard.01 · 像素 68 797 · 建议收"
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
