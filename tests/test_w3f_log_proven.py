"""Task W3f: the pairs the log or the graph already decide are not asked (Qt-free).

* 日志已证明不挡 (:func:`tda.ui.cover_rulings.log_proof`): ``below`` removed /
  opened / displaced while ``above`` was still in the way -> folded, not asked;
  ``above`` opened first -> asked; ``below`` never acted on -> asked; a failed
  action proves nothing; the compound step's earlier action counts; the replay
  agrees with ``validate_sequence`` (the edge it proves impossible is a breach);
* 规则已间接覆盖 (:func:`tda.ui.cover_rulings.indirect_chain`), the D04 shape:
  cover -> screws -> board and cover -> card -> board fold away with the chain;
  no chain (or a rejected link) -> still asked;
* the row text with both states, the 日志 line, the folded rows' texts.
"""
from __future__ import annotations

from tda.core.graph import Edge, validate_sequence
from tda.core.model import ActionRec, InstanceRec
from tda.core.taxonomy import load_taxonomy
from tda.ui import cover_rulings as CR
from tda.ui.cover_rulings import (
    COVER_CLEAR,
    COVER_GATED,
    candidate_text,
    chain_text,
    chain_tip,
    cover_log,
    indirect_chain,
    indirect_text,
    log_line,
    log_proof,
    pending_pairs,
    proven_text,
    triage_pairs,
)

D = 4
TAX = load_taxonomy()


def rec(key: str, cls: str, **kw) -> InstanceRec:
    attrs = kw.pop("attrs", {})
    return InstanceRec(key=key, desktop=D, cls=cls, attrs=attrs, **kw)


def act(step: int, target: str, verb: str, idx: int = 0, result: str = "success") -> ActionRec:
    return ActionRec(desktop=D, step=step, idx=idx, target=target, verb=verb, result=result)


SCENE = {r.key: r for r in (
    rec("cover.01", "cover", attrs={"of": "expansion_slot"}),
    rec("psu.01", "psu"),
    rec("motherboard.01", "motherboard"),
    rec("drive_cage.01", "drive_cage"),
    rec("optical_drive.01", "optical_drive"),
    rec("screw.motherboard.03", "screw", fastens="motherboard.01"),
    rec("expansion_card.wlan.01", "expansion_card", attrs={"kind": "wlan"},
        mounted_on="motherboard.01"),
)}

#: D04's order: the cover opens at 22 with the PSU in place, the PSU leaves at 39.
D04_LOG = [act(22, "cover.01", "open"), act(39, "psu.01", "remove"),
           act(46, "motherboard.01", "remove")]


def keys(cands) -> list[tuple[str, str]]:
    return [(c.above, c.below) for c in cands]


def test_the_gated_verbs_and_the_freeing_states_are_covered_by_s():
    assert COVER_GATED == {"remove", "displace", "open", "unscrew", "disconnect", "release"}
    assert COVER_CLEAR == {"open", "removed", "displaced"}


def test_below_opened_while_above_untouched_is_proven_and_not_asked():
    log = cover_log(SCENE, D04_LOG, TAX)
    move = log_proof(log, "psu.01", "cover.01")
    assert (move.step, move.verb, move.target) == (22, "open", "cover.01")
    assert move.before["psu.01"] == "installed"
    tri = triage_pairs([("psu.01", "cover.01", 1535)], SCENE, [], {}, log=log)
    assert tri.asked == [] and keys(tri.proven) == [("psu.01", "cover.01")]
    assert tri.proven[0].proof == move
    assert pending_pairs([("psu.01", "cover.01", 1535)], SCENE, [], {}, log=log) == []
    # the edge it rules out really is one the log breaks (W3a would skip it)
    edge = Edge(type="covered_by", target="cover.01", blocker="psu.01")
    assert any("open cover.01 violates" in line
               for line in validate_sequence(SCENE, [edge], D04_LOG, TAX))


def test_below_removed_while_above_untouched_is_proven():
    actions = [act(5, "drive_cage.01", "remove"), act(9, "optical_drive.01", "remove")]
    log = cover_log(SCENE, actions, TAX)
    assert log_proof(log, "optical_drive.01", "drive_cage.01").step == 5
    tri = triage_pairs([("optical_drive.01", "drive_cage.01", 9000)], SCENE, [], {}, log=log)
    assert keys(tri.proven) == [("optical_drive.01", "drive_cage.01")] and tri.asked == []


def test_above_out_of_the_way_first_is_asked():
    actions = [act(5, "optical_drive.01", "remove"), act(9, "drive_cage.01", "remove")]
    log = cover_log(SCENE, actions, TAX)
    assert log_proof(log, "optical_drive.01", "drive_cage.01") is None
    assert keys(pending_pairs([("optical_drive.01", "drive_cage.01", 9000)], SCENE, [], {},
                              log=log)) == [("optical_drive.01", "drive_cage.01")]
    # opened (not removed) first is out of the way too
    log = cover_log(SCENE, [act(3, "cover.01", "open"), act(9, "psu.01", "remove")], TAX)
    assert log_proof(log, "cover.01", "psu.01") is None


def test_below_never_acted_on_is_asked():
    log = cover_log(SCENE, [act(3, "psu.01", "remove")], TAX)
    assert log_proof(log, "psu.01", "motherboard.01") is None
    # no class rulings: the shipped W2 rules psu -> motherboard 不挡 (task W3g)
    assert keys(pending_pairs([("psu.01", "motherboard.01", 20_000)], SCENE, [], {},
                              log=log, classes=CR.ClassRulings())) == [
        ("psu.01", "motherboard.01")]


def test_a_failed_action_proves_nothing_and_the_compound_step_counts():
    log = cover_log(SCENE, [act(5, "drive_cage.01", "remove", result="failed")], TAX)
    assert log_proof(log, "optical_drive.01", "drive_cage.01") is None
    # one step: the drive comes out first (idx 0), then the cage (idx 1)
    log = cover_log(SCENE, [act(5, "drive_cage.01", "remove", idx=1),
                            act(5, "optical_drive.01", "remove", idx=0)], TAX)
    assert log_proof(log, "optical_drive.01", "drive_cage.01") is None
    # the other way round within the step
    log = cover_log(SCENE, [act(5, "drive_cage.01", "remove", idx=0),
                            act(5, "optical_drive.01", "remove", idx=1)], TAX)
    assert log_proof(log, "optical_drive.01", "drive_cage.01").idx == 0


def test_a_part_the_machine_lacks_is_not_replayed():
    log = cover_log(SCENE, [act(5, "drive_cage.01", "remove")], TAX,
                    initial={"optical_drive.01": "removed"})
    assert log_proof(log, "optical_drive.01", "drive_cage.01") is None
    log = cover_log(SCENE, [act(5, "drive_cage.01", "remove")], TAX,
                    initial={"drive_cage.01": "removed"})
    assert log.of("drive_cage.01") == []


def test_without_a_log_nothing_is_proven():
    tri = triage_pairs([("psu.01", "cover.01", 1535)], SCENE, [], {})
    assert keys(tri.asked) == [("psu.01", "cover.01")] and tri.proven == []
    assert tri.asked[0].log == ""


# --------------------------------------------------------------------------- #
# 规则已间接覆盖
# --------------------------------------------------------------------------- #
SCREWS = [Edge(type="covered_by", target="screw.motherboard.03", blocker="cover.01"),
          Edge(type="fastened_by", target="motherboard.01", blocker="screw.motherboard.03")]
CARD = [Edge(type="covered_by", target="expansion_card.wlan.01", blocker="cover.01"),
        Edge(type="blocked_by", target="motherboard.01", blocker="expansion_card.wlan.01",
             mode="mounted")]


def test_cover_screws_board_is_indirect():
    chain = indirect_chain(SCREWS, "cover.01", "motherboard.01")
    assert [e.blocker for e in chain] == ["screw.motherboard.03", "cover.01"]
    tri = triage_pairs([("cover.01", "motherboard.01", 3823)], SCENE, SCREWS, {})
    assert tri.asked == [] and keys(tri.indirect) == [("cover.01", "motherboard.01")]
    assert chain_text(tri.indirect[0]) == "cover.01 → screw.motherboard.03 → motherboard.01"
    assert indirect_text(tri.indirect[0]) == \
        "cover.01 → screw.motherboard.03 → motherboard.01 · 像素 3 823"
    assert chain_tip(tri.indirect[0]) == (
        "motherboard.01 等 screw.motherboard.03（fastened_by）\n"
        "screw.motherboard.03 等 cover.01（covered_by）")


def test_cover_card_board_is_indirect_and_wins_over_the_log():
    log = cover_log(SCENE, D04_LOG, TAX)
    tri = triage_pairs([("cover.01", "motherboard.01", 3823)], SCENE, CARD, {}, log=log)
    assert keys(tri.indirect) == [("cover.01", "motherboard.01")]
    assert chain_text(tri.indirect[0]) == "cover.01 → expansion_card.wlan.01 → motherboard.01"


def test_no_chain_is_still_asked():
    assert indirect_chain(SCREWS[:1], "cover.01", "motherboard.01") == ()
    # the chain the wrong way round orders nothing
    upside = [Edge(type="covered_by", target="cover.01", blocker="screw.motherboard.03"),
              Edge(type="fastened_by", target="screw.motherboard.03", blocker="motherboard.01")]
    assert indirect_chain(upside, "cover.01", "motherboard.01") == ()
    # a rejected link breaks it
    dead = [SCREWS[0], Edge(type="fastened_by", target="motherboard.01",
                            blocker="screw.motherboard.03", status="rejected")]
    assert indirect_chain(dead, "cover.01", "motherboard.01") == ()
    assert keys(pending_pairs([("cover.01", "motherboard.01", 3823)], SCENE, dead, {})) == [
        ("cover.01", "motherboard.01")]


def test_a_deeper_chain_is_found_shortest_first():
    edges = [Edge(type="covered_by", target="motherboard.01", blocker="drive_cage.01"),
             Edge(type="covered_by", target="drive_cage.01", blocker="optical_drive.01"),
             Edge(type="covered_by", target="optical_drive.01", blocker="cover.01")]
    chain = indirect_chain(edges, "cover.01", "motherboard.01")
    assert [e.blocker for e in chain] == ["drive_cage.01", "optical_drive.01", "cover.01"]
    shortcut = edges + [Edge(type="connected_to", target="motherboard.01",
                             blocker="optical_drive.01")]
    assert len(indirect_chain(shortcut, "cover.01", "motherboard.01")) == 2


def test_same_pair_pending_counts_the_asked_list_only():
    log = cover_log(SCENE, [act(5, "drive_cage.01", "remove")], TAX)
    scene = dict(SCENE, **{"optical_drive.02": rec("optical_drive.02", "optical_drive"),
                           "drive_cage.02": rec("drive_cage.02", "drive_cage")})
    tri = triage_pairs([("optical_drive.01", "drive_cage.01", 9000),
                        ("optical_drive.02", "drive_cage.02", 8000)], scene, [], {}, log=log)
    assert keys(tri.proven) == [("optical_drive.01", "drive_cage.01")]
    assert [c.same_pair_pending for c in tri.asked] == [0]


# --------------------------------------------------------------------------- #
# the texts
# --------------------------------------------------------------------------- #
def test_the_row_text_says_both_states():
    states = {"cover.01": "open", "motherboard.01": "installed"}
    [cand] = pending_pairs([("cover.01", "motherboard.01", 3823)], SCENE, [], {}, states=states)
    assert candidate_text(cand) == "cover.01（打开）→ motherboard.01（装着） · 像素 3 823 · 其他"


def test_the_log_line():
    log = cover_log(SCENE, D04_LOG, TAX)
    assert log_line(log, "cover.01", "psu.01") == "日志：cover.01 第 22 步 打开，psu.01 第 39 步 拆下"
    assert log_line(log, "drive_cage.01", "optical_drive.01") == \
        "日志：drive_cage.01 从未动过，optical_drive.01 从未拆下"
    both = cover_log(SCENE, [act(3, "cover.01", "open"), act(8, "cover.01", "remove")], TAX)
    assert log_line(both, "cover.01", "psu.01") == \
        "日志：cover.01 第 3 步 打开、第 8 步 拆下，psu.01 从未拆下"
    assert log_line(None, "cover.01", "psu.01") == ""
    [cand] = pending_pairs([("cover.01", "psu.01", 5000)], SCENE, [], {}, log=log)
    assert cand.log == "日志：cover.01 第 22 步 打开，psu.01 第 39 步 拆下"


def test_the_proven_row_text():
    log = cover_log(SCENE, D04_LOG, TAX)
    tri = triage_pairs([("psu.01", "cover.01", 1535)], SCENE, [], {}, log=log,
                       states={"psu.01": "installed", "cover.01": "closed"})
    assert proven_text(tri.proven[0]) == (
        "psu.01（装着）→ cover.01（关上） · 像素 1 535 · 第 22 步 cover.01 打开时 psu.01 还是装着")


def test_the_module_stays_qt_free():
    import subprocess
    import sys

    code = ("import sys, tda.ui.cover_rulings; "
            "print(any(m.startswith('PySide6') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "False"
    assert callable(CR.triage_pairs)
