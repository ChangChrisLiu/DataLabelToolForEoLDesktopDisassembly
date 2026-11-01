"""Task W3g: the user's class-level rulings become rules (Qt-free except the dock test).

* the whitelist W2: an ``edge`` per pair (``covered_by`` / ``blocked_by.contained`` /
  ``blocked_by.physical_path``), a ``not_blocks:`` list, a class pair in both lists
  refused, W1's refusals kept;
* the new ``blocked_by`` mode ``contained``: gates the part's ``remove`` only and wants
  the blocker ``removed`` -- the cage may be swung aside with the drives inside, it may
  not leave until they are out; the Relations tooltip and the independent tables of
  the tests say so too;
* ``pixel-edges``: a pair writes the row of its edge, ``would_violate`` is judged with
  that row's real gating, class-level 不挡 pairs are never written (an explicit Review
  ``blocks`` wins and is logged), ``--undo pixels`` takes the W2 rows away;
* Review: 「用户已定：挡」 tags, the fourth fold 「用户已定不挡（类级）」.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from graph_scenes import DESKTOP
from test_w3a_pixel_edges import (  # noqa: F401 - env / no_app are fixtures
    HDD,
    PSU,
    add_rulings,
    env,
    evidence,
    no_app,
    owned_rows,
    run,
    whitelist_file,
)

from tda.cli import EXIT_ERROR, EXIT_OK
from tda.core import pixel_edges as PE
from tda.core.db import Db
from tda.core.graph import Edge, legal_actions, validate_sequence
from tda.core.graph_rules import (
    BLOCKED_GATES,
    BLOCKED_MODES,
    BLOCKED_REQUIRED,
    CONTAINED_MODE,
    gated_verbs,
    required_states,
)
from tda.core.model import ActionRec, InstanceRec
from tda.core.states import state_at
from tda.core.taxonomy import load_taxonomy
from tda.ui import cover_rulings as CR

TAX = load_taxonomy()
ROOT = Path(__file__).resolve().parents[1]
CAGE, SSD, ODD = "drive_cage.01", "storage_drive.ssd.01", "optical_drive.01"


def shipped() -> PE.Whitelist:
    return PE.load_whitelist(ROOT / PE.DEFAULT_WHITELIST, TAX)


# --------------------------------------------------------------------------- #
# the whitelist W2
# --------------------------------------------------------------------------- #
def test_the_shipped_whitelist_is_the_users_rulings():
    wl = shipped()
    assert wl.version == "W4" and wl.source == "pixels:W4"   # W3: the card rulings of 2026-10-31; W4: + the 47 cards
    edges = {p.label(): p.edge for p in wl.pairs}
    assert len(edges) == 32 and len(wl.not_blocks) == 12
    assert edges["cover -> ram_module"] == "covered_by" and edges["cover -> cpu_cooler"] == "covered_by"   # W4
    for label in ("drive_cage -> motherboard", "optical_drive -> motherboard",
                  "storage_drive.ssd -> motherboard", "storage_drive.hdd -> motherboard",
                  "case_fan -> motherboard", "cooler_bracket -> motherboard",
                  "optical_drive -> storage_drive.ssd", "optical_drive -> storage_drive.hdd",
                  "drive_cage -> storage_drive.hdd", "drive_cage -> ram_module",
                  "optical_drive -> ram_module", "case_fan -> ram_module", "case_fan -> cpu",
                  "case_fan -> cpu_socket_lever", "case_fan -> expansion_card.gpu",
                  "case_fan -> cpu_cooler.fan"):
        assert edges[label] == "covered_by", label
    for label in ("optical_drive -> drive_cage", "storage_drive.ssd -> drive_cage",
                  "storage_drive.hdd -> drive_cage"):              # B1 / B2 / B5
        assert edges[label] == "blocked_by.contained", label
    pair = wl.match("storage_drive.ssd", "drive_cage")
    assert (pair.edge_type, pair.mode) == ("blocked_by", "contained")
    assert wl.match("optical_drive", "motherboard").mode is None
    assert {p.min_frac for p in wl.pairs if p.below in ("cpu", "cpu_socket_lever",
                                                        "ram_module")} == {0.25}
    assert all(p.note for p in wl.pairs) and all(p.note for p in wl.not_blocks)
    assert "看图确认" in wl.match("optical_drive", "storage_drive.ssd").note  # confirmed 2026-10-31
    # 不挡 by class: A4, all of C (any kind of drive and cooler), D4
    no = {p.label() for p in wl.not_blocks}
    assert "psu -> motherboard" in no and "storage_drive.ssd -> ram_module" in no
    assert wl.not_blocking("storage_drive.unknown", "cpu_cooler.heatsink_fan") is not None
    assert wl.not_blocking("drive_cage", "cpu_socket_lever") is not None
    assert wl.not_blocking("case_fan", "cpu") is None                  # C11 is 挡
    assert wl.match("psu", "motherboard") is None
    # W3: the card rulings (B7-B10, A8, C11 ssd, E1-E5) are 挡 ...
    for label in ("storage_drive.ssd -> optical_drive", "storage_drive.ssd -> psu",
                  "psu -> storage_drive.ssd", "psu -> storage_drive.hdd", "misc_part -> motherboard",
                  "case_fan -> storage_drive.ssd", "cover -> cover", "cover -> expansion_card",
                  "storage_drive.ssd -> cover", "cover -> psu", "optical_drive -> optical_drive"):
        assert edges[label] == "covered_by", label
    assert wl.match("cover", "expansion_card.gpu") is not None          # the bare class covers every kind
    # ... and the reversed pairs the cards showed were layer errors, fixed in the data: in neither list
    for above, below in (("storage_drive.hdd", "psu"), ("expansion_card.gpu", "cover")):
        assert wl.match(above, below) is None and wl.not_blocking(above, below) is None
    doc = wl.as_json()
    assert doc["pairs"][6]["edge"] == "blocked_by.contained" and len(doc["not_blocks"]) == 12


@pytest.mark.parametrize("body, said", [
    ("pairs:\n  - {above: psu, below: motherboard, edge: blocks}\n", "edge 'blocks' is none of"),
    ("pairs:\n  - {above: psu, below: motherboard, edge: blocked_by.mounted}\n", "none of"),
    ("pairs:\n  - {above: psu, below: motherboard}\n"
     "not_blocks:\n  - {above: psu, below: motherboard}\n", "also in pairs"),
    # a class and one of its kinds can name the same pair: refused too
    ("pairs:\n  - {above: storage_drive.ssd, below: motherboard}\n"
     "not_blocks:\n  - {above: storage_drive, below: motherboard}\n", "also in pairs"),
    ("pairs:\n  - {above: drive_cage, below: cpu_cooler}\n"
     "not_blocks:\n  - {above: drive_cage, below: cpu_cooler.fan}\n", "also in pairs"),
    ("not_blocks:\n  - {above: spaceship, below: motherboard}\n", "no such class"),
    ("not_blocks:\n  - {above: psu}\n", "needs `above` and `below`"),
    # W1's refusals are kept
    ("pairs:\n  - {above: cover, below: motherboard, edge: blocked_by.contained}\n",
     "decided elsewhere"),
    ("pairs:\n  - {above: ram_module, below: motherboard}\n", "rule edge"),
    ("pairs:\n  - {above: drive_cage, below: screw}\n", "own rules"),
])
def test_the_loader_refuses(tmp_path, body, said):
    path = whitelist_file(tmp_path, "version: W2\n" + body)
    with pytest.raises(PE.WhitelistError) as excinfo:
        PE.load_whitelist(path, TAX)
    assert said in str(excinfo.value)


def test_the_loader_takes_the_three_edges_and_disjoint_lists(tmp_path):
    body = ("version: W2\npairs:\n"
            "  - {above: optical_drive, below: drive_cage, edge: blocked_by.contained}\n"
            "  - {above: psu, below: drive_cage, edge: blocked_by.physical_path}\n"
            "  - {above: drive_cage, below: motherboard}\n"
            "not_blocks:\n  - {above: storage_drive.hdd, below: motherboard, note: n}\n")
    wl = PE.load_whitelist(whitelist_file(tmp_path, body), TAX)
    assert [(p.edge_type, p.mode) for p in wl.pairs] == [
        ("blocked_by", "contained"), ("blocked_by", "physical_path"), ("covered_by", None)]
    assert [p.label() for p in wl.not_blocks] == ["storage_drive.hdd -> motherboard"]
    assert wl.not_blocking("storage_drive.hdd", "motherboard").note == "n"
    assert wl.not_blocking("storage_drive.ssd", "motherboard") is None


# --------------------------------------------------------------------------- #
# the mode `contained`
# --------------------------------------------------------------------------- #
def test_the_contained_mode_gates_remove_only_and_wants_removed():
    assert CONTAINED_MODE == "contained" and CONTAINED_MODE in BLOCKED_MODES
    assert BLOCKED_MODES[0] == "physical_path" and BLOCKED_MODES[-1] == CONTAINED_MODE
    assert BLOCKED_GATES["contained"] == gated_verbs("blocked_by", "contained") == {"remove"}
    assert BLOCKED_REQUIRED["contained"] == required_states("blocked_by", "contained") \
        == {"removed"}
    assert required_states("covered_by", "contained") == required_states("covered_by")


def test_the_relations_tooltip_and_the_independent_tables_know_it():
    from tda.ui.panels.relations import MODE_TOOLTIP

    line = next(l for l in MODE_TOOLTIP.splitlines() if l.strip().startswith("contained:"))
    assert "remove" in line and "displace" not in line
    assert "装在里面 / 上面的零件要先拿下来，架子可以先挪开" in line
    import test_graph_deadlock as deadlock
    import vlm_checker

    assert ("blocked_by", "contained") in deadlock.KINDS
    assert deadlock.MODE_REQUIRED["contained"] == {"removed"}
    assert vlm_checker.BLOCKED_GATES["contained"] == {"remove"}
    assert vlm_checker.BLOCKED_REQUIRED["contained"] == {"removed"}


def rec(key: str, cls: str, **attrs) -> InstanceRec:
    return InstanceRec(key=key, desktop=DESKTOP, cls=cls, attrs=attrs)


SCENE = {r.key: r for r in (rec(CAGE, "drive_cage"), rec(SSD, "storage_drive", kind="ssd"),
                            rec(ODD, "optical_drive"), rec("motherboard.01", "motherboard"))}


def contained(blocker: str = SSD) -> Edge:
    return Edge("blocked_by", CAGE, blocker, mode="contained", source="pixels:W2",
                status="accepted")


def act(step: int, target: str, verb: str) -> ActionRec:
    return ActionRec(desktop=DESKTOP, step=step, idx=0, target=target, verb=verb,
                     result="success")


def legal(state_overrides: dict) -> set:
    events = []
    state = state_at(SCENE, events, 0, TAX, initial=state_overrides)
    return set(legal_actions(SCENE, [contained()], state, TAX))


def test_the_cage_moves_aside_with_the_drive_inside_but_leaves_only_without_it():
    start = legal({})
    assert ("displace", CAGE) in start and ("remove", CAGE) not in start
    # a drive swung aside is still in the cage
    assert ("remove", CAGE) not in legal({SSD: "displaced"})
    assert ("remove", CAGE) in legal({SSD: "removed"})
    # the log: the cage displaced with the SSD installed is no breach (D30 step 9) ...
    log = [act(9, CAGE, "displace"), act(10, SSD, "remove"), act(11, CAGE, "remove")]
    assert validate_sequence(SCENE, [contained()], log, TAX) == []
    # ... where covered_by would make it one -- the reason B2 is not covered_by
    cover = Edge("covered_by", CAGE, SSD, source="pixels:W2", status="accepted")
    [line] = validate_sequence(SCENE, [cover], log, TAX)
    assert line.startswith("step 9: displace drive_cage.01 violates covered_by")
    # taking the cage out before the drive is a breach of `contained`
    early = [act(9, CAGE, "displace"), act(10, CAGE, "remove")]
    [line] = validate_sequence(SCENE, [contained()], early, TAX)
    assert line.startswith("step 10: remove drive_cage.01 violates blocked_by")


# --------------------------------------------------------------------------- #
# pixel-edges (pure)
# --------------------------------------------------------------------------- #
def cover(above: str, below: str, px: int = 20_000, amodal: int = 100_000, frames=(2, 3)):
    return {"above": above, "below": below, "above_class": CR.class_key(SCENE.get(above)),
            "below_class": CR.class_key(SCENE.get(below)), "max_px": px,
            "below_amodal_px": amodal, "frames": list(frames), "n_frames": len(frames),
            "first": frames[0]}


def wl_of(*pairs, not_blocks=()) -> PE.Whitelist:
    return PE.Whitelist("W2", "wl.yaml", 1000, 0.03,
                        tuple(PE.WhitelistPair(a, b, 1000, 0.03, f"note {a}", i, e)
                              for i, (a, b, e) in enumerate(pairs, 1)),
                        tuple(PE.ClassPair(a, b, "不挡 note", i)
                              for i, (a, b) in enumerate(not_blocks, 1)))


def judge(covers, log, whitelist, rulings=None, stored=None, instances=None):
    return PE.plan_desktop(DESKTOP, covers, 3, instances or SCENE, stored or [], log, [],
                           TAX, rulings or {}, whitelist)


def test_a_contained_pair_writes_a_blocked_by_row_judged_by_its_own_gating():
    log = [act(9, CAGE, "displace"), act(10, SSD, "remove"), act(11, CAGE, "remove")]
    plan = judge([cover(SSD, CAGE)], log,
                 wl_of(("storage_drive.ssd", "drive_cage", "blocked_by.contained")))
    [item] = plan.whitelist
    assert (item["edge"], item["edge_type"], item["mode"]) == (
        "blocked_by.contained", "blocked_by", "contained")
    assert item["source"] == "pixels:W2"
    assert item["reason"].endswith("whitelist W2 pair storage_drive.ssd -> drive_cage "
                                   "(blocked_by.contained): note storage_drive.ssd")
    edge = PE.edge_of(item)
    assert (edge.type, edge.target, edge.blocker, edge.mode, edge.necessity, edge.status) == (
        "blocked_by", CAGE, SSD, "contained", "required", "accepted")
    assert PE.edge_text(item) == f"blocked_by({CAGE}, {SSD}, mode=contained)"
    # the same pair as covered_by: the displace at step 9 breaks it
    plan = judge([cover(SSD, CAGE)], log, wl_of(("storage_drive.ssd", "drive_cage", "covered_by")))
    assert plan.whitelist == []
    [skip] = plan.skipped
    assert skip["why"] == "would_violate" and "step 9: displace" in skip["lines"][0]
    # a log that takes the cage out with the drive in it refutes `contained` too
    early = [act(9, CAGE, "remove"), act(10, SSD, "remove")]
    plan = judge([cover(SSD, CAGE)], early,
                 wl_of(("storage_drive.ssd", "drive_cage", "blocked_by.contained")))
    assert plan.whitelist == [] and plan.skipped[0]["why"] == "would_violate"
    covered = PE.edge_of({**item, "edge_type": None, "mode": None})
    assert covered.type == "covered_by" and covered.mode is None


def test_a_row_of_either_kind_keeps_the_pair_out():
    log = [act(10, SSD, "remove"), act(11, CAGE, "remove")]
    wl = wl_of(("storage_drive.ssd", "drive_cage", "blocked_by.contained"))
    rejected = Edge("covered_by", CAGE, SSD, source="manual", status="rejected")
    plan = judge([cover(SSD, CAGE)], log, wl, stored=[rejected])
    assert plan.whitelist == [] and plan.skipped[0]["why"] == "already_an_edge"
    assert plan.skipped[0]["row_status"] == "rejected"


def test_class_not_blocks_are_never_written_unless_review_says_blocks():
    log = [act(10, ODD, "remove"), act(11, "motherboard.01", "remove")]
    wl = wl_of(("drive_cage", "motherboard", "covered_by"),
               not_blocks=[("optical_drive", "motherboard")])
    plan = judge([cover(ODD, "motherboard.01")], log, wl)
    assert plan.whitelist == [] and plan.review == [] and plan.skipped == []
    [no] = plan.class_not_blocks
    assert (no["above"], no["class_pair"], no["note"]) == (ODD, "optical_drive -> motherboard",
                                                          "不挡 note")
    # an explicit Review ruling on that very pair wins, and is logged
    ruling = {(ODD, "motherboard.01"): {"ruling": "blocks", "by": "user", "ts": "2026-11-01",
                                        "view": "scan", "step": 3}}
    plan = judge([cover(ODD, "motherboard.01")], log, wl, rulings=ruling)
    [item] = plan.review
    assert item["source"] == "review:covers" and item["edge_type"] == "covered_by"
    [over] = plan.ruling_overrides_class
    assert over["class_pair"] == "optical_drive -> motherboard" and over["ruling"]["by"] == "user"
    assert plan.class_not_blocks == []
    doc = PE.evidence_json([plan], wl, True, "database", "scan", True, False)
    assert doc["counts"]["ruling_overrides_class"] == 1 and doc["counts"]["class_not_blocks"] == 0
    assert doc["whitelist"]["not_blocks"][0]["above"] == "optical_drive"
    doc = PE.evidence_json([judge([cover(ODD, "motherboard.01")], log, wl)], wl, True,
                           "database", "scan", True, False)
    assert doc["class_not_blocks_by_pair"] == {"optical_drive -> motherboard": 1}
    assert doc["desktops"][0]["class_not_blocks"] == 1


def test_a_ruled_pair_takes_its_class_pairs_edge():
    log = [act(9, CAGE, "displace"), act(10, SSD, "remove"), act(11, CAGE, "remove")]
    ruling = {(SSD, CAGE): {"ruling": "blocks", "by": "user", "ts": "2026-11-01",
                            "view": "scan", "step": 3}}
    plan = judge([cover(SSD, CAGE)], log,
                 wl_of(("storage_drive.ssd", "drive_cage", "blocked_by.contained")),
                 rulings=ruling)
    [item] = plan.review
    assert item["source"] == "review:covers" and item["whitelist_pair"] is None
    assert (item["edge_type"], item["mode"]) == ("blocked_by", "contained")
    # no whitelist: a ruling is covered_by as in W3a
    plan = judge([cover(SSD, CAGE)], [act(10, SSD, "remove")], None, rulings=ruling)
    assert plan.review[0]["edge_type"] == "covered_by"


def test_a_whitelist_row_whose_class_is_now_not_blocks_is_stale():
    row = Edge("blocked_by", CAGE, ODD, mode="contained", source="pixels:W2",
               status="accepted", reason="r")
    wl = wl_of(not_blocks=[("optical_drive", "drive_cage")])
    plan = judge([], [], wl, stored=[row])
    [stale] = plan.stale
    assert stale["why"] == "class_not_blocks" and stale["row"]["type"] == "blocked_by"
    # still backed by its class pair: not stale
    plan = judge([], [], wl_of(("optical_drive", "drive_cage", "blocked_by.contained")),
                 stored=[row])
    assert plan.stale == []


# --------------------------------------------------------------------------- #
# the command
# --------------------------------------------------------------------------- #
def rows(path) -> list[dict]:
    db = Db(str(path))
    try:
        return [dict(r) for r in db.relations(DESKTOP)]
    finally:
        db.close()


def test_apply_writes_a_contained_row_and_undo_pixels_takes_it_away(env, capsys):
    path = whitelist_file(env["tmp"], "version: W2\ndefaults: {min_px: 10, min_frac: 0.1}\n"
                                      "pairs:\n  - {above: psu, below: storage_drive, "
                                      "edge: blocked_by.contained, note: 盘装着电源}\n")
    assert run(env, "--whitelist", str(path), "--allow-unruled") == EXIT_OK
    assert f"would write blocked_by({HDD}, {PSU}, mode=contained)" in capsys.readouterr().out
    assert run(env, "--whitelist", str(path), "--allow-unruled", "--apply") == EXIT_OK
    [row] = [r for r in rows(env["db"]) if r["source"] == "pixels:W2"]
    assert (row["type"], row["target"], row["blocker"], row["mode"], row["status"],
            row["necessity"]) == ("blocked_by", HDD, PSU, "contained", "accepted", "required")
    assert row["reason"].endswith("(blocked_by.contained): 盘装着电源")
    doc = evidence(env)
    assert doc["whitelist_edges"][0]["edge_type"] == "blocked_by"
    assert doc["whitelist"]["pairs"][0]["edge"] == "blocked_by.contained"
    assert run(env, "--undo", "pixels") == EXIT_OK and owned_rows(env["db"]) == []


def test_a_class_not_blocks_pair_is_listed_never_written_and_a_ruling_wins(env):
    path = whitelist_file(env["tmp"], "version: W2\ndefaults: {min_px: 10, min_frac: 0.1}\n"
                                      "pairs: []\nnot_blocks:\n"
                                      "  - {above: psu, below: storage_drive, note: 独立}\n")
    assert run(env, "--whitelist", str(path), "--allow-unruled", "--apply") == EXIT_OK
    assert owned_rows(env["db"]) == []
    doc = evidence(env)
    assert [(i["above"], i["below"]) for i in doc["class_not_blocks"]] == [(PSU, HDD)]
    assert doc["class_not_blocks_by_pair"] == {"psu -> storage_drive": 1}
    add_rulings(env["db"], (PSU, HDD, "blocks"))
    assert run(env, "--whitelist", str(path), "--allow-unruled", "--apply") == EXIT_OK
    [row] = [r for r in rows(env["db"]) if r["source"] == "review:covers"]
    assert (row["type"], row["target"], row["blocker"]) == ("covered_by", HDD, PSU)
    doc = evidence(env)
    assert doc["class_not_blocks"] == [] and len(doc["ruling_overrides_class"]) == 1


def test_a_stale_whitelist_row_of_a_not_blocks_class_is_deleted(env):
    db = Db(str(env["db"]))
    try:
        db.add_relation(DESKTOP, "blocked_by", HDD, PSU, mode="contained", source="pixels:W2",
                        status="accepted")
    finally:
        db.close()
    path = whitelist_file(env["tmp"], "version: W2\nnot_blocks:\n"
                                      "  - {above: psu, below: storage_drive}\n")
    assert run(env, "--whitelist", str(path)) == EXIT_ERROR        # stale: dry run says so
    assert evidence(env)["stale_review_edges"][0]["why"] == "class_not_blocks"
    assert run(env, "--whitelist", str(path), "--apply") == EXIT_OK
    assert owned_rows(env["db"]) == []


# --------------------------------------------------------------------------- #
# Review
# --------------------------------------------------------------------------- #
def review_scene() -> dict:
    return {r.key: r for r in (
        rec("drive_cage.01", "drive_cage"), rec("cpu.01", "cpu"),
        rec("psu.01", "psu"), rec("motherboard.01", "motherboard"),
        rec("optical_drive.01", "optical_drive"),
        rec("storage_drive.hdd.01", "storage_drive", kind="hdd"),
        rec("ram_module.01", "ram_module"), rec("cover.01", "cover"))}


def test_triage_folds_the_class_pairs_ruled_not_blocks():
    inst = review_scene()
    pairs = [("drive_cage.01", "cpu.01", 4406), ("psu.01", "motherboard.01", 17_586),
             ("storage_drive.hdd.01", "ram_module.01", 4092),
             ("optical_drive.01", "motherboard.01", 68_797),
             ("cover.01", "psu.01", 2259)]
    tri = CR.triage_pairs(pairs, inst, [], {})
    assert [(c.above, c.below) for c in tri.class_not_blocks] == [
        ("psu.01", "motherboard.01"), ("drive_cage.01", "cpu.01"),
        ("storage_drive.hdd.01", "ram_module.01")]
    assert {c.tag for c in tri.class_not_blocks} == {CR.TAG_USER_NOT_BLOCKS}
    assert tri.class_not_blocks[0].reason.startswith("A4 不挡")
    assert [(c.above, c.below, c.tag) for c in tri.asked] == [
        ("optical_drive.01", "motherboard.01", CR.TAG_USER_BLOCKS),
        ("cover.01", "psu.01", CR.TAG_USER_BLOCKS)]          # E4: 挡 since W3 (the card rulings)
    assert tri.asked[0].reason == "A2 挡" and tri.asked[1].reason.startswith("E4 挡")
    assert CR.class_not_blocks_text(tri.class_not_blocks[1]) == (
        "drive_cage.01 → cpu.01 · 像素 4 406 · C 不挡：只是拆的习惯顺序")
    # without class rulings every one of them is asked, as before W3g
    plain = CR.triage_pairs(pairs, inst, [], {}, classes=CR.ClassRulings())
    assert plain.class_not_blocks == [] and len(plain.asked) == 5
    # a ruled pair is in no list; an indirect chain wins over the class ruling
    ruled = CR.triage_pairs(pairs, inst, [], {("psu.01", "motherboard.01"): {"ruling": "blocks"}})
    assert ("psu.01", "motherboard.01") not in [(c.above, c.below)
                                                for c in ruled.class_not_blocks]
    chain = [Edge("covered_by", "cpu.01", "ram_module.01"),
             Edge("covered_by", "ram_module.01", "drive_cage.01")]
    tri = CR.triage_pairs([("drive_cage.01", "cpu.01", 4406)], inst, chain, {})
    assert tri.class_not_blocks == [] and len(tri.indirect) == 1


def test_the_class_rulings_are_read_through_the_pixel_edges_loader(tmp_path):
    rulings = CR.class_rulings()
    assert rulings == CR.ClassRulings.from_whitelist(shipped())
    assert CR.class_rulings() is rulings                         # read once
    broken = whitelist_file(tmp_path, "version: nope\n")
    assert CR.class_rulings(str(broken)) == CR.ClassRulings()    # refused: nothing folded


def test_the_dock_folds_them_with_no_button(qapp_offscreen):
    from tda.ui.panels.notes_dock import CLASS_NO_TITLE, RulingsSection

    inst = review_scene()
    tri = CR.triage_pairs([("drive_cage.01", "cpu.01", 4406),
                           ("optical_drive.01", "motherboard.01", 68_797)], inst, [], {})
    section = RulingsSection()
    section.set_rulings(tri.asked, {}, class_not_blocks=tri.class_not_blocks)
    assert section.class_no_toggle.text() == CLASS_NO_TITLE.format(k=1) \
        == "用户已定不挡（类级）（1）"
    assert not section.class_no_toggle.isHidden() and not section.class_no_toggle.isChecked()
    assert section.class_no_texts() == ["drive_cage.01 → cpu.01 · 像素 4 406 · "
                                        "C 不挡：只是拆的习惯顺序"]
    from PySide6.QtWidgets import QPushButton
    assert section.class_no.findChildren(QPushButton) == []
    assert ("drive_cage.01", "cpu.01") in [pair for pair, _row in section.hover_rows]
    assert section.texts()[0].endswith("· 用户已定：挡")
    [(cand, blocks, free)] = section.row_buttons                # 挡 by class: both buttons
    assert (blocks.text(), free.text()) == ("挡着", "不挡")
    section.set_rulings([], {})
    assert section.class_no_toggle.isHidden()


@pytest.fixture
def qapp_offscreen():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_no_new_keys_and_the_docs_say_it():
    from tda.ui import app_actions as A

    for action in A.ACTIONS:                                     # folds only, no key
        assert not any(word in (action.label_zh or "") for word in ("类级", "用户已定"))
    spec = (ROOT / "docs" / "superpowers" / "specs"
            / "2026-09-17-teardown-annotator-design.md").read_text(encoding="utf-8")
    assert "blocked_by(mode ∈ {physical_path, tool_access, cable_tension, mounted, contained})" \
        in spec
    assert "| `blocked_by(mode=contained)`（v1.52，W3g） | Y ∈ {removed}" in spec
    assert "| `blocked_by(mode=contained)`（v1.52，W3g） | 仅 remove |" in spec
    section = spec.split("### 7.3 生成", 1)[1].split("### 7.4", 1)[0]
    [para] = [line for line in section.splitlines()
              if line.startswith("**用户的类级裁定：白名单 W2")]
    for needed in ("`version: W2`", "`blocked_by.contained`", "`contained`", "`not_blocks:`",
                   "`class_not_blocks`", "`ruling_overrides_class`", "「用户已定不挡（类级）（k）」",
                   "drive_latch"):
        assert needed in para, needed
    entries = [line for line in spec.split("## 14. 修订记录", 1)[1].splitlines()
               if line.startswith("- ")]
    # on top when it landed; whatever lands later goes above it
    [entry] = [e for e in entries[:4] if e.startswith("- **v1.52（2026-10-31）**")]
    for needed in ("（W3g，", "**216**", "`would_violate` **0**", "矛盾仍 **24**",
                   "2 070 → 1 854", "13 / 35 / 51 → 13 / 37 / 51", "348 对要问 → 249",
                   "50 → 44"):
        assert needed in entry, needed
    assert spec.index("- **v1.52") < spec.index("- **v1.51")
    handoff = (ROOT / "docs" / "HANDOFF_2026-10-28.md").read_text(encoding="utf-8")
    assert "`pixels:W2`" in handoff and "blocked_by(mode=contained)" in handoff
    quick = (ROOT / "docs" / "trial_quickstart.md").read_text(encoding="utf-8")
    assert "「用户已定不挡（类级）（k）」" in quick and "「建议收」" not in quick
    yaml_text = (ROOT / PE.DEFAULT_WHITELIST).read_text(encoding="utf-8")
    assert "version: W4" in yaml_text and "not_blocks:" in yaml_text   # W3 = W2 + the card rulings; W4 = + the 47 cards
    assert json.loads(json.dumps(shipped().as_json()))["version"] == "W4"
