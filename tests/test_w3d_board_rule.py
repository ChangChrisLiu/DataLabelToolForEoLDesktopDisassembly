"""Everything on the motherboard comes off before the motherboard (task W3d).

The user's rule of 2026-10-31 (「所有主板上的东西都拆掉了，主板才能拆下来」): every
real ``cpu``, ``ram_module``, ``cpu_cooler`` and ``expansion_card`` gives a
``blocked_by(motherboard <- part, mode="mounted")`` rule edge, reason
``"<part> sits on <board>"``. The ``mounted`` mode gates ``remove`` and
``displace`` of the board and wants the part **removed** -- a RAM stick swung
aside does not free the board (round 2: ``covered_by`` accepted ``displaced``,
so every board plan swung the parts aside). S1 extras take no part (either
end), drafts are not instances, two real boards give no edge and one S1 line.
The export side's support table is built from the same class tuple, so Review
still never asks the pair and the export explains the covering by the new edge.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from graph_scenes import DESKTOP, act, bench_instances, good_sequence, state_after
from graph_scenes import inst as _inst
from graph_scenes import triples as _triples

from tda.cli import main
from tda.core.db import Db
from tda.core.export import graph_covers
from tda.core.graph import (
    REQUIRED_STATES,
    applicable_preconditions,
    gated_verbs,
    legal_actions,
    propose_edges,
    remaining_plan,
    required_states,
    unmet,
    unresolved_boards,
    validate_sequence,
)
from tda.core.graph_derive import derive_edges, settled_instances
from tda.core.graph_rules import (
    BLOCKED_MODES,
    MOTHERBOARD_CLASS,
    MOUNTED_MODE,
    MOUNTED_ON_MOTHERBOARD,
    board_of,
)
from tda.core.model import EXTRA_ATTR, LS_PREFIX, InstanceRec
from tda.core.states import initial_state
from tda.core.taxonomy import load_taxonomy

TAX = load_taxonomy()
BOARD = "motherboard.01"
FAN, HEATSINK = "cpu_cooler.fan.01", "cpu_cooler.heatsink.01"
ON_BENCH_BOARD = {"cpu.01", "ram_module.01", FAN}


def board_edges(instances, board: str = BOARD) -> set[tuple[str, str, str]]:
    """The ``blocked_by`` edges onto ``board`` (the only rule that derives one)."""
    return {t for t in _triples(propose_edges(instances, TAX))
            if t[0] == "blocked_by" and t[1] == board}


def extra(rec: InstanceRec) -> InstanceRec:
    rec.attrs = {**rec.attrs, EXTRA_ATTR: "tester"}
    return rec


# --------------------------------------------------------------------------- #
# the mode
# --------------------------------------------------------------------------- #
def test_the_mounted_mode_gates_like_physical_path_and_wants_removed():
    # W3d appended it; W3g's `contained` comes after it
    assert MOUNTED_MODE == "mounted" and BLOCKED_MODES[3] == MOUNTED_MODE
    assert BLOCKED_MODES[0] == "physical_path"      # the Relations panel's default
    assert gated_verbs("blocked_by", "mounted") == frozenset({"remove", "displace"})
    assert required_states("blocked_by", "mounted") == frozenset({"removed"})
    assert required_states("blocked_by", "physical_path") == REQUIRED_STATES["blocked_by"]
    assert required_states("blocked_by") == REQUIRED_STATES["blocked_by"]
    assert required_states("covered_by") == REQUIRED_STATES["covered_by"]
    assert required_states("covered_by", "mounted") == REQUIRED_STATES["covered_by"]
    assert required_states("no_such_type") == frozenset()


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def test_the_bench_gets_exactly_three_new_edges_with_their_reasons():
    bench = bench_instances()
    assert board_edges(bench) == {("blocked_by", BOARD, key) for key in ON_BENCH_BOARD}
    mounted = [e for e in propose_edges(bench, TAX)
               if e.type == "blocked_by" and e.target == BOARD]
    assert {e.blocker: e.reason for e in mounted} == {
        key: f"{key} sits on {BOARD}" for key in ON_BENCH_BOARD}
    for edge in mounted:
        assert (edge.mode, edge.source, edge.status, edge.necessity) == (
            MOUNTED_MODE, "rule", "proposed", "required")
    # no covered_by onto the board any more (round 1's type)
    assert not [e for e in propose_edges(bench, TAX)
                if e.type == "covered_by" and e.target == BOARD]
    assert board_of(bench) == (BOARD, None) and unresolved_boards(bench) == []


def test_both_parts_of_a_two_part_cooler_sit_on_the_board():
    desk = bench_instances()
    desk[HEATSINK] = _inst(HEATSINK, "cpu_cooler", attrs={"kind": "heatsink"})
    assert {("blocked_by", BOARD, FAN), ("blocked_by", BOARD, HEATSINK)} <= board_edges(desk)


def test_every_mounted_class_and_nothing_else():
    desk = bench_instances()
    desk["expansion_card.gpu.01"] = _inst("expansion_card.gpu.01", "expansion_card",
                                          attrs={"kind": "gpu"})
    desk["ram_module.02"] = _inst("ram_module.02", "ram_module")
    desk["cooler_bracket.01"] = _inst("cooler_bracket.01", "cooler_bracket")
    desk["case_fan.01"] = _inst("case_fan.01", "case_fan")
    blockers = {t[2] for t in board_edges(desk)}
    assert blockers == {*ON_BENCH_BOARD, "expansion_card.gpu.01", "ram_module.02"}
    # the PSU, the drive, the bracket and the case fan are not on this rule's list:
    # "the board last" is not a rule (D01 D02 D35 D36 take the PSU out after it)
    assert {desk[k].cls for k in blockers} == set(MOUNTED_ON_MOTHERBOARD)


def test_an_extra_ram_stick_takes_no_edge():
    desk = bench_instances()
    desk["ram_module.02"] = extra(_inst("ram_module.02", "ram_module"))
    assert ("blocked_by", BOARD, "ram_module.02") not in board_edges(desk)
    assert board_edges(desk) == {("blocked_by", BOARD, key) for key in ON_BENCH_BOARD}


def test_an_extra_board_takes_no_edge_at_all():
    desk = bench_instances()
    extra(desk[BOARD])
    assert board_edges(desk) == set()
    assert unresolved_boards(desk) == []


def test_a_draft_second_board_leaves_the_board_unique():
    desk = bench_instances()
    draft = f"{LS_PREFIX}Motherboard#2"
    desk[draft] = _inst(draft, "motherboard")
    assert board_of(desk) == (BOARD, None)
    assert board_of(settled_instances(desk)) == (BOARD, None)
    derived = derive_edges(desk, [], TAX).edges
    assert {("blocked_by", BOARD, key) for key in ON_BENCH_BOARD} <= _triples(derived)
    assert {e.mode for e in derived if e.type == "blocked_by"} == {MOUNTED_MODE}
    assert not [t for t in _triples(derived) if draft in t]
    assert unresolved_boards(settled_instances(desk)) == []


def test_no_board_no_edge():
    desk = bench_instances()
    del desk[BOARD]
    assert not [t for t in _triples(propose_edges(desk, TAX))
                if t[0] == "blocked_by" and t[1].startswith(MOTHERBOARD_CLASS)]
    assert board_of(desk) == (None, None) and unresolved_boards(desk) == []


def test_two_real_boards_give_no_edge_and_one_s1_line():
    desk = bench_instances()
    desk["motherboard.02"] = _inst("motherboard.02", "motherboard")
    assert board_edges(desk) == set() and board_edges(desk, "motherboard.02") == set()
    [line] = unresolved_boards(desk)
    assert line.startswith("unresolved board:")
    assert BOARD in line and "motherboard.02" in line
    assert board_of(desk) == (None, line)


def test_constraints_reports_the_unresolved_board(tmp_path: Path):
    db_path = tmp_path / "tda.sqlite"
    cfg = {"cache_dir": str(tmp_path / "cache"), "db_path": str(db_path),
           "backup_dir": str(tmp_path / "backups"), "raw_logs_dir": str(tmp_path / "raw")}
    (tmp_path / "backups").mkdir()
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    db = Db(str(db_path))
    try:
        db.upsert_desktop(DESKTOP, {"brand": "Bench"})
        for rec in bench_instances().values():
            db.upsert_instance(rec)
        db.upsert_instance(InstanceRec(key="motherboard.02", desktop=DESKTOP,
                                       cls="motherboard"))
        db.replace_steps(DESKTOP, [], good_sequence())
    finally:
        db.close()
    report = tmp_path / "report.md"
    main(["--paths", str(tmp_path / "paths.yaml"), "constraints", "--desktops", str(DESKTOP),
          "--dry-run", "--report", str(report)])
    text = report.read_text(encoding="utf-8")
    assert "unresolved board" in text and "motherboard.02" in text


def test_the_mounted_mode_survives_the_database(tmp_path: Path):
    from tda.core.graph import edges_from_db, edges_to_db

    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        db.upsert_desktop(DESKTOP, {"brand": "Bench"})
        edges_to_db(db, DESKTOP, propose_edges(bench_instances(), TAX))
        back = [e for e in edges_from_db(db, DESKTOP) if e.type == "blocked_by"]
    finally:
        db.close()
    assert {(e.blocker, e.mode) for e in back} == {(k, MOUNTED_MODE) for k in ON_BENCH_BOARD}


# --------------------------------------------------------------------------- #
# what it does to the sequence, the legal set and the plan
# --------------------------------------------------------------------------- #
def test_the_good_teardown_validates_clean():
    bench = bench_instances()
    assert validate_sequence(bench, propose_edges(bench, TAX), good_sequence(), TAX) == []
    # it clears the board first and takes the board last
    order = [a.target for a in good_sequence()]
    assert order[-1] == BOARD
    assert all(order.index(k) < order.index(BOARD) for k in ON_BENCH_BOARD)


def test_the_board_is_not_legal_in_the_initial_state():
    bench = bench_instances()
    got = legal_actions(bench, propose_edges(bench, TAX), initial_state(bench, TAX), TAX)
    assert ("remove", BOARD) not in got and ("displace", BOARD) not in got


def test_the_board_plan_removes_what_is_mounted_and_ends_with_the_board():
    bench = bench_instances()
    plan = remaining_plan(bench, propose_edges(bench, TAX), initial_state(bench, TAX), BOARD,
                          TAX)
    assert plan is not None and plan[-1] == ("remove", BOARD)
    for key in ("ram_module.01", FAN, "cpu.01"):
        assert ("remove", key) in plan
        assert ("displace", key) not in plan
    assert not [p for p in plan if p[0] == "displace"]
    # the cooler is lifted off once, for the CPU and the board alike (the planner
    # takes a part out the first time it is in the way when the goal needs it out)
    assert plan.count(("remove", FAN)) == 1 and len(plan) == len(good_sequence())


def test_only_the_board_goal_insists_on_removal():
    """The CPU plan still swings the cooler aside: nothing there needs it removed."""
    bench = bench_instances()
    plan = remaining_plan(bench, propose_edges(bench, TAX), initial_state(bench, TAX),
                          "cpu.01", TAX)
    assert ("displace", FAN) in plan and plan[-1] == ("remove", "cpu.01")


def test_a_displaced_ram_stick_does_not_free_the_board():
    bench = bench_instances()
    edges = propose_edges(bench, TAX)
    # the good teardown, except that the RAM stick is only swung aside
    log = [act(a.step, a.target, "displace" if a.target == "ram_module.01" else a.verb)
           for a in good_sequence()[:-1]]
    state = state_after(bench, log, TAX)
    assert state["ram_module.01"].state == "displaced"
    gating = applicable_preconditions(edges, ("remove", BOARD))
    assert _triples(unmet(gating, state)) == {("blocked_by", BOARD, "ram_module.01")}
    got = legal_actions(bench, edges, state, TAX)
    assert ("remove", BOARD) not in got and ("displace", BOARD) not in got
    # ... and the replayed log says so
    lines = validate_sequence(bench, edges, [*log, act(15, BOARD, "remove")], TAX)
    assert len(lines) == 1 and "blocked_by(motherboard.01, ram_module.01)" in lines[0]
    # removing it does
    state = state_after(bench, [*log, act(15, "ram_module.01", "remove")], TAX)
    assert unmet(gating, state) == []


# --------------------------------------------------------------------------- #
# the export side and Review read the same class tuple
# --------------------------------------------------------------------------- #
def test_the_support_table_is_the_rule_tuple():
    assert graph_covers.SUPPORT == tuple((cls, MOTHERBOARD_CLASS)
                                         for cls in MOUNTED_ON_MOTHERBOARD)
    assert graph_covers.SUPPORTED_BY == {cls: MOTHERBOARD_CLASS
                                         for cls in MOUNTED_ON_MOTHERBOARD}
    from tda.ui import cover_rulings as ui

    assert ui.SUPPORT is graph_covers.SUPPORT


def test_review_still_never_asks_the_pair():
    from tda.ui import cover_rulings as ui

    bench = bench_instances()
    bench["expansion_card.gpu.01"] = _inst("expansion_card.gpu.01", "expansion_card",
                                           attrs={"kind": "gpu"})
    pairs = [(key, BOARD, 50_000) for key in (*ON_BENCH_BOARD, "expansion_card.gpu.01")]
    # with the rule edges, and without any edge (structural by the support table)
    assert ui.pending_pairs(pairs, bench, propose_edges(bench, TAX), {}) == []
    assert ui.pending_pairs(pairs, bench, [], {}) == []


@pytest.fixture
def board_scene(tmp_path, monkeypatch):
    """The U46 export-graph scene with the motherboard drawn under the CPU."""
    import test_u46_export_graph as U46

    shapes = {"chassis": U46.SHAPES["chassis"], BOARD: (1, 1, 19, 15),
              **{k: v for k, v in U46.SHAPES.items() if k != "chassis"}}
    monkeypatch.setattr(U46, "SHAPES", shapes)
    monkeypatch.setattr(U46, "ORDER", list(shapes))
    return U46.build(tmp_path / "tda.sqlite"), U46


def test_the_export_explains_the_cpu_over_the_board_by_the_rule_edge(board_scene):
    from tda.core.export.graph_export import ReadOnlyDb, build_desktop

    path, U46 = board_scene
    db = ReadOnlyDb(str(path))
    try:
        doc = build_desktop(db, TAX, DESKTOP, U46.VIEW)
    finally:
        db.close()
    covers = {(c["above"], c["below"]): c for c in doc["covers"]}
    cpu = covers[("cpu.01", BOARD)]
    assert cpu["explained_by"] == [f"blocked_by({BOARD}, cpu.01)"]
    # the bench CPU is `mounted_on` the board as well; the support table's reason stays
    assert graph_covers.SUPPORTED_BY_REASON in cpu["structural"]
    assert cpu["candidate"] is False and cpu["no_edge"] is False
    fan = covers[(FAN, BOARD)]
    assert f"blocked_by({BOARD}, {FAN})" in fan["explained_by"] and fan["candidate"] is False
    edge = next(e for e in doc["edges"] if e["label"] == f"blocked_by({BOARD}, cpu.01)")
    assert edge["mode"] == MOUNTED_MODE
    # the board plan removes the parts on it, never swings them aside
    plan = doc["plans"][BOARD]["plan"]
    assert "remove cpu.01" in plan and not [a for a in plan if a.startswith("displace")]


# --------------------------------------------------------------------------- #
# the docs
# --------------------------------------------------------------------------- #
def test_the_docs_say_it():
    root = Path(__file__).resolve().parents[1]
    spec = (root / "docs" / "superpowers" / "specs"
            / "2026-09-17-teardown-annotator-design.md").read_text(encoding="utf-8")
    rules = spec.split("### 7.3 生成", 1)[1].split("### 7.4", 1)[0]
    for needed in ("**主板上的东西先拆（v1.49，W3d）**", "`blocked_by(主板 ← 零件, mode=mounted)`",
                   "`unresolved board: …`", "D01 D02 D35 D36"):
        assert needed in rules, needed
    # right after the W3a cooler paragraph
    assert rules.index("**压在 CPU 上的散热器（v1.47，W3a）**") < rules.index(
        "**主板上的东西先拆（v1.49，W3d）**")
    semantics = spec.split("### 7.1", 1)[1].split("### 7.3", 1)[0]
    assert "mounted" in semantics and "blocked_by(mode=mounted)" in semantics
    entries = [line for line in spec.split("## 14. 修订记录", 1)[1].splitlines()
               if line.startswith("- ")]
    # on top when it landed; W3e's v1.50 has gone above it since
    [entry] = [e for e in entries[:5] if e.startswith("- **v1.49（2026-10-31）**")]
    assert "mode=mounted" in entry
    handoff = (root / "docs" / "HANDOFF_2026-10-28.md").read_text(encoding="utf-8")
    section = handoff.split("## 5. 导出与 HuggingFace 准备", 1)[1].split("\n## ", 1)[0]
    assert sum("MOUNTED_ON_MOTHERBOARD" in line for line in section.splitlines()) == 1
