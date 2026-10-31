"""The cooler on the CPU covers the CPU and the socket lever (task W3a, deliverable 2.1).

The fixed spec-7.3 edge resolved "the cooler" with ``unique_of_class``, so the
nine desktops whose sheet lists a ``cpu_cooler.fan`` *and* a
``cpu_cooler.heatsink`` got no ``covered_by(cpu <- cooler)`` at all, and no
desktop had an edge from the cooler to the socket lever: every CPU plan opened
the lever first. Now one cooler -> it; a heatsink plus a fan -> the heatsink,
with the fan over it; anything else -> no edge and one S1 line.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from graph_scenes import DESKTOP, bench_instances, good_sequence
from graph_scenes import inst as _inst
from graph_scenes import triples as _triples

from tda.cli import main
from tda.core.db import Db
from tda.core.graph import legal_actions, propose_edges, unresolved_coolers
from tda.core.graph_plan import plan_removal
from tda.core.graph_rules import cooler_on_cpu
from tda.core.model import EXTRA_ATTR, InstanceRec
from tda.core.states import state_at
from tda.core.taxonomy import load_taxonomy

TAX = load_taxonomy()
FAN, HEATSINK = "cpu_cooler.fan.01", "cpu_cooler.heatsink.01"
LEVER = "cpu_socket_lever.01"


def two_part_desktop() -> dict[str, InstanceRec]:
    """The bench with a heatsink under its fan, as on D03 / D07 / D08 / ... / D64."""
    bench = bench_instances()
    bench[HEATSINK] = _inst(HEATSINK, "cpu_cooler", attrs={"kind": "heatsink"},
                            mounted_on="motherboard.01", slot_id="heatsink")
    return bench


def cooler_edges(instances) -> set[tuple[str, str, str]]:
    coolers = {k for k, r in instances.items() if r.cls == "cpu_cooler"}
    return {t for t in _triples(propose_edges(instances, TAX))
            if t[0] == "covered_by" and t[2] in coolers}


def test_one_cooler_covers_the_cpu_and_the_lever():
    bench = bench_instances()
    assert cooler_on_cpu(bench) == (FAN, None, None)
    assert cooler_edges(bench) == {("covered_by", "cpu.01", FAN),
                                   ("covered_by", LEVER, FAN)}
    reasons = {e.target: e.reason for e in propose_edges(bench, TAX) if e.blocker == FAN
               and e.type == "covered_by"}
    assert reasons == {"cpu.01": f"{FAN} sits on cpu.01",
                       LEVER: f"{FAN} sits on the socket lever {LEVER}"}
    assert unresolved_coolers(bench) == []


@pytest.mark.parametrize("kind", ["fan", "heatsink", "heatsink_fan"])
def test_a_single_cooler_of_any_kind_is_the_one_on_the_cpu(kind):
    bench = bench_instances()
    del bench[FAN]
    key = f"cpu_cooler.{kind}.01"
    bench[key] = _inst(key, "cpu_cooler", attrs={"kind": kind})
    assert cooler_edges(bench) == {("covered_by", "cpu.01", key), ("covered_by", LEVER, key)}


def test_heatsink_and_fan_the_heatsink_sits_on_the_cpu_and_the_fan_on_it():
    desk = two_part_desktop()
    assert cooler_on_cpu(desk) == (HEATSINK, FAN, None)
    assert cooler_edges(desk) == {("covered_by", "cpu.01", HEATSINK),
                                  ("covered_by", LEVER, HEATSINK),
                                  ("covered_by", HEATSINK, FAN)}
    [over] = [e for e in propose_edges(desk, TAX) if e.target == HEATSINK
              and e.type == "covered_by"]
    assert over.reason == f"{FAN} sits on {HEATSINK}"
    assert unresolved_coolers(desk) == []


def test_every_lever_is_covered_by_class():
    bench = bench_instances()
    bench["cpu_socket_lever.02"] = _inst("cpu_socket_lever.02", "cpu_socket_lever",
                                         parent="motherboard.01", attached=True)
    assert {t[1] for t in cooler_edges(bench)} == {"cpu.01", LEVER, "cpu_socket_lever.02"}


@pytest.mark.parametrize("extra", [
    {"cpu_cooler.fan.02": {"kind": "fan"}},                                   # two fans
    {HEATSINK: {"kind": "heatsink"}, "cpu_cooler.fan.02": {"kind": "fan"}},   # three parts
    {"cpu_cooler.heatsink_fan.01": {"kind": "heatsink_fan"}},                 # fan + combo
    {"cpu_cooler.01": {}},                                                    # no kind
])
def test_any_other_set_gives_no_edge_and_one_s1_line(extra):
    bench = bench_instances()
    for key, attrs in extra.items():
        bench[key] = _inst(key, "cpu_cooler", attrs=attrs)
    assert cooler_edges(bench) == set()
    on_cpu, fan, problem = cooler_on_cpu(bench)
    assert on_cpu is None and fan is None
    [line] = unresolved_coolers(bench)
    assert line == problem and line.startswith("unresolved cooler:")
    for key in (FAN, *extra):
        assert key in line
    # the CPU stays locked by its lever, as before; nothing is guessed
    assert ("locked_by", "cpu.01", LEVER) in _triples(propose_edges(bench, TAX))


def test_an_s1_added_part_takes_no_part_in_the_class_reading():
    bench = bench_instances()
    bench[FAN].attrs = {**bench[FAN].attrs, EXTRA_ATTR: "tester"}
    assert cooler_edges(bench) == set()
    lever_extra = bench_instances()
    lever_extra[LEVER].attrs = {EXTRA_ATTR: "tester"}
    assert cooler_edges(lever_extra) == {("covered_by", "cpu.01", FAN)}


def test_the_lever_can_no_longer_be_opened_first_and_the_cpu_plan_lifts_the_cooler():
    desk = two_part_desktop()
    edges = propose_edges(desk, TAX)
    state = state_at(desk, [], 1, TAX)
    assert ("open", LEVER) not in legal_actions(desk, edges, state, TAX)
    plan = plan_removal(desk, edges, state, "cpu.01", TAX)
    steps = [f"{v} {t}" for v, t in plan.actions]
    lever = steps.index(f"open {LEVER}")
    assert any(s.endswith(HEATSINK) for s in steps[:lever])
    assert any(s.endswith(FAN) for s in steps[:lever])
    assert steps[-1] == "remove cpu.01" and len(steps) > 2


def test_a_correct_log_still_validates():
    """The bench's good teardown takes the fan off before opening the lever."""
    from tda.core.graph import validate_sequence

    bench = bench_instances()
    assert validate_sequence(bench, propose_edges(bench, TAX), good_sequence(), TAX) == []


def test_constraints_reports_the_unresolved_cooler(tmp_path: Path):
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
        db.upsert_instance(InstanceRec(key="cpu_cooler.fan.02", desktop=DESKTOP,
                                       cls="cpu_cooler", attrs={"kind": "fan"}))
        db.replace_steps(DESKTOP, [], good_sequence())
    finally:
        db.close()
    report = tmp_path / "report.md"
    main(["--paths", str(tmp_path / "paths.yaml"), "constraints", "--desktops", str(DESKTOP),
          "--dry-run", "--report", str(report)])
    text = report.read_text(encoding="utf-8")
    assert "unresolved cooler" in text and "cpu_cooler.fan.02" in text


def test_the_docs_say_it():
    root = Path(__file__).resolve().parents[1]
    spec = (root / "docs" / "superpowers" / "specs"
            / "2026-09-17-teardown-annotator-design.md").read_text(encoding="utf-8")
    rules = spec.split("### 7.3 生成", 1)[1].split("### 7.4", 1)[0]
    for needed in ("**压在 CPU 上的散热器（v1.46，W3a）**", "`unresolved cooler: …`",
                   "`covered_by(heatsink ← fan)`", "**用户在复查里裁定的遮挡边（v1.46，W3a）**",
                   "`source = review:covers`", "`pixels:W1`", "**imported**"):
        assert needed in rules, needed
    cv = spec.split("### 8.1 CV 导出", 1)[1].split("### 8.2", 1)[0]
    [bullet] = [line for line in cv.splitlines()
                if line.startswith("- **遮挡裁定成边、支承表、划分（v1.46，W3a）**")]
    for needed in ("`python -m tda.cli pixel-edges", "`python -m tda.cli set-split",
                   '["rule.supported_by"]', "`mounted_on_rule`", "`user.not_blocks`",
                   "`tda_pre_pixel_edges_<时间>.sqlite`"):
        assert needed in bullet, needed
    entries = [line for line in spec.split("## 14. 修订记录", 1)[1].splitlines()
               if line.startswith("- ")]
    assert any(e.startswith("- **v1.46（2026-10-31）**：**压在 CPU 上的散热器、复查裁定的遮挡边")
               for e in entries[:3])
    handoff = (root / "docs" / "HANDOFF_2026-10-28.md").read_text(encoding="utf-8")
    section = handoff.split("## 5. 导出与 HuggingFace 准备", 1)[1].split("\n## ", 1)[0]
    assert sum("cli pixel-edges" in line for line in section.splitlines()) == 1
    assert sum("cli set-split" in line for line in section.splitlines()) == 1
