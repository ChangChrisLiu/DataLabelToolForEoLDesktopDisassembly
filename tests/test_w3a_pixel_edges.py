"""``cli pixel-edges``: Review's ``blocks`` rulings frozen as ``covered_by`` edges (task W3a, 2.2).

The user decides who blocks whom in Review, per instance pair (table
``cover_ruling``, written by task W3c -- created here in the fixture with the
exact DDL). A ``blocks`` pair the confirmed frames show covering becomes an
accepted ``covered_by`` row with ``source = review:covers``; ``not_blocks``
never does. The class-pair whitelist is a suggestion layer: it writes
``pixels:W1`` rows only with ``--allow-unruled``. Plus the export-time pieces:
the support table (2.3) and the ``ruling`` field of ``export-graph``.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import yaml
from graph_scenes import DESKTOP, bench_instances, good_sequence
from test_u46_export_graph import build

from tda.cli import EXIT_ERROR, EXIT_OK, main
from tda.core import pixel_edges as PE
from tda.core.db import Db
from tda.core.export.graph_covers import (
    NOT_BLOCKS_REASON,
    SUPPORTED_BY_REASON,
    cover_rulings,
    structural_reasons,
    supported_by_rule,
)
from tda.core.export.graph_export import ReadOnlyDb, build_desktop, summarize
from tda.core.graph import Edge, edges_from_db, graph_version, propose_edges
from tda.core.graph_derive import derive_edges
from tda.core.taxonomy import load_taxonomy

TAX = load_taxonomy()
VIEW = "scan"
PSU, HDD, SCREW = "psu.01", "storage_drive.hdd.01", "screw.cpu_cooler.01"
RULING_DDL = ("CREATE TABLE cover_ruling(desktop INTEGER, above TEXT, below TEXT, "
              "ruling TEXT CHECK (ruling IN ('blocks','not_blocks')), by TEXT, ts TEXT, "
              "view TEXT, step INTEGER, note TEXT, PRIMARY KEY (desktop, above, below))")


def add_rulings(path: Path, *rows: tuple[str, str, str], view: str = VIEW) -> None:
    conn = sqlite3.connect(str(path))
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='cover_ruling'").fetchone() is None:
            conn.execute(RULING_DDL)
        for above, below, ruling in rows:
            conn.execute("INSERT OR REPLACE INTO cover_ruling VALUES (?,?,?,?,?,?,?,?,?)",
                         (DESKTOP, above, below, ruling, "user", "2026-11-01T10:00:00+00:00",
                          view, 3, None))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def env(tmp_path: Path) -> dict:
    db_path = build(tmp_path / "tda.sqlite")
    cfg = {"cache_dir": str(tmp_path / "cache"), "db_path": str(db_path),
           "backup_dir": str(tmp_path / "backups"), "raw_logs_dir": str(tmp_path / "raw")}
    (tmp_path / "backups").mkdir()
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return {"paths": str(tmp_path / "paths.yaml"), "db": db_path, "tmp": tmp_path}


@pytest.fixture(autouse=True)
def no_app(monkeypatch):
    monkeypatch.setattr("tda.cli_pixel_edges.app_running", lambda: False)
    monkeypatch.setattr("tda.cli_split.app_running", lambda: False)


def run(env: dict, *argv: str) -> int:
    return main(["--paths", env["paths"], "pixel-edges", "--desktops", str(DESKTOP),
                 "--out", str(env["tmp"] / "evidence.json"), *argv])


def evidence(env: dict) -> dict:
    return json.loads((env["tmp"] / "evidence.json").read_text(encoding="utf-8"))


def table(path: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def owned_rows(path: Path) -> list[tuple]:
    return table(path, "SELECT * FROM relation WHERE source LIKE 'pixels:%' "
                       "OR source LIKE 'review:%' ORDER BY id")


def ops(path: Path) -> list[tuple]:
    return table(path, "SELECT kind, desktop, view FROM op_log WHERE kind LIKE 'pixel%' "
                       "ORDER BY id")


def whitelist_file(tmp: Path, body: str) -> Path:
    path = tmp / "wl.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# the whitelist loader
# --------------------------------------------------------------------------- #
def test_the_shipped_whitelist_loads():
    wl = PE.load_whitelist(Path(__file__).resolve().parents[1] / PE.DEFAULT_WHITELIST, TAX)
    assert wl.version == "W1" and wl.source == "pixels:W1"
    assert (wl.min_px, wl.min_frac) == (1000, 0.03)
    labels = [p.label() for p in wl.pairs]
    assert labels[0] == "optical_drive -> motherboard" and len(labels) == 18
    # A1 is structural by the support table: never a candidate, so not the whitelist's
    assert "expansion_card.gpu -> motherboard" not in labels
    assert "drive_cage -> motherboard" not in labels          # 存疑: commented out
    assert {p.min_frac for p in wl.pairs if p.below == "cpu"} == {0.25}


@pytest.mark.parametrize("above, below, said", [
    ("cover", "motherboard", "decided elsewhere"),
    ("expansion_card.wlan", "motherboard", "rides on the board"),
    ("expansion_card.other", "cpu", "rides on the board"),
    ("expansion_card", "motherboard", "rides on the board"),
    ("optical_drive", "screw.motherboard", "own rules"),
    ("optical_drive", "screw", "own rules"),
    ("drive_cage", "connector.front_panel", "own rules"),
    ("drive_cage", "cable_clip", "own rules"),
    ("drive_cage", "loose_cable", "own rules"),
    ("spaceship", "motherboard", "no such class"),
    ("cpu_cooler.turbo", "optical_drive", "no kind"),
    ("expansion_card.gpu", "motherboard", "support table"),
    ("cpu_cooler.fan", "motherboard", "support table"),
    ("ram_module", "motherboard", "support table"),
    ("cpu", "motherboard", "support table"),
])
def test_the_loader_refuses(tmp_path, above, below, said):
    path = whitelist_file(tmp_path, f"version: W1\npairs:\n  - {{above: {above}, below: {below}}}\n")
    with pytest.raises(PE.WhitelistError) as excinfo:
        PE.load_whitelist(path, TAX)
    assert said in str(excinfo.value)


def test_every_refusal_is_listed_at_once_and_a_version_is_required(tmp_path):
    body = ("version: W2\npairs:\n  - {above: cover, below: motherboard}\n"
            "  - {above: drive_cage, below: screw}\n  - {above: psu, below: cover}\n")
    with pytest.raises(PE.WhitelistError) as excinfo:
        PE.load_whitelist(whitelist_file(tmp_path, body), TAX)
    assert "pair 1" in str(excinfo.value) and "pair 2" in str(excinfo.value)
    assert "pair 3" not in str(excinfo.value)
    with pytest.raises(PE.WhitelistError):
        PE.load_whitelist(whitelist_file(tmp_path, "pairs: []\n"), TAX)


def test_threshold_math_with_and_without_the_amodal_area():
    pair = PE.WhitelistPair("optical_drive", "motherboard", 1000, 0.03)
    assert pair.need(None) == 1000 and pair.need(0) == 1000      # unknown: min_px alone
    assert pair.need(100_000) == 3000                            # not 3001 (float noise)
    assert pair.need(10_000) == 1000                             # the floor wins
    small = PE.WhitelistPair("cover", "cpu", 1000, 0.25)
    assert small.need(6524) == 1631
    assert PE.WhitelistPair("optical_drive", "cpu_cooler", 1, 0.0).matches(
        "optical_drive", "cpu_cooler.heatsink")
    assert not PE.WhitelistPair("x", "cpu_cooler.fan", 1, 0.0).matches("x", "cpu_cooler.heatsink")


# --------------------------------------------------------------------------- #
# judging one desktop (pure)
# --------------------------------------------------------------------------- #
def cover(above, below, px=5000, amodal=20000, frames=(2, 3)):
    from tda.core.export.graph_covers import class_key

    bench = bench_instances()
    return {"above": above, "below": below, "above_class": class_key(bench.get(above)),
            "below_class": class_key(bench.get(below)), "max_px": px,
            "below_amodal_px": amodal, "frames": list(frames), "n_frames": len(frames),
            "first": frames[0]}


def ruled(*pairs):
    return {(a, b): {"ruling": r, "by": "user", "ts": "2026-11-01T10:00:00", "view": VIEW,
                     "step": 2, "note": None} for a, b, r in pairs}


def judge(covers, rulings=None, whitelist=None):
    bench = bench_instances()
    return PE.plan_desktop(DESKTOP, covers, 3, bench, propose_edges(bench, TAX),
                           good_sequence(), [], TAX, rulings or {}, whitelist)


def test_a_blocks_ruling_with_pixels_is_an_edge_and_not_blocks_never_is():
    plan = judge([cover(PSU, HDD), cover(SCREW, HDD)],
                 ruled((PSU, HDD, "blocks"), (SCREW, HDD, "not_blocks")))
    [item] = plan.review
    assert (item["above"], item["below"], item["source"]) == (PSU, HDD, "review:covers")
    assert item["frac"] == 0.25 and item["first"] == 2
    assert "ruled blocks in Review by user (2026-11-01, scan step 2)" in item["reason"]
    assert item["reason"].startswith(f"{PSU} covers {HDD} on 2 confirmed scan frames "
                                     f"(max 5000 px, 25.0% of its amodal area)")
    [no] = plan.rejected_by_user
    assert (no["above"], no["below"]) == (SCREW, HDD)


def test_a_ruled_pair_without_pixels_is_listed_not_written():
    plan = judge([], ruled((PSU, HDD, "blocks"), ("cover.01", PSU, "not_blocks")))
    assert plan.review == []
    assert [s["why"] for s in plan.skipped] == ["ruled_no_pixel_evidence"]
    assert [r["below"] for r in plan.rejected_by_user] == [PSU]


def test_already_an_edge_a_reverse_edge_and_a_broken_log_are_not_written():
    plan = judge([cover("cpu_cooler.fan.01", "cpu.01"),
                  cover("motherboard.01", "screw.motherboard.01"),
                  cover(PSU, "motherboard.01")],
                 ruled(("cpu_cooler.fan.01", "cpu.01", "blocks"),
                       ("motherboard.01", "screw.motherboard.01", "blocks"),
                       (PSU, "motherboard.01", "blocks")))
    assert plan.review == []
    why = {(s["above"], s["below"]): s["why"] for s in plan.skipped}
    assert why == {("cpu_cooler.fan.01", "cpu.01"): "already_an_edge",
                   ("motherboard.01", "screw.motherboard.01"): "reverse_edge",
                   (PSU, "motherboard.01"): "would_violate"}
    [broken] = [s for s in plan.skipped if s["why"] == "would_violate"]
    assert broken["lines"] and "motherboard.01" in broken["lines"][0]


def test_a_rejected_rule_row_of_the_same_triple_is_never_overwritten():
    bench = bench_instances()
    stored = [*propose_edges(bench, TAX),
              Edge("covered_by", HDD, PSU, source="override", status="rejected")]
    plan = PE.plan_desktop(DESKTOP, [cover(PSU, HDD)], 3, bench, stored, good_sequence(), [],
                           TAX, ruled((PSU, HDD, "blocks")), None)
    [skip] = plan.skipped
    assert skip["why"] == "already_an_edge" and skip["row_status"] == "rejected"


def wl(min_px=1000, min_frac=0.03, above="psu", below="storage_drive"):
    return PE.Whitelist("W1", "wl.yaml", min_px, min_frac,
                        (PE.WhitelistPair(above, below, min_px, min_frac, "", 1),))


def test_whitelist_thresholds_and_the_flag_for_an_unknown_amodal_area():
    plan = judge([cover(PSU, HDD, px=500, amodal=20000)], whitelist=wl())
    assert plan.whitelist == [] and plan.skipped[0]["why"] == "below_threshold"
    assert plan.skipped[0]["need_px"] == 1000
    plan = judge([cover(PSU, HDD, px=1200, amodal=None)], whitelist=wl())
    [item] = plan.whitelist
    assert item["amodal_unavailable"] is True and item["frac"] is None
    assert item["source"] == "pixels:W1" and "amodal area unknown" in item["reason"]
    plan = judge([cover(PSU, HDD, px=2000, amodal=100_000)], whitelist=wl())
    assert plan.whitelist == [] and plan.skipped[0]["need_px"] == 3000


def test_without_allow_unruled_the_whitelist_only_suggests():
    plan = judge([cover(PSU, HDD)], whitelist=wl())
    PE.split_suggestions(plan, allow_unruled=False)
    assert plan.whitelist == [] and len(plan.suggested) == 1
    assert plan.to_write(False) == [] and plan.to_write(True) == []


def test_structure_refuses_a_whitelist_pair_and_a_ruling_beats_it():
    # the RAM latch rides out on the board: structural, so the whitelist refuses it ...
    latch = cover("ram_latch.01", "motherboard.01")
    plan = judge([latch], whitelist=wl(min_px=1, min_frac=0, above="ram_latch",
                                       below="motherboard"))
    assert plan.skipped[0]["why"] == "structural"
    # ... the support table is a structural reason too (the loader refuses such a
    # pair; a hand-built whitelist that names one is still not written) ...
    gpu = bench_instances()
    gpu["expansion_card.gpu.01"] = gpu["ram_module.01"].__class__(
        key="expansion_card.gpu.01", desktop=DESKTOP, cls="expansion_card",
        attrs={"kind": "gpu"})
    assert structural_reasons(gpu, TAX, "expansion_card.gpu.01", "motherboard.01") == [
        SUPPORTED_BY_REASON]
    card = {**cover("expansion_card.gpu.01", "motherboard.01"),
            "above_class": "expansion_card.gpu"}
    plan = PE.plan_desktop(DESKTOP, [card], 3, gpu,
                           propose_edges(gpu, TAX), [], [], TAX, {},
                           wl(1, 0.0, "expansion_card.gpu", "motherboard"))
    assert plan.whitelist == [] and plan.skipped[0]["why"] == "structural"
    assert plan.skipped[0]["structural"] == [SUPPORTED_BY_REASON]
    # ... and only the user's explicit ruling wins over it
    plan = PE.plan_desktop(DESKTOP, [card], 3, gpu, propose_edges(gpu, TAX), [], [], TAX,
                           ruled(("expansion_card.gpu.01", "motherboard.01", "blocks")), None)
    [item] = plan.review
    assert item["source"] == "review:covers" and item["structural"] == [SUPPORTED_BY_REASON]


# --------------------------------------------------------------------------- #
# the command
# --------------------------------------------------------------------------- #
def test_without_a_ruling_nothing_is_written(env):
    before = table(env["db"], "SELECT * FROM relation ORDER BY id")
    assert run(env, "--apply") == EXIT_OK
    assert table(env["db"], "SELECT * FROM relation ORDER BY id") == before
    doc = evidence(env)
    assert doc["rulings_table"] is True and doc["counts"]["edges"] == 0
    assert ops(env["db"]) == [] and not list((env["tmp"] / "backups").iterdir())


def test_dry_run_writes_nothing_but_the_evidence(env, capsys):
    add_rulings(env["db"], (PSU, HDD, "blocks"), (SCREW, HDD, "not_blocks"))
    before = table(env["db"], "SELECT * FROM relation ORDER BY id")
    assert run(env) == EXIT_OK
    out = capsys.readouterr().out
    assert f"would write covered_by({HDD}, {PSU})" in out
    assert table(env["db"], "SELECT * FROM relation ORDER BY id") == before
    assert ops(env["db"]) == [] and not list((env["tmp"] / "backups").iterdir())
    doc = evidence(env)
    assert doc["applied"] is False and doc["counts"]["edges"] == 1
    assert [(r["above"], r["below"]) for r in doc["rejected_by_user"]] == [(SCREW, HDD)]
    [edge] = doc["edges"]
    assert edge["px"] == 25 and edge["frames"] == [1, 3] and edge["amodal_px"] == 99


def test_apply_writes_the_rows_one_op_and_a_backup(env):
    add_rulings(env["db"], (PSU, HDD, "blocks"), (SCREW, HDD, "not_blocks"))
    version = graph_version_of(env["db"])
    assert run(env, "--apply") == EXIT_OK
    [row] = [dict(r) for r in rows_of(env["db"]) if r["source"].startswith("review:")]
    assert (row["type"], row["target"], row["blocker"]) == ("covered_by", HDD, PSU)
    assert (row["necessity"], row["mode"], row["status"]) == ("required", None, "accepted")
    assert row["evidence_step"] == 1
    assert row["reason"].startswith(f"{PSU} covers {HDD} on 2 confirmed scan frames (max 25 px")
    assert ops(env["db"]) == [("pixel_edges", DESKTOP, "-")]
    [backup] = list((env["tmp"] / "backups").iterdir())
    assert backup.name.startswith("tda_pre_pixel_edges_")
    assert graph_version_of(env["db"]) != version
    # a second run finds the edge already there and writes nothing more
    assert run(env, "--apply") == EXIT_OK
    assert len(owned_rows(env["db"])) == 1 and len(ops(env["db"])) == 1
    assert evidence(env)["skipped"][0]["why"] == "already_an_edge"


def rows_of(path: Path) -> list:
    db = Db(str(path))
    try:
        return db.relations(DESKTOP)
    finally:
        db.close()


def graph_version_of(path: Path):
    db = Db(str(path))
    try:
        return graph_version(db, DESKTOP)
    finally:
        db.close()


def test_constraints_keeps_the_rows_and_export_graph_explains_them(env):
    add_rulings(env["db"], (PSU, HDD, "blocks"), (SCREW, HDD, "not_blocks"))
    assert run(env, "--apply") == EXIT_OK
    frozen, version = owned_rows(env["db"]), graph_version_of(env["db"])
    main(["--paths", env["paths"], "constraints", "--desktops", str(DESKTOP)])
    assert owned_rows(env["db"]) == frozen and graph_version_of(env["db"]) == version
    db = Db(str(env["db"]))
    try:
        derivation = derive_edges(db.instances(DESKTOP), edges_from_db(db, DESKTOP), TAX)
    finally:
        db.close()
    assert [e.source for e in derivation.imported] == ["review:covers"]
    ro = ReadOnlyDb(str(env["db"]))
    try:
        doc = build_desktop(ro, TAX, DESKTOP, VIEW)
    finally:
        ro.close()
    covers = {(c["above"], c["below"]): c for c in doc["covers"]}
    psu = covers[(PSU, HDD)]
    assert psu["explained_by"] == [f"covered_by({HDD}, {PSU})"] and psu["candidate"] is False
    assert psu["ruling"] == "blocks"
    screw = covers[(SCREW, HDD)]
    assert screw["ruling"] == "not_blocks" and screw["candidate"] is False
    assert NOT_BLOCKS_REASON in screw["structural"]
    counts = doc["meta"]["counts"]
    assert counts["rulings"] == {"blocks": 1, "not_blocks": 1}
    assert counts["candidates"] == 0 and counts["candidates_unruled"] == 0
    total = summarize([doc], VIEW)["totals"]
    assert total["rulings"] == {"blocks": 1, "not_blocks": 1}


def test_an_unruled_candidate_and_a_ruled_one_are_counted_apart(env):
    add_rulings(env["db"], (PSU, HDD, "blocks"))
    ro = ReadOnlyDb(str(env["db"]))
    try:
        doc = build_desktop(ro, TAX, DESKTOP, VIEW)
        assert cover_rulings(ro, DESKTOP)[(PSU, HDD)]["by"] == "user"
    finally:
        ro.close()
    counts = doc["meta"]["counts"]
    assert counts["candidates"] == 2
    assert (counts["candidates_ruled"], counts["candidates_unruled"]) == (1, 1)
    row = summarize([doc], VIEW)["per_desktop"][0]
    assert (row["candidates_ruled"], row["candidates_unruled"]) == (1, 1)


def test_undo_deletes_the_rows_with_an_op(env):
    add_rulings(env["db"], (PSU, HDD, "blocks"))
    version = graph_version_of(env["db"])
    assert run(env, "--apply") == EXIT_OK
    assert run(env, "--undo", "all") == EXIT_OK
    assert owned_rows(env["db"]) == []
    assert [k for k, _, _ in ops(env["db"])] == ["pixel_edges", "pixel_edges_undo"]
    assert graph_version_of(env["db"]) == version
    names = sorted(p.name for p in (env["tmp"] / "backups").iterdir())
    assert any(n.startswith("tda_pre_pixel_edges_undo_") for n in names)
    # the rulings stayed in their table, so the review rows come back with --apply
    assert run(env, "--apply") == EXIT_OK and len(owned_rows(env["db"])) == 1


def test_undo_needs_a_selector_and_deletes_only_that_kind(env, capsys):
    add_rulings(env["db"], (PSU, HDD, "blocks"))
    assert run(env, "--apply") == EXIT_OK
    db = Db(str(env["db"]))
    try:          # a whitelist row beside the review one
        db.add_relation(DESKTOP, "covered_by", HDD, SCREW, source="pixels:W1",
                        status="accepted")
    finally:
        db.close()
    capsys.readouterr()
    assert run(env, "--undo") == EXIT_ERROR
    out = capsys.readouterr().out
    assert "pixels" in out and "review" in out and "all" in out
    assert run(env, "--undo", "everything") == EXIT_ERROR
    assert len(owned_rows(env["db"])) == 2
    assert run(env, "--undo", "pixels") == EXIT_OK
    assert [r["source"] for r in rows_of(env["db"]) if r["target"] == HDD
            and r["blocker"] in (PSU, SCREW)] == ["review:covers"]
    assert run(env, "--undo", "review") == EXIT_OK
    assert owned_rows(env["db"]) == []


def test_the_whitelist_writes_only_when_unruled_pairs_are_allowed(env):
    path = whitelist_file(env["tmp"], "version: W1\ndefaults: {min_px: 10, min_frac: 0.1}\n"
                                      "pairs:\n  - {above: psu, below: storage_drive}\n")
    assert run(env, "--whitelist", str(path), "--apply") == EXIT_OK
    assert owned_rows(env["db"]) == []
    assert evidence(env)["counts"]["suggested"] == 1
    assert run(env, "--whitelist", str(path), "--allow-unruled", "--apply") == EXIT_OK
    [row] = rows_of(env["db"])[-1:]
    assert row["source"] == "pixels:W1" and row["target"] == HDD
    doc = evidence(env)
    assert doc["whitelist"]["version"] == "W1" and len(doc["whitelist_edges"]) == 1
    assert doc["thresholds"]["min_frac"] == 0.1


def test_allow_unruled_needs_a_whitelist_and_apply_with_undo_is_refused(env):
    assert run(env, "--allow-unruled") == EXIT_ERROR
    assert run(env, "--apply", "--undo", "all") == EXIT_ERROR


def test_a_running_app_refuses_the_write(env, monkeypatch, capsys):
    add_rulings(env["db"], (PSU, HDD, "blocks"))
    monkeypatch.setattr("tda.cli_pixel_edges.app_running", lambda: True)
    assert run(env, "--apply") == EXIT_ERROR
    assert "annotator app is running" in capsys.readouterr().out
    assert owned_rows(env["db"]) == []


def test_the_evidence_can_come_from_an_export_graph_directory(env):
    add_rulings(env["db"], (PSU, HDD, "blocks"))
    from tda.core.export.graph_export import export_graph

    ro = ReadOnlyDb(str(env["db"]))
    try:
        export_graph(ro, TAX, [DESKTOP], str(env["tmp"] / "graph"), VIEW)
    finally:
        ro.close()
    assert run(env, "--graph-dir", str(env["tmp"] / "graph")) == EXIT_OK
    doc = evidence(env)
    assert doc["evidence_source"].startswith("graph-dir")
    assert [e["px"] for e in doc["edges"]] == [25]


def stamp(path: Path, version: int) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(version),))
        conn.commit()
    finally:
        conn.close()


def test_a_v6_copy_without_the_ruling_table_is_read_with_no_rulings(env):
    """A read-only tool must not demand a migration: a copy taken before W3c's v7."""
    from tda.core.export.graph_export import READ_ONLY_MIN_SCHEMA

    assert READ_ONLY_MIN_SCHEMA == 6
    stamp(env["db"], 6)
    conn = sqlite3.connect(str(env["db"]))
    try:                                  # what a copy taken before W3c looks like
        conn.execute("DROP TABLE IF EXISTS cover_ruling")
        conn.commit()
    finally:
        conn.close()
    ro = ReadOnlyDb(str(env["db"]))
    try:
        assert ro.schema_version == 6 and cover_rulings(ro, DESKTOP) == {}
        doc = build_desktop(ro, TAX, DESKTOP, VIEW)
    finally:
        ro.close()
    assert {c["ruling"] for c in doc["covers"]} == {None}
    assert doc["meta"]["counts"]["rulings"] == {"blocks": 0, "not_blocks": 0}


def test_a_v7_copy_with_the_ruling_table_is_read(env):
    """W3c's v7 adds `cover_ruling`, read through `Db.cover_rulings`."""
    from tda.core.db import SCHEMA_VERSION

    assert SCHEMA_VERSION >= 7
    add_rulings(env["db"], (PSU, HDD, "blocks"))
    stamp(env["db"], 7)
    ro = ReadOnlyDb(str(env["db"]))
    try:
        assert ro.schema_version == 7
        assert cover_rulings(ro, DESKTOP)[(PSU, HDD)]["ruling"] == "blocks"
        assert cover_rulings(ro, DESKTOP)[(PSU, HDD)]["by"] == "user"
        doc = build_desktop(ro, TAX, DESKTOP, VIEW)
    finally:
        ro.close()
    assert {(c["above"], c["below"]): c["ruling"] for c in doc["covers"]}[(PSU, HDD)] == "blocks"


def test_too_old_and_too_new_copies_are_refused(env):
    from tda.core.db import SCHEMA_VERSION

    for version in (5, SCHEMA_VERSION + 1):
        stamp(env["db"], version)
        with pytest.raises(RuntimeError, match=f"schema version {version}"):
            ReadOnlyDb(str(env["db"]))


def test_rulings_come_through_the_db_reader_when_it_has_one(env):
    """W3c's `Db.cover_rulings(desktop)`: used when present, whatever row shape it returns."""
    add_rulings(env["db"], (PSU, HDD, "blocks"))
    ro = ReadOnlyDb(str(env["db"]))
    try:
        class Reader:
            conn = ro.conn

            def cover_rulings(self, desktop):
                assert desktop == DESKTOP
                return [{"above": SCREW, "below": HDD, "ruling": "not_blocks", "by": "w3c",
                         "ts": "t", "view": VIEW, "step": 2, "note": "n"}]

        got = cover_rulings(Reader(), DESKTOP)
    finally:
        ro.close()
    assert got == {(SCREW, HDD): {"ruling": "not_blocks", "by": "w3c", "ts": "t",
                                  "view": VIEW, "step": 2, "note": "n"}}


def test_the_support_table_and_mounted_on_rule(env):
    bench = bench_instances()
    assert supported_by_rule(bench, "cpu.01") == "motherboard.01"
    assert supported_by_rule(bench, "ram_module.01") == "motherboard.01"
    assert supported_by_rule(bench, "psu.01") is None
    assert SUPPORTED_BY_REASON in structural_reasons(bench, TAX, "cpu.01", "motherboard.01")
    assert SUPPORTED_BY_REASON not in structural_reasons(bench, TAX, "motherboard.01", "cpu.01")
    ro = ReadOnlyDb(str(env["db"]))
    try:
        doc = build_desktop(ro, TAX, DESKTOP, VIEW)
    finally:
        ro.close()
    by_key = {i["key"]: i for i in doc["instances"]}
    assert by_key["cpu.01"]["mounted_on_rule"] == "motherboard.01"
    assert by_key["psu.01"]["mounted_on_rule"] is None
    assert by_key["cpu.01"]["mounted_on"] == "motherboard.01"        # as annotated, unchanged


# --------------------------------------------------------------------------- #
# review round 1: S1 stale rows, M1 read-only dry runs, M2 missing graph files,
# M3 one support table, M4 view-independent rulings
# --------------------------------------------------------------------------- #
def set_ruling(path: Path, above: str, below: str, ruling) -> None:
    conn = sqlite3.connect(str(path))
    try:
        if ruling is None:
            conn.execute("DELETE FROM cover_ruling WHERE desktop=? AND above=? AND below=?",
                         (DESKTOP, above, below))
        else:
            conn.execute("UPDATE cover_ruling SET ruling=? WHERE desktop=? AND above=? "
                         "AND below=?", (ruling, DESKTOP, above, below))
        conn.commit()
    finally:
        conn.close()


@pytest.mark.parametrize("now, why", [("not_blocks", "ruled_not_blocks"),
                                      (None, "ruling_cleared")])
def test_a_review_edge_the_user_took_back_is_listed_then_deleted(env, capsys, now, why):
    add_rulings(env["db"], (PSU, HDD, "blocks"))
    assert run(env, "--apply") == EXIT_OK
    [row] = owned_rows(env["db"])
    set_ruling(env["db"], PSU, HDD, now)
    capsys.readouterr()
    # the dry run lists it and fails, so nobody ships a graph that contradicts the user
    assert run(env) == EXIT_ERROR
    assert "STALE" in capsys.readouterr().out
    [stale] = evidence(env)["stale_review_edges"]
    assert (stale["above"], stale["below"], stale["why"]) == (PSU, HDD, why)
    assert stale["row"]["source"] == "review:covers"
    assert owned_rows(env["db"]) == [row]
    # --apply deletes it, with one op whose inverse restores the row
    assert run(env, "--apply") == EXIT_OK
    assert "deleted 1 rows the rulings no longer back" in capsys.readouterr().out
    assert owned_rows(env["db"]) == []
    [(payload, inverse)] = table(env["db"], "SELECT payload_json, inverse_json FROM op_log "
                                            "WHERE kind='pixel_edges' ORDER BY id")[1:]
    assert json.loads(payload)["deleted_stale"][0]["blocker"] == PSU
    assert json.loads(inverse)["restore"][0]["source"] == "review:covers"
    assert run(env) == EXIT_OK                                   # nothing stale any more


def test_a_whitelist_row_the_user_rules_not_blocks_is_stale_too(env):
    db = Db(str(env["db"]))
    try:
        db.add_relation(DESKTOP, "covered_by", HDD, PSU, source="pixels:W1", status="accepted")
    finally:
        db.close()
    assert run(env) == EXIT_OK                       # unruled: a whitelist row is not stale
    add_rulings(env["db"], (PSU, HDD, "not_blocks"))
    assert run(env) == EXIT_ERROR
    assert evidence(env)["stale_review_edges"][0]["row"]["source"] == "pixels:W1"
    assert run(env, "--apply") == EXIT_OK and owned_rows(env["db"]) == []


def sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_v6(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("DROP TABLE IF EXISTS cover_ruling")
        conn.execute("UPDATE meta SET value='6' WHERE key='schema_version'")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def test_dry_runs_leave_a_v6_copy_byte_for_byte(env):
    """M1: the writable Db would migrate the copy to v7 during a run that writes nothing."""
    as_v6(env["db"])
    before = sha(env["db"])
    assert run(env) == EXIT_OK
    assert main(["--paths", env["paths"], "set-split", "--test", str(DESKTOP)]) == EXIT_OK
    assert sha(env["db"]) == before
    assert table(env["db"], "SELECT value FROM meta WHERE key='schema_version'") == [("6",)]
    assert evidence(env)["rulings_table"] is False


def test_a_missing_graph_file_is_said_and_fails(env, capsys):
    """M2: not "ruled but no pixel evidence" -- the evidence was never read."""
    add_rulings(env["db"], (PSU, HDD, "blocks"))
    (env["tmp"] / "empty").mkdir()
    assert run(env, "--graph-dir", str(env["tmp"] / "empty")) == EXIT_ERROR
    assert "skipped: no graph file" in capsys.readouterr().out
    doc = evidence(env)
    assert doc["no_graph_file"] == [f"D{DESKTOP:02d}"]
    assert not [s for s in doc["skipped"] if s["why"] == "ruled_no_pixel_evidence"]
    assert run(env, "--graph-dir", str(env["tmp"] / "empty"), "--apply") == EXIT_ERROR
    assert owned_rows(env["db"]) == []


def test_review_and_the_export_share_one_support_table():
    """M3: the tuple lives in graph_covers; Review's question list imports it."""
    from tda.core.export import graph_covers
    from tda.ui import cover_rulings as ui

    assert ui.SUPPORT is graph_covers.SUPPORT
    bench = bench_instances()
    bench["motherboard.02"] = bench["motherboard.01"].__class__(
        key="motherboard.02", desktop=DESKTOP, cls="motherboard")      # two boards
    for above, below in [("cpu.01", "motherboard.01"), ("cpu.01", "motherboard.02"),
                         ("ram_module.01", "motherboard.01"), ("psu.01", "motherboard.01"),
                         ("motherboard.01", "cpu.01")]:
        mine = SUPPORTED_BY_REASON in structural_reasons(bench, TAX, above, below)
        theirs = "supported_by" in ui.structural_reasons(bench, above, below)
        assert mine == theirs, (above, below)


def test_a_ruling_made_in_another_view_still_counts(env):
    """M4: a ruling is a physical judgement; the pixels come from the export view."""
    add_rulings(env["db"], (PSU, HDD, "blocks"), view="oak1")
    assert run(env, "--apply") == EXIT_OK
    [row] = owned_rows_dicts(env["db"])
    assert (row["target"], row["blocker"], row["source"]) == (HDD, PSU, "review:covers")
    assert "on 2 confirmed scan frames" in row["reason"] and "oak1 step 3" in row["reason"]


def owned_rows_dicts(path: Path) -> list[dict]:
    return [r for r in rows_of(path) if str(r["source"]).startswith(("pixels:", "review:"))]
