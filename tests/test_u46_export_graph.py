"""``python -m tda.cli export-graph``: one precedence-graph JSON per desktop (task U46).

The bench desktop of :mod:`graph_scenes` with its rule edges stored, a log
with one breach (the board leaves with a SATA plug still in), one failed
attempt nothing in the graph explains, a ``reorient`` step and an action on
a part the sheet did not name (``?``), and four scanner frames: three
confirmed, whose stored rows say the CPU fan takes pixels of the CPU (an edge
explains it; a pairwise layering exception ordered it), the PSU takes pixels
of the drive (nothing does; on frame 2 the two visible masks overlap, so the
PSU takes nothing there) and a captive cooler screw takes pixels of the drive
(its fan does not cover the drive, so it stays a candidate), plus a RAM stick
lying on the bench (never read); and one frame not confirmed, which must not
count.

Round 2 (review S1-S4, N1-N6): reorient / unknown targets are not judged,
violations say whether the timeline agrees, bookkeeping attrs are dropped,
``rides_with_parent`` needs the parent's own pair, one snapshot, exit code.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from graph_scenes import DESKTOP, bench_instances

from tda.cli import EXIT_OK, build_parser, main
from tda.core import masks
from tda.core.db import Db
from tda.core.export.coco import load_ctx
from tda.core.export.graph_covers import explaining_edges, structural_reasons
from tda.core.export.graph_export import (
    FAILED_WITHOUT_UNMET,
    README_NAME,
    SUCCESS_VIOLATES,
    SUMMARY_NAME,
    ReadOnlyDb,
    build_desktop,
    desktop_file_name,
    export_graph,
    parse_violation,
    summarize,
)
from tda.core.graph import (
    constraint_edges,
    edges_from_db,
    edges_to_db,
    legal_actions,
    propose_edges,
    validate_sequence,
)
from tda.core.graph_rules import active_edges
from tda.core.model import (
    ActionRec,
    FrameKey,
    FrameOverride,
    PairOverride,
    ShapeKeyframe,
    ShapePart,
    StateEvent,
    StepRec,
)
from tda.core.taxonomy import load_taxonomy

VIEW = "scan"
HW = (20, 40)
CONFIRMED = (1, 2, 3)
UNCONFIRMED = 4
PATCH_STEP = 3
OVERLAP_STEP = 2
LAST_STEP = 10
SCREW = "screw.cpu_cooler.01"
BENCH_PART = "ram_module.01"

#: ``x0, y0, x1, y1`` of each part's whole shape, painted bottom first.
SHAPES = {
    "chassis": (0, 0, 40, 20),
    "cpu.01": (2, 2, 9, 9),
    "storage_drive.hdd.01": (20, 2, 31, 11),
    "cpu_cooler.fan.01": (5, 1, 13, 10),
    "psu.01": (26, 6, 36, 16),
    SCREW: (22, 3, 24, 5),
}
ORDER = list(SHAPES)


def rect(box) -> np.ndarray:
    x0, y0, x1, y1 = box
    out = np.zeros(HW, dtype=bool)
    out[y0:y1, x0:x1] = True
    return out


def act(step, idx, target, verb, result="success", why=None):
    return ActionRec(desktop=DESKTOP, step=step, idx=idx, target=target, verb=verb,
                     result=result, failure_reason=why)


def log():
    steps = [StepRec(DESKTOP, 1, "initial", "Initial conditions")]
    actions = [
        act(2, 0, "cover.01", "open"),
        act(3, 0, "connector.atx_24pin.01", "disconnect"),
        act(4, 0, "psu_latch.01", "open", "failed", "stuck"),
        act(5, 0, "screw.motherboard.01", "unscrew"),
        act(6, 0, "screw.motherboard.02", "unscrew"),
        act(7, 0, "screw.motherboard.03", "unscrew"),
        act(8, 0, "motherboard.01", "remove"),
        act(9, 0, "chassis", "reorient"),
        act(10, 0, "?", "remove"),
    ]
    names = {2: "open the screw cover", 3: "24-pin", 4: "PSU latch", 5: "MB screw 1",
             6: "MB screw 2", 7: "MB screw 3", 8: "motherboard"}
    steps += [StepRec(DESKTOP, s, "normal", n) for s, n in names.items()]
    steps += [StepRec(DESKTOP, 9, "reorient", "Turn the case"),
              StepRec(DESKTOP, 10, "normal", "Remove the obstacle")]
    return steps, actions


def build(path: Path, hand_unplug: bool = False) -> Path:
    """The scene; ``hand_unplug`` adds a hand-written event at step 7 that unplugs the
    SATA plug the log never unplugs -- the replay does not see it, the timeline does."""
    tax = load_taxonomy()
    db = Db(str(path))
    try:
        db.upsert_desktop(DESKTOP, {"brand": "Bench", "model_family": "Bench 1"})
        instances = bench_instances()
        cover = instances["cover.01"]
        cover.attrs = {**cover.attrs, "note": "checked by tester",
                       "added_at": "2026-10-30T12:00:00+00:00"}
        for rec in instances.values():
            db.upsert_instance(rec)
        steps, actions = log()
        db.replace_steps(DESKTOP, steps, actions)
        if hand_unplug:
            db.replace_events(DESKTOP, [StateEvent(
                DESKTOP, 7, "connector.sata_data.02", "state", "plugged", "unplugged",
                evidence_view=VIEW, auto=False)])
        edges_to_db(db, DESKTOP, propose_edges(instances, tax))
        db.set_pose_segment(DESKTOP, VIEW, 1, 1, LAST_STEP, 1, None, None)
        db.set_pair_override(PairOverride(DESKTOP, VIEW, 1, "cpu_cooler.fan.01", "cpu.01"))
        for key, box in SHAPES.items():
            db.add_keyframe(ShapeKeyframe(
                id=None, instance=key, desktop=DESKTOP, view=VIEW, pose_segment=1,
                anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
                parts=[ShapePart("main", masks.encode_rle(rect(box)))]))
        for step in (*CONFIRMED, UNCONFIRMED):
            key = FrameKey(DESKTOP, step, VIEW)
            db.upsert_frame(key, f"scan/{step}.png", {"hw": list(HW)}, None)
            status = "verified" if step in CONFIRMED else "auto"
            for i, inst in enumerate(ORDER):
                visible = rect(SHAPES[inst])
                for upper in ORDER[i + 1:]:
                    if not (step == OVERLAP_STEP and inst == "storage_drive.hdd.01"
                            and upper == "psu.01"):
                        visible &= ~rect(SHAPES[upper])     # frame 2: both claim them
                if step == UNCONFIRMED and inst == "cpu.01":
                    visible = rect(SHAPES[inst])      # would put the CPU over the fan
                db.put_compiled(key, inst, masks.encode_rle(visible), 0.0, "visible",
                                "in_chassis", status, "h",
                                verified_by="tester" if status == "verified" else None)
            # a RAM stick lying on the bench across the chassis: never covering evidence
            db.put_compiled(key, BENCH_PART, masks.encode_rle(rect((14, 12, 20, 18))), 0.0,
                            "visible", "on_bench", status, "h",
                            verified_by="tester" if status == "verified" else None)
        db.set_frame_override(FrameOverride(
            FrameKey(DESKTOP, PATCH_STEP, VIEW), "psu.01",
            visible_rle=masks.encode_rle(rect(SHAPES["psu.01"]))))
    finally:
        db.close()
    return path


@pytest.fixture(scope="module")
def scene(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("u46") / "tda.sqlite")


@pytest.fixture(scope="module")
def doc(scene) -> dict:
    db = ReadOnlyDb(str(scene))
    try:
        return build_desktop(db, load_taxonomy(), DESKTOP, VIEW)
    finally:
        db.close()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# the fields
# --------------------------------------------------------------------------- #
def test_meta_says_which_graph_and_which_view(doc):
    meta = doc["meta"]
    assert meta["desktop"] == DESKTOP
    assert meta["brand"] == "Bench" and meta["model_family"] == "Bench 1"
    # no sheet text: the scanner package's fallback, which is also the name
    assert meta["model"] == "Bench Bench 1" and meta["model_name"] == "Bench Bench 1"
    assert meta["model_note"] is None
    assert meta["export_graph_version"] == 1
    assert meta["graph_version"] and len(meta["graph_version"]) == 16
    assert meta["view"] == VIEW and meta["pixel_views"] == [VIEW]
    assert meta["counts"]["confirmed_frames"] == len(CONFIRMED)
    assert meta["counts"]["edges"] == len(doc["edges"])


def test_instances_carry_relations_and_a_timeline(doc):
    by_key = {i["key"]: i for i in doc["instances"]}
    assert set(by_key) == set(bench_instances())
    cover = by_key["cover.01"]
    assert [(t["step"], t["state"]) for t in cover["timeline"]] == [(0, "closed"), (2, "open")]
    # our bookkeeping (a note to ourselves, when it was added) is not published (S3)
    assert cover["attrs"] == {"of": "motherboard_screws"}
    screw = by_key["screw.cpu_cooler.01"]
    assert screw["parent"] == "cpu_cooler.fan.01" and screw["attached"] is True
    assert screw["fastens"] == "cpu_cooler.fan.01"
    atx = by_key["connector.atx_24pin.01"]
    assert atx["socket_host"] == "motherboard.01" and atx["cable"] == "cable:psu"
    assert atx["implied"] is False and atx["extra"] is False
    board = by_key["motherboard.01"]
    assert board["timeline"][0]["step"] == 0
    assert board["timeline"][-1] == {"step": 8, "state": "removed",
                                     "placement": board["timeline"][-1]["placement"]}


def test_steps_carry_every_action_failed_ones_included(doc):
    by_step = {s["step"]: s for s in doc["steps"]}
    assert by_step[1]["type"] == "initial" and by_step[1]["actions"] == []
    [failed] = by_step[4]["actions"]
    assert failed == {"idx": 0, "verb": "open", "target": "psu_latch.01", "tool": "none",
                      "direction": "none", "result": "failed", "failure_reason": "stuck"}


def test_edges_are_the_active_ones_with_the_step_that_satisfied_them(scene, doc):
    db = ReadOnlyDb(str(scene))
    try:
        stored = active_edges(constraint_edges(edges_from_db(db, DESKTOP)))
    finally:
        db.close()
    assert {e["label"] for e in doc["edges"]} == {e.label() for e in stored}
    by_label = {e["label"]: e for e in doc["edges"]}
    assert by_label["fastened_by(motherboard.01, screw.motherboard.01)"]["demonstrated_at"] == 5
    assert by_label["covered_by(screw.motherboard.01, cover.01)"]["demonstrated_at"] == 2
    # never unplugged, so never satisfied in this log
    assert by_label["connected_to(motherboard.01, connector.sata_data.02)"][
        "demonstrated_at"] is None
    for edge in doc["edges"]:
        assert set(edge) >= {"type", "target", "blocker", "necessity", "mode", "reason",
                             "source", "status", "evidence_step", "demonstrated_at"}


def test_covers_from_the_stored_rows_explained_and_candidate(doc):
    covers = {(c["above"], c["below"]): c for c in doc["covers"]}
    fan = covers[("cpu_cooler.fan.01", "cpu.01")]
    assert fan["explained_by"] == ["covered_by(cpu.01, cpu_cooler.fan.01)"]
    assert fan["candidate"] is False and fan["no_edge"] is False
    assert fan["frames"] == list(CONFIRMED) and fan["n_frames"] == 3
    assert (fan["first"], fan["last"]) == (1, 3)
    # the fan's visible pixels over the CPU's whole shape: x 5..8, y 2..8
    assert fan["max_px"] == 4 * 7
    # the segment's pairwise layering exception put the fan over the CPU
    assert fan["order_source"] == "pair_override"
    assert fan["order_sources"] == {"pair_override": 3}

    psu = covers[("psu.01", "storage_drive.hdd.01")]
    assert psu["explained_by"] == [] and psu["reverse_edges"] == []
    assert psu["structural"] == [] and psu["candidate"] is True
    assert psu["max_px"] == 5 * 5
    # frame 2: the drive's confirmed mask still claims the pixels the PSU's does, so
    # the PSU takes none of them there -- the `- visible(below)` term
    assert psu["frames"] == [1, 3]
    # frame 3 carries a 「只改这一帧」 mask of the PSU; one frame each, the specific wins
    assert psu["order_sources"] == {"per_frame_mask": 1, "zorder": 1}
    assert psu["order_source"] == "per_frame_mask"
    assert psu["below_removed_at"] is None and psu["above_removed_at"] is None
    assert ("storage_drive.hdd.01", "psu.01") not in covers

    # a captive screw rides out with its fan, but the fan does not cover the drive:
    # nothing accounts for the screw over the drive (S4) ...
    loose = covers[(SCREW, "storage_drive.hdd.01")]
    assert loose["structural"] == [] and loose["candidate"] is True
    # ... while over the chassis the fan's own pair is listed
    assert covers[("cpu_cooler.fan.01", "chassis")]
    assert covers[(SCREW, "chassis")]["structural"] == ["above.rides_with_parent",
                                                         "below_fixed"]

    # everything covers the chassis, which never moves: no candidate
    chassis = covers[("cpu.01", "chassis")]
    assert chassis["structural"] == ["below_fixed"] and chassis["candidate"] is False
    # the frame nobody confirmed puts the CPU over the fan: not read
    assert ("cpu.01", "cpu_cooler.fan.01") not in covers
    # the RAM stick on the bench lies across the chassis: not read either
    assert not [c for c in doc["covers"] if BENCH_PART in (c["above"], c["below"])]
    assert doc["meta"]["counts"]["candidates"] == sum(c["candidate"] for c in doc["covers"])


def test_legal_by_step_agrees_with_legal_actions(scene, doc):
    tax = load_taxonomy()
    db = ReadOnlyDb(str(scene))
    try:
        ctx = load_ctx(db, tax, DESKTOP, VIEW)
        edges = active_edges(constraint_edges(edges_from_db(db, DESKTOP)))
        for entry in doc["legal_by_step"]:
            state = ctx.state_at(entry["step"])
            strict = [f"{v} {t}" for v, t in legal_actions(ctx.instances, edges, state, tax)]
            relaxed = [f"{v} {t}" for v, t in legal_actions(ctx.instances, edges, state, tax,
                                                            strict=False)]
            assert entry["legal"] == strict and entry["n_legal"] == len(strict)
            assert entry["legal_relaxed"] == relaxed
    finally:
        db.close()
    by_step = {e["step"]: e for e in doc["legal_by_step"]}
    assert [e["step"] for e in doc["legal_by_step"]] == list(range(0, LAST_STEP + 1))
    assert by_step[1]["next_demonstrated"] == ["open cover.01"]
    assert by_step[1]["next_is_legal"] is True and by_step[1]["next_illegal"] == []
    assert by_step[3]["next_failed"] == ["open psu_latch.01"]
    assert by_step[3]["next_is_legal"] is None          # nothing succeeded at step 4
    # the board leaves with the SATA plug still in: not legal
    assert by_step[7]["next_demonstrated"] == ["remove motherboard.01"]
    assert by_step[7]["next_is_legal"] is False
    assert by_step[7]["next_illegal"] == [{"action": "remove motherboard.01",
                                           "reason": "unmet_edge"}]
    # reorient is gated by nothing and never in a legal set: not judged (S1)
    assert by_step[8]["next_demonstrated"] == ["reorient chassis"]
    assert by_step[8]["next_is_legal"] is None and by_step[8]["next_illegal"] == []
    # a part the sheet did not name: not judged either
    assert by_step[9]["next_unjudged"] == ["remove ?"]
    assert by_step[9]["next_is_legal"] is None and by_step[9]["next_is_legal_relaxed"] is None
    assert by_step[LAST_STEP]["next_step"] is None


def test_plans_from_the_initial_state_and_the_progress_label(doc):
    plans = doc["plans"]
    assert set(plans) == {"motherboard.01", "cpu.01", "psu.01", "all"}
    board = plans["motherboard.01"]
    assert board["plan"][-1] == "remove motherboard.01"
    assert board["length"] == len(board["plan"])
    assert "disconnect connector.sata_data.02" in board["plan"]
    assert board["removed_at"] == 8 and board["demonstrated_length"] == 6
    assert plans["psu.01"]["removed_at"] is None
    assert plans["psu.01"]["demonstrated_length"] is None
    everything = plans["all"]
    # an unscrewed screw is out of the machine too, so the log removes four parts
    assert everything["goals"] == ["screw.motherboard.01", "screw.motherboard.02",
                                   "screw.motherboard.03", "motherboard.01"]
    assert sorted(everything["plan"]) == sorted(board["plan"])
    assert everything["unplanned"] == [] and everything["length"] == board["length"]
    assert everything["removed_at"] == 8 and everything["demonstrated_length"] == 6
    remaining = everything["remaining_steps_by_step"]
    assert len(remaining) == len(doc["legal_by_step"])
    assert remaining[0] == board["length"] and remaining[-1] == 0
    assert remaining == sorted(remaining, reverse=True)


def test_violations_are_the_validate_lines_split(scene, doc):
    tax = load_taxonomy()
    db = ReadOnlyDb(str(scene))
    try:
        edges = active_edges(constraint_edges(edges_from_db(db, DESKTOP)))
        lines = validate_sequence(db.instances(DESKTOP), edges, db.actions(DESKTOP), tax)
    finally:
        db.close()
    assert [v["line"] for v in doc["violations"]] == lines
    kinds = sorted(v["kind"] for v in doc["violations"])
    assert kinds == [FAILED_WITHOUT_UNMET, SUCCESS_VIOLATES]
    [breach] = [v for v in doc["violations"] if v["kind"] == SUCCESS_VIOLATES]
    assert breach == {
        "step": 8, "verb": "remove", "target": "motherboard.01",
        "edge": "connected_to(motherboard.01, connector.sata_data.02)",
        "edge_type": "connected_to", "blocker": "connector.sata_data.02",
        "blocker_state": "plugged", "kind": SUCCESS_VIOLATES, "line": breach["line"],
        "timeline_agrees": True}
    [hint] = [v for v in doc["violations"] if v["kind"] == FAILED_WITHOUT_UNMET]
    assert (hint["step"], hint["verb"], hint["target"]) == (4, "open", "psu_latch.01")
    assert hint["timeline_agrees"] is True
    assert doc["meta"]["counts"]["violations_timeline_disagrees"] == 0
    assert doc["deadlocks"] == [] and doc["dead_ends"] == []


def test_a_hand_written_event_the_replay_ignores_is_flagged(tmp_path):
    """S2: validate_sequence replays actions only; the timeline folds the event in."""
    path = build(tmp_path / "hand.sqlite", hand_unplug=True)
    db = ReadOnlyDb(str(path))
    try:
        doc = build_desktop(db, load_taxonomy(), DESKTOP, VIEW)
    finally:
        db.close()
    [breach] = [v for v in doc["violations"] if v["kind"] == SUCCESS_VIOLATES]
    assert breach["step"] == 8 and breach["timeline_agrees"] is False
    by_step = {e["step"]: e for e in doc["legal_by_step"]}
    assert by_step[7]["next_is_legal"] is True          # the timeline has it unplugged
    summary = summarize([doc], VIEW)
    assert summary["totals"]["violations_timeline_disagrees"] == 1
    assert summary["totals"]["violations_timeline_agrees"] == 1      # the failed attempt


def test_parse_violation_keeps_a_line_it_does_not_know():
    assert parse_violation("something else")["kind"] == "other"
    out = parse_violation("step 3: remove a.01 violates locked_by(a.01, b.01) (b.01 is None)")
    assert out["blocker_state"] is None and out["edge_type"] == "locked_by"


def test_structural_reasons_and_explaining_edges():
    tax = load_taxonomy()
    instances = bench_instances()
    assert structural_reasons(instances, tax, "ram_latch.01", "motherboard.01") == [
        "above.attached_to"]
    # rides with its fan -- only a reason when the fan's own pair is listed (S4)
    assert structural_reasons(instances, tax, "screw.cpu_cooler.01", "motherboard.01",
                              {("cpu_cooler.fan.01", "motherboard.01")}) == [
        "above.rides_with_parent"]
    assert structural_reasons(instances, tax, "screw.cpu_cooler.01", "motherboard.01",
                              {("psu.01", "motherboard.01")}) == []
    assert structural_reasons(instances, tax, "screw.cpu_cooler.01", "motherboard.01") == []
    assert structural_reasons(instances, tax, "connector.atx_24pin.01", "motherboard.01") == [
        "above.socket_host"]
    assert structural_reasons(instances, tax, "psu.01", "storage_drive.hdd.01") == []
    edges = propose_edges(instances, tax)
    assert explaining_edges(edges, "cpu_cooler.fan.01", "cpu.01") == [
        "covered_by(cpu.01, cpu_cooler.fan.01)"]
    assert explaining_edges(edges, "cpu.01", "cpu_cooler.fan.01") == [
        "covered_by(cpu.01, cpu_cooler.fan.01)"]       # either way round
    assert explaining_edges(edges, "motherboard.01", "screw.motherboard.01") == []


# --------------------------------------------------------------------------- #
# the run: files, determinism, read-only, CLI
# --------------------------------------------------------------------------- #
def test_two_runs_are_byte_identical_and_the_database_is_untouched(scene, tmp_path):
    before = sha(scene)
    for name in ("a", "b"):
        assert main(["export-graph", "--db", str(scene), "--desktops", str(DESKTOP),
                     "--out", str(tmp_path / name)]) == EXIT_OK
    assert sha(scene) == before
    files = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert files == sorted([desktop_file_name(DESKTOP), SUMMARY_NAME, README_NAME])
    for name in files:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    text = (tmp_path / "a" / README_NAME).read_text(encoding="utf-8")
    assert "中文摘要" in text and "`legal_by_step[]`" in text and str(scene) not in text
    assert str(tmp_path) not in (tmp_path / "a" / desktop_file_name(DESKTOP)).read_text(
        encoding="utf-8")


def test_the_read_only_database_refuses_a_write(scene):
    db = ReadOnlyDb(str(scene))
    try:
        with pytest.raises(Exception, match="readonly"):
            db.upsert_desktop(DESKTOP, {"brand": "changed"})
    finally:
        db.close()


def test_cli_arguments(scene, tmp_path, capsys):
    ap = build_parser()
    args = ap.parse_args(["export-graph", "--out", "x", "--desktops", "1-3"])
    assert (args.out, args.desktops, args.view) == ("x", "1-3", "scan")
    assert ap.parse_args(["export-graph", "--out", "x", "--view", "oak1"]).view == "oak1"
    with pytest.raises(SystemExit):
        ap.parse_args(["export-graph", "--desktops", "1"])          # --out is required
    with pytest.raises(SystemExit):
        ap.parse_args(["export-graph", "--out", "x", "--view", "cmp_scan"])
    assert main(["export-graph", "--db", str(scene), "--desktops", f"{DESKTOP},901",
                 "--out", str(tmp_path / "o")]) == EXIT_OK
    assert "D901" in capsys.readouterr().out
    summary = json.loads((tmp_path / "o" / SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["desktops"] == [DESKTOP]
    assert main(["export-graph", "--db", str(tmp_path / "missing.sqlite"),
                 "--out", str(tmp_path / "o2")]) != EXIT_OK
    # none of the requested desktops is in the database: an error, nothing written (N1)
    assert main(["export-graph", "--db", str(scene), "--desktops", "901",
                 "--out", str(tmp_path / "o3")]) != EXIT_OK
    assert not (tmp_path / "o3" / SUMMARY_NAME).exists()


def test_the_whole_export_reads_one_snapshot(scene, tmp_path):
    """N6: a write to the file while the export runs is not seen by it."""
    copy = tmp_path / "live.sqlite"
    shutil.copyfile(scene, copy)
    reader = ReadOnlyDb(str(copy))
    try:
        writer = Db(str(copy))
        try:
            writer.upsert_desktop(DESKTOP, {"brand": "changed"})
        finally:
            writer.close()
        assert reader.get_desktop(DESKTOP)["brand"] == "Bench"
    finally:
        reader.close()
    fresh = ReadOnlyDb(str(copy))
    try:
        assert fresh.get_desktop(DESKTOP)["brand"] == "changed"
    finally:
        fresh.close()


def test_export_refuses_a_gallery_view(scene, tmp_path):
    db = ReadOnlyDb(str(scene))
    try:
        with pytest.raises(ValueError):
            export_graph(db, load_taxonomy(), [DESKTOP], str(tmp_path / "g"), "cmp_scan")
    finally:
        db.close()


REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def test_the_docs_say_it():
    spec = SPEC.read_text(encoding="utf-8")
    cv = spec.split("### 8.1 CV 导出", 1)[1].split("### 8.2", 1)[0]
    [bullet] = [line for line in cv.splitlines()
                if line.startswith("- **每机优先级图 JSON（v1.45，U46）**")]
    for needed in ("`python -m tda.cli export-graph", "**只读**", "`covers`", "`legal_by_step`",
                   "`plans`", "`violations`", "`remaining_steps_by_step`", "`summary.json`",
                   "`README.md`", "逐字节相同", "`occluder_mask`", "`timeline_agrees`",
                   "`reorient`", "记账字段"):
        assert needed in bullet, needed
    changes = spec.split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # on top when it landed; U44 / U45 (v1.43 / v1.44) are ordered at their merges
    assert any(e.startswith("- **v1.45（2026-10-30）**：**每机优先级图 JSON**（U46，§8.1）")
               for e in entries[:6])
    handoff = (REPO / "docs" / "HANDOFF_2026-10-28.md").read_text(encoding="utf-8")
    section = handoff.split("## 5. 导出与 HuggingFace 准备", 1)[1].split("\n## ", 1)[0]
    assert sum("export-graph" in line for line in section.splitlines()) == 1


def _fake(desktop: int, covers: list[dict], violations=()) -> dict:
    return {
        "meta": {"desktop": desktop, "graph_version": "x", "model_family": "F",
                 "counts": {"edges": 1, "edges_by_type": {"covered_by": 1},
                            "confirmed_frames": 2, "covers": len(covers),
                            "no_edge": sum(c["no_edge"] for c in covers),
                            "candidates": sum(c["candidate"] for c in covers),
                            "violations": len(violations), "deadlocks": 0,
                            "dead_ends": 0}},
        "covers": covers, "violations": list(violations),
        "legal_by_step": [{"next_is_legal": True}, {"next_is_legal": False},
                          {"next_is_legal": None}],
        "plans": {"all": {"length": 3, "demonstrated_length": 4}},
    }


def _cover(above: str, below: str, candidate: bool = True, below_at=5, above_at=3) -> dict:
    return {"above": above, "below": below, "above_class": above.rsplit(".", 1)[0],
            "below_class": below.rsplit(".", 1)[0], "n_frames": 2, "max_px": 9,
            "candidate": candidate, "no_edge": candidate,
            "above_removed_at": above_at, "below_removed_at": below_at}


def test_summary_groups_candidates_by_class_pair():
    docs = [
        _fake(2, [_cover("cover.01", "cpu_cooler.fan.01"),
                  _cover("cover.02", "cpu_cooler.fan.01", below_at=None),
                  _cover("psu.01", "cpu.01", candidate=False)]),
        _fake(1, [_cover("cover.01", "cpu_cooler.fan.01", above_at=9)],
              [{"kind": SUCCESS_VIOLATES, "line": "step 1: x"}]),
    ]
    summary = summarize(docs, VIEW)
    assert summary["desktops"] == [1, 2]
    [group] = summary["candidates_by_class_pair"]
    assert group["pair"] == "cover → cpu_cooler.fan"
    assert group["desktops"] == [1, 2] and group["n_desktops"] == 2 and group["n_pairs"] == 3
    assert group["below_removed"] == 2 and group["above_removed_first"] == 1
    assert summary["violations"] == [{"desktop": 1, "kind": SUCCESS_VIOLATES,
                                      "line": "step 1: x"}]
    totals = summary["totals"]
    assert totals["candidates"] == 3 and totals["covers"] == 4
    assert totals["next_is_legal"] == {"legal": 2, "judged": 4, "ratio": 0.5}
