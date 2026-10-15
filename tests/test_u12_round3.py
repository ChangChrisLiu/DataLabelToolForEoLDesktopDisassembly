"""Task U12 round 3: the review's blocker (a latch's ``of``) and minors 1-5."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from merge_scenes import DESKTOP, GONE, KEEP, PSU, build, dump
from tda.core import extra as X
from tda.core.db import Db
from tda.core.extra import recompile_events, restamp_graph
from tda.core.graph import edges_from_db
from tda.core.graph_derive import derive_edges
from tda.core.merge_instances import MergeError, merge_instances, plan_merge, undo_merge
from tda.core.merge_replay import OP_REPLAY_MERGES, replay_merges
from tda.core.model import ActionRec, InstanceRec, StepRec
from tda.core.taxonomy import load_taxonomy
from tda.pipeline_logs import import_logs_into_db

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def db(tmp_db_path):
    d = Db(tmp_db_path)
    yield d
    d.close()


# --------------------------------------------------------------------------- #
# B1: a latch's `of` follows the merged module
# --------------------------------------------------------------------------- #
def _ram(db: Db, tax) -> None:
    db.upsert_desktop(1, {})
    for key in ("ram_module.01", "ram_module.02"):
        db.upsert_instance(InstanceRec(key, 1, "ram_module"))
    db.upsert_instance(InstanceRec("ram_latch.01", 1, "ram_latch", attrs={"of": "ram_module.02"}))
    db.upsert_instance(InstanceRec("ram_latch.02", 1, "ram_latch", attrs={"of": "ram_module.01"}))
    db.replace_steps(1, [StepRec(1, s, "normal", f"step {s}") for s in (1, 2, 3)],
                     [ActionRec(1, 2, 0, "ram_latch.01", "open"),
                      ActionRec(1, 3, 0, "ram_module.02", "remove")])
    recompile_events(db, 1, tax)
    derived = derive_edges(db.instances(1), edges_from_db(db, 1), tax)
    for edge in derived.edges:
        db.add_relation(1, edge.type, edge.target, edge.blocker, source=edge.source,
                        reason=edge.reason, status=edge.status)
    restamp_graph(db, 1, tax)


def test_a_latch_of_the_merged_module_follows_and_its_edge_survives(db, tax):
    _ram(db, tax)
    before = dump(db)
    assert ("locked_by", "ram_module.02", "ram_latch.01") in {
        (r["type"], r["target"], r["blocker"]) for r in db.relations(1)}

    report = merge_instances(db, tax, 1, "ram_module.02", "ram_module.01", "tester")

    latches = db.instances(1)
    assert latches["ram_latch.01"].attrs["of"] == "ram_module.01"
    assert latches["ram_latch.02"].attrs["of"] == "ram_module.01"
    assert [r.key for r in report.plan.neighbours_after] == ["ram_latch.01"]
    edges = {(r["type"], r["target"], r["blocker"]) for r in db.relations(1)}
    assert ("locked_by", "ram_module.01", "ram_latch.01") in edges
    # the next S1 Apply / `constraints` run derives exactly that edge: nothing stale
    again = derive_edges(db.instances(1), edges_from_db(db, 1), tax)
    assert again.stale == []
    assert "指向它的实例字段 1" in "\n".join(report.plan.summary_lines())

    undo_merge(db, tax, db.get_op(report.op_id))
    assert dump(db) == before                       # of=ram_module.02 is back, exactly


def test_keeps_own_of_at_gone_is_cleared(db, tax):
    _ram(db, tax)
    latch = db.instances(1)["ram_latch.01"]          # of=ram_module.02
    db.upsert_instance(InstanceRec("ram_latch.03", 1, "ram_latch", attrs={"of": "ram_latch.01"}))
    merge_instances(db, tax, 1, "ram_latch.03", "ram_latch.01", "tester")
    assert db.instances(1)["ram_latch.01"].attrs == latch.attrs
    db.upsert_instance(InstanceRec("ram_latch.04", 1, "ram_latch"))
    keep = db.instances(1)["ram_latch.04"]
    keep.attrs["of"] = "ram_latch.01"
    db.upsert_instance(keep)
    merge_instances(db, tax, 1, "ram_latch.01", "ram_latch.04", "tester")
    assert "of" not in db.instances(1)["ram_latch.04"].attrs


def test_the_forced_re_import_re_points_of_too():
    instances = {
        "ram_module.01": InstanceRec("ram_module.01", 1, "ram_module"),
        "ram_module.02": InstanceRec("ram_module.02", 1, "ram_module"),
        "ram_latch.01": InstanceRec("ram_latch.01", 1, "ram_latch",
                                    attrs={"of": "ram_module.02"}),
    }
    table, _out, report = replay_merges(
        [{"ops": [1], "gone": "ram_module.02", "keep": "ram_module.01", "cls": "ram_module"}],
        instances, [ActionRec(1, 3, 0, "ram_module.02", "remove")])
    assert table["ram_latch.01"].attrs["of"] == "ram_module.01"
    assert instances["ram_latch.01"].attrs["of"] == "ram_module.02"     # input untouched
    assert [e["gone"] for e in report.replayed] == ["ram_module.02"]


# --------------------------------------------------------------------------- #
# minor 2: a merge into a part added in S1 is replayed on --force
# --------------------------------------------------------------------------- #
def test_a_merge_into_an_added_part_survives_a_forced_import(tmp_path, tax):
    folder = tmp_path / "drive"
    folder.mkdir()
    for name in ("desktop_02.csv", "desktop_02_meta.csv"):
        shutil.copy(FIXTURES / name, folder / name)
    db = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(db, str(folder), tax, desktops={2})
    batch = X.plan_extras(db.instances(2), tax, 2, "cover", 1, annotator="tester")
    X.write_batch(db, batch, "tester")
    [added] = batch.keys
    merge_instances(db, tax, 2, "cover.02", added, "tester")
    for _ in range(2):
        run = import_logs_into_db(db, str(folder), tax, desktops={2}, force=True)
        assert "cover.02" not in db.instances(2) and added in db.instances(2)
        assert [a.target for a in db.actions(2) if a.step == 46] == [added]
        op = db.ops_of_kind(2, OP_REPLAY_MERGES)[-1]
        assert [(e["gone"], e["keep"]) for e in op["payload"]["replayed"]] == [
            ("cover.02", added)]
        assert op["payload"]["unmatched"] == []
        assert not any("not matched" in line for line in run.runs[0].corrections)
    db.close()


# --------------------------------------------------------------------------- #
# minor 3: refusals point at 「Apply 并合并」, never at a separate Apply
# --------------------------------------------------------------------------- #
def test_refusals_point_at_apply_and_merge(db):
    scene = build(db, s1=False)
    with pytest.raises(MergeError) as frozen:
        plan_merge(db, scene.tax, DESKTOP, GONE, KEEP)
    scene.set_actions([ActionRec(DESKTOP, 3, 0, KEEP, "open"),
                       ActionRec(DESKTOP, 5, 0, KEEP, "remove"),
                       ActionRec(DESKTOP, 5, 1, GONE, "remove")])
    with pytest.raises(MergeError) as actions:
        plan_merge(db, scene.tax, DESKTOP, GONE, KEEP)
    for caught in (frozen, actions):
        text = str(caught.value)
        assert "「Apply 并合并」" in text and "不用先 Apply" in text
        assert "Apply，再合并" not in text


# --------------------------------------------------------------------------- #
# minor 4: undo never overwrites a later confirmation; it re-queues everything
# --------------------------------------------------------------------------- #
def test_undo_is_refused_once_a_rewritten_frame_was_confirmed_since(db):
    scene = build(db)
    report = merge_instances(db, scene.tax, DESKTOP, GONE, KEEP, "tester")
    scene.refresh_all()
    scene.svc.verify_frame(scene.key(1), "tester")      # cover.01 drawn there now
    before = dump(db)
    with pytest.raises(MergeError, match="合并后又确认过这些帧（scan 第 1 帧）"):
        undo_merge(db, scene.tax, db.get_op(report.op_id))
    assert dump(db) == before


def test_undo_re_queues_every_frozen_frame(db):
    scene = build(db)
    report = merge_instances(db, scene.tax, DESKTOP, GONE, KEEP, "tester")
    scene.refresh_all()
    with db.transaction():
        db.conn.execute("DELETE FROM recheck_queue")   # the sweeper drained it
    undo_merge(db, scene.tax, db.get_op(report.op_id))
    assert db.rechecks(DESKTOP, "scan") == [3, 4, 5, 6]


# --------------------------------------------------------------------------- #
# minor 5: the confirmation names rule edges and gone's own columns
# --------------------------------------------------------------------------- #
def test_the_confirmation_names_rule_edges_and_dropped_columns(db):
    scene = build(db)
    db.delete_relation(DESKTOP, "covered_by", PSU, GONE)
    rec = db.instances(DESKTOP)[KEEP]
    rec.parent, rec.removal_direction = PSU, "+Z"
    db.upsert_instance(rec)
    plan = plan_merge(db, scene.tax, DESKTOP, KEEP, GONE)   # keep cover.02 this time
    text = "\n".join(plan.summary_lines())
    assert "（其中 1 条规则边将按保留实例 cover.02 的属性重算）" in text
    assert ("cover.01 的 parent=psu.01（cover.02 是 空）、removal_direction=+Z（cover.02 是 空）"
            " 也不带过去。") in text
    assert "属性 {'of': 'motherboard_screws'}" in text
