"""Task U12: 「合并到…」 -- fold one instance into another of the same class.

The scene (``merge_scenes``) is D2's cover in miniature: ``cover.02`` carries
the shapes, the verified rows and the layer, ``cover.01`` the log's actions
once S1 has corrected them; merging ``cover.02`` into ``cover.01`` leaves one
cover that is closed until 3, open 3-4 and taken out at 5.
"""
from __future__ import annotations

import pytest

from merge_scenes import DESKTOP, GONE, KEEP, PSU, STEPS, VIEWS, blob, build, dump, mask_kf
from tda.core.db import Db
from tda.core.extra import recompile_events, restamp_graph
from tda.core.merge_instances import (
    OP_MERGE,
    OP_UNDO,
    MergeError,
    merge_instances,
    plan_merge,
    ranges,
    redo_merge,
    undo_merge,
)
from tda.core.model import ActionRec, FrameKey, FrameOverride, InstanceRec, PairOverride, StateEvent
from tda.core.states import state_at, validate_events
from tda.core.truth_inputs import events_of


@pytest.fixture
def db(tmp_db_path):
    d = Db(tmp_db_path)
    yield d
    d.close()


@pytest.fixture
def scene(db):
    return build(db)


def _states(scene, key=KEEP):
    inst = scene.db.instances(DESKTOP)
    events = events_of(scene.db, scene.tax, DESKTOP)
    return [state_at(inst, events, step, scene.tax)[key].state for step in STEPS]


def _merge(scene, gone=GONE, keep=KEEP):
    return merge_instances(scene.db, scene.tax, DESKTOP, gone, keep, "tester")


def _ops(db, kind):
    return [op for op in db.ops(DESKTOP, "-", limit=1000) if op["kind"] == kind]


# --------------------------------------------------------------------------- #
# the D2 case
# --------------------------------------------------------------------------- #
def test_the_d2_cover_merges_into_one_part(scene):
    verified = {s: scene.status(s) for s in STEPS}
    frozen = {s: scene.db.compiled(scene.key(s))[GONE]["visible_rle"]
              for s in (3, 4)}

    report = _merge(scene)

    db = scene.db
    assert GONE not in db.instances(DESKTOP)
    assert [kf.anchor_step for kf in db.keyframes(DESKTOP, "scan", KEEP)] == [2, 4]
    assert db.keyframes(DESKTOP, "scan", GONE) == []
    rows = {s: db.compiled(scene.key(s)) for s in STEPS}
    assert all(GONE not in r for r in rows.values())
    assert {s: rows[s][KEEP]["status"] for s in (3, 4)} == {3: "verified", 4: "verified"}
    assert db.zorder(DESKTOP, "scan", 1).order == [(PSU, "main"), (KEEP, "main")]
    assert _states(scene) == ["closed", "closed", "open", "open", "removed", "removed"]
    assert validate_events(db.instances(DESKTOP), events_of(db, scene.tax, DESKTOP),
                           scene.tax) == []
    # the hand-written "open at 3" is what the retargeted action says anyway
    assert [(r["step"], r["target"]) for r, _why in report.plan.events_dropped] == [(3, GONE)]
    assert "已经是 open" in report.plan.events_dropped[0][1]
    assert (report.moved["shape_keyframe"], report.moved["compiled_mask"],
            report.moved["zorder"]) == (2, 2, 1)
    assert report.dropped["relation"] == 1       # covered_by(psu.01, ...) twice
    # one op row with the whole record of what went
    [op] = _ops(db, OP_MERGE)
    assert op["payload"]["gone"] == GONE and op["payload"]["keep"] == KEEP
    assert op["payload"]["gone_record"]["raw_names"] == ["Front Cover"]
    assert op["payload"]["gone_record"]["attrs"] == {"of": "front_bezel"}

    totals = scene.refresh_all()
    assert totals["conflicts"] == 0 and totals["standing"] == 0
    assert db.conflicts(DESKTOP, open_only=True) == []
    assert {s: scene.status(s) for s in STEPS} == verified          # nothing demoted
    assert {s: db.compiled(scene.key(s))[KEEP]["visible_rle"] for s in (3, 4)} == frozen
    assert KEEP in db.compiled(scene.key(1)) and db.compiled(scene.key(1))[KEEP]["visible_rle"]


def test_keep_keeps_its_record(scene):
    _merge(scene)
    rec = scene.db.instances(DESKTOP)[KEEP]
    assert rec.attrs == {"of": "motherboard_screws"} and rec.raw_names == ["Screw Cover"]


def test_the_other_direction_merges_too(scene):
    report = _merge(scene, gone=KEEP, keep=GONE)
    db = scene.db
    assert KEEP not in db.instances(DESKTOP)
    assert {(a.step, a.target, a.verb) for a in db.actions(DESKTOP)} == {
        (3, GONE, "open"), (5, GONE, "remove")}
    assert _states(scene, GONE) == ["closed", "closed", "open", "open", "removed", "removed"]
    # cover.02's own "open at 3" became a no-op once the action opens it
    assert [(r["step"], r["target"]) for r, _w in report.plan.events_dropped] == [(3, GONE)]
    assert scene.refresh_all()["conflicts"] == 0
    assert scene.status(3) == "verified"


def test_the_summary_says_what_moves_and_what_goes(scene):
    plan = plan_merge(scene.db, scene.tax, DESKTOP, GONE, KEEP)
    text = "\n".join(plan.summary_lines())
    assert "把 cover.02（Front Cover） 合并到 cover.01（Screw Cover）" in text
    assert "形状 2 个（scan 锚在第 2、4 帧）" in text
    assert "已确认的行 2 行（scan 第 3–4 帧）" in text
    assert "缓存行" in text and "手写状态事件 cover.02 第 3 步 closed→open" in text
    assert "{'of': 'front_bezel'}" in text
    assert "不经 Apply" in text


def test_ranges():
    assert ranges([37, 38, 39, 41, 45, 44]) == "37–39、41、44–45"
    assert ranges([]) == ""


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def _refused(scene, gone=GONE, keep=KEEP) -> str:
    before = blob(scene.db)
    with pytest.raises(MergeError) as caught:
        _merge(scene, gone, keep)
    assert blob(scene.db) == before                        # nothing written
    return str(caught.value)


def test_refuses_another_class(scene):
    assert "只有同一类" in _refused(scene, GONE, PSU)


def test_refuses_itself_and_unknown_keys(scene):
    assert "它自己" in _refused(scene, KEEP, KEEP)
    assert "没有实例 cover.09" in _refused(scene, "cover.09", KEEP)


def test_refuses_a_label_studio_draft(scene):
    scene.db.upsert_instance(InstanceRec("ls:Cover#1", DESKTOP, "cover"))
    assert "草稿要采纳" in _refused(scene, "ls:Cover#1", KEEP)
    assert "草稿要采纳" in _refused(scene, KEEP, "ls:Cover#1")


def test_refuses_two_actions_at_one_step(scene):
    scene.set_actions([ActionRec(DESKTOP, 3, 0, KEEP, "open"),
                       ActionRec(DESKTOP, 5, 0, KEEP, "remove"),
                       ActionRec(DESKTOP, 5, 1, GONE, "remove")])
    text = _refused(scene)
    assert "第 5 步 cover.02 和 cover.01 都有动作" in text


def test_refuses_two_shapes_on_one_frame(scene):
    scene.db.add_keyframe(mask_kf(KEEP, (40, 4, 50, 14), 2))
    text = _refused(scene)
    assert "都在、都有形状" in text and "scan 第 1–2 帧" in text


def test_shapes_in_different_frames_are_one_part(db):
    """keep drawn at 1-2 in oak1 only, gone in scan: no frame has both."""
    scene = build(db)
    db.add_keyframe(mask_kf(KEEP, (40, 4, 50, 14), 2, "oak1"))
    _merge(scene)
    assert [kf.anchor_step for kf in db.keyframes(DESKTOP, "oak1", KEEP)] == [2]


def test_refuses_two_verified_rows_on_one_frame(scene):
    db = scene.db
    row = db.compiled(scene.key(3))[GONE]
    db.put_compiled(scene.key(3), KEEP, row["visible_rle"], 0.0, "visible", "in_chassis",
                    "verified", "x", verified_by="tester")
    assert "各有一行已确认" in _refused(scene)


def test_refuses_two_overrides_on_one_frame(scene):
    for key in (GONE, KEEP):
        scene.db.set_frame_override(FrameOverride(scene.key(2), key, visibility="visible"))
    assert "都有单帧覆盖" in _refused(scene)


def test_refuses_what_the_merged_log_cannot_hold(scene):
    """keep re-closed by hand at 6; after the merge it is taken out at 5."""
    scene.set_actions([ActionRec(DESKTOP, 3, 0, KEEP, "open"),
                       ActionRec(DESKTOP, 5, 0, GONE, "remove")])
    scene.db.add_events([StateEvent(DESKTOP, 6, KEEP, "state", "open", "closed", "scan",
                                    auto=False)])
    recompile_events(scene.db, DESKTOP, scene.tax)
    assert "合并后状态日志会出错" in _refused(scene)


def test_refuses_to_contradict_confirmed_frames(db):
    """Merging before S1 is corrected: cover.01 is taken out at 3, but frames
    3-4 were confirmed with the cover drawn on them."""
    scene = build(db, s1=False)
    text = _refused(scene)
    assert "scan 第 3–4 帧：确认时有它，合并后按步骤表它已不在画面里" in text
    assert "先在 Steps 把步骤改对" in text


def test_refuses_a_parent_loop(scene):
    db = scene.db
    psu = db.instances(DESKTOP)[PSU]
    psu.parent = GONE
    db.upsert_instance(psu)
    keep = db.instances(DESKTOP)[KEEP]
    keep.parent = PSU
    db.upsert_instance(keep)
    assert "parent 会成环" in _refused(scene)


# --------------------------------------------------------------------------- #
# what moves, table by table
# --------------------------------------------------------------------------- #
def test_actions_and_overrides_and_conflicts_follow(db):
    scene = build(db)
    scene.set_actions([ActionRec(DESKTOP, 3, 0, KEEP, "open"),
                       ActionRec(DESKTOP, 5, 0, GONE, "remove")])
    db.set_frame_override(FrameOverride(scene.key(2), GONE, visibility="occluded_partial"))
    cid = db.add_conflict(scene.key(4), GONE, None, None, 3)
    db.conn.execute("INSERT INTO instance_frame_flags VALUES(?,?,?,?,?)",
                    (DESKTOP, 2, "oak1", GONE, '["blurry"]'))
    db.conn.commit()

    report = _merge(scene)

    assert [(a.step, a.target) for a in db.actions(DESKTOP)] == [(3, KEEP), (5, KEEP)]
    assert "  · 动作 1（第 5 步改成 cover.01）" in report.plan.summary_lines()
    assert set(db.frame_overrides(scene.key(2))) == {KEEP}
    assert db.get_conflict(cid)["instance"] == KEEP
    assert [tuple(r) for r in db.conn.execute(
        "SELECT instance FROM instance_frame_flags")] == [(KEEP,)]


def test_zorder_with_both_keeps_the_drawn_position(scene):
    db = scene.db
    from tda.core.model import ZOrderRec

    db.set_zorder(ZOrderRec(DESKTOP, "scan", 1, [(KEEP, "main"), (PSU, "main"), (GONE, "main"),
                                                 (KEEP, "lid")]))
    report = _merge(scene)
    assert db.zorder(DESKTOP, "scan", 1).order == [(PSU, "main"), (KEEP, "main"), (KEEP, "lid")]
    [entry] = report.plan.zorder
    assert entry["dropped"] == [(KEEP, "main")]
    assert "层级里 cover.01/main 原来的位置" in "\n".join(report.plan.summary_lines())
    assert _ops(db, OP_MERGE)[0]["payload"]["zorder_dropped"] == [
        {"view": "scan", "seg": 1, "entries": [[KEEP, "main"]]}]


def test_pair_overrides_are_renamed_and_self_pairs_dropped(scene):
    db = scene.db
    db.set_pair_override(PairOverride(DESKTOP, "scan", 1, GONE, PSU))
    db.set_pair_override(PairOverride(DESKTOP, "scan", 1, PSU, KEEP))      # contradicted
    db.set_pair_override(PairOverride(DESKTOP, "scan", 1, GONE, KEEP))     # becomes a self-pair
    report = _merge(scene)
    assert [(p.above, p.below) for p in db.pair_overrides(DESKTOP, "scan", 1)] == [(KEEP, PSU)]
    reasons = sorted(why for _row, why in report.plan.pairs_dropped)
    assert any("它自己" in w for w in reasons) and any("相反" in w for w in reasons)


def test_relations_are_re_pointed_and_deduplicated(scene):
    db = scene.db
    db.add_relation(DESKTOP, "blocked_by", GONE, KEEP, source="manual", mode="physical_path")
    db.add_relation(DESKTOP, "blocked_by", PSU, GONE, source="manual", mode="physical_path")
    report = _merge(scene)
    triples = sorted((r["type"], r["target"], r["blocker"], r["source"])
                     for r in db.relations(DESKTOP))
    assert triples == [("blocked_by", PSU, KEEP, "manual"),
                       ("covered_by", PSU, KEEP, "rule")]
    whys = sorted(why for _row, why in report.plan.relations_dropped)
    assert len(whys) == 2 and any("重复" in w for w in whys) and any("它自己" in w for w in whys)


def test_pointers_at_gone_follow_and_a_self_pointer_is_cleared(scene):
    db = scene.db
    psu = db.instances(DESKTOP)[PSU]
    psu.mounted_on = GONE
    db.upsert_instance(psu)
    keep = db.instances(DESKTOP)[KEEP]
    keep.parent, keep.attached = GONE, True
    db.upsert_instance(keep)
    _merge(scene)
    after = db.instances(DESKTOP)
    assert after[PSU].mounted_on == KEEP
    assert after[KEEP].parent is None and after[KEEP].attached is False


# --------------------------------------------------------------------------- #
# hand-written events: moved or dropped
# --------------------------------------------------------------------------- #
def test_hand_written_events_move_where_they_fit(db):
    """gone was re-closed by hand at 4 (T); keep has its own event at 6."""
    scene = build(db)
    db.add_events([
        StateEvent(DESKTOP, 4, GONE, "state", "open", "closed", "scan", auto=False),
        StateEvent(DESKTOP, 0, GONE, "state", "closed", "open", None, auto=False),
    ])
    recompile_events(db, DESKTOP, scene.tax)
    # gone's step-0 start makes its own "closed -> open at 3" wrong; drop that first
    db.conn.execute("DELETE FROM state_event WHERE target=? AND step=3 AND auto=0", (GONE,))
    db.conn.commit()
    recompile_events(db, DESKTOP, scene.tax)

    report = _merge(scene)

    moved = [(r["step"], r["old"], r["new"]) for r in report.plan.events_moved]
    assert moved == [(4, "open", "closed")]
    dropped = {r["step"]: why for r, why in report.plan.events_dropped}
    assert "初始状态跟着 cover.01" in dropped[0]
    assert _states(scene) == ["closed", "closed", "open", "closed", "removed", "removed"]
    manual = [(e.step, e.target) for e in db.events(DESKTOP) if not e.auto]
    assert manual == [(4, KEEP)]


def test_an_event_at_a_step_keep_already_wrote_is_dropped(db):
    scene = build(db)
    db.add_events([StateEvent(DESKTOP, 4, KEEP, "state", "open", "closed", "scan", auto=False),
                   StateEvent(DESKTOP, 4, GONE, "state", "open", "closed", "scan", auto=False)])
    recompile_events(db, DESKTOP, scene.tax)
    report = _merge(scene)
    dropped = {(r["step"], r["target"]): why for r, why in report.plan.events_dropped}
    assert "cover.01 在第 4 步已有自己的手写事件" in dropped[(4, GONE)]
    assert [(e.step, e.target) for e in db.events(DESKTOP) if not e.auto] == [(4, KEEP)]


# --------------------------------------------------------------------------- #
# settling: derived log, digests, re-checks
# --------------------------------------------------------------------------- #
def test_the_derived_log_digests_and_rechecks_are_settled(scene):
    db = scene.db
    for step in STEPS:
        assert db.frame_digest(scene.key(step)) is not None
    report = _merge(scene)
    assert all(e.target != GONE for e in db.events(DESKTOP))
    auto = [(e.step, e.attr, e.old, e.new) for e in db.events(DESKTOP)
            if e.auto and e.target == KEEP]
    assert auto == [(3, "state", "closed", "open"), (5, "state", "open", "removed"),
                    (5, "placement", "in_chassis", "on_bench")]
    # gone had rows on every scan frame it needed: 1-4
    for step in (1, 2, 3, 4):
        assert db.frame_digest(scene.key(step)) is None
    assert db.frame_digest(scene.key(6)) is not None
    assert db.rechecks(DESKTOP, "scan") == [3, 4, 5, 6]
    assert report.plan.rechecks == {"scan": [3, 4, 5, 6]}


# --------------------------------------------------------------------------- #
# undo / redo
# --------------------------------------------------------------------------- #
def test_undo_restores_every_table_exactly(scene):
    db = scene.db
    db.set_frame_override(FrameOverride(scene.key(2), GONE, visibility="occluded_partial"))
    db.set_pair_override(PairOverride(DESKTOP, "scan", 1, GONE, KEEP))
    db.add_relation(DESKTOP, "blocked_by", PSU, GONE, source="manual", mode="physical_path")
    psu = db.instances(DESKTOP)[PSU]
    psu.mounted_on = GONE
    db.upsert_instance(psu)
    restamp_graph(db, DESKTOP, scene.tax)   # the stamp describes the edge just added
    before = dump(db)

    report = _merge(scene)
    assert dump(db) != before
    undo_id = undo_merge(db, scene.tax, db.get_op(report.op_id), "tester")

    after = dump(db)
    for table in before:
        assert after[table] == before[table], table
    undo = db.get_op(undo_id)
    assert undo["kind"] == OP_UNDO and undo["payload"]["undid"] == report.op_id


def test_redo_gives_the_same_tables_as_the_merge(scene):
    db = scene.db
    first = _merge(scene)
    merged = dump(db)
    undo_id = undo_merge(db, scene.tax, db.get_op(first.op_id))
    again = redo_merge(db, scene.tax, db.get_op(undo_id), "tester")
    assert dump(db) == merged
    assert again.op_id != first.op_id
    # and the redo can be undone in turn
    undo_merge(db, scene.tax, db.get_op(again.op_id))
    assert GONE in db.instances(DESKTOP)


def test_undo_after_a_refresh_brings_the_frames_back(scene):
    db = scene.db
    report = _merge(scene)
    scene.refresh_all()
    undo_merge(db, scene.tax, db.get_op(report.op_id))
    totals = scene.refresh_all()
    assert totals["conflicts"] == 0
    assert db.compiled(scene.key(3))[GONE]["status"] == "verified"
    assert [kf.anchor_step for kf in db.keyframes(DESKTOP, "scan", GONE)] == [2, 4]


def test_undo_is_refused_while_gone_exists_again(scene):
    db = scene.db
    report = _merge(scene)
    db.upsert_instance(InstanceRec(GONE, DESKTOP, "cover"))
    with pytest.raises(MergeError, match="又存在了"):
        undo_merge(db, scene.tax, db.get_op(report.op_id))
