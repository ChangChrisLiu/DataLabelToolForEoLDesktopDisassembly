"""Task U21: 「其实是硬盘架」 / 「其实是卡扣」 -- a logged part reclassed, everything drawn kept.

The scene is D2's drive bay in miniature (``tests/reclass_scenes.py``): the
importer called the SSD cage ``drive_latch.02``; it is drawn and confirmed, the
locker's ``of`` names it, the drive is mounted on it, a manual edge ties them.
A reclass is a merge into a fresh ``drive_cage.01`` with the class check lifted,
and the U20 cage rule follows in the same op: the locker opened before the
cage comes out is hung on it (task U21 scope (b)), else the cage's implied
latch is made.
"""
from __future__ import annotations

import pytest

from merge_scenes import dump, everything
from reclass_scenes import (
    CAGE,
    DESKTOP,
    DRIVE,
    IMPLIED,
    LOCKER,
    SHIELD,
    STEPS,
    build,
)
from tda.core.db import Db
from tda.core.implied import CAGE_LATCH_RULE, is_cage_latch, is_implied
from tda.core.model import ActionRec, InstanceRec
from tda.core.reclass import (
    OP_RECLASS,
    OP_UNDO_RECLASS,
    ReclassError,
    fresh_key,
    plan_reclass,
    reclass_gate,
    reclass_instance,
    redo_reclass,
    undo_reclass,
)
from tda.core.reclass_replay import reclassed_record
from tda.core.state_assert import assert_state
from tda.core.truth_inputs import state_of


@pytest.fixture
def removed(tmp_path):
    """The locker said with Shift+T to be taken out at 3: the cage gets an implied latch."""
    db = Db(str(tmp_path / "tda.sqlite"))
    yield build(db, locker_verb="remove")
    db.close()


@pytest.fixture
def opened(tmp_path):
    """The locker only opened at 3 (after the reverse Shift+T): the cage adopts it."""
    db = Db(str(tmp_path / "tda.sqlite"))
    yield build(db, locker_verb="open")
    db.close()


def states(scene, key, steps=STEPS):
    out = []
    for step in steps:
        held = state_of(scene.db, scene.tax, DESKTOP, step).get(key)
        out.append(None if held is None else held.state)
    return out


def rows_of(db, key):
    """The confirmed rows of ``key``, every column but the instance, by frame."""
    cur = db.conn.execute("SELECT * FROM compiled_mask WHERE desktop=? AND instance=? "
                          "AND status='verified' ORDER BY view, step", (DESKTOP, key))
    names = [d[0] for d in cur.description]
    return [tuple(v for n, v in zip(names, row) if n != "instance") for row in cur.fetchall()]


# --------------------------------------------------------------------------- #
# latch -> cage, the locker taken out: the implied latch
# --------------------------------------------------------------------------- #
def test_everything_moves_to_the_cage(removed):
    db, tax = removed.db, removed.tax
    signed = rows_of(db, SHIELD)
    keyframes = {v: sorted(kf.id for kf in db.keyframes(DESKTOP, v, SHIELD))
                 for v in ("scan", "oak1")}
    report = reclass_instance(db, tax, DESKTOP, SHIELD, annotator="tester")
    plan = report.plan
    assert (plan.old, plan.new, plan.old_cls, plan.new_cls) == (SHIELD, CAGE, "drive_latch",
                                                                "drive_cage")
    inst = db.instances(DESKTOP)
    assert SHIELD not in inst
    cage = inst[CAGE]
    assert cage.cls == "drive_cage" and cage.raw_names == ["remove SSD locker shield"]
    assert cage.attrs == {"of": "ssd", "qualifier": "x"}           # of and the rest carried
    # the shapes (ids unchanged), the confirmed rows byte for byte but the key
    assert {v: sorted(kf.id for kf in db.keyframes(DESKTOP, v, CAGE))
            for v in ("scan", "oak1")} == keyframes
    assert rows_of(db, CAGE) == signed and len(signed) == 5
    # the action, the verb unchanged; the pointers and the of; the edge; z-order
    assert [(a.step, a.target, a.verb) for a in db.actions(DESKTOP) if a.step == 5] == \
        [(5, CAGE, "remove")]
    assert inst[LOCKER].attrs["of"] == CAGE and inst[DRIVE].mounted_on == CAGE
    assert [(r["type"], r["target"], r["blocker"]) for r in db.relations(DESKTOP)] == \
        [("locked_by", CAGE, LOCKER)]
    assert db.zorder(DESKTOP, "scan", 1).order[0] == (CAGE, "main")
    # the T on the shield at 3 (closed -> open): a cage has no open
    dropped = [why for row, why in plan.merge.events_dropped]
    assert dropped == ["drive_cage 没有「open」这个状态"]
    assert not [e for e in db.events(DESKTOP) if not e.auto and e.target in (SHIELD, CAGE)]
    assert states(removed, CAGE) == ["installed"] * 4 + ["removed"] * 4
    # one op row
    [op] = db.ops_of_kind(DESKTOP, OP_RECLASS)
    assert op["payload"]["steps"] == [5] and op["payload"]["verbs"] == [[5, "remove"]]
    assert op["payload"]["raw_names"] == ["remove SSD locker shield"]


def test_the_cage_rule_gives_the_cage_its_latch(removed):
    db, tax = removed.db, removed.tax
    report = reclass_instance(db, tax, DESKTOP, SHIELD, annotator="tester")
    latch = db.instances(DESKTOP)[IMPLIED]          # never the shield's old key .02
    assert is_cage_latch(latch) and is_implied(latch)
    assert (latch.parent, latch.attached) == (CAGE, True)
    assert latch.attrs["implied_rule"] == CAGE_LATCH_RULE and latch.attrs["of"] == "ssd"
    assert states(removed, IMPLIED) == ["closed"] * 3 + ["open"] + ["removed"] * 4
    assert states(removed, LOCKER)[:3] == ["closed", "closed", "removed"]
    assert db.instances(DESKTOP)[LOCKER].parent is None
    # the frames that gain it are predicted, queued, and demoted by the re-check
    assert report.plan.demoted() == {"oak1": [4], "scan": [1, 2, 3, 4]}
    assert set(db.rechecks(DESKTOP, "scan")) >= {1, 2, 3, 4}
    out = removed.refresh_all()
    assert out["conflicts"] == 0                    # the confirmed rows are the cage's
    assert removed.statuses() == {s: "needs_review" if s <= 4 else "verified" for s in STEPS}
    assert removed.status(4, "oak1") == "needs_review"
    assert removed.status(5, "oak1") == "verified"
    line = report.status_line()
    assert line.startswith("drive_latch.02 其实是硬盘架：现在是 drive_cage.01（形状 4、已确认的行 5、"
                           "动作 1 搬过去）")
    assert "机架 drive_cage.01 因此自动得到隐含卡扣 drive_latch.03（关着，默认第 4 帧打开，" \
           "第 5 步随机架拆下）" in line
    assert line.endswith("其中 5 帧会退回待复核")


def test_the_confirmation_says_it_first_and_writes_nothing(removed):
    db, tax = removed.db, removed.tax
    before = everything(db)
    plan = plan_reclass(db, tax, DESKTOP, SHIELD)
    assert everything(db) == before                 # a rehearsal, rolled back
    lines = plan.summary_lines()
    assert lines[0] == ("把 drive_latch.02（remove SSD locker shield） 改成硬盘架 drive_cage.01："
                        "形状、确认过的行、动作、关系和指向它的字段都搬过去，动作不变（第 5 步「remove」）。")
    assert "  · 已确认的行 5 行（oak1 第 4 帧；scan 第 1–4 帧）" in lines
    assert "  · 手写状态事件 drive_latch.02 第 3 步 closed→open：drive_cage 没有「open」这个状态" \
        in lines
    assert ("  · 机架 drive_cage.01 自动得到隐含卡扣 drive_latch.03（关着，默认第 4 帧打开，"
            "第 5 步随机架拆下）") in lines
    assert any(line.endswith("其中 5 帧会退回待复核（oak1 第 4 帧；scan 第 1–4 帧：多了 "
                             "drive_latch.03）。") for line in lines)
    assert lines[-1] == "立即写入，不经 Apply（标注模式里 Ctrl+Z 撤销）。"


# --------------------------------------------------------------------------- #
# latch -> cage, the locker only opened: adopted (scope (b))
# --------------------------------------------------------------------------- #
def test_the_open_locker_is_hung_on_the_new_cage(opened):
    db, tax = opened.db, opened.tax
    report = reclass_instance(db, tax, DESKTOP, SHIELD, annotator="tester")
    inst = db.instances(DESKTOP)
    assert (inst[LOCKER].parent, inst[LOCKER].attached) == (CAGE, True)
    assert not [k for k, r in inst.items() if is_implied(r)]          # nothing implied
    assert states(opened, LOCKER) == ["closed"] * 2 + ["open"] * 2 + ["removed"] * 4
    assert report.plan.cage["adopted"] == [LOCKER]
    assert "drive_latch.01 已挂到 drive_cage.01 上，随机架一起拆下（第 5 步）" in report.status_line()
    # it now leaves with the cage: the frames that had it on 5-8 lose it
    assert report.plan.demoted() == {"oak1": [5, 6, 7, 8], "scan": [5, 6, 7, 8]}
    opened.refresh_all()
    assert opened.statuses() == {s: "needs_review" if s >= 5 else "verified" for s in STEPS}


def test_the_prediction_reads_the_confirmed_rows_not_a_pending_edit(removed):
    """U21 review, M1: D2's §7 sequence -- the reverse Shift+T, no re-check, then the reclass.

    The reverse Shift+T leaves the locker open from 3 with nothing to hang on
    (no cage yet) and queues 4-8 for a re-check that nobody runs. The reclass
    then hangs it on the cage, out at 5. What ``check`` will demote is what
    the confirmed rows no longer match: frame 4 in both views (3 went at once,
    with the Shift+T's own compile) -- not 5-8, which "the parts needed just
    before" would have said.
    """
    db, tax = removed.db, removed.tax
    from tda.core.model import FrameKey
    from tda.ui.session_verb import commit_verb_fix

    commit_verb_fix(db, removed.svc, FrameKey(DESKTOP, 3, "scan"), LOCKER, "open", "tester")
    assert removed.status(3) == "needs_review" and 4 in db.rechecks(DESKTOP, "scan")
    plan = plan_reclass(db, tax, DESKTOP, SHIELD)
    assert plan.cage["adopted"] == [LOCKER]
    assert plan.demoted() == {"oak1": [4], "scan": [4]}
    assert plan.pending == {"scan": [4]}
    line = next(x for x in plan.summary_lines() if "退回待复核" in x)
    assert "其中 2 帧会退回待复核（oak1 第 4 帧；scan 第 4 帧：多了 drive_latch.01）" in line
    assert "这些帧里有 1 帧在这之前就在等复检" in line
    reclass_instance(db, tax, DESKTOP, SHIELD, annotator="tester")
    removed.refresh_all()
    assert removed.statuses() == {1: "verified", 2: "verified", 3: "needs_review",
                                  4: "needs_review", 5: "verified", 6: "verified",
                                  7: "verified", 8: "verified"}
    assert [removed.status(s, "oak1") for s in (4, 5, 6, 7, 8)] == \
        ["needs_review", "verified", "verified", "verified", "verified"]


# --------------------------------------------------------------------------- #
# undo / redo
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture", ["removed", "opened"])
def test_undo_and_redo_are_table_identical(request, fixture):
    scene = request.getfixturevalue(fixture)
    db, tax = scene.db, scene.tax
    before = dump(db)
    report = reclass_instance(db, tax, DESKTOP, SHIELD, annotator="tester")
    after = dump(db)
    undo_id = undo_reclass(db, tax, db.get_op(report.op_id))
    assert dump(db) == before
    again = redo_reclass(db, tax, db.get_op(undo_id))
    assert dump(db) == after
    assert again.plan.new == CAGE
    # the redo's own row can be undone too
    undo_reclass(db, tax, db.get_op(again.op_id))
    assert dump(db) == before
    assert [op["kind"] for op in db.ops_of_kind(DESKTOP, OP_UNDO_RECLASS)] == \
        [OP_UNDO_RECLASS] * 2


def test_undo_is_refused_once_the_implied_latch_has_work(removed):
    db, tax = removed.db, removed.tax
    report = reclass_instance(db, tax, DESKTOP, SHIELD, annotator="tester")
    assert_state(db, tax, DESKTOP, IMPLIED, 2, "open")
    after = everything(db)
    with pytest.raises(ReclassError, match="手设过状态"):
        undo_reclass(db, tax, db.get_op(report.op_id))
    assert everything(db) == after


def test_undo_is_refused_once_something_was_put_on_the_new_key(opened):
    db, tax = opened.db, opened.tax
    report = reclass_instance(db, tax, DESKTOP, SHIELD, annotator="tester")
    db.add_relation(DESKTOP, "covered_by", DRIVE, CAGE, source="manual", reason="later")
    with pytest.raises(ReclassError, match="改类别后 drive_cage.01 上又有了修改"):
        undo_reclass(db, tax, db.get_op(report.op_id))
    assert CAGE in db.instances(DESKTOP)


# --------------------------------------------------------------------------- #
# cage -> latch
# --------------------------------------------------------------------------- #
def test_back_to_a_latch_drops_the_untouched_implied_latch(removed):
    db, tax = removed.db, removed.tax
    reclass_instance(db, tax, DESKTOP, SHIELD, annotator="tester")
    signed = rows_of(db, CAGE)
    back = reclass_instance(db, tax, DESKTOP, CAGE, annotator="tester")
    latch = back.plan.new
    assert latch == "drive_latch.04"         # .02 was the shield, .03 its implied latch
    inst = db.instances(DESKTOP)
    assert CAGE not in inst and IMPLIED not in inst
    assert inst[latch].cls == "drive_latch" and inst[latch].attrs["of"] == "ssd"
    assert rows_of(db, latch) == signed
    assert [r["key"] for r in back.plan.cage["drop"]] == [IMPLIED]
    assert "drive_cage.01 的隐含卡扣 drive_latch.03 用不着了，已删掉（没画过）" in back.status_line()
    assert states(removed, latch) == ["closed"] * 4 + ["removed"] * 4


def test_back_to_a_latch_unhangs_the_locker(opened):
    db, tax = opened.db, opened.tax
    reclass_instance(db, tax, DESKTOP, SHIELD, annotator="tester")
    back = reclass_instance(db, tax, DESKTOP, CAGE, annotator="tester")
    locker = db.instances(DESKTOP)[LOCKER]
    assert (locker.parent, locker.attached) == (None, False)
    assert locker.attrs["of"] == back.plan.new          # its of followed the merge
    assert "drive_latch.01 已不再挂在 drive_cage.01 上（parent 清空）" in back.status_line()


def test_back_to_a_latch_is_refused_while_the_implied_latch_has_work(removed):
    db, tax = removed.db, removed.tax
    reclass_instance(db, tax, DESKTOP, SHIELD, annotator="tester")
    assert_state(db, tax, DESKTOP, IMPLIED, 2, "open")
    why = reclass_gate(db, tax, DESKTOP, CAGE)
    assert why.startswith("drive_cage.01 的隐含卡扣 drive_latch.03 已经画过形状或手设过状态")
    assert "先到 Steps 右键删掉 drive_latch.03" in why
    with pytest.raises(ReclassError, match="卡扣的卡扣"):
        reclass_instance(db, tax, DESKTOP, CAGE)


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_refusals(removed):
    db, tax = removed.db, removed.tax
    db.upsert_instance(InstanceRec("ls:Shield#1", DESKTOP, "drive_latch"))
    db.upsert_instance(InstanceRec("ctx.misc_part.01", DESKTOP, "misc_part",
                                   attrs={"context": True}))
    assert "Label Studio 草稿" in reclass_gate(db, tax, DESKTOP, "ls:Shield#1")
    assert "框外零件" in reclass_gate(db, tax, DESKTOP, "ctx.misc_part.01")
    assert "不是硬盘架或它的卡扣" in reclass_gate(db, tax, DESKTOP, DRIVE)
    assert "没有零件" in reclass_gate(db, tax, DESKTOP, "drive_latch.09")
    assert "只能改成 drive_cage" in reclass_gate(db, tax, DESKTOP, SHIELD, "drive_latch")
    # an action the new class does not take: a latch only opened is no cage
    assert reclass_gate(db, tax, DESKTOP, LOCKER) == ""             # removed at 3: fine
    actions = [a if a.target != LOCKER else ActionRec(DESKTOP, 3, 0, LOCKER, "open")
               for a in db.actions(DESKTOP)]
    db.replace_steps(DESKTOP, db.steps(DESKTOP), actions)
    why = reclass_gate(db, tax, DESKTOP, LOCKER)
    assert why == ("第 3 步 drive_latch.01 是「打开」：硬盘架不会只是打开。它其实拆下来了就先按 "
                   "Shift+T（其实拆下来了），再改类别")
    db.replace_steps(DESKTOP, db.steps(DESKTOP),
                     [a if a.target != LOCKER else ActionRec(DESKTOP, 3, 0, LOCKER, "remove")
                      for a in db.actions(DESKTOP)])
    reclass_instance(db, tax, DESKTOP, SHIELD)
    # a cage displaced is no latch
    actions = [a if a.target != CAGE else ActionRec(DESKTOP, 5, 0, CAGE, "displace")
               for a in db.actions(DESKTOP)]
    db.replace_steps(DESKTOP, db.steps(DESKTOP), actions)
    assert "drive_latch 没有这个动作" in reclass_gate(db, tax, DESKTOP, CAGE)
    before = everything(db)
    with pytest.raises(ReclassError):
        reclass_instance(db, tax, DESKTOP, CAGE)
    assert everything(db) == before


def test_an_implied_latch_is_deleted_not_reclassed(removed):
    db, tax = removed.db, removed.tax
    reclass_instance(db, tax, DESKTOP, SHIELD)
    why = reclass_gate(db, tax, DESKTOP, IMPLIED)
    assert why.startswith("drive_latch.03 是隐含零件（日志里没有）：不用改类别，不要就到 Steps 右键删除")


def test_the_rehearsal_is_refused_inside_another_transaction(removed):
    with removed.db.transaction():
        with pytest.raises(ReclassError, match="另一个事务"):
            plan_reclass(removed.db, removed.tax, DESKTOP, SHIELD)


# --------------------------------------------------------------------------- #
# keys and the record rule
# --------------------------------------------------------------------------- #
def test_a_fresh_key_skips_every_key_the_op_log_names(removed):
    db = removed.db
    assert fresh_key(db, DESKTOP, "drive_cage") == "drive_cage.01"
    db.log_op(DESKTOP, "-", "merge_instances", {"gone": "drive_cage.04"}, {}, "x")
    assert fresh_key(db, DESKTOP, "drive_cage") == "drive_cage.05"
    assert fresh_key(db, DESKTOP, "drive_latch") == "drive_latch.03"
    assert fresh_key(db, DESKTOP, "drive_latch", also={"drive_latch.07"}) == "drive_latch.08"


def test_the_record_rule(removed):
    tax = removed.tax
    latch = InstanceRec("drive_latch.05", DESKTOP, "drive_latch",
                        attrs={"of": "drive_cage.02", "added_by": "tester", "note": "n"},
                        parent="drive_cage.02", attached=True, mounted_on="chassis",
                        fastens="x", removal_direction="up", raw_names=["HDD Shield Locker"])
    made, left = reclassed_record(latch, "drive_cage", "drive_cage.07", tax)
    assert (made.key, made.cls) == ("drive_cage.07", "drive_cage")
    assert made.attrs == {"added_by": "tester", "note": "n"}       # a key is no cage's of
    assert (made.parent, made.attached, made.mounted_on, made.fastens) == \
        (None, False, "chassis", None)
    assert made.removal_direction == "up" and made.raw_names == ["HDD Shield Locker"]
    assert left == {"attrs": {"of": "drive_cage.02"},
                    "columns": {"fastens": "x", "parent": "drive_cage.02"}}
    cage = InstanceRec("drive_cage.07", DESKTOP, "drive_cage", attrs={"of": "hdd"})
    back, left = reclassed_record(cage, "drive_latch", "drive_latch.06", tax)
    assert back.attrs == {"of": "hdd"} and left == {"attrs": {}, "columns": {}}
