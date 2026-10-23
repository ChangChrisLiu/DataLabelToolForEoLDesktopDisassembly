"""Task U21 round 2: 「合并到…」 with ``overlap="keep"`` -- D5's locker and its hand-added twin.

The live D5: the "HDD shield locker" ``drive_latch.01`` was said with Shift+T to
be taken out at 10, so it was drawn closed on frames 1-9 (one keyframe at 9);
the cage's latch was then added by hand as ``drive_latch.02`` (parent the
cage), set open at 13 with ``T`` and drawn there (one keyframe at 13, which
reaches back to frame 1). Frames 1-13 are confirmed. The user: they are one
blue tab. The reverse Shift+T hangs ``.01`` on the cage, open from 10; merging
``.02`` into it is refused (both drawn and confirmed on 1-9) unless
``overlap="keep"``: ``.01``'s closed shape stays on 1-9, ``.02``'s open one
moves over for 10-13.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from merge_scenes import dump, everything
from reclass_scenes import mask_kf, rect
from tda.core import masks
from tda.core.db import Db
from tda.core.extra import recompile_events, restamp_graph
from tda.core.merge_instances import (
    OP_MERGE,
    OVERLAP_KEEP,
    OverlapError,
    merge_instances,
    plan_merge,
    redo_merge,
    undo_merge,
)
from tda.core.model import ActionRec, FrameKey, InstanceRec, StepRec
from tda.core.state_assert import assert_state
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import state_of
from tda.ui import steps_merge as M
from tda.ui.session_verb import commit_verb_fix
from tda.ui.steps_model import StepTableData

D = 5
VIEW = "scan"
STEPS = tuple(range(1, 16))
CAGE, LOCKER, LATCH = "drive_cage.01", "drive_latch.01", "drive_latch.02"
DRIVE = "storage_drive.hdd.01"
A, B = (44, 4, 52, 12), (44, 14, 52, 22)        # the tab closed / open


def kf(key, box, anchor):
    made = mask_kf(key, box, anchor, VIEW)
    made.desktop = D
    return made


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def d5(tmp_path, tax):
    """The live D5 after the reverse Shift+T on the locker (step 1 of the sequence)."""
    db = Db(str(tmp_path / "tda.sqlite"))
    db.upsert_desktop(D, {"brand": "Dell"})
    for rec in (InstanceRec(CAGE, D, "drive_cage", attrs={"of": "hdd"}, raw_names=["HDD shield"]),
                InstanceRec(LOCKER, D, "drive_latch", attrs={"of": "hdd"},
                            raw_names=["HDD shield locker"]),
                InstanceRec(DRIVE, D, "storage_drive", attrs={"kind": "hdd"}, raw_names=["HDD"])):
        db.upsert_instance(rec)
    db.replace_steps(D, [StepRec(D, s, "normal", f"step {s}") for s in STEPS],
                     [ActionRec(D, 10, 0, LOCKER, "remove"),     # Shift+T, before U21
                      ActionRec(D, 13, 0, DRIVE, "remove"),
                      ActionRec(D, 14, 0, CAGE, "remove")])
    recompile_events(db, D, tax)
    data = StepTableData.load(db, D, tax)
    assert data.add_extras("drive_latch", parent=CAGE, attrs={"of": "hdd"}) == [LATCH]
    data.save(db)
    assert_state(db, tax, D, LATCH, 13, "open")                  # T at 13
    db.set_pose_segment(D, VIEW, 1, 1, 15, 15, None, None)
    for step in STEPS:
        db.upsert_frame(FrameKey(D, step, VIEW), f"{VIEW}_{step:03d}.jpg", {"hw": [64, 64]}, None)
    db.add_keyframe(kf(LOCKER, A, 9))                            # closed, 1-9
    db.add_keyframe(kf(LATCH, B, 13))                            # open, reaches back to 1
    db.add_keyframe(kf(CAGE, (4, 4, 40, 40), 13))
    db.add_keyframe(kf(DRIVE, (10, 44, 30, 60), 12))
    svc = TruthService(db, tax, cache_dir=None)
    svc.refresh_range(D, VIEW, list(STEPS))
    for step in range(1, 14):
        svc.verify_frame(FrameKey(D, step, VIEW), "tester")
    # the controller's step 1: the locker was only opened -- hung on the cage
    result = commit_verb_fix(db, svc, FrameKey(D, 14, VIEW), LOCKER, "open", "controller")
    assert result["cage"]["adopted"] == [LOCKER] and result["cage"]["also"] == [LATCH]
    restamp_graph(db, D, tax)          # what the merge re-stamps, so undo compares like
    db.svc = svc
    yield db
    db.close()


def rows(db, key):
    cur = db.conn.execute("SELECT * FROM compiled_mask WHERE desktop=? AND instance=? "
                          "AND status='verified' ORDER BY step", (D, key))
    names = [c[0] for c in cur.description]
    return {r[names.index("step")]: tuple(v for n, v in zip(names, r) if n != "instance")
            for r in cur.fetchall()}


def shape_at(db, tax, key, step):
    """The compiled visible pixels of ``key`` at ``step`` (refreshed), as a box."""
    import json

    [(text,)] = db.conn.execute("SELECT visible_rle_json FROM compiled_mask WHERE desktop=? "
                                "AND step=? AND view=? AND instance=?",
                                (D, step, VIEW, key)).fetchall()
    mask = masks.decode_rle(json.loads(text))
    ys, xs = mask.nonzero()
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def test_the_plain_merge_is_refused_for_the_overlap(d5, tax):
    with pytest.raises(OverlapError, match="scan 第 1–9 帧"):
        plan_merge(d5, tax, D, LATCH, LOCKER)


def test_overlap_keep_keeps_the_locker_on_1_9_and_moves_the_open_tab_onto_10_13(d5, tax):
    locker_rows, latch_rows = rows(d5, LOCKER), rows(d5, LATCH)
    assert sorted(locker_rows) == list(range(1, 10)) and sorted(latch_rows) == list(range(1, 14))
    report = merge_instances(d5, tax, D, LATCH, LOCKER, "controller", overlap=OVERLAP_KEEP)
    plan = report.plan
    assert plan.overlap_frames == {VIEW: list(range(1, 10))}
    assert plan.keyframes_dropped == []              # .02's shape at 13 is not the one on 1-9
    assert plan.verified_dropped == 9
    assert [a for _i, a in plan.keyframes[VIEW]] == [13]
    # the hand-added latch's T at 13 is redundant now: the locker is open from 10
    whys = {(r["step"], r["new"]): why for r, why in plan.events_dropped}
    assert whys[(13, "open")] == "合并后第 13 步它已经是 open，这条多余"
    assert not [e for e in d5.events(D) if not e.auto and e.target in (LATCH, LOCKER)
                and e.step == 13]
    # confirmed rows: the locker's on 1-9 untouched, the tab's on 10-13 renamed
    after = rows(d5, LOCKER)
    assert {s: after[s] for s in range(1, 10)} == locker_rows
    assert {s: after[s] for s in range(10, 14)} == {s: latch_rows[s] for s in range(10, 14)}
    assert LATCH not in d5.instances(D) and not rows(d5, LATCH)
    # recompiled: closed on 1-9, open on 10-13, nothing disagrees, nothing demoted
    out = d5.svc.refresh_range(D, VIEW, list(STEPS), ignore_digest=True)
    assert out["conflicts"] == 0
    assert [d5.get_frame(FrameKey(D, s, VIEW))["review_status"] for s in range(1, 14)] == \
        ["verified"] * 13
    assert shape_at(d5, tax, LOCKER, 5) == A and shape_at(d5, tax, LOCKER, 11) == B
    assert [state_of(d5, tax, D, s)[LOCKER].state for s in (9, 10, 13, 14)] == \
        ["closed", "open", "open", "removed"]
    line = report.status_line()
    assert "重叠处（scan 第 1–9 帧）保留 drive_latch.01 的，丢掉 drive_latch.02 的已确认的行 9" in line
    assert [op["payload"]["overlap"] for op in d5.ops_of_kind(D, OP_MERGE)] == [OVERLAP_KEEP]


def test_undo_is_table_identical_and_redo_does_it_again(d5, tax):
    before = dump(d5)
    report = merge_instances(d5, tax, D, LATCH, LOCKER, "controller", overlap=OVERLAP_KEEP)
    after = dump(d5)
    undone = undo_merge(d5, tax, d5.get_op(report.op_id))
    assert dump(d5) == before
    redo_merge(d5, tax, d5.get_op(undone))
    assert dump(d5) == after


def test_a_keyframe_of_gone_on_an_overlap_frame_goes_and_comes_back(d5, tax):
    """A shape of .02 anchored at 5 would win on 1-5 over the locker's at 9: it goes."""
    d5.add_keyframe(kf(LATCH, (30, 30, 36, 36), 5))
    before = dump(d5)
    report = merge_instances(d5, tax, D, LATCH, LOCKER, "controller", overlap=OVERLAP_KEEP)
    assert [r["anchor_step"] for r in report.plan.keyframes_dropped] == [5]
    assert report.plan.parts_dropped
    assert sorted(kf_.anchor_step for kf_ in d5.keyframes(D, VIEW, LOCKER)) == [9, 13]
    undo_merge(d5, tax, d5.get_op(report.op_id))
    assert dump(d5) == before


def test_a_frame_override_of_gone_there_goes_too(d5, tax):
    from tda.core.model import FrameOverride

    d5.set_frame_override(FrameOverride(FrameKey(D, 3, VIEW), LATCH,
                                        visibility="occluded_partial"))
    d5.set_frame_override(FrameOverride(FrameKey(D, 3, VIEW), LOCKER, visibility="visible"))
    before = everything(d5)
    with pytest.raises(OverlapError):
        plan_merge(d5, tax, D, LATCH, LOCKER)
    assert everything(d5) == before
    before = dump(d5)
    report = merge_instances(d5, tax, D, LATCH, LOCKER, "controller", overlap=OVERLAP_KEEP)
    assert [(r["step"], r["instance"]) for r in report.plan.overrides_dropped] == [(3, LATCH)]
    assert d5.frame_overrides(FrameKey(D, 3, VIEW))[LOCKER].visibility == "visible"
    undo_merge(d5, tax, d5.get_op(report.op_id))
    assert dump(d5) == before


# --------------------------------------------------------------------------- #
# the S1 confirmation's second button
# --------------------------------------------------------------------------- #
@pytest.fixture
def panel(d5, tax, tmp_path):
    from tda.ui.panels.steptable import StepTablePanel

    QApplication.instance() or QApplication([])
    widget = StepTablePanel(d5, D, taxonomy=tax, cache_dir=tmp_path / "cache")
    yield widget
    widget.deleteLater()


def test_the_confirmation_offers_keep_only_for_the_overlap(panel, d5, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(QInputDialog, "getItem",
                        lambda _p, _t, _l, items, *_a, **_k: (
                            next(i for i in items if i.startswith(LOCKER)), True))
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *_a, **_k: pytest.fail("the plain confirmation is not shown"))

    def ask(gone, keep, refusal, lines):
        seen.update(gone=gone, keep=keep, refusal=refusal, lines=lines)
        return True

    panel.ask_merge_overlap = ask
    panel.prompt_merge(LATCH)
    assert (seen["gone"], seen["keep"]) == (LATCH, LOCKER)
    # the plain path's refusal, in U12's words, comes first
    assert seen["refusal"] == ("drive_latch.02 和 drive_latch.01 在同一帧上各有一行已确认的结果"
                               "（scan 第 1–9 帧）：是两个零件，不能合并")
    assert seen["lines"][0] == ("把 drive_latch.02 合并到 drive_latch.01（HDD shield locker）"
                                "（重叠处保留 drive_latch.01）：")
    assert ("  · 重叠处（scan 第 1–9 帧）保留 drive_latch.01 的，drive_latch.02 在这些帧上的"
            "已确认的行 9 行不要了") in seen["lines"]
    assert LATCH not in d5.instances(D)
    assert panel.status.text().startswith("已把 drive_latch.02 合并到 drive_latch.01")
    assert M.KEEP_BUTTON.format(keep=LOCKER) == "合并，重叠处保留 drive_latch.01"


def test_the_second_button_is_the_only_way_on(panel, d5, monkeypatch):
    """The real dialog: 「合并，重叠处保留 drive_latch.01」 merges, Cancel does not."""
    def click(text):
        def run(box):
            [button] = [b for b in box.buttons() if b.text() == text]
            button.click()
            return 0
        return run

    monkeypatch.setattr(QMessageBox, "exec", click("合并，重叠处保留 drive_latch.01"))
    assert panel.ask_merge_overlap(LATCH, LOCKER, "refused", ["line"]) is True
    monkeypatch.setattr(QMessageBox, "exec",
                        lambda box: (next(b for b in box.buttons()
                                          if box.buttonRole(b) == QMessageBox.RejectRole).click(),
                                     0)[1])
    assert panel.ask_merge_overlap(LATCH, LOCKER, "refused", ["line"]) is False
