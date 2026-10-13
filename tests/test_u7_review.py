"""Task U7 in Review and while drawing: ``Shift+N`` (C) and the re-trace warning (E).

On D13/scan a deliberate re-trace of four confirmed clips put 145 conflicts in
the queue, every one of them answered by ``accept_new``, and ``N`` answered one
at a time.  ``Shift+N`` lists the view's conflicts per instance and accepts the
ticked ones through the session -- the path of ``N`` -- until none of those
rows is open.  And the re-trace now says, before the commit, how many confirmed
frames it will put in conflict.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialogButtonBox

import app_scene
import session_scene
from session_scene import CHASSIS, COOLER, DESKTOP, VIEW, cell, draw, rect, seed_shapes
from tda.core import masks
from tda.core.model import FrameKey
from tda.core.truth_conflicts import row_values
from tda.ui import app_actions as A
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.panels.bulk_accept import BulkAcceptDialog, group_conflicts


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(qapp, tmp_path: Path):
    made = session_scene.make_session(tmp_path)
    yield made
    made.close(force=True)


def _open(session) -> list[dict]:
    return session.db.conflicts(DESKTOP, VIEW, open_only=True)


def _two_confirmed_frames(session) -> None:
    """Frames 13 and 12 confirmed with every part drawn (anchored at 14)."""
    session.goto(12)
    seed_shapes(session, 12)
    session.goto(13)
    assert session.confirm_frame() is True        # steps back to 12
    assert session.current().step == 12
    assert session.confirm_frame() is True


def _retrace(session, mask, step: int = 10) -> dict:
    session.goto(step)
    return draw(session, CHASSIS, mask, api.SCOPE_KEYFRAME)


# --------------------------------------------------------------------------- #
# the dialog's model
# --------------------------------------------------------------------------- #
def test_conflicts_are_grouped_per_instance_with_counts_and_frame_ranges():
    queue = ([{"step": s, "instance": "ram_latch.01"} for s in range(17, 42)]
             + [{"step": s, "instance": "motherboard.01"} for s in range(17, 42)]
             + [{"step": 41, "instance": "motherboard.01"}]      # a pre-U7 duplicate
             + [{"step": 40, "instance": "ram_latch.03"}, {"step": 38, "instance": "ram_latch.03"},
                {"step": 41, "instance": "ram_latch.03"}])

    groups = group_conflicts(queue)

    assert [g.instance for g in groups] == ["motherboard.01", "ram_latch.01", "ram_latch.03"]
    assert [g.count for g in groups] == [25, 25, 3]
    assert groups[1].text == "ram_latch.01 ×25（帧 17–41）"
    assert groups[0].text == "motherboard.01 ×25（帧 17–41）"
    assert groups[2].text == "ram_latch.03 ×3（帧 38、40、41）"
    assert group_conflicts([]) == []


def test_the_dialog_ticks_every_instance_and_counts_what_is_ticked(qapp):
    groups = group_conflicts([{"step": s, "instance": i}
                              for i in ("a.01", "b.01") for s in (3, 4, 5)])
    dialog = BulkAcceptDialog(groups)
    ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert dialog.selected_instances() == ["a.01", "b.01"]
    assert dialog.selected_count() == 6 and "6" in ok.text() and ok.isEnabled()

    dialog.item("a.01").setCheckState(Qt.CheckState.Unchecked)
    assert dialog.selected_instances() == ["b.01"]
    assert "3" in ok.text()
    dialog.item("b.01").setCheckState(Qt.CheckState.Unchecked)
    assert not ok.isEnabled()
    dialog.deleteLater()


# --------------------------------------------------------------------------- #
# the session's loop
# --------------------------------------------------------------------------- #
def test_bulk_accept_closes_stacked_and_stale_conflicts(session):
    """A pre-U7 stack on one row, and conflicts the inputs moved past: all closed."""
    session.sweeper_enabled = False                    # deterministic: nothing re-checks
    _two_confirmed_frames(session)
    _retrace(session, rect(2, 34, 34, 62))
    for step in (12, 13):                               # what the sweeper would raise
        session.truth.refresh(FrameKey(DESKTOP, step, VIEW))
    assert sorted(c["step"] for c in _open(session)) == [12, 13]
    first = {c["step"]: c for c in _open(session)}
    # a stack from before U7 on frame 12's row ...
    session.db.add_conflict(FrameKey(DESKTOP, 12, VIEW), CHASSIS, first[12]["old_rle"],
                            masks.encode_rle(rect(2, 30, 30, 62)), 500)
    # ... and the inputs move on before anybody answers (both now stale)
    _retrace(session, rect(4, 4, 40, 40))
    assert len(_open(session)) == 3

    result = session.accept_new_bulk([CHASSIS])

    assert result["open"] == 0 and _open(session) == []
    assert result["rows"] == 2 and result["superseded"] >= 2 and result["accepted"] == 2
    assert 2 <= result["rounds"] <= 6
    # the sweeper is off: the re-checks the re-traces queued are still owed (round 2)
    assert result["pending"] == len(session.db.rechecks(DESKTOP, VIEW)) == 2
    # the frozen rows carry what the inputs say now, and nothing stands any more
    for step in (12, 13):
        key = FrameKey(DESKTOP, step, VIEW)
        row = session.db.compiled(key)[CHASSIS]
        now = row_values(session.truth.compile(key).instances[CHASSIS]).visible_rle
        assert row["status"] == "verified" and row["verified_by"] == "tester"
        assert row["visible_rle"]["counts"] == now["counts"]
    again = session.truth.refresh_range(DESKTOP, VIEW, session.available_steps(),
                                        ignore_digest=True)
    assert again["standing"] == 0 and again["conflicts"] == 0


def test_bulk_accept_leaves_the_instances_not_ticked(session):
    session.sweeper_enabled = False
    _two_confirmed_frames(session)
    _retrace(session, rect(2, 34, 34, 62))
    session.goto(10)
    draw(session, COOLER, cell(40), api.SCOPE_KEYFRAME)
    session.truth.refresh_range(DESKTOP, VIEW, [12, 13])
    assert {c["instance"] for c in _open(session)} == {CHASSIS, COOLER}

    result = session.accept_new_bulk([COOLER])

    assert result["open"] == 0
    assert {c["instance"] for c in _open(session)} == {CHASSIS}


def test_bulk_accept_works_with_the_sweeper_running(session):
    """The sweeper raises the conflicts in the background while the bulk answers them."""
    _two_confirmed_frames(session)
    _retrace(session, rect(2, 34, 34, 62))
    assert session.drain_sweeper(timeout=20.0)
    assert sorted(c["step"] for c in _open(session)) == [12, 13]
    _retrace(session, rect(4, 4, 40, 40))      # re-queues 12 and 13 for the sweeper

    result = session.accept_new_bulk()          # the sweeper may be in the middle of them

    assert result["open"] == 0
    assert session.drain_sweeper(timeout=20.0)
    assert _open(session) == []
    again = session.truth.refresh_range(DESKTOP, VIEW, session.available_steps(),
                                        ignore_digest=True)
    assert again["standing"] == 0


# --------------------------------------------------------------------------- #
# the re-trace warning (E)
# --------------------------------------------------------------------------- #
def test_retrace_reach_counts_the_confirmed_frames_a_keyframe_reaches(session):
    _two_confirmed_frames(session)
    session.goto(10)
    session.begin_edit(CHASSIS)

    reach = session.retrace_reach()

    assert reach["frozen"] == [12, 13] and {12, 13} <= set(reach["steps"])
    session.clear_edit()
    # a part with no shape yet re-traces nothing
    session.db.conn.execute("DELETE FROM shape_keyframe WHERE instance=?", (COOLER,))
    session.db.conn.commit()
    session._invalidate()
    assert session.retrace_reach(COOLER) is None


def test_the_guide_names_shift_n_and_the_retrace_reach():
    review = G.plan_for(G.GuideFacts(mode="review", open_conflicts=145))
    assert "Shift+N" in review.now and "145" in review.now
    assert review.action == "review_accept_all_new"
    single = G.plan_for(G.GuideFacts(mode="review", open_conflicts=1))
    assert "Shift+N" not in single.now and "N 采用新的" in single.now

    loaded = G.plan_for(G.GuideFacts(editing="内存卡扣 ram_latch.01", layer_pixels=True,
                                     retrace_frozen=tuple(range(17, 42))))
    assert "重画会影响 25 个已确认帧，提交后用 Shift+N 一次接受" in loaded.now
    dirty = G.plan_for(G.GuideFacts(editing="x", layer_pixels=True, layer_dirty=True,
                                    retrace_frozen=(40, 41)))
    assert "重画会影响 2 个已确认帧" in dirty.now
    fresh = G.plan_for(G.GuideFacts(editing="x", layer_pixels=True))
    assert "重画会影响" not in fresh.now


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #
@pytest.fixture
def window(qapp, tmp_path):
    session = app_scene.make_session(tmp_path)
    win = MainWindow(session, app_scene.make_paths(tmp_path), "tester",
                     sam_queue=app_scene.StubSamQueue())
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    if win.roi_editing:
        win.act_clear_edit()
    QApplication.processEvents()
    yield win
    app_scene.close_window(win)


def _window_conflicts(win) -> None:
    """Frames 13 and 12 confirmed, then the chassis re-traced at 10: two conflicts."""
    session = win.session
    session.sweeper_enabled = False
    session.goto(12)
    app_scene.seed_shapes(session, 12)
    session.goto(13)
    assert session.confirm_frame() and session.confirm_frame()
    session.goto(10)
    win.on_request_edit(CHASSIS)
    assert "重画会影响 2 个已确认帧（第 12、13 帧），提交后用 Shift+N 一次接受" \
        in win.status_message()
    assert "重画会影响 2 个已确认帧" in win.guide_plan().now
    session.set_editing_mask(rect(2, 34, 34, 62))
    session.commit_edit(api.SCOPE_KEYFRAME)
    session.clear_edit()
    session.truth.refresh_range(DESKTOP, VIEW, [12, 13])


def test_the_window_adapter_falls_back_to_one_answer_per_row():
    """A session without ``accept_new_bulk``: N's call, newest conflict of each row."""
    from tda.ui import app_compat as compat

    class Old:
        def __init__(self):
            self.open = [{"id": 1, "step": 5, "instance": "a"}, {"id": 2, "step": 5, "instance": "a"},
                         {"id": 3, "step": 6, "instance": "a"}, {"id": 4, "step": 6, "instance": "b"}]
            self.calls = []

        def queues(self):
            return {api.QUEUE_CONFLICTS: list(self.open)}

        def resolve_conflict(self, cid, resolution):
            self.calls.append((cid, resolution))
            self.open = [c for c in self.open
                         if (c["step"], c["instance"]) != next(
                             (e["step"], e["instance"]) for e in self.open if e["id"] == cid)]
            return api.Verdict("resolved")

    old = Old()
    out = compat.accept_new_bulk(old, ["a"])
    assert old.calls == [(3, api.RESOLVE_ACCEPT_NEW), (2, api.RESOLVE_ACCEPT_NEW)]
    assert out == {"accepted": 2, "superseded": 0, "refused": 0, "open": 0}
    assert [c["instance"] for c in old.open] == ["b"]


def test_shift_n_is_a_review_key_only():
    combo = (Qt.Key.Key_N, Qt.KeyboardModifier.ShiftModifier)
    assert A.action_for(*combo, A.MODE_REVIEW).name == "review_accept_all_new"
    assert A.action_for(*combo, A.MODE_ANNOTATE) is None
    assert A.action_for(Qt.Key.Key_N, Qt.KeyboardModifier.NoModifier,
                        A.MODE_REVIEW).name == "review_accept_new"
    assert A.action_named("review_accept_all_new").short == "接受全部新值…"


def test_shift_n_opens_the_dialog_and_accepts_what_is_ticked(window):
    _window_conflicts(window)
    window.set_mode(A.MODE_REVIEW)
    window.review.refresh()
    window.refresh_guidance()
    assert window.palette.button("review_accept_all_new").isEnabled()
    assert "Shift+N" in window.guide_plan().now

    window.dispatch(A.action_named("review_accept_all_new"))
    dialog = window.bulk_accept_dialog
    assert dialog.isVisible()
    assert [g.text for g in dialog.groups] == ["chassis ×2（帧 12、13）"]
    dialog.accept()
    QApplication.processEvents()

    assert window.status_message().startswith("已接受 2 个，仍开放 0 个")
    # frames 12 and 13 are still queued for a re-check (round 2)
    assert "后台复检中 2 帧" in window.status_message(), window.status_message()
    assert _open(window.session) == []
    assert window.review.list_for(api.QUEUE_CONFLICTS).count() == 0
    window.refresh_guidance()
    assert not window.palette.button("review_accept_all_new").isEnabled()


def test_a_split_row_does_not_get_the_retrace_clause(window):
    """The reviewer's case: frame 14 confirmed, stand on 13 (✂ row), select ram_latch.01."""
    session = window.session
    session.sweeper_enabled = False
    session.goto(14)
    app_scene.seed_shapes(session, 14)
    assert session.confirm_frame() is True            # steps back to 13
    assert session.current().step == 13
    rows = {r["instance"]: r["kind"] for r in session.task_card()}
    assert rows.get("ram_latch.01") == api.KIND_SPLIT_KEYFRAME
    # what Enter would reach does include the confirmed frame 14 ...
    assert 14 in (session.retrace_reach("ram_latch.01") or {}).get("frozen", [])

    window.on_request_edit("ram_latch.01")

    # ... but the row is committed with Ctrl+K, and nothing says Shift+N
    assert "重画会影响" not in window.status_message(), window.status_message()
    plan = window.guide_plan()
    assert "Ctrl+K" in plan.now and "重画会影响" not in plan.now


def test_shift_n_with_no_conflict_says_so(window):
    window.set_mode(A.MODE_REVIEW)
    window.act_accept_all_new()
    assert "没有开放的冲突" in window.status_message()
    assert getattr(window, "bulk_accept_dialog", None) is None
