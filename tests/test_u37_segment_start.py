"""Task U37: the first frame of a pose segment is a start frame.

D48's step 36 turns the chassis over (``reorient``), so its pose segments are
[1-35] and [36]. Annotated backwards, frame 36 got the start card -- every part
-- and was drawn and confirmed. Frame 35 then showed one row, "nothing to draw
here", because the card was the difference against 36 and a flip moves no part.
But nothing drawn on 36 applies to 35 (no keyframe reaches across a segment),
so 35 is where segment 1 starts and every part on it has to be drawn: ``cli
check`` said so with twelve ``missing_shape`` rows. The user: 「35步让我什么都
不画，直接跳过了……这个是错误的」.

The rule pinned here: a neighbour in another pose segment counts as none. The
card is the start card -- either direction, the walk's backwards and a forward
repair -- and every row carries ``segment_start`` (where the cut is, and
whether it is a flip or a pose break), which is what the card's header and the
guide's title say. Inside a segment, and in a view with no cut at all, nothing
changes. The ✂ rows ``cli check`` and the commit rule read stop at the cut
too, so the card and its Qt-free twin keep the same gates.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

import session_scene as S
from tda.core.db import Db
from tda.core.model import FrameKey, StepRec
from tda.core.pose_breaks import ACCEPTED, KIND_MANUAL
from tda.core.split_rows import split_rows_at
from tda.core.states import diff_states, needs_geom
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import instances_of, same_segment, state_of
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui.panels.taskcard import TaskCardPanel, card_header
from tda.ui.session import AnnotationSession
from tda.ui.session_tasks import SEGMENT_START, segment_cut, task_card_for

D, V = S.DESKTOP, S.VIEW
LAST = S.LAST_STEP          # 14: the D13 scene's last logged step
FLIP = LAST + 1             # 15: a reorient step appended, D48-style
BREAK = 8                   # a manual pose break: segments [1-7], [8-14]


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# --------------------------------------------------------------------------- #
# scenes
# --------------------------------------------------------------------------- #
def _plain(tmp_path: Path, tax, name: str = "plain") -> Db:
    """The D13 scene (steps 1-14, one frame each), one pose segment."""
    db = Db(str(tmp_path / f"{name}.sqlite"))
    S.seed_db(db, tax, tmp_path / f"{name}_cache")
    db.set_pose_segment(D, V, 1, 1, LAST, LAST, None, None)
    return db


def _with_break(tmp_path: Path, tax, at: int = BREAK) -> Db:
    """The same scene with an accepted pose break of the scanner view at ``at``."""
    db = _plain(tmp_path, tax, "break")
    db.add_pose_break(D, V, at, status=ACCEPTED, kind=KIND_MANUAL)
    db.recut_view(D, V)
    assert [(s["seg"], s["start_step"], s["end_step"]) for s in db.pose_segments(D, V)] \
        == [(1, 1, at - 1), (2, at, LAST)]
    return db


def _with_flip(tmp_path: Path, tax) -> Db:
    """D48 in small: a ``reorient`` step after the last logged one, its own segment."""
    db = Db(str(tmp_path / "flip.sqlite"))
    cache = tmp_path / "flip_cache"
    S.seed_db(db, tax, cache)
    steps = db.steps(D) + [StepRec(D, FLIP, "reorient", "Chassis reoriented (turned)")]
    db.replace_steps(D, steps, db.actions(D))
    S._write_frames(db, cache, [FLIP], S.HW)
    db.set_pose_segment(D, V, 1, 1, FLIP, FLIP, None, None)
    db.recut_view(D, V)
    assert [(s["seg"], s["start_step"], s["end_step"]) for s in db.pose_segments(D, V)] \
        == [(1, 1, LAST), (2, FLIP, FLIP)]
    return db


def _strip(card: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k != SEGMENT_START} for row in card]


def _needs(db: Db, tax, step: int) -> set[str]:
    """What the scanner has to draw at ``step``: no staging area in this scene."""
    instances = instances_of(db, D)
    return set(needs_geom(instances, state_of(db, tax, D, step), tax, bench_roi=None))


def _assert_start_card(db: Db, tax, step: int, neighbour: int, cut: dict) -> None:
    """``step`` against ``neighbour`` is the start card, marked with ``cut``."""
    card = task_card_for(db, tax, D, V, step, neighbour)
    assert {row["kind"] for row in card} == {api.KIND_ADD_SHAPE}, card
    assert {row["instance"] for row in card} == _needs(db, tax, step)
    assert all(row[SEGMENT_START] == cut for row in card), card
    # exactly what the walk's own start card would list there, plus the mark
    assert _strip(card) == task_card_for(db, tax, D, V, step, None)


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def test_the_frame_before_a_flip_is_a_start_card_walking_backwards(tmp_path, tax):
    """D48 frame 35: the flip changes no part, and every part has to be drawn."""
    db = _with_flip(tmp_path, tax)
    try:
        # what the old card was made of: no difference at all across the flip
        assert list(diff_states(state_of(db, tax, D, FLIP), state_of(db, tax, D, LAST))) == []
        assert not same_segment(db, D, V, LAST, FLIP)
        _assert_start_card(db, tax, LAST, FLIP, {"cut": FLIP, "reorient": True})
        assert len(task_card_for(db, tax, D, V, LAST, FLIP)) > 10
    finally:
        db.close()


def test_the_sequence_end_after_a_flip_is_the_start_card_it_always_was(tmp_path, tax):
    """D48 frame 36: the walk's start, no neighbour, no segment mark."""
    db = _with_flip(tmp_path, tax)
    try:
        card = task_card_for(db, tax, D, V, FLIP, None)
        assert {row["kind"] for row in card} == {api.KIND_ADD_SHAPE}
        assert not any(SEGMENT_START in row for row in card)
        header = card_header(FLIP, None, [r["kind"] for r in card], first=1, last=FLIP,
                             done=[r["done"] for r in card])
        assert header.startswith(f"第 {FLIP} 帧（起点，已经拆完的样子）：把这一帧里还看得到的零件都画出来")
    finally:
        db.close()


def test_forward_across_the_cut_is_a_start_card_too(tmp_path, tax):
    """Browsing forwards to repair, the frame after the flip has nobody before it."""
    db = _with_flip(tmp_path, tax)
    try:
        _assert_start_card(db, tax, FLIP, LAST, {"cut": FLIP, "reorient": True})
    finally:
        db.close()


def test_a_pose_break_cuts_the_same_way_in_both_directions(tmp_path, tax):
    db = _with_break(tmp_path, tax)
    try:
        cut = {"cut": BREAK, "reorient": False}
        _assert_start_card(db, tax, BREAK - 1, BREAK, cut)       # backwards
        _assert_start_card(db, tax, BREAK, BREAK - 1, cut)       # forwards
        assert segment_cut(db, D, V, BREAK - 1, BREAK) == cut
        assert segment_cut(db, D, V, BREAK, BREAK - 1) == cut
    finally:
        db.close()


def test_a_skipped_frame_does_not_move_the_cut(tmp_path, tax):
    """With frame 7 missing the walk compares 6 with 8: the cut is still at 8."""
    db = _with_break(tmp_path, tax)
    try:
        cut = {"cut": BREAK, "reorient": False}
        _assert_start_card(db, tax, BREAK - 2, BREAK, cut)
        _assert_start_card(db, tax, BREAK + 1, BREAK - 1, cut)
    finally:
        db.close()


def test_inside_a_segment_the_card_is_still_the_difference(tmp_path, tax):
    """Every pair inside one segment gets exactly the card it gets with no cut."""
    plain = _plain(tmp_path, tax)
    cut = _with_break(tmp_path, tax)
    try:
        for step in range(1, LAST + 1):
            for neighbour in (step - 1, step + 1):
                if not 1 <= neighbour <= LAST or (step < BREAK) != (neighbour < BREAK):
                    continue
                expected = task_card_for(plain, tax, D, V, step, neighbour)
                assert task_card_for(cut, tax, D, V, step, neighbour) == expected, \
                    (step, neighbour)
        # ... including a ✂ row right next to the cut: 8 against 9 (connector.02)
        [row] = task_card_for(cut, tax, D, V, BREAK, BREAK + 1)
        assert row["kind"] == api.KIND_SPLIT_KEYFRAME and SEGMENT_START not in row
    finally:
        plain.close()
        cut.close()


def test_without_a_cut_nothing_is_marked(tmp_path, tax):
    """One segment, or none recorded at all: the same cards, no mark anywhere."""
    one = _plain(tmp_path, tax)
    none = Db(str(tmp_path / "none.sqlite"))
    S.seed_db(none, tax, tmp_path / "none_cache")
    try:
        assert none.pose_segments(D, V) == []
        for step in range(1, LAST + 1):
            for neighbour in (None, step - 1, step + 1):
                if neighbour is not None and not 1 <= neighbour <= LAST:
                    continue
                if neighbour is not None:
                    assert same_segment(one, D, V, step, neighbour)
                    assert same_segment(none, D, V, step, neighbour)
                card = task_card_for(one, tax, D, V, step, neighbour)
                assert not any(SEGMENT_START in row for row in card), (step, neighbour)
                assert card == task_card_for(none, tax, D, V, step, neighbour)
    finally:
        one.close()
        none.close()


def test_the_split_rows_stop_at_the_cut(tmp_path, tax):
    """Frame 7 against 8 is a ✂ row (connector.01 plugged) in one segment, none across."""
    plain = _plain(tmp_path, tax)
    cut = _with_break(tmp_path, tax)
    try:
        assert [r[0] for r in split_rows_at(plain, tax, D, V, BREAK - 1, BREAK)] \
            == ["connector.01"]
        assert split_rows_at(cut, tax, D, V, BREAK - 1, BREAK) == []
        # inside the segments they are what they were
        for step in range(1, LAST):
            if step == BREAK - 1:
                continue
            assert split_rows_at(cut, tax, D, V, step, step + 1) \
                == split_rows_at(plain, tax, D, V, step, step + 1), step
    finally:
        plain.close()
        cut.close()


# --------------------------------------------------------------------------- #
# the words
# --------------------------------------------------------------------------- #
def test_the_header_says_where_the_segment_starts():
    kinds = [api.KIND_ADD_SHAPE] * 3
    flip = G.pose_cut_text({"cut": 36, "reorient": True})
    assert flip == "第 36 步翻转机箱"
    header = card_header(35, 36, kinds, first=1, last=36, done=[False] * 3, pose_cut=flip)
    assert header.startswith("第 35 帧（位姿段的起始帧：第 36 步翻转机箱，另起一段，"
                             "这一段的形状要重新画）：把这一帧里还看得到的零件都画出来")
    assert "比第 36 帧多了" not in header
    brk = G.pose_cut_text({"cut": 8, "reorient": False})
    assert brk == "第 8 步是位姿断点"
    header = card_header(8, 7, kinds, first=1, last=14, done=[True, False, False],
                         pose_cut=brk)
    assert header.startswith("第 8 帧（位姿段的起始帧：第 8 步是位姿断点，")
    assert header.endswith("。还剩 2 个，1 个画好了（✔，排在最后）")
    done = card_header(35, 36, kinds, first=1, last=36, done=[True] * 3, pose_cut=flip)
    assert done == ("第 35 帧（位姿段的起始帧：第 36 步翻转机箱，另起一段，这一段的形状要重新画）："
                    "这一帧的零件都画好了，直接 Space 确认")
    # no cut: the walk's own words, unchanged
    assert card_header(35, 36, kinds) == ("第 35 帧：比第 36 帧多了下面这些零件"
                                          "（刚被拆掉的，要把它画回去）")
    assert G.pose_cut_text(None) == ""


def test_the_guide_title_says_it_too():
    facts = G.GuideFacts(step=35, neighbour=36, pose_cut="第 36 步翻转机箱",
                         items=(G.CardItem("chassis", api.KIND_ADD_SHAPE, False, "机箱"),))
    plan = G.plan_for(facts)
    assert plan.title == "第 35 帧（位姿段的起始帧：第 36 步翻转机箱）"
    assert plan.phase == "pick" and "机箱" in plan.now
    assert G.plan_for(G.GuideFacts(step=35, neighbour=36)).title == "第 35 帧（对照第 36 帧）"
    assert G.plan_for(G.GuideFacts(step=36)).title == "第 36 帧（起点：已经拆完的样子）"
    rows = [{"instance": "a", "kind": api.KIND_ADD_SHAPE},
            {"instance": "b", "kind": api.KIND_ADD_SHAPE,
             SEGMENT_START: {"cut": 36, "reorient": True}}]
    assert G.segment_start_of(rows) == {"cut": 36, "reorient": True}
    assert G.segment_start_of(rows[:1]) is None


# --------------------------------------------------------------------------- #
# the session, the panel and Space
# --------------------------------------------------------------------------- #
def test_the_session_card_the_panel_and_space_on_the_frame_before_the_flip(qapp, tmp_path,
                                                                           tax):
    db = _with_flip(tmp_path, tax)
    session = AnnotationSession(db, tax, TruthService(db, tax), str(tmp_path / "flip_cache"),
                                S.ANNOTATOR)
    session.open(D, V)
    panel = TaskCardPanel()
    try:
        session.goto(LAST)
        assert session.task_neighbour() == FLIP          # the neighbour is not changed
        card = session.task_card()
        assert {row["instance"] for row in card} == _needs(db, tax, LAST)
        assert all(row[SEGMENT_START] == {"cut": FLIP, "reorient": True} for row in card)
        panel.set_session(session)
        assert panel.header_text().startswith(
            f"第 {LAST} 帧（位姿段的起始帧：第 {FLIP} 步翻转机箱，另起一段，这一段的形状要重新画）："
            "把这一帧里还看得到的零件都画出来")
        # a start frame's rows leave "came back in with ..." off, as on the walk's start
        assert not any("一起装回来的" in panel.list_widget().item(i).text()
                       for i in range(panel.list_widget().count()))
        # Space: the compiler's missing shapes, one per row
        assert session.confirm_frame() is False
        problems = session.review.problems[LAST]
        missing = {p.split(":", 1)[1] for p in problems if p.startswith("missing_shape:")}
        assert missing == {row["instance"] for row in card}
        assert session.current() == FrameKey(D, LAST, V)
        # one frame further in, the card is the difference again
        session.goto(LAST - 1)
        assert not any(SEGMENT_START in row for row in session.task_card())
        panel.refresh()
        assert "位姿段" not in panel.header_text()
    finally:
        panel.set_session(None)
        session.close(force=True)
        db.close()


def test_the_window_says_it_in_the_header_and_the_guide(qapp, tmp_path):
    """The app scene's two-segment variant (cut at 8, no break row: a pose break)."""
    from app_scene import SEGMENT_CUT, StubSamQueue, close_window, make_paths, make_session
    from tda.ui.app import MainWindow

    session = make_session(tmp_path, two_segments=True)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    try:
        win.session.goto(SEGMENT_CUT - 1)
        win.task_card.refresh()
        cut = f"第 {SEGMENT_CUT} 步是位姿断点"
        assert win.task_card.header_text().startswith(
            f"第 {SEGMENT_CUT - 1} 帧（位姿段的起始帧：{cut}，")
        facts = win.guide_facts()
        assert facts.neighbour == SEGMENT_CUT and facts.pose_cut == cut
        assert G.plan_for(facts).title == f"第 {SEGMENT_CUT - 1} 帧（位姿段的起始帧：{cut}）"
        win.session.goto(SEGMENT_CUT - 2)
        win.task_card.refresh()
        facts = win.guide_facts()
        assert facts.pose_cut == ""
        assert G.plan_for(facts).title == f"第 {SEGMENT_CUT - 2} 帧（对照第 {SEGMENT_CUT - 1} 帧）"
    finally:
        close_window(win)
