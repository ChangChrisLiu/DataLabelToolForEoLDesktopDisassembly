"""Task U37 round 2: a start frame is a start frame whatever its step type.

D34 ends on 「Try to remove the optical drive」, a ``failed`` step. The card
used to answer ``failed`` / ``dupli`` before asking whether the frame is a
start at all, so D34's last frame -- where the reverse walk starts -- showed one
row, "confirm", and the frame before it (nothing changed next door) another.
The chassis never changes state, so it is only ever on a start card: no card
anywhere asked for it, and only ``Space``'s ``missing_shape`` said so.

Now the start comes first: no neighbour, or a neighbour in another pose
segment, is the start card -- every part present, the chassis among them --
and the one-row confirmation of a ``failed`` / ``dupli`` step is kept for a
neighbour inside its own segment, where it still means "nothing changed". The
Qt-free ✂ rows keep the same gates in the same order.
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
from tda.core.split_rows import split_rows_at
from tda.core.states import needs_geom
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import instances_of, state_of
from tda.ui import session_api as api
from tda.ui.panels.taskcard import TaskCardPanel
from tda.ui.session import AnnotationSession
from tda.ui.session_tasks import SEGMENT_START, task_card_for

D, V = S.DESKTOP, S.VIEW
LAST = S.LAST_STEP          # 14: the D13 scene's last logged step
TRY = "Try to remove the optical drive"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _scene(tmp_path: Path, tax, extra: list[tuple[str, str, bool]], name: str = "d34") -> Db:
    """The D13 scene plus steps ``(type, raw name, dupli)`` after step 14, frames for
    them, and the pose segments the step table implies (a ``reorient`` cuts)."""
    db = Db(str(tmp_path / f"{name}.sqlite"))
    cache = tmp_path / f"{name}_cache"
    S.seed_db(db, tax, cache)
    added = [StepRec(D, LAST + 1 + i, kind, raw, dupli=dupli)
             for i, (kind, raw, dupli) in enumerate(extra)]
    db.replace_steps(D, db.steps(D) + added, db.actions(D))
    last = LAST + len(extra)
    S._write_frames(db, cache, range(LAST + 1, last + 1), S.HW)
    db.set_pose_segment(D, V, 1, 1, last, last, None, None)
    db.recut_view(D, V)
    return db


def _needs(db: Db, tax, step: int) -> set[str]:
    """What the scanner has to draw at ``step``: no staging area in this scene."""
    return set(needs_geom(instances_of(db, D), state_of(db, tax, D, step), tax, bench_roi=None))


@pytest.mark.parametrize("kind,dupli", [("failed", False), ("dupli", True)])
def test_a_last_frame_that_is_a_failed_or_dupli_step_gets_the_start_card(tmp_path, tax,
                                                                         kind, dupli):
    """D34 in small: the walk starts on 「Try to remove …」 and lists every part."""
    db = _scene(tmp_path, tax, [(kind, TRY, dupli)])
    try:
        last = LAST + 1
        card = task_card_for(db, tax, D, V, last, None)
        assert {row["kind"] for row in card} == {api.KIND_ADD_SHAPE}, card
        assert {row["instance"] for row in card} == _needs(db, tax, last)
        assert card[0]["instance"] == S.CHASSIS          # bottom-up: the chassis first
        assert not any(SEGMENT_START in row for row in card)
        assert TRY not in {row["instance"] for row in card}
    finally:
        db.close()


@pytest.mark.parametrize("kind,dupli", [("failed", False), ("dupli", True)])
def test_inside_a_segment_a_failed_or_dupli_step_is_still_one_confirmation(tmp_path, tax,
                                                                          kind, dupli):
    """Step 15 fails, step 16 fails again: 15 against 16 (and forwards, 15 against
    14) is the one-row confirmation it always was, titled by the step."""
    db = _scene(tmp_path, tax, [(kind, TRY, dupli), ("failed", "Try again", False)])
    try:
        for neighbour in (LAST + 2, LAST):
            [row] = task_card_for(db, tax, D, V, LAST + 1, neighbour)
            assert row["kind"] == api.KIND_CONFIRM and row["instance"] == TRY, neighbour
            assert SEGMENT_START not in row
            assert split_rows_at(db, tax, D, V, LAST + 1, neighbour) == []
        # the walk's start is the second failed step: the start card
        card = task_card_for(db, tax, D, V, LAST + 2, None)
        assert {row["instance"] for row in card} == _needs(db, tax, LAST + 2)
        # and the frame before the failed one is the difference, as ever: nothing
        [row] = task_card_for(db, tax, D, V, LAST, LAST + 1)
        assert row["kind"] == api.KIND_CONFIRM and row["instance"] == f"step {LAST}"
    finally:
        db.close()


def test_a_failed_step_that_starts_its_pose_segment_gets_the_start_card(tmp_path, tax):
    """Step 15 fails, step 16 turns the chassis: 15 is the end of segment 1."""
    db = _scene(tmp_path, tax, [("failed", TRY, False),
                                ("reorient", "Chassis reoriented", False)])
    try:
        assert [(s["seg"], s["start_step"], s["end_step"]) for s in db.pose_segments(D, V)] \
            == [(1, 1, LAST + 1), (2, LAST + 2, LAST + 2)]
        card = task_card_for(db, tax, D, V, LAST + 1, LAST + 2)
        assert {row["kind"] for row in card} == {api.KIND_ADD_SHAPE}
        assert {row["instance"] for row in card} == _needs(db, tax, LAST + 1)
        assert all(row[SEGMENT_START] == {"cut": LAST + 2, "reorient": True} for row in card)
        assert split_rows_at(db, tax, D, V, LAST + 1, LAST + 2) == []
    finally:
        db.close()


def test_the_session_card_and_the_header_on_d34s_last_frame(qapp, tmp_path, tax):
    db = _scene(tmp_path, tax, [("failed", TRY, False)])
    session = AnnotationSession(db, tax, TruthService(db, tax), str(tmp_path / "d34_cache"),
                                S.ANNOTATOR)
    session.open(D, V)
    panel = TaskCardPanel()
    try:
        session.goto(LAST + 1)
        assert session.task_neighbour() is None
        card = session.task_card()
        assert S.CHASSIS in {row["instance"] for row in card}
        panel.set_session(session)
        assert panel.header_text().startswith(
            f"第 {LAST + 1} 帧（起点，已经拆完的样子）：把这一帧里还看得到的零件都画出来")
        # Space says the same thing the card does
        assert session.confirm_frame() is False
        missing = {p.split(":", 1)[1] for p in session.review.problems[LAST + 1]
                   if p.startswith("missing_shape:")}
        assert missing == {row["instance"] for row in card}
        assert session.current() == FrameKey(D, LAST + 1, V)
    finally:
        panel.set_session(None)
        session.close(force=True)
        db.close()
