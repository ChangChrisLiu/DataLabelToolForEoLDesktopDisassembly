"""Task U7 in the session: re-checks for every frozen frame (B).

On D13/scan 添加零件 demoted the confirmed frames 17-41 to ``needs_review``
-- by design: each gained four clips -- but every row the annotator had signed
stayed frozen.  The re-trace that followed reached all of them, and ``settle``
queued a re-check only for frames whose *flag* said ``verified``: 102
disagreements stayed latent, to surface five at a time on each visit.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from tda.core.extra import plan_extras, write_batch
from tda.core.model import FrameKey
from tda.ui import session_api as api
from tda.ui.session import AnnotationSession
from tda.ui.session_ops import frozen_among
from session_scene import BOARD, CHASSIS, DESKTOP, VIEW, draw, make_session, rect, seed_shapes


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def session(qapp, tmp_path: Path) -> AnnotationSession:
    made = make_session(tmp_path)
    yield made
    made.close(force=True)


def _open_at(session, step: int) -> list[dict]:
    return [c for c in session.db.conflicts(DESKTOP, VIEW, open_only=True)
            if int(c["step"]) == step]


def _confirmed(session, step: int = 12) -> FrameKey:
    session.goto(step)
    seed_shapes(session, step)
    assert session.confirm_frame() is True
    return FrameKey(DESKTOP, step, VIEW)


def _add_a_clip(session) -> list[str]:
    """What 添加零件 writes: one more RAM clip on the motherboard, in every frame."""
    batch = plan_extras(session.db.instances(DESKTOP), session.tax, DESKTOP,
                        "ram_latch", 1, parent=BOARD, annotator="tester")
    write_batch(session.db, batch, "tester")
    session._invalidate()
    return batch.keys


def test_frozen_means_the_flag_or_a_signed_row(session):
    key = _confirmed(session)
    session.truth.demote_frame(key, "the verified frame gained ram_latch.05")

    assert session.db.get_frame(key)["review_status"] == "needs_review"
    assert frozen_among(session.db, DESKTOP, VIEW, [10, 11, 12, 13]) == [12]
    assert session.db.frozen_steps_in(DESKTOP, VIEW, []) == []
    assert session.db.frozen_steps_in(DESKTOP, VIEW, [12, 12, 3]) == [12]


def test_adding_a_part_demotes_the_frame_and_leaves_its_rows_signed(session):
    key = _confirmed(session)
    _add_a_clip(session)

    session.truth.refresh(key)

    assert session.db.get_frame(key)["review_status"] == "needs_review"
    assert any(row["status"] == "verified" for row in session.db.compiled(key).values())
    assert frozen_among(session.db, DESKTOP, VIEW, [12]) == [12]


def test_a_demoted_frame_with_signed_rows_is_rechecked_by_the_sweeper(session):
    """The disagreement is raised in the background, not when the frame is visited."""
    key = _confirmed(session)
    session.truth.demote_frame(key, "the verified frame gained ram_latch.05")
    session.goto(10)

    result = draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)

    assert 12 in result["rechecks"]
    assert 12 in session.db.rechecks(DESKTOP, VIEW) or _open_at(session, 12)
    assert session.drain_sweeper(timeout=20.0)
    assert [c["instance"] for c in _open_at(session, 12)] == [CHASSIS]
    assert session.current().step == 10          # nobody went to frame 12
    assert session.db.rechecks(DESKTOP, VIEW) == []


def test_the_reach_preview_counts_a_demoted_frame_as_frozen(session):
    """The L question ("其中已确认的第 … 帧会被重新检查") asks the same notion."""
    key = _confirmed(session)
    session.truth.demote_frame(key, "the verified frame gained ram_latch.05")
    session.goto(10)
    session.begin_edit(CHASSIS)

    reach = session.preview(api.SCOPE_KEYFRAME)

    assert 12 in reach["steps"] and reach["verified_steps"] == [12]


# --------------------------------------------------------------------------- #
# round 2 (i): opening a view re-checks its frozen frames
# --------------------------------------------------------------------------- #
def test_opening_the_view_rechecks_a_demoted_frame_without_a_visit(session):
    """A latent disagreement an older edit never queued is raised after the view opens.

    Frame 12 is confirmed and demoted, its signed rows kept; the chassis shape
    changes under it without anybody queueing a re-check (what an edit before
    U7 left behind).  Frame 13 has an open conflict already: it was compared,
    and is not re-queued.
    """
    from tda.core import masks
    from tda.core.model import ShapePart

    session.goto(12)
    seed_shapes(session, 12)
    session.goto(13)
    assert session.confirm_frame() is True and session.confirm_frame() is True
    session.truth.demote_frame(FrameKey(DESKTOP, 12, VIEW), "the verified frame gained x")
    planted = session.db.add_conflict(FrameKey(DESKTOP, 13, VIEW), CHASSIS, None, None, 1)
    chassis = session.db.keyframes(DESKTOP, VIEW, CHASSIS)[0]
    chassis.parts = [ShapePart("main", masks.encode_rle(rect(2, 34, 34, 62)))]
    session.db.update_keyframe(chassis)            # no settle: nothing queued
    assert session.db.rechecks(DESKTOP, VIEW) == []
    assert _open_at(session, 12) == []

    session.sweeper_enabled = False
    session.open(DESKTOP, VIEW)
    assert session.db.rechecks(DESKTOP, VIEW) == [12]     # 13 has its open conflict

    session.sweeper_enabled = True
    session.open(DESKTOP, VIEW)
    assert session.drain_sweeper(timeout=20.0)
    assert [c["instance"] for c in _open_at(session, 12)] == [CHASSIS]
    assert [c["id"] for c in _open_at(session, 13)] == [planted]   # not re-checked
    assert session.current().step != 12                             # nobody visited it
    assert session.db.rechecks(DESKTOP, VIEW) == []
