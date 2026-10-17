"""Task U16: ``Enter`` means "from this frame on" (the user's ruling, 2026-10-16).

The annotator walks a view backwards and confirms as they go. Changing a part's
shape on frame k almost always means "from here on in my walk": the frames
behind -- already looked at, often confirmed -- stay as they are. ``Enter``
used to edit the version in force everywhere it applied, which reached those
confirmed frames and turned them red; ``Ctrl+K`` was the key that meant what
they wanted, and they had to remember to press it.

These tests pin the new default:

* the session's judgement (:meth:`enter_scope`): the split where the version in
  force is carried from frames the walk has passed, the re-trace where it starts
  -- in reverse and forward, a frame with no image not counting, a part with no
  shape and a gallery shot always the re-trace;
* the invariant: after ``Enter`` on a carried frame every frame behind it keeps
  byte-identical compiled rows, its review status and no new conflict, and the
  frames from here on carry the new version;
* ``Ctrl+Enter`` 改全部: asks when its reach holds confirmed frames, a second
  press writes and queues their re-check, ``Esc`` takes the question back,
  ``Enter`` and ``L`` do not answer it; with nothing confirmed it writes; on the
  anchor frame it is ``Enter`` and says so;
* ``Ctrl+K`` is ``Enter``; 「改全部」 has the palette slot 「从这帧起新版本」 had;
  the gallery; a layering suggestion; an untouched ✂ row;
* the sentences the annotator reads.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton

import app_scene
import gallery_scene as GS
import session_scene
from session_scene import CHASSIS, COOLER, DESKTOP, VIEW, cell, draw, rect
from test_app_actions import key_event
from tda.core.model import FrameKey
from tda.ui import app_actions as A
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.app_commit import (
    ALL_ASKED,
    ALL_BUTTON,
    ALL_BUTTON_CONFIRM,
    ALL_CANCELLED,
    ALL_IS_ENTER,
    NOTHING_TO_COMMIT,
)
from tda.ui.gallery import GALLERY_REFUSALS
from tda.ui.session import AnnotationSession

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "annotation_guide.md"
QUICKSTART = ROOT / "docs" / "trial_quickstart.md"
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(qapp, tmp_path: Path) -> AnnotationSession:
    made = session_scene.make_session(tmp_path)
    yield made
    made.close(force=True)


def anchors(session, instance: str) -> list[int]:
    return sorted(kf.anchor_step for kf in session.db.keyframes(DESKTOP, VIEW, instance))


def shape_at(session, step: int, instance: str) -> np.ndarray:
    """The part's compiled amodal shape at ``step``, from the inputs, now."""
    return session.truth.compile(FrameKey(DESKTOP, step, VIEW)).instances[instance].amodal


# --------------------------------------------------------------------------- #
# the session's judgement
# --------------------------------------------------------------------------- #
def test_enter_re_traces_where_the_version_starts_and_splits_where_it_is_carried(session):
    session.goto(12)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)     # the cooler leaves at 13
    assert anchors(session, COOLER) == [12]
    session.begin_edit(COOLER)
    assert session.enter_scope() == api.SCOPE_KEYFRAME     # the version starts here
    session.clear_edit()

    session.goto(8)                                        # carried from 12..9
    assert session.enter_scope(COOLER) == api.SCOPE_SPLIT
    # a part with no shape yet takes its default anchor: no "from here on"
    assert CHASSIS not in {kf.instance for kf in session.db.keyframes(DESKTOP, VIEW, CHASSIS)}
    assert session.enter_scope(CHASSIS) == api.SCOPE_KEYFRAME
    assert session.enter_scope() == api.SCOPE_KEYFRAME     # nothing being drawn

    draw(session, COOLER, cell(3), session.enter_scope(COOLER))
    assert anchors(session, COOLER) == [8, 12]
    for step in (9, 10, 11, 12):                           # behind: the old version
        assert np.array_equal(shape_at(session, step, COOLER), cell(0)), step
    for step in (1, 5, 8):                                 # from here on: the new one
        assert np.array_equal(shape_at(session, step, COOLER), cell(3)), step
    # on 8 the new version starts now: Enter re-traces it
    assert session.enter_scope(COOLER) == api.SCOPE_KEYFRAME


def test_the_forward_walk_is_the_mirror_image(session):
    session.goto(12)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    session.goto(8)
    draw(session, COOLER, cell(3), api.SCOPE_SPLIT)        # (..8] and (8..12]
    session.browse_forward()
    session.goto(9)                                        # where (8..12] starts going forward
    assert session.browsing == "forward"
    assert session.enter_scope(COOLER) == api.SCOPE_KEYFRAME
    session.goto(1)                                        # the first frame: nothing behind
    assert session.enter_scope(COOLER) == api.SCOPE_KEYFRAME
    session.goto(11)                                       # 9 and 10 carry it
    assert session.enter_scope(COOLER) == api.SCOPE_SPLIT

    # commit_edit takes the walk's direction: the old version keeps 9-10
    draw(session, COOLER, cell(5), session.enter_scope(COOLER))
    assert anchors(session, COOLER) == [8, 10, 12]
    for step in (9, 10):
        assert np.array_equal(shape_at(session, step, COOLER), cell(0)), step
    for step in (11, 12):
        assert np.array_equal(shape_at(session, step, COOLER), cell(5)), step
    for step in (1, 8):
        assert np.array_equal(shape_at(session, step, COOLER), cell(3)), step


def test_a_forward_split_on_a_versions_first_frame_re_traces(session):
    """Review M2: the forward split pulled the old version onto the anchor below.

    Twice at 6 going forward: the second time 6 is where its version starts
    (the one below ends at 5), so the split would land the old version on 5,
    a version newer than the keyframe there, and replace frames 1-5 with
    nothing re-checked. It re-traces instead and says so (``no_split``).
    """
    session_scene.seed_shapes(session, 3)                  # anchored at 14
    session.browse_forward()
    session.goto(6)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_SPLIT)
    session.clear_edit()
    assert anchors(session, CHASSIS) == [5, 14]
    before = {st: shape_at(session, st, CHASSIS) for st in range(1, 6)}
    again = rect(4, 36, 30, 60)
    result = draw(session, CHASSIS, again, api.SCOPE_SPLIT)
    session.clear_edit()
    assert result["no_split"] is True
    assert anchors(session, CHASSIS) == [5, 14]
    for step in range(1, 6):
        assert np.array_equal(shape_at(session, step, CHASSIS), before[step]), step
    for step in (6, 14):
        assert np.array_equal(shape_at(session, step, CHASSIS), again), step


def test_a_frame_without_an_image_carries_nothing(qapp, tmp_path):
    made = session_scene.make_session(tmp_path, missing=(12,))
    try:
        made.goto(11)
        draw(made, COOLER, cell(0), api.SCOPE_KEYFRAME)
        assert anchors(made, COOLER) == [12]               # its life ends at 12 ...
        # ... which has no image: no frame behind 11 is annotated with it
        assert made.enter_scope(COOLER) == api.SCOPE_KEYFRAME
    finally:
        made.close(force=True)


def test_a_gallery_shot_is_always_the_re_trace(qapp, tmp_path):
    made = GS.make_session(tmp_path)
    try:
        part = made.task_card()[0]["instance"]
        made.begin_edit(part)
        made.set_editing_mask(GS.rect(4, 4, 20, 20))
        assert made.enter_scope() == api.SCOPE_KEYFRAME
        made.commit_edit(made.enter_scope())
        made.begin_edit(part)
        made.set_editing_mask(GS.rect(6, 6, 24, 24))
        assert made.enter_scope() == api.SCOPE_KEYFRAME
        result = made.commit_edit(made.enter_scope())
        assert result["changed"] and result["affected"] == [1]
        assert len(made.db.keyframes(GS.DESKTOP, GS.VIEW, part)) == 1
    finally:
        made.close(force=True)


def test_retrace_reach_is_enters_reach(session):
    """U7's warning counts what Enter writes: from here on, not the frames behind."""
    session.goto(12)
    session_scene.seed_shapes(session, 12)                 # every part anchored at 14
    session.goto(13)
    assert session.confirm_frame() and session.confirm_frame()   # 13 and 12
    session.goto(10)
    session.begin_edit(CHASSIS)
    assert session.retrace_reach()["frozen"] == []
    assert session.retrace_reach(scope=api.SCOPE_KEYFRAME)["frozen"] == [12, 13]


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #
@pytest.fixture
def window(qapp, tmp_path: Path):
    """The 14-step D13 window scene, sweeper off: frames 12-14 drawn, anchored at 14."""
    made = app_scene.make_session(tmp_path)
    made.sweeper_enabled = False
    win = MainWindow(made, app_scene.make_paths(tmp_path), "tester",
                     sam_queue=app_scene.StubSamQueue())
    win.resize(1400, 900)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    if win.roi_editing:
        win.act_clear_edit()
    app_scene.seed_shapes(made, 12)
    QApplication.processEvents()
    yield win
    app_scene.close_window(win)


def goto(win: MainWindow, step: int) -> None:
    win.session.goto(step, force=True)
    QApplication.processEvents()


def confirm_12_and_13(win: MainWindow) -> None:
    goto(win, 13)
    assert win.session.confirm_frame() is True              # steps back to 12
    assert win.session.confirm_frame() is True


def paint(win: MainWindow, instance: str, mask: np.ndarray) -> None:
    win.on_request_edit(instance)
    win.set_editing_mask(mask, undoable=True)
    QApplication.processEvents()


def press(win: MainWindow, slot: str) -> None:
    """The key's slot, and again while the size warning waits for it."""
    getattr(win, slot)()
    for _ in range(2):
        if win._pending_warning is None:
            break
        getattr(win, slot)()


def ctrl_enter(win: MainWindow) -> None:
    """The key, and again while the size warning waits for it (a clip is "too big")."""
    win.handle_key(key_event("Ctrl+Return"))
    if win._pending_warning is not None:
        win.handle_key(key_event("Ctrl+Return"))


def snapshot(session, steps) -> dict:
    """``step -> (review status, compiled rows)`` as stored, after a full refresh."""
    session.truth.refresh_range(DESKTOP, VIEW, session.available_steps(), ignore_digest=True)
    out = {}
    for step in steps:
        key = FrameKey(DESKTOP, step, VIEW)
        out[step] = ((session.db.get_frame(key) or {}).get("review_status"),
                     session.db.compiled(key))
    return out


NEW = rect(2, 34, 34, 62)


def test_enter_on_a_carried_frame_leaves_every_frame_behind_byte_identical(window):
    """The invariant of the ruling, on the D13 scene: 12 and 13 confirmed, Enter at 10."""
    session = window.session
    confirm_12_and_13(window)
    behind = [11, 12, 13, 14]
    before = snapshot(session, behind)
    assert before[12][0] == before[13][0] == api.STATUS_VERIFIED
    assert session.db.conflicts(DESKTOP, VIEW, open_only=True) == []
    goto(window, 10)
    paint(window, CHASSIS, NEW)
    assert window.guide_facts().retrace_frozen == ()        # nothing for Shift+N to answer
    assert "重画会影响" not in window.guide_plan().now

    press(window, "act_commit")

    assert session.editing_instance is None, window.status_message()
    assert anchors(session, CHASSIS) == [10, 14]            # a new version from 10 on
    assert not set(session.db.rechecks(DESKTOP, VIEW)) & set(behind)
    after = snapshot(session, behind)
    assert after == before                                  # rows and review status
    assert session.db.conflicts(DESKTOP, VIEW, open_only=True) == []
    for step in (1, 5, 10):
        assert np.array_equal(shape_at(session, step, CHASSIS), NEW), step
    for step in behind:
        assert not np.array_equal(shape_at(session, step, CHASSIS), NEW), step


def test_enter_names_the_frames_behind_that_keep_the_old_shape(window):
    """The review's gap: a fix made here does not reach the draft two frames back.

    After Enter's split on a carried frame the status line and the canvas say
    which frames still show the old version (and how many are confirmed), and
    which key would have reached them; where the version starts, nothing.
    """
    session = window.session
    confirm_12_and_13(window)
    goto(window, 10)
    paint(window, CHASSIS, NEW)
    press(window, "act_commit")
    said = "已从这帧起新版本；第 11–14 帧（其中已确认 2 帧）仍是旧样子，要一起改用 Ctrl+Enter"
    assert window.status_message() == said
    assert window.canvas.banner_text() == said
    # on 10 the new version starts now: Enter re-traces it, nothing is left behind
    paint(window, CHASSIS, rect(2, 34, 34, 60))
    press(window, "act_commit")
    assert session.editing_instance is None
    assert "仍是旧样子" not in window.status_message(), window.status_message()
    # ... and on the anchor frame of a version nobody is behind of, the same
    goto(window, 14)
    paint(window, CHASSIS, rect(4, 34, 34, 62))
    press(window, "act_commit")
    assert anchors(session, CHASSIS) == [10, 14]
    assert "仍是旧样子" not in window.status_message(), window.status_message()


def kf_rows(session, instance: str) -> list[tuple]:
    """The part's keyframes as rows, the version left out (every update bumps it)."""
    return sorted((kf.id, kf.anchor_step, kf.source, kf.placement,
                   tuple((p.name, str(p.rle), p.box) for p in kf.parts))
                  for kf in session.db.keyframes(DESKTOP, VIEW, instance))


def test_ctrl_enter_after_enters_split_writes_the_shape_behind_too(window):
    """Review M1: the hint is not a dead end -- the reviewer's invariant scene.

    12 and 13 confirmed, Enter at 10 splits; Ctrl+Enter right after (nothing
    loaded: the part the hint named) asks, then writes the version into the
    one behind and drops 10's own: 1-14 carry it, 12 and 13 come back as
    conflicts, one keyframe is left. Ctrl+Z restores the split.
    """
    session = window.session
    confirm_12_and_13(window)
    goto(window, 10)
    paint(window, CHASSIS, NEW)
    press(window, "act_commit")
    assert anchors(session, CHASSIS) == [10, 14]
    split_rows = kf_rows(session, CHASSIS)
    split_shapes = {st: shape_at(session, st, CHASSIS) for st in range(1, 15)}
    assert session.editing_instance is None
    window.refresh_guidance()
    assert window.palette.button("commit_all").isEnabled()     # the hint's part

    window.handle_key(key_event("Ctrl+Return"))                # asks
    assert window.all_asked() and session.editing_instance == CHASSIS
    assert ("把这版形状也写到第 11–14 帧（其中已确认 2 帧，会重新检查）？再按 Ctrl+Enter"
            in window.scope_bar_text()), window.scope_bar_text()
    assert anchors(session, CHASSIS) == [10, 14]               # nothing written yet
    press(window, "act_commit_all")                            # the second press

    assert session.editing_instance is None, window.status_message()
    assert window.status_message() == ("已把这版形状也写到第 11–14 帧"
                                       "（其中已确认 2 帧，会重新检查）")
    assert anchors(session, CHASSIS) == [14]
    for step in range(1, 15):
        assert np.array_equal(shape_at(session, step, CHASSIS), NEW), step
    assert {12, 13} <= set(session.db.rechecks(DESKTOP, VIEW))
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    assert sorted(int(c["step"]) for c in session.db.conflicts(DESKTOP, VIEW, open_only=True)
                  if c["instance"] == CHASSIS) == [12, 13]

    window.act_undo()                                          # one op: both keyframes back
    assert kf_rows(session, CHASSIS) == split_rows
    for step in range(1, 15):
        assert np.array_equal(shape_at(session, step, CHASSIS), split_shapes[step]), step
    # R3: the hint survives the undo -- a bare Ctrl+Enter targets the part again
    window.refresh_guidance()
    assert window.palette.button("commit_all").isEnabled()
    window.handle_key(key_event("Ctrl+Return"))
    assert window.all_asked() and session.editing_instance == CHASSIS
    assert "把这版形状也写到第 11–14 帧" in window.scope_bar_text()


def test_a_first_draw_then_a_fix_at_the_same_frame_reaches_the_draft_behind(window):
    """The rough-draft case: drawn at 10 (it reaches 1-14), fixed at 10 (from 10 on)."""
    session = window.session
    session.db.conn.execute("DELETE FROM shape_keyframe WHERE instance=?", (CHASSIS,))
    session.db.conn.commit()
    session.refresh_all()
    goto(window, 10)
    paint(window, CHASSIS, NEW)
    press(window, "act_commit")                                # the first draw
    assert anchors(session, CHASSIS) == [14]
    fix = rect(4, 36, 30, 60)
    paint(window, CHASSIS, fix)
    press(window, "act_commit")                                # the fix: a split at 10
    assert anchors(session, CHASSIS) == [10, 14]
    assert window.status_message() == ("已从这帧起新版本；第 11–14 帧仍是旧样子，"
                                       "要一起改用 Ctrl+Enter")
    window.handle_key(key_event("Ctrl+Return"))                # nothing confirmed: at once
    assert not window.all_asked() and session.editing_instance is None
    assert window.status_message() == "已把这版形状也写到第 11–14 帧"
    assert anchors(session, CHASSIS) == [14]
    for step in (1, 10, 11, 14):
        assert np.array_equal(shape_at(session, step, CHASSIS), fix), step


def test_ctrl_enter_does_not_fold_a_version_across_a_state_boundary(window):
    """Enter on a ✂ row (open at 12, closed from 13): the frames behind are the other state's."""
    session = window.session
    latch = "ram_latch.02"
    goto(window, 12)
    assert window.instances.select_instance(latch)
    window.act_cycle_state()                                   # open at 12: ✂ here
    window.expire_visibility_note()
    goto(window, 12)
    paint(window, latch, rect(40, 40, 60, 60))
    press(window, "act_commit")
    assert "仍是旧样子" not in window.status_message()          # a ✂ row names nothing
    before = kf_rows(session, latch)
    window.on_request_edit(latch)                              # the stored shape, untouched
    ctrl_enter(window)
    assert "这一帧和后面第 13 步之间是状态边界（✂）" in window.last_error_message()
    assert "只重画这一版用 Enter" not in window.last_error_message()
    assert kf_rows(session, latch) == before
    # R2: with pixels in the layer the refusal names the key that keeps them
    window.set_editing_mask(rect(42, 42, 60, 60), undoable=True)
    ctrl_enter(window)
    assert window.last_error_message().endswith("只重画这一版用 Enter"), \
        window.last_error_message()
    assert kf_rows(session, latch) == before and session.editing_instance == latch


def test_merging_in_keeps_what_a_cut_below_was_showing(window):
    """R1: split at 10, T at 5 makes an un-drawn ✂ below, back at 10 Ctrl+Enter.

    The re-trace cuts the version at the ✂ below; the copy is of the version
    being folded in -- what 1-5 were showing -- not of the one behind (which
    would have put the old shape back on 1-5 while saying 「形状不变」).
    """
    session = window.session
    latch = "ram_latch.02"
    split = rect(40, 40, 60, 60)
    goto(window, 10)
    paint(window, latch, split)
    press(window, "act_commit")
    assert anchors(session, latch) == [10, 14]
    goto(window, 5)
    assert window.instances.select_instance(latch)
    window.act_cycle_state()                                   # open at 5: ✂ at 4 and 5
    window.expire_visibility_note()
    for step in range(1, 11):
        assert np.array_equal(shape_at(session, step, latch), split), step
    goto(window, 10)
    fix = rect(42, 42, 62, 62)
    paint(window, latch, fix)
    ctrl_enter(window)                                         # nothing confirmed: at once
    assert session.editing_instance is None, window.last_error_message()
    assert window.status_message().startswith("已先在第 5 步切开旧版本（形状不变）"), \
        window.status_message()
    for step in range(1, 6):                                   # what they were showing
        assert np.array_equal(shape_at(session, step, latch), split), step
    for step in range(6, 15):
        assert np.array_equal(shape_at(session, step, latch), fix), step


def test_ctrl_enter_inside_a_ctrl_t_span_reaches_the_spans_top(window):
    """Ctrl+T 10-12 open; Enter at 11 leaves 12; Ctrl+Enter at 11 reaches 12, not 13-14."""
    session = window.session
    latch = "ram_latch.02"
    old = {st: shape_at(session, st, latch) for st in range(8, 15)}
    goto(window, 12)
    assert window.instances.select_instance(latch)
    window.ask_state_range = lambda *a: ("open", 10)
    window.act_state_range()
    window.expire_visibility_note()
    goto(window, 11)
    opened = rect(40, 40, 60, 60)
    paint(window, latch, opened)
    press(window, "act_commit")
    assert window.status_message().endswith("已从这帧起新版本；第 12 帧仍是旧样子，"
                                            "要一起改用 Ctrl+Enter"), window.status_message()
    ctrl_enter(window)
    assert session.editing_instance is None, window.last_error_message()
    for step in (10, 11, 12):
        assert np.array_equal(shape_at(session, step, latch), opened), step
    for step in (13, 14):
        assert np.array_equal(shape_at(session, step, latch), old[step]), step


def test_ctrl_enter_asks_then_writes_everywhere_and_queues_the_re_check(window):
    session = window.session
    confirm_12_and_13(window)
    goto(window, 10)
    paint(window, CHASSIS, NEW)
    versions = [(kf.id, kf.version) for kf in session.db.keyframes(DESKTOP, VIEW, CHASSIS)]

    window.handle_key(key_event("Ctrl+Return"))             # asks
    asked = window.scope_bar_text()
    assert window.all_asked() and window.scope_bar.isVisibleTo(window)
    assert "第 1–14 帧（共 14 帧）" in asked, asked
    assert "其中已确认的第 12、13 帧会被重新检查（可能产生冲突）" in asked
    assert "再按 Ctrl+Enter" in asked and "Esc 取消" in asked
    assert window.scope_all_button.text() == ALL_BUTTON_CONFIRM
    assert not window.scope_layer_button.isVisibleTo(window)   # no layering question here
    plan = window.guide_plan()
    assert plan.now.startswith("现在：提示条写着改全部") and plan.action == "commit_all"
    assert [(kf.id, kf.version) for kf in session.db.keyframes(DESKTOP, VIEW, CHASSIS)] \
        == versions                                          # nothing written yet

    window.act_commit()                                     # Enter is not the answer
    assert window.status_message() == ALL_ASKED
    window.act_commit_layering()                            # nor is L
    assert window.status_message() == ALL_ASKED
    assert session.editing_instance == CHASSIS

    press(window, "act_commit_all")                         # the second press writes
    assert session.editing_instance is None, window.status_message()
    assert not window.scope_bar.isVisibleTo(window) and not window.all_asked()
    kfs = session.db.keyframes(DESKTOP, VIEW, CHASSIS)
    assert [(kf.id, kf.anchor_step) for kf in kfs] == [(versions[0][0], 14)]
    assert kfs[0].version == versions[0][1] + 1             # the version in force, re-traced
    assert {12, 13} <= set(session.db.rechecks(DESKTOP, VIEW))
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    assert sorted(int(c["step"]) for c in session.db.conflicts(DESKTOP, VIEW, open_only=True)
                  if c["instance"] == CHASSIS) == [12, 13]


def test_esc_takes_the_question_back_and_keeps_the_layer(window):
    session = window.session
    confirm_12_and_13(window)
    goto(window, 10)
    paint(window, CHASSIS, NEW)
    window.act_commit_all()
    assert window.all_asked()

    window.act_clear_edit()                                 # Esc: the question goes ...
    assert not window.all_asked() and not window.scope_bar.isVisibleTo(window)
    assert window.status_message() == ALL_CANCELLED
    assert session.editing_instance == CHASSIS             # ... the layer stays
    assert window.scope_all_button.text() == ALL_BUTTON
    assert anchors(session, CHASSIS) == [14]
    window.act_clear_edit()                                 # a second Esc discards it
    assert session.editing_instance is None
    assert anchors(session, CHASSIS) == [14]


def test_ctrl_enter_with_nothing_confirmed_in_reach_writes_at_once(window):
    session = window.session
    goto(window, 10)
    paint(window, CHASSIS, NEW)
    press(window, "act_commit_all")
    assert session.editing_instance is None and not window.all_asked()
    assert anchors(session, CHASSIS) == [14]
    for step in (1, 10, 14):
        assert np.array_equal(shape_at(session, step, CHASSIS), NEW), step


def test_a_size_warning_raised_by_ctrl_enter_names_it_and_its_button_presses_it(window):
    """One scope, two keys: the warning remembers which one asked."""
    session = window.session
    goto(window, 10)
    paint(window, CHASSIS, rect(30, 30, 33, 33))           # 9 px: implausibly small
    window.act_commit_all()
    assert window._pending_warning is not None
    assert "Ctrl+Enter 仍然提交" in window.warn_bar_text()
    assert window.warn_commit_button.text() == "Ctrl+Enter 仍然提交"
    plan = window.guide_plan()
    assert "再按一次 Ctrl+Enter" in plan.now and plan.action == "commit_all"
    window._answer_warning()                                # the bar's button
    assert session.editing_instance is None, window.status_message()
    assert anchors(session, CHASSIS) == [14]               # 改全部: the re-trace
    # Enter's own warning names Enter
    paint(window, CHASSIS, rect(31, 31, 33, 33))           # 4 px
    window.act_commit()
    assert window._pending_warning is not None, window.status_message()
    assert window.warn_commit_button.text() == "Enter 仍然提交"
    window._answer_warning()
    assert anchors(session, CHASSIS) == [10, 14]           # Enter: from 10 on


def test_ctrl_enter_on_the_anchor_frame_is_enter_and_says_so(window):
    session = window.session
    goto(window, 14)
    paint(window, CHASSIS, NEW)
    assert session.enter_scope() == api.SCOPE_KEYFRAME
    press(window, "act_commit_all")
    assert session.editing_instance is None
    assert anchors(session, CHASSIS) == [14]
    assert window.status_message() == ALL_IS_ENTER


def test_ctrl_enter_on_an_untouched_layer_says_there_is_nothing(window):
    goto(window, 10)
    window.on_request_edit(CHASSIS)
    reported: list[str] = []
    window.report = reported.append
    window.handle_key(key_event("Ctrl+Return"))             # the greyed button's reason
    assert reported and reported[-1].startswith("改全部：没有可提交的修改")
    window.act_commit_all()                                 # the slot says it too
    assert reported[-1] == NOTHING_TO_COMMIT
    assert not window.all_asked()


def test_enter_on_a_layering_suggestion_writes_the_shape_from_here_on(window, monkeypatch):
    session = window.session
    goto(window, 10)
    paint(window, CHASSIS, NEW)
    monkeypatch.setattr(session, "suggest_scope", lambda *a, **k: "zorder:above:other")
    committed: list[str] = []
    monkeypatch.setattr(session, "commit_edit",
                        lambda scope, *a, **k: committed.append(scope) or {})
    window.act_commit()                                     # the bar
    assert "Ctrl+Enter 改全部" in window.scope_bar_text()
    window.act_commit()                                     # Enter: the shape, from 10 on
    assert committed == [api.SCOPE_SPLIT]

    committed.clear()
    paint(window, CHASSIS, NEW)
    window.act_commit()
    window.act_commit_all()                                 # Ctrl+Enter: the shape, everywhere
    assert committed == [api.SCOPE_KEYFRAME]


def test_the_close_dialogs_save_uses_enters_scope(window, monkeypatch):
    session = window.session
    goto(window, 10)
    paint(window, CHASSIS, NEW)
    committed: list[str] = []
    monkeypatch.setattr(session, "commit_edit",
                        lambda scope, *a, **k: committed.append(scope) or {})
    window.commit_with_suggested_scope()
    assert committed == [api.SCOPE_SPLIT]


def test_ctrl_k_is_enter(window):
    assert A.action_for(Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier).name == "commit"
    assert "commit_split" not in {a.name for a in A.ACTIONS}
    session = window.session
    goto(window, 10)
    paint(window, CHASSIS, NEW)
    window.handle_key(key_event("Ctrl+K"))
    if window._pending_warning is not None:
        window.handle_key(key_event("Ctrl+K"))
    assert session.editing_instance is None, window.status_message()
    assert anchors(session, CHASSIS) == [10, 14]


def test_ctrl_enter_is_bound_in_annotate_mode_only():
    action = A.action_named("commit_all")
    assert action.keys == ("Ctrl+Return", "Ctrl+Enter") and action.slot == "act_commit_all"
    assert action.modes == (A.MODE_ANNOTATE,)
    assert A.key_caption(action) == "Ctrl+Enter"
    for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
        assert A.action_for(key, Qt.KeyboardModifier.ControlModifier).name == "commit_all"
        assert A.action_for(key, Qt.KeyboardModifier.ControlModifier, A.MODE_REVIEW) is None


def test_commit_all_takes_the_palette_slot_of_the_split(window):
    """「改全部」 where 「从这帧起新版本」 was: the 「操作」 group keeps its seven buttons.

    (An ASCII name: ``tmp_path`` is named after the test, and OpenCV cannot
    write the scene's frames under a path with Chinese in it.)
    """
    assert A.PALETTE_ACTIONS == ("commit", "commit_override", "commit_all", "confirm",
                                 "undo", "redo", "toggle_heat")
    button = window.palette.button("commit_all")
    assert button is not None and button.isVisibleTo(window.palette)
    assert A.short_parts(A.action_named("commit_all")) == ("改全部", "")
    assert "Ctrl+Enter" in button.toolTip()
    assert "commit_split" not in window.palette.buttons()
    # greyed with its reason on an untouched layer; live with pixels painted
    goto(window, 10)
    window.on_request_edit(CHASSIS)
    window.refresh_guidance()
    assert not button.isEnabled()
    window.set_editing_mask(NEW, undoable=True)
    window.refresh_guidance()
    assert button.isEnabled(), button.reason()


def test_the_scope_bar_offers_ctrl_enter_not_ctrl_k(window):
    texts = [b.text() for b in window.scope_bar.findChildren(QPushButton)]
    assert ALL_BUTTON in texts and not any("Ctrl+K" in t for t in texts)


# --------------------------------------------------------------------------- #
# the gallery: Enter is a plain commit, 改全部 has nothing to reach
# --------------------------------------------------------------------------- #
@pytest.fixture
def gallery_window(qapp, tmp_path: Path):
    from tda.core.truth import TruthService

    cfg = app_scene.make_paths(tmp_path)
    Path(cfg["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    db, tax, cache_dir = GS.make_db(tmp_path, db_path=cfg["db_path"],
                                    cache_dir=Path(cfg["cache_dir"]))
    made = AnnotationSession(db, tax, TruthService(db, tax), str(cache_dir), "tester")
    made.sweeper_enabled = False
    made.open(GS.DESKTOP, GS.VIEW)
    win = MainWindow(made, cfg, "tester", sam_queue=app_scene.StubSamQueue())
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    yield win
    app_scene.close_window(win)


@pytest.mark.parametrize("key", ["Return", "Ctrl+K"])
def test_enter_on_a_shot_commits_and_re_traces(gallery_window, key):
    win = gallery_window
    part = win.task_card.current_instance()
    for mask in (GS.rect(4, 4, 20, 20), GS.rect(6, 6, 24, 24)):
        paint(win, part, mask)
        win.handle_key(key_event(key))
        if win._pending_warning is not None:
            win.handle_key(key_event(key))
        assert win.session.editing_instance is None, win.status_message()
        assert "refused" not in win.status_message()
    kfs = win.session.db.keyframes(GS.DESKTOP, GS.VIEW, part)
    assert [(kf.anchor_step, kf.version) for kf in kfs] == [(1, 2)]


def test_ctrl_enter_on_a_shot_is_refused_with_its_reason(gallery_window):
    win = gallery_window
    part = win.task_card.current_instance()
    paint(win, part, GS.rect(4, 4, 20, 20))
    reported: list[str] = []
    win.report = reported.append
    win.handle_key(key_event("Ctrl+Return"))
    assert reported and GALLERY_REFUSALS["commit_all"] in reported[-1]
    assert win.session.editing_instance == part
    win.refresh_guidance()
    assert not win.palette.button("commit_all").isEnabled()


# --------------------------------------------------------------------------- #
# the guide's lines
# --------------------------------------------------------------------------- #
FRESH = G.GuideFacts(step=42, neighbour=None)


def test_the_guide_names_enter_for_a_split_row_and_ctrl_enter_for_its_question():
    split = G.plan_for(replace(FRESH, editing="x", layer_pixels=True, layer_dirty=True,
                               editing_kind=api.KIND_SPLIT_KEYFRAME))
    assert "按 Enter（从这帧起新版本）" in split.now and "Ctrl+K" not in split.now
    assert split.action == "commit"
    assert G.COMMIT_SPLIT == "Enter 提交（从这帧起新版本）"
    assert "走过的帧不变" in G.COMMIT and "Ctrl+K" not in G.COMMIT
    asked = G.plan_for(replace(FRESH, editing="x", layer_pixels=True, layer_dirty=True,
                               all_asked=True))
    assert asked.phase == G.PHASE_SCOPE and asked.action == "commit_all"
    assert "再按 Ctrl+Enter" in asked.now and "Esc" in asked.now
    warned = G.plan_for(replace(FRESH, editing="x", layer_pixels=True, layer_dirty=True,
                                warning=True, warning_scope=api.SCOPE_KEYFRAME,
                                warning_action="commit_all"))
    assert "再按一次 Ctrl+Enter" in warned.now and warned.action == "commit_all"
    entered = G.plan_for(replace(FRESH, editing="x", layer_pixels=True, layer_dirty=True,
                                 warning=True, warning_scope=api.SCOPE_SPLIT))
    assert "再按一次 Enter" in entered.now and entered.action == "commit"


# --------------------------------------------------------------------------- #
# the documents
# --------------------------------------------------------------------------- #
def _squash(text: str) -> str:
    return "".join(text.split())


def test_the_task_card_and_the_banners_say_enter_for_a_split_row():
    from tda.ui.app_guide import BANNER_CUT_COPY, BANNER_LOADED_SPLIT, BANNER_SPLIT
    from tda.ui.panels.taskcard import CUT_COPY_SENTENCE, ICON_LEGEND, KIND_SENTENCES

    for text in (KIND_SENTENCES[api.KIND_SPLIT_KEYFRAME], CUT_COPY_SENTENCE, ICON_LEGEND,
                 BANNER_SPLIT, BANNER_LOADED_SPLIT, BANNER_CUT_COPY):
        assert "Ctrl+K" not in text, text
        assert "Enter" in text, text


def test_the_guide_says_enter_is_from_this_frame_on():
    guide = _squash(GUIDE.read_text(encoding="utf-8"))
    assert _squash("**`Enter` 提交**：从这帧起生效，走过的帧不变") in guide
    assert _squash("| `split` | `Enter`") in guide
    assert _squash("| `keyframe` | `Ctrl+Enter`「改全部」") in guide
    assert _squash("（`Ctrl+K` 同 `Enter`；改全部用 `Ctrl+Enter`）") in guide


def test_the_quickstart_says_enter_is_from_this_frame_on():
    quick = _squash(QUICKSTART.read_text(encoding="utf-8"))
    assert _squash("| 形状从这一帧起变了（绝大多数情况） | `Enter`「提交」 |") in quick
    assert _squash("`Ctrl+Enter`「改全部」") in quick
    assert "不要按`Enter`" not in quick


def test_the_spec_has_the_ruling():
    spec = _squash(SPEC.read_text(encoding="utf-8"))
    assert _squash("**v1.14（2026-10-16）**：**`Enter` 默认「从这帧起」**") in spec
    assert _squash("| **拆分关键帧** | 从当前步起按浏览方向生成新版本 | `Enter`") in spec
