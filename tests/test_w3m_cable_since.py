"""W3m: 「被线挡住」 backwards -- from the frame on screen back to the start, undo-able from the middle, and from Review.

The user (2026-11-01, after a day with W3i's tool): 「这个不可见，应该是向前的，比如说我在第24步选定了他不可见，
应该是从24-1 他不可见 … 我需要可以从中间的某一步，再次选中它，然后把他变为可见 … 我需要能在review的部分就能点」.
Pinned here:

* the span rule (:func:`tda.ui.session_cable.since_span`, the mirror of ``until_span``): from the frame on screen
  back to the first annotatable frame on which the part is in its present state -- the frame after its previous own
  state change, or the view's first frame -- skipping ``ignore`` steps and frames without an image (a change on a
  skipped step still counts), across pose segments;
* one ``set_frame_overrides`` op with one inverse for either gesture; the withdrawal's inverse holds the rows it
  removed byte for byte; the 24-then-9 scenario (10..24 stay hidden, 1..9 are visible);
* refusals by name for 完全 (no shape), nothing written; the withdrawal never refused, no op when nothing is held;
* confirmed frames re-signed both ways; the Review path (label + ``cable`` stamps on the confirmed frames; the
  withdrawal with ``computed`` stamps where the compiler then says visible), the line leaving and re-entering the
  queue, a second click a no-op;
* no new key: ``KEYLESS`` grows by two; the two dropdown entries, the two right-click entries, the two Review
  buttons naming the frame on screen; the palette still fits at 1920x1000; the docs.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import vis_review_scene as V
import vis_runs_scene as RS
from app_scene import StubSamQueue, close_window, make_paths
from app_scene import make_session as make_app_session
from session_scene import (
    CHASSIS,
    DESKTOP,
    FAN_CONNECTOR,
    VIEW,
    make_session,
    rect,
    seed_shapes,
)
from tda.core import masks
from tda.core import vis_review as VR
from tda.core.model import OCCLUDER_CABLE, FrameKey, ShapeKeyframe, ShapePart
from tda.core.occluder import FULL, cable_label
from tda.ui import app_actions as A
from tda.ui import session_api as api
from tda.ui import visreview_text as T
from tda.ui.app import MainWindow
from tda.ui.app_cable import SINCE_PICK_FIRST
from tda.ui.panels.instances import (
    CABLE_SINCE_MENU,
    CABLE_SINCE_VISIBLE_MENU,
    CABLE_UNTIL_MENU,
)
from tda.ui.panels.palette import DROPDOWNS
from tda.ui.session_api import SessionRefusal
from tda.ui.session_cable import (
    CABLE_NEEDS_SHAPE,
    CABLE_OTHER_SEGMENT,
    OP_RANGE,
    RULE_SINCE,
    SinceSpan,
    since_span,
)
from tda.ui.session_visreview import OP_KIND as VIS_OP

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
HANDOFF = ROOT / "docs" / "HANDOFF_2026-10-28.md"
GUIDE = ROOT / "docs" / "annotation_guide.md"

LATCH = "ram_latch.01"          # closed until the log opens it at 14
TINY = V.TINY                   # connector.03: too_small, in the queue proper
HIDDEN = V.HIDDEN               # ram_latch.03: wholly under a screw -- computed occluded_full


def key(step: int) -> FrameKey:
    return FrameKey(DESKTOP, int(step), VIEW)


def label_at(session, step: int, instance: str):
    return cable_label(session.db.frame_overrides(key(step)).get(instance))


def labels(session, instance: str, steps=range(1, 15)) -> list:
    return [label_at(session, s, instance) for s in steps]


def ops(session, kind: str) -> list[dict]:
    """The logged ops of ``kind``, oldest first (``Db.ops`` lists the newest first)."""
    found = [op for op in session.db.ops(DESKTOP, VIEW, limit=1000) if op["kind"] == kind]
    return sorted(found, key=lambda op: int(op["id"]))


def stamps(session, instance: str, steps=range(10, 15)) -> list:
    return [(session.db.instance_flags(key(s), instance) or {}).get("verdict") for s in steps]


@pytest.fixture
def scene(tmp_path):
    """The D13 scene, every part of step 10 drawn (anchored at 14), standing on 10."""
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    seed_shapes(session, 10)
    session.goto(10, force=True)
    yield session
    session.close(force=True)


@pytest.fixture
def confirmed(tmp_path):
    """The U24 scene: frames 14-10 confirmed, standing on 14 (the user meets the hidden part late)."""
    session = V.build(tmp_path)
    session.goto(14, force=True)
    yield session
    session.close(force=True)


# --------------------------------------------------------------------------- #
# the span rule
# --------------------------------------------------------------------------- #
def test_the_span_runs_back_to_the_frame_after_the_parts_previous_state_change(scene):
    # the fan connector is unplugged at 12: standing on 13, it has been so since 12 (11 was plugged)
    span = since_span(scene.db, scene.tax, key(13), FAN_CONNECTOR)
    assert span == SinceSpan((12, 13), 11, ("state", "plugged", "unplugged"), 1)
    # the latch the log opens at 14: open since 14 only
    span = since_span(scene.db, scene.tax, key(14), LATCH)
    assert span == SinceSpan((14,), 13, ("state", "closed", "open"), 1)
    # plugged from the view's first frame: back to frame 1, no change found
    assert scene.cable_since_span(FAN_CONNECTOR) == SinceSpan(tuple(range(1, 11)), None, None, 1)
    assert since_span(scene.db, scene.tax, key(10), CHASSIS).steps == tuple(range(1, 11))


def test_missing_and_ignore_steps_are_left_out_but_a_change_on_one_still_bounds_the_span(tmp_path):
    session = make_session(tmp_path, missing=(5,))
    session.sweeper_enabled = False
    try:
        span = since_span(session.db, session.tax, key(10), CHASSIS)
        assert span.steps == (1, 2, 3, 4, 6, 7, 8, 9, 10) and span.since is None
        steps = session.db.steps(DESKTOP)
        for rec in steps:
            if rec.step in (2, 12):
                rec.step_type = "ignore"
        session.db.replace_steps(DESKTOP, steps, session.db.actions(DESKTOP))
        session._invalidate()
        span = since_span(session.db, session.tax, key(10), CHASSIS)
        assert span.steps == (1, 3, 4, 6, 7, 8, 9, 10)         # no ignore step, no missing frame
        # the fan connector is unplugged on the ignored step 12: the change still counts
        span = since_span(session.db, session.tax, key(13), FAN_CONNECTOR)
        assert span == SinceSpan((13,), 11, ("state", "plugged", "unplugged"), 1)
    finally:
        session.close(force=True)


def test_the_span_crosses_pose_segments_where_ctrl_backtick_stops(scene):
    db = scene.db
    db.set_pose_segment(DESKTOP, VIEW, 1, 1, 7, 7, None, None)
    db.set_pose_segment(DESKTOP, VIEW, 2, 8, 14, 14, None, None)
    # the chassis drawn in the second segment too (its own keyframe, as U37 asks); the
    # seeded keyframes are the first segment's
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=CHASSIS, desktop=DESKTOP, view=VIEW, pose_segment=2,
        anchor_step=14, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(rect(4, 4, 40, 40)))]))
    scene._invalidate()
    scene.goto(10, force=True)
    assert scene.cable_bounds() == (8, 14)
    with pytest.raises(SessionRefusal) as err:
        scene.set_cable_range(CHASSIS, FULL, 1)                # Ctrl+` stops at the segment
    assert str(err.value) == CABLE_OTHER_SEGMENT.format(step=1, low=8, high=14)
    assert since_span(db, scene.tax, key(10), CHASSIS).segments == 2
    result = scene.set_cable_since(CHASSIS)
    assert result["frames"] == list(range(1, 11)) and result["segments"] == 2
    assert result["since"] is None
    assert labels(scene, CHASSIS) == [FULL] * 10 + [None] * 4
    [op] = ops(scene, OP_RANGE)
    assert op["payload"]["segments"] == 2 and op["payload"]["rule"] == RULE_SINCE
    assert scene.undo()
    assert labels(scene, CHASSIS) == [None] * 14


# --------------------------------------------------------------------------- #
# one op, one inverse; the 24-then-9 scenario
# --------------------------------------------------------------------------- #
def test_the_op_is_one_set_frame_overrides_with_one_inverse_naming_the_rule(scene):
    result = scene.set_cable_since(FAN_CONNECTOR)
    assert result["frames"] == list(range(1, 11)) and result["since"] is None
    assert result["change"] is None and result["segments"] == 1 and result["rule"] == RULE_SINCE
    assert result["label"] == FULL and result["changed"]
    assert labels(scene, FAN_CONNECTOR) == [FULL] * 10 + [None] * 4
    [op] = ops(scene, OP_RANGE)
    assert op["payload"]["rule"] == RULE_SINCE and op["payload"]["since"] is None
    assert op["inverse"]["rule"] == RULE_SINCE          # both directions name the rule
    assert [f["step"] for f in op["payload"]["frames"]] == list(range(1, 11))
    assert all(not f["exists"] for f in op["inverse"]["frames"])
    for step in (1, 5, 10):
        assert scene.db.frame_overrides(key(step))[FAN_CONNECTOR].occluder == OCCLUDER_CABLE
    assert scene.undo()
    assert labels(scene, FAN_CONNECTOR) == [None] * 14
    assert scene.redo()
    assert labels(scene, FAN_CONNECTOR) == [FULL] * 10 + [None] * 4
    assert len(ops(scene, OP_RANGE)) == 1


def test_visible_again_from_the_middle_leaves_the_later_frames_hidden(scene):
    """「24 起往前都挡住」 then 「9 起往前都可见」: 10..24 stay hidden, 1..9 are visible."""
    scene.set_cable_since(CHASSIS)                          # 1..10 hidden
    scene.goto(5, force=True)
    result = scene.set_cable_since(CHASSIS, None)
    assert result["frames"] == [1, 2, 3, 4, 5] and result["span"] == [1, 2, 3, 4, 5]
    assert result["withdrawn"] == [FULL] and result["label"] is None and result["changed"]
    assert result["rule"] == RULE_SINCE and result["since"] is None
    assert labels(scene, CHASSIS) == [None] * 5 + [FULL] * 5 + [None] * 4
    hidden, cleared = ops(scene, OP_RANGE)
    # the withdrawal's inverse holds the rows it removed, byte for byte
    written = {f["step"]: f for f in hidden["payload"]["frames"]}
    assert cleared["inverse"]["frames"] == [written[s] for s in (1, 2, 3, 4, 5)]
    assert all(not f["exists"] for f in cleared["payload"]["frames"])
    # one Ctrl+Z brings the five back; one more takes all ten away
    assert scene.undo()
    assert labels(scene, CHASSIS) == [FULL] * 10 + [None] * 4
    assert scene.undo()
    assert labels(scene, CHASSIS) == [None] * 14
    assert scene.redo() and scene.redo()
    assert labels(scene, CHASSIS) == [None] * 5 + [FULL] * 5 + [None] * 4


def test_frames_without_a_shape_refuse_the_label_by_name_and_the_withdrawal_is_never_refused(tmp_path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    try:
        # the chassis is drawn in the second pose segment (8-14) only: the first segment's
        # frames 1-7 have no shape of it (a keyframe never reaches across a segment)
        session.db.set_pose_segment(DESKTOP, VIEW, 1, 1, 7, 7, None, None)
        session.db.set_pose_segment(DESKTOP, VIEW, 2, 8, 14, 14, None, None)
        session.db.add_keyframe(ShapeKeyframe(
            id=None, instance=CHASSIS, desktop=DESKTOP, view=VIEW, pose_segment=2,
            anchor_step=14, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(rect(4, 4, 40, 40)))]))
        session._invalidate()
        session.goto(10, force=True)
        with pytest.raises(SessionRefusal) as err:
            session.set_cable_since(CHASSIS)
        assert "第 1–7 步" in str(err.value) and CABLE_NEEDS_SHAPE in str(err.value)
        assert not ops(session, OP_RANGE)
        assert labels(session, CHASSIS) == [None] * 14
        # 撤回 is never refused: nothing held, nothing written, no op
        result = session.set_cable_since(CHASSIS, None)
        assert result["frames"] == [] and not result["changed"] and result["op"] is None
        assert not ops(session, OP_RANGE)
        # over frames that carry the label it withdraws them, the shapeless ones left alone
        assert session.set_cable_range(CHASSIS, FULL, 8)["frames"] == [8, 9, 10]
        result = session.set_cable_since(CHASSIS, None)
        assert result["frames"] == [8, 9, 10] and result["withdrawn"] == [FULL]
        assert result["span"] == list(range(1, 11))
        assert labels(session, CHASSIS) == [None] * 14
    finally:
        session.close(force=True)


def test_nothing_to_change_makes_no_op_either_way(scene):
    scene.set_cable_since(CHASSIS)
    before = len(ops(scene, OP_RANGE))
    again = scene.set_cable_since(CHASSIS)
    assert again["frames"] == [] and not again["changed"] and again["op"] is None
    scene.set_cable_since(CHASSIS, None)
    nothing = scene.set_cable_since(CHASSIS, None)
    assert nothing["frames"] == [] and not nothing["changed"] and nothing["op"] is None
    assert len(ops(scene, OP_RANGE)) == before + 1


# --------------------------------------------------------------------------- #
# confirmed frames
# --------------------------------------------------------------------------- #
def test_confirmed_frames_are_re_signed_both_ways_and_stay_confirmed(confirmed):
    session = confirmed
    result = session.set_cable_since(TINY)
    assert result["frames"] == list(range(1, 15)) and result["since"] is None
    assert result["accepted"] == [10, 11, 12, 13, 14] and result["withheld"] == []
    for step in range(10, 15):
        assert session.db.get_frame(key(step))["review_status"] == "verified"
        assert session.db.compiled_label(key(step), TINY)["visibility"] == FULL
    assert not session.db.conflicts(DESKTOP, VIEW, open_only=True)
    # visible again from 12: the pixels the 完全 replaced were confirmed before it, so the
    # three frames are signed (U15 round 4, _confirmed_once), 13 and 14 keep the label
    session.goto(12, force=True)
    result = session.set_cable_since(TINY, None)
    assert result["frames"] == list(range(1, 13)) and result["withdrawn"] == [FULL]
    assert result["accepted"] == [10, 11, 12] and result["withheld"] == []
    for step in (10, 11, 12):
        assert session.db.get_frame(key(step))["review_status"] == "verified"
        assert session.db.compiled_label(key(step), TINY)["visibility"] == "too_small"
    for step in (13, 14):
        assert session.db.compiled_label(key(step), TINY)["visibility"] == FULL
    assert not session.db.conflicts(DESKTOP, VIEW, open_only=True)
    assert session.undo()
    for step in range(10, 15):
        assert session.db.get_frame(key(step))["review_status"] == "verified"
        assert session.db.compiled_label(key(step), TINY)["visibility"] == FULL


# --------------------------------------------------------------------------- #
# the Review path
# --------------------------------------------------------------------------- #
def test_the_review_button_labels_back_from_the_frame_on_screen_and_stamps_the_confirmed_frames(confirmed):
    session = confirmed
    session.set_visibility_live(True)
    assert any(e["instance"] == TINY for e in session.visibility_queue())
    result = session.review_cable_since(TINY, 14)
    assert result["frames"] == list(range(1, 15)) and result["stamped"] == [10, 11, 12, 13, 14]
    assert result["moved"] == list(range(1, 15)) and result["accepted"] == [10, 11, 12, 13, 14]
    assert result["verdict"] == VR.VERDICT_CABLE and result["rule"] == RULE_SINCE
    assert result["since"] is None and result["segments"] == 1
    assert labels(session, TINY) == [FULL] * 14
    assert stamps(session, TINY) == [VR.VERDICT_CABLE] * 5
    assert not any(e["instance"] == TINY for e in session.visibility_queue())
    [op] = [o for o in session.db.ops(DESKTOP, VIEW, limit=50) if o["kind"] == VIS_OP]
    assert op["payload"]["rule"] == RULE_SINCE and op["payload"]["since"] is None
    # a second click writes nothing
    again = session.review_cable_since(TINY, 14)
    assert not again["changed"] and again["op"] is None
    assert len([o for o in session.db.ops(DESKTOP, VIEW, limit=50) if o["kind"] == VIS_OP]) == 1
    assert session.undo()
    assert labels(session, TINY) == [None] * 14 and stamps(session, TINY) == [None] * 5
    assert any(e["instance"] == TINY for e in session.visibility_queue())


def test_visible_again_from_review_stamps_computed_where_the_compiler_says_visible(confirmed):
    """connector.03 is 4 px wide: the compiler says too_small, so the withdrawn frames come
    back to the queue as computed lines (no stamp); a plainly visible part is stamped."""
    session = confirmed
    session.set_visibility_live(True)
    session.review_cable_since(TINY, 14)
    result = session.review_cable_since(TINY, 12, None)
    assert result["frames"] == list(range(1, 13)) and result["moved"] == list(range(1, 13))
    assert result["withdrawn"] == [FULL] and result["span"] == list(range(1, 13))
    assert result["stamped"] == [] and result["verdict"] == VR.VERDICT_COMPUTED
    assert result["accepted"] == [10, 11, 12]
    assert labels(session, TINY) == [None] * 12 + [FULL] * 2
    assert stamps(session, TINY) == [VR.VERDICT_CABLE] * 5         # the old stamps stay, stale
    # the too_small rows 10..12 are back in the queue, as a computed line; 13-14 stay answered
    mine = [e for e in session.visibility_queue() if e["instance"] == TINY]
    assert [e["steps"] for e in mine] == [[10, 11, 12]] and mine[0]["visibility"] == "too_small"
    assert not mine[0]["manual"]
    reviewed, total = session.visibility_progress()
    assert reviewed == total - 3 - sum(len(e["steps"]) for e in session.visibility_queue()
                                       if e["instance"] != TINY)
    hidden_op, clear_op = ops(session, VIS_OP)
    assert clear_op["payload"]["rule"] == RULE_SINCE and clear_op["payload"]["verdict"] == "computed"
    assert [f["step"] for f in clear_op["payload"]["frames"]] == list(range(1, 13))
    # the inverse holds the rows it removed and their flags as they were
    written = {f["step"]: f["override"] for f in hidden_op["payload"]["frames"]}
    assert [f["override"] for f in clear_op["inverse"]["frames"]] == [written[s] for s in range(1, 13)]
    assert session.undo()
    assert labels(session, TINY) == [FULL] * 14 and stamps(session, TINY) == [VR.VERDICT_CABLE] * 5
    assert not any(e["instance"] == TINY for e in session.visibility_queue())
    assert session.redo()
    assert labels(session, TINY) == [None] * 12 + [FULL] * 2
    # a plainly visible part: hidden, then seen again -- the confirmed frames are stamped
    # computed for visible and the line is gone (answered), a second click writes nothing
    plain = "connector.06"
    session.review_cable_since(plain, 14)
    result = session.review_cable_since(plain, 14, None)
    assert result["stamped"] == [10, 11, 12, 13, 14] and result["moved"] == list(range(1, 15))
    assert stamps(session, plain) == [VR.VERDICT_COMPUTED] * 5
    assert all(session.db.instance_flags(key(s), plain)["visibility"] == "visible" for s in range(10, 15))
    assert not any(e["instance"] == plain for e in session.visibility_queue())
    session.set_visibility_all(True)
    assert not any(e["instance"] == plain for e in session.visibility_queue())   # reviewed: not listed
    ops_before = len([o for o in session.db.ops(DESKTOP, VIEW, limit=50) if o["kind"] == VIS_OP])
    again = session.review_cable_since(plain, 14, None)
    assert not again["changed"] and again["op"] is None
    assert len([o for o in session.db.ops(DESKTOP, VIEW, limit=50) if o["kind"] == VIS_OP]) == ops_before
    assert session.undo()
    assert labels(session, plain) == [FULL] * 14 and stamps(session, plain) == [VR.VERDICT_CABLE] * 5


def test_the_review_withdrawal_puts_a_computed_occlusion_back_in_the_queue(confirmed):
    session = confirmed
    session.set_visibility_live(True)
    before = next(e for e in session.visibility_queue() if e["instance"] == HIDDEN)
    assert before["visibility"] == "occluded_full" and not before["manual"]
    session.review_cable_since(HIDDEN, 14)
    assert not any(e["instance"] == HIDDEN for e in session.visibility_queue())
    result = session.review_cable_since(HIDDEN, 14, None)
    assert result["stamped"] == [] and result["moved"] == list(range(1, 15))
    after = next(e for e in session.visibility_queue() if e["instance"] == HIDDEN)
    assert after["steps"] == before["steps"] and after["visibility"] == "occluded_full"


def test_the_review_buttons_are_refused_where_the_part_has_no_shape_and_from_an_unannotatable_frame(confirmed):
    session = confirmed
    with pytest.raises(SessionRefusal) as err:
        session.review_cable_since("nobody.01", 14)
    assert "帧" in str(err.value)
    assert not [o for o in session.db.ops(DESKTOP, VIEW, limit=50) if o["kind"] == VIS_OP]
    # the withdrawal over a part with nothing held: no op, not refused
    result = session.review_cable_since("nobody.01", 14, None)
    assert not result["changed"] and result["op"] is None


# --------------------------------------------------------------------------- #
# the words
# --------------------------------------------------------------------------- #
def test_the_status_line_says_the_span_and_where_it_stopped_going_back():
    note = T.since_note("接头 connector.03", {
        "frames": list(range(1, 25)), "segments": 2, "since": None,
        "accepted": list(range(18, 25))})
    assert note == ("接头 connector.03：第 1–24 步（跨 2 个位姿段）被线挡住（完全），"
                    "往前直到这个视角第一帧（第 1 步）；已确认的 7 帧已代签（Ctrl+Z 一次撤销）")
    assert T.since_where({"span": [6, 7], "since": 5, "change": ["state", "closed", "open"]}) == \
        "往前直到第 6 步（第 5 步它还是关上）"
    assert T.since_where({"span": [6, 7], "since": 5,
                          "change": ["placement", "on_bench", "in_chassis"]}) == "往前直到第 6 步（第 5 步它还在台面上）"
    cleared = T.since_note("x", {"frames": [1, 2, 3], "span": list(range(1, 10)), "since": None,
                                 "withdrawn": [FULL], "stamped": [1, 2]}, visible=True)
    assert cleared == "x：第 1–3 步撤回被线挡住（往前都可见），往前直到这个视角第一帧（第 1 步）；2 帧记为审过（Ctrl+Z 一次撤销）"
    assert T.SINCE_BUTTON.format(step=24) == "第 24 帧起往前都被线挡住"
    assert T.SINCE_VISIBLE_BUTTON.format(step=24) == "第 24 帧起往前都可见"


# --------------------------------------------------------------------------- #
# no new key: the dropdown entries, the right-click entries, the Review buttons
# --------------------------------------------------------------------------- #
def test_no_new_key_is_bound_and_the_keyless_set_grew_by_two():
    for name, slot in (("cable_since", "act_cable_since"),
                       ("cable_since_visible", "act_cable_since_visible")):
        action = A.action_named(name)
        assert action.keys == () and action.slot == slot and action.modes == (A.MODE_ANNOTATE,)
        assert name not in (A.PALETTE_TOOLS + A.PALETTE_ACTIONS + A.PALETTE_STATE
                            + A.PALETTE_OTHER + A.PALETTE_REVIEW)              # no row of its own
        assert A.key_caption(action) == ""
    assert {a.name for a in A.ACTIONS if not a.keys} == {
        "view_cmp_scan", "cable_until", "cable_since", "cable_since_visible"}
    assert DROPDOWNS["mark_absent"] == (("cable_until", "直到拔掉 / 拆下"),
                                        ("cable_since", "往前直到最开始都被线挡住"),
                                        ("cable_since_visible", "往前直到最开始都可见"))
    assert [e.slot for e in A.MENU_ENTRIES if e.name in ("cable_since_menu", "cable_since_visible_menu")] \
        == ["act_cable_since", "act_cable_since_visible"]
    slots = {a.slot for a in A.ACTIONS}
    assert not {"act_vis_cable_since", "act_vis_cable_since_visible"} & slots    # the panel's buttons
    # the key table itself is unchanged
    assert A.action_named("mark_absent").keys == ("`",)
    assert A.action_named("cable_range").keys == ("Ctrl+`",)
    assert A.action_named("cable_until").keys == ()
    assert A.action_named("vis_review_cable").keys == ("`",)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def answer_roi(win: MainWindow) -> None:
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()


def click(button) -> None:
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     button.rect().center())
    QApplication.processEvents()


@pytest.fixture
def window(qapp, tmp_path: Path):
    """The D13 scene in the window, shapes seeded (anchored at 14), at step 10."""
    session = make_app_session(tmp_path)
    session.sweeper_enabled = False
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(1400, 900)
    answer_roi(win)
    seed_shapes(win.session, 10)
    win.session.goto(10, force=True)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    yield win
    close_window(win)


def dropdown_state(win: MainWindow, item: str) -> tuple[bool, str]:
    win.refresh_guidance()
    return win.palette.dropdown_state(item)


def trigger_dropdown(win: MainWindow, item: str) -> None:
    menu = win.palette.dropdown_menu("mark_absent")
    [entry] = [a for a in menu.actions() if a.data() == item]
    entry.trigger()
    QApplication.processEvents()


def test_the_dropdown_entries_write_backwards_and_withdraw_from_the_middle(window):
    assert "cable_since" not in window.palette.buttons()
    assert "cable_since_visible" not in window.palette.buttons()
    menu = window.palette.dropdown_menu("mark_absent")
    assert [a.text() for a in menu.actions()] == ["直到拔掉 / 拆下", "往前直到最开始都被线挡住",
                                                   "往前直到最开始都可见"]
    window.instances.table().setCurrentCell(-1, -1)
    QApplication.processEvents()
    for item in ("cable_since", "cable_since_visible"):
        enabled, why = dropdown_state(window, item)
        assert not enabled and why.startswith(SINCE_PICK_FIRST), (item, why)
    assert window.instances.select_instance(FAN_CONNECTOR)
    assert dropdown_state(window, "cable_since") == (True, "")
    assert dropdown_state(window, "cable_since_visible") == (True, "")
    trigger_dropdown(window, "cable_since")
    assert labels(window.session, FAN_CONNECTOR) == [FULL] * 10 + [None] * 4
    assert "第 1–10 步被线挡住（完全），往前直到这个视角第一帧（第 1 步）" in window.status_message()
    # from the middle: visible again back to the start, the later frames stay hidden
    window.session.goto(5, force=True)
    QApplication.processEvents()
    assert window.instances.select_instance(FAN_CONNECTOR)
    trigger_dropdown(window, "cable_since_visible")
    assert labels(window.session, FAN_CONNECTOR) == [None] * 5 + [FULL] * 5 + [None] * 4
    assert "第 1–5 步撤回被线挡住（往前都可见）" in window.status_message()
    trigger_dropdown(window, "cable_since_visible")                 # nothing left to withdraw
    assert T.SINCE_VISIBLE_NOTHING.split("：", 1)[1].format(step=5) in window.status_message()
    assert window.act_undo() is None or True
    QApplication.processEvents()
    assert labels(window.session, FAN_CONNECTOR) == [FULL] * 10 + [None] * 4
    # the instance table's right-click entries on the row under the pointer
    menu = window.instances.context_menu_for(CHASSIS)
    assert [a.text() for a in menu.actions()] == [CABLE_UNTIL_MENU, CABLE_SINCE_MENU,
                                                   CABLE_SINCE_VISIBLE_MENU]
    [entry] = [a for a in menu.actions() if a.text() == CABLE_SINCE_MENU]
    entry.trigger()
    QApplication.processEvents()
    assert labels(window.session, CHASSIS) == [FULL] * 5 + [None] * 9
    [entry] = [a for a in menu.actions() if a.text() == CABLE_SINCE_VISIBLE_MENU]
    entry.trigger()
    QApplication.processEvents()
    assert labels(window.session, CHASSIS) == [None] * 14
    assert "撤回被线挡住（往前都可见）" in window.status_message()


def test_a_removed_part_cannot_be_hidden_but_its_withdrawal_is_only_a_no_op(window):
    window.session.goto(14, force=True)                 # the cooler is gone at 14
    QApplication.processEvents()
    window.act_cable_since("cpu_cooler.fan.01")
    assert "没有可标的" in window.status_message() or "不要它的形状" in window.status_message()
    assert labels(window.session, "cpu_cooler.fan.01") == [None] * 14
    window.act_cable_since_visible("cpu_cooler.fan.01")
    assert "没有改" in window.status_message()
    assert not ops(window.session, OP_RANGE)


@pytest.mark.parametrize("size", [(1920, 1000), (1920, 1080)])
def test_the_two_more_entries_add_no_height_and_the_group_stays_in_view(qapp, tmp_path, size):
    """The U13 pins: the dropdown is part of 「看不见」, so three entries move nothing."""
    session = make_app_session(tmp_path)
    session.sweeper_enabled = False
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    try:
        win.resize(*size)
        win.show()
        QApplication.processEvents()
        answer_roi(win)
        QApplication.processEvents()
        scroll = win.palette.scroll
        inner = scroll.widget()
        viewport = scroll.viewport().height()
        button = win.palette.button("mark_absent")
        bottom = button.mapTo(inner, button.rect().bottomLeft()).y()
        assert scroll.verticalScrollBar().value() == 0
        assert bottom <= viewport, (bottom, viewport)
        assert button.height() <= max(win.palette.button(n).height() for n in A.PALETTE_ACTIONS)
        assert len(win.palette.dropdown_items("mark_absent")) == 3
        assert not {"cable_since", "cable_since_visible"} & set(win.palette.buttons())
    finally:
        close_window(win)


@pytest.fixture
def review_window(qapp, tmp_path):
    session = RS.build(tmp_path)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(1400, 900)
    win.show()
    QApplication.processEvents()
    if win.roi_editing:
        win.act_clear_edit()
    QApplication.processEvents()
    win.session.goto(9, force=True)
    win.set_mode(A.MODE_REVIEW)
    win.review.show_visibility()
    QApplication.processEvents()
    yield win
    close_window(win)
    session.close(force=True)


def select_line(win: MainWindow, instance: str) -> None:
    lw = win.review.list_for(api.QUEUE_VISIBILITY)
    for row in range(lw.count()):
        lw.setCurrentRow(row)
        QApplication.processEvents()
        if (win.review.selected_visibility() or {}).get("instance") == instance:
            return
    raise AssertionError(f"no line of {instance}")


def test_the_review_buttons_name_the_frame_on_screen_and_act_from_it(review_window):
    win = review_window
    detail = win.review.run_detail
    assert detail.since_button.text() == T.SINCE_BUTTON.format(step=9)
    assert detail.since_visible_button.text() == T.SINCE_VISIBLE_BUTTON.format(step=9)
    assert not detail.since_button.isEnabled()                 # no line selected yet
    select_line(win, RS.MARKED)                                # jumps to the line's first frame, 10
    assert int(win.session.current().step) == 10
    assert detail.since_button.text() == "第 10 帧起往前都被线挡住" and detail.since_button.isEnabled()
    win.session.goto(12, force=True)                           # the user steps frames
    QApplication.processEvents()
    assert detail.since_button.text() == "第 12 帧起往前都被线挡住"
    assert detail.since_visible_button.text() == "第 12 帧起往前都可见"
    assert labels(win.session, RS.MARKED, range(10, 15)) == [FULL] * 5
    click(detail.since_visible_button)
    assert "往前都可见" in win.status_message() and RS.MARKED in win.status_message()
    assert labels(win.session, RS.MARKED, range(10, 15)) == [None, None, None, FULL, FULL]
    # these frames were confirmed *with* the label, so the pixels the withdrawal reveals were
    # never confirmed: U15 round 4 signs the frame on screen only (12: compiled visible, stamped
    # computed) and leaves 10-11 with their conflicts for Review -- the status line says so
    assert stamps(win.session, RS.MARKED) == [None, None, VR.VERDICT_COMPUTED, None, None]
    assert "第 10–11 帧露出了确认时没有的像素" in win.status_message()
    assert {int(c["step"]) for c in win.session.db.conflicts(RS.DESKTOP, VIEW, open_only=True)
            if c["instance"] == RS.MARKED} == {10, 11}
    mine = [e["steps"] for e in win.session.visibility_queue() if e["instance"] == RS.MARKED]
    assert mine == [[10, 11], [13, 14]]
    # one Ctrl+Z: the labels back, the conflicts closed, the line whole again
    win.act_undo()
    QApplication.processEvents()
    assert labels(win.session, RS.MARKED, range(10, 15)) == [FULL] * 5
    assert not win.session.db.conflicts(RS.DESKTOP, VIEW, open_only=True)
    # hidden again from frame 12 back over a plainly visible part: label on 1..12, the cable
    # stamp on the confirmed 10..12
    plain = "connector.06"
    win.review.set_list_visible(True)
    QApplication.processEvents()
    select_line(win, plain)
    win.session.goto(12, force=True)
    QApplication.processEvents()
    assert detail.since_button.text() == "第 12 帧起往前都被线挡住"
    click(detail.since_button)
    assert "往前直到这个视角第一帧" in win.status_message() and plain in win.status_message()
    assert labels(win.session, plain, range(1, 15)) == [FULL] * 12 + [None, None]
    assert stamps(win.session, plain) == [VR.VERDICT_CABLE] * 3 + [None, None]


# --------------------------------------------------------------------------- #
# the docs
# --------------------------------------------------------------------------- #
def test_the_docs_say_it():
    spec = SPEC.read_text(encoding="utf-8")
    states = spec.split("### 6.2 状态集与默认追踪策略", 1)[1].split("\n### ", 1)[0]
    [para] = [l for l in states.splitlines() if l.startswith("**被线挡住，往前直到最开始**")]
    for needed in ("v1.58，W3m", "since_span", "set_cable_since", "往前直到最开始都可见", "24 起往前都挡住",
                   "rule = since_start", "跨位姿段"):
        assert needed in para, needed
    review = spec.split("### 4.4 审阅模式", 1)[1].split("### 4.5", 1)[0]
    [bullet] = [l for l in review.splitlines() if l.startswith("- **这一帧起往前都被线挡住 / 都可见**")]
    for needed in ("v1.58，W3m", "review_cable_since", "RunDetail.since_button", "set_anchor", "stamp_covers",
                   "回到队列"):
        assert needed in bullet, needed
    entries = [line for line in spec.split("## 14. 修订记录", 1)[1].splitlines() if line.startswith("- ")]
    # on top when it landed; whatever lands later goes above it
    [entry] = [e for e in entries[:1] if e.startswith("- **v1.58（2026-11-01）**")]
    for needed in ("（W3m，", "since_span", "cable_since_start", "review_cable_since", "_review_clear_span",
                   "cable_since_visible", "KEYLESS", "tests/test_w3m_cable_since.py", "D22 scan", "D02 scan"):
        assert needed in entry, needed
    assert spec.index("- **v1.58") < spec.index("- **v1.57") < spec.index("- **v1.56")
    handoff = HANDOFF.read_text(encoding="utf-8")
    export = handoff.split("## 5. 导出与 HuggingFace 准备", 1)[1].split("\n## ", 1)[0]
    [line] = [l for l in export.splitlines() if "W3m" in l]
    for needed in ("v1.58", "往前直到最开始都被线挡住", "往前直到最开始都可见", "第 N 帧起往前"):
        assert needed in line, needed
    # the guide's generated table carries the two entries and the two right-click rows, no prose
    guide = GUIDE.read_text(encoding="utf-8")
    assert A.shortcut_markdown() in guide
    for action in ("cable_since", "cable_since_visible"):
        assert f"| `{A.NO_KEY_MARK}` {A.NO_KEY_TEXT} | {A.action_named(action).label_zh} |" in guide
    for entry in A.MENU_ENTRIES:
        if entry.name in ("cable_since_menu", "cable_since_visible_menu"):
            assert f"| `{entry.where}` |" in guide
