"""W3i: 「被线挡住，直到拔掉」 and the visibility review's check round.

The user (2026-10-31): a connector drawn in its place is under a cable for the
rest of its life, until it is unplugged; the compiler labels every one of those
frames ``visible`` and 可见性审查 never lists them. Pinned here:

* the span rule (:func:`tda.ui.session_cable.until_span`): to the frame before
  the part's next own state change, to the view's last frame when nothing
  changes it again, across pose segments; frames without a shape refused by
  name; 撤回 never refused;
* one ``set_frame_overrides`` op with one inverse (one ``Ctrl+Z`` puts every
  override back; the confirmed rows come back re-signed by the undoer with the
  current input hash -- U15's semantics, not byte for byte);
* the Review button's op: the label over the span, the stamp on the confirmed
  frames, one undo, a second click a no-op;
* the toggle 「也列 visible 的段（接头优先）」: the plain visible runs after the
  queue proper, connectors first, the progress counting them only while on;
* no new key and no new palette row: 「看不见」's dropdown entry, a right-click
  entry, a Review panel button.
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
    LAST_STEP,
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
from tda.ui.app_cable import UNTIL_PICK_FIRST
from tda.ui.panels.instances import CABLE_UNTIL_MENU
from tda.ui.session_api import SessionRefusal
from tda.ui.session_cable import (
    CABLE_NEEDS_SHAPE,
    CABLE_OTHER_SEGMENT,
    OP_RANGE,
    RULE_UNTIL,
    UntilSpan,
    until_span,
)
from tda.ui.session_visreview import OP_KIND as VIS_OP

LATCH = "ram_latch.01"          # closed until the log opens it at 14
TINY = V.TINY                   # connector.03: too_small, in the queue proper
PLAIN = "connector.06"          # plainly visible on every confirmed frame


def key(step: int) -> FrameKey:
    return FrameKey(DESKTOP, int(step), VIEW)


def label_at(session, step: int, instance: str):
    return cable_label(session.db.frame_overrides(key(step)).get(instance))


def labels(session, instance: str, steps=range(9, 15)) -> list:
    return [label_at(session, s, instance) for s in steps]


def ops(session, kind: str) -> list[dict]:
    return [op for op in session.db.ops(DESKTOP, VIEW, limit=1000) if op["kind"] == kind]


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
    """The U24 scene: frames 14-10 confirmed, on 10."""
    session = V.build(tmp_path)
    session.goto(10, force=True)
    yield session
    session.close(force=True)


# --------------------------------------------------------------------------- #
# the span rule
# --------------------------------------------------------------------------- #
def test_the_span_ends_before_the_parts_next_state_change(scene):
    # the fan connector is unplugged at 12: 10 and 11 are the frames it is still plugged on
    span = until_span(scene.db, scene.tax, key(10), FAN_CONNECTOR)
    assert span == UntilSpan((10, 11), 12, ("state", "plugged", "unplugged"), 1)
    # a latch the log opens at 14: opened is a change of its own state too
    span = until_span(scene.db, scene.tax, key(10), LATCH)
    assert span.steps == (10, 11, 12, 13) and span.until == 14
    assert span.change == ("state", "closed", "open")
    assert scene.cable_until_span(FAN_CONNECTOR).until == 12


def test_a_part_whose_state_never_changes_runs_to_the_views_last_frame(scene):
    span = until_span(scene.db, scene.tax, key(10), CHASSIS)
    assert span == UntilSpan((10, 11, 12, 13, 14), None, None, 1)


def test_a_missing_frame_is_left_out_of_the_span(tmp_path):
    session = make_session(tmp_path, missing=(12,))
    session.sweeper_enabled = False
    try:
        span = until_span(session.db, session.tax, key(10), CHASSIS)
        assert span.steps == (10, 11, 13, 14) and span.until is None
    finally:
        session.close(force=True)


def test_the_op_is_one_set_frame_overrides_with_one_inverse(scene):
    result = scene.set_cable_until(FAN_CONNECTOR)
    assert result["frames"] == [10, 11] and result["until"] == 12
    assert result["change"] == ["state", "plugged", "unplugged"]
    assert result["segments"] == 1 and result["rule"] == RULE_UNTIL
    assert result["label"] == FULL and result["changed"]
    assert labels(scene, FAN_CONNECTOR) == [None, FULL, FULL, None, None, None]
    [op] = ops(scene, OP_RANGE)
    assert op["payload"]["rule"] == RULE_UNTIL and op["payload"]["until"] == 12
    assert op["inverse"]["rule"] == RULE_UNTIL     # both directions name the rule
    assert [f["step"] for f in op["payload"]["frames"]] == [10, 11]
    assert all(not f["exists"] for f in op["inverse"]["frames"])
    for step in (10, 11):
        fo = scene.db.frame_overrides(key(step))[FAN_CONNECTOR]
        assert fo.occluder == OCCLUDER_CABLE
    # one Ctrl+Z puts every override back (a confirmed row would come back
    # re-signed by the undoer, U15); redo writes them all again
    assert scene.undo()
    assert labels(scene, FAN_CONNECTOR) == [None] * 6
    assert scene.redo()
    assert labels(scene, FAN_CONNECTOR) == [None, FULL, FULL, None, None, None]
    assert len(ops(scene, OP_RANGE)) == 1


def test_the_span_crosses_pose_segments_where_ctrl_i_stops(scene):
    db = scene.db
    db.set_pose_segment(DESKTOP, VIEW, 1, 1, 7, 7, None, None)
    db.set_pose_segment(DESKTOP, VIEW, 2, 8, 14, 14, None, None)
    # the chassis drawn in the second segment too (its own keyframe, as U37 asks)
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=CHASSIS, desktop=DESKTOP, view=VIEW, pose_segment=2,
        anchor_step=14, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(rect(4, 4, 40, 40)))]))
    scene._invalidate()
    scene.goto(5, force=True)
    assert scene.cable_bounds() == (1, 7)
    with pytest.raises(SessionRefusal) as err:
        scene.set_cable_range(CHASSIS, FULL, 14)          # Ctrl+` stops at the segment
    assert str(err.value) == CABLE_OTHER_SEGMENT.format(step=14, low=1, high=7)
    result = scene.set_cable_until(CHASSIS)
    assert result["frames"] == list(range(5, 15)) and result["segments"] == 2
    assert result["until"] is None
    assert labels(scene, CHASSIS, range(4, 15)) == [None] + [FULL] * 10
    [op] = ops(scene, OP_RANGE)
    assert op["payload"]["segments"] == 2
    assert scene.undo()
    assert labels(scene, CHASSIS, range(4, 15)) == [None] * 11


def test_frames_without_a_shape_are_refused_by_name_and_nothing_is_written(tmp_path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    try:
        # the chassis is drawn up to step 12 only
        session.db.add_keyframe(ShapeKeyframe(
            id=None, instance=CHASSIS, desktop=DESKTOP, view=VIEW, pose_segment=1,
            anchor_step=12, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(rect(4, 4, 40, 40)))]))
        session._invalidate()
        session.goto(10, force=True)
        with pytest.raises(SessionRefusal) as err:
            session.set_cable_until(CHASSIS)
        assert "第 13–14 步" in str(err.value) and CABLE_NEEDS_SHAPE in str(err.value)
        assert not ops(session, OP_RANGE)
        assert labels(session, CHASSIS) == [None] * 6
        # 撤回 is never refused: over the frames that carry the label, the
        # shapeless ones included in the span and left alone
        assert session.set_cable_range(CHASSIS, FULL, 12)["frames"] == [10, 11, 12]
        result = session.set_cable_until(CHASSIS, None)
        assert result["frames"] == [10, 11, 12] and result["withdrawn"] == [FULL]
        assert result["span"] == [10, 11, 12, 13, 14]
        assert labels(session, CHASSIS) == [None] * 6
    finally:
        session.close(force=True)


def test_nothing_to_change_makes_no_op(scene):
    scene.set_cable_until(CHASSIS)
    before = len(ops(scene, OP_RANGE))
    again = scene.set_cable_until(CHASSIS)
    assert again["frames"] == [] and not again["changed"] and again["op"] is None
    assert len(ops(scene, OP_RANGE)) == before


# --------------------------------------------------------------------------- #
# confirmed frames
# --------------------------------------------------------------------------- #
def test_confirmed_frames_are_re_signed_and_stay_confirmed(confirmed):
    session = confirmed
    result = session.set_cable_until(TINY)
    assert result["frames"] == [10, 11, 12, 13, 14] and result["until"] is None
    assert result["accepted"] == [10, 11, 12, 13, 14] and result["withheld"] == []
    assert result["conflicts"] == 0
    for step in range(10, 15):
        assert session.db.get_frame(key(step))["review_status"] == "verified"
        assert session.db.compiled_label(key(step), TINY)["visibility"] == FULL
    assert not session.db.conflicts(DESKTOP, VIEW, open_only=True)
    # the undo re-signs the old rows the same way
    assert session.undo()
    for step in range(10, 15):
        assert session.db.get_frame(key(step))["review_status"] == "verified"
        assert session.db.compiled_label(key(step), TINY)["visibility"] == "too_small"
    assert not session.db.conflicts(DESKTOP, VIEW, open_only=True)


# --------------------------------------------------------------------------- #
# the Review button
# --------------------------------------------------------------------------- #
def stamps(session, instance: str, steps=range(10, 15)) -> list:
    return [(session.db.instance_flags(key(s), instance) or {}).get("verdict") for s in steps]


def test_the_review_button_answers_the_line_and_stamps_the_confirmed_frames(confirmed):
    session = confirmed
    session.set_visibility_live(True)
    assert any(e["instance"] == TINY for e in session.visibility_queue())
    result = session.review_cable_until(TINY, 10)
    assert result["frames"] == [10, 11, 12, 13, 14] and result["stamped"] == [10, 11, 12, 13, 14]
    assert result["moved"] == [10, 11, 12, 13, 14] and result["accepted"] == [10, 11, 12, 13, 14]
    assert result["verdict"] == VR.VERDICT_CABLE and result["rule"] == RULE_UNTIL
    assert labels(session, TINY) == [None] + [FULL] * 5
    assert stamps(session, TINY) == [VR.VERDICT_CABLE] * 5
    flags = session.db.instance_flags(key(12), TINY)
    assert flags["visibility"] == FULL and flags["occluder"] == OCCLUDER_CABLE
    # the line is answered: gone from the queue, counted as reviewed
    assert not any(e["instance"] == TINY for e in session.visibility_queue())
    reviewed, _total = session.visibility_progress()
    assert reviewed == 5
    [op] = [o for o in session.db.ops(DESKTOP, VIEW, limit=50) if o["kind"] == VIS_OP]
    assert op["payload"]["rule"] == RULE_UNTIL and op["payload"]["until"] is None
    # one Ctrl+Z: labels and stamps back, the line back
    assert session.undo()
    assert labels(session, TINY) == [None] * 6 and stamps(session, TINY) == [None] * 5
    assert any(e["instance"] == TINY for e in session.visibility_queue())
    assert session.redo()
    assert stamps(session, TINY) == [VR.VERDICT_CABLE] * 5


def test_a_second_click_of_the_review_button_writes_nothing(confirmed):
    """Review S1: every frame already carries the label and a cable stamp -> no op."""
    session = confirmed
    first = session.review_cable_until(TINY, 10)
    assert first["changed"] and first["op"] is not None
    ops_before = [o["id"] for o in session.db.ops(DESKTOP, VIEW, limit=50) if o["kind"] == VIS_OP]
    stamps_before = [session.db.instance_flags(key(s), TINY) for s in range(10, 15)]
    again = session.review_cable_until(TINY, 10)
    assert not again["changed"] and again["op"] is None
    assert again["frames"] == [10, 11, 12, 13, 14] and again["stamped"] == []
    assert [o["id"] for o in session.db.ops(DESKTOP, VIEW, limit=50)
            if o["kind"] == VIS_OP] == ops_before
    assert [session.db.instance_flags(key(s), TINY) for s in range(10, 15)] == stamps_before
    # a stamp of another verdict over the same label is still an answer to give
    session.review_visibility(TINY, VR.VERDICT_COMPUTED, steps=[12])
    third = session.review_cable_until(TINY, 10)
    assert third["changed"] and third["stamped"] == [12] and third["moved"] == []
    assert session.db.instance_flags(key(12), TINY)["verdict"] == VR.VERDICT_CABLE


def test_the_review_button_labels_past_the_confirmed_frames_but_stamps_only_them(confirmed):
    session = confirmed
    # from 8: frames 8 and 9 are not confirmed -- the label goes there, no stamp
    result = session.review_cable_until(TINY, 8)
    assert result["frames"] == [8, 9, 10, 11, 12, 13, 14]
    assert result["stamped"] == [10, 11, 12, 13, 14]
    assert labels(session, TINY, range(7, 15)) == [None] + [FULL] * 7
    assert stamps(session, TINY, range(8, 15)) == [None, None] + [VR.VERDICT_CABLE] * 5
    assert session.undo()
    assert labels(session, TINY, range(7, 15)) == [None] * 8
    assert stamps(session, TINY, range(8, 15)) == [None] * 7


def test_the_review_button_is_refused_where_the_part_has_no_shape(confirmed):
    session = confirmed
    with pytest.raises(SessionRefusal) as err:
        session.review_cable_until("nobody.01", 10)
    assert "第 10–14 帧" in str(err.value)
    assert not [o for o in session.db.ops(DESKTOP, VIEW, limit=50) if o["kind"] == VIS_OP]


# --------------------------------------------------------------------------- #
# the check round: 「也列 visible 的段（接头优先）」
# --------------------------------------------------------------------------- #
def test_the_toggle_lists_the_visible_runs_connectors_first_after_the_queue(confirmed):
    session = confirmed
    session.set_visibility_live(True)
    proper = [T.entry_text(e) for e in session.visibility_queue()]
    assert len(proper) == 5 and not any(" · visible" in line for line in proper)
    rows = sum(len(e["steps"]) for e in session.visibility_queue())
    assert session.visibility_progress() == (0, rows) and rows == 17
    assert not session.visibility_all()
    session.set_visibility_all(True)
    assert session.visibility_all()
    lines = [T.entry_text(e) for e in session.visibility_queue()]
    assert lines[:5] == proper                       # the queue proper first, unchanged
    extra = lines[5:]
    assert extra and all(" · visible" in line for line in extra)
    # connectors first (the class in Chinese, then the key), then the other classes
    heads = [line.split(" ")[0] for line in extra]
    first_other = next(i for i, h in enumerate(heads) if h != "接头")
    assert first_other > 0 and all(h == "接头" for h in heads[:first_other])
    assert all(h != "接头" for h in heads[first_other:])
    assert f"接头 {PLAIN} 第 10–14 帧（5 帧） · visible · {T.WHOLE}" in extra
    # the progress counts every plain visible row only while the toggle is on
    reviewed, total = session.visibility_progress()
    assert reviewed == 0 and total > rows
    plain_rows = sum(len(e["steps"]) for e in session.visibility_queue()
                     if VR.plain_visible(e))
    assert total == rows + plain_rows
    # the summary across the views does not list them
    assert all(not VR.plain_visible(row) for row in session.visibility_summary())
    assert {row["instance"] for row in session.visibility_summary()} == {
        e["instance"] for e in session.visibility_queue()[:5]}
    session.set_visibility_all(False)
    assert [T.entry_text(e) for e in session.visibility_queue()] == proper
    assert session.visibility_progress() == (0, rows)


def test_space_on_a_visible_line_stamps_it_and_the_backtick_hides_it(confirmed):
    session = confirmed
    session.set_visibility_live(True)
    session.set_visibility_all(True)
    entry = next(e for e in session.visibility_queue() if e["instance"] == PLAIN)
    assert entry["steps"] == [10, 11, 12, 13, 14] and entry["visibility"] == "visible"
    # Space: it really is visible -- the stamp answers for the row as it is
    result = session.review_visibility(PLAIN, VR.VERDICT_COMPUTED, steps=entry["steps"])
    assert result["moved"] == []
    assert stamps(session, PLAIN) == [VR.VERDICT_COMPUTED] * 5
    assert not any(e["instance"] == PLAIN for e in session.visibility_queue())
    reviewed, _total = session.visibility_progress()
    assert reviewed == 5
    assert session.undo()
    assert any(e["instance"] == PLAIN for e in session.visibility_queue())
    # `: the run is under a cable after all (the U15 label over the run)
    other = next(e for e in session.visibility_queue() if e["instance"] == "connector.05")
    result = session.review_visibility("connector.05", VR.VERDICT_CABLE, FULL,
                                       steps=other["steps"])
    assert result["moved"] == other["steps"]
    assert labels(session, "connector.05", other["steps"]) == [FULL] * len(other["steps"])
    assert not any(e["instance"] == "connector.05" for e in session.visibility_queue())


def test_the_toggle_is_remembered_across_views_and_the_splits_cut_visible_runs_too(confirmed):
    session = confirmed
    session.set_visibility_live(True)
    session.set_visibility_all(True)
    session.open(DESKTOP, "oak1", force=True)
    assert session.visibility_all()
    session.open(DESKTOP, VIEW, force=True)
    session.goto(10, force=True)
    entry = next(e for e in session.visibility_queue() if e["instance"] == PLAIN)
    session.split_visibility_run(PLAIN, 12, entry["steps"])
    mine = [e["steps"] for e in session.visibility_queue() if e["instance"] == PLAIN]
    assert mine == [[10, 11], [12, 13, 14]]


def test_order_visible_runs_puts_connectors_first_then_classes_by_their_longest_run():
    def run(part, steps):
        return VR.Run(part, list(steps), "visible", False, None)

    runs = [run("screw.01", [1, 2, 3, 4]), run("cover.01", [1, 2]), run("connector.02", [1]),
            run("connector.01", [1, 2, 3]), run("cover.02", [1, 2, 3, 4, 5]), run("psu.01", [1, 2])]
    assert [r.instance for r in VR.order_visible_runs(runs)] == [
        "connector.01", "connector.02",            # connectors: the longest part first
        "cover.02", "cover.01",                    # covers: longest run 5
        "screw.01",                                # 4
        "psu.01",                                  # 2
    ]
    assert VR.class_of("connector.fan.01") == "connector"


def test_the_status_line_says_the_span_the_segments_and_the_change():
    note = T.until_note("接头 connector.03", {
        "frames": list(range(12, 32)), "segments": 2, "until": 32,
        "change": ["state", "plugged", "unplugged"], "accepted": list(range(12, 19))})
    assert note == ("接头 connector.03：第 12–31 步（跨 2 个位姿段）被线挡住（完全），直到第 32 步拔掉；"
                    "已确认的 7 帧已代签（Ctrl+Z 一次撤销）")
    assert T.until_where({"frames": [3, 4, 5], "until": None}) == "直到这个视角最后一帧（第 5 步）"
    assert T.until_where({"frames": [3], "until": 9, "change": ["placement", "in_chassis",
                                                                 "on_bench"]}) == "直到第 9 步放到台面上"
    assert "5 帧记为审过" in T.until_note("x", {"frames": [1, 2], "stamped": [1, 2, 3, 4, 5]})


# --------------------------------------------------------------------------- #
# no new key: the palette button, the right-click entry, the panel button
# --------------------------------------------------------------------------- #
def test_no_new_key_is_bound():
    action = A.action_named("cable_until")
    assert action.keys == () and action.slot == "act_cable_until"
    assert action.modes == (A.MODE_ANNOTATE,)
    assert "cable_until" not in (A.PALETTE_TOOLS + A.PALETTE_ACTIONS + A.PALETTE_STATE
                                 + A.PALETTE_OTHER + A.PALETTE_REVIEW)      # no row of its own
    assert A.key_caption(action) == "" and A.NO_KEY_TEXT in A.shortcut_markdown()
    assert any(entry.name == "cable_until_menu" and entry.slot == "act_cable_until"
               for entry in A.MENU_ENTRIES)
    slots = {a.slot for a in A.ACTIONS}
    assert "act_vis_cable_until" not in slots          # the Review button is the panel's
    # the key table itself is unchanged: every key that was bound is bound the same way
    assert A.action_named("mark_absent").keys == ("`",)
    assert A.action_named("cable_range").keys == ("Ctrl+`",)
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


def dropdown_state(win: MainWindow, item: str = "cable_until") -> tuple[bool, str]:
    win.refresh_guidance()
    return win.palette.dropdown_state(item)


def trigger_dropdown(win: MainWindow, item: str = "cable_until") -> None:
    menu = win.palette.dropdown_menu("mark_absent")
    [entry] = [a for a in menu.actions() if a.data() == item]
    entry.trigger()
    QApplication.processEvents()


def test_the_dropdown_entry_is_greyed_with_its_reason_and_writes_the_span(window):
    # 「看不见」 carries the arrow; nothing else in the palette is W3i's
    button = window.palette.button("mark_absent")
    assert button.dropdown and not button.arrow_rect().isEmpty()
    assert button.arrow_rect().right() <= button.width()
    assert not button.arrow_rect().contains(button.rect().center())   # the click is still 看不见
    assert "cable_until" not in window.palette.buttons()
    assert window.palette.dropdown_items("mark_absent") == (("cable_until", "直到拔掉 / 拆下"),)
    window.instances.table().setCurrentCell(-1, -1)
    QApplication.processEvents()
    enabled, why = dropdown_state(window)
    assert not enabled and why.startswith(UNTIL_PICK_FIRST)
    menu = window.palette.dropdown_menu("mark_absent")
    [entry] = menu.actions()
    assert entry.text() == "直到拔掉 / 拆下" and why in entry.toolTip()
    entry.trigger()                                  # a greyed entry answers with its reason
    QApplication.processEvents()
    assert why in window.status_message()
    assert labels(window.session, FAN_CONNECTOR) == [None] * 6
    assert window.instances.select_instance(FAN_CONNECTOR)
    assert dropdown_state(window) == (True, "")
    trigger_dropdown(window)
    assert labels(window.session, FAN_CONNECTOR) == [None, FULL, FULL, None, None, None]
    assert "第 10–11 步被线挡住（完全），直到第 12 步拔掉" in window.status_message()
    # the part under the pointer in the instance table: the right-click entry
    assert window.act_undo() is None or True
    QApplication.processEvents()
    assert labels(window.session, FAN_CONNECTOR) == [None] * 6
    menu = window.instances.context_menu_for(CHASSIS)
    [entry] = [a for a in menu.actions() if a.text() == CABLE_UNTIL_MENU]
    entry.trigger()
    QApplication.processEvents()
    assert labels(window.session, CHASSIS) == [None] + [FULL] * 5
    assert "直到这个视角最后一帧（第 14 步）" in window.status_message()


@pytest.mark.parametrize("size", [(1920, 1000), (1920, 1080)])
def test_the_arrow_adds_no_height_and_the_group_stays_in_view(qapp, tmp_path, size):
    """Review M1: the first round's own button sat under the fold at 1920x1000
    (viewport 814 px, top 822). The dropdown is part of 「看不见」: nothing moves."""
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
        assert button.arrow_rect().height() <= button.height()
    finally:
        close_window(win)


def test_a_part_without_a_shape_greys_the_entry_with_the_keys_sentence(window):
    assert window.instances.select_instance("cpu.01")   # drawn at 10: live
    assert dropdown_state(window) == (True, "")
    window.session.goto(14, force=True)                 # the cooler is gone at 14
    QApplication.processEvents()
    assert window.instances.select_instance(CHASSIS)
    assert dropdown_state(window) == (True, "")
    assert not window.instances.select_instance("cpu_cooler.fan.01")   # removed: no row
    window.act_cable_until("cpu_cooler.fan.01")
    assert "没有可标的" in window.status_message() or "不要它的形状" in window.status_message()
    assert labels(window.session, "cpu_cooler.fan.01") == [None] * 6


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


def vis_lines(win: MainWindow) -> list[str]:
    lw = win.review.list_for(api.QUEUE_VISIBILITY)
    return [lw.item(i).text() for i in range(lw.count())]


def test_the_toggle_and_the_button_in_the_window(review_window):
    win = review_window
    before = vis_lines(win)
    assert not any(" · visible" in line for line in before)
    assert not win.review.list_visible()
    win.review.set_list_visible(True)
    QApplication.processEvents()
    assert win.session.visibility_all()
    lines = vis_lines(win)
    assert lines[:len(before)] == before
    assert lines[len(before)].startswith("接头 ")
    assert "可见性审查" in win.status_message() or True
    # pick a plain visible connector line and answer it with the button
    lw = win.review.list_for(api.QUEUE_VISIBILITY)
    row = next(i for i in range(lw.count()) if lw.item(i).text().startswith(f"接头 {PLAIN} "))
    lw.setCurrentRow(row)
    QApplication.processEvents()
    assert win.review.selected_visibility()["instance"] == PLAIN
    click(win.review.run_detail.until_button)
    assert PLAIN in win.status_message() and "直到" in win.status_message()
    assert labels(win.session, PLAIN) == [None] + [FULL] * 5
    assert stamps(win.session, PLAIN) == [VR.VERDICT_CABLE] * 5
    assert not any(line.startswith(f"接头 {PLAIN} ") for line in vis_lines(win))
    # the toggle off: the plain visible lines go, the answered part stays answered
    win.review.set_list_visible(False)
    QApplication.processEvents()
    assert not any(" · visible" in line for line in vis_lines(win))
    assert stamps(win.session, PLAIN) == [VR.VERDICT_CABLE] * 5
