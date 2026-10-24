"""Task U23: 「主视角」 -- on an OAK / RealSense frame, where the part is in the scanner view.

The 14-step D13 scene of the window tests, every chassis part drawn in the
scanner view (``seed_shapes``, compiled and stored), plus a picture per step for
``oak1``, ``oak2`` and ``rs``; the window is opened on ``oak1``. The panel:

* is up on ``oak1`` / ``oak2`` / ``rs`` in Annotate and Review, down on the
  scanner view, in Steps and on a components-gallery shot; a close by the
  annotator is kept in the INI and the top bar's toggle brings it back;
* looks for the part being drawn, else the instance table's selection, else
  the task card's current row;
* shows the scanner frame of the same step (the nearest earlier one when that
  is missing or a dupli step, said in the caption) cropped around the part's
  stored scanner mask -- a click toggles the whole frame, the wheel zooms;
* says why when there is no outline: not on this scanner frame, never drawn in
  the scanner view, the scanner view not annotated at all;
* reads nothing while it is down, and nothing twice for one frame and part;
* leaves the tool palette's height alone (U13's 1920x1000 measurement).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import app_scene as AS
from app_scene import (
    COOLER,
    DESKTOP,
    LAST_STEP,
    StubSamQueue,
    close_window,
    make_db,
    make_paths,
    seed_shapes,
)
from tda.core import masks
from tda.core.cache import VIEW_EXT, cache_path
from tda.core.model import FrameKey, ShapeKeyframe, ShapePart, is_context_key
from tda.core.truth import TruthService
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.app_locator import LOCATOR_VIEWS, SETTING
from tda.ui.canvas.overlay import palette_color
from tda.ui.locator_source import (
    NO_TARGET,
    NOT_ANNOTATED,
    NOT_DRAWN,
    NOT_HERE,
    LocatorSource,
    caption_for,
)
from tda.ui.panels.locator import CROP, FILL, WHOLE
from tda.ui.session import AnnotationSession

import gallery_scene as GS

STEP = 12
CHASSIS = "chassis"
OTHER_HW = (48, 64)          # the OAK / RealSense pictures: not the scanner's size


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def answer_roi(win: MainWindow) -> None:
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()


def write_view(db, cache_dir: Path, view: str, steps=range(1, LAST_STEP + 1)) -> None:
    """A picture per step for one of the other cameras."""
    for step in steps:
        key = FrameKey(DESKTOP, int(step), view)
        path = cache_path(str(cache_dir), key, VIEW_EXT[view])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img = np.full((*OTHER_HW, 3), 40, dtype=np.uint8)
        img[10:30, 10 + step:30 + step] = 180
        cv2.imwrite(path, img)
        db.upsert_frame(key, path, {"cache_path": path, "hw": list(OTHER_HW)}, None)


def seed_other_views(db, skip=()) -> None:
    """Every part has a shape of its own on ``oak1`` / ``oak2`` / ``rs``.

    Since U28 a part met for the first time in a view gets the 「拆前 / 拆后」
    pair instead of the location; these tests are about the location, which
    is what a part already drawn in the view gets (U28 rule (c)).
    """
    keys = sorted(k for k in db.instances(DESKTOP)
                  if not is_context_key(k) and k not in skip)
    for view in LOCATOR_VIEWS:
        for index, key in enumerate(keys):
            mask = np.zeros(OTHER_HW, dtype=bool)
            row, col = divmod(index % 48, 8)
            mask[row * 8 + 1:row * 8 + 7, col * 8 + 1:col * 8 + 7] = True
            db.add_keyframe(ShapeKeyframe(
                id=None, instance=key, desktop=DESKTOP, view=view, pose_segment=1,
                anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
                parts=[ShapePart("main", masks.encode_rle(mask))]))


def make_scene(tmp_path: Path, *, drawn: bool = True, skip=(), own: bool = True,
               own_skip=()):
    """``(session on oak1, paths)``: the scanner view drawn, three other cameras."""
    db, paths, tax = make_db(tmp_path)
    for view in LOCATOR_VIEWS:
        write_view(db, Path(paths["cache_dir"]), view)
    if own:
        seed_other_views(db, skip=own_skip)
    session = AnnotationSession(db, tax, TruthService(db, tax), paths["cache_dir"], "tester")
    session.sweeper_enabled = False
    session.open(DESKTOP, "scan")
    if drawn:
        seed_shapes(session, LAST_STEP, skip=skip)
    session.open(DESKTOP, "oak1", force=True)
    return session, paths


def open_window(session, paths, size=(1400, 900), show=False) -> MainWindow:
    win = MainWindow(session, paths, "tester", sam_queue=StubSamQueue())
    win.resize(*size)
    if show:
        win.show()
    QApplication.processEvents()
    answer_roi(win)
    QApplication.processEvents()
    return win


def goto(win: MainWindow, step: int) -> None:
    win.session.goto(step, force=True)
    QApplication.processEvents()


@pytest.fixture
def window(qapp, tmp_path: Path):
    session, paths = make_scene(tmp_path)
    win = open_window(session, paths)
    goto(win, STEP)
    yield win
    close_window(win)


def select(win: MainWindow, instance: str) -> None:
    assert win.instances.select_instance(instance), instance
    QApplication.processEvents()


def select_nothing(win: MainWindow) -> None:
    win.instances.table().setCurrentCell(-1, -1)
    QApplication.processEvents()


def scene(win: MainWindow):
    return win.locator.scene()


def part(win: MainWindow) -> str:
    """A small part drawn in the scanner view (the cooler is not: it is out at 14)."""
    return next(k for k in AS.chassis_instances(win.session, LAST_STEP)
                if k != CHASSIS and not k.startswith("motherboard"))


def card_target(win: MainWindow) -> str:
    instance = win.task_card.current_instance()
    assert instance, "the card has no current row on this frame"
    return instance


# --------------------------------------------------------------------------- #
# where it shows
# --------------------------------------------------------------------------- #
def test_it_is_up_on_oak1_and_in_the_right_column(window):
    assert window.session.view == "oak1"
    assert window.locator_shown()
    assert window.dockWidgetArea(window.locator_dock) == Qt.DockWidgetArea.RightDockWidgetArea
    assert window.locator_dock.objectName() == "dock_locator"
    assert window.locator_button.isChecked() and window.locator_button.isEnabled()


@pytest.mark.parametrize("view", ["oak2", "rs"])
def test_it_is_up_on_the_other_cameras(window, view):
    window.act_set_view(view)
    QApplication.processEvents()
    assert window.session.view == view
    assert window.locator_shown()
    assert scene(window).asked_step == window.session.current().step


def test_it_is_down_on_the_scanner_view_and_back_on_oak1(window):
    window.act_set_view("scan")
    QApplication.processEvents()
    assert window.session.view == "scan"
    select(window, CHASSIS)            # drawn there: no location, no pair (U28)
    assert not window.locator_shown()
    # since U28 the scanner view shows the 「拆前 / 拆后」 of a part met for the
    # first time, so the top bar's toggle is live there too
    assert window.locator_button.isEnabled()
    window.act_set_view("oak1")
    QApplication.processEvents()
    assert window.locator_shown()


def test_modes_steps_hides_it_review_keeps_it(window):
    window.set_mode(A.MODE_STEPS)
    QApplication.processEvents()
    assert not window.locator_shown()
    window.set_mode(A.MODE_REVIEW)
    QApplication.processEvents()
    assert window.locator_shown()
    window.set_mode(A.MODE_ANNOTATE)
    QApplication.processEvents()
    assert window.locator_shown()


def test_it_is_down_on_a_gallery_shot(qapp, tmp_path):
    cfg = make_paths(tmp_path)
    Path(cfg["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    db, tax, cache_dir = GS.make_db(tmp_path, db_path=cfg["db_path"],
                                    cache_dir=Path(cfg["cache_dir"]))
    session = AnnotationSession(db, tax, TruthService(db, tax), str(cache_dir), "tester")
    session.sweeper_enabled = False
    session.open(GS.DESKTOP, "cmp_oak1")
    win = MainWindow(session, cfg, "tester", sam_queue=StubSamQueue())
    try:
        QApplication.processEvents()
        assert not win.locator_shown()
        assert win.locator_refreshes == 0
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# what for
# --------------------------------------------------------------------------- #
def test_the_target_is_the_card_row_then_the_selection_then_the_edit(qapp, tmp_path):
    # the card asks for the part with no shape on oak1 yet: met for the first
    # time, so since U28 it is shown as its 「拆前 / 拆后」 pair
    session, paths = make_scene(tmp_path, own_skip=(COOLER,))
    window = open_window(session, paths)
    try:
        goto(window, STEP)
        _the_target_is_the_card_row_then_the_selection_then_the_edit(window)
    finally:
        close_window(window)


def _the_target_is_the_card_row_then_the_selection_then_the_edit(window):
    select_nothing(window)
    card = card_target(window)
    assert card == COOLER
    assert window.locator_target() == card
    assert window.locator.pair().instance == card

    other = next(r["key"] for r in window.session.instance_rows()
                 if r["key"] not in (card, CHASSIS) and not r.get("context"))
    select(window, other)
    assert window.locator_target() == other
    assert scene(window).instance == other

    window.on_request_edit(CHASSIS)
    QApplication.processEvents()
    assert window.session.editing_instance == CHASSIS
    assert window.locator_target() == CHASSIS
    assert scene(window).instance == CHASSIS

    window.act_clear_edit()
    QApplication.processEvents()
    select_nothing(window)
    assert window.locator_target() == card
    assert window.locator.pair().instance == card


def test_the_caption_names_the_scanner_frame_the_part_and_its_state(window):
    select(window, CHASSIS)
    found = scene(window)
    assert found.scan_step == STEP and found.instance == CHASSIS
    assert window.locator.caption_text() == f"主视角 第 {STEP} 帧 · 机箱 chassis（在）"
    assert window.locator.note_text() == ""
    assert found.colour == tuple(palette_color(CHASSIS))
    assert found.source == "row"


# --------------------------------------------------------------------------- #
# the picture
# --------------------------------------------------------------------------- #
def test_the_crop_puts_the_part_at_half_the_panel_and_a_click_shows_the_whole(window):
    select(window, part(window))
    view = window.locator.view
    found = scene(window)
    assert found.geometry is not None and view.showing() == CROP
    x0, y0, x1, y1 = found.geometry.box
    src = view.source_rect()
    assert src[0] <= x0 and src[1] <= y0 and src[2] >= x1 and src[3] >= y1
    # 64x64 frame, a 6x6 part: the crop is the panel's floor, not the frame
    assert (src[2] - src[0]) < 64 or (src[3] - src[1]) < 64

    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     view.rect().center())
    assert view.showing() == WHOLE
    assert view.source_rect() == (0.0, 0.0, 64.0, 64.0)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     view.rect().center())
    assert view.showing() == CROP
    assert window.session.current().step == STEP     # nothing else moved


def test_a_large_part_spans_about_half_the_crop(qapp):
    from tda.ui.panels.locator import crop_rect

    box = (1000.0, 1000.0, 1400.0, 1200.0)                    # 400 x 200
    src = crop_rect(box, (3040, 4032), aspect=300 / 210)
    width, height = src[2] - src[0], src[3] - src[1]
    assert abs(400.0 / width - FILL) < 0.01 or abs(200.0 / height - FILL) < 0.01
    assert abs(width / height - 300 / 210) < 0.01
    # near an edge the crop moves into the frame, it is not cut
    edge = crop_rect((0.0, 0.0, 100.0, 100.0), (3040, 4032), aspect=1.0)
    assert edge[0] == 0.0 and edge[1] == 0.0


def test_the_wheel_zooms_the_crop(window):
    select(window, part(window))
    view = window.locator.view
    before = view.source_rect()
    pos = QPointF(view.rect().center())
    event = QWheelEvent(pos, view.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 240),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(view, event)
    after = view.source_rect()
    assert (after[2] - after[0]) < (before[2] - before[0])
    assert view.zoom() < 1.0
    # another part starts over
    select(window, CHASSIS)
    assert view.zoom() == 1.0


def test_a_part_with_two_pieces_gets_their_union_box(qapp, tmp_path):
    session, paths = make_scene(tmp_path, skip=(COOLER,))
    two = np.zeros(AS.HW, dtype=bool)
    two[2:6, 2:6] = True
    two[50:60, 40:52] = True
    session.db.add_keyframe(ShapeKeyframe(
        id=None, instance=COOLER, desktop=DESKTOP, view="scan", pose_segment=1,
        anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("a", masks.encode_rle(_left(two))),
               ShapePart("b", masks.encode_rle(_right(two)))]))
    session.truth.refresh_range(DESKTOP, "scan", range(1, LAST_STEP + 1))
    source = LocatorSource(session.db, session.tax, session.truth, paths["cache_dir"])
    geometry, why, _where = source.geometry(DESKTOP, STEP, COOLER)
    assert why == "" and geometry is not None
    assert geometry.box[0] <= 2 and geometry.box[1] <= 2
    assert geometry.box[2] >= 52 and geometry.box[3] >= 60
    assert len(geometry.outlines) == 2
    session.close(force=True)


def _left(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:, 32:] = False
    return out


def _right(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:, :32] = False
    return out


def test_the_whole_frame_carries_the_scanner_roi(window):
    window.db.conn.execute(
        "UPDATE pose_segment SET roi_json=? WHERE desktop=? AND view='scan'",
        ("[8, 8, 56, 56]", DESKTOP))
    window.db.conn.commit()
    select(window, CHASSIS)
    window.follow_locator(force=True)
    assert scene(window).roi == (8.0, 8.0, 56.0, 56.0)


# --------------------------------------------------------------------------- #
# which scanner frame
# --------------------------------------------------------------------------- #
def test_a_missing_scanner_frame_falls_back_to_the_nearest_earlier_one(window):
    window.db.set_frame_flags(FrameKey(DESKTOP, STEP, "scan"), missing=True)
    select(window, CHASSIS)
    window.follow_locator(force=True)
    found = scene(window)
    assert found.scan_step == STEP - 1 and found.asked_step == STEP
    assert window.locator.caption_text().startswith(
        f"主视角 第 {STEP - 1} 帧（第 {STEP} 帧扫描仪没有图） · 机箱 chassis")


def test_a_dupli_step_falls_back_too_and_says_so(window):
    steps = window.db.steps(DESKTOP)
    for rec in steps:
        if rec.step == STEP:
            rec.dupli = True
    window.db.replace_steps(DESKTOP, steps, window.db.actions(DESKTOP))
    select(window, CHASSIS)
    window.follow_locator(force=True)
    found = scene(window)
    assert found.scan_step == STEP - 1
    assert f"（第 {STEP} 帧是重复帧）" in window.locator.caption_text()


def test_caption_for():
    assert caption_for(7, 7, "", "机箱 chassis", "在") == "主视角 第 7 帧 · 机箱 chassis（在）"
    assert caption_for(6, 7, "missing", "", "") == "主视角 第 6 帧（第 7 帧扫描仪没有图）"


# --------------------------------------------------------------------------- #
# why there is no outline
# --------------------------------------------------------------------------- #
def test_a_part_never_drawn_in_the_scanner_view_says_so(qapp, tmp_path):
    session, paths = make_scene(tmp_path, skip=(COOLER,))
    win = open_window(session, paths)
    try:
        goto(win, STEP)
        select(win, CHASSIS)            # drawn
        assert scene(win).geometry is not None
        select(win, COOLER)             # not drawn in the scanner view
        found = scene(win)
        assert found.instance == COOLER and found.geometry is None
        assert win.locator.note_text() == NOT_DRAWN
        assert found.caption.startswith(f"主视角 第 {STEP} 帧 · ")
        assert win.locator.view.showing() == WHOLE
    finally:
        close_window(win)


def test_drawing_it_in_the_scanner_view_later_is_seen(window):
    """The cached facts follow the scanner view's inputs (``scan_digest``)."""
    window.db.conn.execute("DELETE FROM compiled_mask WHERE desktop=? AND step=? AND view='scan'",
                           (DESKTOP, STEP))
    window.db.conn.commit()
    select(window, COOLER)
    assert window.locator.note_text() == NOT_DRAWN
    shape = np.zeros(AS.HW, dtype=bool)
    shape[40:56, 40:56] = True
    window.db.add_keyframe(ShapeKeyframe(
        id=None, instance=COOLER, desktop=DESKTOP, view="scan", pose_segment=1,
        anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(shape))]))
    window.follow_locator(force=True)
    found = scene(window)
    assert found.geometry is not None and window.locator.note_text() == ""
    assert found.source == "compiled"            # no stored row: compiled, not written
    assert found.geometry.box[0] <= 40 and found.geometry.box[2] >= 56


def test_a_part_not_on_that_scanner_frame_says_so(window):
    key = FrameKey(DESKTOP, STEP, "scan")
    hidden = part(window)
    window.db.put_compiled(key, hidden, None, 1.0, "occluded_full", "in_chassis",
                           "auto", "seeded")
    select(window, hidden)
    window.follow_locator(force=True)
    found = scene(window)
    assert found.geometry is None
    assert window.locator.note_text() == NOT_HERE
    assert window.locator.view.showing() == WHOLE


def test_a_scanner_view_not_annotated_at_all_says_so(qapp, tmp_path):
    session, paths = make_scene(tmp_path, drawn=False)
    win = open_window(session, paths)
    try:
        goto(win, STEP)
        select(win, CHASSIS)
        found = scene(win)
        assert found.geometry is None and found.image is not None
        assert win.locator.note_text() == NOT_ANNOTATED
    finally:
        close_window(win)


def test_nothing_to_look_for_shows_the_frame_and_says_what_to_do(window):
    select_nothing(window)
    source = window.locator_source
    found = source.scene(DESKTOP, STEP, None)
    assert found.instance is None and found.note == NO_TARGET and found.image is not None
    # a card row of a dupli frame names the step, not a part
    assert source.scene(DESKTOP, STEP, "Remove the cover").note == NO_TARGET


def test_a_frame_with_no_stored_rows_is_compiled_once(window):
    window.db.conn.execute("DELETE FROM compiled_mask WHERE desktop=? AND step=? AND view='scan'",
                           (DESKTOP, STEP))
    window.db.conn.commit()
    source = window.locator_source
    before = source.stats["compiles"]
    first, why, where = source.geometry(DESKTOP, STEP, CHASSIS)
    assert first is not None and why == "" and where == "compiled"
    second, _why, _where = source.geometry(DESKTOP, STEP, part(window))
    assert second is not None
    assert source.stats["compiles"] == before + 1
    assert window.db.compiled(FrameKey(DESKTOP, STEP, "scan")) == {}   # nothing written


# --------------------------------------------------------------------------- #
# cost: what is read, and when
# --------------------------------------------------------------------------- #
def test_one_refresh_per_frame_and_part_none_for_the_same_one(window):
    select(window, CHASSIS)
    count = window.locator_refreshes
    for _ in range(3):
        window.refresh_guidance()
    QApplication.processEvents()
    assert window.locator_refreshes == count
    # the card and the table are rebuilt on the way (two or three passes through
    # refresh_guidance with the old selection, no row, the new row): one refresh
    goto(window, STEP - 1)
    assert window.locator_refreshes == count + 1
    assert scene(window).asked_step == STEP - 1
    assert scene(window).instance == window.locator_target()


def test_no_refresh_while_it_is_closed_or_on_the_scanner_view(window):
    window.locator_dock.close()                       # the annotator's close button
    QApplication.processEvents()
    assert not window.locator_shown() and window.locator_wanted is False
    count = window.locator_refreshes
    reads = dict(window.locator_source.stats)
    goto(window, STEP - 1)
    goto(window, STEP - 2)
    assert window.locator_refreshes == count
    assert window.locator_source.stats == reads

    window.act_toggle_locator(True)
    assert window.locator_shown()
    window.act_set_view("scan")
    QApplication.processEvents()
    # on the scanner view it is down for a part drawn there (U28: a part met
    # for the first time shows its 「拆前 / 拆后」 instead)
    select(window, CHASSIS)
    assert not window.locator_shown()
    count = window.locator_refreshes
    reads = dict(window.locator_source.stats)
    select(window, part(window))
    window.refresh_guidance()
    QApplication.processEvents()
    assert not window.locator_shown()
    assert window.locator_refreshes == count
    assert window.locator_source.stats == reads


def test_the_picture_of_a_step_is_read_once(window):
    source = window.locator_source
    select(window, CHASSIS)
    reads = source.stats["thumb_reads"]
    select(window, part(window))
    goto(window, STEP - 1)
    goto(window, STEP)
    assert source.stats["thumb_reads"] == reads + 1     # only STEP - 1 was new
    assert (DESKTOP, STEP) in source.thumbs_cached()


# --------------------------------------------------------------------------- #
# the dock: remembered, and out of the palette's way
# --------------------------------------------------------------------------- #
def test_a_closed_panel_stays_closed_in_the_next_session(qapp, tmp_path):
    session, paths = make_scene(tmp_path)
    win = open_window(session, paths)
    win.locator_dock.close()
    QApplication.processEvents()
    assert str(win.settings.value(SETTING)).lower() == "false"
    win.save_window_state()
    close_window(win)

    session = AnnotationSession(session.db, session.tax, TruthService(session.db, session.tax),
                                paths["cache_dir"], "tester")
    session.sweeper_enabled = False
    session.open(DESKTOP, "oak1")
    again = open_window(session, paths)
    try:
        assert not again.locator_shown()
        assert not again.locator_button.isChecked()
        again.locator_button.click()                   # the top bar's toggle
        QApplication.processEvents()
        assert again.locator_shown() and again.locator_wanted is True
        assert again.locator.scene() is not None
    finally:
        close_window(again)


@pytest.mark.parametrize("size", [(1920, 1000), (1920, 1080)])
def test_the_panel_leaves_the_palette_and_its_state_group_where_they_were(qapp, tmp_path,
                                                                          size):
    """U13's measurement on an oak1 frame: the dock is in the right column."""
    session, paths = make_scene(tmp_path)
    win = open_window(session, paths, size=size, show=True)
    try:
        assert win.locator_dock.isVisible() and not win.locator_dock.isFloating()
        scroll = win.palette.scroll
        with_panel = scroll.viewport().height()
        inner = scroll.widget()
        for name in ("cycle_state", "state_range", "correct_verb", "confirm_cut_copy",
                     "mark_absent"):
            button = win.palette.button(name)
            bottom = button.mapTo(inner, button.rect().bottomLeft()).y()
            assert not button.isHidden() and bottom <= with_panel, (name, bottom, with_panel)
        assert 180 <= win.locator_dock.height() <= 340, win.locator_dock.height()
        win.locator_dock.close()
        QApplication.processEvents()
        assert scroll.viewport().height() == with_panel
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# the budget, on the 12 MP scene
# --------------------------------------------------------------------------- #
OAK_HW = (3040, 4032)
BUDGET_CACHED = 0.015
BUDGET_FIRST = 0.15
#: Generous: a shared machine, a full-suite run (see test_session_perf).
SLACK = 3.0


@pytest.mark.slow
def test_refresh_budget_on_a_12mp_scanner_frame(qapp, tmp_path):
    """First show of a step < 150 ms, a cached one < 15 ms (asserted at 3x)."""
    import session_scene as SS

    session = SS.make_session(tmp_path, last_step=6, hw=OAK_HW)
    session.sweeper_enabled = False
    session.goto(2)
    drawn = SS.seed_shapes(session, 2, grid=7, anchor=6, refresh=False)
    for step in range(1, 7):
        session.truth.refresh(FrameKey(SS.DESKTOP, step, "scan"))
    source = LocatorSource(session.db, session.tax, session.truth, session.cache_dir)
    from tda.ui.panels.locator import LocatorPanel

    panel = LocatorPanel()
    panel.resize(300, 260)
    target = drawn[len(drawn) // 2]
    firsts, cached = [], []
    for step in range(1, 7):
        masks.clear_decode_cache()          # the outline is decoded, not remembered
        start = time.perf_counter()
        panel.show_scene(source.scene(SS.DESKTOP, step, target))
        panel.view.repaint()
        firsts.append(time.perf_counter() - start)
        start = time.perf_counter()
        panel.show_scene(source.scene(SS.DESKTOP, step, drawn[0]))
        panel.view.repaint()
        cached.append(time.perf_counter() - start)
    # a scanner frame nobody compiled yet: compiled once, then every part is free
    session.db.conn.execute("DELETE FROM compiled_mask WHERE desktop=? AND step=? "
                            "AND view='scan'", (SS.DESKTOP, 3))
    session.db.conn.commit()
    start = time.perf_counter()
    panel.show_scene(source.scene(SS.DESKTOP, 3, target))
    compiled = time.perf_counter() - start
    start = time.perf_counter()
    panel.show_scene(source.scene(SS.DESKTOP, 3, drawn[1]))
    after = time.perf_counter() - start
    print(f"\nlocator 12 MP: first {[round(t * 1000, 1) for t in firsts]} ms, "
          f"cached {[round(t * 1000, 1) for t in cached]} ms; no stored rows: "
          f"{compiled * 1000:.0f} ms, then {after * 1000:.1f} ms")
    assert panel.scene().geometry is not None and panel.scene().source == "compiled"
    assert min(firsts) < BUDGET_FIRST * SLACK, firsts
    assert sorted(cached)[len(cached) // 2] < BUDGET_CACHED * SLACK, cached
    session.close(force=True)
