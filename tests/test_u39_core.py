"""Task U39: 画框外 -- the part is there, outside this view's picture.

The user (2026-10-27): 「我需要加入可以改变这个东西是不是在当前画框内能看到的选项……
看不见是这个时候被挡住了，但是不在视野内。比如 D63 的导风罩是存在的，但是在主视角的画面外。」
A part under a cable or behind another part is in the picture and keeps an
amodal shape to draw (被挡住); a part outside the picture has nothing to draw.

Pinned here, on the D13 session scene (every part of step 14 drawn, the cooler
and its four captive screws undrawn on 1-12, the cooler lifted out at 13):

* the rule (:mod:`tda.core.occluder`): 画框外 is the gallery's 「看不见」 row --
  ``out_of_view``, no pixels, no reason; ``I``'s cycle never sets it, and after
  it the next press is 撤回; it needs no shape, only the part in its place;
* the session's ``Ctrl+```: one op over a span, refused (naming the frames)
  where the part is taken out, ``Ctrl+Z`` takes it back, 撤回 over a span takes
  画框外 and the cable labels back together;
* the compiler: no mask, no box, ``out_of_view``; no ``missing_shape`` for it;
  a shape that is there is kept (the amodal estimate) and shows nothing;
* the missing-shape queue (coverage), the task card (done, 「画框外」, the
  start card's ✔ rows after the open ones), ``Space`` and ``cli check``;
* the state and placement are untouched -- it is still in the machine.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

from session_scene import (
    CHASSIS,
    COOLER,
    DESKTOP,
    SCREWS,
    VIEW,
    make_session,
    seed_shapes,
)
from tda.core.model import FrameKey, FrameOverride, Visibility
from tda.core.occluder import (
    CHOICES,
    CYCLE,
    FULL,
    OFF_FRAME,
    PARTIAL,
    cable_label,
    frame_label,
    next_label,
    occluded_override,
    off_frame,
    off_frame_applies,
)
from tda.core.truth_inputs import state_of
from tda.ui import session_api as api
from tda.ui.panels.taskcard import OFF_FRAME_CHIP, OFF_FRAME_SENTENCE, card_header, row_view
from tda.ui.session_api import SessionRefusal
from tda.ui.session_cable import CABLE_NEEDS_SHAPE, CABLE_REMOVED, OP_RANGE
from tda.ui.session_coverage import coverage
from tda.ui.session_tasks import OFF_FRAME_KEY, task_card_for

OUT = Visibility.OUT_OF_VIEW.value
UNDRAWN = (COOLER, *SCREWS)          # no shape on 1-12


@pytest.fixture
def drawn(tmp_path: Path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    seed_shapes(session, 14)
    session.goto(12, force=True)
    yield session
    session.close(force=True)


def override(session, step: int, instance: str):
    return session.db.frame_overrides(FrameKey(DESKTOP, step, VIEW)).get(instance)


def ops(session, kind: str) -> list[dict]:
    return [op for op in session.db.ops(DESKTOP, VIEW, limit=1000) if op["kind"] == kind]


def problems(session, step: int) -> list[str]:
    session.goto(step, force=True)
    return list(session.compiled().problems)


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def test_off_frame_is_the_gallerys_row_out_of_view_with_no_pixels_and_no_reason():
    key = FrameKey(DESKTOP, 12, VIEW)
    row = occluded_override(key, COOLER, None, OFF_FRAME, (0, 0))
    assert (row.visibility, row.visible_rle, row.occluder) == (OUT, None, None)
    assert off_frame(row) and frame_label(row) == OFF_FRAME
    assert cable_label(row) is None                      # not a 被线挡住 label
    # a hand mask under 完全 is not carried into 画框外 (nothing is in the picture)
    hand = FrameOverride(key, COOLER, {"size": [64, 64], "counts": "PP0"}, None)
    assert occluded_override(key, COOLER, hand, OFF_FRAME, (0, 0)).visible_rle is None
    # 撤回 leaves no row behind
    assert occluded_override(key, COOLER, row, None, (0, 0)) is None
    # a digit key's bare out_of_view with no mask is the same statement
    assert off_frame(FrameOverride(key, COOLER, None, OUT))
    assert not off_frame(FrameOverride(key, COOLER, {"size": [1, 1], "counts": "1"}, OUT))


def test_the_key_never_sets_it_and_takes_it_back():
    key = FrameKey(DESKTOP, 12, VIEW)
    assert OFF_FRAME not in CYCLE
    assert CHOICES == (FULL, PARTIAL, None, OFF_FRAME)          # the dialog's order
    assert [next_label(None)] == [FULL]
    held = FrameOverride(key, COOLER, None, OUT)
    assert next_label(held) is None                              # 画框外 -> 撤回
    # the cycle from nothing: 完全 -> 部分 -> 撤回, never 画框外
    seen, current = [], None
    for _ in range(6):
        nxt = next_label(current)
        seen.append(nxt)
        current = occluded_override(key, CHASSIS, current, nxt, (64, 64))
    assert OFF_FRAME not in seen


def test_it_needs_no_shape_only_the_part_in_its_place(drawn):
    db, tax = drawn.db, drawn.tax
    assert off_frame_applies(db, tax, FrameKey(DESKTOP, 12, VIEW), COOLER) == ""
    assert off_frame_applies(db, tax, FrameKey(DESKTOP, 12, VIEW), CHASSIS) == ""
    # lifted out at 13: not in the machine any more
    assert off_frame_applies(db, tax, FrameKey(DESKTOP, 13, VIEW), COOLER) == "removed"


# --------------------------------------------------------------------------- #
# Ctrl+` : one op over a span
# --------------------------------------------------------------------------- #
def test_ctrl_grave_writes_off_frame_over_a_span_in_one_op_and_undoes(drawn):
    before = state_of(drawn.db, drawn.tax, DESKTOP, 11, view=VIEW)[COOLER]
    result = drawn.set_cable_range(COOLER, OFF_FRAME, 10)
    assert result["frames"] == [10, 11, 12] and result["label"] == OFF_FRAME
    for step in (10, 11, 12):
        fo = override(drawn, step, COOLER)
        assert (fo.visibility, fo.visible_rle, fo.occluder) == (OUT, None, None)
    assert override(drawn, 9, COOLER) is None
    assert len(ops(drawn, OP_RANGE)) == 1
    # still in the machine, in the state it was: nothing about the state moved
    assert state_of(drawn.db, drawn.tax, DESKTOP, 11, view=VIEW)[COOLER] == before
    assert drawn.frame_label(COOLER) == OFF_FRAME
    assert COOLER in drawn.off_frame_parts()
    drawn.undo()
    assert all(override(drawn, s, COOLER) is None for s in (10, 11, 12))
    drawn.redo()
    assert all(off_frame(override(drawn, s, COOLER)) for s in (10, 11, 12))


def test_ctrl_grave_refuses_the_frames_the_part_is_taken_out_on(drawn):
    with pytest.raises(SessionRefusal) as refused:
        drawn.set_cable_range(COOLER, OFF_FRAME, 14)
    assert "第 13–14 步" in str(refused.value) and CABLE_REMOVED in str(refused.value)
    assert all(override(drawn, s, COOLER) is None for s in (12, 13, 14))
    # ... and 完全 still wants the shape it does not have (U15 unchanged)
    with pytest.raises(SessionRefusal, match="先画出它的形状"):
        drawn.set_cable_range(COOLER, FULL, 10)
    assert CABLE_NEEDS_SHAPE.startswith("先画出它的形状")


def test_withdraw_over_a_span_takes_off_frame_and_the_cable_back(drawn):
    drawn.goto(14, force=True)
    drawn.set_cable_range(CHASSIS, FULL, 13)                     # 被线挡住 on 13-14
    drawn.goto(12, force=True)
    drawn.set_cable_range(CHASSIS, OFF_FRAME, 11)                # 画框外 on 11-12
    drawn.goto(14, force=True)
    result = drawn.set_cable_range(CHASSIS, None, 11)
    assert result["frames"] == [11, 12, 13, 14]
    assert result["withdrawn"] == sorted({FULL, OFF_FRAME})
    assert all(override(drawn, s, CHASSIS) is None for s in (11, 12, 13, 14))


def test_the_key_on_an_off_frame_part_takes_it_back(drawn):
    drawn.set_cable_range(COOLER, OFF_FRAME, 12)
    assert drawn.cable_gate(COOLER) == (None, "")               # 撤回, never refused
    result = drawn.cycle_cable(COOLER)
    assert result["label"] is None and result["was"] == OFF_FRAME
    assert not result["stale"]                                   # 画框外 needed no shape
    assert override(drawn, 12, COOLER) is None
    # the next press is the cycle again: 完全, refused for want of a shape
    assert drawn.cable_gate(COOLER) == (FULL, CABLE_NEEDS_SHAPE)


# --------------------------------------------------------------------------- #
# the compiler, the queue, the card, Space
# --------------------------------------------------------------------------- #
def test_no_missing_shape_no_mask_no_box(drawn):
    assert f"missing_shape:{COOLER}" in problems(drawn, 11)
    drawn.goto(12, force=True)
    drawn.set_cable_range(COOLER, OFF_FRAME, 10)
    for step in (10, 11, 12):
        found = problems(drawn, step)
        assert f"missing_shape:{COOLER}" not in found
        assert f"missing_shape:{SCREWS[0]}" in found                # the others still are
        inst = drawn.compiled().instances[COOLER]
        assert (inst.visibility, inst.visible, inst.box, inst.amodal) == (OUT, None, None, None)
        assert inst.placement == "in_chassis"
    assert f"missing_shape:{COOLER}" in problems(drawn, 9)


def test_a_shape_that_is_there_is_kept_and_shows_nothing(drawn):
    drawn.goto(14, force=True)
    kf_before = [kf.id for kf in drawn.db.keyframes(DESKTOP, VIEW, CHASSIS)]
    drawn.set_cable_range(CHASSIS, OFF_FRAME, 14)
    inst = drawn.compiled().instances[CHASSIS]
    assert inst.visibility == OUT and inst.visible is None and inst.box is None
    assert inst.amodal is not None and bool(np.any(inst.amodal))  # the whole-part estimate
    assert inst.keyframe_id is not None
    assert [kf.id for kf in drawn.db.keyframes(DESKTOP, VIEW, CHASSIS)] == kf_before
    assert not [p for p in drawn.compiled().problems if CHASSIS in p]


def test_the_missing_shape_queue_drops_it(drawn):
    drawn.set_cable_range(COOLER, OFF_FRAME, 11)
    found = coverage(drawn.db, drawn.tax, DESKTOP, VIEW, [10, 11, 12])
    assert COOLER in found[10].missing
    assert COOLER not in found[11].missing and COOLER not in found[12].missing
    assert SCREWS[0] in found[12].missing


def test_the_card_row_says_off_frame_and_is_done(drawn):
    drawn.set_cable_range(COOLER, OFF_FRAME, 12)
    card = drawn.task_card()
    row = next(r for r in card if r["instance"] == COOLER)
    assert row["kind"] == api.KIND_ADD_SHAPE and row["done"] and row[OFF_FRAME_KEY]
    assert not row.get("invisible")                              # says why on its own
    view = row_view(row)
    assert view["chip"] == OFF_FRAME_CHIP == "画框外"
    assert view["sentence"] == OFF_FRAME_SENTENCE and view["done"]
    for screw in SCREWS:                                          # still to draw
        other = next(r for r in card if r["instance"] == screw)
        assert not other["done"] and not other.get(OFF_FRAME_KEY)
        assert row_view(other)["chip"] != OFF_FRAME_CHIP
    header = card_header(12, 13, [r["kind"] for r in card], done=[r["done"] for r in card])
    assert header.startswith("第 12 帧：比第 13 帧多了下面这些零件")
    # the Qt-free card says the same
    same = task_card_for(drawn.db, drawn.tax, DESKTOP, VIEW, 12, neighbour=13)
    assert next(r for r in same if r["instance"] == COOLER)[OFF_FRAME_KEY]


def test_the_start_card_puts_it_after_the_open_rows(drawn):
    drawn.goto(14, force=True)
    drawn.set_cable_range(CHASSIS, OFF_FRAME, 14)
    card = task_card_for(drawn.db, drawn.tax, DESKTOP, VIEW, 14, neighbour=None)
    chassis = next(r for r in card if r["instance"] == CHASSIS)
    assert chassis["done"] and chassis[OFF_FRAME_KEY]
    assert all(r["done"] for r in card)                       # the rest are drawn
    # a part with no shape: the open row goes first, the 画框外 one with the ✔ rows
    card = task_card_for(drawn.db, drawn.tax, DESKTOP, VIEW, 12, neighbour=None)
    assert not card[0]["done"]
    drawn.goto(12, force=True)
    drawn.set_cable_range(COOLER, OFF_FRAME, 12)
    card = task_card_for(drawn.db, drawn.tax, DESKTOP, VIEW, 12, neighbour=None)
    cooler = next(i for i, r in enumerate(card) if r["instance"] == COOLER)
    first_done = next(i for i, r in enumerate(card) if r["done"])
    assert cooler >= first_done and card[cooler][OFF_FRAME_KEY]


def test_space_is_not_held_up_by_it(drawn):
    assert not drawn.confirm_frame()                            # five shapes missing
    drawn.goto(12, force=True)
    for part in UNDRAWN:
        drawn.set_cable_range(part, OFF_FRAME, 12)
    assert drawn.confirm_frame()
    key = FrameKey(DESKTOP, 12, VIEW)
    assert drawn.db.get_frame(key)["review_status"] == "verified"
    row = drawn.db.compiled_label(key, COOLER)
    assert row["visibility"] == OUT


def test_on_a_confirmed_frame_it_stays_confirmed(drawn):
    drawn.goto(14, force=True)
    assert drawn.confirm_frame()                                  # 14, then on to 13
    key = FrameKey(DESKTOP, 14, VIEW)
    signed = drawn.db.compiled(key)[CHASSIS]
    assert signed["visibility"] != OUT and signed.get("visible_rle")
    drawn.goto(14, force=True)
    drawn.set_cable_range(CHASSIS, OFF_FRAME, 14)
    assert drawn.db.get_frame(key)["review_status"] == "verified"
    row = drawn.db.compiled(key)[CHASSIS]
    assert row["visibility"] == OUT and not row.get("visible_rle")
    assert row["status"] == "verified"
    assert not drawn.db.conflicts(DESKTOP, VIEW, open_only=True)
    drawn.undo()                    # the frame on screen: its pixels are signed again
    assert drawn.db.compiled_label(key, CHASSIS)["visibility"] == signed["visibility"]
    assert drawn.db.get_frame(key)["review_status"] == "verified"


# --------------------------------------------------------------------------- #
# cli check
# --------------------------------------------------------------------------- #
def test_cli_check_does_not_report_it(tmp_path, capsys):
    from app_scene import make_db, write_paths_yaml
    from app_scene import seed_shapes as seed_app
    from tda.cli import main
    from tda.core.truth import TruthService
    from tda.ui.session import AnnotationSession

    db, cfg, tax = make_db(tmp_path)
    paths = write_paths_yaml(tmp_path)

    def listed() -> list[str]:
        main(["--paths", paths, "check", "--desktop", str(DESKTOP), "--view", VIEW,
              "--limit", "500"])
        out = capsys.readouterr().out
        return [line.strip() for line in out.splitlines() if line.strip().startswith("- ")]

    session = AnnotationSession(db, tax, TruthService(db, tax), cfg["cache_dir"], "tester")
    session.sweeper_enabled = False
    session.open(DESKTOP, VIEW)
    try:
        seed_app(session, 14)
        before = listed()
        assert f"- missing_shape:{COOLER}" in before
        session.goto(12, force=True)
        session.set_cable_range(COOLER, OFF_FRAME, 1)               # 画框外 on 1-12
    finally:
        session.close(force=True)
    after = listed()
    assert f"- missing_shape:{COOLER}" not in after
    assert f"- missing_shape:{SCREWS[0]}" in after
    assert not [line for line in after if COOLER in line and "missing_shape" not in line
                and "screw" not in line]
    assert len(before) - len(after) == 12                          # its 12 frames
