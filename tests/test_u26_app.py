"""Task U26, ruling 3: the task card tells a part with no shape from a covered one.

The compiler notes ``empty_visible:<part>`` for both. A part the labelled parts
cover completely is 「这一帧里它被完全挡住了」; one whose shape is empty on this
frame -- nothing drawn, or wholly outside the picture -- is not covered by
anything, and the card now says 「这一帧没有它的形状（没画，或整个在画面外）」.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import vis_review_scene as V
from tda.core import masks
from tda.core.model import ShapeKeyframe, ShapePart
from tda.ui.panels.taskcard import NO_SHAPE_SENTENCE, TaskCardPanel, explain_code

COVERED_SENTENCE = "这一帧里它被完全挡住了"
NO_SHAPE_WORDS = "这一帧没有它的形状（没画，或整个在画面外）"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_the_sentence_follows_the_frame():
    assert COVERED_SENTENCE in explain_code("empty_visible:x.01")
    said = explain_code("empty_visible:x.01", shapeless={"x.01"})
    assert said == NO_SHAPE_SENTENCE.format(what="x.01") == f"x.01：{NO_SHAPE_WORDS}"
    assert COVERED_SENTENCE not in said
    # another part's code, and every other code, are untouched
    assert COVERED_SENTENCE in explain_code("empty_visible:y.01", shapeless={"x.01"})
    assert explain_code("missing_shape:x.01", shapeless={"x.01"}) == explain_code(
        "missing_shape:x.01")


def test_the_card_asks_the_frame_which_one_it_is(qapp, tmp_path):
    session = V.build(tmp_path, confirm=())
    try:
        # a newer shape of connector.03 with nothing in it, on every frame
        session.db.add_keyframe(ShapeKeyframe(
            id=None, instance=V.TINY, desktop=V.DESKTOP, view=V.VIEW, pose_segment=1,
            anchor_step=14, placement="in_chassis", geom_type="mask", version=5,
            parts=[ShapePart("main", masks.encode_rle(np.zeros(V.HW, bool)))]))
        session.refresh_all()
        session.goto(11, force=True)
        assert set(session.current_problems()) >= {f"empty_visible:{V.TINY}",
                                                   f"empty_visible:{V.HIDDEN}"}
        panel = TaskCardPanel()
        panel.set_session(session)
        texts = {row["instance"]: row["text"] for row in panel.problem_rows()
                 if row["code"].startswith("empty_visible:")}
    finally:
        session.close(force=True)
    assert texts[V.TINY] == f"{V.TINY}：{NO_SHAPE_WORDS}"
    assert COVERED_SENTENCE in texts[V.HIDDEN] and NO_SHAPE_WORDS not in texts[V.HIDDEN]


def test_without_a_session_the_card_keeps_the_old_sentence(qapp):
    panel = TaskCardPanel()
    panel._show_arrival(["empty_visible:a.01"])
    [row] = panel.problem_rows()
    assert COVERED_SENTENCE in row["text"]
