"""Task W3f: 「谁压着谁」 rows light the pair up on the canvas, say both states and fold away
what the log or the graph already decides (offscreen Qt).

* resting the pointer on a row outlines **both** parts (the part on top in
  orange, the one underneath in blue, a chip naming each, the box round the
  whole shape) and the status line says which is which; leaving the row takes
  it away; so does a frame change and leaving Review; no op, no selection, no
  editing layer -- on a ruled row and on a folded one too;
* the rows say both parts' states at the frame, the tooltip carries the 日志
  line;
* D04's case: ``psu.01`` lies over ``cover.01`` and the log opened the cover at
  step 2 with the PSU in place -> 「日志已证明不挡（1）」, one 「不挡」 and no
  「挡着」; a pair the graph orders through another part -> 「规则已间接覆盖（1）」
  with the chain, no button.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPointF
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QApplication, QPushButton

import app_scene
from test_w3c_rulings_dock import (  # noqa: F401 - dock_class_rulings: autouse (W3g)
    box,
    dock_class_rulings,
    open_window,
    review,
    ruling_ops,
    seed_pile,
    settle,
)
from tda.core import masks
from tda.core.model import ShapeKeyframe, ShapePart, ZOrderRec
from tda.ui import app_actions as A
from tda.ui.app_rulings import ABOVE_RGB, BELOW_RGB, HOVER_TEXT
from tda.ui.panels.notes_dock import INDIRECT_TITLE, PROVEN_TITLE

DESKTOP, VIEW = app_scene.DESKTOP, app_scene.VIEW


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    win = open_window(tmp_path)
    yield win
    app_scene.close_window(win)


def seed_d04(session) -> dict:
    """The PSU lies over the expansion-slot cover (D04 frame 21); the log opened
    ``cover.01`` at step 2 with the PSU installed.  ``cover.02`` -> ``psu.01`` ->
    ``motherboard.01`` are edges, so ``cover.02 → motherboard.01`` is indirect."""
    shapes = {"motherboard.01": box(0, 64, 0, 64), "cover.01": box(0, 30, 0, 40),
              "psu.01": box(0, 64, 0, 60), "cover.02": box(34, 64, 0, 40)}
    for key, mask in shapes.items():
        session.db.add_keyframe(ShapeKeyframe(
            id=None, instance=key, desktop=DESKTOP, view=VIEW, pose_segment=1,
            anchor_step=app_scene.LAST_STEP, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(mask))]))
    session.db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, [(k, "main") for k in shapes]))
    session.db.add_relation(DESKTOP, "covered_by", "psu.01", "cover.02")
    session.db.add_relation(DESKTOP, "blocked_by", "motherboard.01", "psu.01", mode="path")
    session.refresh_all()
    return shapes


@pytest.fixture
def d04(qapp, tmp_path: Path):
    win = open_window(tmp_path, seed=seed_d04)
    yield win
    app_scene.close_window(win)


def enter(row) -> None:
    QApplication.sendEvent(row, QEnterEvent(QPointF(2, 2), QPointF(2, 2), QPointF(2, 2)))


def leave(row) -> None:
    QApplication.sendEvent(row, QEvent(QEvent.Type.Leave))


def row_of(section, pair):
    return next(row for p, row in section.hover_rows if p == pair)


def untouched(win):
    """What a hover must not change: ops, the selected part, the editing layer."""
    overlay = win.canvas.overlay()
    return (len(win.session.db.ops(DESKTOP, VIEW)), win.instances.selected_instance(),
            None if overlay is None else overlay.editing_instance)


def test_hover_outlines_both_parts_and_leave_takes_them_away(window):
    review(window)
    section = window.notes_panel.rulings
    before = untouched(window)
    pair = ("cover.01", "psu.01")
    enter(row_of(section, pair))
    assert window.rulings_hover == pair
    hints = window.canvas.hint_boxes()
    assert hints == [((0.0, 0.0, 40.0, 30.0), "上：cover.01", tuple(ABOVE_RGB)),
                     ((0.0, 0.0, 60.0, 64.0), "下：psu.01", tuple(BELOW_RGB))]
    assert window.status_message() == HOVER_TEXT.format(
        above="cover.01", below="psu.01", above_state="（打开）", below_state="（装着）")
    assert "橙框" in window.status_message() and "蓝框" in window.status_message()
    assert untouched(window) == before
    leave(row_of(section, pair))
    assert window.canvas.hint_boxes() == [] and window.rulings_hover is None
    assert window.status_message() == ""
    assert untouched(window) == before
    assert ruling_ops(window) == []


def test_moving_to_the_next_row_switches_the_pair(window):
    review(window)
    section = window.notes_panel.rulings
    enter(row_of(section, ("cover.01", "psu.01")))
    leave(row_of(section, ("cover.01", "psu.01")))
    enter(row_of(section, ("psu.01", "motherboard.01")))
    assert [h[1] for h in window.canvas.hint_boxes()] == ["上：psu.01", "下：motherboard.01"]


def test_a_frame_change_clears_the_outlines(window):
    review(window)
    section = window.notes_panel.rulings
    enter(row_of(section, ("cover.01", "psu.01")))
    assert window.canvas.hint_boxes()
    step = int(window.session.current().step)
    window.timeline_goto(step - 1)
    settle()
    assert int(window.session.current().step) == step - 1
    assert window.canvas.hint_boxes() == [] and window.rulings_hover is None
    assert ruling_ops(window) == []


def test_leaving_review_clears_the_outlines(window):
    review(window)
    enter(row_of(window.notes_panel.rulings, ("cover.01", "psu.01")))
    window.set_mode(A.MODE_ANNOTATE)
    settle()
    assert window.canvas.hint_boxes() == [] and window.rulings_hover is None


def test_a_ruled_row_lights_up_too(window):
    review(window)
    section = window.notes_panel.rulings
    section.row_buttons[0][2].click()                             # cover.01 → psu.01 不挡
    settle()
    assert window.canvas.hint_boxes() == []                       # its row is gone
    enter(row_of(section, ("cover.01", "psu.01")))                # now the 已裁定 line
    assert [h[1] for h in window.canvas.hint_boxes()] == ["上：cover.01", "下：psu.01"]
    assert len(ruling_ops(window)) == 1                            # the click, nothing more


def test_rows_say_both_states_and_the_log(window):
    review(window)
    section = window.notes_panel.rulings
    texts = section.texts()
    assert texts[0] == "cover.01（打开）→ psu.01（装着） · 像素 1 200 · 用户已定：挡"
    tips = section.row_tips()
    assert tips[0].splitlines()[-1] == "日志：cover.01 第 2 步 打开，psu.01 从未拆下"
    assert all("日志：" in tip for tip in tips)


def test_d04_the_log_proven_pair_is_folded_with_one_not_blocks(d04):
    review(d04)
    section = d04.notes_panel.rulings
    pairs = [(c.above, c.below) for c in section.candidates]
    assert ("psu.01", "cover.01") not in pairs
    assert [(c.above, c.below) for c in section.proven_candidates] == [("psu.01", "cover.01")]
    assert section.proven_toggle.text() == PROVEN_TITLE.format(k=1)
    assert not section.proven_toggle.isHidden() and not section.proven_toggle.isChecked()
    # W3h: a question, not a proof -- 「不挡」 or 「上下翻转」; the dock's own class
    # rulings say cover → psu 挡, so the row carries the layer tag
    assert section.proven_texts() == [
        "psu.01（装着）→ cover.01（打开） · 像素 1 200 · 第 2 步先动了下面的 cover.01 · "
        "用户已定：挡 → 层序可能画反"]
    [(cand, free, flip)] = section.proven_buttons
    assert (free.text(), flip.text()) == ("不挡", "上下翻转")
    buttons = section.proven.findChildren(QPushButton)
    assert [b.text() for b in buttons] == ["不挡", "上下翻转"]        # never 「挡着」
    # the indirect fold: the chain in words, no button
    assert [(c.above, c.below) for c in section.indirect_candidates] == [
        ("cover.02", "motherboard.01")]
    assert section.indirect_toggle.text() == INDIRECT_TITLE.format(k=1)
    assert section.indirect_texts() == ["cover.02 → psu.01 → motherboard.01 · 像素 1 200"]
    assert section.indirect.findChildren(QPushButton) == []
    # hovering a folded row lights the pair up
    enter(row_of(section, ("psu.01", "cover.01")))
    assert [h[1] for h in d04.canvas.hint_boxes()] == ["上：psu.01", "下：cover.01"]
    leave(row_of(section, ("psu.01", "cover.01")))
    enter(row_of(section, ("cover.02", "motherboard.01")))
    assert [h[1] for h in d04.canvas.hint_boxes()] == ["上：cover.02", "下：motherboard.01"]
    assert ruling_ops(d04) == []
    # 「不挡」 for the record: one op, and the pair moves to 已裁定
    free.click()
    settle()
    assert d04.session.db.cover_ruling(DESKTOP, "psu.01", "cover.01")["ruling"] == "not_blocks"
    assert len(ruling_ops(d04)) == 1
    assert section.proven_candidates == [] and section.proven_toggle.isHidden()
    assert section.ruled_texts()[0].startswith("✗ psu.01 → cover.01 不挡")


def test_no_fold_without_pairs_and_no_new_key(window):
    review(window)
    section = window.notes_panel.rulings
    assert section.proven_toggle.isHidden() and section.indirect_toggle.isHidden()
    for action in A.ACTIONS:                                     # hover and buttons only
        assert not any(word in (action.slot or "") for word in ("rul", "hover"))
        assert not any(word in (action.label_zh or "")
                       for word in ("挡着", "不挡", "裁定", "高亮", "间接", "日志已证明"))


ROOT = Path(__file__).resolve().parents[1]


def test_the_docs_say_it():
    spec = (ROOT / "docs" / "superpowers" / "specs"
            / "2026-09-17-teardown-annotator-design.md").read_text(encoding="utf-8")
    review_section = spec.split("### 4.4 审阅模式", 1)[1].split("### 4.5", 1)[0]
    [para] = [line for line in review_section.splitlines()
              if line.startswith("  **v1.51（W3f）")]
    lines = review_section.splitlines()
    bullet = lines.index(next(b for b in lines if b.startswith("- **谁压着谁的裁定**")))
    assert lines.index(para) == bullet + 2                   # appended to the W3c bullet
    for needed in ("`ImageCanvas.set_hint_boxes`", "**橙框**", "**蓝框**", "「上：key」",
                   "「above（状态）→ below（状态） · 像素 n · <标签>」", "**日志：**",
                   "**规则已间接覆盖（k）**", "**日志已证明不挡（k）**", "`would_violate`",
                   "**443**", "**74**", "**21**", "86 对", "没有快捷键"):
        assert needed in para, needed
    entries = [line for line in spec.split("## 14. 修订记录", 1)[1].splitlines()
               if line.startswith("- ")]
    # on top when it landed; whatever lands later goes above it
    [entry] = [e for e in entries[:6] if e.startswith("- **v1.51（2026-10-31）**")]
    for needed in ("（W3f，§4.4）", "`RulingsSection.sigHover", "`cover_rulings.triage_pairs`",
                   "`indirect_chain`", "`cover_log`", "`log_proof`", "`COVER_CLEAR`",
                   "间接 74、日志 21、要问 348", "间接 26、日志 10、要问 50"):
        assert needed in entry, needed
    assert spec.index("- **v1.51") < spec.index("- **v1.50")
    quick = (ROOT / "docs" / "trial_quickstart.md").read_text(encoding="utf-8")
    review_part = quick.split("## 4. 走完之后：Review", 1)[1].split("### 冲突从哪来", 1)[0]
    item = review_part.split("\n6. ", 1)[1]
    assert item.count("\n   ") == 1                         # two lines
    for needed in ("橙框", "蓝框", "「日志里下面那件先动了（k）」", "「规则已间接覆盖（k）」"):   # W3h's name
        assert needed in item, needed
