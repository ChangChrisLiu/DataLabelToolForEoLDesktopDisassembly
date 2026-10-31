"""Task W3c: 「谁压着谁 · 这台还没裁定的（n）」 in the 「说明」 dock (offscreen Qt).

* the rows render (text, tag, reason as tooltip), in the rule's order, above the
  occlusion list; a batch line for each class pair shared by two or more rows;
* 「挡着」 writes one ``cover_ruling`` op (view = the open view, step = the frame
  on screen) and the row goes; the status line says it; ``Ctrl+Z`` takes it back;
* 「已裁定（k）」 lists the rulings with ✓ / ✗; 「撤销」 clears one (one op);
* the batch line writes its pairs as one op;
* the section shows in Review only and not on a components-gallery view; its
  fold is remembered (``notes/rulings_open``); no key was added; nothing is
  compiled for it.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton

import app_scene
from tda.core import masks
from tda.core.db_rulings import OP_KIND
from tda.core.model import ShapeKeyframe, ShapePart, ZOrderRec
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.app_rulings import RULINGS_SETTING
from tda.ui.cover_rulings import TAG_OTHER, TAG_RECOMMENDED, TAG_UNCERTAIN
from tda.ui.panels.notes_dock import NONE_PENDING, RULINGS_TIP

DESKTOP, VIEW = app_scene.DESKTOP, app_scene.VIEW


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def box(y0, y1, x0, x1) -> np.ndarray:
    mask = np.zeros(app_scene.HW, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def seed_pile(session) -> dict:
    """Bottom-up: the motherboard (the whole frame), the PSU over 60 columns, two
    covers over the PSU's top and bottom -- every overlap >= 1000 px."""
    shapes = {"motherboard.01": box(0, 64, 0, 64), "psu.01": box(0, 64, 0, 60),
              "cover.01": box(0, 30, 0, 40), "cover.02": box(34, 64, 0, 40)}
    for key, mask in shapes.items():
        session.db.add_keyframe(ShapeKeyframe(
            id=None, instance=key, desktop=DESKTOP, view=VIEW, pose_segment=1,
            anchor_step=app_scene.LAST_STEP, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(mask))]))
    session.db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, [(k, "main") for k in shapes]))
    session.refresh_all()
    return shapes


def open_window(tmp_path: Path, seed=seed_pile):
    session = app_scene.make_session(tmp_path)
    if seed is not None:
        seed(session)
    win = MainWindow(session, app_scene.make_paths(tmp_path), "tester",
                     sam_queue=app_scene.StubSamQueue())
    win.resize(1200, 900)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    if win.roi_editing:
        win.act_clear_edit()
    QApplication.processEvents()
    return win


@pytest.fixture
def window(qapp, tmp_path: Path):
    win = open_window(tmp_path)
    yield win
    app_scene.close_window(win)


def settle() -> None:
    for _ in range(3):
        QApplication.processEvents()


def review(win) -> None:
    win.set_mode(A.MODE_REVIEW)
    settle()


def ruling_ops(win) -> list[dict]:
    return win.session.db.ops_of_kind(DESKTOP, OP_KIND)


PENDING = [("cover.01", "psu.01", TAG_RECOMMENDED), ("cover.02", "psu.01", TAG_RECOMMENDED),
           ("psu.01", "motherboard.01", TAG_UNCERTAIN),
           ("cover.01", "motherboard.01", TAG_OTHER), ("cover.02", "motherboard.01", TAG_OTHER)]


def test_the_rows_render_above_the_occlusion_list(window):
    review(window)
    section = window.notes_panel.rulings
    assert not section.isHidden() and section.is_open()
    layout = window.notes_panel.layout()
    assert layout.indexOf(section) < layout.indexOf(window.notes_panel.split)
    assert [(c.above, c.below, c.tag) for c in section.candidates] == PENDING
    assert section.toggle.text() == "谁压着谁 · 这台还没裁定的（5）"
    texts = section.texts()
    assert texts[0] == "cover.01 → psu.01 · 像素 1 200 · 建议收"
    assert texts[2] == "psu.01 → motherboard.01 · 像素 1 440 · 存疑"
    assert texts[5:] == ["同类对本帧全部：cover → psu（2 对）",
                         "同类对本帧全部：cover → motherboard（2 对）"]
    cand, blocks, free = section.row_buttons[2]
    assert (blocks.text(), free.text()) == ("挡着", "不挡")
    assert "PSU 悬在主板一角上方" in section._row_labels[2].toolTip()
    assert section.toolTip() == RULINGS_TIP and "没有快捷键" in RULINGS_TIP
    for button in section.findChildren(QPushButton):
        assert button.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert section.ruled_toggle.text() == "已裁定（0）" and not section.ruled_toggle.isChecked()
    assert window.rulings_builds >= 1 and window.rulings_ms < 50.0


def test_a_click_writes_one_op_and_ctrl_z_takes_it_back(window):
    review(window)
    section = window.notes_panel.rulings
    step = int(window.session.current().step)
    _cand, blocks, _free = section.row_buttons[2]                  # psu.01 → motherboard.01
    blocks.click()
    settle()
    [op] = ruling_ops(window)
    assert (op["view"], op["annotator"]) == (VIEW, "tester")
    row = window.session.db.cover_ruling(DESKTOP, "psu.01", "motherboard.01")
    assert (row["ruling"], row["view"], row["step"], row["by"]) == ("blocks", VIEW, step, "tester")
    assert window.status_message() == f"D13 第 {step} 帧：psu.01 压着 motherboard.01 → 挡着"
    assert ("psu.01", "motherboard.01") not in [(c.above, c.below) for c in section.candidates]
    assert section.toggle.text() == "谁压着谁 · 这台还没裁定的（4）"
    assert section.ruled_texts() == [f"✓ psu.01 → motherboard.01 挡着（{VIEW} 第 {step} 帧）"]
    assert section.ruled_toggle.text() == "已裁定（1）"

    window.act_undo()
    settle()
    assert window.session.db.cover_rulings(DESKTOP) == {}
    assert "「谁压着谁」裁定（1 对）" in window.status_message()
    assert [(c.above, c.below, c.tag) for c in section.candidates] == PENDING
    window.act_redo()
    settle()
    assert window.session.db.cover_ruling(DESKTOP, "psu.01", "motherboard.01")["ruling"] \
        == "blocks"
    assert len(ruling_ops(window)) == 1             # undo / redo replay; they log nothing


def test_undo_in_the_ruled_list_clears_one_ruling_in_one_op(window):
    review(window)
    section = window.notes_panel.rulings
    section.row_buttons[0][2].click()                               # cover.01 → psu.01 不挡
    settle()
    assert section.ruled_texts()[0].startswith("✗ cover.01 → psu.01 不挡")
    section.ruled_toggle.click()
    assert not section.ruled_area.isHidden()
    [(pair, undo)] = section.ruled_buttons
    assert pair == ("cover.01", "psu.01") and undo.text() == "撤销"
    undo.click()
    settle()
    assert window.session.db.cover_rulings(DESKTOP) == {}
    assert len(ruling_ops(window)) == 2
    assert "去掉了「cover.01 压着 psu.01」的裁定" in window.status_message()
    assert [(c.above, c.below) for c in section.candidates][0] == ("cover.01", "psu.01")
    window.act_undo()                                               # the clear, taken back
    settle()
    assert window.session.db.cover_ruling(DESKTOP, "cover.01", "psu.01")["ruling"] == "not_blocks"


def test_the_batch_line_writes_its_pairs_as_one_op(window):
    review(window)
    section = window.notes_panel.rulings
    [(classes, pairs, blocks, _free)] = [b for b in section.batch_buttons
                                         if b[0] == ("cover", "motherboard")]
    assert pairs == [("cover.01", "motherboard.01"), ("cover.02", "motherboard.01")]
    blocks.click()
    settle()
    [op] = ruling_ops(window)
    assert [(r["above"], r["below"], r["ruling"]) for r in op["payload"]["rows"]] == [
        ("cover.01", "motherboard.01", "blocks"), ("cover.02", "motherboard.01", "blocks")]
    step = int(window.session.current().step)
    assert window.status_message() == f"D13 第 {step} 帧：2 对 cover 压着 motherboard → 挡着"
    assert [c.below for c in section.candidates] == ["psu.01", "psu.01", "motherboard.01"]
    assert [b[0] for b in section.batch_buttons] == [("cover", "psu")]
    window.act_undo()
    settle()
    assert window.session.db.cover_rulings(DESKTOP) == {}


def test_hidden_outside_review_and_on_a_gallery_view(window):
    section = window.notes_panel.rulings
    assert window.notes_dock.isHidden() and not window.rulings_apply()      # Annotate
    review(window)
    assert not section.isHidden() and window.rulings_apply()
    session = window.session
    view = session.view
    try:
        session.view = "cmp_oak1"
        window.refresh_rulings()
        assert section.isHidden() and not window.rulings_apply()
    finally:
        session.view = view
    window.refresh_rulings()
    assert not section.isHidden()
    window.set_mode(A.MODE_ANNOTATE)
    settle()
    assert not window.rulings_apply()
    # a click arriving outside Review writes nothing
    window.on_rule_pairs([("psu.01", "motherboard.01")], "blocks", None)
    assert ruling_ops(window) == []


def test_the_fold_is_remembered(qapp, tmp_path: Path):
    win = open_window(tmp_path)
    try:
        review(win)
        section = win.notes_panel.rulings
        section.toggle.click()                                       # fold it
        assert not section.is_open() and section.body.isHidden()
        assert str(win.settings.value(RULINGS_SETTING)).lower() == "false"
        win.set_mode(A.MODE_ANNOTATE)
    finally:
        app_scene.close_window(win)
    again = open_window(tmp_path, seed=None)
    try:
        review(again)
        section = again.notes_panel.rulings
        assert not section.is_open() and section.body.isHidden()
        section.toggle.click()
        assert section.is_open() and not section.body.isHidden()
        assert str(again.settings.value(RULINGS_SETTING)).lower() == "true"
    finally:
        app_scene.close_window(again)


def test_a_frame_with_nothing_to_rule_says_so(qapp, tmp_path: Path):
    win = open_window(tmp_path, seed=None)
    try:
        review(win)
        section = win.notes_panel.rulings
        assert section.candidates == [] and not section.empty.isHidden()
        assert section.empty.text() == NONE_PENDING
    finally:
        app_scene.close_window(win)


def test_an_edge_or_a_ruling_from_elsewhere_drops_the_row(window, monkeypatch):
    review(window)
    window.session.db.add_relation(DESKTOP, "covered_by", "motherboard.01", "psu.01")
    window.session.db.set_cover_ruling(DESKTOP, "cover.02", "psu.01", "blocks", view="oak1",
                                       step=3, by="other")
    compiles = []
    real = window.session.truth.compile
    monkeypatch.setattr(window.session.truth, "compile",
                        lambda *a, **k: compiles.append(a) or real(*a, **k))
    writes = window.session.db.conn.total_changes
    window.refresh_rulings()
    section = window.notes_panel.rulings
    assert [(c.above, c.below) for c in section.candidates] == [
        ("cover.01", "psu.01"), ("cover.01", "motherboard.01"), ("cover.02", "motherboard.01")]
    assert section.ruled_texts() == ["✓ cover.02 → psu.01 挡着（oak1 第 3 帧）"]
    assert compiles == [] and window.session.db.conn.total_changes == writes


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def test_the_docs_say_it():
    spec = SPEC.read_text(encoding="utf-8")
    review_section = spec.split("### 4.4 审阅模式", 1)[1].split("### 4.5", 1)[0]
    bullets = [line for line in review_section.splitlines() if line.startswith("- **")]
    heads = [b.split("**")[1] for b in bullets]
    assert heads.index("谁压着谁的裁定") == heads.index("说明窗与遮挡顺序") + 1
    [bullet] = [b for b in bullets if b.startswith("- **谁压着谁的裁定**（v1.46，W3c）")]
    for needed in ("「谁压着谁需要我结合图去给出", "`cover_ruling (desktop, above, below, ruling ∈ "
                   "{blocks, not_blocks}, by, ts, view, step, note)`", "库结构 **v7**",
                   "**W3a 的 `pixel-edges` 按这些列名读它**", "**≥ 1000**", "**支承表**",
                   "**「挡着」**", "**「不挡」**", "「同类对本帧全部", "「已裁定（k）」", "「撤销」",
                   "`Ctrl+Z`", "**没有快捷键**", "`QSettings notes/rulings_open`",
                   "「D13 第 7 帧：optical_drive.01 压着 motherboard.01 → 挡着」"):
        assert needed in bullet, needed
    changes = spec.split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # on top when it landed; W3a / W3b (v1.46 / v1.47) are ordered at their merges
    assert any(e.startswith("- **v1.46（2026-10-31）**：**复查里的「谁压着谁」裁定**（W3c，§4.4）")
               for e in entries[:3])
    quick = (ROOT / "docs" / "trial_quickstart.md").read_text(encoding="utf-8")
    review_part = quick.split("## 4. 走完之后：Review", 1)[1].split("### 冲突从哪来", 1)[0]
    item = review_part.split("\n5. ", 1)[1]
    assert item.count("\n   ") == 2                        # three lines
    for needed in ("「谁压着谁 · 这台还没裁定的（n）」", "「挡着」", "「不挡」", "「同类对本帧全部」",
                   "`Ctrl+Z`", "「撤销」", "没有快捷键"):
        assert needed in item, needed
    handoff = (ROOT / "docs" / "HANDOFF_2026-10-28.md").read_text(encoding="utf-8")
    section = handoff.split("## 5. 导出与 HuggingFace 准备", 1)[1].split("\n## ", 1)[0]
    assert "`cover_ruling` 表由 W3a 的 `pixel-edges` 读取" in section


def test_no_key_was_added():
    for action in A.ACTIONS:
        assert "rul" not in (action.slot or "")
        assert not any(word in (action.label_zh or "") for word in ("挡着", "不挡", "裁定"))
