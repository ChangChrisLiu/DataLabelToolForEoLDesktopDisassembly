"""Task U39's documents: §4.2 says what 画框外 is and how it differs from 被挡住,
§4.4 how the review treats it, §3.2b no longer says the sequence reports it,
v1.38 is near the top of §14, the quickstart says it in a paragraph, and the
guide's key table names it under ``Ctrl+``` with the prose inside its
7700-character budget and no new key."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from tda.ui import app_actions as A
from tda.ui import visreview_text as T
from tda.ui.app_cable import OFF_FRAME_OFF
from tda.ui.panels.taskcard import OFF_FRAME_SENTENCE
from tda.ui.session_cable import OFF_FRAME_CHOICE

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
GUIDE = ROOT / "docs" / "annotation_guide.md"
HEAD = "**画框外**（v1.38，U39）"


def _spec() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_the_paragraph_is_at_the_end_of_4_2():
    section = _spec().split("### 4.2 倒序标注", 1)[1].split("### 4.3", 1)[0]
    assert section.index("**这台没有这个零件**（v1.33，U34）") < section.index(HEAD)
    para = section.split(HEAD, 1)[1].split("\n", 1)[0]
    for needed in ("D63 的导风罩是存在的，但是在主视角的画面外", "**和「被挡住」不同**",
                   "照画完整形状（amodal）", "**画面之外**", "「这台没有这个零件」",
                   "`visibility = out_of_view`、`visible_rle` 为空、`occluder` 为空",
                   "`tda.core.occluder.OFF_FRAME`",
                   "**入口只在 `` Ctrl+` `` 的对话框里**", f"第四项「{OFF_FRAME_CHOICE}」",
                   "一次 `Ctrl+Z`", "不经过它，免得误触", "按 `` ` `` 就是撤回",
                   "「撤回（被线挡住 / 画框外）」", "不要求先有形状", "已拆下", "台面上",
                   "当作整件的估计", "U10 第二轮", "**不报 `missing_shape`**",
                   "`tda.core.compiler.marked_absent`", "`bench_missing` 不变",
                   "`session_coverage`", "`Space`", "`cli check`", "芯片「画框外」",
                   "实例表可见性列写「框外」", "U2b 第三轮", "⚠ 的「已设为不可见」",
                   "状态、位置照常", "U15 的路子",
                   "COCO 这一帧没有它的行", "`export_scanpack`", "`clean` 不算它", "`KeyError`",
                   "VLM 写 `out_of_view`", "§4.4", "没有 `out_of_view` 的帧覆盖行",
                   "没有新快捷键"):
        assert needed in para, needed
    # the card's own sentence is the one quoted
    assert OFF_FRAME_SENTENCE.split("，不用画")[0] in para


def test_4_4_says_how_the_review_treats_it():
    review = _spec().split("### 4.4 审阅模式", 1)[1].split("### 4.5", 1)[0]
    para = review.split("**可见性审查：按段、跨视角**（v1.35，U36）", 1)[1].split("\n", 1)[0]
    sentence = para.split("**画框外**（v1.38，U39，§4.2）", 1)[1]
    for needed in ("手动的 `out_of_view` 就是画框外", "「out_of_view（原因: 画框外，手动）」",
                   "「画框外 n」", "从不把它当「角度看不到」", "从不同步它",
                   "`SYNC_LABELS` 不再含 `out_of_view`"):
        assert needed in sentence, needed
    assert T.SYNC_REFUSED["off_frame"] in sentence
    assert "只用于看不见的标签（完全 / 部分遮挡；画面外 v1.38 起不同步，见下）" in para


def test_3_2b_no_longer_says_the_sequence_reports_it():
    gallery = _spec().split("### 3.2b 零件合照", 1)[1].split("### 3.2c", 1)[0]
    assert "序列帧上照旧报" not in gallery
    assert "（序列帧上同一行是「画框外」，v1.38 起同样不报，见 §4.2）" in gallery


def test_v1_38_is_near_the_top_of_14():
    changes = _spec().split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # U38's v1.37 lands alongside: the controller orders them at merge
    [entry] = [e for e in entries[:3] if e.startswith("- **v1.38")]
    assert entry.startswith("- **v1.38（2026-10-27）**：**画框外**（U39，§4.2、§4.4、§3.2b）")
    for needed in ("D63 的导风罩", "`tda.core.occluder.OFF_FRAME`", "`off_frame_applies`",
                   f"第四项「{OFF_FRAME_CHOICE}」", "`marked_absent`", "`bench_missing` 不变",
                   "「out_of_view（原因: 画框外，手动）」", "「画框外 n」",
                   "`vis_runs.OFF_FRAME_REASON`", "`SYNC_LABELS` 去掉 `out_of_view`",
                   "`export_scanpack`", "`KeyError`", "U24 / U36 队列行 psu.01",
                   "U10 的", "库结构不变（仍为 v6）", "没有新快捷键"):
        assert needed in entry, needed
    assert changes.index("- **v1.38") < changes.index("- **v1.36")


def test_the_quickstart_says_it():
    body = QUICK.read_text(encoding="utf-8")
    section = body.split("### 被挡住 / 看不见的零件", 1)[1].split("\n### ", 1)[0]
    para = section.split("**零件在机器里，但在这个视角的画面外**", 1)[1]
    for needed in ("D63 的导风罩", "这不是被挡住", f"「{OFF_FRAME_CHOICE}」", "不用先画形状",
                   "任务卡那一条写\n「画框外」", "实例表写「框外」", "`Space` 不再因为它缺形状被拒",
                   "`` ` `` 不会标出画框外"):
        assert needed in para, needed
    assert "撤回画框外" in OFF_FRAME_OFF


def test_the_guide_names_it_with_no_new_key_inside_its_budget():
    body = GUIDE.read_text(encoding="utf-8")
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700
    assert A.shortcut_markdown() in body
    row = next(line for line in body.splitlines() if line.startswith("| `` Ctrl+` `` |"))
    assert "（完全 / 部分 / 撤回 / 画框外；一次撤销）" in row
    assert A.action_named("cable_range").keys == ("Ctrl+`",)
    assert A.action_named("mark_absent").keys == ("`",)
