"""Task U37's documents: §4.2 says the first frame of a pose segment is a start
frame, v1.36 is on top of §14, the quickstart says it under the card's
sentences, and the guide stays inside its 7700-character budget with no new
key in its table."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from tda.ui import app_actions as A
from tda.ui import guide as G

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
GUIDE = ROOT / "docs" / "annotation_guide.md"

#: What the card's header says on D48 frame 35 -- the documents quote it.
HEADER_35 = ("第 35 帧（" + G.POSE_START.format(cut=G.pose_cut_text({"cut": 36, "reorient": True}))
             + "）")


def _spec() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_the_paragraph_is_in_4_2_after_the_missing_frames():
    section = _spec().split("### 4.2 倒序标注", 1)[1].split("### 4.3", 1)[0]
    head = "**位姿段的起始帧**（v1.36，U37）"
    assert section.index("**缺帧处理**") < section.index(head)
    assert section.index(head) < section.index("**这台没有这个零件**（v1.33，U34）")
    para = section.split(head, 1)[1].split("\n", 1)[0]
    for needed in ("`reorient` 步和该视角的位姿断点切开位姿段（§2.5）", "关键帧不跨段沿用",
                   "**不在同一位姿段**", "**起点**处理", "两个方向一样", "段内照旧是差分",
                   "D48 第 36 步翻转机箱", "[1–35]、[36]", "12 行 `missing_shape`",
                   "`tda.core.truth_inputs.same_segment`", "`segment_start`",
                   "`split_rows_at`", "`shape_predates_state`", "和任务卡同一道门",
                   HEADER_35, "「第 k 步是位姿断点」",
                   "「第 35 帧（位姿段的起始帧：第 36 步翻转机箱）」",
                   "仍是「起点，已经拆完的样子」", "差分热图和 `Tab` 仍对照它", "没有新快捷键",
                   # round 2: the start before the step type (D34)
                   "**起点先于步骤类型判断**（第二轮）", "只在同一位姿段里成立",
                   "D34 最后一步「Try to remove the optical drive」是 `failed`",
                   "没有任何一张卡让人画机箱", "段内的 `failed` / `dupli` 帧照旧一行「确认」"):
        assert needed in para, needed
    # ... and item 2's one-key confirmation says where it stops
    assert ("一键确认（但可标记 `in_progress`/手入镜）。起点帧和位姿段的起始帧除外：那里照第 1 条"
            "列出全部零件（v1.36，见下文「位姿段的起始帧」）。") in section


def test_v1_36_is_on_top_of_14():
    changes = _spec().split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    [entry] = [e for e in entries[:2] if e.startswith("- **v1.36")]
    assert entry.startswith("- **v1.36（2026-10-27）**：**位姿段的起始帧按起点处理**（U37，§4.2）")
    for needed in ("「35步让我什么都不画，直接跳过了……这个是错误的」",
                   "`tda.core.truth_inputs.same_segment`",
                   "`segment_start`", "`session_tasks.segment_cut`", "`split_rows_at`",
                   "第 35 帧（位姿段的起始帧：第 36 步翻转机箱）", "D64 第 16–22 步",
                   "**第二轮**：起点先于步骤类型判断", "D34 末步「Try to remove the optical drive」",
                   "全库只有这一处", "段内的 `failed` / `dupli` 帧照旧一行「确认」",
                   "库结构不变（仍为 v6）", "没有新快捷键"):
        assert needed in entry, needed
    assert changes.index("- **v1.36") < changes.index("- **v1.35")


def test_the_quickstart_says_it_with_the_cards_sentences():
    body = QUICK.read_text(encoding="utf-8")
    section = body.split("## 3. 之后每一帧", 1)[1].split("\n2. ", 1)[0]
    assert HEADER_35 in section.replace("\n     ", "")
    for needed in ("翻转机箱（`reorient`）或位姿断点", "形状不跨段沿用", "**每一个**零件",
                   "正向翻过去也一样"):
        assert needed in section, needed


def test_the_guide_stays_in_its_budget_with_no_new_key():
    body = GUIDE.read_text(encoding="utf-8")
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700
    assert A.shortcut_markdown() in body
    section = body.split("## 5.5 位姿断点", 1)[1].split("\n## ", 1)[0]
    assert "段界旁的帧任务卡列出全部零件，没复制过来的重画。" in section
