"""Task U24's documents: the guide (moved, not grown), the quickstart, the spec.

The guide's prose is at its 7700-character budget: the 可见性审查 is the fifth
queue of §6 -- the §7 line about the Ctrl+digit keys moved there, rewritten --
and the generated key table says the digits are Review's. The quickstart has a
short 「最后的可见性审查」 section; the spec has the fifth queue in §4.4,
``viewpoint`` in §6.2 and v1.24 on top of §14.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from tda.ui import app_actions as A

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "annotation_guide.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def _section(body: str, head: str) -> str:
    return body.split(head, 1)[1].split("\n## ", 1)[0]


def test_the_guide_moved_the_label_keys_to_review_without_growing():
    body = GUIDE.read_text(encoding="utf-8")
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700
    assert A.shortcut_markdown() in body                      # regenerated
    for n in range(1, 8):
        row = next(line for line in body.splitlines() if line.startswith(f"| `{n} / Ctrl+{n}`"))
        assert "可见性审查" in row and "可见性审查（Review 模式）" in row
    queues = _section(body, "## 6. ")
    assert "五个标签页" in queues
    assert "- **可见性审查**：标注时不设可见性；最后逐条核对确认帧里非 `visible` 的标签" in queues
    keys = _section(body, "## 7. ")
    assert "`Ctrl+1`–`7`" not in keys and "设可见性" not in keys
    assert "| **Review（复查）** | 只读画布 + 右侧队列 | 五个队列见第 6 节。" in body


def test_the_quickstart_has_the_last_visibility_review():
    body = QUICK.read_text(encoding="utf-8")
    section = body.split("### 最后的可见性审查", 1)[1].split("\n## ", 1)[0]
    assert len(section) < 900
    for needed in ("每个视角各做一次", "「可见性审查」标签页", "「自动」", "「手动」", "原因: 线缆",
                   "`Space`", "`Shift+Space`", "数字键 `1`–`7`", "`` ` `` 被线挡住", "**`Z` 角度看不到**",
                   "`Shift+Z`", "`Ctrl+Z`", "**角度和线缆是两个原因，分开标**", "什么也没挡住它",
                   "`viewpoint`", "`cable`", "可见性审查完成", "status"):
        assert needed in section, needed
    # the walk-through no longer tells the annotator to set labels while drawing
    walk = body.split("## 4. ", 1)[0]
    assert "按 `Ctrl`+数字键设可见性" not in walk
    assert "**标注时不用管可见性标签**" in walk
    assert "右下五个队列" in body


def test_the_spec_has_the_fifth_queue_viewpoint_and_v1_24_on_top():
    body = SPEC.read_text(encoding="utf-8")
    review = body.split("### 4.4 审阅模式", 1)[1].split("### 4.5", 1)[0]
    assert "**可见性审查队列**（v1.24）" in review
    for needed in ("**可见性审查**（v1.24，U24，第五个队列 `visibility`）", "已确认帧",
                   "减去已盖审查戳的", "「ram_latch.03 第 1–13 帧 · occluded_full（被硬盘支架挡住，自动）」",
                   "同一绘制组（placement）", "不编译", "`Space` 确认（按计算）", "`Shift+Space` 整段确认",
                   "`Z` / `Shift+Z` **角度看不到**", "`review_visibility`", "U15 的路子",
                   "`instance_frame_flags.difficulty_flags_json`", "无迁移",
                   "\"vis_reviewed\": true", "回到队列", "「可见性审查完成」",
                   "「可见性审查 r/t」", "`attributes.visibility_reviewed`"):
        assert needed in review, needed
    states = body.split("### 6.2 ", 1)[1].split("### 6.3", 1)[0]
    assert "**角度看不到 `viewpoint`**（v1.24，U24）" in states
    for needed in ("`occluder` 的第三个取值", "和被线挡住是两个原因", "`Z` = 完全",
                   "`Shift+Z` = 部分", "`attributes.occluder = \"viewpoint\"`",
                   "not visible from this viewpoint", "从这个视角看不到"):
        assert needed in states, needed
    changes = body.split("## 14. 修订记录", 1)[1]
    first = next(line for line in changes.splitlines() if line.startswith("- "))
    assert first.startswith("- **v1.24（2026-10-23）**：**可见性审查**（U24，§4.4、§6.2）")
    for needed in ("`QUEUE_VISIBILITY`", "`viewpoint`", "`review_visibility`",
                   "`attributes.visibility_reviewed`", "「可见性审查 n/m」", "无迁移"):
        assert needed in first, needed
    assert changes.index("- **v1.24") < changes.index("- **v1.23")
