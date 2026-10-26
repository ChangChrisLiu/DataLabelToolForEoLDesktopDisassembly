"""Task U36's documents: §4.4 says the review is over runs and across the views,
v1.35 is near the top of §14, the quickstart says it in two sentences, and the
guide stays inside its 7700-character budget with no new key in its table."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from tda.ui import app_actions as A

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
GUIDE = ROOT / "docs" / "annotation_guide.md"


def _spec() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_the_paragraph_is_in_4_4_after_u24s():
    review = _spec().split("### 4.4 审阅模式", 1)[1].split("### 4.5", 1)[0]
    head = "**可见性审查：按段、跨视角**（v1.35，U36）"
    assert review.index("**可见性审查**（v1.24，U24") < review.index(head)
    para = review.split(head, 1)[1].split("\n", 1)[0]
    for needed in ("有的 invisible 是持续了很多 frame 的", "**段是队列的单位**",
                   "**遮挡它的零件不再拆段**", "`changes`", "「第 a–b 帧（n 帧）」", "「全程」",
                   "**零件按最长的一段排**", "看不见**（`occluded_full` / `out_of_view`）",
                   "「—」没标注", "「未确认」只有草稿", "**帧条**", "第一、中间、最后一帧",
                   "**其他视角表**", "只比**已确认帧**", "绿底", "**建议**", "「角度看不到」（`Z` / `Shift+Z`）",
                   "「被挡住」", "**键还是标注员按**", "没有快捷键", "**「从这帧起另算」**", "不写库",
                   "**「同步到其他视角」**", "**从不覆盖判定**", "那边编译说有零件挡着的帧",
                   "`sync_visibility`", "一次 `Ctrl+Z`", "U15 的路子", "`cli check --visibility`",
                   "`visibility_run <零件> a= b= n= label=", "verdict=yes|no",
                   "「可见性审查 r/t（待审 k 段）」"):
        assert needed in para, needed
    # the U24 paragraph says where its run rule was changed
    assert "（v1.35 起遮挡它的零件不再拆段、零件按最长的一段排，见下一条）" in review


def test_v1_35_is_near_the_top_of_14():
    changes = _spec().split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # U33-U35 land v1.32-v1.34 alongside: the controller orders them at merge
    [entry] = [e for e in entries[:4] if e.startswith("- **v1.35")]
    assert entry.startswith("- **v1.35（2026-10-26）**：**可见性审查按段、跨视角**（U36，§4.4）")
    for needed in ("`changes`", "「第 a–b 帧（n 帧）」", "「全程」", "「从这帧起另算」",
                   "「同步到其他视角」", "`sync_visibility`", "`cli check --visibility`",
                   "「可见性审查 r/t（待审 k 段）」", "`tda/core/vis_runs.py`",
                   "库结构不变（仍为 v5）", "830 段 / 8 692 帧", "D13 / D24 / D33 的草稿"):
        assert needed in entry, needed
    assert changes.index("- **v1.35") < changes.index("- **v1.31")


def test_the_quickstart_says_it_in_two_sentences():
    body = QUICK.read_text(encoding="utf-8")
    section = body.split("### 最后的可见性审查", 1)[1].split("\n## ", 1)[0]
    assert len(section) < 1000
    for needed in ("「第 a–b 帧（n 帧）」", "最长的一段排最前", "各视角看不见几帧 / 几段",
                   "首、中、尾三帧", "**其他视角同样这几步**", "绿底", "键还是你按",
                   "「从这帧起另算」", "「同步到其他视角」", "不覆盖审过的"):
        assert needed in section, needed


def test_the_guide_stays_in_its_budget_with_no_new_key():
    body = GUIDE.read_text(encoding="utf-8")
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700
    assert A.shortcut_markdown() in body
    assert ("- **可见性审查**：标注时不设可见性；最后逐条核对确认帧里非 `visible` 的标签"
            "（按段、跨视角；键见第 7 节）。") in body
