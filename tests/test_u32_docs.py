"""Task U32: the spec says what 「重签」 does where the truth table is specified,
Review says where the buttons are, v1.31 is on top of §14, and the quickstart
has its sentence."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"


def _body() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_the_rule_is_in_3_4():
    rule = _body().split("### 3.4 真值表", 1)[1].split("### 3.5", 1)[0]
    head = "**容差以内的差别：「重签」**（v1.31，U32）"
    assert rule.index("**一个像素都看不见 ≠ 看得见**（v1.29，U29）") < rule.index(head)
    para = rule.split(head, 1)[1].split("\n", 1)[0]
    for needed in ("**两个确认掩码重叠**", "5 227 px", "**谁的都不是**", "`TruthService.refreeze`",
                   "`cli refreeze --desktop N --view scan", "`--dry-run`", "--undo last|<run>",
                   "仍是 `verified`", "超出容差的、有开着冲突的仍是冲突", "只报告不代签",
                   "**等冲突处理完**（`deferred`", "D22",
                   "`refreeze_frame` op", "逐列逐字节", "再跑一遍什么都不写",
                   "`stale_within_tolerance: n (deferred m)`", "不影响退出码", "D2–D21",
                   # round 2
                   "**比较的参考是人签的那一版**", "`tda/core/truth_anchor.py`", "`drifted`",
                   "第 2 轮开冲突", "带 `--undo` 时列出会放回的帧", "不带 `--steps`",
                   "`safety_backup`", "**先重签、再做可见性审查**", "`Ctrl+Z` 不撤"):
        assert needed in para, needed


def test_the_buttons_are_in_4_4():
    review = _body().split("### 4.4 审阅模式", 1)[1].split("### 4.5", 1)[0]
    head = "**重签**（v1.31，U32）"
    para = review.split(head, 1)[1].split("\n", 1)[0]
    for needed in ("「重签这一帧」", "「重签全部」", "没有快捷键", "「重签 n 帧（m 行，k px）？」",
                   "「已重签 n 帧、m 行（k px）；重叠 a → b px」", "翻转的标签",
                   "`python -m tda.cli refreeze --desktop N --view V --undo last`", "先备份"):
        assert needed in para, needed


def test_v1_31_is_on_top_of_14():
    changes = _body().split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # on top when it landed; U33-U37's v1.32-v1.36 go above it
    assert entries[0].startswith(("- **v1.31", "- **v1.32", "- **v1.33", "- **v1.34",
                                  "- **v1.35", "- **v1.36",
                                  "- **v1.37", "- **v1.38", "- **v1.39", "- **v1.40"))
    [entry] = [e for e in entries[:10] if e.startswith("- **v1.31")]
    assert entry.startswith("- **v1.31（2026-10-25）**：**重签：容差以内的冻结行改成今天的编译**"
                            "（U32，§3.4、§4.4）")
    for needed in ("262 408", "`tda/core/truth_refreeze.py`", "`cli refreeze", "--undo last",
                   "「重签这一帧」", "`stale_within_tolerance: n`", "`refreeze_frame`",
                   "库结构不变（仍为 v5）", "评审第二轮", "`--undo --dry-run`",
                   "`stale_within_tolerance: n (deferred m)`"):
        assert needed in entry, needed
    assert changes.index("- **v1.31") < changes.index("- **v1.30")


def test_the_quickstart_says_it_once():
    quick = QUICK.read_text(encoding="utf-8")
    review = quick.split("### 冲突从哪来、怎么一次解决", 1)[1].split("### 最后的可见性审查", 1)[0]
    assert "「重签这一帧」/「重签全部」" in review
    assert "`stale_within_tolerance`" in review
    # round 2 (N7 / N8): the undo, and the tolerance as it is
    assert "**撤销不能用 `Ctrl+Z`**" in review and "--undo last" in review
    assert "边界 2 px 以内的差不算，其余的差不超过 max(面积的 2 %, 20 px)" in review
