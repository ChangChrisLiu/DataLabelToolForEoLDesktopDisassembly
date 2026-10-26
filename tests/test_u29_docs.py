"""Task U29: the spec says what changed, where the behaviour is specified, and v1.29 is on top."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def _body() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_v1_29_is_on_top_of_14():
    changes = _body().split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # on top when it landed; L6's v1.30 and U32's v1.31 have gone above it since
    assert entries[0].startswith(("- **v1.29", "- **v1.30", "- **v1.31", "- **v1.32",
                                  "- **v1.33", "- **v1.34", "- **v1.35"))
    [first] = [e for e in entries[:7] if e.startswith("- **v1.29")]
    assert first.startswith("- **v1.29（2026-10-24）**：**标注流程待办清理**（U29")
    for needed in ("小零件上撤销「被线挡住（完全）」", "一段状态中间按 `Q`", "正序也不越过",
                   "captive 翻出去收回旧 fix", "`retires`", "拆前 / 拆后的检测小窗",
                   "1000 → 0 ms", "COCO 导出不再为完整形状编译每一帧", "逐字节相同",
                   "1.67 → 0.21 s", "35 → 4.8 s", "库结构不变（仍为 v5）"):
        assert needed in first, needed
    assert changes.index("- **v1.29") < changes.index("- **v1.28")


def test_the_paragraphs_say_it_where_the_behaviour_lives():
    body = _body()
    rule = body.split("### 3.4 真值表", 1)[1].split("### 3.5", 1)[0]
    assert "**一个像素都看不见 ≠ 看得见**（v1.29，U29）" in rule
    edits = body.split("### 4.3 编辑语义", 1)[1].split("### 4.4", 1)[0]
    assert "**在一段状态的中间帧**（v1.29 核对，U29）" in edits
    assert "**正序也不越过开着的状态边界**（v1.29，U29）" in edits
    assert "这一帧是状态边界（✂）：改全部会把第 a–b 步（状态）的形状一起改掉。用 Q（从这帧起新版本）" in edits
    relations = body.split("### 3.2 实例身份", 1)[1].split("### 3.3", 1)[0]
    assert "**captive 翻转收回旧 fix**（v1.29，U29，U27 R3-N1）" in relations
    pair = body.split("**拆前 / 拆后（v1.28，U28）**", 1)[1].split("### 4.6", 1)[0]
    assert "**缓存与第二个模型**（v1.29，U29，复审 N14 / N18）" in pair
    assert "第一次空下来" in pair and "`heavy`" in pair
    export = body.split("### 8.1 CV 导出", 1)[1].split("### 8.2", 1)[0]
    assert "**代价**（v1.29，U29，U26 复审 N6）" in export
    assert "`tda.core.compiler.keyframe_amodal`" in export
