"""Task U6: the two documents say what the annotator now has to know.

The quickstart's "一般 Enter 接受" is the sentence the D13 annotator followed
into a layering change; it is gone, and both documents say that Enter writes
the shape only, that the layering is ``L``'s and asks first, what its reach
is, that a part never gets a layering suggestion against what it sits on, and
what a failed SAM load looks like.
"""
from __future__ import annotations

from pathlib import Path

from tda.ui import app_actions as A

REPO = Path(__file__).resolve().parents[1]


def _flat(name: str) -> str:
    return " ".join((REPO / "docs" / name).read_text(encoding="utf-8").split())


def test_the_quickstart_no_longer_sends_anybody_to_enter_for_a_layering_change():
    quick = _flat("trial_quickstart.md")
    assert "一般 `Enter` 接受" not in quick and "一般 `Q` 接受" not in quick
    # the commit key is Q since task U22 (Enter its alias)
    assert "`Q`（**只改形状**）" in quick
    assert "层级改动作用于整个位姿段：第 1–40 帧，其中已确认的第 39、40 帧会被重新检查" in quick
    assert "**层级不是形状版本**：它不能按帧拆分" in quick
    # U16: the frame where the version starts is the commit's re-trace now; it
    # was Ctrl+K's "没有拆分" (the key that splits is the commit, Ctrl+K its alias)
    assert "`Q` 就是重画这一版" in quick
    assert "Ctrl+K 等于重画这一版" not in quick.replace(" ", "")
    assert "SAM 加载失败：原因（F5 重试）" in quick
    assert "SAM 没加载：用 R 圆形 / W 画笔 / D 多边形" in quick      # Y / B / P until U22
    assert "检测器找到的螺丝（SAM 未加载： R 拖一下就画好）" in quick or \
        "检测器找到的螺丝（SAM 未加载：R 拖一下就画好）" in quick


def test_the_guide_says_it_in_section_4_and_the_key_table():
    guide = _flat("annotation_guide.md")
    assert "层级只有 `L` 改，其余键只写形状" in guide
    assert "**整个位姿段**" in guide and "不能按帧拆" in guide
    # round 2: every relation the code follows, and the 90 % rule
    assert "零件与所在零件间不出层级建议" in guide
    for word in ("`parent`", "`mounted_on`", "`fastens`", "`socket_host`", "宿主类",
                 "九成以上在对方形状内"):
        assert word in guide, word
    assert "输入恢复一致就自动关（`check`、导出报数）" in guide
    assert "SAM 加载失败：原因（F5 重试）" in guide
    action = A.action_named("commit_layering")
    assert action.keys == ("L",) and action.label_zh in guide
