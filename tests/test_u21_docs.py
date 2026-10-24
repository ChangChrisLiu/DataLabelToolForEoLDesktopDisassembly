"""Task U21: the docs say what 「其实是硬盘架」 does, and the guide's prose does not grow.

The guide gets only its generated ``Shift+T`` row (the action's label); the
quickstart one short section with the D3 example; the spec a paragraph in §3.2
(with the physical model of the locker) and v1.21 on top of §14.
"""
from __future__ import annotations

from pathlib import Path

from tda.ui import app_actions as A

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "annotation_guide.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def test_the_guide_gets_only_its_shift_t_row():
    body = GUIDE.read_text(encoding="utf-8")
    action = next(a for a in A.ACTIONS if a.name == "correct_verb")
    assert "其实是硬盘架 / 其实是卡扣" in action.label_zh
    [row] = [line for line in body.splitlines() if line.startswith("| `Shift+T` |")]
    assert action.label_zh in row
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert not [line for line in prose if "其实是硬盘架" in line]
    assert sum(len(line) for line in prose) <= 7700


def test_the_quickstart_has_one_short_section_with_the_d3_example():
    body = QUICK.read_text(encoding="utf-8")
    head = "### 硬盘架被记成了卡扣（`Shift+T` →「其实是硬盘架」）"
    assert body.count(head) == 1
    section = body.split(head, 1)[1].split("\n### ", 1)[0].replace("\n", "")
    for needed in ("HDD Locker Shield", "第 10 帧", "「其实只是打开了」", "小窗里选「其实是硬盘架」",
                   "`drive_cage.01`", "第 11 步仍是拆下", "第 10 帧多了机架的卡扣",
                   "从不从机架上拆下", "大螺丝", "D4", "隐含卡扣", "回到待复核", "`Ctrl+Z`",
                   "右键「其实是硬盘架…」", "`import-logs --force`"):
        assert needed in section, needed
    assert len(section) < 600


def test_the_spec_has_the_paragraph_in_3_2_and_v1_21_on_top():
    body = SPEC.read_text(encoding="utf-8")
    section = body.split("### 3.2 实例身份", 1)[1].split("### 3.2a", 1)[0]
    assert "**改类别：其实是硬盘架 / 其实是卡扣（v1.21，U21）**" in section
    for needed in ("从不从机架上拆下来", "大螺丝为轴转动", "中心词（最后一个词）",
                   "「… locker shield」是 shield，即机架 `drive_cage`", "D4 的 \"HDD Shield Locker\"",
                   "只有 D2、D3", "合并进另一类的一个新实例", "reclassed_record",
                   "机架规则跟着跑", "`reclass_instance` op", "「其实拆下来了」/「其实只是打开了」",
                   "Steps → Instances 右键「其实是硬盘架…」", "挂上机架", "`cage.repoint`",
                   "reclass_replay.py", "步号和原始名称", "「reclass kept: drive_latch.02 -> "
                   "drive_cage.01 (…)」", "reclass not matched", "已知限制",
                   # review round: M1 and M2
                   "按每个已确认帧**存着的行**", "5 帧，不是 32 帧",
                   "**把新键作 `gone` 并进 S1 加的机架**", "**静默撤掉改类别**",
                   "locker 被解开", "locker 的 `parent` 丢了", "S1 机架的拆下动作全丢"):
        assert needed in section, needed
    changes = body.split("## 14. 修订记录", 1)[1]
    # v1.22 (U22, the left-hand keys) went on top since
    first = next(line for line in changes.splitlines() if line.startswith("- **v1.21"))
    assert first.startswith("- **v1.21（2026-10-23）**：**其实是硬盘架**（U21，§3.2）")
    assert "「合并，重叠处保留 <keep>」" in first
    assert changes.index("- **v1.21") < changes.index("- **v1.20")


def test_the_spec_says_what_overlap_keep_does_in_3_2a():
    body = SPEC.read_text(encoding="utf-8")
    section = body.split("### 3.2a", 1)[1].split("### 3.2b", 1)[0]
    for needed in ("**重叠处保留（v1.21 评审轮，U21）**", "`OverlapError`",
                   "「合并，重叠处保留 <keep>」", "`merge_instances(..., overlap=\"keep\")`",
                   "撤销逐表放回", "D5 的 locker", "形状锚在 9、12、13",
                   "第 10–12 帧用手加那块第 12 帧画的形状", "通常是两个零件", "默认按钮是取消"):
        assert needed in section, needed
