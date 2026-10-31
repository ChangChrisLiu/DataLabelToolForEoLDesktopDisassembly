"""Task U23: the docs say what 「主视角」 does, and the guide's prose does not grow.

The guide is left alone (its prose is at the 7700 limit and the panel explains
itself: caption, tooltip, the sentence under the picture); the quickstart gets
one short section, the spec a paragraph in §4.5 (the layout, where §4.5 already
asked for "another view's same step as reference") and v1.23 on top of §14.
"""
from __future__ import annotations

from pathlib import Path

from tda.ui.locator_source import NOT_ANNOTATED, NOT_DRAWN, NOT_HERE

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "annotation_guide.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def test_the_guide_prose_stays_within_its_limit():
    body = GUIDE.read_text(encoding="utf-8")
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700


def test_the_quickstart_has_one_short_section():
    body = QUICK.read_text(encoding="utf-8")
    head = "### 换到 OAK / RealSense 以后：「主视角」小窗"
    assert body.count(head) == 1
    section = body.split(head, 1)[1].split("\n#", 1)[0].replace("\n", "")
    for needed in ("`oak1` / `oak2` / `rs`", "右栏最下面", "正在画的", "实例表选中的",
                   "任务卡当前行", "主视角 第 12 帧", "单击看整帧", "滚轮", "缺图时用前一帧",
                   NOT_DRAWN, "顶部「主视角」按钮", "记得"):
        assert needed in section, needed
    assert len(section) < 600            # U43's paragraph (这一步的变化) is in it since


def test_the_spec_has_the_paragraph_in_4_5_and_v1_23_on_top():
    body = SPEC.read_text(encoding="utf-8")
    section = body.split("### 4.5 界面布局", 1)[1].split("### 4.6", 1)[0]
    assert "**主视角（v1.23，U23）**" in section
    for needed in ("`oak1`、`oak2`、`rs`", "dupli", "零件约占一半", "帧长边的 10 %",
                   "正在画的 → 实例表选中的", "任务卡当前行", "并集外框", NOT_HERE, NOT_DRAWN,
                   NOT_ANNOTATED, "**存着的** `compiled_mask` 行", "不判断是否过期",
                   "编译一次（`TruthService.compile`，不写库）", "1024 px", "最近 20 步",
                   "离线缩略图（192 px", "什么都不读", "约 60 ms", "约 400 ms",
                   "`locator/open`", "工具面板在画布左边，高度不变", "没有快捷键"):
        assert needed in section, needed
    changes = body.split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # on top when it landed; U24's v1.24, U25's v1.25 and U26's v1.26 have gone
    # above it since
    # (and U27's v1.27, U28's v1.28, U29's v1.29, L6's v1.30 and U32's v1.31)
    assert entries[0].startswith(("- **v1.23", "- **v1.24", "- **v1.25", "- **v1.26",
                                  "- **v1.27",
                                  "- **v1.28", "- **v1.29", "- **v1.30", "- **v1.31",
                                  "- **v1.32", "- **v1.33", "- **v1.34", "- **v1.35",
                                  "- **v1.36",
                                  "- **v1.37", "- **v1.38", "- **v1.39", "- **v1.40",
                                  "- **v1.41", "- **v1.42"))
    [first] = [e for e in entries[:20] if e.startswith("- **v1.23")]
    assert first.startswith("- **v1.23（2026-10-23）**：**主视角**（U23，§4.5）")
    assert changes.index("- **v1.23") < changes.index("- **v1.21")
