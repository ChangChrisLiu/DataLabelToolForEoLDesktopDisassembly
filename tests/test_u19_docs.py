"""Task U19: the docs say what the second SAM head changes, and the guide does not grow.

The keys do not change, so the annotator's guide gets nothing (its prose budget
is full); the quickstart gets one paragraph, the spec a paragraph in §5.1 and
v1.19 on top of §14, the README's layout table the new config.
"""
from __future__ import annotations

from pathlib import Path

from tda.ui.app_assist import HEAD_OFF_NOTE

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "annotation_guide.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
README = ROOT / "README.md"


def test_the_guide_gets_nothing():
    body = GUIDE.read_text(encoding="utf-8")
    assert "微调" not in body and "解码头" not in body
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700


def test_the_quickstart_has_one_paragraph():
    body = QUICK.read_text(encoding="utf-8")
    paras = [p for p in body.split("\n- ") if "SAM 学了你的画法" in p]
    assert len(paras) == 1
    para = paras[0]
    for needed in ("螺丝的掩码贴着螺丝头", "接口、卡扣只点一下", "带框的", "**大零件**",
                   "和内存条、硬盘卡扣（新的在这两样上只点一下反而差）",
                   "还是原来的 SAM，和以前一样", "纠错再点", "`C`", HEAD_OFF_NOTE):
        assert needed in para, needed
    assert len(para) < 450


def test_the_spec_has_the_paragraph_and_v1_19_on_top():
    body = SPEC.read_text(encoding="utf-8")
    section = body.split("### 5.1 交互式分割（MVP）", 1)[1].split("### 5.2", 1)[0]
    assert "**两个解码头（v1.19，U19）**" in section
    for needed in ("141 个张量", "`stock_point_classes`", "ram_module（−0.096 / −0.023）",
                   "drive_latch（−0.027 / −0.191）", "`conv_s0`/`conv_s1`", "逐位相同",
                   "`TDA_SAM_HEAD=off`"):
        assert needed in section, needed
    changes = body.split("## 14. 修订记录", 1)[1]
    first = next(line for line in changes.splitlines() if line.startswith("- "))
    assert first.startswith("- **v1.19（2026-10-17）**：**SAM 两个解码头**（U19")


def test_the_readme_names_the_config():
    row = next(line for line in README.read_text(encoding="utf-8").splitlines()
               if line.startswith("| `configs/` |"))
    assert "SAM decoder heads" in row
