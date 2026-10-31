"""Task U20: the docs say what a drive cage's own latch is, and the guide does not grow.

No key changes, so the annotator's guide gets nothing (its prose budget is full);
the quickstart gets one paragraph with the D5/D6 example, the spec a paragraph
in §3.2 and v1.20 on top of §14.
"""
from __future__ import annotations

from pathlib import Path

from tda.ui.panels.taskcard import cage_latch_hint

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "annotation_guide.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def _flat(path: Path) -> str:
    """The text with the hard line breaks of a paragraph taken out."""
    return path.read_text(encoding="utf-8").replace("\n  ", "")


def test_the_guide_gets_nothing():
    body = GUIDE.read_text(encoding="utf-8")
    assert "机架卡扣" not in body and "drive_cage_latch" not in body
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700


def test_the_quickstart_has_one_paragraph_with_the_d5_d6_example():
    paras = [p for p in _flat(QUICK).split("\n- ") if p.startswith("**硬盘架自带的卡扣**")]
    assert len(paras) == 1
    para = paras[0].split("\n", 1)[0]
    for needed in ("D5/D6", "HDD shield locker", "第 13 帧", "`drive_latch.02`",
                   "「跟 drive_cage.01 一起装回来的」", cage_latch_hint(), "第 12 帧它是 ✂",
                   "`Ctrl+T`", "「添加零件」", "Steps → Instances 右键删掉", "重导不会回来",
                   "回到待复核",
                   # review round: Shift+T latches the cage at once (M2), the dialog
                   # proposes open from the frame (N1), delete rather than merge (M1)
                   "按 `Shift+T` 说它其实拆下了，机架立刻自动得到隐含的 `drive_latch.02`",
                   "`Ctrl+Z` 一起撤销", "到第 11 帧按 `Ctrl+T`（对话框已选「打开」）",
                   "不要合并"):
        assert needed in para, needed
    assert len(para) < 620


def test_the_spec_has_the_paragraph_in_3_2_and_v1_20_on_top():
    body = SPEC.read_text(encoding="utf-8")
    section = body.split("### 3.2 实例身份", 1)[1].split("### 3.2a", 1)[0]
    assert "**机架自带卡扣（v1.20，U20）**" in section
    for needed in ("有机架就有它的卡扣", "implied.py::cage_latches", "「cage latch adopted」",
                   "「cage latch kept」", "「cage latch kept deleted」",
                   "implied_rule: drive_cage_latch",
                   "rule:drive_cage_latch", "c−1", "`delete_implied`", "手工状态优先",
                   "「机架卡扣：默认这里是打开的；更早就开着的话到那一帧 Ctrl+T」",
                   "COCO `implied: true`", "规则只在导入时生效"):
        assert needed in section, needed
    changes = body.split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # on top when it landed; U21's v1.21, U22's v1.22, U23's v1.23, U24's
    # v1.24, U25's v1.25, U26's v1.26, U27's v1.27 and U28's v1.28 and U29's v1.29 and L6's v1.30 have gone above it since
    # (and U32's v1.31)
    assert entries[0].startswith(("- **v1.20", "- **v1.21", "- **v1.22", "- **v1.23",
                                  "- **v1.24", "- **v1.25", "- **v1.26", "- **v1.27",
                                  "- **v1.28", "- **v1.29", "- **v1.30", "- **v1.31",
                                  "- **v1.32", "- **v1.33", "- **v1.34", "- **v1.35",
                                  "- **v1.36",
                                  "- **v1.37", "- **v1.38", "- **v1.39", "- **v1.40",
                                  "- **v1.41", "- **v1.42", "- **v1.43", "- **v1.44",
                                  "- **v1.45", "- **v1.48"))
    assert any(e.startswith("- **v1.20（2026-10-22）**：**机架自带卡扣**（U20，§3.2）")
               for e in entries[:27])
    assert changes.index("- **v1.20") < changes.index("- **v1.19")
