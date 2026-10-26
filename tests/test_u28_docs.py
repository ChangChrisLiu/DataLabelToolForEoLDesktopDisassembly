"""Task U28: the docs say what 「拆前 / 拆后」 does; the guide's prose does not grow.

The guide is left alone (its prose is at the 7700 limit and the pair explains
itself: the captions, the log's line, the toggle's tooltip); the quickstart's
「主视角」 section gets two sentences, the spec a paragraph in §4.5 after the
U23 one, and v1.28 on top of §14.
"""
from __future__ import annotations

from pathlib import Path

from tda.ui.locator_pair import (
    DIFF_GUESS,
    NO_AFTER,
    NO_CHANGE,
    NO_DIFF,
    NO_DIFF_ROI,
    POSE_BREAK,
)

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "annotation_guide.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def test_the_guide_prose_stays_within_its_limit():
    body = GUIDE.read_text(encoding="utf-8")
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700


def test_the_quickstart_says_it_in_two_sentences_of_the_dock_section():
    body = QUICK.read_text(encoding="utf-8")
    head = "### 换到 OAK / RealSense 以后：「主视角」小窗"
    section = body.split(head, 1)[1].split("\n#", 1)[0].replace("\n", "")
    for needed in ("第一次标某个零件时", "扫描仪视角也一样", "「拆前 / 拆后」", "被拆走（或第一次变状态）",
                   "前后两步的扫描仪画面", "裁到同一处", "「第 8 帧：Heatsink · remove」",
                   "「拆前/拆后」按钮切换", "扫描仪视角上小窗收着时按顶部「主视角」",
                   "灰色虚线框只是猜的位置", "别的候选"):
        assert needed in section, needed
    assert "随时" not in section.split("第一次标某个零件时", 1)[1]     # round 2, N10
    assert len(section) < 420


def test_the_spec_has_the_paragraph_after_u23_and_v1_28_on_top():
    body = SPEC.read_text(encoding="utf-8")
    section = body.split("### 4.5 界面布局", 1)[1].split("### 4.6", 1)[0]
    assert section.index("**主视角（v1.23，U23）**") < section.index("**拆前 / 拆后（v1.28，U28）**")
    para = section.split("**拆前 / 拆后（v1.28，U28）**", 1)[1]
    for needed in ("**离开的那一步 r**", "「拆前 第 r−1 帧」/「拆后 第 r 帧」", "**r 取编译后的状态时间线**",
                   "级联已折进去", "`removed` 或 `displaced`", "第一次状态变化", NO_CHANGE,
                   "**两帧裁到同一个窗口**", "r−1 帧上的外框", "最大的几个连通区", "长边 512 px",
                   "ROI 外扩 5 %", "2 % 的边框", POSE_BREAK, NO_DIFF, NO_DIFF_ROI, "0.02 %",
                   "模糊和开闭运算之前数", "放大 1.6 倍",
                   "帧长边的 1/6", "`crop_rect`", "整帧小图", "「第 8 帧：Heatsink · remove」",
                   "「随 散热片 一起拆走」", "(a)", "**当前视角**", "`shape_keyframe`",
                   "切分副本也算形状", "单帧像素", "**任何视角都一样，扫描仪视角也是**", "(b)",
                   "「拆前/拆后」按钮", "换了零件就忘", "不加快捷键", "(c)", "照旧是主视角画面",
                   "小窗照旧隐藏", "(d)", "`locator/open` 的意思不变", "什么都不读",
                   "一次索引查询", "60–75 ms", "0.2 ms", "位姿段及其 ROI",
                   # round 2
                   "**拆后那一步缺图或是 dupli 步时用 r 所在位姿段里更晚的最近一帧**",
                   "「拆后 第 r′ 帧（第 r 帧扫描仪没有图）」", "绝不用更早的一帧", "「（没有图）」",
                   NO_AFTER.format(step="r"), "**差出来的窗口只是猜**", "灰色虚线",
                   DIFF_GUESS, "「候选 1/3」", "44 %（255/578）", "61 %（350/578）",
                   "34 % 到 42 %", "76 %", "其余机器 69 %",
                   "换视角也跟着这个零件", "按顶部「主视角」就是要这个零件的这一对",
                   "（机器、两帧、ROI、扫描仪输入摘要）",
                   # round 3
                   "**消失的检测只作额外候选**", "`configs/detector.yaml` 的 `classes`",
                   "最多 2 个", "差的第一处之后、第二三处之前", "最多四个小图", "画点线",
                   "「检测到消失的 <类名>」", "「检测到一个 <类名> 消失了」",
                   "检测器辅助**作为排序依据**评估过、没有采用", "回到它时重新问",
                   "存了新的扫描仪 ROI",
                   "「候选 n 处」", "检测从不是第一个", "磁盘缓存（只读，不写）", "后台线程",
                   "`DetectionEngine`", "就丢掉", "小窗隐藏时不算", "也不提示",
                   "54 % 到 91 %（374/412）", "61 % 到 92 %（531/578）", "多变化步不变"):
        assert needed in para, needed
    assert "随时" not in para                                          # round 2, N10
    changes = body.split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # on top when it landed; U29's v1.29, L6's v1.30, U32's v1.31 and U33's v1.32 have
    # gone above it since
    assert entries[0].startswith(("- **v1.28", "- **v1.29", "- **v1.30", "- **v1.31",
                                  "- **v1.32"))
    [first] = [e for e in entries[:5] if e.startswith("- **v1.28")]
    assert first.startswith("- **v1.28（2026-10-24）**：**拆前 / 拆后**（U28，§4.5）")
    for needed in ("第一次标", "`removed` / `displaced`", "级联", "1.6 倍", "1/6", "随 … 一起拆走",
                   "扫描仪视角", "「拆前/拆后」按钮", "无快捷键", "`locator/open` 不变",
                   "`pairs` / `diffs`", "第三轮", "消失检测作候选小图", "从不当第一个",
                   "54 % 到 91 %"):
        assert needed in first, needed
    assert changes.index("- **v1.28") < changes.index("- **v1.24")
