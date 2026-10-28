"""Task U27's documents: the spec's paragraph and v1.27, the quickstart's sentence and recipe.

The guide is untouched (its prose stays within 7700 characters). The spec has
the relation fixes in §3.2 with the D23 example and v1.27 on top of §14; the
quickstart says where Steps edits are described that they survive a forced
re-import, and has the D23 recipe.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "annotation_guide.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def test_the_guide_stays_within_its_budget():
    body = GUIDE.read_text(encoding="utf-8")
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700
    assert "relation_fix" not in body and "adopt-relations" not in body


def test_the_spec_has_the_paragraph_in_3_2_and_v1_27_on_top():
    body = SPEC.read_text(encoding="utf-8")
    section = body.split("### 3.2 实例身份", 1)[1].split("### 3.2a", 1)[0]
    [para] = [line for line in section.splitlines()
              if line.startswith("- **关系字段在 Steps 里改，重导保留（v1.27，U27）**")]
    for needed in ("`cable_clip.01`", "`cover.01`", "`parent = cover.01`", "勾 `attached`",
                   "`cable_clip` 只能这样变成 `removed`", "`relation_fix`",
                   "`{desktop, key, raw_name, cls, field, old, new, by, at}`",
                   "改回导入时的值也记", "`replay_relations`", "relation fixes kept",
                   "relation fix not matched", "按 raw name 找", "`parent` 不成环",
                   "被合并掉的零件上的 fix 不套用", "`python -m tda.cli adopt-relations --desktop N`",
                   "`--dry-run`", "U9/U12/U14/U21 的重放不受影响",
                   # round 2
                   "`Captive` 格也写 `captive_fix` op", "captive 回放也按 key + raw name 找螺丝",
                   "「captive differs」", "「not applied (no parent)」", "「rebuilt as <新键>」",
                   "每次导入都会再报一遍", "写入前先备份",
                   "D22–D26、D29、D32、D33（都是 HP）", "D63 是 Dell Precision",
                   # round 3
                   "只取 op id 最大的一条", "relation fixes superseded: n",
                   "captive fixes superseded: n", "记进 `replay_captive` 行",
                   "「rebuilt as」每次导入都说", "在表里勾 `Captive` 和 `Shift+T` 一样",
                   "只在有东西要记时才备份", "captive fixes (Shift+T / Steps)",
                   "用 `Shift+T` 把 `cover.02`（「Heatsink cover」，日志写的是 `open`）改成拆下"):
        assert needed in para, needed
    assert "D63…" not in body
    changes = body.split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # on top when it landed; U28's v1.28 and U29's v1.29 and L6's v1.30 and U32's v1.31
    # have gone above it since
    assert entries[0].startswith(("- **v1.27", "- **v1.28", "- **v1.29", "- **v1.30",
                                  "- **v1.31", "- **v1.32", "- **v1.33", "- **v1.34",
                                  "- **v1.35", "- **v1.36",
                                  "- **v1.37", "- **v1.38", "- **v1.39"))
    [entry] = [e for e in entries[:13] if e.startswith("- **v1.27")]
    assert entry.startswith("- **v1.27（2026-10-24）**：**关系字段重导保留**（U27，§3.2）")
    for needed in ("`relation_fix`", "`replay_relations`", "`adopt-relations",
                   "`tda/core/relation_replay.py`", "手设的螺丝头", "无迁移",
                   "评审轮（round 2）", "（D22、D24–D26、D29、D32、D33）", "captive differs",
                   "第三轮（round 3）", "superseded"):
        assert needed in entry, needed
    assert changes.index("- **v1.27") < changes.index("- **v1.26")


def test_the_quickstart_says_steps_edits_survive_and_has_the_d23_recipe():
    body = QUICK.read_text(encoding="utf-8")
    steps = body.split("## 1. 先到 Steps 模式核对这台机器", 1)[1].split("\n## 2. ", 1)[0]
    flat = steps.replace("\n   ", "")
    assert ("实例表里改的关系字段（`Parent`、`Attached`、`Mounted on`、`Fastens`、`Socket host`、"
            "`Cable` 等）每次 `Apply` 都记下来，`import-logs --force` 重新导入时保留。"
            "`Captive` 格和 `Shift+T` 一样保留；`Head` 格不保留。") in flat
    recipe = steps.split("### 零件其实长在另一个零件上（实例表 `Parent`）", 1)[1]
    flat = recipe.replace("\n", "")
    for needed in ("`cable_clip.01`", "`cover.01`", "Steps → `Instances` → `cable_clip.01` 那一行",
                   "`Parent` 填 `cover.01`、勾 `Attached` → `Apply`",
                   "卡扣第 1 帧关着、第 2 帧打开、第 3 帧起随导风罩拆下", "重新导入也保留",
                   "（D22、D24–D26、D29、D32、D33，表格里写着「case fan clip」的就是这几台）",
                   "D63 是 Dell Precision，它的卡扣在散热器罩上",
                   "用 `Shift+T` 把 `cover.02`（「Heatsink cover」，日志写的是 `open`）改成拆下"):
        assert needed in flat, needed
    assert "D63…" not in recipe
    assert len(recipe) < 800
