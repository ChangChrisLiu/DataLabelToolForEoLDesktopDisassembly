"""Task U33: the spec says it where the states (§6.2) and the verbs (§6.3) are,
and v1.32 is near the top of §14."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def _body() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_the_state_is_in_6_2():
    states = _body().split("### 6.2 状态集与默认追踪策略", 1)[1].split("### 6.3", 1)[0]
    assert "| 虚拟节点 `cable:*` | routed / released / removed（无掩码；removed 见下，v1.32） |" \
        in states
    head = "**松线整根拿走：虚拟节点的 `removed`（v1.32，U33）**"
    assert states.index(head) < states.index("**帧级可见性 `visibility`**")
    para = states.split(head, 1)[1].split("\n", 1)[0]
    for needed in ("「Remove 1 isolated cable from motherboard」", "`cable_clip` + `release`",
                   "`cable_clip.01`", "第 42 步", "`missing_shape:cable_clip.01`", "42 行",
                   "routed / released / removed", "`remove` 作用于它", "`cable:isolated.NN`",
                   "没有实例行", "任务卡、引导、`cli check`", "COCO 导出里没有它",
                   "V3 / V10 / V16",
                   # round 2 (N6): what the VLM export loses on D36's frame 1, and why
                   "V4 ×2", "V5、V6", "326 → 322", "`legal_actions` 对虚拟节点只列 `release`",
                   "已经进入快照"):
        assert needed in para, needed


def test_the_virtual_node_row_in_3_1_has_removed():
    """Round 2 (N6): §3.1's entity table names the third state too."""
    entities = _body().split("### 3.1 实体", 1)[1].split("### 3.2", 1)[0]
    [row] = [line for line in entities.splitlines() if line.startswith("| `VirtualNode` |")]
    assert "状态 routed / released / removed" in row and "v1.32" in row


def test_the_verb_and_the_rule_are_in_6_3():
    verbs = _body().split("### 6.3 动作动词", 1)[1].split("### 6.4", 1)[0]
    assert ("| `remove` | 部件、cover、connector（随线缆整体取出时）、虚拟线缆节点（松线整根拿走，"
            "v1.32） | → removed（附着子零件连带 removed） |") in verbs
    head = "**日志里「拿走一根线」的行（v1.32，U33）**"
    para = verbs.split(head, 1)[1].split("\n", 1)[0]
    for needed in ("`cable_removed`", "`cable_routing` 之前", "remove / take / pull",
                   "clip / locker / holder / tie", "cover / shroud",
                   "「Remove cable from heatsink cover」", "connector / plug",
                   "`cable:<owner>`", "「独立」", "两根线是两个节点", "`is_cable_event`",
                   "`loose_cable_key`", "像翻转机箱", "`release` / `remove`",
                   "53 组不变", "D22–D26、D29、D32、D33",
                   "D28、D30、D31", "66 张表 2 814 行", "35 行", "只有 D36 第 2 步变了",
                   "`tests/test_u33_mapping.py`", "`import-logs --force`",
                   "`cable_event_clips`", "`dropped_instance` op", "`events`",
                   "1 hand-set state event(s) went with it", "but is kept",
                   "别的消失的零件的手设事件照旧是保留理由", "7 行自动编译缓存",
                   "其余表、其余机器逐行不变", "再跑一次什么都不变",
                   # round 2 (N6): the sheets' own wording of the locker rows
                   "「remove cable from cable locker 1」…「4」（D27、D28）",
                   "「Remove cable from cable locker」（D50–D59）",
                   "「Remove cable from two cable locker」（D60）",
                   "「Remove cables from cable locker」（D65）",
                   # round 2 (N2): what keeps it, and advice that S1 can follow
                   "不只是关键帧", "帧覆盖（`` ` `` / `` Ctrl+` `` 的「被线挡住」",
                   "已确认的行", "开着的冲突", "层级", "人定的约束边",
                   "undo it (the keyframe / frame override / confirmed row / conflict / "
                   "relation that holds it), then run import-logs --force again",
                   "S1 会拒绝"):
        assert needed in para, needed
    assert "cable locker N" not in para
    assert "delete the instance in S1" not in para


def test_v1_32_is_near_the_top_of_14():
    changes = _body().split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # U34-U36 landed v1.33-v1.35 the same day; the controller ordered them at the merges
    # (and U37's v1.36 and U38's v1.37 have gone above them since)
    [entry] = [e for e in entries[:6] if e.startswith("- **v1.32")]
    assert entry.startswith("- **v1.32（2026-10-26）**：**日志里拿走一根松线不是卡扣**"
                            "（U33，§6.2、§6.3）")
    for needed in ("`cable_clip.01` + `release`", "42 行 `missing_shape:cable_clip.01`",
                   "`configs/taxonomy.yaml`", "`cable_removed`", "`cable:isolated.NN`",
                   "`tda/core/logs.py`", "53 组", "35 行只有 D36 第 2 步变了",
                   "D63「Remove cable from heatsink cover」照旧", "`import-logs --force`",
                   "`dropped_instance` op", "`tda/pipeline_logs.py`", "`cable_event_clips`",
                   "库结构不变（仍为 v5）",
                   # round 2 (N2 / N6)
                   "D22–D33、D50–D60、D65", "帧覆盖（`` ` `` / `` Ctrl+` `` 的「被线挡住」等）",
                   "已确认的行、开着的冲突、层级、人定的约束边",
                   "then run import-logs --force again", "V4 ×2、V5、V6"):
        assert needed in entry, needed
    assert "delete the instance in S1" not in entry
    assert changes.index("- **v1.32") < changes.index("- **v1.31")
