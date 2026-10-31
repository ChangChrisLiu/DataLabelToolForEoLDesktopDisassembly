"""Task U40's documents: §2.2 states the steps no camera photographed right
after the scanner's ``skipped_steps``, D63 is among the known anomalies, v1.39
is near the top of §14, and the data audit carries the settled D63 line."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
AUDIT = ROOT / "docs" / "data_audit_2026-10-24.md"


def _spec() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_2_2_states_the_rule_next_to_skipped_steps():
    section = _spec().split("### 2.2 统一索引", 1)[1].split("### 2.3", 1)[0]
    head = "**全部相机都没拍的步**（v1.39，U40）"
    skipped = section.index("`scan.<d>.skipped_steps: [k]`")
    assert skipped < section.index(head) < section.index("扫描仪的规则被**断言**")
    # nothing else sits between the skipped_steps sentence and the new one
    assert section.index("k 算作声明缺失。" + head) > skipped
    para = section.split(head, 1)[1].split("扫描仪的规则被**断言**", 1)[0]
    for needed in ("`unphotographed.<d>.steps: [k, …]`", "表格的行 = 拍摄 + 声明的步",
                   "`tda.core.index.capture_steps`", "OAK1/OAK2/RealSense", "两者取并集",
                   "`fix:scan[d]` 行写明各自来源", "四个视角都是缺帧",
                   "`(unphotographed.<d>.steps)`", "`fix:unphotographed[d]`", "`MISMATCH`",
                   "索引仍为 v3", "`folder_steps`", "`scan-align`"):
        assert needed in para, needed


def test_2_2_lists_d63():
    section = _spec().split("### 2.2 统一索引", 1)[1].split("### 2.3", 1)[0]
    line = next(ln for ln in section.splitlines() if ln.startswith("  - D63："))
    for needed in ("`unphotographed.63.steps: [14, 15, 16]`", "拍摄号就是表格的序号",
                   "拍摄 26 CPU 已拆 = 序号 26", "拍摄 28 显卡已拆 = 序号 28", "第 34–37 步",
                   "序号重复 11–13 的第 14–16 步", "拍摄 1→2（蓝色板卡支架）和 3→4（导风罩）",
                   "没有单独照片的动作"):
        assert needed in line, needed
    assert "不确定" not in line
    assert "D63 11–13 重号——已定为三步没拍，`unphotographed`，v1.39" in section


def test_v1_39_is_near_the_top_of_14():
    changes = _spec().split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # U38 / U39 land v1.37 / v1.38 alongside: the controller orders them at merge
    [entry] = [e for e in entries[:8] if e.startswith("- **v1.39")]
    assert entry.startswith("- **v1.39（2026-10-27）**：**全部相机都没拍的步**（U40，§2.2）")
    for needed in ("`unphotographed.<d>.steps: [k, …]`",
                   "`tda.core.index.capture_rows` / `capture_steps`",
                   "`verify_unphotographed`", "`verify_declared_missing`",
                   "`34 +3 unphotographed`", "**D63 的声明与任务原文不同**",
                   "[31, 32, 33]", "第 14–16 步", "拍摄号就是表格的序号",
                   "没有单独照片的动作", "（控制方与审查员各自看图核实）", "第 4 个位姿段",
                   "库结构不变（仍为 v6）"):
        assert needed in entry, needed
    assert "不确定" not in entry and "交用户裁定" not in entry
    assert changes.index("- **v1.39") < changes.index("- **v1.36")


def test_the_audit_says_d63_is_settled():
    text = AUDIT.read_text(encoding="utf-8")
    d63 = text.split("- D63: the sheet has 37 rows for 34 Camera_1 captures", 1)[1]
    settled = d63.split("\n", 2)[1]
    assert settled.startswith("  - **Settled 2026-10-27 (U40):**")
    assert "`unphotographed.63.steps: [14, 15, 16]`" in settled
    assert "the capture number is the sheet's Sequence Number" in settled
    assert "cannot be seen" not in settled
