"""Task U25 docs: spec §2.2/§2.3 say a dupli row is not a step, v1.25 is on top of
§14, and the data audit is committed."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def _body() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_spec_2_2_states_the_collapse_and_the_scanner_rule():
    body = _body()
    section = body.split("### 2.2 统一索引", 1)[1].split("### 2.3", 1)[0]
    for needed in ("**`dupli` 行不是步**（v1.25，U25）", "`r − #(dupli 行 ≤ r)`",
                   "`aux.extra_shots`", "`oak.<d>.dupli_replaces", "`scan.<d>.duplicate_folders`",
                   "`n − #(重复文件夹 ≤ n)`", "`MISMATCH`", "索引文件 v3",
                   "**照片换了步，工作跟着照片走**（v1.25）", "`PhotoMoveConflict`",
                   "`FolderSteps`"):
        assert needed in section, needed
    import_section = body.split("### 2.3 步骤表导入", 1)[1].split("### 2.4", 1)[0]
    assert "`dupli` 行除外（v1.25）" in import_section


def test_v1_25_is_on_top_of_14():
    changes = _body().split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # U26 merged first: its v1.26 is on top, U25's v1.25 right under it; U27's
    # v1.27 and U28's v1.28 and U29's v1.29 and L6's v1.30 have gone above both since
    # (and U32's v1.31)
    assert entries[0].startswith(("- **v1.25", "- **v1.26", "- **v1.27",
                                  "- **v1.28", "- **v1.29", "- **v1.30", "- **v1.31",
                                  "- **v1.32", "- **v1.33", "- **v1.34", "- **v1.35",
                                  "- **v1.36",
                                  "- **v1.37", "- **v1.38", "- **v1.39", "- **v1.40"))
    [first] = [e for e in entries[:17] if e.startswith("- **v1.25")]
    assert first.startswith("- **v1.25（2026-10-24）**：**`dupli` 行不是步；扫描仪按自己的编号对齐**"
                            "（U25，§2.2、§2.3）")
    for needed in ("`scan-align", "`duplicate_folders`", "dupli_replaces",
                   "`docs/data_audit_2026-10-24.md`", "D18 `[42]`"):
        assert needed in first, needed
    assert changes.index("- **v1.25") < changes.index("- **v1.24")


def test_the_data_audit_is_committed():
    audit = REPO / "docs" / "data_audit_2026-10-24.md"
    text = audit.read_text(encoding="utf-8")
    assert text.index("## 1. Per desktop") < text.index("## 2. Desktops that need a human look") \
        < text.index("## 3. Method")
    for d in range(1, 67):
        assert f"\n| {d} |" in text, d


def test_spec_states_the_round_2_rules():
    body = _body()
    section = body.split("### 2.2 统一索引", 1)[1].split("### 2.3", 1)[0]
    for needed in ("`scan.<d>.skipped_steps: [k]`", "`oak.<d>.cam1_order: folder`",
                   "`oak.<d>.gallery_captures: [文件夹]`", "`1000 + 文件夹号`",
                   "手写的状态事件", "`op_log`", "已经没有了", "`desktop.meta.index_dupli_rows`",
                   "**`import-logs` 只在库里的帧按这张表的收拢编号装过时才写**",
                   "`index_v2_20261024.json`", "`extra_shot_drafts_skipped`"):
        assert needed in section, needed
    first = next(line for line in body.split("## 14. 修订记录", 1)[1].splitlines()
                 if line.startswith("- **v1.25"))
    assert "**第二轮**" in first and "D41 `[32]`" in first and "D61 `[26]`" in first


def test_spec_states_the_round_3_rules():
    body = _body()
    section = body.split("### 2.2 统一索引", 1)[1].split("### 2.3", 1)[0]
    assert "只有 `review_status`" in section and "photo column(s) left behind" in section
    assert "`python scripts/precheck_index_moves.py`" in section
    entry = next(line for line in body.split("## 14. 修订记录", 1)[1].splitlines()
                 if line.startswith("- **v1.25"))
    assert "**第三轮**" in entry
