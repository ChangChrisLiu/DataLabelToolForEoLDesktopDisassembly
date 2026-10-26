"""Task U26's documents: §8.1 lists the amodal fields, §9.1 the empty-mask rule,
v1.26 is on §14, and the quickstart says what the new ``check`` line means."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUICK = ROOT / "docs" / "trial_quickstart.md"
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def test_the_spec_lists_the_amodal_fields_and_the_check_rule():
    body = SPEC.read_text(encoding="utf-8")
    cv = body.split("### 8.1 CV 导出", 1)[1].split("### 8.2", 1)[0]
    for needed in ("**每条掩码标注**", "`amodal_segmentation`", "`amodal_bbox`", "`amodal_area`",
                   "`--roi-crop`", "`--no-amodal`", "`info.amodal`", "`iscrowd: 1`",
                   "`null`", "**文件里可见 ⊆ 完整形状**", "`info.amodal_union_rows`",
                   "**被上面标了的零件整个盖住的零件", "`--no-hidden`", "`info.hidden`",
                   # round 3: N1, N3a, N8
                   "只去掉 U26 新加的", "除 `info` 外与 U26 之前的文件逐字节相同",
                   "`amodal = 可见`", "`info.version` 升为 `\"2\"`"):
        assert needed in cv, needed
    checks = body.split("### 9.1 ", 1)[1].split("### 9.2", 1)[0]
    for needed in ("「n 个零件完全被挡住（正常）：m 个零件帧，不算问题」", "`occluded_full`",
                   "不影响退出码", "`empty_visible:<零件> (step N): …`", "`status` 不变",
                   "「这一帧没有它的形状（没画，或整个在画面外）」"):
        assert needed in checks, needed
    occluder = body.split("**遮挡原因 `occluder`**", 1)[1].split("\n", 1)[0]
    assert "v1.26 起每条掩码标注都带" in occluder


def test_v1_26_is_in_the_changelog_above_v1_24():
    changes = SPEC.read_text(encoding="utf-8").split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # on top when it landed; whatever lands later goes above it
    [entry] = [e for e in entries[:8] if e.startswith("- **v1.26")]
    assert entry.startswith("- **v1.26（2026-10-24）**：**正常的遮挡不再算问题；COCO 带完整形状**"
                            "（U26，§8.1、§9.1）")
    for needed in ("`tda.core.empty_visible`", "D3 232 → 0", "`--no-amodal`", "`--no-hidden`",
                   "`info.amodal_union_rows`", "1.68 MB → 2.76 MB", "16 497 → 18 980",
                   "`TruthService.refresh_range`", "「这一帧没有它的形状（没画，或整个在画面外）」",
                   "`info.version` 升为 `\"2\"`"):
        assert needed in entry, needed
    assert changes.index("- **v1.26") < changes.index("- **v1.24")


def test_the_quickstart_explains_the_check_line():
    section = QUICK.read_text(encoding="utf-8").split("## 5. 收工与导出", 1)[1].split("\n## ", 1)[0]
    for needed in ("「n 个零件完全被挡住（正常）」", "不算问题", "形状在这一帧是空的",
                   "「这一帧没有它的形状（没画，或整个在画面外）」", "`amodal_segmentation`",
                   "`--no-amodal`", "`--no-hidden`", "`iscrowd: 1`"):
        assert needed in section, needed
