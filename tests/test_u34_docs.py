"""Task U34: the spec says what 「这台没有这个零件」 is where the reverse walk is
specified (§4.2), v1.33 is on top of §14, the quickstart has the recipe and the
guide has its one line (inside the prose budget)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
GUIDE = ROOT / "docs" / "annotation_guide.md"


def _body() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_the_paragraph_is_in_4_2():
    walk = _body().split("### 4.2 倒序标注", 1)[1].split("### 4.3", 1)[0]
    head = "**这台没有这个零件**（v1.33，U34）"
    assert walk.index("**缺帧处理**") < walk.index(head)
    para = walk.split(head, 1)[1].split("\n", 1)[0]
    for needed in ("D34", "`optical_drive.01`", "`state_assert.assertable_states`",
                   "什么时候拆下由步骤表决定", "**单独的一个操作**", "不是状态断言",
                   "`absent_from_start`", "`tda/core/absent.py`", "step 0", "`auto = 0`",
                   "`evidence_view = s1:absent`", "`in_chassis → elsewhere`",
                   "`elsewhere` 而不是 `on_bench`", "`states.absent_keys`",
                   "**一个事件都不推导**", "空操作", "不报警", "没有它的行，也没有 ✂",
                   "`DesktopCtx.absent`", "**不连带**", "「一开始就不在（这台没有它）」",
                   "最早可标的一帧", "「这台没有这个零件…」", "「其实有…」", "`Ctrl+Z` / `Ctrl+Y`",
                   "`import-logs --force`", "`scripts/absent_from_start.py",
                   # round 2
                   "`T` 的循环不停在这一项", "标着「被线挡住」", "`absent.residues`",
                   "「合并到…」（含「其实是硬盘架…」）涉及「不在」的零件，任一方向都拒绝",
                   "`export_scanpack`", "UTF-8", "`psutil`", "弃置锁事后原样放回",
                   "`python -m tda.cli check --desktop N --views scan,oak1,oak2,rs`"):
        assert needed in para, needed
    # round 2 (ruling N2): T does not cycle onto it
    assert "`T` 循环到它时也是这一项" not in para


def test_v1_33_is_on_top_of_14():
    changes = _body().split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # U33's v1.32 lands beside it: the controller settles the order at the merge
    # U33 / U35 / U36 landed v1.32 / v1.34 / v1.35 the same day; the controller ordered them
    # (and U37's v1.36 has gone above them since)
    [entry] = [e for e in entries[:8] if e.startswith("- **v1.33")]
    assert entry.startswith("- **v1.33（2026-10-26）**：**这台没有这个零件**（U34，§4.2）")
    for needed in ("`absent_from_start`", "`assertable_states`", "`states.absent_keys`",
                   "`events_from_actions`", "「一开始就不在（这台没有它）」",
                   "「这台没有这个零件…」/「其实有…」", "`import-logs --force` 保留",
                   "`scripts/absent_from_start.py`", "`steps_merge.staged_edits`",
                   "D37、D38", "库结构不变（仍为 v5）", "**第二轮**",
                   "从 `T` 的循环里拿掉", "「被线挡住」", "框外零件改名", "`export_scanpack`",
                   "`psutil`"):
        assert needed in entry, needed
    assert "`T` 循环到）" not in entry
    assert changes.index("- **v1.33") < changes.index("- **v1.31")


def test_the_quickstart_has_the_recipe():
    quick = QUICK.read_text(encoding="utf-8")
    head = "### 这台机器根本没有这个零件（「一开始就不在」）"
    assert quick.index("### 画面里的状态和日志不一样（`T`）") < quick.index(head) \
        < quick.index("### 盖板/卡扣其实被拆走了（`Shift+T`）")
    recipe = quick.split(head, 1)[1].split("\n### ", 1)[0]
    for needed in ("D34", "`optical_drive.01`", "「一开始就不在（这台没有它）」",
                   "`T` 不会切到这一项",
                   "光驱 optical_drive.01：一开始就不在（这台没有它），每一帧都不再要它（Ctrl+Z 撤销）",
                   "**不跟着变**", "「这台没有这个零件…」", "「其实有…」", "`import-logs --force`",
                   "标过「被线挡住」会被拒绝"):
        assert needed in recipe, needed
    assert "或按 `T`" not in recipe          # round 2 (N2): the combo's, not T's


def test_the_guide_has_its_line_inside_the_budget():
    body = GUIDE.read_text(encoding="utf-8")
    line = next(x for x in body.splitlines() if x.startswith("**这台根本没有这个零件**"))
    assert "状态框选「一开始就不在」" in line and "按 `T`" not in line   # round 2 (N2)
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700


def test_the_status_line_the_quickstart_quotes_is_the_app_s():
    from tda.ui.app_state import absent_note

    class Patch:
        frames = 0

    quick = QUICK.read_text(encoding="utf-8")
    assert absent_note("光驱 optical_drive.01", Patch()) in quick
