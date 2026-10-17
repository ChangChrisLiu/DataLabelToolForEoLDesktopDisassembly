"""Task U18: the docs say how to make a 框外零件.

The guide's key table (generated) has the right-click row, and the 「添加零件」
paragraph names it in one clause -- the prose budget of
``tests/test_app_actions.py`` still holds. The quickstart has one short section
with the D5 example; the spec has §3.2c and v1.18 on top of §14.
"""
from __future__ import annotations

from pathlib import Path

from tda.ui import app_actions as A
from tda.ui import context_text as T

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "annotation_guide.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def test_the_key_table_has_the_right_click_row_and_the_window_its_slot():
    from tda.ui.app import MainWindow

    [entry] = A.MENU_ENTRIES
    assert entry.where == "右键画布 → 框外零件…" and T.MENU_TEXT in entry.where
    row = f"| `{entry.where}` | {entry.label_zh} |"
    assert row in A.shortcut_markdown()
    assert row in GUIDE.read_text(encoding="utf-8")         # regenerated, not copied
    assert entry.where in A.cheat_sheet_html()
    assert hasattr(MainWindow, entry.slot)
    # not a key: nothing in the index answers it
    assert all(action.name != entry.name for action in A.ACTIONS)


def test_the_add_parts_paragraph_names_it_in_one_clause():
    body = GUIDE.read_text(encoding="utf-8")
    para = next(p for p in body.split("\n\n") if "日志里没有的零件也要标" in p)
    assert "机器外的用画布右键「框外零件…」" in para
    assert para.count("框外零件") == 1
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700


def test_the_quickstart_has_one_short_section_with_the_example():
    body = QUICK.read_text(encoding="utf-8")
    head = "### 机器外面的东西（「框外零件」）"
    assert body.count(head) == 1
    section = body.split(head, 1)[1].split("\n#", 1)[0]
    for needed in ("D5 第 20 帧", "右键", "「框外零件…」", "cooler_bracket", "`Enter`",
                   "第 10 帧", "`I`", "「不在」", "`Ctrl+I`"):
        assert needed in section, needed
    assert len(section) < 900                                # short


def test_the_spec_has_section_3_2c_and_v1_18_on_top():
    body = SPEC.read_text(encoding="utf-8")
    assert "### 3.2c 框外零件（context object，v1.18）" in body
    assert body.index("### 3.2b") < body.index("### 3.2c") < body.index("### 3.3")
    changes = body.split("## 14. 修订记录", 1)[1]
    first = next(line for line in changes.splitlines() if line.startswith("- "))
    assert first.startswith("- **v1.18（2026-10-17）**：**框外零件**（U18")
