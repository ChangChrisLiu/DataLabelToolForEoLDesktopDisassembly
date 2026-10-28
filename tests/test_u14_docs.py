"""Task U14: the guide, the quickstart and the spec say it in the app's own words."""
from __future__ import annotations

from pathlib import Path

from tda.core.captive_fix import CaptiveFix
from tda.ui import app_actions as A
from tda.ui.app_verb import captive_note
from tda.ui.panels.instances import VERB_STAYS_IN, VERB_UNSCREWED
from tda.ui.panels.taskcard import captive_hint

REPO = Path(__file__).resolve().parents[1]


def _flat(*parts: str) -> str:
    return " ".join((REPO.joinpath(*parts)).read_text(encoding="utf-8").split())


def _squash(text: str) -> str:
    return "".join(text.split())


def test_the_key_table_row_names_both_screw_buttons():
    label = A.action_named("correct_verb").label_zh
    assert f"螺丝：{VERB_UNSCREWED} / {VERB_STAYS_IN}" in label
    guide = _flat("docs", "annotation_guide.md")
    assert label in guide                                   # the generated key table


def test_the_quickstart_tells_d3_as_a_worked_example_in_the_past_tense():
    """D3 is fixed in the live database: the paragraph says what was done, and not to redo it."""
    quick = _squash(_flat("docs", "trial_quickstart.md"))
    fix = CaptiveFix(desktop=3, target="screw.cpu_cooler.01", step=2, new=False, at=10,
                     parent="cpu_cooler.fan.01", before={"captive": True},
                     after={"captive": False})
    note = captive_note("散热器螺丝 screw.cpu_cooler.01", {"fix": fix})
    assert _squash(note.split("（Ctrl+Z")[0]) in quick
    assert _squash(captive_hint()) in quick
    for words in ("D3（**已经改好了**）", "五颗风扇螺丝其实第2–6步就拧下拿走了",
                  "改之前任务卡在第7–15帧要画它们", f"当时在实例表点screw.cpu_cooler.01（按钮写"
                  f"「{VERB_UNSCREWED}」）", f"现在D3上这五颗的按钮写「{VERB_STAYS_IN}」——**别再按**",
                  "再按一次`Shift+T`会把它翻回去"):
        assert _squash(words) in quick, words


def test_the_spec_has_the_paragraph_and_the_entry():
    spec = _flat("docs", "superpowers", "specs", "2026-09-17-teardown-annotator-design.md")
    assert "**不脱落螺丝 `captive`（v1.17，U14）**" in spec
    assert "**v1.17（2026-10-16）**：**其实拧下来了**（U14，§3.2）" in spec
    changelog = spec.split("## 14. 修订记录", 1)[1]
    # at the top of §14 when it landed; U18's v1.18, U19's v1.19, U20's v1.20,
    # U21's v1.21, U22's v1.22, U23's v1.23, U24's v1.24, U25's v1.25, U26's
    # v1.26, U27's v1.27 and U28's v1.28 and U29's v1.29 and L6's v1.30 and U32's v1.31
    # have gone on top since
    assert changelog.lstrip().startswith(("- **v1.17", "- **v1.18", "- **v1.19",
                                          "- **v1.20", "- **v1.21", "- **v1.22",
                                          "- **v1.23", "- **v1.24", "- **v1.25",
                                          "- **v1.26", "- **v1.27",
                                          "- **v1.28", "- **v1.29", "- **v1.30",
                                          "- **v1.31", "- **v1.32", "- **v1.33",
                                          "- **v1.34", "- **v1.35", "- **v1.36",
                                          "- **v1.37"))
    assert changelog.index("- **v1.17") < changelog.index("- **v1.16")
