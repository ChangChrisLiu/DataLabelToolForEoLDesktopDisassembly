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


def test_the_quickstart_quotes_the_d3_status_line_and_the_card_hint():
    quick = _squash(_flat("docs", "trial_quickstart.md"))
    fix = CaptiveFix(desktop=3, target="screw.cpu_cooler.01", step=2, new=False, at=10,
                     parent="cpu_cooler.fan.01", before={"captive": True},
                     after={"captive": False})
    note = captive_note("散热器螺丝 screw.cpu_cooler.01", {"fix": fix})
    assert _squash(note) in quick
    assert _squash(captive_hint()) in quick
    for words in ("五颗风扇螺丝其实第2–6步就拧下拿走了", "第7–15帧", f"「{VERB_UNSCREWED}」",
                  f"「{VERB_STAYS_IN}」"):
        assert _squash(words) in quick


def test_the_spec_has_the_paragraph_and_the_entry():
    spec = _flat("docs", "superpowers", "specs", "2026-09-17-teardown-annotator-design.md")
    assert "**不脱落螺丝 `captive`（v1.17，U14）**" in spec
    assert "**v1.17（2026-10-16）**：**其实拧下来了**（U14，§3.2）" in spec
    changelog = spec.split("## 14. 修订记录", 1)[1]
    assert changelog.lstrip().startswith("- **v1.17")              # at the top of §14
