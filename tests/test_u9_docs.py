"""Task U9, E: the guide, the quickstart and the spec say it in the app's own words."""
from __future__ import annotations

from pathlib import Path

from tda.core.verb_fix import VerbFix
from tda.ui import app_actions as A
from tda.ui.app_verb import verb_note
from tda.ui.panels.instances import VERB_ONLY_OPENED, VERB_TAKEN_OUT
from tda.ui.panels.taskcard import taken_out_hint

REPO = Path(__file__).resolve().parents[1]


def _flat(*parts: str) -> str:
    return " ".join((REPO.joinpath(*parts)).read_text(encoding="utf-8").split())


def _squash(text: str) -> str:
    return "".join(text.split())


def test_the_guide_names_the_key_and_both_buttons():
    guide = _flat("docs", "annotation_guide.md")
    assert "`Shift+T`" in guide
    assert VERB_TAKEN_OUT in guide and VERB_ONLY_OPENED in guide
    assert A.action_named("correct_verb").label_zh in guide        # the key table


def test_the_quickstart_quotes_the_d2_status_line_and_the_card_hint():
    quick = _squash(_flat("docs", "trial_quickstart.md"))
    fix = VerbFix(desktop=2, target="cover.01", step=38, idx=0, old="open", new="remove",
                  at=48)
    assert _squash(verb_note("主板螺丝盖板 cover.01", {"fix": fix, "lost": []})) in quick
    assert _squash(taken_out_hint()) in quick
    assert _squash("这些帧上它的形状不再使用（k 帧）") in quick


def test_the_spec_has_the_entry():
    spec = _flat("docs", "superpowers", "specs", "2026-09-17-teardown-annotator-design.md")
    assert "**v1.9（2026-10-14）**：**打开还是拆走**（U9）" in spec
