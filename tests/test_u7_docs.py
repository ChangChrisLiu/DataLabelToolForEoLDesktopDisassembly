"""Task U7, F: both documents say where conflicts come from and how to settle them at once."""
from __future__ import annotations

from pathlib import Path

from tda.ui import app_actions as A
from tda.ui import guide as G
from tda.ui.panels.add_parts import DEMOTION

REPO = Path(__file__).resolve().parents[1]


def _flat(name: str) -> str:
    return " ".join((REPO / "docs" / name).read_text(encoding="utf-8").split())


def test_both_documents_have_the_section():
    for name in ("annotation_guide.md", "trial_quickstart.md"):
        text = _flat(name)
        assert "冲突从哪来、怎么一次解决" in text, name
        assert "添加零件" in text and "需复核" in text, name
        assert "Shift+N" in text, name
        assert "一次画完再提交" in text, name
        # round 2: a new part's first shape over confirmed frames (the motherboard case)
        assert "新零件" in text and "第一笔" in text, name


def test_the_quickstart_says_what_the_pending_rechecks_mean():
    quick = _flat("trial_quickstart.md")
    assert "后台复检中 N 帧" in quick and "再按一次 `Shift+N`" in quick
    assert "打开视角时也会把所有 确认过的帧在后台再核对一遍" in quick or \
        "打开视角时也会把所有确认过的帧在后台再核对一遍" in quick


def test_the_quickstart_quotes_what_the_app_says():
    quick = _flat("trial_quickstart.md")
    sentence = DEMOTION.format(who="这 4 个零件", n=25, steps="17–41")
    assert " ".join(sentence.split()) in quick
    # the status line's clause at the start of a re-trace (app_guide.retrace_note)
    assert "重画会影响 25 个已确认帧（第 17–41 帧），提交后用 Shift+N 一次接受" in quick
    assert G.RETRACE_NOTE.format(n=25).split("，")[0] in quick
    assert "ram_latch.01 ×25（帧 17–41）" in quick
    assert "已接受 k 个，仍开放 m 个" in quick
    assert A.short_parts(A.action_named("review_accept_all_new"))[0] in quick
