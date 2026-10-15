"""Task U12: the guide, the quick start and the spec say how to merge, in the app's words."""
from __future__ import annotations

from pathlib import Path

import pytest

from merge_scenes import DESKTOP, GONE, KEEP, build
from tda.core.db import Db
from tda.core.merge_instances import MergeError, plan_merge
from tda.ui import steps_merge as M

REPO = Path(__file__).resolve().parents[1]


def _flat(*parts: str) -> str:
    return " ".join(REPO.joinpath(*parts).read_text(encoding="utf-8").split())


def test_the_guide_has_the_one_line():
    guide = _flat("docs", "annotation_guide.md")
    assert "两个实例其实是同一个零件？" in guide
    assert "右键「合并到…」" in guide


def test_the_quickstart_walks_through_d2(tmp_path):
    quick = _flat("docs", "trial_quickstart.md")
    assert "两个实例其实是同一个零件？（「合并到…」）" in quick
    assert "「Screw Cover」 → `cover.01`" in quick and "「Front Cover」→ `cover.02`" in quick
    assert M.MENU_TEXT in quick
    tight = "".join(quick.split())
    # round 2: the confirmation's first line with the D2 edits staged, word for word
    assert "".join(M.APPLY_AND_MERGE.format(what="步骤/动作").split()) in tight
    assert "先别点`Apply`" in tight
    assert "形状 7 个（scan 锚在第 37–42、45 帧）" in quick
    assert "已确认的行 8 行（scan 第 38–45 帧）" in quick
    assert "动作 1（第 46 步改成 cover.01）" in quick
    # the refusal quoted is the one the code writes (the miniature scene's frames are 3-4)
    db = Db(str(tmp_path / "t.sqlite"))
    scene = build(db, s1=False)
    with pytest.raises(MergeError) as caught:
        plan_merge(db, scene.tax, DESKTOP, GONE, KEEP)
    db.close()
    said = str(caught.value).replace("第 3–4 帧", "第 38–45 帧")
    # compared without whitespace: the quote is wrapped inside a list item
    assert "".join(said.split()) in tight


def test_the_spec_has_the_rules_and_the_changelog():
    spec = _flat("docs", "superpowers", "specs", "2026-09-17-teardown-annotator-design.md")
    assert "**合并两个实例（v1.11，U12）**" in spec
    assert spec.count("「Apply 并合并」") == 2
    assert "`merge_instances` op" in spec and "`replay_merges` op" in spec
    assert "**v1.11（2026-10-14）**：**合并实例**（U12）" in spec
