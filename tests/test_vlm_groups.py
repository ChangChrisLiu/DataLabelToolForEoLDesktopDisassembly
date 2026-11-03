"""Groups of interchangeable parts and the marks that name them (redesign 2026-11-03, lane A)."""
from __future__ import annotations

import pytest

import vlm_scene as S
from tda.core.db import Db
from tda.core.export.coco import DesktopCtx
from tda.core.export.vlm_groups import mark_box, part_groups
from tda.core.model import InstanceRec
from tda.core.taxonomy import load_taxonomy


def _ctx(instances) -> DesktopCtx:
    return DesktopCtx(desktop=1, tax=load_taxonomy(), instances={r.key: r for r in instances},
                      events=[], actions=[], steps={})


def _screw(key, fastens, role="cpu_cooler", captive=True):
    return InstanceRec(key=key, desktop=1, cls="screw",
                       attrs={"role": role, "head": "PH2", "captive": captive}, fastens=fastens)


def test_cooler_screws_form_one_group():
    fan = "cpu_cooler.fan.01"
    screws = [f"screw.cpu_cooler.{i:02d}" for i in range(1, 5)]
    ctx = _ctx([InstanceRec(key=fan, desktop=1, cls="cpu_cooler", attrs={"kind": "fan"}),
                *(_screw(s, fan) for s in screws)])
    groups = part_groups(ctx)
    assert set(groups) == set(screws)
    group = groups[screws[0]]
    assert {g.gid for g in groups.values()} == {"screw.cpu_cooler@cpu_cooler.fan.01"}
    assert group.members == tuple(screws)
    assert group.label == "CPU cooler screws"
    assert (group.match_class, group.anchor) == ("screw.cpu_cooler", fan)


def test_latches_group_per_module():
    recs = [InstanceRec(key="motherboard.01", desktop=1, cls="motherboard")]
    for m in (1, 2):
        recs.append(InstanceRec(key=f"ram_module.{m:02d}", desktop=1, cls="ram_module"))
        for j in (1, 2):
            recs.append(InstanceRec(key=f"ram_latch.{m}{j}", desktop=1, cls="ram_latch",
                                    attrs={"of": f"ram_module.{m:02d}"}))
    groups = part_groups(_ctx(recs))
    assert groups["ram_latch.11"].gid == "ram_latch@ram_module.01"
    assert groups["ram_latch.22"].gid == "ram_latch@ram_module.02"
    assert groups["ram_latch.11"].members == ("ram_latch.11", "ram_latch.12")
    assert groups["ram_latch.11"].label == "RAM latches"
    assert groups["ram_module.01"].gid == "ram_module@-"
    assert groups["ram_module.01"].label == "RAM modules"
    assert "motherboard.01" not in groups


def test_singleton_is_no_group():
    recs = [InstanceRec(key="psu.01", desktop=1, cls="psu"),
            _screw("screw.psu.01", "psu.01", role="psu", captive=False),
            _screw("screw.motherboard.01", "motherboard.01", role="motherboard", captive=False),
            _screw("screw.motherboard.02", "motherboard.01", role="motherboard", captive=False),
            InstanceRec(key="motherboard.01", desktop=1, cls="motherboard")]
    groups = part_groups(_ctx(recs))
    assert "screw.psu.01" not in groups and "psu.01" not in groups
    assert groups["screw.motherboard.01"].gid == "screw.motherboard@motherboard.01"


@pytest.fixture
def scene(tmp_db_path: str):
    db = Db(tmp_db_path)
    tax = S.build(db)
    yield db, tax
    db.close()


def test_mark_box_uses_amodal_when_hidden(scene):
    db, tax = scene
    tc = S.task_ctx(db, tax)
    frame = tc.frames[4]          # the RAM module is forced `occluded_full` there
    assert frame.rows[S.RAM]["visibility"] == "occluded_full"
    assert S.RAM not in frame.pointable
    x0, y0, x1, y1 = S.RECTS[S.RAM]
    assert mark_box(tc, frame, S.RAM) == [x0, y0, x1 - x0, y1 - y0]
    # a visible part keeps its visible-mask box
    assert mark_box(tc, frame, S.PSU) == list(frame.pointable[S.PSU][1])


def test_mark_box_none_when_removed(scene):
    db, tax = scene
    tc = S.task_ctx(db, tax)
    frame = tc.frames[6]          # the PSU came out at step 5: a bench box, no mark
    assert S.PSU in frame.pointable
    assert mark_box(tc, frame, S.PSU) is None
    assert mark_box(tc, frame, S.SCREW) is None   # unscrewed (removed) at step 2
    assert mark_box(tc, frame, S.RAM) is not None
