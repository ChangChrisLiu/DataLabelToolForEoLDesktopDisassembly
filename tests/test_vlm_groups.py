"""Groups of interchangeable parts and the marks that name them (redesign 2026-11-03, lane A)."""
from __future__ import annotations

import json
import re

import pytest

import vlm_scene as S
import w3b_scene as W
from tda.core.db import Db
from tda.core.export.coco import DesktopCtx
from tda.core.export.vlm import ALL_TASKS, export_vlm
from tda.core.export.vlm_groups import mark_box, part_groups
from tda.core.model import InstanceRec
from tda.core.taxonomy import load_taxonomy
from vlm_checker import Checker


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


# --------------------------------------------------------------------------- #
# A2: naming by mark, never by ordinal
# --------------------------------------------------------------------------- #
ORDINAL = re.compile(r"\b[A-Za-z]+ \d+\b")


def _export(db, tax, out, desktop=S.DESKTOP, view=S.VIEW, **kw):
    kw.setdefault("tasks", ALL_TASKS)
    export_vlm(db, tax, [desktop], view, str(out), **kw)
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]


def _of(records, task):
    return [r for r in records if r["prompt"]["task"] == task]


def test_v2_question_has_mark_and_no_ordinal(scene, tmp_path):
    db, tax = scene
    records = _of(_export(db, tax, tmp_path / "v.jsonl", balance=False), "V2")
    states = [r for r in records if r["label"]["answer_check"]["derive"] == "instance_state"]
    assert states
    for r in states:
        assert "outlined in red" in r["prompt"]["question"]
        assert not ORDINAL.search(r["prompt"]["question"])
        marks = r["prompt"]["marks"]
        assert marks and {m["image"] for m in marks} == {0} and {m["tag"] for m in marks} == {"A"}
        key = r["label"]["answer_check"]["instance"]
        assert marks[0]["bbox"] == r["label"]["evidence"]["bboxes"][key]
        assert r["prompt"]["terms"] == r["prompt"]["state_terms"] == [
            r["label"]["answer_check"]["class"]]


def test_v4_hidden_target_marked_with_amodal_box(scene, tmp_path):
    db, tax = scene
    records = _of(_export(db, tax, tmp_path / "v.jsonl"), "V4")
    hidden = [r for r in records if r["label"]["step"] == 4
              and r["label"]["answer_check"]["target"] == S.RAM]
    assert hidden, "the RAM module is `occluded_full` at step 4 and still asked"
    x0, y0, x1, y1 = S.RECTS[S.RAM]
    assert hidden[0]["prompt"]["marks"] == [
        {"image": 0, "bbox": [x0, y0, x1 - x0, y1 - y0], "tag": "A"}]
    assert "the RAM module outlined in red" in hidden[0]["prompt"]["question"]


def test_v1_v8_v14_have_no_marks(scene, tmp_path):
    db, tax = scene
    records = _export(db, tax, tmp_path / "v.jsonl")
    for task in ("V1", "V8", "V14"):
        assert _of(records, task), task
        for r in _of(records, task):
            assert "marks" not in r["prompt"]
            assert "outlined" not in r["prompt"]["question"]


# --------------------------------------------------------------------------- #
# A3: V8 / V14 / V2 on classes with several instances
# --------------------------------------------------------------------------- #
def test_v8_multi_instance_is_components(tmp_path):
    db = Db(str(tmp_path / "wide.sqlite"))
    tax = S.build_wide(db, desktops=(21,), confirm_step=1)
    records = _of(_export(db, tax, tmp_path / "v.jsonl", desktop=21), "V8")
    sets = [r for r in records if r["label"]["answer_check"]["type"] == "components"]
    assert sets
    for r in sets:
        check = r["label"]["answer_check"]
        comps = r["label"]["answer"]["components"]
        assert len(comps) >= 2 and check["exhaustive"] is True
        assert check["class_filter"] in ("screw.motherboard", "connector.front_panel",
                                         "ram_module", "ram_latch")
        assert "every" in r["prompt"]["question"] or "all" in r["prompt"]["question"] \
            or "Where are" in r["prompt"]["question"]
        assert not ORDINAL.search(r["prompt"]["question"])
    for r in records:
        if r["label"]["answer_check"]["type"] == "boxes":
            # a single box only for a part its class (or a relational phrase) names alone
            assert r["label"]["answer_check"]["expression"] in ("class", "fastens", "of")
    assert Checker(db, tax, 21).check_all(records) == len(records)
    db.close()


def test_v14_mixed_group_skipped(tmp_path):
    db = Db(str(tmp_path / "tiny.sqlite"))
    tax = S.build(db, tiny_screw=True)   # one PSU screw seen, its twin `too_small`
    out = tmp_path / "v.jsonl"
    summary = export_vlm(db, tax, [S.DESKTOP], S.VIEW, str(out), tasks=["V14"])
    records = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x]
    assert summary["skip_reasons"][str(S.DESKTOP)]["V14"]["mixed_group"] >= 1
    for r in records:
        if r["label"]["step"] >= 2:
            continue  # the seen one is out from step 2: its twin is then asked alone
        members = r["label"]["answer_check"].get("members") or [
            r["label"]["answer_check"]["instance"]]
        assert S.SCREW not in members and S.TINY_SCREW not in members
    db.close()


# --------------------------------------------------------------------------- #
# A4: V4 without empty negatives, blockers as groups, no captive tightness
# --------------------------------------------------------------------------- #
GID = f"screw.cpu_cooler@{W.COOLER}"


def test_v4_has_no_state_inapplicable(scene, tmp_path):
    db, tax = scene
    records = _of(_export(db, tax, tmp_path / "v.jsonl"), "V4")
    assert records
    assert "state_inapplicable" not in {r["label"]["truth_source"] for r in records}
    assert all(r["label"]["answer"]["blockers"] for r in records
               if not r["label"]["answer"]["feasible"])


def test_v4_fan_blockers_are_screw_group_and_connector(tmp_path):
    from tda.core.export.vlm_tasks import GRAPH_BLOCKED, V4_QUESTIONS, _v4_record

    db = Db(str(tmp_path / "w.sqlite"))
    tax = W.build(db)
    for rec in db.instances(W.DESKTOP).values():   # cooler screws whose tightness shows
        if rec.key in W.SCREWS:
            rec.attrs = dict(rec.attrs, captive=False)
            db.upsert_instance(rec)
    tc = S.task_ctx(db, tax, W.DESKTOP, W.VIEW)
    made = _v4_record(tc, tc.frames[2], "remove", W.COOLER, V4_QUESTIONS, GRAPH_BLOCKED)
    assert made is not None
    assert made["label"]["answer"]["blockers"] == sorted([GID, W.FAN])
    assert made["label"]["answer_check"]["blocker_members"][GID] == list(W.SCREWS)
    db.close()


def test_v4_captive_dependent_skipped(tmp_path):
    db = Db(str(tmp_path / "w.sqlite"))
    tax = W.build(db)
    out = tmp_path / "v.jsonl"
    summary = export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out), tasks=["V4"])
    records = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x]
    assert summary["skip_reasons"][str(W.DESKTOP)]["V4"]["captive_tightness"] >= 1
    for r in records:
        check = r["label"]["answer_check"]
        assert check["target"] not in (W.COOLER, GID, *W.SCREWS)
        assert not any(e["blocker"] in W.SCREWS for e in check["edges"])
    db.close()


@pytest.mark.parametrize("which", ["vlm", "w3b"])
def test_no_ordinal_in_any_question(tmp_path, which):
    db = Db(str(tmp_path / f"{which}.sqlite"))
    if which == "vlm":
        tax = S.build(db, tiny_screw=True)
        records = _export(db, tax, tmp_path / "v.jsonl")
    else:
        tax = W.build(db)
        records = _export(db, tax, tmp_path / "v.jsonl", desktop=W.DESKTOP, view=W.VIEW)
    assert len({r["prompt"]["task"] for r in records}) >= 8
    for r in records:
        if r["label"]["answer_check"].get("derive") == "screw_count":
            continue  # the one class-level exception (spec 6.5)
        texts = [r["prompt"]["question"],
                 *(o["target_label"] for o in r["prompt"].get("options") or [])]
        for text in texts:
            assert not ORDINAL.search(text), (r["label"]["readable_id"], text)
    db.close()
