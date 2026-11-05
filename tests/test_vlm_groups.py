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


def _keys(record: dict) -> dict:
    """The answer with its keys (``answer_check.answer_keys``; ``label.answer`` names parts
    plainly since the owner's 2026-11-04 round)."""
    label = record["label"]
    return (label.get("answer_check") or {}).get("answer_keys") or label["answer"]


def _set_keys(record: dict, answer: dict) -> None:
    label = record["label"]
    if (label.get("answer_check") or {}).get("answer_keys"):
        label["answer_check"]["answer_keys"] = answer
    else:
        label["answer"] = answer



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


def _plug(key, host, cable, kind=None):
    return InstanceRec(key=key, desktop=1, cls="connector",
                       attrs={"kind": kind} if kind else {}, socket_host=host, cable=cable)


def test_groups_of_one_class_are_told_apart_by_their_anchors():
    """Final review C2: every kind-less connector group read "connectors"; where one class
    has several groups on a desktop the anchor's words are added -- the cable, then the
    host -- and the labels carry no ordinal."""
    recs = [InstanceRec(key="motherboard.01", desktop=1, cls="motherboard"),
            InstanceRec(key="optical_drive.01", desktop=1, cls="optical_drive"),
            InstanceRec(key="storage_drive.ssd.01", desktop=1, cls="storage_drive",
                        attrs={"kind": "ssd"}),
            _plug("connector.01", "motherboard.01", "cable:psu"),
            _plug("connector.02", "motherboard.01", "cable:psu"),
            _plug("connector.03", "storage_drive.ssd.01", "cable:storage_drive"),
            _plug("connector.04", "storage_drive.ssd.01", "cable:storage_drive"),
            _plug("connector.05", "motherboard.01", "cable:optical_drive"),
            _plug("connector.06", "motherboard.01", "cable:optical_drive"),
            _plug("connector.07", "optical_drive.01", "cable:optical_drive"),
            _plug("connector.08", "optical_drive.01", "cable:optical_drive"),
            # a kind of its own: one group, the plain label
            _plug("connector.fan.01", "motherboard.01", None, kind="fan"),
            _plug("connector.fan.02", "motherboard.01", None, kind="fan")]
    groups = part_groups(_ctx(recs))
    labels = {g.gid: g.label for g in groups.values()}
    assert labels == {
        "connector@motherboard.01|cable:psu": "PSU cable connectors",
        "connector@storage_drive.ssd.01|cable:storage_drive": "storage drive cable connectors",
        "connector@motherboard.01|cable:optical_drive":
            "optical drive cable connectors on the motherboard",
        "connector@optical_drive.01|cable:optical_drive":
            "optical drive cable connectors on the optical drive",
        "connector.fan@motherboard.01|-": "fan connectors",
    }
    assert not any(ORDINAL.search(label) for label in labels.values())


def test_anchor_words_that_read_the_same_keep_the_plain_label():
    fan, heat = "cpu_cooler.fan.01", "cpu_cooler.heatsink.01"
    recs = [InstanceRec(key=fan, desktop=1, cls="cpu_cooler", attrs={"kind": "fan"}),
            InstanceRec(key=heat, desktop=1, cls="cpu_cooler", attrs={"kind": "heatsink"}),
            *(_screw(f"screw.cpu_cooler.{i:02d}", fan) for i in (1, 2)),
            *(_screw(f"screw.cpu_cooler.{i:02d}", heat) for i in (3, 4))]
    for m in (1, 2):
        recs.append(InstanceRec(key=f"ram_module.{m:02d}", desktop=1, cls="ram_module"))
        for j in (1, 2):
            recs.append(InstanceRec(key=f"ram_latch.{m}{j}", desktop=1, cls="ram_latch",
                                    attrs={"of": f"ram_module.{m:02d}"}))
    recs += [InstanceRec(key=f"ram_latch.9{j}", desktop=1, cls="ram_latch") for j in (1, 2)]
    labels = {g.gid: g.label for g in part_groups(_ctx(recs)).values()}
    assert labels["screw.cpu_cooler@cpu_cooler.fan.01"] == "CPU cooler screws of the fan CPU cooler"
    heat_label = labels["screw.cpu_cooler@cpu_cooler.heatsink.01"]
    assert heat_label == "CPU cooler screws of the heatsink CPU cooler"
    # two modules' latches read the same however they are qualified: the plain label
    assert labels["ram_latch@ram_module.01"] == labels["ram_latch@ram_module.02"] == "RAM latches"
    assert labels["ram_latch@-"] == "RAM latches without a RAM module"


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


def test_v1_v8_have_no_marks(scene, tmp_path):
    db, tax = scene
    records = _export(db, tax, tmp_path / "v.jsonl")
    for task in ("V1", "V8"):
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
        comps = _keys(r)["components"]
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


def test_v2_cannot_tell_asks_no_group_half_seen(tmp_path, monkeypatch):
    """V14 merged into V2 (owner 2026-11-04): a hidden-part question asks a group only when
    every member still in is hidden -- one PSU screw seen, its twin ``too_small``: no
    cannot_tell question about the pair."""
    from tda.core.export import vlm_tasks as T

    monkeypatch.setattr(T, "V2_HIDDEN_EVERY", 1)
    db = Db(str(tmp_path / "tiny.sqlite"))
    tax = S.build(db, tiny_screw=True)
    out = tmp_path / "v.jsonl"
    export_vlm(db, tax, [S.DESKTOP], S.VIEW, str(out), tasks=["V2"], balance=False)
    records = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x]
    for r in records:
        if r["label"]["step"] >= 2 or not r["label"]["answer_check"].get("hidden"):
            continue  # the seen one is out from step 2: its twin is then asked alone
        members = r["label"]["answer_check"].get("members") or [
            r["label"]["answer_check"]["instance"]]
        assert S.SCREW not in members
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
    assert all(_keys(r)["blockers"] for r in records
               if not _keys(r)["feasible"])


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
    assert _keys(made)["blockers"] == sorted([GID, W.FAN])
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


# --------------------------------------------------------------------------- #
# A5: action tasks on groups, from every camera of a moment
# --------------------------------------------------------------------------- #
ALL_VIEWS = ("scan", "oak1", "oak2", "rs")


@pytest.fixture
def four(tmp_path):
    db = Db(str(tmp_path / "four.sqlite"))
    tax = W.build(db, views=ALL_VIEWS)
    yield db, tax, S.task_ctx(db, tax, W.DESKTOP, W.VIEW, side=ALL_VIEWS[1:])
    db.close()


def _gen(tc, task):
    from tda.core.export.vlm_tasks import GENERATORS

    return [r for step in tc.steps for r in GENERATORS[task](tc, step)]


def test_action_record_images_marked_scan_alone(four):
    """An unmarked action record shows every camera, the scanner first; a marked one (owner
    2026-11-03: V4, V6 multiple choice, V7, V16) the scanner image its marks are drawn on,
    and its question speaks of one picture."""
    db, tax, tc = four
    # V6's multiple choice names its options in words (owner 2026-11-04): no mark, every camera
    for task in ("V4", "V7", "V16"):
        records = [r for r in _gen(tc, task) if task != "V7" or r["prompt"].get("marks")]
        assert records or task == "V7", task
        for r in records:
            step = r["label"]["step"]
            assert r["prompt"]["images"] == [tc.frames[step].image_id], task
            assert r["prompt"]["image_legend"] == [{"view": "scan", "moment": "now"}]
            assert r["label"]["views"] == ["scan"] and r["label"]["image_steps"] == [step]
            assert r["prompt"]["marks"] and {m["image"] for m in r["prompt"]["marks"]} == {0}
            assert not re.search(r"\b(pictures|images|views|cameras)\b",
                                 r["prompt"]["question"]), (task, r["prompt"]["question"])
    for task in ("V6", "V10"):
        records = _gen(tc, task)
        assert records, task
        for r in records:
            step = r["label"]["step"]
            assert not r["prompt"].get("marks"), task
            assert len(r["prompt"]["images"]) == 4, task
            assert r["prompt"]["images"][0] == tc.frames[step].image_id
            assert r["prompt"]["image_legend"] == [{"view": v, "moment": "now"}
                                                   for v in ALL_VIEWS]
            assert r["label"]["views"] == list(ALL_VIEWS)
            assert r["label"]["image_steps"] == [step] * 4
    for task in ("V3", "V12"):
        # the neighbouring pair (the spans of 2-3 steps show the earlier moment instead)
        for r in [r for r in _gen(tc, task) if r["label"].get("v3_form") != "span"
                  and (r["label"].get("v12_span") or 1) == 1]:
            step = r["label"]["step"]
            assert len(r["prompt"]["images"]) == 8
            assert [e["moment"] for e in r["prompt"]["image_legend"]] == \
                ["before"] * 4 + ["after"] * 4
            assert r["label"]["image_steps"] == [step - 1] * 4 + [step] * 4
    records = [r for t in ("V3", "V4", "V6", "V7", "V10", "V12", "V16") for r in _gen(tc, t)]
    assert Checker(db, tax, W.DESKTOP).check_all(records) == len(records)


def test_missing_side_view_shrinks_legend(four):
    db, tax, tc = four
    step = min(r["label"]["step"] for r in _gen(tc, "V10"))
    tc.side_frames["oak2"].pop(step)            # that camera has no picture of the step
    records = [r for r in _gen(tc, "V10") if r["label"]["step"] == step]
    assert records
    for r in records:
        assert r["label"]["views"] == ["scan", "oak1", "rs"]
        assert [e["view"] for e in r["prompt"]["image_legend"]] == ["scan", "oak1", "rs"]
        assert len(r["prompt"]["images"]) == 3
    # a marked record shows the scanner alone whichever side camera is missing
    for r in (r for r in _gen(tc, "V4") if r["label"]["step"] == step):
        assert r["label"]["views"] == ["scan"] and len(r["prompt"]["images"]) == 1


def test_v3_tiny_target_skipped(tmp_path, monkeypatch):
    import tda.core.export.vlm_tasks as T

    monkeypatch.setattr(T, "V3_MIN_PX", 10 ** 9)
    db = Db(str(tmp_path / "w.sqlite"))
    tax = W.build(db)
    out = tmp_path / "v.jsonl"
    summary = export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out), tasks=["V3"])
    records = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x]
    records = [r for r in records if r["label"].get("v3_form") != "span"]
    assert summary["skip_reasons"][str(W.DESKTOP)]["V3"]["invisible_action"] >= 1
    targets = {_keys(r)["target_instance"] for r in records}
    assert W.FAN not in targets                                       # a plug
    assert {W.OPTICAL, W.COOLER, W.SHROUD} <= targets                 # BIG_PARTS stay
    db.close()


def test_v3_accepts_only_the_part_acted_on(tmp_path):
    """Owner 2026-11-03: removing the cooler is answered "the CPU cooler", never by a part
    attached to it (its captive screws ride out with it): ``accept_targets`` is the target
    alone and the grader gives such an answer 0."""
    from tda.core.export.vlm_check import World, grade

    db = Db(str(tmp_path / "w.sqlite"))
    tax = W.build(db)
    out = tmp_path / "v.jsonl"
    records = [r for r in _of(_export(db, tax, out, desktop=W.DESKTOP, view=W.VIEW), "V3")
               if r["label"].get("v3_form") != "span"]
    db.close()
    cooler = next(r for r in records if r["label"]["step"] == 9)
    assert _keys(cooler)["target_instance"] == W.COOLER
    assert cooler["label"]["answer_check"]["accept_targets"] == [W.COOLER]
    # every V3 record accepts its target and nothing else
    for r in records:
        assert r["label"]["answer_check"]["accept_targets"] == [
            _keys(r)["target_instance"]], r["label"]["readable_id"]
    oracle = out.with_name(out.stem + ".oracle.jsonl")
    row = next(json.loads(x) for x in oracle.read_text(encoding="utf-8").splitlines()
               if json.loads(x).get("type") == "desktop")
    world = World.from_oracle(row)
    # the scene's screws are attached to the cooler (parent + attached)
    assert all(world.instances[s]["parent"] == W.COOLER and world.instances[s]["attached"]
               for s in W.SCREWS)

    def reply(target):
        return json.dumps({"verb": "remove", "target": target, "tool": None})

    for name in ("the CPU cooler", W.COOLER):
        assert grade(cooler, reply(name), world, tax).score == 1.0, name
    for name in ("the CPU cooler screws", "a CPU cooler screw", W.SCREWS[0]):
        g = grade(cooler, reply(name), world, tax)
        assert g.score == 0.0 and not g.metrics["target_instance"], name
    # an older record listing the attached parts does not bring the credit back
    old = json.loads(json.dumps(cooler))
    old["label"]["answer_check"]["accept_targets"] = [W.COOLER, *W.SCREWS]
    assert grade(old, reply(W.SCREWS[0]), world, tax).score == 0.0
    assert grade(old, reply("the CPU cooler"), world, tax).score == 1.0


def test_v7_plan_uses_screw_group(tmp_path):
    db = Db(str(tmp_path / "w.sqlite"))
    tax = W.build(db)
    records = _of(_export(db, tax, tmp_path / "v.jsonl", desktop=W.DESKTOP, view=W.VIEW),
                  "V7")
    by_step = {r["label"]["step"]: r for r in records}
    for step in (2, 5, 7, 8):
        plan = [(a["verb"], a["target"]) for a in _keys(by_step[step])["plan"]]
        assert plan.count(("unscrew", GID)) == 1, (step, plan)
        assert not set(W.SCREWS) & {t for _, t in plan}
    # every cooler screw already loose by the log, the cooler still in: still one
    # required group action (spec 6.6), and first
    assert _keys(by_step[7])["plan"][0] == {"verb": "unscrew", "target": GID}
    assert "unscrew" not in [a["verb"] for a in _keys(by_step[9])["plan"]] \
        if 9 in by_step else True
    assert Checker(db, tax, W.DESKTOP).check_all(records) == len(records)
    db.close()


def test_v10_partial_group_is_started(tmp_path):
    db = Db(str(tmp_path / "wide.sqlite"))
    tax = S.build_wide(db, desktops=(21,))
    records = _of(_export(db, tax, tmp_path / "v.jsonl", desktop=21), "V10")
    by_step = {r["label"]["step"]: _keys(r) for r in records}
    screws = {"verb": "unscrew", "target": "screw.motherboard@motherboard.01"}
    assert screws in by_step[2]["started"] and screws not in by_step[2]["done"]
    last = max(by_step)
    assert screws in by_step[last]["done"] and screws not in by_step[last]["started"]
    assert all(not str(d["target"]).startswith("screw.motherboard.")
               for d in by_step[2]["done"])
    assert Checker(db, tax, 21).check_all(records) == len(records)
    db.close()


def test_v10_captive_screws_stay_to_do_until_the_cooler_is_out(tmp_path):
    db = Db(str(tmp_path / "w.sqlite"))
    tax = W.build(db)
    records = _of(_export(db, tax, tmp_path / "v.jsonl", desktop=W.DESKTOP, view=W.VIEW),
                  "V10")
    by_step = {r["label"]["step"]: _keys(r) for r in records}
    group = {"verb": "unscrew", "target": GID}
    assert group not in by_step[7]["done"] and group not in by_step[7]["started"]
    assert group in by_step[9]["done"]


# --------------------------------------------------------------------------- #
# A6: the oracle carries the groups
# --------------------------------------------------------------------------- #
def test_oracle_carries_groups(tmp_path):
    from tda.core.export.vlm import oracle_path

    db = Db(str(tmp_path / "w.sqlite"))
    tax = W.build(db)
    out = tmp_path / "v.jsonl"
    export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out))
    world = next(json.loads(x) for x in oracle_path(str(out)).read_text(
        encoding="utf-8").splitlines() if json.loads(x)["type"] == "desktop")
    assert world["groups"] == {GID: {"members": list(W.SCREWS),
                                     "match_class": "screw.cpu_cooler",
                                     "anchor": W.COOLER, "label": "CPU cooler screws"}}
    for key, inst in world["instances"].items():
        assert inst["group"] == (GID if key in W.SCREWS else None), key
    db.close()


def test_v7_requires_the_captive_screws_as_one_group_while_the_cooler_is_in(tmp_path):
    """Fix round 1: required_actions does not shrink screw by screw as the log loosens them."""
    db = Db(str(tmp_path / "w.sqlite"))
    tax = W.build(db)
    records = _of(_export(db, tax, tmp_path / "v.jsonl", desktop=W.DESKTOP, view=W.VIEW),
                  "V7")
    # the CPU's free plan (V7 asks every major part since 2026-11-04)
    by_step = {r["label"]["step"]: r for r in records
               if r["label"]["answer_check"]["goal"] == W.CPU and r["label"]["v7_form"] == "free"}
    for step in range(2, 9):                    # the cooler comes out at step 9
        required = by_step[step]["label"]["answer_check"]["required_actions"]
        targets = [r["target"] for r in required]
        assert targets.count(GID) == 1, (step, targets)
        assert not set(W.SCREWS) & set(targets), (step, targets)
        item = next(r for r in required if r["target"] == GID)
        assert item["members"] == list(W.SCREWS) and item["states"] == ["loosened", "removed"]
    # identical from step 4 to 8, whatever the log loosened in between
    shapes = {json.dumps(next(r for r in by_step[s]["label"]["answer_check"]["required_actions"]
                              if r["target"] == GID), sort_keys=True) for s in range(4, 9)}
    assert len(shapes) == 1
    assert Checker(db, tax, W.DESKTOP).check_all(records) == len(records)
    db.close()


BANNED_WORDING = ("two images", "single frame", "this frame", "first image", "two frames",
                  "these two", "two photos")


def test_action_task_questions_fit_a_set_of_views(four):
    """Fix round 1: an action-task record shows every camera of a moment (or two moments)."""
    db, tax, tc = four
    from tda.core.export import vlm_tasks as T

    templates = [*T.V3_QUESTIONS, *T.V4_QUESTIONS, *T.V4_FAILED_QUESTIONS, *T.V6_QUESTIONS,
                 *T.V6_MC_QUESTIONS, *T.V7_QUESTIONS, *T.V10_QUESTIONS, *T.V12_QUESTIONS,
                 *T.V16_QUESTIONS]
    for text in templates:
        for banned in BANNED_WORDING:
            assert banned not in text.lower(), (banned, text)
    for task in ("V3", "V4", "V6", "V7", "V10", "V12", "V16"):
        for r in _gen(tc, task):
            low = r["prompt"]["question"].lower()
            assert not any(b in low for b in BANNED_WORDING), (task, low)


def test_v8_set_question_reads_with_a_plural(tmp_path):
    from tda.core.export.vlm_tasks import V8_SET_QUESTIONS

    # the phrase is a plural: never "every {phrase}" ("every visible motherboard screws")
    assert not any(re.search(r"every\s+(visible\s+)?\{phrase\}", t) for t in V8_SET_QUESTIONS)
    db = Db(str(tmp_path / "wide.sqlite"))
    tax = S.build_wide(db, desktops=(21,), confirm_step=1)
    records = _of(_export(db, tax, tmp_path / "v.jsonl", desktop=21), "V8")
    sets = [r for r in records if r["label"]["answer_check"]["type"] == "components"]
    assert sets
    for r in sets:
        question = r["prompt"]["question"]
        assert not re.search(r"\bevery (visible )?[A-Za-z ]*(screws|connectors|modules|latches)\b",
                             question), question
        assert re.search(r"(screws|connectors|modules|latches)\b", question), question
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


def test_every_mark_tag_has_a_name_on_the_label_side(scene, tmp_path):
    """Final review I2: ``label.mark_names`` says what each tag outlines -- a class phrase
    (no ordinal, no key) and the instance keys -- so the text-only references can keep
    "part A" resolvable; the prompt carries nothing new."""
    db, tax = scene
    records = _export(db, tax, tmp_path / "v.jsonl", balance=False)
    marked = [r for r in records if r["prompt"].get("marks")]
    assert marked
    for r in marked:
        names = r["label"]["mark_names"]
        assert set(names) == {m["tag"] for m in r["prompt"]["marks"]}, r["label"]["readable_id"]
        for tag, entry in names.items():
            assert entry["name"] and not ORDINAL.search(entry["name"]) and "." not in entry["name"]
            assert entry["parts"] and all(p in tc_keys(db) for p in entry["parts"])
        assert "mark_names" not in r["prompt"]
    assert not [r for r in records if "mark_names" in r["label"] and not r["prompt"].get("marks")]


def tc_keys(db) -> set:
    return set(db.instances(S.DESKTOP))


def test_an_anchor_with_a_vowel_takes_an():
    recs = [InstanceRec(key="expansion_card.01", desktop=1, cls="expansion_card")]
    recs += [InstanceRec(key=f"card_latch.0{j}", desktop=1, cls="card_latch",
                         attrs={"of": "expansion_card.01"}) for j in (1, 2)]
    recs += [InstanceRec(key=f"card_latch.1{j}", desktop=1, cls="card_latch") for j in (1, 2)]
    labels = {g.gid: g.label for g in part_groups(_ctx(recs)).values()}
    assert labels["card_latch@-"] == "card latches without an expansion card"
    assert labels["card_latch@expansion_card.01"] == "card latches of the expansion card"
