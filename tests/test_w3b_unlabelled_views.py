"""Views without confirmed rows, screw presence and the captive-screw rules (W3b).

* An OAK / RealSense view is exported from the state machine alone: no row of
  it is read, its records say ``view_confirmed: false``, V3 / V12 ask about
  large parts only, V6 / V10 / V16 are asked as anywhere else.
* A screw's state is its presence: tightness was never annotated.
* Unscrewing a captive screw is no visible change: V3 drops it, V12 asks the
  pair as "no change" (``captive_screw_loosened``), V2 / V14 ask nothing of it.
* V5 is out of the default set; ``--tasks V5`` still writes it.
* V2 is thinned by its class prior; the per-desktop skip reasons are in the
  summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import w3b_scene as W
from tda.core.db import Db
from tda.core.export.coco import load_ctx
from tda.core.export.vlm import export_vlm, oracle_path
from tda.core.export.vlm_tasks import BIG_PARTS, TASKS, balance_v2
from vlm_checker import Checker, scan_prompt_leaks


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


@pytest.fixture
def scene(tmp_path: Path):
    db = Db(str(tmp_path / "views.sqlite"))
    tax = W.build(db)
    yield db, tax, tmp_path
    db.close()


def _of(records, task):
    return [r for r in records if r["prompt"]["task"] == task]


def test_an_unlabelled_view_reads_nothing_of_its_own_annotation(scene):
    db, tax, tmp = scene
    out = tmp / "oak1.jsonl"
    summary = export_vlm(db, tax, [W.DESKTOP], W.OTHER, str(out))
    records = _read(out)
    tasks = {r["prompt"]["task"] for r in records}
    assert {"V6", "V10", "V16"} <= tasks
    assert not tasks & {"V1", "V2", "V8", "V14"}
    assert summary["unlabelled_view_desktops"] == [W.DESKTOP]
    for r in records:
        assert r["label"]["view_confirmed"] is False
        assert r["label"]["evidence"]["bboxes"] == {}
        for step in r["label"]["rationale"]["steps"]:
            assert step["op"] != "observe"  # nothing seen, all carried over
    for line in oracle_path(str(out)).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "frame":
            assert row["confirmed"] is False
            assert not any("visibility" in v for v in row["state"].values())
    assert not scan_prompt_leaks(records)
    assert Checker(db, tax, W.DESKTOP).check_all(records) == len(records)


def test_v3_and_v12_on_an_unlabelled_view_are_about_large_parts(scene):
    db, tax, tmp = scene
    out = tmp / "oak1.jsonl"
    export_vlm(db, tax, [W.DESKTOP], W.OTHER, str(out))
    ctx = load_ctx(db, tax, W.DESKTOP, W.OTHER)
    records = _read(out)
    v3 = _of(records, "V3")
    assert v3 and all(ctx.cls_of(r["label"]["answer"]["target_instance"]) in BIG_PARTS
                      for r in v3)
    # the shroud (a cover), the optical drive, the cooler -- not the fan lead,
    # not the screws, and not the CPU: step 11 is the last frame, but its pair
    # (10, 11) is asked, so the CPU is there too
    assert {r["label"]["step"] for r in v3} >= {2, 8, 9}
    assert not {3, 4, 5, 6, 7, 10} & {r["label"]["step"] for r in v3}
    for r in _of(records, "V12"):
        if r["label"]["answer"]["changed"]:
            assert r["label"]["big_change"]
            assert r["label"]["answer_check"]["fields"] == ["changed"]
        else:
            assert r["label"]["reason"] in ("dupli", "captive_screw_loosened")


def test_v6_v10_v16_take_the_image_from_any_view(scene):
    db, tax, tmp = scene
    a, b = tmp / "scan.jsonl", tmp / "oak1.jsonl"
    export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(a))
    export_vlm(db, tax, [W.DESKTOP], W.OTHER, str(b))
    def keyed(path, task):
        return {r["label"]["readable_id"].replace(f"-{r['label']['view']}-", "-"):
                r["label"]["answer"] for r in _of(_read(path), task)}

    for task in ("V6", "V10", "V16"):
        here, there = keyed(a, task), keyed(b, task)
        common = set(here) & set(there)
        assert common
        assert all(here[s] == there[s] for s in common), task


def test_captive_screws_rules(scene):
    db, tax, tmp = scene
    out = tmp / "scan.jsonl"
    # unthinned: the one board screw is present on every frame, so the class
    # prior would take all of its questions away
    summary = export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out), balance=False)
    records = _read(out)
    # V3: steps 4-7 only loosen captive screws -- nothing a pair of pictures shows
    assert not {4, 5, 6, 7} & {r["label"]["step"] for r in _of(records, "V3")}
    assert summary["skip_reasons"][str(W.DESKTOP)]["V3"]["captive_screw_loosened"] == 4
    # V12: those pairs are "no change", a kind of its own, never a "yes"
    captive = [r for r in _of(records, "V12")
               if r["label"].get("negative") == "captive_screw_loosened"]
    assert {r["label"]["step"] for r in captive} == {4, 5, 6, 7}
    for r in captive:
        assert r["label"]["answer"]["changed"] is False
        assert r["label"]["change_visible"] is False
        assert r["label"]["answer_check"]["reason"] == "captive_screw_loosened"
        assert {e["target"] for e in r["label"]["answer_check"]["hidden_events"]} <= set(W.SCREWS)
    assert summary["v12_by_kind"]["captive_screw_loosened"] == 4
    # V2 / V14 never ask a captive screw anything
    for r in _of(records, "V2") + _of(records, "V14"):
        assert not set(r["label"]["evidence"]["instances"]) & set(W.SCREWS)
    # the motherboard screw is asked present / removed, never fastened
    screw = [r for r in _of(records, "V2")
             if r["label"]["answer_check"].get("instance") == W.BOARD_SCREW]
    assert screw and all(r["label"]["answer"]["state"] == "present" for r in screw)
    assert all(r["label"]["answer_check"]["allowed"] == ["present", "removed"] for r in screw)
    assert Checker(db, tax, W.DESKTOP).check_all(records) == len(records)


def test_v5_is_out_of_the_default_set_but_still_one_flag_away(scene):
    db, tax, tmp = scene
    assert "V5" not in TASKS
    out = tmp / "scan.jsonl"
    summary = export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out))
    assert "V5" not in summary["by_task"]
    summary = export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(tmp / "v5.jsonl"), tasks=["V5"])
    assert set(summary["by_task"]) == {"V5"}


def test_the_cli_accepts_v5_and_rejects_a_typo(tmp_path: Path, capsys):
    from app_scene import DESKTOP, VIEW, make_db, write_paths_yaml
    from tda.cli import EXIT_ERROR, main

    db, _paths, _tax = make_db(tmp_path)
    db.close()
    paths = write_paths_yaml(tmp_path)
    out = tmp_path / "v5.jsonl"
    code = main(["--paths", paths, "export-vlm", "--desktops", str(DESKTOP), "--view", VIEW,
                 "--tasks", "V5", "--out", str(out)])
    assert code != EXIT_ERROR or "not a task" not in capsys.readouterr().out
    assert main(["--paths", paths, "export-vlm", "--desktops", str(DESKTOP), "--view", VIEW,
                 "--tasks", "V55", "--out", str(out)]) == EXIT_ERROR
    assert "not a task: V55" in capsys.readouterr().out


def _v2(desktop, i, cls, state, prior):
    return {"id": f"rec_{desktop:04d}{i:012x}", "prompt": {"task": "V2"},
            "label": {"desktop": desktop, "step": i, "answer": {"state": state},
                      "class_prior_answer": prior,
                      "answer_check": {"derive": "instance_state", "class": cls,
                                       "instance": f"{cls}.{i % 3 + 1:02d}"}}}


def _pool(desktop):
    return ([_v2(desktop, i, "ram_latch", "closed", "closed") for i in range(30)]
            + [_v2(desktop, 100 + i, "ram_latch", "open", "closed") for i in range(10)]
            + [_v2(desktop, 200 + i, "psu", "installed", "installed") for i in range(25)]
            + [_v2(desktop, 300 + i, "cpu_socket_lever", "open", "closed")
               for i in range(6)]
            + [_v2(desktop, 400 + i, "cpu_socket_lever", "closed", "closed")
               for i in range(4)])


def test_v2_is_thinned_per_desktop_so_no_answer_is_the_majority():
    kept, stats = balance_v2(_pool(1))

    def of(cls, state):
        return sum(1 for r in kept if r["label"]["answer_check"]["class"] == cls
                   and r["label"]["answer"]["state"] == state)

    assert of("ram_latch", "closed") == 10 and of("ram_latch", "open") == 10
    # a class whose every answer is the same keeps none: it would only measure it
    assert of("psu", "installed") == 0
    # neither value may be the majority -- the non-prior one is capped as well
    assert of("cpu_socket_lever", "open") == 4 and of("cpu_socket_lever", "closed") == 4
    for r in kept:
        prior_right = r["label"]["answer"]["state"] == r["label"]["class_prior_answer"]
        assert r["label"]["prior_is_correct"] is prior_right
    assert stats["state:ram_latch"] == {"before": 40, "after": 20,
                                        "prior_correct_before": 30}
    again, _ = balance_v2(_pool(1))
    assert [r["id"] for r in again] == [r["id"] for r in kept]


def test_v2_thinning_does_not_depend_on_the_other_desktops():
    """D1 alone and D1 inside 1-66 keep exactly the same D1 questions (review M)."""
    alone, _ = balance_v2(_pool(1))
    together, _ = balance_v2(_pool(2) + _pool(1) + _pool(3))
    mine = [r["id"] for r in together if r["label"]["desktop"] == 1]
    assert mine == [r["id"] for r in alone]


def test_v2_records_carry_the_desktops_own_prior(scene):
    db, tax, tmp = scene
    export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(tmp / "s.jsonl"))
    records = [r for r in _read(tmp / "s.jsonl") if r["prompt"]["task"] == "V2"]
    assert records
    for r in records:
        check = r["label"]["answer_check"]
        if check["derive"] == "instance_state" and check["class"] == "cover":
            assert r["label"]["class_prior_answer"] == "closed"  # where it starts
        if check["derive"] == "screw_count":
            assert r["label"]["class_prior_answer"] == 1  # one board screw
        assert r["label"]["prior_is_correct"] is (
            json.dumps(r["label"]["answer"].get("state", r["label"]["answer"].get("count")))
            == json.dumps(r["label"]["class_prior_answer"]))


def test_the_summary_says_why_each_task_was_silent(scene):
    db, tax, tmp = scene
    summary = export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(tmp / "s.jsonl"))
    reasons = summary["skip_reasons"][str(W.DESKTOP)]
    assert "V3" in reasons and "V7" in reasons
    written = json.loads((tmp / "s.summary.json").read_text(encoding="utf-8"))
    assert written["skip_reasons"] == json.loads(json.dumps(summary["skip_reasons"]))
