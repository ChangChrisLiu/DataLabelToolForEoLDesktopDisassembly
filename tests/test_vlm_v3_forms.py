"""V3's two forms (owner 2026-11-04, "两个都要，可以成两种问题").

* ``free`` -- the question as it was, no hint, now also asking where the machine changed
  (``"region"``, a box on the scanner image); the score stays verb + target, region IoU and
  region hit are extra metrics;
* ``roi`` -- the same step with a yellow dashed hint box drawn on both scanner images (the
  gold region grown 2.5 times, at least 300 px a side, clipped), the answer as before.

The two share ``label.v3_pair``, which is the free record's own id, so the eval subset keeps
its V3 ids and takes each one's roi partner along; the audit keeps its ids.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import w3b_scene as W
from tda.core.db import Db
from tda.core.export.vlm import export_vlm, unit_tasks
from tda.core.export.vlm_check import format_instruction, format_key, gold_reply, grade, World
from tda.core.export.vlm_tasks import V3_HINT_MIN, opaque_record_id, v3_hint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vlm_eval_set as E  # noqa: E402
import vlm_io as IO  # noqa: E402
import vlm_score  # noqa: E402
import vlm_to_chat as C  # noqa: E402


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



@pytest.fixture(scope="module")
def multi(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("v3forms")
    db = Db(str(tmp / "s.sqlite"))
    tax = W.build(db)
    out = tmp / "vlm_multi.jsonl"
    export_vlm(db, tax, [W.DESKTOP], "multi", str(out), tasks=unit_tasks("multi"))
    db.close()
    records = list(IO.read_jsonl(out))
    worlds, _ = IO.load_oracle(out)
    return {"tmp": tmp, "out": out, "records": records, "world": worlds[W.DESKTOP]}


def _v3(multi) -> tuple[list[dict], list[dict]]:
    v3 = [r for r in multi["records"] if r["prompt"]["task"] == "V3"]
    return ([r for r in v3 if r["label"]["v3_form"] == "free"],
            [r for r in v3 if r["label"]["v3_form"] == "roi"])


def test_every_v3_step_is_asked_free_and_with_a_hint(multi):
    free, roi = _v3(multi)
    assert free and roi
    by_pair = {r["label"]["v3_pair"]: r for r in free}
    for r in free:
        # the free record keeps the id the subset knows: the readable id never changed
        assert r["id"] == opaque_record_id(r["label"]["readable_id"])
        assert r["label"]["v3_pair"] == r["id"]
        assert "hint_regions" not in r["prompt"]
    for r in roi:
        partner = by_pair[r["label"]["v3_pair"]]
        assert r["label"]["readable_id"] == partner["label"]["readable_id"] + "-roi"
        assert r["label"]["step"] == partner["label"]["step"]
        assert r["prompt"]["images"] == partner["prompt"]["images"]
        # gold = the same verb / target / tool, no region asked
        assert _keys(r) == {k: v for k, v in _keys(partner).items()
                                        if k != "region"}
        assert r["label"]["template_id"].startswith("V3-roi.")


def test_the_free_form_asks_the_region_of_the_change(multi):
    free, _ = _v3(multi)
    asked = [r for r in free if r["label"]["answer_check"].get("region")]
    assert asked
    for r in asked:
        check, answer = r["label"]["answer_check"], _keys(r)
        assert answer["region"] == check["region"]
        assert "where the machine changed" in r["prompt"]["question"]
        assert format_key(r) == "V3.region"
        text = format_instruction(r)
        assert '"region": [x, y, width, height]' in text
        legend = r["prompt"]["image_legend"]
        scanner = [i + 1 for i, e in enumerate(legend) if e["view"] == "scan"]
        # the scanner's before and after pictures (this scene: scan + oak1, so images 1 and 3)
        assert [i + 1 for i in check["region_images"]] == scanner and len(scanner) == 2
        assert f"images {scanner[0]} and {scanner[1]}" in text
        # the gold region is the union of the changed part's scanner boxes before and after
        target = answer["target_instance"]
        x, y, w, h = check["region"]
        ev = r["label"]["evidence"]["bboxes"].get(target)
        if ev:
            assert x <= ev[0] and y <= ev[1] and x + w >= ev[0] + ev[2] and y + h >= ev[1] + ev[3]


def test_the_roi_form_draws_its_hint_on_both_scanner_images(multi):
    free, roi = _v3(multi)
    by_pair = {r["label"]["v3_pair"]: r for r in free}
    for r in roi:
        hints = r["prompt"]["hint_regions"]
        partner = by_pair[r["label"]["v3_pair"]]
        assert [h["image"] for h in hints] == partner["label"]["answer_check"]["region_images"]
        region = partner["label"]["answer_check"]["region"]
        hint = hints[0]["bbox"]
        assert all(h["bbox"] == hint for h in hints)
        assert hint == r["label"]["answer_check"]["hint_region"]
        # it contains the region, is larger, and stays inside the 64 x 64 test image
        assert hint[0] <= region[0] and hint[1] <= region[1]
        assert hint[0] + hint[2] >= region[0] + region[2] and hint[1] + hint[3] >= region[1] + region[3]
        assert 0 <= hint[0] and hint[0] + hint[2] <= W.HW[1] and hint[1] + hint[3] <= W.HW[0]
        assert "yellow dashed box" in r["prompt"]["question"]
        assert format_key(r) == "V3" and "region" not in format_instruction(r)
        assert "marks" not in r["prompt"]


def test_the_hint_box_rule():
    """2.5 times the region about its centre, at least 300 px a side, kept inside the image."""
    assert v3_hint([700, 700, 200, 100], (1600, 1600)) == [550.0, 600.0, 500.0, 300.0]
    assert v3_hint([10, 10, 20, 20], (1600, 1600)) == [0.0, 0.0, 300.0, 300.0]
    assert v3_hint([1590, 1500, 10, 90], (1600, 1600)) == [1300.0, 1300.0, 300.0, 300.0]
    assert v3_hint([10, 10, 20, 20], (64, 64)) == [0.0, 0.0, 64.0, 64.0]
    assert V3_HINT_MIN == 300


def test_gold_replies_score_one_and_the_region_is_graded_apart(multi):
    free, roi = _v3(multi)
    world = multi["world"]
    for r in free + roi:
        g = grade(r, gold_reply(r, world), world)
        assert g.score == 1.0, (r["label"]["readable_id"], g.metrics)
        if r["label"]["answer_check"].get("region"):
            assert g.metrics["region_iou"] == pytest.approx(1.0) and g.metrics["region_hit"]
    r = next(r for r in free if r["label"]["answer_check"].get("region"))
    reply = json.loads(gold_reply(r, world))
    # the region is extra: a wrong box keeps the action's score, a missing one too
    wrong = dict(reply, region=[0, 0, 1, 1])
    g = grade(r, json.dumps(wrong), world)
    assert g.score == 1.0 and g.metrics["region_iou"] < 0.5 and not g.metrics["region_hit"]
    del wrong["region"]
    assert grade(r, json.dumps(wrong), world).score == 1.0
    # a box answered on a picture shown at half size maps back like V1 / V8's
    x, y, w, h = r["label"]["answer_check"]["region"]
    half = dict(reply, region=[x / 2, y / 2, w / 2, h / 2])
    g = grade(r, json.dumps(half), world, transform={"crop": [0, 0], "scale": 0.5})
    assert g.metrics["region_iou"] == pytest.approx(1.0)


def test_the_scorer_rows_v3_and_v3_roi(multi, tmp_path):
    preds = tmp_path / "gold.jsonl"
    world = multi["world"]
    IO.write_jsonl(preds, [{"id": r["id"], "raw_text": gold_reply(r, world)}
                           for r in multi["records"]])
    result = vlm_score.score(multi["out"], preds, resamples=20)
    by_task = result["strict"]["by_task"]
    assert by_task["V3"]["mean"] == 1.0 and by_task["V3-roi"]["mean"] == 1.0
    assert result["strict"]["extra"]["V3"]["region_hit"]["mean"] == 1.0
    assert result["strict"]["extra"]["V3"]["region_iou"]["mean"] == 1.0
    assert result["chance"]["V3-roi"]["mean"] == 0.0 and "hint" in result["chance"]["V3-roi"]["rule"]
    # the hinted form is secondary: the "change" layer counts the free form only
    free, _ = _v3(multi)
    change = [r for r in multi["records"] if r["prompt"]["task"] in ("V3", "V10", "V12")
              and r["label"].get("v3_form") not in ("roi", "span")
              and (r["label"].get("v12_span") or 1) == 1]
    assert result["strict"]["by_layer"]["change"]["n"] == len(change)
    assert vlm_score.task_key(free[0]) == "V3"


def test_the_converter_draws_the_hint_in_yellow(multi, tmp_path):
    from PIL import Image

    _, roi = _v3(multi)
    r = roi[0]
    manifest = IO.load_manifest(multi["out"])
    root = tmp_path / "images"
    for image in r["prompt"]["images"]:
        path = root / manifest[image]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (W.HW[1], W.HW[0]), (90, 90, 90)).save(path)
    row = C.convert(r, manifest, C.Images(str(root), str(tmp_path / "assets"), None), "openai")
    yellow = lambda p: p[0] > 150 and p[1] > 130 and p[2] < 90  # noqa: E731
    for i, path in enumerate(row["meta"]["images"]):
        img = Image.open(path).convert("RGB")
        px = img.load()
        hit = any(yellow(px[x, y]) for x in range(img.size[0]) for y in range(img.size[1]))
        assert hit == (i in {h["image"] for h in r["prompt"]["hint_regions"]}), i
    assert C.CHAT_FORMAT_ID == "chat-v5"


def test_the_eval_subset_keeps_v3_ids_and_takes_the_partner(multi, tmp_path):
    """A V3 unit is keyed by its free record's id -- the draw is the free record's alone --
    and its roi partner rides along; the audit draws free records only."""
    units = E.units_of(multi["records"], "multi")
    v3 = [u for u in units if u.task == "V3"]
    for u in v3:
        assert u.key == u.records[0]["id"] and u.records[0]["label"]["v3_form"] == "free"
        assert [r["label"]["v3_form"] for r in u.records] in (["free"], ["free", "roi"])
    # without the roi records the draw picks the very same free records
    alone = [r for r in multi["records"] if r["label"].get("v3_form") != "roi"]
    with_roi = E.draw(units, seed=0)
    without = E.draw(E.units_of(alone, "multi"), seed=0)
    assert [u.key for u in with_roi["V3"]] == [u.key for u in without["V3"]]
    for task in with_roi:
        assert [u.key for u in with_roi[task]] == [u.key for u in without[task]], task
    out = E.build([multi["out"]], tmp_path / "eval_v3", seed=0, audit=12)
    chosen = list(IO.read_jsonl(tmp_path / "eval_v3" / "multi.jsonl"))
    free = {r["id"] for r in chosen if r["label"].get("v3_form") == "free"}
    roi = [r for r in chosen if r["label"].get("v3_form") == "roi"]
    assert roi and all(r["label"]["v3_pair"] in free for r in roi)
    audit = list(IO.read_jsonl(tmp_path / "eval_v3" / "audit" / "audit.jsonl"))
    assert not [r for r in audit if r["label"].get("v3_form") == "roi"]
    assert out["rows"][[r["task"] for r in out["rows"]].index("V3")]["unit"] == "pair"
