"""The W3b evaluation scripts: references, chat formats, scorer, sampler.

All on the :mod:`tests.w3b_scene` export, with fake model outputs -- no model,
no API call, no download.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

import w3b_scene as W
from tda.core.db import Db
from tda.core.export.vlm import export_vlm
from tda.core.export.vlm_check import gold_reply, parse_lenient, parse_strict

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import vlm_io as IO  # noqa: E402
import vlm_references  # noqa: E402
import vlm_sample  # noqa: E402
import vlm_score  # noqa: E402
import vlm_to_chat  # noqa: E402


@pytest.fixture(scope="module")
def export(tmp_path_factory) -> Path:
    """The scanner (confirmed) and OAK1 (frames only) exports of the scene."""
    tmp = tmp_path_factory.mktemp("w3b_scripts")
    db = Db(str(tmp / "s.sqlite"))
    tax = W.build(db)
    out = tmp / "vlm_scan.jsonl"
    export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out))
    export_vlm(db, tax, [W.DESKTOP], W.OTHER, str(tmp / "vlm_oak1.jsonl"))
    db.close()
    # the image cache the manifest points at: tiny PNGs, one per frame
    from PIL import Image

    root = tmp / "cache"
    for entry in IO.load_manifest(out).values():
        path = root / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (entry["step"] * 20 % 255, 80, 120)).save(path)
    return out


def _records(path) -> list[dict]:
    return list(IO.read_jsonl(path))


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #
def test_the_three_references_keep_ids_and_labels(export, tmp_path):
    assert vlm_references.main(["--export", str(export), "--out", str(tmp_path)]) == 0
    records = {r["id"]: r for r in _records(export)}
    blind = _records(tmp_path / "blind.jsonl")
    oracle = _records(tmp_path / "state_oracle.jsonl")
    assert [r["id"] for r in blind] == list(records)
    assert [r["id"] for r in oracle] == list(records)
    for r in blind:
        assert r["prompt"]["images"] == [] and r["label"]["reference"] == "blind"
        assert r["label"]["answer_check"] == records[r["id"]]["label"]["answer_check"]
        assert r["prompt"]["question"] == records[r["id"]]["prompt"]["question"]
    for r in oracle:
        src = records[r["id"]]
        assert r["prompt"]["images"] == []
        assert len(r["prompt"]["state_tables"]) == len(src["prompt"]["images"])
        assert "image_undecidable" in r["label"]
        for table in r["prompt"]["state_tables"]:
            assert "CPU 1:" in table and "cpu.01" not in table
    # the scanner is confirmed, OAK1 is not: visibility only where it was drawn
    v15 = next(r for r in oracle if r["prompt"]["task"] == "V15")
    scan_table, oak_table = v15["prompt"]["state_tables"]
    assert "visible" in scan_table and "visible" not in oak_table
    undecidable = [r for r in oracle if r["label"]["image_undecidable"]]
    assert undecidable  # V6/V7/V10/V16 while two cooler screws are loose


def test_the_protocol_prior_answers_from_the_previous_action_only(export, tmp_path):
    from tda.core.taxonomy import load_taxonomy

    from tda.core.export.vlm_check import World

    records = _records(export)
    worlds, _ = IO.load_oracle(export)
    # the held-out desktop is never in its own fit: a twin with the same log
    # stands in as the training desktop
    w = worlds[W.DESKTOP]
    worlds[99] = World(99, w.instances, w.edges, w.log)
    variant, preds, report = vlm_references.protocol_prior(
        records, worlds, load_taxonomy(), train={W.DESKTOP, 99})
    assert report["fit"] == "train desktops"
    tasks = {r["prompt"]["task"] for r in variant}
    assert tasks <= {"V6", "V7", "V16"} and "V6" in tasks and "V7" in tasks
    for r in variant:
        assert r["prompt"]["images"] == []
        assert set(r["prompt"]) <= {"task", "images", "question", "previous_action", "options"}
    assert {p["id"] for p in preds} == {r["id"] for r in variant}
    # V6's two forms are reported apart: the open headline, the choice beside it
    assert set(report["by_task"]) == tasks | {"V6-mc"}
    for row in report["by_task"].values():
        assert 0.0 <= row["accuracy"] <= 1.0
    # fitted on the very log it is asked about, the next action is mostly right
    assert report["by_task"]["V6"]["accuracy"] >= 0.5
    assert report["v6_next_verb_class_coarse"] >= 0.5


def test_the_bigram_counts_how_often_a_kind_repeats():
    seqs = [[("open", "cover"), ("unscrew", "screw.cpu_cooler")] * 1
            + [("unscrew", "screw.cpu_cooler")] * 3 + [("remove", "cpu")]]
    model = vlm_references.Bigram(seqs)
    assert model.repeats[("unscrew", "screw.cpu_cooler")] == 4
    assert model.best(("open", "cover")) == ("unscrew", "screw.cpu_cooler")
    coarse = vlm_references.Bigram(seqs, coarse=True)
    assert coarse.best(("open", "cover")) == ("unscrew", "screw")


# --------------------------------------------------------------------------- #
# chat formats
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", ["openai", "qwen", "internvl"])
def test_every_chat_format_carries_the_question_images_and_answer_format(export, tmp_path, fmt):
    out = tmp_path / f"{fmt}.jsonl"
    assert vlm_to_chat.main(["--export", str(export), "--format", fmt,
                             "--image-root", str(export.parent / "cache"),
                             "--out", str(out)]) == 0
    records = {r["id"]: r for r in _records(export)}
    rows = _records(out)
    assert [r["id"] for r in rows] == list(records)
    for row in rows:
        src = records[row["id"]]
        meta = row["meta"]
        assert meta["prompt_template_id"] == f"chat-v1:{src['label']['template_id']}"
        assert len(meta["images"]) == len(src["prompt"]["images"])
        if fmt == "internvl":
            text = row["conversations"][0]["value"]
            assert text.count("<image>") == len(src["prompt"]["images"])
        else:
            parts = row["messages"][1]["content"]
            text = next(p["text"] for p in parts if p["type"] == "text")
            kind = "image_url" if fmt == "openai" else "image"
            assert sum(p["type"] == kind for p in parts) == len(src["prompt"]["images"])
        assert src["prompt"]["question"] in text
        assert "Answer with JSON only" in text
        if src["prompt"].get("options"):
            assert "\nA. " in text and "\nB. " in text
        assert "cpu.01" not in text and "D41" not in text


def test_roi_crop_composite_and_base64(export, tmp_path):
    from PIL import Image

    records = _records(export)
    manifest = IO.load_manifest(export)
    images = vlm_to_chat.Images(str(export.parent / "cache"), str(tmp_path / "assets"), None)
    v8 = next(r for r in records if r["prompt"]["task"] == "V8")
    row = vlm_to_chat.convert(v8, manifest, images, "openai", roi="evidence")
    [transform] = row["meta"]["transforms"]
    crop = Image.open(row["meta"]["images"][0])
    x, y, w, h = v8["label"]["answer"]["bbox"]
    cx, cy = transform["crop"]
    # the crop holds the evidence box, and the offset maps it back
    assert 0 <= x - cx and x - cx + w <= crop.size[0]
    assert 0 <= y - cy and y - cy + h <= crop.size[1]
    assert crop.size != (64, 64) or transform["crop"] == [0, 0]
    pair = next(r for r in records if len(r["prompt"]["images"]) == 2)
    row = vlm_to_chat.convert(pair, manifest, images, "qwen", composite=True)
    assert row["meta"]["composite"] is True and len(row["meta"]["images"]) == 1
    assert Image.open(row["meta"]["images"][0]).size[0] == 64 * 2 + 16
    row = vlm_to_chat.convert(pair, manifest, images, "openai")
    assert len(row["meta"]["images"]) == 2 and row["meta"]["composite"] is False
    row = vlm_to_chat.convert(v8, manifest, images, "openai", inline=True)
    url = row["messages"][1]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    base64.b64decode(url.split(",", 1)[1])


def test_a_state_oracle_record_turns_into_text(export, tmp_path):
    vlm_references.main(["--export", str(export), "--out", str(tmp_path)])
    oracle = _records(tmp_path / "state_oracle.jsonl")
    row = vlm_to_chat.convert(oracle[0], {}, vlm_to_chat.Images(None, None, None), "openai")
    parts = row["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["text"]
    assert "instead of the photo" in parts[0]["text"]


# --------------------------------------------------------------------------- #
# the scorer
# --------------------------------------------------------------------------- #
def test_parsing_strict_and_lenient():
    assert parse_strict('{"changed": true}') == {"changed": True}
    assert parse_strict('```json\n{"changed": true}\n```') == {"changed": True}
    assert parse_strict('Sure! {"changed": true}') is None
    assert parse_lenient('Sure! {"changed": true} hope that helps') == {"changed": True}
    assert parse_lenient("{'valid': False, 'first_error_step': 2}") == \
        {"valid": False, "first_error_step": 2}
    rec = {"prompt": {"task": "V12"}, "label": {"answer_check": {}}}
    assert parse_lenient("Yes, the cover was opened.", rec) == {"changed": True}
    assert parse_lenient("no idea", {"prompt": {"task": "V3"},
                                     "label": {"answer_check": {}}}) is None
    v6 = {"prompt": {"task": "V6", "options": [{}] * 5},
          "label": {"answer_check": {}, "options": [{}] * 5}}
    assert parse_lenient("The answer is option C.", v6) == {"choice": "C"}
    assert parse_lenient("(B)", v6) == {"choice": "B"}
    assert parse_lenient("d", v6) == {"choice": "D"}
    # the pronoun is not option I (and there is no option I)
    assert parse_lenient("I cannot tell.", v6) is None


def test_the_scorer_with_fake_outputs(export, tmp_path):
    records = _records(export)
    worlds, _ = IO.load_oracle(export)
    world = worlds[W.DESKTOP]
    rows = []
    for i, r in enumerate(records):
        gold = gold_reply(r, world)
        if i % 5 == 0:
            raw = "I think it is " + gold + " -- hope that helps"  # lenient only
        elif i % 7 == 0:
            raw = "garbage"                                        # neither
        else:
            raw = gold
        rows.append({"id": r["id"], "raw_text": raw})
    rows = rows[:-1]  # one record never answered
    preds = tmp_path / "fake.jsonl"
    IO.write_jsonl(preds, rows)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"by_task": {"V6-mc": {"accuracy": 0.799}}}),
                     encoding="utf-8")
    assert vlm_score.main(["--truth", str(export), "--pred", str(preds), "--out",
                           str(tmp_path), "--resamples", "200", "--prior", str(prior)]) == 0
    result = json.loads((tmp_path / "fake.score.json").read_text(encoding="utf-8"))
    strict, lenient = result["strict"], result["lenient"]
    assert result["predicted"] == len(records) - 1
    assert strict["parse_failure_rate_all"] > lenient["parse_failure_rate_all"] > 0
    for task, cell in strict["by_task"].items():
        assert cell["mean"] <= lenient["by_task"][task]["mean"] + 1e-9
        lo, hi = cell["ci95"]
        assert lo <= cell["mean"] <= hi
    assert set(strict["by_layer"]) <= {"L1", "L2", "L3", "change", "cross"}
    assert strict["by_template"]
    assert "V7" in strict["extra"] and "required_recall" in strict["extra"]["V7"]
    assert "decidable" in strict["extra"]["V7"]
    md = (tmp_path / "fake.score.md").read_text(encoding="utf-8")
    assert "| V7 |" in md and "text-only upper bound" in md
    mc_row = next(line for line in md.splitlines() if line.startswith("| V6-mc |"))
    assert mc_row.rstrip().endswith("| 79.9% |")
    assert result["headline"]["v6_open"] == strict["by_task"]["V6"]
    # seeded: the same inputs give the same intervals
    again = vlm_score.score(export, preds, resamples=200)
    assert again["strict"]["by_task"] == strict["by_task"]


def test_the_gold_reply_scores_one_everywhere(export):
    records = _records(export) + _records(export.parent / "vlm_oak1.jsonl")
    from tda.core.export.vlm_check import grade
    from tda.core.taxonomy import load_taxonomy

    tax = load_taxonomy()
    worlds, _ = IO.load_oracle(export)
    for r in records:
        g = grade(r, gold_reply(r, worlds[W.DESKTOP]), worlds[W.DESKTOP], tax)
        assert g.score == 1.0, (r["label"]["readable_id"], g.metrics)


def test_the_cluster_bootstrap_resamples_desktops():
    rows = [(1, 1.0)] * 50 + [(2, 0.0)] * 50
    mean, lo, hi = vlm_score.cluster_bootstrap(rows, 500, seed=3)
    assert mean == 0.5
    # two clusters: a resample is all-1, all-0 or half -- a wide interval,
    # where a record-level bootstrap would claim +-0.1
    assert lo == 0.0 and hi == 1.0
    assert vlm_score.cluster_bootstrap(rows, 500, seed=3) == (mean, lo, hi)


# --------------------------------------------------------------------------- #
# the sampler
# --------------------------------------------------------------------------- #
def _fake(task: str, desktop: int, i: int, negative: bool = False) -> dict:
    label = {"desktop": desktop, "view": "scan", "step": i, "answer": {"x": i},
             "answer_check": {}}
    if negative:
        label["negative"] = "graph_blocked"
    return {"id": f"rec_{task}_{desktop}_{i}", "prompt": {"task": task, "images": [],
                                                       "question": f"q{i}"}, "label": label}


def test_the_sampler_is_stratified_capped_and_deterministic(tmp_path):
    records = []
    for d in range(1, 9):
        records += [_fake("V2", d, i) for i in range(200)]
        records += [_fake("V4", d, i, negative=i % 2 == 1) for i in range(20)]
        records += [_fake("V7", d, i) for i in range(3)]
    a = vlm_sample.draw(records, size=60, per_task=25, seed=1)
    b = vlm_sample.draw(records, size=60, per_task=25, seed=1)
    c = vlm_sample.draw(records, size=60, per_task=25, seed=1, purpose="repeat")
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["id"] for r in a] != [r["id"] for r in c]
    by_task = {}
    for r in a:
        by_task[r["prompt"]["task"]] = by_task.get(r["prompt"]["task"], 0) + 1
    assert max(by_task.values()) <= int(0.2 * 60)  # no task above its share
    v4 = [r for r in a if r["prompt"]["task"] == "V4"]
    neg = sum(bool(r["label"].get("negative")) for r in v4)
    assert abs(neg - (len(v4) - neg)) <= 1
    assert {r["label"]["desktop"] for r in a} == set(range(1, 9))
    only = vlm_sample.draw(records, size=60, seed=1, test_desktops={7, 8})
    assert {r["label"]["desktop"] for r in only} == {7, 8}


def test_the_sampler_writes_ids_and_a_reviewer_sheet(export, tmp_path):
    assert vlm_sample.main(["--export", str(export), "--out", str(tmp_path),
                            "--size", "40", "--per-task", "4"]) == 0
    ids = (tmp_path / "audit.ids.txt").read_text(encoding="utf-8").split()
    assert ids and len(ids) == len(set(ids))
    sheet = (tmp_path / "audit.sheet.md").read_text(encoding="utf-8")
    assert sheet.count("\n") == len(ids) + 2
    assert "scan/D41/" in sheet and "| ok? |" in sheet
