"""V1 / V8 boxes (owner 2026-11-04, decisions "V1 / V8（框）"): the IoU >= 0.5 headline stays,
the COCO-style average over 0.50:0.05:0.95 rides beside it (``f1_50_95`` / ``hit_50_95``), and
every answer is graded as given and with every coordinate x 1.25 (a model answering in an
internally resized 0.8 frame), the better grade kept and its frame recorded."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tda.core.export.vlm_check import COCO_IOUS, FRAME_SCALES, gold_reply, grade

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for p in (SCRIPTS, SCRIPTS / "vlm_run"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import vlm_io as IO  # noqa: E402
import vlm_score  # noqa: E402


def _record(rid: str, task: str, answer: dict, check: dict, desktop: int = 1) -> dict:
    return {"id": rid, "prompt": {"task": task, "images": ["img_0"], "question": "?"},
            "label": {"desktop": desktop, "view": "scan", "step": 1, "answer": answer,
                      "answer_check": check, "template_id": f"{task}.t0", "split": "test"}}


def _v1(rid="v1", comps=(("cpu", [0, 0, 100, 100]),), desktop=1) -> dict:
    return _record(rid, "V1", {"components": [{"class": c, "bbox": b} for c, b in comps]},
                   {"type": "components", "iou_threshold": 0.5, "metric": "set_f1"}, desktop)


def _v8(rid="v8", box=(100, 100, 50, 50), metric="box_iou", desktop=1) -> dict:
    return _record(rid, "V8", {"instance": "fan.01", "bbox": list(box)},
                   {"type": "boxes", "iou_threshold": 0.5, "metric": metric}, desktop)


def _v8_set(rid="v8s", boxes=([10, 10, 40, 40], [200, 10, 40, 40]), desktop=1) -> dict:
    return _record(rid, "V8", {"components": [
        {"class": "screw", "instance": f"screw.0{i + 1}", "match_class": "screw.psu", "bbox": b}
        for i, b in enumerate(boxes)]},
        {"type": "components", "class_filter": "screw.psu", "iou_threshold": 0.5,
         "metric": "set_f1+box_iou"}, desktop)


def _scaled(box, s):
    return [v * s for v in box]


def test_the_thresholds_and_frames():
    assert COCO_IOUS == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    assert FRAME_SCALES == (1.0, 1.25)


def test_v1_headline_coco_average_and_frame():
    r = _v1(comps=(("cpu", [0, 0, 100, 100]), ("psu", [200, 200, 80, 40])))
    g = grade(r, gold_reply(r))
    assert g.score == 1.0 and g.metrics["f1_50_95"] == 1.0 and g.metrics["frame"] == 1.0
    # the same boxes answered in a 0.8 frame: read x 1.25 they are the truth
    small = json.dumps({"components": [{"class": "cpu", "bbox": _scaled([0, 0, 100, 100], 0.8)},
                                       {"class": "psu", "bbox": _scaled([200, 200, 80, 40], 0.8)}]})
    g = grade(r, small)
    assert g.score == 1.0 and g.metrics["frame"] == 1.25
    assert g.metrics["f1_50_95"] == pytest.approx(1.0)
    # IoU 0.7: a hit at 0.50..0.70 (5 of 10 thresholds), the headline unchanged
    one = _v1()
    g = grade(one, json.dumps({"components": [{"class": "cpu", "bbox": [0, 0, 100, 70]}]}))
    assert g.score == 1.0 and g.metrics["f1_50_95"] == pytest.approx(0.5)
    # x 1.25 gives the same IoU here: a tie keeps the boxes as given
    assert g.metrics["frame"] == 1.0
    # a box no frame rescues still misses; the wrong class never matches
    assert grade(one, json.dumps({"components": [{"class": "cpu", "bbox": [0, 0, 40, 40]}]})
                 ).score == 0.0
    assert grade(one, json.dumps({"components": [{"class": "psu", "bbox": [0, 0, 100, 100]}]})
                 ).score == 0.0


def test_v8_single_box_hit_average_and_frame():
    r = _v8()
    g = grade(r, gold_reply(r))
    assert g.score == 1.0 and g.metrics["hit_50_95"] == 1.0 and g.metrics["frame"] == 1.0
    g = grade(r, json.dumps({"bbox": _scaled([100, 100, 50, 50], 0.8)}))
    assert g.score == 1.0 and g.metrics["frame"] == 1.25 and g.metrics["hit_50_95"] == 1.0
    # IoU 0.8 as given: hits at 0.50..0.80, 7 of 10
    g = grade(r, json.dumps({"bbox": [100, 100, 50, 40]}))
    assert g.score == 1.0 and g.metrics["hit_50_95"] == pytest.approx(0.7)
    assert g.metrics["frame"] == 1.0
    assert grade(r, json.dumps({"bbox": [0, 0, 20, 20]})).score == 0.0
    # a transform (crop / resize of the shown image) still applies, after the frame
    g = grade(r, json.dumps({"bbox": [50, 50, 25, 25]}), transform={"scale": 0.5})
    assert g.score == 1.0 and g.metrics["frame"] == 1.0
    g = grade(r, json.dumps({"bbox": [40, 40, 20, 20]}), transform={"scale": 0.5})
    assert g.score == 1.0 and g.metrics["frame"] == 1.25


def test_v8_point_in_box_reads_both_frames():
    r = _v8(metric="point_in_box")
    assert grade(r, json.dumps({"point": [120, 120]})).metrics == {"hit": True, "frame": 1.0}
    # (90, 90) is outside; x 1.25 it is (112.5, 112.5), inside
    g = grade(r, json.dumps({"point": [90, 90]}))
    assert g.score == 1.0 and g.metrics["frame"] == 1.25
    assert grade(r, json.dumps({"point": [10, 10]})).score == 0.0


def test_v8_components_set_f1_average_and_frame():
    r = _v8_set()
    g = grade(r, gold_reply(r))
    assert g.score == 1.0 and g.metrics["f1_50_95"] == 1.0 and g.metrics["frame"] == 1.0
    small = json.dumps({"components": [{"bbox": _scaled(b, 0.8)}
                                       for b in ([10, 10, 40, 40], [200, 10, 40, 40])]})
    g = grade(r, small)
    assert g.score == 1.0 and g.metrics["frame"] == 1.25
    # one box right, one missing: F1 2/3 at every threshold
    g = grade(r, json.dumps({"components": [{"bbox": [10, 10, 40, 40]}]}))
    assert g.score == pytest.approx(2 / 3) and g.metrics["f1_50_95"] == pytest.approx(2 / 3)


def test_the_scorer_reports_the_average_and_the_frame_share(tmp_path):
    truth = tmp_path / "truth.jsonl"
    rows = [_v1("a", desktop=1), _v1("b", desktop=2), _v8("c", desktop=1),
            _v8("d", desktop=2), _v8_set("e", desktop=3)]
    IO.write_jsonl(truth, rows)
    replies = {
        "a": gold_reply(rows[0]),
        "b": json.dumps({"components": [{"class": "cpu", "bbox": [0, 0, 80, 80]}]}),
        "c": json.dumps({"bbox": [100, 100, 50, 40]}),            # IoU 0.8, as given
        "d": json.dumps({"bbox": [80, 80, 40, 40]}),              # a 0.8 frame
        "e": gold_reply(rows[4]),
    }
    preds = tmp_path / "pred.jsonl"
    IO.write_jsonl(preds, [{"id": k, "raw_text": v} for k, v in replies.items()])
    assert vlm_score.main(["--truth", str(truth), "--pred", str(preds), "--out", str(tmp_path),
                           "--resamples", "20"]) == 0
    result = json.loads((tmp_path / "pred.score.json").read_text(encoding="utf-8"))
    s = result["strict"]
    # the headline stays IoU >= 0.5: every answer above is right there
    assert s["by_task"]["V1"]["mean"] == 1.0 and s["by_task"]["V8"]["mean"] == 1.0
    assert s["extra"]["V1"]["f1_50_95"]["mean"] == pytest.approx(1.0)
    # V8: c 0.7, d 1.0 (single boxes); e's set F1 average rides as f1_50_95
    assert s["extra"]["V8"]["hit_50_95"]["mean"] == pytest.approx(0.85)
    assert s["extra"]["V8"]["f1_50_95"]["mean"] == pytest.approx(1.0)
    assert s["frame"]["V1"] == {"1.0": 1, "1.25": 1, "share_1.25": 0.5}
    assert s["frame"]["V8"]["1.25"] == 1 and s["frame"]["V8"]["1.0"] == 2
    assert s["extra"]["V8"]["frame_1.25"]["mean"] == pytest.approx(1 / 3, abs=1e-3)
    md = (tmp_path / "pred.score.md").read_text(encoding="utf-8")
    assert "| task | frame 1.0 won | frame 1.25 won | 1.25 share |" in md
    assert "| V1 | 1 | 1 | 50.0% |" in md
