"""V15-moment on every view, without the ``verified`` gate (W3b).

"Are these two the same moment?" is answered by the step alignment and the
state machine; no mask of either view is read. The negative has to differ by a
large part -- a one-screw difference is undecidable at RS / OAK2 resolution.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import w3b_scene as W
from tda.core.db import Db
from tda.core.export.coco import load_ctx
from tda.core.export.vlm import export_vlm
from tda.core.export.vlm_tasks import BIG_PARTS
from tda.core.model import FrameKey
from vlm_checker import Checker


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


@pytest.fixture
def scene(tmp_path: Path):
    db = Db(str(tmp_path / "v15.sqlite"))
    tax = W.build(db)  # scan confirmed, oak1 frames only
    yield db, tax, tmp_path
    db.close()


def _moments(records):
    return [r for r in records if r["prompt"]["task"] == "V15"
            and r["label"].get("v15_variant") == "moment"]


def test_moment_pairs_need_no_verified_mate(scene):
    db, tax, tmp = scene
    out = tmp / "scan.jsonl"
    summary = export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out))
    records = _read(out)
    moments = _moments(records)
    assert moments
    # oak1 has no confirmed frame, so nothing cross-view (which reads its
    # visibility) is asked, and every moment record is unverified
    assert not [r for r in records if r["label"].get("v15_variant") == "cross"]
    for r in moments:
        assert r["label"]["views"] == [W.VIEW, W.OTHER]
        assert r["label"]["verified"] is False
        assert len(r["prompt"]["images"]) == 2
    pos = [r for r in moments if not r["label"].get("negative")]
    neg = [r for r in moments if r["label"].get("negative")]
    assert len(pos) == len(neg)
    assert all(r["label"]["answer_check"]["other_step"] == r["label"]["step"] for r in pos)
    assert summary["v15_moment_by_view"] == {
        f"{W.VIEW}+{W.OTHER}": {"positive": len(pos), "negative": len(neg)}}
    assert Checker(db, tax, W.DESKTOP).check_all(records) == len(records)


def test_a_negative_always_differs_by_a_large_part(scene):
    db, tax, tmp = scene
    out = tmp / "scan.jsonl"
    export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out))
    ctx = load_ctx(db, tax, W.DESKTOP, W.VIEW)
    for r in _moments(_read(out)):
        big = r["label"]["big_change"]
        if r["label"].get("negative"):
            assert big and all(ctx.cls_of(k) in BIG_PARTS for k in big)
            a, b = sorted((r["label"]["step"], r["label"]["answer_check"]["other_step"]))
            old, new = ctx.state_at(a), ctx.state_at(b)
            assert any(old[k].state != new[k].state for k in big)
        else:
            assert big == []
    # step 5 (two cooler screws loose) is two or three steps of screws from
    # either side; its negative has to reach the optical drive (step 8) or
    # the shroud (step 2)
    neg5 = [r for r in _moments(_read(out))
            if r["label"]["step"] == 5 and r["label"].get("negative")]
    assert neg5 and neg5[0]["label"]["answer_check"]["other_step"] in (2, 1, 8, 9, 10, 11)


def test_a_missing_mate_frame_asks_nothing(scene):
    db, tax, tmp = scene
    with db.conn:
        db.conn.execute("UPDATE frame SET missing=1 WHERE desktop=? AND view=? AND step=4",
                        (W.DESKTOP, W.OTHER))
    out = tmp / "scan.jsonl"
    export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out))
    moments = _moments(_read(out))
    assert 4 not in {r["label"]["step"] for r in moments}
    assert 4 not in {r["label"]["answer_check"]["other_step"] for r in moments}


def test_synced_cameras_more_than_twenty_seconds_apart_are_no_moment(tmp_path: Path):
    """Review M: D42's RS steps 16-17 sit ~36 s off the OAKs (a raw-data shift)."""
    db = Db(str(tmp_path / "ts.sqlite"))
    tax = W.build(db, views=(W.OTHER, "rs"), confirm=())
    with db.conn:
        db.conn.execute("UPDATE frame SET ts='2025-06-01T09:00:36' "
                        "WHERE desktop=? AND view='rs' AND step=5", (W.DESKTOP,))
        db.conn.execute("UPDATE frame SET ts='2025-06-01T09:00:15' "
                        "WHERE desktop=? AND view='rs' AND step=6", (W.DESKTOP,))
    out = tmp_path / "oak1.jsonl"
    summary = export_vlm(db, tax, [W.DESKTOP], W.OTHER, str(out))
    moments = _moments(_read(out))
    steps = {r["label"]["step"] for r in moments}
    assert 5 not in steps          # 36 s apart: skipped, both halves
    assert 6 in steps              # 15 s apart: still the same moment
    assert summary["skip_reasons"][str(W.DESKTOP)]["V15"]["moment_ts_mismatch_rs"] == 1
    db.close()


def test_the_scanner_clock_is_not_checked(tmp_path: Path):
    """Scanner ``ts`` is no capture time: its pairs are never dropped for it."""
    db = Db(str(tmp_path / "ts2.sqlite"))
    tax = W.build(db)
    with db.conn:
        db.conn.execute("UPDATE frame SET ts='2025-06-09T09:00:00' "
                        "WHERE desktop=? AND view=?", (W.DESKTOP, W.VIEW))
    out = tmp_path / "scan.jsonl"
    before = len(_moments(_read(out))) if out.exists() else None
    export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(out))
    assert _moments(_read(out)) and before is None
    db.close()


def test_every_later_view_pair_is_asked_once(tmp_path: Path):
    """scan+oak1 from the scanner's file, oak1+... from OAK1's: one file per pair."""
    db = Db(str(tmp_path / "v15b.sqlite"))
    tax = W.build(db, confirm=())  # neither view confirmed: still asked
    a, b = tmp_path / "scan.jsonl", tmp_path / "oak1.jsonl"
    export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(a))
    export_vlm(db, tax, [W.DESKTOP], W.OTHER, str(b))
    assert _moments(_read(a))
    assert not _moments(_read(b))  # oak1 is the last view that has frames here
    assert all(r["label"]["view_confirmed"] is False for r in _moments(_read(a)))
    db.close()
