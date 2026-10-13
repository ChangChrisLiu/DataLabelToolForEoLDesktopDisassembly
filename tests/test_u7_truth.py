"""Task U7, ruling A: one open conflict per frozen row.

On D13/scan the annotator re-traced four confirmed RAM clips at frame 41 in
fourteen small commits.  Every commit changed the compilation of the frames the
clips reach, and the open queue was deduplicated on ``(step, instance, new
value)``: each commit queued one more conflict for the same frozen row -- 13
open for ``(41, motherboard.01)`` alone, twelve of them about shapes that no
longer existed.  A newer disagreement now *replaces* the open one.

The scene is ``truth_scenes``': a PSU and its screw, frame 2 confirmed.
"""
from __future__ import annotations

import collections
import random

import pytest

from tda.core import masks
from tda.core.db import Db
from tda.core.model import ShapePart
from tda.core.truth import (
    AGREE_AGAIN,
    CLOSE_KIND,
    DUPLICATE,
    OVERTAKEN,
    REOPEN_KIND,
    ROW_GONE,
    StaleConflictError,
)
from truth_scenes import DESKTOP, PSU, SCREW, VIEW, Scene, build_scene, rect, replace_parts


@pytest.fixture
def db(tmp_db_path: str):
    d = Db(tmp_db_path)
    yield d
    d.close()


@pytest.fixture
def scene(db: Db) -> Scene:
    built = build_scene(db)
    # a screw big enough that moving it is beyond the re-tracing tolerance
    replace_parts(built, built.screw_kf,
                  [ShapePart("main", masks.encode_rle(rect(14, 14, 34, 34)))])
    built.refresh_all()
    built.svc.verify_frame(built.key(2), "lin")
    return built


def _open(scene: Scene) -> list[dict]:
    return scene.db.conflicts(DESKTOP, VIEW, open_only=True)


def _per_row(scene: Scene) -> collections.Counter:
    return collections.Counter((c["step"], c["instance"]) for c in _open(scene))


def _closings(scene: Scene, reason: str | None = None) -> list[dict]:
    rows = [op for op in scene.db.ops(DESKTOP, VIEW, limit=10_000) if op["kind"] == CLOSE_KIND]
    return [op for op in rows if reason is None or op["payload"]["reason"] == reason]


def _psu(scene: Scene, y1: int) -> None:
    """Re-trace the PSU: ``(10, 10)`` to ``(50, y1)``, far from what was frozen."""
    replace_parts(scene, scene.psu_kf,
                  [ShapePart("main", masks.encode_rle(rect(10, 10, 50, y1)))])


def _screw(scene: Scene, x1: int) -> None:
    replace_parts(scene, scene.screw_kf,
                  [ShapePart("main", masks.encode_rle(rect(14, 14, x1, 34)))])


# --------------------------------------------------------------------------- #
# the invariant
# --------------------------------------------------------------------------- #
def test_fourteen_small_commits_leave_one_open_conflict_per_row(scene: Scene):
    """The D13 story: one re-trace committed in many steps, a refresh after each."""
    ids = []
    for y1 in range(40, 26, -1):           # fourteen commits of one re-trace
        _psu(scene, y1)
        out = scene.svc.refresh(scene.key(2))
        assert out["standing"] == 1
        assert max(_per_row(scene).values()) == 1
        ids.append(next(c["id"] for c in _open(scene) if c["instance"] == PSU))

    assert len(set(ids)) == 14                        # each commit is a new disagreement ...
    assert [c["id"] for c in _open(scene)] == [ids[-1]]   # ... and only the last one is open
    open_now = _open(scene)[0]
    assert masks.bbox(masks.decode_rle(open_now["new_rle"])) == (10, 10, 50, 27)
    for cid in ids[:-1]:
        closed = scene.db.get_conflict(cid)
        assert (closed["status"], closed["resolution"]) == ("resolved", "superseded")


def test_the_same_new_value_is_not_queued_again_and_logs_nothing(scene: Scene):
    _psu(scene, 30)
    first = scene.svc.refresh(scene.key(2))
    rows = len(scene.db.ops(DESKTOP, VIEW, limit=10_000))

    again = scene.svc.refresh(scene.key(2), ignore_digest=True)

    assert (first["conflicts"], again["conflicts"], again["overtaken"]) == (1, 0, 0)
    assert len(_open(scene)) == 1
    assert len(scene.db.ops(DESKTOP, VIEW, limit=10_000)) == rows   # no closing row


def test_a_refresh_overtaking_two_rows_writes_one_op_row(scene: Scene):
    """Both frozen rows of the frame move twice: the second pass replaces both, once."""
    _psu(scene, 30)
    _screw(scene, 24)
    scene.svc.refresh(scene.key(2))
    before = {c["instance"]: c["id"] for c in _open(scene)}
    assert set(before) == {PSU, SCREW}

    _psu(scene, 20)
    _screw(scene, 18)
    out = scene.svc.refresh(scene.key(2))

    assert (out["conflicts"], out["overtaken"], out["standing"]) == (2, 2, 2)
    rows = _closings(scene, OVERTAKEN)
    assert len(rows) == 1
    payload, inverse = rows[0]["payload"], rows[0]["inverse"]
    assert payload == {"step": 2, "conflicts": sorted(before.values()),
                       "instances": sorted([PSU, SCREW]), "resolution": "superseded",
                       "reason": "overtaken by a newer disagreement"}
    assert inverse == {"kind": REOPEN_KIND, "step": 2, "conflicts": sorted(before.values())}
    assert rows[0]["annotator"] == "system"
    assert max(_per_row(scene).values()) == 1


def test_a_database_from_before_u7_is_cleaned_up_by_the_next_refresh(scene: Scene):
    """Thirteen stacked conflicts on one row (the live DB): one refresh leaves one."""
    _psu(scene, 30)
    scene.svc.refresh(scene.key(2))
    current = _open(scene)[0]
    key = scene.key(2)
    for y1 in range(40, 28, -1):     # what twelve earlier commits left open
        scene.db.add_conflict(key, PSU, current["old_rle"],
                              masks.encode_rle(rect(10, 10, 50, y1)), 100)
    assert _per_row(scene)[(2, PSU)] == 13

    out = scene.svc.refresh(key, ignore_digest=True)

    assert [c["id"] for c in _open(scene)] == [current["id"]]   # the one that is still true
    assert (out["conflicts"], out["overtaken"]) == (0, 12)


# --------------------------------------------------------------------------- #
# the accept_new that arrives too late: the same row, the same words
# --------------------------------------------------------------------------- #
def test_a_stale_accept_new_is_recorded_like_a_refresh_overtaking_it(scene: Scene):
    _psu(scene, 30)
    scene.svc.refresh(scene.key(2))
    cid = _open(scene)[0]["id"]
    _psu(scene, 20)                       # the inputs move on, nothing refreshed frame 2

    with pytest.raises(StaleConflictError) as err:
        scene.svc.resolve_conflict(cid, "accept_new", "lin")

    assert "overtaken by a newer disagreement" in str(err.value)
    assert "was queued again" in str(err.value)
    newest = _open(scene)
    assert len(newest) == 1 and newest[0]["id"] != cid
    assert scene.db.get_conflict(cid)["resolution"] == "superseded"
    # the same op-log row a refresh writes when it replaces a conflict
    last = scene.db.ops(DESKTOP, VIEW, limit=1)[0]
    assert last["kind"] == CLOSE_KIND and last["annotator"] == "system"
    # a refresh's row plus one key: who pressed N (round 2)
    assert last["payload"] == {"step": 2, "conflicts": [cid], "instances": [PSU],
                               "resolution": "superseded", "reason": OVERTAKEN,
                               "asked_by": "lin"}
    assert last["inverse"] == {"kind": REOPEN_KIND, "step": 2, "conflicts": [cid]}
    # and no second, human-signed row for the same supersession
    assert not [op for op in scene.db.ops(DESKTOP, VIEW, limit=10)
                if op["kind"] == "resolve_conflict" and op["payload"]["conflict"] == cid]


def test_a_stale_accept_new_that_finds_agreement_says_so(scene: Scene):
    _psu(scene, 30)
    scene.svc.refresh(scene.key(2))
    cid = _open(scene)[0]["id"]
    _psu(scene, 50)                        # back to what was frozen

    with pytest.raises(StaleConflictError):
        scene.svc.resolve_conflict(cid, "accept_new", "lin")

    assert _open(scene) == []
    last = scene.db.ops(DESKTOP, VIEW, limit=1)[0]
    assert (last["kind"], last["payload"]["reason"]) == (CLOSE_KIND, AGREE_AGAIN)


def test_a_refresh_and_a_late_accept_new_leave_the_same_queue(scene: Scene, db: Db, tmp_path):
    """Whichever path closes the old conflict, one open conflict with the current value."""
    _psu(scene, 30)
    scene.svc.refresh(scene.key(2))
    cid = _open(scene)[0]["id"]
    _psu(scene, 20)
    with pytest.raises(StaleConflictError):
        scene.svc.resolve_conflict(cid, "accept_new", "lin")
    late = [(c["instance"], c["new_rle"]["counts"]) for c in _open(scene)]

    scene.svc.refresh(scene.key(2), ignore_digest=True)       # nothing left to replace
    assert [(c["instance"], c["new_rle"]["counts"]) for c in _open(scene)] == late
    assert scene.db.get_conflict(cid)["status"] == "resolved"


# --------------------------------------------------------------------------- #
# reopen_agreed keeps the invariant
# --------------------------------------------------------------------------- #
def test_an_overtaken_closing_is_not_reopened(scene: Scene):
    _psu(scene, 30)
    scene.svc.refresh(scene.key(2))
    _psu(scene, 20)
    scene.svc.refresh(scene.key(2))
    op = _closings(scene, OVERTAKEN)[0]
    rows = len(scene.db.ops(DESKTOP, VIEW, limit=10_000))

    with pytest.raises(ValueError) as err:
        scene.svc.reopen_agreed(op["id"])

    assert OVERTAKEN in str(err.value) and AGREE_AGAIN in str(err.value)
    assert len(scene.db.ops(DESKTOP, VIEW, limit=10_000)) == rows
    assert max(_per_row(scene).values()) == 1


def test_an_agreement_is_not_reopened_over_a_newer_conflict_of_the_row(scene: Scene):
    """Agree, then disagree again: reopening the agreed one would make two."""
    _psu(scene, 30)
    scene.svc.refresh(scene.key(2))
    first = _open(scene)[0]["id"]
    _psu(scene, 50)
    assert scene.svc.refresh(scene.key(2))["closed"] == 1        # agrees again
    agreed = _closings(scene, AGREE_AGAIN)[0]
    _psu(scene, 20)
    scene.svc.refresh(scene.key(2))                               # a new disagreement
    newer = _open(scene)[0]["id"]

    assert scene.svc.reopen_agreed(agreed["id"]) == []
    assert [c["id"] for c in _open(scene)] == [newer]
    assert scene.db.get_conflict(first)["status"] == "resolved"


# --------------------------------------------------------------------------- #
# any sequence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_any_sequence_of_edits_refreshes_and_answers_keeps_one_per_row(scene: Scene,
                                                                        seed: int):
    rng = random.Random(seed)
    key = scene.key(2)
    for _ in range(40):
        move = rng.choice(["psu", "screw", "refresh", "refresh", "accept", "reopen", "keep"])
        if move == "psu":
            _psu(scene, rng.choice([20, 26, 30, 40, 50]))
        elif move == "screw":
            _screw(scene, rng.choice([18, 24, 34]))
        elif move == "refresh":
            scene.svc.refresh(key, ignore_digest=rng.random() < 0.5)
        elif move in ("accept", "keep") and _open(scene):
            target = rng.choice(_open(scene))["id"]
            try:
                scene.svc.resolve_conflict(
                    target, "accept_new" if move == "accept" else "keep_old", "lin")
            except ValueError:
                pass                       # stale (re-queued) or refused: both ordinary
        elif move == "reopen" and _closings(scene):
            try:
                scene.svc.reopen_agreed(rng.choice(_closings(scene))["id"])
            except ValueError:
                pass
        counts = _per_row(scene)
        assert not counts or max(counts.values()) == 1, (move, counts)


# --------------------------------------------------------------------------- #
# round 2
# --------------------------------------------------------------------------- #
def test_a_same_value_duplicate_is_closed_as_a_duplicate_not_overtaken(scene: Scene):
    """A pre-U7 stack of the very same disagreement: the oldest stays, the rest are duplicates."""
    _psu(scene, 30)
    scene.svc.refresh(scene.key(2))
    first = _open(scene)[0]
    for _ in range(2):
        scene.db.add_conflict(scene.key(2), PSU, first["old_rle"], first["new_rle"], 100)
    assert _per_row(scene)[(2, PSU)] == 3

    out = scene.svc.refresh(scene.key(2), ignore_digest=True)

    assert [c["id"] for c in _open(scene)] == [first["id"]]
    assert (out["conflicts"], out["overtaken"]) == (0, 2)
    rows = _closings(scene)
    assert [op["payload"]["reason"] for op in rows] == [DUPLICATE]
    assert DUPLICATE == "duplicate of an open conflict"
    with pytest.raises(ValueError):
        scene.svc.reopen_agreed(rows[0]["id"])


def test_one_refresh_writes_one_row_per_reason(scene: Scene):
    """A stack of a stale value and a duplicate of the current one: two reasons, two rows."""
    _psu(scene, 30)
    scene.svc.refresh(scene.key(2))
    old = _open(scene)[0]
    _psu(scene, 20)
    scene.svc.refresh(scene.key(2))           # replaces `old`
    current = _open(scene)[0]
    scene.db.reopen_conflict(old["id"])       # what a pre-U7 database holds ...
    scene.db.add_conflict(scene.key(2), PSU, current["old_rle"], current["new_rle"], 100)
    before = {op["id"] for op in _closings(scene)}

    scene.svc.refresh(scene.key(2), ignore_digest=True)

    new_rows = [op for op in _closings(scene) if op["id"] not in before]
    assert sorted(op["payload"]["reason"] for op in new_rows) == sorted([OVERTAKEN, DUPLICATE])
    assert [c["id"] for c in _open(scene)] == [current["id"]]


def test_reopening_an_agreement_over_a_stack_reopens_one(scene: Scene):
    """One agreement that closed four stacked conflicts of one row gives that row one back."""
    _psu(scene, 30)
    scene.svc.refresh(scene.key(2))
    first = _open(scene)[0]
    for y1 in (28, 26, 24):                   # a pre-U7 stack on the same row
        scene.db.add_conflict(scene.key(2), PSU, first["old_rle"],
                              masks.encode_rle(rect(10, 10, 50, y1)), 100)
    stack = sorted(c["id"] for c in _open(scene))
    _psu(scene, 50)                            # the row agrees again
    assert scene.svc.refresh(scene.key(2))["closed"] == 4
    op = _closings(scene, AGREE_AGAIN)[0]

    assert scene.svc.reopen_agreed(op["id"]) == [stack[-1]]     # the newest only
    assert _per_row(scene)[(2, PSU)] == 1


def test_only_an_agreement_is_reopened(scene: Scene):
    _psu(scene, 30)
    scene.svc.refresh(scene.key(2))
    cid = _open(scene)[0]["id"]
    with scene.db.transaction():
        scene.db.resolve_conflict(cid, "superseded")
        scene.svc._log_superseded(scene.key(2), [(cid, PSU, ROW_GONE)])
    op = _closings(scene, ROW_GONE)[0]

    with pytest.raises(ValueError):
        scene.svc.reopen_agreed(op["id"])
    assert _open(scene) == []
