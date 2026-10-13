"""Task U6, ruling D: when the inputs agree with a frozen row again, its conflict closes.

On D13/scan an override made at 21:24:33 and undone a minute later left
eighteen conflicts open on frames 39 and 40: the frozen rows agreed with the
inputs again, nothing closed a conflict on agreement, the two frames stayed red
and ``Space`` was refused with nothing for anybody to decide.  The scene here is
the same story in miniature: the screw sits on the PSU, an override puts the
PSU above it, the frozen frame disagrees, the override goes away.
"""
from __future__ import annotations

import pytest

from tda.core.db import Db
from tda.core.model import PairOverride, ShapePart
from tda.core import masks
from tda.core.truth import AGREE_AGAIN, CLOSE_KIND, REOPEN_KIND
from tda.core.truth_fresh import digest_of
from tda.core.truth_inputs import gather
from truth_scenes import DESKTOP, PSU, SCREW, VIEW, Scene, build_scene, rect, replace_parts

OVERRIDE = PairOverride(DESKTOP, VIEW, 1, PSU, SCREW)


@pytest.fixture
def db(tmp_db_path: str):
    d = Db(tmp_db_path)
    yield d
    d.close()


@pytest.fixture
def scene(db: Db) -> Scene:
    built = build_scene(db)
    # a screw big enough that losing it is beyond the re-tracing tolerance
    # (the scene's 8 x 8 one vanishes inside the 2 px band)
    replace_parts(built, built.screw_kf,
                  [ShapePart("main", masks.encode_rle(rect(14, 14, 34, 34)))])
    built.refresh_all()
    built.svc.verify_frame(built.key(2), "lin")
    return built


def _open(scene: Scene) -> list[dict]:
    return scene.db.conflicts(DESKTOP, VIEW, open_only=True)


def _queue_the_trial(scene: Scene) -> list[int]:
    """The override goes in and the frozen frame disagrees: conflicts queued."""
    scene.db.set_pair_override(OVERRIDE)
    out = scene.svc.refresh(scene.key(2))
    assert out["conflicts"] >= 1 and out["closed"] == 0
    ids = [c["id"] for c in _open(scene)]
    assert {c["instance"] for c in _open(scene)} >= {SCREW}
    return ids


def test_undoing_the_override_closes_every_conflict_it_raised(scene: Scene):
    ids = _queue_the_trial(scene)
    scene.db.delete_pair_override(OVERRIDE)

    out = scene.svc.refresh(scene.key(2))

    assert out["closed"] == len(ids) and out["conflicts"] == 0 and out["standing"] == 0
    assert _open(scene) == []
    for cid in ids:
        closed = scene.db.get_conflict(cid)
        assert (closed["status"], closed["resolution"]) == ("resolved", "superseded")
    # one op-log row for the frame, naming them all, with its inverse
    ops = [op for op in scene.db.ops(DESKTOP, VIEW, limit=10) if op["kind"] == CLOSE_KIND]
    assert len(ops) == 1
    payload, inverse = ops[0]["payload"], ops[0]["inverse"]
    assert payload["conflicts"] == ids and payload["step"] == 2
    assert payload["reason"] == AGREE_AGAIN == "inputs agree again"
    assert payload["resolution"] == "superseded"
    assert inverse == {"kind": REOPEN_KIND, "step": 2, "conflicts": ids}
    # the frame is a verified frame again, and Space takes it
    assert scene.review_status(2) == "verified"
    scene.svc.verify_frame(scene.key(2), "lin")
    # and nothing is left to do: the digest was stamped
    again = scene.svc.refresh(scene.key(2))
    assert again["closed"] == 0 and again["compiled"] is None


def test_the_closing_is_undoable_and_reopens_the_same_conflicts(scene: Scene):
    ids = _queue_the_trial(scene)
    scene.db.delete_pair_override(OVERRIDE)
    scene.svc.refresh(scene.key(2))
    op = next(op for op in scene.db.ops(DESKTOP, VIEW, limit=10) if op["kind"] == CLOSE_KIND)

    assert scene.svc.reopen_agreed(op["id"]) == ids
    assert sorted(c["id"] for c in _open(scene)) == ids
    for cid in ids:
        assert scene.db.get_conflict(cid)["resolution"] is None
    assert scene.db.ops(DESKTOP, VIEW, limit=1)[0]["kind"] == REOPEN_KIND
    # a human's resolution is not this function's to undo
    with pytest.raises(ValueError):
        scene.svc.reopen_agreed(scene.db.ops(DESKTOP, VIEW, limit=1)[0]["id"])
    # the inputs still agree, so the next refresh closes them once more
    assert scene.svc.refresh(scene.key(2))["closed"] == len(ids)


def test_reopening_twice_writes_nothing_the_second_time(scene: Scene):
    """Round 2: the second ``reopen_agreed`` of one op is a no-op -- no op row, no digest drop."""
    ids = _queue_the_trial(scene)
    scene.db.delete_pair_override(OVERRIDE)
    scene.svc.refresh(scene.key(2))
    op = next(op for op in scene.db.ops(DESKTOP, VIEW, limit=10) if op["kind"] == CLOSE_KIND)
    assert scene.svc.reopen_agreed(op["id"]) == ids

    key = scene.key(2)
    scene.svc._stamp(key, "a-digest-that-must-survive")
    rows = len(scene.db.ops(DESKTOP, VIEW, limit=10_000))

    assert scene.svc.reopen_agreed(op["id"]) == []
    assert len(scene.db.ops(DESKTOP, VIEW, limit=10_000)) == rows
    assert scene.db.frame_digest(key)["digest"] == "a-digest-that-must-survive"
    assert sorted(c["id"] for c in _open(scene)) == ids


def test_a_conflict_a_human_settled_since_is_not_reopened(scene: Scene):
    """Only a supersession is taken back: ``keep_old`` wrote an override with it."""
    ids = _queue_the_trial(scene)
    scene.db.delete_pair_override(OVERRIDE)
    scene.svc.refresh(scene.key(2))
    op = next(op for op in scene.db.ops(DESKTOP, VIEW, limit=10) if op["kind"] == CLOSE_KIND)
    scene.svc.reopen_agreed(op["id"])
    scene.svc.resolve_conflict(ids[0], "keep_old", "lin")    # a human answers one

    assert ids[0] not in scene.svc.reopen_agreed(op["id"])
    assert scene.db.get_conflict(ids[0])["resolution"] == "keep_old"


def test_the_passes_over_a_view_total_the_closed_conflicts(scene: Scene):
    """Round 2: ``refresh_range`` and ``ensure_fresh`` sum ``closed`` for their callers."""
    ids = _queue_the_trial(scene)
    scene.db.delete_pair_override(OVERRIDE)
    assert scene.svc.refresh_range(DESKTOP, VIEW, [1, 2, 3])["closed"] == len(ids)

    op = next(op for op in scene.db.ops(DESKTOP, VIEW, limit=10) if op["kind"] == CLOSE_KIND)
    scene.svc.reopen_agreed(op["id"])
    fresh = scene.svc.ensure_fresh(DESKTOP, VIEW)
    assert fresh["closed"] == len(ids) and fresh["open_conflicts"] == 0


def test_check_and_the_exports_print_how_many_closed(scene: Scene, capsys):
    from types import SimpleNamespace

    from tda import cli_app

    ids = _queue_the_trial(scene)
    scene.db.delete_pair_override(OVERRIDE)
    cli_app._check_one(scene.db, scene.tax, SimpleNamespace(limit=5), DESKTOP, VIEW)
    out = capsys.readouterr().out
    assert (f"[check] {len(ids)} conflict(s) closed: the inputs agree with the frozen "
            f"rows again") in out
    assert "open conflicts: 0" in out

    op = next(op for op in scene.db.ops(DESKTOP, VIEW, limit=10) if op["kind"] == CLOSE_KIND)
    scene.svc.reopen_agreed(op["id"])
    code, _truth = cli_app._export_prologue(
        SimpleNamespace(refresh=True, allow_conflicts=False), scene.db, [DESKTOP], VIEW,
        "export-coco")
    out = capsys.readouterr().out
    assert code is None, out                                  # nothing left to refuse on
    assert (f"[export-coco] {len(ids)} conflict(s) closed: the inputs agree with the "
            f"frozen rows again") in out


def test_a_current_digest_does_not_hide_an_open_conflict(scene: Scene):
    """The live DB's state: an agreeing refresh had already stamped the frame."""
    ids = _queue_the_trial(scene)
    scene.db.delete_pair_override(OVERRIDE)
    key = scene.key(2)
    # what an agreeing refresh before U6 left behind: the rows describe the
    # inputs, the digest says so, and the conflicts are still open
    scene.svc._stamp(key, digest_of(gather(scene.db, scene.tax, key, None, None),
                                    scene.svc.compiler_version))
    assert scene.svc._digest_is_current(
        key, digest_of(gather(scene.db, scene.tax, key, None, None),
                       scene.svc.compiler_version))

    assert scene.svc.refresh(key)["closed"] == len(ids)
    assert _open(scene) == []


def test_only_the_conflicts_that_agree_close(scene: Scene):
    """The screw is re-traced meanwhile -- a blob added off the PSU: its disagreement stands.

    The PSU's pixels under the screw are the ones it had when it was frozen, so
    the PSU agrees again and only its conflict closes.
    """
    _queue_the_trial(scene)
    scene.db.delete_pair_override(OVERRIDE)
    grown = rect(14, 14, 34, 34) | rect(52, 52, 62, 62)
    replace_parts(scene, scene.screw_kf, [ShapePart("main", masks.encode_rle(grown))])

    out = scene.svc.refresh(scene.key(2))

    still = _open(scene)
    assert {c["instance"] for c in still} == {SCREW}
    assert out["standing"] >= 1
    agreed = [op for op in scene.db.ops(DESKTOP, VIEW, limit=10)
              if op["kind"] == CLOSE_KIND and op["payload"]["reason"] == AGREE_AGAIN]
    assert agreed and all(SCREW not in op["payload"]["instances"] for op in agreed)
    # U7: the screw's disagreement is a newer one, and it replaced the old one
    # instead of standing next to it -- one open conflict per frozen row
    assert len(still) == 1 and out["overtaken"] == 1
