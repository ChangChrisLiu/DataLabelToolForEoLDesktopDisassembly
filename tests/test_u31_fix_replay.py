"""U31-fix: every change ``scripts/fix_u31_zorder.py`` makes is an op with an inverse,
and ``import-logs --force`` leaves it in place.

Scene: D23's real sheet (``tests/fixtures/logs/desktop_23.csv``, an HP Compaq 8100 SFF)
imported as S1 does, the shroud taken out with ``Shift+T`` and the cable clip put on it
through Steps -- the live state after the 2026-10-24 correction -- plus a small synthetic
scanner view (frames 1-12, every part a rectangle of its own) confirmed frame by frame.
Two parts overlap the way U30-QA found them on the HP desktops: the optical drive cage is
drawn *under* the front-panel connector it covers on frames 1-8, and the clip sits on the
heatsink's corner.

The script's own functions are driven:

* ``apply_zorder`` -- one ``set_zorder`` op per gesture, payload and inverse the two orders;
  the inverse written back restores the order (Ctrl+Z), the payload redoes it;
* ``steps_panel_relation`` -- Steps' instance table + Apply: U27 ``relation_fix`` ops;
* ``fix_clip`` -- the clip's frame-2 shape committed on the last frame it is in: a
  ``commit_keyframe`` op whose inverse removes the keyframe again; a clip no longer on the
  shroud is never ruled again;
* ``settle`` -- the conflicts the changes opened accepted (``accept_new``); the frames stay
  confirmed, except where the clip appears (a confirmed frame that gains a part is demoted);
  a conflict the user had open stays open, also when the app supersedes it with a new id.

Then ``import-logs --force``: the layer order (``zorder`` is not rebuilt by an import: it is
not replayed, it is simply never touched), the clip's keyframe, its relation fix (replayed,
U27) and the confirmed rows all come out as they went in, and a refresh finds nothing to
dispute.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from relation_scenes import CLIP, D23, SHROUD, fixes, imported, reimport
from tda.core import masks
from tda.core.model import FrameKey, ShapeKeyframe, ShapePart, ZOrderRec
from tda.core.states import needs_geom
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import instances_of, state_of
from tda.ui.session_ops import apply_zorder as undo_zorder
from tda.ui.session_verb import commit_verb_fix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fix_u31_zorder as U31  # noqa: E402

VIEW = "scan"
HW = (200, 200)
STEPS = range(1, 13)
CAGE, FP1 = "drive_cage.01", "connector.front_panel.01"
HEATSINK = "cpu_cooler.heatsink.01"
CAGE_MOVES = 9                 # "Open Optical drive": displace drive_cage.01
HEATSINK_LEAVES = 8            # "Heatsink": remove cpu_cooler.heatsink.01


def rect(x0, y0, x1, y1) -> np.ndarray:
    m = np.zeros(HW, bool)
    m[y0:y1, x0:x1] = True
    return m


def cell(i: int) -> np.ndarray:
    col, row = i % 16, i // 16
    return rect(col * 12 + 1, row * 12 + 1, col * 12 + 11, row * 12 + 11)


def kf(inst: str, mask: np.ndarray, anchor: int) -> ShapeKeyframe:
    return ShapeKeyframe(id=None, instance=inst, desktop=D23, view=VIEW, pose_segment=1,
                         anchor_step=anchor, placement="in_chassis", geom_type="mask",
                         parts=[ShapePart("main", masks.encode_rle(mask))])


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def scene(qapp, tmp_path, tax):
    db = imported(tmp_path, tax, D23)
    truth = TruthService(db, tax, cache_dir=None)
    commit_verb_fix(db, truth, FrameKey(D23, 3, VIEW), SHROUD, "remove", "chang")
    U31.steps_panel_relation(db, tax, D23, CLIP, SHROUD, True, str(tmp_path / "cache"))
    db.set_pose_segment(D23, VIEW, 1, 1, max(STEPS), max(STEPS), None, None)
    for step in STEPS:
        db.upsert_frame(FrameKey(D23, step, VIEW), f"s{step:03d}.png", {"hw": list(HW)}, None)
    # every part that needs a mask on these frames: one rectangle of its own, anchored on
    # the last frame it needs it (annotated backwards, as the app does)
    insts = instances_of(db, D23)
    last: dict[str, int] = {}
    for step in STEPS:
        for k, kind in needs_geom(insts, state_of(db, tax, D23, step), tax).items():
            if kind == "mask":
                last[k] = step
    order = []
    for i, k in enumerate(sorted(last)):
        if k in (CAGE, FP1, CLIP, HEATSINK):
            continue
        db.add_keyframe(kf(k, cell(i), last[k]))
        order.append((k, "main"))
    # the cage over the connector before it is opened, beside it after
    db.add_keyframe(kf(CAGE, rect(80, 140, 150, 190), CAGE_MOVES - 1))
    db.add_keyframe(kf(CAGE, rect(152, 140, 198, 190), last[CAGE]))
    db.add_keyframe(kf(FP1, rect(100, 150, 120, 170), last[FP1]))
    # the heatsink, and the clip on its corner (frames 1-2)
    db.add_keyframe(kf(HEATSINK, rect(10, 120, 70, 190), last[HEATSINK]))
    db.add_keyframe(kf(CLIP, rect(60, 110, 76, 130), 2))
    floor = [e for e in order if e[0] in ("chassis", "motherboard.01")]
    rest = [e for e in order if e not in floor]
    # as U30-QA found it: the cage low, the connector it covers above it
    db.set_zorder(ZOrderRec(D23, VIEW, 1, floor + [(CAGE, "main")] + rest
                            + [(FP1, "main"), (HEATSINK, "main"), (CLIP, "main")]))
    truth.refresh_range(D23, VIEW, STEPS, ignore_digest=True)
    for step in STEPS:
        truth.verify_frame(FrameKey(D23, step, VIEW), "chang")
    yield db, truth
    db.close()


def verified(db) -> list[int]:
    return [int(r[0]) for r in db.conn.execute(
        "SELECT step FROM frame WHERE desktop=? AND view=? AND review_status='verified' "
        "ORDER BY step", (D23, VIEW))]


def visible_px(db, step: int, inst: str) -> int:
    row = db.compiled(FrameKey(D23, step, VIEW)).get(inst)
    return 0 if not row or not row.get("visible_rle") else int(
        masks.decode_rle(row["visible_rle"]).sum())


def truth_rows(db) -> dict:
    """What an export ships: every row's pixels, label and status."""
    return {(s, k): (str(r.get("visible_rle")), r.get("visibility"), r.get("status"))
            for s in STEPS for k, r in db.compiled(FrameKey(D23, s, VIEW)).items()}


def last_op(db, kind: str) -> dict:
    return db.ops_of_kind(D23, kind)[-1]


def test_the_cage_raise_is_one_set_zorder_op_with_its_inverse(scene, tax):
    db, truth = scene
    assert verified(db) == list(STEPS)
    assert visible_px(db, 1, FP1) == 400            # drawn over the cage
    desk = U31.Desk(db, tax, truth, D23)
    group = U31.Group(D23, "cage", [CAGE], [FP1], qa_last=CAGE_MOVES - 1)
    plan = U31.plan_desktop(desk, [group], lambda *_: None)
    assert [m[:2] for m in plan["moves"]] == [(CAGE, FP1)]          # one raise, no collateral
    before = desk.order(1)
    open0 = U31.open_conflicts(db, D23)
    assert U31.apply_zorder(db, truth, desk, plan, lambda *_: None) == 1
    op = last_op(db, "set_zorder")
    assert [tuple(e) for e in op["inverse"]["order"]] == before
    assert [tuple(e) for e in op["payload"]["order"]] == plan["order1"]
    predicted = {(s, k) for item in plan["groups"] for k, v in item["cand"]["sim"].items()
                 for s in v}
    out = U31.settle(db, truth, desk, open0, predicted, lambda *_: None)
    assert sorted(out["accepted"]) == sorted([(s, FP1) for s in range(1, CAGE_MOVES)]
                                             + [(s, CAGE) for s in range(1, CAGE_MOVES)])
    assert not out["unexpected"] and not out["left_mine"]
    assert verified(db) == list(STEPS)                # accept_new keeps them confirmed
    assert visible_px(db, 1, FP1) == 0 and visible_px(db, CAGE_MOVES, FP1) == 400
    # Ctrl+Z is the inverse written back; the payload redoes it
    undo_zorder(db, truth, op["inverse"])
    assert desk.order(1) == before
    undo_zorder(db, truth, op["payload"])
    assert desk.order(1) == plan["order1"]


def test_the_clip_moves_to_the_heatsink_through_steps_and_gets_its_shape(scene, tax):
    db, truth = scene
    desk = U31.Desk(db, tax, truth, D23)
    open0 = U31.open_conflicts(db, D23)
    m = {"before": 2}
    done = U31.fix_clip(db, tax, truth, desk, m,
                        {"action": "reparent", "parent": HEATSINK}, None, lambda *_: None)
    assert done["relation"] == {"from": [SHROUD, True], "to": [HEATSINK, True]}
    assert fixes(db, D23)[-1:] == [(CLIP, "parent", SHROUD, HEATSINK, "controller")]
    assert U31.clip_frames(db, tax, D23, list(STEPS)) == list(range(1, HEATSINK_LEAVES))
    assert done["shape"]["anchor"] == HEATSINK_LEAVES - 1
    op = last_op(db, "commit_keyframe")
    new = op["payload"]["keyframes"][0]
    assert (new["exists"], new["instance"], new["anchor_step"]) == \
        (True, CLIP, HEATSINK_LEAVES - 1)
    assert op["inverse"]["keyframes"][0]["exists"] is False      # the inverse removes it
    assert op["payload"]["copy_of_frame"] == 2                   # where the pixels came from
    predicted = U31.clip_touches(desk, done["shape"]["frames"])
    out = U31.settle(db, truth, desk, open0, predicted, lambda *_: None)
    assert not out["unexpected"] and not out["left_mine"]
    assert {k for _s, k in out["accepted"]} <= {HEATSINK, CLIP}
    # frames 3-7 gained a part: the app demotes them, the rest stay confirmed
    assert verified(db) == [1, 2] + list(range(HEATSINK_LEAVES, max(STEPS) + 1))
    assert visible_px(db, 3, CLIP) == 320
    # a re-run never re-rules it, whatever a re-measure says: its parent is no longer the
    # shroud (U31 review N4)
    n_ops = int(db.conn.execute("SELECT COUNT(*) FROM op_log").fetchone()[0])
    again = U31.fix_clip(db, tax, truth, desk, {"before": HEATSINK_LEAVES - 1},
                         {"action": "unparent", "parent": None}, None, lambda *_: None)
    assert again["relation"] is None and again["shape"] is None
    assert "not re-ruled" in again["skipped"]
    assert int(db.conn.execute("SELECT COUNT(*) FROM op_log").fetchone()[0]) == n_ops
    assert (db.instances(D23)[CLIP].parent, db.instances(D23)[CLIP].attached) == \
        (HEATSINK, True)


def test_a_conflict_the_user_left_open_stays_open_when_a_move_supersedes_it(scene, tax):
    """U31 review N2: the user's open questions are kept by their (frame, part) keys.

    The raise moves the connector's row on frames 1-8, so the app closes the user's
    conflict there as superseded and opens a new one (one open conflict per row): a new id
    on a key the plan predicted -- which an exclusion by id alone would have accepted.
    """
    db, truth = scene
    # the user's question: the connector redrawn, grown above the cage, after the frames
    # were confirmed -- a conflict on every frame it is drawn on, left open
    shape = db.keyframes(D23, VIEW, FP1)[0]
    shape.parts = [ShapePart("main", masks.encode_rle(rect(100, 150, 120, 170)
                                                      | rect(100, 120, 120, 140)))]
    db.update_keyframe(shape)
    truth.refresh_range(D23, VIEW, STEPS, ignore_digest=True)
    users = U31.open_conflicts(db, D23)
    keys = set(users.values())
    assert {k for _s, k in keys} == {FP1} and {s for s, _k in keys} >= set(range(1, CAGE_MOVES))
    desk = U31.Desk(db, tax, truth, D23)
    group = U31.Group(D23, "cage", [CAGE], [FP1], qa_last=CAGE_MOVES - 1)
    plan = U31.plan_desktop(desk, [group], lambda *_: None)
    assert U31.apply_zorder(db, truth, desk, plan, lambda *_: None) == 1
    predicted = {(s, k) for item in plan["groups"] for k, v in item["cand"]["sim"].items()
                 for s in v}
    assert {(s, FP1) for s in range(1, CAGE_MOVES)} <= predicted
    out = U31.settle(db, truth, desk, users, predicted, lambda *_: None)
    assert sorted(out["accepted"]) == [(s, CAGE) for s in range(1, CAGE_MOVES)]
    assert not out["unexpected"] and not out["left_mine"]
    still = U31.open_conflicts(db, D23)
    assert set(still.values()) == keys                         # every one of them open ...
    assert out["users_open"]["reopened"] == [(s, FP1) for s in range(1, CAGE_MOVES)]
    assert {i for i, k in still.items() if k[0] < CAGE_MOVES}.isdisjoint(users)  # ... new ids


def test_a_forced_re_import_keeps_every_change(scene, tax, tmp_path):
    db, truth = scene
    test_the_cage_raise_is_one_set_zorder_op_with_its_inverse(scene, tax)
    test_the_clip_moves_to_the_heatsink_through_steps_and_gets_its_shape(scene, tax)
    order = db.zorder(D23, VIEW, 1)
    kfs = sorted((k.id, k.instance, k.anchor_step) for k in db.keyframes(D23, VIEW))
    rows = truth_rows(db)
    status = verified(db)
    clip = db.instances(D23)[CLIP]
    reimport(db, tmp_path, tax, D23)
    assert db.zorder(D23, VIEW, 1) == order                     # untouched, not replayed
    assert sorted((k.id, k.instance, k.anchor_step) for k in db.keyframes(D23, VIEW)) == kfs
    again = db.instances(D23)[CLIP]
    assert (again.parent, again.attached) == (clip.parent, clip.attached) == (HEATSINK, True)
    assert U31.clip_frames(db, tax, D23, list(STEPS)) == list(range(1, HEATSINK_LEAVES))
    assert verified(db) == status
    open_before = len(db.conflicts(D23, VIEW))
    res = truth.refresh_range(D23, VIEW, STEPS, ignore_digest=True)
    assert res["conflicts"] == 0 and len(db.conflicts(D23, VIEW)) == open_before
    assert truth_rows(db) == rows
