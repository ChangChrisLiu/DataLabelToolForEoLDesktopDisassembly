"""Task U36, the rule without Qt: runs, the views side by side, the suggestion, the sync plan.

The user (2026-10-26): some 「看不见」 marks last many frames, and the same
part may be hidden in one view and seen in another -- the visibility review
takes both into account. Pinned here on hand-made rows: where a run starts and
ends (the label, the hand label, the reason, a frame with no image, a frame
the part is not on, a cut), what it carries (its frame count, 「全程」, the
frames where the parts hiding it change), the queue's order, a part's 看不见
stretches per view, what every view says on a run's steps, the reason they
point at -- both ways -- and which frames of the other views a verdict may go to.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tda.core import vis_review as VR
from tda.core import vis_runs as RU
from tda.core.model import VIEWS

FULL, PARTIAL = "occluded_full", "occluded_partial"


def row(step, part="a.01", vis=FULL, hand=None, occluder=None, **extra) -> dict:
    return {"step": step, "instance": part, "visibility": vis, "hand_visibility": hand,
            "occluder": occluder, "placement": "in_chassis", "input_hash": f"h{step}",
            **extra}


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
def test_a_run_ends_where_the_label_the_hand_label_or_the_reason_changes():
    rows = [row(1), row(2), row(3, vis=PARTIAL), row(4, vis=PARTIAL),
            row(5, hand=FULL, occluder="cable"), row(6, hand=FULL, occluder="cable"),
            row(7, hand=FULL, occluder="viewpoint"), row(8, hand=FULL)]
    runs = VR.runs_of(rows, list(range(1, 9)))
    assert [(r.steps, r.visibility, r.manual, r.occluder) for r in runs] == [
        ([1, 2], FULL, False, None), ([3, 4], PARTIAL, False, None),
        ([5, 6], FULL, True, "cable"), ([7], FULL, True, "viewpoint"), ([8], FULL, True, None)]


def test_a_frame_with_no_image_does_not_break_a_run_a_frame_without_the_part_does():
    """Step 3 has no image (not in the annotatable order): 2 and 4 are neighbours.
    The part is not on 6 (removed for a while -- no row there): 5 and 7 are
    two runs."""
    rows = [row(s) for s in (1, 2, 4, 5, 7, 8)]
    runs = VR.runs_of(rows, [1, 2, 4, 5, 6, 7, 8])
    assert [r.steps for r in runs] == [[1, 2, 4, 5], [7, 8]]
    entry = runs[0].entry()
    assert (entry["a"], entry["b"], entry["n"]) == (1, 5, 4)     # 第 1–5 帧（4 帧）


def test_the_parts_hiding_it_are_collected_and_their_changes_said_not_cut():
    rows = [row(s) for s in range(1, 7)]
    names = {(1, "a.01"): ("cage.01",), (2, "a.01"): ("cage.01",), (3, "a.01"): ("cage.01",),
             (4, "a.01"): ("psu.01",), (5, "a.01"): ("psu.01",), (6, "a.01"): ("psu.01",)}
    [run] = VR.runs_of(rows, list(range(1, 7)), names)
    assert run.steps == [1, 2, 3, 4, 5, 6]
    assert run.occluded_by == ("cage.01", "psu.01") and run.changes == [4]
    # a cover taken off while another still hides it is no new reason
    names = {(1, "a.01"): ("cage.01", "fan.01"), (2, "a.01"): ("cage.01",),
             (3, "a.01"): ("cage.01", "psu.01"), (4, "a.01"): ("psu.01",),
             (5, "a.01"): (), (6, "a.01"): ("psu.01",)}
    [run] = VR.runs_of(rows, list(range(1, 7)), names)
    assert run.changes == [5, 6]


def test_a_cut_starts_a_run_of_its_own():
    rows = [row(s) for s in range(1, 9)]
    runs = VR.runs_of(rows, list(range(1, 9)), splits={("a.01", 5)})
    assert [r.steps for r in runs] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    # a cut of another part, or at a frame that already starts a run, changes nothing
    assert [r.steps for r in VR.runs_of(rows, list(range(1, 9)),
                                        splits={("b.01", 5), ("a.01", 1)})] == [
        list(range(1, 9))]


def test_reviewed_and_unreviewed_frames_are_runs_apart():
    rows = [row(1, reviewed=True), row(2, reviewed=True), row(3), row(4)]
    runs = VR.runs_of(rows, [1, 2, 3, 4])
    assert [(r.steps, r.reviewed) for r in runs] == [([1, 2], True), ([3, 4], False)]


def test_whole_is_every_confirmed_frame_of_the_part():
    rows = [row(s) for s in (2, 3, 4)] + [row(s, part="b.01") for s in (2, 3)]
    runs = VR.runs_of(rows, [1, 2, 3, 4])
    VR.mark_whole(runs, {"a.01": [2, 3, 4], "b.01": [1, 2, 3]}, [1, 2, 3, 4])
    assert [(r.instance, r.whole) for r in runs] == [("a.01", True), ("b.01", False)]
    # a frame of an ignore step (not in the order) does not count
    VR.mark_whole(runs, {"a.01": [2, 3, 4, 9]}, [1, 2, 3, 4])
    assert runs[0].whole


def test_the_queue_puts_the_part_with_the_longest_run_first():
    rows = ([row(s, part="a.01") for s in (1, 2)] + [row(s, part="a.01") for s in (5,)]
            + [row(s, part="b.01") for s in range(1, 9)] + [row(3, part="c.01")])
    runs = VR.order_runs(VR.runs_of(rows, list(range(1, 9))))
    assert [(r.instance, r.steps[0], len(r.steps)) for r in runs] == [
        ("b.01", 1, 8), ("a.01", 1, 2), ("a.01", 5, 1), ("c.01", 3, 1)]


# --------------------------------------------------------------------------- #
# the views side by side
# --------------------------------------------------------------------------- #
def side(view: str, rows: list[dict], confirmed=None, order=None, state=RU.STATE_OK) -> RU.Side:
    parts: dict = {}
    for r in rows:
        parts.setdefault(r["instance"], {})[r["step"]] = r
    steps = sorted({r["step"] for r in rows}) if confirmed is None else confirmed
    return RU.Side(view, state, parts, frozenset(steps), order or list(range(1, 20)))


def test_hidden_stretches_count_frames_and_stretches_whatever_the_reason():
    s = side("oak2", [row(s, vis=FULL if s not in (5, 6) else "visible") for s in range(1, 11)]
             + [row(11, vis="out_of_view", hand="out_of_view")])
    assert RU.hidden_stretches(s, "a.01") == (9, 2, 5)          # 1-4, 7-11
    assert RU.hidden_stretches(s, "z.01") == (0, 0, 0)
    assert RU.hidden_stretches(RU.Side("rs", RU.STATE_NONE), "a.01") is None
    assert RU.hidden_stretches(RU.Side("rs", RU.STATE_UNCONFIRMED), "a.01") is None


def entry_of(steps, vis=FULL, occluded_by=(), occluder=None, manual=False, part="a.01"):
    return {"instance": part, "steps": list(steps), "visibility": vis,
            "occluded_by": list(occluded_by), "occluder": occluder, "manual": manual}


def test_cross_view_says_what_every_view_has_on_the_same_steps():
    sides = {
        "scan": side("scan", [row(s) for s in range(1, 6)]),
        "oak1": side("oak1", [row(s, vis="visible") for s in (1, 2, 3)]
                     + [row(4, vis=FULL, hand=FULL, occluder="cable")],
                     confirmed=[1, 2, 3, 4]),
        "rs": RU.Side("rs", RU.STATE_UNCONFIRMED),
    }
    sides["oak1"].confirmed = frozenset({1, 2, 3, 4, 5})          # 5: confirmed, part absent
    table = RU.cross_view(entry_of(range(1, 6), occluded_by=["cover.01"]), sides, "scan")
    assert [t["view"] for t in table] == list(VIEWS)
    scan, oak1, oak2, rs = table
    assert (scan["here"], scan["seen"], scan["hidden"], scan["by"]) == (True, 0, 5, ["cover.01"])
    assert (oak1["seen"], oak1["hidden"], oak1["absent"], oak1["reasons"]) == (
        3, 1, 1, {"cable": 1})
    assert oak1["highlight"] and not scan["highlight"]
    assert oak2["state"] == RU.STATE_NONE and rs["state"] == RU.STATE_UNCONFIRMED
    assert not oak2["highlight"] and not rs["highlight"]


def test_cross_view_names_the_parts_over_it_in_the_other_view_on_a_sample():
    asked: list[tuple] = []

    def names(view, step, part, found):
        asked.append((view, step))
        return ("cage.01",) if step < 6 else ("psu.01",)

    sides = {"scan": side("scan", [row(s, vis="visible") for s in range(1, 11)]),
             "oak1": side("oak1", [row(s) for s in range(1, 11)])}
    [_scan, oak1, _o2, _rs] = RU.cross_view(entry_of(range(1, 11), vis=PARTIAL), sides, "scan",
                                            names)
    assert asked == [("oak1", 1), ("oak1", 6), ("oak1", 10)]   # first, middle, last
    assert oak1["by"] == ["cage.01", "psu.01"]


# --------------------------------------------------------------------------- #
# the suggestion, both ways
# --------------------------------------------------------------------------- #
def views_for(here: str, **counts) -> list[dict]:
    out = []
    for view in VIEWS:
        given = counts.get(view)
        base = {"view": view, "here": view == here, "state": RU.STATE_NONE, "seen": 0,
                "hidden": 0, "partial": 0}
        if given is not None:
            base.update(state=RU.STATE_OK, **given)
        out.append(base)
    return out


def test_hidden_here_seen_there_nothing_over_it_is_the_camera_angle():
    views = views_for("scan", scan={"hidden": 5}, oak1={"seen": 5})
    found = RU.suggest(entry_of(range(5), manual=True), views, "scan")
    assert (found["kind"], found["verdict"], found["key"], found["views"]) == (
        "viewpoint", VR.VERDICT_VIEWPOINT, "Z", ["oak1"])
    # the other way: reviewing oak2, the scanner sees it
    views = views_for("oak2", scan={"seen": 3}, oak2={"hidden": 3})
    assert RU.suggest(entry_of(range(3)), views, "oak2")["action"] == "vis_review_viewpoint"
    # 部分: the other view has to see more of it than a partial occlusion
    views = views_for("scan", scan={"seen": 4, "partial": 4}, oak1={"seen": 4, "partial": 4})
    assert RU.suggest(entry_of(range(4), vis=PARTIAL), views, "scan") is None
    views = views_for("scan", scan={"seen": 4, "partial": 4}, oak1={"seen": 4, "partial": 1})
    found = RU.suggest(entry_of(range(4), vis=PARTIAL), views, "scan")
    assert (found["action"], found["key"]) == ("vis_review_viewpoint_partial", "Shift+Z")


def test_something_over_it_and_the_scanner_blind_too_is_hidden():
    # here is oak1: a part over it, and the scanner does not see it either
    views = views_for("oak1", scan={"hidden": 5}, oak1={"hidden": 5})
    found = RU.suggest(entry_of(range(5), occluded_by=["cover.01"]), views, "oak1")
    assert (found["kind"], found["verdict"], found["key"], found["scan_too"]) == (
        "blocked", VR.VERDICT_COMPUTED, "Space", True)
    # the cable as the reason given: the cable key
    found = RU.suggest(entry_of(range(5), occluder="cable", manual=True), views, "oak1")
    assert (found["verdict"], found["action"]) == (VR.VERDICT_CABLE, "vis_review_cable")
    # on the scanner itself, the scanner is "blind" by definition
    views = views_for("scan", scan={"hidden": 5}, oak1={"seen": 5})
    found = RU.suggest(entry_of(range(5), occluded_by=["cover.01"]), views, "scan")
    assert found["verdict"] == VR.VERDICT_COMPUTED and not found["scan_too"]
    # something over it here, but the scanner sees it: the views disagree, no default
    views = views_for("oak1", scan={"seen": 5}, oak1={"hidden": 5})
    assert RU.suggest(entry_of(range(5), occluded_by=["cover.01"]), views, "oak1") is None


def test_no_suggestion_without_another_view_or_for_other_labels():
    views = views_for("scan", scan={"hidden": 5})
    assert RU.suggest(entry_of(range(5)), views, "scan") is None       # nothing to compare
    views = views_for("scan", scan={"seen": 5}, oak1={"seen": 5})
    assert RU.suggest(entry_of(range(5), vis="too_small"), views, "scan") is None
    assert RU.suggest(entry_of(range(5), occluder="viewpoint"), views, "scan") is None


# --------------------------------------------------------------------------- #
# the sync plan
# --------------------------------------------------------------------------- #
class FakeDb:
    """What sync_plan reads: stamps and labels here, the sides of the other views."""

    def __init__(self, here: dict, others: dict, frames=()):
        self.here, self.others, self.frames = here, others, set(frames)

    def compiled_label(self, key, instance):
        found = self.here.get(key.step)
        return None if found is None else {"visibility": found["visibility"]}

    def instance_flags(self, key, instance):
        return self.here.get(key.step, {}).get("flags")

    def frame_overrides(self, key):
        found = self.here.get(key.step, {})
        if found.get("occluder") is None:
            return {}
        return {"a.01": SimpleNamespace(occluder=found["occluder"])}

    def view_annotation(self, desktop, view):
        found = self.others.get(view)
        return (1, sorted(found["confirmed"])) if found else (0, [])

    def vis_view_rows(self, desktop, view):
        return list(self.others[view]["rows"])

    def get_frame(self, key):
        return {"step": key.step} if (key.view, key.step) in self.frames else None


def stamp(verdict, vis, occluder=None):
    return {"vis_reviewed": True, "by": "t", "at": "x", "verdict": verdict,
            "visibility": vis, "occluder": occluder}


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(RU, "order_of", lambda db, d, v: list(range(1, 20)))


def hand_row(step, **kwargs) -> dict:
    """A row of a frame with an override row (``has_override``) for the part."""
    return row(step, has_override=True, **kwargs)


def test_the_sync_goes_only_where_nobody_answered_or_wrote_anything(patched):
    """Round 2 (B1): a person's override of any kind there is never touched."""
    here = {s: {"visibility": FULL, "occluder": "viewpoint",
                "flags": stamp("viewpoint", FULL, "viewpoint")} for s in range(1, 9)}
    oak1 = [
        row(1),                                                   # computed 完全, free: goes
        hand_row(2, hand=FULL, occluder="cable"),                 # a cable mark: theirs
        hand_row(3, flags=stamp("cable", FULL, "cable"), hand=FULL, occluder="cable"),  # verdict
        row(4, vis="visible"),                                    # seen there
        row(5, vis=PARTIAL),                                      # hidden differently
        row(6),                                                   # a part over it there
        hand_row(8, hand=FULL, occluder="viewpoint"),             # the very same reason, by hand
    ]
    for r in oak1:
        r.setdefault("flags", {})
    db = FakeDb(here, {"oak1": {"rows": oak1, "confirmed": {1, 2, 3, 4, 5, 6, 8}}},
                frames={("oak1", 7)})
    plan = RU.sync_plan(db, 13, "scan", "a.01", range(1, 9), shape_why=lambda k, p: "",
                        covered=lambda view, step, part, r: step == 6)
    assert sorted(plan["views"]["oak1"]) == [1]
    assert plan["count"] == 1 and "replaced" not in plan
    assert plan["skipped"] == {"verdict": 1, "different": 2, "manual": 2, "covered": 1,
                               "unconfirmed": 1, "absent": 0, "no_shape": 0}


def test_a_mask_drawn_by_hand_alone_is_a_persons_work_too(patched):
    """Only a single-frame mask, no label and no reason (4 px, stored 完全): left alone."""
    here = {1: {"visibility": FULL, "flags": stamp("computed", FULL)}}
    oak1 = [hand_row(1, flags={})]                     # hand None, occluder None
    db = FakeDb(here, {"oak1": {"rows": oak1, "confirmed": {1}}})
    plan = RU.sync_plan(db, 13, "scan", "a.01", [1])
    assert plan["count"] == 0 and plan["skipped"]["manual"] == 1


def test_the_sync_refuses_without_a_verdict_or_for_a_label_that_is_not_hidden(patched):
    unanswered = {1: {"visibility": FULL, "flags": None}}
    assert RU.sync_plan(FakeDb(unanswered, {}), 13, "scan", "a.01", [1])["refused"] == \
        "no_verdict"
    # a stamp the label moved away from answers for nothing
    moved = {1: {"visibility": PARTIAL, "flags": stamp("computed", FULL)}}
    assert RU.sync_plan(FakeDb(moved, {}), 13, "scan", "a.01", [1])["refused"] == "no_verdict"
    seen = {1: {"visibility": "visible", "flags": stamp("label", "visible")}}
    assert RU.sync_plan(FakeDb(seen, {}), 13, "scan", "a.01", [1])["refused"] == "not_hidden"


def test_a_computed_verdict_goes_where_the_label_is_computed_whatever_covers_it(patched):
    here = {1: {"visibility": FULL, "flags": stamp("computed", FULL)},
            2: {"visibility": FULL, "flags": stamp("computed", FULL)}}
    oak1 = [row(1, flags={}), hand_row(2, hand=FULL, occluder="cable", flags={})]
    db = FakeDb(here, {"oak1": {"rows": oak1, "confirmed": {1, 2}}})
    plan = RU.sync_plan(db, 13, "scan", "a.01", [1, 2], covered=lambda *a: True)
    assert sorted(plan["views"]["oak1"]) == [1]          # a part over it: computed is right
    assert plan["skipped"]["manual"] == 1 and plan["skipped"]["covered"] == 0


def test_a_hand_label_goes_over_as_the_computed_stamp(patched):
    """Round 2 (N1): no hand label is written there -- its computed label is
    already the one, and a hand label would stop following its redraws."""
    here = {1: {"visibility": FULL, "flags": stamp("label", FULL)}}
    db = FakeDb(here, {"oak1": {"rows": [row(1, flags={})], "confirmed": {1}}})
    plan = RU.sync_plan(db, 13, "scan", "a.01", [1])
    assert plan["views"]["oak1"][1] == {"verdict": "computed", "visibility": FULL,
                                        "occluder": None}


def test_a_reason_needs_a_shape_and_nothing_over_it_there(patched):
    here = {s: {"visibility": FULL, "occluder": "cable", "flags": stamp("cable", FULL, "cable")}
            for s in (1, 2, 3, 4)}
    rows = [row(s, flags={}) for s in (1, 2, 3, 4)]
    db = FakeDb(here, {"oak1": {"rows": rows, "confirmed": {1, 2, 3, 4}}})
    why = {1: "no_shape", 2: "removed", 3: "bench", 4: ""}
    plan = RU.sync_plan(db, 13, "scan", "a.01", [1, 2, 3, 4],
                        shape_why=lambda k, p: why[k.step],
                        covered=lambda view, step, part, r: True)
    # the shape's own reasons sorted (round 2, N7); a hand / a layer over it is covered (B2)
    assert plan["count"] == 0
    assert (plan["skipped"]["no_shape"], plan["skipped"]["absent"],
            plan["skipped"]["covered"]) == (2, 1, 1)
