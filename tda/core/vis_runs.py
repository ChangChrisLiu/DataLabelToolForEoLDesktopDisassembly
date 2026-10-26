"""The visibility review over runs and across the views (task U36).

The user (2026-10-26): 「有的 invisible 是持续了很多 frame 的，这个和不同视角之间可能也有
不一样，在 review 中把这个也考虑进去。」 -- a part can stay 看不见 for forty frames, and
the same part on the same steps can be hidden in one view and plainly there in
another (the camera angle). This module is the rule, without Qt and without
the session:

* **runs of a view** -- :func:`view_runs` builds them over every candidate row
  of the view, reviewed or not (a stamp is part of what makes a run here), for
  ``cli check --visibility`` and ``status``'s count of runs still to review
  (:func:`pending_run_count`); the queue's own are
  :func:`~tda.core.vis_review.runs_of` over the unreviewed rows;
* **the views side by side** -- :func:`read_side` reads one view's confirmed
  rows once (no masks); :func:`hidden_stretches` is a part's 看不见 frames and
  stretches in a view (the summary at the top of the queue); :func:`cross_view`
  is, for one run, what every view says about the part on the same steps --
  seen, not seen, why, or 「—」 where the view is not annotated;
* **a suggested reason** -- :func:`suggest`: not seen here, seen in another
  view and nothing over it here -> 角度看不到; something over it here (a part,
  a cable) and not seen by the scanner either -> 被挡住. Only a default: the
  annotator still presses the key;
* **the sync** -- :func:`sync_plan`: which frames of the other views a run's
  verdict may be copied onto -- the same steps, the part as hidden there as
  here, no verdict yet. It never overwrites a verdict, never puts a reason
  where the compiler has a part covering it, and says how many it left and why.

"See" is the label's: :data:`INVISIBLE` (``occluded_full``, ``out_of_view``)
is 看不见; every other label has pixels of the part on the frame (``too_small``
and ``occluded_partial`` too).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from tda.core.model import (
    OCCLUDER_CABLE,
    VIEWS,
    FrameKey,
    Visibility,
)
from tda.core.vis_review import (
    OCCLUDED_LABELS,
    VERDICT_CABLE,
    VERDICT_COMPUTED,
    VERDICT_LABEL,
    VERDICT_VIEWPOINT,
    Run,
    mark_whole,
    order_of,
    runs_of,
    stamp_covers,
)

__all__ = [
    "INVISIBLE",
    "SCAN",
    "STATE_NONE",
    "STATE_OK",
    "STATE_UNCONFIRMED",
    "SYNC_LABELS",
    "Side",
    "cross_view",
    "hidden_stretches",
    "pending_run_count",
    "read_side",
    "sample_steps",
    "suggest",
    "sync_plan",
    "view_runs",
]

#: The labels under which nothing of the part is on the frame: 看不见.
INVISIBLE = frozenset({Visibility.OCCLUDED_FULL.value, Visibility.OUT_OF_VIEW.value})
#: The labels a verdict may be copied to the other views under (「同步到其他视角」):
#: a part hidden, wholly or in part. Never ``visible`` and its kin -- a part
#: seen here is no news about another view.
SYNC_LABELS = frozenset(OCCLUDED_LABELS | {Visibility.OUT_OF_VIEW.value})
#: The scanner's view: the suggestion's "not seen by the scanner either".
SCAN = "scan"

#: A view with no shape and no confirmed frame of the desktop: 「—」.
STATE_NONE = "none"
#: Shapes (drafts, a view begun) but no confirmed frame: nothing to compare yet.
STATE_UNCONFIRMED = "unconfirmed"
#: Confirmed frames: its rows are the truth.
STATE_OK = "ok"


# --------------------------------------------------------------------------- #
# one view, read once
# --------------------------------------------------------------------------- #
@dataclass
class Side:
    """What one view says, read once: its confirmed rows and how far it is."""

    view: str
    state: str
    #: ``part -> step -> row``
    #: (:meth:`~tda.core.db_visreview.VisReviewRowsMixin.vis_view_rows`).
    parts: dict = field(default_factory=dict, repr=False)
    #: The steps with a confirmed frame.
    confirmed: frozenset = frozenset()
    #: The annotatable steps, ascending (what "consecutive" is counted over).
    order: list = field(default_factory=list, repr=False)

    def row(self, instance: str, step: int) -> Optional[dict]:
        """The part's row on a confirmed frame, ``None`` when it has none there."""
        return self.parts.get(str(instance), {}).get(int(step))


def read_side(db, desktop: int, view: str) -> Side:
    """One view's side: one query for its rows, one for its frames and shapes."""
    shapes, confirmed = db.view_annotation(desktop, view)
    if not confirmed:
        return Side(view, STATE_UNCONFIRMED if shapes else STATE_NONE)
    parts: dict[str, dict[int, dict]] = {}
    for row in db.vis_view_rows(desktop, view):
        parts.setdefault(str(row["instance"]), {})[int(row["step"])] = row
    return Side(view, STATE_OK, parts, frozenset(confirmed), order_of(db, desktop, view))


def _covered(row: dict) -> bool:
    return stamp_covers(row.get("flags"), row.get("visibility"), row.get("occluder"))


def _candidate(row: dict) -> bool:
    """Is the row one the review is about (not plain ``visible``, or set by hand)?"""
    return (str(row.get("visibility") or "") != Visibility.VISIBLE.value
            or row.get("hand_visibility") is not None or row.get("occluder") is not None)


# --------------------------------------------------------------------------- #
# runs of a view, reviewed or not
# --------------------------------------------------------------------------- #
def view_runs(db, desktop: int, view: str) -> list[Run]:
    """Every run of the view's candidate rows -- stamped and not -- part by part.

    A run is one part on consecutive annotatable frames with one label, one
    hand label (or none), one reason, and stamped or not; :attr:`Run.whole`
    says it is all of the part's confirmed frames. What ``cli check
    --visibility`` lists. No masks are read.
    """
    rows = [dict(r, reviewed=_covered(r)) for r in db.vis_review_rows(desktop, view)]
    if not rows:
        return []
    order = order_of(db, desktop, view)
    keep = set(order)
    runs = runs_of([r for r in rows if r["step"] in keep], order)
    mark_whole(runs, db.vis_presence(desktop, view), order)
    return runs


def pending_run_count(db, desktop: int, view: str) -> int:
    """How many runs are still to review in one view (``status``)."""
    return sum(1 for run in view_runs(db, desktop, view) if not run.reviewed)


# --------------------------------------------------------------------------- #
# a part in a view: its 看不见 stretches
# --------------------------------------------------------------------------- #
def hidden_stretches(side: Side, instance: str) -> Optional[tuple[int, int, int]]:
    """``(frames, stretches, longest)`` the part is 看不见 in ``side``; ``None`` for 「—」.

    A stretch is consecutive annotatable frames whose label is
    :data:`INVISIBLE`, whatever the reason and whether reviewed -- "hidden for
    forty frames in oak2, three in scan" is what it is for. ``None`` when the
    view has no confirmed frame.
    """
    if side.state != STATE_OK:
        return None
    index = {int(s): i for i, s in enumerate(side.order)}
    hidden = sorted(index[step] for step, row in side.parts.get(str(instance), {}).items()
                    if step in index and str(row.get("visibility") or "") in INVISIBLE)
    if not hidden:
        return 0, 0, 0
    stretches, longest, length = 1, 1, 1
    for before, at in zip(hidden, hidden[1:]):
        if at == before + 1:
            length += 1
        else:
            stretches, length = stretches + 1, 1
        longest = max(longest, length)
    return len(hidden), stretches, longest


# --------------------------------------------------------------------------- #
# one run, every view
# --------------------------------------------------------------------------- #
def sample_steps(steps: Iterable[int]) -> list[int]:
    """The first, middle and last of ``steps`` (fewer when there are fewer)."""
    values = sorted({int(s) for s in steps})
    if len(values) <= 3:
        return values
    return [values[0], values[len(values) // 2], values[-1]]


def cross_view(entry: dict, sides: dict[str, Side], here: str,
               names: Optional[Callable[[str, int, str, dict], tuple]] = None) -> list[dict]:
    """What every view says about the run's part on the run's steps.

    One dict per view of :data:`~tda.core.model.VIEWS` (``sides`` holds them):
    ``view``, ``here``, ``state`` (:data:`STATE_NONE` -> 「—」), ``seen`` /
    ``hidden`` / ``partial`` (frames whose label has pixels of the part / is
    :data:`INVISIBLE` / is ``occluded_partial``), ``absent`` (confirmed frames
    without the part), ``unconfirmed`` (steps not confirmed there),
    ``reasons`` (``{cable|viewpoint|other_unlabeled|manual|computed: frames}``
    over the frames it is not seen on), ``by`` (the parts hiding it there:
    the run's own here, ``names(view, step, part, row)`` on a sample of the
    other view's hidden frames) and ``highlight`` -- another view that sees it
    on these steps.
    """
    instance = str(entry["instance"])
    steps = [int(s) for s in entry.get("steps") or ()]
    out = []
    for view in VIEWS:
        side = sides.get(view)
        row = {"view": view, "here": view == here,
               "state": side.state if side is not None else STATE_NONE,
               "seen": 0, "hidden": 0, "partial": 0, "absent": 0, "unconfirmed": 0,
               "reasons": {}, "by": [], "highlight": False}
        out.append(row)
        if side is None or side.state != STATE_OK:
            continue
        unseen: list[tuple[int, dict]] = []
        for step in steps:
            if step not in side.confirmed:
                row["unconfirmed"] += 1
                continue
            found = side.row(instance, step)
            if found is None:
                row["absent"] += 1
                continue
            label = str(found.get("visibility") or "")
            if label in INVISIBLE:
                row["hidden"] += 1
            else:
                row["seen"] += 1
            if label == Visibility.OCCLUDED_PARTIAL.value:
                row["partial"] += 1
            if label != Visibility.VISIBLE.value or found.get("occluder"):
                why = (str(found["occluder"]) if found.get("occluder")
                       else "manual" if found.get("hand_visibility") is not None
                       else "computed")
                if label != Visibility.VISIBLE.value:
                    row["reasons"][why] = row["reasons"].get(why, 0) + 1
                if why == "computed" and label in OCCLUDED_LABELS:
                    unseen.append((step, found))
        if view == here:
            row["by"] = list(entry.get("occluded_by") or ())
        elif names is not None and unseen:
            picked = {s: r for s, r in unseen}
            for step in sample_steps(picked):
                for name in names(view, step, instance, picked[step]):
                    if name not in row["by"]:
                        row["by"].append(name)
        row["highlight"] = view != here and row["seen"] > 0
    return out


def _clear(row: dict) -> int:
    """Frames of a view with more of the part on them than a partial occlusion."""
    return row["seen"] - row["partial"]


def suggest(entry: dict, views: list[dict], here: str) -> Optional[dict]:
    """The reason the other views point at, or ``None`` (task U36, ruled).

    For a run whose label is an occlusion (完全 / 部分):

    * nothing over it here (no part hides it, no cable reason) and another
      annotated view sees it on these steps (for 部分: sees more of it) ->
      **角度看不到** (``viewpoint``, ``Z`` / ``Shift+Z``);
    * something over it here -- a part, a cable -- and the scanner does not
      see it on these steps either (or this *is* the scanner) -> **被挡住**:
      the computed occlusion as it is (``Space``), or the cable (`` ` `` /
      ``Shift+```) when the cable is the reason given.

    ``{"verdict", "label", "action", "key", "kind": "viewpoint"|"blocked",
    "views"}`` -- ``views`` are the ones that see it (viewpoint) -- or ``None``.
    Only a default: the annotator still presses the key.
    """
    label = str(entry.get("visibility") or "")
    if label not in OCCLUDED_LABELS:
        return None
    full = label == Visibility.OCCLUDED_FULL.value
    reason = entry.get("occluder")
    cable = reason == OCCLUDER_CABLE
    blocked = bool(entry.get("occluded_by")) or cable
    if reason not in (None, OCCLUDER_CABLE):
        # a hand reason other than the cable: 角度看不到 already, or 没标注的东西 --
        # the annotator said why; the other views are no better informed
        return None
    others = [v for v in views if not v["here"] and v["state"] == STATE_OK]
    seeing = [v["view"] for v in others if (v["seen"] if full else _clear(v)) > 0]
    if not blocked and seeing:
        return {"kind": "viewpoint", "verdict": VERDICT_VIEWPOINT, "label": label,
                "action": "vis_review_viewpoint" if full else "vis_review_viewpoint_partial",
                "key": "Z" if full else "Shift+Z", "views": seeing}
    if not blocked:
        return None
    scan = next((v for v in views if v["view"] == SCAN), None)
    scanner_blind = here == SCAN or (
        scan is not None and scan["state"] == STATE_OK
        and (scan["hidden"] if full else scan["hidden"] + scan["partial"]) > 0)
    if not scanner_blind:
        return None
    # ``scan_too``: the scanner is another view that does not see it either
    scan_too = here != SCAN
    if cable:
        return {"kind": "blocked", "verdict": VERDICT_CABLE, "label": label,
                "action": "vis_review_cable" if full else "vis_review_cable_partial",
                "key": "`" if full else "Shift+`", "views": [], "scan_too": scan_too}
    return {"kind": "blocked", "verdict": VERDICT_COMPUTED, "label": label,
            "action": "vis_review_computed", "key": "Space", "views": [],
            "scan_too": scan_too}


# --------------------------------------------------------------------------- #
# 同步到其他视角
# --------------------------------------------------------------------------- #
#: Why a frame of another view is left alone by the sync (the counts it reports).
SKIP_REASONS = ("verdict", "different", "covered", "unconfirmed", "absent", "no_shape")


def source_verdicts(db, desktop: int, view: str, instance: str,
                    steps: Iterable[int]) -> dict[int, dict]:
    """``step -> {"verdict", "visibility", "occluder"}`` the run's frames were answered with.

    Only frames whose stamp still covers their label and reason: a stamp the
    label has moved away from answers for nothing.
    """
    out: dict[int, dict] = {}
    for step in sorted({int(s) for s in steps}):
        key = FrameKey(int(desktop), step, str(view))
        row = db.compiled_label(key, instance)
        if row is None:
            continue
        flags = db.instance_flags(key, instance)
        fo = db.frame_overrides(key).get(str(instance))
        occluder = fo.occluder if fo is not None else None
        if stamp_covers(flags, row.get("visibility"), occluder) and flags.get("verdict"):
            out[step] = {"verdict": str(flags["verdict"]),
                         "visibility": row.get("visibility"), "occluder": occluder}
    return out


def _target(source: dict) -> tuple[Optional[str], Optional[str]]:
    """``(visibility, occluder)`` a verdict leaves on a frame it is copied to."""
    verdict = source["verdict"]
    if verdict == VERDICT_COMPUTED:
        return None, None                  # the frame's own computed label, stamped
    if verdict == VERDICT_LABEL:
        return source["visibility"], None
    return source["visibility"], source["occluder"]


def sync_plan(db, desktop: int, view: str, instance: str, steps: Iterable[int],
              shape_ok: Optional[Callable[[FrameKey, str], bool]] = None,
              covered: Optional[Callable[[str, int, str, dict], bool]] = None) -> dict:
    """Which frames of the other views the run's verdict may go to, and which not.

    For each step the run (in ``view``) was answered on with a hidden label
    (:data:`SYNC_LABELS`), each other sequence view with confirmed frames:
    the same step's row of the part qualifies when it carries **no stamp** (a
    verdict is never overwritten) and has **the same label** as the answered
    frame (as hidden there, in the same way). A hand mark there without a
    verdict -- the 看不见 the annotator said while drawing -- is what the
    review is for, so it takes the verdict (counted in ``replaced`` when its
    reason changes). And by verdict:

    * ``computed`` (as the compiler has it) goes only where the label there
      is computed too;
    * 角度看不到 / 被线挡住 need a shape of it in its place there
      (``shape_ok(key, instance)``) and do not go where the compiler has a
      part covering it there (``covered(view, step, instance, row)``) -- a
      reason that is false there.

    ``{"source": {step: verdict}, "views": {view: {step: verdict}}, "count",
    "replaced", "skipped": {reason: frames}, "refused": str}`` -- ``refused``
    is why nothing can be planned at all (no verdict yet, a label that is not
    hidden), ``""`` otherwise.
    """
    plan = {"source": {}, "views": {}, "count": 0, "replaced": 0,
            "skipped": {k: 0 for k in SKIP_REASONS}, "refused": ""}
    # the other views with confirmed frames: the only ones a verdict can go to
    sides = [side for side in (read_side(db, desktop, other) for other in VIEWS
                               if other != view) if side.state == STATE_OK]
    source = source_verdicts(db, desktop, view, instance, steps)
    plan["source"] = source
    if not source:
        plan["refused"] = "no_verdict"
        return plan
    hidden = {s: v for s, v in source.items() if str(v["visibility"] or "") in SYNC_LABELS}
    if not hidden:
        plan["refused"] = "not_hidden"
        return plan
    for side in sides:
        other = side.view
        chosen: dict[int, dict] = {}
        for step, verdict in sorted(hidden.items()):
            if step not in side.confirmed:
                if (db.get_frame(FrameKey(int(desktop), step, other)) or {}):
                    plan["skipped"]["unconfirmed"] += 1
                continue
            row = side.row(instance, step)
            if row is None:
                plan["skipped"]["absent"] += 1
                continue
            if (row.get("flags") or {}).get("vis_reviewed"):
                plan["skipped"]["verdict"] += 1          # never overwrite a verdict
                continue
            if str(row.get("visibility") or "") != str(verdict["visibility"] or ""):
                plan["skipped"]["different"] += 1        # not hidden the same way there
                continue
            label, occluder = _target(verdict)
            hand = (row.get("hand_visibility"), row.get("occluder") or None)
            has_hand = hand != (None, None)
            if verdict["verdict"] == VERDICT_COMPUTED and has_hand:
                plan["skipped"]["different"] += 1        # not the compiler's label there
                continue
            if verdict["verdict"] in (VERDICT_CABLE, VERDICT_VIEWPOINT):
                if shape_ok is not None and not shape_ok(FrameKey(int(desktop), step, other),
                                                         str(instance)):
                    plan["skipped"]["no_shape"] += 1
                    continue
                if (not has_hand and covered is not None
                        and covered(other, step, str(instance), row)):
                    plan["skipped"]["covered"] += 1      # a part over it there: not this reason
                    continue
            if has_hand and verdict["verdict"] != VERDICT_COMPUTED and hand != (label, occluder):
                plan["replaced"] += 1
            chosen[step] = verdict
        if chosen:
            plan["views"][other] = chosen
            plan["count"] += len(chosen)
    return plan
