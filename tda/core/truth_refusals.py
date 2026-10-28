"""Why a frame cannot be confirmed right now, as data (task U2e).

:meth:`tda.core.truth_verify.VerifyMixin.verify_frame` refuses a frame for five
reasons, checked in this order:

0. the frame has **no photo** -- its row is flagged ``missing`` or has no
   stored path (task U38): there is nothing to confirm, and a cached file at
   its number is another photo's;
1. an **open conflict** about the frame -- a disagreement only a human settles;
2. a **blocking compiler problem** (:data:`BLOCKING_PROBLEMS`);
3. a **frozen disagreement** -- a row somebody confirmed that the current inputs
   no longer produce (another frame changed the part's shape since);
4. an **input race** -- an edit landed while the confirmation was being made.

The window used to find out about the first and the third only by pressing
Space: the pane under the task card listed compiler codes, so a frame with an
open conflict showed nothing and the guide said "按 Space" -- the walk that
makes that likely is fixing a part with Enter on frame j, which reaches frames
already confirmed, whose re-check opens a conflict.

So the reasons are computed in one place, :func:`blocking_reasons`, and both
``verify_frame`` and the session's arrival read it.  Each reason is a
:class:`Blocker` with a *code* shaped like a compiler problem
(``open_conflict:12/chassis``, ``frozen_disagreement:chassis``,
``inputs_changed:``), so a panel lists it next to ``missing_shape:…`` and
:func:`is_blocking` counts it with them.  The race cannot be seen coming; it is
the one reason only a refusal reports (:func:`inputs_changed`).

**Two modes, one function** (U2e round 2).  ``verify_frame`` asks
``thorough=True``: its own order and early stops -- an open conflict refuses
before anything is compiled or decoded -- and every confirmed row behind the
inputs is compared.  The display asks ``thorough=False``: every category, and
the frozen rows are compared only when the frame's stored digest does not
already rule a disagreement out.  Comparing means decoding masks, and in the
normal reverse walk every confirmed frame is *behind but agreeing* (the first
commit of any new part moves the segment's layer order, and an agreeing
re-check leaves the old ``input_hash`` on purpose): decoding them on every
announce put 2 s on a 12 MP arrival.  :func:`frozen_disagreements` is the one
comparison; ``refresh`` queues from it too.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from tda.core.cache import has_no_photo
from tda.core.compiler import CompiledFrame
from tda.core.model import FrameKey
from tda.core.truth_conflicts import (
    disagreement,
    geom_payload,
    label_changes,
    label_text,
    row_payload,
    row_values,
)

__all__ = [
    "BLOCKING_PROBLEMS", "Blocker", "CONFLICT", "FRAME_MISSING", "FROZEN",
    "FROZEN_DISAGREEMENT", "INPUTS_CHANGED", "NO_PHOTO", "OPEN_CONFLICT", "PROBLEM",
    "RACE", "REFUSAL_CODES", "VerifyRefused", "blocking_reasons", "conflict_of",
    "frozen_disagreements", "inputs_changed", "is_blocking", "no_photo",
]

#: Problem prefixes that stop a frame from being verified (spec 3.3 step 3).
#: Everything else -- ``bench_missing``, ``zorder_missing``, ``empty_visible``,
#: ``pose_segment_ambiguous`` -- is a warning the annotator may accept.
BLOCKING_PROBLEMS = ("missing_shape:", "zorder_cycle:", "shape_size_mismatch:")

#: The codes of the four refusals that are not compiler problems.
OPEN_CONFLICT = "open_conflict:"
FROZEN_DISAGREEMENT = "frozen_disagreement:"
INPUTS_CHANGED = "inputs_changed:"
#: The frame's row is flagged ``missing`` or has no path: no photo, nothing to
#: confirm (U38).
FRAME_MISSING = "frame_missing:"
REFUSAL_CODES = (OPEN_CONFLICT, FROZEN_DISAGREEMENT, INPUTS_CHANGED, FRAME_MISSING)

#: :attr:`Blocker.kind`, one per reason, in the order ``verify_frame`` checks them.
NO_PHOTO, CONFLICT, PROBLEM, FROZEN, RACE = "no_photo", "conflict", "problem", "frozen", "race"

VERIFIED = "verified"


def is_blocking(problem: str) -> bool:
    """Does this line stop the frame from being verified?

    A blocking compiler problem or one of the refusals of :func:`blocking_reasons`.
    The one test ``verify_frame`` applies, and the one the task card and the
    guide ask when they say what stands between the annotator and ``Space``
    (tasks U2d, U2e) -- so the two can never name different sets.
    """
    return str(problem).startswith(BLOCKING_PROBLEMS + REFUSAL_CODES)


@dataclass(frozen=True)
class Blocker:
    """One reason the frame cannot be confirmed now."""

    #: :data:`NO_PHOTO`, :data:`CONFLICT`, :data:`PROBLEM`, :data:`FROZEN` or
    #: :data:`RACE`.
    kind: str
    #: What a panel lists it under; :func:`is_blocking` is true for it.
    code: str
    #: The refusal's own words for it (English, the op-log's language).
    text: str
    instance: str = ""
    conflict_id: Optional[int] = None
    #: A frozen disagreement's ``(old payload, new payload, pixels or None)``:
    #: what ``verify_frame`` queues its conflict with.
    queue: Optional[tuple] = field(default=None, compare=False, repr=False)


class VerifyRefused(ValueError):
    """``verify_frame``'s refusal, with the reasons it was refused for."""

    def __init__(self, message: str, blockers: list[Blocker]) -> None:
        super().__init__(message)
        self.blockers = list(blockers)


def inputs_changed() -> Blocker:
    """The input race: an edit landed while the frame was being confirmed.

    Transient, and only ever known after the fact: pressing Space again
    confirms the frame against the inputs it has now.
    """
    return Blocker(RACE, INPUTS_CHANGED, "the inputs changed while confirming; "
                                         "press Space again")


def no_photo() -> Blocker:
    """The frame has no photo: its row is flagged ``missing``, or has no path (task U38).

    A step this view never photographed -- a missed shot the index declares
    (``scan.<d>.skipped_steps``), a camera that did not fire -- carries the
    state machine and the shape anchors (spec 4.2, 缺帧处理) but no picture,
    so there is nothing a confirmation could be about. D61's scanner step 26
    was confirmed anyway, over a cached copy of step 27's photo; with the
    picture gone the frame is refused before anything is compiled.
    """
    return Blocker(NO_PHOTO, FRAME_MISSING,
                   "this view has no photo of this step (a missed shot): there is "
                   "nothing to confirm")


def conflict_of(code: str) -> Optional[tuple[int, str]]:
    """``(conflict id, instance)`` of an ``open_conflict:`` code, else ``None``."""
    if not str(code).startswith(OPEN_CONFLICT):
        return None
    cid, _sep, instance = str(code)[len(OPEN_CONFLICT):].partition("/")
    try:
        return int(cid), instance
    except ValueError:
        return None


#: What :func:`blocking_reasons` gets the frame from: a zero-argument callable
#: returning ``(compiled, frame overrides, stored truth rows)`` -- either of the
#: last two ``None`` to have it read here.  A callable, so that an open conflict
#: refuses a confirmation before anything is compiled (main's order).
FrameSource = Callable[[], tuple]


def blocking_reasons(db, key: FrameKey, frame: FrameSource, *, thorough: bool,
                     ruled_out: Optional[Callable[[CompiledFrame], bool]] = None
                     ) -> list[Blocker]:
    """Why ``verify_frame`` refuses this frame now, as :class:`Blocker` items.

    ``thorough=True`` is ``verify_frame``'s check, in its order, and the
    answer is the first category that has anything: open conflicts (``frame``
    is not even called), else blocking compiler problems, else every confirmed
    row behind the inputs compared (:func:`frozen_disagreements`).

    ``thorough=False`` is what the window shows: every category at once.  A
    frozen disagreement about a part that already has an open conflict is left
    out -- settling the conflict settles it -- so each part is named once.  And
    the confirmed rows are compared only when they can disagree at all: some
    row is behind these inputs (one indexed query), and ``ruled_out(compiled)``
    -- the frame's stored digest is current for them -- is false.  That rule
    holds because a digest is only ever stamped when no disagreement stands,
    and cleared whenever one is queued (see the U2e report's round 2).

    Nothing is written: a frozen disagreement is only *queued*, by the
    confirmation or by a refresh.  The race is not here: :func:`inputs_changed`.

    A frame with no photo of its own -- its row flagged ``missing``, or with no
    stored path (:func:`tda.core.cache.has_no_photo`, the test the image cache
    reads by) -- has one reason in either mode, :func:`no_photo`, and ``frame``
    is not called: there is no picture to compile anything over (task U38).
    """
    if has_no_photo(db.get_frame(key)):
        return [no_photo()]
    conflicts = [Blocker(CONFLICT, f"{OPEN_CONFLICT}{cid}/{instance}",
                         f"conflict {cid} ({instance}) is still open", instance, cid)
                 for cid, instance in db.open_conflicts_at(key)]
    if thorough and conflicts:
        return conflicts
    compiled, overrides, stored = frame()
    problems = [Blocker(PROBLEM, p, p) for p in compiled.problems
                if str(p).startswith(BLOCKING_PROBLEMS)]
    if thorough:
        return problems or frozen_disagreements(db, key, compiled, overrides, stored)
    if stored is None and not db.frozen_rows_behind(key, compiled.input_hash):
        frozen: list[Blocker] = []          # the common case, one indexed query
    elif ruled_out is not None and ruled_out(compiled):
        frozen = []                         # behind, but provably agreeing
    else:
        frozen = frozen_disagreements(db, key, compiled, overrides, stored)
    disputed = {b.instance for b in conflicts}
    return conflicts + problems + [b for b in frozen if b.instance not in disputed]


def frozen_disagreements(db, key: FrameKey, compiled: CompiledFrame,
                         overrides: Optional[dict] = None,
                         stored: Optional[dict] = None) -> list[Blocker]:
    """The confirmed rows these inputs no longer produce, one per instance.

    The one comparison: ``verify_frame`` refuses on it and
    :meth:`~tda.core.truth.TruthService.refresh` queues from it, so the two
    cannot drift.  A confirmed row whose instance left the frame comes first,
    then one whose geometry or labels moved beyond the re-tracing tolerance --
    the order ``verify_frame`` has always queued them in.  A row carrying this
    compilation's own ``input_hash`` *is* this compilation and is skipped
    without decoding anything.

    A row 「重签」 wrote is measured against the row **a person signed** before
    it (:func:`tda.core.truth_anchor.human_rows`, task U32 round 2): the
    tolerance is not transitive, and measured against the re-signed row a part
    could drift one tolerance per re-signing without a conflict ever opening.
    The conflict still shows the stored row as its old side.
    """
    from tda.core.truth_anchor import human_rows

    if stored is None:
        stored = db.compiled(key)
    if overrides is None:
        overrides = db.frame_overrides(key)
    behind = [i for i in sorted(set(stored) & set(compiled.instances))
              if stored[i]["status"] == VERIFIED and stored[i]["input_hash"] != compiled.input_hash]
    signed = human_rows(db, key, behind) if behind else {}
    out: list[Blocker] = []
    for instance in sorted(set(stored) - set(compiled.instances)):
        if stored[instance]["status"] != VERIFIED:
            continue
        out.append(Blocker(FROZEN, f"{FROZEN_DISAGREEMENT}{instance}",
                           f"{instance} is no longer in this frame", instance,
                           queue=(row_payload(stored[instance]), None, None)))
    for instance in sorted(set(stored) & set(compiled.instances)):
        row = stored[instance]
        if row["status"] != VERIFIED or row["input_hash"] == compiled.input_hash:
            continue
        compiled_inst = compiled.instances[instance]
        reference = signed.get(instance, row)
        diff = disagreement(reference, compiled_inst)
        labels = label_changes(reference, compiled_inst, overrides.get(instance))
        if diff is None and not labels:
            continue
        what = label_text(labels) if labels else f"{diff} px differ"
        if instance in signed:
            what += ", against the row signed before re-signing"
        values = row_values(compiled_inst)
        out.append(Blocker(
            FROZEN, f"{FROZEN_DISAGREEMENT}{instance}",
            f"the confirmed {instance} no longer agrees ({what})", instance,
            queue=(row_payload(row), geom_payload(values.visible_rle, values.box, labels),
                   int(diff or 0)),
        ))
    return out
