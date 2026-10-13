"""The persisted truth table: refresh, verification and conflicts (spec 3.4).

:mod:`tda.core.compiler` derives one frame's geometry from explicit inputs;
:class:`TruthService` is what turns that pure function into the stored
``compiled_mask`` table:

* :meth:`TruthService.compile`  -- gather this frame's inputs and compile it.
* :meth:`TruthService.refresh`  -- write the result into the truth table.
* :meth:`TruthService.verify_frame` / :meth:`TruthService.demote_frame`
  -- freeze a frame a human confirmed, or send it back for review.
* :meth:`TruthService.resolve_conflict` -- close one queued disagreement by
  accepting the new value or pinning the frozen one.
* :meth:`TruthService.affected_steps` -- which frames one shape reaches, for
  the canvas' "affects N frames" strip (spec 4.3).

Geometry, the frozen-vs-recompiled comparison and the conflict payloads live
next door in :mod:`tda.core.truth_conflicts`; reading the inputs out of the
database is :mod:`tda.core.truth_inputs`' job.

The rules the table lives by (spec 3.4)
---------------------------------------
* A row is ``auto`` while the compiler owns it: any change to the inputs
  rewrites it.
* A row is ``verified`` once a human confirmed the frame. From then on the
  compiler may **never** overwrite it. When a re-compilation disagrees with a
  frozen row beyond the re-tracing tolerance of :func:`tda.core.masks.is_conflict`
  (box rows: a corner moving more than :data:`BOX_TOL_PX`), the disagreement is
  queued in the ``conflict`` table -- old value, new value and the number of
  differing pixels -- and the row keeps its frozen value until a human resolves
  it with :meth:`TruthService.resolve_conflict`. Nothing is ever silently
  overwritten, and a disagreement already sitting in the open queue is not
  queued twice.
* **One open conflict per frozen row** (task U7): ``(frame, instance)`` has at
  most one. A newer disagreement about the same row *replaces* the open one
  -- closed ``superseded`` with the reason :data:`OVERTAKEN` -- instead of
  stacking next to it: fourteen small commits of one re-trace used to leave
  thirteen conflicts open on one frozen row of D13/scan, each about a shape
  that no longer existed.
* A verified frame that gains or loses an instance drops back to
  ``needs_review``: the human confirmed a different set of objects than the one
  the inputs now describe.

Costs
-----
:meth:`refresh` compiles the frame and then compares hashes: when every stored
row already carries this frame's ``input_hash`` and the instance set is
unchanged, nothing is written at all. :meth:`refresh_range` shares one
:class:`~tda.core.truth_inputs.InputCache` across its steps, so the instance
table, the event log and the keyframes are read once per sweep.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from tda.core import masks
from tda.core.cache import configured_cache_dir
from tda.core.compiler import CompiledFrame, compile_frame, select_keyframe
from tda.core.db import RESOLUTIONS, Db
from tda.core.model import FrameKey, FrameOverride, ShapeKeyframe
from tda.core.states import needs_geom
from tda.core.taxonomy import Taxonomy
from tda.core.truth_fresh import FreshMixin, digest_of
from tda.core.truth_resolve import ResolveMixin, StaleConflictError
from tda.core.truth_verify import BLOCKING_PROBLEMS, VerifyMixin, usable
from tda.core.truth_conflicts import (
    AGREE_AGAIN,
    BOX_TOL_PX,
    CLOSE_KIND,
    DUPLICATE,
    GEOM_BOX,
    OVERTAKEN,
    REOPEN_KIND,
    ROW_GONE,
    SUPERSEDE_REASONS,
    payload_geometry,
    payload_labels,
    row_values,
)
from tda.core.truth_refusals import frozen_disagreements
from tda.core.truth_inputs import (
    FrameInputs,
    InputCache,
    events_of,
    gather,
    instances_of,
    pose_segment_of,
    state_of,
)

__all__ = ["AGREE_AGAIN", "BLOCKING_PROBLEMS", "BOX_TOL_PX", "BY_HAND", "DUPLICATE",
           "OVERTAKEN", "ROW_GONE", "StaleConflictError", "TruthService"]

ACCEPT_NEW = "accept_new"
KEEP_OLD = "keep_old"
SUPERSEDED = "superseded"
OPEN = "open"

#: The resolutions a human may ask for; ``superseded`` is only ever recorded by
#: :meth:`TruthService.resolve_conflict` itself.
BY_HAND = tuple(r for r in RESOLUTIONS if r != SUPERSEDED)

#: ``cache_dir`` was not given: resolve it from the configuration.
_CONFIGURED = object()

AUTO = "auto"
#: The two row statuses the session and the review panel read off this module.
VERIFIED = "verified"
NEEDS_REVIEW = "needs_review"

#: Who writes the service's own op-log rows.  The rows that close conflicts
#: on its own -- the inputs agree again (U6), a newer disagreement replaced
#: them (U7) -- are ``CLOSE_KIND`` with one of ``SUPERSEDE_REASONS``
#: (:mod:`tda.core.truth_conflicts`, where the vocabulary lives so that the
#: mixins can write the same row).
SYSTEM = "system"
#: The per-frame counters of :meth:`TruthService.refresh` that a pass over many
#: frames sums: :meth:`~TruthService.refresh_range`, the re-check drain and
#: :meth:`~tda.core.truth_fresh.FreshMixin.ensure_fresh`.
REFRESH_COUNTERS = ("updated", "conflicts", "standing", "closed", "overtaken", "skipped")


class TruthService(FreshMixin, ResolveMixin, VerifyMixin):
    """Reads the annotations, compiles frames and owns the ``compiled_mask`` table."""

    def __init__(self, db: Db, tax: Taxonomy, compiler_version: str = "1",
                 cache_dir: object = _CONFIGURED):
        """``cache_dir`` is where this database's frames are cached locally.

        Every compilation needs the frame's canvas size, and measuring it off
        the read-only source drive costs a 12 MP decode per frame; the local
        copy costs nothing. Left out, it is resolved from ``configs/paths.yaml``
        -- but only for the database that file names
        (:func:`tda.core.cache.configured_cache_dir`), because a cache belongs
        to the annotations it was built for. Pass it explicitly (``None`` to opt
        out) when the caller knows better.
        """
        self.db = db
        self.tax = tax
        self.compiler_version = compiler_version
        if cache_dir is _CONFIGURED:
            cache_dir = configured_cache_dir(getattr(db, "path", None))
        self.cache_dir: Optional[str] = None if cache_dir is None else str(cache_dir)

    # ------------------------------------------------------------------ compile

    def compile(self, key: FrameKey, cache: Optional[InputCache] = None) -> CompiledFrame:
        """Compile one frame from the database's current state (spec 3.3).

        ``cache`` is a caller's per-pass :class:`InputCache`, as
        :meth:`refresh_range` keeps one.
        """
        return self._compile(key, cache)[1]

    def _compile(
        self, key: FrameKey, cache: Optional[InputCache] = None
    ) -> tuple[FrameInputs, CompiledFrame]:
        """The frame's inputs and its compilation; ``hw`` is needed by callers."""
        inputs = gather(self.db, self.tax, key, cache, self.cache_dir)
        return inputs, self._compile_inputs(key, inputs)

    def _compile_inputs(self, key: FrameKey, inputs: FrameInputs) -> CompiledFrame:
        """Compile from inputs the caller already gathered."""
        return compile_frame(
            key,
            inputs.hw,
            inputs.needs,
            inputs.keyframes,
            inputs.zorder,
            inputs.overrides,
            inputs.occluders,
            inputs.frame_overrides,
            inputs.transform,
            self.compiler_version,
            placements=inputs.placements,
            pose_segment=inputs.pose_segment,
            bench_roi=inputs.bench_roi,
        )

    # ------------------------------------------------------------------ refresh

    def refresh(self, key: FrameKey, cache: Optional[InputCache] = None,
                guard: Optional[str] = None, want_compiled: bool = False,
                ignore_digest: bool = False,
                held: Optional[CompiledFrame] = None) -> dict:
        """Bring one frame's truth rows up to date with the current inputs.

        Returns ``{"updated", "conflicts", "standing", "closed", "overtaken",
        "skipped", "problems", "compiled", "stale"}``: rows written (a deleted
        row counts as written), disagreements this refresh actually **queued**,
        frozen rows in disagreement whether queued now or already open -- a
        sweep of fifty frames over one unresolved conflict reported fifty
        conflicts when it had queued none -- open conflicts it **closed**
        because their frozen row agrees with the inputs again (see
        :meth:`_close_agreeing`), open conflicts a newer disagreement about the
        same row **overtook** (see :meth:`_queue_conflict`), rows
        left alone, the compiler's problem list, the compilation the whole
        decision was made from -- handed back so that a caller which also
        needs the frame does not compile it a second time -- and whether the
        write was abandoned.

        ``cache`` is :meth:`refresh_range`'s way of reading the step-independent
        inputs once; callers outside this module leave it out.

        ``ignore_digest`` compiles the frame whatever the stored digest says,
        which is what the session's "I do not trust the cache" refresh is for.

        A frame whose stored digest still describes its inputs is **not
        compiled at all**: the pixel work is what a batch pass over an untouched
        view used to spend all its time on. ``compiled`` then comes back
        ``None`` unless ``want_compiled`` asks for it, which is what the session
        does when it needs the arrays to draw.

        ``held`` is a compilation the caller already has for this frame -- the
        session keeps the one it compiled, or the one the sweeper prefetched.
        On that path ``want_compiled`` hands it back instead of compiling, and
        on the other it stands in for the compilation the rows are derived
        from (a bulk ``accept_new`` settles a frame against one compilation and
        then refreshes it, task U7) -- but **only when it is still the
        compilation these inputs make** (:func:`~tda.core.truth_verify.usable`,
        asked of the inputs just read, as ``verify_frame`` asks it of its
        ``prepared``); otherwise the frame is compiled. A session's edit epoch
        is a fact about that session, and another writer moves the inputs
        without touching it (U2f).

        ``guard`` is an :meth:`inputs_digest` taken before the compilation, for
        a caller working off the GUI thread. The pixel work happens outside any
        transaction; every write then happens inside **one**, which begins by
        taking the digest again. If it moved, nothing is written and ``stale``
        comes back true: the frame has to be looked at again against the inputs
        it has now, and a conflict describing the old ones would be a conflict
        nobody caused.
        """
        inputs = gather(self.db, self.tax, key, cache, self.cache_dir)
        digest = digest_of(inputs, self.compiler_version)
        if (not ignore_digest and self._digest_is_current(key, digest)
                and not self.db.open_conflicts_at(key)):
            # the rows already describe exactly these inputs: there is nothing
            # to derive and, since nothing moved, nothing a frozen row could
            # disagree with either.  Not with a conflict still open, though: a
            # current digest next to one means the inputs came back to what
            # was frozen (U6), and only a compilation can close it
            result = {"updated": 0, "conflicts": 0, "standing": 0, "closed": 0,
                      "overtaken": 0, "skipped": self.db.compiled_count(key),
                      "problems": [], "compiled": None, "stale": False}
            if want_compiled:
                result["compiled"] = (
                    held if usable(held, key, inputs, self.compiler_version)
                    else self._compile_inputs(key, inputs))
            return result
        # the inputs are already in hand: gathering them a second time reads the
        # whole keyframe table again, which is most of a batch pass's time
        compiled = (held if usable(held, key, inputs, self.compiler_version)
                    else self._compile_inputs(key, inputs))  # no transaction: pixels only
        with self.db.transaction():
            return self._write_refresh(key, compiled, guard, digest,
                                       inputs.frame_overrides)

    def _write_refresh(self, key: FrameKey, compiled: CompiledFrame,
                       guard: Optional[str], digest: str,
                       overrides: dict[str, FrameOverride]) -> dict:
        """The write half of :meth:`refresh`; the caller holds the transaction."""
        result: dict = {
            "updated": 0,
            "conflicts": 0,
            "standing": 0,
            "closed": 0,
            "overtaken": 0,
            "skipped": 0,
            "problems": list(compiled.problems),
            "compiled": compiled,
            "stale": False,
        }
        if guard is not None and self.inputs_digest(key) != guard:
            result["stale"] = True  # nothing written; the block commits nothing
            return result
        stored = self.db.compiled(key)
        new_hash = compiled.input_hash
        self._mark_bench(key, compiled)

        fresh = set(compiled.instances)
        known = set(stored)
        if known == fresh and all(row["input_hash"] == new_hash for row in stored.values()):
            # nothing to write -- but the rows *do* describe these inputs, and
            # saying so is the whole point: a database that predates the digest
            # table is entirely in this state, and without the stamp no pass
            # over it is ever cheaper than the first.  Every frozen row is
            # this compilation, so no open conflict about one still stands.
            result["skipped"] = len(stored)
            result["closed"] = self._close_agreeing(key, stored, fresh, set())
            self._stamp(key, digest)
            return result

        verified_frame = self._frame_is_verified(key, stored)
        queued: Optional[list[dict]] = None
        overtaken: list[tuple[int, str, str]] = []
        # The one comparison of frozen rows with this compilation, the same one
        # verify_frame refuses on (U2e round 2); queued here in this pass's own
        # order -- moved parts, then departed ones -- as it always was.
        disputed = {b.instance: b for b in frozen_disagreements(
            self.db, key, compiled, overrides, stored)}
        result["closed"] = self._close_agreeing(key, stored, fresh, set(disputed))

        for instance in sorted(fresh):
            compiled_inst = compiled.instances[instance]
            row = stored.get(instance)
            if row is not None and row["input_hash"] == new_hash:
                result["skipped"] += 1
                continue
            if row is not None and row["status"] == VERIFIED:
                found = disputed.get(instance)
                if found is None:
                    result["skipped"] += 1
                    continue
                old, new, pixels = found.queue
                queued, inserted = self._queue_conflict(key, instance, old, new,
                                                        pixels, queued, overtaken)
                result["conflicts"] += int(inserted)
                result["standing"] += 1
                continue
            self._put_row(key, instance, row_values(compiled_inst), AUTO, new_hash)
            result["updated"] += 1

        for instance in sorted(known - fresh):
            row = stored[instance]
            if row["status"] == VERIFIED:
                old_payload = disputed[instance].queue[0]
                queued, inserted = self._queue_conflict(
                    key, instance, old_payload, None, self._payload_area(old_payload), queued,
                    overtaken,
                )
                result["conflicts"] += int(inserted)
                result["standing"] += 1
            else:
                self.db.delete_compiled(key, instance)
                result["updated"] += 1
        # one op-log row (per reason) for every conflict this pass replaced (U7)
        result["overtaken"] = self._log_superseded(key, overtaken)

        if verified_frame and (fresh - known or known - fresh):
            self.demote_frame(key, self._demotion_reason(fresh, known))
        if result["standing"]:
            # the frozen rows deliberately still hold their old value, so they
            # do NOT describe these inputs: stamping the digest here would make
            # the next pass skip the frame and the disagreement would never be
            # raised again once the queue entry was resolved (spec 3.4). It is
            # `standing` and not `conflicts`: a disagreement the deduplication
            # suppressed is every bit as unresolved as the one it matched
            self.db.clear_frame_digest(key)
        else:
            self._stamp(key, digest)
        return result

    def _stamp(self, key: FrameKey, digest: str) -> None:
        """Record what the rows just written were derived from (same transaction).

        Only ever called when the rows do describe those inputs: a frame with a
        queued conflict keeps its frozen value instead, so its digest is dropped
        and the next pass looks at it again.
        """
        self.db.set_frame_digest(key, digest, self.compiler_version,
                                 self.db.compiled_count(key))

    def _close_agreeing(self, key: FrameKey, stored: dict[str, dict], fresh: set[str],
                        disputed: set[str]) -> int:
        """Close the open conflicts this compilation shows are over (task U6).

        A conflict says "the frozen row and the inputs disagree".  When the
        inputs come back -- an override undone, a shape re-traced to what was
        confirmed -- the frozen row agrees again and the conflict describes
        nothing, but nothing used to close it: on D13/scan an override undone a
        minute after it was made left eighteen conflicts open, frames 39 and 40
        red and ``Space`` refused, with nothing for anybody to decide.

        Closed here: every open conflict of this frame whose instance has a
        frozen row, is in the compilation, and is not among the ``disputed``
        instances (:func:`~tda.core.truth_refusals.frozen_disagreements`, the
        comparison ``verify_frame`` refuses on).  Each is resolved
        ``superseded``, and the frame gets **one** ``resolve_conflicts`` op-log
        row naming them all with the reason :data:`AGREE_AGAIN`; its inverse
        reopens them (:meth:`reopen_agreed`).  Called inside the refresh's
        transaction.  Returns how many were closed.
        """
        open_here = self.db.open_conflicts_at(key)
        if not open_here:
            return 0
        agree = [(cid, instance) for cid, instance in open_here
                 if instance in fresh and instance not in disputed
                 and (stored.get(instance) or {}).get("status") == VERIFIED]
        for cid, _instance in agree:
            self.db.resolve_conflict(int(cid), SUPERSEDED)
        return self._log_superseded(key, agree, AGREE_AGAIN)

    def _log_superseded(self, key: FrameKey, closed: list[tuple],
                        reason: Optional[str] = None,
                        asked_by: Optional[str] = None) -> int:
        """The op-log row for conflicts the service closed on its own -- one per reason.

        ``closed`` holds ``(id, instance)`` -- closed for ``reason`` -- or
        ``(id, instance, reason)`` of conflicts already resolved ``superseded``
        in the caller's transaction, all of one frame; each reason is one of
        :data:`SUPERSEDE_REASONS`.  Every path that supersedes -- a refresh
        finding the inputs agree again (:meth:`_close_agreeing`), a refresh
        replacing a conflict by a newer one or dropping a duplicate
        (:meth:`_queue_conflict`), an ``accept_new`` the inputs had overtaken
        (:meth:`~tda.core.truth_resolve.ResolveMixin.resolve_conflict`) --
        writes this same ``resolve_conflicts`` row, by :data:`SYSTEM`, whose
        inverse names them for :meth:`reopen_agreed`: one row per reason, in
        the order the reasons first appear.  ``asked_by`` is the human whose
        request led to it (a late ``accept_new``) and goes into the payload as
        one more key; the row stays the service's.  Nothing is written for an
        empty list.  Returns how many conflicts were named.
        """
        groups: dict[str, list[tuple[int, str]]] = {}
        for item in closed:
            why = item[2] if len(item) > 2 else reason
            if why not in SUPERSEDE_REASONS:
                raise ValueError(f"reason must be one of {SUPERSEDE_REASONS}, got {why!r}")
            groups.setdefault(why, []).append((int(item[0]), str(item[1])))
        for why, items in groups.items():
            ids = [cid for cid, _instance in items]
            payload = {"step": key.step, "conflicts": ids,
                       "instances": sorted({instance for _cid, instance in items}),
                       "resolution": SUPERSEDED, "reason": why}
            if asked_by:
                payload["asked_by"] = str(asked_by)
            self.db.log_op(key.desktop, key.view, CLOSE_KIND, payload,
                           {"kind": REOPEN_KIND, "step": key.step, "conflicts": ids},
                           SYSTEM)
        return sum(len(items) for items in groups.values())

    def reopen_agreed(self, op_id: int) -> list[int]:
        """Undo one ``resolve_conflicts`` row of :meth:`_close_agreeing`: reopen them.

        The conflicts it closed go back to ``open`` exactly as they were queued
        (same id, same old and new value), the frame's digest is dropped so the
        next refresh looks at it again, and the reopening is itself logged with
        the inverse that closes them once more.  A conflict that is open again
        already -- or that was since closed by anything but a supersession, a
        human's answer above all -- is left alone.  Returns the ids reopened.

        **One open conflict per frozen row** (task U7) holds here too:

        * only a row closed with the reason :data:`AGREE_AGAIN` is reopened;
          any other is **refused** (``ValueError``).  :data:`OVERTAKEN` and
          :data:`DUPLICATE` closed conflicts whose row has another open
          conflict standing for it -- reopening would put two on it, one of
          them about a shape nobody can see any more; the way back to an
          overtaken value is to move the inputs back (undo the edit), and the
          next refresh queues it again or closes the newer one.
          :data:`ROW_GONE` closed a conflict about a row that no longer exists.
        * the ids are walked newest first, and a row gets **one** conflict
          back: one agreement that closed a stack of conflicts on one row (a
          database from before U7) reopens the newest of them only, and a row
          that has an open conflict now -- the inputs moved again after the
          agreement -- gets none.

        **Idempotent** (U6 round 2): when there is nothing left to reopen --
        the second call on the same op -- nothing is written at all, no op-log
        row and no dropped digest.

        Raises :class:`ValueError` for any other op: a human's ``keep_old`` or
        ``accept_new`` wrote rows and overrides that reopening alone would not
        take back.
        """
        op = self.db.get_op(int(op_id))
        if op is None or op["kind"] != CLOSE_KIND:
            raise ValueError(f"op {op_id} is not a {CLOSE_KIND} row of the truth service")
        reason = (op["payload"] or {}).get("reason")
        if reason != AGREE_AGAIN:
            raise ValueError(
                f"op {op_id} closed conflicts because {reason!r}; only a closing because "
                f"{AGREE_AGAIN!r} is reopened -- the others stand for a row that has its "
                f"open conflict already, or for none")
        inverse = op["inverse"] or {}
        key = FrameKey(int(op["desktop"]), int(inverse["step"]), str(op["view"]))
        taken = {instance for _cid, instance in self.db.open_conflicts_at(key)}
        wanted: list[int] = []
        for cid in sorted((int(c) for c in inverse.get("conflicts") or []), reverse=True):
            found = self.db.get_conflict(cid)
            instance = "" if found is None else str(found["instance"] or "")
            if (found is None or found["status"] == OPEN
                    or found.get("resolution") != SUPERSEDED or instance in taken):
                continue
            wanted.append(cid)
            taken.add(instance)          # one per row: the newest of a stack
        wanted.sort()
        if not wanted:
            return []
        reopened: list[int] = []
        with self.db.transaction():
            for cid in wanted:
                self.db.reopen_conflict(cid)
                reopened.append(cid)
            self.db.clear_frame_digest(key)
            self.db.log_op(
                key.desktop, key.view, REOPEN_KIND,
                {"step": key.step, "conflicts": reopened, "undoes": int(op_id)},
                {"kind": CLOSE_KIND, "step": key.step, "conflicts": reopened},
                SYSTEM,
            )
        return reopened

    def refresh_range(self, desktop: int, view: str, steps: Iterable[int],
                      per_step: bool = False, ignore_digest: bool = False) -> dict:
        """:meth:`refresh` every step of one view, with the totals summed up.

        ``problems`` is the concatenation of the per-step problem lists in the
        order the steps were given -- or, with ``per_step``, a dict keyed by
        step, which is what a caller needs to say *where* a missing shape is
        (spec 4.4 queues).  ``closed`` sums the conflicts the steps closed
        because their frozen rows agree with the inputs again (U6), and
        ``overtaken`` the ones a newer disagreement replaced (U7); ``cli check``
        and the exports print both.
        """
        cache = InputCache()
        total: dict = {counter: 0 for counter in REFRESH_COUNTERS}
        total["problems"] = {} if per_step else []
        for step in steps:
            one = self.refresh(FrameKey(desktop, int(step), view), cache,
                               ignore_digest=ignore_digest)
            for counter in REFRESH_COUNTERS:
                total[counter] += one[counter]
            if per_step:
                total["problems"][int(step)] = list(one["problems"])
            else:
                total["problems"].extend(one["problems"])
        return total

    # ------------------------------------------------------- pending rechecks

    def queue_rechecks(self, desktop: int, view: str, steps: Iterable[int]) -> list[int]:
        """Remember that these frozen frames have to be compared again (spec 3.4).

        Editing a shape changes the inputs of every frame it reaches, and a
        ``verified`` frame among them may now disagree with what a human froze.
        Compiling them all before the annotator can carry on is what made an
        edit cost seconds, so the check is deferred -- and, because a conflict
        that is never raised is worse than a slow tool, the *request* is stored
        rather than kept in memory. Returns the steps queued.
        """
        wanted = [int(s) for s in steps]
        if wanted:
            self.db.add_rechecks(desktop, view, wanted)
        return sorted(set(wanted))

    def queue_rechecks_for_view(self, desktop: int, view: str) -> list[int]:
        """Queue every frozen frame of one view (:meth:`tda.core.db.Db.queue_rechecks_for_view`).

        The service's name for it, so a caller that already holds a
        :class:`TruthService` does not have to reach past it into the database.
        """
        return self.db.queue_rechecks_for_view(desktop, view)

    def open_conflicts(self, desktop: int, view: Optional[str] = None) -> list[dict]:
        """Disagreements of one desktop (or one view) nobody has settled, oldest first.

        A *standing* conflict is the one thing a batch pass cannot work around:
        :meth:`refresh` will not overwrite the frozen row, so the view keeps
        publishing a value a human has already been told is disputed. Exports
        refuse on it unless they are told to go ahead
        (:func:`tda.core.export.coco.export_coco`'s ``allow_conflicts``), and a
        quality check reports it.
        """
        return self.db.conflicts(desktop, view, open_only=True)

    def pending_rechecks(self, desktop: int, view: str) -> list[int]:
        """Frozen frames of one view still waiting to be compared, ascending.

        An export or a quality check must run these first: the truth table is
        only trustworthy once nothing is outstanding.
        """
        return self.db.rechecks(desktop, view)

    def run_pending_rechecks(self, desktop: int, view: str) -> dict:
        """Work the queue off synchronously; same totals as :meth:`refresh_range`.

        This is the batch path -- ``cli check`` and the exports -- next to the
        background sweeper the GUI uses. Each request is retired under the
        generation it was read with, so one that arrives during the drain stays
        queued rather than being cleared unchecked.
        """
        cache = InputCache()
        total: dict = {counter: 0 for counter in REFRESH_COUNTERS}
        total["problems"] = []
        for step, gen in self.db.recheck_items(desktop, view):
            one = self.refresh(FrameKey(desktop, int(step), view), cache)
            for counter in REFRESH_COUNTERS:
                total[counter] += one[counter]
            total["problems"].extend(one["problems"])
            self.db.clear_recheck(desktop, view, step, gen)
        return total

    # ------------------------------------------------------ conflict resolution

    # -------------------------------------------------------------- keyframes

    def affected_steps(
        self, desktop: int, view: str, instance: str, keyframe: ShapeKeyframe
    ) -> list[int]:
        """The logical steps whose compilation would select ``keyframe``.

        A shape reaches a step when that step is in the keyframe's pose segment,
        the instance needs geometry there, it is in the placement the keyframe
        was drawn for, and :func:`tda.core.compiler.select_keyframe` picks this
        keyframe out of the chain -- i.e. exactly the frames an edit to this
        shape would change (spec 4.3).

        "Needs geometry there" is asked of :func:`tda.core.states.needs_geom`
        with this segment's bench ROI, exactly as :func:`gather` asks it: a
        bench box on a view with no staging area reaches no frame at all, and
        saying it reached four was a promise about pixels nobody compiles.
        """
        cache = InputCache()
        instances = instances_of(self.db, desktop, cache)
        events = events_of(self.db, self.tax, desktop, cache)
        bench_roi = self.db.bench_roi(desktop, view, keyframe.pose_segment)
        chain = [
            kf
            for kf in self.db.keyframes(desktop, view, instance)
            if kf.pose_segment == keyframe.pose_segment and kf.placement == keyframe.placement
        ]
        out: list[int] = []
        for rec in self.db.steps(desktop):
            step = rec.step
            key = FrameKey(desktop, step, view)
            if pose_segment_of(self.db, key, cache) != keyframe.pose_segment:
                continue
            state = state_of(self.db, self.tax, desktop, step, cache)
            if instance not in needs_geom(instances, state, self.tax,
                                          bench_roi=bench_roi):
                continue
            if state[instance].placement != keyframe.placement:
                continue
            if self._same_keyframe(select_keyframe(chain, step), keyframe):
                out.append(step)
        return sorted(out)

    # --------------------------------------------------------------- internals

    @staticmethod
    def _same_keyframe(chosen: Optional[ShapeKeyframe], wanted: ShapeKeyframe) -> bool:
        """Is the selected keyframe the one asked about? By id, or by identity."""
        if chosen is None:
            return False
        if chosen.id is not None and wanted.id is not None:
            return int(chosen.id) == int(wanted.id)
        return chosen is wanted

    def _put_row(
        self,
        key: FrameKey,
        instance: str,
        values,
        status: str,
        input_hash: str,
        verified_by: Optional[str] = None,
    ) -> None:
        """Write one truth row from the values :func:`row_values` derived."""
        self.db.put_compiled(
            key, instance, values.visible_rle, values.occlusion_ratio, values.visibility,
            values.placement, status, input_hash, verified_by=verified_by,
            geom_type=values.geom_type, box=values.box,
        )

    @staticmethod
    def _box_mask(box, hw: tuple[int, int]) -> np.ndarray:
        """A filled rectangle clipped to the canvas, ``x1``/``y1`` exclusive."""
        height, width = int(hw[0]), int(hw[1])
        mask = np.zeros((height, width), dtype=bool)
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
        y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
        return mask

    @staticmethod
    def _payload_area(payload: Optional[dict]) -> int:
        """How many pixels one conflict side covers -- its ``sym_diff_px`` alone."""
        geom_type, visible_rle, box = payload_geometry(payload)
        if geom_type == GEOM_BOX and box is not None:
            return int(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
        return 0 if visible_rle is None else masks.area(masks.decode_rle(visible_rle))

    def _queue_conflict(
        self,
        key: FrameKey,
        instance: str,
        old_payload: Optional[dict],
        new_payload: Optional[dict],
        pixels: int,
        queued: Optional[list[dict]],
        overtaken: Optional[list[tuple[int, str, str]]] = None,
    ) -> tuple[list[dict], bool]:
        """Make this disagreement the one open conflict of ``(step, instance)``.

        **One open conflict per frozen row** (task U7).  The open queue is
        looked at for this row only:

        * an open conflict with the **same new value** -- geometry and labels,
          :meth:`_geom_id` -- is this disagreement already: nothing is
          inserted, and refreshing the same frame twice piles nothing up;
        * every open conflict with a **different** new value is about a value
          the inputs no longer produce.  It is closed ``superseded`` and
          recorded in ``overtaken`` as ``(id, instance, OVERTAKEN)``, for the
          caller to log once per refresh (:meth:`_log_superseded`); with no
          ``overtaken`` list given, that row is written here.  Before U7 each
          one stayed open next to the newer one: every commit of a re-traced
          shape queued one more conflict for the same frozen row, thirteen of
          them on D13/scan frame 41;
        * a second open conflict with the **same** new value -- only a
          database from before U7 holds one -- is closed the same way with the
          reason :data:`DUPLICATE`; the oldest stays.

        A conflict a human already resolved is not in the open queue, so it
        may legitimately be raised again -- the resolution writes the inputs
        it settled on, and an identical conflict coming back means the inputs
        moved again.

        Returns ``(the queue it read, whether a row was inserted)``. The queue
        comes back -- without the conflicts closed here, with the one inserted
        -- so one refresh reads it at most once; the flag comes back because a
        caller that tells the annotator what happened must not say "queued
        again" about an insert the deduplication suppressed.  Called inside the
        caller's transaction.
        """
        if queued is None:
            queued = self.db.conflicts(key.desktop, key.view, open_only=True)
        wanted = self._geom_id(new_payload)
        mine = [c for c in queued
                if int(c["step"]) == int(key.step) and c["instance"] == instance]
        same = next((c for c in mine if self._geom_id(c["new_rle"]) == wanted), None)
        replaced = [c for c in mine if c is not same]
        if replaced:
            closed = []
            for conflict in sorted(replaced, key=lambda c: int(c["id"])):
                cid = int(conflict["id"])
                self.db.resolve_conflict(cid, SUPERSEDED)
                closed.append((cid, instance,
                               DUPLICATE if self._geom_id(conflict["new_rle"]) == wanted
                               else OVERTAKEN))
            gone = {cid for cid, _instance, _why in closed}
            queued[:] = [c for c in queued if int(c["id"]) not in gone]
            if overtaken is None:
                self._log_superseded(key, closed)
            else:
                overtaken.extend(closed)
        if same is not None:
            return queued, False
        cid = self.db.add_conflict(key, instance, old_payload, new_payload, int(pixels))
        queued.append({"id": int(cid), "step": int(key.step), "instance": instance,
                       "new_rle": new_payload})
        return queued, True

    @staticmethod
    def _geom_id(payload: Optional[dict]) -> Optional[str]:
        """The cheapest identity of a conflict side: its geometry and its labels.

        The labels are part of the identity because two disagreements about the
        same unchanged outline -- "it is occluded_partial", then "no, it is
        too_small" -- are two different things to decide, and deduplicating them
        onto one queue entry would lose the second.
        """
        if not payload:
            return None
        if "box" in payload:
            geom: Optional[str] = "box:" + ",".join(
                f"{float(v):.3f}" for v in payload["box"]
            )
        else:
            geom = masks.rle_counts(payload)
        labels = payload_labels(payload)
        if not labels:
            return geom
        tag = ";".join(f"{c.get('field')}={c.get('new')}" for c in labels)
        return f"{geom or ''}|{tag}"
