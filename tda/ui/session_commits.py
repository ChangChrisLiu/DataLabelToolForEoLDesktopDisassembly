"""Every gesture that writes, as the session exposes it to the window.

:mod:`tda.ui.session_edit` holds the rules; this is the half that knows *which
frame is open* and *what is being drawn* -- the editing layer, the refusals that
depend on the frame (no image, a part on the bench), the scope suggestion and
the preview of an edit's reach.  It is a mixin rather than a module of functions
because every one of these is about the session's own state.

Split off :mod:`tda.ui.session` only to keep both files readable; the methods
are part of :class:`~tda.ui.session.AnnotationSession`'s public surface and of
the :class:`~tda.ui.session_api.SessionLike` protocol.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

from tda.core.model import FrameKey, is_provisional
from tda.core.split_rows import CUT, keyframe_at
from tda.core.truth_inputs import frame_hw
from tda.ui import session_api as api
from tda.ui.commands import Op
from tda.ui.gallery import gallery_refusal
from tda.ui.session_api import SessionRefusal
from tda.ui import session_edit as edit
from tda.ui.session_ops import GEOM_BOX, ON_BENCH, annotatable_steps, keyframe_state, settle
from tda.ui.session_tasks import has_bench_roi

__all__ = ["CUT_COPY_UNCHANGED", "CommitMixin", "NOT_A_CUT_COPY"]

#: ``Ctrl+K`` with nothing drawn on a frame that shows a cut copy (U11 round 2):
#: it would confirm a placeholder, so it is refused and the explicit key named.
CUT_COPY_UNCHANGED = ("这帧上是另一状态的形状副本：画好这一帧的样子再 Ctrl+K；"
                      "形状确实没变就按 Ctrl+Shift+K（这帧就是这个形状）")
#: ``Ctrl+Shift+K`` where the shape in force is not a cut copy.
NOT_A_CUT_COPY = "{instance} 在这一帧的形状不是切开时留下的副本，不用确认"
#: ``Ctrl+Shift+K`` with pixels painted and not committed.
CONFIRM_NEEDS_CLEAN_LAYER = ("正在画的像素还没提交：画的就是这一帧的样子按 Ctrl+K；"
                             "要确认副本先 Esc 放弃")


def _loggable_extra(extra: Optional[dict]) -> Optional[dict]:
    """Check a caller's op-log note before anything is written.

    The op log is JSON, so a value that cannot be serialised would fail deep
    inside the write transaction -- after the keyframe, with the annotator's
    pixels half in the database. Raising here costs one ``json.dumps`` of a
    handful of flags and leaves the editing layer exactly as it was.
    """
    if extra is None:
        return None
    if not isinstance(extra, dict):
        raise ValueError(f"extra must be a dict, got {type(extra).__name__}")
    try:
        # allow_nan=False: NaN and the infinities are Python's extension to
        # JSON, not JSON, and a reader anywhere else refuses the file the op
        # log would have become
        json.dumps(extra, allow_nan=False)
    except (TypeError, ValueError) as bad:
        raise ValueError(f"extra must be JSON-serialisable: {bad}") from bad
    return dict(extra)


def _refuse_draft(instance: str) -> None:
    """Refuse to draw on a Label Studio draft key (spec 3.2).

    Its keyframes are the record of what the team traced before this tool
    existed; a commit would rewrite one of them in place, still stamped
    ``source="labelstudio"``, and the draft nobody has adopted yet would quietly
    become somebody's annotation.
    """
    if is_provisional(instance):
        raise SessionRefusal(
            f"{instance}：Label Studio 草稿不可直接编辑，"
            f"请先在 S1 中把草稿指派给真实实例"
        )


class CommitMixin:
    """The editing layer and the four edit scopes, for :class:`AnnotationSession`."""

    def _editable_frame(self) -> FrameKey:
        """The current frame, refusing one there is nothing to draw on.

        A step flagged ``missing``, or whose image never made it into the cache,
        has no canvas: its state and its shape anchors are real (spec 4.2
        缺帧处理) but a mask drawn "on" it would be in nobody's coordinates.
        """
        key = self.current()
        if key.step not in self._available or self.image_path(key.step) is None:
            raise SessionRefusal(
                f"step {key.step} of {self.view} has no image: it cannot be annotated"
            )
        return key

    def begin_edit(self, instance: str) -> None:
        """Load the instance's amodal shape into the editing layer (spec 4.3).

        Refuses a Label Studio draft key: its keyframes are the record of what
        the team traced before this tool existed, and a commit would rewrite one
        of them in place, under its own ``labelstudio`` source. A draft is
        resolved onto a real instance in S1 first (spec 3.2); only then is there
        something to draw on.
        """
        key = self._editable_frame()
        _refuse_draft(instance)
        found = self.compiled().instances.get(instance)
        self.layer.begin(instance, None if found is None else found.amodal,
                         frame_hw(self.db, key, self.truth.cache_dir))

    @property
    def editing_instance(self) -> Optional[str]:
        """The instance being drawn, or ``None``."""
        return self.layer.instance

    @property
    def editing_id(self) -> Optional[int]:
        """Identity of the edit being made, or ``None`` when none is open.

        One ``begin_edit`` to the ``clear_edit``/commit that ends it.  The undo
        history is per view and outlives every edit in it, so this is what
        tells "the pixels I am about to commit" from "the pixels I committed
        here an hour ago".
        """
        return self.layer.edit_id

    def editing_mask(self) -> Optional[np.ndarray]:
        """The editing layer as the session last saw it, or ``None``.

        The array belongs to the session: the window paints into its own overlay
        buffer and hands the result over with :meth:`set_editing_mask`.
        """
        return self.layer.mask()

    def set_editing_mask(self, mask: np.ndarray) -> None:
        """Take a copy of the layer the window has been painting into."""
        self.layer.set(mask)

    def erased_mask(self) -> Optional[np.ndarray]:
        """Pixels this edit has deliberately taken off, or ``None`` (E1)."""
        return self.layer.erased

    def set_erased_mask(self, erased: Optional[np.ndarray]) -> None:
        """Replace the protected set without recording an op."""
        self.layer.set_erased(erased)

    def push_stroke(self, before: np.ndarray, after: np.ndarray,
                    adopted: Optional[dict] = None,
                    erased: Optional[np.ndarray] = None) -> None:
        """Record one brush/eraser stroke on the undo stack (spec 4.6).

        The pixels are already painted, so the op is logged rather than applied;
        undoing it hands the earlier mask back on :attr:`sigEditingChanged`.

        ``adopted`` names the Label Studio draft a stroke came from, for the
        commit that later reads its own provenance off this history.
        """
        self.undo_stack.push(self.layer.stroke_op(before, after, adopted, erased),
                             apply=False)
        self._refresh_dirty()

    def adoptions_in_history(self) -> list[dict]:
        """The ``adopted`` notes of every stroke that is applied right now.

        Oldest first, undone strokes left out -- the undo stack's own account
        of which drafts are actually in the layer.
        """
        out = []
        for op in self.undo_stack.ops:
            adopted = (op.payload or {}).get("adopted") if op.kind == "edit_editing_mask" else None
            if adopted:
                out.append(dict(adopted))
        return out

    def clear_edit(self) -> None:
        """Drop the editing layer without writing anything."""
        self.layer.clear()

    def commit_edit(self, scope: str, direction: str = edit.REVERSE, *,
                    extra: Optional[dict] = None) -> dict:
        """Write the editing layer back with the scope the annotator chose.

        ``extra`` is merged into the op-log payload of this commit: what the
        *window* knows about why it was made and the session cannot derive --
        "the annotator was warned the shape is implausibly large and went
        ahead". It belongs in the audit trail of that commit rather than in a
        log nobody joins back to it. Values must be JSON-serialisable, because
        the op log is JSON; anything else raises :class:`ValueError` **before**
        anything is written. Keys the payload already uses are not overwritten.

        ``scope`` is one of :data:`tda.ui.session_api.COMMIT_SCOPES`, or one of
        the two layering answers :func:`tda.ui.session_edit.suggest_scope` gives
        -- ``zorder:above:<B>`` / ``zorder:below:<B>`` -- which write a
        ``PairOverride`` instead of pixels (spec 4.3 改层级).

        A pixel scope with nothing changed is a no-op: loading a shape and
        pressing Enter must not mint a new version of it.  The one exception is
        ``Ctrl+K`` on an open ✂ row of this frame (:meth:`open_split_here`):
        there the boundary *is* the edit, and the result says
        ``unchanged_split``.  An explicit layering scope is always honoured,
        because there the pixels are not the point.

        A layering commit never discards painted pixels.  The editing layer is
        the instance's *amodal* shape, so pixels added inside ``B`` are by
        definition not in ``A``'s shape yet and no order change can make them
        ``A``'s: ``zorder:above:<B>`` with added pixels therefore re-traces the
        keyframe **and** writes the ``PairOverride``, as one undoable op.  The
        eraser direction is different: the pixels erased lie in ``A ∩ B`` and
        saying "B was on top all along" merely hides them, so ``A``'s shape is
        left exactly as it was.
        """
        if not self.layer.active:
            raise RuntimeError("commit_edit() needs begin_edit() first")
        extra = _loggable_extra(extra)
        key, instance = self._editable_frame(), self.layer.instance
        if getattr(self, "gallery", False) and (
                scope == api.SCOPE_SPLIT or edit.splits_the_shape(scope)):
            # a components-gallery shot is one segment of its own (task U10):
            # there is no "from this frame on", only this shot
            raise SessionRefusal(gallery_refusal(key.view, "commit_split"))
        pair = edit.split_zorder_scope(scope)
        unchanged_split = False
        if pair is None and not self.layer.changed():
            if not (scope == api.SCOPE_SPLIT and self.open_split_here(instance)):
                return {"changed": False, "affected": [], "conflicts": 0, "problems": {},
                        "scope": scope}
            if self.cut_copy_here(instance):
                # the row is open *because* the frame shows a placeholder:
                # an untouched Ctrl+K would sign it as this frame's shape
                raise SessionRefusal(CUT_COPY_UNCHANGED)
            # An open ✂ row whose shape is already right on this side (U8
            # round 2b): the clip closed at 39-41 before the board takes it out
            # at 42 leaves the ✂ at 38, where the open shape loaded is the open
            # shape. Ctrl+K cuts the version boundary with it unchanged, so a
            # later Enter on the far side reaches only that state's frames.
            unchanged_split = True
        if pair is not None and self._added_pixels(pair):
            other, _above = pair
            edit.require_instance(self._known_instances(), other)
            self._refuse_mask_on_bench(key, instance)
            result = edit.commit_edit(self.db, self.truth, key, instance,
                                      self.layer.mask(), self._shape_scope(scope),
                                      direction, self.annotator, pair=other,
                                      extra=extra)
            self.layer.settle()   # the pixels went to the database with the pair
        elif pair is not None:
            other, above = pair
            result = edit.commit_pair_override(
                self.db, self.truth, key,
                instance if above else other, other if above else instance,
                self.annotator, known=self._known_instances(), extra=extra,
            )
        else:
            self._refuse_mask_on_bench(key, instance)
            if scope == api.SCOPE_FRAME_OVERRIDE and not self.keyframe_applies(instance):
                # The override would be written and the part still compiled
                # as missing: its row open, Space refused (U2d).
                raise SessionRefusal(api.OVERRIDE_NEEDS_SHAPE)
            result = edit.commit_edit(self.db, self.truth, key, instance, self.layer.mask(),
                                      scope, direction, self.annotator, extra=extra)
            # What was just written is no longer uncommitted: the layer's
            # baseline moves to the mask that went to the database, so the guard
            # on goto/open/close sees a settled layer rather than refusing to
            # leave a frame whose work is already saved.
            #
            # A pure layering scope -- the eraser direction above -- deliberately
            # does **not** settle: it writes a PairOverride and no pixels, so the
            # erased ones are still nobody's but the annotator's to keep or
            # discard.  The window clears them after such a commit; a direct API
            # user is left holding them rather than having them silently marked
            # as written.
            self.layer.settle()
        if unchanged_split:
            result["unchanged_split"] = True
        return self._after_edit(result)

    def cut_copy_here(self, instance: Optional[str]) -> bool:
        """Is the version of ``instance`` in force on this frame a cut copy (U11 r2)?"""
        if not instance or not self.is_open:
            return False
        found = keyframe_at(self.db, self.tax, self.current(), str(instance))
        return found is not None and found.source == CUT

    def confirm_cut_copy(self, instance: Optional[str] = None) -> dict:
        """「这帧就是这个形状」: the cut copy in force here *is* this frame's shape.

        ``Ctrl+Shift+K`` (U11 round 2). Some state changes do not move the
        outline -- a latch seen edge-on -- and the copy a commit left on the
        far side of the boundary is then right as it is. This takes its mark
        off without touching a pixel, so its ✂ row is done and ``cli check``
        stops counting the frames under it. Like every manual-shape write it
        stamps ``source="manual"`` and drops the draft reference. One
        ``commit_keyframe`` op: the row before and after, ``Ctrl+Z`` puts the
        mark (and the reference) back. Refused on a components-gallery shot.

        Refused when the shape in force is not a cut copy, and while painted
        pixels are waiting in the editing layer (they are either this frame's
        shape -- ``Ctrl+K`` -- or not wanted). An untouched layer on the same
        part is closed.
        """
        if getattr(self, "gallery", False):
            # a shot has no Ctrl+K, so no copy -- refused all the same (U10/U11)
            raise SessionRefusal(gallery_refusal(self.view, "confirm_cut_copy"))
        target = str(instance or self.layer.instance or "")
        if not target:
            raise SessionRefusal("先在实例表选中零件（或点任务卡上那一条）")
        key = self._editable_frame()
        if self.layer.active and self.layer.changed():
            raise SessionRefusal(CONFIRM_NEEDS_CLEAN_LAYER)
        kf = keyframe_at(self.db, self.tax, key, target)
        if kf is None or kf.source != CUT:
            raise SessionRefusal(NOT_A_CUT_COPY.format(instance=target))
        if self.layer.active:
            self.layer.clear()
        ref = {"keyframe_id": kf.id}
        before = keyframe_state(kf, ref)
        with self.db.transaction():
            # the annotator's shape now, like every other manual-shape path:
            # source manual, no draft reference (spec 3.1 来源 / 初稿引用)
            kf.source, kf.draft_id = edit.SOURCE_MANUAL, None
            self.db.update_keyframe(kf)
            after = keyframe_state(kf, ref)
            steps = annotatable_steps(
                self.db, key.desktop, key.view,
                set(self.truth.affected_steps(key.desktop, key.view, target, kf))
                | {key.step})
            common = {"desktop": key.desktop, "view": key.view, "step": key.step,
                      "instance": target, "scope": "confirm_cut_copy",
                      "direction": self.browsing, "steps": steps,
                      "zorder": None, "pair": None}
            payload = common | {"keyframes": [after]}
            inverse = common | {"keyframes": [before]}
            self.db.log_op(key.desktop, key.view, "commit_keyframe",
                           edit._loggable(payload), edit._loggable(inverse),
                           self.annotator)
        stats = settle(self.db, self.truth, key.desktop, key.view, steps, key.step)
        result = {"changed": True, "confirmed_cut": int(kf.anchor_step),
                  "affected": steps, "compiled": stats["compiled"],
                  "rechecks": stats["rechecks"], "conflicts": stats["conflicts"],
                  "problems": stats["problems"], "frame": stats["frame"],
                  "op": Op("commit_keyframe", payload, inverse)}
        return self._after_edit(result)

    def open_split_here(self, instance: Optional[str]) -> bool:
        """Does this frame's task card hold an open ✂ row for ``instance``?

        The one case where ``Ctrl+K`` on an untouched layer writes: the
        version boundary itself is the work (U8 round 2b).
        """
        if not instance:
            return False
        return any(row.get("instance") == instance
                   and row.get("kind") == api.KIND_SPLIT_KEYFRAME and not row.get("done")
                   for row in self.task_card())

    @staticmethod
    def _shape_scope(scope: str) -> str:
        """Which pixel scope the shape half of a layering commit is written with.

        ``Enter`` re-traces the keyframe in force; ``Ctrl+K`` on the same
        suggestion (``split+zorder:above:<B>``) cuts a new version from here and
        records the pair with it -- a split that dropped the pair wrote the
        pixels and left them hidden under ``B``, so nothing on screen moved.
        """
        return api.SCOPE_SPLIT if edit.splits_the_shape(scope) else api.SCOPE_KEYFRAME

    def _added_pixels(self, pair: tuple[str, bool]) -> bool:
        """Does this layering gesture carry pixels the shape does not have yet?

        Only the "go above" direction can: ``added = edited & ~before`` is what
        the annotator painted *into* the neighbour, and it is exactly what an
        order change on its own would throw away.
        """
        _other, above = pair
        if not above:
            return False
        mask, before = self.layer.mask(), self.layer.before()
        if mask is None or before is None:
            return False
        return bool(np.any(np.asarray(mask, dtype=bool)
                           & ~np.asarray(before, dtype=bool)))

    def _known_instances(self) -> set[str]:
        """The instances this frame has, which a layering gesture may name."""
        return set(self.compiled().instances)

    def keyframe_applies(self, instance: str) -> bool:
        """Does a keyframe of ``instance`` apply at the open frame?

        The compiler's own answer (``keyframe_id`` of the compiled instance),
        so it is exactly the question behind ``missing_shape``: a frame
        override is drawn *on* an existing shape, and one with no keyframe
        under it leaves the part missing however many pixels it holds (U2d).
        The window asks it to grey ``Alt+Enter``; :meth:`commit_edit` refuses
        with the same sentence.
        """
        found = self.compiled().instances.get(str(instance))
        return found is not None and found.keyframe_id is not None

    def _refuse_mask_on_bench(self, key: FrameKey, instance: str) -> None:
        """A part on the bench is tracked by a rectangle, not by a mask (spec 4.2)."""
        placement = edit.placement_of(self.db, self.tax, key, instance)
        if placement == ON_BENCH:
            raise SessionRefusal(
                f"{instance} is on the bench at step {key.step}: use the bench box tool"
            )

    def commit_box(self, instance: str, box, direction: str = edit.REVERSE) -> dict:
        """Draw the staging-area rectangle of a part on the bench (spec 4.2 S4).

        Refused on a view with no staging-area ROI: a part that has been taken
        out is not in that picture at all (spec 4.2 item 1), so the keyframe
        would be one the compiler never selects, on a frame the part is not in.
        """
        key = self._editable_frame()
        _refuse_draft(instance)  # the one write that names its own instance
        if not has_bench_roi(self.db, key):
            raise SessionRefusal(
                f"本视图没有堆放区 ROI，不能画桌面框：{self.view} 看不到堆放区，"
                f"{instance} 离开机箱后不在这个画面里"
            )
        result = edit.commit_box(self.db, self.truth, key, instance, box,
                                 direction=direction, annotator=self.annotator)
        return self._after_edit(result)

    def commit_occluder(self, mask: np.ndarray, occluder_type: str = "hand") -> dict:
        """Store one occluder layer of the current frame (spec 4.2 step 4)."""
        result = edit.commit_occluder(self.db, self.truth, self._editable_frame(), mask,
                                      occluder_type, self.annotator)
        return self._after_edit(result)

    def preview(self, scope: str, direction: str = edit.REVERSE) -> dict:
        """How far the pending edit would reach, before it is written (spec 4.3).

        ``{"steps", "verified_steps"}`` -- the frames the scope would change and
        the already-confirmed ones among them, i.e. the "影响 N 帧 / 将产生 N 个
        冲突" strip.  Nothing is written and no pixels are touched.

        A layering answer that carries new pixels writes a keyframe as well, so
        its reach is the union of both: the pair holds where both instances are
        mask layers, the shape wherever its keyframe does.
        """
        if self.editing_instance is None:
            return {"steps": [], "verified_steps": []}
        pair = edit.split_zorder_scope(scope)
        if pair is not None:
            other, _above = pair
            edit.require_instance(self._known_instances(), other)
            reach = edit.preview_pair(self.db, self.truth, self.current(),
                                      self.editing_instance, other)
            if not self._added_pixels(pair):
                return reach
            shape = edit.preview(self.db, self.truth, self.current(),
                                 self.editing_instance, self._shape_scope(scope),
                                 direction)
            return {"steps": sorted(set(reach["steps"]) | set(shape["steps"])),
                    "verified_steps": sorted(set(reach["verified_steps"])
                                             | set(shape["verified_steps"]))}
        geom = GEOM_BOX if edit.placement_of(
            self.db, self.tax, self.current(), self.editing_instance
        ) == ON_BENCH else None
        return edit.preview(self.db, self.truth, self.current(), self.editing_instance,
                            scope, direction, **({"geom_type": geom} if geom else {}))

    def retrace_reach(self, instance: Optional[str] = None) -> Optional[dict]:
        """How many confirmed frames re-tracing this shape here would put in conflict (U7).

        ``{"steps", "frozen"}`` for the instance being edited (or ``instance``)
        when a keyframe of it applies at the open frame: the frames ``Enter``
        would rewrite -- :meth:`preview` with keyframe scope, the same chain
        rule -- and the frozen ones among them (the frame confirmed, or a row
        of it signed).  ``None`` for a shape drawn here for the first time:
        that adds a part, it re-traces nothing.

        On D13/scan a re-trace of four confirmed clips at frame 41 put frames
        17-41 in conflict and nothing said so before the commit; the task card
        says it now, from this.  A few state lookups and one query.
        """
        target = instance or self.editing_instance
        if target is None or not self.is_open or not self.keyframe_applies(target):
            return None
        key = self.current()
        geom = GEOM_BOX if edit.placement_of(self.db, self.tax, key, str(target)) == ON_BENCH \
            else edit.GEOM_MASK
        reach = edit.preview(self.db, self.truth, key, str(target), api.SCOPE_KEYFRAME,
                             geom_type=geom)
        return {"steps": list(reach["steps"]), "frozen": list(reach["verified_steps"])}

    def set_visibility(self, instance: str, vis: str) -> None:
        """Override one instance's visibility label on this frame (spec 6.2)."""
        self._after_edit(
            edit.set_visibility(self.db, self.truth, self.current(), instance, vis,
                                self.annotator)
        )

    def clear_visibility(self, instance: str) -> None:
        """Take this frame's hand-set visibility label of ``instance`` back (undoable)."""
        self._after_edit(
            edit.clear_visibility(self.db, self.truth, self.current(), instance,
                                  self.annotator)
        )

    def set_hidden(self, instance: str, hidden: bool) -> None:
        """Show or hide an instance in the canvas.

        A view setting, not data: it is never written to the database and never
        logged, and it is forgotten when another desktop/view is opened.
        """
        if hidden:
            self._hidden.add(instance)
        else:
            self._hidden.discard(instance)

    def set_zorder_move(self, instance: str, above_of: str) -> None:
        """Move ``instance`` directly above ``above_of`` in the layer order."""
        self._after_edit(
            edit.set_zorder_move(self.db, self.truth, self.current(), instance, above_of,
                                 self.annotator, known=self._known_instances())
        )

    def suggest_scope(self, edited: Optional[np.ndarray] = None) -> str:
        """The scope the pending edit would default to (spec 4.3).

        Answered from the pixels that *changed* since :meth:`begin_edit`, so
        opening a shape and touching nothing suggests nothing.
        """
        if not self.layer.active:
            return api.SCOPE_KEYFRAME
        mask = self.layer.mask() if edited is None else edited
        return edit.suggest_scope(self.compiled(), self.layer.instance,
                                  self.layer.before(), mask, hosts=self.scope_hosts())

    def scope_hosts(self) -> dict:
        """``instance -> what it sits on`` for this desktop (task U6).

        :func:`tda.ui.session_scope.hosts_of` over the instance table: the
        relation pointers, the taxonomy's ``host_class`` and the chassis.  Read
        per suggestion -- one small table, and the Steps mode can change it.

        None on a components-gallery shot (task U10): the parts lie side by
        side or across each other on the table, and what sits on what inside
        the machine says nothing about that -- the layering suggestion there is
        the pixels' alone, and ``L`` answers it as anywhere.
        """
        if getattr(self, "gallery", False):
            return {}
        return edit.hosts_of(self.db.instances(self.desktop), self.tax)
