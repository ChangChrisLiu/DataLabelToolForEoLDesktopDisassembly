"""The task card: what has to be annotated on the frame in front of the annotator.

Reverse-order annotation (spec 4.2) arrives at frame ``j`` from ``j + 1``, which
is already done.  The work that belongs to ``j`` is the difference between the
two, read on the image the annotator is looking at: a part that is ``removed``
at ``j + 1`` and installed at ``j`` is a part they can *see* there and have to
draw, a latch that is ``open`` at ``j + 1`` and shut at ``j`` needs a second
version of its shape, and a screw that was merely loosened needs no new pixels
at all.

Everything is therefore expressed in two frames: **``j``, the frame on screen**,
and **``neighbour``, the frame it is diffed against** -- ``j + 1`` browsing
backwards, ``j - 1`` browsing forwards for a repair, and the next one that
actually has an image when the one next door is missing.  ``done`` is always
evaluated at ``j``: the question is whether the geometry the item asks for
exists *here*.

The **start frame** has no neighbour.  Its card is one item per instance that
needs geometry there, the open ones first -- ordered so the chassis is drawn
before the things that sit on top of it -- and the drawn ones after them as ✔
rows (U2b round 3).

So does **the first frame of a pose segment** (task U37).  A ``reorient`` step
or a pose break of the view starts a new segment, and no keyframe reaches
across one (spec 2.5): on D48 the walk came down from frame 36, the flipped
chassis, to frame 35, and nothing drawn on 36 applies there.  The diff against
36 was empty -- a flip moves no part -- so the card said "nothing to draw" on a
frame with twelve shapes missing.  A neighbour in another segment is therefore
no neighbour: the card is the start card, in either direction, and every row
carries :data:`SEGMENT_START` saying where the cut is, so the panel and the
guide can say why.  A start frame is one whatever its step type (round 2): the
single confirmation of a ``failed`` / ``dupli`` step means "nothing changed
next door", which only holds inside a segment.  D34 ends on a failed step, and
that card was the only one that could ever have asked for its chassis.

A part said **画框外** on the frame (task U39: in the machine, outside this
view's picture -- ``Ctrl+```'s fourth choice, the gallery's 「看不见」 row) has
nothing to draw there: its row is done and carries :data:`OFF_FRAME_KEY`, so
the panel says 「画框外」 instead of asking for a shape.

Bench work is asked for only where there is a bench to see.  Spec 4.2 says
若该视角有堆放区 ROI -- *if this view has a staging-area ROI* -- and the scanner
looks straight down at the board, so on it the parts that have been taken out
are simply not in the picture.  ``add_bench_box`` and ``remove_bench_box`` are
therefore emitted only when the frame's pose segment has one recorded, and they
carry their own kind so the window arms the box tool rather than the brush.
"""
from __future__ import annotations

from typing import Optional

from tda.core.compiler import marked_absent, select_keyframe
from tda.core.db import Db
from tda.core.model import FrameKey, InstanceRec, StepType, is_component_view
from tda.core.states import diff_states, needs_geom
from tda.core.split_rows import (  # noqa: F401 - re-exported, the card's vocabulary
    NOTHING_VISIBLE,
    SHAPE_PREDATES_STATE,
    SPLIT_TRANSITIONS,
    STATE_ONLY_TRANSITIONS,
    has_split,
    keyframe_at,
    kind_for,
    split_state,
    stale_split_rows,
)
from tda.core.taxonomy import Taxonomy
from tda.core.truth_inputs import (
    InputCache,
    instances_of,
    pose_segment_of,
    same_segment,
    state_of,
)
from tda.ui import session_api as api
from tda.ui.session_ops import (
    BENCH_KINDS,
    GEOM_BOX,
    GEOM_MASK,
    IN_CHASSIS,
    ON_BENCH,
    chain_for,
    placement_of,
)

__all__ = ["LAYER_RANK", "OFF_FRAME_KEY", "SEGMENT_START", "SHAPE_PREDATES_STATE",
           "SPLIT_TRANSITIONS", "STATE_ONLY_TRANSITIONS", "bench_roi_of", "gallery_card",
           "has_bench_roi", "item_text", "same_segment", "segment_cut", "stale_split_rows",
           "task_card_for"]

REMOVED = "removed"
_VERIFIED = "verified"
_REORIENT = StepType.REORIENT.value

#: The key a start card's rows carry when the frame is the first of its pose
#: segment rather than the start of the walk (task U37): :func:`segment_cut`'s
#: ``{"cut": k, "reorient": bool}``.
SEGMENT_START = "segment_start"
#: The key a row carries when its part is 画框外 on the frame (task U39).
OFF_FRAME_KEY = "off_frame"
#: The rows 画框外 settles: the ones asking for a shape (or its new version).
_SHAPE_KINDS = (api.KIND_ADD_SHAPE, api.KIND_SPLIT_KEYFRAME)

#: Taxonomy group -> how low it sits in a frame, so the start frame's card can
#: be drawn from the bottom up: the chassis is behind everything, the parts
#: bolted into it come next, and the small things that fasten or connect them
#: are on top. Only the order matters, not the numbers.
LAYER_RANK: dict[str, int] = {
    "structure": 0,
    "part": 1,
    "interface": 2,
    "latch": 3,
    "fastener": 4,
}
_CHASSIS_CLASS = "chassis"

#: Order the kinds appear in for one instance: draw first, confirm last.
_KIND_RANK = {
    api.KIND_ADD_SHAPE: 0,
    api.KIND_SPLIT_KEYFRAME: 1,
    api.KIND_STATE_ONLY: 2,
    api.KIND_REMOVE_BENCH_BOX: 3,
    api.KIND_CONFIRM: 4,
}


# --------------------------------------------------------------------------- #
# what already exists
# --------------------------------------------------------------------------- #
def _has_shape(db: Db, tax: Taxonomy, desktop: int, view: str, instance: str, step: int,
               cache: Optional[InputCache] = None,
               geom_type: Optional[str] = None) -> bool:
    """Does a keyframe of the right chain apply to ``instance`` at ``step``?"""
    key = FrameKey(desktop, step, view)
    seg = pose_segment_of(db, key, cache)
    placement = placement_of(db, tax, key, instance, cache)
    chosen = select_keyframe(chain_for(db, key, instance, seg, placement), step)
    if chosen is None:
        return False
    return geom_type is None or chosen.geom_type == geom_type


#: The card's own decisions live in :mod:`tda.core.split_rows` (U8 round 3):
#: ``cli check`` and the commit rule read them without Qt.
_keyframe_at = keyframe_at
_has_split = has_split


def _has_bench_chain(db: Db, desktop: int, view: str, instance: str) -> bool:
    """Has anybody drawn a staging-area box for this instance in this view?"""
    return any(kf.placement == ON_BENCH for kf in db.keyframes(desktop, view, instance))


def bench_roi_of(db: Db, key: FrameKey, cache: Optional[InputCache] = None):
    """The staging area this view can see at this frame, or ``None``.

    What :func:`tda.core.states.needs_geom` wants handed to it, so the card asks
    for exactly the geometry the compiler will look for.
    """
    return db.bench_roi(key.desktop, key.view, pose_segment_of(db, key, cache))


def has_bench_roi(db: Db, key: FrameKey, cache: Optional[InputCache] = None) -> bool:
    """Can this view see a staging area at this frame? (spec 4.2 item 1)

    Without one there is nothing to box: a part that has been taken out is not
    in the picture at all, and asking for thirty rectangles nobody can draw is
    how a task card stops being read.
    """
    return bench_roi_of(db, key, cache) is not None


# --------------------------------------------------------------------------- #
# one change -> one instruction
# --------------------------------------------------------------------------- #
#: The card's kind decision (:func:`tda.core.split_rows.kind_for`); its strings
#: are :mod:`tda.ui.session_api`'s ``KIND_*`` values.
_kind_for = kind_for


def item_text(kind: str, instance: str, changes: dict, rec: Optional[InstanceRec],
              span: Optional[list[int]] = None) -> str:
    """The one-line instruction the task card shows.

    The one formatter: :meth:`AnnotationSession.confirm_frame` says a missing
    shape in exactly these words too, so a panel can match a problem to the item
    it belongs to instead of guessing.
    """
    cls = "" if rec is None else f" ({rec.cls})"
    transition = changes.get("state")
    arrow = "" if transition is None else f": {transition[0]} -> {transition[1]}"
    note = _span_note(span)
    if kind == api.KIND_ADD_SHAPE:
        with_parent = (f", back in with {rec.parent}"
                       if rec is not None and rec.attached and rec.parent else "")
        return f"Draw {instance}{cls} on this frame{with_parent}{note}"
    if kind == api.KIND_ADD_BENCH_BOX:
        return f"Draw the staging-area box of {instance}{cls}{note}"
    if kind == api.KIND_SPLIT_KEYFRAME:
        return f"Split the keyframe of {instance}{cls}{arrow}{note}"
    if kind == api.KIND_REMOVE_BENCH_BOX:
        return f"The staging-area box of {instance}{cls} ends here{note}"
    if kind == api.KIND_CONFIRM:
        nothing = changes.get("nothing")
        if nothing is not None:
            return (f"Nothing to draw here: no change against step {nothing[0]}"
                    f"{note} - confirm the frame")
        return f"Nothing to draw here{arrow}{note} - confirm the frame"
    return f"State only for {instance}{cls}{arrow}{note}"


def _span_note(span: Optional[list[int]]) -> str:
    """Say so when the card covers more than one action step (a missing frame)."""
    if not span or len(span) < 2:
        return ""
    skipped = ", ".join(f"step {s} has no image" for s in span[:-1])
    return f" (covers steps {span[0]}-{span[-1]}; {skipped})"


def _item(instance: str, kind: str, changes: dict, done: bool,
          rec: Optional[InstanceRec], span: Optional[list[int]] = None) -> dict:
    """One card row.

    Besides the four keys every reader relies on, a row carries what the card
    needs to explain itself in the annotator's words (task U2b): the class and
    its attributes (``cover`` of the CPU cooler is 导风罩), the names the step
    sheet used for the part, the parent it came back in with, and the state
    transition behind a split.  All of it is already in hand here, so saying
    it costs the card nothing it did not already pay.
    """
    row = {"instance": instance, "kind": kind,
           "text": item_text(kind, instance, changes, rec, span), "done": bool(done)}
    if rec is not None:
        row["cls"] = str(rec.cls)
        row["attrs"] = dict(rec.attrs or {})
        row["raw_names"] = [str(n) for n in (rec.raw_names or [])]
        if rec.attached and rec.parent:
            row["parent"] = str(rec.parent)
    transition = changes.get("state")
    if transition is not None:
        row["transition"] = [str(transition[0]), str(transition[1])]
    if span and len(span) > 1:
        row["span"] = [int(s) for s in span]
    return row


def _mark_off_frame(db: Db, key: FrameKey, items: list[dict]) -> list[dict]:
    """The rows of parts said 画框外 on ``key`` (task U39), in place.

    A row asking for a shape (✚) or its new version (✂) is done: there is
    nothing of the part in this picture to draw. That row and a state-only one
    carry :data:`OFF_FRAME_KEY`, so the panel says 「画框外」. The label is the
    compiler's :func:`~tda.core.compiler.marked_absent` -- ``out_of_view``
    with no pixels -- which also settles ``missing_shape`` and the ✂ question
    (:func:`tda.core.split_rows.split_state`), so the card agrees with ``Space``.
    """
    try:
        overrides = db.frame_overrides(key) or {}
    except Exception:  # noqa: BLE001 - a missing table is "none set"
        return items
    off = {name for name, fo in overrides.items() if marked_absent(fo)}
    if not off:
        return items
    for row in items:
        kind = row.get("kind")
        if row.get("instance") not in off or kind not in (*_SHAPE_KINDS,
                                                          api.KIND_STATE_ONLY):
            continue
        row[OFF_FRAME_KEY] = True
        if kind in _SHAPE_KINDS:
            row["done"] = True
    return items


def _ordered(items: list[dict], instances: dict[str, InstanceRec]) -> list[dict]:
    """Parent before the children that ride out of the chassis with it."""
    def sort_key(item: dict) -> tuple:
        rec = instances.get(item["instance"])
        parent = rec.parent if rec is not None and rec.attached and rec.parent else None
        return (parent or item["instance"], 0 if parent is None else 1,
                item["instance"], _KIND_RANK.get(item["kind"], 9))

    return sorted(items, key=sort_key)


def _bottom_up(items: list[dict], instances: dict[str, InstanceRec],
               tax: Taxonomy) -> list[dict]:
    """Order a start-frame card so the thing everything sits on is drawn first.

    Open rows first, drawn ones (✔) after them, each bottom-up.
    """
    def sort_key(item: dict) -> tuple:
        rec = instances.get(item["instance"])
        cls = "" if rec is None else rec.cls
        return (1 if item.get("done") else 0,
                0 if cls == _CHASSIS_CLASS else 1,
                LAYER_RANK.get(tax.group_of(cls), len(LAYER_RANK)),
                item["instance"])

    return sorted(items, key=sort_key)


# --------------------------------------------------------------------------- #
# the card
# --------------------------------------------------------------------------- #
def task_card_for(db: Db, tax: Taxonomy, desktop: int, view: str, step: int,
                  neighbour: Optional[int] = None,
                  span: Optional[list[int]] = None) -> list[dict]:
    """What has to be annotated on frame ``step`` (spec 4.2).

    ``neighbour`` is the already-annotated frame the card is diffed against --
    ``step + 1`` in the reverse order annotation runs in, ``step - 1`` when the
    annotator is browsing forwards to repair something, and ``None`` on the
    start frame, which has nobody after it.  A neighbour in another pose
    segment (:func:`same_segment`) counts as none: ``step`` is where its
    segment starts in this walk, nothing drawn next door applies to it, and the
    card is the start card with :data:`SEGMENT_START` on every row (task U37).
    Both are decided before the step type: a ``failed`` / ``dupli`` step gets
    its one-row confirmation only against a neighbour in its own segment.

    A part that is ``removed`` next door and installed here has to be drawn back
    into the chassis, together with the attached children that came back in with
    it, which is why the card groups a parent with them. ``done`` says whether
    the geometry the item asks for already exists **on this frame**, so the
    panel can grey an item out without the annotator having to remember.
    """
    if is_component_view(view):
        return gallery_card(db, tax, desktop, view, step)
    cache = InputCache()
    records = {rec.step: rec for rec in db.steps(desktop)}
    if step not in records:
        return []
    rec = records[step]
    instances = instances_of(db, desktop, cache)
    frame = db.get_frame(FrameKey(desktop, step, view)) or {}
    confirmed = frame.get("review_status") == _VERIFIED

    # The start comes first, whatever the step's type (U37 round 2): a start
    # frame that is a ``failed`` / ``dupli`` step still has every part to draw
    # -- D34 ends on 「Try to remove the optical drive」 (failed), and the
    # chassis, which never changes state, is only ever on a start card.
    if neighbour is None or neighbour not in records:
        return _start_card(db, tax, desktop, view, step, instances, cache, confirmed)
    if not same_segment(db, desktop, view, step, neighbour, cache):
        # a reorient or a pose break between them: this frame starts its
        # segment, and its shapes are drawn anew (task U37, D48 frame 35)
        return _start_card(db, tax, desktop, view, step, instances, cache, confirmed,
                           segment_cut(db, desktop, view, step, neighbour, records))
    if rec.dupli or rec.step_type in ("dupli", "failed"):
        # inside a segment nothing changed next door: one confirmation
        return [_item(rec.raw_name or f"step {step}", api.KIND_CONFIRM, {}, confirmed,
                      None, None)]

    span = span or [neighbour]
    bench_roi = bench_roi_of(db, FrameKey(desktop, step, view), cache)
    bench = bench_roi is not None
    state_here = state_of(db, tax, desktop, step, cache)
    state_there = state_of(db, tax, desktop, neighbour, cache)
    # the work this view is asked for, and the work there is: one owner, two
    # questions. The difference is exactly the staging area a view without a
    # bench ROI cannot see (spec 4.2 item 1), which is not its work.
    needs_here = needs_geom(instances, state_here, tax, bench_roi=bench_roi)
    needs_anywhere = needs_geom(instances, state_here, tax)
    needs_there = needs_geom(
        instances, state_there, tax,
        bench_roi=bench_roi_of(db, FrameKey(desktop, neighbour, view), cache),
    )

    changed: dict[str, dict] = {}
    for instance, attr, there, here in diff_states(state_there, state_here):
        changed.setdefault(instance, {})[attr] = (there, here)

    items: list[dict] = []
    for instance, changes in changed.items():
        if instance not in instances:
            continue  # a virtual cable node: it never carries geometry
        if instance in needs_anywhere and instance not in needs_here:
            continue  # on the bench, and this view cannot see one: not its work
        rec_i = instances.get(instance)
        kind = _kind_for(changes, instance in needs_there, instance in needs_here)
        cut_copy = False
        wants_box = needs_here.get(instance) in BENCH_KINDS
        if wants_box and kind == api.KIND_ADD_SHAPE:
            kind = api.KIND_ADD_BENCH_BOX
        if kind == api.KIND_STATE_ONLY:
            done = True
        elif kind == api.KIND_SPLIT_KEYFRAME:
            done, cut_copy = split_state(db, tax, desktop, view, instance, step, neighbour,
                                         cache)
        else:
            done = _has_shape(db, tax, desktop, view, instance, step, cache,
                              GEOM_BOX if wants_box else None)
        items.append(_item(instance, kind, changes, done, rec_i, span))
        if kind == api.KIND_SPLIT_KEYFRAME and cut_copy:
            # the frame shows a cut copy of the other state's shape: a
            # placeholder, drawn over with Ctrl+K or confirmed (U11 round 2)
            items[-1]["cut_copy"] = True
        if (bench and changes.get("placement") == (ON_BENCH, IN_CHASSIS)
                and _has_bench_chain(db, desktop, view, instance)):
            # the part is back in the chassis here, so its on_bench chain ended
            # next door -- done exactly when no bench box reaches this frame
            retired = not _has_shape(db, tax, desktop, view, instance, step, cache,
                                     GEOM_BOX)
            items.append(_item(instance, api.KIND_REMOVE_BENCH_BOX, changes, retired,
                               rec_i, span))
    if not items:
        return [_item(f"step {step}", api.KIND_CONFIRM,
                      {"nothing": (str(neighbour), str(step))}, confirmed, None, span)]
    # 画框外 (task U39): nothing of the part to draw in this picture
    _mark_off_frame(db, FrameKey(desktop, step, view), items)
    return _ordered(items, instances)


def segment_cut(db: Db, desktop: int, view: str, step: int, neighbour: int,
                records: Optional[dict] = None) -> dict:
    """Where ``step``'s pose segment ends towards ``neighbour`` (task U37).

    ``{"cut": k, "reorient": bool}``: ``k`` is the first step after the
    boundary next to ``step`` -- the end of ``step``'s own segment + 1 walking
    backwards, its start walking forwards, so a skipped frame in between does
    not move it -- and ``reorient`` says whether ``k`` is a ``reorient`` step
    of the step table (the chassis was turned for every view) rather than a
    pose break of this view only.
    """
    here = db.pose_segment_for(FrameKey(desktop, step, view)) or {}
    if int(neighbour) > int(step):
        end = here.get("end_step")
        cut = int(end) + 1 if end is not None and int(end) >= int(step) else int(neighbour)
    else:
        start = here.get("start_step")
        cut = int(start) if start is not None and int(start) <= int(step) else int(step)
    if records is None:
        records = {rec.step: rec for rec in db.steps(desktop)}
    rec = records.get(cut)
    return {"cut": cut, "reorient": rec is not None and rec.step_type == _REORIENT}


def _start_card(db: Db, tax: Taxonomy, desktop: int, view: str, step: int,
                instances: dict[str, InstanceRec], cache: InputCache,
                confirmed: bool, segment_start: Optional[dict] = None) -> list[dict]:
    """Spec 4.2 step 1: on the start frame everything present has to be drawn.

    Every instance that needs geometry here is listed; the ones already drawn
    -- or said 画框外 (task U39) -- are ✔ rows after the open ones (U2b round
    3 ruling: "what are the other items on the last frame?" -- hiding the
    drawn parts took a paragraph to explain).  With nothing that needs geometry at all the card is a single
    confirmation, as on any frame with nothing to draw.

    ``segment_start`` is :func:`segment_cut`'s answer when the frame is the
    first of its pose segment rather than of the walk (task U37); every row,
    the lone confirmation included, carries it as :data:`SEGMENT_START`.
    """
    state = state_of(db, tax, desktop, step, cache)
    bench_roi = bench_roi_of(db, FrameKey(desktop, step, view), cache)
    items = []
    needs = needs_geom(instances, state, tax, bench_roi=bench_roi)
    for instance, geom_kind in sorted(needs.items()):
        on_bench = geom_kind in BENCH_KINDS
        geom = GEOM_BOX if on_bench else GEOM_MASK
        done = _has_shape(db, tax, desktop, view, instance, step, cache, geom)
        kind = api.KIND_ADD_BENCH_BOX if on_bench else api.KIND_ADD_SHAPE
        items.append(_item(instance, kind, {}, done, instances.get(instance), None))
    if not items:
        items = [_item(f"step {step}", api.KIND_CONFIRM, {}, confirmed, None, None)]
    else:
        # 画框外 (task U39) is done, so after the open rows like a drawn one
        _mark_off_frame(db, FrameKey(desktop, step, view), items)
        items = _bottom_up(items, instances, tax)
    if segment_start is not None:
        for row in items:
            row[SEGMENT_START] = dict(segment_start)
    return items


def gallery_card(db: Db, tax: Taxonomy, desktop: int, view: str, shot: int) -> list[dict]:
    """The card of one components-gallery shot (task U10, spec 3.2b).

    One row per part the gallery asks for -- the parts removed by the end of
    the teardown, laid out on the table (:func:`tda.core.states.gallery_state`)
    -- whatever the shot before it held: the layout was re-arranged between
    shots, so there is no neighbour and no change to describe. A row is done
    when the part has a shape on this shot, or when the annotator said it is
    not in it (``absent``: an ``out_of_view`` label with no pixels, 「看不见」);
    open rows first, bottom-up like a start card. Every row carries
    ``gallery: True``, which is what the panel words it by.
    """
    cache = InputCache()
    key = FrameKey(desktop, shot, view)
    if db.get_frame(key) is None:
        return []
    instances = instances_of(db, desktop, cache)
    state = state_of(db, tax, desktop, shot, cache, view=view)
    try:
        overrides = db.frame_overrides(key) or {}
    except Exception:  # noqa: BLE001 - a missing table is "none set"
        overrides = {}
    confirmed = (db.get_frame(key) or {}).get("review_status") == _VERIFIED
    items = []
    for instance in sorted(needs_geom(instances, state, tax)):
        drawn = _has_shape(db, tax, desktop, view, instance, shot, cache)
        absent = not drawn and marked_absent(overrides.get(instance))
        row = _item(instance, api.KIND_ADD_SHAPE, {}, drawn or absent,
                    instances.get(instance), None)
        row["gallery"] = True
        if absent:
            row["absent"] = True
        items.append(row)
    if not items:
        row = _item(f"shot {shot}", api.KIND_CONFIRM, {}, confirmed, None, None)
        row["gallery"] = True
        return [row]
    return _bottom_up(items, instances, tax)
