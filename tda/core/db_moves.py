"""When a re-built index moves a photo to another step, its work moves with it (U25).

An annotation is drawn on a **photo**, not on a step number. Collapsing the
sheet's ``dupli`` rows (spec 2.2, v1.25) renumbers the steps of thirteen
desktops, and mapping the scanner by its duplicate folders moves its photos
too: D18's scanner folder ``RGB491`` was step 49 and is step 48. ``load-index``
therefore compares, per sequence view, where each stored photo was with where
the new index puts it (:func:`plan_view_moves`) and, before it re-points the
frame rows, moves everything keyed by that photo's step:

* the keyframes anchored there (manual and Label Studio drafts alike);
* the frame row's human columns (``review_status``, the hand/tool, in-progress,
  quality and bench flags);
* the frame-keyed inputs -- visibility overrides, instance flags, occluder
  masks, similarity transforms, the frame's note (「这一帧的说明」, U35) -- and
  the view's pose breaks and pose-segment reference / range steps;
* the derived rows (compiled masks, digests, the re-check queue, conflicts),
  which are dropped for a moved frame: the compiler rebuilds them.

A step whose old frame had no photo (a missing frame) moves with its nearest
photographed predecessor, kept inside the new step range, so a keyframe
anchored on D18's old missing step 50 lands on step 48 next to the photo it was
drawn beside. When two old steps land on one (a ``dupli`` row's frame became an
extra shot), the one whose photo is the new step's own frame keeps its rows; a
keyed row of the other is dropped and counted -- a note is not: it is appended
to the main photo's note after a marker naming its old step (U35) -- its
keyframes are kept unless
they would duplicate a non-draft shape of the same part -- or, for a Label
Studio draft, a draft of the same label from the step's own photo (U25 round 2:
D24's oak1 step 36 would otherwise show the part twice). Of its frame row's
human columns only ``review_status`` -- what was done on the step -- is carried
onto the new step where that one has none (counted as dropped where it has its
own); the photo-specific ones (:data:`PHOTO_COLUMNS`: a hand in the frame, the
image quality, an operation in progress, the bench) describe that photo, which
is now an extra shot, so they are left behind and reported (U25 round 3). A
pose break that lands on one already there is counted too.

The hand-written state events (``state_event.auto = 0``) follow the step table,
and the step table follows the OAK1 rows: they move with ``oak1``'s
renumbering (U25 round 2). ``op_log`` payloads are history and keep the step
numbers they were written with.

A desktop is refused before anything is written (:class:`PhotoMoveConflict`)
when a **verified** frame is among the moving ones -- a signed-off frame
changes only through the re-check queue, never by a renumbering -- or when work
sits on a photo the new index no longer has in that view at all
(:meth:`PhotoMoveMixin.photos_left`): moving it to a neighbour would put it on
another photo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from tda.core.model import LS_PREFIX, VIEWS, FrameKey

__all__ = ["FRAME_WORK_COLUMNS", "LOADED_DUPLI_KEY", "PHOTO_COLUMNS", "PhotoMoveConflict",
           "PhotoMoveMixin",
           "ViewMove", "describe_blockers", "describe_left", "loaded_collapse_refusal",
           "plan_view_moves"]

#: Frame columns that hold what a human said about the photo.
FRAME_WORK_COLUMNS = ("review_status", "hand_or_tool_in_frame", "in_progress",
                      "image_quality", "bench_annotated")
#: Of those, the ones that describe the photo itself rather than the step: they
#: never move from a photo that became an extra shot onto the step's own photo.
PHOTO_COLUMNS = ("hand_or_tool_in_frame", "in_progress", "image_quality", "bench_annotated")
#: Inputs keyed by ``(desktop, step, view, ...)``: they move with the photo.
#: The frame's note (``frame_note``, task U35) too -- it is no compiler input,
#: but it is about the photo; two landing on one step are joined, not dropped
#: (:data:`NOTE_JOIN`).
_KEYED_INPUTS = ("frame_override", "instance_frame_flags", "occluder_mask",
                 "frame_transform", "frame_note")
#: How the note of a second old step is appended to the note of the step it
#: lands on (the main photo's, kept first): a marker line naming the old step,
#: who wrote it and when, then its text.
NOTE_JOIN = "\n\n[load-index：以下是原第 {old} 步的说明（{author}，{when}）]\n"
#: Derived from the inputs: dropped for a moved frame, rebuilt by the compiler.
_DERIVED = ("compiled_mask", "frame_digest", "recheck_queue")
#: The view whose rows the step table follows: the sheet's rows are the
#: Camera_1 captures, so the hand-written state events move with ``oak1``.
STEP_TABLE_VIEW = "oak1"
#: Desktop-meta key ``load-index`` writes: the ``dupli`` rows the loaded index
#: collapsed (``[]`` for none). Absent = loaded before U25 (or never).
LOADED_DUPLI_KEY = "index_dupli_rows"


def _new_places(di, view: str) -> tuple[dict[str, int], dict[str, int]]:
    """``(new_main, new_extra)``: where ``di`` puts each photo of ``view``."""
    new_main: dict[str, int] = {}
    new_extra: dict[str, int] = {}
    for key, ff in di.frames.items():
        if key.view != view:
            continue
        new_main[ff.path] = key.step
        for shot in (ff.aux or {}).get("extra_shots") or []:
            path = shot.get("path") if isinstance(shot, dict) else shot
            if path:
                new_extra[path] = key.step
    return new_main, new_extra


def _draft_label(instance: str) -> str:
    """``ls:<label>#<n>`` -> ``<label>``: one part per label on one photo."""
    return str(instance)[len(LS_PREFIX):].rsplit("#", 1)[0]


def _written_columns(row) -> dict:
    """The human frame columns of a row that hold something (``{}`` for none)."""
    if row is None:
        return {}
    out = {}
    for c in FRAME_WORK_COLUMNS:
        v = row[c]
        if v is None or (c == "review_status" and v == "unlabeled"):
            continue
        out[c] = v
    return out


class PhotoMoveConflict(ValueError):
    """A re-built index would move a verified frame's photo to another step."""


@dataclass
class ViewMove:
    """One view's renumbering and what it carried."""

    view: str
    steps: dict[int, int]                       # old step -> new step (changes only)
    main_source: dict[int, int] = field(default_factory=dict)   # new step -> old step
    photos: int = 0
    keyframes: int = 0
    keyed_rows: int = 0
    dropped_rows: int = 0
    dropped_keyframes: int = 0
    derived_rows: int = 0
    #: Label Studio drafts of an extra shot's photo dropped because the step's
    #: own photo has a draft of the same label (N5).
    dropped_drafts: int = 0
    #: ``review_status`` carried from an extra shot's old step onto its new
    #: step, and the ones dropped because the new step had its own (N4).
    carried_columns: int = 0
    dropped_columns: list[str] = field(default_factory=list)
    #: Photo-specific columns of an old step whose photo became an extra shot:
    #: they stay with that photo, not with the step (N16).
    left_columns: list[str] = field(default_factory=list)
    #: Pose breaks that landed on a step that already had one (N4).
    dropped_breaks: int = 0
    #: Hand-written state events renumbered with the step table (oak1 only, B3).
    events: int = 0
    #: Notes of a second old step appended to the note of the step they
    #: landed on (task U35); counted among :attr:`keyed_rows` as well.
    merged_notes: int = 0

    def touched(self) -> set[int]:
        """The old steps that move, and the new steps whose own photo changes."""
        changed = {t for t in self.steps.values()
                   if self.main_source.get(t) is None or self.main_source[t] in self.steps}
        return set(self.steps) | changed

    def line(self, desktop: int) -> str:
        span = ", ".join(f"{a}->{b}" for a, b in sorted(self.steps.items())[:6])
        more = f" ... ({len(self.steps)} steps)" if len(self.steps) > 6 else ""
        extra = []
        if self.events:
            extra.append(f"{self.events} hand-written state event(s) renumbered with the "
                         f"step table")
        if self.dropped_rows or self.dropped_keyframes:
            extra.append(f"dropped {self.dropped_rows} row(s) and {self.dropped_keyframes} "
                         f"keyframe(s) that collided with the step's own photo")
        if self.dropped_drafts:
            extra.append(f"dropped {self.dropped_drafts} Label Studio draft(s) of an extra "
                         f"shot whose label the step's own photo already has")
        if self.carried_columns:
            extra.append(f"{self.carried_columns} review status(es) carried from an extra "
                         f"shot's old step")
        if self.dropped_columns:
            extra.append(f"dropped review status(es) the new step has its own of: "
                         f"{', '.join(self.dropped_columns)}")
        if self.left_columns:
            extra.append(f"{len(self.left_columns)} photo column(s) left behind on an extra "
                         f"shot: {', '.join(self.left_columns)}")
        if self.dropped_breaks:
            extra.append(f"{self.dropped_breaks} pose break(s) landed on one already there")
        if self.merged_notes:
            extra.append(f"{self.merged_notes} frame note(s) landed on a step that has one: "
                         f"appended to it with a [load-index] marker")
        if self.derived_rows:
            extra.append(f"{self.derived_rows} compiled/derived row(s) to rebuild")
        return (f"{self.view}: {self.photos} photo(s) moved step ({span}{more}); "
                f"{self.keyframes} keyframe(s) and {self.keyed_rows} keyed row(s) "
                f"followed them" + (f"; {'; '.join(extra)}" if extra else ""))


def plan_view_moves(old_photos: dict[str, int], old_steps: Iterable[int],
                    new_main: dict[str, int], new_extra: dict[str, int],
                    n_steps: int) -> tuple[dict[int, int], dict[int, int]]:
    """``(moves, main_source)`` for one view.

    ``old_photos`` is ``{path: old step}`` of the stored frames that have a photo,
    ``old_steps`` every old step something is keyed to, ``new_main`` /
    ``new_extra`` where the new index puts each photo (as a step's own frame /
    as an extra shot). ``moves`` maps every old step whose number changes;
    ``main_source`` names, for a new step, the old step whose photo is its own
    frame. An index that moves no photo gives ``({}, {})``.
    """
    placed: dict[int, int] = {}
    main_source: dict[int, int] = {}
    for path, old in old_photos.items():
        if path in new_main:
            placed[old] = new_main[path]
            main_source[new_main[path]] = old
        elif path in new_extra:
            placed[old] = new_extra[path]
    old_steps = {int(v) for v in old_steps}
    # nothing moved -- unless the view has rows past the new last step (a
    # missing frame of a step the collapse removed), which must go too
    if all(a == b for a, b in placed.items()) and not any(s > n_steps for s in old_steps):
        return {}, {}
    full = dict(placed)
    anchors = sorted(placed)
    for s in sorted(old_steps - set(placed)):
        before = [a for a in anchors if a < s]
        after = [a for a in anchors if a > s]
        if before:
            p = before[-1]
            t = placed[p] + (s - p)
            if after:
                t = min(t, max(placed[p], placed[after[0]] - 1))
        elif after:
            q = after[0]
            t = max(1, placed[q] - (q - s))
        else:
            t = s
        full[s] = max(1, min(int(n_steps), t)) if n_steps else t
    moves = {a: b for a, b in full.items() if a != b}
    return moves, {t: s for t, s in main_source.items()}


class PhotoMoveMixin:
    """The ``load-index`` side of :func:`plan_view_moves` (a :class:`Db` mixin)."""

    def plan_photo_moves(self, desktop: int, di) -> dict[str, ViewMove]:
        """Every sequence view of ``desktop`` whose photos ``di`` puts on other steps."""
        out: dict[str, ViewMove] = {}
        for view in VIEWS:
            rows = self.frames_for(desktop, view)
            if not rows:
                continue
            old_photos = {r["path"]: int(r["step"]) for r in rows
                          if r.get("path") and not r.get("missing")}
            keyed = {int(r["step"]) for r in rows}
            keyed |= {int(r[0]) for r in self.conn.execute(
                "SELECT DISTINCT anchor_step FROM shape_keyframe WHERE desktop=? AND view=?",
                (desktop, view))}
            if view == STEP_TABLE_VIEW:
                keyed |= set(self._hand_event_steps(desktop))
            new_main, new_extra = _new_places(di, view)
            moves, main_source = plan_view_moves(old_photos, keyed, new_main, new_extra,
                                                 di.n_steps)
            if moves:
                photos = sum(1 for p, s in old_photos.items() if s in moves)
                out[view] = ViewMove(view, moves, main_source, photos=photos)
        return out

    def photos_left(self, desktop: int, di) -> list[str]:
        """Work on a stored photo the new index no longer has in that view (N3).

        ``"<view> step <n> (<path>): <work>"`` for every sequence frame whose
        photo is neither a frame nor an extra shot of the view in ``di`` and
        that somebody worked on (:meth:`frame_work`, or a human column of the
        frame row): moving it to a neighbouring step would put it on another
        photo. A photo moved into the components gallery (D1's Camera_1 040)
        has left the sequence too.
        """
        out: list[str] = []
        cols = ", ".join(FRAME_WORK_COLUMNS)
        for view in VIEWS:
            new_main, new_extra = _new_places(di, view)
            for r in self.frames_for(desktop, view):
                path = r.get("path")
                if not path or r.get("missing") or path in new_main or path in new_extra:
                    continue
                key = FrameKey(desktop, int(r["step"]), view)
                work = list(self.frame_work(key))
                row = self.conn.execute(
                    f"SELECT {cols} FROM frame WHERE desktop=? AND step=? AND view=?",
                    (desktop, key.step, view)).fetchone()
                written = _written_columns(row)
                if written and not any(w.startswith("review status") for w in work):
                    work.append("frame column(s) " + ", ".join(sorted(written)))
                if work:
                    out.append(f"{view} step {key.step} ({path}): {', '.join(work)}")
        return out

    def _hand_event_steps(self, desktop: int) -> list[int]:
        return [int(r[0]) for r in self.conn.execute(
            "SELECT DISTINCT step FROM state_event WHERE desktop=? AND auto=0", (desktop,))]

    def photo_move_blockers(self, desktop: int, moves: dict[str, ViewMove]) -> list[str]:
        """``"<view> step <n>"`` of every verified frame a move would touch."""
        out = []
        for view, vm in moves.items():
            for step in sorted(vm.touched()):
                row = self.get_frame(FrameKey(desktop, step, view)) or {}
                signed = self.conn.execute(
                    "SELECT 1 FROM compiled_mask WHERE desktop=? AND step=? AND view=? "
                    "AND status='verified' LIMIT 1", (desktop, step, view)).fetchone()
                if row.get("review_status") == "verified" or signed:
                    out.append(f"{view} step {step}")
        return out

    def apply_photo_moves(self, desktop: int, moves: dict[str, ViewMove]) -> dict[int, dict]:
        """Move the work of every moved photo; returns ``{new step: frame work}``
        per view (``{view: {...}}``) for :meth:`restore_frame_work` to put back
        once ``load-index`` has re-pointed the frame rows."""
        work_by_view: dict = {}
        with self.transaction():
            for view, vm in moves.items():
                work_by_view[view] = self._move_view(desktop, view, vm)
            table = moves.get(STEP_TABLE_VIEW)
            if table is not None:
                table.events = self._move_hand_events(desktop, table.steps)
        return work_by_view

    def _move_hand_events(self, desktop: int, steps: dict[int, int]) -> int:
        """Renumber the hand-written state events with the step table (B3)."""
        n = 0
        for r in self.conn.execute(
                "SELECT id, step FROM state_event WHERE desktop=? AND auto=0",
                (desktop,)).fetchall():
            new = steps.get(int(r["step"]))
            if new is not None and new != int(r["step"]):
                self.conn.execute("UPDATE state_event SET step=? WHERE id=?", (new, r["id"]))
                n += 1
        return n

    def restore_frame_work(self, desktop: int, work_by_view: dict) -> None:
        """Put the moved photos' human columns back on their new frame rows."""
        with self.transaction():
            for view, work in work_by_view.items():
                for step, cols in work.items():
                    if not cols:
                        continue
                    sets = ", ".join(f"{c}=?" for c in cols)
                    self.conn.execute(
                        f"UPDATE frame SET {sets} WHERE desktop=? AND step=? AND view=?",
                        (*cols.values(), desktop, step, view))

    # -------------------------------------------------------------- internals
    def _source_order(self, vm: ViewMove, old: int) -> tuple:
        """Sort key: the old step whose photo is the new step's own frame first."""
        new = vm.steps.get(old, old)
        return (0 if vm.main_source.get(new) == old else 1, old)

    def _move_view(self, desktop: int, view: str, vm: ViewMove) -> dict[int, dict]:
        moves = vm.steps
        olds = sorted(moves, key=lambda s: self._source_order(vm, s))
        # the frame rows' human columns: kept aside, the rows themselves deleted.
        # A new step whose own photo does not move keeps its row as it is. The
        # row of an old step whose photo became an extra shot gives its review
        # status to the new step where that has none (else counted as dropped);
        # its photo columns describe the extra shot and are left behind.
        work: dict[int, dict] = {}
        held: dict[int, dict] = {}     # what the new step's row will hold
        cols = ", ".join(FRAME_WORK_COLUMNS)
        sql = f"SELECT {cols} FROM frame WHERE desktop=? AND step=? AND view=?"
        for old in olds:
            new = moves[old]
            src = vm.main_source.get(new)
            row = self.conn.execute(sql, (desktop, old, view)).fetchone()
            if new not in held:
                if src is not None and src not in moves:
                    held[new] = _written_columns(
                        self.conn.execute(sql, (desktop, new, view)).fetchone())
                    work[new] = {}
                elif row is not None:
                    work[new] = {c: row[c] for c in FRAME_WORK_COLUMNS}
                    held[new] = _written_columns(row)
                    continue
                else:
                    held[new], work[new] = {}, {}
                    continue
            for c, v in _written_columns(row).items():
                if c in PHOTO_COLUMNS:
                    vm.left_columns.append(f"step {old}->{new} {c}={v}")
                    continue
                if c not in held[new]:
                    held[new][c] = work[new][c] = v
                    vm.carried_columns += 1
                elif held[new][c] != v:
                    vm.dropped_columns.append(f"step {old}->{new} {c}={v}")
        for table in _DERIVED:
            for step in vm.touched():
                vm.derived_rows += self.conn.execute(
                    f"DELETE FROM {table} WHERE desktop=? AND step=? AND view=?",
                    (desktop, step, view)).rowcount
        # keyed inputs: rows of a moving step, re-inserted at the new step
        for table in _KEYED_INPUTS + ("conflict",):
            self._move_keyed(table, desktop, view, olds, moves, vm)
        self._move_keyframes(desktop, view, olds, moves, vm)
        for old in olds:
            self.conn.execute("DELETE FROM frame WHERE desktop=? AND step=? AND view=?",
                              (desktop, old, view))
        self._move_pose_rows(desktop, view, moves, vm)
        return work

    def _move_keyed(self, table: str, desktop: int, view: str, olds: list[int],
                    moves: dict[int, int], vm: ViewMove) -> None:
        cur = self.conn.execute(f"SELECT * FROM {table} WHERE desktop=? AND view=? "
                                f"AND step IN ({','.join('?' * len(olds))})",
                                (desktop, view, *olds))
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
        if not rows:
            return
        self.conn.execute(f"DELETE FROM {table} WHERE desktop=? AND view=? "
                          f"AND step IN ({','.join('?' * len(olds))})",
                          (desktop, view, *olds))
        order = {old: i for i, old in enumerate(olds)}
        if table == "frame_note":
            self._move_notes(rows, moves, order, vm)
            return
        pk = [r[1] for r in self.conn.execute(f"PRAGMA table_info({table})") if r[5]]
        seen: set[tuple] = set()
        for row in sorted(rows, key=lambda r: order[int(r["step"])]):
            row["step"] = moves[int(row["step"])]
            ident = tuple(row[c] for c in pk)
            if ident in seen:
                vm.dropped_rows += 1
                continue
            exists = self.conn.execute(
                f"SELECT 1 FROM {table} WHERE " + " AND ".join(f"{c}=?" for c in pk),
                ident).fetchone() if pk and pk != ["id"] else None
            if exists:
                vm.dropped_rows += 1
                continue
            seen.add(ident)
            self.conn.execute(
                f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                tuple(row.values()))
            vm.keyed_rows += 1

    def _move_notes(self, rows: list[dict], moves: dict[int, int], order: dict[int, int],
                    vm: ViewMove) -> None:
        """Re-insert the moving notes at their new steps, joining two that meet (U35).

        The rows come in source order -- the old step whose photo is the new
        step's own frame first -- so the main photo's note leads; a note the
        new step already has (its photo does not move) leads as well. Every
        later one is appended after a :data:`NOTE_JOIN` marker; the author and
        time stay the leading note's.
        """
        placed: dict[tuple, dict] = {}
        for row in sorted(rows, key=lambda r: order[int(r["step"])]):
            old = int(row["step"])
            ident = (row["desktop"], row["view"], moves[old])
            here = placed.get(ident)
            if here is None:
                found = self.conn.execute(
                    "SELECT * FROM frame_note WHERE desktop=? AND view=? AND step=?",
                    ident).fetchone()
                here = None if found is None else dict(found)
            if here is None:
                placed[ident] = dict(row, step=moves[old])
            else:
                here["text"] = str(here["text"]) + NOTE_JOIN.format(
                    old=old, author=row.get("author") or "-",
                    when=row.get("updated_at") or "-") + str(row["text"])
                placed[ident] = here
                vm.merged_notes += 1
            vm.keyed_rows += 1
        for row in placed.values():
            self.conn.execute(
                "INSERT INTO frame_note(desktop, view, step, text, author, updated_at) "
                "VALUES(?, ?, ?, ?, ?, ?) ON CONFLICT(desktop, view, step) DO UPDATE SET "
                "text=excluded.text, author=excluded.author, updated_at=excluded.updated_at",
                (row["desktop"], row["view"], row["step"], row["text"], row["author"],
                 row["updated_at"]))

    def _move_keyframes(self, desktop: int, view: str, olds: list[int],
                        moves: dict[int, int], vm: ViewMove) -> None:
        rows = self.conn.execute(
            f"SELECT id, instance, pose_segment, anchor_step, source FROM shape_keyframe "
            f"WHERE desktop=? AND view=? AND anchor_step IN ({','.join('?' * len(olds))})",
            (desktop, view, *olds)).fetchall()
        order = {old: i for i, old in enumerate(olds)}
        targets = sorted(set(moves.values()) - set(moves))
        taken: set[tuple] = set()
        # labels of the Label Studio drafts on each new step's own photo
        main_labels: dict[int, set[str]] = {}
        if targets:     # shapes already on a target step that does not move stay first
            for r in self.conn.execute(
                    f"SELECT instance, pose_segment, anchor_step FROM shape_keyframe "
                    f"WHERE desktop=? AND view=? AND anchor_step IN "
                    f"({','.join('?' * len(targets))})", (desktop, view, *targets)):
                if str(r["instance"]).startswith(LS_PREFIX):
                    main_labels.setdefault(int(r["anchor_step"]), set()).add(
                        _draft_label(r["instance"]))
                else:
                    taken.add((r["instance"], r["pose_segment"], r["anchor_step"]))
        for r in sorted(rows, key=lambda r: (order[int(r["anchor_step"])], r["id"])):
            old = int(r["anchor_step"])
            new = moves[old]
            ident = (r["instance"], r["pose_segment"], new)
            draft = str(r["instance"]).startswith(LS_PREFIX)
            if not draft and ident in taken:
                self.conn.execute("DELETE FROM shape_keyframe WHERE id=?", (r["id"],))
                vm.dropped_keyframes += 1
                continue
            if draft:
                label = _draft_label(r["instance"])
                if vm.main_source.get(new) in (None, old):
                    main_labels.setdefault(new, set()).add(label)
                elif label in main_labels.get(new, ()):
                    # drawn on an extra shot's photo, and the step's own photo
                    # already has this part: two sets would show (N5)
                    self.conn.execute("DELETE FROM shape_keyframe WHERE id=?", (r["id"],))
                    vm.dropped_drafts += 1
                    continue
            taken.add(ident)
            self.conn.execute("UPDATE shape_keyframe SET anchor_step=? WHERE id=?",
                              (new, r["id"]))
            vm.keyframes += 1

    def _move_pose_rows(self, desktop: int, view: str, moves: dict[int, int],
                        vm: Optional[ViewMove] = None) -> None:
        for r in self.conn.execute("SELECT seg, start_step, end_step, ref_step FROM "
                                   "pose_segment WHERE desktop=? AND view=?",
                                   (desktop, view)).fetchall():
            new = {c: moves.get(r[c], r[c]) if r[c] is not None else None
                   for c in ("start_step", "end_step", "ref_step")}
            if any(new[c] != r[c] for c in new):
                self.conn.execute(
                    "UPDATE pose_segment SET start_step=?, end_step=?, ref_step=? "
                    "WHERE desktop=? AND view=? AND seg=?",
                    (new["start_step"], new["end_step"], new["ref_step"], desktop, view,
                     r["seg"]))
        breaks = self.conn.execute("SELECT * FROM pose_break WHERE desktop=? AND view=?",
                                   (desktop, view)).fetchall()
        moving = [dict(b) for b in breaks if int(b["step"]) in moves]
        for b in moving:
            self.conn.execute("DELETE FROM pose_break WHERE desktop=? AND view=? AND step=?",
                              (desktop, view, b["step"]))
        for b in moving:
            b["step"] = moves[int(b["step"])]
            cur = self.conn.execute(
                f"INSERT OR IGNORE INTO pose_break ({', '.join(b)}) "
                f"VALUES ({', '.join('?' * len(b))})", tuple(b.values()))
            if cur.rowcount == 0 and vm is not None:
                vm.dropped_breaks += 1


def describe_left(desktop: int, left: list[str], limit: int = 6) -> str:
    shown = "; ".join(left[:limit]) + (f"; and {len(left) - limit} more"
                                       if len(left) > limit else "")
    return (f"D{desktop:02d}: work sits on photos the new index no longer has in their "
            f"view ({shown}); moving it to a neighbouring step would put it on another "
            f"photo -- check the index fixes, or move the work off those frames first")


def loaded_collapse_refusal(meta: Optional[dict], has_frames: bool,
                            dupli_rows: Iterable[int]) -> Optional[str]:
    """Why the step table may not be renumbered over these frames; ``None`` if it may.

    ``load-index`` records the ``dupli`` rows its index collapsed
    (:data:`LOADED_DUPLI_KEY`). A sheet whose rows collapse differently -- or
    any collapse at all over frames loaded before U25 -- would put the step
    table one or two steps off the photos (U25 round 2, B2). A desktop with no
    frames yet has nothing to be off from.
    """
    if not has_frames:
        return None
    want = sorted({int(r) for r in dupli_rows})
    loaded = (meta or {}).get(LOADED_DUPLI_KEY)
    if loaded is None:
        if not want:
            return None
        return (f"its frames were loaded before the sheet's dupli rows {want} were "
                f"collapsed into steps (an index written before U25), so the step table "
                f"would end up one or two steps off the photos")
    if sorted(int(r) for r in loaded) != want:
        return (f"its frames were loaded with dupli rows {sorted(loaded) or 'none'} "
                f"collapsed, the sheet has {want or 'none'}, so the step table would end "
                f"up off the photos")
    return None


def describe_blockers(desktop: int, blockers: list[str], limit: int = 6) -> str:
    shown = ", ".join(blockers[:limit]) + (f" and {len(blockers) - limit} more"
                                           if len(blockers) > limit else "")
    return (f"D{desktop:02d}: the new index puts other photos on verified frames "
            f"({shown}); a signed-off frame is never renumbered -- re-check it, or move "
            f"the work off it first")
