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
  masks, similarity transforms -- and the view's pose breaks and pose-segment
  reference / range steps;
* the derived rows (compiled masks, digests, the re-check queue, conflicts),
  which are dropped for a moved frame: the compiler rebuilds them.

A step whose old frame had no photo (a missing frame) moves with its nearest
photographed predecessor, kept inside the new step range, so a keyframe
anchored on D18's old missing step 50 lands on step 48 next to the photo it was
drawn beside. When two old steps land on one (a ``dupli`` row's frame became an
extra shot), the one whose photo is the new step's own frame keeps its rows; a
keyed row of the other is dropped and counted, its keyframes are kept unless
they would duplicate a non-draft shape of the same part.

A desktop with a **verified** frame among the moving ones is refused before
anything is written (:class:`PhotoMoveConflict`): a signed-off frame changes
only through the re-check queue, never by a renumbering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from tda.core.model import LS_PREFIX, VIEWS, FrameKey

__all__ = ["FRAME_WORK_COLUMNS", "PhotoMoveConflict", "PhotoMoveMixin", "ViewMove",
           "plan_view_moves"]

#: Frame columns that hold what a human said about the photo.
FRAME_WORK_COLUMNS = ("review_status", "hand_or_tool_in_frame", "in_progress",
                      "image_quality", "bench_annotated")
#: Inputs keyed by ``(desktop, step, view, ...)``: they move with the photo.
_KEYED_INPUTS = ("frame_override", "instance_frame_flags", "occluder_mask",
                 "frame_transform")
#: Derived from the inputs: dropped for a moved frame, rebuilt by the compiler.
_DERIVED = ("compiled_mask", "frame_digest", "recheck_queue")


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

    def touched(self) -> set[int]:
        """The old steps that move, and the new steps whose own photo changes."""
        changed = {t for t in self.steps.values()
                   if self.main_source.get(t) is None or self.main_source[t] in self.steps}
        return set(self.steps) | changed

    def line(self, desktop: int) -> str:
        span = ", ".join(f"{a}->{b}" for a, b in sorted(self.steps.items())[:6])
        more = f" ... ({len(self.steps)} steps)" if len(self.steps) > 6 else ""
        extra = []
        if self.dropped_rows or self.dropped_keyframes:
            extra.append(f"dropped {self.dropped_rows} row(s) and {self.dropped_keyframes} "
                         f"keyframe(s) that collided with the step's own photo")
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
            moves, main_source = plan_view_moves(old_photos, keyed, new_main, new_extra,
                                                 di.n_steps)
            if moves:
                photos = sum(1 for p, s in old_photos.items() if s in moves)
                out[view] = ViewMove(view, moves, main_source, photos=photos)
        return out

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
        return work_by_view

    def restore_frame_work(self, desktop: int, work_by_view: dict) -> None:
        """Put the moved photos' human columns back on their new frame rows."""
        with self.transaction():
            for view, work in work_by_view.items():
                for step, cols in work.items():
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
        # A new step whose own photo does not move keeps its row as it is.
        work: dict[int, dict] = {}
        cols = ", ".join(FRAME_WORK_COLUMNS)
        for old in olds:
            new = moves[old]
            src = vm.main_source.get(new)
            if new in work or (src is not None and src not in moves):
                continue
            row = self.conn.execute(
                f"SELECT {cols} FROM frame WHERE desktop=? AND step=? AND view=?",
                (desktop, old, view)).fetchone()
            if row is not None:
                work[new] = {c: row[c] for c in FRAME_WORK_COLUMNS}
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
        self._move_pose_rows(desktop, view, moves)
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

    def _move_keyframes(self, desktop: int, view: str, olds: list[int],
                        moves: dict[int, int], vm: ViewMove) -> None:
        rows = self.conn.execute(
            f"SELECT id, instance, pose_segment, anchor_step, source FROM shape_keyframe "
            f"WHERE desktop=? AND view=? AND anchor_step IN ({','.join('?' * len(olds))})",
            (desktop, view, *olds)).fetchall()
        order = {old: i for i, old in enumerate(olds)}
        targets = sorted(set(moves.values()) - set(moves))
        taken: set[tuple] = set()
        if targets:     # shapes already on a target step that does not move stay first
            taken = {(r["instance"], r["pose_segment"], r["anchor_step"]) for r in
                     self.conn.execute(
                         f"SELECT instance, pose_segment, anchor_step FROM shape_keyframe "
                         f"WHERE desktop=? AND view=? AND anchor_step IN "
                         f"({','.join('?' * len(targets))})", (desktop, view, *targets))
                     if not str(r["instance"]).startswith(LS_PREFIX)}
        for r in sorted(rows, key=lambda r: (order[int(r["anchor_step"])], r["id"])):
            new = moves[int(r["anchor_step"])]
            ident = (r["instance"], r["pose_segment"], new)
            draft = str(r["instance"]).startswith(LS_PREFIX)
            if not draft and ident in taken:
                self.conn.execute("DELETE FROM shape_keyframe WHERE id=?", (r["id"],))
                vm.dropped_keyframes += 1
                continue
            taken.add(ident)
            self.conn.execute("UPDATE shape_keyframe SET anchor_step=? WHERE id=?",
                              (new, r["id"]))
            vm.keyframes += 1

    def _move_pose_rows(self, desktop: int, view: str, moves: dict[int, int]) -> None:
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
            self.conn.execute(
                f"INSERT OR IGNORE INTO pose_break ({', '.join(b)}) "
                f"VALUES ({', '.join('?' * len(b))})", tuple(b.values()))


def describe_blockers(desktop: int, blockers: list[str], limit: int = 6) -> str:
    shown = ", ".join(blockers[:limit]) + (f" and {len(blockers) - limit} more"
                                           if len(blockers) > limit else "")
    return (f"D{desktop:02d}: the new index puts other photos on verified frames "
            f"({shown}); a signed-off frame is never renumbered -- re-check it, or move "
            f"the work off it first")
