"""框外零件: objects outside the machine, named and drawn on their own (task U18, spec 3.2c).

Some frames show something outside the machine's ROI that nobody listed: another
part lying beside it, a bracket, something unrelated. The annotator names it and
draws it **apart from the main annotation**: a *context object*.

It is an instance (``Db.instances`` returns it) with

* a key ``ctx.<class>.NN`` -- the ordinal continues per class among the context
  objects (:func:`next_key`), never among the machine's parts;
* ``cls`` any taxonomy class, or :data:`~tda.core.model.CONTEXT_OTHER` for
  something no class describes;
* ``attrs = {"context": True, "label": <free text>, "added_by", "added_at"}``
  (``label`` required for ``other``, optional otherwise; an optional ``note``);
* no state, no action, no relation; never implied, never in the log.

Everything that walks the machine's parts skips it
(:func:`tda.core.model.is_context` / :func:`~tda.core.model.machine_only`): the
state machine gives it no state (so no task card, no ``missing_shape``, no
``Space`` refusal), the step table's editors and the constraint graph do not see
it, the guesses (detector, difference map) leave it alone, the VLM export
ignores it. What it *has* is geometry: keyframes in the placement chain
:data:`~tda.core.model.CONTEXT_PLACEMENT`, carried like any part's, compiled in
a placement group of their own (:func:`context_needs` is what
:func:`tda.core.truth_inputs.gather` adds to a frame) -- layered among
themselves by draw order, never against the machine's parts -- stored as truth
rows, frozen with the frame and re-checked like any row. A frame where no shape
of it applies simply does not have it: a missing context shape is never a
problem.

Qt-free. :func:`plan_context` builds one, :func:`write_context` stores it with
one ``op_log`` row (:data:`OP_ADD`, its inverse deletes it); :func:`remove_context`
takes it out with everything that names it -- keyframes in every view, its
layer-order entries, pair overrides and frame overrides -- and returns that as
the record :func:`restore_context` puts back, which is the inverse
:data:`OP_DELETE` logs and an undo of the creation keeps.
"""
from __future__ import annotations

import re
from dataclasses import asdict
from typing import Iterable, Optional

from tda.core.compiler import select_keyframe
from tda.core.db import Db
from tda.core.dbrows import now_iso
from tda.core.model import (
    ALL_VIEWS,
    CONTEXT_ATTR,
    CONTEXT_LABEL,
    CONTEXT_OTHER,
    CONTEXT_PLACEMENT,
    CONTEXT_PREFIX,
    EXTRA_ATTR,
    FrameKey,
    FrameOverride,
    InstanceRec,
    PairOverride,
    ShapeKeyframe,
    ShapePart,
    Visibility,
    ZOrderRec,
    is_context,
    is_context_key,
)
from tda.core.taxonomy import Taxonomy

__all__ = [
    "ADDED_AT", "ADDED_BY", "NOTE_ATTR", "OP_ADD", "OP_DELETE", "OP_LABEL", "OP_VIEW",
    "ContextError", "chain_of", "class_choices", "confirmed_frames", "context_keys",
    "context_label", "context_needs", "context_reach", "frames_of", "keys_in_use",
    "next_key", "plan_context", "relabel_context", "remove_context", "restore_context",
    "said_absent", "write_context",
]

#: ``attrs`` provenance keys (the extras' names, :mod:`tda.core.extra`).
ADDED_BY = EXTRA_ATTR
ADDED_AT = "added_at"
NOTE_ATTR = "note"
#: ``op_log.kind`` of a creation, a deletion and a renaming.
OP_ADD = "add_context_instance"
OP_DELETE = "delete_context_instance"
OP_LABEL = "relabel_context_instance"
#: ``op_log`` is scoped per ``(desktop, view)``; an identity row belongs to none.
OP_VIEW = "-"

_ORDINAL = re.compile(r"\.(\d+)$")
OUT_OF_VIEW = Visibility.OUT_OF_VIEW.value


class ContextError(ValueError):
    """A context object that cannot be made, renamed or removed; the text is UI-ready."""


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def class_choices(tax: Taxonomy) -> list[str]:
    """What a context object may be: every taxonomy class, then ``other``."""
    return [*tax.classes, CONTEXT_OTHER]


def context_keys(instances: dict[str, InstanceRec]) -> list[str]:
    """The keys of the context objects of one instance table, sorted."""
    return sorted(key for key, rec in instances.items()
                  if is_context_key(key) or is_context(rec))


def context_label(rec: Optional[InstanceRec]) -> str:
    """The object's free-text name, ``""`` when it has none."""
    if rec is None:
        return ""
    return str((rec.attrs or {}).get(CONTEXT_LABEL) or "").strip()


def next_key(keys: Iterable[str], cls: str) -> str:
    """``ctx.<cls>.NN``: one past the highest ordinal of ``cls`` among the context keys."""
    prefix = f"{CONTEXT_PREFIX}{cls}."
    top = 0
    for key in keys:
        if not str(key).startswith(prefix):
            continue
        found = _ORDINAL.search(str(key)[len(prefix) - 1:])
        if found:
            top = max(top, int(found.group(1)))
    return f"{prefix}{top + 1:02d}"


#: The tables whose ``instance`` column can still name a context object that
#: is gone -- a confirmed row, a conflict about it, a shape an undo left behind.
_NAMING_TABLES = ("shape_keyframe", "compiled_mask", "conflict", "frame_override")


def keys_in_use(db: Db, desktop: int) -> set[str]:
    """Every context key this desktop has ever had, or that anything still names.

    The instance table's, the ones still named by a shape, a truth row, a
    conflict, a frame override or a pair override -- and every key an
    :data:`OP_ADD` row ever made. :func:`next_key` never hands one of them out
    again: a deleted object's confirmed rows and open conflicts are about
    *that* object, and a new one with its key would inherit them (U18 review).
    """
    desktop = int(desktop)
    like = f"{CONTEXT_PREFIX}%"
    found = {key for key in db.instances(desktop) if is_context_key(key)}
    for table in _NAMING_TABLES:
        found |= {str(r[0]) for r in db.conn.execute(
            f"SELECT DISTINCT instance FROM {table} WHERE desktop=? AND instance LIKE ?",
            (desktop, like)).fetchall()}
    for column in ("above", "below"):
        found |= {str(r[0]) for r in db.conn.execute(
            f"SELECT DISTINCT {column} FROM pair_override WHERE desktop=? AND {column} LIKE ?",
            (desktop, like)).fetchall()}
    for op in db.ops_of_kind(desktop, OP_ADD):
        made = ((op.get("payload") or {}).get("instance") or {}).get("key")
        if made:
            found.add(str(made))
    return found


def confirmed_frames(db: Db, desktop: int, key: str) -> dict[str, list[int]]:
    """``view -> steps`` where a human confirmed a row of this object."""
    out: dict[str, list[int]] = {}
    for row in db.conn.execute(
            "SELECT view, step FROM compiled_mask WHERE desktop=? AND instance=? "
            "AND status='verified' ORDER BY view, step", (int(desktop), str(key))).fetchall():
        out.setdefault(str(row["view"]), []).append(int(row["step"]))
    return out


def plan_context(instances: dict[str, InstanceRec], tax: Taxonomy, desktop: int,
                 cls: str, label: str = "", note: str = "", annotator: str = "",
                 now: Optional[str] = None, taken: Iterable[str] = ()) -> InstanceRec:
    """The record of one new context object, not written yet.

    ``cls`` is a taxonomy class or ``other``; ``label`` is required for
    ``other`` (what else would say what it is) and optional otherwise.
    ``taken`` are keys never to hand out again besides the table's
    (:func:`keys_in_use`).
    """
    cls = str(cls or "").strip()
    if cls != CONTEXT_OTHER and cls not in tax.classes:
        raise ContextError(f"{cls!r} 不是类别 / not a taxonomy class (or {CONTEXT_OTHER!r})")
    label = str(label or "").strip()
    if cls == CONTEXT_OTHER and not label:
        raise ContextError("「其他」要写名字：它是什么 / an 'other' object needs a name")
    key = next_key([*instances, *taken], cls)
    if key in instances:  # pragma: no cover - next_key is one past the highest
        raise ContextError(f"D{desktop:02d} already has {key}")
    attrs: dict = {CONTEXT_ATTR: True, ADDED_BY: str(annotator or "unknown"),
                   ADDED_AT: now or now_iso()}
    if label:
        attrs[CONTEXT_LABEL] = label
    if note and str(note).strip():
        attrs[NOTE_ATTR] = str(note).strip()
    return InstanceRec(key=key, desktop=int(desktop), cls=cls, attrs=attrs, raw_names=[])


def write_context(db: Db, rec: InstanceRec, annotator: str = "") -> int:
    """Store one new context object with its ``op_log`` row; joins a transaction."""
    if not is_context(rec):
        raise ContextError(f"{rec.key} is not a context object")
    with db.transaction():
        if rec.key in db.instances(rec.desktop):
            raise ContextError(f"D{rec.desktop:02d} already has {rec.key}")
        db.upsert_instance(rec)
        return db.log_op(rec.desktop, OP_VIEW, OP_ADD, {"instance": asdict(rec)},
                         {"delete": rec.key}, annotator or str(rec.attrs.get(ADDED_BY) or ""))


def relabel_context(db: Db, desktop: int, key: str, label: str, annotator: str = "") -> int:
    """Rename a context object (its ``label``); refused empty for ``other``."""
    rec = db.instances(int(desktop)).get(str(key))
    if rec is None or not is_context(rec):
        raise ContextError(f"{key} 不是框外零件 / not a context object")
    label = str(label or "").strip()
    if rec.cls == CONTEXT_OTHER and not label:
        raise ContextError("「其他」要写名字：它是什么 / an 'other' object needs a name")
    before = context_label(rec)
    if label:
        rec.attrs[CONTEXT_LABEL] = label
    else:
        rec.attrs.pop(CONTEXT_LABEL, None)
    with db.transaction():
        db.upsert_instance(rec)
        return db.log_op(int(desktop), OP_VIEW, OP_LABEL, {"instance": key, "label": label},
                         {"instance": key, "label": before}, annotator)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def said_absent(override: Optional[FrameOverride]) -> bool:
    """Has 「不在这张里」 (``I``) been said of a context object on this frame?

    The ``out_of_view`` label, whatever pixels the override keeps: a
    single-frame mask drawn there (``Alt+Enter``) stays under the label, so
    taking the label back brings it back (U18 review, M3). The compiler
    shows nothing of a part labelled ``out_of_view`` either way.
    """
    return override is not None and str(override.visibility) == OUT_OF_VIEW


def chain_of(keyframes: Iterable[ShapeKeyframe], key: FrameKey,
             pose_segment: int) -> list[ShapeKeyframe]:
    """A context object's keyframes of ``key``'s view in one pose segment."""
    return [kf for kf in keyframes
            if kf.view == key.view and kf.desktop == key.desktop
            and kf.placement == CONTEXT_PLACEMENT and kf.pose_segment == pose_segment]


def context_needs(instances: dict[str, InstanceRec],
                  keyframes: dict[str, list[ShapeKeyframe]],
                  key: FrameKey, pose_segment: int) -> dict[str, str]:
    """``instance -> geometry`` of the context objects this frame has.

    The ones a shape of applies here -- the smallest anchor at or after the
    step, in this view and pose segment, exactly the compiler's selection: a
    context object has no state to say where it is, so its shapes do. Nothing
    else; a frame no shape reaches does not have it, and that is not a problem.
    """
    out: dict[str, str] = {}
    for name, rec in instances.items():
        if not (is_context_key(name) or is_context(rec)):
            continue
        chosen = select_keyframe(chain_of(keyframes.get(name, ()), key, pose_segment),
                                 key.step)
        if chosen is not None:
            out[name] = "box" if chosen.geom_type == "box" else "mask"
    return out


def context_reach(db: Db, desktop: int, view: str, pose_segment: int,
                  chain: list[ShapeKeyframe], target: Optional[ShapeKeyframe] = None,
                  cache=None) -> list[int]:
    """Steps of this pose segment where ``chain`` selects ``target`` (any keyframe: ``None``).

    The context object's answer to :func:`tda.core.states.needs_geom`: it is
    in a frame exactly where one of its shapes applies.
    """
    from tda.core.truth_inputs import pose_segment_of, view_steps   # import cycle

    out: list[int] = []
    for step in view_steps(db, desktop, view):
        if pose_segment_of(db, FrameKey(desktop, step, view), cache) != pose_segment:
            continue
        chosen = select_keyframe(chain, step)
        if chosen is None:
            continue
        if target is None or chosen is target or (
                chosen.id is not None and target.id is not None
                and int(chosen.id) == int(target.id)):
            out.append(step)
    return out


def frames_of(db: Db, desktop: int, key: str) -> dict[str, list[int]]:
    """``view -> steps`` a context object has a compiled row on (to re-check after a removal)."""
    out: dict[str, list[int]] = {}
    for row in db.conn.execute(
            "SELECT DISTINCT view, step FROM compiled_mask WHERE desktop=? AND instance=? "
            "ORDER BY view, step", (int(desktop), str(key))).fetchall():
        out.setdefault(str(row["view"]), []).append(int(row["step"]))
    return out


# --------------------------------------------------------------------------- #
# removal and restoration
# --------------------------------------------------------------------------- #
def _keyframe_state(kf: ShapeKeyframe) -> dict:
    return asdict(kf)


def _keyframe_of(data: dict) -> ShapeKeyframe:
    parts = [ShapePart(p["name"], p.get("rle"),
                       None if p.get("box") is None else tuple(p["box"]))
             for p in data.get("parts") or []]
    fields = {k: v for k, v in data.items() if k != "parts"}
    return ShapeKeyframe(**fields, parts=parts)


def remove_context(db: Db, desktop: int, key: str) -> dict:
    """Take one context object out with everything that names it. **Joins a transaction.**

    Its keyframes in every view, its entries in every layer order, the pair
    overrides and frame overrides naming it, the identity row and the cached
    (``auto``) truth rows. A row somebody confirmed stays where it is: the
    frame's re-check finds the object gone and says so, as for any part. The
    return value is the record :func:`restore_context` puts back.
    """
    desktop, key = int(desktop), str(key)
    rec = db.instances(desktop).get(key)
    if rec is None or not is_context(rec):
        raise ContextError(f"{key} 不是框外零件 / not a context object")
    record: dict = {"instance": asdict(rec), "keyframes": [], "zorder": [], "pairs": [],
                    "overrides": [], "frames": frames_of(db, desktop, key)}
    with db.transaction():
        for view in ALL_VIEWS:
            for kf in db.keyframes(desktop, view, key):
                record["keyframes"].append(_keyframe_state(kf))
                db.delete_keyframe(int(kf.id))
        for row in db.conn.execute(
                "SELECT view, pose_segment FROM zorder WHERE desktop=? "
                "ORDER BY view, pose_segment", (desktop,)).fetchall():
            view, seg = str(row["view"]), int(row["pose_segment"])
            rec_z = db.zorder(desktop, view, seg)
            order = [tuple(entry) for entry in rec_z.order]
            mine = [(i, list(entry)) for i, entry in enumerate(order) if entry[0] == key]
            if not mine:
                continue
            record["zorder"].append({"view": view, "pose_segment": seg, "entries": mine})
            db.set_zorder(ZOrderRec(desktop, view, seg,
                                    [e for e in order if e[0] != key],
                                    version=rec_z.version + 1))
        for row in db.conn.execute(
                "SELECT view, pose_segment, above, below FROM pair_override "
                "WHERE desktop=? AND (above=? OR below=?)", (desktop, key, key)).fetchall():
            po = PairOverride(desktop, str(row["view"]), int(row["pose_segment"]),
                              str(row["above"]), str(row["below"]))
            record["pairs"].append(asdict(po))
            db.delete_pair_override(po)
        for row in db.conn.execute(
                "SELECT view, step FROM frame_override WHERE desktop=? AND instance=?",
                (desktop, key)).fetchall():
            frame = FrameKey(desktop, int(row["step"]), str(row["view"]))
            fo = db.frame_overrides(frame).get(key)
            if fo is None:
                continue
            record["overrides"].append({"view": frame.view, "step": frame.step,
                                        "visible_rle": fo.visible_rle,
                                        "visibility": fo.visibility,
                                        "occluder": fo.occluder})
            db.delete_frame_override(frame, key)
        db.delete_instance(desktop, key)
    return record


def restore_context(db: Db, record: dict) -> InstanceRec:
    """Put back what :func:`remove_context` took out. **Joins a transaction.**

    The keyframes under their old ids, the layer-order entries at the positions
    they held (clamped to the order as it is now), the overrides as they were.
    """
    rec = InstanceRec(**record["instance"])
    with db.transaction():
        if rec.key in db.instances(rec.desktop):
            raise ContextError(f"{rec.key} 又存在了 / exists again")
        db.upsert_instance(rec)
        for data in record.get("keyframes") or []:
            kf = _keyframe_of(data)
            keep = kf.id
            kf.id = None
            db.add_keyframe(kf, keep_id=keep)
        for entry in record.get("zorder") or []:
            view, seg = str(entry["view"]), int(entry["pose_segment"])
            rec_z = db.zorder(rec.desktop, view, seg)
            order = [tuple(e) for e in rec_z.order if e[0] != rec.key]
            for index, layer in sorted(entry.get("entries") or [], key=lambda e: e[0]):
                order.insert(min(int(index), len(order)), tuple(layer))
            db.set_zorder(ZOrderRec(rec.desktop, view, seg, order, version=rec_z.version + 1))
        for data in record.get("pairs") or []:
            db.set_pair_override(PairOverride(**data))
        for data in record.get("overrides") or []:
            db.set_frame_override(FrameOverride(
                FrameKey(rec.desktop, int(data["step"]), str(data["view"])), rec.key,
                data.get("visible_rle"), data.get("visibility"), data.get("occluder")))
    return rec
