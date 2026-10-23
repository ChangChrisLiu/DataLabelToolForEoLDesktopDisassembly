"""A forced re-import keeps 「其实是硬盘架」 / 「其实是卡扣」 (task U21).

The sheets name the drive cage and its latch inconsistently, and the importer's
name patterns made the cage a latch on three annotated desktops: D2's "remove
SSD locker shield", D3's "HDD Locker Shield", D4's "HDD Shield Locker" (drawn as
the cage and its latch in one mask). An annotator reclasses such a part in the
app (:mod:`tda.core.reclass`): it becomes ``drive_cage.NN``, everything drawn
on it moves along. ``import-logs --force`` parses the sheet again, though, and
would make the latch again under its old key and class -- the shapes on a key
the sheet no longer produces, the cage gone.

So after the merges are replayed (:mod:`tda.core.merge_replay`) and before the
``Shift+T`` verbs are (:mod:`tda.core.verb_replay`, which then sees the new
keys), every reclass **in effect** -- a ``reclass_instance`` row of the op log
whose new key the stored instance table holds with the new class (an undo took
it away) -- is applied to the parsed records:

* **matched** by the instance's own action steps and raw name, as the op
  recorded them -- not by key: the importer's key for the part depends on the
  name patterns, which change (task U21 itself makes "… locker shield" a cage);
* **kept** -- the parsed part is created under the new class and key directly:
  its record goes through :func:`reclassed_record` (the rule the app used), its
  actions and every parsed pointer at it follow, and an action whose verb does
  not apply to the new class (a sheet's ``open`` on what is now a cage) gets
  the verb the step table had when the reclass was made. A part the sheet
  already makes the new class under another key is renamed; a parsed part of
  the new class already holding the new key swaps keys with it;
* **already** -- the sheet already makes it the new class under the new key
  (D2/D3 under the new name patterns), or the part was added in S1 and the
  import keeps it as it is: nothing to do, still reported as kept;
* **not matched** -- no parsed part (or more than one) has an action at those
  steps under that name, or the new key is held by a part of another class;
  reported with why, nothing applied.

The drive-cage latch rule (:func:`tda.core.implied.cage_latches`) runs
afterwards as usual and adopts the cage's locker or keeps its implied latch.
Pure on the parsed records and idempotent; one ``replay_reclass`` row in the
op log says what happened.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable, Optional

from tda.core.db import Db
from tda.core.merge_replay import OF_ATTR, ranges, repointed
from tda.core.model import ActionRec, InstanceRec, is_context, is_extra, is_provisional
from tda.core.taxonomy import Taxonomy

__all__ = [
    "CAPTION", "CLASS_ZH", "OP_RECLASS", "OP_REPLAY_RECLASS", "OP_UNDO_RECLASS", "RECLASS_TO",
    "ReclassReplay",
    "effective_reclasses", "follow", "reclass_renames", "reclass_target", "reclassed_record",
    "replay_reclass_for_import", "replay_reclasses", "verb_applies",
]

#: ``op_log.kind`` of one reclass, of its undo, and of the one row a forced
#: re-import writes about them. Here, in the light module the importer imports.
OP_RECLASS = "reclass_instance"
OP_UNDO_RECLASS = "undo_reclass_instance"
OP_REPLAY_RECLASS = "replay_reclass"
#: The classes a part may be said to be instead: the sheets mix the drive cage
#: ("shield") and its locker up, and nothing else.
RECLASS_TO = {"drive_latch": "drive_cage", "drive_cage": "drive_latch"}
#: What the two classes are called where the annotator reads them, and the
#: option a row offers, by the class it would become.
CLASS_ZH = {"drive_cage": "硬盘架", "drive_latch": "卡扣"}
CAPTION = {"drive_cage": "其实是硬盘架", "drive_latch": "其实是卡扣"}
#: A screw's and a connector's columns: a reclassed bay part never keeps them.
CLEARED_COLUMNS = ("fastens", "socket_host", "cable")
#: Columns a reclassed part keeps unless they name one of the two bay classes:
#: a cage riding on a latch or a cage, or a latch on what was its cage and is
#: now a latch, means nothing (the cage rule decides a latch's ``parent``).
POINTER_COLUMNS = ("parent", "mounted_on")


def reclass_target(cls: Optional[str]) -> Optional[str]:
    """The class a part of ``cls`` may be said to be instead, or ``None``."""
    return RECLASS_TO.get(str(cls or ""))


def verb_applies(tax: Taxonomy, verb: str, cls: str) -> bool:
    """Does ``verb`` apply to ``cls`` (``taxonomy.yaml``'s ``applies_to``)?"""
    return cls in ((tax.verbs.get(verb) or {}).get("applies_to") or ())


def _cls_of_key(key: Optional[str]) -> str:
    return str(key or "").split(".", 1)[0]


def reclassed_record(rec: InstanceRec, new_cls: str, new_key: str,
                     tax: Taxonomy) -> tuple[InstanceRec, dict]:
    """``rec`` as an instance of ``new_cls`` under ``new_key``, and what it does not keep.

    The rule (the app's and the re-import's alike):

    * ``attrs`` are carried -- ``of``, provenance (``added_by``...), the
      parser's bookkeeping -- except an attribute the old class declares and
      the new one does not, and an ``of`` the new class's vocabulary does not
      have (a cage's ``of`` is ``ssd`` / ``hdd`` / ``optical``; a latch's is
      free, so everything a cage had fits);
    * ``raw_names`` are carried;
    * ``parent``/``attached`` and ``mounted_on`` are carried unless they name a
      drive cage or a drive latch (then cleared: a cage does not ride on its
      own latch, and a latch's cage is the cage rule's to decide);
    * ``fastens``, ``socket_host`` and ``cable`` -- a screw's and a
      connector's -- are cleared; every other column is carried.

    Returns ``(record, {"attrs": {...}, "columns": {...}})`` of what was left out.
    """
    old_schema = (tax.classes.get(rec.cls) or {}).get("attrs") or {}
    new_schema = (tax.classes.get(new_cls) or {}).get("attrs") or {}
    attrs: dict = {}
    dropped_attrs: dict = {}
    for name, value in (rec.attrs or {}).items():
        if name in old_schema and name not in new_schema:
            dropped_attrs[name] = value
            continue
        allowed = new_schema.get(name)
        if name == OF_ATTR and allowed and value not in allowed:
            dropped_attrs[name] = value
            continue
        attrs[name] = copy.deepcopy(value)
    new = replace(copy.deepcopy(rec), key=str(new_key), cls=str(new_cls), attrs=attrs)
    columns: dict = {}
    for name in CLEARED_COLUMNS:
        if getattr(new, name, None):
            columns[name] = getattr(new, name)
            setattr(new, name, None)
    for name in POINTER_COLUMNS:
        value = getattr(new, name, None)
        if value and _cls_of_key(value) in RECLASS_TO:
            columns[name] = value
            setattr(new, name, None)
            if name == "parent":
                new.attached = False
    return new, {"attrs": dropped_attrs, "columns": columns}


# --------------------------------------------------------------------------- #
# what the op log holds
# --------------------------------------------------------------------------- #
def _undone(db: Db, desktop: int) -> set[int]:
    return {int((op.get("payload") or {}).get("undid") or 0)
            for op in db.ops_of_kind(int(desktop), OP_UNDO_RECLASS)}


def effective_reclasses(db: Db, desktop: int) -> list[dict]:
    """The reclasses of ``desktop`` in effect, oldest first.

    ``{"op", "old", "new", "old_cls", "new_cls", "steps", "raw_names", "verbs",
    "extra"}`` -- one per reclass row whose new key the stored instance table
    holds with the new class. An undone one lost its key; one reclassed again
    since (latch → cage → latch) lost it to the next row, which is in effect
    instead and matches the same sheet row.
    """
    stored = db.instances(int(desktop))
    undone = _undone(db, desktop)
    out = []
    for op in db.ops_of_kind(int(desktop), OP_RECLASS):
        payload = op.get("payload") or {}
        new = str(payload.get("new") or "")
        held = stored.get(new)
        if int(op["id"]) in undone or held is None or held.cls != payload.get("new_cls"):
            continue
        out.append({"op": int(op["id"]), "old": str(payload.get("old") or ""), "new": new,
                    "old_cls": payload.get("old_cls"), "new_cls": payload.get("new_cls"),
                    "steps": [int(s) for s in payload.get("steps") or ()],
                    "raw_names": list(payload.get("raw_names") or ()),
                    "verbs": [[int(s), str(v)] for s, v in payload.get("verbs") or ()],
                    "extra": is_extra(held)})
    return out


def reclass_renames(db: Db, desktop: int) -> list[tuple[int, str, str]]:
    """``(op id, old key, new key)`` of every reclass not undone, oldest first.

    For the ``Shift+T`` replay: a correction made on ``drive_latch.02`` before
    it became ``drive_cage.01`` is about the cage's action now -- and only one
    made before (:func:`follow`): a later part may be given the old key.
    """
    undone = _undone(db, desktop)
    return [(int(op["id"]), str((op.get("payload") or {}).get("old") or ""),
             str((op.get("payload") or {}).get("new") or ""))
            for op in db.ops_of_kind(int(desktop), OP_RECLASS) if int(op["id"]) not in undone]


def follow(target: str, op_id: int, renames: Optional[Iterable[tuple[int, str, str]]]) -> str:
    """``target`` of the op ``op_id`` followed through the reclasses made after it."""
    for rid, old, new in sorted(renames or ()):
        if int(op_id) < rid and target == old:
            target = new
    return target


# --------------------------------------------------------------------------- #
# the replay
# --------------------------------------------------------------------------- #
@dataclass
class ReclassReplay:
    """What a re-import did with the reclasses in effect.

    Each entry is ``{"ops", "old", "new", "cls", "steps", "raw_names"}``; a
    kept one also has ``"parsed"`` (the importer's key) and ``"verbs"`` (the
    actions whose verb was carried), a not-matched one ``"why"``. ``renames``
    is :func:`reclass_renames`, for the verb replay that follows.
    """

    kept: list[dict] = field(default_factory=list)
    already: list[dict] = field(default_factory=list)
    unmatched: list[dict] = field(default_factory=list)
    inverse: dict = field(default_factory=lambda: {"instances": [], "actions": []})
    renames: list[tuple[int, str, str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.kept or self.already or self.unmatched)

    def payload(self) -> dict:
        return {"kept": self.kept, "already": self.already, "unmatched": self.unmatched}

    def lines(self) -> list[str]:
        """``reclass kept: …`` and ``reclass not matched: …``, one per reclass."""
        out = []
        for e in self.kept:
            carried = "".join(f"; step {s} {sheet} -> {verb}" for s, sheet, verb in e["verbs"])
            out.append(f"reclass kept: {e['parsed']} -> {e['new']} ({e['cls']}; "
                       f"{_where(e)}{carried})")
        for e in self.already:
            how = ("added in S1, kept as it is" if e.get("extra")
                   else "the sheet already says so")
            out.append(f"reclass kept: {e['new']} ({e['cls']}; {_where(e)}; {how})")
        out.extend(f"reclass not matched: {e['old']} -> {e['new']} ({e['why']})"
                   for e in self.unmatched)
        return out


def _where(entry: dict) -> str:
    names = ", ".join(f'"{n}"' for n in entry.get("raw_names") or ()) or "no name"
    return f"step {ranges(entry.get('steps') or ())} {names}"


def _norm(name: str) -> str:
    return " ".join(str(name or "").lower().split())


def _match(table: dict[str, InstanceRec], actions: list[ActionRec], entry: dict) -> list[str]:
    steps = set(entry["steps"])
    names = {_norm(n) for n in entry["raw_names"]}
    hits = []
    for key, rec in sorted(table.items()):
        if is_provisional(key) or is_context(rec):
            continue
        if not steps & {a.step for a in actions if a.target == key}:
            continue
        if names and not names & {_norm(n) for n in rec.raw_names}:
            continue
        hits.append(key)
    return hits


def _rename(table: dict[str, InstanceRec], actions: list[ActionRec], old: str, new: str,
            rec: Optional[InstanceRec] = None) -> None:
    """Move ``old`` to ``new`` in the parsed records: the row, its actions, every pointer."""
    held = table.pop(old)
    table[new] = replace(rec if rec is not None else held, key=new)
    for i, action in enumerate(actions):
        if action.target == old:
            actions[i] = replace(action, target=new)
    for key, other in list(table.items()):
        moved = repointed(other, old, new)
        if moved is not None:
            table[key] = moved


def replay_reclasses(entries: list[dict], instances: dict[str, InstanceRec],
                     actions: list[ActionRec], tax: Taxonomy
                     ) -> tuple[dict[str, InstanceRec], list[ActionRec], ReclassReplay]:
    """The parsed records with every reclass in ``entries`` applied. Pure.

    ``entries`` are :func:`effective_reclasses`. Returns new ``(instances,
    actions, report)``; the inputs are left alone.
    """
    table = dict(instances)
    out = list(actions)
    report = ReclassReplay()
    for e in entries:
        new, cls = e["new"], e["new_cls"]
        entry = {"ops": [e["op"]], "old": e["old"], "new": new, "cls": cls,
                 "steps": list(e["steps"]), "raw_names": list(e["raw_names"])}
        if e.get("extra"):
            report.already.append(entry | {"extra": True})     # the import keeps it as it is
            continue
        hits = _match(table, out, e)
        if len(hits) != 1:
            why = (f"the sheet names no part {_where(entry)}" if not hits
                   else f"several parts of the sheet fit {_where(entry)}: {', '.join(hits)}")
            report.unmatched.append(entry | {"why": why})
            continue
        parsed = hits[0]
        rec = table[parsed]
        if parsed == new and rec.cls == cls:
            report.already.append(entry)
            continue
        holder = table.get(new) if new != parsed else None
        if holder is not None and not (holder.cls == cls == rec.cls):
            report.unmatched.append(entry | {
                "why": f"the sheet's {new} is a {holder.cls} that is no part of this "
                       f"reclass"})
            continue
        report.inverse["instances"].append(asdict(rec))
        report.inverse["actions"].extend({"step": a.step, "idx": a.idx, "target": parsed,
                                          "verb": a.verb} for a in out if a.target == parsed)
        made = rec if rec.cls == cls else reclassed_record(rec, cls, new, tax)[0]
        if holder is not None:              # the importer gave the two keys the other way
            tmp = f"\x00{new}"
            _rename(table, out, parsed, tmp, made)
            _rename(table, out, new, parsed)
            _rename(table, out, tmp, new)
        else:
            _rename(table, out, parsed, new, made)
        recorded = {int(s): str(v) for s, v in e.get("verbs") or ()}
        carried = []
        for i, action in enumerate(out):
            if action.target != new or verb_applies(tax, action.verb, cls):
                continue
            verb = recorded.get(int(action.step))
            if verb and verb_applies(tax, verb, cls):
                carried.append([int(action.step), action.verb, verb])
                out[i] = replace(action, verb=verb)
        report.kept.append(entry | {"parsed": parsed, "verbs": carried})
    return table, out, report


def replay_reclass_for_import(db: Db, desktop: int, instances: dict[str, InstanceRec],
                              actions: list[ActionRec], tax: Taxonomy
                              ) -> tuple[dict[str, InstanceRec], list[ActionRec], ReclassReplay]:
    """:func:`replay_reclasses` with the reclasses in effect on ``desktop``.

    Must be called **before** the import writes: "in effect" is read off the
    stored instance table it is about to replace.
    """
    renames = reclass_renames(db, desktop)
    entries = effective_reclasses(db, desktop)
    if not entries:
        return dict(instances), list(actions), ReclassReplay(renames=renames)
    table, out, report = replay_reclasses(entries, instances, actions, tax)
    report.renames = renames
    return table, out, report
