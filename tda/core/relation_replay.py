"""A forced re-import keeps the relational fields set in Steps (task U27).

On D23 the cable clip (``cable_clip.01``, "remove the cable from the case fan
clip", ``release`` at step 2) is part of the CPU fan shroud (``cover.01``,
removed at step 3): releasing it only lifts the clip to pull the cable out, and
the clip leaves with the shroud. The data model says that with two fields --
``parent = cover.01`` and ``attached`` -- and the cascade of spec 3.3 takes the
clip out at step 3 (``cable_clip`` is ``removed`` only that way). The annotator
sets them in Steps mode's instance table. ``import-logs --force`` re-creates
every instance from the sheet, though, and the heuristic fills the blanks again:
without this, the clip would be back on its own, in the chassis for ever.

**The record.** Every Apply in Steps (:meth:`tda.ui.steps_model.StepTableData.save`)
compares the instance table it writes with the one stored and logs one
:data:`OP_RELATION_FIX` row per changed field of :data:`FIELDS` -- ``{desktop,
key, raw_name, cls, field, old, new, by, at}`` (plus ``old_raw_name`` /
``new_raw_name`` when the value names a part), the inverse the same record with
``old`` and ``new`` swapped. A field set back to the importer's value is a change
like any other and is recorded too: the annotator's "no parent" must survive the
heuristic as much as their "parent = cover.01". Edits of parts staged in the same
Apply (a split compound row, a new target) are not: the re-import loses those
parts with the step table. ``adopt-relations`` records the same rows for edits
made before there was a record (:mod:`tda.cli_relations`).

**The replay.** Inside the import's transaction, after the merges, reclasses,
``Shift+T`` verbs and captive flags are replayed and after
:func:`~tda.core.graph_infer.infer_relational_fields` has filled the blanks,
the **latest** fix of every ``(part, field)`` is applied to the parsed records
(:func:`replay_relations`). Every op is resolved to the part it names in this
import first, and only then grouped (round 3, B2): after a renumbering a part
has fixes under its old key and its new one, and the op with the highest id
wins -- the others are "superseded", counted in one line:

* the instance is found **by key** -- followed through the reclasses made
  after the fix (:mod:`tda.core.reclass_replay`; a field the reclass rule
  cleared stays cleared) -- when the parsed part carries the raw name the fix
  was made on; else **by raw name** (one parsed part of the class carries it:
  the sheet was renumbered); else by key alone (the sheet renamed the row),
  unless the rows shifted under the key (:func:`locate_part`). A fix on a part
  merged away (合并到…, :mod:`tda.core.merge_replay`) is not replayed: the
  part it went into kept its own record;
* a value that names a part is followed through the merges in effect and the
  later reclasses -- as the merge and the reclass re-pointed it -- and found
  the same way (by key and its own raw name), and must still be a valid
  reference: the part exists after the import, it is not the part itself,
  ``fastens`` sits on a screw and ``socket_host`` on a connector, and a
  ``parent`` closes no loop; otherwise it is **not matched** (why) and the
  importer's value stays;
* a screw's ``parent`` / ``attached`` fix is not matched when the screw's
  ``captive`` after the captive replay is not the one it had when the fix was
  made ("captive differs", round 2, B1), and an ``attached = True`` whose
  part's ``parent`` fix was refused is "not applied (no parent)" (N5);
* **kept** -- applied over the importer's value; **agrees** -- the importer
  already says the same, or the part is one the import keeps as it is (added
  in S1, or a vanished part held for its work).

Pure on the parsed records, order-independent (the op ids order the fixes) and
idempotent: a second forced import reads the same rows and the same sheet. One
:data:`OP_REPLAY_RELATIONS` row per import says what happened. A fix that can
never match again (its part merged away, gone from the sheet) is reported on
every import; it stays in the op log (round 2, N3).

Round 2, B1: the captive replay (:mod:`tda.core.captive_replay`) finds its
screws with :func:`locate_part` too, and a ``Captive`` cell edited in Steps is
a ``captive_fix`` row like ``Shift+T``'s, so a screw's ``captive`` and its
``parent`` / ``attached`` come back together.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from tda.core.db import Db
from tda.core.dbrows import now_iso
from tda.core.merge_replay import OF_ATTR, POINTERS, renamed
from tda.core.model import InstanceRec, is_cage_latch, is_context, is_extra, is_provisional
from tda.core.reclass_replay import CLEARED_COLUMNS, POINTER_COLUMNS, RECLASS_TO, follow

__all__ = [
    "FIELDS", "OP_RELATION_FIX", "OP_REPLAY_RELATIONS", "OP_VIEW", "RETIRES", "RelationInputs",
    "RelationReplay", "fix_rows", "locate_part", "record_captive_turns",
    "record_relation_fixes", "relation_changes", "relation_inputs", "replay_relations",
    "value_of", "with_value",
]

#: ``op_log.kind`` of one field set in Steps (or adopted, ``adopt-relations``).
OP_RELATION_FIX = "relation_fix"
#: ``op_log.kind`` of the one row a forced re-import writes about them.
OP_REPLAY_RELATIONS = "replay_relations"
#: ``op_log`` is scoped per ``(desktop, view)``; an instance field belongs to none.
OP_VIEW = "-"
#: The instance fields a fix is recorded for and replayed on: the relational
#: columns of spec 7.1, the cable node, the grouping columns and a latch's
#: ``attrs["of"]`` (``of`` below).
FIELDS = ("parent", "attached", "mounted_on", "fastens", "socket_host", "cable",
          "slot_id", "group_id", "group_order", "removal_direction", OF_ATTR)
#: The fields that always name another instance; ``of`` does when it holds a
#: key (a RAM latch's module), see :func:`_names_a_part`.
REFERENCES = POINTERS
#: Relations only one class may carry (as the Steps table refuses them).
ONLY_ON = {"fastens": "screw", "socket_host": "connector"}
#: ``InstanceRec.group_order``'s default, which a blank cell means.
DEFAULT_ORDER = "unordered"
SCREW = "screw"
#: ``attrs`` entry of an implied instance's note (:data:`tda.core.implied.NOTE_ATTR`).
NOTE = "note"
#: The fix payload's copy of a drive cage's implied latch's note (round 3, R2-N2).
CAGE_NOTE = "cage_latch_note"
#: A screw's fields whose meaning hangs on its ``captive`` (round 2, B1): a
#: captive screw rides out on its ``parent`` when ``attached``, a plain one is
#: taken out at its unscrew. The fix records ``captive`` and is refused when the
#: screw's ``captive`` after the captive replay is another one.
CAPTIVE_BOUND = ("parent", "attached")


# --------------------------------------------------------------------------- #
# one field
# --------------------------------------------------------------------------- #
def value_of(rec: InstanceRec, name: str) -> object:
    """The value of one of :data:`FIELDS` on ``rec``, normalised for comparing.

    ``attached`` is a bool; ``group_order`` its default when blank; every other
    field ``None`` when blank (``""`` and whitespace are what a cleared cell or
    a hand-edited sheet leave), else the text.
    """
    if name == "attached":
        return bool(rec.attached)
    raw = (rec.attrs or {}).get(OF_ATTR) if name == OF_ATTR else getattr(rec, name, None)
    text = "" if raw is None else str(raw).strip()
    if name == "group_order":
        return text or DEFAULT_ORDER
    return text or None


def with_value(rec: InstanceRec, name: str, value: object) -> InstanceRec:
    """A copy of ``rec`` with one of :data:`FIELDS` set to ``value``."""
    new = copy.deepcopy(rec)
    if name == "attached":
        new.attached = bool(value)
    elif name == OF_ATTR:
        if value:
            new.attrs[OF_ATTR] = value
        else:
            new.attrs.pop(OF_ATTR, None)
    elif name == "group_order":
        new.group_order = str(value or DEFAULT_ORDER)
    else:
        setattr(new, name, value or None)
    return new


def _shown(value: object) -> str:
    return "none" if value is None else str(value)


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #
def relation_changes(before: Mapping[str, InstanceRec], after: Mapping[str, InstanceRec]
                     ) -> list[tuple[str, str, object, object]]:
    """``(key, field, old, new)`` of every :data:`FIELDS` value ``after`` changes.

    Only keys both tables hold, and neither a Label Studio draft nor a 框外零件
    (task U18): the import never rewrites those, so there is nothing to replay.
    Sorted by key, then in the order of :data:`FIELDS`.
    """
    out = []
    for key in sorted(set(before) & set(after)):
        old_rec, new_rec = before[key], after[key]
        if is_provisional(key) or is_context(old_rec) or is_context(new_rec):
            continue
        for name in FIELDS:
            old, new = value_of(old_rec, name), value_of(new_rec, name)
            if old != new:
                out.append((key, name, old, new))
    return out


def _first_name(rec: Optional[InstanceRec]) -> Optional[str]:
    return (list(rec.raw_names or ()) or [None])[0] if rec is not None else None


def fix_rows(desktop: int, rec: InstanceRec, name: str, old: object, new: object,
             by: str, at: str, parts: Optional[Mapping[str, InstanceRec]] = None
             ) -> tuple[dict, dict]:
    """``(payload, inverse)`` of one :data:`OP_RELATION_FIX` row.

    ``raw_name`` is the first of the part's raw names -- what the replay finds
    it by when the sheet is renumbered -- and ``cls`` its class. A value that
    names a part carries that part's first raw name too (``new_raw_name``,
    ``old_raw_name``, looked up in ``parts``), for the same reason. A screw's
    ``parent`` / ``attached`` also carries the screw's ``captive`` as it was
    when the fix was made (round 2, B1): the two only mean what they meant on
    a screw that is (not) captive the same way. The inverse is the same record
    with ``old`` and ``new`` swapped.
    """
    common = {"desktop": int(desktop), "key": rec.key, "raw_name": _first_name(rec),
              "cls": rec.cls, "field": name, "by": str(by), "at": str(at)}
    if rec.cls == SCREW and name in CAPTIVE_BOUND:
        common["captive"] = bool((rec.attrs or {}).get("captive"))
    if is_cage_latch(rec):
        # a drive cage's implied latch has no raw name; its note names the
        # cage, which is how a re-import made under a new key is recognised
        common[CAGE_NOTE] = (rec.attrs or {}).get(NOTE)
    parts = parts or {}
    names = {}
    if _names_a_part(name, old) or _names_a_part(name, new):
        names = {"old_raw_name": _first_name(parts.get(str(old))) if old else None,
                 "new_raw_name": _first_name(parts.get(str(new))) if new else None}
    swapped = {"old_raw_name": names.get("new_raw_name"),
               "new_raw_name": names.get("old_raw_name")} if names else {}
    return (common | {"old": old, "new": new} | names,
            common | {"old": new, "new": old} | swapped)


def record_relation_fixes(db: Db, desktop: int, before: Mapping[str, InstanceRec],
                          after: Mapping[str, InstanceRec], by: str
                          ) -> list[tuple[str, str, object, object]]:
    """Log one :data:`OP_RELATION_FIX` row per change; joins the caller's transaction.

    A screw whose ``captive`` this write flips also gets
    :func:`record_captive_turns` for the captive-bound fields it leaves alone.
    Returns :func:`relation_changes` -- the changes logged.
    """
    changes = relation_changes(before, after)
    at = now_iso()
    parts = {**before, **after}
    for key, name, old, new in changes:
        payload, inverse = fix_rows(desktop, after[key], name, old, new, by, at, parts)
        db.log_op(int(desktop), OP_VIEW, OP_RELATION_FIX, payload, inverse, str(by))
    record_captive_turns(db, desktop, before, after, by, at, parts)
    return changes


#: A :data:`OP_RELATION_FIX` row's list of the earlier rows of the same part
#: and field it retires (task U29, U27 R3-N1).
RETIRES = "retires"


def _captive(rec: InstanceRec) -> bool:
    return bool((rec.attrs or {}).get("captive"))


def _same_part(payload: dict, key: str, rec: InstanceRec) -> bool:
    """Is a fix row about this part -- under its key, or by its raw name?"""
    if str(payload.get("key") or "") == key:
        return True
    name = _norm(payload.get("raw_name"))
    return bool(name) and payload.get("cls") == rec.cls and _carries(rec, name)


def record_captive_turns(db: Db, desktop: int, before: Mapping[str, InstanceRec],
                         after: Mapping[str, InstanceRec], by: str,
                         at: Optional[str] = None,
                         parts: Optional[Mapping[str, InstanceRec]] = None) -> list[tuple]:
    """A screw's ``captive`` flipped: settle its captive-bound fixes the flip did not rewrite.

    U27 R3-N1. ``parent`` / ``attached`` fixes record the ``captive`` they were
    set under (round 2, B1) and are "captive differs" when the screw is the
    other kind at a re-import. A flip writes whichever of the two it changes,
    under the new ``captive`` -- but 「其实拧下来了」 keeps ``parent`` (inert, U14),
    so the ``parent`` fix made while it was captive stayed the latest word and
    every forced re-import printed it as not matched, for ever, over the right
    result. So, for each :data:`CAPTIVE_BOUND` field the write leaves as it was
    whose latest fix (by key, or by the screw's raw name) was set under the
    other ``captive``:

    * flipped **out of** captive -- a row that **retires** it (``retires`` =
      the ids of the part's fixes of that field so far): the field's value
      only meant "rides out with" on a captive screw, so no fix of it is in
      force any more and the importer's value stands; the replay lists it
      under ``retired`` and prints nothing;
    * flipped **into** captive -- a row that restates the value the screw kept
      under the new ``captive``, so the part it rides out with is replayed.

    Joins the caller's transaction (the Apply's, or ``Shift+T``'s and its
    undo / redo, which write through the same Apply). One query when a screw
    flipped; none otherwise. Returns ``(key, field, retired ids)``, ``[]`` for
    a restatement.
    """
    flipped = [key for key in sorted(set(before) & set(after))
               if after[key].cls == SCREW and before[key].cls == SCREW
               and _captive(before[key]) != _captive(after[key])]
    if not flipped:
        return []
    at = at or now_iso()
    parts = parts or {**before, **after}
    rows = sorted(db.ops_of_kind(int(desktop), OP_RELATION_FIX), key=lambda op: int(op["id"]))
    out: list[tuple] = []
    for key in flipped:
        rec, now = after[key], _captive(after[key])
        for name in CAPTIVE_BOUND:
            if value_of(before[key], name) != value_of(rec, name):
                continue                        # written just now, under the new captive
            mine = [op for op in rows if (op.get("payload") or {}).get("field") == name
                    and _same_part(op.get("payload") or {}, key, rec)]
            if not mine:
                continue
            last = mine[-1].get("payload") or {}
            if "captive" not in last or bool(last["captive"]) == now:
                continue
            value = value_of(rec, name)
            payload, inverse = fix_rows(desktop, rec, name, value, value, by, at, parts)
            retired: list[int] = []
            if not now:
                retired = [int(op["id"]) for op in mine]
                payload[RETIRES] = retired
            db.log_op(int(desktop), OP_VIEW, OP_RELATION_FIX, payload, inverse, str(by))
            out.append((key, name, retired))
    return out


# --------------------------------------------------------------------------- #
# the replay
# --------------------------------------------------------------------------- #
@dataclass
class RelationReplay:
    """What a re-import did with the relational fields set in Steps.

    Each entry is ``{"key", "field", "value", "ops"}`` -- the key and value as
    the replay followed them -- plus ``"sheet"`` (the importer's value) where
    there was a parsed part, ``"parsed"`` when the part was found under another
    key by its raw name, ``"named"`` (the key the value had) when the part the
    value names was, ``"kept_as_is"`` when the import does not rewrite the
    part, and ``"why"`` on a not-matched one. ``inverse`` holds the importer's
    value of every field the replay changed. ``superseded`` lists the fixes a
    later fix of the same part and field overrode although they were made under
    another key (round 3, B2: the sheet was renumbered in between) --
    ``{"key", "field", "ops", "by"}``, ``by`` being the op that won.
    """

    kept: list[dict] = field(default_factory=list)
    agrees: list[dict] = field(default_factory=list)
    unmatched: list[dict] = field(default_factory=list)
    inverse: list[dict] = field(default_factory=list)
    superseded: list[dict] = field(default_factory=list)
    #: Fields whose latest fix is a retirement (task U29): a screw's flip out of
    #: captive retired the ``parent`` set while it was captive. Not printed.
    retired: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.kept or self.agrees or self.unmatched or self.superseded
                    or self.retired)

    def payload(self) -> dict:
        return {"kept": self.kept, "agrees": self.agrees, "unmatched": self.unmatched,
                "superseded": self.superseded, "retired": self.retired}

    def changed(self) -> set[tuple[str, str]]:
        """``(key, field)`` of every field the replay wrote over the importer's value."""
        return {(e.get("parsed") or e["key"], e["field"]) for e in self.kept}

    def unfilled(self, fills: Iterable[str]) -> list[str]:
        """``fills`` without the inferred fields a fix has just overridden.

        :func:`~tda.core.graph_infer.infer_relational_fields` reports what it
        guessed, for S1 to confirm; a guess the annotator already replaced is
        no question any more.
        """
        gone = {f"{key}.{'attrs.of' if name == OF_ATTR else name} = "
                for key, name in self.changed()}
        return [line for line in fills if not any(line.startswith(p) for p in gone)]

    def kept_line(self) -> str:
        """The fixes that survive, in one line; ``""`` when there are none."""
        if not (self.kept or self.agrees):
            return ""
        parts = [f"{_where(e)} = {_value(e)} (sheet: {_shown(e['sheet'])})"
                 for e in self.kept]
        for e in self.agrees:
            how = ("kept as it is: " + e["kept_as_is"]) if e.get("kept_as_is") \
                else "the sheet agrees"
            parts.append(f"{_where(e)} = {_value(e)} ({how})")
        return "relation fixes kept: " + ", ".join(parts)

    def unmatched_lines(self) -> list[str]:
        """One line per fix the new sheet does not fit."""
        return [f"relation fix not matched: {e['key']}.{e['field']} = {_shown(e['value'])} "
                f"({e['why']})" for e in self.unmatched]

    def superseded_line(self) -> str:
        """``relation fixes superseded: n …``; ``""`` when there are none."""
        n = sum(len(e["ops"]) for e in self.superseded)
        if not n:
            return ""
        return (f"relation fixes superseded: {n} (made under an earlier key of the same "
                f"part, overridden by a later fix of that field)")

    def lines(self) -> list[str]:
        """What the import log says: the kept line, one line per miss, the superseded count."""
        head, tail = self.kept_line(), self.superseded_line()
        return ([head] if head else []) + self.unmatched_lines() + ([tail] if tail else [])


def _where(entry: dict) -> str:
    """``key.field``, saying so when the part was found under another key."""
    parsed = entry.get("parsed")
    if parsed and parsed != entry["key"]:
        return f"{parsed}.{entry['field']} (was {entry['key']}, found by its name)"
    return f"{entry['key']}.{entry['field']}"


def _value(entry: dict) -> str:
    """The value, saying so when the part it names was found under another key."""
    named = entry.get("named")
    return (f"{_shown(entry['value'])} (was {named}, found by its name)" if named
            else _shown(entry["value"]))


def _norm(name: object) -> str:
    return " ".join(str(name or "").lower().split())


def _cls_of_key(key: str) -> str:
    return str(key or "").split(".", 1)[0]


def _carries(rec: InstanceRec, name: str) -> bool:
    return bool(name) and name in {_norm(n) for n in rec.raw_names or ()}


def locate_part(table: Mapping[str, InstanceRec], key: str, raw_name: object,
                stored: Optional[Mapping[str, InstanceRec]] = None,
                cage_note: Optional[str] = None) -> tuple[Optional[str], str]:
    """The parsed part ``key`` (as the fix knew it) is now, or ``(None, why)``.

    1. **by key** when the parsed part carries the raw name the fix recorded
       (or the fix recorded none -- a part added in S1, an implied one);
    2. else **by raw name**: the one parsed part of the class that carries it
       (the sheet was renumbered: a row before it changed class);
    3. else, when nothing carries that name any more, by key alone (the sheet
       renamed the row) -- unless the part now at the key carries the name a
       *different* part had in ``stored`` (the table being replaced): then the
       rows shifted under the key and it is another part.

    Not matched: several parts of the class carry the name, it now names a part
    of another class, or there is neither the key nor the name -- said as
    "rebuilt as <key>" for a drive cage's implied latch the import made again
    under another key (round 2, N4), recognised by ``cage_note`` (the latch's
    note, as the fix recorded it, round 3) or the stored record's, so it is
    said on every import, not only the first. The captive replay finds its screws the
    same way (:func:`tda.core.captive_replay.replay_captive`, round 2, B1), so
    a screw's ``captive`` and its ``parent`` / ``attached`` follow a renumbered
    sheet together.
    """
    name = _norm(raw_name)
    rec = table.get(key)
    if rec is not None and (not name or _carries(rec, name)):
        return key, ""
    cls = rec.cls if rec is not None else _cls_of_key(key)
    named = [other for other, held in sorted(table.items())
             if _carries(held, name) and not is_provisional(other) and not is_context(held)]
    same = [other for other in named if table[other].cls == cls]
    if len(same) == 1:
        return same[0], ""
    if len(same) > 1:
        return None, f'several parts of the sheet are named "{raw_name}": {", ".join(same)}'
    if named:
        return None, (f'the sheet now makes "{raw_name}" a {table[named[0]].cls} '
                      f"({named[0]}), not a {cls}")
    if rec is None:
        rebuilt = _rebuilt_as(table, key, cage_note, (stored or {}).get(key))
        if rebuilt:
            return None, (f"rebuilt as {rebuilt}: the import made the drive cage's implied "
                          f"latch again, under a new key")
        tail = f' nor a part named "{raw_name}"' if name else ""
        return None, f"the sheet no longer has {key}{tail}"
    for now in rec.raw_names or ():
        before = [other for other, held in sorted((stored or {}).items())
                  if other != key and _carries(held, _norm(now))]
        if before:
            return None, (f'the sheet\'s {key} is "{now}" now, which was {before[0]} '
                          f'before; nothing is named "{raw_name}" any more')
    return key, ""


def _rebuilt_as(table: Mapping[str, InstanceRec], key: str, note: Optional[str],
                held: Optional[InstanceRec]) -> str:
    """The key a drive cage's implied latch ``key`` was made again under, or ``""``.

    :func:`tda.core.implied.cage_latches` gives an implied latch its old key
    back only while its ``parent`` is still the cage (task U20); one whose
    parent was changed in Steps comes back under the next free key, with the
    same note naming its cage. ``note`` is the one the fix recorded; a fix made
    before round 3 has none and the stored record's (``held``) is used, which
    only the first re-import still has.
    """
    if not note and held is not None and is_cage_latch(held):
        note = (held.attrs or {}).get(NOTE)
    if not note:
        return ""
    found = [other for other, rec in sorted(table.items())
             if other != key and is_cage_latch(rec) and (rec.attrs or {}).get(NOTE) == note]
    return found[0] if len(found) == 1 else ""


def _names_a_part(name: str, value: object) -> bool:
    """Does ``value`` in ``name`` name another instance?

    The four pointer columns always do. ``of`` does when it holds a key --
    a RAM latch's module, ``ram_module.02`` -- and not when it holds a word of
    its class's vocabulary (a cover's ``cpu_cooler``, a cage's ``hdd``); keys
    have a dot, the words do not.
    """
    if not value:
        return False
    return name in REFERENCES or (name == OF_ATTR and "." in str(value))


def _loops(table: Mapping[str, InstanceRec], key: str, parent: str) -> bool:
    """Would ``parent`` on ``key`` close a parent loop in ``table``?"""
    seen = {key}
    current: Optional[str] = parent
    while current:
        if current in seen:
            return True
        seen.add(current)
        held = table.get(current)
        current = held.parent if held is not None else None
    return False


def _refusal(table: Mapping[str, InstanceRec], target: str, name: str, value: object) -> str:
    """Why ``value`` cannot go into ``target``'s ``name`` after this import, or ``""``.

    ``value`` is already resolved (:func:`_named`): this only asks what the
    Steps table asks of an edit.
    """
    rec = table[target]
    only = ONLY_ON.get(name)
    if value and only and rec.cls != only:
        return f"only a {only} carries {name}; the sheet makes {target} a {rec.cls}"
    if not _names_a_part(name, value):
        return ""
    if value == target:
        return f"{target} cannot name itself"
    if name == "parent" and _loops(table, target, str(value)):
        return f"{value} is already below {target}; the parent would form a loop"
    return ""


def _named(table: Mapping[str, InstanceRec], kept: Mapping[str, InstanceRec],
           stored: Optional[Mapping[str, InstanceRec]], value: str, raw_name: object
           ) -> tuple[Optional[str], str]:
    """The part a fix's value names, after this import, or ``(None, why)``.

    A part the import keeps as it is (``kept``) is itself; any other is
    looked for as :func:`locate_part` looks for the fixed part -- by key and raw
    name -- so a renumbered sheet moves the value with its row.
    """
    if value in kept and value not in table:
        return value, ""
    found, why = locate_part(table, value, raw_name, stored)
    if found is None:
        return None, f"it names {value}: {why}"
    return found, ""


def _captive_differs(rec: InstanceRec, name: str, payload: dict) -> str:
    """Why a screw's ``parent`` / ``attached`` fix no longer fits its ``captive``, or ``""``.

    The fix recorded the screw's ``captive`` (:func:`fix_rows`); after the
    captive replay (``Shift+T`` or the Steps ``Captive`` column) the parsed
    screw may be the other kind -- the two replays found different screws, or
    the flag was never recorded -- and then ticking ``attached`` on a screw
    whose part leaves would keep it loosened in the chassis for ever (B1).
    """
    if rec.cls != SCREW or name not in CAPTIVE_BOUND or "captive" not in payload:
        return ""
    now = bool((rec.attrs or {}).get("captive"))
    if now == bool(payload["captive"]):
        return ""
    return (f"captive differs: {rec.key} is {'captive' if now else 'not captive'} now, the "
            f"fix was set while it was {'captive' if payload['captive'] else 'not captive'}")


def _cleared_by_reclass(name: str, value: object) -> bool:
    """Would the reclass rule have cleared this field (:func:`reclassed_record`)?

    A screw's or a connector's columns always; ``parent`` / ``mounted_on``
    when they name one of the two bay classes.
    """
    if name in CLEARED_COLUMNS:
        return True
    return name in POINTER_COLUMNS and _cls_of_key(str(value or "")) in RECLASS_TO


def _followed(value: object, op_id: int, rename: Optional[dict[str, str]],
              reclassed: Optional[list[tuple[int, str, str]]]) -> object:
    if not isinstance(value, str) or not value:
        return value
    return follow(renamed(value, rename), op_id, reclassed)


def replay_relations(ops: list[dict], fresh: Mapping[str, InstanceRec],
                     kept: Optional[Mapping[str, InstanceRec]] = None,
                     rename: Optional[dict[str, str]] = None,
                     reclassed: Optional[list[tuple[int, str, str]]] = None,
                     stored: Optional[Mapping[str, InstanceRec]] = None,
                     ) -> tuple[dict[str, InstanceRec], RelationReplay]:
    """The parsed instances with the latest fix of every ``(part, field)`` applied.

    ``ops`` are the desktop's :data:`OP_RELATION_FIX` rows (payload decoded);
    ``fresh`` what the import is about to write (merges, reclasses and the
    inference already done); ``kept`` the stored parts the import keeps as
    they are -- added in S1, or vanished and held for their work -- which a
    fix may name and which are not rewritten. ``rename`` is ``gone -> keep``
    of the merges in effect, ``reclassed`` :func:`tda.core.reclass_replay.reclass_renames`,
    ``stored`` the instance table being replaced (it tells a renamed row from
    rows that shifted under a key, :func:`locate_part`). Pure: returns a new
    mapping (changed records are copies) and the report.
    """
    table = dict(fresh)
    kept = dict(kept or {})
    report = RelationReplay()
    #: parts whose ``parent`` fix was refused: their ``attached = True`` fix
    #: would tick a box that carries them nowhere (round 2, N5)
    no_parent: set[str] = set()
    # round 3, B2: every op is first resolved to what it names in THIS import
    # -- the parsed part (by key and raw name), a part the import keeps as it
    # is, a part merged away, or nothing -- and only then grouped, by (that,
    # field): after a renumbering one part has fixes under its old key and its
    # new one, and only the latest op of all of them counts
    groups: dict[tuple, list[tuple[dict, str, str]]] = {}
    resolved: dict[tuple, tuple] = {}
    for op in sorted(ops, key=lambda row: int(row["id"])):
        payload = op.get("payload") or {}
        name = str(payload.get("field") or "")
        if name not in FIELDS:
            continue
        key = str(payload.get("key") or "")
        if key in (rename or {}):           # a merged-away key stays itself: see below
            where: tuple = ("merged", key, "")
        else:
            key = follow(key, int(op["id"]), reclassed)
            ask = (key, payload.get("raw_name"), payload.get(CAGE_NOTE))
            if ask not in resolved:
                if key in kept and key not in table:
                    resolved[ask] = ("kept", key, "")
                else:
                    target, why = locate_part(table, key, ask[1], stored, ask[2])
                    resolved[ask] = ("part", target, "") if target else ("missing", key, why)
            where = resolved[ask]
        groups.setdefault((where[0], where[1], name), []).append((op, key, where[2]))
    for (kind, target, name), rows in sorted(
            groups.items(), key=lambda item: (item[0][1], FIELDS.index(item[0][2]), item[0][0])):
        last, key, missing = rows[-1]
        payload = last.get("payload") or {}
        value = payload.get("new")
        if _names_a_part(name, value):
            value = _followed(value, int(last["id"]), rename, reclassed)
        entry = {"key": key, "field": name, "value": value,
                 "ops": [int(op["id"]) for op, k, _w in rows if k == key]}
        older: dict[str, list[int]] = {}
        for op, k, _w in rows:
            if k != key:
                older.setdefault(k, []).append(int(op["id"]))
        report.superseded.extend({"key": k, "field": name, "ops": ids, "by": int(last["id"])}
                                 for k, ids in sorted(older.items()))
        if payload.get(RETIRES) is not None:
            # a flip out of captive retired the field's fixes (U29, U27 R3-N1):
            # none is in force, the importer's value stands, nothing to say
            report.retired.append(entry)
            continue
        if kind == "merged":
            # 合并到… keeps the record of the part merged into: a field set on
            # the merged-away one went with it (the value's pointers follow)
            report.unmatched.append(entry | {
                "why": f"merged into {renamed(key, rename)}, which kept its own fields"})
            continue
        before = str(payload.get("key") or "")
        if before != key and _cleared_by_reclass(name, value):
            report.unmatched.append(entry | {
                "why": f"the reclass of {before} to {key} cleared it"})
            continue
        if kind == "kept":
            why = ("added in S1" if is_extra(kept[key])
                   else "no longer in the sheet, held for its work")
            report.agrees.append(entry | {"kept_as_is": why})
            continue
        if kind == "missing":
            report.unmatched.append(entry | {"why": missing})
            continue
        if target != key:
            entry["parsed"] = target
        entry["sheet"] = value_of(table[target], name)
        why = _captive_differs(table[target], name, payload)
        if not why and _names_a_part(name, value):
            found, why = _named(table, kept, stored, str(value), payload.get("new_raw_name"))
            if found is not None and found != value:
                entry["named"], value = value, found     # the sheet renumbered it too
                entry["value"] = value
        if not why and name == "attached" and value and target in no_parent:
            why = "not applied (no parent): the parent fix of this part was not matched"
        why = why or _refusal(table, target, name, value)
        if why:
            report.unmatched.append(entry | {"why": why})
            if name == "parent":
                no_parent.add(target)
            continue
        if entry["sheet"] == value:
            report.agrees.append(entry)
            continue
        report.inverse.append({"key": target, "field": name, "value": entry["sheet"]})
        table[target] = with_value(table[target], name, value)
        report.kept.append(entry)
    return table, report


# --------------------------------------------------------------------------- #
# for the import
# --------------------------------------------------------------------------- #
@dataclass
class RelationInputs:
    """What the replay reads off the database, before the import writes (U27)."""

    ops: list[dict] = field(default_factory=list)
    #: The instance table the import is about to replace.
    stored: dict[str, InstanceRec] = field(default_factory=dict)
    #: The stored records of the vanished parts the import keeps for their work.
    held: dict[str, InstanceRec] = field(default_factory=dict)


def relation_inputs(db: Db, desktop: int, holding: Iterable[str] = ()) -> RelationInputs:
    """The fixes of ``desktop``, its stored table and the held parts; read before writing."""
    ops = db.ops_of_kind(int(desktop), OP_RELATION_FIX)
    if not ops:
        return RelationInputs()
    stored = db.instances(int(desktop))
    return RelationInputs(ops=ops, stored=stored,
                          held={key: stored[key] for key in set(holding or ()) if key in stored})
