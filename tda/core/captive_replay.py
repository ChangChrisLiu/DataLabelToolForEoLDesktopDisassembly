"""A forced re-import keeps the captive flips made with ``Shift+T`` (task U14).

``import-logs --force`` re-creates every instance from the sheet, and the
importer guesses ``captive`` again (:func:`tda.core.log_identity.is_captive`:
every CPU-cooler screw of a Dell). A screw an annotator said was taken out
after all (D3's fan screws) would silently become captive again, and the card
would ask for it on frames where it is not.

So, as :mod:`tda.core.verb_replay` does for the verbs, the ``captive_fix``
rows of the op log say **which screws** an annotator flipped -- with ``Shift+T``,
or (task U27 round 2) in the ``Captive`` column of Steps' instance table -- and
for each the value the stored instance table carries right now -- the
annotator's last word, whatever undo/redo led there -- overrides the importer's
guess on the parsed record, matched by key (merged keys followed,
:mod:`tda.core.merge_replay`) while the parsed screw still carries the stored
one's raw name, else by that name (:func:`tda.core.relation_replay.locate_part`,
round 2: the relation replay finds the screw's ``parent`` / ``attached`` the
same way, so the two land on one screw when the sheet is renumbered):

* **kept** -- the stored value is a manual one and differs from the sheet's
  guess: the parsed record gets it (``captive_source="manual"``);
* **agrees** -- the guess already says the same (a manual value still goes in
  as ``captive_source="manual"``: the annotator's word stands, and a later
  change of the guess must not read it as a flip taken back), or the screw is
  one added in S1 (the import keeps its record as it is);
* **not matched** -- the sheet no longer has that key (or it is no screw any
  more), the table being replaced has none, or the flip was taken back
  (Ctrl+Z) and the stored value is the old guess; reported with why, nothing
  applied.

It runs before the relational inference, which fills a captive screw's
``parent`` from what it fastens -- so a screw kept captive rides out with its
part, and one kept loose gets no parent. Order-independent and idempotent.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from tda.core.db import Db
from tda.core.merge_replay import renamed
from tda.core.model import InstanceRec, is_context, is_extra, is_provisional
from tda.core.relation_replay import locate_part

__all__ = ["OP_CAPTIVE_FIX", "OP_REPLAY_CAPTIVE", "CaptiveReplay", "record_captive_fixes",
           "replay_captive", "replay_captive_for_import"]

#: ``op_log.kind`` of one flip; :mod:`tda.core.captive_fix` takes it from here,
#: so this module stays as light as the importer that imports it.
OP_CAPTIVE_FIX = "captive_fix"
#: ``op_log.kind`` of the one row a forced re-import writes about the flips.
OP_REPLAY_CAPTIVE = "replay_captive"
SCREW = "screw"
MANUAL = "manual"


def _word(value: object) -> str:
    return "captive" if value else "not captive"


def _first_name(rec: Optional[InstanceRec]) -> Optional[str]:
    return (list(rec.raw_names or ()) or [None])[0] if rec is not None else None


def _screw(entry: dict) -> str:
    """The screw's key, saying so when the sheet renumbered it (U27 round 2)."""
    parsed = entry.get("parsed")
    if parsed and parsed != entry["target"]:
        return f"{parsed} (was {entry['target']}, found by its name)"
    return entry["target"]


@dataclass
class CaptiveReplay:
    """What a re-import did with the screws ``Shift+T`` flipped.

    Each entry is ``{"target", "ops"}`` plus ``"captive"`` (the stored value)
    and ``"sheet"`` (the importer's guess) where there is one; a ``not
    matched`` entry also has ``"why"``. ``inverse`` holds the guess of every
    record the replay changed -- what undoing it would write back.
    """

    kept: list[dict] = field(default_factory=list)
    agrees: list[dict] = field(default_factory=list)
    unmatched: list[dict] = field(default_factory=list)
    inverse: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.kept or self.agrees or self.unmatched)

    def payload(self) -> dict:
        return {"kept": self.kept, "agrees": self.agrees, "unmatched": self.unmatched}

    def kept_line(self) -> str:
        """The flips that survive, in one line; ``""`` when there are none."""
        if not (self.kept or self.agrees):
            return ""
        parts = []
        if self.kept:
            parts.append(f"kept {len(self.kept)}: " + ", ".join(
                f"{_screw(e)} {_word(e['captive'])} (sheet: {_word(e['sheet'])})"
                for e in self.kept))
        if self.agrees:
            parts.append(f"the sheet already agrees on {len(self.agrees)}: " + ", ".join(
                f"{_screw(e)} {_word(e['captive'])}" for e in self.agrees))
        return "captive fixes (Shift+T) " + "; ".join(parts)

    def unmatched_lines(self) -> list[str]:
        """One line per flipped screw the new sheet (or the table) does not fit."""
        return [f"captive fix not matched: {e['target']} ({e['why']})" for e in self.unmatched]


def replay_captive(ops: list[dict], stored: dict[str, InstanceRec],
                   fresh: dict[str, InstanceRec],
                   rename: Optional[dict[str, str]] = None
                   ) -> tuple[dict[str, InstanceRec], CaptiveReplay]:
    """The parsed instances with the stored ``captive`` of every flipped screw.

    ``ops`` are the desktop's ``captive_fix`` rows (payload decoded) and only
    say which screws were flipped; ``stored`` is the instance table the import
    is about to replace, ``fresh`` what the sheet parses to now. ``rename`` is
    ``gone -> keep`` of the merges in effect. Pure: returns a new mapping
    (records changed are copies) and the :class:`CaptiveReplay`.
    """
    out = dict(fresh)
    report = CaptiveReplay()
    touched: dict[str, list[int]] = {}
    names: dict[str, str] = {}
    for op in sorted(ops, key=lambda row: int(row["id"])):
        payload = op.get("payload") or {}
        target = renamed(str(payload["target"]), rename)
        touched.setdefault(target, []).append(int(op["id"]))
        if payload.get("raw_name"):
            names[target] = str(payload["raw_name"])
    for target, ids in sorted(touched.items()):
        entry: dict = {"target": target, "ops": sorted(ids)}
        # task U27 round 2, B1: a flip recorded with the screw's raw name is
        # found by it -- in the table being replaced (an earlier re-import may
        # have renumbered the screw) and in the parse, the way the relation
        # replay finds the screw's parent / attached fixes; one recorded
        # without (before round 2) by key, and then by the stored screw's name
        name = names.get(target)
        held_key = (locate_part(stored, target, name)[0] if name else None) or target
        held = stored.get(held_key)
        if held is None or held.cls != SCREW:
            report.unmatched.append(entry | {
                "why": f"the instance table being replaced has no screw {target}"})
            continue
        attrs = held.attrs or {}
        entry["captive"] = bool(attrs.get("captive"))
        if held_key not in out and is_extra(held):
            report.agrees.append(entry | {"sheet": None})     # the import keeps it as is
            continue
        found, why = locate_part(out, held_key, name or _first_name(held), stored)
        rec = out.get(found) if found is not None else None
        if rec is None or rec.cls != SCREW:
            gone = found is None and not why.startswith("the sheet no longer has")
            report.unmatched.append(entry | {
                "why": why if gone else f"the sheet no longer has the screw {target}"})
            continue
        if found != target:
            entry["parsed"] = found
        target = found
        entry["sheet"] = bool((rec.attrs or {}).get("captive"))
        manual = attrs.get("captive_source") == MANUAL
        if entry["sheet"] != entry["captive"] and not manual:
            report.unmatched.append(entry | {
                "why": "the flip was taken back (Ctrl+Z) before the import; the sheet's "
                       "guess stands"})
            continue
        if manual:
            # the annotator's word, even where the guess agrees today: a later
            # change of the guess must not read it as a flip taken back
            report.inverse.append({"target": target, "captive": (rec.attrs or {}).get("captive"),
                                   "captive_source": (rec.attrs or {}).get("captive_source")})
            out[target] = replace(rec, attrs={**(rec.attrs or {}), "captive": entry["captive"],
                                              "captive_source": MANUAL})
        (report.agrees if entry["sheet"] == entry["captive"] else report.kept).append(entry)
    return out, report


def replay_captive_for_import(db: Db, desktop: int, fresh: dict[str, InstanceRec],
                              rename: Optional[dict[str, str]] = None
                              ) -> tuple[dict[str, InstanceRec], CaptiveReplay]:
    """:func:`replay_captive` for one desktop, read before the import writes.

    Must be called **before** the instance table is replaced: the stored
    records are the annotator's last word on every flipped screw.
    """
    ops = db.ops_of_kind(int(desktop), OP_CAPTIVE_FIX)
    if not ops:
        return dict(fresh), CaptiveReplay()
    return replay_captive(ops, db.instances(int(desktop)), fresh, rename)


# --------------------------------------------------------------------------- #
# the Captive column of Steps (task U27 round 2, B1)
# --------------------------------------------------------------------------- #
#: ``op_log`` scope of a flip made in Steps: an instance field belongs to no view.
STEPS_VIEW = "-"


def _flip_record(rec: InstanceRec) -> dict:
    """The fields one flip record carries (:meth:`tda.core.captive_fix.CaptiveFix.rows`)."""
    attrs = rec.attrs or {}
    return {"captive": attrs.get("captive"), "captive_source": attrs.get("captive_source"),
            "parent": rec.parent or None, "attached": bool(rec.attached)}


def record_captive_fixes(db: Db, desktop: int, before: dict[str, InstanceRec],
                         after: dict[str, InstanceRec], actions, by: str) -> list[str]:
    """Log a ``captive_fix`` row for every screw whose ``captive`` an Apply in Steps flips.

    The ``Captive`` column of Steps' instance table says the same thing as
    ``Shift+T``'s 「其实拧下来了」 / 「其实留在零件里」, and a forced re-import
    has to keep it the same way (:func:`replay_captive` reads these rows):
    without the row the flag went back to the guess while the ``attached`` set
    with it was replayed (B1). The row has the shape U14's has -- ``desktop``,
    ``target``, ``step`` (the screw's first successful ``unscrew``, ``None``
    when the log has none), ``at`` (``None``: no frame), the four fields,
    ``view``, ``reach`` (empty: Steps re-checks nothing here) and ``annotator``
    -- plus ``"source": "steps"``; the inverse is the record before. Joins the
    caller's transaction; returns the keys flipped.
    """
    flipped = []
    for key in sorted(set(before) & set(after)):
        old, new = before[key], after[key]
        if new.cls != SCREW or old.cls != SCREW or is_provisional(key) or is_context(new):
            continue
        if bool((old.attrs or {}).get("captive")) == bool((new.attrs or {}).get("captive")):
            continue
        steps = [a.step for a in actions
                 if a.target == key and a.verb == "unscrew" and a.result == "success"]
        common = {"desktop": int(desktop), "target": key, "step": min(steps) if steps else None,
                  "at": None, "view": STEPS_VIEW, "reach": {}, "annotator": str(by),
                  "raw_name": _first_name(new), "source": "steps"}
        db.log_op(int(desktop), STEPS_VIEW, OP_CAPTIVE_FIX, common | _flip_record(new),
                  common | _flip_record(old), str(by))
        flipped.append(key)
    return flipped
