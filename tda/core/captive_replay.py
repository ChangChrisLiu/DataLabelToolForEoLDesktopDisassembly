"""A forced re-import keeps the captive flips made with ``Shift+T`` (task U14).

``import-logs --force`` re-creates every instance from the sheet, and the
importer guesses ``captive`` again (:func:`tda.core.log_identity.is_captive`:
every CPU-cooler screw of a Dell). A screw an annotator said was taken out
after all (D3's fan screws) would silently become captive again, and the card
would ask for it on frames where it is not.

So, as :mod:`tda.core.verb_replay` does for the verbs, the ``captive_fix``
rows of the op log say **which screws** an annotator flipped, and for each the
value the stored instance table carries right now -- the annotator's last
word, whatever undo/redo led there -- overrides the importer's guess on the
parsed record, matched by key (merged keys followed, :mod:`tda.core.merge_replay`):

* **kept** -- the stored value is a manual one and differs from the sheet's
  guess: the parsed record gets it (``captive_source="manual"``);
* **agrees** -- the guess already says the same, or the screw is one added in
  S1 (the import keeps its record as it is);
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
from tda.core.model import InstanceRec, is_extra

__all__ = ["OP_CAPTIVE_FIX", "OP_REPLAY_CAPTIVE", "CaptiveReplay", "replay_captive",
           "replay_captive_for_import"]

#: ``op_log.kind`` of one flip; :mod:`tda.core.captive_fix` takes it from here,
#: so this module stays as light as the importer that imports it.
OP_CAPTIVE_FIX = "captive_fix"
#: ``op_log.kind`` of the one row a forced re-import writes about the flips.
OP_REPLAY_CAPTIVE = "replay_captive"
SCREW = "screw"
MANUAL = "manual"


def _word(value: object) -> str:
    return "captive" if value else "not captive"


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
                f"{e['target']} {_word(e['captive'])} (sheet: {_word(e['sheet'])})"
                for e in self.kept))
        if self.agrees:
            parts.append(f"the sheet already agrees on {len(self.agrees)}: " + ", ".join(
                f"{e['target']} {_word(e['captive'])}" for e in self.agrees))
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
    for op in ops:
        target = renamed(str((op.get("payload") or {})["target"]), rename)
        touched.setdefault(target, []).append(int(op["id"]))
    for target, ids in sorted(touched.items()):
        entry: dict = {"target": target, "ops": sorted(ids)}
        held = stored.get(target)
        if held is None or held.cls != SCREW:
            report.unmatched.append(entry | {
                "why": f"the instance table being replaced has no screw {target}"})
            continue
        attrs = held.attrs or {}
        entry["captive"] = bool(attrs.get("captive"))
        rec = out.get(target)
        if rec is None and is_extra(held):
            report.agrees.append(entry | {"sheet": None})     # the import keeps it as is
            continue
        if rec is None or rec.cls != SCREW:
            report.unmatched.append(entry | {
                "why": f"the sheet no longer has the screw {target}"})
            continue
        entry["sheet"] = bool((rec.attrs or {}).get("captive"))
        if entry["sheet"] == entry["captive"]:
            report.agrees.append(entry)
        elif attrs.get("captive_source") != MANUAL:
            report.unmatched.append(entry | {
                "why": "the flip was taken back (Ctrl+Z) before the import; the sheet's "
                       "guess stands"})
        else:
            report.inverse.append({"target": target, "captive": (rec.attrs or {}).get("captive"),
                                   "captive_source": (rec.attrs or {}).get("captive_source")})
            out[target] = replace(rec, attrs={**(rec.attrs or {}), "captive": entry["captive"],
                                              "captive_source": MANUAL})
            report.kept.append(entry)
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
