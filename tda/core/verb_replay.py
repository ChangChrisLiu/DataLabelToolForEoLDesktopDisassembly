"""A forced re-import keeps the verb corrections made with ``Shift+T`` (task U9).

``import-logs --force`` writes the freshly parsed actions over the desktop's
whole step table (:func:`tda.pipeline_logs._write_import`), so every open ⇄
remove correction an annotator made from the pictures (:mod:`tda.core.verb_fix`)
would silently go back to what the sheet says.

**The net effect, not the history.** Undo and redo write nothing to the op
log, so its ``correct_verb`` rows cannot say what the step table ended up with
(Shift+T, 「其实只是打开了」, then Ctrl+Z of the second leaves ``remove`` -- and
a replay of the rows in order gets that wrong). They only say **which actions**
an annotator corrected. For each such ``(step, target)`` the verb the stored
step table carries right now -- the annotator's last word, whatever the
undo/redo path to it -- is copied onto the parsed action:

* **kept** -- the stored verb is ``open`` or ``remove``, applies to the part's
  class, and differs from the sheet's: the parsed action gets it;
* **agrees** -- the sheet already says the same (D2 step 17, which the importer
  now reads as ``remove``; or a correction taken back);
* **not matched** -- either side has no successful action of that target at
  that step (the row names another part now, the part is gone), or the stored
  verb is not one a correction can leave; reported with why, nothing applied.

Order-independent and idempotent: a second forced import finds the stored
table carrying exactly the verbs the first one kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from tda.core.db import Db
from tda.core.merge_replay import renamed
from tda.core.model import ActionRec, InstanceRec
from tda.core.reclass_replay import follow
from tda.core.taxonomy import Taxonomy
from tda.core.verb_fix import FLIP, OP_CORRECT_VERB

__all__ = ["OP_REPLAY", "Replay", "replay_corrections", "replay_for_import"]

#: ``op_log.kind`` of the one row a forced re-import writes about the replay.
OP_REPLAY = "replay_corrections"
SUCCESS = "success"


@dataclass
class Replay:
    """What a re-import did with the actions ``Shift+T`` corrected.

    Each entry is ``{"step", "target", "ops"}`` plus ``"verb"`` (the stored,
    i.e. the annotator's) and ``"sheet"`` (the parsed) where there is one;
    a ``not matched`` entry also has ``"why"``. ``inverse`` holds the sheet's
    verb of every action the replay changed -- what undoing it would write back.
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
        """The corrections that survive, in one line; ``""`` when there are none."""
        if not (self.kept or self.agrees):
            return ""
        parts = []
        if self.kept:
            parts.append(f"kept the annotator's verb on {len(self.kept)} action(s): " + ", ".join(
                f"step {e['step']} {e['target']} {e['verb']} (sheet: {e['sheet']})"
                for e in self.kept))
        if self.agrees:
            parts.append(f"the sheet already agrees on {len(self.agrees)}: " + ", ".join(
                f"step {e['step']} {e['target']} {e['verb']}" for e in self.agrees))
        return "Shift+T corrections: " + "; ".join(parts)

    def unmatched_lines(self) -> list[str]:
        """One line per corrected action the new sheet (or the table) does not fit."""
        return [f"Shift+T correction not matched: step {e['step']} {e['target']} ({e['why']})"
                for e in self.unmatched]


def _find(actions: list[ActionRec], step: int, target: str) -> Optional[int]:
    """Index of the first successful action of ``target`` at ``step``, or ``None``."""
    return next((i for i, a in enumerate(actions)
                 if a.step == step and a.target == target and a.result == SUCCESS), None)


def _named_at(actions: list[ActionRec], step: int) -> str:
    return ", ".join(sorted({a.target for a in actions if a.step == step})) or "nothing"


def replay_corrections(ops: list[dict], stored: list[ActionRec], fresh: list[ActionRec],
                       instances: dict[str, InstanceRec],
                       tax: Taxonomy,
                       rename: Optional[dict[str, str]] = None,
                       reclassed: Optional[list[tuple[int, str, str]]] = None,
                       ) -> tuple[list[ActionRec], Replay]:
    """The parsed actions with the stored verb of every corrected action copied over.

    ``ops`` are the desktop's ``correct_verb`` rows (payload decoded) and only
    say which ``(step, target)`` were corrected; ``stored`` is the step table
    the import is about to replace, ``fresh`` what the sheet parses to now.
    ``rename`` is ``gone -> keep`` of the merges in effect (task U12,
    :mod:`tda.core.merge_replay`): a correction made on ``cover.02`` before it
    was merged into ``cover.01`` is about ``cover.01``'s action now.
    ``reclassed`` is ``(op id, old, new)`` of the reclasses not undone (task
    U21, :func:`tda.core.reclass_replay.reclass_renames`): a correction made on
    ``drive_latch.02`` before it became ``drive_cage.01`` follows it -- one
    made after does not. Pure: returns new action records and the :class:`Replay`.
    """
    out = list(fresh)
    report = Replay()
    touched: dict[tuple[int, str], list[int]] = {}
    for op in ops:
        payload = op.get("payload") or {}
        target = renamed(str(payload["target"]), rename)
        target = follow(target, int(op["id"]), reclassed)
        touched.setdefault((int(payload["step"]), target), []).append(int(op["id"]))
    for (step, target), ids in sorted(touched.items()):
        entry = {"step": step, "target": target, "ops": sorted(ids)}
        held = _find(stored, step, target)
        at = _find(out, step, target)
        if held is None:
            report.unmatched.append(entry | {
                "why": f"the step table being replaced has no {target} at step {step}"})
            continue
        verb = stored[held].verb
        entry["verb"] = verb
        if at is None:
            report.unmatched.append(entry | {
                "why": f"the sheet no longer has {target} at step {step} "
                       f"(it names {_named_at(out, step)} there)"})
            continue
        entry["sheet"] = out[at].verb
        rec = instances.get(target)
        applies = (tax.verbs.get(verb) or {}).get("applies_to") or ()
        if verb not in FLIP:
            report.unmatched.append(entry | {
                "why": f"the step table has {verb} there, which is no open/remove correction"})
        elif rec is not None and rec.cls not in applies:
            report.unmatched.append(entry | {
                "why": f"{verb} does not apply to class {rec.cls}"})
        elif out[at].verb == verb:
            report.agrees.append(entry)
        else:
            report.inverse.append({"step": step, "idx": int(out[at].idx), "target": target,
                                   "verb": out[at].verb})
            out[at] = replace(out[at], verb=verb)
            report.kept.append(entry)
    return out, report


def replay_for_import(db: Db, desktop: int, fresh: list[ActionRec],
                      instances: dict[str, InstanceRec],
                      tax: Taxonomy,
                      rename: Optional[dict[str, str]] = None,
                      reclassed: Optional[list[tuple[int, str, str]]] = None,
                      ) -> tuple[list[ActionRec], Replay]:
    """:func:`replay_corrections` for one desktop, read before the import writes.

    Must be called **before** the step table is replaced: the stored actions
    are the annotator's last word on every corrected action. ``rename`` and
    ``reclassed`` as in :func:`replay_corrections`.
    """
    ops = db.ops_of_kind(int(desktop), OP_CORRECT_VERB)
    if not ops:
        return list(fresh), Replay()
    return replay_corrections(ops, db.actions(int(desktop)), fresh, instances, tax, rename,
                              reclassed)
