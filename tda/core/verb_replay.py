"""A forced re-import keeps the verb corrections made with ``Shift+T`` (task U9 round 2).

``import-logs --force`` writes the freshly parsed actions over the desktop's
whole step table (:func:`tda.pipeline_logs._write_import`), so every open ⇄
remove correction an annotator made from the pictures (:mod:`tda.core.verb_fix`)
would silently go back to what the sheet says. Each correction is an
``op_log`` row of kind ``correct_verb`` -- the action's step, target and the
verb it flipped from and to -- so the import replays them onto the parsed
actions before it writes them, in op order:

* an op whose action is still there with the verb it flipped **from** is
  applied (the verb it flipped **to** must apply to the part's class);
* one whose action already carries the verb it flipped **to** needs nothing
  (C's importer rule now reads "remove SSD locker shield" as ``remove``);
* every other one is reported, with why, and not applied -- the row names
  another part now, the target is gone, the sheet has a third verb there.

Undo does not write the op log, so a correction taken back with ``Ctrl+Z`` is
still a row. What decides is the step table the import is about to replace:
the **latest** op on an action is in effect only when the stored action carries
the verb it flipped to; when it does not, every op on that action was taken
back (or overruled in Steps mode) and none of them is replayed.

Replaying twice is replaying once: the stored step table then carries the
replayed verbs, which is exactly what keeps the same ops in effect.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from tda.core.db import Db
from tda.core.model import ActionRec, InstanceRec
from tda.core.taxonomy import Taxonomy
from tda.core.verb_fix import OP_CORRECT_VERB

__all__ = ["OP_REPLAY", "Replay", "replay_corrections", "replay_for_import"]

#: ``op_log.kind`` of the one row a forced re-import writes about the replay.
OP_REPLAY = "replay_corrections"
SUCCESS = "success"


@dataclass
class Replay:
    """What a re-import did with the desktop's ``correct_verb`` ops.

    Each entry is ``{"op", "step", "target", "old", "new"}``; an ``unmatched``
    one also has ``"why"``. ``inverse`` holds the sheet's verb of every action
    the replay changed, which is what undoing the replay would write back.
    """

    replayed: list[dict] = field(default_factory=list)
    already: list[dict] = field(default_factory=list)
    unmatched: list[dict] = field(default_factory=list)
    inverse: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.replayed or self.already or self.unmatched)

    def payload(self) -> dict:
        return {"replayed": self.replayed, "already": self.already,
                "unmatched": self.unmatched}

    def kept_line(self) -> str:
        """``kept 6 verb correction(s) (Shift+T): ...`` or ``""``."""
        kept = self.replayed + self.already
        if not kept:
            return ""
        parts = [f"step {e['step']} {e['target']} {e['old']}->{e['new']}"
                 + ("" if e in self.replayed else " (already in the sheet)") for e in kept]
        return (f"kept {len(kept)} verb correction(s) made with Shift+T "
                f"({len(self.replayed)} replayed): " + ", ".join(parts))

    def unmatched_lines(self) -> list[str]:
        """One line per correction the new sheet no longer fits."""
        return [f"verb correction not replayed: step {e['step']} {e['target']} "
                f"{e['old']}->{e['new']} ({e['why']})" for e in self.unmatched]


def _entry(op: dict) -> dict:
    payload, inverse = op.get("payload") or {}, op.get("inverse") or {}
    return {"op": int(op["id"]), "step": int(payload["step"]),
            "target": str(payload["target"]), "old": str(inverse["verb"]),
            "new": str(payload["verb"])}


def _find(actions: list[ActionRec], step: int, target: str,
          verb: Optional[str] = None) -> Optional[int]:
    """Index of the successful action of ``target`` at ``step`` (with ``verb`` first)."""
    found = [i for i, a in enumerate(actions)
             if a.step == step and a.target == target and a.result == SUCCESS]
    if verb is not None:
        exact = [i for i in found if actions[i].verb == verb]
        if exact:
            return exact[0]
    return found[0] if found else None


def replay_corrections(ops: list[dict], stored: list[ActionRec], fresh: list[ActionRec],
                       instances: dict[str, InstanceRec],
                       tax: Taxonomy) -> tuple[list[ActionRec], Replay]:
    """The parsed actions with the ``correct_verb`` ops in effect replayed onto them.

    ``ops`` are the desktop's ``correct_verb`` rows (payload/inverse decoded),
    ``stored`` the step table the import is about to replace, ``fresh`` what
    the sheet parses to now. Pure: returns new action records and the
    :class:`Replay`; ``fresh`` is left alone.
    """
    out = list(fresh)
    report = Replay()
    entries = sorted((_entry(op) for op in ops), key=lambda e: e["op"])
    latest: dict[tuple[int, str], dict] = {}
    for entry in entries:
        latest[(entry["step"], entry["target"])] = entry
    for entry in entries:
        step, target, old, new = entry["step"], entry["target"], entry["old"], entry["new"]
        last = latest[(step, target)]
        held = _find(stored, step, target, last["new"])
        if held is None or stored[held].verb != last["new"]:
            had = "no such action" if held is None else stored[held].verb
            report.unmatched.append(entry | {
                "why": f"taken back before the import (the step table had {had} there)"})
            continue
        at = _find(out, step, target, old)
        if at is None:
            there = sorted({a.target for a in out if a.step == step})
            report.unmatched.append(entry | {
                "why": f"the sheet no longer has {target} at step {step} "
                       f"(it names {', '.join(there) or 'nothing'} there)"})
            continue
        if out[at].verb == new:
            report.already.append(entry)
            continue
        if out[at].verb != old:
            report.unmatched.append(entry | {
                "why": f"the sheet has {out[at].verb} there, not {old}"})
            continue
        rec = instances.get(target)
        applies = (tax.verbs.get(new) or {}).get("applies_to") or ()
        if rec is not None and rec.cls not in applies:
            report.unmatched.append(entry | {
                "why": f"{new} does not apply to class {rec.cls}"})
            continue
        report.inverse.append({"step": step, "idx": int(out[at].idx), "target": target,
                               "verb": out[at].verb})
        out[at] = replace(out[at], verb=new)
        report.replayed.append(entry)
    return out, report


def replay_for_import(db: Db, desktop: int, fresh: list[ActionRec],
                      instances: dict[str, InstanceRec],
                      tax: Taxonomy) -> tuple[list[ActionRec], Replay]:
    """:func:`replay_corrections` for one desktop, read before the import writes.

    Must be called **before** the step table is replaced: the stored actions
    are what say which corrections are still in effect.
    """
    ops = db.ops_of_kind(int(desktop), OP_CORRECT_VERB)
    if not ops:
        return list(fresh), Replay()
    return replay_corrections(ops, db.actions(int(desktop)), fresh, instances, tax)
