"""A forced re-import keeps the merges made with 「合并到…」 (task U12).

``import-logs --force`` parses the sheet again, and the sheet still names both
halves of a merged part: D2's row 46 "Front Cover" parses to ``cover.02`` on
every run. Without this, the re-import would create ``cover.02`` again -- with
no shapes, the cover now drawn under ``cover.01`` -- and point step 46 back at
it: the merge silently undone.

So after the parse, every merge still in effect is applied to the parsed
records before anything is written, the way :mod:`tda.core.verb_replay` puts
the ``Shift+T`` verbs back:

* **in effect** = a ``merge_instances`` row of the op log whose ``gone`` is not
  in the stored instance table (an undone merge brought it back, so it is not
  replayed); ``keep`` is followed through later merges (``a`` into ``b``, then
  ``b`` into ``c``: ``a``'s actions end up on ``c``);
* **replayed** -- ``gone`` is in the fresh parse and ``keep`` too (or is a part
  added in S1, which the import keeps), with the merge's class: the parsed
  actions of ``gone`` are retargeted to ``keep``, ``gone`` leaves the parsed
  instance set and every parsed pointer at it -- ``parent``, ``mounted_on``,
  ``fastens``, ``socket_host`` and a latch's ``of`` (:func:`repointed`) --
  follows to ``keep``;
* **already** -- the sheet no longer names ``gone`` (the sheet was corrected):
  nothing to do;
* **not matched** -- the sheet lost ``keep``, the class changed, or the sheet
  now gives both an action at one step; reported with why, nothing applied.

Pure on the parsed records, order-preserving and idempotent: the second forced
import parses the same sheet and finds the same merges in effect. One
``replay_merges`` row in the op log says what happened.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from tda.core.db import Db
from tda.core.model import ActionRec, InstanceRec, is_extra

__all__ = ["OF_ATTR", "OP_MERGE", "OP_REPLAY_MERGES", "OP_UNDO", "OP_VIEW", "POINTERS",
           "MergeReplay", "effective_merges", "ranges", "renamed", "replay_merges",
           "replay_merges_for_import", "repointed"]

# The vocabulary :mod:`tda.core.merge_instances` shares with the importer. It
# lives here because ``tda.pipeline_logs`` imports this module and must stay
# light (no compiler, no pycocotools); the merge itself imports it from here.
#: ``op_log.kind`` of one merge, and of its undo.
OP_MERGE = "merge_instances"
OP_UNDO = "undo_merge_instances"
#: ``op_log`` is scoped per ``(desktop, view)``; a merge belongs to every view.
OP_VIEW = "-"
#: Instance fields that hold another instance's key.
POINTERS = ("parent", "mounted_on", "fastens", "socket_host")
#: The ``attrs`` entry that can hold one too: a latch's ``of``
#: (``ram_latch.01`` of ``ram_module.02``), which the ``locked_by`` rule reads.
OF_ATTR = "of"
#: ``op_log.kind`` of the one row a forced re-import writes about the merges.
OP_REPLAY_MERGES = "replay_merges"


def repointed(rec: InstanceRec, gone: str, keep: str) -> Optional[InstanceRec]:
    """A copy of ``rec`` with every pointer at ``gone`` moved to ``keep``.

    ``None`` when ``rec`` names ``gone`` nowhere. The pointers are
    :data:`POINTERS` and ``attrs["of"]`` (round 3: a latch ``of`` the merged-away
    module would otherwise dangle, and the next rule derivation would drop its
    ``locked_by`` edge in silence). ``keep``'s own pointer at ``gone`` would
    point at itself, so it is cleared -- ``parent`` takes ``attached`` with it,
    ``of`` leaves ``attrs``.
    """
    hits = [name for name in POINTERS if getattr(rec, name) == gone]
    of_hit = (rec.attrs or {}).get(OF_ATTR) == gone
    if rec.key == gone or not (hits or of_hit):
        return None
    new = copy.deepcopy(rec)
    itself = rec.key == keep
    for name in hits:
        setattr(new, name, None if itself else keep)
        if itself and name == "parent":
            new.attached = False
    if of_hit:
        if itself:
            new.attrs.pop(OF_ATTR, None)
        else:
            new.attrs[OF_ATTR] = keep
    return new


def ranges(steps: Iterable[int]) -> str:
    """``[37, 38, 39, 41]`` -> ``"37–39、41"``."""
    nums = sorted({int(s) for s in steps})
    out: list[str] = []
    start = prev = None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}" if start == prev else f"{start}–{prev}")
            start = prev = n
    if start is not None:
        out.append(f"{start}" if start == prev else f"{start}–{prev}")
    return "、".join(out)


@dataclass
class MergeReplay:
    """What a re-import did with the merges in effect.

    Each entry is ``{"ops", "gone", "keep"}``; a replayed one also has
    ``"steps"`` (the parsed actions it retargeted), a not-matched one ``"why"``.
    ``inverse`` is what undoing the replay would put back on the parse.
    """

    replayed: list[dict] = field(default_factory=list)
    already: list[dict] = field(default_factory=list)
    unmatched: list[dict] = field(default_factory=list)
    inverse: dict = field(default_factory=lambda: {"actions": [], "instances": []})
    #: ``gone -> keep`` of every merge in effect, chains resolved.
    rename: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.replayed or self.already or self.unmatched)

    def payload(self) -> dict:
        return {"replayed": self.replayed, "already": self.already,
                "unmatched": self.unmatched}

    def lines(self, skip: Iterable[str] = ()) -> list[str]:
        """What the import log says, one line per merge that did not just work.

        ``skip`` are ``gone`` keys another rule reports on in its own words:
        a drive cage's implied latch merged away is no part the sheet ever
        named, so "the sheet no longer needs" it would mislead (task U20).
        """
        out: list[str] = []
        if self.replayed:
            out.append("merges kept: " + ", ".join(
                f"{e['gone']} -> {e['keep']}"
                + (f" (steps {ranges(e['steps'])})" if e.get("steps") else "")
                for e in self.replayed))
        skipped = set(skip)
        already = [e for e in self.already if e["gone"] not in skipped]
        if already:
            out.append("merges the sheet no longer needs: " + ", ".join(
                f"{e['gone']} -> {e['keep']}" for e in already))
        out.extend(f"merge not matched: {e['gone']} -> {e['keep']} ({e['why']})"
                   for e in self.unmatched)
        return out


def effective_merges(db: Db, desktop: int) -> list[dict]:
    """The merges of ``desktop`` still in effect, oldest first.

    ``{"ops": [ids], "gone", "keep", "cls"}`` -- one per ``gone``, the latest
    row naming it. A ``gone`` the stored instance table holds again (an undo,
    or a re-created key) is not in effect.
    """
    stored = db.instances(int(desktop))
    latest: dict[str, dict] = {}
    ids: dict[str, list[int]] = {}
    for op in db.ops_of_kind(int(desktop), OP_MERGE):
        payload = op.get("payload") or {}
        gone = str(payload.get("gone") or "")
        if not gone:
            continue
        ids.setdefault(gone, []).append(int(op["id"]))
        latest[gone] = {"gone": gone, "keep": str(payload.get("keep") or ""),
                        "cls": payload.get("cls"), "last": int(op["id"])}
    out = []
    for gone, entry in sorted(latest.items(), key=lambda item: item[1]["last"]):
        if gone in stored:
            continue
        out.append({"ops": ids[gone], "gone": gone, "keep": entry["keep"],
                    "cls": entry["cls"]})
    return out


def _resolve(key: str, rename: dict[str, str]) -> str:
    seen = {key}
    while key in rename:
        key = rename[key]
        if key in seen:
            break
        seen.add(key)
    return key


def replay_merges(merges: list[dict], instances: dict[str, InstanceRec],
                  actions: list[ActionRec],
                  kept: Optional[dict[str, InstanceRec]] = None
                  ) -> tuple[dict, list[ActionRec], MergeReplay]:
    """The parsed records with every merge in ``merges`` applied.

    ``kept`` are the stored instances the import keeps although the sheet does
    not name them -- the parts added in S1 (:mod:`tda.core.extra`). A merge
    *into* one of them is as much in effect as one into a sheet part (round 3:
    it used to be reported "the sheet no longer has cover.03").

    Returns new ``(instances, actions, report)``; the inputs are left alone.
    """
    table = dict(instances)
    kept = dict(kept or {})
    out = list(actions)
    report = MergeReplay()
    for merge in merges:
        gone = merge["gone"]
        keep = _resolve(merge["keep"], report.rename)
        entry = {"ops": list(merge["ops"]), "gone": gone, "keep": keep}
        report.rename[gone] = keep
        keep_rec = table.get(keep) or kept.get(keep)
        if gone not in table:
            if keep_rec is not None:
                report.already.append(entry)
            else:
                report.unmatched.append(entry | {
                    "why": f"the sheet names neither {gone} nor {keep}"})
            continue
        if keep_rec is None:
            report.unmatched.append(entry | {"why": f"the sheet no longer has {keep}"})
            continue
        wanted = merge.get("cls")
        if table[gone].cls != keep_rec.cls or (wanted and table[gone].cls != wanted):
            report.unmatched.append(entry | {
                "why": f"the sheet makes {gone} a {table[gone].cls} and {keep} a "
                       f"{keep_rec.cls}"})
            continue
        both = sorted({a.step for a in out if a.target == gone}
                      & {a.step for a in out if a.target == keep})
        if both:
            report.unmatched.append(entry | {
                "why": f"the sheet gives both an action at step {ranges(both)}"})
            continue
        steps: list[int] = []
        for i, action in enumerate(out):
            if action.target == gone:
                report.inverse["actions"].append(
                    {"step": action.step, "idx": action.idx, "target": gone})
                out[i] = ActionRec(**{**vars(action), "target": keep})
                steps.append(action.step)
        report.inverse["instances"].append(asdict(table[gone]))
        del table[gone]
        for key, rec in list(table.items()):
            new = repointed(rec, gone, keep)
            if new is not None:
                table[key] = new
        report.replayed.append(entry | {"steps": sorted(set(steps))})
    return table, out, report


def replay_merges_for_import(db: Db, desktop: int, instances: dict[str, InstanceRec],
                             actions: list[ActionRec]
                             ) -> tuple[dict, list[ActionRec], MergeReplay]:
    """:func:`replay_merges` with the merges in effect on ``desktop``.

    Must be called **before** the import writes: "in effect" is read off the
    stored instance table it is about to replace, and so are the parts added
    in S1 that the import keeps (a merge may have gone into one of them).
    """
    merges = effective_merges(db, desktop)
    if not merges:
        return dict(instances), list(actions), MergeReplay()
    kept = {key: rec for key, rec in db.instances(int(desktop)).items()
            if is_extra(rec) and key not in instances}
    return replay_merges(merges, instances, actions, kept)


def renamed(target: str, rename: Optional[dict[str, str]]) -> str:
    """``target`` followed through the merges in effect (for the verb replay)."""
    return _resolve(target, rename or {})
