"""The part every frame shows but the log never touches (user decision C7).

:mod:`tda.core.logs` only ever creates an instance a *step* names. That is the
right rule almost everywhere -- a desktop the sheet never mentions a PSU latch
on probably has none -- but it breaks on the four teardowns that stop before the
motherboard is lifted out: D49 ends at "CPU", D62 at "Remove half screws on the
motherboard", D63 and D64 unscrew the board and then only "change a direction".
Those sheets unscrew it, unplug it and end, so:

* 33 connectors and screws are left pointing at the *class* ``motherboard``,
  which has no instance on those desktops, and
* the board itself -- plainly visible in every single frame of them -- carries
  no identity, so it can never be labelled in the detection or segmentation
  export.

:func:`implied_instances` fills exactly that gap and nothing else. For a class
``configs/taxonomy.yaml`` lists under ``implied_when_referenced``, and only when
the desktop has no real instance of it yet *and* something references it, it
returns one ``<class>.01`` record carrying ``attrs["implied"] = True``. Nothing
operates on it, so the state machine leaves it ``installed`` / ``in_chassis``
for the whole sequence, which is precisely the truth about a board that was
never taken out. Afterwards the ordinary heuristic of
:mod:`tda.core.graph_infer` resolves the 33 references onto it.

The function is pure and idempotent: one instance per class at most, never one
where a real instance exists, and running it twice creates nothing the second
time. Stage S1 asks about every one of them
(:func:`tda.ui.steps_issues.unresolved_issues`), because "the dataset should
contain this part" is a judgement a human may overrule by deleting the row.

**A drive cage's own latch** (task U20, :func:`cage_latches`) is the second
rule. On the Dell OptiPlex desktops the HDD cage ("HDD shield") has a latch of
its own that has to be opened before the cage comes out, and the sheets mostly
never say so: D5/D6 log a *separate* removable "HDD shield locker" at step 10,
the drive at 13, the cage at 14, and nothing about the cage's latch; D13's
sheet does name it ("SSD shield locker", opened at 7, never removed). The
ruling: if a cage exists, its latch exists with it, and it is asked for when
the cage comes back -- closed from the start, open on the frame before the
cage's removal step (a derived default, :func:`tda.core.states.cage_latch_opens`),
out with the cage by the cascade. A cage the log never removes never comes
back, and gets none.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from tda.core.graph_infer import (
    SCREW_ROLE_CLASSES,
    blank,
    is_provisional,
    real_instances,
)
from tda.core.model import (
    CAGE_LATCH_RULE,
    IMPLIED_RULE_ATTR,
    ActionRec,
    InstanceRec,
    is_cage_latch,
    is_context,
    is_extra,
    machine_only,
)
from tda.core.states import removal_steps
from tda.core.taxonomy import Taxonomy

__all__ = [
    "CAGE_CLASS",
    "CAGE_LATCH_RULE",
    "CageLatches",
    "IMPLIED_ATTR",
    "IMPLIED_RULE_ATTR",
    "LATCH_CLASS",
    "NOTE_ATTR",
    "OP_CAGE_LATCH",
    "OP_DELETE_IMPLIED",
    "OP_KIND",
    "cage_latch_note",
    "cage_latches",
    "implied_instances",
    "implied_keys",
    "is_cage_latch",
    "is_implied",
    "referencing_instances",
]

#: ``attrs`` flag marking a record this module created rather than the log.
IMPLIED_ATTR = "implied"
#: ``attrs`` key holding the one-line explanation shown in stage S1.
NOTE_ATTR = "note"
#: ``op_log.kind`` written for every implied instance, so it can be audited and
#: undone record by record like any other machine-made change.
OP_KIND = "implied_instance"


def is_implied(rec: InstanceRec) -> bool:
    """Was this identity row implied rather than named by a step?"""
    return bool(rec.attrs.get(IMPLIED_ATTR))


def referencing_instances(
    instances: dict[str, InstanceRec], cls: str, tax: Taxonomy
) -> list[str]:
    """Keys of the instances that point at ``cls`` without naming an instance.

    Three references exist in practice and all three come straight out of the
    sheet plus the vocabulary: a connector whose ``socket_host`` is the bare
    class name (``logs.py`` writes ``"motherboard"`` when the step name says
    which part carries the socket but not which one), a screw whose ``role``
    names the class (:data:`~tda.core.graph_infer.SCREW_ROLE_CLASSES`), and an
    instance of a class whose ``host_class`` is this one and whose ``parent`` is
    still blank -- a ``ram_latch`` is a statement that this machine has a
    motherboard, whether or not the sheet ever named one. Those are exactly the
    rows :func:`~tda.core.graph_infer.unresolved_relations` reports as
    ``no candidate``, which is why a latch a human has already given a parent to
    is **not** counted: it is not missing anything.

    Provisional ``ls:*`` drafts are **not** references. They carry no relations
    yet by construction, so nothing about them is unresolved and nothing about
    them is evidence that the machine has this part: D66 holds eight
    ``ls:Motherboard Screw#k`` drafts and no imported sheet at all, and implying
    a board from those would invent an instance out of a drawing.

    The host test is deliberately **not** another arm of the class chain: a
    class that one day declares a ``host_class`` *and* is a connector or a screw
    would otherwise be swallowed by the older arm and silently stop counting.
    ``hit`` is what keeps it from counting such a row twice instead.
    """
    out: list[str] = []
    for key, rec in sorted(instances.items()):
        if is_provisional(key) or is_context(rec):
            continue    # a draft, or a 框外零件 (task U18): no evidence about the machine
        hit = False
        if rec.cls == "connector":
            hit = str(rec.socket_host or "").strip() == cls
        elif rec.cls == "screw":
            hit = cls in SCREW_ROLE_CLASSES.get(str(rec.attrs.get("role") or ""), ())
        if not hit and tax.host_class(rec.cls) == cls and blank(rec.parent):
            hit = True
        if hit:
            out.append(key)
    return out


def _note(referees: list[str]) -> str:
    """The one line stage S1 shows about an implied instance."""
    count = len(referees)
    return (
        f"implied: referenced by {count} instance{'' if count == 1 else 's'}, "
        f"never operated in the log"
    )


def implied_instances(
    instances: dict[str, InstanceRec],
    actions: Optional[Iterable[ActionRec]],
    tax: Taxonomy,
    declined: Optional[Iterable[str]] = None,
) -> list[InstanceRec]:
    """The instances this desktop obviously has but its log never names.

    One record per class of ``tax.implied_when_referenced`` that

    * is not in ``declined`` -- the classes whose implied instance this
      desktop's annotator has already deleted in S1
      (:meth:`tda.core.db.Db.declined_implied`). Implying is a judgement, and
      "no" is an answer that has to outlive the next ``import-logs``: without
      this, the deleted board came back on every re-import with a mask on every
      frame;
    * has **no** real (non-``ls:``) instance on this desktop -- a Label Studio
      draft is a draft, not a settled identity, and cannot stand in for one;
    * is referenced by at least one other instance
      (:func:`referencing_instances`) -- an unresolved ``socket_host``, a screw
      role, or a class that declares this one as its ``host_class``; and
    * has no action aimed at its ``<class>.01`` key, which would mean the log
      *did* operate on it and something else is wrong.

    The caller adds them to ``instances`` **before** running
    :func:`~tda.core.graph_infer.infer_relational_fields`, which then resolves
    the references onto them. Returns an empty list -- never ``None`` -- when
    there is nothing to imply, so a caller can write ``for rec in ...``
    unconditionally.
    """
    instances = machine_only(instances)     # a 框外零件 is never implied (task U18)
    if not instances:
        return []
    desktop = next(iter(instances.values())).desktop
    targeted = {a.target for a in (actions or ())}
    refused = {str(c) for c in (declined or ())}
    out: list[InstanceRec] = []
    for cls in tax.implied_when_referenced:
        key = f"{cls}.01"
        if cls in refused:
            continue
        if key in instances or real_instances(instances, cls) or key in targeted:
            continue
        referees = referencing_instances(instances, cls, tax)
        if not referees:
            continue
        out.append(InstanceRec(
            key=key,
            desktop=desktop,
            cls=cls,
            attrs={IMPLIED_ATTR: True, NOTE_ATTR: _note(referees)},
        ))
    return out


def implied_keys(instances: dict[str, InstanceRec]) -> list[str]:
    """The implied instances of one desktop's table, sorted (``status`` lists them)."""
    return sorted(key for key, rec in instances.items()
                  if is_implied(rec) and not is_context(rec))


# --------------------------------------------------------------------------- #
# a drive cage's own latch (task U20)
# --------------------------------------------------------------------------- #
#: The class whose instances carry a latch of their own, and that latch's class.
CAGE_CLASS = "drive_cage"
LATCH_CLASS = "drive_latch"
#: ``op_log.kind`` of the one row an import writes about the cages' latches:
#: which were adopted, kept, created, kept deleted (inverse: the adopted ones'
#: ``parent``/``attached`` as the sheet left them, and the created keys).
OP_CAGE_LATCH = "cage_latch"
#: ``op_log.kind`` of an implied cage latch deleted in Steps (stage S1); its
#: inverse is the whole record. Every later import reads these rows, so the
#: rule does not make the latch again (:func:`cage_latches`).
OP_DELETE_IMPLIED = "delete_implied"
_SUCCESS = "success"
_OPEN = "open"


def cage_latch_note(cage: str) -> str:
    """The note (``attrs["note"]``) of the latch implied for ``cage``."""
    return f"{cage} 自带的卡扣（日志里没有）"


@dataclass
class CageLatches:
    """What :func:`cage_latches` decided for one desktop's drive cages.

    * ``adopted`` -- ``{"latch", "cage", "opened"}``: a latch the log opens
      and never removes on its own, which is the cage's (D13);
    * ``kept`` -- ``{"latch", "cage", "how"}``: the cage already has its latch,
      added in S1 (D5) or implied before; nothing is done;
    * ``created`` -- the implied latches (records), ``opens`` the step each
      opens at by default (``None``: the cage leaves at step 1, no frame is
      before it);
    * ``kept_deleted`` -- ``{"latch", "cage", "ops"}``: the latch an annotator
      deleted in Steps; not made again;
    * ``unmatched`` -- ``{"latch", "cage", "ops", "why"}``: a deletion this
      sheet no longer needs (the cage has its latch now) or no longer fits (no
      such cage); reported, nothing applied.
    """

    adopted: list[dict] = field(default_factory=list)
    kept: list[dict] = field(default_factory=list)
    created: list[InstanceRec] = field(default_factory=list)
    opens: dict[str, Optional[int]] = field(default_factory=dict)
    kept_deleted: list[dict] = field(default_factory=list)
    unmatched: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.adopted or self.kept or self.created or self.kept_deleted
                    or self.unmatched)

    def payload(self) -> dict:
        """The op-log payload of :data:`OP_CAGE_LATCH`."""
        return {"adopted": self.adopted, "kept": self.kept,
                "created": [{"instance": rec.key, "cage": rec.parent,
                             "attrs": dict(rec.attrs), "opens": self.opens.get(rec.key)}
                            for rec in self.created],
                "kept_deleted": self.kept_deleted, "unmatched": self.unmatched}

    def inverse(self) -> dict:
        """What undoing it would write: the sheet's latches unparented, no new ones."""
        return {"adopted": [{"latch": e["latch"], "parent": None, "attached": False}
                            for e in self.adopted],
                "created": [rec.key for rec in self.created]}

    def implied_lines(self) -> list[str]:
        """One ``implied instance ...`` line per latch created (the import report)."""
        out = []
        for rec in self.created:
            step = self.opens.get(rec.key)
            when = (f"closed, open at step {step}, out with {rec.parent} at {step + 1}"
                    if step is not None else f"closed, out with {rec.parent}")
            out.append(f"implied instance {rec.key}: {rec.attrs.get(NOTE_ATTR, '')} - {when}")
        return out

    def lines(self) -> list[str]:
        """The adopted / kept / kept-deleted / not-matched lines, in that order."""
        out = [f"cage latch adopted: {e['latch']} is {e['cage']}'s own latch (opened at "
               f"step {e['opened']}, never removed on its own) - parent {e['cage']}, attached"
               for e in self.adopted]
        out += [f"cage latch kept: {e['latch']} ({e['how']}) is {e['cage']}'s own latch"
                for e in self.kept]
        out += [f"cage latch kept deleted: {e['latch']} of {e['cage']} was deleted in "
                f"Steps - not made again" for e in self.kept_deleted]
        out += [f"cage latch delete not matched: {e['latch']} of {e['cage']} ({e['why']})"
                for e in self.unmatched]
        return out


def _of(rec: Optional[InstanceRec]) -> str:
    return str(((rec.attrs or {}) if rec is not None else {}).get("of") or "").strip()


def _fits(latch: InstanceRec, cage: InstanceRec) -> tuple[bool, bool]:
    """``(does not contradict, says the same)`` of a latch's ``of`` and a cage's."""
    lo, co = _of(latch), _of(cage)
    same = bool(lo) and (lo == co or lo == cage.key)
    return (not lo or not co or same), same


def _first_steps(instances: dict[str, InstanceRec], actions: list[ActionRec],
                 tax: Taxonomy, cls: str, state: str) -> dict[str, int]:
    """``key -> first step a successful action puts it in `state``, for one class."""
    out: dict[str, int] = {}
    for action in sorted(actions, key=lambda a: (a.step, a.idx)):
        rec = instances.get(action.target)
        if (rec is None or rec.cls != cls or action.result != _SUCCESS
                or action.target in out):
            continue
        if tax.apply_verb(rec.cls, rec.attrs, action.verb) == ("state", state):
            out[action.target] = int(action.step)
    return out


def _follow(key: Optional[str], rename: Optional[dict[str, str]]) -> Optional[str]:
    """``key`` followed through the merges in effect (``gone -> keep``)."""
    seen: set[str] = set()
    while key and rename and key in rename and key not in seen:
        seen.add(key)
        key = rename[key]
    return key


def _deletions(deleted: Optional[Iterable[dict]],
               rename: Optional[dict[str, str]]) -> dict[str, dict]:
    """``cage -> {"latch", "cage", "ops"}`` of the cage latches deleted in Steps."""
    out: dict[str, dict] = {}
    for op in deleted or ():
        payload = op.get("payload") or {}
        if payload.get("rule") != CAGE_LATCH_RULE:
            continue
        cage = _follow(payload.get("cage"), rename) or ""
        entry = out.setdefault(cage, {"latch": str(payload.get("instance") or ""),
                                      "cage": cage, "ops": []})
        entry["latch"] = str(payload.get("instance") or entry["latch"])
        if op.get("id") is not None:
            entry["ops"].append(int(op["id"]))
    return out


def _ordinal(key: str) -> int:
    head, _, tail = str(key).rpartition(".")
    return int(tail) if head == LATCH_CLASS and tail.isdigit() else 0


def _latch_key(cage: str, busy: set[str], stored: dict[str, InstanceRec]) -> str:
    """The key of the latch implied for ``cage``: the one it had, or the next free.

    Re-importing must give the same key back -- shapes are drawn on it -- so a
    latch this rule made for the same cage before keeps its key while nothing
    else (the sheet, a part added in S1) has taken it. Otherwise the next
    ordinal after every ``drive_latch`` key in use or stored, so a key held by
    an instance the import keeps for review is never written over.
    """
    for key, rec in sorted(stored.items()):
        if is_cage_latch(rec) and rec.parent == cage and key not in busy:
            return key
    top = max((_ordinal(k) for k in busy | set(stored)), default=0)
    return f"{LATCH_CLASS}.{top + 1:02d}"


def cage_latches(
    instances: dict[str, InstanceRec],
    actions: Optional[Iterable[ActionRec]],
    tax: Taxonomy,
    extras: Optional[dict[str, InstanceRec]] = None,
    stored: Optional[dict[str, InstanceRec]] = None,
    deleted: Optional[Iterable[dict]] = None,
    rename: Optional[dict[str, str]] = None,
) -> CageLatches:
    """Give every drive cage of the sheet its own latch (task U20). Pure.

    ``instances``/``actions`` are what the sheet parses to (merges and
    ``Shift+T`` corrections replayed); ``extras`` the parts added in S1 the
    import keeps; ``stored`` the instance table being replaced (for the key an
    implied latch had); ``deleted`` the desktop's :data:`OP_DELETE_IMPLIED`
    rows; ``rename`` the merges in effect. For each ``drive_cage`` the log
    removes (its own action, at step ``c``), in that order:

    1. **adopt** -- a ``drive_latch`` of the sheet that the log opens before the
       cage comes out and never removes on its own, whose ``of`` does not
       contradict the cage's (the same, or either missing; the same first, then
       the one opened last), is the cage's: ``parent = cage``, ``attached``
       (D13's "SSD shield locker");
    2. **keep** -- else a latch added in S1, or implied before, whose
       ``parent`` is the cage: nothing to do (D5's, added by hand);
    3. **kept deleted** -- else, if an annotator deleted the latch implied for
       this cage in Steps: nothing is made;
    4. **create** -- else an implied ``drive_latch.NN`` (:func:`_latch_key`),
       ``parent = cage``, ``attached``, closed until the frame before the cage
       comes out and open there (:func:`tda.core.states.cage_latch_opens`).

    A cage the log never removes (or only displaces) is left alone: it never
    "comes back", which is the one frame the ruling asks for its latch on, and
    a latch implied for it would be a closed part to draw on every frame of the
    desktop -- the HP SFF sheets (D22-D33) displace their optical bay and never
    remove it. (An implied latch whose cage stops being removed later, by an
    edit in Steps, simply stays closed: :func:`tda.core.states.cage_latch_opens`.)
    Label Studio drafts and 框外零件 are never cages. Deterministic, and
    idempotent: run on its own result it adopts the same latch and keeps the
    created ones.
    """
    machine = {key: rec for key, rec in machine_only(instances).items()
               if not is_provisional(key)}
    plan = CageLatches()
    gone = _deletions(deleted, rename)
    actions = list(actions or ())
    removed = removal_steps(machine, actions, tax,
                            [key for key, rec in machine.items() if rec.cls == CAGE_CLASS])
    cages = sorted(removed)                 # the cages the log takes out
    opened = _first_steps(machine, actions, tax, LATCH_CLASS, _OPEN)
    alone = set(removal_steps(machine, actions, tax,
                              [k for k, r in machine.items() if r.cls == LATCH_CLASS]))
    side = {key: rec for key, rec in (extras or {}).items()
            if key not in machine and not is_context(rec) and not is_provisional(key)}
    pool = {**side, **machine}
    busy = set(instances) | set(side)
    stored = dict(stored or {})
    assigned: set[str] = set()
    for cage in sorted(cages, key=lambda k: (removed[k], k)):
        rec, out_at = machine[cage], removed[cage]
        found = []
        for key in opened:
            latch = machine[key]
            ok, same = _fits(latch, rec)
            if (key in alone or key in assigned or not ok or opened[key] >= out_at
                    or (latch.parent and latch.parent != cage)):
                continue
            found.append((0 if same else 1, -opened[key], key))
        if found:
            key = sorted(found)[0][2]
            assigned.add(key)
            plan.adopted.append({"latch": key, "cage": cage, "opened": opened[key]})
            continue
        own = sorted(key for key, other in pool.items()
                     if other.cls == LATCH_CLASS and other.parent == cage
                     and key not in assigned and (is_extra(other) or is_cage_latch(other)))
        if own:
            assigned.add(own[0])
            plan.kept.append({"latch": own[0], "cage": cage,
                              "how": "added in S1" if is_extra(pool[own[0]]) else "implied"})
            continue
        if cage in gone:
            plan.kept_deleted.append(gone.pop(cage))
            continue
        key = _latch_key(cage, busy, stored)
        busy.add(key)
        attrs = {IMPLIED_ATTR: True, IMPLIED_RULE_ATTR: CAGE_LATCH_RULE}
        if _of(rec):
            attrs["of"] = _of(rec)
        attrs[NOTE_ATTR] = cage_latch_note(cage)
        plan.created.append(InstanceRec(key=key, desktop=rec.desktop, cls=LATCH_CLASS,
                                        attrs=attrs, parent=cage, attached=True))
        plan.opens[key] = out_at - 1 if out_at > 1 else None
    for cage, entry in sorted(gone.items()):
        held = next((e for e in plan.adopted + plan.kept if e["cage"] == cage), None)
        if held is not None:
            why = f"{cage} has its latch {held['latch']} now"
        elif cage in machine and machine[cage].cls == CAGE_CLASS:
            why = f"the log no longer removes {cage}"
        else:
            why = f"the sheet has no drive cage {cage or '(none recorded)'}"
        plan.unmatched.append({**entry, "why": why})
    return plan
