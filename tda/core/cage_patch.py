"""The drive-cage latches one edit adds, drops or hangs on a cage (tasks U20, U21).

Two edits run the cage rule (:mod:`tda.core.implied`) for one cage in their
own transaction and op, so Ctrl+Z takes the cage's latches back with them:

* ``Shift+T`` on a cage's latch (:mod:`tda.ui.session_verb`): taken out on
  its own, the cage gets its implied latch; only opened again, the latch is
  adopted -- hung on the cage, ``parent = cage`` and ``attached`` -- and the
  cage's untouched implied latch goes (task U21: the blue drive-cage locker is
  never taken off the cage; it pivots open on the cage's big screw and leaves
  with it);
* 「其实是硬盘架」 / 「其实是卡扣」 (:mod:`tda.core.reclass`): a part the log
  called a latch becomes the cage and the rule runs for it; a cage that was a
  latch after all lets go of its latches.

A patch is ``{"desktop", "cage", "add": [records], "drop": [records],
"repoint": [{"key", "before": [parent, attached], "after": [...]}], ...}``;
:func:`mirror` is the other direction. Kept here, below the UI, because the
reclass (a core module) writes one too.
"""
from __future__ import annotations

from typing import Optional

from tda.core.db import Db
from tda.core.model import ALL_VIEWS, InstanceRec, is_cage_latch

__all__ = ["CagePatchError", "apply_cage_patch", "has_work", "mirror"]

#: ``relation.source`` of a derived edge: not somebody's work.
_RULE = "rule"


class CagePatchError(ValueError):
    """One direction of a cage patch that cannot be written; UI-ready Chinese."""


def has_work(db: Db, desktop: int, key: str) -> bool:
    """Did a human do anything on ``key``: a shape, a hand-set state, an override, an edge?"""
    if db.instance_reference_counts(desktop, key):
        return True
    if any(db.keyframes(desktop, view, key) for view in ALL_VIEWS):
        return True
    return any(key in (rel.get("target"), rel.get("blocker")) and rel.get("source") != _RULE
               for rel in db.relations(desktop))


def apply_cage_patch(db: Db, patch: Optional[dict]) -> None:
    """Drop, then re-hang, then add the cage latches one direction carries.

    **Joins the caller's transaction.** A latch to drop that is no longer the
    cage's implied latch, or that has work of its own now, is refused rather
    than deleted with it; a key to add that another part holds is refused too,
    and so is a latch to re-hang that is gone.
    """
    if not patch:
        return
    desktop = int(patch["desktop"])
    stored = db.instances(desktop)
    for record in patch.get("drop") or ():
        key = str(record["key"])
        held = stored.get(key)
        if held is None:
            continue                       # deleted in Steps since: nothing to take back
        if not is_cage_latch(held):
            raise CagePatchError(f"{key} 已经不是机架 {patch.get('cage')} 的隐含卡扣了"
                                 f"（在 Steps 里改过），这一步撤销不了")
        if has_work(db, desktop, key):
            raise CagePatchError(f"{key} 已经画过形状或手设过状态：先撤销那些（或到 Steps 里"
                                 f"处理），再撤销这一步")
        db.delete_auto_events(desktop, key)
        db.delete_instance(desktop, key)
        del stored[key]
    for move in patch.get("repoint") or ():
        key = str(move["key"])
        held = stored.get(key)
        if held is None:
            raise CagePatchError(f"{key} 已经不在了（在 Steps 里删过），它挂在哪个机架上改不回来")
        parent, attached = move["after"]
        held.parent, held.attached = parent, bool(attached)
        db.upsert_instance(held)
    for record in patch.get("add") or ():
        key = str(record["key"])
        held = stored.get(key)
        if held is not None and not is_cage_latch(held):
            raise CagePatchError(f"{key} 已经是另一个零件（{held.cls}）的键了：机架的隐含卡扣"
                                 f"加不回来")
        db.upsert_instance(InstanceRec(**record))


def mirror(patch: Optional[dict]) -> Optional[dict]:
    """The other direction of a cage patch: added is dropped, re-hung is hung back."""
    if not patch:
        return None
    return {**patch, "add": patch.get("drop") or [], "drop": patch.get("add") or [],
            "repoint": [{"key": m["key"], "before": m["after"], "after": m["before"]}
                        for m in patch.get("repoint") or ()],
            "kept": [], "also": [], "adopted": []}
