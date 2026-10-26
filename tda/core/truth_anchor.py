"""What a frozen row is compared against: the row a person signed (task U32, round 2).

A frozen row is compared with every re-compilation, and a difference beyond
the re-tracing tolerance is a conflict (spec 3.4).  「重签」
(:mod:`tda.core.truth_refreeze`) writes today's compile into rows that differ
only *within* it -- and the tolerance is not transitive: compared against the
row the last re-signing wrote, a row re-signed again and again could walk any
distance from what anybody confirmed without a conflict ever opening (the U32
review's five re-traces of three columns each: 0 conflicts, a 400 px screw at
700 px).

So the comparison is made against the row **a person signed**: when the
stored row is exactly what a ``refreeze_frame`` op wrote, the op's inverse is
followed back to the row it replaced, and on through earlier re-signings,
until a row no re-signing wrote -- an annotator's confirmation, an
``accept_new``, a 可见性审查 verdict, or the original an undo put back.  The
stored row is never more than one tolerance from that row, and the compile
can move no further than one tolerance from it before the ordinary refresh
raises the conflict.

:func:`anchor_history` reads the view's ``refreeze_frame`` ops once and keeps
them on the connection until another op is logged (``MAX(id)`` of the log),
so the common case -- no re-signing ever happened on this frame -- costs one
indexed query and no decoding.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

from tda.core import dbrows as R

__all__ = ["REFREEZE_OP", "anchor_history", "decoded", "human_rows", "rows_differ",
           "signed_by_hand"]

#: The op a re-signing logs per frame (:mod:`tda.core.truth_refreeze`).
REFREEZE_OP = "refreeze_frame"


def rows_differ(now: Optional[dict], wanted: dict) -> bool:
    """Is the stored raw row ``now`` not exactly ``wanted`` (every column ``wanted`` names)?"""
    return now is None or any(now.get(c) != wanted.get(c) for c in wanted)


def anchor_history(db, desktop: int, view: str) -> dict[int, list[dict]]:
    """The ``refreeze_frame`` ops of one view, per step, oldest first; cached per log state."""
    conn = getattr(db, "conn", None)
    if conn is None:            # a stand-in without a database: nothing was re-signed
        return {}
    try:
        newest = conn.execute("SELECT MAX(id) FROM op_log").fetchone()[0]
        cache = db.__dict__.setdefault("_refreeze_history", {})
        held = cache.get((int(desktop), str(view)))
        if held is not None and held[0] == newest:
            return held[1]
        rows = conn.execute(
            "SELECT * FROM op_log WHERE desktop=? AND view=? AND kind=? ORDER BY id",
            (int(desktop), str(view), REFREEZE_OP),
        ).fetchall()
    except (sqlite3.Error, AttributeError, TypeError):
        return {}
    out: dict[int, list[dict]] = {}
    for row in rows:
        op = R.json_row(row, "payload", "inverse")
        out.setdefault(int((op.get("payload") or {}).get("step", -1)), []).append(op)
    cache[(int(desktop), str(view))] = (newest, out)
    return out


def signed_by_hand(row: dict, instance: str, ops: list[dict]) -> Optional[dict]:
    """The raw row a person signed, when ``row`` is one a re-signing wrote; else ``None``.

    ``row`` is the stored raw row (every column); ``ops`` the frame's
    ``refreeze_frame`` ops.  While the row is exactly what one of them wrote
    for ``instance`` (its ``payload.rows``), step to what that op replaced (its
    ``inverse.rows``) -- the newest such op first, so a row re-signed by two
    runs is walked through both.  The first row no re-signing wrote is the
    person's.  ``None`` when ``row`` itself is that row (the common case).
    """
    current, walked = row, False
    for _ in range(len(ops) + 1):
        writer = None
        for op in reversed(ops):
            wrote = ((op.get("payload") or {}).get("rows") or {}).get(instance)
            if wrote is not None and not rows_differ(current, wrote):
                writer = op
                break
        before = None if writer is None else \
            ((writer.get("inverse") or {}).get("rows") or {}).get(instance)
        if before is None:
            return current if walked else None
        current, walked = before, True
    return current


def decoded(raw: dict) -> dict:
    """A raw row shaped like :meth:`tda.core.db.Db.compiled`'s (``visible_rle``, ``box``)."""
    out = {k: v for k, v in raw.items() if k not in ("visible_rle_json", "box_json")}
    out["visible_rle"] = R.loads(raw.get("visible_rle_json"))
    out["box"] = R.loads(raw.get("box_json"))
    return out


def human_rows(db, key, instances) -> dict[str, dict]:
    """``instance -> decoded row a person signed`` for those of ``instances`` a re-signing wrote.

    Empty -- one cached query, nothing decoded -- on a frame no re-signing touched.
    """
    ops = anchor_history(db, key.desktop, key.view).get(int(key.step))
    if not ops or not callable(getattr(db, "compiled_raw", None)):
        return {}
    raw = db.compiled_raw(key)
    out = {}
    for instance in instances:
        if instance not in raw:
            continue
        human = signed_by_hand(raw[instance], instance, ops)
        if human is not None:
            out[instance] = decoded(human)
    return out
