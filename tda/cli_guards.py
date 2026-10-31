"""The guards a one-off writing command takes before it touches the database (task W3a).

``pixel-edges`` and ``set-split`` write a handful of rows the paper's numbers
depend on, once, usually while the annotator's app is closed. Before the first
write each of them:

1. refuses while ``python -m tda.cli app`` is running (:func:`app_running`, by
   argv tokens; fail-closed: without ``psutil`` nobody can tell, and "yes" is
   the answer that keeps the live database safe);
2. holds the single-user lock (:func:`tda.cli_common.session`);
3. probes for a writer with ``BEGIN IMMEDIATE`` (:func:`writer_busy`);
4. backs the database up under a name that says what came next
   (:func:`named_backup`: ``tda_pre_<what>_<YYYYmmdd_HHMMSS>.sqlite`` in
   ``paths.yaml``'s ``backup_dir``, which follows the raw drive to today's
   letter).

A dry run takes none of them except the lock: it writes nothing.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from tda import pipeline as P

__all__ = ["app_running", "named_backup", "writer_busy"]


def app_running() -> bool:
    """Is ``python -m tda.cli app`` running? Fail-closed when ``psutil`` is missing."""
    try:
        import psutil
    except ImportError:
        return True
    me = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info.get("pid") == me:
                continue
            args = list(proc.info.get("cmdline") or [])
        except Exception:  # noqa: BLE001 - a process that went away
            continue
        if any(a == "tda.cli" and i + 1 < len(args) and args[i + 1] == "app"
               for i, a in enumerate(args)):
            return True
    return False


def writer_busy(db) -> Optional[str]:
    """``None`` when a write transaction can start now, else sqlite's reason."""
    try:
        db.conn.execute("BEGIN IMMEDIATE")
        db.conn.execute("COMMIT")
    except sqlite3.OperationalError as exc:
        return str(exc)
    return None


def named_backup(paths: dict, db, command: str, what: str) -> Optional[str]:
    """Back up into ``backup_dir`` as ``tda_pre_<what>_<stamp>.sqlite``; ``None`` (printed) on failure.

    The copy is the verified one :meth:`tda.core.db.Db.backup` makes, renamed
    afterwards so the rotation of the everyday copies never prunes it.
    """
    try:
        out = Path(db.backup(P.ready_backup_dest(paths), P.backup_keep(paths)))
        target = out.with_name(f"tda_pre_{what}_{time.strftime('%Y%m%d_%H%M%S')}.sqlite")
        n = 1
        while target.exists():
            target = out.with_name(f"tda_pre_{what}_{time.strftime('%Y%m%d_%H%M%S')}_{n}.sqlite")
            n += 1
        os.replace(out, target)
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"[{command}] 备份失败，什么也没写 / backup failed: {exc}; nothing was written")
        return None
    print(f"[{command}] 先备份了数据库 / backed the database up first -> {target}")
    return str(target)
