"""``notes``: what the annotator wrote about the frames of one machine (task U35).

::

    python -m tda.cli notes --desktop N [--view V] [--format md] [--out FILE]

Review's 「这一帧的说明」 box keeps one note per frame (``frame_note``); this
prints them -- step, author, time and the text -- per view, so whoever checks
the data reads what the annotator found odd, decided or saw.  ``--format md``
gives Markdown (a section per view, a heading per frame, the text quoted);
``--out`` writes the same text to a UTF-8 file instead of printing it.

**Read-only, for real** (review round 2, N2): the file is opened through a
``file:...?mode=ro`` URI and *not* through :class:`~tda.core.db.Db`, whose
schema bootstrap would create the table and stamp version 6 -- so reading the
notes never upgrades a database, and the live file's first v6 open is the
annotator's app after the update.  No lock either.  A database still at
version 5 has no notes yet: that is said, and the exit code is 0.

The text is printed as UTF-8 whatever the console's code page (a note is
Chinese more often than not; a Windows pipe defaults to cp1252).  In
PowerShell, redirecting or piping that output decodes it with the console's
own code page and garbles the Chinese: use ``--out FILE``, or switch the
console to UTF-8 first (``chcp 65001``).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from tda import pipeline as P
from tda.cli_common import EXIT_ERROR, EXIT_OK, load_paths
from tda.core.db_notes import FrameNotesMixin
from tda.core.frame_notes import notes_markdown, notes_text
from tda.core.model import ALL_VIEWS

__all__ = ["ENCODING_HINT", "NO_TABLE", "_add_notes", "cmd_notes", "read_only_uri"]

#: What a database without the table (schema version 5) says.
NO_TABLE = ("D{desktop:02d}: 没有说明（库结构 v{version}，应用升级后第一次打开会建表）/ "
            "no frame notes (schema version {version}; the updated app creates the table "
            "the first time it opens the file)")
#: ``--help``'s word on the console (review round 2, N5).
ENCODING_HINT = ("在 PowerShell 里转存或用管道时用 --out FILE，或先 chcp 65001 把控制台设成 UTF-8，"
                 "否则中文会乱码 / in PowerShell, redirect with --out FILE or run chcp 65001 "
                 "first: piped output is decoded with the console's code page")


def read_only_uri(path: str) -> str:
    """``file:`` URI of ``path`` that sqlite may only read."""
    absolute = os.path.abspath(str(path)).replace("\\", "/")
    return f"file:{quote(absolute, safe='/:')}?mode=ro"


class _NoteReader(FrameNotesMixin):
    """:class:`~tda.core.db_notes.FrameNotesMixin` over a read-only connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def has_table(self) -> bool:
        return self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                 "AND name='frame_note'").fetchone() is not None

    def schema_version(self) -> Optional[str]:
        try:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        except sqlite3.Error:
            return None
        return None if row is None else str(row[0])

    def has_desktop(self, desktop: int) -> bool:
        return self.conn.execute("SELECT 1 FROM desktop WHERE id=?",
                                 (int(desktop),)).fetchone() is not None


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):   # not a real stream, or already set
        pass


def cmd_notes(args: argparse.Namespace) -> int:
    """Print (or write) the notes of one desktop, optionally one view."""
    paths = load_paths(args.paths)
    db_path = args.db or P.require(paths, "db_path")
    if not os.path.exists(db_path):
        print(f"[notes] {db_path} does not exist")
        return EXIT_ERROR
    try:
        conn = sqlite3.connect(read_only_uri(db_path), uri=True)
    except sqlite3.Error as exc:
        print(f"[notes] cannot open {db_path} read-only: {exc}")
        return EXIT_ERROR
    conn.row_factory = sqlite3.Row
    try:
        reader = _NoteReader(conn)
        desktop = int(args.desktop)
        if not reader.has_desktop(desktop):
            print(f"[notes] desktop {desktop} is not in the database")
            return EXIT_ERROR
        if not reader.has_table():
            _utf8_stdout()
            print(NO_TABLE.format(desktop=desktop, version=reader.schema_version() or "?"))
            return EXIT_OK
        notes = reader.frame_notes(desktop, args.view)
    finally:
        conn.close()
    render = notes_markdown if args.format == "md" else notes_text
    text = render(desktop, notes)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8", newline="\n")
        print(f"[notes] D{desktop:02d}: {len(notes)} note(s) -> {args.out}")
        return EXIT_OK
    _utf8_stdout()
    print(text)
    return EXIT_OK


def _add_notes(sub) -> None:
    p = sub.add_parser("notes", help="print the annotator's notes on the frames (Review's "
                                     "这一帧的说明) of one desktop; read-only",
                       description="Print the notes Review's 这一帧的说明 box keeps, per view: "
                                   "step, author, time, text. Opens the database read-only "
                                   "(never upgrades it).",
                       epilog=ENCODING_HINT)
    p.add_argument("--desktop", type=int, required=True)
    p.add_argument("--view", default=None, choices=list(ALL_VIEWS),
                   help="one view only (default: every view)")
    p.add_argument("--format", default="text", choices=("text", "md"),
                   help="plain text (default) or Markdown")
    p.add_argument("--out", default=None,
                   help="write to this UTF-8 file instead of printing (use it in PowerShell)")
    p.set_defaults(func=cmd_notes)
