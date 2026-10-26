"""``notes``: what the annotator wrote about the frames of one machine (task U35).

::

    python -m tda.cli notes --desktop N [--view V] [--format md] [--out FILE]

Review's 「这一帧的说明」 box keeps one note per frame (``frame_note``); this
prints them -- step, author, time and the text -- per view, so whoever checks
the data reads what the annotator found odd, decided or saw.  ``--format md``
gives Markdown (a section per view, a heading per frame, the text quoted);
``--out`` writes the same text to a UTF-8 file instead of printing it.

Read-only: no lock, nothing written.  The text is printed as UTF-8 whatever
the console's code page (a note is Chinese more often than not, and the
default Windows pipe is cp1252).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tda.cli_common import EXIT_ERROR, EXIT_OK, session
from tda.core.frame_notes import notes_markdown, notes_text
from tda.core.model import ALL_VIEWS

__all__ = ["_add_notes", "cmd_notes"]


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):   # not a real stream, or already set
        pass


def cmd_notes(args: argparse.Namespace) -> int:
    """Print (or write) the notes of one desktop, optionally one view."""
    with session(args) as (_paths, db):
        desktop = int(args.desktop)
        if db.get_desktop(desktop) is None:
            print(f"[notes] desktop {desktop} is not in the database")
            return EXIT_ERROR
        notes = db.frame_notes(desktop, args.view)
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
                                     "这一帧的说明) of one desktop")
    p.add_argument("--desktop", type=int, required=True)
    p.add_argument("--view", default=None, choices=list(ALL_VIEWS),
                   help="one view only (default: every view)")
    p.add_argument("--format", default="text", choices=("text", "md"),
                   help="plain text (default) or Markdown")
    p.add_argument("--out", default=None, help="write to this UTF-8 file instead of printing")
    p.set_defaults(func=cmd_notes)
