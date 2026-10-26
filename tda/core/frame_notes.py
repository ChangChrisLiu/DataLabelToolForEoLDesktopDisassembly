"""「这一帧的说明」: one undoable write, and how the notes read back (task U35).

The annotator types a note about the frame on screen in Review (what is odd
on it, what they decided, what the picture shows); the controller reads them
later with ``python -m tda.cli notes --desktop N``.  The rows are
:mod:`tda.core.db_notes`; this module is the one way they change and the text
the command line prints.

**The op.**  Every change is one ``set_frame_note`` op whose payload and
inverse are two *states* of the same row -- ``{"exists", "text", "author",
"updated_at"}`` -- the way every other op of the window is written
(:mod:`tda.ui.session_ops`): undoing puts the old row back byte for byte (or
removes the row a first note made), redoing puts the new one back.
:func:`set_note` writes it and its ``op_log`` row in one transaction;
:func:`apply_note` is both directions of the undo stack.

The text is kept as typed, line breaks included; only the whitespace around
it goes, and a text that is nothing but whitespace is "no note" (the row is
removed).  Saving the text a frame already has writes nothing and logs
nothing.  The note is no input of the compiler: nothing is recompiled, signed
or queued.
"""
from __future__ import annotations

from typing import Iterable, Optional

from tda.core import dbrows as R
from tda.core.model import FrameKey, is_component_view

__all__ = ["NOTE_PREFIX", "OP_KIND", "apply_note", "clean_text", "first_line", "note_state",
           "notes_by_step", "notes_markdown", "notes_text", "set_note", "when_text",
           "write_note_state"]

#: The op one change logs (``tda.ui.commands.KINDS`` lists it too).
OP_KIND = "set_frame_note"
#: What the task card and the timeline's tooltip put in front of a note.
NOTE_PREFIX = "说明："


def clean_text(text: Optional[str]) -> str:
    """The note as stored: line breaks normalised, the whitespace around it gone.

    ``""`` means no note.
    """
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace(" ", "\n")
    return value.strip()


def first_line(text: str, limit: int = 0) -> str:
    """The first non-empty line of a note (``…`` when more follows or it is cut)."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    head = lines[0]
    cut = limit and len(head) > limit
    if cut:
        head = head[:limit].rstrip()
    return head + ("…" if cut or len(lines) > 1 else "")


def note_state(note: Optional[dict]) -> dict:
    """One row (or none) as an op's state."""
    if note is None:
        return {"exists": False, "text": None, "author": None, "updated_at": None}
    return {"exists": True, "text": note["text"], "author": note.get("author"),
            "updated_at": note.get("updated_at")}


def write_note_state(db, key: FrameKey, state: dict) -> None:
    """Make the row of ``key`` look like ``state``; ``exists: False`` removes it."""
    if state.get("exists"):
        db.put_frame_note(key, state["text"], state.get("author"), state.get("updated_at"))
    else:
        db.delete_frame_note(key)


def _key(payload: dict) -> FrameKey:
    return FrameKey(int(payload["desktop"]), int(payload["step"]), str(payload["view"]))


def set_note(db, key: FrameKey, text: Optional[str], author: str) -> Optional[tuple[dict, dict]]:
    """Write the note of ``key`` (``""`` clears it); ``(payload, inverse)`` of the op.

    ``None`` -- nothing written, nothing logged -- when the frame already says
    exactly that.  One transaction: the row and its ``op_log`` row.
    """
    value = clean_text(text)
    before = db.frame_note(key)
    if (before["text"] if before else "") == value:
        return None
    common = {"desktop": key.desktop, "view": key.view, "step": key.step}
    after = ({"exists": True, "text": value, "author": str(author), "updated_at": R.now_iso()}
             if value else note_state(None))
    payload = common | after
    inverse = common | note_state(before)
    with db.transaction():
        write_note_state(db, key, after)
        db.log_op(key.desktop, key.view, OP_KIND, payload, inverse, str(author))
    return payload, inverse


def apply_note(db, payload: dict) -> FrameKey:
    """Both directions of a ``set_frame_note`` op; returns the frame it was about."""
    key = _key(payload)
    with db.transaction():
        write_note_state(db, key, payload)
    return key


# --------------------------------------------------------------------------- #
# reading them back (cli notes, the Steps table)
# --------------------------------------------------------------------------- #
def notes_by_step(db, desktop: int) -> dict[int, list[dict]]:
    """``{step: notes}`` of one desktop's sequence views (the components gallery's
    shots are not steps), each step's notes by view."""
    out: dict[int, list[dict]] = {}
    for note in db.frame_notes(desktop):
        if not is_component_view(note["view"]):
            out.setdefault(int(note["step"]), []).append(note)
    return out


def when_text(stamp: Optional[str]) -> str:
    """``2026-10-26T06:03:00+00:00`` -> ``2026-10-26 06:03 UTC`` (as stored otherwise)."""
    value = str(stamp or "")
    if len(value) >= 16 and value[10] == "T":
        zone = " UTC" if value.endswith(("+00:00", "Z")) else ""
        return f"{value[:10]} {value[11:16]}{zone}"
    return value or "-"


def _grouped(notes: Iterable[dict]) -> dict[tuple[int, str], list[dict]]:
    out: dict[tuple[int, str], list[dict]] = {}
    for note in notes:
        out.setdefault((int(note["desktop"]), str(note["view"])), []).append(note)
    return out


def notes_text(desktop: int, notes: list[dict]) -> str:
    """Plain text: a heading per view, then step, author, time and the text indented."""
    if not notes:
        return f"D{desktop:02d}: 没有说明 / no frame notes"
    lines = [f"D{desktop:02d}: {len(notes)} 条说明 / frame note(s)"]
    for (_d, view), mine in _grouped(notes).items():
        lines += ["", f"[{view}] {len(mine)}"]
        for note in mine:
            lines.append(f"step {note['step']}  {note.get('author') or '-'}  "
                         f"{when_text(note.get('updated_at'))}")
            lines += [f"    {line}" for line in str(note["text"]).splitlines()]
    return "\n".join(lines)


def notes_markdown(desktop: int, notes: list[dict]) -> str:
    """Markdown: a section per view, a heading per frame, the text as a quote."""
    lines = [f"# D{desktop:02d} 这一帧的说明 / frame notes ({len(notes)})", ""]
    if not notes:
        return "\n".join(lines + ["没有说明 / no frame notes", ""])
    for (_d, view), mine in _grouped(notes).items():
        lines += [f"## {view}", ""]
        for note in mine:
            lines += [f"### 第 {note['step']} 帧 / step {note['step']} · "
                      f"{note.get('author') or '-'} · {when_text(note.get('updated_at'))}", ""]
            lines += [f"> {line}" if line.strip() else ">"
                      for line in str(note["text"]).splitlines()]
            lines.append("")
    return "\n".join(lines)
