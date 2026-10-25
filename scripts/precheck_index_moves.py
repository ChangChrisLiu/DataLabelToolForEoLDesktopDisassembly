"""Pre-checks before a re-index renumbers steps (task U25, the live sequence of R2.4).

::

    python scripts/precheck_index_moves.py [--db <tda.sqlite>] [--desktops 1,18,22,...]

Read-only (the database is opened ``mode=ro``). For every desktop of the list it
counts what ``load-index`` / ``import-logs --force`` would refuse over, or could
not carry to the new step numbers, and prints one line per check, the counts per
desktop, and a last line ``OK`` or ``NOT OK: ...``:

* **blocking** (``NOT OK``): verified frames and confirmed compiled rows
  (``load-index`` refuses the desktop, and ``import-logs`` then waits for it);
  the Shift+T corrections ``correct_verb`` / ``assert_state`` / ``captive_fix``
  (``import-logs --force`` replays them by ``(step, target)``; after a
  renumbering they report "not matched" and must be redone by hand); relations
  with an evidence step that are not Label Studio's (nothing moves them);
* **noted**: hand-written state events (``load-index`` moves them with the step
  table and says how many), ``commit_keyframe`` ops (history: their payload keeps
  the old step numbers, nothing replays them), hand-written frame columns (the
  photo columns stay with their photo, ``review_status`` follows the step).

Exit code 0 for ``OK``, 1 for ``NOT OK``, 2 when the database cannot be opened.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
#: The desktops U25 renumbers: the 13 with dupli rows, and D41 / D61 whose
#: scanner skipped a step.
DEFAULT_DESKTOPS = (1, 18, 22, 24, 28, 31, 36, 40, 41, 48, 52, 55, 60, 61, 62)

#: (name, blocking, SQL with one ``{q}`` placeholder for the desktop list); every
#: query returns ``(desktop, [kind,] count)`` rows.
CHECKS: tuple[tuple[str, bool, str], ...] = (
    ("verified frames", True,
     "SELECT desktop, view, COUNT(*) FROM frame WHERE review_status='verified' "
     "AND desktop IN ({q}) GROUP BY desktop, view"),
    ("confirmed compiled rows", True,
     "SELECT desktop, COUNT(*) FROM compiled_mask WHERE status='verified' "
     "AND desktop IN ({q}) GROUP BY desktop"),
    ("Shift+T corrections (op_log correct_verb / assert_state / captive_fix)", True,
     "SELECT desktop, kind, COUNT(*) FROM op_log WHERE kind IN "
     "('correct_verb', 'assert_state', 'captive_fix') AND desktop IN ({q}) "
     "GROUP BY desktop, kind"),
    ("relations with an evidence step, not Label Studio's", True,
     "SELECT desktop, COUNT(*) FROM relation WHERE evidence_step IS NOT NULL "
     "AND source<>'labelstudio' AND desktop IN ({q}) GROUP BY desktop"),
    ("hand-written state events (state_event.auto=0; load-index moves them)", False,
     "SELECT desktop, COUNT(*) FROM state_event WHERE auto=0 AND desktop IN ({q}) "
     "GROUP BY desktop"),
    ("op_log commit_keyframe (history; payload keeps the old steps)", False,
     "SELECT desktop, kind, COUNT(*) FROM op_log WHERE kind='commit_keyframe' "
     "AND desktop IN ({q}) GROUP BY desktop, kind"),
    ("hand-written frame columns (photo columns stay with their photo)", False,
     "SELECT desktop, COUNT(*) FROM frame WHERE (hand_or_tool_in_frame IS NOT NULL "
     "OR in_progress IS NOT NULL OR image_quality IS NOT NULL OR bench_annotated IS NOT NULL) "
     "AND desktop IN ({q}) GROUP BY desktop"),
)


def _default_db() -> Optional[str]:
    try:
        import yaml

        with open(REPO / "configs" / "paths.yaml", encoding="utf-8") as fh:
            return (yaml.safe_load(fh) or {}).get("db_path")
    except OSError:
        return None


def run(db: str, desktops: tuple[int, ...]) -> tuple[bool, list[str]]:
    """``(ok, lines)`` for ``db`` (opened read-only) and ``desktops``."""
    uri = "file:" + Path(db).resolve().as_posix() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        q = ",".join("?" * len(desktops))
        lines = [f"pre-checks on {db} for D{', D'.join(str(d) for d in desktops)} "
                 f"(read-only):"]
        blocking: list[str] = []
        for name, blocks, sql in CHECKS:
            rows = con.execute(sql.format(q=q), desktops).fetchall()
            tag = "blocks" if blocks else "note"
            if not rows:
                lines.append(f"  [{tag}] {name}: none")
                continue
            cells = ", ".join(
                f"D{r[0]:02d}" + (f" {r[1]}" if len(r) == 3 else "") + f": {r[-1]}"
                for r in rows)
            lines.append(f"  [{tag}] {name}: {cells}")
            if blocks:
                blocking.append(f"{name} on D{', D'.join(sorted({f'{r[0]:02d}' for r in rows}))}")
    finally:
        con.close()
    lines.append("OK" if not blocking else "NOT OK: " + "; ".join(blocking))
    return not blocking, lines


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=None, help="database (default: paths.yaml db_path)")
    ap.add_argument("--desktops", default=",".join(str(d) for d in DEFAULT_DESKTOPS))
    args = ap.parse_args(argv)
    db = args.db or _default_db()
    if not db or not Path(db).is_file():
        print(f"cannot open the database {db!r}")
        return 2
    desktops = tuple(int(d) for d in args.desktops.split(",") if d.strip())
    ok, lines = run(db, desktops)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
