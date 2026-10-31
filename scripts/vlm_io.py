"""Shared reading for the W3b VLM scripts (``vlm_references``, ``vlm_to_chat``,
``vlm_score``, ``vlm_sample``).

Everything reads an export of ``python -m tda.cli export-vlm`` and its sidecars
-- ``<name>.images_manifest.jsonl``, ``<name>.oracle.jsonl``,
``<name>.summary.json`` -- and never the database: the evaluation layer has to
run on a frozen release, on a machine that has none.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tda.core.export.vlm_check import World  # noqa: E402


def read_jsonl(path) -> Iterator[dict]:
    """Every non-empty line of a JSONL file, parsed."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows: Iterable[dict]) -> int:
    """Write ``rows`` one per line (``\\n``, UTF-8); returns how many."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
            n += 1
    return n


def sidecar(export_jsonl, suffix: str) -> Path:
    """``vlm_scan.jsonl`` + ``oracle.jsonl`` -> ``vlm_scan.oracle.jsonl``."""
    p = Path(export_jsonl)
    return p.with_name(p.stem + "." + suffix)


def load_oracle(export_jsonl) -> tuple[dict[int, World], dict[str, dict]]:
    """``(desktop -> World, image id -> frame table)`` from the oracle sidecar."""
    worlds: dict[int, World] = {}
    frames: dict[str, dict] = {}
    path = sidecar(export_jsonl, "oracle.jsonl")
    if not path.exists():
        return worlds, frames
    for row in read_jsonl(path):
        if row.get("type") == "desktop":
            worlds[int(row["desktop"])] = World.from_oracle(row)
        elif row.get("type") == "frame":
            frames[row["image"]] = row
    return worlds, frames


def load_manifest(export_jsonl) -> dict[str, dict]:
    """``image id -> {path, desktop, view, step, roi?}``."""
    path = sidecar(export_jsonl, "images_manifest.jsonl")
    if not path.exists():
        return {}
    return {row["image"]: row for row in read_jsonl(path) if row.get("type") != "header"}


def load_summary(export_jsonl) -> dict:
    """The export's summary (counts, skips, the shortcut table), or ``{}``."""
    path = sidecar(export_jsonl, "summary.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def desktop_of(record: dict) -> int:
    return int(record["label"]["desktop"])


def is_positive(record: dict) -> Optional[bool]:
    """``True`` / ``False`` for a record with a negative half, ``None`` otherwise.

    Positive = no ``label.negative``; tasks that carry no negatives at all
    (V1 V2 V3 V7 V8 V10) answer ``None``.
    """
    task = record["prompt"]["task"]
    if task in ("V4", "V12", "V14", "V15", "V16"):
        return not record["label"].get("negative")
    return None
