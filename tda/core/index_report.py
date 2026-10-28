"""Markdown summary of a built frame index (see :mod:`tda.core.index`)."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from tda.core.model import VIEWS, is_component_view

_HEADER = (
    "| desktop | n_steps | rows (dupli) | frames | extra shots | missing scan | "
    "missing oak1 | missing oak2 | missing rs | gallery shots | issues |"
)


def _gallery(di: Any) -> int:
    """How many components-gallery shots (task U10) the desktop has, all cameras."""
    return sum(1 for key in di.frames if is_component_view(key.view))


def _extra_shots(di: Any) -> int:
    """Extra shots (task U25): ``dupli`` captures and scanner duplicate folders."""
    return sum(len((ff.aux or {}).get("extra_shots") or []) for ff in di.frames.values())


def _unphotographed(di: Any) -> list[int]:
    """The steps no camera photographed (``unphotographed.<d>.steps``, task U40)."""
    return sorted(int(s) for s in getattr(di, "unphotographed", None) or [])


def _rows(di: Any) -> str:
    """``50 (30, 43)``: the Camera_1 captures and the dupli rows collapsed out;
    ``34 +3 unphotographed`` when steps no camera photographed are declared (U40)."""
    dupli = list(getattr(di, "dupli_rows", None) or [])
    rows = int(getattr(di, "n_rows", 0) or 0) or di.n_steps
    text = f"{rows} ({', '.join(str(r) for r in dupli)})" if dupli else str(rows)
    unph = _unphotographed(di)
    return text + (f" +{len(unph)} unphotographed" if unph else "")


def compact(steps: list[int]) -> str:
    """``[1, 2, 3, 7]`` -> ``"1-3, 7 (n=4)"``."""
    out: list[str] = []
    start = prev = None
    for s in steps:
        if start is None:
            start = prev = s
        elif s == prev + 1:
            prev = s
        else:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = s
    if start is not None:
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return f"{', '.join(out)} (n={len(steps)})"


def write_report(idx: dict[int, Any], path: str) -> None:
    """Write a per-desktop markdown report: step counts, missing frames, issues."""
    n_steps = sum(di.n_steps for di in idx.values())
    n_gallery = sum(_gallery(di) for di in idx.values())
    n_frames = sum(len(di.frames) for di in idx.values()) - n_gallery
    n_extra = sum(_extra_shots(di) for di in idx.values())
    n_dupli = sum(len(getattr(di, "dupli_rows", None) or []) for di in idx.values())
    unph = {d: _unphotographed(di) for d, di in sorted(idx.items()) if _unphotographed(di)}
    lines = [
        "# Frame index report",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- desktops: {len(idx)}",
        f"- logical steps: {n_steps} (the sheet's dupli rows collapsed: {n_dupli})",
        f"- frames: {n_frames}",
        f"- extra shots (dupli captures, scanner duplicate folders): {n_extra}",
        f"- components gallery shots: {n_gallery}",
    ]
    if unph:
        lines.append(
            f"- steps no camera photographed (`unphotographed`): "
            f"{sum(len(s) for s in unph.values())} on "
            + ", ".join(f"D{d} ({compact(s)})" for d, s in unph.items())
        )
    lines += [
        "",
        "## Per desktop",
        "",
        _HEADER,
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    totals = {v: 0 for v in VIEWS}
    for d, di in sorted(idx.items()):
        per = {v: sum(1 for k in di.missing if k.view == v) for v in VIEWS}
        for v in VIEWS:
            totals[v] += per[v]
        lines.append(
            f"| {d} | {di.n_steps} | {_rows(di)} | {len(di.frames) - _gallery(di)} | "
            f"{_extra_shots(di)} | {per['scan']} | "
            f"{per['oak1']} | {per['oak2']} | {per['rs']} | {_gallery(di)} | "
            f"{len(di.issues)} |"
        )
    lines.append(
        f"| **total** | {n_steps} | | {n_frames} | {n_extra} | {totals['scan']} | "
        f"{totals['oak1']} | {totals['oak2']} | {totals['rs']} | {n_gallery} | "
        f"{sum(len(di.issues) for di in idx.values())} |"
    )

    lines += ["", "## Missing frames", ""]
    for d, di in sorted(idx.items()):
        if not di.missing:
            continue
        lines.append(f"- **Desktop {d}**")
        declared = set(unph.get(d) or [])
        for view in VIEWS:
            steps = sorted(k.step for k in di.missing if k.view == view)
            if steps:
                own = sorted(set(steps) & declared)
                note = (f" -- of which unphotographed (no view): {compact(own)}"
                        if own else "")
                lines.append(f"  - `{view}`: {compact(steps)}{note}")

    if unph:
        lines += ["", "## Steps no camera photographed", "",
                  "Declared in `configs/index_fixes.yaml` (`unphotographed.<d>.steps`, "
                  "task U40): the operator did them but no view has a capture; every "
                  "view's captures sit on the other steps.", ""]
        for d, steps in unph.items():
            lines.append(f"- **Desktop {d}**: steps {compact(steps)}")

    lines += ["", "## Issues and applied fixes", ""]
    for d, di in sorted(idx.items()):
        if not di.issues:
            continue
        lines.append(f"- **Desktop {d}**")
        lines += [f"  - {msg}" for msg in di.issues]
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
