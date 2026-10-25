"""Resolution and verification of the declared anomaly fixes.

The rules live in ``configs/index_fixes.yaml`` (spec 2.2).  OAK ``Camera_2`` is
normally paired to ``Camera_1`` by identical filename timestamp, so a declared
fix serves two purposes: it is the *fallback* when no timestamp partner exists,
and it is an *assertion* on the pairing the timestamps produced.  Every declared
fix therefore emits a line into ``DesktopIndex.issues`` - a confirmation naming
the resolved logical step, or a ``MISMATCH``.  Nothing is applied silently.

Folder-name vocabulary: ``from_dir`` / ``to_cam1_dir`` / the ``cam2_swap`` pairs
are Camera_*folder* names (``"004"``), never logical step numbers.  A folder maps
to a logical step through ``step_of_dir``, which already accounts for
``cam1_reorder_by_ts``.
"""
from __future__ import annotations

from typing import Optional

from tda.core.model import FrameKey


def cam2_fix_step(
    oak_fix: dict, src_dir: str, step_of_dir: dict[str, int]
) -> Optional[int]:
    """Logical step a Camera_2 folder is declared to hold, or ``None``.

    Consulted only when timestamp pairing found no Camera_1 partner.
    """
    shift = oak_fix.get("cam2_shift") or {}
    if shift.get("from_dir") == src_dir:
        return step_of_dir.get(str(shift["to_cam1_dir"]))
    for pair in oak_fix.get("cam2_swap") or []:
        if src_dir == pair[0]:
            return step_of_dir.get(pair[1])
        if src_dir == pair[1]:
            return step_of_dir.get(pair[0])
    if oak_fix.get("cam2_extra_from") == src_dir and "as_step" in oak_fix:
        return step_of_dir.get(f"{int(oak_fix['as_step']):03d}")
    return None


def verify_cam2_fixes(
    desktop: int,
    oak_fix: dict,
    step_of_dir: dict[str, int],
    assigned: dict[str, list[int]],
) -> list[str]:
    """Check the resulting pairing against every declared Camera_2 fix."""
    issues: list[str] = []

    def check(label: str, src_dir: str, want_dir: str) -> None:
        want = step_of_dir.get(want_dir)
        got = assigned.get(src_dir, [])
        head = f"fix:oak[{desktop}] {label}"
        if want is not None and want in got:
            issues.append(
                f"{head}: cam2 {src_dir} -> cam1 dir {want_dir} = logical step {want}"
            )
        else:
            issues.append(
                f"{head} MISMATCH: cam2 {src_dir} expected -> cam1 dir {want_dir} = "
                f"logical step {want}, got steps {got}"
            )

    shift = oak_fix.get("cam2_shift") or {}
    if shift:
        check("cam2_shift", shift["from_dir"], str(shift["to_cam1_dir"]))
    for pair in oak_fix.get("cam2_swap") or []:
        check(f"cam2_swap {pair[0]}<->{pair[1]}", pair[0], pair[1])
        check(f"cam2_swap {pair[0]}<->{pair[1]}", pair[1], pair[0])
    extra = oak_fix.get("cam2_extra_from")
    if extra and "as_step" in oak_fix:
        check("cam2_extra_from", extra, f"{int(oak_fix['as_step']):03d}")
    return issues


def declared_missing(fixes: dict, view: str, desktop: int) -> set[int]:
    """The steps ``missing_steps`` / ``missing_steps_range`` declare for one view.

    A scanner's ``skipped_steps`` (U25 round 2) are missing too: the scanner
    never photographed them.
    """
    entry = (fixes.get(view) or {}).get(desktop)
    if not isinstance(entry, dict):
        return set()
    steps = {int(s) for s in entry.get("missing_steps") or []}
    rng = entry.get("missing_steps_range")
    if rng:
        steps |= set(range(int(rng[0]), int(rng[1]) + 1))
    if view == "scan":
        steps |= {int(s) for s in entry.get("skipped_steps") or []}
    return steps


def verify_declared_missing(
    desktop: int, fixes: dict, n_steps: int, missing: list[FrameKey]
) -> list[str]:
    """Check the ``scan``/``rs`` gaps declared in the yaml against the real gaps.

    ``missing_steps`` / ``missing_steps_range`` are **logical steps** -- since
    v1.25 the collapsed ones (spec 2.2): the steps the sheet's rows make once its
    ``dupli`` rows are dropped. A declared-missing frame that turns out to exist
    is a ``MISMATCH`` - the yaml is then stale.
    """
    issues: list[str] = []
    for view in ("scan", "rs"):
        declared_all = declared_missing(fixes, view, desktop)
        if not declared_all:
            continue
        declared = {s for s in declared_all if 1 <= s <= n_steps}
        beyond = sorted(declared_all - declared)
        actual = {k.step for k in missing if k.view == view}
        present = sorted(declared - actual)
        if present:
            issues.append(
                f"fix:{view}[{desktop}] MISMATCH: declared-missing logical steps "
                f"present in index: {present}"
            )
        else:
            note = (
                f" ({len(beyond)} declared step(s) are beyond the {n_steps} steps: {beyond})"
                if beyond
                else ""
            )
            issues.append(
                f"fix:{view}[{desktop}] declared missing steps {sorted(declared)} "
                f"confirmed absent" + note
            )
    return issues


def verify_scan_rule(desktop: int, scan_fix: dict, dupli_rows: list[int], n_steps: int,
                     frames: dict, missing: list[FrameKey],
                     leftover: Optional[list[str]] = None) -> list[str]:
    """Assert the scanner rule of a desktop with ``dupli`` rows or a declaration (U25).

    Every collapsed step must have a scanner frame except the declared missing
    ones, and every folder must have found a step: a folder left over, or a
    step without a frame that nothing declares, is a ``MISMATCH`` -- the
    declaration (or its absence) does not fit the folders on the drive, which is
    what ``scan-align`` is for. Silent for every other desktop.
    """
    scan_fix = scan_fix or {}
    declared = "duplicate_folders" in scan_fix or "skipped_steps" in scan_fix
    if not declared and not dupli_rows:
        return []
    dup = sorted(int(n) for n in (scan_fix.get("duplicate_folders") or []))
    scan = {k.step: ff for k, ff in frames.items() if k.view == "scan"}
    folders = len(scan) + sum(len((ff.aux or {}).get("extra_shots") or [])
                              for ff in scan.values())
    head = f"fix:scan[{desktop}] duplicate_folders {dup}"
    replacing = sorted(int(n) for n in (scan_fix.get("replacing_folders") or []))
    if replacing:
        head += f" (replacing {replacing})"
    skipped = sorted(int(n) for n in (scan_fix.get("skipped_steps") or []))
    if skipped:
        head += f", skipped steps {skipped}"
    if not declared:
        head = (f"scan[{desktop}] (no scan.{desktop}.duplicate_folders declared -- run "
                f"scan-align; dupli rows {dupli_rows} taken as all skipped by the scanner)")
    absent = sorted(k.step for k in missing if k.view == "scan")
    expected = sorted(declared_missing({"scan": {desktop: scan_fix}}, "scan", desktop))
    undeclared = [s for s in absent if s not in expected]
    extras = folders - len(scan)
    shown = (f"{folders} folders -> {len(scan)} of {n_steps} steps"
             + (f" + {extras} extra shot(s)" if extras else ""))
    problems = []
    if leftover:
        problems.append(f"folders {leftover} have no step")
    if undeclared:
        problems.append(f"steps {undeclared} have no scanner frame and are not declared "
                        f"missing")
    if problems:
        return [f"{head} MISMATCH: {shown}; " + "; ".join(problems)]
    return [f"{head}: {shown}" + (f", declared missing {expected}" if expected else
                                  ", every step photographed")]
