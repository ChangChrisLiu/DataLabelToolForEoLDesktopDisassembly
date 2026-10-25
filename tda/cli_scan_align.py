"""``scan-align``: which OAK steps did the scanner photograph?  (task U25)

::

    python -m tda.cli scan-align --desktop 18 [--view oak1|oak2|rs] [--render DIR]

Reads the frame index, the desktop's sheet (its ``dupli`` rows and notes), the
scanner folders on the raw drive and -- opened **read-only** -- the database's
pose-segment ROIs, then ranks every mapping of the scanner folders onto the
OAK captures (:mod:`tda.core.scan_align`) and prints the table, the verdict and
the ``scan.<d>.duplicate_folders`` declaration the best complete mapping needs
in ``configs/index_fixes.yaml``. Writes nothing but the frame cache it is given
(``--cache``, default ``<cache_dir>/scan_align``) and the montages of
``--render``; takes no lock, so it can run while the annotator is open.
"""
from __future__ import annotations

import argparse
import os

from tda import pipeline as P
from tda.cli_common import EXIT_ERROR, EXIT_OK, load_paths

__all__ = ["_add_scan_align", "cmd_scan_align"]


def cmd_scan_align(args: argparse.Namespace) -> int:
    from tda.core import rawroot
    from tda.core import scan_align as A
    from tda.core.index import load_index
    from tda.core.logs import sheet_rows_by_desktop

    paths = load_paths(args.paths)   # also finds the raw drive (rawroot.configure)
    scanner_root = rawroot.resolve_raw(P.require(paths, "scanner_root"))
    if scanner_root is None:
        print("[scan-align] raw data drive not connected; nothing to compare")
        return EXIT_ERROR
    target = args.index or P.index_path(paths)
    try:
        index = load_index(target)
    except OSError as exc:
        print(f"[scan-align] cannot read {target}: {exc}; run build-index first")
        return EXIT_ERROR
    di = index.get(int(args.desktop))
    if di is None:
        print(f"[scan-align] D{int(args.desktop):02d} is not in {target}")
        return EXIT_ERROR
    sheet = sheet_rows_by_desktop(args.logs or P.drive_dir(paths)).get(int(args.desktop))
    db_path = args.db or paths.get("db_path")
    cache = args.cache or P.cache_file(paths, "scan_align")
    try:
        analysis = A.analyse(int(args.desktop), di, scanner_root=scanner_root, sheet=sheet,
                             db_path=db_path, view=args.view, cache_dir=cache,
                             pairs=not args.no_pairs)
    except (OSError, ValueError) as exc:
        print(f"[scan-align] {exc}")
        return EXIT_ERROR
    print(A.describe(analysis, top=args.top))
    if args.render:
        j = analysis.judgement
        phis = {"declared": j.declared.hyp.phi}
        other = j.best if j.best is not j.declared else j.runner_up
        if other is not None:
            phis["best" if other is j.best else "runner-up"] = other.hyp.phi
        if j.complete and j.complete[0].hyp.phi not in phis.values():
            phis["complete"] = j.complete[0].hyp.phi
        stem = os.path.join(args.render, f"D{int(args.desktop):02d}_{args.view}")
        print(f"[scan-align] wrote {A.render_decisive(analysis, stem + '.png', phis)}")
        # what changed between the scanner folders the answers disagree on
        target = (j.answer if j.answer is not j.declared else (j.runner_up or j.answer))
        pairs = sorted(A.disputed_pairs(analysis, target.hyp.phi))
        crops = [(f"scan {analysis.folder_names[i]}", analysis.folder_paths[i],
                  f"{analysis.folder_names[i + 1]}", analysis.folder_paths[i + 1])
                 for i in pairs]
        if crops:
            print(f"[scan-align] wrote "
                  f"{A.render_changes(crops, 'scan', analysis.roi['scan'][0], stem + '_changes.png')}")
    return EXIT_OK


def _add_scan_align(sub) -> None:
    p = sub.add_parser("scan-align",
                       help="rank how the scanner folders map onto the OAK steps "
                            "(read-only)")
    p.add_argument("--desktop", required=True, type=int)
    p.add_argument("--view", default="oak1", choices=("oak1", "oak2", "rs"),
                   help="the reference sequence (default oak1)")
    p.add_argument("--index", default=None, help="index JSON (default <cache_dir>/index.json)")
    p.add_argument("--logs", default=None, help="Drive sheets (default <raw_logs_dir>/drive)")
    p.add_argument("--cache", default=None,
                   help="where the downscaled frames are kept (default <cache_dir>/scan_align)")
    p.add_argument("--render", default=None, help="write the decisive frames here")
    p.add_argument("--top", type=int, default=12, help="rows of the ranked table")
    p.add_argument("--no-pairs", action="store_true",
                   help="leave out the skip-plus-re-shot hypotheses")
    p.set_defaults(func=cmd_scan_align)
