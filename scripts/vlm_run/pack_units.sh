#!/usr/bin/env bash
# Print the export units a pack holds, space-separated, in the pack's order:
# manifest.json "units" (the camera views with records, then multi). A unit with no
# record is left out of the pack (export_vlmpack.py "empty_units"), so the job arrays
# read this list instead of assuming all five.
#   bash pack_units.sh <pack>          (or PACK=<pack> bash pack_units.sh)
# Without a manifest (or an old one without "units") it prints all five; run_eval.sh
# skips a unit that is not there.
set -euo pipefail
PACK_DIR="${1:-${PACK:?usage: pack_units.sh <pack>}}"
PYTHON="${PYTHON:-python}"
"$PYTHON" - "$PACK_DIR/manifest.json" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        units = json.load(fh).get("units")
except (OSError, ValueError):
    units = None
print(" ".join(units or ["scan", "oak1", "oak2", "rs", "multi"]))
PY
