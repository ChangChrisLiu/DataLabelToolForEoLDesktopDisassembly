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
# PYTHON unset: python, else python3 (a login node may have only python3)
if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON=python
  elif command -v python3 >/dev/null 2>&1; then PYTHON=python3
  else echo "pack_units.sh: neither python nor python3 on PATH; set PYTHON" >&2; exit 2
  fi
fi
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
