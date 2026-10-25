"""Pin caches, temp dirs, threads and the process priority; the GPU rule.

Import this module **first** in every L6 entry point::

    from experiments.l6_models import env  # noqa: F401  (side effects first)

It imports L4's ``env`` (which imports L3's and L2's) and L5's ``env``, so their
modules can be reused unchanged, then overrides every variable with L6's own
locations: everything L6 writes lives under ``D:/DataSet/.cache/tmp/l6``
(:data:`ROOT`); fold weights go to ``D:/DataSet/models/weights`` as
``*_l6_*``.  Nothing lands on C:, nothing is downloaded.

**The database.**  :data:`LIVE_COPY` is the one backup-API copy of
``annotations/tda.sqlite`` taken at the start (2026-10-24 21:33, the live file
opened ``mode=ro``); every L6 read goes through it, opened ``mode=ro``
(:func:`connect`).  ``export-coco`` writes (it refreshes the truth table and
takes the single-user lock), so it runs on :data:`EXPORT_DB`, a second copy
made from :data:`LIVE_COPY`.

**The GPU rule.**  The annotator works in the app on this machine (``python -m
tda.cli app``, SAM on the GPU).  A GPU-heavy phase -- a training chunk, an
inference or evaluation batch -- starts only while no such process exists
(:func:`app_running`, the brief's PowerShell test ``Name like 'python%'`` and
``CommandLine -like '*tda.cli*app*'``, case-insensitive like ``-like``); when
one does, :func:`wait_for_gpu` polls every 5 minutes and logs every wait to
:data:`WAITS`.  Training checks between epochs (detector) or every
:data:`SAM_CHUNK` steps (SAM) and resumes from its checkpoint.  The user's
process is never touched.
"""
from __future__ import annotations

import fnmatch
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# the reused experiments' env modules first (each sets its own variables) ...
from experiments.l4_detector import env as l4env  # noqa: E402
from experiments.l5_sam import env as l5env  # noqa: E402

# ... then L6's
ROOT = Path("D:/DataSet/.cache/tmp/l6")
CACHE = ROOT / "cache"
TMP = CACHE / "tmp"
LIVE_COPY = ROOT / "live_copy.sqlite"
EXPORT_DB = ROOT / "export_work.sqlite"
DB = str(LIVE_COPY)
DB_URI = f"file:{DB}?mode=ro"
OUT = ROOT
STOP = ROOT / "STOP"
WAITS = ROOT / "logs" / "gpu_waits.jsonl"
WEIGHTS = Path("D:/DataSet/models/weights")
CACHE_DIR = "D:/DataSet/cache"          # the app's local frame cache (read only)
APP_ENV_PY = Path("D:/Anaconda/envs/tda/python.exe")
TRAIN_ENV_PY = Path("D:/Anaconda/envs/tda_l3/python.exe")
MAX_THREADS = 4
MAX_WORKERS = 4
POLL_S = 300
SAM_CHUNK = 250
APP_PATTERN = "*tda.cli*app*"

_ENV = {
    "TMP": str(TMP),
    "TEMP": str(TMP),
    "TMPDIR": str(TMP),
    "PIP_CACHE_DIR": str(CACHE / "pip"),
    "TORCH_HOME": str(CACHE / "torch"),
    "HF_HOME": str(CACHE / "hf"),
    "HUGGINGFACE_HUB_CACHE": str(CACHE / "hf" / "hub"),
    "TRANSFORMERS_CACHE": str(CACHE / "hf" / "transformers"),
    "XDG_CACHE_HOME": str(CACHE / "xdg"),
    "YOLO_CONFIG_DIR": "D:/DataSet/.cache/ultralytics_cfg",
    "RF_HOME": str(WEIGHTS),
    "ROBOFLOW_HOME": str(WEIGHTS),
    "MPLCONFIGDIR": str(TMP / "mpl"),
    "TRITON_CACHE_DIR": str(CACHE / "triton"),
    "CUDA_CACHE_PATH": "D:/DataSet/.cache/nv_compute",
    "WANDB_MODE": "disabled",
    "WANDB_DIR": str(TMP / "wandb"),
    "YOLO_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "SAM2_BUILD_CUDA": "0",
    "SAM2_BUILD_ALLOW_ERRORS": "1",
    # rfdetr's rich tables crash a cp1252 stdout redirected to a log file
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "OMP_NUM_THREADS": str(MAX_THREADS),
    "MKL_NUM_THREADS": str(MAX_THREADS),
    "OPENBLAS_NUM_THREADS": str(MAX_THREADS),
    "NUMEXPR_NUM_THREADS": str(MAX_THREADS),
}
for _k, _v in _ENV.items():
    os.environ[_k] = _v
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - not a text stream; leave it
        pass
for _p in (ROOT, TMP, TMP / "mpl", CACHE / "torch", CACHE / "hf", CACHE / "xdg",
           ROOT / "logs"):
    _p.mkdir(parents=True, exist_ok=True)

# the reused modules read these at call time
l4env.DB = DB
l4env.DB_URI = DB_URI
l4env.CACHE_DIR = CACHE_DIR
l5env.DB = DB
l5env.DB_URI = DB_URI
l5env.CACHE_DIR = CACHE_DIR


def connect():
    """The L6 copy of the database, read-only; refuses the live file."""
    import sqlite3

    if "annotations" in Path(DB).resolve().parts:
        raise SystemExit("refusing to open the live database")
    con = sqlite3.connect(DB_URI, uri=True)
    con.row_factory = sqlite3.Row
    return con


def low_priority() -> None:
    try:
        import psutil

        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception as exc:  # noqa: BLE001 - say so, keep going
        print(f"[l6] could not lower the priority: {exc}", flush=True)


def limit_threads() -> None:
    l5env.limit_threads()


def stop_requested() -> bool:
    return STOP.exists()


# --------------------------------------------------------------------------- #
# the GPU rule
# --------------------------------------------------------------------------- #
def app_processes() -> list[dict]:
    """The annotator's app processes: python*, command line like ``*tda.cli*app*``."""
    import psutil

    me = os.getpid()
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            if not name.startswith("python") or p.info["pid"] == me:
                continue
            cmd = " ".join(p.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if fnmatch.fnmatchcase(cmd.lower(), APP_PATTERN):
            found.append({"pid": p.info["pid"], "cmd": cmd[:200]})
    return found


def app_running() -> bool:
    return bool(app_processes())


def _log_wait(rec: dict) -> None:
    WAITS.parent.mkdir(parents=True, exist_ok=True)
    with WAITS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def wait_for_gpu(what: str, poll_s: int = POLL_S) -> float:
    """Block while the app runs (poll every ``poll_s``); the seconds waited."""
    procs = app_processes()
    if not procs:
        return 0.0
    t0 = time.time()
    _log_wait({"event": "wait_start", "what": what, "t": time.strftime("%Y-%m-%d %H:%M:%S"),
               "app": procs})
    print(f"[l6] the annotator's app is open ({procs[0]['pid']}): {what} waits, "
          f"polling every {poll_s} s", flush=True)
    while app_running():
        if stop_requested():
            break
        time.sleep(poll_s)
    waited = time.time() - t0
    _log_wait({"event": "wait_end", "what": what, "t": time.strftime("%Y-%m-%d %H:%M:%S"),
               "waited_s": round(waited)})
    print(f"[l6] app closed after {waited / 60:.0f} min: {what} resumes", flush=True)
    return waited


low_priority()
