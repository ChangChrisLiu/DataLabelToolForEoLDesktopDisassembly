"""Pin caches, temp dirs, threads and the process priority before torch loads.

Import this module **first** in every L5 entry point::

    from experiments.l5_sam import env  # noqa: F401  (side effects)

Caches live under ``D:/DataSet/.cache/l5``; outputs under
``D:/DataSet/experiments_out/l5_sam``; the database is the controller's copy
``D:/DataSet/.cache/tmp/l5/tda_l5.sqlite`` opened ``mode=ro`` -- never the live
``annotations/tda.sqlite``.  Nothing is downloaded (``HF_HUB_OFFLINE=1``).

The annotator may start the app on the same GPU at any time, so every process
runs at BELOW_NORMAL priority with at most 4 CPU threads, checks :data:`STOP`
between epochs / folds, and refuses to start a GPU job when less than
:data:`MIN_FREE_GB` would remain free.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CACHE = Path("D:/DataSet/.cache/l5")
TMP = CACHE / "tmp"
DBTMP = Path("D:/DataSet/.cache/tmp/l5")
DB = str(DBTMP / "tda_l5.sqlite")
DB_URI = f"file:{DB}?mode=ro"
STOP = DBTMP / "STOP"
OUT = Path("D:/DataSet/experiments_out/l5_sam")
WEIGHTS = Path("D:/DataSet/models/weights")
SAM2_CKPT = WEIGHTS / "sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"   # the app's DEFAULT_CONFIG
CACHE_DIR = "D:/DataSet/cache"                    # the app's local frame cache
MAX_THREADS = 4
#: GPU memory that must stay free for the annotator's app (GiB).
MIN_FREE_GB = 10.0

_ENV = {
    "TMP": str(TMP),
    "TEMP": str(TMP),
    "TMPDIR": str(TMP),
    "TORCH_HOME": str(CACHE / "torch"),
    "HF_HOME": str(CACHE / "hf"),
    "HUGGINGFACE_HUB_CACHE": str(CACHE / "hf" / "hub"),
    "XDG_CACHE_HOME": str(CACHE / "xdg"),
    "MPLCONFIGDIR": str(TMP / "mpl"),
    "TRITON_CACHE_DIR": str(CACHE / "triton"),
    "CUDA_CACHE_PATH": "D:/DataSet/.cache/nv_compute",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "SAM2_BUILD_CUDA": "0",
    "SAM2_BUILD_ALLOW_ERRORS": "1",
    "OMP_NUM_THREADS": str(MAX_THREADS),
    "MKL_NUM_THREADS": str(MAX_THREADS),
    "OPENBLAS_NUM_THREADS": str(MAX_THREADS),
    "NUMEXPR_NUM_THREADS": str(MAX_THREADS),
}
for _k, _v in _ENV.items():
    os.environ[_k] = _v
for _p in (TMP, TMP / "mpl", DBTMP, OUT, CACHE / "torch", CACHE / "hf", CACHE / "xdg"):
    _p.mkdir(parents=True, exist_ok=True)


def low_priority() -> None:
    """BELOW_NORMAL for this process."""
    try:
        import psutil

        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception as exc:  # noqa: BLE001 - say so, keep going
        print(f"[l5] could not lower the priority: {exc}", flush=True)


def limit_threads() -> None:
    """Cap torch and OpenCV CPU threads (call after importing torch)."""
    try:
        import torch

        torch.set_num_threads(MAX_THREADS)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2

        cv2.setNumThreads(MAX_THREADS)
    except Exception:  # noqa: BLE001
        pass


def stop_requested() -> bool:
    return STOP.exists()


def gpu_free_gb() -> float:
    """Free memory on GPU 0 as the driver reports it (all processes)."""
    import torch

    free, _total = torch.cuda.mem_get_info(0)
    return free / 1024 ** 3


def connect():
    """The controller's DB copy, read-only; refuses the live database."""
    import sqlite3

    if "annotations" in Path(DB).resolve().parts:
        raise SystemExit("refusing to open the live database")
    con = sqlite3.connect(DB_URI, uri=True)
    con.row_factory = sqlite3.Row
    return con


low_priority()
