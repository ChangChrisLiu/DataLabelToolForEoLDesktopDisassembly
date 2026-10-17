"""Pin every cache, temp dir, thread pool and the process priority before torch loads.

Import this module **first** in every L4 entry point.  It imports L3's ``env``
(which imports L2's), so their modules can be reused unchanged, and then
overrides every variable with L4's own locations: caches under
``D:/DataSet/.cache/l4``, outputs under ``D:/DataSet/experiments_out/l4_detector``,
the database copy ``D:/DataSet/.cache/tmp/l4/tda_l4.sqlite`` (opened ``mode=ro``,
never the live ``annotations/tda.sqlite``).  Nothing lands on C:, nothing is
downloaded.

The annotator works in the app on the same GPU while this runs, so every L4
process runs at BELOW_NORMAL priority with 4 CPU threads (:func:`low_priority`,
called on import), and training stops cleanly after the current epoch when the
controller creates :data:`STOP`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.l3_detector import env as l3env  # noqa: E402  (L2's, then L3's values)

CACHE = Path("D:/DataSet/.cache/l4")
TMP = CACHE / "tmp"
DBTMP = Path("D:/DataSet/.cache/tmp/l4")
DB = str(DBTMP / "tda_l4.sqlite")       # the controller's copy, opened mode=ro
DB_URI = f"file:{DB}?mode=ro"
STOP = DBTMP / "STOP"
OUT = Path("D:/DataSet/experiments_out/l4_detector")
WEIGHTS = Path("D:/DataSet/models/weights")
L2OUT = l3env.L2OUT
L3OUT = l3env.OUT
PRODUCTION = WEIGHTS / "tda_det_rfdetr_small_all.pt"
L3_HOLD13 = L3OUT / "rfdetr_small" / "runs" / "hold13" / "last_ema.pth"
CACHE_DIR = "D:/DataSet/cache"         # the app's local frame cache (configs/paths.yaml)
MAX_THREADS = 4
MAX_WORKERS = 4

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
    "YOLO_CONFIG_DIR": str(CACHE / "ultralytics_cfg"),
    "RF_HOME": str(WEIGHTS),
    "ROBOFLOW_HOME": str(WEIGHTS),
    "MPLCONFIGDIR": str(TMP / "mpl"),
    "CUDA_CACHE_PATH": "D:/DataSet/.cache/nv_compute",
    "WANDB_MODE": "disabled",
    "WANDB_DIR": str(TMP / "wandb"),
    "YOLO_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
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
    """BELOW_NORMAL for this process (its loader workers inherit it on Windows)."""
    try:
        import psutil

        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
    except Exception as exc:  # noqa: BLE001 - say so, keep going
        print(f"[l4] could not lower the priority: {exc}", flush=True)


def limit_threads() -> None:
    """Cap torch and OpenCV CPU threads (call after importing torch)."""
    l3env.limit_threads()


def stop_requested() -> bool:
    return STOP.exists()


def bind_l2(out: Path) -> Path:
    """Point L2's reused modules at L4's database copy and ``out`` (read-only DB)."""
    from experiments.l2_detector import data as D
    from experiments.l2_detector import env as l2env

    D.DB = DB
    D.DB_URI = DB_URI
    D.GEOM = DBTMP / "geom" / "ls_boxes_all.json"
    if not D.GEOM.exists():
        D.build_boxes(D.GEOM)
    l2env.TMP = DBTMP
    out.mkdir(parents=True, exist_ok=True)
    l2env.OUT = out
    return out


low_priority()
