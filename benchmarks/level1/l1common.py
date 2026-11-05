"""Small helpers shared by the Level-1 benchmark scripts (no third-party imports at module level)."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

TEST_DESKTOPS = [4, 10, 15, 20, 24, 30, 31, 32, 46, 48, 49, 65]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def dump_json(obj, path, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def parse_desktops(text):
    """'1,2,D05,7-9' -> [1, 2, 5, 7, 8, 9]; '' or 'none' -> []."""
    if text is None:
        return None
    text = str(text).strip()
    if text.lower() in ("", "none"):
        return []
    out = []
    for tok in text.replace(" ", ",").split(","):
        tok = tok.strip().upper().lstrip("D")
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b.lstrip("D")) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def dname(d):
    return f"D{int(d):02d}"


def set_seed(seed, deterministic=False):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def image_root(coco, coco_path, override=None):
    """Directory the COCO file_name entries are relative to (the scanner pack folder)."""
    if override:
        return Path(override)
    root = coco.get("info", {}).get("image_root")
    if root:
        p = Path(root)
        if not p.is_absolute():
            p = (Path(coco_path).parent / p)
        return p
    return Path(coco_path).parent


def contiguous_categories(categories):
    """Sorted category ids -> 0..K-1 (YOLO) and back."""
    ids = sorted(c["id"] for c in categories)
    to_idx = {cid: i for i, cid in enumerate(ids)}
    names = {c["id"]: c["name"] for c in categories}
    return ids, to_idx, names


def rle_encode(mask):
    """Boolean HxW numpy mask -> COCO compressed RLE with a str 'counts'."""
    import numpy as np
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


class Logger:
    """Print and append to a log file."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def __call__(self, *msg):
        line = " ".join(str(m) for m in msg)
        stamp = f"[{time.strftime('%Y-%m-%d %H:%M:%S')} +{time.time() - self.t0:7.1f}s] "
        print(stamp + line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(stamp + line + "\n")


def env_report():
    info = {"python": sys.version.split()[0], "platform": sys.platform}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    for mod in ("torchvision", "ultralytics", "pycocotools", "numpy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except ImportError:
            pass
    return info
