"""Export L5's fine-tuned decoder head as a delta on stock SAM 2.1 (task U19).

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l5_sam.export_head [--dry-run]

L5 trained the prompt encoder and the mask decoder and left the image encoder
frozen, so ``sam2.1_hiera_large_tda_f5_v2.pt`` (898 MB, a full state dict) and
the stock ``sam2.1_hiera_large.pt`` differ only there.  This compares the two
tensor by tensor -- byte for byte -- and refuses unless

* both hold the same tensor names with the same shapes and dtypes,
* every tensor that differs belongs to ``sam_prompt_encoder`` or
  ``sam_mask_decoder`` (:data:`tda.models.sam_heads.HEAD_MODULES`),
* the image encoder is byte-identical, and
* the number of differing tensors is ``--expect`` (141, REPORT.md section 7);

then writes the differing tensors, and the stock checkpoint's fingerprint,
to ``sam2.1_tda_head_f5_v2.pt`` (the format :func:`tda.models.sam_heads.read_head`
reads) with a manifest ``.json`` beside it: both source files and their
SHA-256, the tensor names, and the training provenance and blind D5 numbers
from ``experiments/l5_sam/REPORT.md``.  The weights directory is not in git.
"""
from __future__ import annotations

from experiments.l5_sam import env  # noqa: F401  (side effects first)

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

from tda.models.sam_heads import (
    FINGERPRINT_KEYS,
    HEAD_FORMAT,
    HEAD_MODULES,
    fingerprint,
    read_head,
)

WEIGHTS = env.WEIGHTS
SOURCE = WEIGHTS / "sam2.1_hiera_large_tda_f5_v2.pt"
STOCK = WEIGHTS / "sam2.1_hiera_large.pt"
OUT = WEIGHTS / "sam2.1_tda_head_f5_v2.pt"
EXPECT = 141
TRAIN_MANIFEST = env.OUT / "train" / "f5_v2" / "manifest.json"
GATE = env.OUT / "tables" / "f5_v2_gate.json"

#: REPORT.md section 4, fold f5 recipe v2 (held-out D5, blind): the numbers the
#: controller ruled on.  Prompt-averaged (box, point, point_box) IoU / BF of
#: the shown candidate, stock -> tuned.
D5_BLIND = {
    "masks": 1262,
    "instances": 51,
    "small": {"stock": [0.640, 0.746], "tuned": [0.727, 0.866]},
    "screw": {"stock": [0.720, 0.816], "tuned": [0.800, 0.929]},
    "connector": {"stock": [0.554, 0.581], "tuned": [0.718, 0.771]},
    "latch": {"stock": [0.645, 0.676], "tuned": [0.716, 0.764]},
    "large": {"stock": [0.526, 0.193], "tuned": [0.570, 0.166]},
    "loose_box_iou": {"small_macro": [0.639, 0.750], "large": [0.435, 0.529]},
    "refine_small_macro_iou": [0.766, 0.828],
    "gate": {
        "G1_macro_delta_iou": [0.101, 0.068, 0.138],
        "G1_macro_delta_bf": [0.128, 0.083, 0.179],
        "G3_large_delta_iou": {"box": 0.052, "point": -0.021, "point_box": 0.102},
        "G4_refine_delta_iou": 0.062,
        "G5_decode_ms": {"stock": 8.27, "tuned": 8.26},
        "G3b_loose_large": 0.094,
        "G6_loose_small": 0.111,
        "passed": False,
        "failed": "G3 (large parts, point only: -0.021)",
    },
}


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def same_bytes(a, b) -> bool:
    """Same dtype, shape and bytes (``torch.equal`` would call NaN != NaN)."""
    import torch

    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    fa = a.contiguous().reshape(-1)
    fb = b.contiguous().reshape(-1)
    if fa.numel() == 0:
        return True
    return bool(torch.equal(fa.view(torch.uint8), fb.view(torch.uint8)))


def compare(stock: dict, tuned: dict) -> dict:
    """Which tensors differ, and the checks the docstring lists."""
    if set(stock) != set(tuned):
        only_s = sorted(set(stock) - set(tuned))[:5]
        only_t = sorted(set(tuned) - set(stock))[:5]
        raise SystemExit(f"the two checkpoints name different tensors: only stock "
                         f"{only_s}, only tuned {only_t}")
    for key in stock:
        if stock[key].dtype != tuned[key].dtype or stock[key].shape != tuned[key].shape:
            raise SystemExit(f"{key}: {stock[key].dtype} {tuple(stock[key].shape)} vs "
                             f"{tuned[key].dtype} {tuple(tuned[key].shape)}")
    prefixes = tuple(f"{m}." for m in HEAD_MODULES)
    differ = sorted(k for k in stock if not same_bytes(stock[k], tuned[k]))
    outside = [k for k in differ if not k.startswith(prefixes)]
    encoder = [k for k in stock if k.startswith("image_encoder.")]
    in_modules = [k for k in stock if k.startswith(prefixes)]
    return {
        "differ": differ,
        "outside": outside,
        "encoder": encoder,
        "encoder_differ": [k for k in encoder if k in set(differ)],
        "in_modules": in_modules,
        "unchanged_in_modules": sorted(set(in_modules) - set(differ)),
        "others": [k for k in stock if not k.startswith(prefixes) and k not in set(encoder)],
        "params": int(sum(tuned[k].numel() for k in differ)),
        "per_module": {m: sum(1 for k in differ if k.startswith(f"{m}.")) for m in HEAD_MODULES},
    }


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--stock", type=Path, default=STOCK)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--expect", type=int, default=EXPECT)
    ap.add_argument("--dry-run", action="store_true", help="compare and report, write nothing")
    args = ap.parse_args(argv)

    import torch

    env.limit_threads()
    print(f"[export] stock  {args.stock}", flush=True)
    print(f"[export] source {args.source}", flush=True)
    stock = torch.load(str(args.stock), map_location="cpu", weights_only=True, mmap=True)["model"]
    tuned = torch.load(str(args.source), map_location="cpu", weights_only=True, mmap=True)["model"]
    found = compare(stock, tuned)
    differ = found["differ"]
    print(f"[export] {len(stock)} tensors per checkpoint; {len(differ)} differ "
          f"({found['params'] / 1e6:.3f} M parameters: {found['per_module']})")
    print(f"[export] image encoder: {len(found['encoder'])} tensors, "
          f"{len(found['encoder_differ'])} differ; other tensors outside the head modules: "
          f"{len(found['others'])}, differing {len(found['outside'])}")
    print(f"[export] head modules hold {len(found['in_modules'])} tensors; unchanged: "
          f"{found['unchanged_in_modules']}")
    if found["outside"]:
        raise SystemExit(f"tensors outside {HEAD_MODULES} differ: {found['outside'][:10]}")
    if found["encoder_differ"]:
        raise SystemExit("the image encoder is not byte-identical")
    if len(differ) != args.expect:
        raise SystemExit(f"{len(differ)} tensors differ, expected {args.expect}")
    keys = list(FINGERPRINT_KEYS) + found["unchanged_in_modules"]
    base_print = fingerprint(stock, keys)
    if args.dry_run:
        print("[export] dry run: nothing written")
        return 0

    stock_sha = sha256_file(args.stock)
    source_sha = sha256_file(args.source)
    print(f"[export] sha256 stock {stock_sha[:16]}..., source {source_sha[:16]}...", flush=True)
    state = {k: tuned[k].detach().clone().contiguous() for k in differ}
    blob = {
        "format": HEAD_FORMAT,
        "modules": list(HEAD_MODULES),
        "base": {"file": args.stock.name, "sha256": stock_sha, "fingerprint": base_print},
        "source": {"file": args.source.name, "sha256": source_sha},
        "state_dict": state,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    torch.save(blob, str(partial))
    os.replace(partial, args.out)

    # read it back the way the app does, and compare tensor by tensor
    back = read_head(args.out)
    assert sorted(back["state_dict"]) == differ
    for key in differ:
        assert same_bytes(back["state_dict"][key], tuned[key]), key
    assert back["base"]["fingerprint"] == base_print
    head_sha = sha256_file(args.out)

    train = _read_json(TRAIN_MANIFEST) or {}
    gate = _read_json(GATE) or {}
    manifest = {
        "format": HEAD_FORMAT,
        "what": "SAM 2.1 Hiera-L prompt encoder + mask decoder fine-tuned on the annotator's "
                "verified scan masks (L5 recipe v2, fold f5), as a delta on the stock checkpoint",
        "head": {"file": args.out.name, "sha256": head_sha, "bytes": args.out.stat().st_size},
        "source": {"file": args.source.name, "sha256": source_sha,
                   "bytes": args.source.stat().st_size},
        "base": {"file": args.stock.name, "sha256": stock_sha,
                 "bytes": args.stock.stat().st_size, "fingerprint": base_print},
        "sam2_config": env.SAM2_CONFIG,
        "modules": list(HEAD_MODULES),
        "tensors": {"count": len(differ), "params": found["params"],
                    "per_module": found["per_module"], "names": differ},
        "unchanged_in_modules": found["unchanged_in_modules"],
        "identical": {"image_encoder_tensors": len(found["encoder"]),
                      "other_tensors": len(found["others"]),
                      "checked": "byte for byte"},
        "training": {
            "experiment": "L5 (experiments/l5_sam/REPORT.md)",
            "recipe": "v2 (amendment 1: half of the box prompts pushed out by up to 25 %)",
            "fold": "f5",
            "trained_on_desktops": [2, 3, 4, 13],
            "held_out": 5,
            "held_out_blind": True,
            "date": "2026-10-17",
            "trained_modules": "prompt encoder + mask decoder (incl. conv_s0/conv_s1); "
                               "image encoder frozen",
            "train_manifest": train,
        },
        "evaluation_d5_blind": D5_BLIND,
        "gate_json": gate,
        "ruling": "REPORT.md section 0: one image encoder, two decoder heads; boxes -> tuned, "
                  "point-only large classes -> stock, other point-only -> tuned, refinements "
                  "-> the head that produced the mask (task U19, configs/sam.yaml)",
        "exported_by": "experiments/l5_sam/export_head.py",
        "exported_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    out_json = args.out.with_suffix(".json")
    out_json.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[export] wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, sha256 "
          f"{head_sha[:16]}...) and {out_json.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
