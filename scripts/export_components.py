"""The components-gallery package: the parts laid out after the teardown ("all components").

::

    python scripts/export_components.py --db <copy of tda.sqlite> \
        [--out D:/DataSet/exports/componentpack_2026-11-02] [--cache-dir <dir>] \
        [--paths configs/paths.yaml] [--work <dir>] [--force] [--no-hf-check]

After a desktop was taken apart its parts were laid out on the table and
photographed once more by every camera: the structured-light scanner
(``cmp_scan``, the scanner's ``allcompoentsrgb`` folders, 1600 x 1600, a burst
per shot), the two OAK-D cameras (``cmp_oak1`` / ``cmp_oak2``, the ``Components``
folders, 4032 x 3040) and the RealSense (``cmp_rs``). These photos are not
teardown steps; the annotator drew each part on them separately (tasks U10,
U42, U45). Only shots the annotator **confirmed** are packed.

Per view and desktop, ``<view>/D<nn>/`` holds

* ``images/<shot>.<ext>`` -- the picture the masks were drawn on, copied byte
  for byte from the raw drive (for ``cmp_scan`` the burst image the frame
  cache chose);
* ``labels/<shot>.png`` -- a 16-bit instance-id map of the confirmed visible
  masks (0 = background);
* ``annotations.json`` -- :func:`tda.core.export.coco.export_coco` of the view
  (confirmed rows only, amodal shapes), each annotation with ``instance_id``;
  each image with ``hf_path`` (the same file under ``raw/`` on the Hugging Face
  repository) instead of a machine path;
* ``instances.json`` -- per part: id, key, class, public attrs, whether the
  teardown sequence has it (``in_sequence``).

**Instance ids are the scanner package's.** A part's id here is the id the
scanner package (``annotations/scanner/D<nn>/instances.json``) gives the same
key, so a gallery mask and the part's masks during the teardown join on
``(desktop, instance_id)``. A part that only the gallery has (added from a
gallery shot, ``gallery_only``) gets the ids after the sequence's, in key
order.

Read-only on the database it is given (a backup-API copy is made into
``--work``), on the raw drive and on the cache. Deterministic.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from export_scanpack import (  # noqa: E402
    LICENSE_NOTE,
    PackError,
    _class_line,
    _decode,
    _work_copy,
    desktop_dir,
    instance_ids,
    sha256_of,
    write_json,
    write_label_png,
)
from tda.core.db import Db  # noqa: E402
from tda.core.export.coco import EXPORT_VERSION, category_ids, export_coco, load_ctx  # noqa: E402
from tda.core.export.release import model_names, public_attrs  # noqa: E402
from tda.core.model import is_provisional  # noqa: E402
from tda.core.taxonomy import load_taxonomy  # noqa: E402
from tda.core.truth import TruthService  # noqa: E402

PACK_VERSION = "1"
DEFAULT_OUT = "D:/DataSet/exports/componentpack_2026-11-02"
VIEWS = ("cmp_scan", "cmp_oak1", "cmp_oak2", "cmp_rs")
VIEW_TEXT = {
    "cmp_scan": "structured-light scanner, top-down, 1600 x 1600 PNG (one image of the shot's burst)",
    "cmp_oak1": "OAK-D Pro camera 1, 12 MP JPEG",
    "cmp_oak2": "OAK-D Pro camera 2, 12 MP JPEG",
    "cmp_rs": "Intel RealSense, 1280 x 720 PNG",
}
#: raw folder below the dataset root -> candidate folders on the Hugging Face repository
HF_ROOTS = (
    ("UGA DATA/", ("raw/scanner/",)),
    ("OAKD Capture/Desktop_Datacollection/DesktopData/", ("raw/oak/",)),
    ("Realsense Capture/Dataset Information/Exp/", ("raw/realsense/", "raw/realsense/Components/")),
)


def hf_candidates(below: str) -> list[str]:
    for prefix, roots in HF_ROOTS:
        if below.startswith(prefix):
            rest = below[len(prefix):]
            out = [r + rest for r in roots]
            # RealSense components may sit as Components/Desktop N/... on the repository
            if prefix.startswith("Realsense") and "/Components/" in rest:
                desk, _, tail = rest.partition("/Components/")
                out.append(f"raw/realsense/Components/{desk}/{tail}")
            return out
    return []


def shot_name(shot: int) -> str:
    return f"C{shot:02d}" if shot < 1000 else f"C{shot}"


def build(db_path: str, out_dir: str, *, cache_dir: Optional[str], work_dir: str,
          resolve: Callable, tail: Callable, force: bool,
          hf_exists: Optional[Callable[[list[str]], set]] = None) -> dict:
    out, work = Path(out_dir), Path(work_dir)
    if out.exists():
        if not force:
            raise PackError(f"{out} exists (pass --force)")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tax = load_taxonomy()
    cat_of = category_ids(tax)
    db = Db(str(_work_copy(db_path, work)))
    summary: dict = {"package": "componentpack", "pack_version": PACK_VERSION,
                     "coco_export_version": EXPORT_VERSION, "views": {}}
    by_class: Counter = Counter()
    hf_paths: dict[str, list[str]] = {}
    try:
        truth = TruthService(db, tax, cache_dir=cache_dir)
        for view in VIEWS:
            desktops = [int(r[0]) for r in db.conn.execute(
                "select distinct desktop from frame where view=? and review_status='verified' "
                "order by desktop", (view,))]
            if not desktops:
                continue
            doc = export_coco(db, tax, desktops, view, str(work / f"coco_{view}.json"),
                              truth=truth, cache_dir=cache_dir)
            if doc["info"]["version"] != EXPORT_VERSION:
                raise PackError(f"export version {doc['info']['version']} != {EXPORT_VERSION}")
            anns_of = defaultdict(list)
            for a in doc["annotations"]:
                anns_of[a["image_id"]].append(a)
            vstats = {"desktops": desktops, "shots": 0, "annotations": 0, "crowd": 0,
                      "parts": 0, "overlap_px": 0, "per_desktop": []}
            for desktop in desktops:
                ctx = load_ctx(db, tax, desktop, "scan")
                seq_ids = instance_ids(db, desktop, skip={**ctx.absent, **ctx.gallery_only})
                extra_keys = sorted(k for k in ctx.gallery_only if not is_provisional(k))
                key_to_id = dict(seq_ids)
                for i, k in enumerate(extra_keys, start=len(seq_ids) + 1):
                    key_to_id[k] = i
                ddir = out / view / desktop_dir(desktop)
                imgs = [im for im in doc["images"] if int(im["extra"]["desktop"]) == desktop]
                d_anns, d_images, used = [], [], set()
                for im in sorted(imgs, key=lambda i: int(i["extra"]["shot"])):
                    shot = int(im["extra"]["shot"])
                    src = im["extra"].get("source")
                    real = resolve(src) if src else None
                    below = tail(src) if src else None
                    if not real or not Path(real).is_file() or below is None:
                        raise PackError(f"{view} D{desktop:02d} shot {shot}: source {src} not found")
                    below = below.replace("\\", "/").lstrip("/")
                    ext = Path(real).suffix.lower()
                    name = shot_name(shot)
                    img_rel = f"images/{name}{ext}"
                    (ddir / "images").mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(real, ddir / img_rel)
                    hw = (int(im["height"]), int(im["width"]))
                    anns = sorted(anns_of.get(im["id"], []),
                                  key=lambda a: (-int(a.get("area") or 0), a["attributes"]["instance_key"]))
                    label = np.zeros(hw, dtype=np.uint16)
                    claimed = np.zeros(hw, dtype=np.uint16)
                    for a in anns:                      # big parts first, small ones on top
                        key = a["attributes"]["instance_key"]
                        if key not in key_to_id:
                            raise PackError(f"{view} D{desktop:02d} shot {shot}: {key} has no id")
                        m = _decode(a["segmentation"], hw)
                        a["instance_id"] = key_to_id[key]
                        a["shared_px"] = int((m & (claimed > 0)).sum())
                        label[m] = key_to_id[key]
                        claimed[m] += 1
                        used.add(key)
                        vstats["annotations"] += 1
                        vstats["crowd"] += int(a.get("iscrowd") or 0)
                        by_class[(view, tax_cls(db, desktop, key))] += 1
                    for a in anns:
                        m = _decode(a["segmentation"], hw)
                        a["label_px"] = int((label[m] == a["instance_id"]).sum()) if m.any() else 0
                    vstats["overlap_px"] += int((claimed > 1).sum())
                    write_label_png(ddir / "labels" / f"{name}.png", label)
                    cands = hf_candidates(below)
                    hf_paths[f"{view}/{desktop_dir(desktop)}/{img_rel}"] = cands
                    extra = {k: v for k, v in im["extra"].items() if k != "source"}
                    extra.update(source_folder=below.rpartition("/")[0],
                                 file_name=below.rpartition("/")[2], hf_path=cands[0] if cands else None,
                                 sha256=sha256_of(ddir / img_rel), label=f"labels/{name}.png")
                    d_images.append(dict(im, file_name=img_rel, extra=extra))
                    d_anns += anns
                    vstats["shots"] += 1
                meta = db.get_desktop(desktop) or {}
                model, model_sheet, model_note = model_names(meta)
                insts = db.instances(desktop)
                inst_out = {}
                for k, iid in sorted(key_to_id.items(), key=lambda kv: kv[1]):
                    if k not in used:
                        continue
                    rec = insts[k]
                    inst_out[str(iid)] = {
                        "id": iid, "key": k, "class": rec.cls, "category_id": cat_of.get(rec.cls),
                        "class_description": _class_line(rec),
                        "attrs": public_attrs(rec.attrs), "in_sequence": k in seq_ids,
                        "raw_names": list(rec.raw_names or []),
                    }
                head = {"desktop": desktop, "view": view, "brand": meta.get("brand"),
                        "model": model, "model_sheet": model_sheet, "model_note": model_note}
                coco = {"info": dict(doc["info"], description="E3D components gallery (pre-release)",
                                     desktops=[desktop]),
                        "categories": doc["categories"], "images": d_images, "annotations": d_anns}
                write_json(ddir / "annotations.json", coco)
                write_json(ddir / "instances.json", dict(head, instances=inst_out))
                vstats["parts"] += len(inst_out)
                vstats["per_desktop"].append({"desktop": desktop, "shots": len(d_images),
                                              "annotations": len(d_anns), "parts": len(inst_out)})
            summary["views"][view] = vstats
    finally:
        db.close()
    summary["annotations_by_class"] = {f"{v}:{c}": n for (v, c), n in sorted(by_class.items())}
    if hf_exists is not None:
        flat = sorted({c for cs in hf_paths.values() for c in cs})
        found = hf_exists(flat)
        missing = []
        for rel, cs in sorted(hf_paths.items()):
            hit = [c for c in cs if c in found]
            if not hit:
                missing.append(rel)
        if missing:
            raise PackError(f"{len(missing)} pictures have no copy under raw/ on the repository: {missing[:5]}")
        # rewrite hf_path to the candidate that exists
        for view in summary["views"]:
            for d in summary["views"][view]["desktops"]:
                path = out / view / desktop_dir(d) / "annotations.json"
                doc = json.loads(path.read_text(encoding="utf-8"))
                for im in doc["images"]:
                    cs = hf_paths[f"{view}/{desktop_dir(d)}/{im['file_name']}"]
                    im["extra"]["hf_path"] = next(c for c in cs if c in found)
                write_json(path, doc)
        summary["hf_paths_checked"] = len(hf_paths)
    (out / "LICENSE_NOTE.txt").write_text(LICENSE_NOTE, encoding="utf-8", newline="\n")
    (out / "README.md").write_text(readme_text(summary), encoding="utf-8", newline="\n")
    files = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = path.relative_to(out).as_posix()
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256_of(path)}
    write_json(out / "manifest.json", dict(summary, files=files))
    return summary


_CLS_CACHE: dict = {}


def tax_cls(db: Db, desktop: int, key: str) -> str:
    if desktop not in _CLS_CACHE:
        _CLS_CACHE[desktop] = {k: r.cls for k, r in db.instances(desktop).items()}
    return _CLS_CACHE[desktop].get(key, "?")


def readme_text(s: dict) -> str:
    rows = []
    for view, v in s["views"].items():
        ds = v["desktops"]
        span = (f"D{ds[0]:02d}-D{ds[-1]:02d}" if len(ds) > 1 and ds == list(range(ds[0], ds[-1] + 1))
                else ", ".join(f"D{d:02d}" for d in ds))
        rows.append(f"| `{view}` | {VIEW_TEXT[view]} | {span} | {v['shots']} | {v['parts']} | "
                    f"{v['annotations']} |")
    classes = Counter()
    for k, n in s["annotations_by_class"].items():
        classes[k.split(":", 1)[1]] += n
    cls_line = ", ".join(f"`{c}` {n}" for c, n in classes.most_common())
    return f"""\
# E3D components gallery (pre-release)

After each desktop was taken apart, its parts were laid out on the table and
photographed by every camera ("all components" shots). These shots are **not
teardown steps**: each part is drawn on them separately, so a part can be
recognised out of the machine as well as inside it. Only shots the annotator
confirmed are in this package.

| view | camera | desktops | shots | parts | annotations |
|---|---|---|---|---|---|
{chr(10).join(rows)}

Annotations by class (all views): {cls_line}.

## Layout

```
<view>/D<nn>/
  images/C<shot>.<ext>   the picture the masks were drawn on (byte for byte the raw file)
  labels/C<shot>.png     16-bit instance-id map of the confirmed visible masks (0 = background)
  annotations.json       COCO (RLE masks, amodal_segmentation, attributes per annotation)
  instances.json         per part: id, key, class, attrs, in_sequence
LICENSE_NOTE.txt, manifest.json (counts, sha256 of every file)
```

`<shot>` is the shot number inside the desktop's gallery folder (`C01`, `C02`,
...). Each image record's `extra` names the raw file: `source_folder` and
`file_name` below the dataset root, `hf_path` the same file under `raw/` on this
repository, and for the scanner `burst_choice` (which image of the burst was
drawn on: `p0` = `P_0.png`, `p0_dark` = a brighter one because `P_0` is dark).

## Joining with the teardown

An `instance_id` here is the id the scanner package
(`annotations/scanner/D<nn>/instances.json`) gives the same part key, so a
gallery mask joins the part's masks through the teardown on
`(desktop, instance_id)` (or on `key`). A part that only the gallery shows
(added from a gallery shot) has `in_sequence: false` and an id after the
sequence's. The parts are on the table, so every annotation's `placement` is
`on_bench` and its `state` the part's state after the teardown (`removed`).

## Masks

`segmentation` is the confirmed visible mask; `amodal_segmentation` the whole
shape. A part the annotator marked as present but not visible in a shot has an
empty `segmentation` and `visibility: occluded_full`. Overlaps between parts
are rare on the table: the label map paints the larger parts first; each
annotation says how many of its pixels another mask also claims (`shared_px`)
and how many the map gives it (`label_px`).

Pre-release data; terms in `LICENSE_NOTE.txt`.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--paths", default=str(REPO / "configs" / "paths.yaml"))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--work", default=str(REPO / ".cache" / "componentpack_work"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-hf-check", action="store_true",
                    help="do not check that every picture has a copy under raw/ on the repository")
    args = ap.parse_args(argv)
    from tda.cli_common import load_paths
    from tda.core import rawroot
    paths = load_paths(args.paths)
    raw = rawroot.current()
    hf_exists = None
    if not args.no_hf_check:
        def hf_exists(cands: list[str]) -> set:
            from huggingface_hub import HfApi
            api, found = HfApi(), set()
            for i in range(0, len(cands), 200):
                for info in api.get_paths_info("ChangChrisLiu/E3D-Desktop-Disassembly",
                                               cands[i:i + 200], repo_type="dataset"):
                    found.add(info.path)
            return found
    try:
        s = build(args.db, args.out, cache_dir=args.cache_dir or paths.get("cache_dir"),
                  work_dir=args.work, resolve=rawroot.resolve_raw,
                  tail=(raw.tail if raw is not None else (lambda p: None)), force=args.force,
                  hf_exists=hf_exists)
    except PackError as exc:
        print(f"[componentpack] refused: {exc}")
        return 1
    for view, v in s["views"].items():
        print(f"[componentpack] {view}: {len(v['desktops'])} desktops, {v['shots']} shots, "
              f"{v['parts']} parts, {v['annotations']} annotations, overlap {v['overlap_px']} px")
    print(f"[componentpack] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
