"""The GT-mask overlay reference of the VLM evaluation (W4-C; paper plan 6.2).

::

    # 1. the masks, once per export, from the scanner pack's COCO (confirmed rows)
    python scripts/vlm_overlay.py masks --export <vlm_scan.jsonl> \\
        --coco-root exports/scanpack_<date> [--out <vlm_scan.masks.jsonl>]

    # 2. the variant: every eligible record with its frames painted
    python scripts/vlm_overlay.py render --export <truth.jsonl> [--records <subset.jsonl>] \\
        --image-root <images> --assets <dir> --out <mask_overlay.jsonl> \\
        [--masks <masks.jsonl>] [--force]

The "minimal set" reference of the paper: the frame a model sees, with the
**confirmed visible masks painted** in one class colour each (the order of
``configs/taxonomy.yaml``, a fixed alpha, a thin outline of the same colour).
Nothing else is drawn -- no text, no box, no label, nothing that depends on the
record -- so what a model scores on it over the plain frame is the price of
perception: how much it misses because it cannot find the parts, not because it
does not know what to do with them.

* A record is eligible when every frame it shows is a **confirmed** frame of
  its view (the oracle's ``frame.confirmed``) that has masks, and the task is
  not V8 (the box *is* the answer: a painted box would be the answer). V15's two
  pictures are both painted when both frames are confirmed (the scanner is the
  only confirmed view on most desktops, so almost no V15 pair qualifies).
* A ``multi`` record (``label.view == "multi"``: one moment from every camera,
  the redesign of 2026-11-03) has **only its scanner frames painted** -- image 0,
  and for V3 / V12 also the scanner frame of the second moment (:func:`painted_images`);
  those must be confirmed with masks, the side cameras' pictures are shown as they
  are. The variant's manifest points those at the plain frame, relative to
  ``--assets`` (``../images/...`` in the pack), with ``"painted": false``.
* The masks are the rows the export itself read: the instances the oracle's
  frame table lists with a ``visibility`` (confirmed rows, no screw hole, no
  框外零件). Their visible RLE comes from a ``<export>.masks.jsonl`` sidecar
  that this script builds from the scanner pack's COCO
  (``scripts/export_scanpack.py``: ``D<nn>/annotations.json``, confirmed rows
  only, full-frame coordinates). A row's visibility in the COCO must agree
  with the oracle's or the build stops: a scanner pack from another snapshot
  paints masks the questions were not asked about.
* The overlay of a frame is written once, to ``--assets/<view>/D<nn>/s<nn>.png``
  (the manifest's own relative path, ``.png``), and the variant's records keep
  their ids: a manifest sidecar beside the records maps the ids to the
  overlays, so ``vlm_to_chat.py --export <mask_overlay.jsonl> --image-root
  <assets>`` shows them. ``label.reference = "mask_overlay"``; the label side is
  otherwise untouched, so ``vlm_score.py --truth <the plain truth file>`` grades
  the replies.

No ``tda`` import: ``json``, PyYAML (the class order) and Pillow (the painting),
which is what the evaluation pack's ``tools/`` has. The COCO RLE is decoded
here in pure Python (column-major, the LEB128-like string of pycocotools).
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional

__all__ = ["ALPHA", "OUTLINE_PX", "PALETTE", "OverlayError", "class_colour",
           "decode_counts", "eligibility", "load_masks", "mask_image", "masks_from_coco",
           "paint", "painted_images", "render", "render_export", "taxonomy_classes", "write_masks"]

#: Painted colour weight, 0..255 (0.45): the picture stays readable under it.
ALPHA = 115
#: Outline width in pixels, inside the mask, full colour.
OUTLINE_PX = 2
#: One colour per taxonomy class, by the class's position in
#: ``configs/taxonomy.yaml`` ``classes`` (chassis first). Fixed for every
#: desktop and view; 26 entries for the 25 classes.
PALETTE = (
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (100, 149, 237), (154, 205, 50), (255, 105, 180), (0, 255, 127), (139, 69, 19),
    (75, 0, 130),
)
REFERENCE = "mask_overlay"
#: The export unit whose records show one moment from every camera; only its scanner
#: frames are painted.
MULTI_VIEW = "multi"
#: The view a multi record's truth, evidence and marks rest on.
MULTI_PRIMARY = "scan"
MASKS_SUFFIX = "masks.jsonl"
#: Sidecars copied beside the variant so it is a truth file of its own.
COPIED_SIDECARS = ("oracle.jsonl", "summary.json")


class OverlayError(RuntimeError):
    """The inputs do not support what the overlay promises."""


# --------------------------------------------------------------------------- #
# small io (a copy of vlm_io's, so this file has no import of its own)
# --------------------------------------------------------------------------- #
def read_jsonl(path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows: Iterable[dict]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def sidecar(export_jsonl, suffix: str) -> Path:
    p = Path(export_jsonl)
    return p.with_name(p.stem + "." + suffix)


def load_manifest(export_jsonl) -> dict[str, dict]:
    path = sidecar(export_jsonl, "images_manifest.jsonl")
    if not path.exists():
        return {}
    return {row["image"]: row for row in read_jsonl(path) if row.get("type") != "header"}


def load_frames(export_jsonl) -> dict[str, dict]:
    """``image id -> the oracle's frame line`` (``confirmed``, ``state``)."""
    path = sidecar(export_jsonl, "oracle.jsonl")
    if not path.exists():
        return {}
    return {row["image"]: row for row in read_jsonl(path) if row.get("type") == "frame"}


# --------------------------------------------------------------------------- #
# the palette
# --------------------------------------------------------------------------- #
def taxonomy_classes(path: Optional[str] = None) -> list[str]:
    """The class names in ``configs/taxonomy.yaml`` order (the palette's index)."""
    import yaml

    candidates = [Path(path)] if path else [
        Path(__file__).resolve().parent / "configs" / "taxonomy.yaml",        # the pack
        Path(__file__).resolve().parents[1] / "configs" / "taxonomy.yaml",    # the repo
    ]
    for cand in candidates:
        if cand.is_file():
            doc = yaml.safe_load(cand.read_text(encoding="utf-8"))
            classes = list((doc or {}).get("classes") or {})
            if not classes:
                raise OverlayError(f"{cand}: no classes")
            if len(classes) > len(PALETTE):
                raise OverlayError(f"{len(classes)} classes but {len(PALETTE)} colours")
            return classes
    raise OverlayError("configs/taxonomy.yaml not found; pass --taxonomy")


def class_colour(cls: str, classes: list[str]) -> tuple[int, int, int]:
    """The fixed colour of a class; unknown classes (none expected) take the last."""
    try:
        return PALETTE[classes.index(cls)]
    except ValueError:
        return PALETTE[-1]


def palette_table(classes: list[str]) -> dict[str, str]:
    return {cls: "#%02x%02x%02x" % class_colour(cls, classes) for cls in classes}


# --------------------------------------------------------------------------- #
# COCO RLE in pure Python
# --------------------------------------------------------------------------- #
def decode_counts(text: str) -> list[int]:
    """pycocotools' compressed string -> run lengths (background first)."""
    counts: list[int] = []
    p, m = 0, 0
    n = len(text)
    while p < n:
        x, k, more = 0, 0, 1
        while more:
            c = ord(text[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += counts[m - 2]
        counts.append(x)
        m += 1
    return counts


def mask_image(size: list, counts: str):
    """A Pillow ``L`` image (0 / 255) of a COCO RLE, ``size = [h, w]``."""
    from PIL import Image

    h, w = int(size[0]), int(size[1])
    runs = decode_counts(counts) if isinstance(counts, str) else [int(c) for c in counts]
    parts = []
    on = False
    for run in runs:
        parts.append((b"\xff" if on else b"\x00") * int(run))
        on = not on
    buf = b"".join(parts)
    if len(buf) != h * w:
        raise OverlayError(f"RLE of {len(buf)} pixels for a {w}x{h} mask")
    # COCO runs are column-major: a row of this image is a column of the mask
    return Image.frombytes("L", (h, w), buf).transpose(Image.Transpose.TRANSPOSE)


# --------------------------------------------------------------------------- #
# the masks sidecar
# --------------------------------------------------------------------------- #
def _confirmed_keys(frame: dict) -> set[str]:
    """The instances the export read on this frame: those with a visibility."""
    return {k for k, e in (frame.get("state") or {}).items() if "visibility" in e}


def masks_from_coco(coco_root, manifest: dict[str, dict], frames: dict[str, dict],
                    log=None) -> tuple[list[dict], dict]:
    """Mask rows for every confirmed frame of ``manifest`` from ``coco_root``.

    ``coco_root`` holds ``**/annotations.json`` files (the scanner pack:
    ``D<nn>/annotations.json``), each a COCO document whose ``images[].extra``
    names ``desktop``, ``view``, ``step`` and whose ``annotations[].attributes``
    carry ``instance_key`` and ``visibility``. A row is kept when its instance
    is one the oracle's frame table lists with a visibility; its visibility
    must agree.

    Returns ``(rows sorted by desktop / view / step, stats)``; ``stats`` names
    the confirmed frames no COCO file covers (``confirmed_without_masks``).
    """
    root = Path(coco_root)
    files = sorted(p for p in root.rglob("annotations.json") if p.is_file())
    if not files:
        raise OverlayError(f"{coco_root}: no annotations.json under it")
    wanted: dict[tuple[int, str, int], str] = {}
    for image_id, entry in manifest.items():
        frame = frames.get(image_id)
        if frame and frame.get("confirmed"):
            wanted[(int(entry["desktop"]), str(entry["view"]), int(entry["step"]))] = image_id
    rows: dict[str, dict] = {}
    disagreements: list[str] = []
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in doc.get("categories") or []}
        by_image: dict[int, list[dict]] = {}
        for ann in doc.get("annotations") or []:
            by_image.setdefault(ann["image_id"], []).append(ann)
        for image in doc.get("images") or []:
            extra = image.get("extra") or {}
            if not {"desktop", "view", "step"} <= set(extra):
                continue
            key = (int(extra["desktop"]), str(extra["view"]), int(extra["step"]))
            image_id = wanted.get(key)
            if image_id is None or image_id in rows:
                continue
            frame = frames[image_id]
            keys = _confirmed_keys(frame)
            masks = []
            for ann in sorted(by_image.get(image["id"], []),
                              key=lambda a: str((a.get("attributes") or {}).get("instance_key"))):
                attrs = ann.get("attributes") or {}
                inst = attrs.get("instance_key")
                if inst not in keys:
                    continue  # a screw hole, a part outside the machine: not asked about
                seg = ann.get("segmentation") or {}
                if not isinstance(seg, dict) or not seg.get("counts"):
                    continue  # a box-only row has no visible mask to paint
                want = frame["state"][inst].get("visibility")
                if attrs.get("visibility") != want:
                    disagreements.append(f"D{key[0]:02d} {key[1]} s{key[2]:03d} {inst}: "
                                         f"COCO {attrs.get('visibility')} vs oracle {want}")
                    continue
                masks.append({"instance": inst, "class": cats.get(ann["category_id"], "?"),
                              "visibility": want, "counts": seg["counts"]})
            size = [int(image["height"]), int(image["width"])]
            for ann in by_image.get(image["id"], []):
                seg = ann.get("segmentation")
                if isinstance(seg, dict) and seg.get("size") and list(seg["size"]) != size:
                    raise OverlayError(f"{path}: an RLE size differs from the image size "
                                       f"(ROI-cropped COCO? the overlay needs full-frame masks)")
            rows[image_id] = {"type": "frame", "image": image_id, "desktop": key[0],
                              "view": key[1], "step": key[2], "size": size, "masks": masks}
        if log:
            log(f"[vlm-overlay] {path.parent.name}: {len(rows)}/{len(wanted)} confirmed frames "
                f"have masks")
    if disagreements:
        shown = "; ".join(disagreements[:5]) + (" ..." if len(disagreements) > 5 else "")
        raise OverlayError(f"{len(disagreements)} confirmed rows whose visibility in {coco_root} "
                           f"differs from the export's oracle -- the scanner pack is not from "
                           f"this snapshot: {shown}")
    missing = sorted(k for k in wanted if wanted[k] not in rows)
    ordered = sorted(rows.values(), key=lambda r: (r["desktop"], r["view"], r["step"]))
    stats = {"confirmed_frames": len(wanted), "frames_with_masks": len(ordered),
             "masks": sum(len(r["masks"]) for r in ordered),
             "confirmed_without_masks": [f"D{d:02d} {v} s{s:03d}" for d, v, s in missing],
             "source": root.name}
    return ordered, stats


def write_masks(path, rows: list[dict], classes: list[str], source: str) -> int:
    header = {"type": "header", "format": "coco_rle", "order": "column-major",
              "source": source, "classes": classes, "reference": REFERENCE}
    return write_jsonl(path, [header, *rows])


def load_masks(path) -> dict[str, dict]:
    """``image id -> {size, masks}`` of a masks sidecar (``{}`` when absent)."""
    path = Path(path)
    if not path.exists():
        return {}
    return {row["image"]: row for row in read_jsonl(path) if row.get("type") == "frame"}


def cut_masks(src, dst, image_ids: set[str]) -> int:
    """The sidecar restricted to ``image_ids`` (the header kept)."""
    rows = [row for row in read_jsonl(src)
            if row.get("type") == "header" or row.get("image") in image_ids]
    return write_jsonl(dst, rows)


# --------------------------------------------------------------------------- #
# painting
# --------------------------------------------------------------------------- #
def paint(image, masks: list[dict], size: list, classes: list[str], *, alpha: int = ALPHA,
          outline: int = OUTLINE_PX):
    """``image`` (Pillow, RGB) with every mask painted; returns a new image.

    Big masks first (by area), so a screw stays visible on its cooler; inside
    each mask the colour is blended at ``alpha`` and its inner ``outline``
    pixels are the full colour. Pixels under no mask are untouched.
    """
    from PIL import Image, ImageChops, ImageFilter

    if list(image.size) != [int(size[1]), int(size[0])]:
        raise OverlayError(f"image is {image.size[0]}x{image.size[1]} but the masks are "
                           f"{size[1]}x{size[0]}")
    out = image.convert("RGB")
    W, H = out.size
    decoded = []
    for m in masks:
        mask = mask_image(size, m["counts"])
        box = mask.getbbox()
        if box is None:
            continue  # a part hidden completely: nothing visible to paint
        area = (box[2] - box[0]) * (box[3] - box[1])
        decoded.append((-area, m["instance"], m["class"], mask, box))
    decoded.sort(key=lambda t: (t[0], t[1]))
    painted = Counter()
    for _, _, cls, mask, box in decoded:
        colour = class_colour(cls, classes)
        pad = outline + 1
        x0, y0 = max(0, box[0] - pad), max(0, box[1] - pad)
        x1, y1 = min(W, box[2] + pad), min(H, box[3] + pad)
        crop = mask.crop((x0, y0, x1, y1))
        region = out.crop((x0, y0, x1, y1))
        weight = crop.point(lambda v: alpha if v else 0)
        region = Image.composite(Image.new("RGB", region.size, colour), region, weight)
        if outline > 0:
            edge = ImageChops.subtract(crop, crop.filter(ImageFilter.MinFilter(2 * outline + 1)))
            region.paste(colour, mask=edge)
        out.paste(region, (x0, y0))
        painted[cls] += 1
    return out, painted


def painted_images(record: dict, frames: dict[str, dict]) -> list[str]:
    """The images of a record the overlay paints: all of them, except in a ``multi``
    record, where only the scanner frames are (image 0, and the scanner frame of the
    second moment of V3 / V12); a side camera's picture is shown as it is."""
    images = list(record["prompt"].get("images") or [])
    if record["label"].get("view") != MULTI_VIEW:
        return images
    views = record["label"].get("views")
    if isinstance(views, list) and len(views) == len(images):
        return [i for i, v in zip(images, views) if v == MULTI_PRIMARY]
    picked = [i for i in images if (frames.get(i) or {}).get("view") == MULTI_PRIMARY]
    return picked or images[:1]


def eligibility(record: dict, frames: dict[str, dict], masks: dict[str, dict]) -> Optional[str]:
    """``None`` when the record is eligible, else why not."""
    if record["prompt"]["task"] == "V8":
        return "V8"
    images = painted_images(record, frames)
    if not images:
        return "no_image"
    for image_id in images:
        frame = frames.get(image_id)
        if not frame or not frame.get("confirmed"):
            return "unconfirmed"
    for image_id in images:
        if image_id not in masks:
            return "no_masks"
    return None


def _relative_to(src: Path, assets: Path) -> str:
    """``src`` as a POSIX path relative to ``assets`` (``../images/...`` in the pack)."""
    import os

    try:
        return Path(os.path.relpath(Path(src).resolve(), Path(assets).resolve())).as_posix()
    except ValueError:  # another drive: no relative path exists
        return Path(src).resolve().as_posix()


def overlay_path(entry: dict) -> str:
    """The overlay's path under ``--assets``: the manifest's, as ``.png``."""
    return Path(entry["path"]).with_suffix(".png").as_posix()


def render(records: Iterable[dict], manifest: dict[str, dict], frames: dict[str, dict],
           masks: dict[str, dict], image_root, assets, classes: list[str], *,
           force: bool = False, log=None) -> tuple[list[dict], list[dict], dict]:
    """``(variant records, overlay manifest rows, summary)``; overlays written to ``assets``."""
    from PIL import Image

    root = Path(image_root) if image_root else None
    assets = Path(assets)
    out_records: list[dict] = []
    out_manifest: dict[str, dict] = {}
    skipped: Counter = Counter()
    painted_classes: Counter = Counter()
    written = reused = unpainted = 0
    done: dict[str, str] = {}
    n_in = 0
    for record in records:
        n_in += 1
        why = eligibility(record, frames, masks)
        if why:
            skipped[why] += 1
            continue
        paint_these = set(painted_images(record, frames))
        for image_id in record["prompt"]["images"]:
            if image_id in done:
                continue
            entry = manifest.get(image_id)
            if entry is None:
                raise OverlayError(f"image {image_id} is not in the manifest")
            if image_id not in paint_these:
                # a side camera of a multi record: the plain frame, by a path relative to
                # the assets (the variant's manifest is read with --image-root <assets>)
                src = (root / entry["path"]) if root else Path(entry["path"])
                rel = _relative_to(src, assets)
                done[image_id] = rel
                out_manifest[image_id] = ({k: v for k, v in entry.items() if k != "path"}
                                          | {"path": rel, "painted": False})
                unpainted += 1
                continue
            rel = overlay_path(entry)
            target = assets / rel
            if target.exists() and not force:
                reused += 1
            else:
                src = (root / entry["path"]) if root else Path(entry["path"])
                if not src.is_file():
                    raise OverlayError(f"frame {src} is missing")
                img = Image.open(src)
                img.load()
                row = masks[image_id]
                out, counts = paint(img, row["masks"], row["size"], classes)
                painted_classes.update(counts)
                target.parent.mkdir(parents=True, exist_ok=True)
                out.save(target, format="PNG")
                written += 1
            done[image_id] = rel
            out_manifest[image_id] = {k: v for k, v in entry.items() if k != "path"} | {"path": rel}
            if log and (written + reused) % 200 == 0:
                log(f"[vlm-overlay]   {written + reused} overlays")
        out = copy.deepcopy(record)
        out["label"]["reference"] = REFERENCE
        out_records.append(out)
    manifest_rows = [{"type": "header", "reference": REFERENCE,
                      "note": "the same image ids as the truth file; paths are overlays"}]
    manifest_rows += [out_manifest[k] for k in sorted(out_manifest)]
    summary = {"reference": REFERENCE, "records_in": n_in, "records": len(out_records),
               "skipped": dict(sorted(skipped.items())),
               "overlays": len(done) - unpainted, "overlays_written": written,
               "overlays_reused": reused, "unpainted_side_images": unpainted,
               "alpha": round(ALPHA / 255, 3), "outline_px": OUTLINE_PX,
               "palette": palette_table(classes),
               "masks_painted_by_class": dict(sorted(painted_classes.items())),
               "excluded": {"V8": "the box is the answer",
                            "unconfirmed": "a frame of the record is not a confirmed frame",
                            "no_masks": "a confirmed frame without masks in the source",
                            "no_image": "the record shows no picture"},
               "multi_rule": "only the scanner frames of a multi record are painted"}
    return out_records, manifest_rows, summary


def render_export(export, out, image_root, assets, *, records=None, masks_path=None,
                  coco_root=None, taxonomy=None, force: bool = False, log=print) -> dict:
    """The whole variant for one truth file; returns the summary (also written).

    Masks: ``masks_path``, else ``<export>.masks.jsonl`` beside the export, else
    built in memory from ``coco_root``. Writes ``out`` (the records),
    ``<out>.images_manifest.jsonl`` (ids -> overlays), copies the oracle and
    summary sidecars beside it, and ``<out>.summary.json``.
    """
    export = Path(export)
    out = Path(out)
    classes = taxonomy_classes(taxonomy)
    manifest = load_manifest(export)
    frames = load_frames(export)
    if not manifest or not frames:
        raise OverlayError(f"{export}: the manifest and oracle sidecars are needed")
    source = None
    masks_file = Path(masks_path) if masks_path else sidecar(export, MASKS_SUFFIX)
    if masks_file.exists():
        masks = load_masks(masks_file)
        source = masks_file.name
    elif coco_root:
        rows, stats = masks_from_coco(coco_root, manifest, frames, log)
        masks = {r["image"]: r for r in rows}
        source = f"COCO {stats['source']}"
    else:
        raise OverlayError(f"no masks: neither {masks_file.name} beside the export nor "
                           f"--coco-root")
    rows_in = read_jsonl(records) if records else read_jsonl(export)
    variant, manifest_rows, summary = render(rows_in, manifest, frames, masks, image_root,
                                             assets, classes, force=force, log=log)
    write_jsonl(out, variant)
    write_jsonl(sidecar(out, "images_manifest.jsonl"), manifest_rows)
    for suffix in COPIED_SIDECARS:
        src = sidecar(export, suffix)
        if src.exists() and src.resolve() != sidecar(out, suffix).resolve():
            shutil.copyfile(src, sidecar(out, suffix))
    summary["masks_source"] = source
    summary["assets"] = str(Path(assets).name)
    sidecar(out, "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    if log:
        log(f"[vlm-overlay] {summary['records']} of {summary['records_in']} records painted "
            f"({summary['overlays']} overlays, {written_s(summary)}); skipped "
            + (", ".join(f"{k} {n}" for k, n in summary["skipped"].items()) or "none")
            + f" -> {out}")
    return summary


def written_s(summary: dict) -> str:
    return f"{summary['overlays_written']} written, {summary['overlays_reused']} reused"


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("masks", help="the masks sidecar of an export from a scanner pack's COCO")
    m.add_argument("--export", required=True, help="the export-vlm JSONL (manifest + oracle beside it)")
    m.add_argument("--coco-root", required=True, help="directory with D<nn>/annotations.json")
    m.add_argument("--out", default=None, help=f"default <export>.{MASKS_SUFFIX}")
    m.add_argument("--taxonomy", default=None)
    r = sub.add_parser("render", help="the mask_overlay variant of a truth file")
    r.add_argument("--export", required=True, help="the truth file (manifest, oracle, masks beside it)")
    r.add_argument("--records", default=None, help="paint these records instead of the export's")
    r.add_argument("--image-root", required=True, help="what the manifest paths are relative to")
    r.add_argument("--assets", required=True, help="where the overlays go: <view>/D<nn>/s<nn>.png")
    r.add_argument("--out", required=True, help="the variant's records JSONL")
    r.add_argument("--masks", default=None, help=f"a masks sidecar (default <export>.{MASKS_SUFFIX})")
    r.add_argument("--coco-root", default=None, help="build the masks from this COCO when no sidecar")
    r.add_argument("--taxonomy", default=None)
    r.add_argument("--force", action="store_true", help="repaint overlays that already exist")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "masks":
            classes = taxonomy_classes(args.taxonomy)
            manifest, frames = load_manifest(args.export), load_frames(args.export)
            rows, stats = masks_from_coco(args.coco_root, manifest, frames, print)
            out = args.out or sidecar(args.export, MASKS_SUFFIX)
            write_masks(out, rows, classes, stats["source"])
            print(f"[vlm-overlay] masks: {stats['frames_with_masks']} of "
                  f"{stats['confirmed_frames']} confirmed frames, {stats['masks']} masks -> {out}")
            if stats["confirmed_without_masks"]:
                print(f"[vlm-overlay] {len(stats['confirmed_without_masks'])} confirmed frames "
                      f"have no masks in {args.coco_root}: "
                      + ", ".join(stats["confirmed_without_masks"][:8])
                      + (" ..." if len(stats["confirmed_without_masks"]) > 8 else ""))
        else:
            render_export(args.export, args.out, args.image_root, args.assets,
                          records=args.records, masks_path=args.masks, coco_root=args.coco_root,
                          taxonomy=args.taxonomy, force=args.force)
    except OverlayError as exc:
        print(f"[vlm-overlay] refused: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
