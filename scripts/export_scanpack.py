"""The scanner-view segmentation package for the 3D collaborators (task U30-PACK).

::

    python scripts/export_scanpack.py --db <copy of tda.sqlite> \
        [--out D:/DataSet/exports/scanpack_2026-10-26] [--desktops 2,3,4,13,19,23,24,28,30,31] \
        [--paths configs/paths.yaml] [--cache-dir <dir>] [--work <dir>] [--force]

The collaborators hold 3D reconstructions from the same structured-light
scanner whose top-down 1600 x 1600 RGB frames are our main view. They want to
pull each annotated part's points out of a reconstruction with our 2D masks.
Per desktop ``D<nn>/`` this writes:

* ``images/S<nn>.png`` -- the exact scanner picture the masks were drawn on:
  the burst image the frame cache chose (``<cache>/scan/D<nn>/manifest.json``),
  copied byte for byte **from the raw drive**; the cache's own copy is only
  compared against it (bytes, then pixels when the bytes differ);
* ``labels/S<nn>.png`` -- a 16-bit instance-id map of the confirmed visible
  masks (0 = background, ids from ``instances.json``);
* ``annotations.json`` -- :func:`tda.core.export.coco.export_coco` (version 2,
  default flags: confirmed rows only, full-frame coordinates, amodal shapes,
  the parts covered completely as crowd rows) with, per annotation,
  ``instance_id``, ``occluders``, ``clean`` and the rest of :data:`ANN_FIELDS`;
* ``frames.json``, ``instances.json``, ``sequence.json``;

and at the top ``README.md``, ``clean_parts.csv``, ``LICENSE_NOTE.txt`` and
``manifest.json`` (counts and the sha256 of every other file).

**What the masks are.** The truth table's *confirmed* rows -- what the
annotator verified -- read through ``export_coco(only_verified=True)``, never
a fresh compilation. Every exported frame is compiled once more all the same,
and the build stops (:class:`PackError`) unless the confirmed rows agree with
it by the app's own rule (:func:`tda.core.truth_refusals.frozen_disagreements`,
the comparison Space and the re-check queue use; same instance set both
ways). Pixel-exact agreement is not required -- a confirmed row may differ from
today's compilation inside the re-tracing tolerance -- and is counted
(``manifest.json`` ``truth_check``).

**The label map.** Confirmed rows confirmed at different times, and a frame's
「只改这一帧」 mask, can claim the same pixel (the compiler never subtracts a
per-frame mask from the others). The map gives such a pixel to the part higher
in the frame's layer order: the compiler's painting order (``in_chassis``,
then ``on_bench``, then ``context``, each bottom-up), its pairwise overrides,
then the parts with a per-frame mask on top. Each annotation says how many of
its visible pixels another one also claims (``shared_px``) and how many the map
gives it (``label_px``).

``clean`` is the set to start with: ``visible == amodal`` to the pixel, label
``visible``, the amodal box inside the pose segment's ROI, the amodal shape the
compiler's own (not widened by a visible mask reaching outside it), and every
visible pixel the label map's.

Read-only on the database it is given (a ``mode=ro`` backup-API copy is made
into ``--work`` and only that copy is opened -- the export's re-check pass
writes to it), on the raw drive and on the frame cache. Deterministic: two runs
over the same inputs write the same bytes (no wall clock, no machine path in
any file but the recorded raw paths).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tda.core import masks  # noqa: E402
from tda.core.db import Db  # noqa: E402
from tda.core.export.coco import (  # noqa: E402
    EXPORT_VERSION,
    category_ids,
    export_coco,
    load_ctx,
    roi_of,
)
from tda.core.implied import is_implied  # noqa: E402
from tda.core.model import (  # noqa: E402
    IMPLIED_RULE_ATTR,
    FrameKey,
    is_context,
    is_extra,
    is_provisional,
)
from tda.core.taxonomy import Taxonomy, load_taxonomy  # noqa: E402
from tda.core.truth import TruthService  # noqa: E402
from tda.core.truth_refusals import frozen_disagreements  # noqa: E402
from tda.ui.locator_pair import leaving_steps  # noqa: E402

__all__ = ["CLASS_EN", "DEFAULT_DESKTOPS", "PACK_VERSION", "PackError", "build_pack", "main"]

#: The ten desktops of the first package: five Dell, five HP.
DEFAULT_DESKTOPS = (2, 3, 4, 13, 19, 23, 24, 28, 30, 31)
DEFAULT_OUT = "D:/DataSet/exports/scanpack_2026-10-26"
VIEW = "scan"
PACK_VERSION = "1"
VERIFIED = "verified"
VISIBLE = "visible"
#: Paint order of the compiler's placement groups in the label map (bottom first).
GROUP_RANK = {"in_chassis": 0, "on_bench": 1, "context": 2}
#: What the package adds to every COCO annotation, at its top level.
ANN_FIELDS = ("instance_id", "visibility", "occlusion_ratio", "occluders",
              "occluded_unlabelled_px", "shared_px", "label_px", "amodal_is_shape", "clean")
#: A crowd row whose amodal shape the label map leaves mostly empty: nothing
#: labelled covers it, yet nothing of it is seen (a QA note, not a rule).
UNLABELLED_SHARE = 0.5

#: One line of English per taxonomy class. ``configs/taxonomy.yaml`` carries
#: no prose, so the words are written here; the class names are the taxonomy's.
CLASS_EN: dict[str, str] = {
    "chassis": "The computer case: the metal enclosure everything else sits in.",
    "drive_cage": "A cage or bracket holding storage or optical drives.",
    "cover": "A removable cover, shroud or duct (attrs.of says over what, e.g. the "
             "air shroud over the CPU cooler).",
    "cooler_bracket": "The bracket that holds the CPU cooler down.",
    "motherboard": "The main circuit board.",
    "cpu": "The processor in its socket.",
    "cpu_cooler": "The CPU cooler: heatsink and/or fan (attrs.kind).",
    "ram_module": "A memory (RAM) stick.",
    "psu": "The power supply unit.",
    "storage_drive": "A hard disk or SSD (attrs.kind).",
    "optical_drive": "A CD/DVD drive.",
    "expansion_card": "An add-in card in a slot (GPU, Wi-Fi card, ...).",
    "case_fan": "A case fan.",
    "misc_part": "Another removable part (attrs.name says which).",
    "screw": "A screw (attrs.role: what it fastens; a captive screw stays on its part).",
    "ram_latch": "The latch at the end of a RAM slot; moulded into the motherboard.",
    "cpu_socket_lever": "The CPU socket's retention lever; part of the motherboard.",
    "psu_latch": "A latch or bracket holding the power supply.",
    "drive_latch": "A latch holding a drive or a drive cage.",
    "card_latch": "A latch holding an expansion card.",
    "cooler_latch": "A latch or clip holding the CPU cooler.",
    "cable_clip": "A clip holding a cable.",
    "connector": "A cable plug in its socket (tracked only while plugged in).",
    "other": "An object outside the machine (context), named by its label.",
}


class PackError(RuntimeError):
    """The inputs cannot make a package; the message says what to look at."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def png_size(path) -> Optional[tuple[int, int]]:
    """``(width, height)`` from a PNG's IHDR chunk, or ``None`` for another format."""
    with open(path, "rb") as handle:
        head = handle.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", head[16:24])
    return int(width), int(height)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=1) + "\n", encoding="utf-8",
                    newline="\n")


def frame_name(step: int) -> str:
    return f"S{int(step):02d}"


def desktop_dir(desktop: int) -> str:
    return f"D{int(desktop):02d}"


def _decode(rle: Optional[dict], hw: tuple[int, int]) -> np.ndarray:
    if not rle:
        return np.zeros(hw, dtype=bool)
    out = masks.decode_rle(rle).astype(bool, copy=False)
    if out.shape != tuple(hw):
        raise PackError(f"a mask of size {out.shape} on a {hw} frame")
    return out


def _bbox_inside(box: Optional[Sequence[float]], roi: Optional[Sequence[int]]) -> bool:
    """Does COCO box ``[x, y, w, h]`` lie inside ROI ``[x0, y0, x1, y1]``?"""
    if box is None:
        return False
    if roi is None:
        return True
    x, y, w, h = (float(v) for v in box)
    return x >= roi[0] and y >= roi[1] and x + w <= roi[2] and y + h <= roi[3]


# --------------------------------------------------------------------------- #
# the source pictures
# --------------------------------------------------------------------------- #
def load_cache_manifest(cache_dir: Optional[str], desktop: int) -> dict:
    """``<cache>/scan/D<nn>/manifest.json``: which burst image each step chose."""
    if not cache_dir:
        return {}
    path = Path(cache_dir) / VIEW / desktop_dir(desktop) / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _source_record(frame: dict, chosen: Optional[dict]) -> dict:
    """Which stored file is the picture of a frame, and how it was chosen.

    The cache's choice wins (it is what the canvas showed while the masks were
    drawn); the frame row's ``path`` is the fallback, and a frame row naming
    another file of the same burst is recorded (the cache re-chose a dark
    ``P_0`` without the row following).
    """
    row_path = str(frame.get("path") or "")
    if chosen and chosen.get("src"):
        src = str(chosen["src"])
        metrics = chosen.get("metrics") or []
        index = chosen.get("chosen")
        if isinstance(index, int) and 0 <= index < len(metrics):
            listed = metrics[index].get("path")
            if listed and listed != src:
                raise PackError(f"cache manifest names {src} but its metrics[{index}] "
                                f"is {listed}")
        choice = {"by": "cache_manifest", "index": index,
                  "p_index": (metrics[index].get("p_index")
                              if isinstance(index, int) and 0 <= index < len(metrics) else None),
                  "reason": chosen.get("reason"), "burst_size": len(metrics) or None}
    else:
        src = row_path
        choice = {"by": "frame_row", "index": None, "p_index": None, "reason": None,
                  "burst_size": len(((frame.get("aux") or {}).get("burst")) or []) or None}
    return {"recorded": src, "choice": choice,
            "frame_row_path": row_path if row_path != src else None}


def _source_folder(recorded: str, tail: Callable[[str], Optional[str]]) -> tuple[str, str]:
    """``(folder below the dataset root, file name)`` of a recorded raw path."""
    below = tail(recorded)
    text = (below if below is not None else recorded).replace("\\", "/").lstrip("/")
    folder, _, name = text.rpartition("/")
    return folder, name


def _pixels_equal(a: str, b: str) -> bool:
    one = cv2.imread(a, cv2.IMREAD_UNCHANGED)
    two = cv2.imread(b, cv2.IMREAD_UNCHANGED)
    return one is not None and two is not None and one.shape == two.shape \
        and bool(np.array_equal(one, two))


# --------------------------------------------------------------------------- #
# one frame: the label map and what each annotation adds
# --------------------------------------------------------------------------- #
def paint_order(compiled, patched: set[str], present: Sequence[str]) -> tuple[list[str], list]:
    """``(bottom-up keys, pairs)`` the label map paints a frame's visible masks in.

    The compiler's painting order per placement group (:data:`GROUP_RANK`),
    preceded by any part with pixels but no layer of its own (its contest is
    lost), and followed by the parts with a per-frame visible mask -- the
    annotator's explicit "this is what is seen here", which the compiler never
    subtracts from the others. ``pairs`` are the pairwise overrides that held.
    """
    groups = sorted(compiled.painted, key=lambda g: (GROUP_RANK.get(g, 9), g))
    layered = [k for g in groups for k in compiled.painted[g] if k not in patched]
    seen = set(layered) | set(patched)
    loose = sorted(k for k in present if k not in seen)
    pairs = [pair for g in groups for pair in (compiled.pairs or {}).get(g, [])]
    return loose + layered + sorted(k for k in patched if k in present), pairs


def frame_label_map(anns: list[dict], hw: tuple[int, int], key_to_id: dict[str, int],
                    compiled, patched: set[str]) -> tuple[np.ndarray, dict]:
    """The 16-bit instance-id map of one frame, and the decoded masks.

    ``anns`` are the frame's annotations; their ``segmentation`` (the confirmed
    visible mask) is what is painted -- nothing is recompiled here.
    """
    visible = {a["attributes"]["instance_key"]: _decode(a["segmentation"], hw) for a in anns}
    order, pairs = paint_order(compiled, patched,
                               [k for k, m in visible.items() if m.any()])
    label = np.zeros(hw, dtype=np.uint16)
    tail = [k for k in order if k in patched]
    for key in order[:len(order) - len(tail)]:
        label[visible[key]] = key_to_id[key]
    for upper, lower in pairs:
        if upper in visible and lower in visible:
            label[(label == key_to_id[lower]) & visible[upper]] = key_to_id[upper]
    for key in tail:
        label[visible[key]] = key_to_id[key]
    return label, visible


def annotate_frame(anns: list[dict], label: np.ndarray, visible: dict[str, np.ndarray],
                   hw: tuple[int, int], key_to_id: dict[str, int], roi, compiled) -> None:
    """Add :data:`ANN_FIELDS` to each annotation of one frame, in place."""
    claims = np.zeros(hw, dtype=np.uint8)
    for mask in visible.values():
        claims += mask
    for ann in anns:
        attrs = ann["attributes"]
        key = attrs["instance_key"]
        iid = key_to_id[key]
        seen = visible[key]
        amodal = _decode(ann.get("amodal_segmentation"), hw)
        hidden = amodal & ~seen
        found = np.bincount(label[hidden].astype(np.int64), minlength=1)
        occluders = sorted(((int(i), int(n)) for i, n in enumerate(found) if i and n),
                           key=lambda t: (-t[1], t[0]))
        inst = compiled.instances.get(key)
        shape = None if inst is None else inst.amodal
        is_shape = bool(shape is not None and np.array_equal(np.asarray(shape, bool), amodal))
        label_px = int(np.count_nonzero(label[seen] == iid))
        area = int(np.count_nonzero(seen))
        clean = bool(area > 0 and not ann.get("iscrowd")
                     and area == int(np.count_nonzero(amodal))
                     and attrs.get("visibility") == VISIBLE
                     and _bbox_inside(ann.get("amodal_bbox"), roi)
                     and is_shape and label_px == area)
        ann.update({
            "instance_id": iid,
            "visibility": attrs.get("visibility"),
            "occlusion_ratio": attrs.get("occlusion_ratio"),
            "occluders": [{"id": i, "px": n} for i, n in occluders],
            "occluded_unlabelled_px": int(found[0]) if len(found) else 0,
            "shared_px": int(np.count_nonzero(claims[seen] > 1)),
            "label_px": label_px,
            "amodal_is_shape": is_shape,
            "clean": clean,
        })


def write_label_png(path: Path, label: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", label, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise PackError(f"could not encode {path}")
    path.write_bytes(buf.tobytes())


# --------------------------------------------------------------------------- #
# one desktop
# --------------------------------------------------------------------------- #
def instance_ids(db: Db, desktop: int) -> dict[str, int]:
    """``key -> id``, 1..N in key order over every real instance (drafts left out)."""
    keys = sorted(k for k in db.instances(desktop) if not is_provisional(k))
    if len(keys) >= np.iinfo(np.uint16).max:
        raise PackError(f"desktop {desktop}: {len(keys)} instances do not fit a 16-bit map")
    return {k: i for i, k in enumerate(keys, start=1)}


def _class_line(rec) -> str:
    text = CLASS_EN.get(rec.cls, rec.cls)
    detail = ", ".join(f"{k}={rec.attrs[k]}" for k in ("of", "kind", "role", "name", "label")
                       if (rec.attrs or {}).get(k) not in (None, ""))
    return f"{text} [{detail}]" if detail else text


def _runs(steps: list[int], states: dict[int, Optional[tuple[str, str]]]) -> list[dict]:
    out: list[dict] = []
    for step in steps:
        value = states.get(step)
        if value is None:
            continue
        if out and out[-1]["to"] == steps[steps.index(step) - 1] \
                and (out[-1]["state"], out[-1]["placement"]) == value:
            out[-1]["to"] = step
        else:
            out.append({"from": step, "to": step, "state": value[0], "placement": value[1]})
    return out


def _leaving(desktop_actions, leaving, frame_steps: list[int], seg_of) -> dict[str, dict]:
    """``key -> {step r, kind, verb, left_with, before/after frame}`` (U28's timeline)."""
    verbs: dict[tuple[int, str], str] = {}
    for action in sorted(desktop_actions, key=lambda a: (a.step, a.idx)):
        if action.result == "success":
            verbs.setdefault((int(action.step), str(action.target)), str(action.verb))
    out = {}
    for key, found in leaving.items():
        r = int(found.step)
        verb = verbs.get((r, key))
        if verb is None and found.left_with:
            verb = verbs.get((r, str(found.left_with)))
        before = max((s for s in frame_steps if s < r), default=None)
        after = min((s for s in frame_steps if s >= r), default=None)
        same_pose = (before is not None and after is not None
                     and seg_of(before) == seg_of(after))
        out[key] = {"step": r, "kind": found.kind, "old": found.old, "new": found.new,
                    "verb": verb, "left_with": found.left_with,
                    "before_frame": before, "after_frame": after,
                    "same_pose_segment": same_pose}
    return out


def build_desktop(db: Db, tax: Taxonomy, truth: TruthService, desktop: int, out: Path,
                  work: Path, cache_dir: Optional[str], resolve: Callable, tail: Callable,
                  qa: dict) -> dict:
    """Write ``out/D<nn>/`` for one desktop; returns its counts."""
    ddir = out / desktop_dir(desktop)
    ctx = load_ctx(db, tax, desktop, VIEW)
    key_to_id = instance_ids(db, desktop)
    all_insts = {k: r for k, r in db.instances(desktop).items() if k in key_to_id}
    cat_of = category_ids(tax)

    try:
        # default flags: confirmed rows only (after the re-check queue is drained,
        # on the copy), full frame, amodal shapes, covered parts as crowd rows;
        # an open conflict refuses -- a disputed row is not a confirmed one
        doc = export_coco(db, tax, [desktop], VIEW, str(work / f"coco_{desktop:02d}.json"),
                          truth=truth)
    except RuntimeError as exc:
        raise PackError(f"D{desktop:02d}: {exc}") from None
    if doc["info"]["version"] != EXPORT_VERSION:
        raise PackError(f"export version {doc['info']['version']} != {EXPORT_VERSION}")
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in doc["annotations"]:
        by_image[ann["image_id"]].append(ann)
    images = {img["extra"]["step"]: img for img in doc["images"]}

    chosen = load_cache_manifest(cache_dir, desktop)
    frames = db.frames_for(desktop, VIEW)
    frame_steps = [int(f["step"]) for f in frames]
    stale = sorted(int(s) for s in chosen if int(s) not in set(frame_steps))
    if stale:
        qa["manifest_steps_without_frame"].append({"desktop": desktop, "steps": stale})
    steps_tab = ctx.steps
    actions = db.actions(desktop)
    acts_at: dict[int, list] = defaultdict(list)
    for a in actions:
        acts_at[int(a.step)].append(a)

    def action_rows(step: int) -> list[dict]:
        return [{"idx": a.idx, "target": a.target, "target_id": key_to_id.get(a.target),
                 "verb": a.verb, "tool": a.tool, "direction": a.direction,
                 "result": a.result} for a in sorted(acts_at.get(step, []), key=lambda a: a.idx)]

    def seg_of(step: int):
        return db.pose_segment_for(FrameKey(desktop, step, VIEW)).get("seg")

    truth_rows = truth_identical = 0
    truth_max_px = 0
    frames_out: dict[str, dict] = {}
    counts = Counter()
    per_inst = defaultdict(Counter)
    for frame in frames:
        step = int(frame["step"])
        key = FrameKey(desktop, step, VIEW)
        name = frame_name(step)
        roi = roi_of(db, key)
        img = images.get(step)
        source = _source_record(frame, chosen.get(str(step)))
        folder, file_name = _source_folder(source["recorded"], tail)
        entry = {
            "step": step, "exported": img is not None,
            "image": None, "label": None,
            "source_folder": folder, "file_name": file_name,
            "source_recorded_path": source["recorded"],
            "burst_choice": source["choice"],
            "frame_row_path": source["frame_row_path"],
            "sha256": None, "bytes": None, "width": None, "height": None,
            "cache_identical": None, "cache_pixels_identical": None,
            "roi": None if roi is None else list(roi),
            "pose_segment": seg_of(step),
            "step_name": (steps_tab[step].raw_name if step in steps_tab else None),
            "step_type": ctx.step_type(step),
            "actions": action_rows(step),
            "review_status": frame.get("review_status"),
            "label_sha256": None, "annotations": 0, "clean": 0, "crowd": 0,
        }
        if source["frame_row_path"]:
            qa["frame_row_names_other_image"].append(
                {"desktop": desktop, "step": step, "frame_row": source["frame_row_path"],
                 "chosen": source["recorded"], "reason": source["choice"]["reason"]})
        frames_out[str(step)] = entry
        if img is None:
            continue
        # -- the picture: copied from the raw drive, compared with the cache's
        raw = resolve(source["recorded"])
        if not raw or not Path(raw).is_file():
            raise PackError(f"D{desktop:02d} {name}: the source picture {source['recorded']} "
                            f"is not readable (resolved to {raw}); is the raw drive connected?")
        dest = ddir / "images" / f"{name}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw, dest)
        digest = sha256_of(dest)
        if digest != sha256_of(raw):
            raise PackError(f"{dest}: the copy does not match {raw}")
        size = png_size(dest)
        if size is None:
            raise PackError(f"{raw} is not a PNG")
        hw = (int(img["height"]), int(img["width"]))
        if (size[1], size[0]) != hw:
            raise PackError(f"D{desktop:02d} {name}: picture is {size[0]}x{size[1]}, "
                            f"masks are {hw[1]}x{hw[0]}")
        cached = (Path(cache_dir) / VIEW / desktop_dir(desktop) / f"s{step:03d}.png"
                  if cache_dir else None)
        if cached is not None and cached.is_file():
            entry["cache_identical"] = sha256_of(cached) == digest
            entry["cache_pixels_identical"] = (True if entry["cache_identical"]
                                               else _pixels_equal(str(cached), str(dest)))
            if not entry["cache_identical"]:
                qa["cache_differs_from_raw"].append(
                    {"desktop": desktop, "step": step,
                     "pixels_identical": entry["cache_pixels_identical"]})
        entry.update({"image": f"images/{name}.png", "sha256": digest,
                      "bytes": dest.stat().st_size, "width": size[0], "height": size[1]})

        # -- the confirmed rows against a fresh compilation (the app's rule)
        stored = {k: r for k, r in db.compiled(key).items()}
        compiled = truth.compile(key)
        if set(stored) != set(compiled.instances):
            raise PackError(
                f"D{desktop:02d} {name}: confirmed rows and today's compilation name other "
                f"parts (only confirmed: {sorted(set(stored) - set(compiled.instances))}, "
                f"only compiled: {sorted(set(compiled.instances) - set(stored))})")
        unverified = sorted(k for k, r in stored.items() if r.get("status") != VERIFIED)
        if unverified:
            qa["unconfirmed_rows_left_out"].append(
                {"desktop": desktop, "step": step, "instances": unverified})
        problems = frozen_disagreements(db, key, compiled, stored=stored)
        if problems:
            raise PackError(f"D{desktop:02d} {name}: confirmed rows disagree with the inputs: "
                            + "; ".join(p.text for p in problems))
        for inst, row in stored.items():
            if row.get("status") != VERIFIED:
                continue
            truth_rows += 1
            fresh = compiled.instances[inst].visible
            old = _decode(row.get("visible_rle"), hw)
            new = np.zeros(hw, bool) if fresh is None else np.asarray(fresh, bool)
            diff = int(np.count_nonzero(old ^ new))
            if diff == 0 and row.get("visibility") == compiled.instances[inst].visibility:
                truth_identical += 1
            truth_max_px = max(truth_max_px, diff)

        # -- the label map and the per-annotation fields
        anns = sorted(by_image.get(img["id"], []), key=lambda a: a["id"])
        for ann in anns:
            if ann["attributes"]["instance_key"] not in key_to_id:
                raise PackError(f"annotation of unknown instance "
                                f"{ann['attributes']['instance_key']}")
        written = {a["attributes"]["instance_key"] for a in anns}
        for inst in sorted(stored):
            row = stored[inst]
            if row.get("status") == VERIFIED and inst not in written:
                qa["confirmed_rows_without_annotation"].append(
                    {"desktop": desktop, "step": step, "instance": inst,
                     "visibility": row.get("visibility"), "geom_type": row.get("geom_type"),
                     "placement": row.get("placement")})
        patched = {k for k, fo in db.frame_overrides(key).items() if fo.visible_rle}
        label, visible = frame_label_map(anns, hw, key_to_id, compiled, patched)
        annotate_frame(anns, label, visible, hw, key_to_id, roi, compiled)
        lpath = ddir / "labels" / f"{name}.png"
        write_label_png(lpath, label)
        entry.update({"label": f"labels/{name}.png", "label_sha256": sha256_of(lpath)})
        img["extra"]["cache_file_name"] = img["file_name"]
        img["extra"]["roi"] = entry["roi"]
        img["file_name"] = entry["image"]
        img["label_file"] = entry["label"]
        for ann in anns:
            attrs = ann["attributes"]
            k = attrs["instance_key"]
            entry["annotations"] += 1
            entry["clean"] += int(ann["clean"])
            entry["crowd"] += int(bool(ann["iscrowd"]))
            per_inst[k]["annotations"] += 1
            per_inst[k]["seen"] += int(ann["area"] > 0)
            per_inst[k]["crowd"] += int(bool(ann["iscrowd"]))
            per_inst[k]["clean"] += int(ann["clean"])
            amodal_area = int(ann.get("amodal_area") or 0)
            if ann["iscrowd"] and amodal_area and \
                    ann["occluded_unlabelled_px"] >= UNLABELLED_SHARE * amodal_area:
                qa["empty_but_nothing_labelled_covers_it"].append(
                    {"desktop": desktop, "step": step, "instance": k,
                     "amodal_area": amodal_area,
                     "unlabelled_px": ann["occluded_unlabelled_px"],
                     # 「被线挡住」/「角度看不到」: the annotator said why
                     "stated_reason": attrs.get("occluder")})
            if attrs.get("visibility") == "occluded_full" and ann["area"] > 0:
                qa["occluded_full_label_with_pixels"].append(
                    {"desktop": desktop, "step": step, "instance": k, "area": ann["area"]})
            if not ann["amodal_is_shape"] and ann["area"] > 0:
                qa["amodal_widened_by_visible"].append(
                    {"desktop": desktop, "step": step, "instance": k})
            if ann["shared_px"]:
                counts["shared_px"] += ann["shared_px"]
        if any(a["shared_px"] for a in anns):
            pairs = []
            keys = [a["attributes"]["instance_key"] for a in anns if a["shared_px"]]
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    n = int(np.count_nonzero(visible[a] & visible[b]))
                    if n:
                        pairs.append({"a": a, "b": b, "px": n})
            qa["overlapping_confirmed_masks"].append(
                {"desktop": desktop, "step": step, "pairs": pairs})
        counts["frames"] += 1

    # -- the timeline: states per step, the leaving step, flicker
    steps_all = sorted(set(steps_tab) | set(frame_steps))
    states = {s: ctx.state_at(s) for s in steps_all}
    leaving = _leaving(actions, leaving_steps(ctx.instances, ctx.events, tax),
                       sorted(int(s) for s in images), seg_of)
    _flicker(doc, desktop, states, qa)
    _hidden_before_leaving(doc, desktop, leaving, qa)

    instances_out: dict[str, dict] = {}
    timeline: dict[str, dict] = {}
    for k, iid in sorted(key_to_id.items(), key=lambda t: t[1]):
        rec = all_insts[k]
        attrs = dict(rec.attrs or {})
        context = is_context(rec)
        per_step = None if context else {
            str(s): (None if states[s].get(k) is None else
                     {"state": states[s][k].state, "placement": states[s][k].placement})
            for s in steps_all}
        lv = leaving.get(k)
        if lv is not None:
            lv = dict(lv, left_with_id=key_to_id.get(lv["left_with"]) if lv["left_with"] else None)
        instances_out[str(iid)] = {
            "id": iid, "key": k, "class": rec.cls,
            "category_id": cat_of.get(rec.cls),
            "class_description": _class_line(rec),
            "class_group": (tax.classes.get(rec.cls) or {}).get("group"),
            "attrs": attrs,
            "parent": rec.parent, "parent_id": key_to_id.get(rec.parent) if rec.parent else None,
            "attached": bool(rec.attached),
            "mounted_on": rec.mounted_on, "fastens": rec.fastens,
            "socket_host": rec.socket_host, "cable": rec.cable,
            "captive": attrs.get("captive"),
            "context": context, "extra": is_extra(rec), "implied": is_implied(rec),
            "implied_rule": attrs.get(IMPLIED_RULE_ATTR),
            "raw_names": list(rec.raw_names or []),
            "leaving": lv,
            "frames": {f: int(per_inst[k][f]) for f in ("annotations", "seen", "crowd", "clean")},
            "states": per_step,
        }
        if not context:
            timeline[str(iid)] = {"key": k, "runs": _runs(steps_all, {
                s: (None if states[s].get(k) is None else
                    (states[s][k].state, states[s][k].placement)) for s in steps_all})}

    steps_out = []
    for s in steps_all:
        rows = action_rows(s)
        ok = [r for r in rows if r["result"] == "success"]
        steps_out.append({
            "step": s, "name": steps_tab[s].raw_name if s in steps_tab else None,
            "step_type": ctx.step_type(s),
            "frame": frame_name(s) if s in images else None,
            "verbs": sorted({r["verb"] for r in ok}),
            "targets": [r["target"] for r in ok],
            "target_ids": [r["target_id"] for r in ok],
            "actions": rows,
        })
    leaving_order = sorted(
        ({"step": v["step"], "id": key_to_id[k], "key": k, "kind": v["kind"],
          "verb": v["verb"], "left_with": v["left_with"]} for k, v in leaving.items()),
        key=lambda r: (r["step"], r["key"]))

    meta = db.get_desktop(desktop) or {}
    head = {"desktop": desktop, "brand": meta.get("brand"), "model": meta.get("model_family"),
            "view": VIEW}
    doc["info"]["scanpack"] = dict(head, pack_version=PACK_VERSION,
                                   fields=list(ANN_FIELDS),
                                   label_maps="labels/S<nn>.png, uint16, 0 = background")
    write_json(ddir / "annotations.json", doc)
    write_json(ddir / "frames.json", dict(head, frames=frames_out))
    write_json(ddir / "instances.json", dict(head, steps=steps_all, instances=instances_out))
    write_json(ddir / "sequence.json", dict(head, steps=steps_out, leaving=leaving_order,
                                            timeline=timeline))

    clean_rows = [
        {"desktop": desktop, "step": img_step(a), "instance_id": a["instance_id"],
         "class": instances_out[str(a["instance_id"])]["class"], "area": int(a["area"]),
         "instance_key": a["attributes"]["instance_key"]}
        for a in doc["annotations"] if a.get("clean")]
    return {
        "desktop": desktop, "brand": head["brand"], "model": head["model"],
        "frames": len(images), "frames_in_view": len(frames),
        "instances": len(key_to_id),
        "instances_annotated": sum(1 for k in key_to_id if per_inst[k]["annotations"]),
        "annotations": len(doc["annotations"]),
        "crowd": sum(1 for a in doc["annotations"] if a["iscrowd"]),
        "clean": len(clean_rows),
        "clean_instances": len({r["instance_id"] for r in clean_rows}),
        "amodal_union_rows": int(doc["info"].get("amodal_union_rows") or 0),
        "shared_px": int(counts["shared_px"]),
        "truth_check": {"rows": truth_rows, "identical_to_fresh_compile": truth_identical,
                        "within_app_tolerance": truth_rows - truth_identical,
                        "max_px_diff": truth_max_px},
        "_clean_rows": clean_rows,
    }


def img_step(ann: dict) -> int:
    """The step of an annotation, read off its stable image id (see coco.image_id)."""
    return (int(ann["image_id"]) % 100_000) // 10


def _flicker(doc: dict, desktop: int, states: dict, qa: dict) -> None:
    """A part seen on the frames either side but not on this one, its state unchanged."""
    area: dict[str, dict[int, int]] = defaultdict(dict)
    for ann in doc["annotations"]:
        area[ann["attributes"]["instance_key"]][img_step(ann)] = int(ann["area"])
    steps = sorted({img_step(a) for a in doc["annotations"]})
    for key, seen in area.items():
        for i in range(1, len(steps) - 1):
            a, b, c = steps[i - 1], steps[i], steps[i + 1]
            if seen.get(b, None) == 0 and seen.get(a, 0) > 0 and seen.get(c, 0) > 0:
                same = all(states.get(s, {}).get(key) == states.get(b, {}).get(key)
                           for s in (a, c))
                if same:
                    qa["empty_between_seen_frames"].append(
                        {"desktop": desktop, "step": b, "instance": key,
                         "area_before": seen[a], "area_after": seen[c]})


def _hidden_before_leaving(doc: dict, desktop: int, leaving: dict, qa: dict) -> None:
    """A part removed at ``r`` whose frame ``r - 1`` shows nothing of it."""
    area = {(a["attributes"]["instance_key"], img_step(a)): int(a["area"])
            for a in doc["annotations"]}
    for key, lv in leaving.items():
        if lv["kind"] != "removed" or lv["before_frame"] is None:
            continue
        seen = area.get((key, lv["before_frame"]))
        if seen == 0:
            qa["hidden_on_the_frame_before_leaving"].append(
                {"desktop": desktop, "instance": key, "r": lv["step"],
                 "left_with": lv["left_with"]})


# --------------------------------------------------------------------------- #
# the package
# --------------------------------------------------------------------------- #
QA_KEYS = ("overlapping_confirmed_masks", "occluded_full_label_with_pixels",
           "empty_but_nothing_labelled_covers_it", "empty_between_seen_frames",
           "hidden_on_the_frame_before_leaving", "amodal_widened_by_visible",
           "confirmed_rows_without_annotation", "frame_row_names_other_image",
           "cache_differs_from_raw", "manifest_steps_without_frame",
           "unconfirmed_rows_left_out")


def _prepare_out(out: Path, force: bool) -> None:
    if out.exists() and any(out.iterdir()):
        marker = out / "manifest.json"
        is_pack = marker.is_file() and '"scanpack"' in marker.read_text(encoding="utf-8")[:400]
        if not force:
            raise PackError(f"{out} is not empty; pass --force to replace a package there")
        if not is_pack:
            raise PackError(f"{out} is not empty and holds no scanpack manifest; "
                            f"refusing to delete it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def _work_copy(db_path: str, work: Path) -> Path:
    """A backup-API copy of the input database, opened read-only."""
    work.mkdir(parents=True, exist_ok=True)
    dest = work / "work.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(str(dest) + suffix).unlink(missing_ok=True)
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    src = sqlite3.connect(uri, uri=True)
    try:
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return dest


def build_pack(db_path: str, out_dir: str, desktops: Sequence[int] = DEFAULT_DESKTOPS, *,
               cache_dir: Optional[str] = None, work_dir: Optional[str] = None,
               resolve: Optional[Callable[[str], Optional[str]]] = None,
               tail: Optional[Callable[[str], Optional[str]]] = None,
               tax: Optional[Taxonomy] = None, force: bool = False) -> dict:
    """Write the package; returns the summary (counts per desktop, the QA lists).

    ``resolve`` turns a recorded raw path into the file to read today and
    ``tail`` into its part below the dataset root (both default to the identity
    / ``None``: :mod:`tda.core.rawroot` is the CLI's business). ``work_dir``
    receives the database copy, the plain COCO exports and ``qa.json``.
    """
    out = Path(out_dir)
    work = Path(work_dir) if work_dir else REPO / ".cache" / "scanpack_work"
    resolve = resolve or (lambda p: p)
    tail = tail or (lambda p: None)
    tax = tax or load_taxonomy()
    _prepare_out(out, force)
    db = Db(str(_work_copy(db_path, work)))
    qa: dict[str, list] = {k: [] for k in QA_KEYS}
    try:
        truth = TruthService(db, tax, cache_dir=cache_dir)
        per_desktop = []
        for desktop in [int(d) for d in desktops]:
            if db.get_desktop(desktop) is None:
                raise PackError(f"desktop {desktop} is not in {db_path}")
            per_desktop.append(build_desktop(db, tax, truth, desktop, out, work, cache_dir,
                                             resolve, tail, qa))
    finally:
        db.close()
    clean_rows = [r for d in per_desktop for r in d.pop("_clean_rows")]
    _write_clean_csv(out / "clean_parts.csv", clean_rows)
    by_class = Counter(r["class"] for r in clean_rows)
    summary = {
        "package": "scanpack", "pack_version": PACK_VERSION, "view": VIEW,
        "coco_export_version": EXPORT_VERSION,
        "desktops": [d["desktop"] for d in per_desktop],
        "totals": {k: sum(d[k] for d in per_desktop)
                   for k in ("frames", "instances", "annotations", "crowd", "clean",
                             "amodal_union_rows", "shared_px")},
        "per_desktop": per_desktop,
        "clean_by_class": dict(sorted(by_class.items())),
        "clean_by_desktop_and_class": {
            desktop_dir(d["desktop"]): dict(sorted(Counter(
                r["class"] for r in clean_rows if r["desktop"] == d["desktop"]).items()))
            for d in per_desktop},
    }
    (out / "LICENSE_NOTE.txt").write_text(LICENSE_NOTE, encoding="utf-8", newline="\n")
    (out / "README.md").write_text(readme_text(summary), encoding="utf-8", newline="\n")
    files = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = path.relative_to(out).as_posix()
        if rel == "manifest.json":
            continue
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256_of(path)}
    write_json(out / "manifest.json", dict(summary, files=files))
    write_json(work / "qa.json", qa)
    return dict(summary, qa=qa)


def _write_clean_csv(path: Path, rows: list[dict]) -> None:
    fields = ("desktop", "step", "instance_id", "class", "area", "instance_key")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["desktop"], r["step"], r["instance_id"])):
            writer.writerow(row)


LICENSE_NOTE = """\
Internal test data -- not for redistribution.

This package (scanner pictures, masks, annotations) is shared with our research
collaborators for an internal test of point-cloud extraction only. Do not
redistribute it, publish it, upload it to a public service or use it to train
models for release without written permission from the dataset owners. The
annotations are a work in progress and may change.
"""


# --------------------------------------------------------------------------- #
# README
# --------------------------------------------------------------------------- #
README_EXAMPLE = '''\
import json
import numpy as np
from PIL import Image                        # pip install pillow
from pycocotools import mask as mask_utils   # pip install pycocotools

desk = "D02"                                 # any desktop folder of the package
coco = json.load(open(f"{desk}/annotations.json"))
inst = json.load(open(f"{desk}/instances.json"))["instances"]
ann = next(a for a in coco["annotations"] if a["clean"])         # a clean part
img = next(i for i in coco["images"] if i["id"] == ann["image_id"])
labels = np.array(Image.open(f"{desk}/{img['label_file']}"))     # uint16 ids
iid = ann["instance_id"]
ys, xs = np.nonzero(labels == iid)           # its pixels: x = column, y = row
print(img["file_name"], inst[str(iid)]["key"], inst[str(iid)]["class"], len(xs), "px")
print("first pixels (x, y):", list(zip(xs[:5].tolist(), ys[:5].tolist())))
visible = mask_utils.decode(ann["segmentation"]).astype(bool)
amodal = mask_utils.decode(ann["amodal_segmentation"]).astype(bool)
print("label map == visible mask:", bool(np.array_equal(visible, labels == iid)))
print("hidden pixels (amodal minus visible):", int((amodal & ~visible).sum()))
'''


def readme_text(summary: dict) -> str:
    t = summary["totals"]
    rows = "\n".join(
        f"| {desktop_dir(d['desktop'])} | {d['brand']} {d['model']} | {d['frames']} | "
        f"{d['instances_annotated']} | {d['annotations']} | {d['crowd']} | {d['clean']} | "
        f"{d['clean_instances']} |"
        for d in summary["per_desktop"])
    classes = ", ".join(f"{k} {v}" for k, v in sorted(summary["clean_by_class"].items(),
                                                       key=lambda kv: (-kv[1], kv[0])))
    desks = ", ".join(desktop_dir(d) for d in summary["desktops"])
    return f"""# Scanner-view segmentation package (scanpack v{summary['pack_version']})

## 中文摘要

本包是 {len(summary['desktops'])} 台台式机（{desks}）结构光扫描仪俯视 RGB 帧（1600×1600）的二维分割标注，
供 3D 合作方测试"用 2D 掩码从重建点云里取出每个零件的点"。每台机器一个文件夹 `D<nn>/`：
`images/S<nn>.png` 是标注时画掩码所用的那张原始扫描图（从原始盘逐字节拷贝），
`labels/S<nn>.png` 是 16 位实例 id 图（0 = 背景，id 见 `instances.json`），
`annotations.json` 是 COCO 格式（可见掩码 + 完整形状 amodal 掩码 + 遮挡信息），
`frames.json` / `sequence.json` 给出每一步的名称、动作和每个零件的状态时间线。
坐标就是图像像素 (x, y)，原点在左上角。建议先用 `clean_parts.csv`（共 {t['clean']} 条）：
这些零件在那一帧上没有被任何东西挡住（可见掩码与完整形状逐像素相同、标签为 visible、在 ROI 内），
掩码下的像素就是它的全部可见表面。被遮挡的零件有两条路：(1) 只取可见掩码下的点；
(2) 用拆卸顺序：零件在第 r 步离开，取第 r-1 帧重建里有、第 r 帧重建里没有、且落在它 amodal 掩码内的点。
amodal 掩码是完整轮廓，不是点集——被挡住的表面扫描仪看不到。内部测试数据，请勿外传（见 `LICENSE_NOTE.txt`）。

## What this is

Two-dimensional instance segmentation of the top-down RGB frames of the
structured-light scanner, for {len(summary['desktops'])} desktop computers being
taken apart step by step. Every frame was annotated and then confirmed by a
human annotator; the masks here are exactly the confirmed ones. The goal of the
package is to test whether these masks let you select each annotated part's
points in your 3D reconstruction of the same scanner frame.

| Desktop | Model | Frames | Parts | Annotations | Fully covered (crowd) | Clean | Clean parts |
|---|---|---|---|---|---|---|---|
{rows}
| **Total** | | {t['frames']} | | {t['annotations']} | {t['crowd']} | {t['clean']} | |

*Parts* counts the instances with at least one annotation; *Clean* counts
(frame, part) annotations, *Clean parts* the distinct parts among them.
Clean annotations by class: {classes}.

## Layout

```
README.md            this file
LICENSE_NOTE.txt     internal test data, not for redistribution
manifest.json        counts, and size + sha256 of every other file
clean_parts.csv      desktop, step, instance_id, class, area, instance_key
D<nn>/
  images/S<nn>.png   the scanner picture of step <nn> (original file, byte for byte)
  labels/S<nn>.png   16-bit instance-id map of the visible masks (0 = background)
  annotations.json   COCO instance segmentation, one annotation per (frame, part)
  frames.json        per step: source file, sha256, size, ROI, step name, actions
  instances.json     per part: id, key, class, relations, states per step, leaving step
  sequence.json      the step table and the state timeline per part
```

`S<nn>` is the step number. A step's picture is taken **after** that step's
action: on `S05` the part removed at step 5 is already gone.

## Coordinate frame

Everything is in image pixels of the picture `frames.json` names: `(x, y)` with
the origin at the top-left corner, x to the right (column), y down (row),
1600 x 1600 for every frame here. Masks, boxes (`[x, y, w, h]`), the label maps
and the ROI all use this frame, unshifted and uncropped. `images/S<nn>.png` is
the very file the masks were drawn on: `frames.json` gives its original folder
and file name on our scanner drive (`source_folder`, `file_name`; the scanner
writes a burst `P_0..P_9` per step and one was chosen -- `burst_choice`), its
sha256 and its size, so you can match it with the scanner frame your
reconstruction was built from.

## Selecting points

If your reconstruction keeps, for each 3D point, the pixel of this image it
came from (an organized point cloud, or a depth map registered to this RGB
image), then **the points under a part's visible mask are that part's points**:
read the label map at the pixel (`labels/S<nn>.png == instance_id`) or decode
the annotation's `segmentation`. The scanner only measures surfaces it can see,
so a hidden surface has no points: the `amodal_segmentation` is the part's full
silhouette on the image (what it would cover if nothing were in front), not a
set of points.

### Start with the clean set

`clean_parts.csv` lists the annotations with `clean: true` ({t['clean']} rows): the
part is not covered by anything on that frame, so its visible mask *is* its
whole outline and the points under it are all of the part the scanner saw.
Precisely, `clean` is true when

* the visible mask equals the amodal mask to the pixel (0 px difference),
* its label (`visibility`) is `visible`,
* the amodal box lies inside the frame's ROI,
* the amodal shape is the annotated shape itself (`amodal_is_shape`), and
* the label map gives the part every pixel of its visible mask (`label_px == area`).

### Occluded parts: two routes

1. **Visible mask.** Take the points under the visible mask only. This is what
   the scanner could see of the part on that frame; it is the amodal shape
   minus the parts above it (`occlusion_ratio` says how much is missing).
2. **Disassembly order.** `instances.json` gives each part's leaving step `r`
   (`leaving.step`, with the verb and, for a part that rode out inside another
   one, `left_with`). The part is present on frame `r-1` (`leaving.before_frame`)
   and gone on frame `r` (`leaving.after_frame`). Its points are the ones that
   are in the reconstruction of `r-1` and not in the reconstruction of `r`,
   restricted to its amodal mask on frame `r-1`. Other things can change at
   step `r` too (a part that leaves with it, a cable moved aside), so check the
   other parts leaving at `r` in `sequence.json` (`leaving`). `kind` is
   `removed` for a part taken out, `displaced` for one moved aside but still in
   the picture, and `changed` for a part that never leaves but changes state
   (a connector unplugged, a latch opened).

### `occlusion_ratio` and `occluders`

`occlusion_ratio = 1 - area / amodal_area`: the share of the part's silhouette
hidden on that frame. `occluders` lists who takes the hidden pixels, read off
the label map: `[{{"id": instance_id, "px": pixels}}, ...]`, largest first.
`occluded_unlabelled_px` is the rest -- hidden pixels no labelled part shows
(a cable, a part nobody annotates, a hand). The three add up:
`sum(px) + occluded_unlabelled_px == amodal_area - area`.

### The ROI

`frames.json` gives each frame's `roi` `[x0, y0, x1, y1]` (x1, y1 exclusive):
the rectangle around the chassis that annotation concentrated on (it is the
same for every frame of a desktop here). Parts outside it -- set down on the
bench -- are rare; a part whose amodal box leaves the ROI is never `clean`.

### Fully covered parts (crowd rows)

A part that is in its place but completely covered on a frame is still an
annotation: an empty `segmentation`, `area` 0, `bbox` = its amodal box,
`iscrowd: 1`, and its `amodal_segmentation`. It has no points on that frame;
use route 2 (or another frame) for it.

### Where two confirmed masks overlap

A few confirmed visible masks claim the same pixels ({t['shared_px']} pixel claims in the
whole package): rows confirmed at different times, or a mask corrected on one
frame only. The label map gives such a pixel to the part higher in the frame's
layer order (a part corrected on that frame wins). Each annotation's
`shared_px` says how many of its visible pixels another mask also claims and
`label_px` how many the label map gives it; `segmentation` itself is kept as it
was confirmed.

## Files in detail

* `annotations.json` -- COCO (`images`, `annotations`, `categories`) as our
  annotation tool exports it (version {summary['coco_export_version']}): `segmentation` (visible mask) and
  `amodal_segmentation` are compressed RLE (`pycocotools.mask.decode`), plus
  `amodal_bbox`, `amodal_area` and an `attributes` block (`instance_key`,
  `state`, `placement`, `visibility`, `verified`, ...). The package adds at the
  top of each annotation: `instance_id`, `visibility`, `occlusion_ratio`,
  `occluders`, `occluded_unlabelled_px`, `shared_px`, `label_px`,
  `amodal_is_shape`, `clean`. Each image has `file_name` (`images/S<nn>.png`)
  and `label_file` (`labels/S<nn>.png`). `visibility` is one of `visible`,
  `occluded_partial`, `occluded_full`, `visible_tiny` (a human may label a part
  `occluded_full` with a sliver still showing).
* `labels/S<nn>.png` -- single-channel 16-bit PNG, value = `instance_id`
  (0 = background). Read it with Pillow (`np.array(Image.open(p))`) or OpenCV
  (`cv2.imread(p, cv2.IMREAD_UNCHANGED)`).
* `instances.json` -- `instances[id]`: `key` (our stable name, e.g.
  `screw.cpu_cooler.03`), `class` and `class_description`, `category_id` (the
  COCO category), `attrs`, `parent` / `attached` (an attached part leaves with
  its parent), `mounted_on`, `fastens`, `socket_host`, `captive` (a captive
  screw stays on its part), `extra` (added by the annotator, not named in the
  log), `implied` (assumed present, e.g. the motherboard of a machine whose log
  never names it), `leaving` (see route 2), `frames` (annotation counts) and
  `states` (state and placement at every step). ids are this package's;
  `key` is the identity to use across versions.
* `sequence.json` -- `steps`: step, name (as written in the teardown log),
  step type, verbs, targets and the actions; `leaving`: every part's leaving
  step in order; `timeline`: per part, runs of `(from, to, state, placement)`.
* `frames.json` -- per step: `image`, `label`, `source_folder`, `file_name`,
  `sha256`, `width`, `height`, `roi`, `step_name`, `step_type`, `actions`,
  `review_status`, annotation counts.
* `manifest.json` -- the counts above per desktop and per class, and the size
  and sha256 of every other file.

## Python example

Run from the package root:

```python
{README_EXAMPLE}```
"""


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def _parse_desktops(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", required=True,
                        help="the database to read (a copy; it is only ever opened read-only)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="the package folder")
    parser.add_argument("--desktops", default=",".join(str(d) for d in DEFAULT_DESKTOPS),
                        help="e.g. 2,3,4,13 or 20-25")
    parser.add_argument("--paths", default=str(REPO / "configs" / "paths.yaml"),
                        help="paths.yaml: the raw drive and the frame cache")
    parser.add_argument("--cache-dir", default=None, help="overrides paths.yaml's cache_dir")
    parser.add_argument("--work", default=str(REPO / ".cache" / "scanpack_work"),
                        help="scratch folder: the database copy, plain exports, qa.json")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing package in --out")
    args = parser.parse_args(argv)

    from tda.cli_common import load_paths
    from tda.core import rawroot

    paths = load_paths(args.paths)            # configures the raw-drive resolver
    raw = rawroot.current()
    print(raw.log_line if raw is not None else "raw root: not configured")
    cache_dir = args.cache_dir or paths.get("cache_dir")
    try:
        summary = build_pack(
            args.db, args.out, _parse_desktops(args.desktops), cache_dir=cache_dir,
            work_dir=args.work, resolve=rawroot.resolve_raw,
            tail=(raw.tail if raw is not None else None), force=args.force)
    except PackError as exc:
        print(f"[scanpack] refused: {exc}")
        return 1
    for d in summary["per_desktop"]:
        print(f"[scanpack] D{d['desktop']:02d}: {d['frames']} frames, {d['instances']} parts, "
              f"{d['annotations']} annotations, {d['crowd']} crowd, {d['clean']} clean, "
              f"truth {d['truth_check']}")
    print(f"[scanpack] clean by class: {summary['clean_by_class']}")
    for key in QA_KEYS:
        print(f"[scanpack] qa {key}: {len(summary['qa'][key])}")
    print(f"[scanpack] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
