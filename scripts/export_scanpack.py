"""The scanner-view segmentation package for the 3D collaborators (task U30-PACK).

::

    python scripts/export_scanpack.py --db <copy of tda.sqlite> \
        [--out D:/DataSet/exports/scanpack_2026-10-26] [--desktops 2,3,4,13,19,23,24,28,30,31] \
        [--exclude "D2:1,D4:1=why"] [--note "text"] \
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
(``manifest.json`` ``confirmed_vs_today``). The amodal shapes are today's (the export
derives them from the keyframes), and so is everything this script measures on
top; the README says which field is which.

**The label map.** Confirmed rows confirmed at different times, and a frame's
「只改这一帧」 mask, can claim the same pixel (the compiler never subtracts a
per-frame mask from the others). The map gives such a pixel to the part higher
in the frame's layer order: the compiler's painting order (``in_chassis``,
then ``on_bench``, then ``context``, each bottom-up), its pairwise overrides,
then the parts with a per-frame mask on top (by key among themselves). Each
annotation says how many of its visible pixels another one also claims
(``shared_px``) and how many the map gives it (``label_px``).

``clean`` is the set to start with: ``visible == amodal`` to the pixel, label
``visible``, the amodal box inside the pose segment's ROI, the amodal shape the
compiler's own (not widened by a visible mask reaching outside it), every
visible pixel the label map's, and no pixel shared with another confirmed mask
(round 2: three such rows were over-drawn latches).

``--exclude D2:1,D4:1=<reason>`` leaves listed frames out of the package (no
picture, label map or annotation; out of every count) and lists them under the
README's Known issues with the reason; ``--note`` adds a line there. Both are
recorded in ``manifest.json``.

Read-only on the database it is given (a ``mode=ro`` backup-API copy is made
into ``--work`` and only that copy is opened -- the export's re-check pass
writes to it), on the raw drive and on the frame cache. Deterministic: two runs
over the same inputs write the same bytes (no wall clock, no machine path in
any shipped file).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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
from tda.core.compiler_visibility import (  # noqa: E402
    OCCLUDED_FULL_RATIO,
    OCCLUDED_PARTIAL_RATIO,
    TOO_SMALL_PX,
    VISIBLE_TINY_PX,
)
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

__all__ = ["CLASS_EN", "DEFAULT_DESKTOPS", "MODEL_NAMES", "PACK_VERSION", "PackError",
           "build_pack", "main", "model_names", "parse_exclusions"]

#: The ten desktops of the first package: five Dell, five HP.
DEFAULT_DESKTOPS = (2, 3, 4, 13, 19, 23, 24, 28, 30, 31)
DEFAULT_OUT = "D:/DataSet/exports/scanpack_2026-10-26"
VIEW = "scan"
PACK_VERSION = "1"
VERIFIED = "verified"
VISIBLE = "visible"
OCCLUDED_FULL = "occluded_full"
#: Paint order of the compiler's placement groups in the label map (bottom first).
GROUP_RANK = {"in_chassis": 0, "on_bench": 1, "context": 2}
#: What the package adds to every COCO annotation, at its top level.
ANN_FIELDS = ("instance_id", "visibility", "occlusion_ratio", "occluders",
              "occluded_unlabelled_px", "shared_px", "label_px", "amodal_is_shape", "clean")
#: A crowd row whose amodal shape the label map leaves mostly empty: nothing
#: labelled covers it, yet nothing of it is seen (a QA note, not a rule).
UNLABELLED_SHARE = 0.5
#: A class whose share of clean annotations is below this is "rarely clean".
RARELY_CLEAN = 0.15
#: Instance attrs that are our bookkeeping (who added a part, when, how a flag
#: was guessed, a note to ourselves) rather than facts about the part.
DROP_ATTRS = ("added_by", "added_at", "captive_source", "reason", "note",
              "implied", IMPLIED_RULE_ATTR)
#: Reason given to an ``--exclude`` that names none.
DEFAULT_EXCLUDE_REASON = "left out of this package on request"

#: ``(brand, model as the teardown sheet spells it) -> (the model's real name,
#: a caveat or None)``. The sheet's text is kept beside it (``model_sheet``), as
#: written; the caveat goes to ``model_note``, never into the name.
NAMEPLATE = "to be verified on the nameplate: the teardown sheet says '{sheet}'"
MODEL_NAMES: dict[tuple[str, str], tuple[str, Optional[str]]] = {
    ("Dell", "Optiplex 7020"): ("Dell OptiPlex 7020", None),
    ("Dell", "Optiplex 7010"): ("Dell OptiPlex 7010", None),
    ("Dell", "Optiplex 3020"): ("Dell OptiPlex 3020", None),
    ("Dell", "Optiplex 990"): ("Dell OptiPlex 990", None),
    ("HP", "Compoq 8100 Elite Small Form Fador"): ("HP Compaq 8100 Elite SFF", None),
    ("HP", "Compoq 8300 Elite Small Form Fador"): ("HP Compaq 8300 Elite SFF", None),
    ("HP", "EliteDesk 800 GI SFF"): ("HP EliteDesk 800 G1 SFF", None),
    ("HP", "ProDesk 4300 GI SFF"): ("HP Compaq Pro 4300 SFF",
                                    NAMEPLATE.format(sheet="ProDesk 4300 GI SFF")),
    ("HP", "ProDesk 4300"): ("HP Compaq Pro 4300 SFF", NAMEPLATE.format(sheet="ProDesk 4300")),
}
#: What the README promises about every frame; the build checks each one and
#: stops when a frame breaks it (round 3, R2-N3).
EXPECTED_HW = (1600, 1600)
EXPECTED_TIER = "gold"
#: The one reason the README gives for a picture other than ``P_0``.
DARK_P0 = "p0_dark"
FIRST_IMAGE = "P_0.png"

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
    "cooler_latch": "What holds the CPU cooler in place: a latch, a clip, or (on some Dell "
                    "machines) a frame around the fan.",
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


def model_names(meta: dict) -> tuple[str, str, Optional[str]]:
    """``(model, model_sheet, model_note)``: the real name, the teardown sheet's
    text as is, and a caveat about the name (or ``None``)."""
    brand = str(meta.get("brand") or "").strip()
    family = str(meta.get("model_family") or "").strip()
    sheet = str(meta.get("brand_model_raw") or f"{brand} {family}").strip()
    name, note = MODEL_NAMES.get((brand, family), (f"{brand} {family}".strip(), None))
    return name, sheet, note


def parse_exclusions(specs: Sequence[str]) -> dict[tuple[int, int], str]:
    """``["D2:1,D4:1=why", ...]`` -> ``{(2, 1): "why", (4, 1): "why"}``."""
    out: dict[tuple[int, int], str] = {}
    for spec in specs or ():
        frames, _, reason = str(spec).partition("=")
        reason = reason.strip() or DEFAULT_EXCLUDE_REASON
        for item in frames.split(","):
            item = item.strip()
            if not item:
                continue
            found = re.fullmatch(r"[Dd]?(\d+)\s*:\s*[Ss]?(\d+)", item)
            if found is None:
                raise PackError(f"--exclude: {item!r} is not D<desktop>:<step>")
            out[(int(found.group(1)), int(found.group(2)))] = reason
    return out


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
        listed_ok = isinstance(index, int) and 0 <= index < len(metrics)
        if listed_ok and metrics[index].get("path") and metrics[index]["path"] != src:
            raise PackError(f"cache manifest names {src} but its metrics[{index}] "
                            f"is {metrics[index]['path']}")
        choice = {"by": "annotation_tool", "index": index,
                  "p_index": metrics[index].get("p_index") if listed_ok else None,
                  "reason": chosen.get("reason"), "burst_size": len(metrics) or None}
    else:
        src = row_path
        choice = {"by": "frame_record", "index": None, "p_index": None, "reason": None,
                  "burst_size": len(((frame.get("aux") or {}).get("burst")) or []) or None}
    return {"recorded": src, "choice": choice,
            "frame_row_path": row_path if row_path != src else None}


def _source_folder(recorded: str, tail: Callable[[str], Optional[str]]) -> tuple[str, str]:
    """``(folder below the dataset root, file name)`` of a recorded raw path.

    A path the dataset root does not hold would ship a machine's drive and
    folders in ``frames.json``: the build stops instead (R2-N6).
    """
    below = tail(recorded)
    if below is None:
        raise PackError(f"{recorded} is not below the raw dataset's root: its folder "
                        f"cannot be named without this machine's path")
    text = below.replace("\\", "/").lstrip("/")
    if re.match(r"^[A-Za-z]:", text):
        raise PackError(f"{recorded}: the folder below the dataset root still names a drive")
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


def masks_ratio(area: int, amodal_area: int) -> float:
    """``1 - area / amodal_area`` of the masks in the file, in ``[0, 1]``."""
    if amodal_area <= 0:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - area / amodal_area)))


def annotate_frame(anns: list[dict], label: np.ndarray, visible: dict[str, np.ndarray],
                   hw: tuple[int, int], key_to_id: dict[str, int], roi, compiled) -> Counter:
    """Add :data:`ANN_FIELDS` to each annotation of one frame, in place; frame counts.

    ``occlusion_ratio`` is measured on the masks the file carries; the
    confirmed value stays in ``attributes.occlusion_ratio`` (it is what the
    ``visibility`` label was decided from).
    """
    stats: Counter = Counter()
    claims = np.zeros(hw, dtype=np.uint8)
    for mask in visible.values():
        claims += mask
    stats["overlap_px"] = int(np.count_nonzero(claims > 1))
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
        shared = int(np.count_nonzero(claims[seen] > 1))
        area = int(np.count_nonzero(seen))
        whole = int(np.count_nonzero(amodal))
        ratio = masks_ratio(area, whole)
        confirmed = attrs.get("occlusion_ratio")
        rules = {
            "equal": area > 0 and not ann.get("iscrowd") and area == whole,
            "visible": attrs.get("visibility") == VISIBLE,
            "roi": _bbox_inside(ann.get("amodal_bbox"), roi),
            "shape": is_shape,
            "label": label_px == area,
            "shared": shared == 0,
        }
        clean = all(rules.values())
        for rule, held in rules.items():
            if not held and all(v for r, v in rules.items() if r != rule):
                stats[f"only_{rule}"] += 1          # clean but for this one rule
        stats["roi_outside"] += int(not rules["roi"])
        stats["ratio_differs"] += int(confirmed is None or abs(float(confirmed) - ratio) > 1e-9)
        ann.update({
            "instance_id": iid,
            "visibility": attrs.get("visibility"),
            "occlusion_ratio": ratio,
            "occluders": [{"id": i, "px": n} for i, n in occluders],
            "occluded_unlabelled_px": int(found[0]) if len(found) else 0,
            "shared_px": shared,
            "label_px": label_px,
            "amodal_is_shape": is_shape,
            "clean": clean,
        })
    return stats


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
    for index, step in enumerate(steps):
        value = states.get(step)
        if value is None:
            continue
        if out and index and out[-1]["to"] == steps[index - 1] \
                and (out[-1]["state"], out[-1]["placement"]) == value:
            out[-1]["to"] = step
        else:
            out.append({"from": step, "to": step, "state": value[0], "placement": value[1]})
    return out


def _leaving(desktop_actions, leaving, frame_steps: list[int], seg_of) -> dict[str, dict]:
    """``key -> {step r, kind, verb, left_with, before/after frame}`` (U28's timeline).

    ``frame_steps`` are the frames in the package: an excluded frame is never
    a part's before or after picture.
    """
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
        # unknown, not "a pose change", when one of the two frames is missing
        same_pose = (None if before is None or after is None
                     else seg_of(before) == seg_of(after))
        out[key] = {"step": r, "kind": found.kind, "old": found.old, "new": found.new,
                    "verb": verb, "left_with": found.left_with,
                    "before_frame": before, "after_frame": after,
                    "same_pose_segment": same_pose}
    return out


def build_desktop(db: Db, tax: Taxonomy, truth: TruthService, desktop: int, out: Path,
                  work: Path, cache_dir: Optional[str], resolve: Callable, tail: Callable,
                  qa: dict, excluded: dict[int, str],
                  expected_hw: tuple[int, int] = EXPECTED_HW,
                  expected_tier: str = EXPECTED_TIER) -> dict:
    """Write ``out/D<nn>/`` for one desktop; returns its counts.

    ``excluded`` is ``step -> reason`` of the frames left out of the package.
    The README's fixed claims are checked frame by frame and stop the build
    when one fails (R2-N3): every picture ``expected_hw`` (height, width), the
    view's tier ``expected_tier``, and a picture other than ``P_0`` chosen only
    because ``P_0`` is dark (with the frame record, if it differs, naming
    ``P_0``).
    """
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
    # the excluded frames leave here: nothing below sees them
    dropped = {img["id"] for img in doc["images"] if int(img["extra"]["step"]) in excluded}
    doc["images"] = [img for img in doc["images"] if img["id"] not in dropped]
    doc["annotations"] = [a for a in doc["annotations"] if a["image_id"] not in dropped]
    step_of = {img["id"]: int(img["extra"]["step"]) for img in doc["images"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in doc["annotations"]:
        by_image[ann["image_id"]].append(ann)
    images = {step_of[img["id"]]: img for img in doc["images"]}

    chosen = load_cache_manifest(cache_dir, desktop)
    frames = db.frames_for(desktop, VIEW)
    frame_steps = [int(f["step"]) for f in frames]
    missing = sorted(s for s in excluded if s not in set(frame_steps))
    if missing:
        raise PackError(f"--exclude: D{desktop:02d} has no frame {missing}")
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
    stats: Counter = Counter()
    per_inst = defaultdict(Counter)
    other_image: list[dict] = []
    segments: set = set()
    rois: set = set()
    for frame in frames:
        step = int(frame["step"])
        key = FrameKey(desktop, step, VIEW)
        name = frame_name(step)
        roi = roi_of(db, key)
        img = images.get(step)
        source = _source_record(frame, chosen.get(str(step)))
        folder, file_name = _source_folder(source["recorded"], tail)
        row_file = (None if source["frame_row_path"] is None
                    else _source_folder(source["frame_row_path"], tail)[1])
        entry = {
            "step": step, "exported": img is not None,
            "excluded": excluded.get(step),
            "image": None, "label": None,
            "source_folder": folder, "file_name": file_name,
            "burst_choice": source["choice"],
            "frame_record_file": row_file,
            "sha256": None, "bytes": None, "width": None, "height": None,
            "identical_to_annotated_copy": None, "annotated_copy_same_pixels": None,
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
            if img is not None:
                other_image.append({"step": step, "frame_record_file": row_file,
                                    "file_name": file_name})
        frames_out[str(step)] = entry
        if img is None:
            continue
        if file_name != FIRST_IMAGE and source["choice"]["reason"] != DARK_P0:
            raise PackError(f"D{desktop:02d} {name}: the picture is {file_name}, chosen for "
                            f"{source['choice']['reason']!r}; the README explains only a dark "
                            f"P_0 ({DARK_P0!r})")
        if row_file is not None and row_file != FIRST_IMAGE:
            raise PackError(f"D{desktop:02d} {name}: the frame record names {row_file}; the "
                            f"README says such a record names {FIRST_IMAGE}")
        if img.get("tier") != expected_tier:
            raise PackError(f"D{desktop:02d} {name}: tier {img.get('tier')!r}, the README "
                            f"says {expected_tier!r}")
        stats[f"burst_{int(source['choice']['burst_size'] or 0)}"] += 1
        stats[f"burst_reason:{source['choice']['reason'] or 'frame_record'}"] += 1
        segments.add(entry["pose_segment"])
        rois.add(None if roi is None else tuple(roi))
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
        if hw != tuple(expected_hw):
            raise PackError(f"D{desktop:02d} {name}: picture is {hw[1]}x{hw[0]}, the README "
                            f"says {expected_hw[1]}x{expected_hw[0]} for every frame")
        cached = (Path(cache_dir) / VIEW / desktop_dir(desktop) / f"s{step:03d}.png"
                  if cache_dir else None)
        if cached is not None and cached.is_file():
            same = sha256_of(cached) == digest
            entry["identical_to_annotated_copy"] = same
            entry["annotated_copy_same_pixels"] = (True if same
                                                   else _pixels_equal(str(cached), str(dest)))
            if not same:
                qa["cache_differs_from_raw"].append(
                    {"desktop": desktop, "step": step,
                     "pixels_identical": entry["annotated_copy_same_pixels"]})
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
        frame_stats = annotate_frame(anns, label, visible, hw, key_to_id, roi, compiled)
        stats.update(frame_stats)
        stats["overlap_frames"] += int(frame_stats["overlap_px"] > 0)
        if frame_stats["overlap_px"]:
            # which shared pixels a per-frame mask claims (the tool never takes
            # one out of its neighbours) and which come from rows alone
            claims = np.zeros(hw, dtype=np.uint8)
            for mask in visible.values():
                claims += mask
            corrected = np.zeros(hw, dtype=bool)
            for k in patched & set(visible):
                corrected |= visible[k]
            plain = int(np.count_nonzero((claims > 1) & ~corrected))
            stats["overlap_px_plain"] += plain
            stats["overlap_frames_plain"] += int(plain > 0)
        lpath = ddir / "labels" / f"{name}.png"
        write_label_png(lpath, label)
        entry.update({"label": f"labels/{name}.png", "label_sha256": sha256_of(lpath)})
        img["extra"] = {"desktop": desktop, "step": step, "view": VIEW,
                        "review_status": frame.get("review_status"), "roi": entry["roi"]}
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
            if attrs.get("tier") != expected_tier:
                raise PackError(f"D{desktop:02d} {name} {k}: tier {attrs.get('tier')!r}, the "
                                f"README says {expected_tier!r}")
            stats[f"visibility:{attrs.get('visibility')}"] += 1
            stats[f"placement:{attrs.get('placement')}"] += 1
            if not _bbox_inside(ann.get("amodal_bbox"), roi):
                stats[f"roi_outside:{all_insts[k].cls}"] += 1
            stats["shared_px"] += ann["shared_px"]
            amodal_area = int(ann.get("amodal_area") or 0)
            if ann["iscrowd"] and amodal_area and \
                    ann["occluded_unlabelled_px"] >= UNLABELLED_SHARE * amodal_area:
                qa["empty_but_nothing_labelled_covers_it"].append(
                    {"desktop": desktop, "step": step, "instance": k,
                     "amodal_area": amodal_area,
                     "unlabelled_px": ann["occluded_unlabelled_px"],
                     # 「被线挡住」/「角度看不到」: the annotator said why
                     "stated_reason": attrs.get("occluder")})
            if attrs.get("visibility") == OCCLUDED_FULL and ann["area"] > 0:
                stats["occluded_full_with_pixels"] += 1
                qa["occluded_full_label_with_pixels"].append(
                    {"desktop": desktop, "step": step, "instance": k, "area": ann["area"]})
            if not ann["amodal_is_shape"] and ann["area"] > 0:
                stats["amodal_widened"] += 1
                qa["amodal_widened_by_visible"].append(
                    {"desktop": desktop, "step": step, "instance": k})
        if frame_stats["overlap_px"]:
            pairs = []
            keys = [a["attributes"]["instance_key"] for a in anns if a["shared_px"]]
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    n = int(np.count_nonzero(visible[a] & visible[b]))
                    if n:
                        pairs.append({"a": a, "b": b, "px": n})
            qa["overlapping_confirmed_masks"].append(
                {"desktop": desktop, "step": step, "pairs": pairs})

    # -- the timeline: states per step, the leaving step, flicker
    steps_all = sorted(set(steps_tab) | set(frame_steps))
    states = {s: ctx.state_at(s) for s in steps_all}
    leaving = _leaving(actions, leaving_steps(ctx.instances, ctx.events, tax),
                       sorted(images), seg_of)
    _flicker(doc, step_of, desktop, states, qa)
    _hidden_before_leaving(doc, step_of, desktop, leaving, qa)

    instances_out: dict[str, dict] = {}
    timeline: dict[str, dict] = {}
    values = {"state": set(), "placement": set()}
    for k, iid in sorted(key_to_id.items(), key=lambda t: t[1]):
        rec = all_insts[k]
        attrs = {a: v for a, v in (rec.attrs or {}).items() if a not in DROP_ATTRS}
        context = is_context(rec)
        per_step = None
        if not context:
            per_step = {}
            for s in steps_all:
                here = states[s].get(k)
                per_step[str(s)] = (None if here is None else
                                    {"state": here.state, "placement": here.placement})
                if here is not None:
                    values["state"].add(here.state)
                    values["placement"].add(here.placement)
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
            "captive": (rec.attrs or {}).get("captive"),
            "context": context, "extra": is_extra(rec), "implied": is_implied(rec),
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
            "excluded": excluded.get(s),
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
    model, sheet, note = model_names(meta)
    head = {"desktop": desktop, "brand": meta.get("brand"), "model": model,
            "model_sheet": sheet, "model_note": note, "view": VIEW}
    doc["info"] = {
        "description": f"Scanner-view segmentation package v{PACK_VERSION}, desktop "
                       f"{desktop_dir(desktop)}: confirmed visible masks, amodal shapes "
                       f"and occlusion, one annotation per (frame, part)",
        "version": doc["info"]["version"],
        "desktop": desktop, "brand": head["brand"], "model": model, "model_sheet": sheet,
        "model_note": note, "view": VIEW,
        "confirmed_rows_only": True, "full_frame_coordinates": True,
        "fully_covered_parts_as_crowd": True,
        "amodal_widened_rows": int(stats["amodal_widened"]),
        "excluded_frames": [{"step": s, "reason": r} for s, r in sorted(excluded.items())],
        "fields": list(ANN_FIELDS),
        "label_maps": "labels/S<nn>.png, uint16, 0 = background",
    }
    write_json(ddir / "annotations.json", doc)
    write_json(ddir / "frames.json", dict(head, frames=frames_out))
    write_json(ddir / "instances.json", dict(head, steps=steps_all, instances=instances_out))
    write_json(ddir / "sequence.json", dict(head, steps=steps_out, leaving=leaving_order,
                                            timeline=timeline))

    by_class = defaultdict(Counter)
    clean_rows = []
    for a in doc["annotations"]:
        cls = instances_out[str(a["instance_id"])]["class"]
        by_class[cls]["annotations"] += 1
        if a.get("clean"):
            by_class[cls]["clean"] += 1
            clean_rows.append({"desktop": desktop, "step": step_of[a["image_id"]],
                               "instance_id": a["instance_id"], "class": cls,
                               "area": int(a["area"]),
                               "instance_key": a["attributes"]["instance_key"]})
    return {
        "desktop": desktop, "brand": head["brand"], "model": model, "model_sheet": sheet,
        "model_note": note,
        "frames": len(images), "frames_in_view": len(frames),
        "pose_segments": len(segments), "rois": len(rois),
        "excluded": [{"step": s, "reason": r} for s, r in sorted(excluded.items())],
        "instances": len(key_to_id),
        "instances_annotated": sum(1 for k in key_to_id if per_inst[k]["annotations"]),
        "annotations": len(doc["annotations"]),
        "crowd": sum(1 for a in doc["annotations"] if a["iscrowd"]),
        "clean": len(clean_rows),
        "clean_instances": len({r["instance_id"] for r in clean_rows}),
        "amodal_widened_rows": int(stats["amodal_widened"]),
        "overlap_frames": int(stats["overlap_frames"]),
        "overlap_px": int(stats["overlap_px"]),
        # of which no per-frame mask is involved: confirmed rows alone
        "overlap_px_without_per_frame_mask": int(stats["overlap_px_plain"]),
        "overlap_frames_without_per_frame_mask": int(stats["overlap_frames_plain"]),
        "shared_px": int(stats["shared_px"]),
        # every confirmed row against what today's annotation draws for it
        "confirmed_vs_today": {"rows_compared": truth_rows,
                               "pixel_identical": truth_identical,
                               "differ_within_tolerance": truth_rows - truth_identical,
                               "largest_difference_px": truth_max_px},
        "_clean_rows": clean_rows,
        "_stats": stats,
        "_by_class": by_class,
        "_values": values,
        "_other_image": other_image,
    }


def _flicker(doc: dict, step_of: dict, desktop: int, states: dict, qa: dict) -> None:
    """A part seen on the frames either side but not on this one, its state unchanged."""
    area: dict[str, dict[int, int]] = defaultdict(dict)
    for ann in doc["annotations"]:
        area[ann["attributes"]["instance_key"]][step_of[ann["image_id"]]] = int(ann["area"])
    steps = sorted(set(step_of.values()))
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


def _hidden_before_leaving(doc: dict, step_of: dict, desktop: int, leaving: dict,
                           qa: dict) -> None:
    """A part removed at ``r`` whose frame ``r - 1`` shows nothing of it."""
    area = {(a["attributes"]["instance_key"], step_of[a["image_id"]]): int(a["area"])
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
               tax: Optional[Taxonomy] = None, force: bool = False,
               exclude: Optional[dict[tuple[int, int], str]] = None,
               notes: Sequence[str] = (),
               expected_hw: tuple[int, int] = EXPECTED_HW,
               expected_tier: str = EXPECTED_TIER) -> dict:
    """Write the package; returns the summary (counts per desktop, the QA lists).

    ``resolve`` turns a recorded raw path into the file to read today and
    ``tail`` into its part below the dataset root (the CLI passes
    :mod:`tda.core.rawroot`'s; without one every source folder is refused
    rather than shipped as a machine path). ``work_dir`` receives the database
    copy, the plain COCO exports and ``qa.json``. ``exclude`` is ``(desktop,
    step) -> reason`` of frames to leave out; ``notes`` are lines for the
    README's Known issues. ``expected_hw`` (height, width) and
    ``expected_tier`` are what the README promises of every frame; a frame
    that breaks either stops the build.
    """
    out = Path(out_dir)
    work = Path(work_dir) if work_dir else REPO / ".cache" / "scanpack_work"
    resolve = resolve or (lambda p: p)
    tail = tail or (lambda p: None)
    tax = tax or load_taxonomy()
    exclude = dict(exclude or {})
    wanted = [int(d) for d in desktops]
    stray = sorted(k for k in exclude if k[0] not in wanted)
    if stray:
        raise PackError(f"--exclude names desktops that are not packed: {stray}")
    _prepare_out(out, force)
    db = Db(str(_work_copy(db_path, work)))
    qa: dict[str, list] = {k: [] for k in QA_KEYS}
    try:
        truth = TruthService(db, tax, cache_dir=cache_dir)
        per_desktop = []
        for desktop in wanted:
            if db.get_desktop(desktop) is None:
                raise PackError(f"desktop {desktop} is not in {db_path}")
            mine = {s: r for (d, s), r in exclude.items() if d == desktop}
            per_desktop.append(build_desktop(db, tax, truth, desktop, out, work, cache_dir,
                                             resolve, tail, qa, mine,
                                             expected_hw=tuple(expected_hw),
                                             expected_tier=expected_tier))
    finally:
        db.close()
    clean_rows = [r for d in per_desktop for r in d.pop("_clean_rows")]
    stats: Counter = Counter()
    by_class: dict[str, Counter] = defaultdict(Counter)
    values = {"state": set(), "placement": set()}
    other_image = []
    for d in per_desktop:
        stats.update(d.pop("_stats"))
        for cls, c in d.pop("_by_class").items():
            by_class[cls].update(c)
        for k, v in d.pop("_values").items():
            values[k] |= v
        other_image += [dict(r, desktop=d["desktop"]) for r in d.pop("_other_image")]
    _write_clean_csv(out / "clean_parts.csv", clean_rows)
    summary = {
        "package": "scanpack", "pack_version": PACK_VERSION, "view": VIEW,
        "coco_export_version": EXPORT_VERSION,
        "desktops": [d["desktop"] for d in per_desktop],
        # checked on every frame (R2-N3)
        "frame_size": {"width": int(expected_hw[1]), "height": int(expected_hw[0])},
        "tier": expected_tier,
        "totals": {k: sum(d[k] for d in per_desktop)
                   for k in ("frames", "instances", "annotations", "crowd", "clean",
                             "amodal_widened_rows", "overlap_frames", "overlap_px",
                             "overlap_px_without_per_frame_mask",
                             "overlap_frames_without_per_frame_mask",
                             "shared_px")},
        "per_desktop": per_desktop,
        "clean_by_class": {c: int(by_class[c]["clean"]) for c in sorted(by_class)
                           if by_class[c]["clean"]},
        "annotations_by_class": {c: int(by_class[c]["annotations"]) for c in sorted(by_class)},
        "clean_by_desktop_and_class": {
            desktop_dir(d["desktop"]): dict(sorted(Counter(
                r["class"] for r in clean_rows if r["desktop"] == d["desktop"]).items()))
            for d in per_desktop},
        "visibility_counts": {k.split(":", 1)[1]: int(v) for k, v in sorted(stats.items())
                              if k.startswith("visibility:")},
        "placement_counts": {k.split(":", 1)[1]: int(v) for k, v in sorted(stats.items())
                             if k.startswith("placement:")},
        "state_values": sorted(values["state"]),
        "placement_values": sorted(values["placement"]),
        "burst_sizes": {k.split("_", 1)[1]: int(v) for k, v in sorted(stats.items())
                        if k.startswith("burst_") and not k.startswith("burst_reason:")},
        "burst_reasons": {k.split(":", 1)[1]: int(v) for k, v in sorted(stats.items())
                          if k.startswith("burst_reason:")},
        "amodal_box_outside_roi_by_class": dict(sorted(
            ((k.split(":", 1)[1], int(v)) for k, v in stats.items()
             if k.startswith("roi_outside:")), key=lambda kv: (-kv[1], kv[0]))),
        "checks": {
            "occlusion_ratio_differs_from_confirmed": int(stats["ratio_differs"]),
            "occluded_full_with_pixels": int(stats["occluded_full_with_pixels"]),
            "amodal_box_outside_roi": int(stats["roi_outside"]),
            "clean_but_for_roi": int(stats["only_roi"]),
            "clean_but_for_shared_px": int(stats["only_shared"]),
        },
        "frame_record_names_other_image": sorted(
            other_image, key=lambda r: (r["desktop"], r["step"])),
        "exclusions": [{"desktop": d, "step": s, "reason": r}
                       for (d, s), r in sorted(exclude.items())],
        "notes": [str(n) for n in notes],
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
inst = json.load(open(f"{desk}/instances.json"))["instances"]   # keys are strings
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


def _pct(value: float) -> str:
    return f"{value:g}"


def _n(count: int, word: str, plural: Optional[str] = None) -> str:
    """``"1 annotation"`` / ``"3 annotations"``."""
    return f"{count} {word if count == 1 else (plural or word + 's')}"


def few_clean(summary: dict) -> list[tuple[str, int, int]]:
    """``(class, clean, annotations)`` of the classes under :data:`RARELY_CLEAN` clean."""
    out = []
    for cls, n in sorted(summary["annotations_by_class"].items(), key=lambda kv: (-kv[1], kv[0])):
        clean = summary["clean_by_class"].get(cls, 0)
        if clean < RARELY_CLEAN * n:
            out.append((cls, clean, n))
    return out


def few_clean_text(summary: dict) -> tuple[str, str]:
    """``(English, Chinese)`` lines naming the classes with few or no clean rows.

    Generated from the share alone and worded neutrally (R2-N1): which classes
    fall under the line depends on the desktops packed, and no reason is given
    that a class -- a RAM stick, a latch -- might not share.
    """
    share = int(round(RARELY_CLEAN * 100))
    rows = few_clean(summary)
    if not rows:
        return (f"Every class has at least {share} % of its annotations clean.",
                f"每一类都至少有 {share}% 的标注是 clean。")
    en = ", ".join(f"{cls} ({clean} of {n})" for cls, clean, n in rows)
    zh = "、".join(f"{cls}（{clean}/{n}）" for cls, clean, n in rows)
    return (f"Classes with few or no clean rows (under {share} % of their annotations): "
            f"{en}. For these use the visible mask or the disassembly order "
            f"(*Occluded parts*).",
            f"clean 行占比低于 {share}% 的类：{zh}；这些类请用可见掩码或拆卸顺序取点。")


def roi_paragraph(summary: dict) -> str:
    """The README's ROI section body, every sentence from the data (R2-N2, R2-N3)."""
    c = summary["checks"]
    bench = int(summary["placement_counts"].get("on_bench", 0))
    out = ["`frames.json` gives each frame's `roi` `[x0, y0, x1, y1]` (x1, y1 exclusive): "
           "the rectangle around the chassis the annotation concentrated on."]
    several = [d for d in summary["per_desktop"]
               if d.get("pose_segments", 1) > 1 or d.get("rois", 1) > 1]
    if not several:
        out.append("Each desktop here was scanned from one camera pose and has one ROI, "
                   "shared by all its frames.")
    else:
        which = ", ".join(f"{desktop_dir(d['desktop'])} "
                          f"({_n(d['pose_segments'], 'camera pose')}, {_n(d['rois'], 'ROI')})"
                          for d in several)
        out.append(f"An ROI is drawn once per camera pose (`pose_segment` in `frames.json`); "
                   f"{which} {'has' if len(several) == 1 else 'have'} more than one, so "
                   f"read `roi` frame by frame.")
    if bench == 0:
        out.append("Every annotation in this package is of a part in the chassis; none is "
                   "of a part set down on the bench.")
    else:
        out.append(f"{_n(bench, 'annotation')} {'is' if bench == 1 else 'are'} of parts set "
                   f"down on the bench (`placement: on_bench`), which may lie outside the "
                   f"ROI; the others are of parts in the chassis.")
    crossing = int(c["amodal_box_outside_roi"])
    if crossing == 0:
        out.append("Every amodal box lies inside its frame's ROI, so rule 3 of *clean* "
                   "excludes nothing here.")
    else:
        classes = ", ".join(f"{k} {v}" for k, v in
                            summary["amodal_box_outside_roi_by_class"].items())
        only = int(c["clean_but_for_roi"])
        tail = (f"; for {only} of them it is the only reason" if only
                else "; each of them also fails another rule")
        out.append(f"The rectangle is not a tight fit: on {_n(crossing, 'annotation')} "
                   f"({classes}) the amodal box reaches past it, so by rule 3 "
                   f"{'it is' if crossing == 1 else 'they are'} not clean{tail}.")
    return "\n".join(out)


#: What each burst-choice ``reason`` of the frame cache means.
REASON_TEXT = {
    "p0": "the first image",
    "p0_dark": "the first image was a dark exposure, so the sharpest of the others",
    "p0_saturated": "the first image was over-exposed, so the sharpest of the others",
    "p0_outlier": "the first image differed from the rest, so the sharpest of the others",
    "p0_missing": "the burst has no first image, so the lowest-numbered one",
}


def _reason_text(reason: str) -> str:
    if reason in REASON_TEXT:
        return REASON_TEXT[reason]
    if reason.endswith("_kept"):
        return "no image of the burst passed the checks, so the first was kept"
    return reason


def overlap_cause(summary: dict) -> str:
    """Why confirmed masks overlap here, from the counts (final build, item e)."""
    t = summary["totals"]
    total = int(t["overlap_px"])
    plain = int(t.get("overlap_px_without_per_frame_mask", 0))
    corrected = ("a mask corrected on that frame only (the annotation tool does not take a "
                 "per-frame mask out of its neighbours)")
    rows = ("rows confirmed at different times (an outline adjusted later, within the "
            "tool's tolerance)")
    if plain == 0:
        return (f"every one of them is {corrected}; masks without such a correction do "
                f"not overlap anywhere in this package")
    if plain == total:
        return f"they are {rows}"
    return (f"{total - plain} of them are {corrected}, and {plain} (on "
            f"{_n(int(t.get('overlap_frames_without_per_frame_mask', 0)), 'frame')}) are "
            f"{rows}")


def overlap_intro(summary: dict) -> str:
    """The first sentence of *Where two confirmed masks overlap*, zero included."""
    t = summary["totals"]
    if not t["overlap_px"]:
        return "No two confirmed visible masks claim the same pixel in this package."
    return (f"On {_n(t['overlap_frames'], 'frame')} two confirmed visible masks claim the "
            f"same {t['overlap_px']} pixels; {overlap_cause(summary)}.")


def _known_issues(summary: dict) -> str:
    t, c = summary["totals"], summary["checks"]
    items = []
    if t["overlap_frames"]:
        items.append(
            f"**Overlapping confirmed masks.** On {_n(t['overlap_frames'], 'frame')}, "
            f"{t['overlap_px']} pixels are claimed by two confirmed visible masks: "
            f"{overlap_cause(summary)}. The label map gives each such pixel to one part "
            f"(see *Where two confirmed masks overlap*); take points from the label map. "
            f"Every row involved has `shared_px > 0` and none is clean.")
    if c["occluded_full_with_pixels"]:
        items.append(
            f"**`occluded_full` with a sliver.** {c['occluded_full_with_pixels']} annotations "
            f"are labelled `occluded_full` yet keep a few visible pixels: the label follows "
            f"the rule occlusion ratio >= {_pct(OCCLUDED_FULL_RATIO)} (on the confirmed "
            f"ratio), not \"no pixel visible\". The pixels are real; use them like any "
            f"other visible pixels.")
    other = summary["frame_record_names_other_image"]
    if other:
        where = ", ".join(f"{desktop_dir(r['desktop'])} {frame_name(r['step'])}" for r in other)
        items.append(
            f"**Picture vs. our frame record.** For {len(other)} frames our database's frame "
            f"record names `P_0.png`, but the picture the masks were drawn on -- the one in "
            f"this package -- is another image of the burst (`P_0` is a dark exposure): "
            f"{where}. `frames.json` gives the file in `file_name` and the record's name in "
            f"`frame_record_file`. The masks match the packaged picture.")
    for ex in summary["exclusions"]:
        reason = str(ex["reason"]).strip()
        reason += "" if reason.endswith((".", "!", "?")) else "."
        items.append(f"**{desktop_dir(ex['desktop'])} {frame_name(ex['step'])} is left out** "
                     f"(no picture, label map or annotation; not in any count): {reason}")
    items += [str(n) for n in summary["notes"]]
    return "\n".join(f"{i}. {text}" for i, text in enumerate(items, start=1)) or "None known."


def readme_text(summary: dict) -> str:
    t, c = summary["totals"], summary["checks"]
    noted = [d for d in summary["per_desktop"] if d.get("model_note")]
    rows = "\n".join(
        f"| {desktop_dir(d['desktop'])} | {d['model']}{' *' if d.get('model_note') else ''} | "
        f"{d['frames']} | {d['instances_annotated']} | {d['annotations']} | {d['crowd']} | "
        f"{d['clean']} | {d['clean_instances']} |"
        for d in summary["per_desktop"])
    # each note a paragraph of its own: a line right under a table is a table row
    model_notes = "".join(f"\n\\* {desktop_dir(d['desktop'])}: {d['model_note']}.\n"
                          for d in noted)
    classes = ", ".join(f"{k} {v}" for k, v in sorted(summary["clean_by_class"].items(),
                                                       key=lambda kv: (-kv[1], kv[0])))
    desks = ", ".join(desktop_dir(d) for d in summary["desktops"])
    few_en, few_zh = few_clean_text(summary)
    vis = summary["visibility_counts"]
    bursts = summary["burst_sizes"]
    longest = max((int(k) for k in bursts), default=0)
    odd_bursts = sum(v for k, v in bursts.items() if int(k) < longest)
    reasons = "; ".join(f"`{r}` ({_n(n, 'frame')}): {_reason_text(r)}"
                        for r, n in summary["burst_reasons"].items())
    width, height = summary["frame_size"]["width"], summary["frame_size"]["height"]
    full, part = _pct(OCCLUDED_FULL_RATIO), _pct(OCCLUDED_PARTIAL_RATIO)
    small, tiny = TOO_SMALL_PX, VISIBLE_TINY_PX
    excluded = ", ".join(f"{desktop_dir(e['desktop'])} {frame_name(e['step'])}"
                         for e in summary["exclusions"]) or "none"
    return f"""# Scanner-view segmentation package (scanpack v{summary['pack_version']})

## 中文摘要

本包是 {len(summary['desktops'])} 台台式机（{desks}）结构光扫描仪俯视 RGB 帧（{width}×{height}）的二维分割标注，
供 3D 合作方测试"用 2D 掩码从重建点云里取出每个零件的点"。每台机器一个文件夹 `D<nn>/`：
`images/S<nn>.png` 是标注时画掩码所用的那张原始扫描图（从原始盘逐字节拷贝），
`labels/S<nn>.png` 是 16 位实例 id 图（0 = 背景，id 见 `instances.json`，只在本台机器内有效），
`annotations.json` 是 COCO 格式（可见掩码 + 完整形状 amodal 掩码 + 遮挡信息），
`frames.json` / `sequence.json` 给出每一步的名称、动作和每个零件的状态时间线。
坐标就是图像像素 (x, y)，原点在左上角。S01 是初始状态，没有动作；其余每张图都拍在该步动作**之后**
（S05 上第 5 步拆下的零件已经不在了）。

**取点以 label map 为准**：零件的点就是 `labels/S<nn>.png == instance_id` 的像素下的点。
确认的可见掩码在 {t['overlap_frames']} 帧上有重叠，`segmentation` 保持确认时的原样，重叠像素在 label map 里只归一个零件。

建议先用 `clean_parts.csv`（共 {t['clean']} 条）。`clean` 要同时满足六条：
(1) 可见掩码与完整形状逐像素相同；(2) 标签为 `visible`；(3) amodal 框在 ROI 内；
(4) amodal 就是标注的形状本身（没有被可见掩码撑大）；(5) label map 把它的每个可见像素都给了它；
(6) 它的可见像素没有一个同时被别的确认掩码占用（`shared_px == 0`）。
这样的零件在那一帧上没有被任何东西挡住，掩码下的点就是扫描仪看到的它的全部表面。
{few_zh}
被遮挡的零件有两条路：(1) 只取可见掩码（label map）下的点；
(2) 用拆卸顺序：零件在第 r 步离开，取第 r-1 帧重建里有、第 r 帧重建里没有、且落在它第 r-1 帧 amodal 掩码内的点。
完全被挡住的零件是 crowd 行（`iscrowd: 1`，可见掩码为空），那一帧上没有它的点。
amodal 掩码是完整轮廓，不是点集——被挡住的表面扫描仪看不到。已知问题见文末 *Known issues*。
内部测试数据，请勿外传（见 `LICENSE_NOTE.txt`）。

## What this is

Two-dimensional instance segmentation of the top-down RGB frames of the
structured-light scanner, for {len(summary['desktops'])} desktop computers being
taken apart step by step. Every frame was annotated and then confirmed by a
human annotator; the visible masks here are exactly the confirmed ones (see
*Confirmed and derived fields* for what else is in the files). The goal of the
package is to test whether these masks let you select each annotated part's
points in your 3D reconstruction of the same scanner frame.

| Desktop | Model | Frames | Parts | Annotations | Fully covered (crowd) | Clean | Clean parts |
|---|---|---|---|---|---|---|---|
{rows}
| **Total** | | {t['frames']} | | {t['annotations']} | {t['crowd']} | {t['clean']} | |
{model_notes}

*Parts* counts the instances with at least one annotation; *Clean* counts
(frame, part) annotations, *Clean parts* the distinct parts among them.
Frames left out of this package: {excluded} (see *Known issues*). `frames.json`,
`instances.json` and `annotations.json` give the model name (`model`), the
teardown sheet's own spelling of it (`model_sheet`) and a caveat about the
name where there is one (`model_note`, else null).

Clean annotations by class: {classes}.

{few_en}

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

## Steps and pictures

`S<nn>` is the step number. **S01 is the initial condition** (step name
"Initial conditions", no action). Every other picture is taken **after** that
step's action: on `S05` the part removed at step 5 is already gone.

The scanner writes a burst of images per step (`P_0.png`, `P_1.png`, ...) and
one of them is the picture the masks were drawn on. `frames.json` names it:
`source_folder` (below the dataset root on our scanner drive) and `file_name`,
with `burst_choice`: `index` is its position in the burst as listed,
`p_index` the number in its file name (`P_<p_index>.png`), `burst_size` how
many images the burst has, `reason` why it was chosen and `by` who chose it
(`annotation_tool`: the tool that showed it to the annotator). Reasons here:
{reasons}. Burst sizes here: {', '.join(f'{v} frames with {k} images' for k, v in sorted(bursts.items(), key=lambda kv: int(kv[0])))}{'.' if not odd_bursts else f'; in the {odd_bursts} shorter bursts an image is missing, so `index` and `p_index` can differ.'}
`sha256`, `width`, `height` identify the file; `identical_to_annotated_copy`
says that the annotation tool's working copy of the picture was byte for byte
this file.

## Coordinate frame

Everything is in image pixels of the picture `frames.json` names: `(x, y)` with
the origin at the top-left corner, x to the right (column), y down (row),
{width} x {height} for every frame here (checked on each one when the package
was built). Masks, boxes (`[x, y, w, h]`), the label maps
and the ROI all use this frame, unshifted and uncropped.

## Selecting points

If your reconstruction keeps, for each 3D point, the pixel of this image it
came from (an organized point cloud, or a depth map registered to this RGB
image), then **the points of a part are the points under its pixels in the
label map**: `labels/S<nn>.png == instance_id`. Use the label map, not the
annotation's `segmentation`: the segmentation is the confirmed mask as it was
signed off and, on {t['overlap_frames']} frames, it overlaps another part's
(*Where two confirmed masks overlap*); the label map gives every pixel to one
part. Where they do not overlap the two are identical. The scanner only
measures surfaces it can see, so a hidden surface has no points: the
`amodal_segmentation` is the part's full silhouette on the image (what it would
cover if nothing were in front), not a set of points.

### Start with the clean set

`clean_parts.csv` lists the annotations with `clean: true` ({t['clean']} rows): the
part is not covered by anything on that frame, so its visible mask *is* its
whole outline and the points under it are all of the part the scanner saw.
`clean` is true when all six hold:

1. the visible mask equals the amodal mask to the pixel (0 px difference);
2. its label (`visibility`) is `visible`;
3. the amodal box lies inside the frame's ROI;
4. the amodal shape is the annotated shape itself (`amodal_is_shape`);
5. the label map gives the part every pixel of its visible mask (`label_px == area`);
6. no other confirmed mask claims any of its pixels (`shared_px == 0`).

### Occluded parts: two routes

1. **Visible mask.** Take the points under the part's pixels in the label map.
   This is what the scanner could see of the part on that frame
   (`occlusion_ratio` says how much of its silhouette is missing).
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
   (a connector unplugged, a latch opened). A frame left out of the package is
   never a `before_frame` or `after_frame`: they are the nearest frames in the
   package before and at `r`, and null when there is none.
   `leaving.same_pose_segment` is true when the two frames were taken from the
   same camera pose (compare them directly), false when the pose changed
   between them, and null when one of them is missing.

### Visibility labels

`visibility` is one of six values. The annotation tool decides it from the
occlusion ratio of the confirmed row and the shorter side of the visible
mask's bounding box: the rules below are tried top to bottom and the first one
that holds gives the label (an annotator may override it, and then
`attributes.occluder` may give the reason: `cable` -- hidden by a cable nobody
annotates -- or `viewpoint`):

| Value | Rule (first match wins) | Annotations here |
|---|---|---|
| `out_of_view` | no geometry on the frame | 0 (such a row is never an annotation) |
| `occluded_full` | occlusion ratio >= {full} | {vis.get('occluded_full', 0)} |
| `too_small` | shorter side < {small} px | {vis.get('too_small', 0)} |
| `visible_tiny` | shorter side {small} to {tiny - 1} px | {vis.get('visible_tiny', 0)} |
| `occluded_partial` | occlusion ratio >= {part} (and < {full}) | {vis.get('occluded_partial', 0)} |
| `visible` | everything else: occlusion ratio < {part} | {vis.get('visible', 0)} |

So **a `visible` part may still be partly covered** (up to {int(float(part) * 100)} % of its
silhouette); `clean` is the test for "not covered at all". An `occluded_full`
part may keep a sliver of visible pixels (*Known issues*).

### `occlusion_ratio` and `occluders`

`occlusion_ratio` (top level) is `1 - area / amodal_area`, measured on the two
masks in the file: the share of the part's silhouette hidden on that frame.
`attributes.occlusion_ratio` is the value recorded when the row was confirmed
(the one `visibility` was decided from); the two differ on
{c['occlusion_ratio_differs_from_confirmed']} annotations, where today's amodal shape is not the one the
row was confirmed against. `occluders` lists who takes the hidden pixels, read
off the label map: `[{{"id": instance_id, "px": pixels}}, ...]`, largest first.
`occluded_unlabelled_px` is the rest -- hidden pixels no labelled part shows
(a cable, a part nobody annotates, a hand). The three add up:
`sum(px) + occluded_unlabelled_px == amodal_area - area`.

### The ROI

{roi_paragraph(summary)}

### Fully covered parts (crowd rows)

A part that is in its place but completely covered on a frame is still an
annotation: an empty `segmentation`, `area` 0, `bbox` = its amodal box,
`iscrowd: 1`, and its `amodal_segmentation`. It has no points on that frame;
use route 2 (or another frame) for it.

### Where two confirmed masks overlap

{overlap_intro(summary)} The
label map gives such a pixel to the part higher in the frame's annotated layer
order (a part whose mask was corrected on that frame is on top). Each
annotation's `shared_px` says how many of its visible pixels another mask also
claims and `label_px` how many the label map gives it; `segmentation` itself
is kept as it was confirmed. No annotation with `shared_px > 0` is clean.

## Confirmed and derived fields

* **Confirmed** (as the annotator signed the frame off): `segmentation`,
  `area`, `bbox` (for a crowd row: the amodal box), `iscrowd`, and in
  `attributes`: `visibility`, `occlusion_ratio`, `occluder`, `placement`,
  `verified`.
* **Derived from today's annotation** (the part's drawn shape on that frame):
  `amodal_segmentation`, `amodal_bbox`, `amodal_area`; on {t['amodal_widened_rows']}
  annotations the shape is widened to hold the confirmed visible mask
  (`amodal_is_shape: false`).
* **Computed for this package** from the two: the label maps, the top-level
  `occlusion_ratio`, `occluders`, `occluded_unlabelled_px`, `shared_px`,
  `label_px`, `amodal_is_shape`, `clean`; and from the teardown log and the
  annotation: `states`, `leaving`, `sequence.json`.

## Files in detail

* **ids and keys.** `instance_id` is valid within one desktop folder: on every
  frame of that desktop it names the same part, and each desktop counts from 1.
  `key` (e.g. `screw.cpu_cooler.03`) is our name for the part and the identity
  to use across versions of this package. JSON object keys are strings:
  `instances["12"]`, `frames["5"]`.
* `annotations.json` -- COCO (`images`, `annotations`, `categories`):
  `segmentation` (visible mask) and `amodal_segmentation` are compressed RLE
  (`pycocotools.mask.decode`), plus `amodal_bbox`, `amodal_area` and an
  `attributes` block: `instance_key`, `state`, `placement`, `visibility`,
  `occluder`, `visibility_reviewed` (a person re-checked the label),
  `occlusion_ratio` (confirmed), `amodal_complete` (the drawn shape covers the
  whole part), `implied`, `tier` (the annotation standard of this camera:
  `{summary['tier']}` on every annotation here) and `verified`. The package adds at the top of each annotation:
  `instance_id`, `visibility`, `occlusion_ratio`, `occluders`,
  `occluded_unlabelled_px`, `shared_px`, `label_px`, `amodal_is_shape`,
  `clean`. Each image has `file_name` (`images/S<nn>.png`) and `label_file`
  (`labels/S<nn>.png`); `extra.step` is its step.
* `labels/S<nn>.png` -- single-channel 16-bit PNG, value = `instance_id`
  (0 = background). Read it with Pillow (`np.array(Image.open(p))`) or OpenCV
  (`cv2.imread(p, cv2.IMREAD_UNCHANGED)`).
* `instances.json` -- `instances["<id>"]`: `key`, `class` and
  `class_description`, `category_id` (the COCO category), `attrs`, `parent` /
  `attached` (an attached part leaves with its parent), `mounted_on`,
  `fastens`, `socket_host`, `captive` (a captive screw stays on its part),
  `extra` (added by the annotator, not named in the teardown log), `implied`
  (assumed present, e.g. the motherboard of a machine whose log never names
  it), `leaving` (see route 2), `frames` (annotation counts) and `states`
  (`states["<step>"]`: state and placement at every step). States:
  {', '.join(f'`{s}`' for s in summary['state_values'])}; placements:
  {', '.join(f'`{p}`' for p in summary['placement_values'])}. `attrs` carries what
  the teardown sheet says about the part, where it says it: `role` (what a
  screw fastens), `kind`, `of` (what a cover or latch belongs to), `captive`,
  `sheet_no` (the part's number on the sheet), `cable_owner` (the part whose
  cable a connector is on) and `qualifier` (further words from the sheet).
* `sequence.json` -- `steps`: step, name (as written in the teardown log),
  step type, verbs, targets and the actions; `leaving`: every part's leaving
  step in order; `timeline`: per part, runs of `(from, to, state, placement)`.
* `frames.json` -- `frames["<step>"]`: `image`, `label`, `source_folder`,
  `file_name`, `burst_choice`, `frame_record_file`, `sha256`, `width`,
  `height`, `roi`, `step_name`, `step_type`, `actions`, `review_status`,
  annotation counts, and `excluded` (the reason, for a frame left out).
* `manifest.json` -- the counts above per desktop and per class, the
  exclusions and notes, and the size and sha256 of every other file. Per
  desktop, `confirmed_vs_today` says how every confirmed visible mask compares
  with what today's annotation draws for that part on that frame:
  `pixel_identical`, or `differ_within_tolerance` (inside the annotation tool's
  re-tracing tolerance, so still the confirmed mask), with the largest such
  difference in pixels (`largest_difference_px`); `pose_segments` and `rois`
  count the camera poses and ROIs of its frames.

## Known issues

{_known_issues(summary)}

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
    parser.add_argument("--exclude", action="append", default=[],
                        help="frames to leave out, 'D2:1,D4:1=reason' (repeatable)")
    parser.add_argument("--note", action="append", default=[],
                        help="a line for the README's Known issues (repeatable)")
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
            tail=(raw.tail if raw is not None else None), force=args.force,
            exclude=parse_exclusions(args.exclude), notes=args.note)
    except PackError as exc:
        print(f"[scanpack] refused: {exc}")
        return 1
    for d in summary["per_desktop"]:
        print(f"[scanpack] D{d['desktop']:02d}: {d['frames']} frames, {d['instances']} parts, "
              f"{d['annotations']} annotations, {d['crowd']} crowd, {d['clean']} clean, "
              f"confirmed vs today {d['confirmed_vs_today']}")
    print(f"[scanpack] totals {summary['totals']}")
    print(f"[scanpack] checks {summary['checks']}")
    print(f"[scanpack] clean by class: {summary['clean_by_class']}")
    for key in QA_KEYS:
        print(f"[scanpack] qa {key}: {len(summary['qa'][key])}")
    print(f"[scanpack] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
