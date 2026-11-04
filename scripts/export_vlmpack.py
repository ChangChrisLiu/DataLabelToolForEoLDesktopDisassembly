"""The self-contained VLM evaluation pack (task W4-A; paper plan 6.2, spec 8.2).

::

    python scripts/export_vlmpack.py --db <read-only copy of tda.sqlite> \
        [--out D:/DataSet/exports/vlmpack_<date>] [--cache-dir D:/DataSet/cache] \
        [--views scan,oak1,oak2,rs] [--desktops 1-66] [--seed 0] \
        [--images all|eval|none] [--work <dir>] [--allow-conflicts] [--force] \
        [--no-selfcheck] [--coco-root D:/DataSet/exports/scanpack_<date>] [--overlays eval|none] \
        [--no-multi]

The models are run elsewhere (another machine, the HPRC), so everything the
evaluation needs travels in one directory that neither the annotation tool nor
its database has to be present for::

    README.md                 the data card (English): tasks, fields, formats, how to run
    manifest.json             what was built, counts, and size + sha256 of every other file
    data/vlm_<unit>.jsonl     the five export units of ``export_vlm`` (all records): the
                              four camera views (scan, oak1, oak2, rs) with single-view
                              perception and V15, and ``multi`` with the action tasks, each
                              record one moment seen from every camera (scanner first)
    data/vlm_<unit>.images_manifest.jsonl / .oracle.jsonl / .summary.json
    data/vlm_<unit>.masks.jsonl   (with --coco-root) the confirmed visible masks of every
                              confirmed frame, COCO RLE, from the scanner pack's COCO
    images/<view>/D<nn>/s<nn>.<ext>   the frames the manifests name, byte for byte
    reasoning/<unit>.jsonl    the reasoning side product: ``{id, task, question, answer,
                              reasoning}`` per record of ``data/`` (the gold reply in plain
                              names and the rationale text: what is seen, the rule, the answer)
    eval_v3/                  the frozen evaluation subset (``scripts/vlm_eval_set.py``):
                              ids.txt, <unit>.jsonl + sidecars (+ .masks.jsonl), table.md,
                              table.json, audit/ (the author-audit sample, the fillable HTML
                              sheet and its CSV skeleton -- ``scripts/vlm_audit_sheet.py``),
                              mask_overlay/<unit>.jsonl (+ manifest) -- the GT-mask overlay
                              variant of the subset (W4-C), painted into overlays/
    overlays/<view>/D<nn>/s<nn>.png   the subset's frames with the confirmed masks painted
    tools/                    the scorer, converters, references, the overlay painter, samplers,
                              the audit / human-baseline sheets and the self-check, with the
                              ``tda`` modules they import vendored
                              (``tools/tda``, ``tools/configs``) and ``requirements.txt``

**The database is never written.** The script refuses the live database by path
(:data:`LIVE_DB`) and copies whatever file it is given with the sqlite backup
API into ``--work``; only the copy is opened, because the export's
``ensure_fresh`` writes compiled rows. The frame cache is only read.

**Images.** Every image a manifest names (``--images all``, the default), or
only those the evaluation subset names (``eval``), is copied from
``--cache-dir`` under ``images/`` with the manifest's own relative path, so
``tools/vlm_to_chat.py --image-root <pack>/images`` finds them; the copy is
hashed as it is written. **A missing image stops the build before anything is
copied** (:class:`PackError`), as does too little free space. ``none`` is a dry
run (export + subset + tools, no pictures) and the manifest says so.

**Masks and overlays (W4-C).** The export's oracle carries no mask, so with
``--coco-root`` (the scanner pack, ``scripts/export_scanpack.py``: confirmed
rows only, full-frame RLE) the build writes ``data/vlm_<view>.masks.jsonl``
(and the subset's cut) through :mod:`vlm_overlay`, which refuses a COCO whose
visibility disagrees with the oracle's; then it paints the subset's
``mask_overlay`` variant (``--overlays eval``, the default; ``none`` leaves the
painting to the receiving machine, which can do it from the sidecar). Without
``--coco-root`` nothing of this is built and the card says so.

**Self-check.** The build ends by running ``tools/vlm_selfcheck.py`` from the
pack in a subprocess (``python -I``: no cwd, no ``PYTHONPATH``): the gold reply
of every record of the subset must score 1.0 on every task with the vendored
tools alone, and the ``tda`` it imported must be the pack's. The clean-virtual-
environment run is the separate ``scripts/vlmpack_cleanvenv_check.py``.

Deterministic: two runs over the same database and cache write the same bytes
(no wall clock and no machine path in any shipped file; the pack directory's
date is only its name).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import vlm_eval_set  # noqa: E402
import vlm_io as IO  # noqa: E402
import vlm_overlay  # noqa: E402
from tda.core.db import Db  # noqa: E402
from tda.core.export.vlm import (  # noqa: E402
    MULTI_PRIMARY,
    MULTI_VIEW,
    export_vlm,
    manifest_path,
    unit_tasks,
)
from tda.core.export.vlm import siblings as vlm_siblings  # noqa: E402
from tda.core.export.vlm_check import CAPABILITY_LAYERS, RESPONSE_FORMATS  # noqa: E402
from tda.core.export.vlm_shortcuts import shortcut_report  # noqa: E402
from tda.core.export.vlm_tasks import ALL_TASKS, MARKED_ACTION_TASKS, TRUTH_SOURCES  # noqa: E402
from tda.core.model import VIEWS, is_component_view  # noqa: E402
from tda.core.taxonomy import Taxonomy, load_taxonomy  # noqa: E402
from tda.core.truth import TruthService  # noqa: E402
from tda.pipeline import parse_desktops  # noqa: E402

__all__ = ["IMAGE_MODES", "LIVE_DB", "OVERLAY_MODES", "PACK_VERSION", "REDESIGN_TOOLS", "TOOLS",
           "VENDORED", "PackError", "build_pack", "main", "masks_sidecars", "pack_units",
           "readme_text", "refuse_live", "render_overlays", "require_tools", "sha256_of",
           "work_copy"]

#: The annotation tool's live database: never opened here, by name.
LIVE_DB = "D:/DataSet/annotations/tda.sqlite"
PACK_VERSION = "1"
PACK_NAME = "vlmpack"
DEFAULT_CACHE_DIR = "D:/DataSet/cache"
DEFAULT_WORK = "D:/DataSet/.cache/tmp/w4a/work"
IMAGE_MODES = ("all", "eval", "none")
#: Which overlays the build paints: the subset's, or none (the sidecar still ships).
OVERLAY_MODES = ("eval", "none")
#: The W3b scripts shipped under ``tools/`` (W4-A's own two, W4-C's painter, W4-E's three and
#: the redesign's glossary and audit carry-over included).
TOOLS = ("vlm_io.py", "vlm_to_chat.py", "vlm_score.py", "vlm_references.py",
         "vlm_overlay.py", "vlm_sample.py", "vlm_eval_set.py", "vlm_selfcheck.py",
         "vlm_audit_sheet.py", "vlm_audit_summary.py", "vlm_human_sheet.py",
         "vlm_glossary.py", "vlm_audit_carry.py")
#: Tools the redesign of 2026-11-03 added (the converter's glossary, the audit carry-over):
#: a build without them stops with a message naming them; no pack ships without them.
REDESIGN_TOOLS = ("vlm_glossary.py", "vlm_audit_carry.py")
#: Items per page of the audit sheet the build writes into ``eval_v3/audit/``.
AUDIT_BATCH = 50
#: The ``tda`` modules :mod:`tda.core.export.vlm_check` needs, copied verbatim
#: under ``tools/tda``: the scorer's whole import closure, nothing of the tool.
VENDORED = ("tda/__init__.py", "tda/core/__init__.py", "tda/core/taxonomy.py",
            "tda/core/model.py", "tda/core/states.py", "tda/core/graph_rules.py",
            "tda/core/graph_infer.py", "tda/core/graph_plan.py",
            "tda/core/graph_templates.py", "tda/core/graph.py",
            "tda/core/export/vlm_check.py")
#: The configuration the vendored taxonomy reads (``tools/configs/``).
VENDORED_CONFIGS = ("configs/taxonomy.yaml", "configs/taxonomy_map.yaml")
REQUIREMENTS = """\
# The pack's tools run on Python >= 3.11 with these two packages; nothing else
# (no numpy, no OpenCV, no Qt, no `tda` install -- the modules the scorer needs
# are vendored under tools/tda).
PyYAML>=6.0
# Pillow is needed only by vlm_to_chat.py --roi view / --composite / --max-side
# (crops and composites); the scorer and the self-check do not import it.
Pillow>=10.0
"""
#: One line per task for the data card.
TASK_MEANINGS = {
    "V1": "which parts are visible, and where (class + box for every visible part)",
    "V2": "the state of a named part, or how many screws of a role are still in place",
    "V3": "what the operator just did between two frames (verb, target, tool): free form "
          "(headline, no hint; also asks the region where the machine changed) and roi form "
          "(secondary, a yellow dashed hint box drawn there; same step, `v3_pair`)",
    "V4": "can this action be done to this part right now; if not, what blocks it",
    "V5": "(not in the default set) every action that is legal now",
    "V6": "every action that can be carried out now (the frontier, grouped): open form "
          "(headline, F1 over groups), multiple choice (one option possible, the others "
          "blocked; same frame, `v6_pair`) and constrained (one extra rule in words)",
    "V7": "the plan to take the CPU out under the rules (free) or under one extra rule in "
          "words (constrained: also counts the plugs to unplug), from a frame where that "
          "work has started",
    "V8": "localise a named part (box)",
    "V10": "which actions have already been carried out; progress bin",
    "V12": "did the machine's state change between two frames",

    "V15": "two cameras, two pictures: the same moment of the teardown or not",
    "V16": "is a candidate plan legal from this state; the first step that is not",
}
LICENSE_NOTE = """\
E3D: End-of-Life Desktop Disassembly Dataset -- pre-release.

Published on Hugging Face (ChangChrisLiu/E3D-Desktop-Disassembly) by its owners at
Texas A&M University. A licence and a citation are added with the first tagged
release; until then the data is provided for inspection and research evaluation.
Please do not redistribute copies or train models for release on it before then
without permission from the dataset owners (contact: ChangChrisLiu on Hugging
Face or GitHub). The annotations may still change between releases.
"""


class PackError(RuntimeError):
    """The build stops: what the README would promise is not true of the inputs."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _same_file(a: str, b: str) -> bool:
    na = os.path.normcase(os.path.abspath(str(a)))
    nb = os.path.normcase(os.path.abspath(str(b)))
    if na == nb:
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def refuse_live(db_path: str, live: str = LIVE_DB) -> None:
    """Stop before anything else when ``db_path`` is the annotation tool's database."""
    if _same_file(db_path, live):
        raise PackError(f"{db_path} is the live database; pass a copy "
                        f"(sqlite3 backup API into D:/DataSet/.cache/tmp/w4a/)")
    if not Path(db_path).is_file():
        raise PackError(f"{db_path}: no such database file")


def work_copy(db_path: str, work: Path) -> Path:
    """A backup-API copy of the input, made through a read-only connection."""
    work.mkdir(parents=True, exist_ok=True)
    dest = work / "vlmpack_work.sqlite"
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


#: Written first, removed last: a directory with it is a build that did not finish.
BUILDING = ".vlmpack_building"


def _prepare_out(out: Path, force: bool) -> None:
    if out.exists() and any(out.iterdir()):
        marker = out / "manifest.json"
        is_pack = (out / BUILDING).is_file()
        if marker.is_file():
            # the real manifest's "package" key sits past the first few hundred bytes
            # (the file list comes first), so read the JSON instead of a text prefix
            try:
                is_pack = is_pack or json.loads(marker.read_text(encoding="utf-8")).get("package") == PACK_NAME
            except (OSError, ValueError):
                pass
        if not force:
            raise PackError(f"{out} is not empty; pass --force to replace a pack there")
        if not is_pack:
            raise PackError(f"{out} is not empty and holds no {PACK_NAME} manifest; "
                            f"refusing to delete it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / BUILDING).write_text("a build of scripts/export_vlmpack.py is in progress\n",
                                encoding="utf-8")


def _drop_pycache(root: Path) -> None:
    """No ``__pycache__`` ships: a .pyc carries a timestamp, and the manifest hashes."""
    for cache in sorted(root.rglob("__pycache__"), reverse=True):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)


def _git_head(repo: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


# --------------------------------------------------------------------------- #
# 1. the export
# --------------------------------------------------------------------------- #
def pack_units(views: Sequence[str], multi: bool = True) -> list[str]:
    """The export units of a pack: the camera views asked for, then ``multi`` (the
    action tasks, every camera at once) when the scanner -- its primary -- is among them."""
    units = [str(v) for v in views]
    if multi and MULTI_PRIMARY in units and MULTI_VIEW not in units:
        units.append(MULTI_VIEW)
    return units


def export_views(db: Db, tax: Taxonomy, desktops: Sequence[int], views: Sequence[str],
                 data_dir: Path, cache_dir: Optional[str], allow_conflicts: bool,
                 log=print) -> dict[str, dict]:
    """``export_vlm`` per unit into ``data/``; returns ``unit -> summary``.

    ``views`` are export units (:func:`pack_units`): camera views (single-view
    perception and V15) and ``multi`` (the action tasks); each writes its own task
    set (:func:`tda.core.export.vlm.unit_tasks`).
    """
    service = TruthService(db, tax, cache_dir=cache_dir)
    out: dict[str, dict] = {}
    for view in views:
        if is_component_view(view):
            raise PackError(f"{view} is a components-gallery view: no VLM questions")
        target = data_dir / f"vlm_{view}.jsonl"
        log(f"[vlmpack] export {view}: {len(desktops)} desktops ...")
        stats = export_vlm(db, tax, list(desktops), view, str(target), tasks=unit_tasks(view),
                           only_verified=False, truth=service, allow_conflicts=allow_conflicts,
                           balance=True, split="all")
        if not stats.get("records"):
            if view == MULTI_VIEW:
                raise PackError(f"{view}: the export wrote no record")
            # a camera unit holds only confirmed perception and the V15 pairs it comes
            # first in: a view nothing of which is confirmed (and the last view, rs) can
            # be empty. Its pictures still ship through the multi unit.
            log(f"[vlmpack] export {view}: no record (nothing confirmed in this view); "
                f"the unit is left out")
            for path in (target, *vlm_siblings(str(target))):
                Path(path).unlink(missing_ok=True)
            continue
        log(f"[vlmpack] export {view}: {stats['records']} records, {stats['images']} images"
            + (f", {stats['open_conflicts']} frames with open conflicts (unverified)"
               if stats.get("open_conflicts") else ""))
        out[view] = stats
    return out


def _summary_for_manifest(stats: dict) -> dict:
    keys = ("records", "images", "by_task", "by_source", "by_truth_source", "by_split",
            "open_conflicts", "illegal_steps", "excluded_desktops", "v10_excluded",
            "unlabelled_view_desktops", "records_without_confirmed_view", "v12_by_kind",
            "v15_moment_by_view", "v16_corruptions", "v6_options",
            "image_undecidable_by_task", "shortcuts", "v6_option_shortcut", "tasks",
            "primary_view", "side_views", "by_form", "non_stepwise", "skip_reasons")
    return {k: stats.get(k) for k in keys if k in stats}


# --------------------------------------------------------------------------- #
# 2. the images
# --------------------------------------------------------------------------- #
def manifest_entries(jsonl_paths: Iterable[Path]) -> dict[str, dict]:
    """``relative path -> {desktop, view, step}`` over several manifests."""
    out: dict[str, dict] = {}
    for path in jsonl_paths:
        for row in IO.load_manifest(path).values():
            out.setdefault(row["path"], {"desktop": int(row["desktop"]),
                                         "view": row["view"], "step": int(row["step"])})
    return out


def images_needed(data_dir: Path, views: Sequence[str], mode: str,
                  eval_dir: Optional[Path]) -> dict[str, dict]:
    if mode == "none":
        return {}
    if mode == "all":
        return manifest_entries(data_dir / f"vlm_{v}.jsonl" for v in views)
    if mode == "eval":
        if eval_dir is None:
            raise ValueError("--images eval needs the evaluation subset")
        return manifest_entries(p for p in (eval_dir / f"{v}.jsonl" for v in views)
                                if p.exists())
    raise ValueError(f"--images {mode!r}: one of {', '.join(IMAGE_MODES)}")


def copy_images(cache_dir: str, needed: dict[str, dict], out: Path, log=print
                ) -> dict[str, dict]:
    """Copy every needed frame under ``out/images``; sha256 while writing.

    Checks **all** sources and the free space first: a build that copies 20 GB
    and then finds the last frame missing has wasted an evening.
    """
    root = Path(cache_dir)
    missing = [rel for rel in sorted(needed) if not (root / rel).is_file()]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise PackError(f"{len(missing)} of {len(needed)} images are not in {cache_dir}: "
                        f"{shown}. Build the cache for those views first "
                        f"(python -m tda.cli build-cache --views <view>)")
    total = sum((root / rel).stat().st_size for rel in needed)
    dest_root = out / "images"
    dest_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest_root).free
    if total > free:
        raise PackError(f"the images need {total / 2**30:.1f} GB but {out} has only "
                        f"{free / 2**30:.1f} GB free")
    log(f"[vlmpack] copying {len(needed)} images ({total / 2**30:.2f} GB) from {cache_dir}")
    files: dict[str, dict] = {}
    done = 0
    for rel in sorted(needed):
        src = root / rel
        dst = dest_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with open(src, "rb") as fh, open(dst, "wb") as oh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
                oh.write(chunk)
                size += len(chunk)
        files["images/" + rel] = {"bytes": size, "sha256": digest.hexdigest(),
                                  **needed[rel]}
        done += 1
        if done % 1000 == 0:
            log(f"[vlmpack]   {done}/{len(needed)} images")
    return files


# --------------------------------------------------------------------------- #
# 3. the tools
# --------------------------------------------------------------------------- #
def require_tools(repo: Path = REPO) -> None:
    """Stop (:class:`PackError`) when a tool the pack ships is not in ``repo/scripts``,
    naming the redesign's glossary / audit carry-over first: a pack without them would
    convert records without their glossary and could not carry the audit over."""
    missing = [name for name in REDESIGN_TOOLS if not (Path(repo) / "scripts" / name).is_file()]
    if missing:
        raise PackError(f"the redesign's tools {', '.join(missing)} are missing from "
                        f"{Path(repo) / 'scripts'}: the converter needs its glossary and the "
                        f"audit its carry-over; build from a tree that has them")
    for name in TOOLS:
        src = Path(repo) / "scripts" / name
        if not src.is_file():
            raise PackError(f"tool {src} is missing")


def vendor_tools(tools: Path, repo: Path = REPO) -> list[str]:
    """Copy the scripts and the scorer's ``tda`` closure under ``tools/``.

    Two files are generated rather than copied: ``tda/core/export/__init__.py``
    (the repository's imports the COCO and VLM exporters, which need the
    database layer) and ``tda/core/export/vlm_tasks.py`` (3 000 lines of
    generators, of which the references need one dictionary).
    """
    written: list[str] = []
    require_tools(repo)
    tools.mkdir(parents=True, exist_ok=True)
    for name in TOOLS:
        src = repo / "scripts" / name
        if not src.is_file():
            raise PackError(f"tool {src} is missing")
        shutil.copyfile(src, tools / name)
        written.append(name)
    for rel in VENDORED + VENDORED_CONFIGS:
        src = repo / rel
        if not src.is_file():
            raise PackError(f"vendored file {src} is missing")
        dst = tools / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        written.append(rel)
    # W4-B's model driver, its Slurm templates and requirements travel with the pack
    # (``tools/vlm_run/``), the run guide beside the card (``docs/vlm_run_hprc.md``):
    # the pack is what gets copied to HPRC, nothing else.
    driver = repo / "scripts" / "vlm_run"
    if not driver.is_dir():
        raise PackError(f"driver folder {driver} is missing")
    for src in sorted(p for p in driver.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        rel = (Path("vlm_run") / src.relative_to(driver)).as_posix()
        dst = tools / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        written.append(rel)
    guide = repo / "docs" / "vlm_run_hprc.md"
    if not guide.is_file():
        raise PackError(f"run guide {guide} is missing")
    docs = tools.parent / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(guide, docs / "vlm_run_hprc.md")
    write_text(tools / "tda" / "core" / "export" / "__init__.py",
               '"""The scorer\'s export subset (vendored by scripts/export_vlmpack.py).\n\n'
               'Only :mod:`tda.core.export.vlm_check` is here; the exporters stay in the\n'
               'annotation tool.\n"""\n')
    written.append("tda/core/export/__init__.py")
    write_text(tools / "tda" / "core" / "export" / "vlm_tasks.py", vlm_tasks_stub())
    written.append("tda/core/export/vlm_tasks.py")
    write_text(tools / "requirements.txt", REQUIREMENTS)
    written.append("requirements.txt")
    return written


def vlm_tasks_stub() -> str:
    """The few names of :mod:`tda.core.export.vlm_tasks` the scorer side uses.

    The real module is 3 000 lines of generators over the database layer. The
    scorer needs ``SCREW_PRESENCE`` (the state-oracle reference) and
    ``class_label`` (V1 grading, imported lazily by ``vlm_check``) with its
    helper; their **source is read off the live module** at build time, so the
    stub cannot drift from the tool, and the build's self-check exercises it.
    """
    import inspect

    from tda.core.export import vlm_tasks as T

    parts = [
        '"""The names of tda.core.export.vlm_tasks the scorer side uses (vendored).',
        "",
        "Generated by scripts/export_vlmpack.py from the live module's source; the",
        "generators themselves stay in the annotation tool.",
        '"""',
        "from __future__ import annotations",
        "",
        "from typing import Optional",
        "",
        "from tda.core.taxonomy import Taxonomy",
        "",
        f"INITIALISMS = frozenset({json.dumps(sorted(T.INITIALISMS))})",
        f"SCREW_PRESENCE = {json.dumps(T.SCREW_PRESENCE, sort_keys=True)}",
        "",
        "",
        inspect.getsource(T._words).rstrip("\n"),
        "",
        "",
        inspect.getsource(T.class_label).rstrip("\n"),
        "",
    ]
    return "\n".join(parts)


def _isolated_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # nothing lands in the pack but what is listed
    return env


def smoke_import(tools: Path, python: str = sys.executable) -> None:
    """Import every tool from the pack alone, in a subprocess with ``-I``."""
    code = ("import sys, pathlib; tools = pathlib.Path(sys.argv[1]); "
            "sys.path.insert(0, str(tools)); "
            "import vlm_io, vlm_to_chat, vlm_score, vlm_references, vlm_overlay, vlm_sample, "
            "vlm_eval_set, vlm_selfcheck, vlm_audit_sheet, vlm_audit_summary, "
            "vlm_human_sheet, vlm_glossary, vlm_audit_carry, tda, "
            "tda.core.export.vlm_check as C; "
            "vlm_overlay.taxonomy_classes(); "
            "from tda.core.taxonomy import load_taxonomy; load_taxonomy(); "
            "where = pathlib.Path(tda.__file__).resolve().parent; "
            "assert where == (tools / 'tda').resolve(), where; print(where)")
    proc = subprocess.run([python, "-I", "-B", "-c", code, str(tools)], capture_output=True,
                          text=True, cwd=str(tools.parent), env=_isolated_env())
    if proc.returncode != 0:
        raise PackError(f"the pack's tools do not import on their own:\n{proc.stderr}")


def run_selfcheck(out: Path, views: Sequence[str], work: Path,
                  python: str = sys.executable, log=print) -> dict:
    """``tools/vlm_selfcheck.py`` on the subset, from the pack, in a subprocess."""
    truth = [out / vlm_eval_set.EVAL_VERSION / f"{v}.jsonl" for v in views]
    truth = [p for p in truth if p.exists()]
    target = work / "selfcheck"
    cmd = [python, "-I", "-B", str(out / "tools" / "vlm_selfcheck.py"), "--truth",
           *map(str, truth), "--out", str(target)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work),
                          env=_isolated_env())
    for line in proc.stdout.splitlines():
        log(line)
    if proc.returncode != 0:
        raise PackError(f"the self-check failed (exit {proc.returncode}):\n{proc.stderr}")
    report = json.loads((target / "selfcheck.json").read_text(encoding="utf-8"))
    where = Path(report["tda_module"]).resolve()
    if where != (out / "tools" / "tda").resolve():
        raise PackError(f"the self-check imported tda from {where}, not from the pack")
    return {"ok": report["ok"], "files": [
        {"file": f["file"], "records": f["records"],
         "tasks": {t: row["strict_mean"] for t, row in f["rows"].items()}}
        for f in report["files"]]}


# --------------------------------------------------------------------------- #
# 3b. the masks and the overlay variant (W4-C)
# --------------------------------------------------------------------------- #
def masks_sidecars(data_dir: Path, eval_dir: Path, views: Sequence[str], coco_root: str,
                   log=print) -> dict:
    """``data/vlm_<unit>.masks.jsonl`` and ``eval_v3/<unit>.masks.jsonl`` from the COCO.

    One pass over the scanner pack for all views (an image id is the same in
    every export file), then one sidecar per view with the rows its manifest
    names. Returns per view: confirmed frames, frames with masks, masks, and the
    confirmed frames the COCO does not cover (the OAK frames of a V15 pair).
    """
    classes = vlm_overlay.taxonomy_classes(str(REPO / "configs" / "taxonomy.yaml"))
    manifest: dict[str, dict] = {}
    frames: dict[str, dict] = {}
    per_view: dict[str, tuple[dict, dict]] = {}
    for view in views:
        export = data_dir / f"vlm_{view}.jsonl"
        m, f = vlm_overlay.load_manifest(export), vlm_overlay.load_frames(export)
        per_view[view] = (m, f)
        manifest.update(m)
        frames.update(f)
    try:
        rows, stats = vlm_overlay.masks_from_coco(coco_root, manifest, frames, log)
    except vlm_overlay.OverlayError as exc:
        raise PackError(f"masks: {exc}") from exc
    by_image = {r["image"]: r for r in rows}
    out: dict = {"source": stats["source"], "views": {}}
    for view in views:
        m, f = per_view[view]
        mine = [by_image[i] for i in sorted(m) if i in by_image]
        mine.sort(key=lambda r: (r["desktop"], r["view"], r["step"]))
        confirmed = [i for i, fr in f.items() if fr.get("confirmed") and i in m]
        missing = sorted(f"D{m[i]['desktop']:02d} {m[i]['view']} s{m[i]['step']:03d}"
                         for i in confirmed if i not in by_image)
        target = data_dir / f"vlm_{view}.masks.jsonl"
        vlm_overlay.write_masks(target, mine, classes, stats["source"])
        row = {"confirmed_frames": len(confirmed), "frames_with_masks": len(mine),
               "masks": sum(len(r["masks"]) for r in mine),
               "confirmed_without_masks": len(missing),
               "confirmed_without_masks_examples": missing[:20]}
        subset = eval_dir / f"{view}.jsonl"
        if subset.exists():
            used = {img for r in IO.read_jsonl(subset) for img in r["prompt"].get("images") or []}
            n = vlm_overlay.cut_masks(target, eval_dir / f"{view}.masks.jsonl", used) - 1
            row["eval_frames_with_masks"] = n
        out["views"][view] = row
        log(f"[vlmpack] masks {view}: {len(mine)} of {len(confirmed)} confirmed frames, "
            f"{row['masks']} masks" + (f", {len(missing)} confirmed frames without masks"
                                       if missing else ""))
    return out


def render_overlays(out: Path, eval_dir: Path, views: Sequence[str], log=print) -> dict:
    """The subset's ``mask_overlay`` variant per unit: ``eval_v3/mask_overlay/<unit>.jsonl``
    and the painted frames under ``overlays/``; returns the painter's summary per view."""
    result: dict = {}
    for view in views:
        truth = eval_dir / f"{view}.jsonl"
        if not truth.exists() or not vlm_overlay.sidecar(truth, vlm_overlay.MASKS_SUFFIX).exists():
            continue
        try:
            summary = vlm_overlay.render_export(
                truth, eval_dir / "mask_overlay" / f"{view}.jsonl", out / "images",
                out / "overlays", taxonomy=str(REPO / "configs" / "taxonomy.yaml"),
                log=lambda line: log(line.replace("[vlm-overlay]", "[vlmpack] overlay")))
        except vlm_overlay.OverlayError as exc:
            raise PackError(f"overlay {view}: {exc}") from exc
        result[view] = {k: summary[k] for k in ("records_in", "records", "skipped", "overlays",
                                                 "masks_painted_by_class", "alpha", "outline_px")}
    return result


# --------------------------------------------------------------------------- #
# 4. composition (for the card)
# --------------------------------------------------------------------------- #
def write_reasoning(data_dir: Path, out_dir: Path, units: Sequence[str]) -> dict:
    """``reasoning/<unit>.jsonl``: the training side product of every record of ``data/``
    (owner 2026-11-04) -- ``{id, task, question, answer, reasoning}``: the question as the
    model reads it (the options lettered), the gold reply with every part in plain names
    (no ordinal) and the record's ``rationale.text``. Returns ``{unit: rows}``."""
    from tda.core.export.vlm_check import LETTERS, gold_reply

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for unit in units:
        src = data_dir / f"vlm_{unit}.jsonl"
        if not src.exists():
            continue
        worlds, _ = IO.load_oracle(src)

        def rows():
            for r in IO.read_jsonl(src):
                prompt = r["prompt"]
                question = prompt["question"]
                if prompt.get("options"):
                    question += "\n" + "\n".join(
                        f"{LETTERS[i]}. {o['verb']} {o['target_label']}"
                        for i, o in enumerate(prompt["options"]))
                answer = json.loads(gold_reply(r, worlds.get(IO.desktop_of(r)), plain=True))
                yield {"id": r["id"], "task": prompt["task"], "question": question,
                       "answer": answer,
                       "reasoning": (r["label"].get("rationale") or {}).get("text") or ""}
        counts[unit] = IO.write_jsonl(out_dir / f"{unit}.jsonl", rows())
    return counts


def composition(data_dir: Path, views: Sequence[str]) -> dict:
    """What the card says beyond ``summary.json``: V2 by class, the optical drive cages."""
    v2: Counter = Counter()
    v2_prior_wrong = 0
    v2_total = 0
    bracket_desktops: set[int] = set()
    desktops: set[int] = set()
    for view in views:
        path = data_dir / f"vlm_{view}.jsonl"
        for rec in IO.read_jsonl(path):
            if rec["prompt"]["task"] == "V2":
                v2[vlm_eval_set.v2_class(rec)] += 1
                v2_total += 1
                if rec["label"].get("prior_is_correct") is False:
                    v2_prior_wrong += 1
        oracle = IO.sidecar(path, "oracle.jsonl")
        if oracle.exists():
            for row in IO.read_jsonl(oracle):
                if row.get("type") != "desktop":
                    continue
                desktops.add(int(row["desktop"]))
                for inst in row.get("instances", {}).values():
                    if inst.get("class") == "drive_cage" and \
                            (inst.get("attrs") or {}).get("of") == "optical":
                        bracket_desktops.add(int(row["desktop"]))
    return {"v2_by_class": dict(sorted(v2.items(), key=lambda kv: (-kv[1], kv[0]))),
            "v2_records": v2_total, "v2_prior_wrong": v2_prior_wrong,
            "optical_bracket_desktops": sorted(bracket_desktops),
            "desktops": sorted(desktops)}


# --------------------------------------------------------------------------- #
# the build
# --------------------------------------------------------------------------- #
def build_pack(db_path: str, out_dir: str, *, cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
               work_dir: Optional[str] = None, views: Sequence[str] = VIEWS,
               desktops: Optional[Sequence[int]] = None, seed: int = vlm_eval_set.SEED,
               images: str = "all", allow_conflicts: bool = False, force: bool = False,
               selfcheck: bool = True, audit: int = 300, tax: Optional[Taxonomy] = None,
               repo: Path = REPO, python: str = sys.executable, log=print,
               live_db: str = LIVE_DB, coco_root: Optional[str] = None,
               overlays: str = "eval", multi: bool = True) -> dict:
    """Build the pack at ``out_dir``; returns the manifest's summary.

    ``views`` are the camera views exported as single-view units; ``multi`` adds the
    ``multi`` unit (the action tasks, every camera at once) when the scanner is among
    them (:func:`pack_units`). Every later step -- the subset, the images, the masks,
    the overlays, the self-check -- runs over all units.
    """
    if images not in IMAGE_MODES:
        raise ValueError(f"images={images!r}: one of {', '.join(IMAGE_MODES)}")
    if overlays not in OVERLAY_MODES:
        raise ValueError(f"overlays={overlays!r}: one of {', '.join(OVERLAY_MODES)}")
    if coco_root and not Path(coco_root).is_dir():
        raise PackError(f"--coco-root {coco_root}: no such directory")
    refuse_live(db_path, live_db)
    require_tools(repo)  # before an hour of export, not after it
    views = [str(v) for v in views]
    if not views:
        raise PackError("no view asked for")
    units = pack_units(views, multi)
    out = Path(out_dir)
    work = Path(work_dir) if work_dir else Path(DEFAULT_WORK)
    _prepare_out(out, force)
    tax = tax or load_taxonomy()
    db = Db(str(work_copy(db_path, work)))
    try:
        wanted = sorted(int(d) for d in desktops) if desktops else db.desktop_ids()
        for d in wanted:
            if db.get_desktop(d) is None:
                raise PackError(f"desktop {d} is not in {db_path}")
        if not wanted:
            raise PackError(f"{db_path} has no desktop")
        data_dir = out / "data"
        exports = export_views(db, tax, wanted, units, data_dir, cache_dir, allow_conflicts,
                               log)
    finally:
        db.close()
    if not exports:
        raise PackError("no unit wrote a record")
    empty_units = [u for u in units if u not in exports]
    units = [u for u in units if u in exports]
    # the reasoning side product (owner 2026-11-04): question, gold reply, reasoning chain
    reasoning = write_reasoning(data_dir, out / "reasoning", units)
    log(f"[vlmpack] reasoning: " + " ".join(f"{u} {n}" for u, n in reasoning.items()))

    # the frozen subset and its audit sample
    eval_dir = out / vlm_eval_set.EVAL_VERSION
    log(f"[vlmpack] drawing {vlm_eval_set.EVAL_VERSION} (seed {seed}) ...")
    eval_result = vlm_eval_set.build([data_dir / f"vlm_{v}.jsonl" for v in units], eval_dir,
                                     seed=seed, audit=audit)
    log(f"[vlmpack] {vlm_eval_set.EVAL_VERSION}: {eval_result['records']} records: "
        + " ".join(f"{r['task']}={r['records']}" for r in eval_result["rows"]))
    # the subset's own text-only bounds (the exports' are over the whole pool); kept out
    # of table.json, which tools/vlm_eval_set.py re-creates byte for byte without them
    eval_shortcuts = shortcut_report(
        r for v in units if (eval_dir / f"{v}.jsonl").exists()
        for r in IO.read_jsonl(eval_dir / f"{v}.jsonl"))
    # the fillable audit sheet (W4-E): static pages + CSV skeleton beside audit.jsonl,
    # image paths relative to the page, so the pack travels whole
    audit_dir = eval_dir / "audit"
    if eval_result.get("audit") and (audit_dir / "audit.jsonl").is_file():
        import vlm_audit_sheet

        sheet = vlm_audit_sheet.build(
            list(IO.read_jsonl(audit_dir / "audit.jsonl")),
            [p for p in (eval_dir / f"{v}.jsonl" for v in units) if p.exists()],
            out / "images", audit_dir, batch=AUDIT_BATCH, name="audit")
        eval_result["audit"]["sheet"] = {"files": sheet["files"], "batches": sheet["batches"],
                                         "batch_size": sheet["batch_size"]}
        log(f"[vlmpack] audit sheet: {sheet['items']} items in {sheet['batches']} pages")

    # the pictures
    needed = images_needed(data_dir, units, images, eval_dir)
    image_files = copy_images(cache_dir or "", needed, out, log) if needed else {}

    # the confirmed masks (from the scanner pack's COCO) and the subset's overlay variant
    references: dict = {"masks": None, "mask_overlay": None, "overlay_mode": overlays}
    if coco_root:
        log(f"[vlmpack] masks from {coco_root} ...")
        references["masks"] = masks_sidecars(data_dir, eval_dir, units, coco_root, log)
        if overlays == "eval" and images != "none":
            log("[vlmpack] painting the subset's mask overlays ...")
            references["mask_overlay"] = render_overlays(out, eval_dir, units, log)
        elif overlays == "eval":
            references["mask_overlay_skipped"] = "no images in this pack (--images none)"
    else:
        references["skipped"] = "no --coco-root: no masks sidecar, no overlay variant"

    # the tools, imported from the pack alone, then the self-check from the pack
    tools = out / "tools"
    tool_files = vendor_tools(tools, repo)
    smoke_import(tools, python)
    check = run_selfcheck(out, units, work, python, log) if selfcheck else None

    comp = composition(data_dir, units)
    summary = {
        "package": PACK_NAME, "pack_version": PACK_VERSION,
        "eval_version": vlm_eval_set.EVAL_VERSION,
        "views": views, "units": units, "desktops": wanted,
        "tasks": [t for t in ALL_TASKS if any(t in unit_tasks(u) for u in units)],
        "unit_tasks": {u: list(unit_tasks(u)) for u in units},
        "empty_units": empty_units,
        "image_mode": images, "images": len(image_files),
        "image_bytes": sum(f["bytes"] for f in image_files.values()),
        "allow_conflicts": bool(allow_conflicts),
        "seed": seed,
        "source_commit": _git_head(repo),
        "exports": {v: _summary_for_manifest(s) for v, s in exports.items()},
        "eval": eval_result,
        "eval_shortcuts": eval_shortcuts,
        "reasoning": {"files": {u: f"reasoning/{u}.jsonl" for u in reasoning},
                      "rows": reasoning},
        "composition": comp,
        "references": references,
        "selfcheck": check,
        "tools": tool_files,
    }
    write_text(out / "LICENSE_NOTE.txt", LICENSE_NOTE)
    write_text(out / "README.md", readme_text(summary))
    _drop_pycache(out)
    (out / BUILDING).unlink(missing_ok=True)
    files = dict(image_files)
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = path.relative_to(out).as_posix()
        if rel == "manifest.json" or rel in files:
            continue
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256_of(path)}
    write_json(out / "manifest.json", dict(summary, files=files))
    log(f"[vlmpack] {len(files)} files -> {out}")
    return summary


# --------------------------------------------------------------------------- #
# the data card
# --------------------------------------------------------------------------- #
def _task_sort(task: str) -> tuple:
    digits = "".join(ch for ch in task[1:] if ch.isdigit())
    return (int(digits or 0), task)


def _n(count: int, word: str, plural: Optional[str] = None) -> str:
    return f"{count} {word if count == 1 else (plural or word + 's')}"


def _units(summary: dict) -> list[str]:
    """The export units of a summary (camera views, then multi); old summaries: the views."""
    return list(summary.get("units") or summary["views"])


def _counts_table(summary: dict) -> str:
    views = _units(summary)
    tasks = sorted({t for s in summary["exports"].values() for t in s["by_task"]},
                   key=_task_sort)
    head = "| task | layer | " + " | ".join(views) + " | all | what it asks |"
    sep = "|---|---|" + "---|" * len(views) + "---|---|"
    lines = [head, sep]
    for t in tasks:
        cells = [str(summary["exports"][v]["by_task"].get(t, 0)) for v in views]
        total = sum(summary["exports"][v]["by_task"].get(t, 0) for v in views)
        lines.append(f"| {t} | {CAPABILITY_LAYERS.get(t, '-')} | " + " | ".join(cells)
                     + f" | {total} | {TASK_MEANINGS.get(t, '')} |")
    totals = [str(summary["exports"][v]["records"]) for v in views]
    lines.append("| **all** | | " + " | ".join(totals)
                 + f" | {sum(s['records'] for s in summary['exports'].values())} | |")
    return "\n".join(lines)


def _eval_table(summary: dict) -> str:
    rows = summary["eval"]["rows"]
    lines = ["| task | target | units | records | pool (records) | by view | desktops | "
             "max desktop share |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        views = ", ".join(f"{v} {n}" for v, n in r["by_view"].items())
        lines.append(f"| {r['task']} | {r['target']} | {r['units']} | {r['records']} | "
                     f"{r['pool_records']} | {views} | {r['desktops']} | "
                     f"{r['max_desktop_share']:.1%} |")
    return "\n".join(lines)


def _shortcut_table(summary: dict) -> str:
    views = _units(summary)
    tasks = sorted({t for s in summary["exports"].values()
                    for t in (s.get("shortcuts") or {})}, key=_task_sort)
    lines = ["| task | " + " | ".join(views) + " |", "|---|" + "---|" * len(views)]
    for t in tasks:
        cells = []
        for v in views:
            row = (summary["exports"][v].get("shortcuts") or {}).get(t)
            if not row:
                cells.append("-")
                continue
            best = max((row.get(k) or 0.0) for k in ("verb", "class", "verb_class",
                                                    "template", "majority", "group",
                                                    "match_class", "group_class"))
            cells.append(f"{best:.1%}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    mc = []
    for v in views:
        opt = summary["exports"][v].get("v6_option_shortcut") or {}
        if opt.get("records"):
            mc.append(f"{v} {opt['shortcut']:.1%}")
    if mc:
        random = next(opt["random"] for v in views
                      for opt in [summary["exports"][v].get("v6_option_shortcut") or {}]
                      if opt.get("records"))
        lines += ["", f"The V6 multiple choice has its own shortcut, the option's verb alone: "
                  + ", ".join(mc) + f" (random {random:.1%})."]
    v4 = (summary.get("eval_shortcuts") or {}).get("V4")
    if v4:
        ev = summary.get("eval_version") or vlm_eval_set.EVAL_VERSION
        lines += ["", f"V4 in `{ev}` ({v4['records']} records, drawn balanced within each verb + "
                  f"class): majority {v4['majority']:.1%}, class {v4['class']:.1%}, match "
                  f"class {v4['match_class']:.1%}, group {v4['group']:.1%}, verb "
                  f"{v4['verb']:.1%}, verb + class {v4['verb_class']:.1%}."]
    return "\n".join(lines)


def _formats() -> str:
    return "\n".join(f"* `{key}`: {text}" for key, text in RESPONSE_FORMATS.items()
                     if not key.startswith("V5"))


def _rules_text(summary: dict) -> str:
    """The card's paragraph on the rules the action questions are graded by, and on the
    reasoning side product."""
    non_step = sorted({int(d) for s in summary["exports"].values()
                       for d, why in (s.get("skip_reasons") or {}).items()
                       if isinstance(why, dict) and why.get("graph_tasks") == "not_stepwise"}
                      | {int(d) for d in summary.get("non_stepwise") or []})
    reasoning = (summary.get("reasoning") or {}).get("rows") or {}
    text = (
        "**The rules the action questions are graded by** (the owner's rulings of "
        "2026-11-04). V4, V6, V7 and V16 rest on the stored constraint graph -- the "
        "general rules R1-R17 and the per-machine rulings -- and on two rules of the moment, "
        "computed from the scanner frame of the step and never stored: **dynamic "
        "occlusion** (a part Y above part X that covers at least 20 % of X's whole shape "
        "has to be taken out, moved aside or opened before X can be touched; cables never "
        "count, a drive cage never counts over a plug, a bracket that never moves never "
        "counts, owner exceptions in `configs/occlusion_exceptions.yaml`) and **loose parts "
        "first** (a part lying loose in the case from the start comes out before anything "
        "else is done). V6's gold is every action possible now under them (the frontier), "
        "not the one the operator did; V7's required steps are the minimal prerequisites "
        "under them, not the demonstrated sequence; V16's validity is a replay of the "
        "sequence under them, never \"it was demonstrated\". "
        + (f"D{', D'.join(f'{d:02d}' for d in non_step)} took several parts out at once "
           f"(not stepwise): no V4 / V6 / V7 / V16 there. " if non_step else
           "D66 took several parts out at once (not stepwise): no V4 / V6 / V7 / V16 there. "))
    if reasoning:
        text += ("\n\n**Reasoning side product.** `reasoning/<unit>.jsonl` holds one row per "
                 "record of `data/` -- `{\"id\", \"task\", \"question\", \"answer\", "
                 "\"reasoning\"}`: the question as the model reads it, the gold reply with "
                 "every part named plainly (no ordinal) and the reasoning chain in words "
                 "(what is seen, which rule applies, so the answer) -- for training only; "
                 "nothing in it is shown to a model under evaluation. Rows: "
                 + ", ".join(f"{u} {n}" for u, n in reasoning.items()) + ".")
    return text


def _known_issues(summary: dict) -> str:
    ev = summary.get("eval_version") or vlm_eval_set.EVAL_VERSION
    items = []
    comp = summary["composition"]
    v2 = comp["v2_by_class"]
    if comp["v2_records"]:
        top = next(iter(v2.items()))
        share = top[1] / comp["v2_records"]
        items.append(
            f"**V2 class skew.** After the per-desktop class-prior balancing (no answer above "
            f"50 % of a class's questions on a desktop; a class whose answer never changes "
            f"keeps none) the V2 pool is {comp['v2_records']} questions of which "
            f"`{top[0]}` is {top[1]} ({share:.0%}); classes whose state is the same in "
            f"every frame of a desktop (CPU, motherboard, RAM modules, expansion cards, "
            f"most screws) have few or no state questions. `{ev}` draws V2 **equally "
            f"over classes** (and half `prior_is_correct = false`), so its V2 is not the "
            f"pool's mix -- compare models on it, not on the pool's class frequencies. "
            f"Full pool by class: " + ", ".join(f"{k} {n}" for k, n in list(v2.items())[:12])
            + (" ..." if len(v2) > 12 else "") + ".")
    unconfirmed = {v: s.get("records_without_confirmed_view") or {}
                   for v, s in summary["exports"].items()}
    parts = [f"{v}: " + ", ".join(f"{t} {n}" for t, n in sorted(d.items(), key=lambda kv:
                                                                 _task_sort(kv[0])))
             for v, d in unconfirmed.items() if d]
    empty = list(summary.get("empty_units") or [])
    if parts or empty:
        items.append(
            "**Unconfirmed views.** On OAK1, OAK2 and RealSense almost no frame has "
            "confirmed masks (the scanner is the annotated view). Their own files hold V15 "
            "(it needs no mask) and V1 / V2 / V8 only where a frame is confirmed; such "
            "records say `label.view_confirmed: false`, carry no evidence boxes and no "
            "visibility. The action tasks are not asked per side view any more: in "
            "`vlm_multi.jsonl` the side cameras' pictures are shown beside the scanner's, "
            "whose confirmed frame carries the truth, the evidence and the marks."
            + (" Records without a confirmed view: " + "; ".join(parts) + "." if parts else "")
            + (f" No record at all, so no file of their own: {', '.join(empty)} (their "
               f"pictures ship through `vlm_{MULTI_VIEW}.jsonl`)." if empty else ""))
    items.append(
        "**V15 is the `moment` variant only.** It asks whether two pictures from two cameras "
        "show the same moment; it needs no masks, so it exists on all four views, but a "
        "pair whose camera clocks disagree by more than 20 s is not asked (OAK/RealSense; "
        "the scanner's timestamps are not capture times and are not checked). The `cross` "
        "variant (the state of a part one view cannot see, read from another) needs both "
        "views confirmed and is not in this release. Each unordered view pair is written "
        "once, in the file of the view that comes first (scan < oak1 < oak2 < rs).")
    bracket = comp["optical_bracket_desktops"]
    if bracket:
        items.append(
            f"**The optical drive cage.** On {len(bracket)} desktops (D"
            + ", D".join(f"{d:02d}" for d in bracket) + ") the optical drive is never taken "
            "out -- installed to the end, or only opened / pushed out -- and it is annotated "
            "together with the bracket it sits in as class `drive_cage` with "
            "`attrs.of = \"optical\"` (the oracle's `instances[key].attrs.of`). The questions "
            "call it **\"Optical drive cage N\"** (an HDD / SSD cage stays \"Drive cage N\"); "
            "class `optical_drive` is only a drive that is taken out. The scorer grounds an "
            "answer by that wording (or the key) and does **not** accept \"optical drive\" "
            "for the cage (owner, 2026-11-03).")
    illegal = {v: s.get("illegal_steps") or {} for v, s in summary["exports"].items()}
    scan_illegal = illegal.get(MULTI_VIEW) or illegal.get(summary["views"][0]) or {}
    if scan_illegal:
        listed = ", ".join(f"D{int(d):02d} {sorted(int(s) for s in steps)}"
                           for d, steps in sorted(scan_illegal.items(), key=lambda kv:
                                                  int(kv[0])))
        items.append(
            f"**Illegal steps.** On {_n(len(scan_illegal), 'desktop')} the logged action of a "
            f"step contradicts the constraint graph; V4 / V6 / V16 say nothing about the "
            f"moment before it (`summary.json` `illegal_steps`): {listed}.")
    v10 = sorted({int(d) for s in summary["exports"].values()
                  for d in (s.get("v10_excluded") or {})})
    if v10:
        items.append(
            f"**V10 excluded desktops.** D" + ", D".join(f"{d:02d}" for d in v10)
            + ": their logs name an action target nobody resolved, and V10 is scored "
            "exhaustively, so they have no V10 question.")
    excluded = sorted({int(d) for s in summary["exports"].values()
                       for d in (s.get("excluded_desktops") or {})})
    if excluded:
        items.append("**Excluded desktops (graph tasks).** D"
                     + ", D".join(f"{d:02d}" for d in excluded)
                     + ": no usable constraint graph; V4 / V6 / V7 / V16 are not asked.")
    sources = Counter()
    for s in summary["exports"].values():
        sources.update(s.get("by_truth_source") or {})
    failed = sources.get("failed_attempt", 0)
    if failed == 0:
        items.append(
            "**No `failed_attempt` record.** No V4 question of this export comes from a real "
            "failed attempt in the log; every infeasible answer is the graph's or the "
            "state's.")
    elif failed <= 3:
        items.append(
            f"**`failed_attempt` is rare.** {_n(failed, 'V4 record')} "
            f"{'comes' if failed == 1 else 'come'} from a real failed attempt in the log; "
            "the other infeasible answers are the graph's or the state's. Report it as a "
            "case, not a row.")
    items.append(
        "**Shortcut bounds are in-sample.** The `summary.json` `shortcuts` table (and the "
        "scorer's *text-only upper bound* column) is measured on the **full export** of a "
        f"unit, not on `{ev}`; `{ev}/<unit>.summary.json` is a copy of the same file. "
        "The pack's `manifest.json` `eval_shortcuts` gives the same bounds measured on "
        f"`{ev}` itself (V4: drawn balanced within each verb + class).")
    items.append(
        "**Captive screws.** A truth that rests on a captive screw's tightness is marked "
        "`label.image_undecidable: true` (no 2D view shows it); the scorer reports every "
        "task also on the decidable records (`extra.<task>.decidable`).")
    conflicts = {v: s.get("open_conflicts") for v, s in summary["exports"].items()
                 if s.get("open_conflicts")}
    if conflicts:
        items.append("**Open conflicts at export time.** "
                     + ", ".join(f"{v}: {n} frames" for v, n in conflicts.items())
                     + " were exported with `verified: false` (`--allow-conflicts`).")
    return "\n\n".join(f"{i}. {text}" for i, text in enumerate(items, start=1))


def _references_text(summary: dict) -> str:
    """The card's paragraph on the reference lines: what each is, for, and not for."""
    refs = summary.get("references") or {}
    para = (
        "Every model row is read against reference lines, none of which is a model "
        "(`tools/vlm_references.py`, `tools/vlm_overlay.py`, and two rows the scorer "
        "prints itself). **chance** is the expected score of a uniform guess over the "
        "record's answer space (binary tasks 0.5, `V6-mc` 1/|options|, V2 1/|allowed| -- its "
        "count form (1 + 0.8 k)/(n+1) over the guesses 0..n, k the gold's neighbours --, "
        "open tasks 0; the rule per task is printed under "
        "the score table). |allowed| counts every state of the part's class and "
        "`cannot_tell`. **frequency** is the most frequent gold reply over the truth "
        "file's `train` desktops (`label.split`; for `V6-mc` the most frequent letter) given "
        "as the answer to every record and graded like a model -- the floor a model has to "
        "clear to have read anything at all. **blind** is the record without its picture "
        "(question and options only; like the state oracle it drops the mark wording -- "
        "\"the RAM module outlined in red\" becomes \"the RAM module\" -- and a part named "
        "by its mark tag alone keeps the tag with its class phrase beside it, \"part A (a RAM "
        "module)\", the state oracle's table tagging that part's rows): what it scores is "
        "template leakage, and the export's "
        "*text-only upper bound (in-sample)* is the ceiling of the same thing. "
        "**protocol_prior** answers V6 / V7 / V16 from nothing but the previous demonstrated "
        "action, by a bigram fitted on the other desktops (leave-one-desktop-out): the line a "
        "text model reading the log starts from. **state_oracle** replaces the pictures by "
        "the structured state table of each moment (perfect perception, screws shown "
        "present / removed only): one table per image in a camera view's file, one per "
        "moment in `multi` (the four cameras of one step share one table, the scanner's; "
        "V3 / V12 have two); it has no V2 `cannot_tell` items, because those ask whether the view can "
        "settle a part's state and the table's visibility column is that answer. "
        "**mask_overlay** shows the frame with the confirmed visible masks painted in one "
        "fixed colour per class (the order of `configs/taxonomy.yaml`, alpha 0.45, a 2 px "
        "outline, no text, no box, nothing that depends on the record), for every record "
        "whose frames are all confirmed frames with masks -- in `multi` the scanner's "
        "frames, which alone are painted (the side cameras' stay as they are) -- except V8 "
        "(the box is the answer): the gap between it and the plain frame is the price of perception. Use "
        "the references to place a model -- above chance and frequency, below the oracle "
        "lines -- and to split its error into perception and reasoning. Do **not** report a "
        "reference as a model result, do not compare `mask_overlay` or `state_oracle` "
        "against the model on different records (they answer fewer: score them on the same "
        "truth file and read `mask_overlay.summary.json` for what was left out), do not "
        "train or tune anything on the overlays or the state tables, and never crop or "
        "paint anything else into a picture a model is scored on."
    )
    masks = refs.get("masks")
    if not masks:
        status = ("**This copy carries no masks sidecar and no overlay variant** (built without "
                  "`--coco-root`): `blind`, `protocol_prior`, `state_oracle`, `chance` and "
                  "`frequency` are available; for `mask_overlay` build the pack with the "
                  "scanner pack (`--coco-root exports/scanpack_<date>`) or run "
                  "`tools/vlm_overlay.py masks --coco-root <scanner pack>` where it exists.")
    else:
        rows = masks.get("views") or {}
        per_view = ", ".join(f"{v} {r['frames_with_masks']}/{r['confirmed_frames']}"
                             for v, r in rows.items())
        without = sum(r.get("confirmed_without_masks", 0) for r in rows.values())
        painted = refs.get("mask_overlay") or {}
        if painted:
            lines = []
            for v, s in painted.items():
                skipped = ", ".join(f"{k} {n}" for k, n in sorted(s["skipped"].items())) or "none"
                lines.append(f"{v} {s['records']} of {s['records_in']} records "
                             f"({s['overlays']} overlays; left out: {skipped})")
            painted_text = ("The subset's overlay variant is pre-painted (`eval_v3/mask_overlay/`, "
                            "`overlays/`): " + "; ".join(lines) + ".")
        else:
            painted_text = ("The overlays are **not** pre-painted in this copy "
                            f"({refs.get('mask_overlay_skipped') or 'built with --overlays none'}): "
                            "`tools/vlm_overlay.py render` paints them from the sidecar.")
        status = (f"The masks come from the scanner pack `{masks.get('source')}` (confirmed "
                  f"rows, visibility checked against the oracle) as `data/vlm_<unit>.masks.jsonl` "
                  f"and `eval_v3/<unit>.masks.jsonl`; confirmed frames with masks per unit: "
                  f"{per_view}" + (f"; {without} confirmed frames have no masks (OAK frames of "
                                   f"V15 pairs -- their records are left out of the variant)."
                                   if without else ".") + " " + painted_text)
    return para + "\n\n" + status


def readme_text(summary: dict) -> str:
    views = summary["views"]
    units = _units(summary)
    ev_name = summary.get("eval_version") or vlm_eval_set.EVAL_VERSION
    # D00 is an empty placeholder desktop (no graph, no frames): not counted
    n_desktops = len([d for d in summary["desktops"] if int(d) != 0])
    n_records = sum(s["records"] for s in summary["exports"].values())
    ev = summary["eval"]
    images_line = {
        "all": (f"`images/` holds every frame the manifests name ({summary['images']} "
                f"files, {summary['image_bytes'] / 2**30:.1f} GB)."),
        "eval": (f"`images/` holds only the frames `{ev_name}` names ({summary['images']} "
                 f"files, {summary['image_bytes'] / 2**30:.1f} GB); the full question set "
                 f"in `data/` needs the dataset's frame cache for the rest."),
        "none": ("**This copy carries no images** (built with `--images none`, a dry run); "
                 "point `--image-root` at the dataset's frame cache."),
    }[summary["image_mode"]]
    audit = ev.get("audit") or {}
    sheet = audit.get("sheet") or {}
    pages = ""
    if sheet:
        which = ("the fillable page `audit.sheet.b01.html`" if sheet["batches"] == 1 else
                 f"the fillable pages `audit.sheet.b01.html` ... `b{sheet['batches']:02d}` "
                 f"({sheet['batch_size']} items each)")
        pages = (f", {which} and the CSV skeleton `audit.sheet.csv` "
                 f"(see *Author audit*)")
    audit_line = (f"`{ev_name}/audit/` is the {audit['size']}-item author-audit sample drawn "
                  f"from inside the subset (`tools/vlm_sample.py --purpose audit`: about 25 per "
                  f"task, positives and negatives balanced, every desktop represented, "
                  f"{audit['desktops']} desktops): `audit.ids.txt`, `audit.jsonl`, the "
                  f"Markdown sheet `audit.sheet.md` (image path, question, answer, an empty "
                  f"`ok?` column){pages}." if audit else
                  "No audit sample was drawn (`--audit 0`).")
    check = summary.get("selfcheck")
    check_line = (f"The build ran it: every task row of every `{ev_name}/<unit>.jsonl` scored "
                  "1.0 with the pack's tools alone." if check and check.get("ok") else
                  "The build did not run it (`--no-selfcheck`).")
    bracket = summary["composition"]["optical_bracket_desktops"]
    bracket_line = (f" The optical drive cage note above applies to "
                    f"{_n(len(bracket), 'desktop')}." if bracket else "")
    multi = MULTI_VIEW in units
    # the references' example unit: the protocol prior answers V6 / V7 / V16, which live
    # in the multi unit only (final review M1)
    ref_unit = MULTI_VIEW if multi else units[0]
    ref_why = ("the action tasks, the protocol prior's V6 / V7 / V16 among them" if multi
               else "this copy has no action task, so no protocol prior")
    unit_list = ", ".join(f"`{u}`" for u in units)
    truth_files = " ".join(f"{ev_name}/{u}.jsonl" for u in units)
    multi_tasks = list(summary.get("unit_tasks", {}).get(MULTI_VIEW) or [])
    every_camera = [t for t in multi_tasks if t not in MARKED_ACTION_TASKS]
    scanner_only = [("V7 (when the CPU is in sight)" if t == "V7" else t) for t in multi_tasks
                    if t in MARKED_ACTION_TASKS]
    multi_line = (
        f"`vlm_{MULTI_VIEW}.jsonl` holds the action tasks "
        f"({', '.join(multi_tasks)}): one record per step and question. "
        + (f"The multi-view ones ({', '.join(every_camera)}) show that moment seen by every "
           f"camera. " if every_camera else "")
        + (f"The ones whose parts are outlined in red ({', '.join(scanner_only)}) show the "
           f"marked scanner image alone." if scanner_only else "")
        if multi else
        f"This copy has no `{MULTI_VIEW}` unit (built without the scanner view or with "
        f"`--no-multi`), so it holds no action task.")
    text = f"""# VLM evaluation pack ({PACK_NAME} v{summary['pack_version']}, `{ev['version']}`)

Template-generated questions about desktop-PC teardowns, with programmatic
answers, over {_n(n_desktops, 'desktop')} and {_n(len(views), 'camera view')}
({', '.join(views)}), in {_n(len(units), 'export unit')} ({unit_list}):
**{n_records} records** in `data/`, a frozen **evaluation subset of {ev['records']}
records** in `{ev_name}/`, the frames they show, and the tools that turn records
into chat requests and grade a model's replies. Everything runs without the
annotation tool or its database: `tools/` is self-contained (Python >= 3.11,
PyYAML; Pillow for crops, scaling and drawing the marks).

{images_line}

## Layout

```
README.md                 this file
LICENSE_NOTE.txt          pre-release terms (licence and citation come with the first tagged release)
manifest.json             build parameters, counts, size + sha256 of every other file
data/
  vlm_<unit>.jsonl                   one record per line (the whole question set of a unit)
  vlm_<unit>.images_manifest.jsonl   opaque image id -> path under images/, desktop, view, step, roi
  vlm_<unit>.oracle.jsonl            label-side world: instances, part groups, edges, log per desktop;
                                     state table per image
  vlm_<unit>.summary.json            counts, silences, shortcut bounds of that export
  vlm_<unit>.masks.jsonl             (with masks) the confirmed visible masks of every confirmed frame, COCO RLE
images/<view>/D<nn>/s<nn>.<ext>      the frames, byte for byte the dataset's cached files (the source's extension)
overlays/<view>/D<nn>/s<nn>.png      (with masks) the subset's frames with the confirmed masks painted (mask_overlay)
{ev_name}/
  ids.txt                 the subset's record ids, task order
  <unit>.jsonl            the subset's records of that unit, with .images_manifest.jsonl,
                          .oracle.jsonl (cut to the subset), .summary.json (the export's) and,
                          with masks, .masks.jsonl (cut to the subset) beside it
  mask_overlay/<unit>.jsonl   (with masks) the subset's GT-mask overlay variant: eligible records,
                          their .images_manifest.jsonl pointing into overlays/, .summary.json (what was left out)
  table.md / table.json   the n table: target, drawn, by unit, strata, desktops
  audit/                  the author-audit sample: audit.ids.txt, audit.jsonl, audit.sheet.md,
                          the fillable pages audit.sheet.b01.html ... and the CSV skeleton audit.sheet.csv
reasoning/<unit>.jsonl    the reasoning side product (training only): {{id, task, question, answer, reasoning}}
tools/
  vlm_to_chat.py          records -> OpenAI / Qwen-VL / InternVL chat JSONL (images by path or base64;
                          draws the marks, writes the image legend and the glossary)
  vlm_glossary.py         the class descriptions and state meanings the converter appends
  vlm_score.py            {{id, raw_text}} replies -> scores per task / layer / template, 95 % CIs,
                          with the chance and frequency reference rows beside every task
  vlm_references.py       the references: blind, protocol prior (bigram), state oracle (no V2 cannot_tell item), mask overlay
  vlm_overlay.py          the confirmed-mask painter (masks sidecar from a scanner pack's COCO; overlays)
  vlm_sample.py           stratified seeded samples (audit / repeat)
  vlm_eval_set.py         the procedure that drew {ev_name} (re-runnable, byte-identical)
  vlm_selfcheck.py        gold replies must score 1.0 everywhere -- run it first
  vlm_audit_sheet.py      the fillable audit pages (model view first, reviewer boxes on request) + CSV
  vlm_audit_summary.py    filled audit CSVs -> error rates with Wilson 95 % intervals, flagged items
  vlm_audit_carry.py      carries audit verdicts over to a re-drawn subset where the item is unchanged
  vlm_human_sheet.py      a timed answer form without answers (not part of the evaluation plan)
  vlm_io.py               shared reading
  tda/, configs/          the scorer's modules and the taxonomy, vendored
  requirements.txt
```

## Tasks

Each record belongs to one task (`prompt.task`) and one unit (`label.view`).
The four camera units hold **perception**: V1 V2 V8 read the compiled masks,
so they are asked only about frames a human confirmed in that view (V14 is merged
into V2: a V2 state question on a part this view cannot see -- wholly covered, too
small, outside the picture -- is answered `cannot_tell`), and V15
(two cameras, same moment or not) needs no mask at all. {multi_line} Its
truth, rationale and evidence come from the step log, the state machine, the
constraint graph and the **scanner** frame (the only confirmed view); the side
cameras are shown, not read. Counts are records in `data/` per unit:

{_counts_table(summary)}

{_rules_text(summary)}

`layer` is the capability layer (`tools/tda/core/export/vlm_check.py`
`CAPABILITY_LAYERS`): L1 perception, L2 affordance, L3 planning, change
(V3 V12 V10), cross (V15). V5 (every legal action) is not exported.

### Multi-view records

A multi-view `{MULTI_VIEW}` record (V3, V6 in all its forms, V10, V12, and V7 when the CPU is
out of sight: no part outlined) has as
`prompt.images` the frames of one step in the fixed order scanner, OAK 1, OAK 2,
RealSense; a camera without a photo of that step is left out, so a record has 1
to 4 images (V3 / V12: the cameras of the step before, then those of the step
after, up to 8). A `{MULTI_VIEW}` record whose parts are outlined in red (V4,
V16, and V7 when the CPU can be seen) shows the marked scanner image of its step alone:
the outline says where to look, so no side camera is added (owner, 2026-11-03);
`label.views` is then `["scan"]`. `label.views` lists the views present and
`label.image_steps` the step of each image, in image order;
`prompt.image_legend` (`{{"view", "moment"}}` per image) is what the converter
turns into the one line naming the images ("Image 1: the top-down scanner." on
a one-image record). The question text never names a view.
`label.view_confirmed` is the scanner frame's.

### Marks, groups, glossary

**Marks.** A question never names a part by an ordinal ("RAM module 3"): every
part it is about is outlined on the scanner image. `prompt.marks` lists
`{{"image", "bbox": [x, y, w, h], "tag"}}` (the image's index in
`prompt.images`, the box in that image's native pixels, the tag `A`, `B`, ...);
the question says "the RAM module outlined in red" or "part A". The converter
draws each box as a thin **dashed** red outline at 60 % opacity, just outside the
box, so the part under it stays visible; the tag sits in a small filled label
**outside** the outline (above its top-left corner, or below it where there is no
room), never on the part, one label per tag and image (a group's label beside its
top-most outline). It draws on a copy of the picture (scaled with it), written under
`--assets` (`tools/vlm_run/run_eval.sh` passes one) or, without it, under
`<out dir>/<out stem>_marked/` beside the chat file; the files under `images/`
are never changed. The box is the part's visible-mask box on
the confirmed scanner frame, or its amodal box when the part is present but
hidden; a part that cannot be outlined is not asked about -- except where the owner
rulings name it in words: a V2 `cannot_tell` question on a part too small or outside the
picture, a V7 goal out of sight ("the CPU"), and V6, whose answers and options name the
actions directly ("unplug the connectors", "take out the PSU"), no mark, no tag. V1 and V8
carry no mark (the box would be the answer). V2's screw counts are class questions without
a mark.

**Groups.** Interchangeable parts -- the screws of one fan, the latches of one
RAM module, the connectors of one cable, the RAM modules -- are asked about as
one group, every member still in the asked situation outlined with the same tag
("the CPU cooler screws outlined in red"). The oracle's desktop line has
`groups` (`group id -> members, match_class, anchor, label`) and each instance
its `group`; gold action targets (V4 blockers, V6, V7 / V16 plans, V10's done
set) are instance keys or group ids, and the scorer grounds a reply's wording
("the cooler screws", "a CPU cooler screw") to the group.

**Glossary.** `prompt.terms` (classes the question or the options name) and
`prompt.state_terms` (classes whose state it asks) are turned by the converter
into a short glossary appended to the prompt: what the class is and, for state
questions, what each state means (`tools/vlm_glossary.py`).

## Records

One JSON object per line:

```
{{"id": "rec_<16 hex>",
 "prompt": {{"task": "V4", "images": ["img_<16 hex>", ...], "question": "...",
            "options": [{{"verb", "target_label"}}, ...],     # V6 multiple choice only (words)
            "marks": [{{"image", "bbox", "tag"}}, ...],         # parts outlined in red
            "hint_regions": [{{"image", "bbox"}}, ...],         # V3 roi form: yellow dashed box
            "image_legend": [{{"view", "moment"}}, ...],        # multi-view records only
            "terms": [...], "state_terms": [...]               # glossary entries
           }},
 "label":  {{"answer": {{...}}, "answer_check": {{...}}, "evidence": [...], "rationale": {{...}},
            "desktop", "step", "view", "views", "image_steps", "layer", "template_id", "tier",
            "verified", "view_confirmed", "graph_version", "model_family", "chassis_type",
            "split", "truth_source", "negative", "image_undecidable", "readable_id", ...}}}}
```

**Nothing on the prompt side reveals the answer.** `prompt.images` are opaque
ids (a salted hash of desktop, view and step); the manifest maps them to files
and the label side carries desktop, step and view. Show a model `prompt` only
(through `tools/vlm_to_chat.py`, which draws the marks and adds the legend and
the glossary).

* `label.answer` -- the structured truth (per task, see *Answer formats*).
* `label.answer_check` -- how the checker re-derives and grades the answer
  (type, fields, metric, the eligible set for V6, the plan for V16, the
  target for V3, ...). `tools/vlm_score.py` implements it; nothing
  here is shown to a model.
* `label.evidence` / `label.rationale` -- boxes of the parts the answer rests
  on and the reasoning chain (`observe` / `recall_relation` /
  `check_precondition` / ... steps); label side only. **Every** record also carries
  `rationale.text`, the reasoning in words of what the picture shows -- what is seen,
  which rule applies, so the answer ("No: the CPU socket lever still locks the CPU.";
  V16 step by step) -- no graph vocabulary, no ordinal; `reasoning/<unit>.jsonl` collects
  it with the question and the gold reply (in plain names) as a training side product.
* V4's `answer.reason` is that reason in words -- the prerequisites now met ("both its RAM
  latches are open, so the RAM module can be taken out") or what still holds the part --,
  `answer.why` its machine-readable list (`{"part", "state", "relation"}`, a plain verb:
  "locks", "covers", "fastens", "is plugged into", "is mounted on", ...), and
  `answer_check.reason_code` the internal code (`violates_edge`, `covered_now` -- a part above
  still covers it --, `loose_part_first`, or none on a "yes"). V16's
  `answer.reason` is the first broken step in words, its code in `answer_check.reason_code`.
  Parts are named without ordinals everywhere a person or a model reads them.
* `label.truth_source` (V4, V7, V16) -- where an answer's "yes" or "no" comes
  from: `{'`, `'.join(TRUTH_SOURCES)}`. `demonstrated`: the operator did it
  next (and the rules allow it). `simulated` (V16 valid): a replay of the whole sequence
  under the rules is clean -- validity never rests on "it was demonstrated".
  `occlusion_blocked`: a part above still covers it in this frame. `loose_first`: a part
  lying loose in the case has to come out first.
  `graph_blocked`: a constraint edge forbids it. `graph_blocked_pixels`:
  forbidden only by an edge that was imported from the pixels / a reviewer's
  occlusion ruling ("class rule x overlap in the annotation").
  `state_inapplicable`: the part is not in a state the verb starts from
  (already open, already removed) -- no longer used for V4's negatives.
  `failed_attempt`: a real failed attempt in the log. A V4 "yes" on a **group**
  (its members outlined together, "can the motherboard screws outlined in red be
  unscrewed?") is demonstrated for the one member the operator took next and
  graph-permitted for the others: no edge forbids them and they are
  interchangeable with it, but nobody did them at that moment.
* `label.negative` -- the kind of negative half, or absent on a positive: V4 the
  truth source; V12 `captive_screw_loosened` (the log moves, the picture does
  not); V2 `cannot_tell`; V15 `different_moment`; V16 the corruption
  (`substitute` / `move` / `swap`).
* `label.tier` is the annotation standard of the view the truth rests on (scan
  and oak1 gold, oak2 silver, rs bronze; `{MULTI_VIEW}`: the scanner's);
  `label.verified` says whether a human confirmed this frame and every row the
  answer was read off; `label.view_confirmed` says whether anything of this
  view was confirmed on this desktop at all.
* `label.split` -- `test` / `train` as the platform-disjoint split stamped it
  (12 / 54 desktops); the zero-shot evaluation ignores it and uses all
  desktops.
* `label.prior_is_correct` (V2) -- whether the class prior (the part's state
  at the start of the teardown; for counts the starting number of screws)
  already gives the answer. The scorer's `V2` row (the headline) is every V2
  question; `V2-prior-wrong` is the `false` subset, a shortcut check.
* `label.v6_form` / `label.v6_pair` (V6) -- `open`, `mc` or `constrained`; the open and
  multiple-choice forms of one step share the pair id. `answer_check.groups` is the
  frontier (every action possible now, grouped: all connectors one group, all RAM
  latches, the RAM modules, screws by the part they fasten, else one group per part; a
  target only a cable hides stays, `hidden_by_cable`), `gold_groups` the keys graded,
  `blocked_groups` what cannot be done yet; the constrained form adds `constraint`
  (`kind`, `rule` -- the words the prompt adds --, the parts held, the condition and the
  edges a replay adds) and `forbidden_groups`.
* `label.v7_form` (V7) -- `free` or `constrained`; `answer_check.moment` is the moment's own
  rules (what covers a part in this frame, the loose parts still in), `constraint` the
  rule of the constrained form and `count` the plugs it has to unplug.
* `label.v3_form` / `label.v3_pair` (V3, owner 2026-11-04) -- two questions about one step,
  scored as the rows `V3` and `V3-roi`. **`free`** is the question with no hint; it also asks
  `"region": [x, y, width, height]`, where the machine changed, in pixels of the top-down
  scanner image (its before and after pictures, `answer_check.region_images`). The gold
  region (`answer.region` = `answer_check.region`) is the union of the changed part's scanner
  boxes before and after (its group's where the part itself has none); the score stays verb +
  target, and region IoU / region hit (IoU >= 0.5) are extra rows -- a box on a cropped or
  scaled picture maps back like V1 / V8's. **`roi`** draws a yellow dashed box with no tag
  (`prompt.hint_regions`, `answer_check.hint_region`) on both scanner pictures -- the gold
  region grown 2.5 times about its centre, at least 300 px a side, kept inside and clipped to
  the image -- says the change happened inside it, and asks the action as before (no region).
  A step without a gold region (no confirmed scanner box) is asked free, without the region,
  and has no roi partner. `v3_pair` is the free record's own id.
* `label.image_undecidable` -- the truth rests on a captive screw's tightness.
* `label.views` -- V15: the two views of the pair; `{MULTI_VIEW}`: the views
  shown, in `prompt.images` order.

## Answer formats

The chat converter appends the format sentence of the record's task
(`format_instruction`); the scorer parses the reply **strict** (one JSON object,
a code fence allowed) and **lenient** (the last well-formed object in the text,
else the answer words) and reports both parse-failure rates. `{{allowed}}` is the
class's state vocabulary, filled per record from `answer_check.allowed`.

{_formats()}

## Images, crops, ROI

`images/<view>/D<nn>/s<nn>.<ext>` is the frame of step `<nn>` of desktop `<nn>`
in that view, byte for byte the dataset's cached frame. Two crops are allowed
and the converter implements both: none (`--roi none`, the default) or the
view's **fixed chassis ROI** (`--roi view`: the manifest's `roi` `[x0, y0, x1, y1]`,
one crop for every record of a desktop and view).
**Never crop around a record's evidence**: the evidence boxes are label side,
and a crop built from them puts the answer in the picture (every V2 state question
on a visible part has a box and nearly no `cannot_tell` one does; V8's box is its answer). Box
answers (V1, V8) and the marks are in pixel coordinates of the image shown;
with `--roi view` or `--max-side` the converter scales the marks with the
picture, writes the crop offset and scale to `meta.transforms`, and
`vlm_score.py --chat` maps box answers back. With `--roi view` a mark that
crosses the edge of the view's ROI is clipped there: the model sees only the
part of the outline inside the crop. A multi-image question is one image
part per picture, or one side-by-side picture with `--composite` (fixed per
model).

## The evaluation subset (`{ev_name}`)

Drawn once over all {n_desktops} desktops (zero-shot; the split is not used),
seed {ev['seed']}, by `tools/vlm_eval_set.py`: per task at least {ev['targets']['binary']}
records for the yes / no tasks ({', '.join(ev['targets']['binary_tasks'])}) and
{ev['targets']['other']} for the others where the pool allows; V6 is drawn as pairs
(open + multiple choice of one frame), V6 constrained and V7 constrained as buckets of their
own, and V3 as its free record with its roi partner (the unit is
keyed by the free record's id, so the V3 ids drawn are the free form's alone; the audit
sample draws free records only). Perception is drawn from the camera units, the action
tasks from `{MULTI_VIEW}`; a task found in several units is allocated over them in
proportion to the export, then equally over the task's strata (V4 feasible /
infeasible then `truth_source`; V2 `prior_is_correct` then class; V12 changed /
not then kind; V6 by frontier size, V6 / V7 constrained by kind of rule; V15 same /
different then view pair; V16 valid / invalid then corruption; V3 by verb, V8 by class,
V10 by progress bin; inside every stratum the ids of `eval_v2` first, wherever the
record survives -- {ev.get('kept', {}).get('drawn_again', 0)} of them are drawn again),
desktops take turns inside a stratum, and no desktop holds more than
{ev['max_desktop_share']:.0%} of a task's target (a pool smaller than the target gives
everything it has). Two runs are byte-identical. `{ev_name}` replaces `eval_v2` (the final
round of 2026-11-04: the corrected graph with dynamic occlusion and loose parts first, V6
every action possible now, V7 free / constrained, V14 merged into V2; chat format
`chat-v5`); the tools refuse an `eval_v1` or `eval_v2` path. Run every open model on the
same `{ev_name}`; the closed
model answers the items of `{ev_name}/audit/audit.ids.txt` once (the same items
the author audits), and the open models are also scored on those ids for the
comparison; intervals by cluster bootstrap over desktops (the scorer does this).

{_eval_table(summary)}

The strata counts are in `{ev_name}/table.md`. {audit_line}

## Author audit

The error rate of the generated truth. Open `{ev_name}/audit/audit.sheet.b01.html`
(and the following pages) in a browser from the pack directory -- plain HTML
with inline script, no server, no network; the frames are loaded by relative
path (`../../images/...`). Every item shows the picture(s) exactly as the model gets
them -- the red dashed marks (and V3's yellow dashed hint box) drawn the converter's way,
nothing else --, the question, the gold answer in words (no ordinals) and as JSON, and
the rationale (the reason in what the picture shows); the evidence and answer boxes are a
separate reviewer layer, hidden until **show reviewer boxes / 显示审核框** is ticked, and a
legend line says the model never sees them. The reviewer marks 正确 / 错误 / 存疑
(correct / wrong / doubtful) and may leave a note. Answers stay in the browser
until **Export answers** downloads a CSV (`id, task, readable_id, desktop, view,
step, batch, verdict, note, reviewer, answered_at`); `audit.sheet.csv` is the
same table empty, for a spreadsheet. Then

```
python tools/vlm_audit_summary.py --csv runs/audit/*.csv --records {ev_name}/audit/audit.jsonl \\
    --out runs/audit                   # audit.summary.md / .json
```

reports, per task and overall, the error rate (marked wrong) and the error-or-doubt
rate with **Wilson 95 % intervals** (0 of 300 -> [0, 1.3 %]), the items marked wrong or
doubtful with their notes, and the disagreements where several reviewers marked one
item (an item then takes its worst verdict). `tools/vlm_audit_carry.py` carries the
verdicts of an earlier audit over to items that survive unchanged in substance. The
audit pages contain the answers: do not show them to anyone answering the questions.

## Reference lines

{_references_text(summary)}

### Text-only shortcut upper bounds (full export, in-sample)

The best of five prompt-only classifiers (majority, verb, class, verb + class,
template) per task, from `summary.json`; a model below these has read nothing
from the picture. The scorer prints it as *text-only upper bound
(in-sample)*: it is measured on the full export, not on `{ev_name}`, and it is a
ceiling for template leakage, not a model.

{_shortcut_table(summary)}

## How to run

```
python -m venv venv                                  # then activate it, and
pip install -r tools/requirements.txt                # PyYAML (+ Pillow for crops and marks)

# 0. the pack stands alone and the truth is consistent: every gold reply scores 1.0
python tools/vlm_selfcheck.py --truth {truth_files} \\
    --out runs/selfcheck

# 1. chat requests for one unit's subset (OpenAI-style; also qwen, internvl)
python tools/vlm_to_chat.py --export {ev_name}/{MULTI_VIEW if multi else views[0]}.jsonl --format openai \\
    --image-root images --out runs/{MULTI_VIEW if multi else views[0]}.openai.jsonl   # [--base64] [--roi view --assets runs/crops]

# 2. run your model: one line per request, {{"id": ..., "raw_text": "<the reply>"}}

# 3. score
python tools/vlm_score.py --truth {ev_name}/{MULTI_VIEW if multi else views[0]}.jsonl --pred runs/<model>.{MULTI_VIEW if multi else views[0]}.jsonl \\
    --chat runs/{MULTI_VIEW if multi else views[0]}.openai.jsonl --out runs/scores --name <model>-{MULTI_VIEW if multi else views[0]}

# references: blind, protocol prior, state oracle (no V2 cannot_tell item) and, with --assets, the mask overlay
# ({ref_unit}: {ref_why}; run the same on every unit)
python tools/vlm_references.py --export {ev_name}/{ref_unit}.jsonl --out runs/refs/{ref_unit} \\
    --assets overlays --image-root images                     # overlays/ is pre-painted for {ev_name}
python tools/vlm_to_chat.py --export {ev_name}/{ref_unit}.jsonl --records runs/refs/{ref_unit}/blind.jsonl \\
    --format openai --out runs/{ref_unit}.blind.jsonl
python tools/vlm_to_chat.py --export runs/refs/{ref_unit}/mask_overlay.jsonl --format openai \\
    --image-root overlays --out runs/{ref_unit}.overlay.jsonl       # the variant carries its own manifest
python tools/vlm_score.py --truth {ev_name}/{ref_unit}.jsonl --pred runs/refs/{ref_unit}/protocol_prior.predictions.jsonl \\
    --out runs/scores --name protocol-prior-{ref_unit}
# (or, for any subset: python tools/vlm_overlay.py render --export <truth.jsonl> --image-root images \\
#      --assets <dir> --out <dir>/mask_overlay.jsonl; tools/vlm_run/run_eval.sh VARIANT=overlay does this)
```

`vlm_score.py` writes `<name>.score.json` and `<name>.score.md`: per task, per
layer and per question template, strict and lenient, 95 % cluster-bootstrap
intervals over desktops, the text-only upper bound beside each task, V6 open
(headline: F1 over the frontier's groups, precision / recall beside it), `V6-mc` (one
option possible) and `V6-constrained` (F1 plus `constraint_violation`), V7 and
`V7-constrained` (valid = every step legal under the rules and the goal reached; plus
`constraint_kept`, `count_exact` and `count_abs_error`: the plugs the plan unplugs against
the plugs that are in), V2's `cannot_tell_accuracy`, V3 (headline: the free form; extra rows region_iou /
region_hit) and `V3-roi` (the hinted form), V2 (headline: every V2 question) and `V2-prior-wrong`
(V2's count form, `derive = screw_count`, scores the exact count 1.0, one off 0.8 and anything
else 0; the extra row `exact` is the exact matches alone, `count_exact` those of the counts), V7's violation-free /
required-recall / goal-reached, V4's blockers F1, V10's bin accuracy. Score the
units separately (each truth file has its own oracle) and pool in your
analysis; `{ev_name}/ids.txt` is the subset. The model driver (`tools/vlm_run/vlm_run.py`:
OpenAI-compatible APIs, vLLM, transformers, and a `gold` pseudo backend), its Slurm
templates (an array over the units {' '.join(units)}) and `run_eval.sh` ship in
`tools/vlm_run/`; `docs/vlm_run_hprc.md` is the step-by-step run guide for the
cluster. A `{MULTI_VIEW}` request carries up to 8 images: give the model that many
(`--max-images 8` for vLLM). This card documents the data and the scorer.

**Self-check.** {check_line} Re-run it on the receiving machine before anything
else; it needs no image.

## Known limitations

{_known_issues(summary)}

## Provenance

`manifest.json` lists every file with its size and sha256, the desktops, views,
units, seed, image mode and the annotation tool's commit the export ran from
(`source_commit`). Records are reproducible from the database: the question
template is chosen by a checksum of the record id, hard negatives are drawn in
a permutation seeded by desktop and step, and the files are written in a fixed
order.{bracket_line}
"""
    return text


# --------------------------------------------------------------------------- #
def _parse_views(text: str) -> list[str]:
    views = [v.strip() for v in str(text).split(",") if v.strip()]
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise PackError(f"unknown view(s) {', '.join(unknown)}; one of {', '.join(VIEWS)}")
    return views


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--db", required=True, help="a read-only copy of tda.sqlite (never the live one)")
    ap.add_argument("--out", default=None,
                    help=f"pack directory (default D:/DataSet/exports/{PACK_NAME}_<today>)")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="the frame cache (read only)")
    ap.add_argument("--views", default=",".join(VIEWS),
                    help="the camera views exported as single-view units (the multi unit is added "
                         "when scan is among them)")
    ap.add_argument("--no-multi", action="store_true",
                    help="leave out the multi unit (the action tasks): a perception-only pack")
    ap.add_argument("--desktops", default=None, help="e.g. 1-66 (default: every desktop)")
    ap.add_argument("--seed", type=int, default=vlm_eval_set.SEED)
    ap.add_argument("--images", default="all", choices=IMAGE_MODES,
                    help="all frames the manifests name, only the subset's, or none (dry run)")
    ap.add_argument("--audit", type=int, default=300, help="size of the author-audit sample")
    ap.add_argument("--work", default=DEFAULT_WORK, help="the database copy and scratch files")
    ap.add_argument("--allow-conflicts", action="store_true",
                    help="export frames with an open disagreement as unverified")
    ap.add_argument("--no-selfcheck", action="store_true")
    ap.add_argument("--force", action="store_true", help="replace a pack at --out")
    ap.add_argument("--coco-root", default=None,
                    help="the scanner pack (D<nn>/annotations.json): the confirmed masks for the "
                         "masks sidecars and the GT-mask overlay variant")
    ap.add_argument("--overlays", default="eval", choices=OVERLAY_MODES,
                    help="paint the subset's overlays into the pack (eval), or ship only the "
                         "masks sidecar (none)")
    args = ap.parse_args(argv)
    out = args.out or f"D:/DataSet/exports/{PACK_NAME}_{date.today().isoformat()}"
    try:
        views = _parse_views(args.views)
        wanted = sorted(parse_desktops(args.desktops)) if args.desktops else None
        build_pack(args.db, out, cache_dir=args.cache_dir, work_dir=args.work, views=views,
                   desktops=wanted, seed=args.seed, images=args.images,
                   allow_conflicts=bool(args.allow_conflicts), force=bool(args.force),
                   selfcheck=not args.no_selfcheck, audit=args.audit,
                   coco_root=args.coco_root, overlays=args.overlays,
                   multi=not args.no_multi)
    except PackError as exc:
        print(f"[vlmpack] refused: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
