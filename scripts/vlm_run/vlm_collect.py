"""Glue around ``vlm_run.py`` for a run directory (W4-B): subset, split, table.

* ``subset --export <view.jsonl> --out <dir>/<name>.jsonl (--ids <file> | --records <jsonl>)``
  -- the records of an export restricted to some ids, **with the export's
  sidecars copied beside it** (``.images_manifest.jsonl``, ``.oracle.jsonl``,
  ``.summary.json``, and ``.masks.jsonl`` when the export has one) so
  ``vlm_to_chat.py --export`` and ``vlm_score.py --truth`` take the subset as
  a truth file of its own. The pack's ``eval_v1/<view>.jsonl`` and a pilot ids
  file both go through here.
* ``blind --records <jsonl> --out <jsonl>`` -- the image-ablated variant of the
  records (``prompt.images = []``, ``label.reference = "blind"``), the same
  rows ``vlm_references.py`` writes as ``blind.jsonl``, for a run directory
  that has no ``tda`` and may hold a single desktop (the protocol prior there
  refuses, and takes the other variants with it).
* ``split --pred <pred.jsonl> [--out-dir <dir>]`` -- a ``--repeat k`` prediction
  file into ``<stem>.r1.jsonl .. <stem>.rk.jsonl``, each with ``raw_text`` =
  ``raw_text_i``, so every repeat is scored like a run of its own.
* ``table --scores <*.score.json ...> --out <table.md> [--json <table.json>]``
  -- one markdown table over many ``vlm_score.py`` outputs: a row per
  ``runs/<model>/<variant>/<view>`` (read off the path), strict accuracy per
  task with the bootstrap interval, parse-failure rates, and for repeats
  (``*.r1 .. *.rk``) mean ± sample std over the repeats instead.

No ``tda`` import; ``json``, ``shutil``, ``statistics`` only.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

SIDECARS = ("images_manifest.jsonl", "oracle.jsonl", "summary.json")
#: Copied when the export has one: the confirmed masks (W4-C's overlay variant).
OPTIONAL_SIDECARS = ("masks.jsonl",)
_REPEAT = re.compile(r"^(?P<stem>.+)\.r(?P<k>\d+)$")


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


def sidecar(path, suffix: str) -> Path:
    p = Path(path)
    return p.with_name(p.stem + "." + suffix)


# --------------------------------------------------------------------------- #
def subset(export, out, ids_file: Optional[str] = None, records: Optional[str] = None) -> int:
    """Write the records of ``export`` whose id is in ``ids_file`` / ``records``, plus sidecars."""
    if bool(ids_file) == bool(records):
        raise SystemExit("subset: exactly one of --ids / --records")
    if ids_file:
        keep = {ln.strip() for ln in Path(ids_file).read_text(encoding="utf-8").splitlines()
                if ln.strip()}
    else:
        keep = {str(r["id"]) for r in read_jsonl(records)}
    rows = [r for r in read_jsonl(export) if str(r["id"]) in keep]
    missing = keep - {str(r["id"]) for r in rows}
    if missing:
        raise SystemExit(f"subset: {len(missing)} ids not in {export} "
                         f"(first: {sorted(missing)[:3]})")
    out = Path(out)
    if out.resolve() == Path(export).resolve():
        raise SystemExit("subset: --out must differ from --export")
    n = write_jsonl(out, rows)
    for suffix in SIDECARS + OPTIONAL_SIDECARS:
        src = sidecar(export, suffix)
        if src.exists():
            shutil.copyfile(src, sidecar(out, suffix))
    return n


def blind(records, out) -> int:
    """Every record with its images taken away; question, options and label unchanged."""
    def strip(r: dict) -> dict:
        r = json.loads(json.dumps(r))
        r["prompt"]["images"] = []
        r["label"]["reference"] = "blind"
        return r

    return write_jsonl(out, (strip(r) for r in read_jsonl(records)))


def split(pred, out_dir: Optional[str] = None) -> list[Path]:
    """``pred.jsonl`` with ``raw_text_1..k`` -> ``pred.r1.jsonl .. pred.rk.jsonl``."""
    pred = Path(pred)
    rows = list(read_jsonl(pred))
    k = 0
    for row in rows:
        i = 1
        while f"raw_text_{i}" in row:
            i += 1
        k = max(k, i - 1)
    if k == 0:
        raise SystemExit(f"split: {pred} has no raw_text_1..k (not a --repeat run)")
    out_dir = Path(out_dir) if out_dir else pred.parent
    paths = []
    for i in range(1, k + 1):
        path = out_dir / f"{pred.stem}.r{i}.jsonl"
        write_jsonl(path, ({"id": r["id"], "raw_text": r.get(f"raw_text_{i}"),
                            "meta": {**(r.get("meta") or {}), "repeat": i}} for r in rows))
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- #
def _task_order(task: str) -> tuple:
    digits = "".join(ch for ch in task[1:] if ch.isdigit())
    return (int(digits or 0), task)


def _row_key(path: Path) -> tuple[str, str]:
    """``(run, name)``: ``.../<model>/<variant>/<view>/score/x.score.json`` -> ``model/variant/view``."""
    name = path.name[:-len(".score.json")] if path.name.endswith(".score.json") else path.stem
    parts = [p for p in path.parts[:-1] if p not in ("score", "scores")]
    run = "/".join(parts[-3:]) if len(parts) >= 3 else "/".join(parts)
    return run, name


def collect(score_files: list[str]) -> dict:
    """Group score files by run, fold repeats, return the table as data."""
    files: list[Path] = []
    for pattern in score_files:
        hits = sorted(glob.glob(pattern, recursive=True))
        files += [Path(h) for h in hits] if hits else [Path(pattern)]
    plain: dict[tuple[str, str], list[dict]] = defaultdict(list)
    repeats: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in files:
        if not f.is_file():
            raise SystemExit(f"table: {f} is not a file")
        run, name = _row_key(f)
        m = _REPEAT.match(name)
        data = json.loads(f.read_text(encoding="utf-8"))
        if m:
            repeats[(run, m.group("stem"))].append(data)
        else:
            plain[(run, name)].append(data)
    # a repeated run's plain file is its first repeat again: the repeats stand for it
    groups = {key: repeats.get(key) or plain[key] for key in set(plain) | set(repeats)}
    rows = []
    tasks: set[str] = set()
    for (run, name), results in sorted(groups.items()):
        strict = [r["strict"] for r in results]
        task_names = sorted({t for s in strict for t in s["by_task"]}, key=_task_order)
        tasks.update(task_names)
        cells = {}
        for t in task_names:
            means = [s["by_task"][t]["mean"] for s in strict if t in s["by_task"]
                     and s["by_task"][t]["mean"] is not None]
            if not means:
                continue
            cell = {"n": strict[0]["by_task"][t]["n"], "mean": round(statistics.fmean(means), 4),
                    "repeats": len(means)}
            if len(means) > 1:
                cell["std"] = round(statistics.stdev(means), 4)
            else:
                cell["ci95"] = strict[0]["by_task"][t].get("ci95")
            cells[t] = cell
        rows.append({"run": run, "name": name, "repeats": len(results),
                     "records": results[0].get("records"),
                     "predicted": round(statistics.fmean(r.get("predicted", 0) for r in results)),
                     "parse_failure_strict": round(statistics.fmean(
                         s["parse_failure_rate_all"] for s in strict), 4),
                     "parse_failure_lenient": round(statistics.fmean(
                         r["lenient"]["parse_failure_rate_all"] for r in results), 4),
                     "by_task": cells})
    return {"tasks": sorted(tasks, key=_task_order), "rows": rows}


def markdown(table: dict) -> str:
    tasks = table["tasks"]

    def fmt(cell):
        if not cell:
            return "–"
        if "std" in cell:
            return f"{cell['mean']:.3f} ± {cell['std']:.3f} (k={cell['repeats']})"
        ci = cell.get("ci95")
        if ci and ci[0] is not None:
            return f"{cell['mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        return f"{cell['mean']:.3f}"

    lines = ["# VLM runs -- strict accuracy per task", "",
             "Cells: mean [95 % cluster bootstrap over desktops]; for repeated runs "
             "mean ± sample std over the k repeats. Parse failure: strict / lenient, "
             "over every record of the truth file (unanswered records count).", "",
             "| run | name | n | answered | parse fail | " + " | ".join(tasks) + " |",
             "|---|---|---|---|---|" + "---|" * len(tasks)]
    for row in table["rows"]:
        cells = " | ".join(fmt(row["by_task"].get(t)) for t in tasks)
        lines.append(f"| {row['run']} | {row['name']} | {row['records']} | {row['predicted']} | "
                     f"{row['parse_failure_strict']:.1%} / {row['parse_failure_lenient']:.1%} | "
                     f"{cells} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset", help="records of an export by id, with sidecars")
    s.add_argument("--export", required=True)
    s.add_argument("--out", required=True, help="the subset JSONL (sidecars go beside it)")
    s.add_argument("--ids", default=None)
    s.add_argument("--records", default=None, help="a JSONL whose ids to keep")
    b = sub.add_parser("blind", help="the image-ablated variant of a records file")
    b.add_argument("--records", required=True)
    b.add_argument("--out", required=True)
    p = sub.add_parser("split", help="a --repeat prediction file into one per repeat")
    p.add_argument("--pred", required=True)
    p.add_argument("--out-dir", default=None)
    t = sub.add_parser("table", help="one table over many *.score.json")
    t.add_argument("--scores", nargs="+", required=True, help="files or globs")
    t.add_argument("--out", required=True, help="markdown table")
    t.add_argument("--json", default=None, help="the same as JSON")
    args = ap.parse_args(argv)
    if args.cmd == "subset":
        n = subset(args.export, args.out, args.ids, args.records)
        print(f"[vlm-collect] subset: {n} records -> {args.out}")
    elif args.cmd == "blind":
        n = blind(args.records, args.out)
        print(f"[vlm-collect] blind: {n} records without images -> {args.out}")
    elif args.cmd == "split":
        paths = split(args.pred, args.out_dir)
        print(f"[vlm-collect] split: {len(paths)} repeats -> {paths[0].parent}")
    else:
        table = collect(args.scores)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown(table), encoding="utf-8")
        if args.json:
            Path(args.json).write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
        print(f"[vlm-collect] table: {len(table['rows'])} runs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
