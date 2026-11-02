"""The pack's self-check: the gold replies score 1.0 on every task (W4-A).

``python tools/vlm_selfcheck.py --truth <eval_v1/scan.jsonl> [<eval_v1/oak1.jsonl> ...]
--out <dir> [--resamples 50]``

For each truth file this writes the reply a perfect model would give to every
record (:func:`tda.core.export.vlm_check.gold_reply`, instance keys spelled
the way the oracle labels them) as a predictions file ``<out>/<view>.gold.jsonl``
-- ``{"id", "raw_text"}`` per line, the format ``vlm_score.py --pred`` reads --
and scores it with :mod:`vlm_score`. It exits 0 only when, in **strict**
parsing, every task row of every file has mean 1.0 and no parse failure; a
pack whose scorer, oracle and records disagree anywhere exits 1 and names the
rows.

Everything it imports lives in the pack's ``tools/`` (the vendored ``tda``
subset): run it in a fresh virtual environment with nothing but PyYAML
installed to prove the pack stands alone. ``<out>/selfcheck.json`` records the
result per file and task.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vlm_io as IO  # noqa: E402
import vlm_score  # noqa: E402
from tda.core.export.vlm_check import gold_reply  # noqa: E402


def write_gold(truth_path, out_path) -> int:
    """The gold predictions file of one truth file; returns how many lines."""
    worlds, _ = IO.load_oracle(truth_path)
    rows = ({"id": r["id"], "raw_text": gold_reply(r, worlds.get(IO.desktop_of(r)))}
            for r in IO.read_jsonl(truth_path))
    return IO.write_jsonl(out_path, rows)


def sidecars_cover(truth_path) -> dict:
    """Does the oracle hold every record's world and every image's state table,
    and the manifest every image? The scorer and the references need all three."""
    records = list(IO.read_jsonl(truth_path))
    worlds, frames = IO.load_oracle(truth_path)
    manifest = IO.load_manifest(truth_path)
    desks = {IO.desktop_of(r) for r in records}
    images = {img for r in records for img in r["prompt"].get("images") or []}
    return {"missing_worlds": sorted(desks - set(worlds)),
            "missing_frames": sorted(images - set(frames)),
            "missing_manifest": sorted(images - set(manifest))}


def check(truth_path, out_dir, *, resamples: int = 50, seed: int = 0) -> dict:
    """Score the gold replies of one file; ``{"file", "records", "rows", "ok", ...}``.

    ``ok`` also needs the sidecars to cover the file (:func:`sidecars_cover`): a
    gold reply can score 1.0 against a missing world (the keys are accepted
    raw), a model's wording cannot.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    view = Path(truth_path).stem
    pred = out / f"{view}.gold.jsonl"
    n = write_gold(truth_path, pred)
    result = vlm_score.score(truth_path, pred, resamples=resamples, seed=seed)
    strict = result["strict"]
    rows = {}
    cover = sidecars_cover(truth_path)
    ok = n > 0 and result["predicted"] == n and not any(cover.values())
    for task, cell in sorted(strict["by_task"].items()):
        fail = strict["parse_failure_rate"].get(task) or 0.0
        good = (cell["mean"] is not None and abs(cell["mean"] - 1.0) < 1e-9 and fail == 0.0)
        rows[task] = {"n": cell["n"], "strict_mean": cell["mean"], "parse_failure_rate": fail,
                      "ok": good}
        ok = ok and good
    (out / f"{view}.gold.score.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    return {"file": str(Path(truth_path).name), "records": n,
            "predicted": result["predicted"], "rows": rows, "ok": ok, **cover}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--truth", nargs="+", required=True, help="export-vlm JSONL files")
    ap.add_argument("--out", required=True, help="where the gold predictions and scores go")
    ap.add_argument("--resamples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    results = [check(p, args.out, resamples=args.resamples, seed=args.seed)
               for p in args.truth]
    import tda

    report = {"ok": all(r["ok"] for r in results), "files": results,
              "tda_module": str(Path(tda.__file__).resolve().parent),
              "python": sys.version.split()[0]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selfcheck.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    for r in results:
        bad = [t for t, row in r["rows"].items() if not row["ok"]]
        gaps = [f"{k} {len(v)}" for k in ("missing_worlds", "missing_frames", "missing_manifest")
                for v in [r[k]] if v]
        print(f"[vlm-selfcheck] {r['file']}: {r['records']} records, "
              f"{len(r['rows'])} task rows, "
              + ("all 1.0, sidecars complete" if r["ok"] else
                 "FAILED: " + "; ".join(([f"not 1.0 on {', '.join(bad)}"] if bad else [])
                                        + gaps) or "the record count"))
    print(f"[vlm-selfcheck] tda imported from {report['tda_module']}")
    print(f"[vlm-selfcheck] {'OK' if report['ok'] else 'FAILED'} -> {out / 'selfcheck.json'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
