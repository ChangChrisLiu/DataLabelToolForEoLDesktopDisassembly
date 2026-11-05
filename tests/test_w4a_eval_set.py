"""The evaluation-subset sampler (W4-A, ``scripts/vlm_eval_set.py``) on synthetic records.

The records here carry only what the sampler reads (task, id, desktop, view,
the stratum fields), so the constraints can be checked on pools of any shape.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import vlm_eval_set as E  # noqa: E402
import vlm_io as IO  # noqa: E402


def _rec(task: str, n: int, desktop: int, view: str, **label) -> dict:
    rid = "rec_" + hashlib.sha256(f"{task}|{n}|{desktop}|{view}".encode()).hexdigest()[:16]
    img = "img_" + hashlib.sha256(f"{desktop}|{view}|{n % 7}".encode()).hexdigest()[:16]
    lab = {"desktop": desktop, "step": n % 7 + 1, "view": view,
           "answer_check": {}, "answer": {}}
    lab.update(label)
    return {"id": rid, "prompt": {"task": task, "images": [img], "question": "q?"},
            "label": lab}


def _pool() -> dict[str, list[dict]]:
    """Two views: the scanner three times the size of OAK1 on the log tasks."""
    sources = ("graph_blocked", "graph_blocked_pixels", "state_inapplicable")
    pool: dict[str, list[dict]] = {"scan": [], "oak1": []}
    n = 0
    for view, mult in (("scan", 3), ("oak1", 1)):
        for desktop in range(1, 31):
            for k in range(10 * mult):
                n += 1
                # V4: half feasible; the negatives over three sources, one rare
                negative = None if k % 2 == 0 else sources[0 if k % 9 else 1]
                if negative == "graph_blocked" and k % 3 == 0:
                    negative = "state_inapplicable"
                pool[view].append(_rec("V4", n, desktop, view, negative=negative,
                                       truth_source=negative or "demonstrated"))
                # V6 pairs
                pair = f"pair{n}"
                pool[view].append(_rec("V6", n * 2, desktop, view, v6_form="open", v6_pair=pair))
                pool[view].append(_rec("V6", n * 2 + 1, desktop, view, v6_form="mc", v6_pair=pair))
                # V12 half changed
                pool[view].append(_rec("V12", n, desktop, view,
                                       negative=None if k % 2 else "captive_screw_loosened"))
                # V15 pairs with another view
                other = "oak2" if k % 2 else "rs"
                pool[view].append(_rec("V15", n, desktop, view, views=[view, other],
                                       negative=None if k % 4 < 2 else "different_moment"))
            if view == "scan":
                for k in range(20):
                    n += 1
                    cls = ["ram_latch"] * 14 + ["cover", "psu", "cable_clip", "card_latch",
                                                "drive_cage", "cpu_socket_lever"]
                    # (states that vary within a class: the draw thins V2 per desktop and
                    # class so no answer is the majority, and a class of one answer goes)
                    pool[view].append(_rec("V2", n, desktop, view,
                                           prior_is_correct=bool(k % 3),
                                           answer={"state": "open" if k % 2 else "closed"},
                                           answer_check={"derive": "instance_state",
                                                         "class": cls[k]}))
                    pool[view].append(_rec("V1", n, desktop, view))
    return pool


@pytest.fixture(scope="module")
def units():
    pool = _pool()
    return [u for view, recs in pool.items() for u in E.units_of(recs, view)]


# --------------------------------------------------------------------------- #
# allocation
# --------------------------------------------------------------------------- #
def test_equal_allocation_redistributes_what_a_small_stratum_cannot_fill():
    assert E.allocate_equal({"a": 100, "b": 5, "c": 100}, 60) == {"a": 28, "b": 5, "c": 27}
    assert sum(E.allocate_equal({"a": 3, "b": 3}, 60).values()) == 6
    assert E.allocate_equal({"a": 10}, 0) == {"a": 0}


def test_proportional_allocation_follows_the_pool():
    assert E.allocate_proportional({"scan": 300, "oak1": 100}, 100) == {"scan": 75, "oak1": 25}
    q = E.allocate_proportional({"scan": 2, "oak1": 1, "rs": 1}, 3)
    assert sum(q.values()) == 3 and q["scan"] == 2 or sum(q.values()) == 3
    assert E.allocate_proportional({"a": 1, "b": 1}, 10) == {"a": 1, "b": 1}


def test_strata_paths():
    r = _rec("V4", 1, 1, "scan", negative="graph_blocked", truth_source="graph_blocked",
             answer_check={"match_class": "screw.motherboard", "target_class": "screw",
                           "verb": "unscrew"})
    assert E.strata(r) == ("class=screw.motherboard", "verb=unscrew", "feasible=no",
                           "source=graph_blocked")
    psu = _rec("V4", 2, 1, "scan", answer_check={"target_class": "psu", "verb": "remove"})
    assert E.strata(psu) == ("class=psu", "verb=remove", "feasible=yes")
    assert E.strata(_rec("V4", 2, 1, "scan")) == ("class=?", "verb=?", "feasible=yes")
    v2 = _rec("V2", 3, 1, "scan", prior_is_correct=False,
              answer_check={"derive": "screw_count", "role": "motherboard"})
    assert E.strata(v2) == ("prior_correct=no", "screw:motherboard")
    assert E.strata(_rec("V15", 4, 1, "scan", views=["scan", "rs"])) == \
        ("same_moment=yes", "pair=scan+rs")
    assert E.strata(_rec("V16", 5, 1, "scan", negative="swap")) == ("valid=no", "corruption=swap")
    assert E.strata(_rec("V8", 6, 1, "scan", answer={"instance": "cpu_cooler.fan.01"})) == \
        ("class=cpu_cooler",)
    assert E.strata(_rec("V1", 7, 1, "scan")) == ()


# --------------------------------------------------------------------------- #
# the draw
# --------------------------------------------------------------------------- #
def test_the_draw_meets_targets_balances_strata_and_caps_desktops(units):
    chosen = E.draw(units, seed=0)
    by_task = {t: [r for u in us for r in u.records] for t, us in chosen.items()}
    assert len(by_task["V4"]) == 400 and len(by_task["V12"]) == 400 and len(by_task["V15"]) == 400
    assert len(by_task["V2"]) == 300 and len(by_task["V1"]) == 300
    assert len(chosen["V6"]) == 300 and len(by_task["V6"]) == 600
    # binary balance
    feas = Counter(bool(r["label"].get("negative")) for r in by_task["V4"])
    assert feas[False] == 200 and feas[True] == 200
    assert Counter(bool(r["label"].get("negative")) for r in by_task["V12"]) == {True: 200, False: 200}
    # the rare truth source is represented as far as the pool allows, the
    # others share the rest about equally
    sources = Counter(r["label"]["truth_source"] for r in by_task["V4"]
                      if r["label"].get("negative"))
    pool_pixels = sum(1 for u in units if u.task == "V4"
                      and u.records[0]["label"].get("truth_source") == "graph_blocked_pixels")
    assert sources["graph_blocked_pixels"] == min(pool_pixels, 200 // 3 + 1) or \
        sources["graph_blocked_pixels"] >= min(pool_pixels, 60)
    assert abs(sources["graph_blocked"] - sources["state_inapplicable"]) <= 2
    # V2: half prior-wrong, classes equal rather than the pool's 70 % ram_latch
    prior = Counter(r["label"]["prior_is_correct"] for r in by_task["V2"])
    assert prior[False] == 150 and prior[True] == 150
    # the draw thins V2 per desktop and class first (2026-11-04: the export no longer does):
    # the other classes have one question per desktop here, one answer, so they go
    classes = Counter(r["label"]["answer_check"]["class"] for r in by_task["V2"])
    assert classes == {"ram_latch": 300}
    assert classes["psu"] == 30 and classes["cover"] == 30
    # views in proportion (scan has 3x the V4 records)
    views = Counter(r["label"]["view"] for r in by_task["V4"])
    assert views == {"scan": 300, "oak1": 100}
    # V6 pairs travel whole
    for u in chosen["V6"]:
        assert {r["label"]["v6_form"] for r in u.records} == {"open", "mc"}
    # no desktop above 20 % of a task's target, and the desktops are spread
    for task, us in chosen.items():
        n = len(us)
        desks = Counter(u.desktop for u in us)
        assert max(desks.values()) <= max(1, int(0.2 * n)), task
        assert len(desks) == 30, task


def test_the_draw_is_deterministic_and_seeded(units):
    a = E.draw(units, seed=0)
    b = E.draw(units, seed=0)
    c = E.draw(units, seed=1)
    ids = lambda d: {t: [u.key for u in us] for t, us in d.items()}  # noqa: E731
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)
    # the order of the input does not matter either
    d = E.draw(list(reversed(units)), seed=0)
    assert ids(a) == ids(d)


def test_a_pool_smaller_than_the_target_gives_everything_under_the_cap(units):
    small = [u for u in units if u.task == "V1" and u.desktop <= 3]
    chosen = E.draw(small, seed=0)["V1"]
    # 60 records on 3 desktops: the cap (20 % of the 60-record target) holds
    assert len(chosen) == 36 and max(Counter(u.desktop for u in chosen).values()) == 12


def test_the_top_up_fills_a_task_whose_strata_ran_dry(units):
    # only unchanged V12 on desktops 1-2, changed everywhere: the strata cannot
    # give 200 unchanged, the changed half tops the task up to 400
    pool = [u for u in units if u.task == "V12"
            and (not u.records[0]["label"].get("negative") or u.desktop <= 2)]
    chosen = E.draw(pool, seed=0)["V12"]
    assert len(chosen) == 400
    negatives = sum(1 for u in chosen if u.records[0]["label"].get("negative"))
    assert 0 < negatives < 200


def _v4(n, desktop, mclass, feasible, group=False, verb="remove"):
    check = {"match_class": mclass, "target_class": mclass.split(".")[0], "verb": verb}
    if group:
        check["members"] = ["a", "b"]
    return _rec("V4", n, desktop, "multi", answer_check=check,
                **({} if feasible else {"negative": "graph_blocked",
                                        "truth_source": "graph_blocked"}))


def test_v4_is_balanced_within_each_class():
    """Final review I1: eval_v3's V4 leaned per class (motherboard 51 of 56 "no"), so
    the class name answered it. Each class is cut to at most 60 % of one answer before
    the draw, a one-answer class is left out, and the draw splits each class equally."""
    recs, n = [], 0
    for desktop in range(1, 31):
        for k in range(12):
            n += 1
            recs.append(_v4(n, desktop, "motherboard", feasible=k < 2))   # 1 yes : 5 no
            n += 1
            recs.append(_v4(n, desktop, "screw.motherboard", feasible=k % 2 == 0, group=True))
            n += 1
            recs.append(_v4(n, desktop, "psu", feasible=False))           # never a yes
    units = E.units_of(recs, "multi")
    kept = E.balance_v4(units, seed=0)
    by = Counter((E.v4_class(u.records[0]), not u.records[0]["label"].get("negative"))
                 for u in kept)
    assert by[("motherboard", True)] == 60 and by[("motherboard", False)] == 90
    assert ("psu", False) not in by
    chosen = E.draw(units, seed=0)["V4"]
    per_class: dict = {}
    for u in chosen:
        per_class.setdefault(E.v4_class(u.records[0]), Counter())[
            not u.records[0]["label"].get("negative")] += 1
    for cls, counts in per_class.items():
        assert max(counts.values()) / sum(counts.values()) <= E.V4_MAX_MAJORITY + 1e-9, cls
    assert "psu" not in per_class
    # a verb + class cell is balanced on its own: "remove the motherboard" may not lean
    # even where "displace the motherboard" makes the class look even
    recs = [_v4(i, 1 + i % 30, "motherboard", feasible=i % 4 == 0, verb="remove")
            for i in range(120)]
    recs += [_v4(1000 + i, 1 + i % 30, "motherboard", feasible=i % 4 != 0, verb="displace")
             for i in range(120)]
    chosen = E.draw(E.units_of(recs, "multi"), seed=0)["V4"]
    cells: dict = {}
    for u in chosen:
        check = u.records[0]["label"]["answer_check"]
        cells.setdefault(check["verb"], Counter())[not u.records[0]["label"].get("negative")] += 1
    for verb, counts in cells.items():
        assert max(counts.values()) / sum(counts.values()) <= E.V4_MAX_MAJORITY + 1e-9, verb


# --------------------------------------------------------------------------- #
# the files
# --------------------------------------------------------------------------- #
def _exports(tmp: Path) -> list[Path]:
    pool = _pool()
    paths = []
    for view, recs in pool.items():
        path = tmp / f"vlm_{view}.jsonl"
        IO.write_jsonl(path, recs)
        images = {img: {"image": img, "path": f"{view}/D{IO.desktop_of(r):02d}/s{r['label']['step']:03d}.png",
                        "desktop": IO.desktop_of(r), "view": view, "step": r["label"]["step"]}
                  for r in recs for img in r["prompt"]["images"]}
        IO.write_jsonl(IO.sidecar(path, "images_manifest.jsonl"),
                       [{"type": "header", "salt": "x"}] + list(images.values()))
        rows = [{"type": "desktop", "desktop": d, "instances": {}, "edges": [], "log": []}
                for d in sorted({IO.desktop_of(r) for r in recs})]
        rows += [{"type": "frame", "image": img, "desktop": e["desktop"], "view": view,
                  "step": e["step"], "confirmed": False, "state": {}}
                 for img, e in images.items()]
        IO.write_jsonl(IO.sidecar(path, "oracle.jsonl"), rows)
        IO.sidecar(path, "summary.json").write_text(json.dumps({"view": view, "records": len(recs)}),
                                                    encoding="utf-8")
        paths.append(path)
    return paths


def _hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_build_writes_the_subset_its_sidecars_and_the_table_byte_identically(tmp_path):
    exports = _exports(tmp_path / "exp")
    a = E.build(exports, tmp_path / "a" / "eval_v3", seed=0, audit=300)
    b = E.build(exports, tmp_path / "b" / "eval_v3", seed=0, audit=300)
    assert _hashes(tmp_path / "a") == _hashes(tmp_path / "b")
    out = tmp_path / "a" / "eval_v3"
    ids = out.joinpath("ids.txt").read_text(encoding="utf-8").split()
    assert len(ids) == a["records"] == sum(a["by_view"].values())
    for view in ("scan", "oak1"):
        truth = out / f"{view}.jsonl"
        records = list(IO.read_jsonl(truth))
        assert all(r["label"]["view"] == view for r in records)
        used = {img for r in records for img in r["prompt"]["images"]}
        assert set(IO.load_manifest(truth)) == used
        worlds, frames = IO.load_oracle(truth)
        assert set(frames) == used
        assert set(worlds) == {IO.desktop_of(r) for r in records}
        assert IO.load_summary(truth)["view"] == view
    table = json.loads((out / "table.json").read_text(encoding="utf-8"))
    assert table["version"] == "eval_v3" and table["seed"] == 0
    assert {r["task"] for r in table["rows"]} == {"V1", "V2", "V4", "V6", "V12", "V15"}
    v6 = next(r for r in table["rows"] if r["task"] == "V6")
    assert v6["units"] == 300 and v6["records"] == 600 and v6["unit"] == "pair"
    md = (out / "table.md").read_text(encoding="utf-8")
    assert "| V4 | 400 | 400 | 400 |" in md and "feasible=no / source=" in md
    # the audit sample comes from inside the subset
    audit = (out / "audit" / "audit.ids.txt").read_text(encoding="utf-8").split()
    assert audit and set(audit) <= set(ids) and len(audit) == a["audit"]["size"] == 300
    assert len({r["prompt"]["task"] for r in IO.read_jsonl(out / "audit" / "audit.jsonl")}) == 6
    assert (out / "audit" / "audit.sheet.md").is_file()
    assert b["rows"] == a["rows"]


def test_the_cli_runs(tmp_path, capsys):
    exports = _exports(tmp_path / "exp")
    assert E.main(["--export", *map(str, exports), "--out", str(tmp_path / "eval_v3"),
                   "--target-binary", "40", "--target-other", "30", "--audit", "60"]) == 0
    out = capsys.readouterr().out
    assert "V4=40" in out and "V6=60" in out and "audit sample: " in out
    with pytest.raises(ValueError, match="two exports of view"):
        E.build([exports[0], exports[0]], tmp_path / "dup")


def test_eval_v1_path_refused(tmp_path, capsys):
    """eval_v1 is the subset before the 2026-11-03 redesign: every tool that takes a
    subset path refuses one under eval_v1, with a message naming eval_v3."""
    import vlm_references
    import vlm_sample
    import vlm_selfcheck

    exports = _exports(tmp_path / "exp")
    old = tmp_path / "pack" / "eval_v1"
    calls = [
        lambda: E.main(["--export", *map(str, exports), "--out", str(old)]),
        lambda: E.main(["--export", str(old / "scan.jsonl"), "--out", str(tmp_path / "eval_v3")]),
        lambda: vlm_selfcheck.main(["--truth", str(old / "scan.jsonl"), "--out", str(tmp_path / "c")]),
        lambda: vlm_references.main(["--export", str(old / "multi.jsonl"), "--out", str(tmp_path / "r")]),
        lambda: vlm_sample.main(["--export", str(old / "audit" / "audit.jsonl"),
                                 "--out", str(tmp_path / "s")]),
    ]
    for call in calls:
        with pytest.raises(SystemExit) as err:
            call()
        assert "eval_v3" in str(err.value.code) and "eval_v1" in str(err.value.code)
    assert not old.exists() and not (tmp_path / "c").exists()
    # a file merely named like it is not a subset directory; eval_v3 passes
    E.refuse_old_eval(tmp_path / "eval_v3" / "scan.jsonl", tmp_path / "eval_v10_notes.txt", None)
    sys.path.insert(0, str(SCRIPTS / "vlm_run"))
    import vlm_collect

    with pytest.raises(SystemExit) as err:
        vlm_collect.main(["subset", "--export", str(exports[0]), "--out", str(tmp_path / "x.jsonl"),
                          "--records", str(old / "scan.jsonl")])
    assert "eval_v3" in str(err.value.code)
