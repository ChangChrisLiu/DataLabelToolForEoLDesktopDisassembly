"""``scripts/export_scanpack.py --update``: rebuild some desktops, take the others as they are.

On the two U30 scenes (D2 and D13, :mod:`test_u30_scanpack`) in one database:
an update that rebuilds nothing writes the old package again byte for byte; one
that rebuilds D13 after a change to it writes exactly what a full build over the
changed database writes, D2's files untouched; a package whose files no longer
match its manifest is refused, and so is an output folder inside the old one.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import export_scanpack as S  # noqa: E402
import test_u30_scanpack as U  # noqa: E402
from tda.core.db import Db  # noqa: E402

EXCLUDE = {(d, U.EXCLUDED[d]): U.EXCLUDE_REASON for d in (2, 13)}


def files_of(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def pack(root: Path, db: Path, name: str, desktops, **kw) -> dict:
    kw.setdefault("tail", U.below_raw)
    kw.setdefault("expected_hw", U.HW)
    return S.build_pack(str(db), str(root / name), desktops, cache_dir=str(root / "cache"),
                        work_dir=str(root / f"work_{name}"), **kw)


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    root = tmp_path_factory.mktemp("scanpack_update")
    U.build_scene(root, 2)
    db = U.build_scene(root, 13)          # the same database: both desktops
    pack(root, db, "old", [2, 13], exclude=EXCLUDE, notes=[U.NOTE])
    # the change: a part of D13 gains an attribute (as the RAM latches' `of`)
    changed = root / "changed.sqlite"
    shutil.copyfile(db, changed)
    handle = Db(str(changed))
    key = sorted(k for k in handle.instances(13) if k.startswith("ram_latch."))[0]
    rec = handle.instances(13)[key]
    rec.attrs = dict(rec.attrs or {}, of="ram_module.01")
    handle.upsert_instance(rec)
    handle.close()
    return {"root": root, "db": db, "changed": changed, "old": root / "old", "key": key}


def test_rebuilding_nothing_writes_the_old_package_again(scenes):
    root = scenes["root"]
    pack(root, scenes["db"], "noop", [], update_from=str(scenes["old"]))
    assert files_of(root / "noop") == files_of(scenes["old"])


def test_rebuilding_one_desktop_is_a_full_build_of_the_changed_database(scenes):
    root = scenes["root"]
    pack(root, scenes["changed"], "full_changed", [2, 13], exclude=EXCLUDE, notes=[U.NOTE])
    summary = pack(root, scenes["changed"], "upd", [13], update_from=str(scenes["old"]),
                   exclude={(13, U.EXCLUDED[13]): U.EXCLUDE_REASON})
    new, old = files_of(root / "upd"), files_of(scenes["old"])
    assert new == files_of(root / "full_changed")
    assert summary["exclusions"] == [{"desktop": d, "step": s, "reason": U.EXCLUDE_REASON}
                                     for d, s in sorted(EXCLUDE)]
    assert summary["notes"] == [U.NOTE]                    # carried from the old package
    assert {r for r in new if r.startswith("D02/")} == {r for r in old if r.startswith("D02/")}
    assert all(new[r] == old[r] for r in new if r.startswith("D02/"))
    changed = sorted(r for r in new if new[r] != old.get(r))
    assert "D13/instances.json" in changed and "manifest.json" in changed
    assert not [r for r in changed if r.startswith("D02/")]
    assert scenes["key"].encode() in new["D13/instances.json"]
    assert b'"of": "ram_module.01"' in new["D13/instances.json"]


def test_a_package_that_no_longer_matches_its_manifest_is_refused(scenes):
    root = scenes["root"]
    edited = root / "edited"
    shutil.copytree(scenes["old"], edited)
    target = edited / "D02" / "frames.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(S.PackError, match="not the file its manifest lists"):
        pack(root, scenes["db"], "from_edited", [13], update_from=str(edited),
             exclude={(13, U.EXCLUDED[13]): U.EXCLUDE_REASON})
    shutil.rmtree(edited)
    shutil.copytree(scenes["old"], edited)
    (edited / "D02" / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(S.PackError, match="exactly the files its manifest lists"):
        pack(root, scenes["db"], "from_stray", [13], update_from=str(edited),
             exclude={(13, U.EXCLUDED[13]): U.EXCLUDE_REASON}, force=True)


def test_the_output_may_not_be_the_old_package_or_inside_it(scenes):
    with pytest.raises(S.PackError, match="folder of its own"):
        S.build_pack(str(scenes["db"]), str(scenes["old"] / "new"), [13],
                     cache_dir=str(scenes["root"] / "cache"), tail=U.below_raw,
                     expected_hw=U.HW, update_from=str(scenes["old"]),
                     work_dir=str(scenes["root"] / "work_inside"))
