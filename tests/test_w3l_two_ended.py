"""Task W3l: ``attrs.two_ended`` on a connector -- its cable is detachable at both ends whatever its kind.

The HP SFF family (D22-D33) powers the optical drive from a board header: the sheet's 「Optical drive - motherboard
connector」 is the board end, the drive-end plugs were never pulled (user 2026-11-01: 「这两个 connector 在这次一直没拆下来」).
A ``sata_power`` kind alone makes :func:`connector_owner` hand the plug to the PSU harness, and the PSU leaves at step 18
with the plug still in -- a violation the picture does not show.  The flag keeps the plug off the PSU; it still gates its
own socket host (the drive cage the drive is one piece with).
"""
from __future__ import annotations

from pathlib import Path

from tda.core.graph_rules import TWO_ENDED_ATTR, connector_owner, propose_edges
from tda.core.model import InstanceRec
from tda.core.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
HANDOFF = ROOT / "docs" / "HANDOFF_2026-10-28.md"


def _rec(key: str, cls: str, **kw) -> InstanceRec:
    attrs = kw.pop("attrs", {})
    return InstanceRec(key=key, desktop=23, cls=cls, attrs=dict(attrs), **kw)


def _table(power_attrs: dict) -> dict:
    return {
        "chassis": _rec("chassis", "chassis"),
        "motherboard.01": _rec("motherboard.01", "motherboard"),
        "psu.01": _rec("psu.01", "psu"),
        "drive_cage.01": _rec("drive_cage.01", "drive_cage"),
        "connector.04": _rec("connector.04", "connector", socket_host="motherboard.01", cable="cable:optical_drive"),
        "connector.sata_data.01": _rec("connector.sata_data.01", "connector", attrs={"kind": "sata_data"},
                                       parent="drive_cage.01", attached=True, socket_host="drive_cage.01",
                                       cable="cable:optical_drive"),
        "connector.sata_power.01": _rec("connector.sata_power.01", "connector", attrs={"kind": "sata_power", **power_attrs},
                                        parent="drive_cage.01", attached=True, socket_host="drive_cage.01",
                                        cable="cable:optical_drive"),
    }


def test_the_flag_takes_a_power_plug_off_the_psu_harness():
    plain = _table({})
    assert connector_owner(plain["connector.sata_power.01"], plain) == "psu.01"
    flagged = _table({TWO_ENDED_ATTR: True})
    assert connector_owner(flagged["connector.sata_power.01"], flagged) is None
    # a data lead was two-ended already; the flag changes nothing for it
    assert connector_owner(flagged["connector.sata_data.01"], flagged) is None
    assert connector_owner(_table({TWO_ENDED_ATTR: False})["connector.sata_power.01"], plain) == "psu.01"


def test_the_edges_follow():
    tax = load_taxonomy()

    def connected(table):
        return sorted((e.target, e.blocker) for e in propose_edges(table, tax) if e.type == "connected_to"
                      and e.blocker in ("connector.sata_power.01", "connector.sata_data.01"))

    assert connected(_table({})) == [("drive_cage.01", "connector.sata_data.01"), ("drive_cage.01", "connector.sata_power.01"),
                                     ("psu.01", "connector.sata_power.01")]
    assert connected(_table({TWO_ENDED_ATTR: True})) == [("drive_cage.01", "connector.sata_data.01"),
                                                         ("drive_cage.01", "connector.sata_power.01")]


def test_the_docs_say_it():
    spec = SPEC.read_text(encoding="utf-8")
    rules = spec.split("### 7.2 硬约束语义", 1)[1].split("### 7.3", 1)[0]
    [line] = [l for l in rules.splitlines() if l.startswith("**接口规则**")]
    for needed in ("two_ended", "W3l", "v1.57"):
        assert needed in line, needed
    entries = [line for line in spec.split("## 14. 修订记录", 1)[1].splitlines() if line.startswith("- ")]
    # on top when it landed; whatever lands later goes above it
    [entry] = [e for e in entries[:1] if e.startswith("- **v1.57（2026-11-01）**")]
    for needed in ("（W3l，", "two_ended", "connector_owner", "sata_power", "drive_cage.01", "tests/test_w3l_two_ended.py"):
        assert needed in entry, needed
    assert spec.index("- **v1.57") < spec.index("- **v1.56") < spec.index("- **v1.55")
    handoff = HANDOFF.read_text(encoding="utf-8")
    export = handoff.split("## 5. 导出与 HuggingFace 准备", 1)[1].split("\n## ", 1)[0]
    [line] = [l for l in export.splitlines() if "W3l" in l]
    for needed in ("two_ended", "v1.57"):
        assert needed in line, needed
