"""Task U9 (importer): a sheet name that says ``remove`` / ``take`` is believed.

``cover`` and the four removable latches may be opened in place *or* taken
out; the importer drafted ``open`` for every one of them, which on D2 left six
parts the task card kept asking for after they had gone on the pile. Only the
pictures can say which most of them were, so the default stays ``open`` -- but
a name that *says* the part came out is taken at its word.
"""
from __future__ import annotations

import pytest

from tda.core.logs import import_log, sheet_verb
from tda.core.taxonomy import load_taxonomy, parse_raw_name

COLUMNS = ("Sequence Number", "Sequence Name", "Target Nest Group", "Tool Utility", "Notes")


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def synth(names, tax, desktop=99):
    names = ["Initial Conditions", *names]
    rows = [{c: "" for c in COLUMNS} | {"Sequence Number": str(i + 1), "Sequence Name": n}
            for i, n in enumerate(names)]
    return import_log(desktop, rows, {}, tax)


def test_the_two_way_classes_are_read_off_the_verbs(tax):
    assert tax.opened_or_taken_out() == frozenset(
        {"cover", "psu_latch", "drive_latch", "card_latch", "cooler_latch"})


#: The raw names of the 118 ``open`` actions on those classes (desktop counts
#: from the dataset survey), and what each one drafts now.
RAW_NAMES = [
    ("power module locker", "psu_latch", "open"),
    ("case cover for motherboard screws", "cover", "open"),
    ("gpu locker", "card_latch", "open"),
    ("cpu fan cover", "cover", "open"),
    ("ssd shield locker", "drive_latch", "open"),
    ("cpu heatsink cover", "cover", "open"),
    ("screw cover", "cover", "open"),
    ("cpu fan locker", "cooler_latch", "open"),
    ("gpu/screw cover", "cover", "open"),
    ("open cpu heatsink cover", "cover", "open"),
    ("Screw Cover", "cover", "open"),
    ("Front Cover", "cover", "open"),
    ("Power Module Blocker", "psu_latch", "open"),
    ("SSD Locker", "drive_latch", "open"),
    ("release cpu fan", "cooler_latch", "open"),
    ("remove SSD locker shield", "drive_latch", "remove"),
    ("Remove GPU locker", "card_latch", "remove"),
    ("take out power module locker", "psu_latch", "remove"),
    ("CPU fan locker - TAKE", "cooler_latch", "remove"),
    ("Remove RAM cover", "cover", "remove"),            # a prefix verb, as before
    # the word, not the letters: neither of these says the part came out
    ("removed? ssd locker", "drive_latch", "open"),
    ("ssd locker (takeaway)", "drive_latch", "open"),
]


@pytest.mark.parametrize("raw,cls,verb", RAW_NAMES, ids=[r[0] for r in RAW_NAMES])
def test_sheet_verb_table(tax, raw, cls, verb):
    parsed = parse_raw_name(raw)
    assert parsed.cls == cls
    assert sheet_verb(parsed.cls, raw, parsed.verb, tax) == verb


@pytest.mark.parametrize("raw,cls,verb", [
    ("remove RAM clip 1", "ram_latch", "open"),        # moulded into the board
    ("take CPU locker", "cpu_socket_lever", "open"),
    ("remove cable locker", "cable_clip", "release"),
    ("Remove motherboard screw 1", "screw", "unscrew"),
], ids=lambda v: str(v))
def test_only_the_two_way_classes_are_affected(tax, raw, cls, verb):
    parsed = parse_raw_name(raw)
    assert parsed.cls == cls
    assert sheet_verb(parsed.cls, raw, parsed.verb, tax) == verb


def test_the_import_drafts_remove_and_keeps_the_parts_apart(tax):
    """D2 steps 12 and 17: two parts, the second one's name says it came out.

    Which part a row names is still decided on the drafted verb: with
    ``remove`` the unnumbered displace-then-remove rule would have merged the
    shield into the locker opened before it.
    """
    li = synth(["SSD Locker", "remove SSD locker shield", "Screw Cover"], tax)
    assert [(a.step, a.target, a.verb) for a in li.actions] == [
        (2, "drive_latch.01", "open"),
        (3, "drive_latch.02", "remove"),
        (4, "cover.01", "open"),
    ]
    assert not [i for i in li.issues if "does not apply" in i or "reuses" in i]
