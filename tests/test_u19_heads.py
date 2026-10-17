"""Task U19: SAM with two decoder heads -- the routing table and the configuration.

The ruling (``experiments/l5_sam/REPORT.md`` section 0): one image encoder,
two (prompt encoder, mask decoder) pairs.  A box -> the fine-tuned head; no box
and a class in ``stock_point_classes`` (the large classes, and two measured
worse on a single click) -> the stock head; no box otherwise -> the fine-tuned
head; a refinement -> the head that produced the mask it refines.  All pure: the
service is ``test_u19_service.py`` (a toy model) and ``test_u19_weights.py``
(the real one), the window ``test_u19_app.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from tda.models import sam_heads as H
from tda.models.sam_heads import (
    STOCK,
    TUNED,
    Routing,
    SamHeadConfigError,
    choose_head,
    large_classes_from_bands,
    load_sam_head_config,
    parse_sam_head_config,
)

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "configs" / "sam.yaml"
BANDS = ROOT / "configs" / "prompt_box_bands.yaml"
#: The members of ``stock_point_classes`` by the reason each is there.
BY_BAND = frozenset({"cpu_cooler", "cpu_socket_lever", "drive_cage", "motherboard",
                     "optical_drive", "psu", "storage_drive"})
NAMED_BY_THE_RULING = frozenset({"chassis", "expansion_card"})
#: L5, per-class single-click IoU of the tuned head minus the stock head's on
#: the two held-out desktops, each with its own fold's model: D5 (f5_v2, blind)
#: / D13 (f13_v2, not blind) -- the U19 review's addition.
MEASURED_WORSE = {"ram_module": (-0.096, -0.023), "drive_latch": (-0.027, -0.191)}
STOCK_POINT = BY_BAND | NAMED_BY_THE_RULING | frozenset(MEASURED_WORSE)
ROUTING = Routing(stock_point_classes=STOCK_POINT)


# --------------------------------------------------------------------------- #
# the routing table (pure)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("has_box, cls, mask_input_head, expected", [
    # a box -- detector, difference map, a drag, point+box -- is the tuned head's
    (True, "screw", None, TUNED),
    (True, "motherboard", None, TUNED),
    (True, "chassis", None, TUNED),
    (True, None, None, TUNED),
    # no box: a class in stock_point_classes goes to the stock head ...
    (False, "motherboard", None, STOCK),
    (False, "chassis", None, STOCK),
    (False, "cpu_cooler", None, STOCK),
    (False, "expansion_card", None, STOCK),
    (False, "cpu_socket_lever", None, STOCK),
    (False, "ram_module", None, STOCK),       # measured worse with the tuned head
    (False, "drive_latch", None, STOCK),
    (True, "ram_module", None, TUNED),        # ... on a single click, not with a box
    # ... every other class, and no class at all, to the tuned one
    (False, "screw", None, TUNED),
    (False, "connector", None, TUNED),
    (False, "ram_latch", None, TUNED),
    (False, "psu_latch", None, TUNED),
    (False, "cover", None, TUNED),            # its scan band tops out at 17 004 px
    (False, None, None, TUNED),
    (False, "other", None, TUNED),            # a 框外零件 of no class (U18)
    (False, "no_such_class", None, TUNED),
    # a refinement goes back to the head whose answer it refines, box or not
    (True, "screw", STOCK, STOCK),
    (False, "motherboard", TUNED, TUNED),
    (False, "screw", STOCK, STOCK),
    # a tag that names no head is no tag
    (False, "motherboard", "sam3", STOCK),
    (True, "motherboard", "", TUNED),
])
def test_the_routing_table(has_box, cls, mask_input_head, expected):
    assert choose_head(ROUTING, has_box=has_box, instance_cls=cls,
                       mask_input_head=mask_input_head) == expected


def test_the_routes_are_configuration_not_code():
    flipped = Routing(box_prompts=STOCK, point_listed=TUNED, point_other=STOCK,
                      stock_point_classes=frozenset({"psu"}))
    assert choose_head(flipped, has_box=True, instance_cls="screw") == STOCK
    assert choose_head(flipped, has_box=False, instance_cls="psu") == TUNED
    assert choose_head(flipped, has_box=False, instance_cls="screw") == STOCK


# --------------------------------------------------------------------------- #
# the shipped configuration
# --------------------------------------------------------------------------- #
def _shipped() -> dict:
    return yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))


def test_the_shipped_file_is_the_ruling():
    config = parse_sam_head_config(_shipped(), SHIPPED)
    assert config.off == ""
    assert config.head == Path("D:/DataSet/models/weights/sam2.1_tda_head_f5_v2.pt")
    assert config.routing == Routing(box_prompts=TUNED, point_listed=STOCK, point_other=TUNED,
                                     stock_point_classes=STOCK_POINT)


def test_the_stock_point_classes_are_the_bands_the_ruling_and_two_measured():
    """The scan bands above 20 000 px, two large classes they do not cover, two measured.

    When ``prompt_box_bands.yaml`` is re-fitted and a class crosses the line,
    this says so: the list in ``configs/sam.yaml`` is what the app reads.
    """
    bands = yaml.safe_load(BANDS.read_text(encoding="utf-8"))
    by_band = set(large_classes_from_bands(bands))
    assert by_band == BY_BAND
    shipped = set(_shipped()["stock_point_classes"])
    assert by_band <= shipped
    scan = bands["views"]["scan"]
    assert shipped - by_band == NAMED_BY_THE_RULING | {"ram_module", "drive_latch"}
    for cls in NAMED_BY_THE_RULING:          # large, with no scan band to say so
        assert cls not in scan["classes"]
    # the two measured additions: not large by any band, worse on both held-out
    # desktops (each with its own fold's model)
    assert scan["classes"]["ram_module"]["hi_px"] < 20_000
    assert "drive_latch" not in scan["classes"]
    for d5, d13 in MEASURED_WORSE.values():
        assert d5 < 0 and d13 < 0
    # cover was named in the ruling's draft list, and the rule leaves it out
    assert scan["classes"]["cover"]["hi_px"] < 20_000 and "cover" not in shipped


def test_every_member_says_why_it_is_there():
    reasons = {cls: "band > 20 000 px" for cls in BY_BAND}
    reasons.update({cls: "named by the ruling" for cls in NAMED_BY_THE_RULING})
    reasons.update({cls: "measured worse" for cls in MEASURED_WORSE})
    lines = SHIPPED.read_text(encoding="utf-8").split("\nstock_point_classes:\n", 1)[1]
    members = [line for line in lines.splitlines() if line.startswith("  - ")]
    assert len(members) == len(STOCK_POINT)
    for line in members:
        cls, _, comment = line[4:].partition("#")
        assert reasons[cls.strip()] in comment, line
    for cls, (d5, d13) in MEASURED_WORSE.items():
        line = next(m for m in members if m[4:].split()[0] == cls)
        assert f"{d5:.3f} (D5)" in line and f"{d13:.3f} (D13)" in line, line


# --------------------------------------------------------------------------- #
# the parser's refusals
# --------------------------------------------------------------------------- #
def _good() -> dict:
    return {"enabled": True, "head": "head.pt",
            "route": {"box_prompts": "tuned",
                      "point_only": {"stock_point_classes": "stock", "other": "tuned"}},
            "stock_point_classes": ["motherboard", "psu"]}


def _with(path: tuple, value) -> dict:
    data = _good()
    target = data
    for key in path[:-1]:
        target = target[key]
    if value is _DROP:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return data


_DROP = object()


@pytest.mark.parametrize("data, says", [
    (["enabled", True], "not a mapping"),
    (_with(("stock_point_class",), ["psu"]), "unknown keys ['stock_point_class']"),
    (_with(("large_classes",), ["psu"]), "unknown keys ['large_classes']"),   # the old name
    (_with(("enabled",), "yes"), "enabled = 'yes'"),
    (_with(("head",), None), "head must be a path"),
    (_with(("head",), "  "), "head must be a path"),
    (_with(("route",), "tuned"), "route must be a mapping"),
    (_with(("route", "box_prompt"), "tuned"), "route has unknown keys ['box_prompt']"),
    (_with(("route", "box_prompts"), _DROP), "route is missing ['box_prompts']"),
    (_with(("route", "box_prompts"), "fine"), "route.box_prompts = 'fine'"),
    (_with(("route", "point_only"), "stock"), "route.point_only must be a mapping"),
    (_with(("route", "point_only", "small"), "tuned"), "unknown keys ['small']"),
    (_with(("route", "point_only", "other"), _DROP), "route.point_only is missing ['other']"),
    (_with(("route", "point_only", "stock_point_classes"), 1),
     "route.point_only.stock_point_classes = 1"),
    (_with(("route", "point_only", "large"), "stock"), "unknown keys ['large']"),
    (_with(("stock_point_classes",), []), "stock_point_classes must be a non-empty list"),
    (_with(("stock_point_classes",), "motherboard"),
     "stock_point_classes must be a non-empty list"),
    (_with(("stock_point_classes",), ["psu", 3]), "stock_point_classes must be a non-empty list"),
    (_with(("stock_point_classes",), ["mother_board"]),
     "['mother_board'] are not taxonomy classes"),
])
def test_the_parser_refuses(data, says):
    with pytest.raises(SamHeadConfigError) as refused:
        parse_sam_head_config(data, Path("x/sam.yaml"))
    assert says in str(refused.value)


def test_a_bare_head_name_is_in_the_weights_dir_and_an_absolute_one_is_kept():
    """U19 review N2: the head follows the weights_dir the app was started with."""
    source = Path("C:/cfg/sam.yaml")
    weights = Path("E:/other_disk/weights")
    bare = parse_sam_head_config(_good(), source, weights_dir=str(weights))
    assert bare.head == weights / "head.pt"
    nested = parse_sam_head_config(_with(("head",), "heads/head.pt"), source, weights_dir=weights)
    assert nested.head == weights / "heads" / "head.pt"
    absolute = parse_sam_head_config(_with(("head",), "D:/DataSet/models/weights/h.pt"), source,
                                     weights_dir=weights)
    assert absolute.head == Path("D:/DataSet/models/weights/h.pt")
    # no weights_dir to go by (library use): beside the configuration file
    for empty in (None, "", "  "):
        assert parse_sam_head_config(_good(), source, weights_dir=empty).head == \
            Path("C:/cfg/head.pt")


def test_load_hands_the_weights_dir_to_the_head(tmp_path, caplog):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "head.pt").write_bytes(b"x")
    path = _write(tmp_path, _good())
    with caplog.at_level(logging.INFO):
        found = load_sam_head_config(str(path), weights_dir=str(weights))
        missing = load_sam_head_config(str(path))           # beside the file: not there
    assert found.wanted and found.head == weights / "head.pt"
    assert not missing.wanted and "does not exist" in missing.off


def test_a_relative_head_is_beside_the_file_and_enabled_false_is_off():
    config = parse_sam_head_config(_good(), Path("C:/cfg/sam.yaml"))
    assert config.head == Path("C:/cfg/head.pt")
    off = parse_sam_head_config(_with(("enabled",), False), Path("C:/cfg/sam.yaml"))
    assert off.off.startswith("enabled: false") and not off.wanted
    assert off.routing.stock_point_classes == frozenset({"motherboard", "psu"})


def _write(tmp_path: Path, data, name: str = "sam.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data) if not isinstance(data, str) else data,
                    encoding="utf-8")
    return path


def _warnings(caplog, logger: str = "tda.models") -> list:
    return [r for r in caplog.records if r.levelno >= logging.WARNING
            and r.name.startswith(logger)]


@pytest.mark.parametrize("content, says", [
    (None, "does not exist"),
    ("route: [unclosed", "cannot be read"),
    ("- a list", "not a mapping"),
])
def test_an_unusable_file_is_one_warning_and_off(tmp_path, caplog, content, says):
    path = tmp_path / "sam.yaml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with caplog.at_level(logging.INFO):
        config = load_sam_head_config(str(path))
    assert not config.wanted and says in config.off
    assert len(_warnings(caplog)) == 1
    assert "stock head answers every prompt" in _warnings(caplog)[0].getMessage()


def test_a_missing_head_file_is_one_warning_and_off(tmp_path, caplog):
    path = _write(tmp_path, _with(("head",), str(tmp_path / "nowhere.pt")))
    with caplog.at_level(logging.INFO):
        config = load_sam_head_config(str(path))
    assert not config.wanted and "nowhere.pt does not exist" in config.off
    assert len(_warnings(caplog)) == 1


def test_enabled_false_is_one_info_line_and_no_warning(tmp_path, caplog):
    path = _write(tmp_path, _with(("enabled",), False))
    with caplog.at_level(logging.INFO):
        config = load_sam_head_config(str(path))
    assert config.off.startswith("enabled: false") and _warnings(caplog) == []
    assert any("SAM fine-tuned head off" in r.getMessage() for r in caplog.records)


def test_the_environment_says_off_or_names_another_file(tmp_path, caplog, monkeypatch):
    monkeypatch.setenv(H.ENV_VAR, "OFF")
    with caplog.at_level(logging.INFO):
        config = load_sam_head_config()
    assert config.off == "TDA_SAM_HEAD=off" and not config.wanted and _warnings(caplog) == []

    (tmp_path / "head.pt").write_bytes(b"x")
    other = _write(tmp_path, _good(), "other.yaml")
    monkeypatch.setenv(H.ENV_VAR, str(other))
    config = load_sam_head_config()
    assert config.wanted and config.source == other
    assert config.head == tmp_path / "head.pt"

    # an explicit path wins over the environment
    monkeypatch.setenv(H.ENV_VAR, "off")
    assert load_sam_head_config(str(other)).wanted


def test_the_suite_runs_with_the_head_off():
    """conftest.py: a window that loads SAM in a test loads the stock head alone."""
    assert load_sam_head_config().off == "TDA_SAM_HEAD=off"


