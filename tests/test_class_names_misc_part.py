"""A named misc part reads by its name: the chassis I/O panel (user, 2026-10-28, D36 step 24
"The USB part on the case" -- 「把这个其他零件换成我们之前说的机箱接口」)."""
from tda.ui.class_names import class_zh, instance_label


def test_the_front_io_module_reads_as_the_chassis_io_panel():
    assert class_zh("misc_part", {"name": "front_io_module"}) == "机箱接口面板"
    assert instance_label("misc_part.01", "misc_part", {"name": "front_io_module"}) == "机箱接口面板 misc_part.01"


def test_an_unnamed_misc_part_still_reads_as_other():
    assert class_zh("misc_part", {}) == "其它零件"
    assert class_zh("misc_part", {"name": "something_else"}) == "其它零件"
