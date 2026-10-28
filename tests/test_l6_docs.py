"""L6: the spec and the two configuration files say what changed (round 4).

The spec gets one paragraph in §4.5 (the detector's v3 and ``conf_by_class``,
the asymmetric rule of the 「拆前 / 拆后」 chips), one in §5.1 (the v3 head) and
v1.30 on top of §14; ``configs/detector.yaml``'s step 1 says a class may have
its own floor; ``configs/sam.yaml``'s header carries L6's numbers and a
commented f5_v2 ``head:`` line.  The live values of both files are not
changed here: the controller flips them in the ship commit.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
DETECTOR = ROOT / "configs" / "detector.yaml"
SAM = ROOT / "configs" / "sam.yaml"


def test_the_spec_has_the_detector_paragraph_in_4_5():
    body = SPEC.read_text(encoding="utf-8")
    section = body.split("### 4.5 界面布局", 1)[1].split("### 4.6", 1)[0]
    head = "**小件检测器 v3 与按类下限（v1.30，L6）**"
    assert section.index("**拆前 / 拆后（v1.28，U28）**") < section.index(head)
    para = section.split(head, 1)[1].split("\n", 1)[0]
    for needed in ("`tda_det_rfdetr_small_all_v3.pt`", "923 帧、20 台", "D27 HP ProDesk 400 G1 SFF",
                   "**`conf_by_class`**", "缓存照旧存所有 ≥ `merge_floor` 的框",
                   "`conf_by_class: {screw: 0.15}`", "**数字都是同配方、不含这两台的折模型 F 的**",
                   "把 0.15 搬到 v3 上是假设", "32 对 22", "「v3 + 螺丝 0.15」对「现役 v2 + 0.05」",
                   "15/15", "接头 +7、内存条 +3", "**不对称**", "r 帧上按 `conf`（0.05）",
                   "12/15", "13/15", "还没存扫描仪 ROI"):
        assert needed in para, needed


def test_the_spec_has_the_head_bullet_in_5_1():
    body = SPEC.read_text(encoding="utf-8")
    section = body.split("### 5.1 交互式分割（MVP）", 1)[1].split("### 5.2", 1)[0]
    head = "**解码头 v3（v1.30，L6）**"
    assert section.index("**两个解码头（v1.19，U19）**") < section.index(head)
    para = section.split(head, 1)[1].split("\n", 1)[0]
    for needed in ("`sam2.1_tda_head_v3.pt`", "24 359 个可见掩码", "141 个张量",
                   "**数字是不含这两台、18 台训练的折模型 S_F 的**", "0.742→0.763", "0.471→0.504",
                   "0.664→0.705", "0.791→0.816", "−0.0065", "−0.034", "注释掉的 `head:` 一行是 f5_v2"):
        assert needed in para, needed


def test_v1_30_is_on_top_of_14():
    changes = SPEC.read_text(encoding="utf-8").split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # on top when it landed; U32's v1.31 has gone above it since (and U33-U37's v1.32-v1.36)
    assert entries[0].startswith(("- **v1.30", "- **v1.31", "- **v1.32", "- **v1.33",
                                  "- **v1.34", "- **v1.35", "- **v1.36",
                                  "- **v1.37", "- **v1.38", "- **v1.39"))
    [entry] = [e for e in entries[:10] if e.startswith("- **v1.30")]
    assert entry.startswith("- **v1.30（2026-10-25）**：**检测器与 SAM 头用全部已确认帧重训；"
                            "按类置信度下限**（L6，§4.5、§5.1）")
    for needed in ("923 个已确认扫描仪帧", "`conf_by_class: {screw: 0.15}`", "F / S_F", "是假设",
                   "32 对 22", "15/15 对 14/15", "不对称", "12/15 → 13/15", "生效值由上线提交切换"):
        assert needed in entry, needed
    assert changes.index("- **v1.30") < changes.index("- **v1.28")


def test_the_detector_files_step_one_names_the_class_floor():
    text = DETECTOR.read_text(encoding="utf-8")
    assert "conf >= `conf` or the class's\n#      own floor (`conf_by_class`, below)" in text
    assert "conf_by_class:\n" in text and "screw: 0.15" in text
    # the live values are the ship commit's to change (tests/test_detector.py pins them)


def test_the_sam_files_header_has_l6_and_the_f5_v2_fallback():
    text = SAM.read_text(encoding="utf-8")
    for needed in ("# L6 (experiments/l6_models/REPORT.md)", "24 359 visible masks of 923 frames",
                   "the fold model's", "box 0.742 -> 0.763", "single click 0.471 -> 0.504",
                   "-0.0065", "-0.034", "# head: D:/DataSet/models/weights/sam2.1_tda_head_f5_v2.pt"):
        assert needed in text, needed
    # the live head is the ship commit's to change (tests/test_u19_heads.py pins it)
