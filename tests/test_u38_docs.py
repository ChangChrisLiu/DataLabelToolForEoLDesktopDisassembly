"""Task U38's documents: §4.2's 缺帧处理 says a missing frame has no image and
cannot be confirmed and that its cache goes with a renumbering, v1.37 is on top
of §14, and the guide is untouched -- still inside its 7700-character budget,
no new key in its table."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from tda.ui import app_actions as A
from tda.ui.panels.taskcard import NO_PHOTO_SENTENCE

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
GUIDE = ROOT / "docs" / "annotation_guide.md"


def _spec() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_the_missing_frames_paragraph_says_no_image_no_confirmation_no_cache():
    section = _spec().split("### 4.2 倒序标注", 1)[1].split("### 4.3", 1)[0]
    para = section.split("**缺帧处理**：", 1)[1].split("\n", 1)[0]
    assert para.startswith("扫描仪缺第 k 步时，S2 跳过该帧掩码，状态与锚点按逻辑步照常；"
                           "其他视角对无扫描仪对应帧的步回退为原生倒序标注。")
    for needed in ("**没有图像、不能确认**（v1.37，U38）", "帧行 `missing`，或没有路径",
                   "缓存里同号的文件、`aux` 里留下的 `cache_path` 都是别的照片的",
                   "画布和时间轴都不读", "编译也不量它（只用视角的推断尺寸）",
                   "`Space` 和复查的 `Enter` 都由 `verify_frame` 拒绝",
                   "「step k 没有照片（漏拍），不能确认」", "`cli check` 不为它写行也不报它",
                   "换号时它的缓存作废", "`load-index` 清掉量得的尺寸和 `cache_path`",
                   "副本、缩略图和 manifest 记录", "`--prune-missing` 删掉"):
        assert needed in para, needed


def test_v1_37_is_on_top_of_14():
    changes = _spec().split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    [entry] = [e for e in entries[:3] if e.startswith("- **v1.37")]
    assert entry.startswith("- **v1.37（2026-10-27）**：**缺帧没有图像、不能确认**（U38，§4.2）")
    for needed in ("`scan.61.skipped_steps: [26]`", "23 行签名", "`s026.png`", "`RGB261`",
                   "`tda.core.cache.has_no_photo`", "`frame_hw` 不量 `missing` 行",
                   "`frame_missing:`", "`no_photo()`",
                   "this view has no photo of this step (a missed shot): there is nothing "
                   "to confirm", NO_PHOTO_SENTENCE,
                   "step k 没有照片（漏拍），不能确认", "`run_pending_rechecks`",
                   "U10 第三轮的测试钉着", "`--prune-missing`",
                   "D1 扫描仪第 27、33 步，D41 第 31 步，D61 第 26 步",
                   "库结构不变（仍为 v6）", "没有新快捷键"):
        assert needed in entry, needed
    assert changes.index("- **v1.37") < changes.index("- **v1.36")


def test_the_guide_is_untouched_and_in_its_budget():
    body = GUIDE.read_text(encoding="utf-8")
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700
    assert A.shortcut_markdown() in body
