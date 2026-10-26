"""U30-QA, pass 6: the report's tables, as Markdown text (``<root>/report_tables.txt``).

Reads ``ranked.json`` / ``all_items.json`` (``u30_rank.py``), the analysis and the
sheet index; writes nothing but the text file. The narrative of the report is
written by hand around these tables.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import u30_common as U  # noqa: E402

DELL = [d for d in U.DESKTOPS if d <= 21]
HP = [d for d in U.DESKTOPS if d >= 22]


def short(text: str, n: int = 150) -> str:
    text = str(text).replace("|", "/")
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> int:
    R = json.load(open(os.path.join(U.QA_ROOT, "ranked.json"), encoding="utf-8"))
    items = json.load(open(os.path.join(U.QA_ROOT, "all_items.json"), encoding="utf-8"))
    sheets = json.load(open(os.path.join(U.QA_ROOT, "sheets", "index_desktops.json"),
                            encoding="utf-8"))
    out = []
    w = out.append
    # ---- 1. check
    w("### T1 cli check\n")
    w("| 台 | 帧 | problems | shape_predates_state | cut_copy_in_force | zorder_floor | "
      "missing_shape / empty_visible | 完全被挡住（零件 / 零件帧） | open conflicts |")
    w("|---|---|---|---|---|---|---|---|---|")
    for d in U.DESKTOPS:
        k = R["desktops"][str(d)]["check"]
        c = k["codes"]
        w(f"| D{d:02d} | {R['desktops'][str(d)]['summary']['frames']} | {k['problems']} | "
          f"{c.get('shape_predates_state', 0)} | {c.get('cut_copy_in_force', 0)} | "
          f"{c.get('zorder_floor', 0)} | {c.get('missing_shape', 0)} / {c.get('empty_visible', 0)} | "
          f"{k['covered_parts']} / {k['covered_part_frames']} | {k['open_conflicts']} |")
    w("")
    # ---- 2. flags per desktop
    codes = ["a_empty", "b_floor", "c_below", "d_roi", "e_dup", "vis_outside_amodal", "jump",
             "amodal_jump", "frame_unchanged", "leave_no_change", "zorder_pixels",
             "visible_changed", "change_unexplained"]
    w("### T2 自动标记（high / 全部）\n")
    w("| 台 | " + " | ".join(codes) + " | 负担 |")
    w("|---|" + "---|" * (len(codes) + 1))
    for d in U.DESKTOPS:
        fl = [f for f in items if f["desktop"] == d and f.get("source") == "auto"]
        cells = []
        for c in codes:
            n = sum(1 for f in fl if f["code"] == c)
            h = sum(1 for f in fl if f["code"] == c and f["severity"] == "high")
            cells.append("·" if n == 0 else f"{h}/{n}")
        w(f"| D{d:02d} | " + " | ".join(cells) + f" | {R['desktops'][str(d)]['burden']} |")
    w("")
    # ---- 3. SAM
    sam = R.get("sam") or {}
    if sam:
        w(f"### T3 SAM 一致度（{sam['n']} 个 (帧, 零件)）\n")
        o = sam["overall"]
        w(f"全部：IoU(SAM, 我们的可见) 出厂头 中位数 {o['tuned_vis']['median']}（p10 "
          f"{o['tuned_vis']['p10']}，< 0.5 的 {o['tuned_vis']['lt_0.5']}）；原版头 中位数 "
          f"{o['stock_vis']['median']}（p10 {o['stock_vis']['p10']}，< 0.5 的 "
          f"{o['stock_vis']['lt_0.5']}）；IoU(SAM 出厂头, 非模态) 中位数 {o['tuned_amodal']['median']}\n")
        w("| 类别 | n | 出厂头 中位数 / p10 / <0.5 | 原版头 中位数 / p10 / <0.5 | 出厂头 vs 非模态 中位数 |")
        w("|---|---|---|---|---|")
        for cls, t in sam["by_class"].items():
            a, b, m = t["tuned_vis"], t["stock_vis"], t["tuned_amodal"]
            w(f"| {cls} | {a['n']} | {a['median']} / {a['p10']} / {a['lt_0.5']} | "
              f"{b['median']} / {b['p10']} / {b['lt_0.5']} | {m['median']} |")
        w("")
        w("| 台 | n | 出厂头 中位数 / p10 / <0.5 | 原版头 中位数 / p10 / <0.5 |")
        w("|---|---|---|---|")
        for d, t in sam["by_desktop"].items():
            a, b = t["tuned_vis"], t["stock_vis"]
            w(f"| D{int(d):02d} | {a['n']} | {a['median']} / {a['p10']} / {a['lt_0.5']} | "
              f"{b['median']} / {b['p10']} / {b['lt_0.5']} |")
        w("")
        w("### T3b 一致度最低的 200 行（前 60 行；全部在 ranked.json 的 sam.low200，图在 low_01…）\n")
        kinds = Counter(r.get("kind") for r in sam["low200"])
        w("200 行的类型：" + "；".join(f"{k} {v}" for k, v in kinds.most_common()) + "\n")
        w("| # | 台 | 帧 | 零件 | 标签 / 遮挡 | 可见 / 非模态 px | SAM px | 出厂头 IoU(可见 / 非模态) | 原版头 IoU | 类型 |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for n, r in enumerate(sam["low200"][:60], 1):
            w(f"| {n} | D{r['desktop']:02d} | {r['step']} | {r['inst']} | {r['visibility']} / "
              f"{(r['occ'] or 0):.2f} | {r['vis_area']} / {r['amodal_area']} | {r['sam_area']} | "
              f"{r['tuned_vis']:.2f} / {r['tuned_amodal']:.2f} | {r['stock_vis']:.2f} | "
              f"{r.get('kind')} |")
        w("")
    # ---- 4. jumps
    w("### T4 面积跳变（状态不变、遮挡者不变）\n")
    w("| 台 | 零件 | 帧 | 说明 |")
    w("|---|---|---|---|")
    for f in sorted((f for f in items if f["code"] in ("jump", "amodal_jump")),
                    key=lambda f: (f["desktop"], f["inst"], f["code"])):
        w(f"| D{f['desktop']:02d} | {f['inst']} | {f['frames']} | {f['code']}: {short(f['detail'])} |")
    w("")
    # ---- 5. log vs picture
    w("### T5 日志 vs 画面\n")
    w("| 台 | 帧 | 标记 | 零件 | 说明 |")
    w("|---|---|---|---|---|")
    for f in sorted((f for f in items if f["code"] in ("frame_unchanged", "leave_no_change",
                                                         "zorder_pixels")
                     and f["severity"] in ("high", "medium")),
                    key=lambda f: (f["desktop"], f["steps"][0])):
        w(f"| D{f['desktop']:02d} | {f['frames']} | {f['code']} | {f['inst']} | {short(f['detail'], 180)} |")
    w("")
    w("### T5b 画面变了、标注没跟上（visible_changed，高）：按台、按步分组\n")
    w("| 台 | 帧 | 挪动的零件 | 被画在它上面的零件 |")
    w("|---|---|---|---|")
    grp = defaultdict(list)
    for f in items:
        if f["code"] == "visible_changed" and f["severity"] == "high":
            grp[(f["desktop"], f["frames"], ", ".join(f.get("movers") or []))].append(f["inst"])
    for (d, fr, mv), insts in sorted(grp.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        w(f"| D{d:02d} | {fr} | {mv} | {', '.join(sorted(insts))} |")
    w("")
    # ---- 6. per desktop
    w("### T6 每台\n")
    for d in U.DESKTOPS:
        s = R["desktops"][str(d)]
        sm = s["summary"]
        fl = [f for f in items if f["desktop"] == d]
        w(f"#### D{d:02d}（{sm['brand']} {sm['model']}）\n")
        kf = ", ".join(f"{k} {v}" for k, v in sorted(sm["kf_sources"].items(), key=str))
        w(f"- {sm['frames']} 帧（已确认 {sm['verified']}），{sm['instances']} 个零件，"
          f"{sm['part_frames']} 个零件帧；ROI {sm['roi']}；在用形状来源（零件帧）：{kf}；"
          f"单帧覆盖 {sm['fo']}")
        k = s["check"]
        w(f"- cli check：problems {k['problems']} {k['codes'] or ''}，open conflicts "
          f"{k['open_conflicts']}；负担 {s['burden']}")
        hi = [f for f in fl if f["severity"] in ("high",) and f["code"] != "change_unexplained"]
        med = [f for f in fl if f["severity"] == "medium" and f["code"] != "change_unexplained"]
        for f in sorted(hi, key=lambda f: -f["score"])[:12]:
            w(f"  - **{f['code']}** {f['inst']} @ {f['frames']}：{short(f['detail'], 200)}")
        if len(hi) > 12:
            w(f"  - …另有 {len(hi) - 12} 条 high（ranked.json / all_items.json）")
        if med:
            cnt = Counter(f["code"] for f in med)
            w(f"  - medium：{dict(cnt)}")
        w(f"- 图：{', '.join(sheets.get(str(d), []))}\n")
    # ---- 7. top
    w("### T7 排序后的可疑项（前 40）\n")
    w("| # | 台 | 标记 | 零件 | 帧 | 分 | 说明 |")
    w("|---|---|---|---|---|---|---|")
    for n, f in enumerate(R["top"][:40], 1):
        w(f"| {n} | D{f['desktop']:02d} | {f['code']} | {f['inst']} | {f['frames']} | {f['score']} | "
          f"{short(f['detail'], 200)} |")
    w("")
    # ---- 8. package ranking
    w("### T8 数据包候选（负担 = high 1 分、medium 0.3 分，不算 change_unexplained）\n")
    w("| 台 | 品牌 | 负担 | open conflicts | check problems | high 条目 |")
    w("|---|---|---|---|---|---|")
    for d in sorted(U.DESKTOPS, key=lambda d: R["desktops"][str(d)]["burden"]):
        s = R["desktops"][str(d)]
        hi = Counter(f["code"] for f in items if f["desktop"] == d and f["severity"] == "high"
                     and f["code"] != "change_unexplained")
        w(f"| D{d:02d} | {s['summary']['brand']} | {s['burden']} | {s['check']['open_conflicts']} | "
          f"{s['check']['problems']} | {dict(hi)} |")
    with open(os.path.join(U.QA_ROOT, "report_tables.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(len(out), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
