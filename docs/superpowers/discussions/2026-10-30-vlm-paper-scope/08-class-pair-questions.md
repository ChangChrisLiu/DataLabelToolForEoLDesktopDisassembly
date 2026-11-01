# 08 · 「谁压着谁」类级逻辑清单（请用户逐条回答）

2026-10-31 晚。用户：「你可以把你所有不确定的逻辑给我列出来，然后我和你说，这样的话基本所有的逻辑就理清楚了」。

**怎么读**：每条是一个类别对「上面那件 → 下面那件」。**挡着** = 不先把上面那件打开 / 拿走 / 挪开，下面那件物理上拆不下来；
**不挡** = 只是画面上重叠。你的回答会成为类级规则：凡这一对在某台确认帧上确实有像素重叠（≥ 1000 px）就写成那台的
`covered_by(下 ← 上)` 边（W3a 的白名单机制，`pixel-edges --allow-unruled`）；日志里顺序相反的台会被自动跳过并列出，不会写。
数据列（`.cache/tmp/day2/graph_live`，66 台扫描仪确认帧）：**台 / 对** = 多少台、多少实例对有这种重叠；**上先动** = 日志里上面那件
先被打开 / 拆下 / 挪开，之后下面那件才拆；**下未拆** = 下面那件在日志里从未拆下（无法用日志判断）。

**已经不用问的**（规则已经覆盖，Review 里也会折起来，W3f）：
- 装在主板上的零件挡主板（CPU、内存、散热器、扩展卡，W3d）；压在 CPU 上的散热器挡 CPU 和拉杆（W3a）。
- 盖板的作用链：`of = motherboard_screws` → 盖板挡主板螺丝 → 螺丝固定主板；`of = expansion_slot` → 盖板挡卡 → 卡挡主板；
  `of = cpu_cooler`（导风罩）→ 盖板挡散热器 → 散热器挡 CPU / 拉杆 → 主板；`of = ram` → 盖板挡内存 → 内存挡主板。
  所以 **cover → motherboard / cpu / cpu_socket_lever / ram_module / cpu_cooler 都已间接覆盖**，不再问（34 + 14 + 11 + 3 + 2 台）。
- 下面是螺丝 / 接头 / 线缆的对：由它们自己的规则决定（拧 / 拔），不问。
- 日志已证明不挡的对（下面那件先拆了、上面那件当时还没动过）：不问，例如 D04 `psu → cover`、`cpu_cooler.fan → cooler_latch`。

---

## A. 大件 → 主板（主板最后拆，这些大件在日志里都先拆了）

| # | 上 → 下 | 台 / 对 | 中位 px | 上先动 / 下未拆 | 我的猜测 | 你的回答 |
|---|---|---|---|---|---|---|
| A1 | drive_cage → motherboard | 33 / 33 | 74 021 | 32 / 1 | 挡着（刚性硬盘架压在主板上；Dell MT 的架子是翻开的——翻开也算让路） | |
| A2 | optical_drive → motherboard | 22 / 22 | 68 797 | 21 / 1 | 挡着 | |
| A3 | storage_drive.ssd → motherboard | 22 / 23 | 27 572 | 21 / 2 | 挡着？SSD 连支架一起压在主板边上 | |
| A4 | psu → motherboard | 11 / 11 | 17 586 | 11 / 0 | 挡着（电源悬在主板一角上方）。注意 D01 D02 D35 D36 是先拆主板再拆电源，那几台不会写 | |
| A5 | storage_drive.hdd → motherboard | 7 / 7 | 34 524 | 6 / 1 | 挡着 | |
| A6 | case_fan → motherboard | 5 / 5 | 42 656 | 5 / 0 | 挡着？机箱风扇装在机箱上，但压着主板边 | 挡。「这个机箱风扇是散装掉下来的，这个时候会影响主板和别的东西，他是最前面的要被移走的」→ W2 已有 |
| A7 | cooler_bracket → motherboard | 2 / 2 | 12 130 | 2 / 0 | 不挡（支架随主板走，是主板的一部分） | 挡。「A7 和 A8 都会挡住」→ W2 已有 |
| A8 | misc_part → motherboard（D36） | 1 / 1 | 1 555 | 1 / 0 | 看图定 | 挡。「A8 之前我们不是把这个 misc part 改名为机箱上的接口了吗」（界面上 `front_io_module` 本来就显示为「机箱接口面板」）→ W3 `misc_part → motherboard` |

## B. 硬盘架 / 光驱 / 硬盘 / 电源之间

| # | 上 → 下 | 台 / 对 | 中位 px | 上先动 / 下未拆 | 我的猜测 | 你的回答 |
|---|---|---|---|---|---|---|
| B1 | optical_drive → drive_cage | 19 / 19 | 103 484 | 19 / 0 | 挡着 | |
| B2 | storage_drive.ssd → drive_cage | 16 / 17 | 16 689 | 16 / 1 | 挡着？SSD 装在架子上，拆架子前要先拿下 SSD | |
| B3 | optical_drive → storage_drive.ssd | 12 / 13 | 18 757 | 13 / 0 | 挡着 | |
| B4 | optical_drive → storage_drive.hdd | 6 / 7 | 42 305 | 7 / 0 | 挡着 | |
| B5 | storage_drive.hdd → drive_cage | 5 / 5 | 39 676 | 4 / 1 | 挡着？硬盘在架子里 | |
| B6 | drive_cage → storage_drive.hdd | 2 / 2 | 63 606 | 2 / 0 | 挡着？架子翻开才拿得到硬盘 | 挡。「B6，会挡住，拿不下来」→ W2 已有 |
| B7 | storage_drive.ssd → optical_drive | 2 / 2 | 37 463 | 2 / 0 | 看图定 | 挡。「B7 也会挡住」→ W3 `storage_drive.ssd → optical_drive` |
| B8 | storage_drive.hdd → psu | 6 / 6 | 61 974 | 6 / 0 | 挡着？硬盘压在电源上 | **画反了，电源在上**。「PSU 的模块在这里是要翻一下打开，然后才能把里面的 hdd 或者 ssd 拿出来 … 移走 psu 不需要考虑 hdd 这样的，因为 psu 在 hdd 前面」→ W3 `psu → storage_drive.hdd`；D22 D23 D26 D29 D32 D33 的层序已改（`pair_override`） |
| B9 | storage_drive.ssd → psu | 5 / 5 | 20 411 | 4 / 1 | 挡着？ | 第一张挡（「需要移走 ssd 才能动 psu」→ W3 `storage_drive.ssd → psu`）；D24 那张**画反了**（「这个 psu 会叠下来然后压在 SSD 上面，所以第一帧这里是看不到 ssd 的，你的遮挡关系算错了」→ D24 D25 D30 层序已改） |
| B10 | psu → storage_drive.ssd（D31） | 1 / 1 | 32 480 | 1 / 0 | 看图定 | 挡。「B10 和 B8 是一个问题，psu 会挡住 ssd」→ W3 `psu → storage_drive.ssd` |

## C. 大件 → CPU / 拉杆 / 散热片（主要是两件散热器那 8 台：D03 D07 D08 D11 D14 D16 D18 D19）

| # | 上 → 下 | 台 / 对 | 中位 px | 上先动 / 下未拆 | 我的猜测 | 你的回答 |
|---|---|---|---|---|---|---|
| C1 | drive_cage → cpu_cooler.heatsink | 8 / 8 | 18 744 | 8 / 0 | 挡着（架子压在散热片上） | |
| C2 | drive_cage → cpu_socket_lever | 8 / 8 | 7 579 | 8 / 0 | 挡着 | |
| C3 | drive_cage → cpu | 8 / 8 | 4 406 | 8 / 0 | 挡着 | |
| C4 | optical_drive → cpu_cooler.heatsink | 8 / 8 | 19 415 | 8 / 0 | 挡着 | |
| C5 | optical_drive → cpu_socket_lever | 8 / 8 | 8 300 | 8 / 0 | 挡着 | |
| C6 | optical_drive → cpu | 8 / 8 | 5 514 | 8 / 0 | 挡着 | |
| C7 | storage_drive.ssd → cpu_socket_lever | 10 / 10 | 1 795 | 8 / 2 | 看图定（像素少，可能只是边缘） | |
| C8 | storage_drive.ssd → cpu_cooler.heatsink | 8 / 8 | 2 614 | 8 / 0 | 看图定 | |
| C9 | storage_drive.ssd → cpu | 3 / 3 | 6 473 | 3 / 0 | 看图定 | |
| C10 | storage_drive.hdd → heatsink / lever / cpu | 2 / 2 / 1 | 9 236 / 4 549 / 3 005 | 全部上先动 | 看图定 | |
| C11 | case_fan → cpu / lever / expansion_card.gpu / cpu_cooler.fan（D43、D54） | 1–2 | 6 458–22 685 | 上先动 | 看图定 | 挡。「C11 几个都是挡住的 … 这个 case fan，都是散落掉在机箱里面的，是最先需要被拿走的部分」→ W2 已有 + W3 `case_fan → storage_drive.ssd` |

## D. 大件 → 内存条（像素都很少：悬在内存插槽上方的架子 / 光驱位）

| # | 上 → 下 | 台 / 对 | 中位 px | 上先动 / 下未拆 | 我的猜测 | 你的回答 |
|---|---|---|---|---|---|---|
| D1 | drive_cage → ram_module | 19 / 27 | 2 719 | 27 / 0 | 存疑：Dell MT 架子悬在内存上方，内存往上拔可能拔不出；但也可能拔得出 | |
| D2 | optical_drive → ram_module | 16 / 27 | 1 795 | 27 / 0 | 同上 | |
| D3 | case_fan → ram_module | 3 / 6 | 1 597 | 6 / 0 | 看图定 | 挡。「D3 同理」→ W2 已有 |
| D4 | storage_drive.ssd / hdd → ram_module | 5 / 6，2 / 6 | 1 277 / 4 092 | 上先动 | 看图定 | |

## E. 零散

| # | 上 → 下 | 台 / 对 | 中位 px | 上先动 / 下未拆 | 我的猜测 | 你的回答 |
|---|---|---|---|---|---|---|
| E1 | cover → cover（D39 等） | 3 / 3 | 79 171 | 1 / 2 | 看图定（外盖压内盖？） | 第一张挡（→ W3 `cover → cover`）；D49 **画反了**（「这个导风罩 cover01 压下来，会压住其他所有的别的，他在这里是最上面的，你的遮挡关系错了，你怎么这个没按照我们拆的顺序来」→ D49 D62 的 cover.01 层移到其它盖板之上） |
| E2 | expansion_card.gpu → cover | 3 / 3 | 2 830 | 0 / 3 | 盖板从未拆下，无法用日志判断；看图定 | **画反了**：「需要先把蓝色的 cover 打开，才能去拆到 expansion card，你又搞错顺序了」→ W3 `cover → expansion_card`；D24 D29 D32 的 cover.02 已改到显卡之上 |
| E3 | storage_drive.ssd → cover | 2 / 4 | 31 491 | 2 / 2 | 看图定 | 挡。「这个 ssd 在这个导风罩上面，然后导风罩压住了剩下的东西，所以要先把 ssd 拿走，你还是要用我们的拆解的顺序作为主要参考」→ W3 `storage_drive.ssd → cover` |
| E4 | cover → psu（D02） | 2 / 2 | 2 259 | 2 / 0 | 不挡？ | 挡。「要先打开然后拆 psu，第二个 E4 也要先拿走 cover01，然后再拿走 psu」→ W3 `cover → psu` |
| E5 | storage_drive.ssd → storage_drive.ssd（D02）、optical_drive → optical_drive（D36）、drive_cage → storage_drive.hdd 等单台 | 1 | — | 上先动 | 看图定 | D02：「两个 ssd 实际上就是一个，标注有问题，把他们合并了」（已合并）；D36 挡：「要先拆掉橙色的，这个不也是我们拆解的顺序吗」→ W3 `optical_drive → optical_drive` |

## 另一类问题：盖板的 `of` 标错

D04 `cover.01`（日志名「Case locker above WLAN card」，现记为 `of = expansion_slot` → 挡 WLAN 卡）。用户说它的作用是盖住主板螺丝。
`of` 只能填一个：**A** 改成 `motherboard_screws`（挡 6 颗主板螺丝，不再挡 WLAN 卡）；**B** 保持 `expansion_slot`；**C** 两者都挡
（保持 `of`，另加 6 条手工边 `covered_by(screw.motherboard.* ← cover.01)`）。请选。

---

## 2026-10-31 深夜 · 看图裁定与层序修正

用户看了 32 张对卡片（`.cache/tmp/day2/pairs_v2/*.jpg`），上面各行的「你的回答」是原话；贯穿的原则：**「你的遮挡关系算错了 … 要用我们的拆解的顺序作为主要参考」**——日志先动了下面那件的覆盖不是「不挡」的证明，而是层序画反的信号。写到了哪里：

- **白名单 W3**（`configs/covers_whitelist.yaml`，`version: W3`）：W2 的 19 条 + 11 条 `covered_by`（A8、B7、B8、B9、B10、C11 的 `case_fan → storage_drive.ssd`、E1–E5），`not_blocks` 不变；线上 `pixel-edges --allow-unruled --apply` 写了 `pixels:W3` 241 行（`covered_by` 201、`blocked_by(contained)` 40），矛盾 24 → 22。
- **层序修正**（数据，不进白名单）：B8 / B9 九台 `psu.01` 在 hdd / ssd 之上、E2 三台 `cover.02` 在显卡之上的 `pair_override`（线上第一批 12 条），E1 D49 D62 导风罩 `cover.01` 层移到其它盖板之上，D02 的两个 SSD 合并成一个；之后 66 台 `refreeze`、14 条 `drive_latch.of → drive_cage.01`（B1 的后半句）。
- **复查**（W3h，规格 §4.4 / §7.3 v1.53）：W3f 的「日志已证明不挡（k）」改为「日志里下面那件先动了（k）」，每行「不挡」或「上下翻转」由用户看图定；类别对在 W3 `pairs:` 里（正着或反着）的加标「用户已定：挡 → 层序可能画反」。
- 本清单里「已经不用问的」第四条（日志已证明不挡的对不问）**不再成立**：那些对现在列在复查的「日志里下面那件先动了」里等用户看图。
- 另一类问题（D04 `cover.01` 的 `of`）仍待用户选。
