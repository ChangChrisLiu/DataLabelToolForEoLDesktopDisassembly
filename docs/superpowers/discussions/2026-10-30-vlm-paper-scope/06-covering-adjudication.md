# 「谁压着谁」裁定单（第 1 天，供用户逐行核对）

来源：2026-10-31 早新副本 `.cache/tmp/day1/tda.sqlite` 上的 `export-graph`（66 台，2 664 边，8 410 遮挡对，2 411 个无边候选）。
每行是一个**类对**：上面的零件在已确认扫描仪帧里盖住下面的零件，而图里没有任何边、也没有结构理由解释它。
列：台数 / 实例对数；像素 = 重叠像素的中位 / 最小 / 最大；**顺序** = 日志里 above 在 below 被第一次动手之前就已离开的比例（你说的「sequence 反映绝大多数信息」就是这一列）；间隔 = 两者相差的步数中位。
建议：**建议收** = 进白名单，按每台像素实例化成 `covered_by(below ← above)`；**存疑** = 需要你判断物理上是否真的挡；**规则边** / **支承表** = 由 §6 规则或 `mounted_on` 支承表处理，不进白名单；**不收** / **不在白名单** / **本版不做** = 这一版不出边。
你只需在最后一列写 ✔ / ✘ / 改法。像素覆盖 ≠ 物理阻挡的几种情形：翻起的铰链件（HP 硬盘架 / PSU 翻开后仍重叠）、悬空件（PSU 悬在主板上方）、随主板走的小卡、悬在内存上方的硬盘架。

## A. 类对总表

| # | above | below | 台数 / 对数 | 像素 中位 / 最小 / 最大 | 顺序 | 间隔 | 建议 | 理由 | 你的裁定 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | expansion_card.gpu | motherboard | 24 / 24 | 16935 / 7058 / 30300 | 100% | 9.0 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 2 | optical_drive | motherboard | 22 / 22 | 68797 / 15583 / 95174 | 95% | 37 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 3 | optical_drive | drive_cage | 19 / 19 | 103484 / 79093 / 106887 | 100% | 5 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 4 | cover | cpu_socket_lever | 13 / 13 | 6328 / 4158 / 13000 | 100% | 27 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 5 | optical_drive | storage_drive.ssd | 12 / 13 | 18757 / 10206 / 39406 | 100% | 4 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 6 | cover | cpu | 12 / 12 | 1865 / 748 / 6606 | 100% | 28.0 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 7 | drive_cage | cpu_socket_lever | 9 / 9 | 7448 / 6 / 8431 | 100% | 14 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 8 | optical_drive | cpu_cooler.heatsink | 8 / 8 | 19415 / 16755 / 28149 | 100% | 12.0 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 9 | drive_cage | cpu_cooler.heatsink | 8 / 8 | 18744 / 14304 / 20387 | 100% | 7.0 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 10 | optical_drive | cpu_socket_lever | 8 / 8 | 8300 / 7011 / 10539 | 100% | 17.0 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 11 | optical_drive | cpu | 8 / 8 | 5514 / 4685 / 6318 | 100% | 18.0 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 12 | drive_cage | cpu | 8 / 8 | 4406 / 3218 / 5121 | 100% | 13.5 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 13 | optical_drive | storage_drive.hdd | 6 / 7 | 42305 / 33085 / 92727 | 100% | 4 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 14 | drive_cage | storage_drive.hdd | 2 / 2 | 63606 / 53027 / 74186 | 100% | 2.0 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 15 | cpu_cooler.heatsink | cooler_bracket | 2 / 2 | 2494 / 90 / 4898 | 100% | 3.0 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 16 | cover | psu | 2 / 2 | 2259 / 1455 / 3064 | 100% | 9.0 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 17 | cover | cpu_cooler.heatsink | 2 / 2 | 1417 / 1152 / 1683 | 100% | 16.5 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 18 | optical_drive | optical_drive | 1 / 1 | 92310 / 92310 / 92310 | 100% | 1 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 19 | optical_drive | cover | 1 / 1 | 1869 / 1869 / 1869 | 100% | 1 | **建议收** | 刚性大件压着；日志每台都先拆 above |  |
| 20 | drive_cage | motherboard | 34 / 34 | 73461 / 788 / 100304 | 65% | 32 | **存疑** | 日志里 above 先离开的比例只有 65% |  |
| 21 | drive_cage | ram_latch | 26 / 103 | 327 / 8 / 780 | 81% | 9 | **存疑** | 上方悬着的大件下面内存可能拔得出（Dell MT 硬盘架 / 光驱位在内存插槽上方） |  |
| 22 | drive_cage | ram_module | 21 / 32 | 2255 / 20 / 6537 | 72% | 11 | **存疑** | 上方悬着的大件下面内存可能拔得出（Dell MT 硬盘架 / 光驱位在内存插槽上方） |  |
| 23 | optical_drive | ram_module | 19 / 32 | 1678 / 411 / 6328 | 100% | 15.0 | **存疑** | 上方悬着的大件下面内存可能拔得出（Dell MT 硬盘架 / 光驱位在内存插槽上方） |  |
| 24 | psu | cover | 17 / 17 | 68 / 13 / 3365 | 71% | 1.0 | **存疑** | 日志里 above 先离开的比例只有 71% |  |
| 25 | psu | motherboard | 12 / 12 | 17359 / 3 / 21962 | 100% | 21.5 | **存疑** | PSU 悬在主板一角上方，主板可能斜抽；HP SFF 10 台日志都先动 PSU |  |
| 26 | optical_drive | ram_latch | 10 / 32 | 318 / 41 / 780 | 100% | 13.0 | **存疑** | 上方悬着的大件下面内存可能拔得出（Dell MT 硬盘架 / 光驱位在内存插槽上方） |  |
| 27 | cover | ram_latch | 4 / 22 | 247 / 3 / 638 | 95% | 9.5 | **存疑** | 上方悬着的大件下面内存可能拔得出（Dell MT 硬盘架 / 光驱位在内存插槽上方） |  |
| 28 | cover | cover | 3 / 3 | 79171 / 1533 / 80649 | 33% | -1 | **存疑** | 日志里 above 先离开的比例只有 33% |  |
| 29 | expansion_card.gpu | cover | 3 / 3 | 2830 / 1879 / 4204 | 0% | -1 | **存疑** | 日志里 above 先离开的比例只有 0% |  |
| 30 | cover | ram_module | 3 / 5 | 2654 / 83 / 4073 | 100% | 19 | **存疑** | 上方悬着的大件下面内存可能拔得出（Dell MT 硬盘架 / 光驱位在内存插槽上方） |  |
| 31 | cpu_cooler.fan | ram_latch | 3 / 8 | 133 / 1 / 399 | 100% | 15.0 | **存疑** | 上方悬着的大件下面内存可能拔得出（Dell MT 硬盘架 / 光驱位在内存插槽上方） |  |
| 32 | cpu_cooler.fan | cooler_latch | 2 / 2 | 45712 / 45372 / 46053 | 0% | -13.5 | **存疑** | 日志里 above 先离开的比例只有 0% |  |
| 33 | expansion_card.gpu | card_latch | 2 / 2 | 1670 / 1348 / 1992 | 0% | -1.0 | **存疑** | 日志里 above 先离开的比例只有 0% |  |
| 34 | optical_drive | cpu_cooler.fan | 2 / 2 | 529 / 43 / 1016 | 100% | 7.5 | **存疑** | 像素中位数 < 1000 |  |
| 35 | cpu_cooler.fan | ram_module | 2 / 2 | 1 / 1 / 2 | 100% | 10.0 | **存疑** | 上方悬着的大件下面内存可能拔得出（Dell MT 硬盘架 / 光驱位在内存插槽上方） |  |
| 36 | psu | storage_drive.ssd | 1 / 1 | 32480 / 32480 / 32480 | 0% | -7 | **存疑** | 日志里 above 先离开的比例只有 0% |  |
| 37 | drive_cage | optical_drive | 1 / 1 | 882 / 882 / 882 | 100% | 2 | **存疑** | 像素中位数 < 1000 |  |
| 38 | psu | cpu_socket_lever | 1 / 1 | 461 / 461 / 461 | 0% | -10 | **存疑** | 日志里 above 先离开的比例只有 0% |  |
| 39 | cpu_cooler.fan | drive_cage | 1 / 1 | 62 / 62 / 62 | 100% | 9 | **存疑** | 像素中位数 < 1000 |  |
| 40 | expansion_card.gpu | expansion_card.other | 1 / 1 | 5 / 5 / 5 | 100% | 1 | **存疑** | 像素中位数 < 1000 |  |
| 41 | cpu_cooler.fan | cpu_socket_lever | 40 / 40 | 12094 / 10168 / 22604 | 100% | 20.0 | **规则边** | §6 扩展 FIXED_COVERED_BY 覆盖（压 CPU 的那件压拉杆；fan 压 heatsink） |  |
| 42 | cpu_cooler.heatsink | cpu_socket_lever | 20 / 20 | 10989 / 8720 / 16487 | 100% | 20.0 | **规则边** | §6 扩展 FIXED_COVERED_BY 覆盖（压 CPU 的那件压拉杆；fan 压 heatsink） |  |
| 43 | cpu_cooler.fan | cpu_cooler.heatsink | 8 / 8 | 64391 / 48234 / 66194 | 100% | 5.0 | **规则边** | §6 扩展 FIXED_COVERED_BY 覆盖（压 CPU 的那件压拉杆；fan 压 heatsink） |  |
| 44 | cpu_cooler.heatsink | cpu | 8 / 8 | 6580 / 6349 / 7018 | 100% | 6.5 | **规则边** | §6 扩展 FIXED_COVERED_BY 覆盖（压 CPU 的那件压拉杆；fan 压 heatsink） |  |
| 45 | cpu_cooler.heatsink_fan | cpu_socket_lever | 4 / 4 | 10359 / 7549 / 11712 | 100% | 23.0 | **规则边** | §6 扩展 FIXED_COVERED_BY 覆盖（压 CPU 的那件压拉杆；fan 压 heatsink） |  |
| 46 | cpu | motherboard | 65 / 65 | 6643 / 6168 / 12283 | 100% | 11.0 | **支承表** | 按 mounted_on 类级支承排除，不出边 |  |
| 47 | ram_module | motherboard | 60 / 126 | 2694 / 1478 / 6537 | 100% | 15 | **支承表** | 按 mounted_on 类级支承排除，不出边 |  |
| 48 | cpu_cooler.fan | motherboard | 48 / 48 | 48939 / 45708 / 79699 | 100% | 32.5 | **支承表** | 按 mounted_on 类级支承排除，不出边 |  |
| 49 | cpu_cooler.heatsink | motherboard | 20 / 20 | 57422 / 30626 / 108914 | 100% | 34.0 | **支承表** | 按 mounted_on 类级支承排除，不出边 |  |
| 50 | cpu_cooler.heatsink_fan | motherboard | 4 / 4 | 31442 / 30250 / 41511 | 100% | 34.0 | **支承表** | 按 mounted_on 类级支承排除，不出边 |  |
| 51 | cover | motherboard | 34 / 58 | 5847 / 39 / 117983 | 36% | 39.5 | **不收** | 像素连续分布无断点；HP 大导风罩可单独裁定 |  |
| 52 | expansion_card.other | motherboard | 3 / 9 | 9256 / 4434 / 16916 | 100% | 9.5 | **不收** | 小卡随主板走，覆盖≠阻挡 |  |
| 53 | expansion_card.wlan | motherboard | 2 / 2 | 4675 / 4523 / 4828 | 100% | 15.5 | **不收** | 小卡随主板走，覆盖≠阻挡 |  |
| 54 | expansion_card.other | expansion_card.other | 1 / 1 | 787 / 787 / 787 | 100% | 1 | **不收** | 小卡随主板走，覆盖≠阻挡 |  |
| 55 | ram_module | ram_latch | 39 / 142 | 34 / 1 / 144 | 22% | -3 | **不在白名单** | above 不是刚性大件 |  |
| 56 | storage_drive.ssd | motherboard | 22 / 23 | 27572 / 1431 / 46197 | 100% | 33 | **不在白名单** | above 不是刚性大件 |  |
| 57 | storage_drive.ssd | drive_cage | 16 / 17 | 16689 / 2640 / 44058 | 94% | 1 | **不在白名单** | above 不是刚性大件 |  |
| 58 | storage_drive.ssd | cpu_socket_lever | 11 / 11 | 1784 / 8 / 11506 | 100% | 15 | **不在白名单** | above 不是刚性大件 |  |
| 59 | storage_drive.ssd | ram_module | 9 / 14 | 549 / 37 / 1529 | 100% | 15 | **不在白名单** | above 不是刚性大件 |  |
| 60 | storage_drive.hdd | motherboard | 8 / 8 | 34261 / 78 / 81615 | 100% | 34 | **不在白名单** | above 不是刚性大件 |  |
| 61 | storage_drive.ssd | cpu_cooler.heatsink | 8 / 8 | 2614 / 1102 / 35250 | 100% | 8.0 | **不在白名单** | above 不是刚性大件 |  |
| 62 | storage_drive.hdd | psu | 6 / 6 | 61974 / 57015 / 69044 | 0% | -1.5 | **不在白名单** | above 不是刚性大件 |  |
| 63 | storage_drive.ssd | cpu | 6 / 6 | 3206 / 4 / 7101 | 100% | 18.0 | **不在白名单** | above 不是刚性大件 |  |
| 64 | case_fan | motherboard | 5 / 5 | 42656 / 12606 / 62155 | 100% | 43 | **不在白名单** | above 不是刚性大件 |  |
| 65 | storage_drive.hdd | drive_cage | 5 / 5 | 39676 / 33999 / 100963 | 100% | 1 | **不在白名单** | above 不是刚性大件 |  |
| 66 | storage_drive.ssd | psu | 4 / 4 | 12018 / 3276 / 37758 | 25% | -2.0 | **不在白名单** | above 不是刚性大件 |  |
| 67 | storage_drive.ssd | ram_latch | 4 / 5 | 126 / 9 / 321 | 100% | 9.0 | **不在白名单** | above 不是刚性大件 |  |
| 68 | storage_drive.ssd | optical_drive | 3 / 3 | 25316 / 56 / 49611 | 100% | 7 | **不在白名单** | above 不是刚性大件 |  |
| 69 | storage_drive.hdd | ram_module | 3 / 8 | 3869 / 641 / 5983 | 100% | 18.5 | **不在白名单** | above 不是刚性大件 |  |
| 70 | case_fan | ram_module | 3 / 6 | 1597 / 1277 / 1771 | 100% | 29 | **不在白名单** | above 不是刚性大件 |  |
| 71 | case_fan | ram_latch | 3 / 12 | 342 / 187 / 658 | 100% | 26 | **不在白名单** | above 不是刚性大件 |  |
| 72 | storage_drive.ssd | cover | 2 / 4 | 31491 / 6395 / 53825 | 100% | 1.5 | **不在白名单** | above 不是刚性大件 |  |
| 73 | case_fan | cpu_cooler.fan | 2 / 2 | 22685 / 4317 / 41054 | 100% | 6.0 | **不在白名单** | above 不是刚性大件 |  |
| 74 | cooler_bracket | motherboard | 2 / 2 | 12130 / 10025 / 14235 | 100% | 33.5 | **不在白名单** | above 不是刚性大件 |  |
| 75 | case_fan | cpu_socket_lever | 2 / 2 | 9365 / 146 / 18585 | 100% | 31.0 | **不在白名单** | above 不是刚性大件 |  |
| 76 | storage_drive.hdd | cpu_cooler.heatsink | 2 / 2 | 9236 / 3763 / 14710 | 100% | 10.5 | **不在白名单** | above 不是刚性大件 |  |
| 77 | storage_drive.hdd | cpu_socket_lever | 2 / 2 | 4549 / 2828 / 6270 | 100% | 22.0 | **不在白名单** | above 不是刚性大件 |  |
| 78 | storage_drive.hdd | cpu | 2 / 2 | 1509 / 14 / 3005 | 100% | 23.0 | **不在白名单** | above 不是刚性大件 |  |
| 79 | ram_module | drive_cage | 2 / 3 | 633 / 578 / 890 | 0% | -20 | **不在白名单** | above 不是刚性大件 |  |
| 80 | storage_drive.hdd | ram_latch | 2 / 12 | 333 / 1 / 780 | 100% | 5.5 | **不在白名单** | above 不是刚性大件 |  |
| 98 | drive_cage | screw.motherboard | 29 / 62 | 199 / 4 / 392 | 87% | 28.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 99 | drive_cage | connector.front_panel | 27 / 66 | 455 / 43 / 2599 | 50% | 14 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 100 | connector | drive_cage | 26 / 83 | 1012 / 14 / 3311 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 101 | connector | motherboard | 22 / 76 | 928 / 14 / 3311 | 1% | 26 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 102 | optical_drive | screw.motherboard | 21 / 21 | 246 / 111 / 349 | 100% | 33 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 103 | connector | connector | 21 / 28 | 87 / 1 / 828 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 104 | expansion_card.gpu | screw.motherboard | 20 / 31 | 200 / 9 / 416 | 100% | 5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 105 | drive_cage | connector.fan | 19 / 20 | 521 / 15 / 748 | 100% | 11.5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 106 | optical_drive | connector | 18 / 37 | 1443 / 175 / 3311 | 100% | 3 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 107 | optical_drive | connector.fan | 18 / 18 | 525 / 345 / 748 | 100% | 14.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 108 | optical_drive | connector.front_panel | 18 / 28 | 298 / 15 / 1825 | 100% | 18.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 109 | connector | cpu_socket_lever | 17 / 34 | 297 / 1 / 1450 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 110 | drive_latch | screw.motherboard | 16 / 23 | 504 / 165 / 672 | 100% | 31 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 111 | drive_cage | connector | 15 / 28 | 767 / 8 / 3610 | 100% | 13.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 112 | cover | connector.fan | 14 / 14 | 479 / 28 / 640 | 100% | 22.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 113 | connector.sata_data | connector.sata_power | 13 / 13 | 461 / 1 / 676 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 114 | cover | screw.cpu_cooler | 13 / 22 | 244 / 70 / 636 | 100% | 2.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 115 | ram_latch | ram_latch | 12 / 21 | 8 / 1 / 24 | 10% | -28 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 116 | card_latch | expansion_card.gpu | 11 / 11 | 1422 / 40 / 1884 | 91% | 1 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 117 | storage_drive.ssd | connector | 11 / 15 | 828 / 41 / 1849 | 100% | 14 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 118 | screw.motherboard | ram_latch | 11 / 18 | 32 / 2 / 511 | 39% | -23 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 119 | psu | screw.motherboard | 10 / 16 | 267 / 7 / 389 | 100% | 21.5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 120 | cover | connector.front_panel | 9 / 10 | 322 / 144 / 571 | 100% | 30.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 121 | cover | screw.motherboard | 9 / 10 | 298 / 212 / 459 | 100% | 34.5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 122 | connector | cpu_cooler.heatsink | 8 / 22 | 840 / 175 / 3109 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 123 | connector | storage_drive.ssd | 8 / 13 | 723 / 81 / 1695 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 124 | storage_drive.ssd | connector.fan | 8 / 10 | 498 / 127 / 748 | 100% | 14.5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 125 | connector.sata_power | connector.sata_data | 8 / 8 | 334 / 1 / 2860 | 12% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 126 | optical_drive | screw.cpu_cooler | 8 / 17 | 264 / 46 / 421 | 100% | 8 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 127 | drive_cage | screw.cpu_cooler | 8 / 16 | 260 / 141 / 421 | 100% | 6.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 128 | screw.motherboard | cpu_socket_lever | 8 / 8 | 250 / 78 / 363 | 0% | -15.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 129 | drive_latch | connector.front_panel | 7 / 7 | 558 / 430 / 698 | 100% | 16 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 130 | connector | ram_module | 7 / 16 | 251 / 66 / 1122 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 131 | storage_drive.ssd | screw.cpu_cooler | 7 / 9 | 146 / 7 / 543 | 100% | 7 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 132 | connector.sata_power | optical_drive | 7 / 8 | 67 / 8 / 1856 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 133 | connector.sata_power | storage_drive.ssd | 6 / 7 | 741 / 164 / 1407 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 134 | connector.front_panel | drive_cage | 6 / 20 | 446 / 67 / 685 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 135 | cpu_cooler.heatsink | connector.front_panel | 6 / 10 | 231 / 27 / 1018 | 100% | 13.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 136 | connector | connector.fan | 6 / 7 | 90 / 3 / 290 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 137 | connector | optical_drive | 6 / 9 | 61 / 14 / 1577 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 138 | cover | connector | 6 / 6 | 44 / 4 / 900 | 100% | 19.5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 139 | connector.front_panel | connector.front_panel | 6 / 6 | 24 / 13 / 45 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 140 | cooler_latch | motherboard | 5 / 5 | 60064 / 1919 / 101365 | 100% | 44 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 141 | connector.sata_power | motherboard | 5 / 5 | 1944 / 759 / 3068 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 142 | connector.sata_data | motherboard | 5 / 5 | 1000 / 561 / 2109 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 143 | ram_latch | drive_cage | 5 / 8 | 241 / 30 / 406 | 0% | -33.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 144 | cpu_cooler.fan | connector.front_panel | 5 / 8 | 156 / 29 / 984 | 100% | 19.5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 145 | storage_drive.hdd | connector.fan | 5 / 5 | 96 / 36 / 455 | 100% | 15 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 146 | storage_drive.hdd | connector | 4 / 4 | 1485 / 614 / 1771 | 100% | 15.5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 147 | cooler_latch | connector.front_panel | 4 / 4 | 395 / 357 / 516 | 100% | 34.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 148 | case_fan | connector.front_panel | 4 / 4 | 364 / 278 / 416 | 100% | 23.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 149 | connector.front_panel | cover | 4 / 6 | 275 / 3 / 2090 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 150 | case_fan | screw.motherboard | 4 / 5 | 266 / 119 / 332 | 100% | 35 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 151 | screw.cpu_cooler | cover | 4 / 10 | 216 / 48 / 255 | 0% | -14.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 152 | connector | ram_latch | 4 / 6 | 22 / 3 / 186 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 153 | cooler_latch | cpu_cooler.fan | 3 / 3 | 45841 / 1861 / 46398 | 100% | 6 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 154 | storage_drive.hdd | cable_clip | 3 / 5 | 3193 / 1004 / 7362 | 100% | 13.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 155 | connector.sata_power | cpu_socket_lever | 3 / 3 | 1711 / 636 / 2269 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 156 | case_fan | connector | 3 / 3 | 1544 / 1305 / 1561 | 100% | 14 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 157 | connector.sata_power | cpu | 3 / 3 | 915 / 749 / 1692 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 158 | case_fan | connector.sata_data | 3 / 3 | 694 / 507 / 709 | 100% | 15 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 159 | connector.fan | cover | 3 / 6 | 390 / 287 / 631 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 160 | screw.motherboard | drive_cage | 3 / 4 | 340 / 275 / 437 | 0% | -26.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 161 | connector.sata_data | optical_drive | 3 / 3 | 241 / 5 / 1802 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 162 | connector | storage_drive.hdd | 3 / 6 | 234 / 48 / 658 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 163 | storage_drive.ssd | screw.motherboard | 3 / 4 | 228 / 53 / 506 | 100% | 35.5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 164 | storage_drive.hdd | screw.motherboard | 3 / 4 | 192 / 111 / 245 | 100% | 31.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 165 | connector | screw.cpu_cooler | 3 / 4 | 117 / 79 / 152 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 166 | cpu_cooler.fan | connector | 3 / 3 | 6 / 2 / 6 | 100% | 11 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 167 | cooler_latch | cpu_socket_lever | 2 / 2 | 16515 / 11963 / 21068 | 100% | 29.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 168 | cooler_latch | cpu | 2 / 2 | 6388 / 6168 / 6608 | 100% | 30.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 169 | drive_cage | cable_clip | 2 / 4 | 4424 / 1534 / 6829 | 25% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 170 | connector.sata_power | cover | 2 / 2 | 2865 / 2662 / 3068 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 171 | connector.sata_power | cpu_cooler.heatsink | 2 / 2 | 2865 / 2662 / 3068 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 172 | connector | cover | 2 / 2 | 1716 / 1696 / 1736 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 173 | connector.sata_data | cpu_cooler.heatsink | 2 / 2 | 1076 / 1000 / 1152 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 174 | connector.sata_data | cover | 2 / 3 | 1000 / 8 / 1152 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 175 | connector | psu | 2 / 4 | 853 / 18 / 1620 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 176 | storage_drive.hdd | connector.front_panel | 2 / 2 | 721 / 697 / 746 | 100% | 12.5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 177 | connector.sata_data | cpu_socket_lever | 2 / 2 | 596 / 41 / 1152 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 178 | connector | connector.sata_power | 2 / 3 | 595 / 586 / 948 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 179 | connector | connector.front_panel | 2 / 3 | 495 / 380 / 755 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 180 | storage_drive.ssd | connector.front_panel | 2 / 3 | 467 / 329 / 606 | 100% | 22 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 181 | card_latch | motherboard | 2 / 2 | 438 / 105 / 772 | 50% | 8 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 182 | expansion_card.other | connector.front_panel | 2 / 2 | 397 / 151 / 644 | 100% | 8.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 183 | connector.sata_data | storage_drive.ssd | 2 / 2 | 393 / 69 / 717 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 184 | case_fan | cable_clip | 2 / 2 | 352 / 28 / 677 | 100% | 21.5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 185 | expansion_card.wlan | screw.motherboard | 2 / 2 | 334 / 209 / 459 | 100% | 12.5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 186 | expansion_card.other | screw.motherboard | 2 / 3 | 333 / 269 / 336 | 100% | 6 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 187 | storage_drive.hdd | screw.cpu_cooler | 2 / 3 | 323 / 252 / 332 | 100% | 9 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 188 | connector.sata_power | connector | 2 / 2 | 310 / 55 / 565 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 189 | case_fan | screw.cpu_cooler | 2 / 4 | 212 / 166 / 277 | 100% | 3.5 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 190 | cpu_cooler.heatsink | screw.cooler_bracket | 2 / 3 | 200 / 29 / 293 | 100% | 2 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 191 | connector | cpu | 2 / 2 | 107 / 11 / 204 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 192 | ram_latch | screw.motherboard | 2 / 2 | 85 / 83 / 88 | 0% | -4.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 193 | cable_clip | motherboard | 2 / 2 | 48 / 19 / 78 | 0% | — | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |
| 194 | drive_latch | connector | 2 / 2 | 42 / 2 / 83 | 100% | 17.0 | **本版不做** | 小件 / 卡扣，像素在噪声量级或层级已知反序 |  |

（台数 1 且不进白名单的类对省略；共 242 个类对。）

## B. 建议收 / 存疑 类对的逐台清单（实例级，供扫一眼）

| 台 | above | below | 像素 | 帧 | above 离开步 | below 首次动手步 |
|---|---|---|---|---|---|---|
| D39 | cover.01 | cover.02 | 1533 | 1–40 (40) | 41 | 42 |
| D49 | cover.02 | cover.01 | 80649 | 1–4 (4) | 6 | 5 |
| D62 | cover.02 | cover.01 | 79171 | 1–4 (4) | 6 | 5 |
| D09 | cover.01 | cpu.01 | 6336 | 1–1 (1) | 2 | 24 |
| D13 | cover.01 | cpu.01 | 6606 | 1–1 (1) | 2 | 21 |
| D20 | cover.01 | cpu.01 | 6359 | 1–1 (1) | 2 | 24 |
| D22 | cover.01 | cpu.01 | 748 | 1–2 (2) | 3 | 34 |
| D23 | cover.01 | cpu.01 | 1918 | 1–2 (2) | 3 | 32 |
| D24 | cover.01 | cpu.01 | 2172 | 1–2 (2) | 3 | 31 |
| D25 | cover.01 | cpu.01 | 1957 | 1–2 (2) | 3 | 31 |
| D26 | cover.01 | cpu.01 | 819 | 1–2 (2) | 3 | 28 |
| D29 | cover.01 | cpu.01 | 1625 | 1–2 (2) | 3 | 34 |
| D31 | cover.01 | cpu.01 | 1813 | 1–2 (2) | 3 | 31 |
| D32 | cover.01 | cpu.01 | 1319 | 1–2 (2) | 3 | 34 |
| D33 | cover.01 | cpu.01 | 1516 | 1–2 (2) | 3 | 32 |
| D49 | cover.02 | cpu_cooler.heatsink.01 | 1683 | 1–5 (5) | 6 | 30 |
| D62 | cover.02 | cpu_cooler.heatsink.01 | 1152 | 1–5 (5) | 6 | 15 |
| D09 | cover.01 | cpu_socket_lever.01 | 11430 | 1–1 (1) | 2 | 23 |
| D13 | cover.01 | cpu_socket_lever.01 | 13000 | 1–1 (1) | 2 | 20 |
| D20 | cover.01 | cpu_socket_lever.01 | 12323 | 1–1 (1) | 2 | 23 |
| D22 | cover.01 | cpu_socket_lever.01 | 5785 | 1–2 (2) | 3 | 33 |
| D23 | cover.01 | cpu_socket_lever.01 | 6881 | 1–2 (2) | 3 | 31 |
| D24 | cover.01 | cpu_socket_lever.01 | 6213 | 1–2 (2) | 3 | 30 |
| D25 | cover.01 | cpu_socket_lever.01 | 6328 | 1–2 (2) | 3 | 30 |
| D26 | cover.01 | cpu_socket_lever.01 | 6087 | 1–2 (2) | 3 | 27 |
| D29 | cover.01 | cpu_socket_lever.01 | 6218 | 1–2 (2) | 3 | 33 |
| D30 | cover.01 | cpu_socket_lever.01 | 4158 | 1–2 (2) | 3 | 25 |
| D31 | cover.01 | cpu_socket_lever.01 | 6425 | 1–2 (2) | 3 | 30 |
| D32 | cover.01 | cpu_socket_lever.01 | 6074 | 1–2 (2) | 3 | 33 |
| D33 | cover.01 | cpu_socket_lever.01 | 6347 | 1–2 (2) | 3 | 31 |
| D02 | cover.01 | psu.01 | 1455 | 1–37 (37) | 46 | 48 |
| D48 | cover.01 | psu.01 | 3064 | 5–5 (1) | 8 | 24 |
| D08 | cover.01 | ram_latch.02 | 156 | 23–37 (15) | 未离开 | 23 |
| D48 | cover.01 | ram_latch.05 | 252 | 1–5 (5) | 8 | — |
| D49 | cover.02 | ram_latch.01 | 61 | 1–5 (5) | 6 | 12 |
| D49 | cover.02 | ram_latch.02 | 464 | 1–5 (5) | 6 | 13 |
| D49 | cover.02 | ram_latch.03 | 213 | 1–5 (5) | 6 | 14 |
| D49 | cover.02 | ram_latch.04 | 124 | 1–5 (5) | 6 | 15 |
| D49 | cover.02 | ram_latch.05 | 124 | 1–5 (5) | 6 | — |
| D49 | cover.02 | ram_latch.06 | 420 | 1–5 (5) | 6 | — |
| D49 | cover.02 | ram_latch.07 | 427 | 1–5 (5) | 6 | — |
| D49 | cover.02 | ram_latch.08 | 464 | 1–5 (5) | 6 | — |
| D49 | cover.02 | ram_latch.09 | 638 | 1–5 (5) | 6 | — |
| D49 | cover.02 | ram_latch.10 | 543 | 1–5 (5) | 6 | — |
| D49 | cover.02 | ram_latch.11 | 243 | 1–5 (5) | 6 | — |
| D49 | cover.02 | ram_latch.12 | 226 | 1–5 (5) | 6 | — |
| D62 | cover.02 | ram_latch.01 | 37 | 1–5 (5) | 6 | 16 |
| D62 | cover.02 | ram_latch.02 | 159 | 1–5 (5) | 6 | 17 |
| D62 | cover.02 | ram_latch.03 | 419 | 1–5 (5) | 6 | 18 |
| D62 | cover.02 | ram_latch.04 | 491 | 1–5 (5) | 6 | 19 |
| D62 | cover.02 | ram_latch.05 | 77 | 1–5 (5) | 6 | — |
| D62 | cover.02 | ram_latch.06 | 3 | 1–5 (5) | 6 | — |
| D62 | cover.02 | ram_latch.07 | 408 | 1–5 (5) | 6 | — |
| D62 | cover.02 | ram_latch.08 | 453 | 1–5 (5) | 6 | — |
| D48 | cover.01 | ram_module.01 | 83 | 6–6 (1) | 8 | 27 |
| D49 | cover.02 | ram_module.03 | 2654 | 1–5 (5) | 6 | — |
| D49 | cover.02 | ram_module.04 | 4073 | 1–5 (5) | 6 | — |
| D62 | cover.02 | ram_module.03 | 2122 | 1–5 (5) | 6 | — |
| D62 | cover.02 | ram_module.04 | 3028 | 1–5 (5) | 6 | — |
| D04 | cpu_cooler.fan.01 | cooler_latch.01 | 45372 | 1–1 (1) | 15 | 2 |
| D05 | cpu_cooler.fan.01 | cooler_latch.01 | 46053 | 1–1 (1) | 16 | 2 |
| D02 | cpu_cooler.fan.01 | drive_cage.01 | 62 | 1–7 (7) | 8 | 17 |
| D02 | cpu_cooler.fan.01 | ram_latch.03 | 1 | 1–7 (7) | 8 | 20 |
| D02 | cpu_cooler.fan.01 | ram_latch.07 | 18 | 1–7 (7) | 8 | — |
| D46 | cpu_cooler.fan.01 | ram_latch.05 | 107 | 1–6 (4) | 7 | — |
| D46 | cpu_cooler.fan.01 | ram_latch.06 | 160 | 3–3 (1) | 7 | — |
| D46 | cpu_cooler.fan.01 | ram_latch.07 | 314 | 3–4 (2) | 7 | — |
| D46 | cpu_cooler.fan.01 | ram_latch.08 | 399 | 1–6 (6) | 7 | — |
| D48 | cpu_cooler.fan.01 | ram_latch.02 | 44 | 6–7 (2) | 8 | 26 |
| D48 | cpu_cooler.fan.01 | ram_latch.08 | 316 | 6–7 (2) | 8 | — |
| D02 | cpu_cooler.fan.01 | ram_module.02 | 1 | 1–7 (7) | 8 | 23 |
| D09 | cpu_cooler.fan.01 | ram_module.01 | 2 | 5–5 (1) | 16 | 21 |
| D35 | cpu_cooler.heatsink.01 | cooler_bracket.01 | 90 | 1–3 (3) | 4 | 7 |
| D36 | cpu_cooler.heatsink.01 | cooler_bracket.01 | 4898 | 1–4 (4) | 5 | 8 |
| D03 | drive_cage.01 | cpu.01 | 3599 | 1–10 (10) | 11 | 38 |
| D07 | drive_cage.01 | cpu.01 | 3218 | 1–13 (13) | 14 | 35 |
| D08 | drive_cage.01 | cpu.01 | 3702 | 1–13 (13) | 14 | 26 |
| D11 | drive_cage.01 | cpu.01 | 4498 | 1–13 (13) | 14 | 26 |
| D14 | drive_cage.01 | cpu.01 | 4314 | 1–13 (13) | 14 | 26 |
| D16 | drive_cage.01 | cpu.01 | 4876 | 11–11 (1) | 12 | 24 |
| D18 | drive_cage.01 | cpu.01 | 5121 | 1–12 (12) | 13 | 28 |
| D19 | drive_cage.01 | cpu.01 | 4681 | 1–13 (13) | 14 | 35 |
| D03 | drive_cage.01 | cpu_cooler.heatsink.01 | 14304 | 1–10 (10) | 11 | 27 |
| D07 | drive_cage.01 | cpu_cooler.heatsink.01 | 15290 | 1–13 (13) | 14 | 21 |
| D08 | drive_cage.01 | cpu_cooler.heatsink.01 | 16737 | 1–13 (13) | 14 | 21 |
| D11 | drive_cage.01 | cpu_cooler.heatsink.01 | 19758 | 1–13 (13) | 14 | 21 |
| D14 | drive_cage.01 | cpu_cooler.heatsink.01 | 18771 | 1–13 (13) | 14 | 21 |
| D16 | drive_cage.01 | cpu_cooler.heatsink.01 | 20387 | 3–11 (9) | 12 | 19 |
| D18 | drive_cage.01 | cpu_cooler.heatsink.01 | 19500 | 1–12 (12) | 13 | 20 |
| D19 | drive_cage.01 | cpu_cooler.heatsink.01 | 18718 | 1–13 (13) | 14 | 21 |
| D02 | drive_cage.01 | cpu_socket_lever.01 | 6 | 8–16 (9) | 17 | 36 |
| D03 | drive_cage.01 | cpu_socket_lever.01 | 5931 | 1–10 (10) | 11 | 37 |
| D07 | drive_cage.01 | cpu_socket_lever.01 | 6721 | 1–13 (13) | 14 | 34 |
| D08 | drive_cage.01 | cpu_socket_lever.01 | 7448 | 1–13 (13) | 14 | 25 |
| D11 | drive_cage.01 | cpu_socket_lever.01 | 7859 | 1–13 (13) | 14 | 25 |
| D14 | drive_cage.01 | cpu_socket_lever.01 | 7710 | 1–13 (13) | 14 | 25 |
| D16 | drive_cage.01 | cpu_socket_lever.01 | 8431 | 4–11 (8) | 12 | 23 |
| D18 | drive_cage.01 | cpu_socket_lever.01 | 7742 | 1–12 (12) | 13 | 27 |
| D19 | drive_cage.01 | cpu_socket_lever.01 | 6981 | 1–13 (13) | 14 | 34 |
| D02 | drive_cage.01 | motherboard.01 | 59187 | 1–16 (16) | 17 | 45 |
| D03 | drive_cage.01 | motherboard.01 | 71625 | 1–10 (10) | 11 | 48 |
| D04 | drive_cage.01 | motherboard.01 | 75457 | 1–12 (12) | 13 | 46 |
| D05 | drive_cage.01 | motherboard.01 | 80917 | 1–13 (13) | 14 | 46 |
| D06 | drive_cage.01 | motherboard.01 | 83139 | 1–13 (13) | 14 | 49 |
| D07 | drive_cage.01 | motherboard.01 | 92613 | 1–13 (13) | 14 | 56 |
| D08 | drive_cage.01 | motherboard.01 | 93886 | 1–13 (13) | 14 | 46 |
| D09 | drive_cage.01 | motherboard.01 | 82553 | 1–13 (13) | 14 | 46 |
| D10 | drive_cage.01 | motherboard.01 | 72901 | 1–12 (12) | 13 | 42 |
| D11 | drive_cage.01 | motherboard.01 | 94813 | 1–13 (13) | 14 | 47 |
| D12 | drive_cage.01 | motherboard.01 | 84984 | 1–13 (13) | 14 | 46 |
| D13 | drive_cage.01 | motherboard.01 | 78197 | 1–10 (10) | 11 | 42 |
| D14 | drive_cage.01 | motherboard.01 | 94064 | 1–13 (13) | 14 | 48 |
| D15 | drive_cage.01 | motherboard.01 | 74021 | 1–12 (12) | 13 | 42 |
| D16 | drive_cage.01 | motherboard.01 | 100304 | 1–11 (11) | 12 | 45 |
| D17 | drive_cage.01 | motherboard.01 | 83827 | 1–13 (13) | 14 | 46 |
| D18 | drive_cage.01 | motherboard.01 | 97391 | 1–12 (12) | 13 | 48 |
| D19 | drive_cage.01 | motherboard.01 | 96529 | 1–13 (13) | 14 | 55 |
| D20 | drive_cage.01 | motherboard.01 | 76835 | 1–13 (13) | 14 | 46 |
| D21 | drive_cage.01 | motherboard.01 | 82482 | 1–13 (13) | 14 | 43 |
| D22 | drive_cage.01 | motherboard.01 | 23841 | 1–43 (43) | 未离开 | 44 |
| D23 | drive_cage.01 | motherboard.01 | 25946 | 1–8 (8) | 未离开 | 42 |
| D24 | drive_cage.01 | motherboard.01 | 26541 | 1–41 (41) | 未离开 | 42 |
| D25 | drive_cage.01 | motherboard.01 | 25152 | 1–8 (8) | 未离开 | 41 |
| D26 | drive_cage.01 | motherboard.01 | 18477 | 1–8 (8) | 未离开 | 38 |
| D27 | drive_cage.01 | motherboard.01 | 16627 | 1–11 (11) | 未离开 | 43 |
| D28 | drive_cage.01 | motherboard.01 | 788 | 1–10 (10) | 未离开 | 45 |
| D29 | drive_cage.01 | motherboard.01 | 29169 | 1–44 (44) | 未离开 | 45 |
| D30 | drive_cage.01 | motherboard.01 | 24285 | 1–8 (8) | 未离开 | 36 |
| D31 | drive_cage.01 | motherboard.01 | 26518 | 1–8 (8) | 未离开 | 41 |
| D32 | drive_cage.01 | motherboard.01 | 25903 | 1–8 (8) | 未离开 | 45 |
| D33 | drive_cage.01 | motherboard.01 | 19324 | 1–41 (41) | 未离开 | 42 |
| D48 | drive_cage.01 | motherboard.01 | 45735 | 1–8 (8) | 9 | — |
| D61 | drive_cage.01 | motherboard.01 | 16654 | 1–7 (7) | 8 | 40 |
| D48 | drive_cage.01 | optical_drive.01 | 882 | 1–8 (8) | 9 | 11 |
| D02 | drive_cage.01 | ram_latch.02 | 62 | 1–16 (16) | 17 | 19 |
| D02 | drive_cage.01 | ram_latch.03 | 371 | 1–16 (16) | 17 | 20 |
| D02 | drive_cage.01 | ram_latch.06 | 82 | 1–16 (16) | 17 | — |
| D02 | drive_cage.01 | ram_latch.07 | 573 | 1–16 (16) | 17 | — |
| D03 | drive_cage.01 | ram_latch.01 | 335 | 1–10 (10) | 11 | 17 |
| D03 | drive_cage.01 | ram_latch.02 | 310 | 1–10 (10) | 11 | 18 |
| D03 | drive_cage.01 | ram_latch.03 | 315 | 9–10 (2) | 11 | 19 |
| D03 | drive_cage.01 | ram_latch.04 | 351 | 9–10 (2) | 11 | 20 |
| D03 | drive_cage.01 | ram_latch.05 | 327 | 1–10 (10) | 11 | — |
| D03 | drive_cage.01 | ram_latch.06 | 267 | 1–10 (10) | 11 | — |
| D03 | drive_cage.01 | ram_latch.07 | 780 | 9–10 (2) | 11 | — |
| D03 | drive_cage.01 | ram_latch.08 | 369 | 9–10 (2) | 11 | — |
| D05 | drive_cage.01 | ram_latch.02 | 308 | 1–13 (13) | 14 | 18 |
| D05 | drive_cage.01 | ram_latch.03 | 33 | 1–13 (13) | 14 | 19 |
| D05 | drive_cage.01 | ram_latch.06 | 456 | 1–13 (13) | 14 | — |
| D05 | drive_cage.01 | ram_latch.07 | 119 | 1–13 (13) | 14 | — |
| D06 | drive_cage.01 | ram_latch.03 | 269 | 1–13 (13) | 14 | 19 |
| D06 | drive_cage.01 | ram_latch.04 | 218 | 1–13 (13) | 14 | 20 |
| D06 | drive_cage.01 | ram_latch.07 | 468 | 1–13 (13) | 14 | — |
| D06 | drive_cage.01 | ram_latch.08 | 343 | 1–13 (13) | 14 | — |
| D07 | drive_cage.01 | ram_latch.01 | 248 | 9–13 (5) | 14 | 22 |
| D07 | drive_cage.01 | ram_latch.02 | 282 | 9–13 (5) | 14 | 23 |
| D07 | drive_cage.01 | ram_latch.03 | 199 | 9–13 (5) | 14 | 24 |
| D07 | drive_cage.01 | ram_latch.04 | 269 | 9–13 (5) | 14 | 25 |
| D07 | drive_cage.01 | ram_latch.05 | 260 | 1–13 (13) | 14 | 26 |
| D07 | drive_cage.01 | ram_latch.06 | 335 | 1–13 (13) | 14 | 27 |
| D07 | drive_cage.01 | ram_latch.07 | 351 | 1–13 (13) | 14 | 28 |
| D07 | drive_cage.01 | ram_latch.08 | 241 | 1–13 (13) | 14 | 29 |
| D08 | drive_cage.01 | ram_latch.01 | 279 | 10–13 (4) | 14 | 22 |
| D08 | drive_cage.01 | ram_latch.02 | 216 | 1–13 (13) | 14 | 23 |
| D08 | drive_cage.01 | ram_latch.03 | 373 | 10–13 (4) | 14 | — |
| D08 | drive_cage.01 | ram_latch.04 | 357 | 10–13 (4) | 14 | — |
| D08 | drive_cage.01 | ram_latch.05 | 337 | 1–13 (13) | 14 | — |
| D08 | drive_cage.01 | ram_latch.06 | 290 | 1–13 (13) | 14 | — |
| D08 | drive_cage.01 | ram_latch.07 | 350 | 1–13 (13) | 14 | — |
| D08 | drive_cage.01 | ram_latch.08 | 349 | 1–13 (13) | 14 | — |
| D09 | drive_cage.01 | ram_latch.05 | 487 | 1–13 (13) | 14 | — |
| D09 | drive_cage.01 | ram_latch.06 | 529 | 1–13 (13) | 14 | — |
| D09 | drive_cage.01 | ram_latch.07 | 257 | 1–13 (13) | 14 | — |
| D09 | drive_cage.01 | ram_latch.08 | 146 | 1–13 (13) | 14 | — |
| D11 | drive_cage.01 | ram_latch.01 | 327 | 9–13 (5) | 14 | 22 |
| D11 | drive_cage.01 | ram_latch.02 | 256 | 1–13 (13) | 14 | 23 |
| D11 | drive_cage.01 | ram_latch.03 | 372 | 1–13 (13) | 14 | — |
| D11 | drive_cage.01 | ram_latch.04 | 383 | 9–13 (5) | 14 | — |
| D11 | drive_cage.01 | ram_latch.05 | 337 | 1–13 (13) | 14 | — |
| D11 | drive_cage.01 | ram_latch.06 | 305 | 9–13 (5) | 14 | — |
| D11 | drive_cage.01 | ram_latch.07 | 418 | 1–13 (13) | 14 | — |
| D11 | drive_cage.01 | ram_latch.08 | 380 | 9–13 (5) | 14 | — |
| D12 | drive_cage.01 | ram_latch.02 | 321 | 1–13 (13) | 14 | 18 |
| D12 | drive_cage.01 | ram_latch.06 | 537 | 1–13 (13) | 14 | — |
| D13 | drive_cage.01 | ram_latch.02 | 301 | 1–10 (10) | 11 | 15 |
| D13 | drive_cage.01 | ram_latch.06 | 243 | 1–10 (10) | 11 | — |
| D14 | drive_cage.01 | ram_latch.01 | 415 | 9–13 (5) | 14 | 22 |
| D14 | drive_cage.01 | ram_latch.02 | 402 | 1–13 (13) | 14 | 23 |
| D14 | drive_cage.01 | ram_latch.03 | 284 | 1–13 (13) | 14 | — |
| D14 | drive_cage.01 | ram_latch.04 | 463 | 9–13 (5) | 14 | — |
| D16 | drive_cage.01 | ram_latch.01 | 449 | 11–11 (1) | 12 | 20 |
| D16 | drive_cage.01 | ram_latch.02 | 430 | 1–11 (11) | 12 | 21 |
| D16 | drive_cage.01 | ram_latch.03 | 471 | 11–11 (1) | 12 | — |
| D16 | drive_cage.01 | ram_latch.04 | 394 | 1–11 (11) | 12 | — |
| D17 | drive_cage.01 | ram_latch.02 | 297 | 1–13 (13) | 14 | 18 |
| D17 | drive_cage.01 | ram_latch.03 | 81 | 1–13 (13) | 14 | 19 |
| D17 | drive_cage.01 | ram_latch.07 | 397 | 1–13 (13) | 14 | — |
| D17 | drive_cage.01 | ram_latch.08 | 71 | 1–13 (13) | 14 | — |
| D18 | drive_cage.01 | ram_latch.01 | 342 | 9–12 (4) | 13 | 21 |
| D18 | drive_cage.01 | ram_latch.02 | 321 | 9–12 (4) | 13 | 22 |
| D18 | drive_cage.01 | ram_latch.03 | 346 | 1–12 (12) | 13 | 23 |
| D18 | drive_cage.01 | ram_latch.04 | 302 | 1–12 (12) | 13 | 24 |
| D18 | drive_cage.01 | ram_latch.05 | 394 | 1–12 (12) | 13 | — |
| D18 | drive_cage.01 | ram_latch.06 | 327 | 1–12 (12) | 13 | — |
| D18 | drive_cage.01 | ram_latch.07 | 306 | 9–12 (4) | 13 | — |
| D18 | drive_cage.01 | ram_latch.08 | 322 | 9–12 (4) | 13 | — |
| D19 | drive_cage.01 | ram_latch.01 | 242 | 9–13 (5) | 14 | 22 |
| D19 | drive_cage.01 | ram_latch.02 | 298 | 9–13 (5) | 14 | 23 |
| D19 | drive_cage.01 | ram_latch.03 | 271 | 9–13 (5) | 14 | 24 |
| D19 | drive_cage.01 | ram_latch.04 | 306 | 9–13 (5) | 14 | 25 |
| D19 | drive_cage.01 | ram_latch.05 | 301 | 1–13 (13) | 14 | 26 |
| D19 | drive_cage.01 | ram_latch.06 | 327 | 1–13 (13) | 14 | 27 |
| D19 | drive_cage.01 | ram_latch.07 | 337 | 1–13 (13) | 14 | 28 |
| D19 | drive_cage.01 | ram_latch.08 | 282 | 1–13 (13) | 14 | 29 |
| D21 | drive_cage.01 | ram_latch.01 | 315 | 1–13 (13) | 14 | 17 |
| D21 | drive_cage.01 | ram_latch.05 | 339 | 1–13 (13) | 14 | — |
| D22 | drive_cage.01 | ram_latch.03 | 317 | 1–10 (10) | 未离开 | 29 |
| D22 | drive_cage.01 | ram_latch.06 | 416 | 1–10 (10) | 未离开 | — |
| D23 | drive_cage.01 | ram_latch.03 | 374 | 1–8 (8) | 未离开 | 27 |
| D23 | drive_cage.01 | ram_latch.04 | 407 | 1–8 (8) | 未离开 | 28 |
| D23 | drive_cage.01 | ram_latch.07 | 8 | 1–8 (8) | 未离开 | — |
| D23 | drive_cage.01 | ram_latch.08 | 467 | 1–8 (8) | 未离开 | — |
| D24 | drive_cage.01 | ram_latch.02 | 287 | 1–8 (3) | 未离开 | 28 |
| D24 | drive_cage.01 | ram_latch.06 | 407 | 1–8 (3) | 未离开 | — |
| D24 | drive_cage.01 | ram_latch.07 | 83 | 1–8 (3) | 未离开 | — |
| D25 | drive_cage.01 | ram_latch.04 | 356 | 1–8 (8) | 未离开 | — |
| D26 | drive_cage.01 | ram_latch.05 | 565 | 1–8 (3) | 未离开 | — |
| D26 | drive_cage.01 | ram_latch.06 | 484 | 1–8 (3) | 未离开 | — |
| D26 | drive_cage.01 | ram_latch.07 | 251 | 1–8 (3) | 未离开 | — |
| D29 | drive_cage.01 | ram_latch.04 | 128 | 1–8 (3) | 未离开 | 30 |
| D29 | drive_cage.01 | ram_latch.06 | 401 | 1–8 (3) | 未离开 | — |
| D30 | drive_cage.01 | ram_latch.03 | 426 | 1–8 (3) | 未离开 | — |
| D31 | drive_cage.01 | ram_latch.04 | 553 | 1–8 (3) | 未离开 | — |
| D32 | drive_cage.01 | ram_latch.03 | 409 | 1–8 (3) | 未离开 | 29 |
| D32 | drive_cage.01 | ram_latch.04 | 47 | 1–8 (3) | 未离开 | 30 |
| D32 | drive_cage.01 | ram_latch.07 | 430 | 1–8 (3) | 未离开 | — |
| D48 | drive_cage.01 | ram_latch.06 | 319 | 1–8 (8) | 9 | — |
| D02 | drive_cage.01 | ram_module.02 | 2374 | 1–16 (16) | 17 | 23 |
| D03 | drive_cage.01 | ram_module.01 | 3458 | 1–10 (10) | 11 | 21 |
| D03 | drive_cage.01 | ram_module.02 | 4106 | 1–10 (10) | 11 | 22 |
| D05 | drive_cage.01 | ram_module.02 | 3221 | 1–13 (13) | 14 | 22 |
| D06 | drive_cage.01 | ram_module.03 | 3184 | 1–13 (13) | 14 | 25 |
| D07 | drive_cage.01 | ram_module.01 | 5463 | 1–13 (13) | 14 | 30 |
| D07 | drive_cage.01 | ram_module.02 | 5691 | 1–13 (13) | 14 | 31 |
| D07 | drive_cage.01 | ram_module.03 | 6528 | 1–13 (13) | 14 | 32 |
| D07 | drive_cage.01 | ram_module.04 | 6537 | 1–13 (13) | 14 | 33 |
| D08 | drive_cage.01 | ram_module.01 | 5648 | 1–13 (13) | 14 | 24 |
| D11 | drive_cage.01 | ram_module.01 | 2019 | 1–13 (13) | 14 | 24 |
| D12 | drive_cage.01 | ram_module.02 | 2202 | 1–13 (13) | 14 | 22 |
| D13 | drive_cage.01 | ram_module.02 | 1810 | 1–10 (10) | 11 | 19 |
| D14 | drive_cage.01 | ram_module.01 | 3511 | 1–13 (13) | 14 | 24 |
| D16 | drive_cage.01 | ram_module.01 | 4862 | 1–11 (11) | 12 | 22 |
| D17 | drive_cage.01 | ram_module.02 | 1840 | 1–13 (13) | 14 | 22 |
| D18 | drive_cage.01 | ram_module.01 | 3598 | 1–12 (12) | 13 | 25 |
| D18 | drive_cage.01 | ram_module.02 | 4166 | 1–12 (12) | 13 | 26 |
| D19 | drive_cage.01 | ram_module.01 | 1686 | 1–13 (13) | 14 | 30 |
| D19 | drive_cage.01 | ram_module.02 | 1863 | 1–13 (13) | 14 | 31 |
| D19 | drive_cage.01 | ram_module.03 | 1673 | 1–13 (13) | 14 | 32 |
| D19 | drive_cage.01 | ram_module.04 | 2308 | 1–13 (13) | 14 | 33 |
| D21 | drive_cage.01 | ram_module.01 | 2719 | 1–13 (13) | 14 | 19 |
| D22 | drive_cage.01 | ram_module.02 | 1079 | 1–10 (10) | 未离开 | 32 |
| D23 | drive_cage.01 | ram_module.01 | 1145 | 1–8 (8) | 未离开 | 29 |
| D23 | drive_cage.01 | ram_module.02 | 805 | 1–8 (8) | 未离开 | 30 |
| D24 | drive_cage.01 | ram_module.01 | 934 | 1–8 (3) | 未离开 | 29 |
| D25 | drive_cage.01 | ram_module.01 | 20 | 1–8 (8) | 未离开 | 29 |
| D29 | drive_cage.01 | ram_module.01 | 800 | 1–8 (3) | 未离开 | 31 |
| D29 | drive_cage.01 | ram_module.02 | 1107 | 1–8 (3) | 未离开 | 32 |
| D32 | drive_cage.01 | ram_module.01 | 656 | 1–8 (3) | 未离开 | 31 |
| D32 | drive_cage.01 | ram_module.02 | 1113 | 1–8 (3) | 未离开 | 32 |
| D03 | drive_cage.01 | storage_drive.hdd.01 | 74186 | 1–10 (10) | 11 | 14 |
| D48 | drive_cage.01 | storage_drive.hdd.01 | 53027 | 1–8 (8) | 9 | 10 |
| D63 | expansion_card.gpu.01 | card_latch.01 | 1348 | 17–30 (14) | 31 | 30 |
| D65 | expansion_card.gpu.01 | card_latch.01 | 1992 | 1–33 (33) | 34 | 33 |
| D24 | expansion_card.gpu.01 | cover.02 | 1879 | 1–32 (32) | 33 | 32 |
| D29 | expansion_card.gpu.01 | cover.02 | 4204 | 1–35 (35) | 36 | 35 |
| D32 | expansion_card.gpu.01 | cover.02 | 2830 | 1–35 (35) | 36 | 35 |
| D39 | expansion_card.gpu.01 | expansion_card.other.01 | 5 | 1–42 (42) | 43 | 44 |
| D03 | expansion_card.gpu.01 | motherboard.01 | 7058 | 1–35 (35) | 36 | 48 |
| D24 | expansion_card.gpu.01 | motherboard.01 | 7943 | 1–32 (32) | 33 | 42 |
| D29 | expansion_card.gpu.01 | motherboard.01 | 22672 | 1–35 (35) | 36 | 45 |
| D32 | expansion_card.gpu.01 | motherboard.01 | 16638 | 1–35 (35) | 36 | 45 |
| D35 | expansion_card.gpu.01 | motherboard.01 | 16557 | 1–32 (32) | 33 | 41 |
| D39 | expansion_card.gpu.01 | motherboard.01 | 15726 | 1–42 (42) | 43 | 54 |
| D42 | expansion_card.gpu.01 | motherboard.01 | 15230 | 1–32 (32) | 33 | 42 |
| D43 | expansion_card.gpu.01 | motherboard.01 | 17278 | 1–38 (38) | 39 | 47 |
| D44 | expansion_card.gpu.01 | motherboard.01 | 13444 | 1–29 (29) | 30 | 39 |
| D45 | expansion_card.gpu.01 | motherboard.01 | 16783 | 1–28 (28) | 29 | 36 |
| D46 | expansion_card.gpu.01 | motherboard.01 | 17221 | 1–24 (24) | 25 | 36 |
| D47 | expansion_card.gpu.01 | motherboard.01 | 13730 | 1–26 (26) | 27 | 42 |
| D49 | expansion_card.gpu.01 | motherboard.01 | 17087 | 1–7 (7) | 8 | — |
| D50 | expansion_card.gpu.01 | motherboard.01 | 12023 | 1–33 (33) | 34 | 43 |
| D53 | expansion_card.gpu.01 | motherboard.01 | 24024 | 1–36 (36) | 37 | 48 |
| D54 | expansion_card.gpu.01 | motherboard.01 | 18415 | 1–36 (36) | 37 | 46 |
| D55 | expansion_card.gpu.01 | motherboard.01 | 22679 | 1–36 (36) | 37 | 48 |
| D56 | expansion_card.gpu.01 | motherboard.01 | 28170 | 1–34 (34) | 35 | 46 |
| D57 | expansion_card.gpu.01 | motherboard.01 | 22040 | 1–35 (35) | 36 | 47 |
| D58 | expansion_card.gpu.01 | motherboard.01 | 15768 | 1–36 (36) | 37 | 46 |
| D60 | expansion_card.gpu.01 | motherboard.01 | 20040 | 1–36 (36) | 37 | 46 |
| D61 | expansion_card.gpu.01 | motherboard.01 | 7647 | 1–30 (29) | 31 | 40 |
| D63 | expansion_card.gpu.01 | motherboard.01 | 30300 | 17–30 (14) | 31 | — |
| D65 | expansion_card.gpu.01 | motherboard.01 | 20190 | 1–33 (33) | 34 | 43 |
| D17 | optical_drive.01 | cover.01 | 1869 | 1–3 (3) | 4 | 5 |
| D03 | optical_drive.01 | cpu.01 | 5683 | 1–8 (8) | 9 | 38 |
| D07 | optical_drive.01 | cpu.01 | 5300 | 1–8 (8) | 9 | 35 |
| D08 | optical_drive.01 | cpu.01 | 5346 | 1–9 (9) | 10 | 26 |
| D11 | optical_drive.01 | cpu.01 | 4924 | 1–8 (8) | 9 | 26 |
| D14 | optical_drive.01 | cpu.01 | 4685 | 1–8 (8) | 9 | 26 |
| D16 | optical_drive.01 | cpu.01 | 6318 | 2–10 (9) | 11 | 24 |
| D18 | optical_drive.01 | cpu.01 | 5934 | 1–8 (8) | 9 | 28 |
| D19 | optical_drive.01 | cpu.01 | 6271 | 1–8 (8) | 9 | 35 |
| D16 | optical_drive.01 | cpu_cooler.fan.01 | 43 | 4–10 (7) | 11 | 14 |
| D17 | optical_drive.01 | cpu_cooler.fan.01 | 1016 | 1–3 (3) | 4 | 16 |
| D03 | optical_drive.01 | cpu_cooler.heatsink.01 | 17827 | 1–8 (8) | 9 | 27 |
| D07 | optical_drive.01 | cpu_cooler.heatsink.01 | 16755 | 1–8 (8) | 9 | 21 |
| D08 | optical_drive.01 | cpu_cooler.heatsink.01 | 17541 | 1–9 (9) | 10 | 21 |
| D11 | optical_drive.01 | cpu_cooler.heatsink.01 | 21223 | 1–8 (8) | 9 | 21 |
| D14 | optical_drive.01 | cpu_cooler.heatsink.01 | 19001 | 1–8 (8) | 9 | 21 |
| D16 | optical_drive.01 | cpu_cooler.heatsink.01 | 28149 | 1–10 (10) | 11 | 19 |
| D18 | optical_drive.01 | cpu_cooler.heatsink.01 | 19829 | 1–8 (8) | 9 | 20 |
| D19 | optical_drive.01 | cpu_cooler.heatsink.01 | 22411 | 1–8 (8) | 9 | 21 |
| D03 | optical_drive.01 | cpu_socket_lever.01 | 7462 | 1–8 (8) | 9 | 37 |
| D07 | optical_drive.01 | cpu_socket_lever.01 | 7011 | 1–8 (8) | 9 | 34 |
| D08 | optical_drive.01 | cpu_socket_lever.01 | 7864 | 1–9 (9) | 10 | 25 |
| D11 | optical_drive.01 | cpu_socket_lever.01 | 8875 | 1–8 (8) | 9 | 25 |
| D14 | optical_drive.01 | cpu_socket_lever.01 | 8245 | 1–8 (8) | 9 | 25 |
| D16 | optical_drive.01 | cpu_socket_lever.01 | 10539 | 1–10 (10) | 11 | 23 |
| D18 | optical_drive.01 | cpu_socket_lever.01 | 8355 | 1–8 (8) | 9 | 27 |
| D19 | optical_drive.01 | cpu_socket_lever.01 | 8726 | 1–8 (8) | 9 | 34 |
| D02 | optical_drive.01 | drive_cage.01 | 79093 | 1–10 (10) | 11 | 17 |
| D03 | optical_drive.01 | drive_cage.01 | 87724 | 1–8 (8) | 9 | 11 |
| D04 | optical_drive.01 | drive_cage.01 | 104693 | 1–8 (8) | 9 | 13 |
| D05 | optical_drive.01 | drive_cage.01 | 103056 | 1–8 (8) | 9 | 14 |
| D06 | optical_drive.01 | drive_cage.01 | 103514 | 1–8 (8) | 9 | 14 |
| D07 | optical_drive.01 | drive_cage.01 | 101766 | 1–8 (8) | 9 | 14 |
| D08 | optical_drive.01 | drive_cage.01 | 104876 | 1–9 (9) | 10 | 14 |
| D09 | optical_drive.01 | drive_cage.01 | 103936 | 1–8 (8) | 9 | 14 |
| D10 | optical_drive.01 | drive_cage.01 | 101592 | 1–8 (8) | 9 | 13 |
| D11 | optical_drive.01 | drive_cage.01 | 106887 | 1–8 (8) | 9 | 14 |
| D12 | optical_drive.01 | drive_cage.01 | 102783 | 1–8 (8) | 9 | 14 |
| D14 | optical_drive.01 | drive_cage.01 | 103484 | 1–8 (8) | 9 | 14 |
| D15 | optical_drive.01 | drive_cage.01 | 106585 | 1–8 (8) | 9 | 13 |
| D16 | optical_drive.01 | drive_cage.01 | 104145 | 1–10 (10) | 11 | 12 |
| D17 | optical_drive.01 | drive_cage.01 | 101812 | 1–3 (3) | 4 | 14 |
| D18 | optical_drive.01 | drive_cage.01 | 104337 | 1–8 (8) | 9 | 13 |
| D19 | optical_drive.01 | drive_cage.01 | 102919 | 1–8 (8) | 9 | 14 |
| D20 | optical_drive.01 | drive_cage.01 | 101311 | 1–8 (8) | 9 | 14 |
| D21 | optical_drive.01 | drive_cage.01 | 106287 | 1–11 (11) | 12 | 14 |
| D02 | optical_drive.01 | motherboard.01 | 63605 | 1–10 (10) | 11 | 45 |
| D03 | optical_drive.01 | motherboard.01 | 75883 | 1–8 (8) | 9 | 48 |
| D04 | optical_drive.01 | motherboard.01 | 60887 | 1–8 (8) | 9 | 46 |
| D05 | optical_drive.01 | motherboard.01 | 68425 | 1–8 (8) | 9 | 46 |
| D06 | optical_drive.01 | motherboard.01 | 69020 | 1–8 (8) | 9 | 49 |
| D07 | optical_drive.01 | motherboard.01 | 77466 | 1–8 (8) | 9 | 56 |
| D08 | optical_drive.01 | motherboard.01 | 77921 | 1–9 (9) | 10 | 46 |
| D09 | optical_drive.01 | motherboard.01 | 68469 | 1–8 (8) | 9 | 46 |
| D10 | optical_drive.01 | motherboard.01 | 59160 | 1–8 (8) | 9 | 42 |
| D11 | optical_drive.01 | motherboard.01 | 82856 | 1–8 (8) | 9 | 47 |
| D12 | optical_drive.01 | motherboard.01 | 71585 | 1–8 (8) | 9 | 46 |
| D14 | optical_drive.01 | motherboard.01 | 80042 | 1–8 (8) | 9 | 48 |
| D15 | optical_drive.01 | motherboard.01 | 60012 | 1–8 (8) | 9 | 42 |
| D16 | optical_drive.01 | motherboard.01 | 95174 | 1–10 (10) | 11 | 45 |
| D17 | optical_drive.01 | motherboard.01 | 82043 | 1–3 (3) | 4 | 46 |
| D18 | optical_drive.01 | motherboard.01 | 81003 | 1–8 (8) | 9 | 48 |
| D19 | optical_drive.01 | motherboard.01 | 84860 | 1–8 (8) | 9 | 55 |
| D20 | optical_drive.01 | motherboard.01 | 63779 | 1–8 (8) | 9 | 46 |
| D21 | optical_drive.01 | motherboard.01 | 68575 | 1–11 (11) | 12 | 43 |
| D35 | optical_drive.01 | motherboard.01 | 17980 | 1–22 (22) | 23 | 41 |
| D36 | optical_drive.01 | motherboard.01 | 15583 | 1–10 (10) | 11 | 41 |
| D48 | optical_drive.01 | motherboard.01 | 28478 | 1–10 (10) | 未离开 | — |
| D36 | optical_drive.01 | optical_drive.02 | 92310 | 1–10 (10) | 11 | 12 |
| D03 | optical_drive.01 | ram_latch.03 | 315 | 1–8 (8) | 9 | 19 |
| D03 | optical_drive.01 | ram_latch.04 | 351 | 1–8 (8) | 9 | 20 |
| D03 | optical_drive.01 | ram_latch.07 | 780 | 1–8 (8) | 9 | — |
| D03 | optical_drive.01 | ram_latch.08 | 369 | 1–8 (8) | 9 | — |
| D07 | optical_drive.01 | ram_latch.01 | 248 | 1–8 (8) | 9 | 22 |
| D07 | optical_drive.01 | ram_latch.02 | 282 | 1–8 (8) | 9 | 23 |
| D07 | optical_drive.01 | ram_latch.03 | 199 | 1–8 (8) | 9 | 24 |
| D07 | optical_drive.01 | ram_latch.04 | 269 | 1–8 (8) | 9 | 25 |
| D08 | optical_drive.01 | ram_latch.01 | 279 | 1–9 (9) | 10 | 22 |
| D08 | optical_drive.01 | ram_latch.03 | 373 | 1–9 (9) | 10 | — |
| D08 | optical_drive.01 | ram_latch.04 | 357 | 1–9 (9) | 10 | — |
| D08 | optical_drive.01 | ram_latch.05 | 324 | 1–9 (9) | 10 | — |
| D11 | optical_drive.01 | ram_latch.01 | 327 | 1–8 (8) | 9 | 22 |
| D11 | optical_drive.01 | ram_latch.04 | 383 | 1–8 (8) | 9 | — |
| D11 | optical_drive.01 | ram_latch.06 | 305 | 1–8 (8) | 9 | — |
| D11 | optical_drive.01 | ram_latch.08 | 380 | 1–8 (8) | 9 | — |
| D14 | optical_drive.01 | ram_latch.01 | 415 | 1–8 (8) | 9 | 22 |
| D14 | optical_drive.01 | ram_latch.04 | 463 | 1–8 (8) | 9 | — |
| D16 | optical_drive.01 | ram_latch.01 | 449 | 1–10 (10) | 11 | 20 |
| D16 | optical_drive.01 | ram_latch.03 | 471 | 1–10 (10) | 11 | — |
| D17 | optical_drive.01 | ram_latch.02 | 195 | 1–3 (3) | 4 | 18 |
| D17 | optical_drive.01 | ram_latch.07 | 41 | 1–3 (3) | 4 | — |
| D18 | optical_drive.01 | ram_latch.01 | 342 | 1–8 (8) | 9 | 21 |
| D18 | optical_drive.01 | ram_latch.02 | 321 | 1–8 (8) | 9 | 22 |
| D18 | optical_drive.01 | ram_latch.07 | 306 | 1–8 (8) | 9 | — |
| D18 | optical_drive.01 | ram_latch.08 | 322 | 1–8 (8) | 9 | — |
| D19 | optical_drive.01 | ram_latch.01 | 242 | 1–8 (8) | 9 | 22 |
| D19 | optical_drive.01 | ram_latch.02 | 298 | 1–8 (8) | 9 | 23 |
| D19 | optical_drive.01 | ram_latch.03 | 271 | 1–8 (8) | 9 | 24 |
| D19 | optical_drive.01 | ram_latch.04 | 306 | 1–8 (8) | 9 | 25 |
| D36 | optical_drive.01 | ram_latch.03 | 114 | 1–10 (10) | 11 | 28 |
| D36 | optical_drive.01 | ram_latch.07 | 262 | 1–10 (10) | 11 | — |
| D02 | optical_drive.01 | ram_module.02 | 1265 | 1–10 (10) | 11 | 23 |
| D03 | optical_drive.01 | ram_module.01 | 4776 | 1–8 (8) | 9 | 21 |
| D03 | optical_drive.01 | ram_module.02 | 5771 | 1–8 (8) | 9 | 22 |
| D04 | optical_drive.01 | ram_module.02 | 739 | 1–8 (8) | 9 | 21 |
| D05 | optical_drive.01 | ram_module.01 | 1161 | 1–8 (8) | 9 | 21 |
| D05 | optical_drive.01 | ram_module.02 | 1591 | 1–8 (8) | 9 | 22 |
| D06 | optical_drive.01 | ram_module.02 | 1241 | 1–8 (8) | 9 | 24 |
| D06 | optical_drive.01 | ram_module.03 | 1528 | 1–8 (8) | 9 | 25 |
| D07 | optical_drive.01 | ram_module.01 | 5380 | 1–8 (8) | 9 | 30 |
| D07 | optical_drive.01 | ram_module.02 | 5550 | 1–8 (8) | 9 | 31 |
| D07 | optical_drive.01 | ram_module.03 | 6307 | 1–8 (8) | 9 | 32 |
| D07 | optical_drive.01 | ram_module.04 | 6328 | 1–8 (8) | 9 | 33 |
| D08 | optical_drive.01 | ram_module.01 | 5484 | 1–9 (9) | 10 | 24 |
| D09 | optical_drive.01 | ram_module.02 | 1046 | 1–8 (8) | 9 | 22 |
| D10 | optical_drive.01 | ram_module.01 | 951 | 1–8 (8) | 9 | 18 |
| D11 | optical_drive.01 | ram_module.01 | 1890 | 1–8 (8) | 9 | 24 |
| D12 | optical_drive.01 | ram_module.01 | 878 | 1–8 (8) | 9 | 21 |
| D12 | optical_drive.01 | ram_module.02 | 1728 | 1–8 (8) | 9 | 22 |
| D14 | optical_drive.01 | ram_module.01 | 3321 | 1–8 (8) | 9 | 24 |
| D16 | optical_drive.01 | ram_module.01 | 4702 | 1–10 (10) | 11 | 22 |
| D17 | optical_drive.01 | ram_module.01 | 1131 | 1–3 (3) | 4 | 21 |
| D17 | optical_drive.01 | ram_module.02 | 1776 | 1–3 (3) | 4 | 22 |
| D18 | optical_drive.01 | ram_module.01 | 3433 | 1–8 (8) | 9 | 25 |
| D18 | optical_drive.01 | ram_module.02 | 4019 | 1–8 (8) | 9 | 26 |
| D19 | optical_drive.01 | ram_module.01 | 1628 | 1–8 (8) | 9 | 30 |
| D19 | optical_drive.01 | ram_module.02 | 1795 | 1–8 (8) | 9 | 31 |
| D19 | optical_drive.01 | ram_module.03 | 1611 | 1–8 (8) | 9 | 32 |
| D19 | optical_drive.01 | ram_module.04 | 2200 | 1–8 (8) | 9 | 33 |
| D20 | optical_drive.01 | ram_module.01 | 411 | 1–8 (8) | 9 | 21 |
| D20 | optical_drive.01 | ram_module.02 | 548 | 1–8 (8) | 9 | 22 |
| D21 | optical_drive.01 | ram_module.01 | 1319 | 1–11 (11) | 12 | 19 |
| D36 | optical_drive.01 | ram_module.02 | 1391 | 1–10 (10) | 11 | 31 |
| D03 | optical_drive.01 | storage_drive.hdd.01 | 83523 | 1–8 (8) | 9 | 14 |
| D04 | optical_drive.01 | storage_drive.hdd.01 | 42305 | 1–8 (8) | 9 | 12 |
| D05 | optical_drive.01 | storage_drive.hdd.01 | 37046 | 1–8 (8) | 9 | 13 |
| D06 | optical_drive.01 | storage_drive.hdd.01 | 33789 | 1–8 (8) | 9 | 13 |
| D07 | optical_drive.01 | storage_drive.hdd.01 | 33085 | 1–8 (8) | 9 | 13 |
| D36 | optical_drive.01 | storage_drive.hdd.01 | 92727 | 1–10 (10) | 11 | 15 |
| D36 | optical_drive.02 | storage_drive.hdd.01 | 74755 | 11–11 (1) | 12 | 15 |
| D02 | optical_drive.01 | storage_drive.ssd.01 | 34281 | 1–10 (10) | 11 | 13 |
| D02 | optical_drive.01 | storage_drive.ssd.02 | 35913 | 1–10 (10) | 11 | 16 |
| D08 | optical_drive.01 | storage_drive.ssd.01 | 10206 | 1–9 (9) | 10 | 13 |
| D09 | optical_drive.01 | storage_drive.ssd.01 | 36778 | 1–8 (8) | 9 | 13 |
| D10 | optical_drive.01 | storage_drive.ssd.01 | 14264 | 1–8 (8) | 9 | 12 |
| D11 | optical_drive.01 | storage_drive.ssd.01 | 15801 | 1–8 (8) | 9 | 13 |
| D12 | optical_drive.01 | storage_drive.ssd.01 | 30918 | 1–8 (8) | 9 | 13 |
| D14 | optical_drive.01 | storage_drive.ssd.01 | 18757 | 1–8 (8) | 9 | 13 |
| D15 | optical_drive.01 | storage_drive.ssd.01 | 39406 | 1–8 (8) | 9 | 12 |
| D17 | optical_drive.01 | storage_drive.ssd.01 | 36031 | 1–3 (3) | 4 | 13 |
| D18 | optical_drive.01 | storage_drive.ssd.01 | 16030 | 1–8 (8) | 9 | 12 |
| D19 | optical_drive.01 | storage_drive.ssd.01 | 16625 | 1–8 (8) | 9 | 13 |
| D20 | optical_drive.01 | storage_drive.ssd.01 | 16689 | 1–8 (8) | 9 | 13 |
| D04 | psu.01 | cover.01 | 1535 | 1–21 (21) | 39 | 22 |
| D05 | psu.01 | cover.01 | 73 | 1–36 (36) | 37 | 39 |
| D06 | psu.01 | cover.01 | 60 | 1–40 (40) | 41 | 42 |
| D07 | psu.01 | cover.01 | 38 | 1–46 (46) | 47 | 48 |
| D08 | psu.01 | cover.01 | 74 | 1–36 (36) | 37 | 38 |
| D09 | psu.01 | cover.02 | 21 | 1–37 (37) | 38 | 39 |
| D10 | psu.01 | cover.02 | 13 | 1–33 (33) | 34 | 35 |
| D11 | psu.01 | cover.01 | 42 | 1–36 (36) | 37 | 38 |
| D12 | psu.01 | cover.02 | 234 | 1–36 (36) | 39 | 37 |
| D13 | psu.01 | cover.02 | 68 | 1–33 (33) | 34 | 35 |
| D14 | psu.01 | cover.01 | 13 | 1–36 (36) | 37 | 40 |
| D15 | psu.01 | cover.02 | 32 | 1–33 (33) | 34 | 35 |
| D18 | psu.01 | cover.01 | 31 | 1–38 (38) | 39 | 40 |
| D19 | psu.01 | cover.01 | 430 | 1–45 (45) | 46 | 47 |
| D20 | psu.01 | cover.02 | 901 | 1–34 (34) | 39 | 35 |
| D21 | psu.01 | cover.02 | 394 | 1–29 (29) | 36 | 30 |
| D62 | psu.01 | cover.04 | 3365 | 1–6 (6) | 未离开 | 7 |
| D47 | psu.01 | cpu_socket_lever.01 | 461 | 31–32 (2) | 41 | 31 |
| D08 | psu.01 | motherboard.01 | 1171 | 1–36 (36) | 37 | 46 |
| D09 | psu.01 | motherboard.01 | 3 | 1–37 (37) | 38 | 46 |
| D22 | psu.01 | motherboard.01 | 16591 | 1–11 (11) | 20 | 44 |
| D23 | psu.01 | motherboard.01 | 17132 | 1–9 (9) | 18 | 42 |
| D24 | psu.01 | motherboard.01 | 19606 | 1–9 (9) | 20 | 42 |
| D25 | psu.01 | motherboard.01 | 17992 | 1–9 (9) | 20 | 41 |
| D26 | psu.01 | motherboard.01 | 16517 | 1–9 (9) | 20 | 38 |
| D29 | psu.01 | motherboard.01 | 21962 | 1–9 (9) | 20 | 45 |
| D30 | psu.01 | motherboard.01 | 18454 | 1–9 (9) | 15 | 36 |
| D31 | psu.01 | motherboard.01 | 17586 | 1–9 (9) | 20 | 41 |
| D32 | psu.01 | motherboard.01 | 16963 | 1–9 (9) | 20 | 45 |
| D33 | psu.01 | motherboard.01 | 21020 | 1–9 (9) | 18 | 42 |
| D31 | psu.01 | storage_drive.ssd.01 | 32480 | 1–9 (9) | 20 | 13 |

共 485 行。

## C. 两件散热器的机器（§6 规则 + 白名单都要看）

- D03（Optiplex 7010）：cpu_cooler.fan.01, cpu_cooler.heatsink.01；今天的 CPU 计划：['open cpu_socket_lever.01', 'remove cpu.01']
- D07（Optiplex 7020）：cpu_cooler.fan.01, cpu_cooler.heatsink.01；今天的 CPU 计划：['open cpu_socket_lever.01', 'remove cpu.01']
- D08（Optiplex 7010）：cpu_cooler.fan.01, cpu_cooler.heatsink.01；今天的 CPU 计划：['open cpu_socket_lever.01', 'remove cpu.01']
- D11（Optiplex 7010）：cpu_cooler.fan.01, cpu_cooler.heatsink.01；今天的 CPU 计划：['open cpu_socket_lever.01', 'remove cpu.01']
- D14（Optiplex 3010）：cpu_cooler.fan.01, cpu_cooler.heatsink.01；今天的 CPU 计划：['open cpu_socket_lever.01', 'remove cpu.01']
- D16（Optiplex 3010）：cpu_cooler.fan.01, cpu_cooler.heatsink.01；今天的 CPU 计划：['open cpu_socket_lever.01', 'remove cpu.01']
- D18（Optiplex 7010）：cpu_cooler.fan.01, cpu_cooler.heatsink.01；今天的 CPU 计划：['open cpu_socket_lever.01', 'remove cpu.01']
- D19（Optiplex 990）：cpu_cooler.fan.01, cpu_cooler.heatsink.01；今天的 CPU 计划：['open cpu_socket_lever.01', 'remove cpu.01']
- D64（G4）：cpu_cooler.fan.01, cpu_cooler.heatsink.01；今天的 CPU 计划：None

这些机器上 drive_cage / optical_drive → cpu_cooler.heatsink / cpu / cpu_socket_lever 的对在 A 表里（建议收：硬盘架、光驱压着散热块，日志每台先拆它们）。
