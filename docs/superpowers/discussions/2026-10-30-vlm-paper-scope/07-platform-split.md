# 平台聚类与划分提案（第 1 天）

来源：新副本 `desktop` 表（品牌、型号族、外形尺寸 L*W*H 英寸、机箱类型）+ 实例表（散热器配置、硬盘架）。型号族拼写已规范化（Compoq / Compac → Compaq，GI → G1，Small Form Fador → SFF），**SFF / CMT / TWR 后缀保留**：同一型号线的 SFF 和 CMT 是两个机箱平台。
**尺寸**是最硬的平台证据：同一机箱平台尺寸一致到 1/8 英寸；一个族里尺寸不同就是两个平台。划分按族整体留出（平台不相交）；零样本评测的结论不依赖划分，划分是给将来微调者和数据卡的。

数据录入疑点（请看一眼）：D55 品牌 Dell 但型号族写的是 Compaq 8200 Elite；D45 型号族「customized desktop from alternative」；D36 / D39 / D54「Regular model」；desktop 表里还有一行 id 0。

## A. 平台表（按台数排序）

| 品牌 | 型号族（规范化） | 台（尺寸 L*W*H）| 台数 | 散热器配置 | 硬盘架 | 全部已确认 |
|---|---|---|---|---|---|---|
| Dell | optiplex 7020 | 12.25*11.5*3.75: D06,07,09,12,13,17,21; 12.3125*11.5*3.75: D02; 12.375*11.5*3.75: D05; 16.25*14.25*6.9375: D37; 16.25*14.25*7: D38 | 11 | fan×10, fan+heatsink×1 | 9/11 | 是 |
| Dell | optiplex 7010 | 12.25*11.5*3.75: D03,11; 16.375*14.25*6.9375: D40,41; 12.1875*11.4375*3.75: D08; 12.375*11.5*3.75: D18 | 6 | fan+heatsink×4, fan×2 | 4/6 | 是 |
| Dell | optiplex 3020 | 12.375*11.5*3.75: D10,15,20; 12.25*11.5*3.75: D04; 15.75*14.1875*6.75: D65 | 5 | fan×5 | 4/5 | 是 |
| HP | compaq 8200 elite | 17.625*17.375*7.125: D53,57,59,66; 17.625*16.875*7.125: D52 | 5 | fan×5 | 0/5 | 是 |
| HP | elitedesk 800 g1 | 17.375*15.75*6.75: D44,47; 17.375*15.8125*6.9375: D50,51; 15.75*17.375*6.8125: D42 | 5 | fan×5 | 0/5 | 是 |
| Dell | optiplex 3010 | 12.375*11.5*3.75: D14; 12.3125*11.375*3.75: D16 | 2 | fan+heatsink×2 | 2/2 | 是 |
| Dell | optiplex 9020 | 16.25*14.25*7: D43; 16.25*14.25*6.9375: D60 | 2 | fan×2 | 0/2 | 是 |
| Dell | regular model | 15.5*13.6875*4.4375: D36; 15.8125*14.25*6.9375: D39 | 2 | heatsink×1, fan×1 | 0/2 | 是 |
| HP | compaq 8100 elite sff | 14.9375*13.375*4: D23,26 | 2 | heatsink×2 | 2/2 | 是 |
| HP | compaq 8200 elite sff | 14.9375*13.375*4: D22,29 | 2 | heatsink×2 | 2/2 | 是 |
| HP | compaq 8300 elite sff | 14.9375*13.375*4: D24; 15.375*13.375*4.0625: D32 | 2 | heatsink×2 | 2/2 | 是 |
| ? | customized desktop from alternative | 16.375*13.75*7: D45 | 1 | fan×1 | 0/1 | 是 |
| Apple | g4 | ?: D64 | 1 | fan+heatsink×1 | 0/1 | 否：D64 |
| Dell | 5500 | 17.625*17.6875*6.625: D62 | 1 | heatsink×1 | 0/1 | 是 |
| Dell | compaq 8200 elite cmt | 17.625*17.375*7: D55 | 1 | fan×1 | 0/1 | 是 |
| Dell | optiplex 380 | 15.75*14*4.25: D35 | 1 | heatsink×1 | 0/1 | 是 |
| Dell | optiplex 990 | 12.25*11.5*3.75: D19 | 1 | fan+heatsink×1 | 1/1 | 是 |
| Dell | precision t5500 | 17.75*17.5*7: D49 | 1 | heatsink×1 | 0/1 | 是 |
| Dell | precision t7500 | 22.625*20.375*8.375: D63 | 1 | heatsink×1 | 0/1 | 否：D63 |
| Dell | regular model 2 | 15.75*14.25*6.875: D54 | 1 | fan×1 | 0/1 | 是 |
| HP | compaq 4300 elite sff | 14.9375*13.375*4: D25 | 1 | heatsink×1 | 1/1 | 是 |
| HP | compaq 8100 elite | 15.375*13.375*4.0625: D33 | 1 | heatsink×1 | 1/1 | 是 |
| HP | elitedesk 800 g1 sff | 15*13.3725*4: D28 | 1 | heatsink_fan×1 | 1/1 | 是 |
| HP | elitedesk 800 g2 | 16.75*15.3125*6.875: D58 | 1 | fan×1 | 0/1 | 是 |
| HP | elitedesk 800 g2 twr | 16.75*15.375*6.875: D01 | 1 | fan×1 | 0/1 | 否：D01 |
| HP | pavilion a6257c | 15.6875*14.4375*6.5625: D61 | 1 | fan×1 | 1/1 | 是 |
| HP | pro 3500 series mt | 14.5625*14.0625*6.5625: D34 | 1 | fan×1 | 0/1 | 是 |
| HP | prodesk 400 g1 sff | 15*13.3725*4: D27 | 1 | heatsink_fan×1 | 1/1 | 是 |
| HP | prodesk 4300 | 14.875*13.375*4: D31 | 1 | heatsink_fan×1 | 1/1 | 是 |
| HP | prodesk 4300 g1 sff | 14.875*13.375*4: D30 | 1 | heatsink_fan×1 | 1/1 | 是 |
| HP | z230 | 17.4375*15.75*6.625: D56 | 1 | fan×1 | 0/1 | 是 |
| HP | z440 | 17.5*17*6.625: D46 | 1 | fan×1 | 0/1 | 是 |
| Lenovo | m91p | 14.625*13.25*4: D48 | 1 | fan×1 | 1/1 | 是 |

注意：Optiplex 7020 / 7010 / 3020 各族里都同时有 SFF（约 12.3×11.5×3.75）和 MT（约 16×14×7）两种尺寸，即两个机箱平台；留出 3020 整族时两种形态都在测试侧。

## B. 测试划分提案（整族留出，约 12 台，两个品牌都有，全部已确认）

原则：留出的族里每一台都已全部确认；最大的两族（Optiplex 7020 ×11、7010 ×6）留在训练侧，测试侧要的是「没见过的平台」；留出 1 个中等 Dell 族 + 2 个小 HP 族 + 3 个单台族覆盖 SFF / MT / 工作站形态。

- Dell optiplex 3020：D04,10,15,20,65（5 台，全部已确认；散热器 fan）
- HP compaq 8300 elite sff：D24,32（2 台，全部已确认；散热器 heatsink）
- HP prodesk 4300（g1 sff）：D30,31（2 台，全部已确认；散热器 heatsink_fan）
- Lenovo m91p：D48（1 台，全部已确认；散热器 fan）
- Dell precision t5500：D49（1 台，全部已确认；散热器 heatsink）
- HP z440：D46（1 台，全部已确认；散热器 fan）

合计 12 台 → `desktop.split = test`；其余 `train`（D63 / D64 收尾后归 train）。你改族或换台都行，只要整族一起动。

## C. 备选的全确认小族（可换）

- Dell optiplex 3020：D04,10,15,20,65（5 台）
- HP compaq 8200 elite：D52,53,57,59,66（5 台）
- HP elitedesk 800 g1：D42,44,47,50,51（5 台）
- Dell optiplex 3010：D14,16（2 台）
- Dell optiplex 9020：D43,60（2 台）
- Dell regular model：D36,39（2 台）
- HP compaq 8100 elite sff：D23,26（2 台）
- HP compaq 8200 elite sff：D22,29（2 台）
- HP compaq 8300 elite sff：D24,32（2 台）
- ? customized desktop from alternative：D45（1 台）
- Dell 5500：D62（1 台）
- Dell compaq 8200 elite cmt：D55（1 台）
- Dell optiplex 380：D35（1 台）
- Dell optiplex 990：D19（1 台）
- Dell precision t5500：D49（1 台）
- Dell regular model 2：D54（1 台）
- HP compaq 4300 elite sff：D25（1 台）
- HP compaq 8100 elite：D33（1 台）
- HP elitedesk 800 g1 sff：D28（1 台）
- HP elitedesk 800 g2：D58（1 台）
- HP pavilion a6257c：D61（1 台）
- HP pro 3500 series mt：D34（1 台）
- HP prodesk 400 g1 sff：D27（1 台）
- HP prodesk 4300（g1 sff）：D31（1 台）
- HP prodesk 4300 g1 sff：D30（1 台）
- HP z230：D56（1 台）
- HP z440：D46（1 台）
- Lenovo m91p：D48（1 台）
