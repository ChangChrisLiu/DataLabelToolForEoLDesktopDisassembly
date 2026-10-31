# 立场稿补充（2026-10-31 00:xx，查 U46 实跑 `.cache/tmp/u46/r2b/D*.json` 所得）

## A. 计划长度（§2 的数字核实）

| 目标 | 台数 | 图上计划长度 min / 中位 / max | 日志实际到达该目标的步数 min / 中位 / max |
|---|---|---|---|
| cpu.01 | 65 | 2 / 7 / 8 | 19 / 30 / 45 |
| motherboard.01 | 66 | 10 / 21 / 27 | 33 / 42.5 / 55 |
| psu.01 | 65 | 1 / 4 / 8 | 12 / 26 / 47 |
| all | 66 | 9 / 41 / 54 | 14 / 42 / 56 |

CPU 目标确实是「短目标」，主板目标是「长目标」，与 §2 的提法一致。PSU 的 D64 计划长度 1（`remove psu.01`，没有任何边）是 D64 未收尾所致，不算。

## B. §6 的精确诊断：缺的不是一条边，是两条

D03 / D07 / D08 / D11 / D14 / D16 / D18 / D19 的 CPU 计划是 `open cpu_socket_lever.01 → remove cpu.01` 两步，散热器还在上面；
D45 是 `open lever → displace fan → remove cpu`。看边：

- 规则边 `covered_by(cpu.01 ← cpu_cooler.fan.NN)`（reason「sits on cpu」）**只对 kind = fan 生成**。D03 这种「风扇 + 散热块」两件的机器，压在 CPU 上的是 heatsink，而 heatsink 没有这条边，于是图认为 CPU 裸露 → 两步计划。heatsink 22 台、heatsink_fan 4 台里凡是没有 fan 件的都会这样。
- **没有任何 `blocked_by(cpu_socket_lever ← cpu_cooler.*)` 边**，所以即便风扇机器（D01）计划长度对了，顺序也是先开拉杆再拆散热器——拉杆被散热器的固定架压着，物理上开不了。65/65 台日志都是先拆散热器后开拉杆，0 反例。

所以 §6 应改成两条规则，都来自同一个事实「散热器压在 CPU 插座上」：
1. `covered_by(cpu.01 ← cpu_cooler.<任意 kind>)`，必要性 required；一台同时有 heatsink 与 fan 时，heatsink 压 CPU、fan 压 heatsink（`covered_by(heatsink ← fan)`），不要让 fan 直接压 CPU。
2. `blocked_by(cpu_socket_lever.N ← cpu_cooler.<压在 CPU 上的那件>)`，required。

效果：CPU 计划在 heatsink 机器上会从 2 步变成「拧螺丝 ×4 → 拆散热块 → 开拉杆 → 取 CPU」；V5 的合法集合里不再有第 0 步「打开 CPU 拉杆」。
验证方式：在副本上加规则后重跑 `cli constraints --dry-run --validate` 看违规数不涨（预期仍 24），再跑 `export-graph` 比较 `plans.cpu.01.length` 分布与候选数（cpu_cooler.*→cpu_socket_lever 的 56 台候选应消失）。

## C. 顺带的数据发现

- 主板目标：图上中位 21 步 vs 日志中位 42.5 步。差额主要是日志里拆了但主板不需要的零件（硬盘、光驱、PSU、内存、CPU…），不是漏边；论文里讲「演示顺序 vs 最短合法计划」时要把这一点说清，否则审稿人会以为图漏了一半。
