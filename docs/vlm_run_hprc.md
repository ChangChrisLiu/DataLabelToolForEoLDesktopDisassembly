# 在 HPRC 上跑 VLM 评测：从拷包到分数表

> **2026-11-05（VLM 最后一轮）**：本文与英文 `HANDOFF.md` §7–§8 对齐，两者不一致时以 `HANDOFF.md` 为准。
> - 数据版本是最终封存快照 **v7**（`tda_sealed_20261104_v7.sqlite`，sha256 `f99a4f8870dc70e3451496ded89bdb8bc1e01840fdb790ab32b5457bf1e798aa`；
>   v5、v6 是中间版本）；包是 `vlm/vlmpack_2026-11-05/`（下文 `$PACK`），评测子集 `eval_v3`（下文 `$EVAL`），对话格式 `chat-v5`。
>   11-03 的 `vlmpack_2026-11-02` / `eval_v2` 已作废并从 HF 删除；`run_eval.sh`、`vlm_collect.py` 等工具拒绝 `eval_v1` / `eval_v2` 的路径。
>   包：184 263 条（scan 81 388、oak1 10 393、oak2 5 068、multi 87 414），`eval_v3` 5 347 条（scan 1 097、oak1 137、oak2 66、multi 4 047），
>   图 8 551 张 / 23.9 GB，整包 31.8 GB / 11 019 个文件；源码 commit `c12dcbb`（并入 main 为 `675dca2`）；标准答案在 184 263 条上全 1.0。
> - 包保持只读，所有输出进 `$WORK`；处处设 `PYTHONDONTWRITEBYTECODE=1`；`RUNS=$WORK/runs`；`MAX_TOKENS` 默认 2048；
>   overlay 只跑 scan 和 multi（只有扫描仪帧有确认的掩码）。
> - **闭源（负责人 11-04 定）：GPT-6、思考强度 Medium、只用 Codex，每题一个新的、隔离的会话，只做 `$EVAL/audit/audit.ids.txt`
>   那 300 条，跑一次，不重复**（见 §5.1）；`submit_matrix.sh` 的 `CLOSED_MODELS` 留空，不需要重复子集和 `REPEAT_IDS`。
> - 单元：相机单元 `scan` / `oak1` / `oak2` / `rs` 只放感知题（V1 V2 V8 V15；V14 已取消），动作题（V3 V4 V6 V7 V10 V12 V16）
>   只在 `multi` 里（一个时刻的所有相机画面；V3 / V12 前后两个时刻最多 8 张；画了红框的题只给扫描仪图）。没有记录的单元不进包，
>   作业 array 读 `tools/vlm_run/pack_units.sh` 列出的单元，不假定五个都在。

这份是**照着做**的流程（W4-B，2026-11）。模型不在本工作站上跑：评测包（`exports/<包文件夹>/`，HF 上 `vlm/<包文件夹>/`）
拷到 TAMU HPRC（或任何有 GPU 的机器），用包里 `tools/` 的脚本完成
导出子集 → 输入变体 → 对话请求 → 模型 → 评分 → 汇总 一条链，全程不需要 `tda` 包，也不碰数据库。

## 0. 测什么、怎么判分（摘要，全文见 `HANDOFF.md` §7）

- **题型**（按论文图的分组）：
  - Perception：V1 Enumeration（列零件 + 框）、V8 Grounding（找一个零件的框）、V2 State（状态，多一个 `cannot_tell`「从这张图看不出来」，
    V14 取消；含 Screw count 数螺丝）、V15 Cross-view（两台相机是否同一时刻）；
  - Understanding：V3 Action（不给提示时同时答 `region` 变化区域 / 给黄色虚线提示框 / 多步：相隔 2–3 步做了哪些）、
    V12 State change（相邻步和相隔 2、3 步，有没有变）、V10 Progress（进度）；
  - Planning：V4 Feasibility（现在能不能做、先要处理什么）、V6 Next action（开放 `{"actions": [...], "why"}` / 选择：同一动词、恰好一个能做 / 给约束）、
    V7 Plan（不给约束：标准答案是最短合法计划 / 给约束）、V16 Plan check（一串步骤能否执行、第一错步，按规则重放）。
- **真值**：AND/OR 图 + 规则 R1–R17 + 动态遮挡（上面的零件盖住 ≥ 20 % 先拿走；线缆、硬盘架盖插头、不动的支架不算）+ 掉落零件优先；
  D66 不出 V4 / V6 / V7 / V16；题目、答案、推理都不带序号。
- **判分**：strict / lenient 两种解析；V1 / V8 按 IoU ≥ 0.5（另报 COCO 式 0.50:0.05:0.95 平均 `f1_50_95` / `hit_50_95`），
  评分器对每个框按 ×1.0 和 ×1.25 两种坐标各算一次取高，并记下胜出的坐标；V2 计数对 1、差 1 个 0.8；V3 区域 IoU / 命中、多步集合 F1；
  V6 开放 / 给约束按动作组算精确率 / 召回 / F1（给约束另报 `constraint_violation`）；V7 合法性、到达目标、必需步骤召回（给约束另报
  `count_exact`、`count_abs_error`）；V16 valid 和第一错步（真值来源 `simulated`）；
  参照线 chance / frequency / blind / protocol prior / state oracle / mask overlay；按台聚类自助法 95 % 区间（1 000 次、种子 0）。
- **推理链** `$PACK/reasoning/<unit>.jsonl`（每行 `{id, task, question, answer, reasoning}`）是训练用的附带产物，**绝不放进给被测模型的提示里**。

驱动脚本在仓库里是 `scripts/vlm_run/`，W4-A 把它拷进包的 `tools/vlm_run/`：

| 文件 | 作用 |
|---|---|
| `vlm_run.py` | 调模型：`openai`（任何 OpenAI 兼容端点）/ `vllm`（离线）/ `transformers`（回退）/ `gold`（伪模型，从标签作答） |
| `vlm_collect.py` | `subset`（按 id 取子集并带上 sidecar）/ `blind`（去图像变体）/ `split`（拆 `--repeat` 文件，本项目不用）/ `table`（汇总分数表） |
| `run_eval.sh` | 一个作业：子集 → 变体 → `vlm_to_chat` → `vlm_run` → `vlm_score`，全靠环境变量 |
| `collect_scores.sh` | 把 `runs/` 下所有 `*.score.json` 汇成 `runs/score.md` |
| `slurm/gold_check.sbatch` | 集群上的 gold 自检（CPU，array 包里的各单元） |
| `slurm/open_model.sbatch` | 开源模型 × 一种输入变体，array 包里的各单元，一张 GPU |
| `pack_units.sh` | 打印包里实际有的单元（`manifest.json` 的 `units`；缺 `python` 时用 `python3`） |
| `slurm/closed_model.sbatch` | 闭源模型（API），CPU，key 只从环境变量来 |
| `slurm/submit_matrix.sh` | 模型 × 变体全部提交 |
| `requirements.txt` / `requirements-vllm.txt` | 环境（见 §2） |

输出目录固定为 `runs/<model_tag>/<variant>/<unit>/`：`truth/`（本次跑的记录 + sidecar）、`chat.jsonl`、`pred.jsonl`、
`run.log`、`score/<model_tag>.score.{json,md}`（重复跑另有 `pred.r1..rk.jsonl` 与 `<model_tag>.r1..rk.score.*`；本项目不做重复跑）。

---

## 1. 拷包

推荐从 HF 下载（`HANDOFF.md` §8.1）：

```bash
pip install -U huggingface_hub
export PACK_NAME='vlmpack_2026-11-05' EVAL='eval_v3'
huggingface-cli download ChangChrisLiu/E3D-Desktop-Disassembly --repo-type dataset \
    --include "vlm/$PACK_NAME/*" --local-dir $SCRATCH/e3d       # 公开仓库，读不需要 token
export PACK=$SCRATCH/e3d/vlm/$PACK_NAME
```

从工作站直接拷也行：

```bash
# 工作站 → HPRC（Grace 用 grace.hprc.tamu.edu；大包走数据传输节点或 Globus）
rsync -avP exports/$PACK_NAME/ USER@grace.hprc.tamu.edu:/scratch/user/USER/$PACK_NAME/
```

到了集群先按包的 `manifest.json` 逐文件核对 sha256（`HANDOFF.md` §8.1 有脚本），再：

```bash
ls $PACK            # data/（vlm_<unit>.jsonl + sidecar）、$EVAL/、images/、overlays/、reasoning/、tools/、manifest.json
bash $PACK/tools/vlm_run/pack_units.sh $PACK   # 例如 scan oak1 oak2 multi
```

`run_eval.sh` 依次在 `$PACK/data/vlm_<unit>.jsonl`、`$PACK/<unit>.jsonl`、`$PACK/vlm_<unit>.jsonl` 找全量导出，图像根目录默认
`$PACK/images`；别的路径用 `TRUTH=` / `IMAGE_ROOT=` 覆盖。包里的 `.sh` / `.sbatch` 已经是 LF，**不要在包上跑 `dos2unix`**
（会改变 sha256）。工作目录：`export WORK=$SCRATCH/e3d_eval; mkdir -p $WORK/logs; export PYTHONDONTWRITEBYTECODE=1`，
把 `$PACK/tools/vlm_run/slurm` 拷到 `$WORK/slurm` 再改；所有输出都进 `$WORK`，包保持只读。

## 2. 建环境（一次）

```bash
# TODO(hprc)：按集群当时的模块名改；vllm 0.11 对应 CUDA 12.8
module purge
module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0
python -m venv $SCRATCH/venvs/vlmpack && source $SCRATCH/venvs/vlmpack/bin/activate
pip install -U pip
pip install -r $PACK/tools/vlm_run/requirements.txt                 # 转换器 + 评分器 + gold + openai 后端
pip install -r $PACK/tools/vlm_run/requirements-vllm.txt            # 开源模型（GPU 节点才需要）
pip freeze > $WORK/requirements.lock                                 # 装成功后锁一份，写进论文附录（写到工作目录，不写进包）
```

权重放 scratch，不放 `$HOME`（配额小）。登录节点有网，先下好，计算节点离线跑：

```bash
export HF_HOME=$SCRATCH/hf
hf download Qwen/Qwen3-VL-8B-Instruct
hf download OpenGVLab/InternVL3_5-8B
```

`sbatch` 模板里 `HF_HUB_OFFLINE=1`；计算节点要联网（闭源 API）时 `module load WebProxy`（模板里是注释，按集群文档开）。

## 3. gold 自检（不用模型，必须每项 1.000）

先跑包自带的自检（标准答案过一遍评分器），再在登录节点用 gold 后端跑一个单元，几十秒（动作题都在 `multi`，再跑一次 `VIEW=multi`）：

```bash
cd $WORK
T=""; for U in $(bash $PACK/tools/vlm_run/pack_units.sh $PACK); do T="$T $PACK/$EVAL/$U.jsonl"; done
python $PACK/tools/vlm_selfcheck.py --truth $T --out $WORK/runs/selfcheck
PACK=$PACK RUNS=$WORK/runs VIEW=scan BACKEND=gold MODEL_TAG=gold bash $PACK/tools/vlm_run/run_eval.sh
cat $WORK/runs/gold/image/scan/score/gold.score.md
```

期望：`parse failures: strict 0.0%, lenient 0.0%`，每个任务、题式、层、模板、附加指标行（IoU@0.5、COCO 式平均、V6 前沿 F1、
V7 合法性 / 必需步骤召回 / 数量准确率、V16 第一错步）strict = lenient = `1.000 [1.000, 1.000]`；
越低越好的比率是 0.000，chance / frequency 列是参照线，不是这次的结果。
不是 1.000 说明包、转换器或评分器三者之一在这台机器上不一致——先修这个，再花 GPU 时。所有单元一起：

```bash
cd $WORK && sbatch --export=ALL,PACK=$PACK,RUNS=$WORK/runs $WORK/slurm/gold_check.sbatch
```

去图像变体的 gold 也应是 1.000（gold 从标签作答，不看图）：`VARIANT=blind`。去图像变体把题面里的
"outlined in red" 去掉，按标记字母点名的零件写成 "part A (a RAM module)"，不丢题。

## 4. 试点（一个开源模型，scan 单元）

试点只在 **train 台**上做，从评测集里按任务分层抽，每个任务最多 9 条（按单元各抽一次：scan 36、oak1 12、oak2 9、multi 63 条；见 HANDOFF §8.4）：

```bash
cd $WORK                     # pilot_ids.txt 写在工作目录；multi 的试点把下面的 scan 换成 multi
python - <<'EOF'
import hashlib, json, collections, os
rows = [json.loads(l) for l in open(f"{os.environ['PACK']}/{os.environ['EVAL']}/scan.jsonl", encoding="utf-8")]
rows = [r for r in rows if r["label"].get("split") == "train"]
rows.sort(key=lambda r: hashlib.sha256(("pilot0|" + r["id"]).encode()).hexdigest())
per = collections.Counter(); keep = []
for r in rows:
    t = r["prompt"]["task"]
    if per[t] < 9: per[t] += 1; keep.append(r["id"])
open("pilot_ids.txt", "w").write("\n".join(keep[:100]) + "\n")
print(len(keep[:100]), dict(per))
EOF
```

**先 dry-run**（不加载模型，校验每张图都在，打印第一条完整提示）：

```bash
VIEW=scan; OUT=$WORK/runs/pilot_qwen3vl8b/image/$VIEW
python $PACK/tools/vlm_run/vlm_collect.py subset --export $PACK/data/vlm_$VIEW.jsonl --out $OUT/truth/$VIEW.jsonl --ids $WORK/pilot_ids.txt
python $PACK/tools/vlm_to_chat.py --export $OUT/truth/$VIEW.jsonl --format openai --image-root $PACK/images --out $OUT/chat.jsonl
python $PACK/tools/vlm_run/vlm_run.py --backend vllm --model Qwen/Qwen3-VL-8B-Instruct --chat $OUT/chat.jsonl \
  --out $OUT/pred.jsonl --image-root $PACK/images --max-side 1568 --dry-run
```

然后提交（`--array=0` 只跑 `pack_units.sh` 列出的第一个单元，即 scan；动作题的试点把 `scan` 换成 `multi`，并用 `VIEW_LIST=multi`）：

```bash
sbatch --array=0 --export=ALL,PACK=$PACK,RUNS=$WORK/runs,MODEL=Qwen/Qwen3-VL-8B-Instruct,SUBSET=$WORK/pilot_ids.txt,MODEL_TAG=pilot_qwen3vl8b \
  $WORK/slurm/open_model.sbatch
```

看四样东西：`score/pilot_qwen3vl8b.score.md` 的 **parse failure**（strict 高、lenient 低 = 模型爱加说明文字，
提示词或 `--max-tokens` 要调）；`pred.jsonl` 里有没有 `finish_reason == "length"`（V1 长列表、V6 开放题的全部动作、V7 计划都可能被截断；
有就把 `MAX_TOKENS` 全部改成 4096 并记下）；`run.log` 的每批耗时（推全量用）；`meta.prompt_tokens` /
`completion_tokens` 的均值（粗估闭源开销，见 §7）。试点题留在 `$EVAL` 里；在它们上调过的东西（提示、token、图尺寸）要在论文里说明。回退路线：`BACKEND=transformers`（一条一条生成，慢十倍，
需要 HF 格式权重，InternVL 用 `*-hf` 仓库；vllm 两种都认）。

## 5. 全量评测

编辑 `$WORK/slurm/submit_matrix.sh`（包里那份的副本）的开源模型清单（`CLOSED_MODELS` 留空），然后：

```bash
cd $WORK && PACK=$PACK RUNS=$WORK/runs bash $WORK/slurm/submit_matrix.sh
squeue -u $USER
```

每个作业 = 一个模型 × 一种变体（`image` 原图 / `blind` 去图像，题面与选项只去掉标记措辞）× array 包里的各单元，评测集
`$EVAL/<unit>.jsonl`（所有开源模型同一子集）。`multi` 的请求最多 8 张图（`open_model.sbatch` 对 multi 设 `--max-images 8`）；
sbatch 默认 `MAX_SIDE=1568`、`MAX_TOKENS=2048`。模型：Qwen/Qwen3-VL-8B-Instruct、OpenGVLab/InternVL3_5-8B，vLLM、temperature 0、各跑一次。

**300 题对比行**：开源模型的全量 `pred.jsonl` 也用 300 道审核题的真值再评一次（带 `--chat`），和闭源、作者审核的错误率放在一起：

```bash
for U in $(bash $PACK/tools/vlm_run/pack_units.sh $PACK); do
  python $PACK/tools/vlm_run/vlm_collect.py subset --export $PACK/$EVAL/$U.jsonl \
      --ids $PACK/$EVAL/audit/audit.ids.txt --out $WORK/audit300/truth/$U.jsonl
  for M in Qwen_Qwen3-VL-8B-Instruct OpenGVLab_InternVL3_5-8B; do
    D=$WORK/runs/$M/image/$U
    python $PACK/tools/vlm_score.py --truth $WORK/audit300/truth/$U.jsonl --pred $D/pred.jsonl \
        --chat $D/chat.jsonl --out $WORK/audit300/score --name $M-audit300-$U
  done
done
```

（某个单元没有审核题时真值文件是空的，跳过。）

### 5.1 闭源：GPT-6 + Codex，只做 300 条

负责人 11-04 定：**GPT-6、思考强度 Medium、只用 Codex、每题一个新的隔离会话**，只答 `$EVAL/audit/audit.ids.txt` 的 300 条
（64 台；multi 225、scan 65、oak1 8、oak2 2；也就是作者审核的同一批），跑一次 `image`，不重复、不报标准差（Codex 不暴露 temperature）；
要不要加一遍 `blind` 先问负责人。

接力包 `vlm/closed_model_run_2026-11-05/`（`huggingface-cli download ChangChrisLiu/E3D-Desktop-Disassembly --repo-type dataset --include "vlm/closed_model_run_2026-11-05/*" --local-dir $SCRATCH/e3d`，
先读它的 README）已备好下面第 1–2 步的结果：`items.jsonl`（300 题全部，每行 n、id、task、form、unit、text、images；`text` 是一字不差的题面，
`images` 是按顺序要给的图）、`images/`、`chat/<unit>.jsonl`（评分换算框要用）、`score_all.py`。

1. 切题：`python $PACK/tools/vlm_run/vlm_collect.py subset --export $PACK/$EVAL/<unit>.jsonl --ids $PACK/$EVAL/audit/audit.ids.txt --out $WORK/closed/truth/<unit>.jsonl`。
2. 生成题面和图：`python $PACK/tools/vlm_to_chat.py --export $WORK/closed/truth/<unit>.jsonl --format openai --image-root $PACK/images --assets $WORK/closed/assets/<unit> --max-side 1568 --out $WORK/closed/chat/<unit>.jsonl`。
   每行的 `messages`：系统说明一行，然后用户文本（图像说明、题目、选项、答题格式、术语表）和按顺序的图（画好标记的副本在 `--assets` 里）。
   给 Codex 的文字 = 系统说明 + 空行 + 用户文本，一字不差；图按 "Image 1…" 的顺序给，不再缩放裁剪。`chat/<unit>.jsonl` 要留着，评分要用。
3. **隔离**：Codex 能读文件、跑命令。每题在 `$PACK`、`$WORK`、接力包之外新建空目录，只放这一题的图；新会话，GPT-6 + Medium，
   只读沙箱，除模型外不联网，不用记忆和自定义指令；取最终回复作 `raw_text`。不能让它看到包、真值、接力包的 `items.jsonl` / `chat/` 或别的题。
   实际用的 Codex 版本和设置记进 `meta`（第 4 步）。
4. 记录：写 `$WORK/runs/gpt6_medium_codex/image/<unit>/pred.jsonl`，每行 `{"id", "raw_text", "meta"}`，`meta` 记 Codex 版本、
   Codex 报告的模型名、推理强度、送图尺寸、日期、尝试次数。驱动脚本放 `$WORK`，不进包。
5. 投递失败（空回答、会话丢了、报错）：新会话重发，最多 3 次，记到 `attempts.md`；拿到有效回答后不再重问（答错不算投递失败）。
6. 两台机器：按行对半分，各写各的文件，`cat` 合并，检查每个 id 只出现一次。
7. 评分：`python $PACK/tools/vlm_score.py --truth $WORK/closed/truth/<unit>.jsonl --pred <pred.jsonl> --chat $WORK/closed/chat/<unit>.jsonl --out <目录> --name gpt6-medium-<unit>`。
   V1 / V8：IoU ≥ 0.5 为主，另报 COCO 式 0.50:0.05:0.95 平均（`f1_50_95` / `hit_50_95`）；每个框在送去的图坐标（×1.0）和 ×1.25 下
   各算一次取高，所有模型同一规则，评分器自动做并记下胜出的坐标，不用加开关。
   用接力包时：把所有回答合成一个 `replies.jsonl`（每行 `{"id", "raw_text", "meta"}`），`python score_all.py --pack $PACK --replies replies.jsonl`。
8. 300 题答完后，把「意思对、用词没对上」的题挑出来给负责人看一眼，确认判分器不太严。

**旧的网页版结果**（11-04，112 题，chat-v3 旧题面）：作废，一条都不用；`vlm/closed_model_run_2026-11-04/` 已从 HF 删除。

第三种变体 `VARIANT=overlay`（W4-C）：把**已确认的可见掩码**按类别颜色画在原图上（固定透明度、细描边，不写字不画框），
只对每张图都是已确认帧且有掩码的题出（V8 除外——框就是答案；`multi` 只画扫描仪那几张，侧面相机原样），是论文的"感知参照线"。包里 `overlays/` 已预先画好
`$EVAL` 的那份（scan 1 097 条里 598 条可答，multi 4 047 条全可答，oak1 / oak2 0 条）；别的子集会从 `<unit>.masks.jsonl` sidecar 现画（`OVERLAYS=` 指到可写目录）。它答的题比原图少，
`run_eval.sh` 仍按完整子集评分（没答的记为 parse failure），看 `truth/mask_overlay.summary.json` 知道少了哪些；
`vlm_score.py` 的 `score.md` 同时给出 `chance` / `frequency` 两行参照。`submit_matrix.sh` 的清单默认只有
`image blind`，要跑 overlay 自己加进 `VARIANTS`。**overlay 只对 scan 和 multi 有意义**（其他相机没有确认的掩码，
可答 0 条）；与原图比较时，两者都按 `truth/mask_overlay.jsonl` 的 id 重新评分。

**中断 / 失败重跑**：同一命令再提交一次即可——`vlm_run.py` 跳过 `pred.jsonl` 里已答的 id，只补 `raw_text = null`
的行；退出码 3 = 有请求失败（行已写、带 `meta.error`），0 = 全部答完。

**闭源的 key**（只在用 API 跑闭源时需要；本项目的闭源走 Codex，见上）：`echo 'export OPENAI_API_KEY=...' > ~/.vlm_keys.env && chmod 600 ~/.vlm_keys.env`。
`closed_model.sbatch` source 这个文件，`API_KEY_ENV` 只传**变量名**；`vlm_run.py` 不打印、不写入、错误信息里也替换掉。
别的端点：`BASE_URL=https://.../v1 API_KEY_ENV=MY_KEY`（Gemini / Anthropic 的 OpenAI 兼容层、或集群内 `vllm serve`
加 `RUN_ARGS=--no-api-key`）。

常用旋钮（都走环境变量进 `run_eval.sh`）：`MAX_SIDE`（送图最长边；sbatch 默认 1568，单独跑 `run_eval.sh` 时默认不缩放）、`MAX_TOKENS`（默认 2048：V1 要列出全部可见零件，512 会截断）、
`ROI=view`（按视角固定底盘裁切；跨出裁切框的红色标记框会在边缘被截断）、`CHAT_FORMAT`、`RUN_ARGS="--tensor-parallel 4 --dtype bf16 --batch-size 8
--gpu-mem 0.85 --max-model-len 16384"`。OOM 先降 `--batch-size` 与 `MAX_SIDE`，再加卡 + `--tensor-parallel`。

## 6. 收分数

```bash
PACK=$PACK RUNS=$WORK/runs bash $PACK/tools/vlm_run/collect_scores.sh
cat $WORK/runs/score.md
```

`vlm_collect.py table` 把每个 `runs/<model>/<variant>/<unit>/score/*.score.json` 变成一行（strict 分数 +
按台聚类自助法 95 % 区间）。单个作业的细表看
`score/<model_tag>.score.md`：按任务 / 题式 / 能力层 / 模板，附加指标（V1 / V8 的 IoU@0.5 和 `f1_50_95` / `hit_50_95`、V3 区域 IoU / 命中和 tool、
V4 blockers F1、V6 按组 F1 和 `constraint_violation`、V7 合法性 / 到达目标 / 必需步骤召回 / `count_exact` / `count_abs_error`、V10 进度分箱、V16 第一错步）、文本捷径上界列。

参照线：`python $PACK/tools/vlm_references.py --export $PACK/$EVAL/<unit>.jsonl --out $WORK/runs/refs/<unit> --assets $WORK/overlays --image-root $PACK/images`
写出 blind、state_oracle、mask_overlay 和 protocol prior（`--assets` 不要指到 `$PACK/overlays`，否则会画进包里）；
protocol prior 只在 multi 上有意义（V6 / V7 / V16），用 `vlm_score.py --truth $PACK/$EVAL/multi.jsonl --pred $WORK/runs/refs/multi/protocol_prior.predictions.jsonl --out $WORK/runs/scores --name protocol-prior-multi` 评分；
状态 oracle 要让模型作答：`vlm_to_chat.py --export $PACK/$EVAL/<unit>.jsonl --records $WORK/runs/refs/<unit>/state_oracle.jsonl --format openai --out …` → `vlm_run.py` → `vlm_score.py`。

**作者审核**（负责人做）：300 题来自 64 台（V1、V2、V3 不给提示、V3 跨步、V4、V6 开放 / 选择 / 给约束、V7 不给约束 / 给约束、V8、V10 每桶 19 题，
V12 36 题含跨步，V15、V16 各 18 题），其中 142 题沿用自 `eval_v2` 的审核样本。去掉 V1 / V8 要审 262 题，12 个 V2 结论沿用第一轮
（`vlm/owner_review/eval_v3_audit_carried.csv`），实际还剩 250 题。打开 `$PACK/$EVAL/audit/audit.sheet.b01.html` 起的各页（和以前一样，每页 50 题），每题先显示模型看到的图，
审核层（标准答案、证据框、推理链）在开关后面、绝不给模型看；只审生成的答案（V1 / V8 不审）；每页 **Export answers** 导出 CSV，
`python $PACK/tools/vlm_audit_summary.py --csv <CSV> --records $PACK/$EVAL/audit/audit.jsonl --out <目录>` 给出错误率和 Wilson 95 % 区间。

**论文要报的**（详见 `HANDOFF.md` §8.8）：
- parse-failure 率两种（strict / lenient），分母是真值文件的全部记录，没答的算失败；
- 开源 temperature 0 只跑一次；闭源（Codex）在 300 条上跑一次，和开源在同 300 条上的分数、审核错误率并排；
- V1 / V8：IoU ≥ 0.5 和 COCO 式平均，两种坐标规则写明；
- 主表按论文图的分组：Perception（V1 Enumeration、V8 Grounding、V2 State 含 Screw count、V15 Cross-view）、
  Understanding（V3 Action、V12 State change、V10 Progress）、Planning（V4 Feasibility、V6 Next action、V7 Plan、V16 Plan check）；
- V2 的主行 `V2` 是全部 V2 题（含 `cannot_tell`），`V2-prior-wrong`（类别先验猜错的那一半）是次要的捷径检查；
- 置信区间 = 按台聚类自助法（簇 = 66 台，1 000 次，`vlm_score.py --resamples 1000 --seed 0`，`run_eval.sh` 用默认值）；
- 每个任务旁边的文本捷径上界与（V6 / V7 / V16）protocol prior；
- 文件里不写任何会议或期刊的名字；
- 七轴：模型快照 `meta.model_snapshot`、送图尺寸 `meta.image_sizes_sent`、`max_tokens`、temperature、
  变体、提示模板 id `meta.prompt_template_id`、`requirements.lock`——全在 `pred.jsonl` 的 `meta` 里。

## 7. 闭源的开销

闭源只在 300 条上跑一次（§5），不再做全量预算表。试点（§4）的 `prompt_tokens` / `completion_tokens` 均值乘以 300
可以粗估 token 数；以 Codex 实际报告的为准，记进 `meta`。

## 8. 常见问题

- `bash: $'\r': command not found` → 脚本带 CRLF：只对 `$WORK` 里的副本 `dos2unix`，不要动包。
- `environment variable X is not set` → `~/.vlm_keys.env` 没 source 或变量名写错；`--dry-run` 会打印 key 是否已设。
- `HTTP 429` / `503` → 已自动指数退避重试（`--max-retries`，尊重 `Retry-After`）；仍失败的行留 `meta.error`，重跑补齐。
- vllm `ValueError: ... images` → 多图题超过 `--max-images`（相机单元默认 2，V15 两图；`multi` 要 8，V3 / V12 前后各 4 张）。
- `transformers` 后端 `apply_chat_template` 报错 → 版本太旧或权重不是 HF 格式；驱动会退到「模板文本 + processor」两步法，
  再不行换 `-hf` 仓库或走 vllm。
- `--dry-run` 报 `missing image` → `IMAGE_ROOT` 与 manifest 的相对路径不配，看包 README。
- 想在工作站上先过一遍链路：`tests/test_w4b_vlm_run.py` 用 gold 后端与本机假服务器把全链跑通（不装 vllm）。

## 9. 后端接口（要加模型时）

`vlm_run.py` 把三种聊天格式都归一成 `Request(id, system, parts[("text"|"image", value)], meta)`；一个后端是
`Backend` 的子类：`describe() -> dict`（写进每行 `meta`）、`generate(list[Request]) -> list[Reply]`（同序，
`Reply(text, error, latency_s, prompt_tokens, completion_tokens, model_snapshot, finish_reason, image_sizes, attempts)`），
`batch_size` 决定一次送几条。图像统一经 `ImagePrep.load(ref) -> Picture(data, mime, size, source_size)`
（`--max-side` 在这里缩放），`openai_messages(req, prep)` 生成带 `data:` 图的 OpenAI 消息，vllm 与 openai 后端都用它。
`vllm` / `transformers` / `torch` 只在对应后端 `_ensure()` 时导入；`--dry-run` 与 `gold` / `openai` 后端在没有它们的机器上照常跑。
