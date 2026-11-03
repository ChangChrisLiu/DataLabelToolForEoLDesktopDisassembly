# 在 HPRC 上跑 VLM 评测：从拷包到分数表

> **2026-11-03**：本文已与英文 `HANDOFF.md` §8 对齐：包保持只读，所有输出进 `$WORK`；处处设 `PYTHONDONTWRITEBYTECODE=1`；`RUNS=$WORK/runs`；
> `MAX_TOKENS` 默认 2048；overlay 只跑 scan；重复子集生成到 `$WORK/repeat/` 并经 `REPEAT_IDS` 传入。两者不一致时以 `HANDOFF.md` 为准。

这份是**照着做**的流程（W4-B，2026-11）。模型不在本工作站上跑：评测包（W4-A 的 `exports/vlmpack_<date>/`）
拷到 TAMU HPRC（或任何有 GPU / 有 API 出口的机器），用包里 `tools/` 的脚本完成
导出子集 → 输入变体 → 对话请求 → 模型 → 评分 → 汇总 一条链，全程不需要 `tda` 包，也不碰数据库。

驱动脚本在仓库里是 `scripts/vlm_run/`，W4-A 把它拷进包的 `tools/vlm_run/`：

| 文件 | 作用 |
|---|---|
| `vlm_run.py` | 调模型：`openai`（任何 OpenAI 兼容端点）/ `vllm`（离线）/ `transformers`（回退）/ `gold`（伪模型，从标签作答） |
| `vlm_collect.py` | `subset`（按 id 取子集并带上 sidecar）/ `blind`（去图像变体）/ `split`（拆 `--repeat` 文件）/ `table`（汇总分数表） |
| `run_eval.sh` | 一个作业：子集 → 变体 → `vlm_to_chat` → `vlm_run` → `vlm_score`，全靠环境变量 |
| `collect_scores.sh` | 把 `runs/` 下所有 `*.score.json` 汇成 `runs/score.md` |
| `slurm/gold_check.sbatch` | 集群上的 gold 自检（CPU，array 四个视角） |
| `slurm/open_model.sbatch` | 开源模型 × 一种输入变体，array 四个视角，一张 GPU |
| `slurm/closed_model.sbatch` | 闭源模型（API），CPU，key 只从环境变量来 |
| `slurm/submit_matrix.sh` | 模型 × 变体全部提交 |
| `requirements.txt` / `requirements-vllm.txt` | 环境（见 §2） |

输出目录固定为 `runs/<model_tag>/<variant>/<view>/`：`truth/`（本次跑的记录 + sidecar）、`chat.jsonl`、`pred.jsonl`、
`run.log`、`score/<model_tag>.score.{json,md}`（重复跑另有 `pred.r1..rk.jsonl` 与 `<model_tag>.r1..rk.score.*`）。

---

## 1. 拷包

```bash
# 工作站 → HPRC（Grace 用 grace.hprc.tamu.edu；大包走数据传输节点或 Globus）
rsync -avP exports/vlmpack_2026-11-02/ USER@grace.hprc.tamu.edu:/scratch/user/USER/vlmpack_2026-11-02/
```

到了集群先核对包的 `manifest.json`（sha256，见包内 README），再：

```bash
export PACK=$SCRATCH/vlmpack_2026-11-02
ls $PACK            # data/（vlm_<view>.jsonl + sidecar）、eval_v1/、images/、overlays/、tools/、manifest.json
```

`run_eval.sh` 依次在 `$PACK/data/vlm_<view>.jsonl`、`$PACK/<view>.jsonl`、`$PACK/vlm_<view>.jsonl` 找全量导出，图像根目录默认
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

先在登录节点跑一个视角，几十秒：

```bash
cd $WORK
PACK=$PACK RUNS=$WORK/runs VIEW=scan BACKEND=gold MODEL_TAG=gold bash $PACK/tools/vlm_run/run_eval.sh
cat $WORK/runs/gold/image/scan/score/gold.score.md
```

期望：`parse failures: strict 0.0%, lenient 0.0%`，每个任务、层、模板、附加指标行 strict = lenient = `1.000 [1.000, 1.000]`；
`blocked_rate` 0.000 也是对的（越低越好），chance / frequency 列是参照线，不是这次的结果。
不是 1.000 说明包、转换器或评分器三者之一在这台机器上不一致——先修这个，再花 GPU 时。四个视角一起：

```bash
cd $WORK && sbatch --export=ALL,PACK=$PACK,RUNS=$WORK/runs $WORK/slurm/gold_check.sbatch
```

去图像变体的 gold 也应是 1.000（gold 从标签作答，不看图）：`VARIANT=blind`。

## 4. 100 题试点（一个开源模型，一个视角）

试点只在 **train 台**上做（论文 §6.2），从评测集里按任务分层抽 100 条：

```bash
cd $WORK                     # pilot_ids.txt 写在工作目录
python - <<'EOF'
import hashlib, json, collections
rows = [json.loads(l) for l in open(f"{__import__('os').environ['PACK']}/eval_v1/scan.jsonl", encoding="utf-8")]
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

然后提交（`--array=0` 只跑 scan）：

```bash
sbatch --array=0 --export=ALL,PACK=$PACK,RUNS=$WORK/runs,MODEL=Qwen/Qwen3-VL-8B-Instruct,SUBSET=$WORK/pilot_ids.txt,MODEL_TAG=pilot_qwen3vl8b \
  $WORK/slurm/open_model.sbatch
```

看三样东西：`score/pilot_qwen3vl8b.score.md` 的 **parse failure**（strict 高、lenient 低 = 模型爱加说明文字，
提示词或 `--max-tokens` 要调）；`run.log` 的每批耗时（推全量用）；`pred.jsonl` 里 `meta.prompt_tokens` /
`completion_tokens` 的均值（闭源预算用，见 §7）。回退路线：`BACKEND=transformers`（一条一条生成，慢十倍，
需要 HF 格式权重，InternVL 用 `*-hf` 仓库；vllm 两种都认）。

## 5. 全量评测

编辑 `$WORK/slurm/submit_matrix.sh`（包里那份的副本）的两张模型清单，先生成重复子集（见下），然后：

```bash
cd $WORK && PACK=$PACK RUNS=$WORK/runs REPEAT_IDS=$WORK/repeat/repeat.ids.txt bash $WORK/slurm/submit_matrix.sh
squeue -u $USER
```

每个作业 = 一个模型 × 一种变体（`image` 原图 / `blind` 去图像，题面与选项不变）× array 四个视角，评测集
`eval_v1/<view>.jsonl`（所有模型同一子集）。闭源模型另跑 **重复子集**（`$WORK/repeat/repeat.ids.txt`，经 `REPEAT_IDS` 传入，`REPEAT=3`，
每题问三次写 `raw_text_1..3`）。

**重复子集不在包里，第一次跑闭源前先生成（确定性，300 条，每题 25 条，覆盖 66 台）**：

```bash
cd $WORK && cat $PACK/eval_v1/scan.jsonl $PACK/eval_v1/oak1.jsonl $PACK/eval_v1/oak2.jsonl $PACK/eval_v1/rs.jsonl > eval_all.jsonl
PYTHONDONTWRITEBYTECODE=1 python $PACK/tools/vlm_sample.py --export eval_all.jsonl --out repeat \
    --purpose repeat --seed 0 --test-desktops 1-66
# 核对 repeat/repeat.ids.txt：300 行，第一行 rec_68740f3181306b86；LF 行尾的 sha256 = 0801ccab8bddc17d9e2d7e580aadcda61143390a42ebdb78e29e31f760e8b160
```

第三种变体 `VARIANT=overlay`（W4-C）：把**已确认的可见掩码**按类别颜色画在原图上（固定透明度、细描边，不写字不画框），
只对每张图都是已确认帧且有掩码的题出（V8 除外——框就是答案），是论文的"感知参照线"。包里 `overlays/` 已预先画好
`eval_v1` 的那份；别的子集会从 `<view>.masks.jsonl` sidecar 现画（`OVERLAYS=` 指到可写目录）。它答的题比原图少，
`run_eval.sh` 仍按完整子集评分（没答的记为 parse failure），看 `truth/mask_overlay.summary.json` 知道少了哪些；
`vlm_score.py` 的 `score.md` 同时给出 `chance` / `frequency` 两行参照。`submit_matrix.sh` 的清单默认只有
`image blind`，要跑 overlay 自己加进 `VARIANTS`（闭源模型多一份预算）。**overlay 只对 scan 有意义**（其他视角没有确认的掩码，
可答 0 条）；与原图比较时，两者都按 `truth/mask_overlay.jsonl` 的 id 重新评分。

**中断 / 失败重跑**：同一命令再提交一次即可——`vlm_run.py` 跳过 `pred.jsonl` 里已答的 id，只补 `raw_text = null`
的行；退出码 3 = 有请求失败（行已写、带 `meta.error`），0 = 全部答完。

**闭源的 key**：`echo 'export OPENAI_API_KEY=...' > ~/.vlm_keys.env && chmod 600 ~/.vlm_keys.env`。
`closed_model.sbatch` source 这个文件，`API_KEY_ENV` 只传**变量名**；`vlm_run.py` 不打印、不写入、错误信息里也替换掉。
别的端点：`BASE_URL=https://.../v1 API_KEY_ENV=MY_KEY`（Gemini / Anthropic 的 OpenAI 兼容层、或集群内 `vllm serve`
加 `RUN_ARGS=--no-api-key`）。

常用旋钮（都走环境变量进 `run_eval.sh`）：`MAX_SIDE`（送图最长边；sbatch 默认 1568，单独跑 `run_eval.sh` 时默认不缩放）、`MAX_TOKENS`（默认 2048：V1 要列出全部可见零件，512 会截断）、
`ROI=view`（按视角固定底盘裁切）、`CHAT_FORMAT`、`RUN_ARGS="--tensor-parallel 4 --dtype bf16 --batch-size 8
--gpu-mem 0.85 --max-model-len 16384"`。OOM 先降 `--batch-size` 与 `MAX_SIDE`，再加卡 + `--tensor-parallel`。

## 6. 收分数

```bash
PACK=$PACK RUNS=$WORK/runs bash $PACK/tools/vlm_run/collect_scores.sh
cat $WORK/runs/score.md
```

`vlm_collect.py table` 把每个 `runs/<model>/<variant>/<view>/score/*.score.json` 变成一行（strict 准确率 +
按台聚类自助法 95 % 区间），重复跑（`*.r1..rk`）折成 **均值 ± 样本标准差 (k=3)**。单个作业的细表看
`score/<model_tag>.score.md`：按任务 / 能力层 / 模板，附加指标（V6 blocked_rate、V7 violation_free、
V16 localised、V4 blockers_f1、V10 bin_accuracy）、文本捷径上界列。

**论文要报的**（§6.2）：
- parse-failure 率两种（strict / lenient），分母是真值文件的全部记录，没答的算失败；
- 闭源三次重复的标准差（重复子集），开源 temperature 0 只跑一次；
- 置信区间 = 按台聚类自助法（簇 = 66 台，1 000 次，`vlm_score.py --resamples 1000 --seed 0`，`run_eval.sh` 用默认值）；
- 每个任务旁边的文本捷径上界与（V6 / V7 / V16）protocol prior；
- 七轴：模型快照 `meta.model_snapshot`、送图尺寸 `meta.image_sizes_sent`、`max_tokens`、temperature、
  变体、提示模板 id `meta.prompt_template_id`、`requirements.lock`——全在 `pred.jsonl` 的 `meta` 里。

## 7. 闭源模型预算

从试点（§4）读 `meta.prompt_tokens` / `completion_tokens` 的均值，再填下表（价格按供应商当天的报价）：

```
tokens_in  = N_items × ( T_text + n_img × T_img )          # T_img：供应商的图 token 规则；用试点实测均值更准
tokens_out = N_items × T_out
cost       = tokens_in × P_in + tokens_out × P_out
```

| 量 | 值 | 来源 |
|---|---|---|
| N_items（image 变体，四视角合计） | ⟨ ⟩ | `eval_v1/table.md` 各任务 n 之和 |
| N_items（blind 变体） | ⟨同上⟩ | 同一子集，n_img = 0 |
| n_img 均值 | ⟨ ⟩ | 单图任务 1，V3 / V12 / V15 2 |
| T_text（文字 token 均值） | ⟨ ⟩ | blind 试点的 `prompt_tokens` |
| T_img（每图 token，MAX_SIDE=⟨ ⟩） | ⟨ ⟩ | image 试点的 `prompt_tokens` − T_text |
| T_out | ⟨ ⟩ | 试点 `completion_tokens` 均值（JSON 短答一般 < 60） |
| 重复子集 | N_rep = 300 × (k−1 = 2) | `$WORK/repeat/repeat.ids.txt` |
| P_in / P_out（每百万 token） | ⟨ ⟩ / ⟨ ⟩ | 供应商报价 |
| 总计（每个闭源模型） | ⟨ ⟩ | 上式 ×（image + blind）+ 重复子集 |

试点 100 题 × 一个模型就能把 T_* 全部实测出来，不要靠猜。

## 8. 常见问题

- `bash: $'\r': command not found` → 脚本带 CRLF：只对 `$WORK` 里的副本 `dos2unix`，不要动包。
- `environment variable X is not set` → `~/.vlm_keys.env` 没 source 或变量名写错；`--dry-run` 会打印 key 是否已设。
- `HTTP 429` / `503` → 已自动指数退避重试（`--max-retries`，尊重 `Retry-After`）；仍失败的行留 `meta.error`，重跑补齐。
- vllm `ValueError: ... images` → 多图题超过 `--max-images`（默认 2，V3 / V12 / V15 都是两图）。
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
