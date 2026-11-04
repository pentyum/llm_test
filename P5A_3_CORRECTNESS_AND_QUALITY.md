# P5a.3 实验执行计划：MoE BF16 等价性、完整质量评测与可信性能统计

> **交付对象**：Coding Agent（可直接执行）  
> **项目**：`https://github.com/pentyum/llm_test`  
> **基线分支**：`main`（从开始执行时最新提交创建实验分支）  
> **建议实验分支**：`experiment/p5a3-correctness-quality`  
> **实验平台**：Ubuntu、RTX 4060 Laptop 8GB VRAM、48GB RAM、NVMe SSD、Conda  
> **实验模型**：`allenai/OLMoE-1B-7B-0125-Instruct`  
> **日期**：2026-11-04  
> **本轮目标**：先查清 BF16 透传不一致的根因，再验证完整 Prompt 质量结果、动态精度调度与 token 级延迟。**本轮不以更高 tokens/s 为主要目标。**

---

## 0. 给 Agent 的执行约束（必须遵守）

1. **先审计、再改代码、最后跑实验**。不要先根据猜测改 Router、调度阈值或量化算法。先以证据确定原始模型与替换层第一次出现数值分歧的位置。
2. **严禁在线重新量化**。BF16、`int8_g128`、`int4_g128` 的真实权重由既有离线导出流程提供；在线允许加载、解包和反量化，不能重新计算并存储另一套量化权重。
3. **BF16 等价性检查不过关时，不得宣布 S4 质量有提升**。可以继续收集诊断性结果，但必须标记 `BLOCKED_BY_BF16_PARITY`。
4. **不得把存储精度称作原生计算精度**。现有 GPU INT4 路径依然是 GPU 解包/反量化至 BF16，再进行矩阵计算。记录 `weight_storage_precision` 与 `compute_dtype/kernel` 两个字段。
5. **不要更改既有 P1–P5a.2 原始结果文件**。新结果全部落到 `results/p5a3/`，新报告在 `docs/p5a3/`。禁止用新数据回填旧 CSV、JSONL、报告。
6. **不要把 8 条 Prompt 的测试写成 32 条**，也不要把“生成匹配率”作为唯一精度指标。所有结论必须基于真实生成的 CSV/JSONL 聚合。
7. **RTX 4060 是主实验设备**，不要求 5080。限制缓存预算，优先保证 8GB 显存内稳定运行；测试出现 OOM 记录失败后缩小批次，而不是静默跳过。
8. **保留失败证据**：未通过的门禁要保存日志、配置和最小复现，并在报告中标记 `FAIL/BLOCKED`，不能改写为 `PASS`。
9. **不擅自修改远端 main**。从 main 创建实验分支；在分支中提交源码、测试、脚本、报告与小体积结果。大型模型文件、专家二进制、临时缓存不提交 Git。
10. **所有阈值需要标注来源和用途**。本计划的参考数值是初始验收目标，不是通过调参就能证明模型正确的科学定律。不能为了通过阈值而在评测集合上反复拟合策略。

---

## 1. 已知事实、待核实问题与工作假设

### 1.1 P5a.2 当前记录（供对照，不视为本轮结果）

当前 `main` 的 `docs/p5a2/P5A2_REPORT.md` / `results/p5a2/results_summary.json` 记录：

| 模式 | 平均 Decode tokens/s | 生成 Token 匹配率 | 说明 |
|---|---:|---:|---|
| 原模型 | 14.30 | 100.00% | Reference |
| Layer 0 BF16 透传 | 11.23 | **85.94%** | 需要排查 |
| Layer 0 固定 INT8 | 12.50 | 90.62% | 不能直接证明优于 BF16 |
| Layer 0 固定 INT4（GPU 解包） | 12.94 | 86.72% | 已去除旧 CPU 往返 |
| Layer 0 动态 S4 | 12.85 | 93.75% | 质量结论须待 BF16 对齐复核 |

**数据口径纠正**：旧 `run_p5a2_e2e_eval.py` 中 `load_eval_prompts()` 返回 `prompts[:8]`，因此当时是 **8 Prompt × 3 repeats × 5 modes = 120 次生成**，不是全 32 Prompt。`results_summary.json` 也写明 `num_prompts=8`。本轮必须重新执行完整评测。

### 1.2 重点审计的现有文件

在开始时以仓库实物为准，不要假定路径或 API 永远不变：

- `src/p5a/dynamic_moe_layer.py`：原始 Router 的使用、Top-k 选择、逐专家 forward、加权累加；
- `src/tiered_cache.py`：专家缓存、`forward_expert_swiglu`、解包与计算路径；
- `src/quant_codec.py`：BF16 / INT8 / INT4 的反量化、GPU 解包；
- `src/p5a/generation_runner.py`：Prefill、KV Cache、Decode、TTFT、TPOT、EOS；
- `src/p5a/measured_cost_policy.py`：S4 质量约束、实测成本和精度切换；
- `scripts/run_p5a2_e2e_eval.py`：8/32 Prompt 截断、参考输出与结果聚合；
- `configs/p5a2/eval_prompts.jsonl`：32 条输入，四类各 8 条；
- `results/p5a2/`：上一阶段原始记录。

**优先排查的候选差异，不得预先断言其就是根因**：

- 官方 `OlmoeSparseMoeBlock` 的 Router 与专家执行方式和自定义实现是否完全一致；
- `norm_topk_prob` 是否与 checkpoint config 一致，Top-k 权重是否在相同阶段转 dtype；
- `gate/up/down` 矩阵从 checkpoint 到自定义专家文件是否有 transpose、chunk、拼接、scale、dtype 或索引错误；
- `SwiGLU` 激活函数及 gate/up 对应关系；
- 专家输出的加权与 `index_add_` / 累加顺序、BF16 舍入差异；
- 缓存命中和 SSD 加载是否使用完全相同的预存权重；
- Reference 的 tokenizer、`generation_config`、EOS/Pad、KV Cache、`torch_dtype`、attention backend 和模型模式是否一致。

**重要原则**：BF16 低精度浮点运算有非结合性。即使张量值相同，不同累加顺序也可能造成有限的数值差异，并在贪婪生成时被放大。必须区分“权重映射错误”与“合法浮点舍入差异”，而不是只以 token 是否匹配作为判断。

---

## 2. 预期目录与交付物

保留现有目录，建议新增：

```text
llm_test/
├── configs/p5a3/
│   ├── default.yaml
│   └── eval_manifest.json
├── scripts/
│   ├── run_p5a3_parity.py
│   ├── run_p5a3_teacher_forced.py
│   ├── run_p5a3_generation_eval.py
│   └── aggregate_p5a3.py
├── src/p5a3/
│   ├── __init__.py
│   ├── parity_capture.py
│   ├── quality_metrics.py
│   ├── eval_runner.py
│   └── timing.py
├── tests/p5a3/
│   ├── test_weight_mapping.py
│   ├── test_bf16_parity.py
│   ├── test_generation_metrics.py
│   ├── test_eval_manifest.py
│   └── test_scheduler_integrity.py
├── results/p5a3/
│   ├── environment.json
│   ├── run_manifest.json
│   ├── parity/
│   │   ├── weight_mapping.csv
│   │   ├── router_parity.csv
│   │   ├── expert_parity.csv
│   │   ├── layer_output_parity.csv
│   │   ├── first_divergence.json
│   │   └── debug_logs/
│   ├── quality/
│   │   ├── teacher_forced_logits.csv
│   │   ├── free_generation.jsonl
│   │   ├── per_token_quality.csv
│   │   └── summary.csv
│   ├── performance/
│   │   ├── token_latency.jsonl
│   │   ├── cache_stats.jsonl
│   │   ├── memory_samples.csv
│   │   └── summary.csv
│   ├── ablation/
│   │   ├── scheduler_decisions.jsonl
│   │   └── summary.csv
│   └── plots/
└── docs/p5a3/
    ├── BF16_PARITY_FINDINGS.md
    ├── QUALITY_AND_SPEED_FINDINGS.md
    └── P5A3_REPORT.md
```

实际命名可做小幅适配，但必须有对应等价产物。每个 CSV/JSONL 要有字段说明。默认 **不提交大量逐 token 的完整 logits 数组**，只提交必要统计与摘要；需要调试时可保存局部小张量/哈希以便复现。

---

## 3. 阶段 G0：环境冻结、可复现和基线核验

### 任务

1. 检出最新 main，保存 commit SHA 和工作区状态；从该 SHA 创建 `experiment/p5a3-correctness-quality`。
2. 优先复用项目已有 Conda 环境；没有合适环境时创建新环境 `moe-p5a3`。不要升级 torch/transformers 等核心依赖而不记录原因。
3. 检查 CUDA、NVIDIA 驱动、GPU 显存、RAM、磁盘剩余空间；保存 Python、PyTorch、Transformers、Safetensors、NumPy、Triton（如果有）的版本。
4. 检查模型及 `artifacts/real_experts` 是否存在；记录模型 revision、文件 manifest、SHA256（大文件可以记录 revision + 文件名/大小，完整 checksum 选择性执行）。
5. 跑现有 smoke tests；不必重新做大型离线量化导出，除非发现专家映射不一致。

### 建议命令

```bash
cd ~/llm_test
git fetch origin
git switch main
git pull --ff-only
BASE_SHA=$(git rev-parse HEAD)
git switch -c experiment/p5a3-correctness-quality
printf 'BASE_SHA=%s\n' "$BASE_SHA"

# 确保已加载 Conda shell hook
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate moe-p5a3 2>/dev/null || conda create -y -n moe-p5a3 python=3.11
conda activate moe-p5a3

python -V
python -c 'import torch; print("torch",torch.__version__,"cuda",torch.version.cuda,"cuda_available",torch.cuda.is_available())'
nvidia-smi
free -h
df -h .
python -m pytest -q tests/p5a2 tests/p5a1  # 若这些目录存在
```

如果新建环境缺少依赖，按仓库实际 requirements/已验证版本安装；优先沿用上一阶段的 `ENVIRONMENT.md`，严禁盲目执行不受控的 `pip install -U`。

### G0 验收

- [ ] 记录 Git SHA、Conda 环境、Python/CUDA/GPU/依赖版本；
- [ ] 原模型与离线专家文件可访问，且不会因路径硬编码导致失败；
- [ ] 现有基线脚本可以启动；
- [ ] 保存 `results/p5a3/environment.json`、`run_manifest.json`。

---

## 4. 阶段 G1：离线专家权重映射与数值等价审计（最高优先级）

**不要一上来跑 480 次完整生成。先找出 BF16 透传在相同输入下为何与原模型不同。**

### G1.1 矩阵级权重对照

针对 Layer 0 至少选取 8 个专家（包含实际 Router 高频专家、低频专家和一个预先指定专家），然后扩展到所有 64 个专家：

1. 从**原模型的官方专家权重**读取该 expert 的 gate/up/down 对应部分；
2. 从 WeightStore 读取同一 `(layer_id=0, expert_id, bf16)` 的三个预存矩阵；
3. 检查 `shape`、`dtype`、转置方向、排列方式、是否存在 bias，以及 `gate/up` 的切分顺序；
4. 逐矩阵比较最大绝对误差、相对 L2、是否完全按位一致（只在布局/类型一致时有意义）；
5. 特别检查实际使用的 Transformers 版本中 OLMoE 的专家权重是否是 `gate_up_proj` 合并矩阵，以及切分顺序。

此处 **BF16 原始数据映射应该满足逐元素一致，或能够解释清楚转换过程**；若发生明显差异，先修复离线导出/映射，别继续做质量评测。

文件：`parity/weight_mapping.csv`。列至少包括 `layer_id, expert_id, matrix, orig_shape, stored_shape, dtype, exact_equal, max_abs_error, rel_l2_error, status`。

### G1.2 Router 数值一致性

在固定 hidden states 上对比官方 `orig_layer0.gate` 与替换层 `.gate`：

- `router_logits` 的最大/平均误差；
- 64 个专家的 softmax 概率差异；
- Top-k **专家 ID 和顺序** 是否完全一致；
- Top-k 的权重值、权重和以及 `norm_topk_prob` 设置是否一致；
- 是否存在 `.float()` / `.to(BF16)` 的时机差异。

**验收：相同输入、相同 Router 参数时，Top-k ID 顺序应一致**；否则先修复路由，不要调 S4。

### G1.3 单专家独立计算对照

在相同 hidden states 上比较：

- 官方参考专家执行：gate/up/down + 激活；
- 自定义 BF16 权重经 `WeightStore → TieredCache → forward_expert_swiglu` 的输出；
- 额外用高精度/FP32 诊断路径确定误差源，但不改变线上执行语义。

同时测量：最大绝对误差、相对 L2、余弦相似度、含 NaN/Inf 的数量。对多个输入规模测试：单 token、短序列、多个 token；固定随机张量 + 至少若干个真实模型捕获的 hidden states。

### G1.4 MoE 整层输出对照

固定相同 hidden states，不进行自回归，比较：

- 官方 MoE 层输出；
- 当前自定义 BF16 层输出；
- 若有差异，构造一个严格复现官方执行次序的 **reference-compatible BF16 路径**（仅用于定位与校验），特别检查 router 权重、累加 dtype、累加顺序和 `index_add_` 行为；
- 不得通过改变参考模型的权重/Tokenizer 或直接复制参考输出伪造对齐。

建议在 **层输入 → Router → 单专家输出 → 加权后输出 → MoE 聚合 → 模型 logits** 五个阶段自动保存统计，并输出**第一个显著分歧的位置**。

初始数值目标（不是可随意调宽的“通过率开关”）：

- 权重原始映射：期望 exact equal；
- Router Top-k：所有测试输入的 ID/顺序相同；
- BF16 专家/层输出：目标 `rel_l2 <= 5e-3`，且无 NaN/Inf；
- 与官方 reference 的算子差异若无法达标，必须提供解释、误差分布与“参考兼容执行路径”的对照，而不是修改指标名字。

**关键补充**：设立一个“计算顺序对照”分支，仅改变累加方式，观察误差是否显著下降；比较时 GPU 侧需同步测时，但数值准确性测试本身不需要人为依赖 token 生成的最终匹配率。

### G1 验收及阻断条件

- [ ] BF16 的 gate/up/down 权重来源、排列与数值一致；
- [ ] Router Top-k 一致；
- [ ] 单专家与整层输出差异已缩小至可解释的浮点范围；
- [ ] `first_divergence.json` 明确记录最大分歧的阶段与原因；
- [ ] 若仍存在不明的显著误差，**停止 S4 质量宣称**，转入修复任务（可继续测试功能与性能，但报告必须 `BLOCKED`）。

---

## 5. 阶段 G2：生成过程一致性与 KV Cache 核验

在完成 G1 后，才展开生成流程检查。

### 任务

1. 同一 Prompt、同一 tokenizer、相同模型 checkpoint、相同 `eval()` 模式下运行原模型与 BF16 替换层。
2. 固定 CPU/CUDA 随机种子；greedy decode，禁止采样；设置 `temperature` 等实际无关参数时不要误以为能解决数值分歧。
3. Reference 与替换模型必须使用同一种 KV Cache 模式。验证 `past_key_values` 正确传递，并确认每一步 input shape 与位置索引一致。
4. 对 **第一处生成 token 分歧** 输出该步 logits 的：Top-10 token、top-1 与 top-2 margin、参考与替换模型的各自 top-1、最大绝对差、KL / JS 差异。
5. 进行 **Teacher-forced（固定共同前缀）** 对照，避免自回归分歧后 hidden states 不再一致、无法判断误差来源。
6. 分别测试 `prefill_only`、`one_decode_step`、`full_decode_16`、`full_decode_64` 的 logits 与 KV Cache 行为。

### 两种评估必须分开

**A. Teacher-forced / same-prefix**：所有模式看到相同 reference 前缀，可比较每一步的 logits 和量化误差。是**模型数值正确性**的主要指标。

**B. Free-running greedy generation**：每个模式用自己生成的 token 继续解码，可测实际输出、速度与分歧传播。是**端到端行为**指标，不能替代 A。

记录 `prefill`/`decode` 两个阶段的差异，确保 Prompt token 不进入“生成 token 匹配率”的分母。

### G2 验收

- [ ] 原模型同设置连续运行可重复；
- [ ] Teacher-forced logits 指标可复现；
- [ ] BF16 替换层首次分歧来源已定位，或标记数值非结合性并给出充分证据；
- [ ] KV Cache 与无 Cache 参考在适当容差下相符（比较 logits；不要要求 GPU 浮点运算位级相同）；
- [ ] 日志包含第一处分歧位置及 Prompt ID。

---

## 6. 阶段 G3：完整 32 Prompt 与评估数据协议

### G3.1 正确读取评测集

修改/新建评测入口，不再使用 `prompts[:8]`。**必须检查：**

- 输入恰好 32 个唯一 `id`；
- `architecture`、`systems`、`code`、`reasoning` 四类各 8 个；
- 任务中不要把“配置文件有 32 条”误当成“实际执行了 32 条”；
- 输入顺序允许随机化，但运行 manifest 固定记录 shuffle seed 与最终顺序；
- 传参允许 `--num-prompts 8` 进行 smoke run，但**最终报告必须完整 32**；
- `--max-new-tokens 16` 用于与旧数据比较，并增加一个 64-token 的可选补充评测（资源不足可降为子集，须明确区分）。

### G3.2 配置与隔离

最低支持以下配置（可通过 YAML/CLI 实现）：

```yaml
model_id: allenai/OLMoE-1B-7B-0125-Instruct
model_dir: models/hf_original
expert_store_dir: artifacts/real_experts
sensitivity_csv: results/p45_real_sensitivity.csv
prompt_file: configs/p5a2/eval_prompts.jsonl
output_dir: results/p5a3
seed: 20261104
modes:
  - ref_unmodified
  - bf16_passthrough
  - fixed_int8
  - fixed_int4
  - s4_measured_cost
repeats: 3
max_new_tokens: 16
decode_mode: incremental_kv
vram_cache_mib: 64
ram_cache_mib: 128
schedule_quality_budget: 0.025
run_teacher_forced: true
run_free_generation: true
```

所有路径以仓库根目录相对路径为主，并允许 CLI 覆盖；旧机器的 `/home/yaoyao/...` 不得作为唯一可用路径。固定 `seed`、Prompt 列表、模式顺序、预热策略，并写入 manifest。

**缓存公平性**：每个策略拥有独立 `TieredCache` / scheduler state；给出冷启动与预热后的两组成绩。若在 Prompt 间复用缓存，必须显式说明，不得跨模式共享缓存。EOS 提前结束时记录有效生成 token 数、终止原因，不能悄悄用固定 16 当实际分母。

### G3 验收

- [ ] 32 个 Prompt × 3 次重复 × 5 个模式 = **480 条生成记录**（若某次 OOM/错误须保留失败记录；完整成功时为 480）；
- [ ] 每类 Prompt 都在各个模式下执行；
- [ ] 每条记录包含实际生成 token 数、EOS 信息、参考配置和成功/失败状态；
- [ ] 不发生跨模式缓存或调度状态串扰。

---

## 7. 阶段 G4：质量指标——从匹配率升级到 logits、KL 和困惑度

### G4.1 Teacher-forced logits（必做）

同一输入序列/前缀，测量每个模型与 BF16 Reference 的输出分布：

- `reference_top1_id` 与 `candidate_top1_id`；
- `top1_agreement`（逐位置一致率）；
- `logits_max_abs`、`logits_mean_abs`、`logits_rel_l2`；
- `KL(p_ref || p_candidate)`（以 logits 做稳定的 `log_softmax`，避免上溢；若只保留 Top-K 概率，要明确标注近似）；
- `JS_divergence`（可选）；
- `reference_top1_margin`（top-1 与 top-2 的概率/logit 间隔）；
- `candidate_ref_token_nll` 或 reference 下一 token 的 NLL 差值。

汇总时按 Prompt 类别、层/版本、Prefill/Decode、position 分组；同时报告均值、p50、p95、最大值，不能只报单个总体百分比。对于 EOS 或变长样本，明确掩码策略。

### G4.2 Free-running greedy generation（必做）

- 生成 token 逐位置匹配率与 **首个分歧位置**；
- **整段完全一致比例**；
- 生成长度、提前 EOS、停用词处理；
- 新增 token 的文本内容（可保存有限长度摘要）；
- S4 同一专家真正切换的次数，精度比例、完整选择事件与缓存命中来源。

明确标注：生成匹配率不是模型质量的普适评分，更不是 BF16 数值等价性的充分证据；不同精度的贪婪生成有时会“碰巧一致”。

### G4.3 PPL / NLL（建议做；算力或时间不足可标为未完成）

可用固定的公开/合法测试文本（独立于 S4 阈值校准数据），对 reference、BF16 替换、固定 INT8、固定 INT4 和 S4 计算相同 token 序列的平均 NLL 或困惑度；若动态 S4 的行为依赖生成轨迹，应明示 teacher-forced 模式下策略所见 Router 与缓存状态。

不得将 Prompt 问答的 token 匹配率直接叫作“准确率”或“困惑度”。

### G4 验收

- [ ] 参考与各策略的 same-prefix KL、logits 差异能够计算；
- [ ] 原模型 vs BF16 透传差异来源与程度已被清晰记录；
- [ ] 四类 Prompt 全覆盖；
- [ ] PPL 若跳过必须写明原因，报告只能对已做的指标下结论。

---

## 8. 阶段 G5：S4 混合精度消融与有效性检验

**只有 G1/G2 的 BF16 对齐问题得到解决，才把本阶段的“质量改进”标记为可信。** 否则本阶段仅作为调度行为测试。

### 必做对照模式

| 组别 | 目的 |
|---|---|
| `ref_unmodified` | 原始模型参考 |
| `bf16_passthrough` | 自定义层执行等价性 |
| `fixed_int4` | 低带宽/低驻留成本基线 |
| `fixed_int8` | 精度较高的固定量化基线 |
| `s4_measured_cost` | 真实成本感知选择（INT4/INT8/BF16） |
| `s4_no_cache_awareness` | 移除缓存驻留成本，查看贡献（可新增） |
| `s4_no_quality_constraint` | 移除质量约束，检查是否退化到固定精度（可新增） |

### 必做切换测试

1. **确定性功能测试**：对同一 `(layer, expert)` 强制 `INT4 → INT8 → INT4`，逐次检查加载版本、执行结果、日志和原始 SSD 文件来源；不能调用在线量化。
2. **真实 Router 自主选择测试**：不能使用强制模式伪装成自主切换。至少记录一组**实际由 S4 选择**的同一专家跨 token 多精度事件；若当前数据没出现，记录 `NOT_OBSERVED`，不能通过调整事件统计造出切换。
3. **INT4 ↔ INT8 单独消融** 和 **INT4 ↔ BF16 单独消融**：确定中间精度是否真的有价值，防止 S4 只在 INT4 与 BF16 间切换。
4. **质量阈值扫描**（例如 `0.010 / 0.020 / 0.025 / 0.040`）：扫描前固定配置；必须使用独立校准集合确定策略，不在最终 32 Prompt 上选最优阈值后再报告无偏质量。
5. **缓存容量扫描**（如 VRAM `32/64/128 MiB`，RAM `64/128/256 MiB`）：测量真实缓存命中、容量峰值和延迟；内存压力异常时降到安全配置并记录。

### S4 统计必须区分

- `requested_precision` vs `loaded_precision` vs `weight_storage_precision`；
- `compute_dtype`（现阶段通常是 BF16）；
- 精度按**专家调用次数**的比例、按**生成 token** 的比例；
- `same_expert_decode_switch_count`，而不是全局不同专家之间切换数；
- VRAM 目标精度命中 / RAM 目标精度命中 / SSD miss；
- Router 分数分桶、每个专家离线敏感度，以及估计质量损失；
- `scheduler_predicted_ms` 与 `observed_fetch/dequant/gemm_ms`，检查成本模型误差。

如果 `max_loss_budget` 实际只是单专家 `router_score × local_error`，报告必须称为 **“局部代理质量约束”**，不可声称是整模型质量上界。

### G5 验收

- [ ] 强制切换正确且无在线量化；
- [ ] S4 自主切换统计直接由真实事件生成；
- [ ] 三种精度的选择原因和误差/延迟成本可追溯；
- [ ] 公平缓存隔离、无泄漏、完整记录失败与 OOM；
- [ ] 如果 S4 质量优于 INT4，需同时呈现对 INT8 的代价；不得只选有利的一组比较。

---

## 9. 阶段 G6：token 级性能统计与设备资源记录

### 性能测量口径

1. **TTFT**：以本项目测量定义为准，明确是否包含 tokenizer、权重加载/冷缓存、Prompt prefill、首 token 选择。至少将 `prefill_compute_ms` 与 `first_token_total_ms` 区分出来。
2. **TPOT**：保存每一个 Decode step 的独立 wall-clock 延迟 `step_latency_ms`，最终从**全体有效 Decode steps**计算总体 p50/p95/p99；不要用每条生成“平均 TPOT”的中位数冒充 token 级 p50。
3. **Decode tokens/s**：报告 `sum(decode_steps) / sum(decode_wall_time)` 的加权总体吞吐；也可额外报告“每条 Prompt tps 均值”，名称必须不同。
4. **GPU 计时**：CPU wall-clock 前后适当调用 `torch.cuda.synchronize()`；测纯 CUDA kernel 时用 CUDA Events，区分两种指标，不混用。
5. **系统内存**：记录进程 RSS、GPU 总占用/进程峰值（运行期间采样），说明 `nvidia-smi` 采样间隔可能漏掉极短峰值；PyTorch CUDA 分配峰值作为补充。
6. **缓存与 I/O**：报告逻辑 SSD 读取字节、操作系统 Page Cache 情况（如可获取）、RAM→VRAM 字节；不要在缺少观测的情况下宣称发生了真实物理 NVMe 读取。
7. **区分冷/热缓存**，至少给出一种稳定的热缓存可重复测试；冷缓存的 OS Page Cache 难以控制时明确写为“应用级冷缓存”，不能称为物理冷盘。

### 每个 Decode step 的最小字段

```json
{
  "run_id": "...",
  "commit_sha": "...",
  "mode": "s4_measured_cost",
  "prompt_id": "sys_01",
  "repeat": 1,
  "decode_step": 0,
  "input_token_id": 42,
  "output_token_id": 123,
  "step_latency_ms": 0.0,
  "cuda_peak_allocated_bytes": 0,
  "cache_vram_used_bytes": 0,
  "cache_ram_used_bytes": 0,
  "ssd_logical_read_bytes_delta": 0,
  "expert_calls": 0,
  "precision_switches": 0
}
```

数值 `0.0` 仅是 JSON 格式例子，Agent 必须写实际测量值。

### 统计输出

- Mode 级：`TTFT mean / p50 / p95`，`TPOT p50 / p95 / p99`，整体 Decode t/s；
- Prompt 类别级：同上，加上生成分歧/质量指标；
- Cache 级：精度版本命中率、显存/RAM 峰值、SSD 逻辑读量和预取量；
- 重复测量：至少 3 次；针对核心对照计算按 Prompt 分组的 paired delta，并提供 bootstrap 95% CI（可选）。

### G6 验收

- [ ] `token_latency.jsonl` 包含每个成功 Decode step；
- [ ] p50/p95/p99 来自原始 token 级样本；
- [ ] 显存测量发生在进程执行期间；
- [ ] 真实 NVMe I/O 与 Page Cache 命中不混淆；
- [ ] 能完整复算报告表格。

---

## 10. 阶段 G7：数据质量与报告真实性自动检查

至少编写以下测试：

```bash
python -m pytest -q tests/p5a3
```

测试项：

1. `test_eval_manifest.py`：32 个 Prompt 唯一 ID、四类各 8 个；`--num-prompts` 限制与最终全量执行分离。
2. `test_weight_mapping.py`：一组固定专家 BF16 权重形状/内容可复现，确认不存在专家 ID 错位。
3. `test_bf16_parity.py`：Router Top-k 不漂移，输出差异符合已文档化容差或明确 FAIL。
4. `test_generation_metrics.py`：针对人工构造的 token 序列核对首分歧位置、长度不同、EOS、空生成、Teacher-forced KL、TPOT quantile 等边界条件。
5. `test_scheduler_integrity.py`：同一专家强制精度切换时请求/加载/实际精度相符，无在线量化、无跨模式状态污染。
6. `test_data_integrity.py`（可新增）：复算 480 条记录/失败数、每组样本数、统计口径；报告中的数值必须从已落盘数据生成。
7. `test_no_hardcoded_report.py`（可新增）：报告脚本不允许将某个模式的精度比例、速度、准确率、通过状态直接写死。

所有统计流程严格使用 `run_manifest.json` 对应的数据文件；不能混合不同 SHA、设备、预算、模型版本产生的一组数字。缺失的数据应显示 `N/A` 或 `NOT_RUN`，不能补零。

---

## 11. 阶段 G8：一键运行方式与推荐执行顺序

以 Agent 实际实现的 CLI 为准，建议提供以下入口；**这些命令是待实现的接口规范，并非声称仓库当前已有这些脚本**。

```bash
# 在 Git 仓库根目录、Conda 环境激活后

# 1. 快速代码/数据自检
python -m pytest -q tests/p5a3

# 2. 先审计 BF16 映射、Router 和 MoE 输出
python scripts/run_p5a3_parity.py --config configs/p5a3/default.yaml

# 3. G1/G2 无阻断后跑 teacher-forced logits（全 32 Prompt）
python scripts/run_p5a3_teacher_forced.py --config configs/p5a3/default.yaml

# 4. 快速 smoke：8 prompts × 1 repeat × 5 modes
python scripts/run_p5a3_generation_eval.py \
  --config configs/p5a3/default.yaml \
  --num-prompts 8 --repeats 1

# 5. 完整评测：32 prompts × 3 repeats × 5 modes
python scripts/run_p5a3_generation_eval.py \
  --config configs/p5a3/default.yaml \
  --num-prompts 32 --repeats 3

# 6. 聚合与报告生成
python scripts/aggregate_p5a3.py --input results/p5a3 --output docs/p5a3/P5A3_REPORT.md
```

**建议分批执行**：先 Layer 0 BF16/Reference parity；确认无异常后，再运行固定精度和 S4；S4 的阈值扫描在独立校准集合上做，不挤占最终 32 Prompt 的无偏评估。

### 跑满 32 Prompt 的验证脚本应包含断言

```python
expected = 32 * 3 * 5
assert len(unique_prompt_ids) == 32
assert counts_by_category == {
    "architecture": 8,
    "systems": 8,
    "code": 8,
    "reasoning": 8,
}
assert n_success + n_failure == expected
# 最终正式通过需要 n_failure == 0；否则记录 NOT_COMPLETE
```

---

## 12. 阶段 G9：报告模板与结论约束

`docs/p5a3/P5A3_REPORT.md` 至少包括：

1. 运行时环境、Conda/Python、GPU、模型版本、commit SHA、全部运行配置；
2. P5a.2 已知问题对照表（8/32 Prompt、BF16 85.94%、平均 TPOT vs token 级 p50 等）；
3. BF16 一致性审计：权重/Router/专家/层输出/logits 的对照及**首个分歧根因**；
4. 32 Prompt 完整覆盖证明、有效记录条数、类别分布和失败记录；
5. Teacher-forced logits 指标与 Free-running 指标分别成表；
6. 固定 INT4/INT8/BF16/S4 的质量—速度—内存联合对照；
7. S4 每种精度比例、同一专家实际切换事件、质量阈值与成本预测误差；
8. Token 级 TPOT、p50/p95/p99、显存/RAM 和 SSD I/O；
9. 每项 G0–G9 门禁：`PASS/FAIL/BLOCKED/SKIPPED` 和对应真实文件路径；
10. 已确认的问题、仍不确定的问题、下一阶段（P5b）的具体前提条件。

报告中禁止出现：

- 在 BF16 parity 没有通过时宣称“量化后匹配率比 BF16 高，所以质量更好”；
- 使用 `prompts[:8]` 却声称全部 32 条；
- 将存储 INT4 的 BF16 矩阵乘法称为原生 INT4 GEMM；
- 把 S4 的局部代理误差约束称为整模型误差上界；
- 用逻辑读取字节代替已测量的物理 NVMe 读取；
- 将从 5 个不同运行目录挑出的最好数字拼成一个“同次实验”表格。

---

## 13. 项目决策门禁：何时可以进入 P5b？

| Gate | 必须满足的条件 | 未通过时怎么做 |
|---|---|---|
| **G0** | 环境、仓库 SHA 与模型可复现 | 修复环境，不跑新质量评测 |
| **G1** | BF16 专家权重映射及 Router 一致，层输出误差来源得到解释 | **阻断 S4 质量结论**，优先修复模型执行 |
| **G2** | KV Cache / Decode 对照可复现，首次 logits 分歧已定位 | 修复 Runner 与 BF16 parity |
| **G3** | 正式 32 Prompt × 3 repeats × 5 modes，完整成功记录 | 保留失败报告，补齐缺失运行 |
| **G4** | Teacher-forced logits + Free-running 质量统计齐全 | 不声称模型级质量改善 |
| **G5** | S4 真正自主调度，不误把强制切换计入自主切换 | 修改事件/统计并重跑 |
| **G6** | 真 token TPOT、显存峰值、各级缓存/I/O 可复算 | 重新测量并修正指标 |
| **G7** | 自动测试与数据质量检查通过 | 修复测试/统计，不编造通过 |
| **G8/G9** | 一键复现、报告数据自动聚合 | 报告标记未完成 |

**进入 P5b（多层动态精度）建议满足的额外条件**：

- BF16 透传在真实固定输入下已达到有据可依的数值一致性，生成分歧可解释，不能有不明的错位/矩阵错误；
- INT4/INT8/S4 的误差与性能测量可信；
- 同一专家在线切换确实发生，且无需在线重新量化；
- 运行过程中 VRAM 不 OOM，缓存预算不超限；
- 所有 P5a.3 结论可由提交的脚本和原始数据复算。

**如上述条件未满足：不要进入 P5b。** 先提交本轮不完整但真实的诊断成果，再围绕失败 Gate 开下一轮修复。

---

## 14. Agent 最终提交清单

- [ ] 新分支 `experiment/p5a3-correctness-quality`，记录基线 `main` SHA；
- [ ] 修复必要的 BF16 透传数值不一致（若确认属于实现错误），并提供针对性回归测试；
- [ ] 新建/修改脚本以运行真正的 32 Prompt 全量评测；
- [ ] 所有代码可通过配置控制路径、Prompt 数量、repeats、质量预算及缓存容量；
- [ ] `tests/p5a3/` 全部通过，若有 BLOCKED 测试明确记录原因；
- [ ] `results/p5a3/` 保存本次原始实验数据、环境、SHA、事件与绘图；
- [ ] `docs/p5a3/BF16_PARITY_FINDINGS.md` 与 `docs/p5a3/P5A3_REPORT.md`；
- [ ] 表格/图由原始结果自动生成，没有任何手工写死的实验成绩；
- [ ] 明确列出未实现和不确定部分；
- [ ] 提交 Git commit，并在最终回复中报告 commit SHA、通过/失败 Gate、关键性能及可信度边界。

### 向用户汇报时建议采用下面的简明结构

```text
P5a.3 实验完成情况
- Git branch / commit:
- 环境：RTX 4060 / Ubuntu / Conda / 模型 SHA:
- BF16 parity：PASS / FAIL / BLOCKED；根因：
- 正式评测：成功 x/480，类别覆盖：
- Teacher-forced logits 误差：
- Free-running Token Match：
- S4 自主同专家精度切换次数：
- token 级 TPOT p50/p95/p99：
- 显存峰值 / RAM 峰值 / SSD 逻辑读量：
- 下一阶段 P5b 是否放行：YES / NO；原因：
```

---

## 15. 本轮最重要的一句话

> **P5a.3 的成功标准不是让 S4 的匹配率数字更漂亮，而是让“原始模型 → BF16 替换 → INT8/INT4 → S4”的数值差异与性能差异都可追溯、可解释、可复现。**

只有先建立可信的 BF16 对照，后续的多层动态精度、SSD 专家卸载与缓存调度优化，才有可靠的比较基础。
