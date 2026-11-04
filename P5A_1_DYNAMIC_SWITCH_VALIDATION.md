# P5a.1 实验执行计划：真实 MoE 专家动态精度切换验证

> **交付对象：Coding Agent**  
> **目标仓库**：<https://github.com/pentyum/llm_test>  
> **起始分支**：`experiment/p4-5-validation`  
> **建议工作分支**：`experiment/p5a1-dynamic-switch`  
> **主要设备**：Ubuntu · RTX 4060 Laptop 8GB VRAM · 48GB RAM · NVMe SSD  
> **模型**：`allenai/OLMoE-1B-7B-0125-Instruct`  
> **阶段定位**：P5a 的真实性、正确性与计时修复；**不是**宣称性能超越 llama.cpp 的优化阶段。

---

## 0. Agent 的执行契约（必须遵守）

1. **先检查实际代码、配置、数据和当前 Git 状态，再修改。** 不要假定 README 与分支代码完全同步。记录 `git rev-parse HEAD` 和环境信息。
2. 在 `experiment/p4-5-validation` 上新建分支；不要覆盖 P1–P5a 的原始 CSV、日志、报告，也不要对现有模型文件执行破坏性操作。
3. **禁止在线权重量化。** 在线推理允许从 SSD 读取已离线生成的 BF16/INT8_G128/INT4_G128 权重、反量化、执行 GEMM；禁止调用 `quantize_int4_g128`、`quantize_int8_g128` 等在线量化入口。把“反量化执行 BF16 GEMM”和“原生 INT4/INT8 GEMM”明确区分。
4. 不要为了让 S4 看起来成功而伪造路由分数、伪造精度比例、人工修改实验 CSV 或将强制切换测试包装成 S4 自动决策结果。
5. 不允许把一个仅在 **Prefill** 中切换的案例报告为“跨 Decode token 切换”；必须用全局调用编号和生成 token 位置识别。
6. 任何测试失败，保留原始失败记录并写明原因；允许交付部分结果，**不允许硬编码 PASS 或成功结论**。
7. 对所有数字写明口径（均值/p50/p95、逻辑读取/物理 SSD 读取、请求精度/实际执行精度、加载延迟/计算延迟、CPU 或 GPU 执行）。
8. 单卡 8GB VRAM 为硬性约束。优先复用已有原始模型和专家存储，不重复下载大文件。避免同时驻留整模型的 BF16 GPU 权重。
9. 不要改动系统级配置、清除全系统 page cache、使用 `sudo`，除非用户明确批准。不能测物理冷读时如实标记为 `page-cache-uncontrolled`。
10. 完成后提交代码、测试、可追溯的测量数据、报告；每条结论都应能从提交的 JSONL/CSV 中计算得到。

## 1. 本次实验为什么必须做

已知 `experiment/p4-5-validation` 分支中存在以下需要验证/修正的点（具体实现以当前 checkout 为准）：

- `scripts/run_p5a.py` 的 `dynamic_s4` 实测精度选择比例为 **INT4 100%、INT8 0%**，因此尚未证明 S4 在实际生成中进行了双精度切换。
- 该脚本把 `bf16_passthrough`、`fixed_int8`、`fixed_int4`、`dynamic_s4` 放在**同一个缓存实例**中顺序运行；策略间存在状态污染。
- 解码循环反复执行 `model(generated_ids)`，属于全前缀重算，不能将其报告为使用 KV Cache 的标准增量 Decode 速度。
- 精度事件日志只有每次 forward 内的 `token_idx`，没有可靠的全局解码步编号；不能证明“同一专家在不同生成 token 上切换”。
- 现有 P5a 报告的动态切换文字为预先写死，与 CSV 不一致。
- 现有 Layer 0 的 INT4 解包可能通过 CPU NumPy 路径，然后以 BF16 GEMM 执行；`tokens/s` 应标注为**功能性实现的真实生成吞吐**，不得宣称原生 INT4 CUDA 内核收益。

**本阶段的核心研究问题**：

> 在不进行在线量化的前提下，能否证明同一个 `(layer_id, expert_id)` 在不同自回归 Decode 步，从 SSD/RAM/VRAM 中得到事先存储的不同精度版本，并执行正确的专家计算？其后，S4 是否能够在不作弊、不改变原始路由的情况下，自主触发这样的切换，且其质量—延迟权衡可测？

## 2. 分阶段目标与门槛

| 阶段 | 内容 | 必须交付 | 通过条件 |
|---|---|---|---|
| A | 环境与基线冻结 | `docs/P5A1_ENV.md`、代码版本、数据清单 | 原 P5a 可复查，历史文件未覆盖 |
| B | 事件与指标可信化 | 精度事件 JSONL、统计工具、单测 | `requested/loaded/executed` 一致，报告源于数据 |
| C | **确定性切换功能验证** | 同一专家跨调用 INT4→INT8→INT4 的测试 | 每步执行正确；无在线量化 |
| D | **S4 自主切换验证** | 新策略、消融、真实 Decode 事件 | 至少 1 个同一专家跨 Decode 步出现 2 种实际执行精度；否则标记未通过 |
| E | 独立缓存与增量 Decode | 独立模式运行、KV Cache 增量验证 | 与全前缀参考输出数值/Token 一致性达标 |
| F | 公平性能与质量评估 | CSV/JSONL、图、报告、tests | 有重复、口径完整，不声称未经验证的收益 |

> **严格区分 C 与 D**：C 可以通过“测试专用强制精度策略”证明系统可切换；D 必须由**正式 S4**根据真实 Router 分数、真实敏感度及缓存状态自主选择。仅通过 C 不构成 D 的通过。

## 3. Phase A：检查代码、数据与运行环境

### A.1 安全拉取并新建分支

```bash
cd /path/to/llm_test
git status --short
git fetch origin experiment/p4-5-validation
git switch experiment/p4-5-validation
git pull --ff-only origin experiment/p4-5-validation
git switch -c experiment/p5a1-dynamic-switch

git rev-parse HEAD
nvidia-smi
free -h
df -h .
conda env list
```

若目录有未提交的用户修改，**先停止覆盖操作并向用户报告**；不得强制 `reset --hard` 或 `git clean -fdx`。若工作分支已存在，复用或创建不冲突的分支名。

### A.2 Python 环境

优先复用现有可复现 Conda 环境。如果缺失，单独新建：

```bash
conda create -n moe-p5a1 python=3.11 -y
conda activate moe-p5a1
python --version
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

优先依据仓库实际 `requirements*.txt`、`environment*.yml` 或已有安装脚本安装依赖；**不可无依据地覆盖已验证的 CUDA/PyTorch 组合**。锁定最终版本到 `docs/P5A1_ENV.md`，保存 `pip freeze`、`nvidia-smi`、Git commit。离线或依赖失败时先报告环境缺口。

### A.3 数据清单

读取并验证存在性、数量、文件哈希或可识别元数据：

- 原始 Hugging Face 模型：`models/hf_original`（如本地路径不同，使用 CLI 参数或环境变量）。
- 多精度专家目录：`artifacts/real_experts`（至少 Layer 0、各精度对应版本）。
- 真实 Router 轨迹：`artifacts/traces/trace_real_router.jsonl`（若存在）。
- 敏感度数据：`results/p45_real_sensitivity.csv`（含 Layer 0 相关专家）。
- 历史 P5a：`results/p5a_e2e.csv` 和 `docs/P5A_REPORT.md`。

如果部分专家只有单一精度版本，必须登记缺口，不得在 Online 阶段补做量化。允许单独调用 **offline** 导出脚本产生缺失版本，并在独立日志中说明输入来源与生成命令。

## 4. Phase B：可信事件系统与统计 API

### B.1 事件最小 schema

每次实际使用一个专家，写入一条 JSONL。建议每条包含：

```json
{
  "run_id": "example",
  "mode": "dynamic_s4",
  "phase": "decode",
  "forward_id": 18,
  "generated_token_step": 17,
  "token_idx_in_forward": 0,
  "layer_id": 0,
  "expert_id": 12,
  "topk_rank": 3,
  "router_score": 0.093,
  "requested_precision": "int8_g128",
  "loaded_precision": "int8_g128",
  "executed_precision": "int8_g128",
  "hit_tier": "RAM",
  "cache_key": "0:12:int8_g128:1.0",
  "fetch_ms": 0.15,
  "dequant_ms": 0.07,
  "matmul_ms": 0.09,
  "source_weight_sha256": "..."
}
```

以上只是 **字段模板，不是实验数据**。哈希可以引用 manifest 中事先计算的哈希，避免每步重新计算。没有单独测量的时间写 `null`，不能填 0 冒充已测；某些精度格式只有反量化实现，也必须正确记录 `executed_precision` 的语义，例如另增 `compute_dtype="bf16"`、`kernel_type="dequant_then_bf16_gemm"`，而非误称原生 INT4 GEMM。

注意 `forward_id` 应在每次自回归 model forward 调用时全局递增；`generated_token_step` 只有 Decode 阶段才有意义。Prefill 包含多个 prompt token，必须与 Decode 独立统计。

### B.2 报告统计从事件派生

至少实现：

- `precision_request_share_pct` 与 `precision_execute_share_pct`（分开）；
- `unique_experts_per_precision`；
- `same_expert_decode_switch_count`：同一 `(layer_id,expert_id)` 在**不同 generated_token_step** 的已执行精度不同；
- `switch_transitions`：INT4→INT8、INT8→INT4 等；
- `vram_target_hit_pct`、`ram_target_hit_pct`、`ssd_miss_pct`，分母严格一致；
- `ssd_logical_read_mib`（由 WeightStore 记录的文件长度，不等于物理 NVMe 流量）；
- 按进程统计的 `read_bytes`（若可用）作为 OS 层近似物理读取指标，并注明 page cache、后台程序和文件系统限制；
- 真正完整的 `fetch/dequant/compute` 时间以及端到端 `TTFT`、增量 `TPOT`。

代码不能写死“INT4 60%、INT8 40%”或“动态切换成功”。生成报告必须检查计数，**若切换次数为零，输出 `NOT VERIFIED`**。

### B.3 关键单元测试

新增 `tests/p5a1/test_event_schema.py` 与 `tests/p5a1/test_report_consistency.py`，验证：

- 多次 forward 后 `forward_id` 单调增长，不因 `clear_events()` 重置错误的全局计数；
- Prefill 的 `token_idx=0` 不会与 Decode 第 0 步混淆；
- 同一专家仅在 Prefill 选到不同精度，不计作跨 Decode 切换；
- 事件中缺少时间字段时写 `null`，不是虚假的零延迟；
- 从合成事件计算得到的精度比例、转换次数与报告一致；
- `requested_precision != executed_precision` 时必须记录原因（例如 fallback），不得静默吞掉。

## 5. Phase C：先独立证明“同一专家能切换”

这一阶段可以使用**测试专用策略**，但不得写入 S4 正式策略中。

### C.1 测试专用 `ForcedSequencePolicy`

在 `src/p5a/` 下提供一个可注入策略（名字可调整）：

```python
# 伪代码：不得作为实际实验结果
policy = ForcedSequencePolicy({
    (0, 12): ["int4_g128", "int8_g128", "int4_g128"]
})
```

对同一专家及相同固定输入张量重复执行 3 次；每次从预存权重版本加载或者命中缓存，不能实时执行权重量化。先单独调用专家计算路径，再执行 `DynamicMoELayer` 层级测试。确保 `requested_precision` 与实际加载的文件、`executed_precision` 一致。

### C.2 数值与内存验收

- INT4/INT8 解码后权重或专家输出，与**离线生成的对应版本**数值一致；选 BF16 作为误差参考。
- 同精度重复执行的结果应稳定（按 CUDA/CPU、dtype 设定合理的 `atol/rtol`），不同精度不要求输出完全相同。
- 输出 shape、dtype、device 正确；没有 NaN/Inf。
- 0 预算和小于单个专家大小时，缓存不越界：`ram_used <= ram_budget`、`vram_used <= vram_budget`。
- 有明确证据证明本阶段没有调用任何在线量化入口：利用测试 monkeypatch 将 `quantize_*` 替换为抛异常的函数，在在线路径中调用即失败。
- 记录每种精度的实际 tensor 文件路径、manifest 精度和可选 checksum。

**C 通过标准**：测试用例实际记录同一 `(layer_id,expert_id)` 的 `INT4 → INT8 → INT4`，三个版本参与计算并分别通过精度检查；这是*功能验证*，不是 S4 算法验证。

## 6. Phase D：让 S4 真正自主选择精度

### D.1 不要靠改变测试数据造出结果

先诊断 S4 成本：对一批**真实 Decode Router 事件**输出每个候选精度的：

- `router_score`；
- `estimated_error`（敏感度表给出的离线局部误差）；
- `quality_cost`；
- `load_cost`（SSD / RAM / VRAM）；
- `total_cost`、候选合法性及最终选择。

检查为何现有 `Cost = 12 × score × error + load_cost` 总选 INT4。注意不同成本维度可能单位不统一：一个是无量纲误差，另一个是硬编码的相对加载成本。不能仅为获得混合比例而随意调常数。

### D.2 推荐最小策略：质量约束 + 真实成本

保留原始 S4 作为 `S4_legacy` 比较；新增一个可配置的 `S4_quality_budget`，在每次候选选择时执行：

1. 使用离线**实测**的 `(layer_id, expert_id, precision)` 敏感度；缺失敏感度时采用明确的保守 fallback（如 INT8 或 BF16），不得使用 `expert_id % N` 冒充目标专家误差。
2. 用 Router 权重及专家误差估算一个局部损失代理 `estimated_loss`，注明它**不是模型级真实精度损失**。
3. 过滤掉超过质量阈值 `max_estimated_loss` 的候选版本。
4. 在允许的候选中，按预计完整完成时间（缓存命中、数据搬运、反量化、计算）选择；初期可用离线校准的延迟表，不要给“0/0.2/1/4”打上物理毫秒的标签。
5. 若全部候选不满足约束，选择更高精度保底或显式返回不可行错误；不能偷偷降级到 INT4。
6. 将 `max_estimated_loss` 设为 CLI 参数，在验证集上跑**预先定义**的多个预算（如 `0.01 / 0.03 / 0.06`），并说明数值只是探索网格、不是预期最优值。

为避免频繁换入换出，可加入最小驻留时间/切换滞回，但必须做消融，且不能因此掩盖完全没有切换的结果。

### D.3 两层验收

**D1（决策层）**：针对真实采集 Router 分数与敏感度，至少一个真实专家在不同事件中自主产生两种精度选择。记录完整成本分解。若未出现，报告未达标；不能修改 Router 值或者用强制策略冒充。

**D2（真实 Decode 层）**：在一条或多条真实 OLMoE 贪婪解码过程中，同一个 `(layer_id, expert_id)` 在**不同 Decode 步**实际执行了两种精度，并且所有事件能追溯到合法离线权重。优先用多条自然 Prompt 增加重访概率；若仍没有切换，报告 `D2 NOT VERIFIED`，而不是人为注入伪造真实生成事件。

允许 S4 的某些参数配置没有任何切换；这些情况也是重要结论，必须完整记录参数值及命中率。不要设置硬性“必须达到 40% INT8”之类比例指标。

## 7. Phase E：公平模式比较与正确的 KV Cache 解码

### E.1 每种模式独立缓存

`ref_unmodified`、`bf16_passthrough`、`fixed_int8`、`fixed_int4`、`S4_legacy`、`S4_quality_budget` 应使用**独立的缓存对象、计数器和模型包装状态**；不要在同一 `TieredCache` 只清掉两个 stats 字段就当成冷启动。避免上一模式为下一模式预热。

- **冷策略运行**：新建缓存对象，清空项目进程持有的 RAM/VRAM 专家缓存；操作系统 Page Cache 不受控，必须注明。
- **热策略运行**：用事先说明的固定序列进行预热，**只在预热后重置统计**；注意预热时不要泄漏到测量结果。
- 每种模式保持相同 Prompt、生成长度、温度、随机种子、模型权重、线程数和预算。
- 至少 3 次重复，建议 5 次；随机化策略运行顺序，同时保留原始逐次结果。
- 对于 4060 的 8GB VRAM，设置独立**专家缓存预算**（例如 32/64/128 MiB 探索值），并记录总进程显存峰值与 CUDA allocator 的实际峰值；不可把缓存预算当成总显存占用。

### E.2 修正解码路径

当前 `model(generated_ids)` 每个 token 全前缀重算。实现支持 `past_key_values`/`use_cache=True` 的**增量解码**：

```python
# 示意：实际调用需匹配已锁定 transformers 版本
# Prefill: outputs = model(input_ids, use_cache=True, ...)
# Decode: outputs = model(last_token_id, past_key_values=..., use_cache=True, ...)
```

保持 `attention_mask` 和位置编码处理与 Hugging Face 模型实现一致。先做**一致性测试**：同一模型、同一 Prompt，增量 KV Cache 与完整前缀重算在前 N 步的 next-token logits（有误差容忍）和 greedy token 序列一致或说明不一致原因。不要把两个模式混合比较。

不支持时保留 `full_prefix_recompute` 独立模式，报告 `KV Cache 未验证`，不得继续宣称标准 TPOT。

### E.3 正确计时与质量指标

- 将 Prefill 首 token 耗时独立标为 `TTFT`。
- Decode 用每次真实增量 forward 的完整墙钟延迟统计 TPOT mean/p50/p95/p99 和 tokens/s；GPU 计时前后做适当 `torch.cuda.synchronize()`，避免计到仅提交 kernel 的时间。
- Separate `fetch_ms`、`dequant_ms`、`expert_compute_ms`（涉及 GPU 的子阶段用 CUDA Event 或明确同步）。
- `torch.cuda.max_memory_allocated()` 和 `max_memory_reserved()` 在单次测量开始前按需重置；补充进程级显存与系统内存峰值，不能使用 GPU 总容量冒充实测占用。
- 计算 token match 只比较 **生成部分**，不把 Prompt token 纳入分母；比较时分母是生成 token 个数，长度不同要明确处理。
- 增加至少一项分布层面质量指标（如同前缀的 next-token logits KL/Top-1 agreement），可再测小批语料 perplexity；只把生成文本“通顺”作为定性现象，不当作精度充分验证。
- 对比过程中把参考模型的 KV/模型设备配置固定，报告“其余 15 层在 CPU、自定义专家在 GPU/CPU 的功能验证路径”是否仍成立。

## 8. Phase F：实验矩阵与统计要求

### F.1 推荐最小实验矩阵

| 组别 | 策略 | Trace / 推理 | 缓存 | 目的 |
|---|---|---|---|---|
| T0 | 固定 INT4 | 单专家固定输入 | 冷、热 | 数据与数值基线 |
| T1 | 固定 INT8 | 单专家固定输入 | 冷、热 | 数据与数值基线 |
| T2 | ForcedSequencePolicy | 同一专家调用 3 次 | 独立 | 功能性 INT4→INT8→INT4 |
| T3 | S4_legacy | 真实 Router 事件 | 独立冷、热 | 说明旧 S4 为何倾向 INT4 |
| T4 | S4_quality_budget | 真实 Router 事件 | 独立冷、热 | 自主选择的成本—误差权衡 |
| T5 | 固定 BF16/INT8/INT4 | 真实增量 Decode | 独立 | 模型级正确性与性能基准 |
| T6 | S4_legacy/S4_quality_budget | 真实增量 Decode | 独立 | 是否出现真正跨 Decode 动态切换 |

若算力、时间或磁盘压力较大，先完成 T0–T4，**如实标记 T5–T6 未完成**，不得将 P4.5 Trace 回放数据伪装成端到端性能。为减少初期费用，可以先取 3 条以上覆盖不同长度的 Prompt、每条 16～32 个新 token，再扩展到不少于 10 条 Prompt；优先保证测量可信度。

### F.2 统计标准

每条模式/预算记录：

- 硬件型号、驱动、CUDA/PyTorch/Transformers 版本、commit、数据目录和权重格式版本；
- `run_id`、Prompt ID、seed、冷/热标签、实际生成 token 数；
- 请求与执行精度占比、同专家跨 Decode 切换次数、Router 分数分布；
- VRAM/RAM 命中率、淘汰次数、SSD 逻辑读量与可获得的物理 I/O 读量；
- TTFT、TPOT mean/p50/p95/p99、tokens/s、运行总墙钟、CPU/RAM/VRAM 峰值；
- 相对 BF16 的数值误差代理、next-token logit 差异、生成 token 匹配率；
- 重复测试均值与分散度，不要把一次偶然最优数值写成确定性收益。

**关于“优化成功”的定义**：S4 只要证明了正确的动态精度执行，就完成**功能目标**；是否比固定 INT4/INT8 更快或质量更高，是独立的性能结论，必须由公平测试支持。

## 9. 建议的代码改动范围（可按实际项目微调）

```text
src/p5a/
  dynamic_moe_layer.py         # 事件与真实执行精度记录、全局 step、策略注入
  precision_policies.py       # ForcedSequencePolicy、S4_legacy、S4_quality_budget
  generation_runner.py        # 使用 KV Cache 的统一生成器
  metrics.py                  # JSONL → CSV / 报告统计

scripts/
  run_p5a1.py                # 总入口：分组独立运行、冷热策略、可配置参数
  verify_p5a1_artifacts.py  # 检查报告指标和事件文件一致

tests/p5a1/
  test_forced_switch.py
  test_no_online_quantization.py
  test_event_schema.py
  test_cache_isolation.py
  test_incremental_decode.py
  test_report_consistency.py

results/p5a1/
  run_manifest.json
  switch_events_*.jsonl
  decision_trace_*.jsonl
  policy_comparison.csv
  generation_metrics.csv
  validation_summary.json
  figures/*.png

docs/
  P5A1_ENV.md
  P5A1_REPORT.md
```

若仓库已有同功能代码，复用已有模块，不要为满足目录示例而重复造轮子。`results/p5a1` 可以只提交小规模脱敏统计和示例事件，原始大体量 Trace 若受 Git 体积限制则保存本地并给出哈希与复现实例；不得提交模型权重、密码或敏感 Prompt。

## 10. 最小 CLI 规范（Agent 自己实现）

希望能够用以下**约定接口**执行；命令属于开发目标，并非当前仓库已存在的功能：

```bash
# 单元测试，不依赖大模型（除被标记为 integration 的测试）
PYTHONPATH=. pytest -q tests/p5a1

# 功能测试：强制切换，验证加载/执行精度
python scripts/run_p5a1.py \
  --phase forced-switch \
  --device cuda \
  --output-dir results/p5a1/forced

# 决策回放：真实 Router 事件，严格区分强制和自动决策
python scripts/run_p5a1.py \
  --phase policy-replay \
  --policies s4_legacy s4_quality_budget \
  --output-dir results/p5a1/replay

# 真实增量推理：各模式独立缓存
python scripts/run_p5a1.py \
  --phase generation \
  --policies ref_unmodified fixed_int4 fixed_int8 s4_legacy s4_quality_budget \
  --decode-mode incremental_kv \
  --repeats 3 \
  --output-dir results/p5a1/generation

# 验证报告不是硬编码或混用其他 run 的数据
python scripts/verify_p5a1_artifacts.py --input-dir results/p5a1
```

必须支持 `--model-dir`、`--expert-store`、`--sensitivity-csv` 以及 `--trace-file` 等可配置参数；所有默认相对路径从仓库根目录推导，不要写死 `/home/yaoyao/llm_test`。离线缺模型时给出清楚错误消息，不要未经批准自动下载几十 GB。

## 11. 明确的验收检查清单

以下各项在 `results/p5a1/validation_summary.json` 中都要有 `status: PASS|FAIL|NOT_VERIFIED|SKIPPED`、证据文件、失败原因：

- [ ] **G1**：`git commit`、Conda 环境、GPU 信息、输入文件版本已记录。
- [ ] **G2**：所有在线路径禁止运行时权重量化；测试通过。
- [ ] **G3**：确定性强制切换 `INT4→INT8→INT4` 成功，载入的是对应预量化版本，专家输出正确。
- [ ] **G4**：报告从原始事件重新计算；与 CSV 中精度比例、切换次数、缓存命中率一致。
- [ ] **G5**：S4 的决策层至少在真实 Router 事件上自主给出两种精度；否则记录 `NOT_VERIFIED`。
- [ ] **G6**：真实自回归 **Decode** 发生过同一专家跨 token 的两种**实际执行**精度；否则记录 `NOT_VERIFIED`（不得用 G3 代替）。
- [ ] **G7**：每个模式缓存隔离，零预算不会产生命中且不会超容量；冷热指标分别输出。
- [ ] **G8**：增量 KV Cache 解码已对照全前缀计算验证；否则报告只允许 `full_prefix_recompute` 标签。
- [ ] **G9**：所有 CPU/GPU 计时完成必要同步；TTFT/TPOT 与子阶段指标口径明确。
- [ ] **G10**：原始 P1–P5a 数据保持不变，`pytest` 通过，代码已提交实验分支。

**本轮最低可接受交付**：G1–G4、G7、G10 全部 PASS，G5/G6/G8/G9 明确 PASS 或留下可复现的失败/限制；不许“跳过但写成功”。

**进入 P5b 的门槛**：G1–G10 均达到有效 PASS（G5/G6 特别重要），并在独立冷/热缓存实验下得到可信性能数据。即便动态推理更慢，也可以进入性能优化，但报告必须如实标记“动态切换正确、加速尚未实现”。

## 12. 失败与排查优先级

1. **S4 仍 100% INT4**：先看真实 Router 分数分布和候选成本，检查质量阈值是否从未生效；不要强行设定预期精度比例。
2. **找不到同一专家跨 Decode 的重访**：换用自然 Prompt、增加合理的生成步数，统计专家重访次数；不要将 Prefill 事件混进 Decode；可让 G3 先通过，G6 记为未验证。
3. **GPU OOM**：缩小专家缓存预算/Prompt 长度/并发，模型其余部分保持 CPU；记录峰值与 OOM 环境；不得隐式改变模型精度。
4. **KV Cache 与全前缀输出不一致**：检查 attention mask、位置编码、past length、输出 logits 与是否为贪婪解码；在修复前不报告标准 Decode TPOT。
5. **INT4 很慢**：单独分析 NumPy 解包、CPU↔GPU 传输和反量化 GEMM；本阶段不强制开发 CUDA/Triton 原生 INT4 内核。
6. **SSD 读取量不一致**：区分 `WeightStore` 逻辑文件字节与 `/proc/<pid>/io` 实际 `read_bytes`，后者可能因 Page Cache 很小。不可把逻辑读取量直接称为物理 SSD 实读。
7. **缓存预取线程导致异常**：要求异常上报、运行结束 drain、任务去重和可靠时间戳；不要悄悄吞掉异常。
8. **运行时间过长**：先完成快速的功能与决策测试，再增加端到端 Prompt 数和重复次数，逐步扩展；在报告中写清采样规模。

## 13. 最终提交格式

Agent 完成后，最终回复必须提供：

1. 新分支名、最新 commit SHA、变更文件列表。
2. 实际执行的命令及测试结果（如 `NN passed, MM skipped`）；未运行命令也需注明。
3. `results/p5a1/validation_summary.json` 中 G1–G10 状态表。
4. **最重要的两条独立结论**：`ForcedSwitch verified?`、`RealDecodeS4Switch verified?`。
5. 对照表：固定 INT4、固定 INT8、S4_legacy、新 S4 的真实执行精度比例、跨 Decode 切换次数、TTFT、TPOT、SSD 逻辑读量、缓存命中率和生成质量。
6. `docs/P5A1_REPORT.md` 的相对路径与公开分支链接；失败时列出真实 blocker 和复现步骤。
7. 下一阶段建议必须依据本轮结果，不可预设“动态精度已经加速”。

---

## 附录：建议报告的判定语句

**当仅 G3 成功时：**

> 已在测试用相同专家上验证预存 INT4/INT8 权重的实际切换与计算正确性；尚未验证 S4 在真实自回归 Decode 中自主触发跨 token 精度切换。

**当 G6 成功但速度下降时：**

> 已验证真实 OLMoE 解码中同一专家跨生成 token 发生实际 INT4/INT8 切换，在线不重新量化；当前实现由于权重搬运/反量化/计算开销，尚未证明端到端加速。

**当 G6 与性能质量对照均支持优势时：**

> 已在指定硬件、Prompt 集、缓存预算和测试口径下观察到动态调度的性能—质量改善；该结论仅适用于所测试的配置，尚不能直接外推到更大规模 MoE。

**本计划的成功定义是“证据可信”，不是“指标必须好看”。**
