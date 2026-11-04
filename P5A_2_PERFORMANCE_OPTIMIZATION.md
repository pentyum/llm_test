# P5a.2 实验计划：MoE 多精度推理性能优化与成本感知调度

> **交给 Coding Agent 的执行文档**  
> 版本：v1.0 · 2026-11-04  
> 代码仓库：<https://github.com/pentyum/llm_test>  
> 起始分支：`experiment/p5a1-dynamic-switch`  
> 建议工作分支：`experiment/p5a2-gpu-dequant-cost`  
> 优先设备：**Ubuntu 26.04 / RTX 4060 Laptop 8GB VRAM / 48GB RAM / NVMe SSD**  
> 目标模型：`allenai/OLMoE-1B-7B-0125-Instruct`

---

## 0. Agent 执行契约（首先阅读）

本实验的目标不是再次证明“可以动态切换权重”，而是**定位并降低 INT4 的执行开销，让动态精度调度使用真实可比较的成本指标**。必须保留 P5a.1 已验证的真实增量 Decode、同一专家 INT4↔INT8 切换能力和离线预量化权重仓库。

### 不可违反的约束

1. **禁止在线量化**：推理阶段不得调用 `quantize_int4_g128()`、`quantize_int8_g128()` 或重新从 BF16 生成低精度权重。允许将**已经离线量化**的权重在 GPU 上**反量化**为 BF16（这是本阶段主要研究对象）。
2. **性能测试必须区分**：`SSD/RAM/VRAM 取权重`、`INT4 解包`、`反量化`、`SwiGLU/GEMM`、`Router/调度`、`整 token Decode`。不得只凭一个 `decode_tps` 猜测瓶颈。
3. **权重版本和计算内核必须分开命名**：`int4_g128` 仅描述磁盘/缓存权重格式；若解码为 BF16 再调用 `torch.matmul()`，记录 `compute_kernel=bf16_matmul_after_dequant`，不得称为原生 INT4 GEMM。
4. **以 RTX 4060 的显存为硬限制**：实际 `torch.cuda.max_memory_allocated()`、`max_memory_reserved()` 与 `nvidia-smi` 进程显存都要记录；模型必须能在 8GB 设备上运行。发生 OOM 必须缩小 batch/context/cache，不能默认转移到 5080。
5. **保护历史证据**：不得覆盖 P1–P5a.1 的 CSV/日志或伪造缺失指标。新结果写入带 run_id 的目录。禁止将“未测”“跳过”或“编译失败”写为 `PASS`。
6. **不得为通过验收硬编码结果**：报告必须由本次原始 CSV/JSON、实际运行命令及环境信息生成，阈值在运行前写入配置。
7. **每项实验至少区分 warm-up 和 timed runs**；同一对照组保持模型、Prompt、精度规则、缓存预算与评测输入一致。需要随机化执行顺序，避免 GPU 频率/温度及 OS 缓存影响。
8. **先通过正确性，再优化**：每个阶段失败应记录原因、回退到最后可用提交，不得跳过失败门禁声称全阶段成功。
9. 不要改动系统级 CUDA 驱动，不要使用 `sudo` 全局清理 Linux Page Cache，不要为某项测试主动下载超出硬盘预算的大模型。

### 结果表达规范

- `VERIFIED`：存在可复现测试与证据文件；`PARTIAL`：部分运行成功但未满足门禁；`FAILED`：门禁失败；`NOT_RUN`：未执行；`BLOCKED`：客观依赖不可用。
- `Triton` 为**可选增强**，只在 PyTorch GPU 解包基线正确、可度量后尝试；如果 PyTorch GPU 路径更快，允许保留为最终实现。
- 本阶段**不要求动态 S4 一定比固定 INT8 更快**；如果最终固定 INT8 更快，应如实报告并分析何时动态调度才有价值。

---

## 1. 已知基线、已知风险与待证假设

以下数值**仅用于标记先前 P5a.1 报告的参考数据，必须在新环境重新复测**：

| 模式 | 旧报告 Decode 吞吐 | 旧报告 TPOT mean | 说明 |
|---|---:|---:|---|
| 原始模型 BF16 | 14.70 tokens/s | 68.01 ms | 不替换 MoE Layer 0 |
| BF16 替换层 | 11.17 tokens/s | 89.50 ms | 单层自定义执行器开销 |
| 固定 INT8 | 12.57 tokens/s | 79.56 ms | 已离线预量化，执行前反量化 |
| 固定 INT4 | 3.25 tokens/s | 307.93 ms | CPU NumPy 解包路径疑似显著瓶颈 |
| S4 质量预算策略 | 3.33 tokens/s | 299.85 ms | 96.97% INT4、3.03% INT8；仅报告观测到一次同专家切换 |

旧报告：<https://github.com/pentyum/llm_test/blob/experiment/p5a1-dynamic-switch/docs/P5A1_REPORT.md>

### 需要直接核查的代码位置

- `src/quant_codec.py::dequantize_int4_g128()`：目前使用 `packed_bytes.cpu().numpy()` 和 NumPy 解包，再转回 GPU；这是第一优化对象。
- `src/tiered_cache.py::forward_expert_swiglu()`：每次执行都反量化三个专家矩阵，然后做 BF16 `matmul`；应分离测量解包和矩阵计算。
- `src/p5a/dynamic_moe_layer.py::DynamicMoELayer.forward()`：对每个 token、每个 Top-k 专家串行执行并同步/取 `.item()`，存在调度及同步开销。
- `src/p5a/precision_policies.py::S4QualityBudgetPolicy`：目前成本为固定的**加载延迟估计**，没有完整考虑反量化及执行时间。
- P5a.1 测试脚本、event logger 与 `results/`：请在本地仓库**实际检索文件名**后复用，不要假定某个脚本路径存在。

### 本轮研究假设

- **H1**：把 INT4 的 CPU NumPy 解包换为纯 GPU 解包，能明显降低 INT4 专家执行延迟及真实 Decode TPOT。
- **H2**：INT4 权重更省存储/带宽，不意味着它的**当前执行路径**比 INT8 更快；基于完整执行成本的调度应避免错误选择。
- **H3**：在相同质量和缓存预算约束下，联合考虑加载、反量化与计算成本，可改善 S4 的延迟—质量权衡；如果没有改善，应分析原因。
- **H4**：小 batch（Decode 常见单 token / 小 token 批）下，Python per-expert 调用与同步开销可能独立于解包成为主要瓶颈。

---

## 2. 代码与环境准备（Gate G0）

在本地仓库运行：

```bash
set -euo pipefail
cd /path/to/llm_test
git status --short
git fetch origin
git switch experiment/p5a1-dynamic-switch
git pull --ff-only origin experiment/p5a1-dynamic-switch
git switch -c experiment/p5a2-gpu-dequant-cost
mkdir -p results/p5a2 docs/p5a2 tests/p5a2
```

如分支已存在，请切换已有分支，不要强制覆盖；如工作树非空，先保存用户改动，不得擅自删除文件。

建议新建独立 Conda 环境，并**优先继承 P5a.1 已验证的依赖版本**：

```bash
conda create -n moe-p5a2 python=3.11 -y
conda activate moe-p5a2
python -m pip install --upgrade pip
# 随后按仓库实际 pyproject.toml/environment.yml/依赖锁文件安装，
# 不存在时再参考旧环境 pip freeze 选择 PyTorch + CUDA + Transformers + Safetensors + pytest。
# 不要盲目执行 pip install --upgrade torch，避免改变已验证 CUDA 组合。
```

预检命令：

```bash
python - <<'PY'
import sys, platform, torch
print('python:', sys.version)
print('platform:', platform.platform())
print('torch:', torch.__version__, 'torch CUDA:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('compute capability:', torch.cuda.get_device_capability(0))
PY
nvidia-smi
free -h
df -h .
git rev-parse HEAD
```

预期 RTX 4060 为 Ada 架构，计算能力通常为 8.9；以**运行时输出为准**。Triton 必须先确认当前 PyTorch/CUDA/Triton 版本兼容，再决定是否安装。

**G0 验收：** 新分支创建成功、GPU 可用、P5a.1 的已有基础正确性测试能够运行，记录 `git SHA`、Conda 包版本、驱动版本、CUDA、GPU 型号、SSD 型号与空闲空间。

交付：`results/p5a2/<run_id>/environment.json`、`commands.log`、`preflight.log`。

---

## 3. P0：复测 P5a.1 并进行真正的性能分解（Gate G1）

**不要先改算法。** 先把旧吞吐结果在同设备上重新复现，确认误差范围，并增加可信的执行阶段计时。

### 测量维度（至少）

每个专家调用：

1. `policy_select_wall_ms`：精度策略计算耗时（Python wall-clock）。
2. `weight_lookup_wall_ms`：缓存查找与状态判定。
3. `ssd_load_wall_ms`：用户态从存储加载所需时间；需另报 `logical_bytes_read`，**不得自动等同物理磁盘读取**。
4. `h2d_wall_ms`、`h2d_cuda_ms`：CPU→GPU 传输（对同步与异步路径分别测量）。
5. `int4_unpack_cuda_ms`：位操作/解包，**单独**测。
6. `scale_apply_cuda_ms`：按 group 应用 scale 和转换为目标 dtype。
7. `expert_compute_cuda_ms`：gate/up/down GEMM、SiLU 与乘法。
8. `expert_e2e_wall_ms`：一次专家调用总墙钟耗时，包含 Python 调度与同步。
9. `moe_layer_e2e_ms`：整个 MoE 层（含 Top-k 个专家）耗时。

整段生成：

- `TTFT`：首次 token 产生所需时长，和预加载/Prefill 的起止事件明确区分。
- `TPOT`：后续每个生成 token 的完整实际墙钟时间，报告 mean、p50、p95、p99。
- `decode_tps`：`decode_tokens / decode_wall_seconds`；不要把 Prompt token 计入。
- `GPU peak allocated/reserved`、`RAM RSS peak`、`CPU↔GPU bytes`、预取/SSD 逻辑流量、缓存命中分层统计。

### GPU 计时要求

- CUDA microbenchmark 用 `torch.cuda.Event(enable_timing=True)`、warmup、合理重复、最终同步。
- Python 调度、文件 I/O、进程计时用 `time.perf_counter()`；不能用 CUDA Event 测磁盘 I/O。
- 不要在**生产正常路径**每个小算子后强制 `torch.cuda.synchronize()`；可在 profile mode 进行阶段分解，并单独报告同步开销。
- 对多 stream 的拷贝与执行，正确建立事件依赖；不可把异步发射时间冒充拷贝完成时间。
- 对 SSD 区分 `cold process`、`warm page cache`；Linux 文件读取走 Page Cache 时禁止称之为物理 SSD 读取。若无安全隔离手段，不必强行实现真实冷盘实验，只需明确 `NOT_VERIFIED`。

### 对照条件

- 至少比较：`ref_unmodified`、`bf16_passthrough`、`fixed_int8`、`fixed_int4`、`s4_budget`。
- 先用 P5a.1 原始 Prompt 和 token 数重跑，随后在 P5 独立长短 Prompt 集中扩展。
- 运行环境固定：同样的量化仓库、模型参数、KV Cache、cache budget、解码规则（temperature=0 / greedy）、最大新 token 数。
- **每个策略新建独立 cache**；记录冷启动与预热后两种情况，并用同样规则比较。

**G1 验收：** 新的端到端数据有原始事件级 CSV 和执行环境信息；延迟阶段覆盖无漏项；能根据 breakdown 判断 INT4 真实瓶颈。若旧数据不能复现，先报告原因并保留旧结果，不要继续做性能优劣声明。

交付：`results/p5a2/<run_id>/baseline_e2e.csv`、`expert_timing.csv`、`baseline_profile.json`、`docs/p5a2/BASELINE_PROFILE.md`。

---

## 4. P1：纯 GPU INT4 解包/反量化（Gate G2–G4）

### 核心改动

保留原始 NumPy 解包路径作为**测试参考/回退实现**；新增：

- `dequantize_int4_g128_torch_cuda(packed, scales, shape, group_size=128, target_dtype=torch.bfloat16)`（名称可调整）。
- 不调用 `.cpu()`、`.numpy()`、`torch.from_numpy()`、CPU 中转或运行时重新量化。
- 输入：GPU 上 `uint8 packed`，与当前离线 INT4_G128 完全一致的 `scales`、`shape`、`group_size`。
- 约定：**低 nibble 是第一个数，高 nibble 是第二个数**；按补码解码，`8..15 → -8..-1`；现有离线量化实际常用范围为 `[-7,7]`，但解码本身仍应正确处理 `-8`。
- 输出：与原实现一致的 shape/device/dtype，按最后一维每 128 元素使用对应 scale，数值与 CPU 参考实现一致（允许浮点舍入的严格公差）。
- 禁止在生产路径中永久缓存全部反量化 BF16 权重并把它当成“4-bit 显存占用”。

### PyTorch CUDA 解包算法提示（示意，不要求逐字照搬）

```python
# packed: cuda uint8, flattened; low nibble precedes high nibble
low = (packed & 0x0F).to(torch.int16)
high = ((packed >> 4) & 0x0F).to(torch.int16)
low = torch.where(low >= 8, low - 16, low)
high = torch.where(high >= 8, high - 16, high)
q = torch.empty((packed.numel() * 2,), device=packed.device, dtype=torch.int8)
q[0::2] = low.to(torch.int8)
q[1::2] = high.to(torch.int8)
# 后续 reshape(-1, group_size) 并乘 scales.reshape(-1,1)
# 严格遵循现有 CPU 参考的 float32 scale 计算及最终 BF16 cast 语义。
```

> 这是功能性基线，不保证最快。要测量它产生的临时张量、kernel launch 和峰值显存，再决定如何进一步融合。

### 正确性测试（必须自动化）

- 静态 nibble 编码边界：`[-8,-7,-1,0,1,7]`、低/高半字节交换、补码负值。
- 随机量化离线权重（固定 seed）和真实 OLMoE 专家三矩阵，比较 CPU 参考 vs GPU 实现。
- 多种可用 shape、group_size=128、dtype=BF16；无效 shape/group_size 应显式报错。
- 不允许 silent reshape、错位 scales 或在权重来自非目标 CUDA device 时自动拷回 CPU。
- 使用 `torch.testing.assert_close()` 对比反量化结果；建议浮点误差容忍度 `atol=0, rtol=0` 优先尝试，若 GPU 算子顺序导致不完全相同，报告最大绝对误差和相对误差后设定严格的、事先记录的公差。
- 专家级输出 `SwiGLU(x)` 的相对 L2 和余弦相似度与现有 CPU 解包版本一致；对 BF16 参考只要求量化预期误差，不要求完全相同。
- 一个调用链上的负面断言：线上 GPU 解包函数不调用 `Tensor.cpu()`、`numpy()`；可以通过 monkeypatch / profiler 辅助验证。

**G2（正确性）**：上述测试全部通过；`INT4 → INT8 → INT4` 功能回归测试保持通过。

### 纯解包性能基准

针对真实专家的 gate/up/down 三矩阵，至少各测：

- 原 CPU NumPy 解包 + 回 GPU（旧路径）的**端到端 wall time**。
- PyTorch GPU 解包 + scale 的 CUDA Event 时长、端到端 wall time、临时显存峰值。
- 相同次数的 BF16 `matmul`，与解包耗时明确分离。
- 20 次 warmup + 至少 100 次 timed calls（若 OOM 或非常慢，明确记录改用次数与原因）。

注意：CPU 旧路径的耗时不得只用 GPU Event 计时，否则会漏掉 CPU/PCIe 往返。

**G3（性能可解释性）**：记录 GPU 新路径对旧路径的提升比例，同时保留绝对时长和显存峰值；若新路径更慢，正确性仍通过但标记 `PERFORMANCE_NOT_IMPROVED`，分析 kernel 启动、临时张量与同步成本，不得虚报加速。

### 可选 Triton 优化（P1b）

在 G2 通过、PyTorch GPU 版已有可信 profile 后，可尝试单 kernel 融合：

- 一个 CUDA/Triton kernel 负责 packed nibble → signed int4 → group-wise scale → BF16 写回。
- 对 8GB 4060 优先考虑流式 tile/分块输出，避免为完整专家同时分配过多 BF16 临时 buffer。
- 对每种 shape 验证结果与 PyTorch CUDA 参考一致；评估首次 JIT 编译时间与稳定运行时间，前者不计入 steady-state TPOT，但要报告部署成本。
- 不要引入当前 GPU 不支持的架构特性；不要求 tensor core 原生 INT4 GEMM。

**G4（可选）**：Triton 正确且优于 PyTorch GPU 时纳入候选；否则选择更简单稳定的 PyTorch GPU 版，原因记录在报告中。

交付建议：`tests/p5a2/test_int4_unpack.py`、`scripts/bench_p5a2_dequant.py`（若目录不存在，创建）、`results/p5a2/<run_id>/dequant_bench.csv`、`docs/p5a2/DEQUANT_FINDINGS.md`。

---

## 5. P2：把新解包接入单层真实 OLMoE（Gate G5）

1. 在 `TieredCache.forward_expert_swiglu()`（或抽象出的 Executor）增加明确可配置后端：
   - `legacy_numpy`：原路径，仅作诊断；
   - `torch_cuda`：新增正式路径；
   - `triton`：仅在 G4 通过时启用。
2. 保留三精度数据格式读取：BF16 原生权重、INT8_G128、INT4_G128，不要更改离线权重的结构/scale 定义。
3. 给执行事件添加 `storage_precision`、`compute_kernel`、`dequant_backend`、`unpack_ms`、`scale_ms`、`gemm_ms`、`expert_total_ms`，别再把 `meta['precision']` 当作内核精度。
4. 使用完整模型、**只替换 Layer 0**，先固定 INT4 测试，再运行固定 INT8、BF16 passthrough。
5. BF16 passthrough 与官方未替换模型应继续在固定 greedy 测试中生成相同 token；如测试中出现偏差，先检查 Router、Top-k 归一化、dtype、shape 与 KV Cache。
6. 使用相同 Prompt、cache 设置和解码参数，比较旧 `legacy_numpy` 与新 `torch_cuda` 的真实 Decode TPOT；如速度没有改善，输出 profiler 的剩余耗时构成。
7. 不应将 PyTorch CPU fallback 误判成 CUDA 优化成功；记录每个 Tensor 设备与调度决策。

**G5 验收：** 新后端能完成真实增量 KV Decode，结果与旧 INT4 后端一致或满足明确的数值容忍度；强制切换和 S4 动态切换回归测试通过；延迟原始事件能定位 INT4 路径的主要开销。

---

## 6. P3：实测成本驱动的 S4 调度器（Gate G6–G8）

### 6.1 先收集真实成本，而不是直接改权重

在当前机器、固定 batch/序列长度与真实专家矩阵形状下，为每个精度测以下值：

| 维度 | 必测候选 |
|---|---|
| `precision` | `int4_g128`, `int8_g128`, `bf16` |
| `cache_tier` | `VRAM`, `RAM`, `SSD`（SSD 读取另分 page cache 状态） |
| `dequant_backend` | `torch_cuda`; 可用时增加 `triton` |
| `cost_components` | fetch / unpack / scale / GEMM / Python overhead / end-to-end |
| `load_bytes` | 实际缓存字节数、权重字节数 |
| `shape` | 三矩阵形状与目标批大小 |

生成 `cost_profile.json`，包含版本、机器、基准配置、观测样本数、p50/p95 与异常值处理规则。**不要在重跑前无条件重用旧机器/旧版本的成本模型**。

### 6.2 定义调度目标

保留 `S4QualityBudgetPolicy` 旧策略以便对照。新增 `S4MeasuredCostPolicy`，在每次需要执行某专家时：

1. 按当前 `(layer_id, expert_id)` 和 Router 分数获取 `estimated_loss(precision)`，使用 P4.5 真实敏感度映射；缺失条目使用可解释的 fallback 并计数，不得静默使用专家 ID 取模。
2. 在质量约束 `estimated_loss ≤ max_loss_budget` 下筛选候选精度；未通过的精度不得仅因快而被选中。
3. 查询当前缓存层级（只是状态窥视，不得改变 LRU 次序）并预测对应版本的实际加载代价。
4. 预测完整 `T_expected = T_fetch + T_unpack + T_scale + T_compute + T_scheduler`。注意只有不重叠的成本可以直接相加；如果引入异步预取，单独表示 `T_unhidden`，避免重复计算。
5. 在有效候选中选预测成本最低者；代价接近时可加入 hysteresis（切换惩罚）避免反复抖动，但必须公开其参数和副作用。
6. 如高精度版本未缓存，允许策略决定加载或保持低精度，前提是**满足当前质量约束**。不得悄悄退化到低于质量底线的版本。
7. 缓存预算是硬约束；`max_memory_allocated` 与 `max_memory_reserved` 的高水位独立报告。

### 6.3 策略消融与公平对照

至少对照：

- `Fixed-INT4`（存储少，但新算子是否够快需要实际测）；
- `Fixed-INT8`（速度基线）；
- `S4-Legacy`（原无约束 S4）；
- `S4-QualityBudget`（P5a.1 版本）；
- `S4-MeasuredCost`（本轮新增）；
- `Oracle-trace-cost`（**可选离线分析**：在已记录路由与完备成本下算理论调度下界，不能伪称在线可实现）。

对比时应：

- 相同 Prompt、相同模型权重、相同质量预算、相同缓存上限、相同 Decode 步数；
- 对于状态相关策略，另报从冷缓存开始与预热缓存开始的结果；
- 显示专家级 precision 使用占比（按调用次数与按调用权重两种口径）、实际精度切换次数及每次切换前后缓存位置；
- 将策略选择开销本身计入 TPOT；
- 不要求新策略一定出现 `INT4 ↔ INT8` 交替：若真实工作负载下只选一种，应如实分析，另以确定性测试保证功能仍然支持切换。

**G6（模型可信）**：实测成本配置完整且未硬编码结果；策略能正确读取缓存状态和对应成本。  
**G7（质量约束）**：所有在线选择均满足配置的误差约束或明确记录 fallback/例外，测试给定阈值下正确选择高精度。  
**G8（实际效果）**：有完整策略对照图；无论新策略是否提速，能解释取舍，不得仅以 INT4 使用比例或命中率代替生成质量结论。

交付：`src/p5a/measured_cost_policy.py`（示例名）、`tests/p5a2/test_measured_policy.py`、`results/p5a2/<run_id>/cost_profile.json`、`scheduler_ablation.csv`。

---

## 7. P4：多 Prompt 模型质量与端到端性能（Gate G9–G11）

### 7.1 评测数据

至少覆盖三类 Prompt，各 **≥10 条**（合计 ≥30 条，内容和哈希随报告保存）：

1. 中文/英文问答与简短推理；
2. 长文本续写或信息抽取；
3. 包含重复指令或长上下文的稳定性测试。

每类保留短 / 中 / 长输入长度差异；如果 OOM 或运行太慢，先做 3 条 Smoke Test，再分批执行完整集，不得把 Smoke Test 当最终结果。

生成参数统一固定为 `temperature=0` / greedy、相同最大新增 token 数、相同 EOS/停止规则。明确单请求 batch=1 测试只说明单请求性能，不代表多用户吞吐。

### 7.2 正确性与质量

- 只计算**生成 token 区间**的 token match，排除 Prompt tokens；同时统计第一次分歧位置、序列长度差异。
- 保留原始 `logits`（或至少对应生成步的 logits 差异）抽样：Top-1/Top-5 一致率、KL 或概率差异、相对误差；注意 BF16 logits 很接近但生成仍可分叉。
- 在独立小规模固定文本评测集上统计 teacher-forced NLL / perplexity（若任务格式允许）。
- 对真实 Layer 0 专家输出统计相对 L2 误差/余弦相似度；与模型级质量指标明确分开。
- 不得将 100% token match 解释为“无损量化”或“总体质量完全相同”。

### 7.3 性能测量

- 每条 Prompt 独立记录 `TTFT, TPOT_mean, TPOT_p50, TPOT_p95, TPOT_p99, decode_tps`、完整样本数；总量报告给出聚合方式（按 token / 按请求），不混用。
- 重复运行至少 3 次正式测量，计时前预热；报告均值、标准差/区间与跨 run 变化。
- 捕获 VRAM（allocated/reserved/进程总占用）、CPU RSS peak、SSD 逻辑读取流量、Page Cache 说明、cache hit by tier、精度切换次数。
- 如显存紧张，先关闭非必要服务，降低 cache budget 或 context 长度；保留降级原因，不使用 5080“代替”4060 报告主结果。

### 7.4 建议生成图表

- `INT4 旧解包 vs GPU 解包`：专家执行分解堆叠图（仅解释性展示，包含重叠 I/O 时不可直接相加）；
- `Fixed INT8 vs INT4 vs S4`：每 token TPOT 分布箱线图；
- `quality vs TPOT`：散点或 Pareto 图；
- `cache budget vs VRAM hit / SSD traffic / TPOT`；
- `precision use over decode steps`：按 token 展示精度选择轨迹。

**G9（质量）**：保留可复现测试集、生成答案和 logits/NLL 结果，不只报告 token-match。  
**G10（性能）**：端到端实测含至少 30 条 Prompt、至少 3 个正式重复 run，并包含冷/热缓存状态说明。  
**G11（诚实结论）**：如果 GPU 解包提速但 S4 仍慢于固定 INT8，报告写明原因和下一阶段优化方向，而不是更改对照组或缩短无利样本。

---

## 8. P5：诊断性实验（可选，不得代替主结果）

### 8.1 BF16 解码权重缓存

仅作诊断，用有限预算缓存已反量化的 BF16 `gate/up/down`。目的是判断当前瓶颈是否来自重复反量化，而不是作为最终“低显存 4-bit”方案。

务必分开记录：

- 量化存储字节、反量化缓存的 BF16 字节、临时工作区字节；
- 真实 peak VRAM；
- 在相同总显存预算下是否还能放入足够专家。

若 BF16 解码缓存使 TPOT 大幅下降，但显存占用失控，结论是**需要融合低比特执行核**，而不是声称多精度方案全面加速。

### 8.2 Python 调度与同步消融

- profile `selected_experts[..., rank].item()` 和 `routing_weights[..., rank].item()` 的 GPU→CPU 同步损失；
- 比较串行 Python for-loop 与适度批处理/分组执行；
- 统计 torch kernel launch 次数、每专家 `.synchronize()` 次数；
- 不破坏原生 Top-k 路由语义及专家权重加权；
- 批量优化失败时保留基线版本。

### 8.3 真正原生 INT4/INT8 GEMM（仅探索）

如果 GPU 解包已不再是主要瓶颈，仍需要显著提速，可设计单 token / 小 batch 的 weight-only quantized GEMV/GEMM kernel；先进行最小独立 prototype。**本轮不要求交付，必须与 BF16 matmul-after-dequant 的结果和显存开销分开报告。**

---

## 9. 测试与验收总表

| 门禁 | 必须完成 | 核心证据 | 未通过时的动作 |
|---|---|---|---|
| G0 环境 | 是 | `environment.json`, Git SHA, smoke test | 修复环境、冻结依赖 |
| G1 性能基线 | 是 | baseline CSV + profile | 不得直接声称提速 |
| G2 GPU INT4 正确性 | 是 | nibble / scales / shape / expert tests | 回退新后端 |
| G3 GPU 解包测量 | 是 | CPU 旧路径与 GPU 新路径公平对照 | 报告未改善并诊断 |
| G4 Triton | 否 | Triton 测试 + profile | 不影响主计划 |
| G5 真实单层 Decode | 是 | 生成 + 强制切换回归 + 事件日志 | 暂停调度优化 |
| G6 实测成本 | 是 | `cost_profile.json` | 不启用新策略 |
| G7 质量约束 | 是 | policy unit tests | 修复成本/阈值逻辑 |
| G8 策略消融 | 是 | `scheduler_ablation.csv` | 不得声称更优 |
| G9 质量评估 | 是 | prompts + logits/NLL + outputs | 不得推断质量无损 |
| G10 E2E 性能 | 是 | 30+ prompts, 3+ runs, TPOT/TTFT | 标记 PARTIAL |
| G11 真实报告 | 是 | 自动生成、值与 CSV 相符 | 修复报告生成 |

**“P5a.2 已完成”**要求 G0–G3、G5–G11 均通过。G4 可跳过。是否形成**性能加速**必须单独给出，不是完成门禁的先决条件。

### 硬性回归测试

- BF16 passthrough 单层与官方参考的确定性输出一致（固定环境条件）。
- Forced `INT4→INT8→INT4` 仍然成功，且来源是离线权重文件，无在线量化。
- 自动 S4 能够正常运行，事件记录精度与实际加载文件一致。
- Cache 容量在所有插入、预取、驱逐路径保持 `used ≤ budget`；零预算不缓存。
- 对比模式的 cache 实例隔离，不共享策略运行结果。
- 仅本阶段新增的 GPU 反量化实现需要 GPU；CPU 旧实现作为测试参考保留。
- `torch.cuda.max_memory_allocated`/reserved、`nvidia-smi` 进程显存同时保存，未发生 OOM 才算运行完成。

---

## 10. 推荐目录结构与交付物

可以根据仓库实际结构微调，但必须保留稳定的 CSV 字段与可复现入口：

```text
llm_test/
├── src/
│   ├── quant_codec.py                 # 保留 CPU 参考，增加 GPU 反量化
│   ├── tiered_cache.py                # 可配置反量化后端和细分事件
│   └── p5a/
│       ├── dynamic_moe_layer.py       # 事件时序与内核元数据
│       ├── precision_policies.py      # 保留已有策略
│       └── measured_cost_policy.py    # 新策略（建议）
├── scripts/
│   ├── bench_p5a2_dequant.py          # 新增微基准（建议）
│   ├── profile_p5a2_e2e.py            # 新增端到端 profile（建议）
│   └── eval_p5a2_policies.py          # 新增策略消融（建议）
├── tests/p5a2/
│   ├── test_int4_unpack.py
│   ├── test_expert_forward.py
│   ├── test_measured_policy.py
│   └── test_dynamic_switch_regression.py
├── configs/p5a2/
│   ├── bench.yaml
│   ├── policies.yaml
│   └── eval_prompts.jsonl
├── results/p5a2/<run_id>/
│   ├── environment.json
│   ├── commands.log
│   ├── baseline_e2e.csv
│   ├── expert_timing.csv
│   ├── dequant_bench.csv
│   ├── cost_profile.json
│   ├── scheduler_ablation.csv
│   ├── generation_details.jsonl
│   ├── quality_metrics.csv
│   ├── results_summary.json
│   └── plots/
└── docs/p5a2/
    ├── BASELINE_PROFILE.md
    ├── DEQUANT_FINDINGS.md
    └── P5A2_REPORT.md
```

**注意**：上面部分文件是“建议新增”，而非声称当前分支已存在。Agent 应先探索仓库，使用已有模块和入口；不能为了匹配目录树而重复造轮子。

### CSV 最低字段协议

每次真实专家调用至少记录：

```text
run_id,git_sha,mode,phase,layer_id,expert_id,decode_step,
router_score,storage_precision,requested_precision,loaded_precision,
compute_kernel,dequant_backend,cache_hit_tier,
fetch_ms,unpack_ms,scale_ms,gemm_ms,expert_e2e_wall_ms,
logical_ssd_read_bytes,h2d_bytes,ram_cache_bytes,vram_cache_bytes
```

每个 Decode token 至少记录：

```text
run_id,prompt_id,mode,repeat_id,decode_step,tpot_ms,
precision_switches,expert_calls,allocated_mib,reserved_mib,
process_gpu_memory_mib,quality_budget,cache_state
```

**必须提供字段数据字典**；`null` 明确表示未测，`0` 表示确实观测为 0，不能混用。

---

## 11. Agent 推荐执行顺序

```text
[ ] 读取分支和先前报告；冻结环境并完成 G0
[ ] 原样复跑 P5a.1；新增 profiling，完成 G1
[ ] 为 INT4 nibble/scale/shape 写测试，先让旧版参考通过
[ ] 实现 PyTorch 纯 GPU unpack + scale，完成 G2
[ ] 做 100+ 次 GPU microbench 和 wall-time 对照，完成 G3
[ ] [可选] 研究 Triton 融合，实现并检验 G4
[ ] 接入 DynamicMoELayer 单层真实 KV Decode，完成 G5
[ ] 测量 cache-tier × precision 的真实成本，完成 G6
[ ] 实现质量约束的 measured-cost S4 并单测，完成 G7
[ ] 所有 baseline 与新 S4 公平消融，完成 G8
[ ] 30+ Prompt、独立质量和 3+ 次重复测量，完成 G9/G10
[ ] 自动汇总结果，生成真实 P5A2_REPORT.md，完成 G11
[ ] git commit；在新分支推送；给出 commit SHA 与可复现命令
```

执行每一个任务时遵循小步提交策略：功能正确性提交、性能实验提交、调度器提交、报告提交。若性能优化导致正确性回归，优先回退该优化并保留测量数据。

---

## 12. 最终报告必须回答的 8 个问题

1. 旧 INT4 路径究竟花多少时间在 CPU NumPy 解包、数据往返、GPU 反量化、GEMM、Python 调度与同步？
2. 纯 GPU PyTorch 反量化的输出与旧路径差异有多大？解包加速多少，临时显存增加多少？
3. 在真实 Decode 中，固定 INT4 的 TPOT 与吞吐率改善多少？是否仍比固定 INT8 慢？
4. 如固定 INT4 仍慢于 INT8，主要瓶颈转移到了哪里？
5. 加入完整执行成本后，新 S4 选择 INT4/INT8/BF16 的比例与切换次数如何变化？
6. 在相同质量阈值和缓存预算下，新 S4 的 TPOT、SSD 流量、VRAM 与质量表现，相较旧 S4 和固定 INT8 有何差异？
7. 与原始模型相比，生成 token 的一致率、logits 差异、NLL/PPL 是否出现可解释的质量退化？
8. 下一阶段最有价值的是 **融合低比特 GEMV/GEMM**、**多层专家执行与分层卸载**，还是 **预取/调度优化**？必须基于实测瓶颈给出选择。

### 结论模板（必须由实际数据填入，不得提前宣告成功）

```text
GPU_INT4_UNPACK_CORRECT: VERIFIED / PARTIAL / FAILED
GPU_INT4_UNPACK_SPEEDUP: <实测值或 NOT_VERIFIED>
INT4_REAL_DECODE_TPOT_CHANGE: <实测值或 NOT_VERIFIED>
DYNAMIC_SWITCH_REGRESSION: VERIFIED / FAILED
MEASURED_COST_POLICY_CORRECT: VERIFIED / FAILED
S4_BETTER_THAN_FIXED_INT8_AT_SAME_QUALITY: YES / NO / NOT_VERIFIED
8GB_VRAM_FEASIBLE: YES / NO / NOT_VERIFIED
NEXT_RESEARCH_DIRECTION: <以数据论证的选择>
```

---

## 13. 参考与边界

本计划基于项目 `experiment/p5a1-dynamic-switch` 的已提交代码、报告以及用户前几轮的实验目标，主要检查对象：

- <https://raw.githubusercontent.com/pentyum/llm_test/experiment/p5a1-dynamic-switch/src/quant_codec.py>
- <https://raw.githubusercontent.com/pentyum/llm_test/experiment/p5a1-dynamic-switch/src/tiered_cache.py>
- <https://raw.githubusercontent.com/pentyum/llm_test/experiment/p5a1-dynamic-switch/src/p5a/dynamic_moe_layer.py>
- <https://raw.githubusercontent.com/pentyum/llm_test/experiment/p5a1-dynamic-switch/src/p5a/precision_policies.py>
- <https://raw.githubusercontent.com/pentyum/llm_test/experiment/p5a1-dynamic-switch/docs/P5A1_REPORT.md>
- <https://pytorch.org/get-started/locally/>
- <https://developer.nvidia.com/cuda/gpus>

**重要边界：** 本轮核心目标为“预量化权重的 GPU 解包/反量化 + 实测成本调度”，并不是完全消除 BF16 中间张量的原生低比特 GPU GEMM。若要继续追求 INT4 在推理速度上的优势，应由本轮 profiling 决定是否投入更复杂的融合量化算子开发。
