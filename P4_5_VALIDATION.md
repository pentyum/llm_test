# P4.5 验证修复与 P5a 单层动态精度集成实验计划

> **执行对象**：Coding Agent（Ubuntu / Conda / NVIDIA CUDA）  
> **代码仓库**：https://github.com/pentyum/llm_test  
> **主实验平台**：RTX 4060 Laptop 8 GiB VRAM / 48 GB RAM / NVMe SSD  
> **模型**：`allenai/OLMoE-1B-7B-0125-Instruct`  
> **优先级**：P4.5 必做；P5a 仅在 P4.5 门禁通过后执行  
> **实验原则**：离线生成 BF16、INT8_G128、INT4_G128 权重；**在线不得重新量化权重**。在线解码低比特权重为 BF16 是可允许的兼容路径，但必须单独计时，不得声称是原生 INT4/INT8 GEMM。

## 0. Agent 的执行契约（必须遵守）

1. **先读仓库，后改代码**：阅读 `README.md`、`ENVIRONMENT.md`、`DECISIONS.md`、`REPORT.md`、`src/tiered_cache.py`、`src/precision_scheduler.py`、`src/quant_codec.py`、`src/replay.py`、`src/export_real_experts.py`、`scripts/run_baseline.py` 以及现有测试。若文件或接口已变化，以实际代码为准，在 `docs/P4_5_FINDINGS.md` 中注明差异。
2. **留存可复现基线**：记录 `git rev-parse HEAD`，在新分支 `experiment/p4-5-validation` 工作；不覆盖 `results/p1_*`、`results/p2_*`、`results/p3_*`、`results/p4_*` 和已有 `REPORT.md`。建议先对原始 CSV 做 SHA256 并保存到 `results/p45_baseline_manifest.json`。
3. **不虚构结果**：无法运行的项目必须写明 `SKIPPED`、原因和已尝试的命令；不得把模拟路由说成真实路由，不得把文件缓存命中说成物理 SSD 读取，不得把访问耗时说成端到端推理耗时。
4. **资源约束**：只使用 RTX 4060 主机；默认保留至少 1.5 GiB VRAM 给模型/上下文/框架，CPU RAM 避免接近 48 GB；避免触发 Swap。遇到 CUDA OOM 时缩小 batch、序列长度或工作集，不强行搬入全模型。
5. **依赖管理**：优先复用现有 Conda 环境 `moe-precision-lab`；如需升级会破坏 P1–P4 的 PyTorch/CUDA 环境，另建 `moe-precision-lab-p45`。禁止无说明地覆盖现有环境。记录 `conda list --explicit` 与 `pip freeze`。
6. **先测试再压测**：每项代码修复必须先有单元测试。运行五次独立重复的性能实验，每次设置确定的种子，记录预热轮次和运行顺序。对比算法必须回放同一访问轨迹、相同预算和相同机器负载条件。
7. **结果可追溯**：所有结果必须关联 commit SHA、模型 revision、量化版本、系统环境、脚本命令、实验种子、trace hash、配置文件、计时语义及容量单位。
8. **保护现有成果**：不要删除 P1–P4 源码，不提交 GGUF、safetensors、量化权重、原始大型 trace 或缓存文件；忽略生成的模型文件和大型日志。
9. **权限与风险**：不要调用 `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` 影响全系统；不随意清理用户目录、重分区或覆盖其他 NVMe 数据。确需 `O_DIRECT` / `fio`，只使用本项目新建的测试文件。
10. **失败即报告**：P4.5 核心正确性门禁未通过时不继续 P5a，不得通过放宽断言或移除错误样本掩盖问题。

## 1. 研究目标与当前疑点

### 1.1 已有成果

现有 P1–P4 已建立：固定精度的 OLMoE/llama.cpp 推理基线、预量化专家文件、SSD→RAM→VRAM 分层缓存、INT4/INT8/BF16 精度调度，以及局部专家误差评估。

### 1.2 必须复核的具体问题

| ID | 问题/待证实假设 | 原因 | 关键修复 |
|---|---|---|---|
| B1 | 0 字节预算下仍可能插入缓存项 | `_evict()` 驱逐后未必拒绝超预算对象 | 入库前容量检查、旁路返回及单测 |
| B2 | `target_precision_hits` 混合了 RAM/VRAM 命中 | 难以判断真正的 GPU 常驻率 | 分别记录同精度 VRAM / RAM 命中 |
| B3 | `prefetch()` 重复读取、错误吞掉或漏计 I/O | 可能虚高策略收益 | 去重、生命周期、异常传播及统一计费 |
| B4 | P1 显存指标可能在子进程退出后采样 | 进程峰值与总 GPU 使用量混淆 | 运行中按 PID 采样与峰值统计 |
| B5 | `mean_latency_ms` 被误写为 p50 | 均值与中位数不是同一指标 | 明确输出 mean/p50/p95/p99 |
| B6 | P4 工作集只有少量真实专家，路由分数为合成 | 很多 INT4 专家可完全放入 RAM | 真实 Router trace + 跨层工作集 |
| B7 | 量化输出先解码为 BF16 再执行 GEMM | 不能推断原生量化内核的速度 | 分离 load / decode / copy / GEMM 时间 |
| B8 | P4 原始 CSV 与报告部分数字不一致 | 结果口径和版本不明 | 按统一元数据重新生成独立报告 |
| B9 | 缓存记账 bytes 可能只是序列化文件大小 | VRAM 的 tensors 实际分配量不同 | 分开记录逻辑文件大小、张量实际字节、allocator 峰值 |

> 注：B1–B8 是现有审查指出的问题或高风险点，**Agent 必须用当前代码和测试逐条确认**；B9 是新增加的测量要求。不能仅根据本计划断言每个问题仍存在。

## 2. 验收目标（严格区分正确性与性能）

### P4.5 核心正确性门禁（全部通过才能进入 P5a）

- [ ] C1：每次缓存操作后 `ram_accounted_bytes <= ram_budget_bytes` 且 `vram_accounted_bytes <= vram_budget_bytes`；`budget=0` 时绝不能插入；对象比预算大时必须旁路，不允许溢出。
- [ ] C2：相同 `(layer_id, expert_id, precision, format_version)` 的多个并发预取请求最多触发一个真实加载；预取失败能被观测、记录并回退处理；不死锁。
- [ ] C3：访问命中指标被拆分为 `vram_target_hit`、`ram_target_hit`、`any_precision_hit`；分别给出请求分母及数值，禁止重复计算。
- [ ] C4：所有同步 H2D/解码/GEMM 计时在对应阶段的 CUDA 工作完成后才停止；另有单独的异步提交耗时字段。
- [ ] C5：预取触发的读量包含在总读量中；`bytes_from_store`（逻辑流量）与 `physical_disk_read_bytes`（设备实际流量，若可测）明确区分。
- [ ] C6：真实 Router trace 来源可追溯到实际模型 forward，包含准确的 layer/expert/top-k/权重/请求与 token 位置，且可以在不重跑模型时稳定回放。
- [ ] C7：同一 trace/预算下 S0/S1/S2/S3/S4 能完成运行，生成 CSV、JSON、图；指标能由原始事件日志重新聚合得到。
- [ ] C8：`pytest` 全绿（既有测试 + 新增测试）；新报告忠实反映当前原始数据，不能将 mean 当 p50、不能将访问延迟标为 token 延迟。

### 性能与科学性门禁（不是预设成功结论）

- 在相同 VRAM/RAM 预算、相同 trace、相同计算后端的条件下，评估 S4 是否处于质量–延迟–I/O 的非支配前沿；**允许结论是“不优于固定 INT4/INT8”**。
- 结果给出 mean/p50/p95/p99、标准差或 95% 置信区间，至少 5 次重复；展示各次原始结果而非只挑选最好的一次。
- 冷启动、warm Page Cache、RAM hit、VRAM hit 必须独立报告。若无法可靠测得物理 SSD I/O，应标 `N/A` 并说明原因。

## 3. 阶段 A：仓库检查、环境固定和旧结果审计

### A1. 记录运行环境与代码状态

在仓库根目录执行（目录名称以实际为准）：

```bash
cd ~/llm_test

git status --short
git rev-parse HEAD
nvidia-smi
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
free -h
conda activate moe-precision-lab
python --version
python -c "import torch; print('torch',torch.__version__,'CUDA',torch.version.cuda,'available',torch.cuda.is_available())"
python -m pip freeze > logs/p45_pip_freeze.txt
conda list --explicit > logs/p45_conda_explicit.txt
```

首次运行前确保 `logs/` 存在；若已存在未提交修改，先确认修改意图并保留，不可粗暴执行 `git reset --hard`。

### A2. 旧结果一致性核对

1. 检查 `REPORT.md` 与 `results/p4_policy_comparison.csv` 的 `mean_latency_ms`、`p95_latency_ms`、`ssd_read_mb`、`avg_weighted_error` 是否一致。
2. 明确旧表中的 `MB` 实际是十进制 MB 还是 MiB；代码若除以 `1024**2`，应标 `MiB`。
3. 对旧结果写入审计 CSV `results/p45_legacy_audit.csv`：`metric,report_value,csv_value,unit,match,reason`。
4. 旧报告不直接覆盖；在 `docs/P4_5_FINDINGS.md` 增加“历史结果不能直接对比的原因”。

**阶段 A 产物**：环境快照、基线清单、旧结果对账记录。  
**进入 B 的条件**：仓库状态与源数据已固定，测试环境可运行。

## 4. 阶段 B：修复三级缓存的正确性与可观测性

### B1. 容量与对象生命周期

修改 `src/tiered_cache.py`：

- 插入 RAM/VRAM 前，先计算该版本实际需要缓存的字节；对象 `size > budget` 或 `budget == 0` 时**不缓存**。
- 驱逐后仍无法容纳的对象必须走临时加载路径，不插入缓存；临时 Tensor 的生命周期应覆盖本次计算，使用后能释放。
- 任何操作完成后均满足 `used_bytes <= budget_bytes`；同时记录由框架/allocator 产生但不计入逻辑缓存的显存额外开销。
- 如果实际缓存 GPU tensor 为多个张量，记账使用所有 Tensor 的 `numel() * element_size()` 合计（若存在共享 storage，避免重复计）；磁盘 file bytes 另存。
- `get_expert()` 返回的计时和来源必须反映真实路径：VRAM / RAM / SSD；不能因为某版本在 RAM 就算作 VRAM 命中。
- 根据需要增加 `get_expert(..., allow_fallback)` 的来源/实际精度元信息，保证下游知道实际执行精度。

建议统一数据结构（字段名可调整，但含义不得丢失）：

```python
CacheKey = tuple[int, int, str, str]  # layer, expert, precision, format_version

# 每个访问事件至少记录：
# requested_precision, actual_precision, hit_tier,
# bypass_reason, logical_bytes, physical_read_bytes_if_available,
# t_lookup_ms, t_store_load_ms, t_h2d_ms,
# t_decode_ms, t_gemm_ms, t_end_to_end_ms
```

### B2. 预取 I/O 统计、去重和并发

- 一个 `(CacheKey, target_tier)` 对应一个 in-flight future/state，重复预取不得重复读取。
- 状态至少包含 `queued -> loading -> ready / failed / cancelled`。
- **不要持有全局缓存锁执行 SSD 读取、CUDA 拷贝或长耗时 `synchronize()`**。仅在检查状态、提交、发布新缓存项及更新记账时持锁。
- 预取和前台加载共享统一的 `store_read_count` / `store_read_bytes` 计费接口；只计一次物理读调用。
- 前台访问命中正在加载中的同版本时，应按可配置策略等待该 future 或执行受控回退，避免二次 I/O。
- 不要用 `except Exception: pass`；记录异常、重试上限、降级路径并在测试中暴露异常。
- 程序退出前可选择 drain/cancel 预取任务，不能遗留后台线程或中断输出的结果统计。

### B3. 新增测试（必须覆盖）

在 `tests/` 添加对应单测/集成测试：

1. `budget=0`，连续 100 次不同专家访问后 `ram_used == vram_used == 0`，缓存命中为零（不考虑 OS Page Cache）。
2. 专家对象大于缓存预算时旁路，不出现负计数或溢出。
3. 恰好等于预算、多个对象驱逐、相同专家不同精度并存、同精度重复请求。
4. 预取与普通访问同时请求同一专家，真实存储加载次数应为一次（允许仅首次/竞态受控路径；若设计不能保证单次必须清晰声明并修复）。
5. 预取时模拟文件损坏或 I/O 异常，任务失败能被上层感知且缓存状态可恢复。
6. 多线程随机并发访问 1000 次，不死锁、不负计数、最终使用量不超预算。
7. 使用 `cuda.synchronize` 验证异步 H2D 完成前后访问行为正确；不能返回未就绪的权重给消费者。

**阶段 B 产物**：核心缓存修复、测试、`results/p45_cache_correctness.json`。  
**进入 C 的条件**：全部缓存与预取正确性测试通过。

## 5. 阶段 C：重新建立可解释的 I/O 和 GPU 微基准

### C1. 四种路径分别测量

| 场景 | 计时范围 | 必须保留的区分 |
|---|---|---|
| VRAM 命中 | 索引查找、可选本地 kernel 执行 | `lookup_only` 与 `lookup+compute` |
| RAM 命中 | pinned/pageable RAM → VRAM | H2D submit vs complete、拷贝量 |
| Page Cache 命中 | 文件 API 从 Linux 缓存得到数据 | 不可叫做物理 SSD 冷读 |
| NVMe 读取 | 存储设备向 RAM 提供数据 | 逻辑读取、实测设备读取与调度延迟 |

- GPU 测量使用 CUDA Events 计 GPU 工作耗时，host 侧使用 `time.perf_counter_ns()`，必要时在计时边界同步；不同工具的数字不能直接混为一谈。
- 显存峰值同时记录 `torch.cuda.max_memory_allocated/reserved`（若使用 PyTorch）及进程运行中 NVML/`nvidia-smi` 采样。标清设备总占用、当前进程显存和 PyTorch allocator 数值之间的区别。
- 对文件系统缓存：分别测 `warm` 和 `attempted_cold`；`posix_fadvise(DONTNEED)` 仅代表冷读尝试，并非绝对保证物理读取。确需直接 I/O 时可用单独 `fio --direct=1` 基准，并满足 I/O 对齐与文件系统限制；不得将 `fio` 设备能力测试替代真实加载器的实测。
- 单次专家文件 3–12 MiB，应额外报告打开文件、解析 safetensors、创建张量以及 memcpy 的细分开销。
- 比较 pinned 与 pageable RAM，但仅将**已实测**结果纳入结论；不假定 `non_blocking=True` 总能异步重叠。

### C2. 新增统一结果字段

至少输出以下字段（可拆成事件日志 + 聚合表）：

```text
run_id, timestamp_utc, commit_sha, model_revision, device_name,
trace_sha256, seed, strategy, precision_format, workload_kind,
ram_budget_mib, vram_budget_mib, access_count,
mean_access_ms, p50_access_ms, p95_access_ms, p99_access_ms,
mean_h2d_complete_ms, mean_decode_ms, mean_gemm_ms,
vram_target_hits, ram_target_hits, any_precision_hits,
ssd_store_read_count, ssd_logical_read_mib,
physical_disk_read_mib, prefetch_store_read_mib,
ram_peak_mib, vram_logical_peak_mib,
nvml_process_vram_peak_mib, torch_reserved_peak_mib,
quality_proxy_error, repeat_index
```

未测或不适用写为 `null`/`NaN`，**不填 0**。定义每个字段的开始、结束、同步点、单位及分母，写进 `docs/METRICS_DEFINITIONS.md`。

### C3. 重复试验规范

- 预热至少 100 次访问，随后每组测量至少 1000 次（如果真实 trace 较短则多次无损循环并保留边界说明）。
- 每组至少 5 次独立重复，种子固定且输出每次的 CSV。
- S0–S4 的顺序轮换或随机打乱，避免热机、温度和缓存状态系统性偏向某一策略。
- 分别测量**冷启动**与**稳态**；报告稳定阶段时必须明确是否从空缓存起跑、预热是否计入统计。
- 观察温度、功耗限制、GPU 时钟、CPU/SSD 背景负载、swap in/out；若有干扰，记录而非偷偷删去。

**阶段 C 产物**：`results/p45_io_microbench.csv`、`results/p45_gpu_microbench.csv`、`results/p45_latency_distributions.png` 和测量定义文档。  
**进入 D 的条件**：每个性能指标均有明确时间边界和对应验证测试。

## 6. 阶段 D：真实 OLMoE Router trace 与量化敏感度

### D1. 收集真实推理路由

以 OLMoE 的真实前向传播收集 `layer_id`、`expert_id`、`router_score`、`topk_rank`、`request_id`、`sequence_id`、`token_index`、`phase`（prefill/decode）等字段。

- 优先查阅模型实际 `transformers` 实现是否支持返回 `router_logits` / `output_router_logits`；若接口不同，通过最小侵入 hook 捕获 Router 的真实 Top-k，**不要猜测张量维度**。
- 捕获后校验每层选择的专家数量与模型配置吻合，记录归一化后的 Top-k 路由权重；如有特殊 routing normalization，保留模型原计算语义。
- 数据集建议：至少 3 类输入（短问答、长上下文摘要/检索、代码/推理），每类若干不同 prompt，使用相同参数进行确定性生成；记录完整 prompt 清单和 tokenizer/model revision。
- trace 可分为：`real_prefill`、`real_decode`、`synthetic_zipf`、`synthetic_bursty`。真实和合成结果在 CSV 与图表中必须分开，**不允许混称**。
- 如果完整 BF16 OLMoE 无法在 8GiB VRAM 加载，使用 CPU offload、小 batch、短上下文或已有可运行的量化基线收集路由。路由来自量化模型时记录这一事实，并与原始 BF16 路由可能存在差异。
- 目标至少覆盖**全部 16 层、每层 64 个专家的索引空间**，不是要求 1024 个专家全部被激活；报告真实覆盖率和热点分布。

建议 trace 事件 schema（JSONL/Parquet，字段类型明确）：

```json
{
  "request_id": "req-001",
  "phase": "decode",
  "token_index": 17,
  "layer_id": 4,
  "expert_id": 23,
  "topk_rank": 2,
  "router_score": 0.147,
  "trace_source": "real_router",
  "model_revision": "<actual-commit-or-revision>"
}
```

### D2. 敏感度不能只来自单层随机输入

- 从**真实模型隐藏状态**抽取被激活专家的输入，离线对比该专家 BF16、INT8_G128、INT4_G128 输出。允许在同一次样本中计算反事实量化误差，但不可在在线解码循环中生成新的量化权重。
- 至少覆盖早期、中间、后期 MoE 层，并按热度和历史误差采样；报告实际覆盖的层/专家/输入数，不要暗示全专家都已测过。
- 除 `relative_L2` 外记录 cosine、局部 `p_i * ||E_i(x) - Ê_i(x)||`；注意多个专家误差**不能直接视为模型级质量损失**。
- 不对没有校准数据的专家通过 `% len(error_table)` 强行套用其他专家误差；应采用保守默认值、同层分位数或“不确定”标记，并做独立消融。
- 可在有限模型级场景评估 next-token logits KL、perplexity 和生成质量（如可复现的任务精度），区分局部代理误差与实际语言模型质量。

**阶段 D 产物**：`artifacts/traces/`（不提交大型文件）、`results/p45_router_coverage.csv`、`results/p45_real_sensitivity.csv`、`docs/P4_5_TRACE_PROTOCOL.md`。  
**进入 E 的条件**：真实 trace 可回放且至少多层真实敏感度数据存在；若不能获得真实 trace，停止并报告阻碍，不宣称 P4.5 完成。

## 7. 阶段 E：公平回放 S0–S4 和消融实验

### E1. 必须比较的策略

- `S0`：固定 INT4（相同存储、设备、缓存预算）。
- `S1`：固定 INT8。
- `S2`：仅 Router score 的精度选择。
- `S3`：Router score + 离线敏感度。
- `S4`：Router score + 敏感度 + 缓存/I/O 成本联合调度。
- **可选 `S4-no-prefetch` / `S4-no-cost` / `S4-no-sensitivity`**：定位收益来源。

**公平性要求**：各策略用相同真实 trace、相同工作集、相同内存和显存预算、相同版本的权重和计算后端。精度策略不同带来字节数不同是研究对象，不能人为固定相同访问字节数。每组从相同缓存初始状态开始，分别统计冷启动和稳态。

### E2. 工作集和缓存预算

- trace 工作集：16（历史对照）、64、256、512、最多 1024 个真实专家实例，或按实际 trace 覆盖的不同 `(layer, expert)` 数报告。
- RAM 缓存预算：0 / 32 / 64 / 128 / 256 / 512 MiB；VRAM 专家缓存预算：0 / 24 / 48 / 96 / 192 MiB。数值指**实验缓存**，不包括模型、KV Cache、CUDA context；如果 P5a 接入真实模型时显存不足，要相应缩小。
- 至少两种访问分布：真实 trace 的原序列，以及保持边际热度但打乱时序的对照 trace；用于区分时间局部性与单纯热度带来的收益。
- 必须输出缓存**目标精度**命中率、**任意精度**命中率、**直接 VRAM 命中率**以及 SSD 读取量。

### E3. 质量–延迟 Pareto 与统计检验

- 横轴至少分别绘制访问延迟、SSD 读取量、峰值显存，纵轴为局部代理误差；质量指标的解释范围必须注明。
- 在相同精度误差上限下比较延迟，或在相同延迟预算下比较误差；**不能只比较 S4 与固定 INT4 的误差、再比较 S4 与固定 INT8 的延迟，就宣称全面更优**。
- 若 S4 被 S0/S1 严格支配（同预算下同时更慢、误差更大、且 I/O 不小），必须忠实记录并分析，不得挑选特殊区间掩盖。
- 对每个关键配置报告 5 次重复的均值与分布，标出样本量和 95% 置信区间或离散程度。

**阶段 E 产物**：`results/p45_policy_comparison.csv`、`results/p45_pareto.png`、`results/p45_ablation.csv`、`docs/P4_5_FINDINGS.md`。  
**进入 P5a 的条件**：C1–C8 全部通过，至少一组真实 Router trace 上能够严格复现结果；性能收益可以为负，只要测量可信即可进入下一阶段探索。

## 8. P5a：仅集成一个真实 MoE 层（条件执行）

> 本阶段目标是**功能与正确性**，不是要求超过全 GPU 的固定 Q4_K_M llama.cpp 吞吐。P5a 不应通过修改全部推理引擎代码来完成。

### F1. 实施顺序

1. 选择一个中间 MoE 层，例如 `layer_id=8`；先确认真实模型模块结构、输入形状、Top-k 语义和专家输出合并方式。
2. 构建**该层**的多精度专家仓库与 `ExpertProvider`：`load(layer, expert, precision) -> tensors + metadata`。精度必须来自**离线生成**的文件；没有目标精度时应抛异常或按显式策略回退。
3. 首先实现 `BF16 -> BF16` 原样替换，保持其他所有层完全不变；用固定 token 序列对比未改模型的该层输出和 final logits，设数值误差阈值并解释可接受浮点差异。
4. 其次实现 `fixed INT8` 和 `fixed INT4` 的单层替换，验证权重形状、scale、重排、专家聚合和 deterministic decode 是否正确。
5. 最后只在这一层启用 `S4`：每次 router Top-k 决策后为该 token 的活跃专家选择 INT4/INT8（BF16 可选），从 RAM/VRAM/SSD 获取对应离线版本，执行该专家，再按模型原始路由权重聚合输出。
6. 显示每次切换的 `token_index, expert_id, requested_precision, actual_precision, source_tier, latency_ms`；确保实际精度与调度器决策一致。
7. 先对 batch size=1、短序列、greedy decode 运行，再考虑 prefill、多请求和异步预取；不要在首次集成时加入 MTP、Tensor Parallel 或复杂 CUDA 自定义核。

### F2. 必须分开的三种执行后端

| 后端 | 目标 | 测量说明 |
|---|---|---|
| `dequant_bf16` | 低精度文件读取后解码并用 BF16 GEMM 执行 | **正确性后端**；将解码、拷贝、GEMM 单独计时 |
| `native_int8`（可选） | 使用已验证正确的 INT8 量化算子 | 必须说明硬件、格式适配和内核版本 |
| `native_int4`（可选） | 直接执行量化 INT4 kernel | 不能把 unpack+BF16 GEMM 命名为原生 INT4 |

如果 PyTorch/HF 集成需要将其余模型放入 CPU RAM：允许，但必须记录层间 offload 与内存开销。即使速度很慢，只要单层改造产生正确 token 和可追溯精度切换，就算 P5a 功能成功；不得与 llama.cpp Q4 GPU 全驻留基线做不公平的加速比较。

### F3. P5a 验收测试

- [ ] `BF16 passthrough`：替换前后相同输入的该层输出、logits 和 greedy token 一致或在预先制定且解释清楚的数值容差内。
- [ ] `fixed INT8` / `fixed INT4`：与离线量化参考输出一致；真实 token 生成可完成。
- [ ] `dynamic INT4/INT8`：同一专家能在不同 token 被选用不同预存版本；线上日志确实反映精度变化，而不是仅在 replay 中变化。
- [ ] 所有在线量化入口受到测试保护；没有在 decode loop 中重新计算 weight scales 或重新打包低比特权重。
- [ ] 运行中峰值显存不超过设备容量且无 OOM；如果单层集成也无法在目标设备运行，记录具体阻碍，不能假称成功。
- [ ] 输出至少两类真实推理指标：TTFT 和 decode `ms/token` / `tokens/s`；与访问微基准在报告中分表。
- [ ] 对比固定 INT4、固定 INT8、S4 的生成 logits、至少一个简单任务数据集上的指标与完整延迟；小样本只做探索，不能外推普遍质量收益。

**P5a 产物**：`src/p5a/` 或等效模块、`tests/test_p5a_*.py`、`scripts/run_p5a.sh`、`results/p5a_e2e.csv`、`docs/P5A_REPORT.md`。

## 9. 推荐的脚本接口（Agent 实现后才可执行）

Agent 可调整具体文件位置，但应保留类似的一键入口、`--config` 和 `--output-dir`：

```bash
# P4.5: 自动执行环境检查、修复测试、真实 trace、回放消融
bash scripts/run_p45.sh --config configs/p45.yaml --output-dir results/p45_run

# P5a: 仅在 P4.5 所有核心门禁通过后运行
bash scripts/run_p5a.sh --config configs/p5a.yaml --output-dir results/p5a_run

# 所有测试
PYTHONPATH=. python -m pytest -q tests/
```

推荐配置项：

```yaml
# configs/p45.yaml（示例；以实际模块 API 为准）
seed: 42
repeat: 5
warmup_accesses: 100
measured_accesses: 2000
model_id: allenai/OLMoE-1B-7B-0125-Instruct
model_revision: null  # Agent 实际下载时固定为 commit hash
precision_formats: [bf16, int8_g128, int4_g128]
ram_cache_mib: [0, 32, 64, 128, 256, 512]
vram_cache_mib: [0, 24, 48, 96, 192]
workloads: [real_prefill, real_decode, synthetic_zipf, synthetic_bursty]
strategies: [S0, S1, S2, S3, S4]
metrics:
  capture_cuda_events: true
  capture_process_gpu_peak: true
  capture_prefetch_bytes: true
  separate_os_page_cache: true
```

如果脚本或真实模型资源不足，必须执行可行的子集并在报告中逐项记录 `SKIPPED` 和门禁状态；**不得伪造未执行脚本的输出**。

## 10. 交付目录和最终报告格式

建议最终目录如下：

```text
llm_test/
├── P4_5_VALIDATION.md          # 本计划，放仓库根目录
├── configs/p45.yaml            # 新增配置
├── configs/p5a.yaml            # 门禁通过后再创建
├── scripts/run_p45.sh          # 可重复实验入口
├── scripts/run_p5a.sh          # 可选、按阶段创建
├── src/                       # 修改/新增缓存、路由 trace、测量与 P5a
├── tests/                     # 新增正确性、并发与精度测试
├── docs/
│   ├── METRICS_DEFINITIONS.md  # 指标口径
│   ├── P4_5_TRACE_PROTOCOL.md  # 真实 Router 数据获取
│   ├── P4_5_FINDINGS.md        # 问题复核、修复、实测与门禁
│   └── P5A_REPORT.md           # 若 P5a 执行
├── results/
│   ├── p45_legacy_audit.csv
│   ├── p45_baseline_manifest.json
│   ├── p45_cache_correctness.json
│   ├── p45_io_microbench.csv
│   ├── p45_gpu_microbench.csv
│   ├── p45_router_coverage.csv
│   ├── p45_real_sensitivity.csv
│   ├── p45_policy_comparison.csv
│   ├── p45_ablation.csv
│   ├── p45_pareto.png
│   └── p5a_e2e.csv              # 若执行 P5a
└── logs/                       # 原始运行日志（大文件不入 Git）
```

### 最终 `docs/P4_5_FINDINGS.md` 至少回答

1. 旧实验哪几个数据和当前 CSV 不一致，原因是什么？
2. 修复后 0 容量缓存是否严格无命中、无超预算？是否有并发错误？
3. P1 全 GPU 与 CPU MoE 的**运行时真实峰值显存**是多少？相比旧报告发生何种变化？
4. 真实 NVMe 物理 I/O、Page Cache 读取、RAM→VRAM、VRAM 命中各自的耗时是多少？哪项测量仍不可靠？
5. 使用真实 Router trace 后，在相同预算下 S4 处于 Pareto 前沿吗？若失败，是因为算法、工作集、I/O 还是内核开销？
6. 为什么当前运行时的低比特解码与原生低比特 GEMM 不等价？不同后端测到的结果各是什么？
7. P4.5 C1–C8 哪些 PASS/FAIL/SKIPPED？是否批准进入 P5a？
8. 如果执行 P5a，同一专家在**真实生成**过程中是否发生真实的 INT4↔INT8 精度切换？对 logits、TTFT、TPOT、显存产生了什么影响？

## 11. 停止条件与后续方向

**立即停止 P5a、优先修复 P4.5** 的情形：缓存超预算、预取与前台竞争死锁、目标精度与实际加载版本不一致、无法获取可核验的真实 Router trace、统计无法由事件日志复算。

**可以继续但需要如实标注局限** 的情形：无法测得物理 SSD I/O、无法获得原生低比特 kernel、P5a 只能使用 BF16 解码后计算、S4 的性能没有超过固定 INT4 或 INT8。

**后续 P5b/P6（本计划不自动实施）**：多层协同、异步预取、原生量化 kernel、以及超出 8 GiB 显存的更大 MoE。只有当 P5a 证明真实生成精度切换正确，才值得投入这些工程优化。

---

## 12. 给 Agent 的最终指令

> 你正在维护 `pentyum/llm_test`，本任务为 **P4.5 验证修复，门禁通过后再进行 P5a 单层真实推理**。请先读取仓库与此文档，不要直接大规模重构。按 A→B→C→D→E 顺序逐项实现、执行测试并保存结果；每完成一阶段，在 `docs/P4_5_FINDINGS.md` 记录修改文件、实际运行命令、测试结果、尚存问题和下一阶段是否允许继续。C1–C8 不全部通过就停止在 P4.5，并给出客观的失败报告。若条件允许进入 P5a，只改一个真实 MoE 层，首先做到 BF16 原样替换，再做固定 INT8/INT4，最后做同一专家不同 token 的动态版本切换。全过程不在线量化权重，不覆盖历史成果，不杜撰数据。最终提供 commit SHA、完整文件差异清单、通过/失败的测试列表、可复现命令以及结论的适用边界。

### 参考链接

- 项目仓库：https://github.com/pentyum/llm_test
- 已有报告：https://github.com/pentyum/llm_test/blob/main/REPORT.md
- 分层缓存实现：https://github.com/pentyum/llm_test/blob/main/src/tiered_cache.py
- 精度调度：https://github.com/pentyum/llm_test/blob/main/src/precision_scheduler.py
- 量化编解码：https://github.com/pentyum/llm_test/blob/main/src/quant_codec.py

*本文件是针对当前仓库和历史报告制定的下一轮执行方案，不是声称 P4.5/P5a 已运行或验证。*
