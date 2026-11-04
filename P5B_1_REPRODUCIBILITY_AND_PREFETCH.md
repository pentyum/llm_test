# P5b.1 实验计划：多层 MoE 可复现性、专家恢复与真实预取验证

> **交给 Coding Agent 的执行规范（可直接作为任务书）**  
> 项目：[`pentyum/llm_test`](https://github.com/pentyum/llm_test)  
> 起点：`main`，执行时记录实际 Git commit SHA；建议新分支 `experiment/p5b1-reproducibility-prefetch`  
> 设备：Ubuntu / NVIDIA RTX 4060 Laptop 8 GB VRAM / 48 GB RAM / NVMe SSD  
> 目标模型：`allenai/OLMoE-1B-7B-0125-Instruct`  
> 默认替换层：`[0, 4, 8, 12]`；分级回归 `[0] → [0, 4] → [0, 4, 8, 12]`  
> 优先级：**正确性与可复现性 > 指标真实性 > 预取收益 > 吞吐优化**  
> 状态：**待 Agent 实际运行、验收；本文件不是实验完成报告**

---

## 0. 任务边界与已核对事实

### 0.1 最终要回答的五个问题

1. 全新 checkout、可记录的 Conda 环境中，P5b 四层卸载是否能从零运行、稳定重复？
2. 模型替换与卸载后，`restore()` 能否安全恢复可用的原始 MoE？如果不能，是否显式拒绝而不是产生损坏的模型？
3. 当前的预取回调是否在 **真实模型前向路径** 中触发？预取任务是否在专家访问前真正完成、保留在缓存中、减少了关键路径等待？
4. RAM / VRAM 预算、逻辑 SSD 读取和物理磁盘 I/O 的测量是否可审计、口径一致？
5. 在同一输入、同一精度策略、同样的缓存和暖机条件下，预取是否真的改善了 TTFT / TPOT，提升是否超过运行波动？

### 0.2 与上一轮审查的关键纠正

**不要预设 `prefetch_manager` 构造参数仍然缺失。** 本任务书编写时，公开 `main` 的以下代码可见：

- `src/p5b/offloaded_moe.py` 把 `prefetch_manager` 传给 `DynamicMoELayer`。
- `src/p5a/dynamic_moe_layer.py` 的构造函数**已经接受** `prefetch_manager=None`；`forward()` 中**已经调用** `on_layer_enter()` / `on_expert_access()`。
- 当前 `src/p5b/offloaded_moe.py` 在 `release_orig_experts=True` 时把原始专家设为 `None`，但 `restore()` 没有重建专家。这是应优先验证并修复的逻辑。
- 当前 `src/p5b/prefetch_manager.py` 的“命中”主要通过预取键集合统计，还不能直接证明预取完成、仍然驻留和关键路径等待被隐藏；预取字节计数中还存在按固定 3,200,000 字节估算的逻辑。

因此 **G0 必须重新核对代码和可执行结果；对于已修复的问题，不得为了满足计划而重复改代码**。仓库随后可能再次更新，以实际锁定的 SHA 为准。

参考源码：

- https://github.com/pentyum/llm_test/blob/main/src/p5b/offloaded_moe.py
- https://github.com/pentyum/llm_test/blob/main/src/p5a/dynamic_moe_layer.py
- https://github.com/pentyum/llm_test/blob/main/src/p5b/prefetch_manager.py
- https://github.com/pentyum/llm_test/blob/main/docs/p5b/P5B_REPORT.md

### 0.3 不允许的做法

- 不允许覆盖 `results/p5b/`、`docs/p5b/` 原有历史结果；新数据统一放 `results/p5b1/` 和 `docs/p5b1/`。
- 不允许在推理阶段重新量化 BF16 权重；可以**反量化**预存 INT4/INT8 参与计算，并准确标注为“反量化 + BF16 GEMM”，不是原生 INT4 GEMM。
- 不允许只通过文件 `open()` 或读取次数断言“物理 SSD 冷读取”。
- 不允许将“预取任务发出”“预取键后来被访问”算作“有效预取命中”。
- 不允许将 Oracle 使用未来真实路由信息的结果宣传为可部署的在线预取算法。
- 不允许修改 Prompt 集合、评测长度、随机种子、运行顺序或缓存条件后仍直接与旧 P5b 数值做精确对比。
- 不允许在未运行成功时报告 PASS，也不得写死实验结论、表格数值或加速比。
- 不允许为通过测试而跳过失败样本、自动换更大的 GPU、放宽显存限制或悄悄保留全部原始 BF16 专家。
- 不允许使用 `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` 作为默认测试路径；只能在明确授权、独占测试机且有风险说明时作为可选方法。

---

## G0. 固定输入、环境和 Git 基线（P0，强制）

### G0.1 在干净工作区执行

```bash
set -euo pipefail
cd /path/to/llm_test

git status --short
# 如存在未提交改动，先保存/提交；不得直接覆盖。
git fetch origin
git switch main
git pull --ff-only origin main
# 若分支已存在，请继续使用该分支，不要强制覆盖。
git switch -c experiment/p5b1-reproducibility-prefetch

mkdir -p results/p5b1 docs/p5b1 logs/p5b1
BASE_SHA=$(git rev-parse HEAD)
printf '%s\n' "$BASE_SHA" > results/p5b1/base_commit.txt
```

若机器上暂缺模型/权重，Agent 需寻找仓库已有的文件路径与下载/导出脚本，**先记录缺失内容和估算磁盘占用**，不得自动重复下载几十 GB 模型。路径全部由 `--model-path`、`--store-path` 或环境变量输入，绝不保留 `/home/yaoyao/...` 等硬编码绝对路径。

### G0.2 独立 Conda 环境

```bash
conda create -n moe-p5b1 python=3.11 -y
conda activate moe-p5b1
python -V
nvidia-smi
```

安装依赖时先检查仓库现有安装文档、环境锁定文件和历史 `ENVIRONMENT.md`；优先重现 P5b 报告版本（当时记录 `PyTorch 2.6.0+cu124` 和 `Transformers 5.22.1`），**不得盲目升级 Transformers 或 CUDA**。若无法安装原版本，则选择可用组合并在报告中将其列为“新环境”，不得声称严格复现。

```bash
# 根据仓库实际文件决定依赖安装命令，不能假设 requirements.txt 一定存在
ls -1 | grep -E 'requirements|environment|pyproject|setup' || true
python -m pip freeze > results/p5b1/pip_freeze.txt
conda list --explicit > results/p5b1/conda_explicit.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
  > results/p5b1/gpu_info.csv
lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINT > results/p5b1/disks.txt
uname -a > results/p5b1/kernel.txt
```

环境信息补充：CPU、RAM、SSD 文件系统和挂载点、PCIe 链路信息（可读时）、可用磁盘空间、CUDA Runtime、torch CUDA 能力、测试时的其他 GPU 进程。

### G0.3 运行前健康检查与代码审计

必须记录以下项目（建议生成 `results/p5b1/preflight.json`）：

- `DynamicMoELayer.__init__` 是否接受 `prefetch_manager`，且传入 `None`/对象时可实例化。
- 真正调用 `on_layer_enter()` 与 `on_expert_access()` 的文件与行号。
- `release_orig_experts=True` 之后原专家对象是否被其他引用持有；不能仅通过设置 `None` 就断言 CPU RAM 已释放。
- `restore()` 是否能恢复专家参数完整性与输出可用性。
- 预取策略有哪些；实际使用的是哪个；能否复现单次模型生成。
- P5b 原始 CSV、JSON、报告的 SHA、参数与运行数是否能一一对应。

推荐最小签名检查：

```bash
PYTHONPATH=. python - <<'PY'
import inspect
from src.p5a.dynamic_moe_layer import DynamicMoELayer
from src.p5b.offloaded_moe import MultiLayerOffloadedMoE
print('DynamicMoELayer', inspect.signature(DynamicMoELayer.__init__))
print('MultiLayerOffloadedMoE', inspect.signature(MultiLayerOffloadedMoE.__init__))
PY

PYTHONPATH=. pytest -q tests/ \
  > logs/p5b1/pytest_initial.txt 2>&1 || true
```

测试允许失败，但日志必须保留；不要通过 `|| true` 把测试失败当成最终 PASS。记录每个失败测试的名称和 traceback。

**G0 验收门槛**：可在固定 SHA 中导入所有关键模块，产出环境快照；若真实模型文件缺失，则仅执行不依赖模型的单元测试，并标记 `BLOCKED_MODEL_ARTIFACTS`。

---

## G1. 多层集成回归与 `restore()` 语义（P0，先于性能测试）

### G1.1 测试矩阵

| 场景 | 替换层 | `release_orig_experts` | 预期 |
|---|---|---:|---|
| T1 | `[0]` | `False` | attach → forward → restore → reference forward 全部可执行 |
| T2 | `[0,4]` | `False` | 同上 |
| T3 | `[0,4,8,12]` | `False` | 同上，记录峰值显存 |
| T4 | `[0]` | `True` | attach 与动态前向可用；恢复必须安全 |
| T5 | `[0,4,8,12]` | `True` | 卸载后真实依赖 store；恢复不能留下损坏原层 |
| T6 | `[0,4,8,12]` | `True` | 重复 attach/restore、抛异常和提前退出的资源清理 |

### G1.2 `restore()` 的允许设计（二选一；必须写清楚）

**方案 A（推荐）**：恢复时从原始 checkpoint 重新实例化或加载缺失专家，然后将完整模块挂回模型。必须核对权重张量哈希、dtype、device、forward 数值差异。若原始 checkpoint 无法获取，调用 `restore()` 应抛出明确的异常而不是静默恢复半残缺对象。

**方案 B（工程上更简单）**：声明 `release_orig_experts=True` 是不可逆卸载，`restore()` 必须立即报出 `RestorationUnavailableError`（或明确 `RuntimeError`），提示重载原模型；同进程继续使用原模型前必须重新加载，不能把 `experts=None` 的模块恢复到执行路径。`release_orig_experts=False` 必须仍然支持正常恢复。

**禁止方案**：忽略丢失的专家、只恢复模块引用、假装已恢复并在稍后前向时崩溃。

### G1.3 内存真实性测试

在同一进程内采集以下 4 个快照：

1. `baseline_loaded`：原始模型全部加载之后；
2. `attached_retain`：attach 且不卸载专家；
3. `attached_release`：释放原专家、GC、CUDA 同步之后；
4. `restored_or_explicitly_unavailable`：恢复成功或明确不可恢复。

记录：RSS、USS（可用时）、`torch.cuda.memory_allocated()`、`memory_reserved()`、GPU 进程显存、参数数量、参数引用检查；不同进程运行时不要误将系统 Page Cache 算作 RSS。

**G1 验收门槛**：T1–T6 的行为都符合已声明语义；不能出现不完整恢复；有单元测试和内存快照。若此项失败，**停止后续全部性能宣传**。

建议新增：`tests/p5b1/test_attach_restore.py`、`tests/p5b1/test_release_memory.py`。

---

## G2. 预取状态机与端到端钩子验证（P0）

### G2.1 将预取从“键集合”升级为可观测任务状态

以 `(layer_id, expert_id, precision, store_version)` 作为唯一键，至少支持：

```text
NOT_REQUESTED → QUEUED → READING → RAM_READY → CONSUMED
                            ↘ FAILED
RAM_READY → EVICTED
QUEUED/READING → CANCELLED（仅安全取消）
```

任务记录至少包含：

- `request_id`, `run_id`, `token_step`, `layer_id`, `expert_id`, `precision`
- `request_ts_ns`, `read_start_ns`, `read_done_ns`, `ready_ts_ns`, `access_ts_ns`
- `bytes_requested`, `bytes_read_logical`, `bytes_disk_observed`（若无法归因则为 `null`）
- `cache_tier_at_access`, `state_at_access`, `evicted_before_access`
- `wait_ms_at_access`, `io_error`, `cancel_reason`
- `prefetch_source`: `static_hot`, `oracle`, `manual` 等

必须区分：

- **issued**：发出预取请求；
- **ready**：数据已经加载完成、可安全使用；
- **used**：随后有真实专家访问；
- **effective_hit**：访问时相同精度版本已经 ready 且仍驻留，实际省去同步加载；
- **late_prefetch**：访问时尚未 ready，仍需要等待；
- **wasted**：任务完成但未被使用，或使用前已驱逐；
- **coalesced**：对同一键重复请求被合并。

现有 `prefetched_keys` / `hit_keys` 可以保留为兼容统计，但报告中的 `effective_prefetch_hit_rate` **只能按上述严格定义计算**。

### G2.2 严格并发约束

- 同一键最多一个物理加载任务；多位等待者可以共享 `Future`。
- 预取失败必须可查询、可上报，不能在线程中悄悄吞掉异常。
- 不得在持有全局缓存锁时进行长时间 SSD 读取或同步 H2D 复制。
- 预算不足必须安全拒绝预取，不能“先超限，之后再驱逐”。
- “ready” 必须以数据拷贝完成为准，不能仅凭异步任务提交；CUDA 异步复制如被引入，需要正确的 Event/Stream 同步。
- 避免请求预取和同步 `get_expert()` 对同一键重复读盘；适用时让前台等待在途加载。
- 禁止将不同量化格式、不同 store 版本的权重错误地复用为命中。
- 预取线程池必须有上限，可 shutdown 并 `join`，避免测试结束后仍在写结果。

### G2.3 确认 hook 真正来自模型执行

在 `DynamicMoELayer.forward()` 进入时调用 `on_layer_enter()`，在请求专家权重时调用 `on_expert_access()`。当前公开源码已有类似 hook，但还要验证：

1. 这些调用在真正运行 `model.forward()` / 增量 `decode` 时触发，而非只在独立演示脚本中触发；
2. `current_decode_step` 随 token 正确变化，prefill/decode 不混淆；
3. 不同 Layer 的事件顺序符合实际模型执行顺序；
4. 每次访问使用的精度与 store 中加载精度一致；
5. 没有未来信息泄漏到 `static_layer_prefetch`；Oracle 仅保留为离线上限对照；
6. 跨层预取应在下层专家需要之前启动，并记录可用的重叠窗口。

建议新增：`tests/p5b1/test_prefetch_state_machine.py`、`test_prefetch_integration.py`、`test_prefetch_concurrency.py`。

### G2.4 必须执行的可控故障注入

- 读取延迟注入（例如可配置 20ms）→ 正确区分 ready / late；
- 同一键连续发起 10 次预取 → 只产生 1 次实际读取或明确的去重计数；
- 预取读取期间立刻访问 → 返回正确权重，不能竞争读盘或返回未完成数据；
- 预取后主动驱逐再访问 → 不能计入 effective hit；
- 文件缺失/损坏 → FAILED 被观测到，并安全回退或抛错；
- 预算 0、专家超过预算 → 不会越界、不会死锁；
- 终止和恢复进程 → 无残留线程泄漏、无部分结果被误认有效。

**G2 验收门槛**：全部状态与故障测试通过，日志可追踪一次专家从预取发起到实际使用的完整因果链。

---

## G3. 严格缓存与 I/O 观测（P0）

### G3.1 缓存字节预算

建议测试预算：

| 维度 | 小 | 中 | 大 |
|---|---:|---:|---:|
| VRAM 专家缓存 | 256 MiB | 512 MiB | 1024 MiB |
| RAM 专家缓存 | 128 MiB | 512 MiB | 2048 MiB |

这些是**专家缓存预算**，不包含模型其他权重、KV Cache、CUDA context 和工作区。真实 8GB VRAM 机器上应先做 `torch.cuda.mem_get_info()` 检查：如果剩余显存不足，跳过超预算配置并记录 `SKIPPED_OOM_RISK`，不能盲目 OOM。每个访问之后断言缓存拥有的有效权重大小 `<= budget`。另外记录 `torch.cuda.max_memory_allocated()` 和 GPU 进程的**真实峰值显存**，不能以缓存预算代替总显存。

### G3.2 三类 I/O 口径必须分开

| 指标 | 如何解释 | 观测建议 |
|---|---|---|
| 逻辑文件读取 | 代码从 SSD store 文件 API 请求了多少字节 | 在 `WeightStore` / 缓存边界按实际 `read()`/加载体积计数 |
| 进程物理块读取 | 读取是否引发底层 storage I/O | Linux `/proc/<pid>/io` 的 `read_bytes` 差值，注意 mmap 与子进程归因限制 |
| 设备级磁盘活动 | 目标 NVMe 设备的真实吞吐 | `iostat -dx 1` 或等价工具，注意后台其他进程干扰 |

`read_bytes=0` **不能单独证明**所有数据均来自缓存；`read_bytes>0` 也要排除模型、日志及其他文件 I/O。需配合独占 NVMe、文件路径追踪、Warm/Cold 分组和设备级计数。

### G3.3 Warm / Cold / Page Cache 分组

- **Warm-Cache**：先读相同文件一次，然后重复执行，不主动清理系统缓存，标为 `warm_buffered`。
- **Process-Restart**：新进程启动，并不等于冷 SSD 读（Page Cache 可能仍然很热）。
- **Cold-Verified（可选）**：只有在有权限、确认确实发生设备级读取且不污染其他程序时才可以使用该称呼。可使用独立测试文件集、适当的 `posix_fadvise`（仅是提示，需验证结果）或受控 `O_DIRECT` 路径（要求对齐）；不要把技术措施本身当成冷读取证据。
- **SSD 未验证**：若无法证明物理读取，则在报告中统一称 `buffered/offloaded file path`，不能称 `physical cold SSD`。

对于 `mmap` 加载，要单独讨论 Page Fault 和内核文件缓存，不能把文件内存映射直接解释为完全通过磁盘读取。

### G3.4 建议保存的数据

- `io_events.jsonl`：每次文件/预取访问的字节数、路径所属 store、状态、时间；
- `process_io.csv`：固定频率采样 `rchar`, `read_bytes`, `syscr` 等；
- `device_io.csv`：NVMe 设备级采样；
- `memory.csv`：CPU RSS/USS、GPU allocated/reserved、进程 GPU memory；
- `cache_events.jsonl`：对象的插入、命中、驱逐、旁路、容量变化。

所有读写数据应经过同一个 `run_id` 关联。只将实际字节纳入统计，删除 `3_200_000` 等固定大小作为“实测 SSD 字节数”的写法；若必须预测大小，应单列 `estimated_bytes`。

**G3 验收门槛**：同一次运行能解释 `逻辑请求字节 → RAM/VRAM 缓存命中 → 实际等待 → 设备物理读取是否发生`。不可证的部分标记 `unknown`，而非填 0。

---

## G4. 多层 BF16 一致性及路由回归（P0/P1）

### G4.1 对照路径

相同模型 checkpoint、tokenizer、Prompt、随机种子、数据类型、设备设置和生成参数，分别测试：

1. `ref_unmodified`：原始模型；
2. `L1_bf16`：只替换 Layer 0；
3. `L2_bf16`：替换 Layer 0/4；
4. `L4_bf16`：替换 Layer 0/4/8/12；
5. 对 L4 的 fixed INT8 / INT4 / S4 保留为**量化质量对照**，不能混为 BF16 等价性测试。

### G4.2 比较维度

- 同一 hidden state 输入下，Router logits、Top-k IDs、routing weights；
- 每个专家的 SwiGLU 输出与加权聚合输出；
- 每个被替换 MoE 层的相对 L2 误差、最大绝对误差、Cosine；
- 整个模型的 teacher-forced logits 相对差、Top-1 一致率、平均 KL；
- Prefill 与增量 KV Cache Decode 的一致性，保留逐 token 独立时间戳。

当前历史实验 L4 BF16 自回归匹配率不完美。不能要求 BF16 位级相等作为不现实的唯一门槛；但必须**定位并量化**数值差异，证明其来自可解释的设备 / 顺序 / dtype 路径，而非错误的专家映射、丢失专家或路由计算错误。

### G4.3 建议正确性门槛

- 所有关键权重的形状、索引、dtype、哈希对应正确；
- 同样输入下 Router Top-k IDs 应一致；若不一致，视为 P0 失败并定位；
- BF16 层输出相对 L2 误差按原 P5b 已验证阈值（如 0.005）做回归，同时提供最大值和分布，而不是只给均值；
- 数值无 NaN / Inf；
- 生成 Token 匹配率只作为**诊断项**，不得单独作为量化质量结论；
- 如果误差超既有阈值，停止 P5b.1 的性能结论传播，保留失败报告。

新增 `tests/p5b1/test_multilayer_bf16_parity.py`。

---

## G5. 真正公平的预取消融实验（P1）

### G5.1 比较组

| 组别 | 预取 | 数据可见性 | 目的 |
|---|---|---|---|
| A | `sync_no_prefetch` | 当前 Router | 正式基线 |
| B | `static_layer_prefetch` | 仅使用历史 hot expert/当前已知信息 | 可部署预取 |
| C | `oracle_prefetch` | 允许使用未来路由轨迹 | 离线上限，不可部署 |
| D（可选） | 消融关闭后台线程，仅同步 warm RAM | 不使用未来信息 | 区分线程与 Page Cache 收益 |

**公平性要求：** 首先固定权重精度为 INT4 对比 A/B/C；再固定 INT8 对比；最后才使用 S4。不同策略的精度选取差异会改变权重大小和访存量，禁止直接将不同精度策略的时间差全部归功于预取。

### G5.2 实验参数

- 层数：`[0]` smoke；`[0,4]` 回归；`[0,4,8,12]` 正式。
- 输入：仓库既有 32 条 Prompt（若存在且可用），至少覆盖多类别；先用 4–8 条 smoke，再完整 32 条。
- 生成长度：统一，例如 `max_new_tokens=64`；早停数量和实际解码 token 数必须记录。
- 随机性：greedy 模式优先；固定 seed。不能让不同模式使用不同采样随机数流。
- 重复次数：正式运行每组至少 5 次独立 run，交替或随机平衡 A/B/C 的运行顺序，记录机器温度和后台负载。
- 预算：固定 VRAM/RAM 字节预算，所有组独立 Cache 实例；每次重新创建模型/重置缓存的顺序固定；分 warm/cold 两套实验，不交叉汇总。
- GPU：RTX 4060；不能遇到 OOM 就切换 RTX 5080 后将数据并入同表。

### G5.3 核心测量

**端到端**：

- TTFT（明确包含 tokenizer 与否、是否包含首次文件读取）；
- 每个实际 Decode token 的 TPOT `p50/p95/p99/mean`，不是各次平均 TPOT 的分位数；
- Decode tokens/s、实际输出 token 数、峰值 VRAM、峰值 RAM、OOM 次数；
- teacher-forced KL、Top-1 一致率与生成文本（用于防止非预期改变模型质量）。

**预取本身**：

- requests / completed / cancelled / failed / coalesced；
- issued keys used vs ready-at-access effective hits vs late-prefetch vs evicted-before-use；
- `wait_ms_at_access` 分布、主动预取 I/O、无用预取 I/O；
- 预取与前台实际 GPU 计算重叠的窗口，必须有时间戳或 Trace 证明；
- 设备读量与逻辑读量，可能时按文件区分。

**指标示例（必须标注分母）**：

```text
effective_prefetch_hit_rate =
  accesses_served_by_ready_resident_prefetch / eligible_expert_accesses

late_prefetch_rate =
  accesses_blocking_on_in_flight_prefetch / eligible_expert_accesses

prefetch_waste_ratio =
  bytes_completed_but_never_consumed_or_evicted / bytes_prefetch_completed
```

要记录 0 分母情形为 `null`/N/A，不能写成 0%。可以单独保留“预取请求后使用率”，但不许把它标成 effective hit。

### G5.4 预取获得真实收益的判断

- 若 B 比 A 快：提供同一批 Prompt、相同缓存预算、相同精度的配对延迟差异分布、均值与置信区间。
- 若 B 没有显著提升：报告 `NO_CONFIRMED_SPEEDUP`，分析 Page Cache、工作集、预取完成太晚、执行开销占优、线程竞争等原因；**不能因为预取命中率高就标为性能成功**。
- 若 C 明显优于 B：报告潜在预测上限，但不能将 C 的收益归于当前在线预测器。
- 若 B 的预取额外 I/O 使尾延迟变差：列出负收益及关键 Trace，保留失败样例。

建议使用配对 bootstrap 的 95% CI（以 Prompt/run 为合理采样单位，避免把同一次生成内大量 token 当作独立样本）；也可直接报告每次结果及样本标准差，不用强行追求显著性。

**G5 验收门槛**：A/B/C 都能在四层真实 Decode 下完成至少 3 次可复现运行，正式 5 次；有严格 effective hit 统计，并明确结论是否证实加速（允许否定结论）。

---

## G6. 跨模式数据一致性和自动报告（P0/P1）

### G6.1 每次运行必须有不可变的元数据

建议 JSON：

```json
{
  "run_id": "p5b1_...",
  "git_sha": "<实际SHA>",
  "branch": "experiment/p5b1-reproducibility-prefetch",
  "model_id": "allenai/OLMoE-1B-7B-0125-Instruct",
  "checkpoint_fingerprint": "<sha256或可复现文件清单>",
  "device": "RTX 4060 Laptop 8GB",
  "target_layers": [0, 4, 8, 12],
  "precision_mode": "fixed_int4",
  "prefetch_strategy": "static_layer_prefetch",
  "vram_cache_mib": 512,
  "ram_cache_mib": 512,
  "io_condition": "warm_buffered",
  "prompt_set_hash": "<实际SHA256>",
  "seed": 42,
  "repeat_index": 0,
  "status": "COMPLETED"
}
```

`<...>` 是文档占位符，Agent 在实际运行时必须填入真实值；无从得知的字段写 `null` 与 `reason_unknown`，**不可伪造**。

### G6.2 原始数据作为唯一真相

将一切摘要/图表从原始事件生成。建议：

```text
results/p5b1/
  base_commit.txt
  env/
  run_manifest.jsonl
  runs/<run_id>/
    metadata.json
    generated_tokens.jsonl
    token_latencies.jsonl
    expert_access_events.jsonl
    prefetch_events.jsonl
    cache_events.jsonl
    process_io.csv
    device_io.csv
    memory.csv
    stderr.log
    summary.json
  aggregate/
    result_rows.csv
    prefetch_ablation.csv
    parity_summary.csv
    regression_status.json
    speedup_ci.csv
    plots/*.png

docs/p5b1/
  P5B1_REPORT.md
  P5B1_REPRO_STEPS.md
  P5B1_KNOWN_LIMITATIONS.md
```

事件 schema 的字段允许在代码中调整，但必须写明版本、单位（ns/ms/byte/MiB/GB）、分母和采集来源。

### G6.3 机器可检查的不变量

报告生成前自动断言：

1. 报告的所有行都关联存在的 `run_id` 和 SHA；
2. 每个解码 token 只有合法 step ID，token 个数与吞吐计算分母一致；
3. `max(cache_owned_bytes) <= budget_bytes`；
4. 无缓存缺失事件在统计上被误标为 VRAM 命中；
5. `effective_prefetch_hits <= eligible_accesses`；
6. `completed_prefetch_bytes` 来自真实张量/文件长度，而不是硬编码常量；
7. `quality` 不是通过 prompt token 匹配率冒充；
8. 每项 speedup 都能从原始 A/B 行重新计算，且时间/字节单位一致；
9. `PASS/FAIL/NOT_VERIFIED` 从测试结果计算，**不是手工写死**；
10. 生成报告时如果缺少原始数据，应失败并列出缺失文件。

新增建议脚本：

- `scripts/p5b1/run_repro.py`：单组可复现实验 CLI（不要假定仓库现有脚本已支持这些参数）；
- `scripts/p5b1/run_ablation.py`：矩阵调度与断点续跑；
- `scripts/p5b1/build_report.py`：CSV/图表/报告自动构建；
- `scripts/p5b1/validate_results.py`：检查数据完整性和报告一致性。

必须提供 `--dry-run` 列出参数矩阵、预期硬盘空间和命令，不实际加载模型；必须支持 `--output-dir` / `--model-path` / `--store-path`。

---

## G7. 建议执行顺序、停止条件与交付

### G7.1 建议分组执行（不要一次跑完所有组合）

| 顺序 | 内容 | 若失败则 |
|---|---|---|
| 1 | G0 环境、代码签名、依赖健康检查 | 停止，修复环境 |
| 2 | G1 attach/restore、原专家释放、内存真实性 | 停止，修复模型完整性 |
| 3 | G2 预取状态机与故障注入 | 停止，修复预取正确性 |
| 4 | G3 缓存容量、I/O 计数 | 停止，不做性能结论 |
| 5 | G4 `[0]`→`[0,4]`→`[0,4,8,12]` BF16 回归 | 停止，定位误差 |
| 6 | G5 少量 Prompt A/B/C smoke | 先排除异常 |
| 7 | G5 全量 32 Prompt、5 次配对重复 | 统计收益或无收益 |
| 8 | G6 自动汇总、审计与提交 | 任何断言失败需标注 FAIL |

### G7.2 必须交付

- [ ] `experiment/p5b1-reproducibility-prefetch` 分支上的最小、可审查代码变更。
- [ ] 所有新测试与各阶段日志、失败实例；不得只提交最终成功日志。
- [ ] 完整 `run_manifest.jsonl`，含 Git SHA、环境、Prompt SHA 和配置。
- [ ] `restore()` 的明确语义与单元测试；不能恢复时必须显式失败。
- [ ] 真实前向执行的预取事件和至少一条完整有效命中链。
- [ ] 数据按 `logical_reads`、`process_read_bytes`、`device_reads` 区分。
- [ ] 无预取、静态预取、Oracle 对照的可复现实验结果。
- [ ] 逐 token TPOT 数据和成对 CI / 运行方差。
- [ ] 完整四层 BF16 parity 指标。
- [ ] `docs/p5b1/P5B1_REPORT.md`、`P5B1_REPRO_STEPS.md`、`P5B1_KNOWN_LIMITATIONS.md`。
- [ ] 每项关键结论都能追溯到具体原始数据与提交 SHA。

### G7.3 最终 Gate（全部满足才允许进入更大规模 P5b.2）

**必须 PASS：**

1. 在干净 checkout 上能够启动并完成真实四层推理（实际 GPU 环境）；
2. 不存在因不完整 `restore()` 导致的静默模型损坏；
3. 缓存预算严格执行，资源释放稳定；
4. 预取任务的 ready / late / evicted / used 定义明确、实测正确；
5. 同一精度下 A/B/C 的评测配置公平；
6. BF16 parity 通过既有数值回归要求，或被明确判为未通过并停止宣传；
7. 统计报告由原始 CSV/JSONL 重新生成，没有硬编码通过结论。

**可以是 FAIL 但必须如实报告：**

- 静态预取没有带来显著 Decode 加速；
- SSD 测试仅达到 buffered file path，不能验证物理冷读；
- S4 比固定 INT4 慢或质量无显著优势；
- BF16 自回归 token 不完全一致（在数值回归检验通过、且差异有解释时）。

如果某项只能部分验证，状态写 `PARTIAL`/`NOT_VERIFIED`，不能概括成全部成功。

### G7.4 报告模板（必须由脚本填充）

```markdown
# P5b.1 实验报告

## 运行基线
- Git SHA / 模型 fingerprint / 环境 / GPU / SSD

## G0–G4 正确性
- 接口与 restore 行为：PASS / FAIL
- RAM/VRAM 预算检查：PASS / FAIL
- BF16 L1/L2/L4 parity：数值表 + 失败样例
- 历史 P5b 与本轮是否严格可比：YES / NO，说明原因

## 预取准确性
- issued / ready / effective hits / late / evicted / wasted
- 实际 SSD logical bytes 与 device 级 I/O（或 NOT_VERIFIED）
- 可追溯的事件时间轴

## 端到端性能
- 相同 INT4 的 A/B/C 结果（TTFT、TPOT p50/p95/p99、t/s、置信区间）
- 相同 INT8 的 A/B/C 结果
- S4 作为附加分析
- 质量指标和显存峰值

## 最终结论
- 已证实收益
- 尚未证实/负收益
- 可以进入 P5b.2 吗？YES / NO，按 Gate 条件自动判定
```

---

## 附录 A：给 Agent 的执行指令（可直接复制）

> 你是该项目的实验工程 Agent。以 `pentyum/llm_test` 当前 `main` 为基线，新建 `experiment/p5b1-reproducibility-prefetch` 分支，先执行 G0 进行事实核对，再按 G1–G7 顺序修改、测试、记录。请勿假设旧审查中所有问题仍存在：公开 `main` 代码已经含有 `prefetch_manager` 构造参数和前向回调；优先核查 `restore()` 的卸载恢复、预取“真正 ready 并驻留”的命中语义和物理 SSD I/O 证据。原 P5b 数据不可覆盖；只在 `results/p5b1/`、`docs/p5b1/` 写新结果。保留所有失败、日志与运行 SHA。必须在 RTX 4060 8GB 环境中执行，不能在线重量化。对无法证明的性能结论写 `NOT_VERIFIED`。先解决可复现与数值一致性，再进行配对预取消融；完成后提交测试代码、原始事件、自动报告及结论。

## 附录 B：本轮明确不做的工作

- 不接入更多 MoE 层或更大的模型；
- 不开始 Triton / 自定义融合 INT4 GEMM 大规模重构；
- 不上多机协同和 RTX 5080；
- 不将静态预测器替换成新的复杂机器学习路由预测模型；
- 不引入额外在线重量化精度；
- 不为了追求 tokens/s 而移除正确性测试和证据链。

**一句话验收标准：能从干净提交复现四层 MoE，并用严格、可追溯的事件证明预取是否真的让专家在访问前准备完毕，以及是否缩短了真实生成延迟。**
