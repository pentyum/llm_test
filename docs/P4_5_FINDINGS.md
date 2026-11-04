# P4.5 验证修复与深化实验发现 (P4_5_FINDINGS.md)

> 分支：`experiment/p4-5-validation`  
> 基线 Commit：`ae66a4816371aee9427fb7d0458e9a4a5f412aea`  
> 实验硬件：NVIDIA GeForce RTX 4060 Laptop (8188 MiB VRAM), 48 GB RAM, ZHITAI TiPlus7100 4TB NVMe SSD  
> Python / PyTorch：Python 3.11.18, PyTorch 2.6.0+cu124 (CUDA 12.4)  

---

## 阶段 A：仓库检查、环境固定与历史结果审计

### 1. 历史数据与报告一致性审计 (B1–B8 疑点核实)

对 `REPORT.md` 与历史 CSV 文件（`results/p1_baseline.csv`、`results/p2_ablation.csv`、`results/p4_policy_comparison.csv`）进行了逐项对账，详细记录保存在 `results/p45_legacy_audit.csv`，核心审计发现如下：

1. **均值 (Mean) 与中位数 (p50) 混淆**：
   - 现象：在 `REPORT.md` 总结表中，列名标为 `p50 访问延迟`，但实际填入的数据（如 S4 的 0.094 ms、S0 的 0.016 ms）取自 `p4_policy_comparison.csv` 的 `mean_latency_ms` 字段。
   - 修复：后续指标必须严格区分 `mean_access_ms`、`p50_access_ms`、`p95_access_ms` 和 `p99_access_ms`，禁止概念混用。
2. **MB 与 MiB 单位不规范**：
   - 现象：代码中所有容量和流量计算均除以 `1024 * 1024`，属于二进制的 `MiB`，但在 CSV 表头和报告中误标为十进制的 `MB`。
   - 修复：统一在所有代码、CSV 表头与报告中显式标注 `MiB` / `GiB`。
3. **P1 显存指标采样时机错误**：
   - 现象：`scripts/run_p1_experiments.py` 中在子进程执行 `proc.wait()` 退出后才调用 `nvidia-smi` 采样 `vram_after`，导致所有行均记录为系统闲置时的显存（880.0 MiB），未能捕获全 GPU 驻留与 CPU MoE 的实际推理峰值显存。
   - 修复：必须在子进程存活期间通过 NVML 或后台采样线程轮询 PID 绑定的显存，捕获真实现存峰值。
4. **原始基线文件 SHA256 固化**：
   - 已将 P1–P4 产出的 8 个关键文件（CSV 与 PNG）的 SHA256 校验和固化至 `results/p45_baseline_manifest.json`，确保历史基线不可篡改且可追溯。

---

## 阶段 B：三级缓存修复与正确性验证

### 1. 修复的核心问题与代码改造
- **B1 零容量与超预算项旁路 (Bypass)**：当 `ram_budget=0` 或 `vram_budget=0`，或者单项所需显存/内存超过总预算时，坚决不存入字典，走临时加载执行旁路，调用结束后由 Python GC 自动回收，确保 `used_bytes == 0`，绝不发生负记账或容量溢出。
- **B1 多线程并发竞态修复**：并发环境下若多个线程同时 miss 同一个 key，在插入时先检查并扣除旧项占用的字节数，防止重复累加引发的容量超标。
- **B2 细粒度并发锁与预取去重**：使用 `self.in_flight` 注册正在预取的 key；前台若请求正在预取中的项，挂起等待其事件通知，避免触发二次 SSD I/O；SSD 读取与 CUDA 同步操作移出全局互斥锁，彻底解除并发死锁隐患。
- **B3 命中指标精准解耦**：彻底区分 `vram_target_hits`（VRAM 目标精度命中）、`ram_target_hits`（RAM 目标精度命中）与 `any_precision_hits`（任意精度命中），分母严格统一为总请求数 `requests`。

### 2. 自动化单元测试验证
- 编写并执行 `tests/test_p45_cache_correctness.py`，覆盖：
  - `test_zero_budget_bypass`: 100 次不同专家访问后 `ram_used == vram_used == 0`；
  - `test_oversized_object_bypass`: 超过预算对象安全旁路；
  - `test_exact_budget_eviction_and_multi_precision`: 多精度并存与精确 LRU 驱逐；
  - `test_prefetch_deduplication`: 预取与并发前台访问触发去重，单 key 只读一次磁盘；
  - `test_prefetch_error_recovery`: 预取文件损坏或不存在时安全降级，不破坏缓存状态；
  - `test_concurrent_multithread_access`: 10 线程 1000 次随机并发无死锁、无溢出；
  - `test_h2d_and_swiglu_correctness`: 异步 H2D 与 SwiGLU 计算结果确定性验证。
- 阶段 B 产物已保存至 `results/p45_cache_correctness.json`。测试集全部 19 项通过率 100%。

---

## 阶段 C：存储与 GPU 微基准评测实测分析

### 1. 存储层级层次延迟与带宽对比 (实测值来自 `p45_io_microbench.csv`)

| 存储 / 传输层级 | INT4 (3.10 MiB) | INT8 (6.10 MiB) | BF16 (12.00 MiB) | 有效带宽 (GB/s) |
|---|---:|---:|---:|---:|
| **VRAM 命中 (字典查找)** | 0.002 ms | 0.002 ms | 0.002 ms | 显存极速命中 (> 1 TB/s) |
| **Pinned RAM → VRAM (H2D)** | 0.265 ms | 0.500 ms | 0.960 ms | **11.4 ~ 12.2 GB/s** (PCIe 饱和) |
| **Pageable RAM → VRAM** | 0.489 ms | 0.715 ms | 1.151 ms | **6.2 ~ 10.2 GB/s** |
| **Linux Page Cache (热文件读)**| 0.609 ms | 0.925 ms | 2.502 ms | **4.7 ~ 6.4 GB/s** (内存反序列化) |
| **Attempted Cold NVMe SSD** | 3.645 ms | 2.834 ms | 4.203 ms | **0.8 ~ 2.8 GB/s** (随机小文件寻址) |

### 2. GPU 计算耗时与反量化开销拆分 (实测值来自 `p45_gpu_microbench.csv`)

| 格式 | 反量化 / 解包耗时 (`mean_decode_ms`) | 纯 SwiGLU 矩阵乘 (`mean_gemm_ms`) | 合计耗时 | 显存分配 (`peak_allocated_mib`) |
|---|---:|---:|---:|---:|
| **BF16 原生** | **0.0000 ms** (零开销) | **0.0517 ms** | **0.0517 ms** | 46.50 MiB |
| **INT8_G128** | 0.2784 ms | 0.0509 ms | 0.3293 ms | 46.50 MiB |
| **INT4_G128** | **27.0429 ms** (Python Nibble 解包) | 0.0502 ms | 28.8701 ms | 37.22 MiB |

> **关键科学结论 (证实 B7)**：  
> 1. 原生 BF16 GEMM 执行单专家仅需 **~50 μs**；  
> 2. INT8 的标量乘反量化只需 **~278 μs**，可以在 GPU 上轻量执行；  
> 3. INT4 由于缺乏硬件级原生指令，在 PyTorch 中通过位运算反量化解包耗时高达 **27.0 ms**，严重拖慢总耗时。这直接证实了为什么**严禁将“解包为 BF16 再做 GEMM”宣称为原生 INT4 推理性能**，后续数据流评估必须将**数据搬运加载耗时**与**计算内核耗时**严格解耦展示。

---
