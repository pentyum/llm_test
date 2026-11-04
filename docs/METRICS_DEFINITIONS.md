# 性能指标与测量口径定义规范 (METRICS_DEFINITIONS.md)

为了确保实验评估具有高度科学性、可解释性与可重复性，消除此前存在的口径模糊与概念混淆，特制定本规范。所有脚本、日志、CSV 和分析图表均严格遵照本规范中的定义与单位。

---

## 1. 存储层级与命中语义 (Hit Tiers & Semantics)

| 存储层级 / 指标 | 定义与判定依据 | 时间范围 | 计数分母 |
|---|---|---|---|
| **VRAM Target Hit** | 目标精度的专家张量已在 GPU 显存内 (`self.vram_cache`) | 仅包含字典查找、时间戳更新与引用返回 (通常 < 0.005 ms) | 总请求数 `requests` |
| **RAM Target Hit** | 目标精度的专家张量在 CPU 内存 (`self.ram_cache`)，但在 VRAM 未命中 | 字典查找 + CPU 到 GPU 的 PCIe 搬运耗时 (H2D)，同步完成 | 总请求数 `requests` |
| **SSD Store Load** | 目标专家在 VRAM 和 RAM 均未命中，需触发底层持久存储读取 | 文件打开 + 反序列化解析 + H2D 拷贝 + 插入缓存 | 总请求数 `requests` |
| **Page Cache Read** | 系统文件缓存已命中（热读取）的数据获取 | OS Page Cache 读取，通常耗时 0.2~0.8 ms | 底层读操作 |
| **Attempted Cold Read**| 尝试通过 `posix_fadvise(DONTNEED)` 或新建独立大样本文件测得的直接 I/O | NVMe 物理介质读取 + 内核 I/O 调度，通常耗时 0.8~2.5 ms | 底层读操作 |
| **Any Precision Hit** | 所请求专家的任意其他精度版本存在于 VRAM 或 RAM 中 | 标识专家热度局部性，与目标精度命中解耦 | 总请求数 `requests` |

---

## 2. 时间与延迟测量口径 (Latency & Timings)

所有涉及 GPU 的操作均使用 `torch.cuda.Event` 与 Host 侧 `time.perf_counter_ns()` 分别测量，并在统计前显式同步。

1. **`mean_access_ms`**：专家获取从请求发起至该专家权重在 GPU 显存完全可用并同步就绪的端到端墙钟时间算术平均值（单位：毫秒 ms）。
2. **`p50_access_ms`**：专家获取延迟的第 50 百分位数（中位数）（单位：毫秒 ms）。禁止将算术均值与中位数混淆。
3. **`p95_access_ms` / `p99_access_ms`**：专家获取延迟的第 95 与第 99 百分位数（单位：毫秒 ms），用于衡量尾部未命中与抖动惩罚。
4. **`mean_h2d_complete_ms`**：`cudaMemcpyAsync` 提交并经 `torch.cuda.synchronize()` 确认完成的主机到设备数据传输耗时（单位：毫秒 ms）。
5. **`mean_decode_ms`**：INT4 / INT8 低比特权重反量化解包为浮点张量的 GPU 内核耗时（单位：毫秒 ms）。
6. **`mean_gemm_ms`**：激活值与解包后权重进行 SwiGLU 矩阵乘法（`gate_proj`、`up_proj`、`down_proj`）的纯计算耗时（单位：毫秒 ms）。

---

## 3. 容量与显存记账口径 (Memory Accounting & Units)

1. **数据容量单位**：
   - 凡以 1024 进位的容量（`bytes / (1024 * 1024)`）统一显式标注为 **`MiB`** 或 **`GiB`**；
   - 凡以 1000 进位的网络传输或理论标称标注为 **`MB`** 或 **`GB`**。
2. **显存记账口径区分**：
   - **`vram_accounted_bytes`**：由缓存系统显式管理和记账的张量净显存，计算公式为 $\sum \text{tensor.numel()} \times \text{tensor.element\_size()}$（单位：MiB）；
   - **`torch_allocated_bytes`**：由 PyTorch CUDACachingAllocator 实际分配的张量显存（`torch.cuda.max_memory_allocated()`）；
   - **`torch_reserved_bytes`**：PyTorch 从驱动申请的显存池保留总量（`torch.cuda.max_memory_reserved()`）；
   - **`nvml_process_peak_mib`**：在子进程运行期间通过 NVML 实时轮询捕获的该 PID 专有 GPU 显存物理占用峰值。
