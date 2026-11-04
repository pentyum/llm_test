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
