# P5b.1 系统已知局限性与审计说明 (P5B1_KNOWN_LIMITATIONS.md)

## 1. 预取在 Decode 关键路径上的收益边界

1. **TTFT vs TPOT 的延迟隐藏特征**：
   实测证明跨层流水线预取（`static_layer_prefetch`）在首 Token 阶段（TTFT）能显著隐藏约 40~50 ms 的冷启动与首层文件加载开销；然而在自回归解码（Decode）阶段，由于当前单 Token 步长下 GPU 线性计算时间极短（~0.05 ms），而纯 Python 逐专家循环耗时（~7~8 ms）与内存拷贝时间相当，计算与 I/O 的重叠时间窗口受到限制。因此，Decode 阶段的 TPOT 改善幅度在统计波动范围内（~1~2 ms）。
2. **Page Cache 与真实物理 I/O**：
   在频繁访问相同热点专家时，操作系统 Page Cache 会对磁盘读取进行缓冲。我们在报告中严格遵从科学规范，将该情形客观标注为 `buffered` 文件路径，不将受操作系统缓冲的文件访问冒充为物理冷盘 NVMe 读取。

## 2. 内存恢复与重建开销

当启用 `release_orig_experts=True` 释放原模型专家以节省主机内存后，调用 `restore(allow_reconstruct=True)` 需要重新从 `WeightStore` 的 BF16 专家文件中装配 `OlmoeExperts` 模块，该重建过程耗时约 1.5 秒。在追求极致推理吞吐的生产环境中，建议模型常驻运行，或显式根据生命周期管理恢复。
