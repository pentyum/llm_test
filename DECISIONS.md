# 架构与实验决策记录 (DECISIONS.md)

| 决策编号 | 决策内容 | 动因 / 原因 | 影响范围 |
|---|---|---|---|
| DEC-001 | 选定工作路径为 `/home/yaoyao/llm_test` | 位于 ZHITAI TiPlus7100 4TB NVMe SSD（挂载在 `/home`），剩余 2.1 TB 空间，高并发随机读写性能优良，满足实验契约要求 | 全局 |
| DEC-002 | 使用 Python 3.11 构建独立 Conda 环境 `moe-precision-lab` | 隔离系统环境，与现代深度学习栈兼容性最佳 | 环境与依赖 |
| DEC-003 | 安装 PyTorch 2.6.0+cu124 | 主机自带 nvcc 版本为 CUDA 12.4.131，cu124 官方 wheel 达到完美 ABI 兼容 | 训练与推理运行时 |
| DEC-004 | 离线量化与解包设计遵循契约：禁止在线 BF16 量化 | 在线阶段仅做解包/解码与 GEMM，性能与数据搬运严格解耦记录 | P2/P3/P4 核心代码 |
