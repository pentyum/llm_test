# 架构与实验决策记录 (DECISIONS.md)

| 决策编号 | 决策内容 | 动因 / 原因 | 影响范围 |
|---|---|---|---|
| **DEC-001** | 选定工作路径为 `/home/yaoyao/llm_test` | 位于 ZHITAI TiPlus7100 4TB NVMe SSD（挂载在 `/home`），剩余 2.1 TB 空间，高并发随机读写性能优良，满足实验契约要求 | 全局 |
| **DEC-002** | 使用 Python 3.11 构建独立 Conda 环境 `moe-precision-lab` | 隔离系统环境，与现代深度学习栈兼容性最佳 | 环境与依赖 |
| **DEC-003** | 安装 PyTorch 2.6.0+cu124 | 主机自带 nvcc 版本为 CUDA 12.4.131，cu124 官方 wheel 达到完美 ABI 兼容 | 运行时 |
| **DEC-004** | 遵循契约：禁止在线 BF16 动态量化 | 在线阶段仅做解包/解码与 GEMM，性能与数据搬运严格解耦记录 | P2/P3/P4 核心代码 |
| **DEC-005** | llama-cli 基线评测必须追加 `-st` (`--single-turn`) 与 `--perf` 参数 | 新版 llama.cpp 的 llama-cli 在 prompt 生成完毕后会默认进入多轮对话 REPL 循环等待 stdin，导致自动化脚本无响应超时 | P1 基线脚本 |
| **DEC-006** | 子进程标准输出直接重定向至磁盘日志文件，废弃内存 pipe | 当 llama-cli 开启 `-v` 详细日志时，输出量迅速突破 Linux 管道缓冲区（64 KiB），导致子进程 write 阻塞而父进程 poll 死锁；文件重定向彻底规避死锁 | 自动化评测工具 |
| **DEC-007** | P3 阶段精细化单 shard 下载 (`model-00001-of-00003.safetensors`, 4.65 GB) | 分析 `model.safetensors.index.json` 发现 Layer 0 全部 64 个专家均位于 shard 1，下载单个 shard 即可满足 >8 个真实专家的评估需求，节省 8+ GB 磁盘与带宽 | P3 数据获取 |
| **DEC-008** | P4 调度器基于实测专家误差建立联合成本函数 $\text{Cost} = \lambda_q \cdot \text{Err} + \lambda_l \cdot \text{LoadCost}$ | 避免盲目按照 Router 分数请求 12MB 的 BF16 导致 VRAM 发生雪崩式抖动；在有限显存下动态在 INT4 与 INT8 之间优化 | P4 调度算法 |
