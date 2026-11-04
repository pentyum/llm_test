# RTX 4060 上的 MoE 多精度专家预存储与分层缓存实验

本项目在消费级硬件（NVIDIA GeForce RTX 4060 Laptop 8GB VRAM + 48GB RAM + NVMe SSD）上，系统性研究与验证面向 MoE（以 OLMoE-1B-7B 为代表）的多精度离线预存储、分层缓存（SSD→RAM→VRAM）与动态精度调度策略。

## 目录结构

```text
moe-precision-lab/
├── configs/            # 实验配置文件 (default.yaml)
├── docs/               # 设计发现与基线结论 (P1_FINDINGS.md)
├── src/                # 核心实现源码
│   ├── quant_codec.py          # INT4/INT8 编解码与 Nibble 打包
│   ├── weight_store.py         # SSD 预存专家文件管理与 Checksum 校验
│   ├── tiered_cache.py         # SSD->RAM->VRAM 三级缓存与淘汰机制
│   ├── export_experts.py       # 离线合成专家打包
│   ├── export_real_experts.py  # 真实 OLMoE 专家提取与 SwiGLU 误差计算
│   ├── trace_generator.py      # 四类长序列工作负载轨迹生成
│   ├── replay.py               # 轨迹回放器与消融测试矩阵 (M0-M5)
│   └── precision_scheduler.py  # 精度感知调度器与 Pareto 权衡 (S0-S4)
├── scripts/            # 自动化环境与实验运行脚本
│   ├── setup_env.sh            # P0 环境安装与 CUDA 校验
│   ├── build_llama_cpp.sh      # P1 llama.cpp 构建与参数检查
│   ├── download_q4_gguf.sh     # P1 Q4_K_M GGUF 单文件下载
│   ├── run_baseline.py         # P1 基线性能自动化采集
│   ├── run_baseline.sh         # P1 基线评测入口
│   └── download_real_shard1.sh # P3 真实权重单 shard 下载
├── tests/              # 完整的 pytest 自动化单元测试 (12 项测试全部通过)
├── models/             # GGUF 与 safetensors 权重 (不入 Git)
├── artifacts/          # 多精度预存储专家与轨迹数据 (不入 Git)
├── logs/               # 原始日志文件 (不入 Git)
├── results/            # 实验产出的 CSV 数据与高质量对比图表
│   ├── p1_baseline.csv & p1_baseline.png
│   ├── p2_ablation.csv & p2_ablation.png
│   ├── p3_quant_error.csv & p3_quant_error.png
│   └── p4_policy_comparison.csv & p4_pareto.png
├── ENVIRONMENT.md      # 硬件平台详细参数与依赖环境记录
├── DECISIONS.md        # 关键设计与工程决策记录
└── REPORT.md           # 最终实验报告与 RQ1~RQ6 实测结论
```

## 一键复现流程

### 1. 环境准备 (P0)
```bash
bash scripts/setup_env.sh
```

### 2. llama.cpp 基线测试 (P1)
```bash
bash scripts/build_llama_cpp.sh
bash scripts/download_q4_gguf.sh
bash scripts/run_baseline.sh
```
产出：`results/p1_baseline.csv`、`results/p1_baseline.png` 与 `docs/P1_FINDINGS.md`。

### 3. 多精度专家存储与分层缓存消融 (P2)
```bash
# 运行单元测试
PYTHONPATH=. pytest -v tests/

# 离线打包与轨迹生成
python src/export_experts.py
python src/trace_generator.py

# 运行 M0-M5 消融矩阵
python src/replay.py
```
产出：`results/p2_ablation.csv` 与 `results/p2_ablation.png`。

### 4. 真实 OLMoE 专家量化与误差分析 (P3)
```bash
bash scripts/download_real_shard1.sh
python src/export_real_experts.py
```
产出：`results/p3_quant_error.csv` 与 `results/p3_quant_error.png`。

### 5. 精度感知调度策略对比与 Pareto 分析 (P4)
```bash
python src/precision_scheduler.py
```
产出：`results/p4_policy_comparison.csv` 与 `results/p4_pareto.png`。

---

## 核心实验结果速览

1. **RQ1（存储开销）**：INT8_G128 实际占用 6.10 MB（开销 1.6%），INT4_G128 实际占用 3.10 MB（开销 3.3%）；
2. **RQ2（分层延迟）**：SSD 冷读取 p50 为 0.85ms，VRAM 命中加速至 0.002ms（加速 400×）；
3. **RQ3（INT4 vs 多精度）**：相同预算下 INT4 专家容量翻倍，命中率高；但多精度能使高敏感专家输出误差从 0.21 降至 0.01；
4. **RQ4（调度策略）**：联合缓存感知调度（S4）相比纯 Router 调度延迟降低 87.8%，加权输出误差相比固定 INT4 降低 40.2%，取得最优 Pareto 前沿；
5. **RQ5（真实误差）**：真实 OLMoE 专家的 INT8 SwiGLU 输出相对误差为 1.3%（CosSim 0.9999），INT4 为 20.8%（CosSim 0.978），不同专家展现出明显的量化异质性；
6. **RQ6（端到端）**：固定 Q4 全 GPU 为 257.2 t/s，CPU MoE 缓存 0MB 为 75.3 t/s，1024MB 缓存为 38.3 t/s。真实端到端多精度内核在线替换严格标为 **未验证**。
