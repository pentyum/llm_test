# P5a.1 实验环境与数据基线记录 (P5A1_ENV.md)

## 1. 软件与硬件环境

| 配置项 | 详细信息 | 备注 |
|---|---|---|
| **分支名称** | `experiment/p5a1-dynamic-switch` | 基于 `experiment/p4-5-validation` 创建 |
| **基准 Commit** | `3b1155dd4e3962b2afe146d7d242a2cb1094b29c` | P5a 初版交付 Commit |
| **主机环境** | Ubuntu 26.04.1 LTS (Resolute Raccoon), Linux 7.0.0-42-generic x86_64 | YAOYAO-TIANXUAN |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM) | PCI Bus: `0000:01:00.0`, Driver 595.104.02 |
| **CUDA Runtime**| CUDA 12.4 (nvcc 12.4.131) | 驱动支持 CUDA 13.2 |
| **系统内存** | 46 GiB RAM (~48 GB 物理内存), 63 GiB Swap | 空闲内存充裕 |
| **存储介质** | ZHITAI TiPlus7100 4TB NVMe SSD (ext4, 挂载于 `/home`) | 剩余空间 ~2.1 TB |
| **Python 环境** | Python 3.11.18 (Conda 环境: `moe-precision-lab`) | 独立环境 |
| **PyTorch** | PyTorch 2.6.0+cu124 | 官方 prebuilt wheel |
| **Transformers**| Transformers 5.22.1 | 支持 OLMoE 模型结构 |

## 2. 核心数据资产校验清单

- [x] **原始模型 (`models/hf_original/`)**：包含 `config.json`, `tokenizer.json`, `model-00001-of-00003.safetensors`, `model-00002-of-00003.safetensors`, `model-00003-of-00003.safetensors`（13 GB 完整官方权重）；
- [x] **离线预量化专家库 (`artifacts/real_experts/`)**：
  - `bf16/`：64 个专家的原始 BF16 权重文件与元数据 JSON；
  - `int8_g128/`：64 个专家的对称 INT8 权重 + scale 文件与元数据 JSON；
  - `int4_g128/`：64 个专家的双值打包 uint8 权重 + scale 文件与元数据 JSON；
- [x] **真实前向 Router 轨迹 (`artifacts/traces/trace_real_router.jsonl`)**：包含 17,536 个真实路由事件，覆盖模型全部 16 层；
- [x] **真实输入敏感度表 (`results/p45_real_sensitivity.csv`)**：覆盖 Layer 0, Layer 4, Layer 8, Layer 12 的专家真实输入 SwiGLU 误差数据；
- [x] **历史成果保护**：`results/p1_*`, `results/p2_*`, `results/p3_*`, `results/p4_*`, `results/p45_*`, `results/p5a_e2e.csv`, `REPORT.md` 均被完好保留，未发生任何覆盖。
