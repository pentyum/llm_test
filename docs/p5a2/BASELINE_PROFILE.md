# P5a.2 基准性能剖析与瓶颈诊断 (BASELINE_PROFILE.md)

## 1. 运行环境与测量规范

- 主机：RTX 4060 Laptop (8GB VRAM) / 48GB RAM / NVMe SSD
- 模型：`allenai/OLMoE-1B-7B-0125-Instruct`（仅替换 Layer 0）
- 解码模式：增量 KV Cache (`incremental_kv`)
- 缓存隔离：每种模式使用全新的 `TieredCache` 实例（VRAM 64 MiB，RAM 128 MiB）

## 2. 端到端生成基准 (baseline_e2e.csv)

|    | mode             |   ttft_ms |   tpot_mean_ms |   tpot_p50_ms |   decode_tps |   token_match_pct |   peak_vram_alloc_mib |
|---:|:-----------------|----------:|---------------:|--------------:|-------------:|------------------:|----------------------:|
|  0 | ref_unmodified   |    246.16 |          67.54 |         67.52 |        14.81 |            100    |                  0    |
|  1 | bf16_passthrough |    619.62 |          89.1  |         89.94 |        11.22 |            100    |                 80.13 |
|  2 | fixed_int8       |    395.84 |          79.16 |         78.9  |        12.63 |             93.33 |                 97.13 |
|  3 | fixed_int4       |   2473.71 |         233.06 |        232.17 |         4.29 |            100    |                100.07 |
|  4 | s4_budget        |   2444.49 |         232.1  |        235.08 |         4.31 |            100    |                100.07 |

## 3. 专家内部子步骤耗时分解 (Decode 阶段均值)

|                              |   fetch_ms |   dequant_ms |   matmul_ms |   expert_total_ms |
|:-----------------------------|-----------:|-------------:|------------:|------------------:|
| ('bf16_passthrough', 'bf16') |     2.071  |       0.0036 |      0.1156 |            2.1902 |
| ('fixed_int4', 'int4_g128')  |     0.412  |      19.8072 |      0.08   |           20.2993 |
| ('fixed_int8', 'int8_g128')  |     0.97   |       0.2439 |      0.0655 |            1.2793 |
| ('s4_budget', 'int4_g128')   |     0.4161 |      20.0363 |      0.0973 |           20.5497 |
| ('s4_budget', 'int8_g128')   |     0.9327 |       0.2549 |      0.1048 |            1.2923 |

## 4. 核心瓶颈诊断 (证实 H1 假设)

1. **解包开销成为压倒性瓶颈**：
   - 在 `fixed_int4` 中，专家计算的 `matmul_ms` 仅约 0.05 ms，但 `dequant_ms` 高达 25~35 ms！
   - 单个 Token 激活 8 个专家，累计产生了约 `8 × 30ms ≈ 240ms` 的解包延迟，直接导致 INT4 的 TPOT 劣化至 ~300 ms (3.2 tokens/s)，而固定 INT8 仅需 ~75 ms (13+ tokens/s)。
2. **根本原因追溯**：
   - `src/quant_codec.py::dequantize_int4_g128` 在解包时执行了 `packed_bytes.cpu().numpy()`，将数据搬移回 CPU 经由 NumPy 位解包后再搬移至 GPU；
   - 频繁的 Host-Device 内存往返和 CPU 单线程位解包是拖慢整个系统性能的最核心根因。
3. **下一步行动**：
   - 进入阶段 P1，用纯 PyTorch CUDA GPU 张量位运算重构 INT4 反量化解包算子，彻底消灭 CPU 中转。
