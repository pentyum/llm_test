# P5a.1 实验总结报告：真实 MoE 专家动态精度切换验证

> 分支：`experiment/p5a1-dynamic-switch`  
> 模型：`allenai/OLMoE-1B-7B-0125-Instruct`  
> 平台：NVIDIA RTX 4060 Laptop (8GB VRAM) / 48GB RAM / NVMe SSD  

## 1. 核心结论摘要

- **ForcedSwitch (确定性强制切换功能)**：【已验证 VERIFIED】。单测证明同一专家在保持完全不在线量化的前提下，能够依序从离线预存库载入 INT4→INT8→INT4 权重并完成精准前向计算。
- **RealDecodeS4Switch (真实 Decode 自主切换)**：【已验证 VERIFIED】。在增量 KV Cache 自回归文本生成中，同一个专家根据上下文动态演变的 Router 权重与显存成本，成功自主触发了真实的双精度切换（观测到 1 次切换）。
- **BF16 Passthrough 完全保真**：Layer 0 替换后，增量生成的 Token 序列与未修改官方模型达到 **100.0% 完全逐字一致**。

## 2. 端到端增量生成性能与质量对照表 (实测值来自 `t5_t6_generation.csv`)

|    | mode                |   ttft_ms |   tpot_mean_ms |   tpot_p50_ms |   decode_tps |   token_match_pct |   decode_switches |   pct_int4 |   pct_int8 |
|---:|:--------------------|----------:|---------------:|--------------:|-------------:|------------------:|------------------:|-----------:|-----------:|
|  0 | ref_unmodified      |    255.05 |          68.01 |         68.03 |        14.7  |               100 |                 0 |       0    |       0    |
|  1 | T5_bf16_passthrough |    536.72 |          89.5  |         87.55 |        11.17 |               100 |                 0 |       0    |       0    |
|  2 | T5_fixed_int8       |    395.56 |          79.56 |         78.69 |        12.57 |                95 |                 0 |       0    |     100    |
|  3 | T5_fixed_int4       |   3728.05 |         307.93 |        305.48 |         3.25 |               100 |                 0 |     100    |       0    |
|  4 | T6_s4_legacy        |   2954.11 |         302.81 |        301.52 |         3.3  |               100 |                 0 |     100    |       0    |
|  5 | T6_s4_budget        |   3176.39 |         299.85 |        306.27 |         3.33 |               100 |                 1 |      96.97 |       3.03 |

## 3. G1–G10 十大验收门禁状态

{
  "G1_branch_and_assets": "PASS",
  "G2_no_online_quantization": "PASS",
  "G3_forced_switch_verified": "PASS",
  "G4_event_schema_strict": "PASS",
  "G5_s4_decision_dual_precision": "PASS",
  "G6_real_decode_s4_switch": "PASS",
  "G7_cache_isolation": "PASS",
  "G8_kv_cache_incrementality": "PASS",
  "G9_bf16_passthrough_exact_match": "PASS",
  "G10_comprehensive_report": "PASS",
  "independent_conclusions": {
    "ForcedSwitch_verified": true,
    "RealDecodeS4Switch_verified": true,
    "Decode_switches_observed": 1,
    "BF16_token_match_pct": 100.0
  }
}

## 4. 关键发现与局限说明

1. **旧 S4 全选 INT4 的根本原因**：旧版成本公式中加载开销项（0.2 / 0.4 / 1.0）远大于 Router 加权误差项（0.005 ~ 0.02），导致未受显式质量约束的调度器无条件偏好最小体积的 INT4；新版 `S4QualityBudgetPolicy` 引入质量硬约束，成功恢复了自然的双精度自适应能力。
2. **缓存隔离与增量 KV 解码**：各策略使用独立 `TieredCache` 实例，杜绝了跨策略状态污染；引入 `past_key_values` 增量解码消除了全前缀重算的冗余开销。
3. **性能边界与计算瓶颈**：当前实现中其余 15 层在 CPU，Layer 0 在 GPU 执行反量化解包；INT4 位解包需 ~27ms，因此当前 TPOT 瓶颈主要在非原生量化 GEMM 的 Python 解包开销，后续工程若集成 Triton/CUDA 原生量化内核将迎来质的加速。
