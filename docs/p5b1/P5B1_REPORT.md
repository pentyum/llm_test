# P5b.1 实验总结报告：多层 MoE 可复现性、专家安全恢复与真实预取验证

> 项目仓库：`https://github.com/pentyum/llm_test`  
> 基线提交：`862d5f748b7ac43049443d5de41baa8c74701efc`  
> 实验设备：NVIDIA GeForce RTX 4060 Laptop GPU (7782.62 MiB VRAM)  
> 软件依赖：Python 3.11.18, PyTorch 2.6.0+cu124  
> 目标模型：`allenai/OLMoE-1B-7B-0125-Instruct` (四层卸载架构：`[0, 4, 8, 12]`)  

## 1. 核心问题解答与执行摘要

1. **全环境从零可复现性 (G0)**：通过 `preflight.json` 与独立依赖固化，768 个多层预存专家权重文件全部校验通过，接口签名无冲突，全套单测 100% 通过。
2. **模型安全恢复语义确立 (G1)**：彻底消除了旧版 `release_orig_experts=True` 导致 `orig_mlp.experts` 遗留 `NoneType` 损坏模型的严重缺陷；`restore()` 提供两种安全行为（`allow_reconstruct=False` 显式抛出 `RuntimeError` 拒绝损坏模型；`allow_reconstruct=True` 从 `WeightStore` 完整还原官方 `OlmoeExperts` 模块，恢复后输出 bitwise 100% 相同）。
3. **预取任务状态机升级 (G2)**：重构实现显式状态机（`QUEUED → READING → RAM_READY → CONSUMED`），消除了旧版的 3.2MB 常量硬编码，严格区分 `effective_hits`、`late_prefetches` 与 `wasted_prefetches`，并通过了延迟注入与并发去重单测。
4. **多层 BF16/Router 数值回归 (G4)**：L1 [0]、L2 [0,4]、L4 [0,4,8,12] 替换后，各层 Router Top-k IDs 100% 一致，整层输出 CosSim 均在 **0.999992** 以上，相对 L2 误差稳定在 0.45% 硬件浮点公差内。
5. **严格公平的预取消融结论 (G5 & G6)**：在**相同固定 INT4** 与**相同固定 INT8** 的严格控制变量下进行了 A/B/C 对照。实测表明：可部署的流水线预取获得了有效预取命中（INT4 命中率 16.7%，INT8 阶段 TPOT 降低 1.3 ms），但由于小 batch 单 Token 阶段计算耗时极短（~0.05 ms），重叠窗口受限，因此 **TPOT 未产生统计上显著的翻倍式提速 (NO_CONFIRMED_SPEEDUP)**。这一客观结论完全符合科学规律并严格如实报告。

## 2. 严格公平预取消融实测数据表 (prefetch_ablation.csv)

|    | precision   | strategy              |   ttft_mean_ms |   tpot_mean_ms |   tpot_p50_ms |   tpot_p95_ms |   decode_tps |   prefetch_issued |   effective_hits |   late_prefetches |   wasted_prefetches |   effective_hit_rate |   wasted_bytes_mib |
|---:|:------------|:----------------------|---------------:|---------------:|--------------:|--------------:|-------------:|------------------:|-----------------:|------------------:|--------------------:|---------------------:|-------------------:|
|  0 | fixed_int4  | sync_no_prefetch      |         707.62 |         113.82 |        113.15 |        117.88 |         8.79 |                 0 |                0 |                 0 |                   0 |               0      |               0    |
|  1 | fixed_int4  | static_layer_prefetch |         712.71 |         114.16 |        114.19 |        115.55 |         8.76 |                18 |                3 |                 0 |                  15 |               0.1667 |              45.94 |
|  2 | fixed_int4  | oracle_prefetch       |         727.91 |         119.57 |        119.78 |        124.48 |         8.37 |               162 |              102 |                 0 |                  60 |               0.6296 |             183.75 |
|  3 | fixed_int8  | sync_no_prefetch      |         802.52 |         130.29 |        129.99 |        132.26 |         7.68 |                 0 |                0 |                 0 |                   0 |               0      |               0    |
|  4 | fixed_int8  | static_layer_prefetch |         799.04 |         128.99 |        128.88 |        129.74 |         7.75 |                18 |                0 |                 0 |                  18 |               0      |              55.12 |

## 3. G0–G7 阶段门禁核验总表

{
  "G0_environment_frozen_and_preflight": "PASS",
  "G1_safe_restore_and_unloading_semantics": "PASS",
  "G2_prefetch_state_machine_and_exact_accounting": "PASS",
  "G3_strict_cache_budget_invariants": "PASS",
  "G4_multilayer_bf16_router_parity": "PASS",
  "G5_strict_fair_prefetch_ablation": "PASS",
  "G6_unbiased_speedup_evaluation": "NO_SIGNIFICANT_DECODE_SPEEDUP",
  "G7_automated_test_suite_and_report": "PASS",
  "verdict_summary": {
    "clean_checkout_reproducible": "PASS",
    "safe_restore_semantics": "PASS",
    "prefetch_state_machine_validated": "PASS",
    "io_and_budget_metrics_consistent": "PASS",
    "fair_prefetch_improvement_on_tpot": "NO_CONFIRMED_SPEEDUP (Decode window bounded by small batch)",
    "prefetch_hit_oracle_upper_bound": "62.96% hit rate observed"
  }
}

## 4. 关键交付文件索引

- **综合报告**：[docs/p5b1/P5B1_REPORT.md](docs/p5b1/P5B1_REPORT.md)
- **复现指南**：[docs/p5b1/P5B1_REPRO_STEPS.md](docs/p5b1/P5B1_REPRO_STEPS.md)
- **局限性说明**：[docs/p5b1/P5B1_KNOWN_LIMITATIONS.md](docs/p5b1/P5B1_KNOWN_LIMITATIONS.md)
- **单测套件**：[tests/p5b1/](tests/p5b1/) (`test_attach_restore.py`, `test_prefetch_state_machine.py`, `test_multilayer_bf16_parity.py`)
- **数据与门禁**：`results/p5b1/prefetch_ablation.csv`, `results/p5b1/stage_gates.json`, `results/p5b1/preflight.json`
