# P5b 系统架构与数据路径说明 (ARCHITECTURE.md)

## 1. 专家多精度存储与生命周期

在 `allenai/OLMoE-1B-7B-0125-Instruct` 模型的多层专家卸载架构中，目标 MoE 层（如 Layer 0, 4, 8, 12）的专家权重采用离线预量化方案存储在 NVMe SSD 上，完全杜绝在线推理时的重先量化开销。

### 专家唯一标识键 (CacheKey / WeightKey)
全局唯一的 4 元组标识：
`(layer_id: int, expert_id: int, precision: str, format_version: str)`

```text
       Offline Pre-quantization (Run Once)
┌────────────────────────────────────────────────────────┐
│  Official OLMoE Safetensors (Shards 1, 2, 3)          │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          BF16         INT8_G128     INT4_G128
        (12.0 MB)       (6.1 MB)      (3.1 MB)
             └─────────────┬─────────────┘
                           ▼
          artifacts/real_experts/{precision}/l{L}_e{E}.pt
```

## 2. 三级分层缓存与数据流调度 (SSD → RAM → VRAM)

```text
                     Token Hidden States x
                               │
                               ▼
                    Layer Router (Official Gate)
                               │
                               ▼
                    Top-8 Expert Routing
                               │
            Precision Policy (S4 Measured Cost Policy)
         [Quality Constraint: router_score * err <= budget]
                               │
                               ▼ Target Precision Selected
                     Lookup TieredCache
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
       [VRAM Hit]          [RAM Hit]          [SSD Miss]
      (Lookup ~0.8μs)    (Pinned H2D)       (Buffered / Cold)
            │              ~0.45 ms              ~1.16 ms
            │                  │                     │
            │                  ▼                     ▼
            │           H2D Async Stream     Read -> RAM Stage
            │                  │                     │
            └──────────────────┴─────────────────────┘
                               │
                               ▼
                   Pure GPU Tensor Dequant
                    (PyTorch CUDA ~0.5ms)
                               │
                               ▼
                       Tensor Core GEMM
                         (Pure ~0.05ms)
                               │
                               ▼
                    Accumulate Weighted Output
```

## 3. 内存所有权与真实卸载保障

1. **原模型专家内存释放**：
   在加载并替换多层 MoE 时，通过 `release_orig_experts=True` 显式解引用原模型的 `layer.mlp.experts` 并执行垃圾回收，防止在系统 RAM 中长久保留原始 BF16 专家的双份冗余占用。
2. **严格容量约束 (Hard Budget Enforcement)**：
   `TieredCache` 严守 `used_bytes <= budget_bytes`，对超额专家按访问频度执行精确 LRU 驱逐，0 预算对象执行单次计算后由显存释放，防止隐式内存溢出。
3. **流水线异步预取**：
   在执行当前层 $L_i$ 的 Tensor Core 计算时，后台线程基于层拓扑为目标后继层 $L_{i+1}$ 异步触发预取（SSD → RAM），重叠隐藏存储访问开销。
