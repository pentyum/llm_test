#!/usr/bin/env python3
"""
P2 Tiered Cache Trace Replayer and Ablation Study
Ablation Matrix:
  M0: Fixed INT4, No explicit cache (Cold SSD read benchmark)
  M1: Fixed INT4, LRU Cache
  M2: Fixed INT8, LRU Cache
  M3: Trace Specified Multi-Precision, LRU Cache
  M4: Trace Specified Multi-Precision, Cost-Aware Cache
  M5: Trace Specified Multi-Precision, Cost-Aware Cache + Prefetch
Evaluated on 4 workloads: uniform, zipf, bursty, mixed_precision (10,000 requests each).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.trace_generator import generate_traces

TRACE_DIR = "/home/yaoyao/llm_test/artifacts/traces"
RESULTS_CSV = "/home/yaoyao/llm_test/results/p2_ablation.csv"
RESULTS_PNG = "/home/yaoyao/llm_test/results/p2_ablation.png"

# Fixed identical budget for fair comparison across all M1-M5
# For 64 synthetic experts:
# Total INT4 size ~ 198 MB, INT8 ~ 390 MB, BF16 ~ 768 MB
# Set RAM budget = 128 MiB (~30-40% of experts), VRAM budget = 48 MiB (~10-15% of experts)
RAM_BUDGET_BYTES = 128 * 1024 * 1024
VRAM_BUDGET_BYTES = 48 * 1024 * 1024

WORKLOADS = ["uniform", "zipf", "bursty", "mixed_precision"]

ABLATION_GROUPS = [
    {"group": "M0", "prec_policy": "fixed_int4", "cache_policy": "none", "prefetch": False},
    {"group": "M1", "prec_policy": "fixed_int4", "cache_policy": "lru", "prefetch": False},
    {"group": "M2", "prec_policy": "fixed_int8", "cache_policy": "lru", "prefetch": False},
    {"group": "M3", "prec_policy": "trace_mixed", "cache_policy": "lru", "prefetch": False},
    {"group": "M4", "prec_policy": "trace_mixed", "cache_policy": "cost_aware", "prefetch": False},
    {"group": "M5", "prec_policy": "trace_mixed", "cache_policy": "cost_aware", "prefetch": True},
]

def load_trace(workload: str):
    file_path = os.path.join(TRACE_DIR, f"trace_{workload}.jsonl")
    if not os.path.exists(file_path):
        generate_traces()
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def run_ablation_experiment(store: WeightStore, ablation_cfg: dict, workload: str, trace_records: list):
    group = ablation_cfg["group"]
    prec_policy = ablation_cfg["prec_policy"]
    cache_policy = ablation_cfg["cache_policy"]
    enable_prefetch = ablation_cfg["prefetch"]

    # In M0 (no cache), budget is 0
    ram_b = 0 if cache_policy == "none" else RAM_BUDGET_BYTES
    vram_b = 0 if cache_policy == "none" else VRAM_BUDGET_BYTES

    cache = TieredCache(
        store,
        ram_budget_bytes=ram_b,
        vram_budget_bytes=vram_b,
        eviction_policy=cache_policy if cache_policy != "none" else "lru",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    latencies_ms = []
    t_start = time.perf_counter()

    # Pre-create dummy activation tensor for GEMM measurement
    dummy_x = torch.randn(1, 2048, dtype=torch.bfloat16, device=cache.device)

    total_requests = len(trace_records)

    for i, req in enumerate(trace_records):
        layer_id = req["layer_id"]
        expert_id = req["expert_id"]

        # Determine precision by policy
        if prec_policy == "fixed_int4":
            target_prec = "int4_g128"
        elif prec_policy == "fixed_int8":
            target_prec = "int8_g128"
        elif prec_policy == "trace_mixed":
            target_prec = req["precision"]
        else:
            target_prec = req["precision"]

        # Prefetch lookahead if enabled
        if enable_prefetch:
            p_dist = req.get("prefetch_distance", 2)
            if i + p_dist < total_requests:
                next_req = trace_records[i + p_dist]
                next_prec = next_req["precision"] if prec_policy == "trace_mixed" else ("int4_g128" if prec_policy == "fixed_int4" else "int8_g128")
                cache.prefetch(next_req["layer_id"], next_req["expert_id"], next_prec, target_tier="RAM")

        # Access expert
        gpu_data, meta, timings = cache.get_expert(layer_id, expert_id, target_prec)
        
        # Periodic sample GEMM computation to avoid excessive overhead
        if i % 10 == 0:
            _, dec_ms, gemm_ms = cache.forward_expert_swiglu(dummy_x, gpu_data, meta)
            timings["decode_ms"] = dec_ms
            timings["gemm_ms"] = gemm_ms

        latencies_ms.append(timings["total_ms"])

    t_end = time.perf_counter()
    total_wall_ms = (t_end - t_start) * 1000.0

    stats = cache.get_stats()
    l_arr = np.array(latencies_ms)
    p50 = float(np.percentile(l_arr, 50))
    p95 = float(np.percentile(l_arr, 95))
    p99 = float(np.percentile(l_arr, 99))
    mean_lat = float(np.mean(l_arr))

    res = {
        "group": group,
        "workload": workload,
        "prec_policy": prec_policy,
        "cache_policy": cache_policy,
        "prefetch": enable_prefetch,
        "requests": total_requests,
        "p50_ms": round(p50, 4),
        "p95_ms": round(p95, 4),
        "p99_ms": round(p99, 4),
        "mean_latency_ms": round(mean_lat, 4),
        "total_wall_s": round(total_wall_ms / 1000.0, 2),
        "vram_hit_rate": stats["vram_hit_rate"],
        "ram_hit_rate": stats["ram_hit_rate"],
        "target_precision_hit_rate": stats["target_hit_rate"],
        "ssd_read_bytes_mb": round(stats["ssd_read_bytes"] / (1024 * 1024), 2),
        "ram_to_vram_bytes_mb": round(stats["ram_to_vram_bytes"] / (1024 * 1024), 2),
        "vram_evictions": stats["vram_evictions"],
        "ram_evictions": stats["ram_evictions"]
    }
    print(f"[{group:2s} | {workload:15s}] p50={p50:.3f}ms, p95={p95:.3f}ms, target_hit={stats['target_hit_rate']*100:.1f}%, SSD_read={res['ssd_read_bytes_mb']}MB", flush=True)
    return res

def main():
    print("=== Starting P2 Multi-Precision Tiered Cache Ablation ===", flush=True)
    store = WeightStore()

    all_results = []

    for w in WORKLOADS:
        print(f"\n--- Loading workload: {w} ---", flush=True)
        trace = load_trace(w)
        for grp in ABLATION_GROUPS:
            res = run_ablation_experiment(store, grp, w, trace)
            all_results.append(res)

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nAll ablation results saved to {RESULTS_CSV}", flush=True)

    # Plot summary charts
    plt.figure(figsize=(15, 6))

    # Subplot 1: p95 latency across groups and workloads
    plt.subplot(1, 2, 1)
    pivot_p95 = df.pivot(index="group", columns="workload", values="p95_ms")
    pivot_p95.plot(kind="bar", ax=plt.gca(), colormap="viridis", width=0.8)
    plt.ylabel("p95 Latency (ms)")
    plt.title("Expert Access p95 Latency by Policy & Workload")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)

    # Subplot 2: Target precision hit rate across groups
    plt.subplot(1, 2, 2)
    pivot_hit = df.pivot(index="group", columns="workload", values="target_precision_hit_rate") * 100
    pivot_hit.plot(kind="bar", ax=plt.gca(), colormap="plasma", width=0.8)
    plt.ylabel("Hit Rate (%)")
    plt.title("Target Precision Cache Hit Rate (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(RESULTS_PNG, dpi=200)
    print(f"Ablation chart saved to {RESULTS_PNG}", flush=True)

    print("=== P2 Ablation Completed Successfully ===", flush=True)

if __name__ == "__main__":
    main()
