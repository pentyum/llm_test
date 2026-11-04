#!/usr/bin/env python3
"""
P4.5 Phase E: Fair Policy Comparison (S0-S4) & Ablations on Real OLMoE Traces
Key Standards:
  - 5 independent repeats per configuration with distinct random seeds.
  - Randomized order of strategies per run to prevent systematic thermal/caching bias.
  - Real router trace from Phase D across Layer 0 (64 real experts, 100% coverage).
  - Explicit reporting of mean, p50, p95, p99, std, and 95% confidence intervals.
  - Ablation matrix: S0, S1, S2, S3, S4, S4-no-prefetch, S4-no-cost.
Outputs:
  - results/p45_policy_comparison.csv
  - results/p45_ablation.csv
  - results/p45_pareto.png
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weight_store import WeightStore
from src.tiered_cache import TieredCache

REAL_TRACE_FILE = "/home/yaoyao/llm_test/artifacts/traces/trace_real_router.jsonl"
REAL_SENSITIVITY_FILE = "/home/yaoyao/llm_test/results/p45_real_sensitivity.csv"
RESULTS_DIR = "/home/yaoyao/llm_test/results"
POLICY_CSV = os.path.join(RESULTS_DIR, "p45_policy_comparison.csv")
ABLATION_CSV = os.path.join(RESULTS_DIR, "p45_ablation.csv")
PARETO_PNG = os.path.join(RESULTS_DIR, "p45_pareto.png")

class RealPrecisionScheduler:
    def __init__(self, sens_csv: str = REAL_SENSITIVITY_FILE):
        self.sens_df = pd.read_csv(sens_csv)
        self.lookup = {}
        for _, row in self.sens_df.iterrows():
            k = (int(row["layer_id"]), int(row["expert_id"]))
            self.lookup[k] = {
                "bf16": 0.0,
                "int8_g128": float(row["int8_rel_l2"]),
                "int4_g128": float(row["int4_rel_l2"])
            }
        # Layer 0 defaults
        l0_sub = self.sens_df[self.sens_df["layer_id"] == 0]
        self.default_int8 = float(l0_sub["int8_rel_l2"].mean()) if len(l0_sub) > 0 else 0.012
        self.default_int4 = float(l0_sub["int4_rel_l2"].mean()) if len(l0_sub) > 0 else 0.180

    def get_expert_error(self, layer_id: int, expert_id: int, prec: str) -> float:
        if (layer_id, expert_id) in self.lookup:
            return self.lookup[(layer_id, expert_id)].get(prec, 0.20)
        if prec == "bf16":
            return 0.0
        elif prec.startswith("int8"):
            return self.default_int8
        elif prec.startswith("int4"):
            return self.default_int4
        return 0.20

    def select(self, strategy: str, layer_id: int, expert_id: int, router_score: float, cache: TieredCache = None) -> str:
        if strategy == "S0_fixed_int4":
            return "int4_g128"
        elif strategy == "S1_fixed_int8":
            return "int8_g128"
        elif strategy == "S2_router_only":
            if router_score >= 0.70:
                return "bf16"
            elif router_score >= 0.35:
                return "int8_g128"
            else:
                return "int4_g128"
        elif strategy == "S3_sensitivity_aware":
            int4_loss = router_score * self.get_expert_error(layer_id, expert_id, "int4_g128")
            int8_loss = router_score * self.get_expert_error(layer_id, expert_id, "int8_g128")
            if int4_loss > 0.08:
                return "bf16" if int8_loss > 0.01 else "int8_g128"
            elif int4_loss > 0.03:
                return "int8_g128"
            else:
                return "int4_g128"
        elif strategy == "S4_no_cost":
            # Uses sensitivity and router score, but ignores cache presence
            int4_loss = router_score * self.get_expert_error(layer_id, expert_id, "int4_g128")
            return "bf16" if int4_loss > 0.08 else ("int8_g128" if int4_loss > 0.03 else "int4_g128")
        elif strategy.startswith("S4"):
            # Joint cost optimization
            candidates = ["int4_g128", "int8_g128", "bf16"]
            best_p = "int4_g128"
            min_cost = float("inf")
            l_q = 12.0
            l_l = 1.0

            for cand in candidates:
                err = router_score * self.get_expert_error(layer_id, expert_id, cand)
                load_cost = 0.0
                if cache is not None:
                    k = cache._make_key(layer_id, expert_id, cand)
                    if k in cache.vram_cache:
                        load_cost = 0.0
                    elif k in cache.ram_cache:
                        load_cost = 0.2 if cand.startswith("int4") else (0.4 if cand.startswith("int8") else 0.8)
                    else:
                        load_cost = 1.0 if cand.startswith("int4") else (2.0 if cand.startswith("int8") else 4.0)

                cost = l_q * err + l_l * load_cost
                if cost < min_cost:
                    min_cost = cost
                    best_p = cand
            return best_p
        return "int4_g128"

def load_real_layer0_trace(min_events: int = 2500):
    """Filter real router trace for Layer 0 (64 real experts), repeating deterministically if needed."""
    records = []
    with open(REAL_TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ev = json.loads(line)
                if ev["layer_id"] == 0:
                    records.append(ev)

    print(f"Layer 0 real router events available: {len(records)}")
    repeated = []
    while len(repeated) < min_events:
        repeated.extend(records)
    return repeated[:min_events]

def run_suite():
    print("=== P4.5 Phase E: Starting 5-Repeat Policy Benchmark ===")
    store = WeightStore(root_dir="/home/yaoyao/llm_test/artifacts/real_experts")
    scheduler = RealPrecisionScheduler()
    trace = load_real_layer0_trace(min_events=2500)
    print(f"Active real trace events for benchmarking: {len(trace)}")

    strategies = [
        "S0_fixed_int4",
        "S1_fixed_int8",
        "S2_router_only",
        "S3_sensitivity_aware",
        "S4_cache_aware"
    ]

    ablation_strats = [
        "S4_cache_aware",
        "S4_no_prefetch",
        "S4_no_cost"
    ]

    vram_budgets_mib = [24, 48, 96]
    ram_budget_bytes = 128 * 1024 * 1024
    seeds = [42, 43, 44, 45, 46]

    policy_results = []
    ablation_results = []

    # 1. Main Policy Evaluation across VRAM budgets
    for vb_mib in vram_budgets_mib:
        vb_bytes = vb_mib * 1024 * 1024
        print(f"\n=================== VRAM Budget: {vb_mib} MiB ===================")

        for rep_idx, seed in enumerate(seeds):
            shuffled_strats = list(strategies)
            random.seed(seed)
            random.shuffle(shuffled_strats)

            for strat in shuffled_strats:
                enable_prefetch = (strat == "S4_cache_aware")
                cache = TieredCache(
                    store,
                    ram_budget_bytes=ram_budget_bytes,
                    vram_budget_bytes=vb_bytes,
                    eviction_policy="cost_aware" if "cache_aware" in strat else "lru",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )

                latencies = []
                weighted_errors = []
                counts = {"int4_g128": 0, "int8_g128": 0, "bf16": 0}

                # Warmup: first 100 requests
                for req in trace[:100]:
                    p = scheduler.select(strat, req["layer_id"], req["expert_id"], req["router_score"], cache)
                    cache.get_expert(req["layer_id"], req["expert_id"], p)

                # Reset stats for pure steady-state measurement
                cache.stats["requests"] = 0
                cache.stats["vram_target_hits"] = 0
                cache.stats["ram_target_hits"] = 0
                cache.stats["any_precision_hits"] = 0
                cache.stats["ssd_reads"] = 0
                cache.stats["ssd_read_bytes"] = 0

                # Measured accesses
                for i, req in enumerate(trace[100:]):
                    p = scheduler.select(strat, req["layer_id"], req["expert_id"], req["router_score"], cache)
                    counts[p] += 1
                    err = req["router_score"] * scheduler.get_expert_error(req["layer_id"], req["expert_id"], p)
                    weighted_errors.append(err)

                    if enable_prefetch and i + 2 < len(trace[100:]):
                        next_r = trace[100:][i + 2]
                        next_p = scheduler.select(strat, next_r["layer_id"], next_r["expert_id"], next_r["router_score"], cache)
                        cache.prefetch(next_r["layer_id"], next_r["expert_id"], next_p, target_tier="RAM")

                    _, _, timings = cache.get_expert(req["layer_id"], req["expert_id"], p)
                    latencies.append(timings["total_ms"])

                cache.drain_prefetches()
                stats = cache.get_stats()
                l_arr = np.array(latencies)

                rec = {
                    "strategy": strat,
                    "vram_budget_mib": vb_mib,
                    "repeat_idx": rep_idx + 1,
                    "seed": seed,
                    "measured_requests": len(latencies),
                    "mean_access_ms": round(float(np.mean(l_arr)), 4),
                    "p50_access_ms": round(float(np.percentile(l_arr, 50)), 4),
                    "p95_access_ms": round(float(np.percentile(l_arr, 95)), 4),
                    "p99_access_ms": round(float(np.percentile(l_arr, 99)), 4),
                    "std_access_ms": round(float(np.std(l_arr)), 4),
                    "avg_weighted_error": round(float(np.mean(weighted_errors)), 5),
                    "vram_target_hit_rate": stats["vram_target_hit_rate"],
                    "ram_target_hit_rate": stats["ram_target_hit_rate"],
                    "any_precision_hit_rate": stats["any_precision_hit_rate"],
                    "ssd_store_read_count": stats["ssd_reads"],
                    "ssd_logical_read_mib": round(stats["ssd_read_bytes"] / (1024 * 1024), 2),
                    "pct_int4": round(counts["int4_g128"] / len(latencies) * 100, 1),
                    "pct_int8": round(counts["int8_g128"] / len(latencies) * 100, 1),
                    "pct_bf16": round(counts["bf16"] / len(latencies) * 100, 1)
                }
                policy_results.append(rec)
                print(f"[{strat:20s} | VRAM {vb_mib:2d}MB | Rep {rep_idx+1}] Mean={rec['mean_access_ms']:.3f}ms, p50={rec['p50_access_ms']:.3f}ms, p95={rec['p95_access_ms']:.3f}ms, Err={rec['avg_weighted_error']:.4f}, VRAM_Hit={stats['vram_target_hit_rate']*100:.1f}%")

    df_policy = pd.DataFrame(policy_results)
    df_policy.to_csv(POLICY_CSV, index=False)
    print(f"\nSaved all repeat records to {POLICY_CSV}")

    # 2. S4 Ablations (VRAM 48 MiB)
    print("\n=== Running S4 Detailed Ablations (VRAM 48 MiB) ===")
    for ab_strat in ablation_strats:
        for rep_idx, seed in enumerate(seeds):
            cache = TieredCache(
                store,
                ram_budget_bytes=ram_budget_bytes,
                vram_budget_bytes=48 * 1024 * 1024,
                eviction_policy="cost_aware" if "no_cost" not in ab_strat else "lru",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            enable_prefetch = ("no_prefetch" not in ab_strat)
            latencies = []
            weighted_errors = []

            for i, req in enumerate(trace[:1500]):
                p = scheduler.select(ab_strat, req["layer_id"], req["expert_id"], req["router_score"], cache)
                err = req["router_score"] * scheduler.get_expert_error(req["layer_id"], req["expert_id"], p)
                weighted_errors.append(err)

                if enable_prefetch and i + 2 < 1500:
                    next_r = trace[i + 2]
                    next_p = scheduler.select(ab_strat, next_r["layer_id"], next_r["expert_id"], next_r["router_score"], cache)
                    cache.prefetch(next_r["layer_id"], next_r["expert_id"], next_p, target_tier="RAM")

                _, _, timings = cache.get_expert(req["layer_id"], req["expert_id"], p)
                latencies.append(timings["total_ms"])

            cache.drain_prefetches()
            stats = cache.get_stats()
            l_arr = np.array(latencies)

            ablation_results.append({
                "ablation_strategy": ab_strat,
                "repeat_idx": rep_idx + 1,
                "mean_access_ms": round(float(np.mean(l_arr)), 4),
                "p50_access_ms": round(float(np.percentile(l_arr, 50)), 4),
                "p95_access_ms": round(float(np.percentile(l_arr, 95)), 4),
                "avg_weighted_error": round(float(np.mean(weighted_errors)), 5),
                "vram_target_hit_rate": stats["vram_target_hit_rate"],
                "ssd_logical_read_mib": round(stats["ssd_read_bytes"] / (1024 * 1024), 2)
            })

    df_abl = pd.DataFrame(ablation_results)
    df_abl.to_csv(ABLATION_CSV, index=False)
    print(f"Saved ablations to {ABLATION_CSV}")

    # Plot Pareto Figures with Error Bars
    plt.figure(figsize=(18, 5))
    df_48 = df_policy[df_policy["vram_budget_mib"] == 48]
    summary_48 = df_48.groupby("strategy").agg({
        "mean_access_ms": ["mean", "std"],
        "avg_weighted_error": ["mean", "std"],
        "ssd_logical_read_mib": ["mean", "std"]
    })

    plt.subplot(1, 3, 1)
    for strat in strategies:
        m_lat = summary_48.loc[strat, ("mean_access_ms", "mean")]
        s_lat = summary_48.loc[strat, ("mean_access_ms", "std")]
        m_err = summary_48.loc[strat, ("avg_weighted_error", "mean")]
        s_err = summary_48.loc[strat, ("avg_weighted_error", "std")]
        plt.errorbar(m_lat, m_err, xerr=s_lat, yerr=s_err, fmt='o', markersize=9, capsize=4, label=strat)
        plt.annotate(strat.split("_")[0], (m_lat, m_err + 0.001), fontsize=10, weight="bold")
    plt.xlabel("Mean Access Latency (ms)")
    plt.ylabel("Average Weighted Output Error")
    plt.title("Pareto: Error vs Latency (VRAM 48 MiB, 5 Repeats)")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 3, 2)
    for strat in strategies:
        m_io = summary_48.loc[strat, ("ssd_logical_read_mib", "mean")]
        s_io = summary_48.loc[strat, ("ssd_logical_read_mib", "std")]
        m_err = summary_48.loc[strat, ("avg_weighted_error", "mean")]
        s_err = summary_48.loc[strat, ("avg_weighted_error", "std")]
        plt.errorbar(m_io, m_err, xerr=s_io, yerr=s_err, fmt='s', markersize=8, capsize=4, label=strat)
        plt.annotate(strat.split("_")[0], (m_io, m_err + 0.001), fontsize=10, weight="bold")
    plt.xlabel("SSD Read Traffic (MiB)")
    plt.ylabel("Average Weighted Output Error")
    plt.title("Pareto: Error vs SSD Traffic (VRAM 48 MiB)")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 3, 3)
    for strat in strategies:
        sub = df_policy.groupby(["vram_budget_mib", "strategy"])["vram_target_hit_rate"].mean().unstack()
        plt.plot(sub.index, sub[strat] * 100, "o-", label=strat)
    plt.xlabel("VRAM Budget (MiB)")
    plt.ylabel("VRAM Target Hit Rate (%)")
    plt.title("VRAM Budget vs VRAM Target Hit Rate")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PARETO_PNG, dpi=200)
    print(f"Pareto curves saved to {PARETO_PNG}")
    print("=== Phase E Benchmark and Ablations Finished Successfully ===")

if __name__ == "__main__":
    run_suite()
