#!/usr/bin/env python3
"""
P4 Precision-Aware Scheduler for MoE Multi-Precision Experts
Implements Strategies:
  - S0: Fixed INT4
  - S1: Fixed INT8
  - S2: Router-only Thresholding
  - S3: Sensitivity-aware (Router Weight x Real Expert Quant Error)
  - S4: Cache-aware (Error vs SSD/RAM Load Cost Trade-off)
Evaluates on Real OLMoE Quantization Error Table from P3.
Outputs results/p4_policy_comparison.csv & results/p4_pareto.png.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weight_store import WeightStore
from src.tiered_cache import TieredCache

P3_ERROR_CSV = "/home/yaoyao/llm_test/results/p3_quant_error.csv"
RESULTS_CSV = "/home/yaoyao/llm_test/results/p4_policy_comparison.csv"
RESULTS_PNG = "/home/yaoyao/llm_test/results/p4_pareto.png"
REAL_EXPERTS_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

class PrecisionScheduler:
    def __init__(self, error_csv_path: str = P3_ERROR_CSV):
        self.error_df = pd.read_csv(error_csv_path)
        # Build lookup table: expert_id -> {int8_err, int4_err}
        self.expert_errors = {}
        for _, row in self.error_df.iterrows():
            e = int(row["expert_id"])
            self.expert_errors[e] = {
                "bf16": 0.0,
                "int8_g128": float(row["int8_rel_l2"]),
                "int4_g128": float(row["int4_rel_l2"]),
            }

    def get_error(self, expert_id: int, precision: str) -> float:
        e = expert_id % len(self.expert_errors)
        return self.expert_errors[e].get(precision, 0.20)

    def select_precision(
        self,
        strategy: str,
        layer_id: int,
        expert_id: int,
        router_score: float,
        cache: TieredCache = None
    ) -> str:
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
            # Weight error by router importance
            e = expert_id % len(self.expert_errors)
            int4_loss = router_score * self.expert_errors[e]["int4_g128"]
            int8_loss = router_score * self.expert_errors[e]["int8_g128"]

            # If INT4 loss exceeds threshold, upgrade to INT8 or BF16
            if int4_loss > 0.12:
                return "bf16" if int8_loss > 0.01 else "int8_g128"
            elif int4_loss > 0.05:
                return "int8_g128"
            else:
                return "int4_g128"

        elif strategy == "S4_cache_aware":
            # Joint optimization: Cost = lambda_quality * Error + lambda_latency * LoadCost
            candidates = ["int4_g128", "int8_g128", "bf16"]
            best_prec = "int4_g128"
            min_cost = float("inf")

            lambda_quality = 10.0
            lambda_latency = 1.0

            for cand in candidates:
                err = router_score * self.get_error(expert_id, cand)
                # Check latency penalty based on cache location
                load_cost = 0.0
                if cache is not None:
                    key = (layer_id, expert_id, cand)
                    if key in cache.vram_cache:
                        load_cost = 0.0
                    elif key in cache.ram_cache:
                        load_cost = 0.2 if cand == "int4_g128" else (0.4 if cand.startswith("int8") else 0.8)
                    else:
                        load_cost = 1.0 if cand == "int4_g128" else (2.0 if cand.startswith("int8") else 4.0)

                total_c = lambda_quality * err + lambda_latency * load_cost
                if total_c < min_cost:
                    min_cost = total_c
                    best_prec = cand

            return best_prec
        else:
            return "int4_g128"

def run_policy_benchmark():
    print("=== P4: Running Precision-Aware Scheduler Comparison ===")
    scheduler = PrecisionScheduler()
    store = WeightStore(root_dir=REAL_EXPERTS_DIR)

    # 2,000 synthetic trace requests across 16 real experts
    np.random.seed(42)
    num_requests = 2000
    trace = []
    # Zipfian access with random router scores
    ranks = np.arange(1, 17)
    weights = 1.0 / np.power(ranks, 1.2)
    probs = weights / np.sum(weights)

    for i in range(num_requests):
        exp_id = int(np.random.choice(ranks, p=probs) - 1)
        score = float(np.random.uniform(0.05, 1.0))
        trace.append((0, exp_id, score))

    strategies = [
        "S0_fixed_int4",
        "S1_fixed_int8",
        "S2_router_only",
        "S3_sensitivity_aware",
        "S4_cache_aware"
    ]

    # Test under multiple VRAM budgets to evaluate Pareto trade-offs
    vram_budgets = [24 * 1024 * 1024, 48 * 1024 * 1024, 96 * 1024 * 1024]
    ram_budget = 128 * 1024 * 1024

    results = []

    for vb in vram_budgets:
        vb_mb = int(vb / (1024 * 1024))
        print(f"\n--- Testing VRAM Budget = {vb_mb} MiB ---")

        for strat in strategies:
            cache = TieredCache(
                store,
                ram_budget_bytes=ram_budget,
                vram_budget_bytes=vb,
                eviction_policy="cost_aware" if strat == "S4_cache_aware" else "lru",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            total_weighted_error = 0.0
            latencies = []
            prec_counts = {"int4_g128": 0, "int8_g128": 0, "bf16": 0}

            for layer_id, exp_id, score in trace:
                prec = scheduler.select_precision(strat, layer_id, exp_id, score, cache=cache)
                prec_counts[prec] += 1

                # Account weighted error
                err = score * scheduler.get_error(exp_id, prec)
                total_weighted_error += err

                # Fetch expert and track latency
                _, _, timings = cache.get_expert(layer_id, exp_id, prec)
                latencies.append(timings["total_ms"])

            stats = cache.get_stats()
            l_arr = np.array(latencies)
            mean_lat = float(np.mean(l_arr))
            p95_lat = float(np.percentile(l_arr, 95))
            avg_error = total_weighted_error / num_requests

            rec = {
                "strategy": strat,
                "vram_budget_mib": vb_mb,
                "avg_weighted_error": round(avg_error, 5),
                "mean_latency_ms": round(mean_lat, 4),
                "p95_latency_ms": round(p95_lat, 4),
                "target_hit_rate": stats["target_hit_rate"],
                "ssd_read_mb": round(stats["ssd_read_bytes"] / (1024 * 1024), 2),
                "pct_int4": round(prec_counts["int4_g128"] / num_requests * 100, 1),
                "pct_int8": round(prec_counts["int8_g128"] / num_requests * 100, 1),
                "pct_bf16": round(prec_counts["bf16"] / num_requests * 100, 1)
            }
            results.append(rec)
            print(f"[{strat:22s} | VRAM {vb_mb:2d}MB] Error={avg_error:.4f}, Latency={mean_lat:.3f}ms, TargetHit={stats['target_hit_rate']*100:.1f}%, INT4/8/BF16={rec['pct_int4']}/{rec['pct_int8']}/{rec['pct_bf16']}%")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nP4 policy comparison table saved to {RESULTS_CSV}")

    # Plot Pareto curves
    plt.figure(figsize=(18, 5))

    # Curve 1: Error vs Latency Pareto (at 48 MiB VRAM budget)
    plt.subplot(1, 3, 1)
    df_48 = df[df["vram_budget_mib"] == 48]
    for _, row in df_48.iterrows():
        plt.scatter(row["mean_latency_ms"], row["avg_weighted_error"], s=120, label=row["strategy"])
        plt.annotate(row["strategy"].split("_")[0], (row["mean_latency_ms"], row["avg_weighted_error"] + 0.002), fontsize=10, weight="bold")
    plt.xlabel("Mean Access Latency (ms)")
    plt.ylabel("Average Weighted Output Error")
    plt.title("Error vs Latency Pareto (VRAM 48 MiB)")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Curve 2: Error vs SSD Read Traffic (at 48 MiB VRAM budget)
    plt.subplot(1, 3, 2)
    for _, row in df_48.iterrows():
        plt.scatter(row["ssd_read_mb"], row["avg_weighted_error"], s=120, label=row["strategy"])
        plt.annotate(row["strategy"].split("_")[0], (row["ssd_read_mb"], row["avg_weighted_error"] + 0.002), fontsize=10, weight="bold")
    plt.xlabel("SSD Read Traffic (MB)")
    plt.ylabel("Average Weighted Output Error")
    plt.title("Error vs SSD I/O Traffic (VRAM 48 MiB)")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Curve 3: VRAM Budget vs Target Hit Rate
    plt.subplot(1, 3, 3)
    for strat in strategies:
        sub = df[df["strategy"] == strat]
        plt.plot(sub["vram_budget_mib"], sub["target_hit_rate"] * 100, "o-", label=strat)
    plt.xlabel("VRAM Budget (MiB)")
    plt.ylabel("Target Precision Hit Rate (%)")
    plt.title("VRAM Budget vs Hit Rate")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_PNG, dpi=200)
    print(f"Pareto curves saved to {RESULTS_PNG}")
    print("=== P4 Completed Successfully ===")

if __name__ == "__main__":
    run_policy_benchmark()
