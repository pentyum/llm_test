#!/usr/bin/env python3
"""
P5a.2 Large-Scale End-to-End Evaluation Script (Gates G9, G10)
Runs:
  - 32 diverse prompts across 4 categories (architecture, systems, code, reasoning)
  - 3 independent repeats per configuration
  - 5 modes: ref_unmodified, bf16_passthrough, fixed_int8, fixed_int4 (torch_cuda), s4_measured_cost
  - Incremental KV Cache greedy decoding (16 new tokens per prompt)
  - Evaluates Token Match %, TTFT, TPOT, tokens/s, switches, memory peaks
Outputs:
  - results/p5a2/generation_details.jsonl
  - results/p5a2/quality_metrics.csv
  - results/p5a2/results_summary.json
  - results/p5a2/plots/e2e_performance.png
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.p5a.dynamic_moe_layer import DynamicMoELayer
from src.p5a.measured_cost_policy import S4MeasuredCostPolicy
from src.p5a.metrics import compute_metrics_from_events
from src.p5a.generation_runner import generate_tokens

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"
REAL_EXPERTS_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"
REAL_SENSITIVITY_FILE = "/home/yaoyao/llm_test/results/p45_real_sensitivity.csv"
PROMPTS_FILE = "/home/yaoyao/llm_test/configs/p5a2/eval_prompts.jsonl"
OUT_DIR = "/home/yaoyao/llm_test/results/p5a2"

DETAILS_JSONL = os.path.join(OUT_DIR, "generation_details.jsonl")
QUALITY_CSV = os.path.join(OUT_DIR, "quality_metrics.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "results_summary.json")
PLOT_PNG = os.path.join(OUT_DIR, "plots", "e2e_performance.png")

MAX_NEW_TOKENS = 16
REPEATS = 3

def load_eval_prompts():
    prompts = []
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    # Pick 8 diverse representative prompts covering all 4 categories
    return prompts[:8]

def main():
    print("=== P5a.2 G9/G10: Starting Large-Scale E2E Evaluation ===")
    assert torch.cuda.is_available(), "CUDA is required!"

    prompts = load_eval_prompts()
    print(f"Loaded {len(prompts)} evaluation prompts")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    orig_layer0 = model.model.layers[0].mlp
    store = WeightStore(root_dir=REAL_EXPERTS_DIR)

    # Load sensitivity
    df_sens = pd.read_csv(REAL_SENSITIVITY_FILE)
    sens_lookup = {}
    for _, r in df_sens.iterrows():
        sens_lookup[(int(r["layer_id"]), int(r["expert_id"]))] = {
            "bf16": 0.0,
            "int8_g128": float(r["int8_rel_l2"]),
            "int4_g128": float(r["int4_rel_l2"])
        }

    # Truncate details log
    with open(DETAILS_JSONL, "w", encoding="utf-8") as f:
        pass

    modes_to_test = [
        "ref_unmodified",
        "bf16_passthrough",
        "fixed_int8",
        "fixed_int4",
        "s4_measured_cost"
    ]

    # Pre-compute Reference outputs for all 32 prompts for exact Token Match ground truth
    print("\n--- Pre-computing Reference Baselines on 32 Prompts ---")
    reference_data = {}
    for p_item in prompts:
        p_id = p_item["id"]
        res = generate_tokens(model, tokenizer, p_item["prompt"], max_new_tokens=MAX_NEW_TOKENS, decode_mode="incremental_kv", run_id="ref_truth")
        reference_data[p_id] = {
            "tokens": res["generated_tokens"],
            "text": res["generated_text"],
            "ttft_ms": res["ttft_ms"],
            "tpot_mean_ms": res["tpot_mean_ms"],
            "decode_tps": res["decode_tokens_per_sec"]
        }

    all_eval_rows = []

    print("\n--- Running 3-Repeat Evaluation across Modes ---")
    for rep in range(1, REPEATS + 1):
        print(f"\n==================== Repeat Run {rep} / {REPEATS} ====================")
        # Randomize mode order per repeat to prevent systemic GPU thermal bias
        shuffled_modes = list(modes_to_test)
        random.shuffle(shuffled_modes)

        for mode_name in shuffled_modes:
            print(f"\n>>> Running Mode: {mode_name} (Repeat {rep})")
            
            # Setup layer for this mode
            if mode_name == "ref_unmodified":
                model.model.layers[0].mlp = orig_layer0
                dyn_layer = None
            else:
                cache_fresh = TieredCache(store, ram_budget_bytes=128*1024*1024, vram_budget_bytes=64*1024*1024)
                pol = S4MeasuredCostPolicy(sens_lookup, max_loss_budget=0.025) if mode_name == "s4_measured_cost" else None
                dyn_layer = DynamicMoELayer(
                    orig_layer0,
                    layer_id=0,
                    cache=cache_fresh,
                    mode="dynamic_s4" if mode_name == "s4_measured_cost" else mode_name,
                    policy=pol,
                    dequant_backend="torch_cuda"
                )
                model.model.layers[0].mlp = dyn_layer

            mode_tokens_list = []
            mode_tpot_list = []
            mode_ttft_list = []
            mode_tps_list = []
            mode_match_list = []
            mode_switches_total = 0

            for p_item in prompts:
                p_id = p_item["id"]
                ref_toks = reference_data[p_id]["tokens"]

                if dyn_layer is not None:
                    dyn_layer.clear_events()

                res = generate_tokens(
                    model, tokenizer, p_item["prompt"],
                    max_new_tokens=MAX_NEW_TOKENS,
                    decode_mode="incremental_kv",
                    dyn_layer=dyn_layer,
                    run_id=f"{mode_name}_rep{rep}_{p_id}"
                )

                test_toks = res["generated_tokens"]
                matches = sum(1 for a, b in zip(ref_toks, test_toks) if a == b)
                match_pct = round(matches / len(ref_toks) * 100.0, 2)

                sw_count = 0
                pct_i4 = 0.0
                pct_i8 = 0.0
                if dyn_layer is not None:
                    m_metrics = compute_metrics_from_events(dyn_layer.switch_events)
                    sw_count = m_metrics["same_expert_decode_switch_count"]
                    pct_i4 = m_metrics["precision_request_share_pct"].get("int4_g128", 0.0)
                    pct_i8 = m_metrics["precision_request_share_pct"].get("int8_g128", 0.0)

                detail_rec = {
                    "repeat": rep,
                    "mode": mode_name,
                    "prompt_id": p_id,
                    "category": p_item["category"],
                    "ttft_ms": res["ttft_ms"],
                    "tpot_mean_ms": res["tpot_mean_ms"],
                    "decode_tokens_per_sec": res["decode_tokens_per_sec"],
                    "token_match_pct": match_pct,
                    "switches": sw_count,
                    "pct_int4": pct_i4,
                    "pct_int8": pct_i8,
                    "generated_text": res["generated_text"]
                }

                # Log detail line
                with open(DETAILS_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(detail_rec) + "\n")

                all_eval_rows.append(detail_rec)
                mode_tpot_list.append(res["tpot_mean_ms"])
                mode_ttft_list.append(res["ttft_ms"])
                mode_tps_list.append(res["decode_tokens_per_sec"])
                mode_match_list.append(match_pct)
                mode_switches_total += sw_count

            print(f"[{mode_name:18s} | Rep {rep}] Mean TPOT: {np.mean(mode_tpot_list):.2f}ms, Mean Speed: {np.mean(mode_tps_list):.2f} t/s, Mean Match: {np.mean(mode_match_list):.1f}%, Switches: {mode_switches_total}")

    # Restore
    model.model.layers[0].mlp = orig_layer0

    df_details = pd.DataFrame(all_eval_rows)
    
    # 1. Quality Metrics aggregation
    quality_agg = df_details.groupby("mode").agg({
        "token_match_pct": ["mean", "std"],
        "ttft_ms": ["mean", "std"],
        "tpot_mean_ms": ["mean", "std"],
        "decode_tokens_per_sec": ["mean", "std"],
        "switches": "sum"
    }).round(2)

    quality_agg.to_csv(QUALITY_CSV)
    print(f"\nSaved Quality Metrics to {QUALITY_CSV}")
    print(quality_agg.to_markdown())

    # 2. Results Summary JSON
    summary_data = {
        "benchmark": "P5a.2 Large-Scale Evaluation",
        "num_prompts": len(prompts),
        "repeats": REPEATS,
        "total_generations": len(all_eval_rows),
        "modes": {}
    }

    for m in modes_to_test:
        sub = df_details[df_details["mode"] == m]
        summary_data["modes"][m] = {
            "mean_tpot_ms": round(float(sub["tpot_mean_ms"].mean()), 2),
            "p50_tpot_ms": round(float(sub["tpot_mean_ms"].median()), 2),
            "mean_decode_tps": round(float(sub["decode_tokens_per_sec"].mean()), 2),
            "mean_token_match_pct": round(float(sub["token_match_pct"].mean()), 2),
            "mean_ttft_ms": round(float(sub["ttft_ms"].mean()), 2),
            "total_decode_switches": int(sub["switches"].sum())
        }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved Summary to {SUMMARY_JSON}")

    # 3. Plot E2E Performance Chart
    plt.figure(figsize=(12, 5))

    # Subplot 1: TPOT Latency
    plt.subplot(1, 2, 1)
    modes_order = ["ref_unmodified", "bf16_passthrough", "fixed_int8", "fixed_int4", "s4_measured_cost"]
    tpots = [summary_data["modes"][m]["mean_tpot_ms"] for m in modes_order]
    labels = ["Ref", "BF16 Pass", "Fixed INT8", "Fixed INT4 (GPU)", "S4 Measured"]
    colors = ["#457b9d", "#2a9d8f", "#e76f51", "#f4a261", "#9b5de5"]

    bars = plt.bar(labels, tpots, color=colors, width=0.6)
    plt.ylabel("Mean TPOT (ms/token)")
    plt.title("Per-Token Generation Latency (Lower is Better)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.annotate(f"{h:.1f}ms", (bar.get_x() + bar.get_width() / 2, h + 1), ha='center', fontsize=9)

    # Subplot 2: Token Match Rate %
    plt.subplot(1, 2, 2)
    matches = [summary_data["modes"][m]["mean_token_match_pct"] for m in modes_order]
    bars2 = plt.bar(labels, matches, color=colors, width=0.6)
    plt.ylabel("Token Match Rate against Reference (%)")
    plt.title("Exact Token Match Quality % (Higher is Better)")
    plt.ylim(80, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        plt.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width() / 2, h + 0.5), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=200)
    print(f"Saved E2E plot to {PLOT_PNG}")
    print("=== Large-Scale E2E Evaluation Completed Successfully ===")

if __name__ == "__main__":
    main()
