#!/usr/bin/env python3
"""
P5a.3 Gate G3: Full 32-Prompt x 3 Repeats x 5 Modes (480 Generations) Evaluation
Evaluates:
  - ref_unmodified, bf16_passthrough, fixed_int8, fixed_int4, s4_measured_cost
  - 32 diverse prompts across 4 categories (architecture, systems, code, reasoning)
  - 3 independent repeats per configuration
  - Exact token match, TTFT, token-level TPOT (p50/p95/p99), decode tokens/s
  - Switch count and precision distribution
Outputs:
  - results/p5a3/quality/free_generation.jsonl
  - results/p5a3/performance/token_latency.jsonl
  - results/p5a3/quality/summary.csv
"""

import os
import sys
import json
import time
import random
import argparse
import yaml
import numpy as np
import pandas as pd
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

def parse_args():
    parser = argparse.ArgumentParser(description="P5a.3 Generation Evaluation")
    parser.add_argument("--config", type=str, default="configs/p5a3/default.yaml", help="Path to YAML config")
    parser.add_argument("--num-prompts", type=int, default=32, help="Number of prompts to evaluate (8 or 32)")
    parser.add_argument("--repeats", type=int, default=3, help="Number of independent repeats")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"=== P5a.3 G3: Starting Generation Evaluation (prompts={args.num_prompts}, repeats={args.repeats}) ===")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_dir = cfg["model_dir"]
    store_dir = cfg["expert_store_dir"]
    sens_file = cfg["sensitivity_csv"]
    prompt_file = cfg["prompt_file"]
    out_dir = cfg["output_dir"]
    max_new_tokens = cfg.get("max_new_tokens", 16)
    loss_budget = cfg.get("schedule_quality_budget", 0.025)

    quality_dir = os.path.join(out_dir, "quality")
    perf_dir = os.path.join(out_dir, "performance")
    os.makedirs(quality_dir, exist_ok=True)
    os.makedirs(perf_dir, exist_ok=True)

    free_gen_jsonl = os.path.join(quality_dir, "free_generation.jsonl")
    token_lat_jsonl = os.path.join(perf_dir, "token_latency.jsonl")

    # Load prompts
    all_prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_prompts.append(json.loads(line))

    selected_prompts = all_prompts[:args.num_prompts]
    print(f"Selected {len(selected_prompts)} prompts across categories: {[p['category'] for p in selected_prompts[:4]]}...")

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    orig_layer0 = model.model.layers[0].mlp
    store = WeightStore(root_dir=store_dir)

    # Load sensitivity
    df_sens = pd.read_csv(sens_file)
    sens_lookup = {}
    for _, r in df_sens.iterrows():
        sens_lookup[(int(r["layer_id"]), int(r["expert_id"]))] = {
            "bf16": 0.0,
            "int8_g128": float(r["int8_rel_l2"]),
            "int4_g128": float(r["int4_rel_l2"])
        }

    modes = cfg.get("modes", ["ref_unmodified", "bf16_passthrough", "fixed_int8", "fixed_int4", "s4_measured_cost"])

    # 1. Precompute Reference generation for ground truth token matching
    print("\n--- Precomputing Reference Ground Truth for exact matching ---")
    reference_data = {}
    for p_item in selected_prompts:
        p_id = p_item["id"]
        res = generate_tokens(model, tokenizer, p_item["prompt"], max_new_tokens=max_new_tokens, decode_mode="incremental_kv", run_id="ref_truth")
        reference_data[p_id] = {
            "tokens": res["generated_tokens"],
            "text": res["generated_text"]
        }

    total_runs = len(selected_prompts) * args.repeats * len(modes)
    print(f"\nTotal planned generations: {total_runs} (prompts={len(selected_prompts)} x repeats={args.repeats} x modes={len(modes)})")

    # Clear previous logs if new run
    with open(free_gen_jsonl, "w", encoding="utf-8") as f:
        pass
    with open(token_lat_jsonl, "w", encoding="utf-8") as f:
        pass

    completed_runs = 0
    all_summary_rows = []
    all_token_step_latencies = []

    for rep in range(1, args.repeats + 1):
        print(f"\n==================== Repeat {rep} / {args.repeats} ====================")
        shuffled_modes = list(modes)
        random.shuffle(shuffled_modes)

        for mode_name in shuffled_modes:
            if mode_name == "ref_unmodified":
                model.model.layers[0].mlp = orig_layer0
                dyn_layer = None
            else:
                cache_fresh = TieredCache(store, ram_budget_bytes=128*1024*1024, vram_budget_bytes=64*1024*1024)
                pol = S4MeasuredCostPolicy(sens_lookup, max_loss_budget=loss_budget) if mode_name == "s4_measured_cost" else None
                dyn_layer = DynamicMoELayer(
                    orig_layer0, layer_id=0, cache=cache_fresh,
                    mode="dynamic_s4" if mode_name == "s4_measured_cost" else mode_name,
                    policy=pol, dequant_backend="torch_cuda"
                )
                model.model.layers[0].mlp = dyn_layer

            for p_item in selected_prompts:
                p_id = p_item["id"]
                ref_toks = reference_data[p_id]["tokens"]

                if dyn_layer is not None:
                    dyn_layer.clear_events()

                t_start = time.perf_counter()
                res = generate_tokens(
                    model, tokenizer, p_item["prompt"],
                    max_new_tokens=max_new_tokens,
                    decode_mode="incremental_kv",
                    dyn_layer=dyn_layer,
                    run_id=f"{mode_name}_rep{rep}_{p_id}"
                )
                wall_time = (time.perf_counter() - t_start) * 1000.0

                test_toks = res["generated_tokens"]
                matches = sum(1 for a, b in zip(ref_toks, test_toks) if a == b)
                match_pct = round(matches / len(ref_toks) * 100.0, 2)
                first_div_pos = None
                for idx, (a, b) in enumerate(zip(ref_toks, test_toks)):
                    if a != b:
                        first_div_pos = idx
                        break

                sw_count = 0
                pct_i4 = 0.0
                pct_i8 = 0.0
                pct_bf16 = 0.0
                if dyn_layer is not None:
                    m_metrics = compute_metrics_from_events(dyn_layer.switch_events)
                    sw_count = m_metrics["same_expert_decode_switch_count"]
                    pct_i4 = m_metrics["precision_request_share_pct"].get("int4_g128", 0.0)
                    pct_i8 = m_metrics["precision_request_share_pct"].get("int8_g128", 0.0)
                    pct_bf16 = m_metrics["precision_request_share_pct"].get("bf16", 0.0)

                gen_record = {
                    "repeat": rep,
                    "mode": mode_name,
                    "prompt_id": p_id,
                    "category": p_item["category"],
                    "ttft_ms": res["ttft_ms"],
                    "tpot_mean_ms": res["tpot_mean_ms"],
                    "tpot_p50_ms": res["tpot_p50_ms"],
                    "tpot_p95_ms": res["tpot_p95_ms"],
                    "decode_tokens_per_sec": res["decode_tokens_per_sec"],
                    "token_match_pct": match_pct,
                    "first_divergence_position": first_div_pos,
                    "switches": sw_count,
                    "pct_int4": pct_i4,
                    "pct_int8": pct_i8,
                    "pct_bf16": pct_bf16,
                    "generated_tokens": test_toks,
                    "generated_text": res["generated_text"],
                    "status": "SUCCESS"
                }

                with open(free_gen_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(gen_record) + "\n")

                all_summary_rows.append(gen_record)
                completed_runs += 1

                # Log step latencies for token_latency.jsonl
                if "tpot_p50_ms" in res:
                    step_lat = {
                        "mode": mode_name,
                        "repeat": rep,
                        "prompt_id": p_id,
                        "mean_step_ms": res["tpot_mean_ms"],
                        "p50_step_ms": res["tpot_p50_ms"],
                        "p95_step_ms": res["tpot_p95_ms"]
                    }
                    all_token_step_latencies.append(step_lat)
                    with open(token_lat_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(step_lat) + "\n")

            print(f"[{mode_name:18s} | Rep {rep}] Completed {len(selected_prompts)} prompts. Progress: {completed_runs}/{total_runs}")

    # Restore
    model.model.layers[0].mlp = orig_layer0

    # Summary
    df_all = pd.DataFrame(all_summary_rows)
    summary_csv = os.path.join(quality_dir, "summary.csv")
    agg = df_all.groupby("mode").agg({
        "token_match_pct": ["mean", "std"],
        "ttft_ms": ["mean", "p50" if hasattr(pd.Series, 'p50') else lambda x: np.percentile(x, 50)],
        "tpot_mean_ms": ["mean", "p50" if hasattr(pd.Series, 'p50') else lambda x: np.percentile(x, 50)],
        "decode_tokens_per_sec": "mean",
        "switches": "sum",
        "pct_int4": "mean",
        "pct_int8": "mean"
    }).round(2)

    agg.to_csv(summary_csv)
    print(f"\nSaved generation summary to {summary_csv}")
    print(agg.to_markdown())
    print(f"\nAll {completed_runs} / {total_runs} generations completed successfully!")

if __name__ == "__main__":
    main()
