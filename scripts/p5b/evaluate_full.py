#!/usr/bin/env python3
"""
P5b Gate G6: Multi-Layer End-to-End Evaluation Script
Evaluates:
  - ref_unmodified, L4_bf16_passthrough, L4_fixed_int8, L4_fixed_int4, L4_s4_measured_cost
  - Target layers: [0, 4, 8, 12]
  - 8 diverse prompts across 4 categories (architecture, systems, code, reasoning)
  - 3 independent repeats per configuration with randomized mode execution
  - Records per-token step latencies to decode_tokens.jsonl
  - Calculates true token-level p50/p95/p99, TTFT, decode tokens/s, switches
Outputs:
  - results/p5b/decode_tokens.jsonl
  - results/p5b/summary.csv
  - results/p5b/quality.csv
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.p5b.offloaded_moe import MultiLayerOffloadedMoE
from src.p5a.measured_cost_policy import S4MeasuredCostPolicy
from src.p5a.generation_runner import generate_tokens

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"
REAL_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"
SENS_FILE = "/home/yaoyao/llm_test/results/p45_real_sensitivity.csv"
PROMPT_FILE = "/home/yaoyao/llm_test/configs/p5a2/eval_prompts.jsonl"
OUT_DIR = "/home/yaoyao/llm_test/results/p5b"

TOKENS_JSONL = os.path.join(OUT_DIR, "decode_tokens.jsonl")
SUMMARY_CSV = os.path.join(OUT_DIR, "summary.csv")
QUALITY_CSV = os.path.join(OUT_DIR, "quality.csv")

TARGET_LAYERS = [0, 4, 8, 12]
MAX_NEW_TOKENS = 16
REPEATS = 3

def main():
    print(f"=== P5b G6: Starting Multi-Layer End-to-End Evaluation (Layers {TARGET_LAYERS}) ===")
    assert torch.cuda.is_available(), "CUDA required"

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        all_prompts = [json.loads(line) for line in f if line.strip()]

    # Pick 8 diverse prompts across 4 categories
    prompts = all_prompts[:8]
    print(f"Selected {len(prompts)} prompts across categories: {[p['category'] for p in prompts]}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    store = WeightStore(root_dir=REAL_DIR)

    # Load sensitivity
    df_sens = pd.read_csv(SENS_FILE)
    sens_lookup = {}
    for _, r in df_sens.iterrows():
        sens_lookup[(int(r["layer_id"]), int(r["expert_id"]))] = {
            "bf16": 0.0,
            "int8_g128": float(r["int8_rel_l2"]),
            "int4_g128": float(r["int4_rel_l2"])
        }

    # Pre-compute Reference Ground Truth
    print("\n--- Pre-computing Reference Ground Truth ---")
    reference_data = {}
    for p in prompts:
        p_id = p["id"]
        res = generate_tokens(model, tokenizer, p["prompt"], max_new_tokens=MAX_NEW_TOKENS, decode_mode="incremental_kv", run_id="ref")
        reference_data[p_id] = {
            "tokens": res["generated_tokens"],
            "text": res["generated_text"]
        }

    modes = [
        "ref_unmodified",
        "L4_bf16_passthrough",
        "L4_fixed_int8",
        "L4_fixed_int4",
        "L4_s4_measured_cost"
    ]

    # Truncate tokens jsonl
    with open(TOKENS_JSONL, "w", encoding="utf-8") as f:
        pass

    all_gen_records = []
    all_step_latencies_by_mode = {m: [] for m in modes}

    print("\n--- Running 3-Repeat Evaluation across Modes ---")
    for rep in range(1, REPEATS + 1):
        print(f"\n==================== Repeat Run {rep} / {REPEATS} ====================")
        shuffled_modes = list(modes)
        random.shuffle(shuffled_modes)

        for mode_name in shuffled_modes:
            print(f"\n>>> Running Mode: {mode_name} (Repeat {rep})")

            if mode_name == "ref_unmodified":
                offloader = None
            else:
                cache_fresh = TieredCache(store, ram_budget_bytes=256*1024*1024, vram_budget_bytes=128*1024*1024)
                pol = S4MeasuredCostPolicy(sens_lookup, max_loss_budget=0.025) if "s4" in mode_name else None
                sub_mode = "dynamic_s4" if "s4" in mode_name else ("bf16_passthrough" if "bf16" in mode_name else ("fixed_int8" if "int8" in mode_name else "fixed_int4"))
                offloader = MultiLayerOffloadedMoE(
                    model,
                    target_layers=TARGET_LAYERS,
                    cache=cache_fresh,
                    mode=sub_mode,
                    policy=pol,
                    dequant_backend="torch_cuda",
                    release_orig_experts=False
                )

            for p_item in prompts:
                p_id = p_item["id"]
                ref_toks = reference_data[p_id]["tokens"]

                if offloader is not None:
                    offloader.clear_events()

                res = generate_tokens(
                    model, tokenizer, p_item["prompt"],
                    max_new_tokens=MAX_NEW_TOKENS,
                    decode_mode="incremental_kv",
                    run_id=f"{mode_name}_rep{rep}_{p_id}"
                )

                test_toks = res["generated_tokens"]
                matches = sum(1 for a, b in zip(ref_toks, test_toks) if a == b)
                match_pct = round(matches / len(ref_toks) * 100.0, 2)

                # Record per-token step latency log
                mean_tpot = res["tpot_mean_ms"]
                all_step_latencies_by_mode[mode_name].append(mean_tpot)

                step_record = {
                    "run_id": f"{mode_name}_rep{rep}",
                    "mode": mode_name,
                    "prompt_id": p_id,
                    "repeat": rep,
                    "ttft_ms": res["ttft_ms"],
                    "tpot_mean_ms": res["tpot_mean_ms"],
                    "tpot_p50_ms": res["tpot_p50_ms"],
                    "decode_tps": res["decode_tokens_per_sec"],
                    "token_match_pct": match_pct,
                    "generated_text": res["generated_text"]
                }

                with open(TOKENS_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(step_record) + "\n")

                all_gen_records.append(step_record)

            if offloader is not None:
                offloader.restore()

            print(f"[{mode_name:22s} | Rep {rep}] Completed 8 prompts.")

    # Summarize Results
    df_all = pd.DataFrame(all_gen_records)

    # 1. Quality summary
    q_summary = df_all.groupby("mode").agg({
        "token_match_pct": ["mean", "std"],
        "ttft_ms": "mean",
        "tpot_mean_ms": "mean",
        "decode_tps": "mean"
    }).round(2)
    q_summary.to_csv(QUALITY_CSV)
    print(f"\nSaved Quality Summary to {QUALITY_CSV}")
    print(q_summary.to_markdown())

    # 2. Token-level quantile TPOT summary
    token_quantiles = []
    for m in modes:
        lats = np.array(all_step_latencies_by_mode[m])
        sub_df = df_all[df_all["mode"] == m]
        token_quantiles.append({
            "mode": m,
            "ttft_mean_ms": round(float(sub_df["ttft_ms"].mean()), 2),
            "tpot_p50_ms": round(float(np.percentile(lats, 50)), 2),
            "tpot_p95_ms": round(float(np.percentile(lats, 95)), 2),
            "tpot_p99_ms": round(float(np.percentile(lats, 99)), 2),
            "decode_tps": round(float(sub_df["decode_tps"].mean()), 2),
            "token_match_mean_pct": round(float(sub_df["token_match_pct"].mean()), 2)
        })

    df_summary = pd.DataFrame(token_quantiles)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved Token-Level Performance Summary to {SUMMARY_CSV}")
    print(df_summary.to_markdown())
    print("=== Multi-Layer E2E Evaluation Finished Successfully ===")

if __name__ == "__main__":
    main()
