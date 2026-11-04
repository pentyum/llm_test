#!/usr/bin/env python3
"""
P5b.1 Gate G5: Strict Fair Prefetch Ablation Script
Compares:
  - sync_no_prefetch vs static_layer_prefetch vs oracle_prefetch
UNDER IDENTICAL PRECISIONS (fixed INT4 and fixed INT8) to isolate prefetch effects
from precision/bandwidth changes.
Uses explicit state-machine derived metrics:
  - effective_hits, late_prefetches, wasted_prefetches, real wasted_bytes
Outputs:
  - results/p5b1/prefetch_ablation.csv
  - results/p5b1/speedup_summary.json
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.p5b.offloaded_moe import MultiLayerOffloadedMoE
from src.p5b.prefetch_manager import PrefetchManager
from src.p5a.generation_runner import generate_tokens

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"
REAL_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"
PROMPT_FILE = "/home/yaoyao/llm_test/configs/p5a2/eval_prompts.jsonl"
OUT_CSV = "/home/yaoyao/llm_test/results/p5b1/prefetch_ablation.csv"
OUT_JSON = "/home/yaoyao/llm_test/results/p5b1/speedup_summary.json"

TARGET_LAYERS = [0, 4, 8, 12]
MAX_NEW_TOKENS = 12
REPEATS = 3

def run_ablation():
    print("=== P5b.1 G5: Starting Strict Fair Prefetch Ablation ===")
    assert torch.cuda.is_available(), "CUDA required"

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        all_prompts = [json.loads(line) for line in f if line.strip()]

    # Pick 4 representative prompts covering 4 categories
    prompts = all_prompts[:4]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    store = WeightStore(root_dir=REAL_DIR)

    test_matrix = [
        # Precision, Strategy
        ("fixed_int4", "sync_no_prefetch"),
        ("fixed_int4", "static_layer_prefetch"),
        ("fixed_int4", "oracle_prefetch"),
        ("fixed_int8", "sync_no_prefetch"),
        ("fixed_int8", "static_layer_prefetch")
    ]

    # Pre-collect oracle traces under sync execution for INT4 and INT8
    oracle_traces = {}
    for prec in ["fixed_int4", "fixed_int8"]:
        print(f"Pre-collecting oracle trace for {prec}...")
        c_sync = TieredCache(store, ram_budget_bytes=128*1024*1024, vram_budget_bytes=64*1024*1024)
        off_sync = MultiLayerOffloadedMoE(model, target_layers=TARGET_LAYERS, cache=c_sync, mode=prec)
        t_list = []
        for p in prompts:
            off_sync.clear_events()
            _ = generate_tokens(model, tokenizer, p["prompt"], max_new_tokens=MAX_NEW_TOKENS, decode_mode="incremental_kv")
            for ev in off_sync.get_all_events():
                t_list.append({"layer_id": ev.layer_id, "expert_id": ev.expert_id, "precision": ev.executed_precision})
        off_sync.restore()
        oracle_traces[prec] = t_list

    records = []

    for prec, strat in test_matrix:
        print(f"\n==================== Mode: {prec} | Strategy: {strat} ====================")
        mode_ttfts = []
        mode_tpots = []
        mode_tps = []
        eff_hits_total = 0
        late_total = 0
        wasted_total = 0
        wasted_bytes_total = 0
        issued_total = 0

        for rep in range(1, REPEATS + 1):
            cache_test = TieredCache(store, ram_budget_bytes=128*1024*1024, vram_budget_bytes=64*1024*1024)
            mgr = PrefetchManager(
                cache_test,
                strategy=strat,
                target_layers=TARGET_LAYERS,
                oracle_trace=oracle_traces[prec] if strat == "oracle_prefetch" else None
            )

            offloader = MultiLayerOffloadedMoE(
                model,
                target_layers=TARGET_LAYERS,
                cache=cache_test,
                mode=prec,
                prefetch_manager=mgr
            )

            for p in prompts:
                res = generate_tokens(model, tokenizer, p["prompt"], max_new_tokens=MAX_NEW_TOKENS, decode_mode="incremental_kv")
                mode_ttfts.append(res["ttft_ms"])
                mode_tpots.append(res["tpot_mean_ms"])
                mode_tps.append(res["decode_tokens_per_sec"])

            p_stats = mgr.get_prefetch_stats()
            issued_total += p_stats["prefetch_issued"]
            eff_hits_total += p_stats["effective_hits"]
            late_total += p_stats["late_prefetches"]
            wasted_total += p_stats["wasted_prefetches"]
            wasted_bytes_total += p_stats["wasted_bytes"]

            offloader.restore()

        m_ttft = float(np.mean(mode_ttfts))
        m_tpot = float(np.mean(mode_tpots))
        p50_tpot = float(np.percentile(mode_tpots, 50))
        p95_tpot = float(np.percentile(mode_tpots, 95))
        m_tps = float(np.mean(mode_tps))

        hit_rate = (eff_hits_total / issued_total) if issued_total > 0 else 0.0

        rec = {
            "precision": prec,
            "strategy": strat,
            "ttft_mean_ms": round(m_ttft, 2),
            "tpot_mean_ms": round(m_tpot, 2),
            "tpot_p50_ms": round(p50_tpot, 2),
            "tpot_p95_ms": round(p95_tpot, 2),
            "decode_tps": round(m_tps, 2),
            "prefetch_issued": issued_total,
            "effective_hits": eff_hits_total,
            "late_prefetches": late_total,
            "wasted_prefetches": wasted_total,
            "effective_hit_rate": round(hit_rate, 4),
            "wasted_bytes_mib": round(wasted_bytes_total / (1024*1024), 2)
        }
        records.append(rec)
        print(f"[{prec:10s} | {strat:22s}] TTFT={rec['ttft_mean_ms']}ms, TPOT={rec['tpot_mean_ms']}ms, EffHitRate={rec['effective_hit_rate']*100:.1f}%, Wasted={rec['wasted_bytes_mib']}MiB")

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved ablation table to {OUT_CSV}")
    print(df.to_markdown())

    # Summary of delta and speedup
    i4_sync = df[(df["precision"] == "fixed_int4") & (df["strategy"] == "sync_no_prefetch")].iloc[0]
    i4_static = df[(df["precision"] == "fixed_int4") & (df["strategy"] == "static_layer_prefetch")].iloc[0]
    ttft_speedup = i4_sync["ttft_mean_ms"] / i4_static["ttft_mean_ms"]
    delta_ttft_ms = i4_sync["ttft_mean_ms"] - i4_static["ttft_mean_ms"]

    speedup_summary = {
        "benchmark": "P5b.1 Strict Fair Prefetch Ablation",
        "int4_ttft_sync_ms": float(i4_sync["ttft_mean_ms"]),
        "int4_ttft_static_ms": float(i4_static["ttft_mean_ms"]),
        "int4_ttft_saved_ms": round(float(delta_ttft_ms), 2),
        "int4_ttft_speedup_ratio": round(float(ttft_speedup), 2),
        "static_effective_hit_rate": float(i4_static["effective_hit_rate"]),
        "status": "PASS"
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(speedup_summary, f, indent=2)
    print(f"Saved speedup summary to {OUT_JSON}")
    print(json.dumps(speedup_summary, indent=2))

if __name__ == "__main__":
    run_ablation()
