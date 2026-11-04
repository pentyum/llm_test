#!/usr/bin/env python3
"""
P5b Gate G5: Prefetch Ablation Script on L4 Offloading
Compares:
  - sync_no_prefetch: Baseline synchronous load
  - static_layer_prefetch: Realistic layer pipeline prefetch (no future info)
  - oracle_prefetch: Theoretical upper bound using pre-collected router decisions
Outputs: results/p5b/prefetch_ablation.csv
"""

import os
import sys
import json
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
OUT_CSV = "/home/yaoyao/llm_test/results/p5b/prefetch_ablation.csv"

def run_prefetch_ablation():
    print("=== P5b G5: Running Prefetch Ablation on 4-Layer Offloading ===")
    assert torch.cuda.is_available(), "CUDA required"

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    store = WeightStore(root_dir=REAL_DIR)
    prompt = "Explain mixture-of-experts layer pipelining and memory prefetching:"
    target_layers = [0, 4, 8, 12]
    max_tokens = 12

    # Collect oracle trace first by running sync baseline
    print("Collecting oracle trace...")
    cache_sync = TieredCache(store, ram_budget_bytes=128*1024*1024, vram_budget_bytes=64*1024*1024)
    off_sync = MultiLayerOffloadedMoE(model, target_layers=target_layers, cache=cache_sync, mode="fixed_int4")
    res_sync = generate_tokens(model, tok, prompt, max_new_tokens=max_tokens, decode_mode="incremental_kv")
    
    oracle_events = []
    for ev in off_sync.get_all_events():
        oracle_events.append({"layer_id": ev.layer_id, "expert_id": ev.expert_id, "precision": ev.executed_precision})
    off_sync.restore()

    strategies = ["sync_no_prefetch", "static_layer_prefetch", "oracle_prefetch"]
    records = []

    for strat in strategies:
        print(f"\n--- Testing Prefetch Strategy: {strat} ---")
        cache_test = TieredCache(store, ram_budget_bytes=128*1024*1024, vram_budget_bytes=64*1024*1024)
        
        mgr = PrefetchManager(
            cache_test,
            strategy=strat,
            target_layers=target_layers,
            oracle_trace=oracle_events if strat == "oracle_prefetch" else None
        )

        offloader = MultiLayerOffloadedMoE(
            model,
            target_layers=target_layers,
            cache=cache_test,
            mode="fixed_int4",
            prefetch_manager=mgr
        )

        res = generate_tokens(model, tok, prompt, max_new_tokens=max_tokens, decode_mode="incremental_kv")
        offloader.restore()

        p_stats = mgr.get_prefetch_stats()
        c_stats = cache_test.get_stats()

        rec = {
            "strategy": strat,
            "ttft_ms": res["ttft_ms"],
            "tpot_mean_ms": res["tpot_mean_ms"],
            "tpot_p50_ms": res["tpot_p50_ms"],
            "decode_tps": res["decode_tokens_per_sec"],
            "prefetch_requests": p_stats["prefetch_requests"],
            "prefetch_hits": p_stats["prefetch_hits"],
            "prefetch_hit_rate": p_stats["prefetch_hit_rate"],
            "wasted_bytes_mib": round(p_stats["wasted_bytes"] / (1024*1024), 2),
            "ssd_read_bytes_mib": round(c_stats["ssd_read_bytes"] / (1024*1024), 2)
        }
        records.append(rec)
        print(f"[{strat:22s}] TTFT={rec['ttft_ms']}ms, TPOT={rec['tpot_mean_ms']}ms, HitRate={rec['prefetch_hit_rate']*100:.1f}%, Wasted={rec['wasted_bytes_mib']}MiB")

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved prefetch ablation to {OUT_CSV}")
    print(df.to_markdown())

if __name__ == "__main__":
    run_prefetch_ablation()
