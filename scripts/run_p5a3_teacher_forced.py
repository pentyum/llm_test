#!/usr/bin/env python3
"""
P5a.3 Gate G2: Teacher-Forced Logits and KV Cache Parity Script
Evaluates candidate modes against official reference on FIXED identical prefixes:
  - top1_agreement
  - KL divergence: KL(p_ref || p_cand)
  - logits_max_abs, logits_rel_l2
  - reference top-1 margin
Outputs:
  - results/p5a3/quality/teacher_forced_logits.csv
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.p5a.dynamic_moe_layer import DynamicMoELayer
from src.p5a.measured_cost_policy import S4MeasuredCostPolicy
from src.p5a.generation_runner import generate_tokens

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"
REAL_EXPERTS_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"
REAL_SENSITIVITY_FILE = "/home/yaoyao/llm_test/results/p45_real_sensitivity.csv"
PROMPTS_FILE = "/home/yaoyao/llm_test/configs/p5a2/eval_prompts.jsonl"
OUT_CSV = "/home/yaoyao/llm_test/results/p5a3/quality/teacher_forced_logits.csv"

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

MAX_NEW_TOKENS = 16

def run_teacher_forced_eval():
    print("=== P5a.3 G2: Starting Teacher-Forced Logits Evaluation ===")
    assert torch.cuda.is_available(), "CUDA required"

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    # Run on all 32 prompts
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

    modes = [
        "bf16_passthrough",
        "fixed_int8",
        "fixed_int4",
        "s4_measured_cost"
    ]

    records = []

    print("\n--- Evaluating Teacher-Forced Logits per Prompt ---")
    for idx, p_item in enumerate(prompts):
        p_id = p_item["id"]
        p_text = p_item["prompt"]

        # 1. Generate reference prefix and next tokens
        model.model.layers[0].mlp = orig_layer0
        inputs = tokenizer(p_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        # Get reference generation and all step logits
        ref_tokens = input_ids.clone()
        ref_step_logits = []

        with torch.no_grad():
            out = model(ref_tokens, use_cache=True)
            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]
            next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
            ref_step_logits.append(next_logits.float().cpu())
            ref_tokens = torch.cat([ref_tokens, next_tok], dim=-1)

            for _ in range(MAX_NEW_TOKENS - 1):
                out = model(next_tok, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_logits = out.logits[:, -1, :]
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
                ref_step_logits.append(next_logits.float().cpu())
                ref_tokens = torch.cat([ref_tokens, next_tok], dim=-1)

        # 2. For each candidate mode, run with EXACT SAME reference tokens (Teacher-Forcing)
        for m_name in modes:
            cache_fresh = TieredCache(store, ram_budget_bytes=128*1024*1024, vram_budget_bytes=64*1024*1024)
            pol = S4MeasuredCostPolicy(sens_lookup, max_loss_budget=0.025) if m_name == "s4_measured_cost" else None
            dyn_layer = DynamicMoELayer(
                orig_layer0, layer_id=0, cache=cache_fresh,
                mode="dynamic_s4" if m_name == "s4_measured_cost" else m_name,
                policy=pol, dequant_backend="torch_cuda"
            )
            model.model.layers[0].mlp = dyn_layer

            # Teacher-forced feed of identical prefix
            with torch.no_grad():
                # Prefill with original prompt
                dyn_layer.current_phase = "prefill"
                out = model(input_ids, use_cache=True)
                past_kv_cand = out.past_key_values
                cand_logits = out.logits[:, -1, :].float().cpu()

                # Step 0 comparison
                ref_l = ref_step_logits[0]
                top1_ref = torch.argmax(ref_l, dim=-1).item()
                top1_cand = torch.argmax(cand_logits, dim=-1).item()

                p_ref = F.softmax(ref_l, dim=-1)
                log_p_cand = F.log_softmax(cand_logits, dim=-1)
                kl = F.kl_div(log_p_cand, p_ref, reduction="batchmean").item()
                max_abs = (ref_l - cand_logits).abs().max().item()
                rel_l2 = (torch.norm(ref_l - cand_logits) / torch.norm(ref_l)).item()

                sorted_ref, _ = torch.sort(ref_l, descending=True)
                margin = (sorted_ref[0, 0] - sorted_ref[0, 1]).item()

                records.append({
                    "prompt_id": p_id,
                    "category": p_item["category"],
                    "mode": m_name,
                    "step": 0,
                    "top1_agreement": (top1_ref == top1_cand),
                    "kl_divergence": round(kl, 6),
                    "logits_max_abs": round(max_abs, 5),
                    "logits_rel_l2": round(rel_l2, 6),
                    "ref_top1_margin": round(margin, 5)
                })

                # Subsequent steps with Teacher Forcing
                dyn_layer.current_phase = "decode"
                for step in range(1, MAX_NEW_TOKENS):
                    feed_tok = ref_tokens[:, prompt_len + step - 1 : prompt_len + step]
                    dyn_layer.current_decode_step = step
                    out = model(feed_tok, past_key_values=past_kv_cand, use_cache=True)
                    past_kv_cand = out.past_key_values
                    cand_l = out.logits[:, -1, :].float().cpu()

                    ref_l = ref_step_logits[step]
                    top1_ref = torch.argmax(ref_l, dim=-1).item()
                    top1_cand = torch.argmax(cand_l, dim=-1).item()

                    p_ref = F.softmax(ref_l, dim=-1)
                    log_p_cand = F.log_softmax(cand_l, dim=-1)
                    kl = F.kl_div(log_p_cand, p_ref, reduction="batchmean").item()
                    max_abs = (ref_l - cand_l).abs().max().item()
                    rel_l2 = (torch.norm(ref_l - cand_l) / torch.norm(ref_l)).item()

                    sorted_ref, _ = torch.sort(ref_l, descending=True)
                    margin = (sorted_ref[0, 0] - sorted_ref[0, 1]).item()

                    records.append({
                        "prompt_id": p_id,
                        "category": p_item["category"],
                        "mode": m_name,
                        "step": step,
                        "top1_agreement": (top1_ref == top1_cand),
                        "kl_divergence": round(kl, 6),
                        "logits_max_abs": round(max_abs, 5),
                        "logits_rel_l2": round(rel_l2, 6),
                        "ref_top1_margin": round(margin, 5)
                    })

        if (idx + 1) % 4 == 0 or (idx + 1) == len(prompts):
            print(f"Processed {idx + 1} / {len(prompts)} prompts...")

    model.model.layers[0].mlp = orig_layer0

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved teacher-forced logits evaluation to {OUT_CSV}")

    # Summary
    summary = df.groupby("mode").agg({
        "top1_agreement": "mean",
        "kl_divergence": ["mean", "max"],
        "logits_max_abs": "mean",
        "logits_rel_l2": "mean"
    }).round(6)
    print("\n=== Teacher-Forced Distribution-Level Parity Summary ===")
    print(summary.to_markdown())

if __name__ == "__main__":
    run_teacher_forced_eval()
