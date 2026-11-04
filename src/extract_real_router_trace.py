#!/usr/bin/env python3
"""
P4.5 Phase D: Extract Real OLMoE Router Traces & Real Hidden States Sensitivity
- Runs forward passes on official allenai/OLMoE-1B-7B-0125-Instruct
- Extracts real Top-k router logits, rankings, scores across all 16 layers
- Captures real MoE input hidden states
- Evaluates real input sensitivity (BF16 vs INT8_G128 vs INT4_G128) across multiple layers
Outputs:
  - artifacts/traces/trace_real_router.jsonl
  - results/p45_router_coverage.csv
  - results/p45_real_sensitivity.csv
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quant_codec import quantize_int8_g128, dequantize_int8_g128, quantize_int4_g128, dequantize_int4_g128

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"
TRACE_DIR = "/home/yaoyao/llm_test/artifacts/traces"
RESULTS_DIR = "/home/yaoyao/llm_test/results"
REAL_TRACE_FILE = os.path.join(TRACE_DIR, "trace_real_router.jsonl")
COVERAGE_CSV = os.path.join(RESULTS_DIR, "p45_router_coverage.csv")
SENSITIVITY_CSV = os.path.join(RESULTS_DIR, "p45_real_sensitivity.csv")

os.makedirs(TRACE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

PROMPTS = [
    ("prompt_qa", "Explain how mixture-of-experts language models route tokens."),
    ("prompt_summary", "Discuss the trade-offs between static integer quantization and dynamic precision caching in large language models."),
    ("prompt_code", "Write a high-performance Python class implementing an LRU cache with tiered memory eviction."),
    ("prompt_reasoning", "Solve the following step by step: A cache system with 3 tiers has latencies 2us, 200us, and 2ms. If the hit rates are 80%, 15%, and 5% respectively, calculate the average memory access time."),
    ("prompt_multiturn", "User: What are experts in MoE?\nAssistant: Experts are feed-forward networks activated conditionally.\nUser: How does the router select top-k experts?")
]

def run_swiglu_forward(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
    g_out = F.linear(x, gate)
    u_out = F.linear(x, up)
    inter = F.silu(g_out) * u_out
    out = F.linear(inter, down)
    return out

def main():
    print("=== Loading OLMoE Model and Tokenizer on CPU ===")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_experts
    top_k = model.config.num_experts_per_tok
    norm_topk = getattr(model.config, "norm_topk_prob", False)

    print(f"Model config: {num_layers} layers, {num_experts} experts/layer, Top-{top_k}, norm_topk_prob={norm_topk}")

    # Storage for hidden states per layer: layer_idx -> list of tensors
    captured_hidden_states = {l: [] for l in range(num_layers)}

    def make_pre_hook(l_idx):
        def _hook(module, args):
            # args[0] is hidden_states: shape (batch, seq_len, 2048)
            with torch.no_grad():
                captured_hidden_states[l_idx].append(args[0].detach().cpu())
        return _hook

    hooks = []
    for l in range(num_layers):
        h = model.model.layers[l].mlp.register_forward_pre_hook(make_pre_hook(l))
        hooks.append(h)

    trace_records = []
    global_req_id = 0

    print("=== Running Forward Passes on Prompts ===")
    for p_id, p_text in PROMPTS:
        inputs = tok(p_text, return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)

        router_logits_all = outputs.router_logits  # tuple of 16 tensors: each (seq_len, 64)

        for l_idx in range(num_layers):
            logits_l = router_logits_all[l_idx].float()  # (seq_len, 64)
            # Softmax routing weights
            probs = F.softmax(logits_l, dim=-1)
            topk_weights, topk_indices = torch.topk(probs, top_k, dim=-1)

            if norm_topk:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

            for tok_idx in range(seq_len):
                for rank in range(top_k):
                    exp_id = int(topk_indices[tok_idx, rank].item())
                    score = float(topk_weights[tok_idx, rank].item())

                    rec = {
                        "request_id": global_req_id,
                        "prompt_id": p_id,
                        "token_index": tok_idx,
                        "layer_id": l_idx,
                        "expert_id": exp_id,
                        "topk_rank": rank,
                        "router_score": round(score, 5),
                        "phase": "prefill",
                        "prefetch_distance": rank + 1
                    }
                    trace_records.append(rec)
                    global_req_id += 1

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"Total router access events captured: {len(trace_records)}")

    # Write trace JSONL
    with open(REAL_TRACE_FILE, "w", encoding="utf-8") as f:
        for rec in trace_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Real router trace saved to {REAL_TRACE_FILE}")

    # Coverage Analysis
    cov_data = []
    df_trace = pd.DataFrame(trace_records)
    for l_idx in range(num_layers):
        sub = df_trace[df_trace["layer_id"] == l_idx]
        active_experts = sub["expert_id"].unique()
        top_exp = sub["expert_id"].value_counts().head(3).to_dict()
        cov_data.append({
            "layer_id": l_idx,
            "total_calls": len(sub),
            "unique_experts_activated": len(active_experts),
            "coverage_pct": round(len(active_experts) / num_experts * 100, 2),
            "top_3_experts": str(top_exp)
        })

    df_cov = pd.DataFrame(cov_data)
    df_cov.to_csv(COVERAGE_CSV, index=False)
    print(f"\nRouter coverage table saved to {COVERAGE_CSV}:")
    print(df_cov.to_markdown())

    # --- Real Sensitivity Evaluation Across Multiple Layers ---
    print("\n=== Evaluating Real Expert Sensitivity on Captured Inputs ===")
    # Pick layers 0, 4, 8, 12 and sample 4 active experts per layer
    eval_layers = [0, 4, 8, 12]
    sens_records = []

    group_size = 128

    for l_idx in eval_layers:
        # Get captured real hidden states for this layer
        if not captured_hidden_states[l_idx]:
            continue
        # Concat tokens across prompts: shape (total_tokens, 2048)
        real_x_all = torch.cat(captured_hidden_states[l_idx], dim=1).squeeze(0).to(torch.bfloat16)

        mlp_layer = model.model.layers[l_idx].mlp
        sub_trace = df_trace[df_trace["layer_id"] == l_idx]
        most_active_experts = sub_trace["expert_id"].value_counts().index[:6].tolist()

        for e_id in most_active_experts:
            # In transformers OlmoeExperts, parameters are batched across experts:
            # gate_up_proj: (64, 2048, 2048) -> gate is [:1024, :], up is [1024:, :]
            # down_proj: (64, 2048, 1024)
            gate_up = mlp_layer.experts.gate_up_proj[e_id].data
            gate_bf16 = gate_up[:1024, :].clone()
            up_bf16 = gate_up[1024:, :].clone()
            down_bf16 = mlp_layer.experts.down_proj[e_id].data.clone()

            # Sample 32 real token embeddings
            sample_x = real_x_all[:32]

            # 1. Baseline BF16 SwiGLU forward
            out_bf16 = run_swiglu_forward(sample_x, gate_bf16, up_bf16, down_bf16)

            # 2. Quantize INT8 & Forward
            g_i8, g_s8 = quantize_int8_g128(gate_bf16, group_size=group_size)
            u_i8, u_s8 = quantize_int8_g128(up_bf16, group_size=group_size)
            d_i8, d_s8 = quantize_int8_g128(down_bf16, group_size=group_size)

            g_dec_i8 = dequantize_int8_g128(g_i8, g_s8, group_size)
            u_dec_i8 = dequantize_int8_g128(u_i8, u_s8, group_size)
            d_dec_i8 = dequantize_int8_g128(d_i8, d_s8, group_size)
            out_i8 = run_swiglu_forward(sample_x, g_dec_i8, u_dec_i8, d_dec_i8)

            # 3. Quantize INT4 & Forward
            g_i4, g_s4, g_shp = quantize_int4_g128(gate_bf16, group_size=group_size)
            u_i4, u_s4, u_shp = quantize_int4_g128(up_bf16, group_size=group_size)
            d_i4, d_s4, d_shp = quantize_int4_g128(down_bf16, group_size=group_size)

            g_dec_i4 = dequantize_int4_g128(g_i4, g_s4, g_shp, group_size)
            u_dec_i4 = dequantize_int4_g128(u_i4, u_s4, u_shp, group_size)
            d_dec_i4 = dequantize_int4_g128(d_i4, d_s4, d_shp, group_size)
            out_i4 = run_swiglu_forward(sample_x, g_dec_i4, u_dec_i4, d_dec_i4)

            # Metrics
            diff_i8 = out_i8.float() - out_bf16.float()
            rel_l2_i8 = (torch.norm(diff_i8) / torch.norm(out_bf16.float())).item()
            cos_i8 = F.cosine_similarity(out_i8.float().flatten(), out_bf16.float().flatten(), dim=0).item()

            diff_i4 = out_i4.float() - out_bf16.float()
            rel_l2_i4 = (torch.norm(diff_i4) / torch.norm(out_bf16.float())).item()
            cos_i4 = F.cosine_similarity(out_i4.float().flatten(), out_bf16.float().flatten(), dim=0).item()

            sens_records.append({
                "layer_id": l_idx,
                "expert_id": e_id,
                "int8_rel_l2": round(rel_l2_i8, 5),
                "int8_cosine_sim": round(cos_i8, 5),
                "int4_rel_l2": round(rel_l2_i4, 5),
                "int4_cosine_sim": round(cos_i4, 5)
            })
            print(f"[Layer {l_idx:2d} | Expert {e_id:2d}] INT8 RelL2={rel_l2_i8:.4f}, CosSim={cos_i8:.4f} | INT4 RelL2={rel_l2_i4:.4f}, CosSim={cos_i4:.4f}")

    df_sens = pd.DataFrame(sens_records)
    df_sens.to_csv(SENSITIVITY_CSV, index=False)
    print(f"\nReal sensitivity table saved to {SENSITIVITY_CSV}")

if __name__ == "__main__":
    main()
