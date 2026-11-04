#!/usr/bin/env python3
"""
P3: Export Real OLMoE Expert Weights & SwiGLU Quantization Error Analysis
Processes Layer 0 experts from model-00001-of-00003.safetensors:
  - Exports BF16, INT8_G128, and INT4_G128
  - Evaluates SwiGLU output reconstruction errors (MSE, Rel L2, Cosine Sim, Max Err)
  - Saves results to results/p3_quant_error.csv & results/p3_quant_error.png
"""

import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from safetensors import safe_open

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quant_codec import quantize_int8_g128, dequantize_int8_g128, quantize_int4_g128, dequantize_int4_g128
from src.weight_store import WeightStore

SHARD_PATH = "/home/yaoyao/llm_test/models/hf_original/model-00001-of-00003.safetensors"
REAL_EXPERTS_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"
RESULTS_CSV = "/home/yaoyao/llm_test/results/p3_quant_error.csv"
RESULTS_PNG = "/home/yaoyao/llm_test/results/p3_quant_error.png"

def run_swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
    # OLMoE weights in safetensors are (out_features, in_features)
    # x: (batch, in_features)
    g_out = F.linear(x, gate)
    u_out = F.linear(x, up)
    inter = F.silu(g_out) * u_out
    out = F.linear(inter, down)
    return out

def export_and_evaluate_real_experts(num_experts_to_eval: int = 16):
    print(f"=== P3: Exporting Real Experts from {SHARD_PATH} ===")
    os.makedirs(REAL_EXPERTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    store = WeightStore(root_dir=REAL_EXPERTS_DIR)
    group_size = 128
    layer_id = 0

    records = []

    # Fix test input for SwiGLU validation
    torch.manual_seed(42)
    # 32 tokens x 2048 hidden_size
    test_x = torch.randn(32, 2048, dtype=torch.bfloat16)

    with safe_open(SHARD_PATH, framework="pt", device="cpu") as f:
        available_keys = f.keys()

        for e in range(num_experts_to_eval):
            gate_key = f"model.layers.{layer_id}.mlp.experts.{e}.gate_proj.weight"
            up_key = f"model.layers.{layer_id}.mlp.experts.{e}.up_proj.weight"
            down_key = f"model.layers.{layer_id}.mlp.experts.{e}.down_proj.weight"

            if gate_key not in available_keys:
                print(f"Skipping expert {e}: key not found in shard")
                continue

            gate_bf16 = f.get_tensor(gate_key)
            up_bf16 = f.get_tensor(up_key)
            down_bf16 = f.get_tensor(down_key)

            # 1. Save BF16
            bf16_data = {"gate_proj": gate_bf16, "up_proj": up_bf16, "down_proj": down_bf16}
            bf16_meta = {
                "source_shard": "model-00001-of-00003.safetensors",
                "dtype": "bfloat16",
                "format": "raw_bf16",
                "shapes": {"gate_proj": list(gate_bf16.shape), "up_proj": list(up_bf16.shape), "down_proj": list(down_bf16.shape)}
            }
            store.save_expert(layer_id, e, "bf16", bf16_data, bf16_meta)

            # 2. Quantize & Save INT8_G128
            g_i8, g_s8 = quantize_int8_g128(gate_bf16, group_size=group_size)
            u_i8, u_s8 = quantize_int8_g128(up_bf16, group_size=group_size)
            d_i8, d_s8 = quantize_int8_g128(down_bf16, group_size=group_size)
            i8_data = {
                "gate_proj": g_i8, "gate_scales": g_s8,
                "up_proj": u_i8, "up_scales": u_s8,
                "down_proj": d_i8, "down_scales": d_s8
            }
            i8_meta = {
                "source_shard": "model-00001-of-00003.safetensors",
                "dtype": "int8",
                "format": "int8_g128",
                "group_size": group_size,
                "shapes": {"gate_proj": list(gate_bf16.shape), "up_proj": list(up_bf16.shape), "down_proj": list(down_bf16.shape)}
            }
            store.save_expert(layer_id, e, "int8_g128", i8_data, i8_meta)

            # 3. Quantize & Save INT4_G128
            g_i4, g_s4, g_shp = quantize_int4_g128(gate_bf16, group_size=group_size)
            u_i4, u_s4, u_shp = quantize_int4_g128(up_bf16, group_size=group_size)
            d_i4, d_s4, d_shp = quantize_int4_g128(down_bf16, group_size=group_size)
            i4_data = {
                "gate_proj": g_i4, "gate_scales": g_s4, "gate_shape": g_shp,
                "up_proj": u_i4, "up_scales": u_s4, "up_shape": u_shp,
                "down_proj": d_i4, "down_scales": d_s4, "down_shape": d_shp
            }
            i4_meta = {
                "source_shard": "model-00001-of-00003.safetensors",
                "dtype": "int4",
                "format": "int4_g128",
                "group_size": group_size,
                "shapes": {"gate_proj": list(gate_bf16.shape), "up_proj": list(up_bf16.shape), "down_proj": list(down_bf16.shape)}
            }
            store.save_expert(layer_id, e, "int4_g128", i4_data, i4_meta)

            # --- Numerical Verification & Error Computation ---
            # Baseline BF16 SwiGLU forward
            out_bf16 = run_swiglu(test_x, gate_bf16, up_bf16, down_bf16)

            # Dequantize INT8 & Forward
            g_dec_i8 = dequantize_int8_g128(g_i8, g_s8, group_size)
            u_dec_i8 = dequantize_int8_g128(u_i8, u_s8, group_size)
            d_dec_i8 = dequantize_int8_g128(d_i8, d_s8, group_size)
            out_i8 = run_swiglu(test_x, g_dec_i8, u_dec_i8, d_dec_i8)

            # Dequantize INT4 & Forward
            g_dec_i4 = dequantize_int4_g128(g_i4, g_s4, g_shp, group_size)
            u_dec_i4 = dequantize_int4_g128(u_i4, u_s4, u_shp, group_size)
            d_dec_i4 = dequantize_int4_g128(d_i4, d_s4, d_shp, group_size)
            out_i4 = run_swiglu(test_x, g_dec_i4, u_dec_i4, d_dec_i4)

            # Metrics for INT8
            diff_i8 = out_i8.float() - out_bf16.float()
            rel_l2_i8 = (torch.norm(diff_i8) / torch.norm(out_bf16.float())).item()
            cos_i8 = F.cosine_similarity(out_i8.float().flatten(), out_bf16.float().flatten(), dim=0).item()
            mse_i8 = torch.mean(diff_i8 ** 2).item()
            max_err_i8 = torch.max(torch.abs(diff_i8)).item()

            # Metrics for INT4
            diff_i4 = out_i4.float() - out_bf16.float()
            rel_l2_i4 = (torch.norm(diff_i4) / torch.norm(out_bf16.float())).item()
            cos_i4 = F.cosine_similarity(out_i4.float().flatten(), out_bf16.float().flatten(), dim=0).item()
            mse_i4 = torch.mean(diff_i4 ** 2).item()
            max_err_i4 = torch.max(torch.abs(diff_i4)).item()

            records.append({
                "layer_id": layer_id,
                "expert_id": e,
                "int8_rel_l2": round(rel_l2_i8, 5),
                "int8_cosine_sim": round(cos_i8, 5),
                "int8_mse": round(mse_i8, 6),
                "int8_max_err": round(max_err_i8, 5),
                "int4_rel_l2": round(rel_l2_i4, 5),
                "int4_cosine_sim": round(cos_i4, 5),
                "int4_mse": round(mse_i4, 6),
                "int4_max_err": round(max_err_i4, 5),
            })

            print(f"[Layer {layer_id} | Expert {e:2d}] INT8 RelL2={rel_l2_i8:.4f}, CosSim={cos_i8:.4f} | INT4 RelL2={rel_l2_i4:.4f}, CosSim={cos_i4:.4f}")

    store.save_manifest()

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nReal expert quantization error table saved to {RESULTS_CSV}")

    # Plot Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df["expert_id"], df["int8_rel_l2"], "o-", label="INT8 Relative L2", color="#2a9d8f")
    plt.plot(df["expert_id"], df["int4_rel_l2"], "s-", label="INT4 Relative L2", color="#e76f51")
    plt.xlabel("Expert ID")
    plt.ylabel("Relative L2 Output Error")
    plt.title("SwiGLU Output Relative L2 Error Across Real Experts")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(df["expert_id"], df["int8_cosine_sim"], "o-", label="INT8 Cosine Sim", color="#2a9d8f")
    plt.plot(df["expert_id"], df["int4_cosine_sim"], "s-", label="INT4 Cosine Sim", color="#e76f51")
    plt.xlabel("Expert ID")
    plt.ylabel("Cosine Similarity")
    plt.title("SwiGLU Output Cosine Similarity Across Real Experts")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_PNG, dpi=200)
    print(f"Plot saved to {RESULTS_PNG}")

if __name__ == "__main__":
    export_and_evaluate_real_experts(16)
