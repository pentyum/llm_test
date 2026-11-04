"""
Offline Synthetic Expert Exporter
Generates 2 layers x 32 experts with deterministic random seed.
Exports BF16, INT8_G128, and INT4_G128 versions.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.quant_codec import quantize_int8_g128, quantize_int4_g128
from src.weight_store import WeightStore

def export_synthetic_experts(num_layers: int = 2, num_experts: int = 32, seed: int = 42):
    torch.manual_seed(seed)
    store = WeightStore()
    
    hidden_size = 2048
    intermediate_size = 1024
    group_size = 128

    print(f"=== Exporting {num_layers} layers x {num_experts} experts ===")
    
    stats = {"bf16": [], "int8_g128": [], "int4_g128": []}

    for l in range(num_layers):
        for e in range(num_experts):
            # Create deterministic expert weights in BF16
            gate = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
            up = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
            down = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16)

            # 1. BF16
            bf16_data = {"gate_proj": gate, "up_proj": up, "down_proj": down}
            bf16_meta = {
                "dtype": "bfloat16",
                "format": "raw_bf16",
                "shapes": {
                    "gate_proj": list(gate.shape),
                    "up_proj": list(up.shape),
                    "down_proj": list(down.shape),
                }
            }
            store.save_expert(l, e, "bf16", bf16_data, bf16_meta)
            p_bf16 = store.get_expert_path(l, e, "bf16")
            stats["bf16"].append(os.path.getsize(p_bf16))

            # 2. INT8_G128
            g_i8, g_s8 = quantize_int8_g128(gate, group_size=group_size)
            u_i8, u_s8 = quantize_int8_g128(up, group_size=group_size)
            d_i8, d_s8 = quantize_int8_g128(down, group_size=group_size)
            int8_data = {
                "gate_proj": g_i8, "gate_scales": g_s8,
                "up_proj": u_i8, "up_scales": u_s8,
                "down_proj": d_i8, "down_scales": d_s8
            }
            int8_meta = {
                "dtype": "int8",
                "format": "int8_g128",
                "group_size": group_size,
                "shapes": {
                    "gate_proj": list(gate.shape),
                    "up_proj": list(up.shape),
                    "down_proj": list(down.shape),
                }
            }
            store.save_expert(l, e, "int8_g128", int8_data, int8_meta)
            p_i8 = store.get_expert_path(l, e, "int8_g128")
            stats["int8_g128"].append(os.path.getsize(p_i8))

            # 3. INT4_G128
            g_i4, g_s4, g_shp = quantize_int4_g128(gate, group_size=group_size)
            u_i4, u_s4, u_shp = quantize_int4_g128(up, group_size=group_size)
            d_i4, d_s4, d_shp = quantize_int4_g128(down, group_size=group_size)
            int4_data = {
                "gate_proj": g_i4, "gate_scales": g_s4, "gate_shape": g_shp,
                "up_proj": u_i4, "up_scales": u_s4, "up_shape": u_shp,
                "down_proj": d_i4, "down_scales": d_s4, "down_shape": d_shp
            }
            int4_meta = {
                "dtype": "int4",
                "format": "int4_g128",
                "group_size": group_size,
                "shapes": {
                    "gate_proj": list(gate.shape),
                    "up_proj": list(up.shape),
                    "down_proj": list(down.shape),
                }
            }
            store.save_expert(l, e, "int4_g128", int4_data, int4_meta)
            p_i4 = store.get_expert_path(l, e, "int4_g128")
            stats["int4_g128"].append(os.path.getsize(p_i4))

    store.save_manifest()
    
    print("\n=== Storage Statistics (RQ1 Analysis) ===")
    num_total = num_layers * num_experts
    for prec in ["bf16", "int8_g128", "int4_g128"]:
        sizes = stats[prec]
        avg_mb = (sum(sizes) / len(sizes)) / (1024 * 1024)
        total_mb = sum(sizes) / (1024 * 1024)
        print(f"{prec:10s}: Avg per expert = {avg_mb:.2f} MB | Total {num_total} experts = {total_mb:.2f} MB")

if __name__ == "__main__":
    export_synthetic_experts()
