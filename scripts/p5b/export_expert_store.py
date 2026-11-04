#!/usr/bin/env python3
"""
P5b Gate G2: Multi-Layer Offline Expert Weight Export
Exports BF16, INT8_G128, and INT4_G128 for Layer 4, 8, 12 across all 64 experts.
Layer 0 is already exported; this completes L1, L2, L4 expert store.
Strictly offline: Zero online quantization calls during inference.
Outputs:
  - artifacts/real_experts/{bf16, int8_g128, int4_g128}/l{layer}_e{expert}.pt
  - artifacts/real_experts/manifest.json
"""

import os
import sys
import json
import torch
from safetensors import safe_open

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.quant_codec import quantize_int8_g128, quantize_int4_g128
from src.weight_store import WeightStore

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"
REAL_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

SHARD_MAP = {
    0: "model-00001-of-00003.safetensors",
    4: "model-00001-of-00003.safetensors",
    8: "model-00002-of-00003.safetensors",
    12: "model-00003-of-00003.safetensors"
}

def export_layers(target_layers=[4, 8, 12], group_size=128):
    print(f"=== P5b G2: Starting Offline Export for Layers {target_layers} ===")
    store = WeightStore(root_dir=REAL_DIR)

    for layer_id in target_layers:
        shard_name = SHARD_MAP[layer_id]
        shard_path = os.path.join(MODEL_DIR, shard_name)
        assert os.path.exists(shard_path), f"Missing shard: {shard_path}"

        print(f"\n--- Exporting Layer {layer_id} from {shard_name} ---")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for e in range(64):
                gate_k = f"model.layers.{layer_id}.mlp.experts.{e}.gate_proj.weight"
                up_k = f"model.layers.{layer_id}.mlp.experts.{e}.up_proj.weight"
                down_k = f"model.layers.{layer_id}.mlp.experts.{e}.down_proj.weight"

                gate = f.get_tensor(gate_k)
                up = f.get_tensor(up_k)
                down = f.get_tensor(down_k)

                # 1. BF16
                store.save_expert(layer_id, e, "bf16", {
                    "gate_proj": gate, "up_proj": up, "down_proj": down
                }, {
                    "source_shard": shard_name, "precision": "bf16", "layer_id": layer_id, "expert_id": e
                })

                # 2. INT8_G128
                g_i8, g_s8 = quantize_int8_g128(gate, group_size=group_size)
                u_i8, u_s8 = quantize_int8_g128(up, group_size=group_size)
                d_i8, d_s8 = quantize_int8_g128(down, group_size=group_size)
                store.save_expert(layer_id, e, "int8_g128", {
                    "gate_proj": g_i8, "gate_scales": g_s8,
                    "up_proj": u_i8, "up_scales": u_s8,
                    "down_proj": d_i8, "down_scales": d_s8
                }, {
                    "source_shard": shard_name, "precision": "int8_g128", "group_size": group_size, "layer_id": layer_id, "expert_id": e
                })

                # 3. INT4_G128
                g_i4, g_s4, g_shp = quantize_int4_g128(gate, group_size=group_size)
                u_i4, u_s4, u_shp = quantize_int4_g128(up, group_size=group_size)
                d_i4, d_s4, d_shp = quantize_int4_g128(down, group_size=group_size)
                store.save_expert(layer_id, e, "int4_g128", {
                    "gate_proj": g_i4, "gate_scales": g_s4, "gate_shape": g_shp,
                    "up_proj": u_i4, "up_scales": u_s4, "up_shape": u_shp,
                    "down_proj": d_i4, "down_scales": d_s4, "down_shape": d_shp
                }, {
                    "source_shard": shard_name, "precision": "int4_g128", "group_size": group_size, "layer_id": layer_id, "expert_id": e
                })

                if (e + 1) % 16 == 0:
                    print(f"Layer {layer_id}: Exported {e + 1} / 64 experts...")

    store.save_manifest()
    print("\nAll target layers exported and manifest updated successfully!")

if __name__ == "__main__":
    export_layers()
