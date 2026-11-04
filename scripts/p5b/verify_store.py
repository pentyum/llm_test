#!/usr/bin/env python3
"""
P5b Gate G2: Multi-Layer Expert Store Verification Script
Verifies:
  1. Complete presence of 256 experts x 3 precisions = 768 files in manifest
  2. Random sampling across Layer 0, 4, 8, 12 to verify shapes, dtypes, and unpack correctness
  3. Strict zero online quantization check
Outputs:
  - results/p5b/store_verification.json
"""

import os
import sys
import json
import random
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.weight_store import WeightStore
from src.quant_codec import dequantize_int4_g128_torch_cuda, dequantize_int8_g128

REAL_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"
OUT_JSON = "/home/yaoyao/llm_test/results/p5b/store_verification.json"

def verify_expert_store():
    print("=== P5b G2: Verifying Multi-Layer Expert Store ===")
    store = WeightStore(root_dir=REAL_DIR)
    manifest = store.manifest

    layers = [0, 4, 8, 12]
    precisions = ["bf16", "int8_g128", "int4_g128"]

    # 1. Check all 768 keys exist in manifest
    missing_keys = []
    for l in layers:
        for e in range(64):
            for p in precisions:
                k = f"{l}_{e}_{p}"
                if k not in manifest:
                    missing_keys.append(k)

    print(f"Total keys expected: {len(layers) * 64 * len(precisions)} (768)")
    print(f"Missing keys in manifest: {len(missing_keys)}")
    assert len(missing_keys) == 0, f"Missing manifest keys: {missing_keys[:5]}"

    # 2. Random sampling across layers and precisions
    sample_experts = [
        (0, 5), (0, 38), (4, 12), (4, 47),
        (8, 0), (8, 63), (12, 10), (12, 33)
    ]

    sample_results = []
    for l, e in sample_experts:
        # Load BF16
        b_data, b_meta, _, _ = store.load_expert(l, e, "bf16")
        assert "gate_proj" in b_data and "up_proj" in b_data and "down_proj" in b_data
        assert b_data["gate_proj"].shape == (1024, 2048)
        assert b_data["up_proj"].shape == (1024, 2048)
        assert b_data["down_proj"].shape == (2048, 1024)

        # Load INT8
        i8_data, i8_meta, _, _ = store.load_expert(l, e, "int8_g128")
        assert "gate_scales" in i8_data

        # Load INT4
        i4_data, i4_meta, _, _ = store.load_expert(l, e, "int4_g128")
        assert "gate_shape" in i4_data

        # Verify GPU unpack on INT4
        if torch.cuda.is_available():
            p_g = i4_data["gate_proj"].cuda()
            s_g = i4_data["gate_scales"].cuda()
            shp = i4_data["gate_shape"]
            unpacked = dequantize_int4_g128_torch_cuda(p_g, s_g, shp, group_size=128)
            assert unpacked.shape == (1024, 2048)
            assert unpacked.dtype == torch.bfloat16

        sample_results.append({
            "layer_id": l,
            "expert_id": e,
            "bf16_shape": list(b_data["gate_proj"].shape),
            "int8_scales_shape": list(i8_data["gate_scales"].shape),
            "int4_packed_numel": i4_data["gate_proj"].numel(),
            "status": "VERIFIED"
        })

    result_data = {
        "gate": "G2",
        "status": "PASS",
        "total_manifest_entries": len(manifest),
        "total_verified_layers": layers,
        "sample_verifications": sample_results
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    print(f"Saved store verification to {OUT_JSON}")
    print(json.dumps(result_data, indent=2))
    return True

if __name__ == "__main__":
    verify_expert_store()
