import pytest
import os
import sys
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.quant_codec
from src.weight_store import WeightStore
from src.tiered_cache import TieredCache

REAL_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

def test_strictly_no_online_quantization_during_multilayer_inference(monkeypatch):
    """G2 Verification: Ensures no online quantize_* is called during multi-layer inference."""
    def _forbidden(*args, **kwargs):
        raise RuntimeError("VIOLATION: Online quantization was called during inference!")

    monkeypatch.setattr(src.quant_codec, "quantize_int4_g128", _forbidden)
    monkeypatch.setattr(src.quant_codec, "quantize_int8_g128", _forbidden)

    store = WeightStore(root_dir=REAL_DIR)
    cache = TieredCache(store)

    dummy_x = torch.randn(1, 2048, dtype=torch.bfloat16, device="cuda" if torch.cuda.is_available() else "cpu")

    # Access across multiple layers [0, 4, 8, 12]
    for l in [0, 4, 8, 12]:
        for p in ["int4_g128", "int8_g128", "bf16"]:
            gpu_data, meta, timings = cache.get_expert(l, 5, p)
            out, _, _ = cache.forward_expert_swiglu(dummy_x, gpu_data, meta)
            assert out.shape == (1, 2048)
