import pytest
import os
import sys
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.quant_codec
from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.p5a.precision_policies import ForcedSequencePolicy

REAL_EXPERTS_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

def test_strictly_no_online_quantization(monkeypatch):
    """
    G2 Verification: Ensures that online inference strictly loads pre-quantized weights
    and never invokes quantize_int4_g128 or quantize_int8_g128.
    """
    def _forbidden_quant(*args, **kwargs):
        raise RuntimeError("VIOLATION: Online quantization was invoked during inference!")

    monkeypatch.setattr(src.quant_codec, "quantize_int4_g128", _forbidden_quant)
    monkeypatch.setattr(src.quant_codec, "quantize_int8_g128", _forbidden_quant)

    store = WeightStore(root_dir=REAL_EXPERTS_DIR)
    cache = TieredCache(store)

    policy = ForcedSequencePolicy({
        (0, 5): ["int4_g128", "int8_g128", "int4_g128"]
    })

    dummy_x = torch.randn(1, 2048, dtype=torch.bfloat16, device=cache.device)

    # Execute 3 calls - must succeed by loading pre-saved weights without calling quantize_*
    for step in range(3):
        target_prec = policy.select(0, 5)
        gpu_data, meta, timings = cache.get_expert(0, 5, target_prec)
        out, dec_ms, gemm_ms = cache.forward_expert_swiglu(dummy_x, gpu_data, meta)
        assert out.shape == (1, 2048)
        assert meta["precision"] == target_prec
