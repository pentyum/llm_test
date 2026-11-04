import pytest
import torch
from src.weight_store import WeightStore
from src.tiered_cache import TieredCache

def test_quantized_expert_shapes_and_reproducibility():
    store = WeightStore()
    cache = TieredCache(store)

    x = torch.randn(1, 2048, dtype=torch.bfloat16, device=cache.device)

    # 1. BF16
    bf16_data, bf16_meta, _ = cache.get_expert(0, 0, "bf16")
    out_bf16_1, _, _ = cache.forward_expert_swiglu(x, bf16_data, bf16_meta)
    out_bf16_2, _, _ = cache.forward_expert_swiglu(x, bf16_data, bf16_meta)
    assert torch.equal(out_bf16_1, out_bf16_2), "BF16 forward must be deterministic"

    # 2. INT8
    int8_data, int8_meta, _ = cache.get_expert(0, 0, "int8_g128")
    out_int8, _, _ = cache.forward_expert_swiglu(x, int8_data, int8_meta)
    assert out_int8.shape == out_bf16_1.shape

    # 3. INT4
    int4_data, int4_meta, _ = cache.get_expert(0, 0, "int4_g128")
    out_int4, _, _ = cache.forward_expert_swiglu(x, int4_data, int4_meta)
    assert out_int4.shape == out_bf16_1.shape

    # Check relative L2 errors
    err_int8 = (torch.norm(out_int8.float() - out_bf16_1.float()) / torch.norm(out_bf16_1.float())).item()
    err_int4 = (torch.norm(out_int4.float() - out_bf16_1.float()) / torch.norm(out_bf16_1.float())).item()

    assert err_int8 < 0.05, f"INT8 relative error too high: {err_int8}"
    assert err_int4 < 0.25, f"INT4 relative error too high: {err_int4}"
    assert err_int8 < err_int4, "INT8 error should be smaller than INT4 error"
