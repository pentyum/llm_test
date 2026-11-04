import pytest
import os
import sys
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.quant_codec import (
    pack_int4, unpack_int4,
    dequantize_int4_g128_legacy_numpy,
    dequantize_int4_g128_torch_cuda,
    dequantize_int4_g128,
    quantize_int4_g128
)
from src.weight_store import WeightStore

REAL_EXPERTS_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

def test_nibble_boundary_values():
    """G2 & Step 1: Tests edge case values [-8, -7, -1, 0, 1, 7] in GPU unpack."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    base_values = np.array([-8, 7, -7, 6, -1, 0, 1, -2, -3, 3, -4, 4, -5, 5, -6, 2], dtype=np.int8)
    # Repeat 8 times to reach length 128 (group_size 128)
    values = np.tile(base_values, 8)
    packed_np = pack_int4(values)
    packed_cuda = torch.from_numpy(packed_np).cuda()

    scales_cuda = torch.ones((1, 1), dtype=torch.bfloat16, device="cuda")

    out_legacy = dequantize_int4_g128_legacy_numpy(packed_cuda, scales_cuda, (1, 128), group_size=128)
    out_gpu = dequantize_int4_g128_torch_cuda(packed_cuda, scales_cuda, (1, 128), group_size=128)

    assert torch.equal(out_legacy, out_gpu), f"Legacy {out_legacy} != GPU {out_gpu}"
    expected = torch.tensor(values, dtype=torch.bfloat16, device="cuda").reshape(1, 128)
    assert torch.equal(out_gpu, expected)

def test_random_weights_exact_match():
    """G2 & Step 2: Random weights quantization and unpack exactness between CPU and GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    torch.manual_seed(42)
    shape = (256, 512)
    w = torch.randn(shape, dtype=torch.bfloat16) * 2.0
    packed, scales, _ = quantize_int4_g128(w, group_size=128)

    packed_cuda = packed.cuda()
    scales_cuda = scales.cuda()

    out_legacy = dequantize_int4_g128_legacy_numpy(packed_cuda, scales_cuda, shape, group_size=128)
    out_gpu = dequantize_int4_g128_torch_cuda(packed_cuda, scales_cuda, shape, group_size=128)

    assert torch.equal(out_legacy, out_gpu), "GPU CUDA unpack must match CPU NumPy unpack exactly!"

def test_real_expert_matrices_unpack():
    """G2 & Step 3: Verifies exact unpack on real OLMoE expert gate/up/down matrices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    store = WeightStore(root_dir=REAL_EXPERTS_DIR)
    data, meta, _, _ = store.load_expert(0, 0, "int4_g128")

    for proj in ["gate", "up", "down"]:
        packed = data[f"{proj}_proj"].cuda()
        scales = data[f"{proj}_scales"].cuda()
        shp = data[f"{proj}_shape"]

        out_legacy = dequantize_int4_g128_legacy_numpy(packed, scales, shp, group_size=128)
        out_gpu = dequantize_int4_g128_torch_cuda(packed, scales, shp, group_size=128)

        assert torch.equal(out_legacy, out_gpu), f"Mismatch in real expert matrix: {proj}"

def test_no_cpu_transfer_during_gpu_unpack(monkeypatch):
    """
    G2 & Negative Assertion: Ensures dequantize_int4_g128_torch_cuda does NOT call
    Tensor.cpu() or numpy().
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    orig_cpu = torch.Tensor.cpu
    def _forbidden_cpu(self, *args, **kwargs):
        raise RuntimeError("VIOLATION: Tensor.cpu() was called during GPU unpack!")

    monkeypatch.setattr(torch.Tensor, "cpu", _forbidden_cpu)

    packed = torch.randint(0, 255, (1024,), dtype=torch.uint8, device="cuda")
    scales = torch.ones((16, 1), dtype=torch.bfloat16, device="cuda")

    # Should run smoothly without calling cpu()
    out = dequantize_int4_g128_torch_cuda(packed, scales, (16, 128), group_size=128)
    assert out.is_cuda
    assert out.shape == (16, 128)
