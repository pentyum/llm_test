import pytest
import numpy as np
import torch
from src.quant_codec import pack_int4, unpack_int4, quantize_int4_g128, dequantize_int4_g128, quantize_int8_g128, dequantize_int8_g128

def test_pack_unpack_roundtrip_all_values():
    # Test all 4-bit signed integer values from -8 to 7
    all_vals = np.array([-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int8)
    packed = pack_int4(all_vals)
    unpacked = unpack_int4(packed)
    assert np.array_equal(all_vals, unpacked), f"Expected {all_vals}, got {unpacked}"

def test_pack_unpack_random_roundtrip():
    np.random.seed(42)
    for length in [100, 1024, 8192]:
        vals = np.random.randint(-8, 8, size=length, dtype=np.int8)
        packed = pack_int4(vals)
        assert len(packed) == length // 2
        unpacked = unpack_int4(packed)
        assert np.array_equal(vals, unpacked)

def test_odd_length_handling():
    # Odd length should raise AssertionError
    odd_vals = np.array([1, 2, 3], dtype=np.int8)
    with pytest.raises(AssertionError):
        pack_int4(odd_vals)

def test_quantize_dequantize_int4():
    torch.manual_seed(42)
    # 2048 x 1024 matrix
    weight = torch.randn(2048, 1024, dtype=torch.bfloat16)
    packed, scales, shape = quantize_int4_g128(weight, group_size=128)
    
    assert packed.shape[0] == (2048 * 1024) // 2
    assert scales.shape == (2048, 1024 // 128)
    assert shape == (2048, 1024)
    
    dequant = dequantize_int4_g128(packed, scales, shape, group_size=128, target_dtype=torch.bfloat16)
    assert dequant.shape == weight.shape
    assert dequant.dtype == torch.bfloat16
    
    # Check MSE is reasonable for 4-bit quantization
    mse = torch.mean((weight.float() - dequant.float()) ** 2).item()
    assert mse < 0.05, f"MSE too high: {mse}"

def test_quantize_dequantize_int8():
    torch.manual_seed(42)
    weight = torch.randn(2048, 1024, dtype=torch.bfloat16)
    int8_w, scales = quantize_int8_g128(weight, group_size=128)
    
    assert int8_w.shape == (2048, 1024)
    assert int8_w.dtype == torch.int8
    assert scales.shape == (2048, 1024 // 128)
    
    dequant = dequantize_int8_g128(int8_w, scales, group_size=128, target_dtype=torch.bfloat16)
    assert dequant.shape == weight.shape
    
    # INT8 MSE should be very small (< 0.001)
    mse = torch.mean((weight.float() - dequant.float()) ** 2).item()
    assert mse < 0.001, f"INT8 MSE too high: {mse}"
