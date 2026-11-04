"""
Quantization Codec for MoE Experts
Supports:
  - BF16: Identity
  - INT8 (group_size 128, symmetric)
  - INT4 (group_size 128, symmetric, 2 values packed per byte)
"""

import numpy as np
import torch

def quantize_int8_g128(weight: torch.Tensor, group_size: int = 128):
    """
    Symmetric INT8 group quantization along the last dimension.
    weight: shape (..., K)
    Returns:
      int8_weight: torch.int8 tensor
      scales: torch.float32 / torch.bfloat16 tensor
    """
    orig_shape = weight.shape
    orig_dtype = weight.dtype
    K = orig_shape[-1]
    assert K % group_size == 0, f"Last dimension {K} must be divisible by group_size {group_size}"

    flat = weight.reshape(-1, group_size).float()
    max_val = torch.amax(torch.abs(flat), dim=-1, keepdim=True)
    scales = torch.clamp(max_val / 127.0, min=1e-8)
    quantized = torch.clamp(torch.round(flat / scales), -127, 127).to(torch.int8)

    int8_weight = quantized.reshape(orig_shape)
    scales = scales.reshape(*orig_shape[:-1], K // group_size).to(orig_dtype)
    return int8_weight, scales

def dequantize_int8_g128(int8_weight: torch.Tensor, scales: torch.Tensor, group_size: int = 128, target_dtype=torch.bfloat16):
    """
    Dequantize symmetric INT8 to target float dtype.
    """
    orig_shape = int8_weight.shape
    K = orig_shape[-1]
    flat_w = int8_weight.reshape(-1, group_size).float()
    flat_s = scales.reshape(-1, 1).float()
    dequant = (flat_w * flat_s).to(target_dtype)
    return dequant.reshape(orig_shape)

def pack_int4(int4_values: np.ndarray) -> np.ndarray:
    """
    Pack 1D array of 4-bit signed integers (range -8..7) into uint8 bytes.
    nibble 0 (low 4 bits): first value
    nibble 1 (high 4 bits): second value
    """
    assert len(int4_values) % 2 == 0, "Length of int4 array must be even"
    u4 = (int4_values.astype(np.int32) & 0x0F).astype(np.uint8)
    low = u4[0::2]
    high = u4[1::2]
    packed = (low | (high << 4)).astype(np.uint8)
    return packed

def unpack_int4(packed_bytes: np.ndarray) -> np.ndarray:
    """
    Unpack uint8 bytes into 1D array of 4-bit signed integers (-8..7).
    """
    low = (packed_bytes & 0x0F).astype(np.int32)
    high = ((packed_bytes >> 4) & 0x0F).astype(np.int32)
    # Sign extension
    low = np.where(low >= 8, low - 16, low).astype(np.int8)
    high = np.where(high >= 8, high - 16, high).astype(np.int8)

    unpacked = np.empty(len(packed_bytes) * 2, dtype=np.int8)
    unpacked[0::2] = low
    unpacked[1::2] = high
    return unpacked

def quantize_int4_g128(weight: torch.Tensor, group_size: int = 128):
    """
    Symmetric INT4 group quantization along the last dimension.
    Range: [-7, 7]
    Returns:
      packed_bytes: torch.uint8 tensor of length numel // 2
      scales: float tensor
      shape: original shape
    """
    orig_shape = tuple(weight.shape)
    orig_dtype = weight.dtype
    K = orig_shape[-1]
    assert K % group_size == 0, f"Last dimension {K} must be divisible by group_size {group_size}"

    flat = weight.reshape(-1, group_size).float()
    max_val = torch.amax(torch.abs(flat), dim=-1, keepdim=True)
    scales = torch.clamp(max_val / 7.0, min=1e-8)
    quantized = torch.clamp(torch.round(flat / scales), -7, 7).to(torch.int8)

    # Pack into bytes
    q_np = quantized.cpu().numpy().flatten()
    packed_np = pack_int4(q_np)
    packed_bytes = torch.from_numpy(packed_np)

    scales = scales.reshape(*orig_shape[:-1], K // group_size).to(orig_dtype)
    return packed_bytes, scales, orig_shape

def dequantize_int4_g128(packed_bytes: torch.Tensor, scales: torch.Tensor, shape: tuple, group_size: int = 128, target_dtype=torch.bfloat16):
    """
    Dequantize packed INT4 bytes to target float dtype.
    """
    packed_np = packed_bytes.cpu().numpy()
    unpacked_np = unpack_int4(packed_np)
    unpacked = torch.from_numpy(unpacked_np).to(device=scales.device)

    K = shape[-1]
    flat_w = unpacked.reshape(-1, group_size).float()
    flat_s = scales.reshape(-1, 1).float()
    dequant = (flat_w * flat_s).to(target_dtype)
    return dequant.reshape(shape)
