import pytest
import os
import sys
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.p5a.dynamic_moe_layer import DynamicMoELayer
from src.p5a.generation_runner import generate_tokens

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"
REAL_EXPERTS_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

def test_torch_cuda_swiglu_exact_match_with_legacy_numpy():
    """
    G5 Verification 1: Pure GPU SwiGLU forward must match legacy CPU NumPy SwiGLU exactly.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    store = WeightStore(root_dir=REAL_EXPERTS_DIR)
    cache = TieredCache(store)

    data, meta, _ = cache.get_expert(0, 0, "int4_g128")
    x = torch.randn(2, 2048, dtype=torch.bfloat16, device="cuda")

    out_legacy, _, _ = cache.forward_expert_swiglu(x, data, meta, dequant_backend="legacy_numpy")
    out_gpu, _, _ = cache.forward_expert_swiglu(x, data, meta, dequant_backend="torch_cuda")

    assert torch.equal(out_legacy, out_gpu), "GPU SwiGLU must produce bitwise identical output to legacy NumPy path!"

def test_incremental_decode_tokens_identical_between_backends():
    """
    G5 Verification 2: Verifies that incremental generation in fixed_int4 with torch_cuda
    produces identical token sequences to legacy_numpy, but runs significantly faster.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    orig_layer0 = model.model.layers[0].mlp
    store = WeightStore(root_dir=REAL_EXPERTS_DIR)

    prompt = "Explain MoE routing in one sentence:"
    n_tokens = 6

    # 1. Run with legacy_numpy backend
    cache_leg = TieredCache(store, ram_budget_bytes=64*1024*1024, vram_budget_bytes=32*1024*1024)
    dyn_leg = DynamicMoELayer(orig_layer0, layer_id=0, cache=cache_leg, mode="fixed_int4", dequant_backend="legacy_numpy")
    model.model.layers[0].mlp = dyn_leg
    res_leg = generate_tokens(model, tok, prompt, max_new_tokens=n_tokens, decode_mode="incremental_kv")

    # 2. Run with torch_cuda backend
    cache_gpu = TieredCache(store, ram_budget_bytes=64*1024*1024, vram_budget_bytes=32*1024*1024)
    dyn_gpu = DynamicMoELayer(orig_layer0, layer_id=0, cache=cache_gpu, mode="fixed_int4", dequant_backend="torch_cuda")
    model.model.layers[0].mlp = dyn_gpu
    res_gpu = generate_tokens(model, tok, prompt, max_new_tokens=n_tokens, decode_mode="incremental_kv")

    # Restore
    model.model.layers[0].mlp = orig_layer0

    # Assert exact token identity
    assert res_leg["generated_tokens"] == res_gpu["generated_tokens"], \
        f"Mismatch: legacy {res_leg['generated_tokens']} != gpu {res_gpu['generated_tokens']}"
    
    print(f"Legacy NumPy TPOT: {res_leg['tpot_mean_ms']} ms")
    print(f"Torch CUDA TPOT:   {res_gpu['tpot_mean_ms']} ms")
    assert res_gpu["tpot_mean_ms"] < res_leg["tpot_mean_ms"], "Torch CUDA backend must be faster than legacy NumPy!"
