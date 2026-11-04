import pytest
import os
import sys
import gc
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.p5b.offloaded_moe import MultiLayerOffloadedMoE
from src.p5a.generation_runner import generate_tokens

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"
REAL_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

def test_true_ssd_offloading_and_memory_unloading():
    """
    G4 Verification: True SSD-backed execution without retaining original BF16 experts in RAM.
    """
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    store = WeightStore(root_dir=REAL_DIR)
    # Zero RAM cache, 32 MiB VRAM budget to force continuous SSD loads
    cache = TieredCache(store, ram_budget_bytes=0, vram_budget_bytes=32*1024*1024)

    # Attach with release_orig_experts=True
    offloader = MultiLayerOffloadedMoE(
        model,
        target_layers=[0, 4],
        cache=cache,
        mode="fixed_int4",
        release_orig_experts=True
    )

    # 1. Assert that original experts are genuinely None (not retained in memory)
    assert offloader.orig_mlps[0].experts is None, "Layer 0 experts must be deleted!"
    assert offloader.orig_mlps[4].experts is None, "Layer 4 experts must be deleted!"

    # 2. Execute incremental generation
    prompt = "Explain MoE offloading:"
    res = generate_tokens(model, tok, prompt, max_new_tokens=5, decode_mode="incremental_kv")

    # 3. Verify that SSD loads actually occurred
    stats = cache.get_stats()
    print("TieredCache stats after execution:", stats)
    assert stats["ssd_reads"] > 0, "Must have triggered true SSD file reads!"
    assert stats["ssd_read_bytes"] > 0
    assert len(res["generated_tokens"]) == 5

    offloader.restore()
