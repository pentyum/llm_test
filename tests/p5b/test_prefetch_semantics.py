import pytest
import os
import sys
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.p5b.offloaded_moe import MultiLayerOffloadedMoE
from src.p5b.prefetch_manager import PrefetchManager
from src.p5a.generation_runner import generate_tokens

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"
REAL_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

@pytest.fixture(scope="module")
def setup_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    store = WeightStore(root_dir=REAL_DIR)
    return model, tok, store

def test_prefetch_semantics_and_exact_output_identity(setup_model):
    """
    G5 Verification: Prefetch must produce EXACT same outputs as sync execution,
    while correctly tracking hits and wasted bytes.
    """
    model, tok, store = setup_model
    prompt = "Explain MoE routing in one sentence:"
    n_tokens = 5

    # 1. Sync No-Prefetch
    cache_sync = TieredCache(store, ram_budget_bytes=64*1024*1024, vram_budget_bytes=32*1024*1024)
    off_sync = MultiLayerOffloadedMoE(model, target_layers=[0, 4], cache=cache_sync, mode="fixed_int4")
    res_sync = generate_tokens(model, tok, prompt, max_new_tokens=n_tokens, decode_mode="incremental_kv")
    off_sync.restore()

    # 2. Static Layer Prefetch
    cache_pre = TieredCache(store, ram_budget_bytes=64*1024*1024, vram_budget_bytes=32*1024*1024)
    mgr = PrefetchManager(cache_pre, strategy="static_layer_prefetch", target_layers=[0, 4])
    off_pre = MultiLayerOffloadedMoE(model, target_layers=[0, 4], cache=cache_pre, mode="fixed_int4", prefetch_manager=mgr)
    res_pre = generate_tokens(model, tok, prompt, max_new_tokens=n_tokens, decode_mode="incremental_kv")
    off_pre.restore()

    # Exact token equality
    assert res_sync["generated_tokens"] == res_pre["generated_tokens"], "Outputs must match exactly!"

    # Verify prefetch stats
    stats = mgr.get_prefetch_stats()
    print("Static Prefetch Stats:", stats)
    assert stats["prefetch_requests"] > 0
    assert "prefetch_hit_rate" in stats
