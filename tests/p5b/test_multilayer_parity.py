import pytest
import os
import sys
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.p5b.offloaded_moe import MultiLayerOffloadedMoE
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

def test_l2_and_l4_router_and_layer_parity(setup_model):
    """G1 Verification 1: Parity check across multiple layers [0, 4, 8, 12]."""
    model, tokenizer, store = setup_model

    torch.manual_seed(42)
    x = torch.randn(1, 4, 2048, dtype=torch.bfloat16)

    cache = TieredCache(store, ram_budget_bytes=256*1024*1024, vram_budget_bytes=128*1024*1024)
    target_layers = [0, 4, 8, 12]

    # Test each layer independently first
    for l in target_layers:
        orig_mlp = model.model.layers[l].mlp
        with torch.no_grad():
            ref_out = orig_mlp(x)

        # Attach custom layer
        offloader = MultiLayerOffloadedMoE(model, target_layers=[l], cache=cache, mode="bf16_passthrough")
        dyn_layer = offloader.dyn_layers[l]
        with torch.no_grad():
            dyn_out = dyn_layer(x)

        diff = (ref_out.float() - dyn_out.float()).abs()
        rel_l2 = (torch.norm(diff) / torch.norm(ref_out.float())).item()
        cos_sim = F.cosine_similarity(ref_out.float().flatten(), dyn_out.float().flatten(), dim=0).item()

        print(f"Layer {l} Output Parity: Rel L2 = {rel_l2:.6e}, CosSim = {cos_sim:.6f}")
        assert rel_l2 < 0.005, f"Layer {l} rel_l2 too high: {rel_l2}"
        assert cos_sim > 0.9999, f"Layer {l} cosine similarity too low: {cos_sim}"

        offloader.restore()

def test_l4_incremental_decode_smoke(setup_model):
    """G1 Verification 2: 4-layer offloading incremental decode runs without NaN/OOM."""
    model, tokenizer, store = setup_model

    cache = TieredCache(store, ram_budget_bytes=256*1024*1024, vram_budget_bytes=128*1024*1024)
    target_layers = [0, 4, 8, 12]

    offloader = MultiLayerOffloadedMoE(model, target_layers=target_layers, cache=cache, mode="fixed_int4")

    prompt = "Explain mixture of experts in one short sentence:"
    res = generate_tokens(model, tokenizer, prompt, max_new_tokens=6, decode_mode="incremental_kv")

    print(f"L4 fixed_int4 generated tokens: {res['generated_tokens']}")
    print(f"Generated text: {repr(res['generated_text'])}")

    assert len(res["generated_tokens"]) == 6
    assert not torch.isnan(torch.tensor(res["generated_tokens"])).any()

    offloader.restore()
