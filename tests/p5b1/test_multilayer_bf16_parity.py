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

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"
REAL_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

@pytest.fixture(scope="module")
def setup_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    store = WeightStore(root_dir=REAL_DIR)
    return model, tok, store

def test_l1_l2_l4_bf16_and_router_parity(setup_model):
    """G4 Verification: Strict regression across L1 [0], L2 [0, 4], and L4 [0, 4, 8, 12]."""
    model, tokenizer, store = setup_model

    torch.manual_seed(42)
    x = torch.randn(2, 4, 2048, dtype=torch.bfloat16)

    cache = TieredCache(store, ram_budget_bytes=256*1024*1024, vram_budget_bytes=128*1024*1024)
    layers_to_test = [0, 4, 8, 12]

    # Test L4 simultaneously attached
    offloader = MultiLayerOffloadedMoE(
        model,
        target_layers=layers_to_test,
        cache=cache,
        mode="bf16_passthrough"
    )

    for l in layers_to_test:
        dyn_layer = offloader.dyn_layers[l]
        orig_mlp = offloader.orig_mlps[l]

        # 1. Router logits and Top-k ID match
        flat_x = x.view(-1, 2048)
        orig_logits, _, orig_topk = orig_mlp.gate(flat_x)
        dyn_logits, _, dyn_topk = dyn_layer.gate(flat_x)

        assert torch.equal(orig_topk, dyn_topk), f"Layer {l} Top-k experts mismatch!"
        assert (orig_logits.float() - dyn_logits.float()).abs().max().item() == 0.0

        # 2. Layer output numerical parity
        with torch.no_grad():
            ref_out = orig_mlp(x)
            dyn_out = dyn_layer(x)

        diff = (ref_out.float() - dyn_out.float()).abs()
        rel_l2 = (torch.norm(diff) / torch.norm(ref_out.float())).item()
        cos_sim = F.cosine_similarity(ref_out.float().flatten(), dyn_out.float().flatten(), dim=0).item()

        print(f"Layer {l}: Rel L2 = {rel_l2:.6e}, CosSim = {cos_sim:.6f}")
        assert rel_l2 < 0.005, f"Layer {l} Rel L2 too high: {rel_l2}"
        assert cos_sim > 0.9999, f"Layer {l} Cosine similarity too low: {cos_sim}"

    # Invariant: Cache within budget
    assert cache.vram_used <= cache.vram_budget
    assert cache.ram_used <= cache.ram_budget

    offloader.restore(allow_reconstruct=True)
