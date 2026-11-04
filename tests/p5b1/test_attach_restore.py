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

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"
REAL_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

@pytest.fixture(scope="module")
def setup_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    store = WeightStore(root_dir=REAL_DIR)
    return model, tok, store

def test_safe_restore_without_expert_release(setup_model):
    """G1 Verification 1: Safe restore when release_orig_experts=False."""
    model, tokenizer, store = setup_model
    x = torch.randn(1, 4, 2048, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_out = model.model.layers[0].mlp(x)

    cache = TieredCache(store)
    offloader = MultiLayerOffloadedMoE(model, target_layers=[0], cache=cache, mode="bf16_passthrough", release_orig_experts=False)
    assert offloader.is_attached

    offloader.restore()
    assert not offloader.is_attached

    with torch.no_grad():
        restored_out = model.model.layers[0].mlp(x)

    assert torch.equal(ref_out, restored_out), "Model output after restore must be bitwise identical!"

def test_explicit_error_when_restoring_released_experts_without_reconstruct(setup_model):
    """G1 Verification 2: Explicit error when allow_reconstruct=False on released experts."""
    model, tokenizer, store = setup_model
    cache = TieredCache(store)

    offloader = MultiLayerOffloadedMoE(model, target_layers=[0], cache=cache, mode="fixed_int4", release_orig_experts=True)
    assert offloader.orig_mlps[0].experts is None

    with pytest.raises(RuntimeError) as exc_info:
        offloader.restore(allow_reconstruct=False)

    assert "Cannot restore layer 0 because original experts were released" in str(exc_info.value)
    # Clean up with reconstruction for subsequent tests
    offloader.restore(allow_reconstruct=True)

def test_safe_reconstruction_from_store_after_release(setup_model):
    """G1 Verification 3: Reconstructs OlmoeExperts from WeightStore BF16 on restore."""
    model, tokenizer, store = setup_model
    x = torch.randn(1, 4, 2048, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_out = model.model.layers[0].mlp(x)

    cache = TieredCache(store)
    offloader = MultiLayerOffloadedMoE(model, target_layers=[0], cache=cache, mode="fixed_int4", release_orig_experts=True)
    assert offloader.orig_mlps[0].experts is None

    # Restore with reconstruction
    offloader.restore(allow_reconstruct=True)
    assert offloader.model.model.layers[0].mlp.experts is not None

    with torch.no_grad():
        restored_out = model.model.layers[0].mlp(x)

    assert torch.equal(ref_out, restored_out), "Reconstructed experts must produce bitwise identical output to reference!"
