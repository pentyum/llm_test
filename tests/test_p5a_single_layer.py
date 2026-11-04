import pytest
import os
import sys
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM
from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.p5a.dynamic_moe_layer import DynamicMoELayer

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"
REAL_EXPERTS_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

@pytest.fixture(scope="module")
def setup_layer0():
    # Load model on CPU
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu")
    orig_moe = model.model.layers[0].mlp

    store = WeightStore(root_dir=REAL_EXPERTS_DIR)
    cache = TieredCache(store, ram_budget_bytes=128*1024*1024, vram_budget_bytes=64*1024*1024)

    dyn_layer = DynamicMoELayer(orig_moe, layer_id=0, cache=cache, mode="bf16_passthrough")
    return orig_moe, dyn_layer

def test_bf16_passthrough_numerical_match(setup_layer0):
    """F3 & Step 1: BF16 passthrough output matches original reference layer output."""
    orig_moe, dyn_layer = setup_layer0
    dyn_layer.set_mode("bf16_passthrough")

    torch.manual_seed(42)
    # 4 tokens x 2048 hidden
    x = torch.randn(1, 4, 2048, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_out = orig_moe(x)
        dyn_out = dyn_layer(x)

    assert dyn_out.shape == ref_out.shape
    # Check relative L2 difference is within CPU vs GPU bfloat16 accumulation tolerance
    rel_l2 = (torch.norm(dyn_out.float() - ref_out.float()) / torch.norm(ref_out.float())).item()
    print(f"BF16 passthrough RelL2: {rel_l2:.6f}")
    assert rel_l2 < 0.01, f"BF16 passthrough error too high: {rel_l2}"

def test_fixed_int8_and_int4_modes(setup_layer0):
    """F3 & Step 2: Fixed INT8 and INT4 forward execute cleanly with bounded errors."""
    orig_moe, dyn_layer = setup_layer0
    torch.manual_seed(42)
    x = torch.randn(1, 4, 2048, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_out = orig_moe(x)

        # INT8
        dyn_layer.set_mode("fixed_int8")
        out_i8 = dyn_layer(x)
        rel_l2_i8 = (torch.norm(out_i8.float() - ref_out.float()) / torch.norm(ref_out.float())).item()
        assert rel_l2_i8 < 0.05, f"INT8 error too high: {rel_l2_i8}"

        # INT4
        dyn_layer.set_mode("fixed_int4")
        out_i4 = dyn_layer(x)
        rel_l2_i4 = (torch.norm(out_i4.float() - ref_out.float()) / torch.norm(ref_out.float())).item()
        assert rel_l2_i4 < 0.25, f"INT4 error too high: {rel_l2_i4}"

def test_dynamic_switching_records_events(setup_layer0):
    """F3 & Step 3: Dynamic mode executes and records online precision switch events."""
    orig_moe, dyn_layer = setup_layer0
    dyn_layer.set_mode("dynamic_s4")
    dyn_layer.clear_events()

    x = torch.randn(1, 8, 2048, dtype=torch.bfloat16)
    with torch.no_grad():
        out_dyn = dyn_layer(x)

    events = dyn_layer.switch_events
    assert len(events) == 8 * 8  # 8 tokens x 8 experts/token
    assert any(e["requested_precision"] == "int4_g128" for e in events)
    assert any(e["requested_precision"] == "int8_g128" for e in events)
