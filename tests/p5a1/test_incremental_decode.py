import pytest
import os
import sys
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.p5a.generation_runner import generate_tokens

MODEL_DIR = "/home/yaoyao/llm_test/models/hf_original"

@pytest.fixture(scope="module")
def setup_model_and_tok():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    return model, tok

def test_incremental_kv_matches_full_prefix_recompute(setup_model_and_tok):
    """
    G8 Verification: Proves that incremental KV Cache generation produces
    EXACTLY identical token sequences to full prefix recomputation.
    """
    model, tokenizer = setup_model_and_tok
    prompt = "Explain MoE routing in one concise sentence:"
    max_tokens = 6

    res_full = generate_tokens(
        model, tokenizer, prompt, max_new_tokens=max_tokens, decode_mode="full_prefix_recompute"
    )

    res_kv = generate_tokens(
        model, tokenizer, prompt, max_new_tokens=max_tokens, decode_mode="incremental_kv"
    )

    assert res_full["generated_tokens"] == res_kv["generated_tokens"], \
        f"KV tokens {res_kv['generated_tokens']} != Full tokens {res_full['generated_tokens']}"
    assert res_full["generated_text"] == res_kv["generated_text"]
    print(f"Verified token match: {res_kv['generated_text']}")
