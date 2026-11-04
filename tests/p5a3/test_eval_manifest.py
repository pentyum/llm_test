import pytest
import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PROMPT_FILE = "/home/yaoyao/llm_test/configs/p5a2/eval_prompts.jsonl"
MANIFEST_FILE = "/home/yaoyao/llm_test/results/p5a3/run_manifest.json"

def test_eval_manifest_completeness_and_balance():
    """G7 Verification: 32 unique prompt IDs, exactly 8 in each of 4 categories."""
    assert os.path.exists(PROMPT_FILE), "Prompt file must exist"
    
    prompts = []
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    assert len(prompts) == 32, f"Expected exactly 32 prompts, got {len(prompts)}"
    
    # Check unique IDs
    ids = [p["id"] for p in prompts]
    assert len(set(ids)) == 32, "All 32 prompt IDs must be unique!"

    # Check 4 balanced categories
    cat_counts = {}
    for p in prompts:
        cat = p["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    expected_cats = {"architecture": 8, "systems": 8, "code": 8, "reasoning": 8}
    assert cat_counts == expected_cats, f"Category distribution mismatch: {cat_counts}"

def test_run_manifest_metadata():
    assert os.path.exists(MANIFEST_FILE)
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["num_prompts"] == 32
    assert m["total_expected_runs"] == 32 * 3 * 5
    assert m["repeats"] == 3
    assert len(m["modes"]) == 5
