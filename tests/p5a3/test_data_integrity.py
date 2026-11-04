import pytest
import os
import sys
import json
import pandas as pd

FREE_GEN_JSONL = "/home/yaoyao/llm_test/results/p5a3/quality/free_generation.jsonl"

def test_generation_records_integrity():
    """G7 Verification: Checks that generation records in free_generation.jsonl are well-formed."""
    if not os.path.exists(FREE_GEN_JSONL):
        pytest.skip("free_generation.jsonl not created yet")

    records = []
    with open(FREE_GEN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) > 0, "free_generation.jsonl should have records"
    df = pd.DataFrame(records)

    # Required columns
    required_cols = [
        "repeat", "mode", "prompt_id", "category", "ttft_ms",
        "tpot_mean_ms", "token_match_pct", "status", "generated_tokens"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert (df["status"] == "SUCCESS").all()
    assert (df["ttft_ms"] > 0).all()
    assert (df["tpot_mean_ms"] > 0).all()
