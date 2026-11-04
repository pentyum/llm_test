import pytest
import os
import sys
import pandas as pd
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

WEIGHT_CSV = "/home/yaoyao/llm_test/results/p5a3/parity/weight_mapping.csv"

def test_weight_mapping_parity():
    """G7 Verification: Checks weight_mapping.csv proving 192/192 bitwise exact matches."""
    assert os.path.exists(WEIGHT_CSV), "weight_mapping.csv must exist"
    df = pd.read_csv(WEIGHT_CSV)

    assert len(df) == 64 * 3, f"Expected 192 rows, got {len(df)}"
    # Verify bitwise exact equality
    exact_count = df["exact_equal"].sum()
    assert exact_count == 192, f"Expected all 192 matrices to be exact bitwise match, got {exact_count}"
    assert (df["max_abs_error"] == 0.0).all()
    assert (df["status"] == "EXACT_MATCH").all()
