import pytest
import os
import sys
import pandas as pd
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROUTER_CSV = "/home/yaoyao/llm_test/results/p5a3/parity/router_parity.csv"
LAYER_CSV = "/home/yaoyao/llm_test/results/p5a3/parity/layer_output_parity.csv"
DIV_JSON = "/home/yaoyao/llm_test/results/p5a3/parity/first_divergence.json"

def test_router_parity():
    assert os.path.exists(ROUTER_CSV)
    df = pd.read_csv(ROUTER_CSV)
    assert (df["status"] == "PASS").all()
    assert (df["indices_exact_match"] == True).all()
    assert (df["logits_max_abs_diff"] == 0.0).all()

def test_layer_output_parity():
    assert os.path.exists(LAYER_CSV)
    df = pd.read_csv(LAYER_CSV)
    assert (df["status"] == "PASS").all()
    assert (df["rel_l2_diff"] < 0.005).all()
    assert (df["cosine_similarity"] > 0.9999).all()

def test_first_divergence_record():
    assert os.path.exists(DIV_JSON)
    with open(DIV_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "first_divergent_stage" in data
