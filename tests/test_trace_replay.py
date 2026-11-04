import pytest
import os
import json
from src.trace_generator import generate_traces

def test_trace_generation_deterministic():
    traces1 = generate_traces(num_requests=50, num_layers=2, num_experts=8, seed=123)
    
    with open(traces1["zipf"], "r") as f:
        lines1 = f.readlines()

    traces2 = generate_traces(num_requests=50, num_layers=2, num_experts=8, seed=123)
    with open(traces2["zipf"], "r") as f:
        lines2 = f.readlines()

    assert lines1 == lines2, "Same seed must produce identical traces"
    data = [json.loads(line) for line in lines1]
    assert len(data) == 50
    assert data[0]["request_id"] == 0
    assert "layer_id" in data[0]
    assert "expert_id" in data[0]
