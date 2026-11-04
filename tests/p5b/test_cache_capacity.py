import pytest
import os
import sys
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.weight_store import WeightStore
from src.tiered_cache import TieredCache

REAL_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

def test_cache_hard_budget_invariant():
    """G3 Verification: Strict invariant used <= budget on VRAM and RAM."""
    store = WeightStore(root_dir=REAL_DIR)
    vram_budget = 30 * 1024 * 1024  # 30 MiB (~9 INT4 experts)
    ram_budget = 60 * 1024 * 1024   # 60 MiB

    cache = TieredCache(store, ram_budget_bytes=ram_budget, vram_budget_bytes=vram_budget)

    # Access 30 different experts across layers
    for l in [0, 4, 8, 12]:
        for e in range(10):
            cache.get_expert(l, e, "int4_g128")
            # Invariant check at every single step
            assert cache.vram_used <= cache.vram_budget, f"VRAM exceeded: {cache.vram_used} > {cache.vram_budget}"
            assert cache.ram_used <= cache.ram_budget, f"RAM exceeded: {cache.ram_used} > {cache.ram_budget}"

def test_zero_budget_bypass_and_oversized():
    """G3 Verification: Zero budget forces bypass and does not accumulate bytes."""
    store = WeightStore(root_dir=REAL_DIR)
    cache = TieredCache(store, ram_budget_bytes=0, vram_budget_bytes=0)

    for e in range(5):
        data, meta, _ = cache.get_expert(0, e, "int4_g128")
        assert cache.vram_used == 0
        assert cache.ram_used == 0
        assert len(cache.vram_cache) == 0
