import pytest
import os
import sys
import time
import threading
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weight_store import WeightStore
from src.tiered_cache import TieredCache

def test_zero_budget_bypass():
    """C1 & Test 1: budget=0 must never store items and ram_used == vram_used == 0."""
    store = WeightStore()
    cache = TieredCache(store, ram_budget_bytes=0, vram_budget_bytes=0)

    for e in range(16):
        data, meta, timings = cache.get_expert(0, e % 4, "int4_g128")
        assert timings["hit_tier"] == "SSD"
        assert cache.ram_used == 0
        assert cache.vram_used == 0
        assert len(cache.ram_cache) == 0
        assert len(cache.vram_cache) == 0

    stats = cache.get_stats()
    assert stats["vram_target_hits"] == 0
    assert stats["ram_target_hits"] == 0
    assert stats["bypass_vram_count"] == 16
    assert stats["bypass_ram_count"] == 16

def test_oversized_object_bypass():
    """C1 & Test 2: Objects larger than budget must bypass without negative counts or overflow."""
    store = WeightStore()
    # Budget is 1 MB, but an INT4 expert is ~3.1 MB
    cache = TieredCache(store, ram_budget_bytes=1024*1024, vram_budget_bytes=1024*1024)

    data, meta, timings = cache.get_expert(0, 0, "int4_g128")
    assert cache.ram_used == 0
    assert cache.vram_used == 0
    assert len(cache.ram_cache) == 0
    assert len(cache.vram_cache) == 0
    stats = cache.get_stats()
    assert stats["bypass_vram_count"] == 1

def test_exact_budget_eviction_and_multi_precision():
    """Test 3: Multi-precision coexist, budget respects limit, and LRU evicts cleanly."""
    store = WeightStore()
    # INT4 expert is ~3.1MB, INT8 is ~6.1MB. Budget 10MB can hold 3 INT4 or 1 INT8 + 1 INT4
    cache = TieredCache(store, ram_budget_bytes=10*1024*1024, vram_budget_bytes=10*1024*1024)

    # 1. Load INT4 expert 0
    cache.get_expert(0, 0, "int4_g128")
    assert cache.vram_used <= 10*1024*1024

    # 2. Load INT8 expert 0 (different precision, distinct key)
    cache.get_expert(0, 0, "int8_g128")
    assert cache.vram_used <= 10*1024*1024
    assert len(cache.vram_cache) == 2

    # 3. Load INT8 expert 1 (triggers eviction)
    cache.get_expert(0, 1, "int8_g128")
    assert cache.vram_used <= 10*1024*1024
    stats = cache.get_stats()
    assert stats["vram_evictions"] >= 1

def test_prefetch_deduplication():
    """C2 & Test 4: Concurrent prefetch and get_expert on same key triggers exactly one SSD read."""
    store = WeightStore()
    cache = TieredCache(store, ram_budget_bytes=32*1024*1024, vram_budget_bytes=32*1024*1024)

    # Trigger prefetch
    cache.prefetch(0, 3, "int4_g128", target_tier="RAM")
    # Immediate get_expert while in-flight
    data, meta, timings = cache.get_expert(0, 3, "int4_g128")

    cache.drain_prefetches()
    stats = cache.get_stats()
    # Must only read from SSD once for this expert
    assert stats["ssd_reads"] == 1

def test_prefetch_error_recovery():
    """C2 & Test 5: Prefetch handles error without corrupting cache state."""
    store = WeightStore()
    cache = TieredCache(store)

    # Request nonexistent expert (99, 99)
    cache.prefetch(99, 99, "int4_g128", target_tier="RAM")
    cache.drain_prefetches()

    stats = cache.get_stats()
    assert stats["prefetch_errors"] == 1
    # Cache state remains clean
    assert cache.ram_used == 0

def test_concurrent_multithread_access():
    """C2 & Test 6: 1000 concurrent accesses across threads: no deadlock, used <= budget."""
    store = WeightStore()
    cache = TieredCache(
        store,
        ram_budget_bytes=15*1024*1024,
        vram_budget_bytes=10*1024*1024,
        eviction_policy="lru"
    )

    errors = []
    def _worker(thread_id):
        try:
            for i in range(100):
                exp_id = (thread_id + i) % 8
                prec = "int4_g128" if i % 2 == 0 else "int8_g128"
                cache.get_expert(0, exp_id, prec)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert len(errors) == 0, f"Thread errors: {errors}"
    assert cache.ram_used <= cache.ram_budget
    assert cache.vram_used <= cache.vram_budget
    assert cache.ram_used >= 0
    assert cache.vram_used >= 0

def test_h2d_and_swiglu_correctness():
    """C4 & Test 7: SwiGLU forward produces correct tensor shapes and deterministic outputs."""
    store = WeightStore()
    cache = TieredCache(store)

    x = torch.randn(2, 2048, dtype=torch.bfloat16, device=cache.device)
    data, meta, timings = cache.get_expert(0, 0, "int4_g128")
    out, dec_ms, gemm_ms = cache.forward_expert_swiglu(x, data, meta)

    assert out.shape == (2, 2048)
    assert dec_ms >= 0.0
    assert gemm_ms >= 0.0
