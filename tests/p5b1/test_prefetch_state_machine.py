import pytest
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.p5b.prefetch_manager import PrefetchManager, PrefetchState

REAL_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

def test_prefetch_state_machine_effective_hit():
    """G2 Verification 1: Normal flow: trigger -> RAM_READY -> on_access -> CONSUMED."""
    store = WeightStore(root_dir=REAL_DIR)
    cache = TieredCache(store)
    mgr = PrefetchManager(cache, strategy="static_layer_prefetch")

    mgr.trigger_prefetch(0, 10, "int4_g128")
    k = "0:10:int4_g128"
    assert k in mgr.tasks
    assert mgr.tasks[k].state == PrefetchState.RAM_READY

    # Access expert
    mgr.on_expert_access(0, 10, "int4_g128")
    assert mgr.tasks[k].state == PrefetchState.CONSUMED

    stats = mgr.get_prefetch_stats()
    assert stats["prefetch_issued"] == 1
    assert stats["effective_hits"] == 1
    assert stats["late_prefetches"] == 0

def test_prefetch_late_detection_with_simulated_stall():
    """G2 Verification 2: Detects LATE prefetch when access happens before ready."""
    store = WeightStore(root_dir=REAL_DIR)
    cache = TieredCache(store)
    mgr = PrefetchManager(cache, strategy="static_layer_prefetch")

    # Manually insert a task in READING state
    k = "4:12:int4_g128"
    from src.p5b.prefetch_manager import PrefetchTask
    mgr.tasks[k] = PrefetchTask(
        request_id="req_test", layer_id=4, expert_id=12, precision="int4_g128",
        state=PrefetchState.READING, issued_ts=time.perf_counter()
    )

    # Access immediately while still READING
    mgr.on_expert_access(4, 12, "int4_g128")
    assert mgr.tasks[k].state == PrefetchState.LATE

    stats = mgr.get_prefetch_stats()
    assert stats["late_prefetches"] == 1
    assert stats["effective_hits"] == 0

def test_prefetch_coalescing():
    """G2 Verification 3: Consecutive duplicate prefetch requests are safely coalesced."""
    store = WeightStore(root_dir=REAL_DIR)
    cache = TieredCache(store)
    mgr = PrefetchManager(cache)

    for _ in range(5):
        mgr.trigger_prefetch(0, 5, "int4_g128")

    stats = mgr.get_prefetch_stats()
    assert stats["prefetch_issued"] == 1
    assert stats["coalesced_requests"] == 4
