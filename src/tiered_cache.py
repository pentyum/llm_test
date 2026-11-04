"""
Robust Tiered Cache for MoE Multi-Precision Experts (P4.5 Production Grade)
Strict constraints:
  - Strict budget enforcement: used_bytes <= budget_bytes at all times.
  - Zero-budget or oversized objects are bypassed and never stored.
  - Precise decoupling of hit metrics: VRAM hit vs RAM hit vs any-precision hit.
  - Fine-grained locking: SSD I/O and CUDA sync never hold the global lock.
  - De-duplicated background prefetch with in-flight future tracking and error recovery.
"""

import os
import sys
import time
import threading
from typing import Dict, Tuple, Optional, Any, List
from collections import OrderedDict
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weight_store import WeightStore
from src.quant_codec import dequantize_int8_g128, dequantize_int4_g128

CacheKey = Tuple[int, int, str, str]  # (layer_id, expert_id, precision, format_version)

def calculate_tensors_bytes(data: Dict[str, Any]) -> int:
    """Calculate actual memory occupied by PyTorch tensors."""
    total = 0
    for v in data.values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
    return total

class TieredCache:
    def __init__(
        self,
        weight_store: WeightStore,
        ram_budget_bytes: int = 2 * 1024 * 1024 * 1024,   # 2 GiB default
        vram_budget_bytes: int = 512 * 1024 * 1024,       # 512 MiB default
        eviction_policy: str = "lru",                     # "lru" or "cost_aware"
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        format_version: str = "1.0"
    ):
        self.store = weight_store
        self.ram_budget = max(0, ram_budget_bytes)
        self.vram_budget = max(0, vram_budget_bytes)
        self.policy = eviction_policy
        self.device = device
        self.format_version = format_version

        self.ram_used = 0
        self.vram_used = 0

        # Cache key -> (data_dict, meta_dict, size_bytes, last_accessed_time, access_count)
        self.ram_cache: Dict[CacheKey, Tuple[Dict[str, Any], Dict[str, Any], int, float, int]] = OrderedDict()
        self.vram_cache: Dict[CacheKey, Tuple[Dict[str, Any], Dict[str, Any], int, float, int]] = OrderedDict()

        # In-flight request tracking for de-duplication: CacheKey -> threading.Event
        self.in_flight: Dict[CacheKey, threading.Event] = {}
        self.in_flight_results: Dict[CacheKey, Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Exception], int]] = {}

        self.lock = threading.RLock()
        self.prefetch_threads: List[threading.Thread] = []

        # Granular metrics (All counts strictly defined)
        self.stats = {
            "requests": 0,
            "vram_target_hits": 0,
            "ram_target_hits": 0,
            "any_precision_hits": 0,
            "ssd_reads": 0,
            "ssd_read_bytes": 0,
            "ram_to_vram_bytes": 0,
            "vram_evictions": 0,
            "ram_evictions": 0,
            "bypass_vram_count": 0,
            "bypass_ram_count": 0,
            "prefetch_requests": 0,
            "prefetch_duplicates": 0,
            "prefetch_hits": 0,
            "prefetch_errors": 0,
            "total_io_ms": 0.0,
            "total_copy_ms": 0.0,
        }

    def _make_key(self, layer_id: int, expert_id: int, precision: str) -> CacheKey:
        return (layer_id, expert_id, precision, self.format_version)

    def _evict_tier_locked(self, tier: str, needed_bytes: int):
        """Must be called while holding self.lock."""
        target_cache = self.vram_cache if tier == "vram" else self.ram_cache
        used = self.vram_used if tier == "vram" else self.ram_used
        budget = self.vram_budget if tier == "vram" else self.ram_budget

        if budget == 0:
            return

        while used + needed_bytes > budget and target_cache:
            if self.policy == "lru":
                evict_key, entry = target_cache.popitem(last=False)
            elif self.policy == "cost_aware":
                now = time.time()
                min_key = None
                min_score = float("inf")
                for k, v in target_cache.items():
                    size = v[2]
                    count = v[4]
                    recency = max(now - v[3], 0.001)
                    score = (count / recency) / max(size, 1024)
                    if score < min_score:
                        min_score = score
                        min_key = k
                evict_key = min_key
                entry = target_cache.pop(evict_key)
            else:
                evict_key, entry = target_cache.popitem(last=False)

            entry_bytes = entry[2]
            used -= entry_bytes
            if tier == "vram":
                self.stats["vram_evictions"] += 1
            else:
                self.stats["ram_evictions"] += 1

        if tier == "vram":
            self.vram_used = used
        else:
            self.ram_used = used

    def prefetch(self, layer_id: int, expert_id: int, precision: str, target_tier: str = "RAM"):
        """De-duplicated non-blocking prefetch."""
        key = self._make_key(layer_id, expert_id, precision)
        event = None

        with self.lock:
            self.stats["prefetch_requests"] += 1
            # Check already cached
            if target_tier == "VRAM" and key in self.vram_cache:
                self.stats["prefetch_hits"] += 1
                return
            if target_tier == "RAM" and (key in self.ram_cache or key in self.vram_cache):
                self.stats["prefetch_hits"] += 1
                return

            # Check in-flight
            if key in self.in_flight:
                self.stats["prefetch_duplicates"] += 1
                return

            event = threading.Event()
            self.in_flight[key] = event

        def _worker():
            data_cpu = None
            meta = None
            err = None
            f_bytes = 0
            try:
                data_cpu, meta, io_ms, f_bytes = self.store.load_expert(layer_id, expert_id, precision)
            except Exception as e:
                err = e
            
            with self.lock:
                self.in_flight_results[key] = (data_cpu, meta, err, f_bytes)
                event.set()

                if err is not None:
                    self.stats["prefetch_errors"] += 1
                    del self.in_flight[key]
                    del self.in_flight_results[key]
                    return

                # Record SSD stats once
                self.stats["ssd_reads"] += 1
                self.stats["ssd_read_bytes"] += f_bytes

                tensor_bytes = calculate_tensors_bytes(data_cpu)
                size_to_account = max(tensor_bytes, f_bytes)

                # Store into RAM if fits
                if self.ram_budget > 0 and size_to_account <= self.ram_budget:
                    if key in self.ram_cache:
                        self.ram_used -= self.ram_cache[key][2]
                    self._evict_tier_locked("ram", size_to_account)
                    if self.ram_used + size_to_account <= self.ram_budget:
                        self.ram_cache[key] = (data_cpu, meta, size_to_account, time.time(), 1)
                        self.ram_used += size_to_account
                    else:
                        self.stats["bypass_ram_count"] += 1
                else:
                    self.stats["bypass_ram_count"] += 1

                # Clean up in-flight
                if key in self.in_flight:
                    del self.in_flight[key]
                if key in self.in_flight_results:
                    del self.in_flight_results[key]

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        self.prefetch_threads.append(th)

    def drain_prefetches(self):
        """Wait for all running prefetch threads to finish."""
        for th in self.prefetch_threads:
            if th.is_alive():
                th.join(timeout=5.0)
        self.prefetch_threads.clear()

    def get_expert(
        self,
        layer_id: int,
        expert_id: int,
        target_precision: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
        """
        Retrieves expert into device VRAM.
        Returns: (tensors_on_device, meta_dict, timings_dict)
        """
        key = self._make_key(layer_id, expert_id, target_precision)
        t_start = time.perf_counter()

        timings = {
            "lookup_ms": 0.0,
            "ssd_read_ms": 0.0,
            "h2d_copy_ms": 0.0,
            "total_ms": 0.0,
            "hit_tier": "SSD"
        }

        # 1. Fast path: Check VRAM Hit under lock
        with self.lock:
            self.stats["requests"] += 1

            # Check any precision hit
            any_hit = any(k[0] == layer_id and k[1] == expert_id for k in self.vram_cache) or \
                      any(k[0] == layer_id and k[1] == expert_id for k in self.ram_cache)
            if any_hit:
                self.stats["any_precision_hits"] += 1

            if key in self.vram_cache:
                entry = self.vram_cache[key]
                self.vram_cache.move_to_end(key)
                self.vram_cache[key] = (entry[0], entry[1], entry[2], time.time(), entry[4] + 1)
                self.stats["vram_target_hits"] += 1
                timings["hit_tier"] = "VRAM"
                timings["lookup_ms"] = (time.perf_counter() - t_start) * 1000.0
                timings["total_ms"] = timings["lookup_ms"]
                return entry[0], entry[1], timings

            # Check if this key is in-flight prefetching
            in_flight_event = self.in_flight.get(key)

        # If in-flight, wait without holding lock
        if in_flight_event is not None:
            in_flight_event.wait(timeout=5.0)

        # 2. Check RAM Hit under lock
        data_cpu = None
        meta = None
        size_to_account = 0
        from_ssd = False

        with self.lock:
            if key in self.ram_cache:
                entry = self.ram_cache[key]
                self.ram_cache.move_to_end(key)
                self.ram_cache[key] = (entry[0], entry[1], entry[2], time.time(), entry[4] + 1)
                data_cpu, meta, size_to_account = entry[0], entry[1], entry[2]
                self.stats["ram_target_hits"] += 1
                timings["hit_tier"] = "RAM"
            else:
                from_ssd = True

        # 3. Cold SSD Load (Execute outside lock!)
        if from_ssd:
            t_io_0 = time.perf_counter()
            data_cpu, meta, _, f_bytes = self.store.load_expert(layer_id, expert_id, target_precision)
            t_io_1 = time.perf_counter()
            timings["ssd_read_ms"] = (t_io_1 - t_io_0) * 1000.0
            timings["hit_tier"] = "SSD"

            tensor_bytes = calculate_tensors_bytes(data_cpu)
            size_to_account = max(tensor_bytes, f_bytes)

            with self.lock:
                self.stats["ssd_reads"] += 1
                self.stats["ssd_read_bytes"] += f_bytes
                self.stats["total_io_ms"] += timings["ssd_read_ms"]

                # Insert to RAM if budget allows
                if self.ram_budget > 0 and size_to_account <= self.ram_budget:
                    if key in self.ram_cache:
                        self.ram_used -= self.ram_cache[key][2]
                    self._evict_tier_locked("ram", size_to_account)
                    if self.ram_used + size_to_account <= self.ram_budget:
                        self.ram_cache[key] = (data_cpu, meta, size_to_account, time.time(), 1)
                        self.ram_used += size_to_account
                    else:
                        self.stats["bypass_ram_count"] += 1
                else:
                    self.stats["bypass_ram_count"] += 1

        # 4. Host-to-Device Copy (Outside lock)
        t_copy_0 = time.perf_counter()
        gpu_data = {}
        for k, v in data_cpu.items():
            if isinstance(v, torch.Tensor):
                gpu_data[k] = v.to(self.device, non_blocking=False)
            else:
                gpu_data[k] = v

        if self.device == "cuda":
            torch.cuda.synchronize()
        t_copy_1 = time.perf_counter()
        timings["h2d_copy_ms"] = (t_copy_1 - t_copy_0) * 1000.0

        # 5. Insert into VRAM under lock (or bypass if budget=0 or too large)
        with self.lock:
            self.stats["ram_to_vram_bytes"] += size_to_account
            self.stats["total_copy_ms"] += timings["h2d_copy_ms"]

            if self.vram_budget > 0 and size_to_account <= self.vram_budget:
                if key in self.vram_cache:
                    self.vram_used -= self.vram_cache[key][2]
                self._evict_tier_locked("vram", size_to_account)
                if self.vram_used + size_to_account <= self.vram_budget:
                    self.vram_cache[key] = (gpu_data, meta, size_to_account, time.time(), 1)
                    self.vram_used += size_to_account
                else:
                    self.stats["bypass_vram_count"] += 1
            else:
                self.stats["bypass_vram_count"] += 1

        timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0
        return gpu_data, meta, timings

    def forward_expert_swiglu(
        self,
        x: torch.Tensor,
        gpu_data: Dict[str, Any],
        meta: Dict[str, Any]
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Executes SwiGLU forward: E(x) = down( silu(gate(x)) * up(x) )
        Synchronizes boundaries for exact timing separation: decode_ms vs pure gemm_ms.
        """
        precision = meta.get("precision", "bf16")
        group_size = meta.get("group_size", 128)

        if self.device == "cuda":
            torch.cuda.synchronize()
        t_dec_0 = time.perf_counter()

        if precision == "bf16":
            gate_w = gpu_data["gate_proj"]
            up_w = gpu_data["up_proj"]
            down_w = gpu_data["down_proj"]
        elif precision.startswith("int8"):
            gate_w = dequantize_int8_g128(gpu_data["gate_proj"], gpu_data["gate_scales"], group_size)
            up_w = dequantize_int8_g128(gpu_data["up_proj"], gpu_data["up_scales"], group_size)
            down_w = dequantize_int8_g128(gpu_data["down_proj"], gpu_data["down_scales"], group_size)
        elif precision.startswith("int4"):
            gate_w = dequantize_int4_g128(gpu_data["gate_proj"], gpu_data["gate_scales"], gpu_data["gate_shape"], group_size)
            up_w = dequantize_int4_g128(gpu_data["up_proj"], gpu_data["up_scales"], gpu_data["up_shape"], group_size)
            down_w = dequantize_int4_g128(gpu_data["down_proj"], gpu_data["down_scales"], gpu_data["down_shape"], group_size)
        else:
            raise ValueError(f"Unsupported precision: {precision}")

        if self.device == "cuda":
            torch.cuda.synchronize()
        t_dec_1 = time.perf_counter()
        decode_ms = (t_dec_1 - t_dec_0) * 1000.0

        # Ensure input x is on the same device as the weights
        x = x.to(device=gate_w.device, dtype=gate_w.dtype)

        t_gemm_0 = time.perf_counter()
        gate_out = torch.matmul(x, gate_w.T)
        silu_out = torch.nn.functional.silu(gate_out)
        up_out = torch.matmul(x, up_w.T)
        inter = silu_out * up_out
        out = torch.matmul(inter, down_w.T)

        if self.device == "cuda":
            torch.cuda.synchronize()
        t_gemm_1 = time.perf_counter()
        gemm_ms = (t_gemm_1 - t_gemm_0) * 1000.0

        return out, decode_ms, gemm_ms

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            s = dict(self.stats)
            reqs = max(s["requests"], 1)
            s["vram_target_hit_rate"] = round(s["vram_target_hits"] / reqs, 5)
            s["ram_target_hit_rate"] = round(s["ram_target_hits"] / reqs, 5)
            s["ssd_miss_rate"] = round(s["ssd_reads"] / reqs, 5)
            s["any_precision_hit_rate"] = round(s["any_precision_hits"] / reqs, 5)
            s["vram_accounted_mib"] = round(self.vram_used / (1024 * 1024), 2)
            s["ram_accounted_mib"] = round(self.ram_used / (1024 * 1024), 2)
            s["vram_budget_mib"] = round(self.vram_budget / (1024 * 1024), 2)
            s["ram_budget_mib"] = round(self.ram_budget / (1024 * 1024), 2)
            return s
