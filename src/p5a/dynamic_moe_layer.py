"""
P5a.1: Single-Layer Dynamic Precision MoE Module for Real OLMoE Inference
Replaces an OlmoeSparseMoeBlock with a tiered-cache-backed expert executor.
Supports:
  1. bf16_passthrough: 100% numerical verification against reference
  2. fixed_int8: All experts load offline pre-quantized INT8
  3. fixed_int4: All experts load offline pre-quantized INT4
  4. dynamic_s4: Token-level runtime precision switching based on S4 policies
  5. forced: Injected test-only ForcedSequencePolicy
Logs high-precision PrecisionEvent instances into EventLogger & switch_events.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tiered_cache import TieredCache
from src.weight_store import WeightStore
from src.p5a.metrics import PrecisionEvent, EventLogger

class DynamicMoELayer(nn.Module):
    def __init__(
        self,
        orig_moe_block: nn.Module,
        layer_id: int,
        cache: TieredCache,
        mode: str = "bf16_passthrough",
        policy = None,
        dequant_backend: str = "torch_cuda",
        prefetch_manager = None
    ):
        super().__init__()
        self.layer_id = layer_id
        self.cache = cache
        self.mode = mode
        self.policy = policy
        self.dequant_backend = dequant_backend
        self.prefetch_manager = prefetch_manager

        # Retain identical gate network
        self.gate = orig_moe_block.gate
        self.num_experts = getattr(self.gate, "num_experts", 64)
        self.top_k = getattr(self.gate, "top_k", 8)

        # Contextual tracking fields set by generation runner
        self.run_id: str = "default"
        self.current_phase: str = "decode"
        self.current_forward_id: int = 1
        self.current_decode_step: Optional[int] = 0
        self.event_logger: Optional[EventLogger] = None

        self.switch_events: List[PrecisionEvent] = []

    def set_mode(self, new_mode: str):
        self.mode = new_mode

    def set_policy(self, new_policy):
        self.policy = new_policy

    def clear_events(self):
        self.switch_events.clear()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch_size, sequence_length, hidden_dim)
        Returns: final_hidden_states (same signature as OlmoeSparseMoeBlock)
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)  # (N, hidden_dim)
        num_tokens = flat_hidden_states.shape[0]

        # 1. Gate computation
        gate_out = self.gate(flat_hidden_states)
        if isinstance(gate_out, tuple):
            router_logits, routing_weights, selected_experts = gate_out
        else:
            router_logits = gate_out
            probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(probs, self.top_k, dim=-1)

        routing_weights = routing_weights.to(hidden_states.dtype)
        if self.prefetch_manager is not None:
            self.prefetch_manager.on_layer_enter(self.layer_id, self.current_decode_step or 0)
        final_hidden_states = torch.zeros_like(flat_hidden_states)

        # 2. Token-level Expert Execution Loop
        for tok_idx in range(num_tokens):
            tok_x = flat_hidden_states[tok_idx : tok_idx + 1]

            for rank in range(self.top_k):
                exp_id = int(selected_experts[tok_idx, rank].item())
                score = float(routing_weights[tok_idx, rank].item())

                # Determine target precision
                if self.mode == "bf16_passthrough":
                    target_prec = "bf16"
                elif self.mode == "fixed_int8":
                    target_prec = "int8_g128"
                elif self.mode == "fixed_int4":
                    target_prec = "int4_g128"
                elif self.policy is not None:
                    target_prec = self.policy.select(self.layer_id, exp_id, score, self.cache)
                else:
                    target_prec = "int4_g128"

                if self.prefetch_manager is not None:
                    self.prefetch_manager.on_expert_access(self.layer_id, exp_id, target_prec)
                # Load pre-quantized weights from TieredCache
                gpu_data, meta, timings = self.cache.get_expert(self.layer_id, exp_id, target_prec)

                # Execute SwiGLU forward with configured dequant_backend
                out_exp, dec_ms, gemm_ms = self.cache.forward_expert_swiglu(tok_x, gpu_data, meta, dequant_backend=self.dequant_backend)
                final_hidden_states[tok_idx] += score * out_exp.squeeze(0).to(hidden_states.device)

                # Record strict PrecisionEvent
                ev = PrecisionEvent(
                    run_id=self.run_id,
                    mode=self.mode,
                    phase=self.current_phase,
                    forward_id=self.current_forward_id,
                    generated_token_step=self.current_decode_step if self.current_phase == "decode" else None,
                    token_idx_in_forward=tok_idx,
                    layer_id=self.layer_id,
                    expert_id=exp_id,
                    topk_rank=rank,
                    router_score=round(score, 5),
                    requested_precision=target_prec,
                    loaded_precision=meta["precision"],
                    executed_precision=meta["precision"],
                    hit_tier=timings.get("hit_tier", "VRAM"),
                    cache_key=f"{self.layer_id}:{exp_id}:{meta['precision']}:1.0",
                    fetch_ms=round(timings.get("total_ms", 0.0), 4),
                    dequant_ms=round(dec_ms, 4),
                    matmul_ms=round(gemm_ms, 4),
                    total_ms=round(timings.get("total_ms", 0.0) + dec_ms + gemm_ms, 4)
                )

                self.switch_events.append(ev)
                if self.event_logger is not None:
                    self.event_logger.log(ev)

        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        return final_hidden_states
