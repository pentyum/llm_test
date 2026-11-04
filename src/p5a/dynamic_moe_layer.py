"""
P5a: Single-Layer Dynamic Precision MoE Module for Real OLMoE Inference
Replaces an OlmoeSparseMoeBlock with a tiered-cache-backed expert executor.
Supports:
  1. bf16_passthrough: 100% numerical verification against reference
  2. fixed_int8: All experts load offline pre-quantized INT8
  3. fixed_int4: All experts load offline pre-quantized INT4
  4. dynamic_s4: Token-level runtime precision switching based on S4 cost model
Strictly prohibits online quantization of model weights.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tiered_cache import TieredCache
from src.weight_store import WeightStore

class DynamicMoELayer(nn.Module):
    def __init__(
        self,
        orig_moe_block: nn.Module,
        layer_id: int,
        cache: TieredCache,
        mode: str = "bf16_passthrough",
        scheduler = None
    ):
        super().__init__()
        self.layer_id = layer_id
        self.cache = cache
        self.mode = mode
        self.scheduler = scheduler

        # Retain identical gate network
        self.gate = orig_moe_block.gate
        self.num_experts = getattr(self.gate, "num_experts", 64)
        self.top_k = getattr(self.gate, "top_k", 8)

        # Logging of online precision switches
        self.switch_events: List[Dict[str, Any]] = []

    def set_mode(self, new_mode: str):
        self.mode = new_mode

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

        # 1. Official Gate computation: returns (router_logits, routing_weights, selected_experts)
        gate_out = self.gate(flat_hidden_states)
        if isinstance(gate_out, tuple):
            router_logits, routing_weights, selected_experts = gate_out
        else:
            router_logits = gate_out
            probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(probs, self.top_k, dim=-1)

        routing_weights = routing_weights.to(hidden_states.dtype)

        # Output accumulator on same device as hidden_states
        final_hidden_states = torch.zeros_like(flat_hidden_states)

        # 2. Expert Execution Loop (Token-level)
        for tok_idx in range(num_tokens):
            tok_x = flat_hidden_states[tok_idx : tok_idx + 1]  # (1, hidden_dim)

            for rank in range(self.top_k):
                exp_id = int(selected_experts[tok_idx, rank].item())
                score = float(routing_weights[tok_idx, rank].item())

                # Determine precision
                if self.mode == "bf16_passthrough":
                    prec = "bf16"
                elif self.mode == "fixed_int8":
                    prec = "int8_g128"
                elif self.mode == "fixed_int4":
                    prec = "int4_g128"
                elif self.mode == "dynamic_s4":
                    if self.scheduler is not None:
                        prec = self.scheduler.select("S4_cache_aware", self.layer_id, exp_id, score, self.cache)
                    else:
                        prec = "int8_g128" if score > 0.15 else "int4_g128"
                else:
                    prec = "bf16"

                # Load offline pre-quantized weights from TieredCache
                gpu_data, meta, timings = self.cache.get_expert(self.layer_id, exp_id, prec)

                # Record online switch event
                self.switch_events.append({
                    "token_idx": tok_idx,
                    "layer_id": self.layer_id,
                    "expert_id": exp_id,
                    "topk_rank": rank,
                    "router_score": round(score, 5),
                    "requested_precision": prec,
                    "hit_tier": timings.get("hit_tier", "VRAM"),
                    "fetch_latency_ms": round(timings.get("total_ms", 0.0), 3)
                })

                # Execute SwiGLU forward
                out_exp, _, _ = self.cache.forward_expert_swiglu(tok_x, gpu_data, meta)
                final_hidden_states[tok_idx] += score * out_exp.squeeze(0).to(hidden_states.device)

        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        return final_hidden_states
