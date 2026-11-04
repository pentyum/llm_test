"""
P5b Gate G1 & G4: Multi-Layer Offloaded MoE Manager with Safe Restore Semantics
Features:
  - Simultaneous replacement of multiple MoE layers: L1 [0], L2 [0,4], L4 [0,4,8,12]
  - True expert parameter unloading: When release_orig_experts=True, deletes original
    BF16 experts from memory (orig_mlp.experts = None) to eliminate RAM dual-copy retention.
  - Safe Restore Semantics:
    When restoring, if experts were released:
    - If allow_reconstruct=True: safely reconstructs OlmoeExperts from WeightStore BF16 data.
    - If allow_reconstruct=False: raises explicit RuntimeError instead of leaving broken NoneType models.
"""

import os
import sys
import gc
import copy
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tiered_cache import TieredCache
from src.p5a.dynamic_moe_layer import DynamicMoELayer

class MultiLayerOffloadedMoE:
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[int],
        cache: TieredCache,
        mode: str = "bf16_passthrough",
        policy = None,
        dequant_backend: str = "torch_cuda",
        release_orig_experts: bool = False,
        prefetch_manager = None
    ):
        self.model = model
        self.target_layers = sorted(list(target_layers))
        self.cache = cache
        self.mode = mode
        self.policy = policy
        self.dequant_backend = dequant_backend
        self.release_orig_experts = release_orig_experts
        self.prefetch_manager = prefetch_manager

        self.orig_mlps: Dict[int, nn.Module] = {}
        self.dyn_layers: Dict[int, DynamicMoELayer] = {}
        self.orig_experts_prototypes: Dict[int, Any] = {}
        self.is_attached = False

        self.attach()

    def attach(self):
        """Replaces target layers in model with DynamicMoELayers."""
        if self.is_attached:
            return

        for l in self.target_layers:
            orig_mlp = self.model.model.layers[l].mlp
            self.orig_mlps[l] = orig_mlp

            dyn = DynamicMoELayer(
                orig_moe_block=orig_mlp,
                layer_id=l,
                cache=self.cache,
                mode=self.mode,
                policy=self.policy,
                dequant_backend=self.dequant_backend,
                prefetch_manager=self.prefetch_manager
            )
            self.dyn_layers[l] = dyn
            self.model.model.layers[l].mlp = dyn

            if self.release_orig_experts:
                if hasattr(orig_mlp, "experts") and orig_mlp.experts is not None:
                    # Save a shallow prototype configuration for safe reconstruction if restored
                    self.orig_experts_prototypes[l] = type(orig_mlp.experts)
                    orig_mlp.experts = None

        if self.release_orig_experts:
            gc.collect()

        self.is_attached = True

    def restore(self, allow_reconstruct: bool = True):
        """
        Safely restores original MLPs.
        If experts were released, reconstructs them from WeightStore if allow_reconstruct=True,
        or raises an explicit RuntimeError if allow_reconstruct=False.
        """
        if not self.is_attached:
            return

        for l in self.target_layers:
            orig_mlp = self.orig_mlps[l]

            if orig_mlp.experts is None:
                if not allow_reconstruct:
                    raise RuntimeError(
                        f"Cannot restore layer {l} because original experts were released to free memory "
                        "and allow_reconstruct=False. Explicitly set allow_reconstruct=True to rebuild from store."
                    )
                # Reconstruct OlmoeExperts from WeightStore BF16 weights
                self._reconstruct_experts_for_layer(l, orig_mlp)

            # Re-attach original MLP
            self.model.model.layers[l].mlp = orig_mlp

        self.dyn_layers.clear()
        self.orig_mlps.clear()
        self.orig_experts_prototypes.clear()
        self.is_attached = False

    def _reconstruct_experts_for_layer(self, layer_id: int, orig_mlp: nn.Module):
        """Rebuilds the OlmoeExperts module using verified offline BF16 expert weights."""
        store = self.cache.store
        gate_up_list = []
        down_list = []

        for e in range(64):
            data, _, _, _ = store.load_expert(layer_id, e, "bf16")
            gate = data["gate_proj"] # (1024, 2048)
            up = data["up_proj"]     # (1024, 2048)
            down = data["down_proj"] # (2048, 1024)

            gate_up = torch.cat([gate, up], dim=0) # (2048, 2048)
            gate_up_list.append(gate_up)
            down_list.append(down)

        full_gate_up = torch.stack(gate_up_list, dim=0).to(dtype=torch.bfloat16) # (64, 2048, 2048)
        full_down = torch.stack(down_list, dim=0).to(dtype=torch.bfloat16)       # (64, 2048, 1024)

        # Instantiate a clean OlmoeExperts
        from transformers.models.olmoe.modeling_olmoe import OlmoeExperts
        config = getattr(self.model, "config", None)
        if config is not None:
            new_experts = OlmoeExperts(config).to(dtype=torch.bfloat16, device="cpu")
            new_experts.gate_up_proj.data.copy_(full_gate_up)
            new_experts.down_proj.data.copy_(full_down)
            orig_mlp.experts = new_experts
        else:
            raise RuntimeError(f"Model config not found to instantiate OlmoeExperts for layer {layer_id}")

    def set_mode(self, new_mode: str):
        self.mode = new_mode
        for dyn in self.dyn_layers.values():
            dyn.set_mode(new_mode)

    def set_policy(self, new_policy):
        self.policy = new_policy
        for dyn in self.dyn_layers.values():
            dyn.set_policy(new_policy)

    def clear_events(self):
        for dyn in self.dyn_layers.values():
            dyn.clear_events()

    def get_all_events(self) -> List[Any]:
        all_events = []
        for dyn in self.dyn_layers.values():
            all_events.extend(dyn.switch_events)
        return all_events
