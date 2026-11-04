"""
P5b Gate G1: Multi-Layer Offloaded MoE Manager
Manages simultaneous replacement of multiple MoE layers:
  - L1: [0]
  - L2: [0, 4]
  - L4: [0, 4, 8, 12]
Ensures distinct (layer_id, expert_id) keying, strict gate weight identity,
and support for restore / mode switching across all offloaded layers.
"""

import os
import sys
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
        dequant_backend: str = "torch_cuda"
    ):
        self.model = model
        self.target_layers = sorted(list(target_layers))
        self.cache = cache
        self.mode = mode
        self.policy = policy
        self.dequant_backend = dequant_backend

        self.orig_mlps: Dict[int, nn.Module] = {}
        self.dyn_layers: Dict[int, DynamicMoELayer] = {}
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
                dequant_backend=self.dequant_backend
            )
            self.dyn_layers[l] = dyn
            self.model.model.layers[l].mlp = dyn

        self.is_attached = True

    def restore(self):
        """Restores original MLPs in the model."""
        if not self.is_attached:
            return

        for l in self.target_layers:
            self.model.model.layers[l].mlp = self.orig_mlps[l]

        self.dyn_layers.clear()
        self.orig_mlps.clear()
        self.is_attached = False

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
