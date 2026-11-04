"""
Weight Store for Pre-quantized MoE Experts on SSD
Handles:
  - Metadata serialization and SHA256 checksum verification
  - Per-expert multi-precision file loading and storage accounting
  - Strict separation of offline saved artifacts vs online reading
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, Tuple
import torch

class WeightStore:
    def __init__(self, root_dir: str = "/home/yaoyao/llm_test/artifacts/experts"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        self.index_file = os.path.join(self.root_dir, "manifest.json")
        self.manifest: Dict[str, Any] = {}
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except Exception:
                self.manifest = {}

    @staticmethod
    def compute_sha256(filepath: str) -> str:
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_expert_path(self, layer_id: int, expert_id: int, precision: str) -> str:
        prec_dir = os.path.join(self.root_dir, precision)
        os.makedirs(prec_dir, exist_ok=True)
        return os.path.join(prec_dir, f"l{layer_id}_e{expert_id}.pt")

    def get_meta_path(self, layer_id: int, expert_id: int, precision: str) -> str:
        prec_dir = os.path.join(self.root_dir, precision)
        return os.path.join(prec_dir, f"l{layer_id}_e{expert_id}.json")

    def save_expert(self, layer_id: int, expert_id: int, precision: str, data: Dict[str, Any], meta: Dict[str, Any]):
        file_path = self.get_expert_path(layer_id, expert_id, precision)
        meta_path = self.get_meta_path(layer_id, expert_id, precision)

        torch.save(data, file_path)
        file_bytes = os.path.getsize(file_path)
        checksum = self.compute_sha256(file_path)

        meta.update({
            "version": "1.0",
            "layer_id": layer_id,
            "expert_id": expert_id,
            "precision": precision,
            "file_bytes": file_bytes,
            "checksum": checksum,
            "saved_at": time.time()
        })

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        key = f"{layer_id}_{expert_id}_{precision}"
        self.manifest[key] = {
            "file_path": file_path,
            "meta_path": meta_path,
            "file_bytes": file_bytes,
            "checksum": checksum,
            "precision": precision
        }

    def save_manifest(self):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def load_expert(self, layer_id: int, expert_id: int, precision: str, verify_checksum: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], float, int]:
        """
        Loads expert data and metadata from SSD.
        Returns: (data_dict, meta_dict, io_ms, bytes_read)
        """
        file_path = self.get_expert_path(layer_id, expert_id, precision)
        meta_path = self.get_meta_path(layer_id, expert_id, precision)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expert file not found: {file_path}")

        t0 = time.perf_counter()
        file_bytes = os.path.getsize(file_path)
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        t1 = time.perf_counter()
        io_ms = (t1 - t0) * 1000.0

        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        if verify_checksum:
            curr_hash = self.compute_sha256(file_path)
            if curr_hash != meta.get("checksum"):
                raise ValueError(f"Checksum mismatch for {file_path}: expected {meta.get('checksum')}, got {curr_hash}")

        return data, meta, io_ms, file_bytes
