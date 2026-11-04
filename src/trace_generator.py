"""
Trace Generator for MoE Expert Access
Generates 4 synthetic workloads (10,000 requests each):
  1. uniform: Uniform random across all experts (low locality upper bound)
  2. zipf: Long-tail popular experts (parameter alpha=1.2)
  3. bursty: Temporal burstiness (local hot-set shifts periodically)
  4. mixed_precision: Requests alternate between INT4, INT8, and BF16
Saves traces as JSONL files to ensure exact reproducibility across all ablations.
"""

import os
import json
import random
import numpy as np

TRACE_DIR = "/home/yaoyao/llm_test/artifacts/traces"
os.makedirs(TRACE_DIR, exist_ok=True)

def generate_traces(num_requests: int = 10000, num_layers: int = 2, num_experts: int = 32, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    workloads = ["uniform", "zipf", "bursty", "mixed_precision"]
    precisions = ["int4_g128", "int8_g128", "bf16"]

    generated_files = {}

    for w_name in workloads:
        file_path = os.path.join(TRACE_DIR, f"trace_{w_name}.jsonl")
        records = []

        if w_name == "uniform":
            for req_id in range(num_requests):
                layer = random.randint(0, num_layers - 1)
                expert = random.randint(0, num_experts - 1)
                prec = "int4_g128"
                score = round(random.uniform(0.1, 1.0), 4)
                records.append({
                    "request_id": req_id,
                    "layer_id": layer,
                    "expert_id": expert,
                    "precision": prec,
                    "router_score": score,
                    "prefetch_distance": random.randint(1, 4)
                })

        elif w_name == "zipf":
            # Zipf distribution over expert IDs (1..num_experts)
            alpha = 1.2
            ranks = np.arange(1, num_experts + 1)
            weights = 1.0 / np.power(ranks, alpha)
            probs = weights / np.sum(weights)
            
            for req_id in range(num_requests):
                layer = random.randint(0, num_layers - 1)
                expert = int(np.random.choice(ranks, p=probs) - 1)
                prec = "int4_g128"
                score = round(random.uniform(0.1, 1.0), 4)
                records.append({
                    "request_id": req_id,
                    "layer_id": layer,
                    "expert_id": expert,
                    "precision": prec,
                    "router_score": score,
                    "prefetch_distance": random.randint(1, 4)
                })

        elif w_name == "bursty":
            # Periodically shift active hot set (e.g. every 200 requests, choose 4 hot experts)
            hot_set_size = 4
            current_hot = random.sample(range(num_experts), hot_set_size)

            for req_id in range(num_requests):
                if req_id % 200 == 0:
                    current_hot = random.sample(range(num_experts), hot_set_size)

                layer = random.randint(0, num_layers - 1)
                # 80% probability pick from hot set, 20% random
                if random.random() < 0.8:
                    expert = random.choice(current_hot)
                else:
                    expert = random.randint(0, num_experts - 1)

                prec = "int4_g128"
                score = round(random.uniform(0.1, 1.0), 4)
                records.append({
                    "request_id": req_id,
                    "layer_id": layer,
                    "expert_id": expert,
                    "precision": prec,
                    "router_score": score,
                    "prefetch_distance": random.randint(1, 4)
                })

        elif w_name == "mixed_precision":
            # Zipf popularity with precision determined by router_score
            alpha = 1.1
            ranks = np.arange(1, num_experts + 1)
            weights = 1.0 / np.power(ranks, alpha)
            probs = weights / np.sum(weights)

            for req_id in range(num_requests):
                layer = random.randint(0, num_layers - 1)
                expert = int(np.random.choice(ranks, p=probs) - 1)
                # Router score determines requested precision
                # High router score -> BF16, Medium -> INT8, Low -> INT4
                score = round(random.uniform(0.01, 1.0), 4)
                if score > 0.75:
                    prec = "bf16"
                elif score > 0.40:
                    prec = "int8_g128"
                else:
                    prec = "int4_g128"

                records.append({
                    "request_id": req_id,
                    "layer_id": layer,
                    "expert_id": expert,
                    "precision": prec,
                    "router_score": score,
                    "prefetch_distance": random.randint(1, 4)
                })

        with open(file_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        print(f"Generated {w_name} trace: {len(records)} requests -> {file_path}")
        generated_files[w_name] = file_path

    return generated_files

if __name__ == "__main__":
    generate_traces()
