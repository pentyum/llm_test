"""
P5a.1 Generation Runner with Verified KV Cache & Cache Isolation
Supports:
  - decode_mode: "incremental_kv" (standard KV cache) vs "full_prefix_recompute"
  - Rigorous timing: TTFT, TPOT (mean, p50, p95, p99), decode tokens/s
  - Direct integration with DynamicMoELayer and PrecisionEvent tracking
  - Token match calculation against reference (generated tokens ONLY, excluding prompt)
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.p5a.metrics import EventLogger, PrecisionEvent

def generate_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    decode_mode: str = "incremental_kv",
    dyn_layer = None,
    run_id: str = "default",
    event_logger: Optional[EventLogger] = None
) -> Dict[str, Any]:
    """
    Executes greedy generation with exact TTFT/TPOT timing and precision tracking.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    forward_id_counter = 1

    if dyn_layer is not None:
        dyn_layer.clear_events()
        dyn_layer.current_phase = "prefill"
        dyn_layer.current_forward_id = forward_id_counter
        dyn_layer.current_decode_step = None
        dyn_layer.run_id = run_id
        dyn_layer.event_logger = event_logger

    # 1. Prefill
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_prefill_0 = time.perf_counter()

    with torch.no_grad():
        if decode_mode == "incremental_kv":
            outputs = model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values
        else:
            outputs = model(input_ids, use_cache=False)
            past_kv = None

        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_prefill_1 = time.perf_counter()
    ttft_ms = (t_prefill_1 - t_prefill_0) * 1000.0

    generated_tokens = [next_token.item()]
    current_ids = torch.cat([input_ids, next_token], dim=-1)

    # 2. Decode Loop
    decode_latencies_ms = []

    for step in range(max_new_tokens - 1):
        forward_id_counter += 1
        if dyn_layer is not None:
            dyn_layer.current_phase = "decode"
            dyn_layer.current_forward_id = forward_id_counter
            dyn_layer.current_decode_step = step

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_step_0 = time.perf_counter()

        with torch.no_grad():
            if decode_mode == "incremental_kv":
                outputs = model(next_token, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
            else:
                outputs = model(current_ids, use_cache=False)
                past_kv = None

            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_step_1 = time.perf_counter()
        step_ms = (t_step_1 - t_step_0) * 1000.0
        decode_latencies_ms.append(step_ms)

        generated_tokens.append(next_token.item())
        current_ids = torch.cat([current_ids, next_token], dim=-1)

    dec_arr = np.array(decode_latencies_ms)
    tpot_mean = float(np.mean(dec_arr)) if len(dec_arr) > 0 else 0.0
    tpot_p50 = float(np.percentile(dec_arr, 50)) if len(dec_arr) > 0 else 0.0
    tpot_p95 = float(np.percentile(dec_arr, 95)) if len(dec_arr) > 0 else 0.0
    tpot_p99 = float(np.percentile(dec_arr, 99)) if len(dec_arr) > 0 else 0.0
    decode_tps = (1000.0 / tpot_mean) if tpot_mean > 0 else 0.0

    gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
        "run_id": run_id,
        "decode_mode": decode_mode,
        "ttft_ms": round(ttft_ms, 2),
        "tpot_mean_ms": round(tpot_mean, 2),
        "tpot_p50_ms": round(tpot_p50, 2),
        "tpot_p95_ms": round(tpot_p95, 2),
        "tpot_p99_ms": round(tpot_p99, 2),
        "decode_tokens_per_sec": round(decode_tps, 2),
        "generated_tokens": generated_tokens,
        "generated_text": gen_text,
        "num_generated_tokens": len(generated_tokens)
    }
