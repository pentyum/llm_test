#!/bin/bash
set -ex
source /home/yaoyao/miniconda3/etc/profile.d/conda.sh
conda activate moe-precision-lab

mkdir -p /home/yaoyao/llm_test/models/hf_original

python - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = "allenai/OLMoE-1B-7B-0125-Instruct"
local_dir = "/home/yaoyao/llm_test/models/hf_original"

files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors"
]

for fname in files:
    print(f"Downloading {fname}...")
    p = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=local_dir)
    print(f"Downloaded {fname} ({os.path.getsize(p) / (1024**2):.2f} MB)")
PY

ls -lh /home/yaoyao/llm_test/models/hf_original/
