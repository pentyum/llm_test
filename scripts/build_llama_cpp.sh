#!/bin/bash
set -ex
cd /home/yaoyao/llm_test

if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
fi

cd llama.cpp
git rev-parse HEAD | tee /home/yaoyao/llm_test/logs/llama_cpp_commit.txt

# Configure with CUDA enabled for RTX 4060 (sm_89)
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89"
cmake --build build -j "$(nproc)" --target llama-cli

# Check help options
./build/bin/llama-cli --help > /home/yaoyao/llm_test/logs/llama_help.txt 2>&1

echo "=== llama.cpp build complete ==="
grep -E -- '--cpu-moe|--moe-cache-mib|--n-gpu-layers|--gpu-layers|-ngl' /home/yaoyao/llm_test/logs/llama_help.txt || true
