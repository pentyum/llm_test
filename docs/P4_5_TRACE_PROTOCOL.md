# 真实模型路由轨迹采集协议 (P4_5_TRACE_PROTOCOL.md)

## 1. 采集模型与配置
- **模型**：`allenai/OLMoE-1B-7B-0125-Instruct`（官方 HuggingFace 原始权重）
- **拓扑架构**：16 层 Transformer Blocks，每层包含 64 个专家，每 Token 激活 Top-8 专家
- **参数规格**：`hidden_size = 2048`，`intermediate_size = 1024`，`norm_topk_prob = False`
- **权重参数形态**：`OlmoeExperts` 模块以融合张量存储，其中 `gate_up_proj` 为 `(64, 2048, 2048)`，`down_proj` 为 `(64, 2048, 1024)`

## 2. 真实前向输入数据集 (Prompts)
涵盖问答、长文摘要、代码实现、多步数学推理和多轮对话 5 类典型工作负载：
1. `prompt_qa`: `"Explain how mixture-of-experts language models route tokens."`
2. `prompt_summary`: `"Discuss the trade-offs between static integer quantization and dynamic precision caching in large language models."`
3. `prompt_code`: `"Write a high-performance Python class implementing an LRU cache with tiered memory eviction."`
4. `prompt_reasoning`: `"Solve the following step by step: A cache system with 3 tiers has latencies 2us, 200us, and 2ms. If the hit rates are 80%, 15%, and 5% respectively, calculate the average memory access time."`
5. `prompt_multiturn`: 多轮对话结构 prompt

## 3. 路由与隐藏状态提取流程
1. 在模型执行 forward 时开启 `output_router_logits=True`，获取所有 16 层的完整路由未归一化 logits；
2. 严格按模型官方算子执行 Softmax 计算路由概率，并提取 Top-8 专家的选中索引与概率权重；
3. 挂接 `forward_pre_hook` 拦截进入 MoE 层的真实隐藏状态张量（形状 `(batch, seq_len, 2048)`）；
4. 输出的每一行事件包含：`request_id`, `prompt_id`, `token_index`, `layer_id`, `expert_id`, `topk_rank`, `router_score`, `phase="prefill"`；
5. 结果持久化于 `artifacts/traces/trace_real_router.jsonl`（总计 17,536 个真实路由事件）。
