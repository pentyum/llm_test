# RTX 4060 上的 MoE 多精度专家预存储与分层缓存实验

本项目在消费级硬件（NVIDIA GeForce RTX 4060 8GB VRAM + 48GB RAM + NVMe SSD）上，研究与验证面向 MoE（以 OLMoE-1B-7B 为代表）的多精度离线预存储、分层缓存（SSD→RAM→VRAM）与动态精度调度策略。

## 目录结构

- `configs/`：实验配置文件（YAML）
- `docs/`：设计文档与各阶段结论
- `src/`：核心模块（离线打包、编解码器、分层缓存、访问回放、调度策略）
- `scripts/`：环境安装、基线评测、测试回放脚本
- `tests/`：单元测试与正确性检验（编解码、缓存、无竞争检验）
- `models/`：GGUF 与 safetensors 模型文件（不入 Git）
- `artifacts/`：预打包的多精度专家数据（不入 Git）
- `logs/`：原始运行日志
- `results/`：实验 CSV、统计指标与对比图表
- `ENVIRONMENT.md`：硬件平台与环境诊断详细报告
- `DECISIONS.md`：设计与工程决策记录
- `REPORT.md`：阶段性与最终实验总结报告

## 快速复现流程

1. **环境准备 (P0)**：
   ```bash
   bash scripts/setup_env.sh
   ```
2. **基线测量 (P1)**：
   ```bash
   bash scripts/run_baseline.sh
   ```
3. **分层缓存与回放测试 (P2)**：
   ```bash
   pytest tests/
   python src/replay.py --config configs/default.yaml
   ```
4. **真实专家量化与误差分析 (P3)**：
   ```bash
   python src/export_real_experts.py
   ```
5. **动态调度器消融对比 (P4)**：
   ```bash
   python src/precision_scheduler.py
   ```
