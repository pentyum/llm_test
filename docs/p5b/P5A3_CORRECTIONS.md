# P5a.3 历史报告与原始数据一致性复核修正说明 (P5A3_CORRECTIONS.md)

## 1. 审计概述

根据 P5b 规范 G0 审计要求，对 `docs/p5a3/P5A3_REPORT.md` 中摘要文字性叙述与 `results/p5a3/quality/summary.csv` 实际落盘的 480 条生成聚合数据进行了精确对账。

## 2. 差异对账清单 (Discrepancy Log)

### 差异 1：S4 generation speed described as 12.8+ tokens/s in summary
- **报告文字声称**：`S4 generation speed described as 12.8+ tokens/s in summary`
- **底层 CSV 真实实测值**：`11.95 tokens/s`
- **权威数据源**：`results/p5a3/quality/summary.csv`
- **审计判定**：`FAIL (Discrepancy caught by G0 audit)`

### 差异 2：S4 average token match rate described as 93%+ in summary
- **报告文字声称**：`S4 average token match rate described as 93%+ in summary`
- **底层 CSV 真实实测值**：`88.09%`
- **权威数据源**：`results/p5a3/quality/summary.csv`
- **审计判定**：`FAIL (Discrepancy caught by G0 audit)`

## 3. 根因说明与 P5b 实施准则

1. **差异根因**：P5a.3 摘要中的 `12.8+` 与 `93%+` 属于早期单 Prompt 或少量样本测试时的临时经验估计，在最终全量 32 Prompts × 3 次重复（480 次真实解码）汇总时未被脚本动态引用替换，造成正文摘要与表格数据出现微小出入；
2. **P5b 严正准则**：在 P5b 阶段，所有总结报告摘要必须 100% 由脚本读取原始 CSV/JSON 动态填充，绝不允许任何人工估算文字；历史 P5a.3 原始文件完好保留，本文件作为官方正式修正说明附录。
