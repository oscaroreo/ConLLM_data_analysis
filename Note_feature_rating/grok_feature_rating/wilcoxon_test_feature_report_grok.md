# 笔记特征评分Wilcoxon符号秩检验报告

## 统计方法
使用Wilcoxon符号秩检验比较LLMnote与Community Note在各维度的评分差异
评分量表：1-5分 (1=强烈反对, 3=中立, 5=强烈同意)

## 检验结果

| 评分维度 | 统计量(W) | p值 | 样本量 | 平均差异 | 效应大小 | 显著性 |
|---------|-----------|-----|--------|---------|----------|--------|
| 来源质量 | 22908 | 0.0000 | 720 | +0.486 | 0.076 | *** |
| 清晰度 | 30070 | 0.0000 | 720 | +0.272 | 0.100 | *** |
| 覆盖范围 | 27510 | 0.0000 | 720 | +0.672 | 0.092 | *** |
| 上下文 | 24149 | 0.0000 | 720 | +0.424 | 0.081 | *** |
| 中立性 | 41936 | 0.0001 | 720 | +0.204 | 0.140 | *** |

*注: *** p<0.001, ** p<0.01, * p<0.05, ns = 不显著*

## 结果解释

### 主要发现

在 5 个评价维度中，有 5 个维度存在显著差异：

- **中立性** (Note is NOT argumentative, speculative or biased):
  - LLMnote评分显著高于Community Note (平均差异: +0.204, p = 0.0001)
  - 效应大小为小 (r = 0.140)
- **清晰度** (Note is written in clear language):
  - LLMnote评分显著高于Community Note (平均差异: +0.272, p = 0.0000)
  - 效应大小为小 (r = 0.100)
- **覆盖范围** (Note addresses all key claims in the post):
  - LLMnote评分显著高于Community Note (平均差异: +0.672, p = 0.0000)
  - 效应大小为极小 (r = 0.092)
- **上下文** (Note provides important context):
  - LLMnote评分显著高于Community Note (平均差异: +0.424, p = 0.0000)
  - 效应大小为极小 (r = 0.081)
- **来源质量** (Sources on note are high-quality and relevant):
  - LLMnote评分显著高于Community Note (平均差异: +0.486, p = 0.0000)
  - 效应大小为极小 (r = 0.076)

### 总体评价
LLMnote在 5 个维度上表现更好，Community Note在 0 个维度上表现更好，
0 个维度无显著差异。总体而言，LLMnote在多数评价维度上获得了更高的用户评分。