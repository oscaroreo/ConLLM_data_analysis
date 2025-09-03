# 笔记特征评分Wilcoxon符号秩检验报告

## 统计方法
使用Wilcoxon符号秩检验比较LLMnote与Community Note在各维度的评分差异
评分量表：1-5分 (1=强烈反对, 3=中立, 5=强烈同意)

## 检验结果

| 评分维度 | 统计量(W) | p值 | 样本量 | 平均差异 | 效应大小 | 显著性 |
|---------|-----------|-----|--------|---------|----------|--------|
| 来源质量 | 38664 | 0.0000 | 720 | +0.267 | 0.129 | *** |
| 清晰度 | 30666 | 0.0000 | 720 | +0.208 | 0.102 | *** |
| 覆盖范围 | 37718 | 0.0000 | 720 | +0.360 | 0.126 | *** |
| 上下文 | 50614 | 0.0004 | 720 | +0.204 | 0.169 | *** |
| 中立性 | 43386 | 0.0095 | 720 | +0.149 | 0.145 | ** |

*注: *** p<0.001, ** p<0.01, * p<0.05, ns = 不显著*

## 结果解释

### 主要发现

在 5 个评价维度中，有 5 个维度存在显著差异：

- **上下文** (Note provides important context):
  - LLMnote评分显著高于Community Note (平均差异: +0.204, p = 0.0004)
  - 效应大小为小 (r = 0.169)
- **中立性** (Note is NOT argumentative, speculative or biased):
  - LLMnote评分显著高于Community Note (平均差异: +0.149, p = 0.0095)
  - 效应大小为小 (r = 0.145)
- **来源质量** (Sources on note are high-quality and relevant):
  - LLMnote评分显著高于Community Note (平均差异: +0.267, p = 0.0000)
  - 效应大小为小 (r = 0.129)
- **覆盖范围** (Note addresses all key claims in the post):
  - LLMnote评分显著高于Community Note (平均差异: +0.360, p = 0.0000)
  - 效应大小为小 (r = 0.126)
- **清晰度** (Note is written in clear language):
  - LLMnote评分显著高于Community Note (平均差异: +0.208, p = 0.0000)
  - 效应大小为小 (r = 0.102)

### 总体评价
LLMnote在 5 个维度上表现更好，Community Note在 0 个维度上表现更好，
0 个维度无显著差异。总体而言，LLMnote在多数评价维度上获得了更高的用户评分。