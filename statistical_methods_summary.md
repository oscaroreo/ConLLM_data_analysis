# 统计方法汇总

## 1. 有用性分析 (Helpfulness Analysis)

### 1.1 Wilcoxon符号秩检验
- **文件**: `Helpfulness/wilcoxon_calculation.py`
- **方法**: `scipy.stats.wilcoxon()`
- **参数设置**:
  ```python
  stats.wilcoxon(community_scores, llm_scores, 
                 alternative='two-sided', 
                 zero_method='wilcox')
  ```
- **统计量计算**:
  - 使用配对数据（同一用户对同一推文的两种笔记评分）
  - 计算差值：`differences = llm_scores - community_scores`
  - 返回Wilcoxon统计量W和p值

- **额外检验**:
  - **单侧检验**: 分别测试 `alternative='greater'` 和 `alternative='less'`
  - **效应大小**: Rank-biserial correlation: `r = 1 - (2 * W) / (n * (n + 1))`
  - **正态性检验**: Shapiro-Wilk test 验证是否使用非参数检验
  - **配对t检验**: 作为对比，使用 `stats.ttest_rel()`
  - **符号检验**: 使用 `stats.binomtest()` 检验正负差值的比例

### 1.2 线性混合效应模型
- **文件**: `Helpfulness/linear_mixed_effects_analysis.py`
- **方法**: `statsmodels.formula.api.mixedlm`
- **模型公式**:
  ```python
  model = mixedlm("helpfulness_score ~ is_llmnote", df, groups=df["user_id"])
  fitted_model = model.fit(method='powell', maxiter=1000)
  ```
- **固定效应**: `is_llmnote` (LLMnote=1, Community=0)
- **随机效应**: `user_id` (参与者随机截距)
- **备选方法**: 当statsmodels不可用时，使用OLS回归

## 2. 胜率分析 (Winning Rate Analysis)

### 2.1 二项分布检验
- **文件**: `Winning_rate/winning_rate_analysis.py`
- **方法**: 
  ```python
  # 新版scipy
  from scipy.stats import binomtest
  result = binomtest(llm_wins, total_comparisons, p=0.5, alternative='two-sided')
  
  # 旧版scipy
  stats.binom_test(llm_wins, total_comparisons, p=0.5, alternative='two-sided')
  ```
- **原假设**: LLMnote被选择的概率 = 50%
- **备择假设**: LLMnote被选择的概率 ≠ 50%

### 2.2 Wilson置信区间
- **用途**: 计算胜率的95%置信区间
- **公式实现**:
  ```python
  p_hat = successes / total
  z = stats.norm.ppf((1 + confidence) / 2)
  denominator = 1 + z**2 / total
  center = (p_hat + z**2 / (2 * total)) / denominator
  margin = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denominator
  ```

### 2.3 混合效应逻辑回归（可选）
- **模型**: `mixedlm("llm_selected ~ 1", df, groups=df["participant_name"])`
- **因变量**: 二元变量（LLMnote是否被选中）
- **随机效应**: 参与者

## 3. 特征评分分析 (Feature Rating Analysis)

### 3.1 多维度Wilcoxon检验
- **文件**: `Note_feature_rating/feature_rating_visualization.py`
- **检验维度**: 
  - source_quality (来源质量)
  - clarity (清晰度)
  - coverage (覆盖范围)
  - context (上下文)
  - impartiality (中立性)

- **方法**:
  ```python
  statistic, p_value = wilcoxon(llm_scores, community_scores, alternative='two-sided')
  ```

- **效应大小计算**:
  ```python
  z_score = statistic / np.sqrt(n * (n + 1) * (2 * n + 1) / 6)
  effect_size = abs(z_score) / np.sqrt(n)
  ```

## 4. 统计显著性标准

所有分析中统一使用的显著性水平：
- `***` : p < 0.001 (极显著)
- `**` : p < 0.01 (非常显著)
- `*` : p < 0.05 (显著)
- `ns` : p ≥ 0.05 (不显著)

## 5. 主要发现

1. **有用性评分**: 
   - Wilcoxon检验 p = 0.591 (无显著差异)
   - 线性混合效应模型也支持无显著差异的结论

2. **胜率分析**: 
   - 总体胜率 51.1%，接近50%
   - 二项检验显示与50%无显著差异

3. **特征评分**: 
   - 覆盖范围: LLMnote显著更好 (p = 0.0003)
   - 中立性: Community Note显著更好 (p = 0.0004)
   - 其他维度无显著差异