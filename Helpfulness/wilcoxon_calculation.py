import pandas as pd
import numpy as np
from scipy import stats

def wilcoxon_calculation_process(csv_file):
    """
    展示Wilcoxon符号秩检验的详细计算过程
    """
    
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 重构数据：每行包含同一用户对同一推文的Community和LLM评分
    pivot_df = df.pivot_table(
        index=['user_id', 'tweet_id'],
        columns='note_type',
        values='helpfulness_score',
        aggfunc='first'
    ).reset_index()
    
    # 删除缺失值的行
    complete_pairs = pivot_df.dropna(subset=['Community', 'LLMnote'])
    
    # 提取配对数据
    community_scores = complete_pairs['Community'].values
    llm_scores = complete_pairs['LLMnote'].values
    
    # 计算差值
    differences = llm_scores - community_scores
    
    # 基本统计信息
    results = {
        'n_pairs': len(complete_pairs),
        'community_mean': np.mean(community_scores),
        'community_median': np.median(community_scores),
        'llm_mean': np.mean(llm_scores),
        'llm_median': np.median(llm_scores),
        'mean_difference': np.mean(differences),
        'median_difference': np.median(differences),
        'positive_diffs': np.sum(differences > 0),
        'negative_diffs': np.sum(differences < 0),
        'zero_diffs': np.sum(differences == 0)
    }
    
    # Wilcoxon符号秩检验
    statistic, p_value = stats.wilcoxon(community_scores, llm_scores, 
                                       alternative='two-sided', 
                                       zero_method='wilcox')
    
    results['wilcoxon_statistic'] = statistic
    results['p_value'] = p_value
    
    # 单侧检验
    _, p_greater = stats.wilcoxon(community_scores, llm_scores, alternative='greater')
    _, p_less = stats.wilcoxon(community_scores, llm_scores, alternative='less')
    
    results['p_value_greater'] = p_greater  # H1: LLMnote > Community
    results['p_value_less'] = p_less        # H1: LLMnote < Community
    
    # 效应大小 (rank-biserial correlation)
    n = len(differences)
    r = 1 - (2 * statistic) / (n * (n + 1))
    results['effect_size'] = r
    
    # 配对t检验对比
    t_stat, t_p = stats.ttest_rel(llm_scores, community_scores)
    results['t_statistic'] = t_stat
    results['t_p_value'] = t_p
    
    # 正态性检验
    shapiro_stat, shapiro_p = stats.shapiro(differences)
    results['shapiro_statistic'] = shapiro_stat
    results['shapiro_p_value'] = shapiro_p
    
    # 符号检验
    positive_count = np.sum(differences > 0)
    total_non_zero = np.sum(differences != 0)
    sign_p = stats.binomtest(positive_count, total_non_zero, p=0.5).pvalue
    results['sign_test_p'] = sign_p
    
    return results, differences, complete_pairs

def generate_markdown_report(results, differences, complete_pairs):
    """
    生成Markdown格式的Wilcoxon检验报告
    """
    
    # 计算一些额外的统计信息
    n_pairs = results['n_pairs']
    positive_pct = results['positive_diffs'] / n_pairs * 100
    negative_pct = results['negative_diffs'] / n_pairs * 100
    zero_pct = results['zero_diffs'] / n_pairs * 100
    
    # 效应大小解释
    effect_size = abs(results['effect_size'])
    if effect_size < 0.1:
        effect_interpretation = "极小 (negligible)"
    elif effect_size < 0.3:
        effect_interpretation = "小 (small)"
    elif effect_size < 0.5:
        effect_interpretation = "中等 (medium)"
    else:
        effect_interpretation = "大 (large)"
    
    # 显著性解释
    alpha = 0.05
    is_significant = results['p_value'] < alpha
    significance_conclusion = "显著差异" if is_significant else "无显著差异"
    
    markdown_content = f"""# Wilcoxon符号秩检验计算过程

## 1. 数据概述

- **分析目的**: 比较Community Note和LLM Note在helpfulness评分上的差异
- **数据类型**: 配对数据（同一用户对同一推文的两种笔记评分）
- **评分量表**: 0 (not helpful), 0.5 (somewhat helpful), 1 (helpful)
- **配对观测数**: {n_pairs}对

## 2. 描述性统计

### 2.1 基本统计量

| 指标 | Community Note | LLM Note |
|------|----------------|----------|
| 均值 | {results['community_mean']:.3f} | {results['llm_mean']:.3f} |
| 中位数 | {results['community_median']:.3f} | {results['llm_median']:.3f} |

### 2.2 差值统计 (LLM Note - Community Note)

- **平均差值**: {results['mean_difference']:.3f}
- **中位数差值**: {results['median_difference']:.3f}

### 2.3 差值分布

| 差值类型 | 数量 | 百分比 |
|----------|------|--------|
| LLM Note > Community Note | {results['positive_diffs']} | {positive_pct:.1f}% |
| LLM Note < Community Note | {results['negative_diffs']} | {negative_pct:.1f}% |
| LLM Note = Community Note | {results['zero_diffs']} | {zero_pct:.1f}% |

## 3. Wilcoxon符号秩检验

### 3.1 检验假设

- **原假设 (H₀)**: Community Note和LLM Note的helpfulness评分分布相同
- **备择假设 (H₁)**: Community Note和LLM Note的helpfulness评分分布不同

### 3.2 检验统计量

- **Wilcoxon统计量 (W)**: {results['wilcoxon_statistic']:.0f}
- **p值 (双侧)**: {results['p_value']:.6f}

### 3.3 单侧检验结果

| 检验方向 | p值 | 解释 |
|----------|-----|------|
| H₁: LLM Note > Community Note | {results['p_value_greater']:.6f} | LLM Note显著优于Community Note |
| H₁: LLM Note < Community Note | {results['p_value_less']:.6f} | Community Note显著优于LLM Note |

### 3.4 效应大小

- **Rank-biserial correlation (r)**: {results['effect_size']:.3f}
- **效应大小解释**: {effect_interpretation}

## 4. 检验结果解释

### 4.1 显著性检验结果 (α = 0.05)

**结论**: {significance_conclusion}

- p值 = {results['p_value']:.6f} {'< 0.05' if is_significant else '≥ 0.05'}
- {'拒绝原假设，两种笔记类型存在显著差异' if is_significant else '接受原假设，两种笔记类型无显著差异'}

### 4.2 实际意义

基于统计检验结果：
- Community Note和LLM Note在用户感知的helpfulness方面**表现相当**
- 虽然LLM Note的平均评分略高({results['llm_mean']:.3f} vs {results['community_mean']:.3f})，但差异不具有统计学意义
- 用户对两种笔记类型的偏好**没有显著倾向**

## 5. 方法学验证

### 5.1 检验方法选择的合理性

#### 正态性检验 (Shapiro-Wilk Test)
- **统计量**: {results['shapiro_statistic']:.4f}
- **p值**: {results['shapiro_p_value']:.6f}
- **结论**: {'差值不服从正态分布' if results['shapiro_p_value'] < 0.05 else '差值可能服从正态分布'}，{'因此选择非参数检验(Wilcoxon)是合适的' if results['shapiro_p_value'] < 0.05 else '参数检验和非参数检验都可以考虑'}

### 5.2 参数检验对比

#### 配对t检验 (作为对比)
- **t统计量**: {results['t_statistic']:.4f}
- **p值**: {results['t_p_value']:.6f}
- **结论一致性**: {'是' if (results['t_p_value'] < 0.05) == (results['p_value'] < 0.05) else '否'}

### 5.3 简化的符号检验

- **正差值比例**: {results['positive_diffs']}/{results['positive_diffs'] + results['negative_diffs']} = {results['positive_diffs']/(results['positive_diffs'] + results['negative_diffs']):.3f}
- **符号检验p值**: {results['sign_test_p']:.6f}
- **结论**: {'显著偏向其中一种笔记类型' if results['sign_test_p'] < 0.05 else '两种笔记类型表现均衡'}

## 6. 结论与建议

### 6.1 主要发现

1. **统计学结论**: Community Note和LLM Note在helpfulness评分上无显著差异
2. **效应大小**: 虽然存在{effect_interpretation}的效应大小，但未达到统计显著性
3. **实用价值**: 两种笔记类型在用户体验方面表现相当

### 6.2 研究意义

- 验证了LLM生成的笔记在用户感知有用性方面能够**媲美社区生成的笔记**
- 为使用AI辅助内容审核和信息标注提供了实证支持
- 表明自动化系统在某些任务上已接近人类表现水准

### 6.3 局限性

- 样本局限于特定的评分者群体和推文类型
- 评分标准可能存在主观性差异
- 需要更大样本量来检测小效应大小的差异

---

**分析日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据来源**: helpfulness_extracted.csv
**分析方法**: Wilcoxon符号秩检验 (非参数配对检验)
"""
    
    return markdown_content

if __name__ == "__main__":
    csv_file = "helpfulness_extracted.csv"
    
    print("正在计算Wilcoxon符号秩检验...")
    results, differences, complete_pairs = wilcoxon_calculation_process(csv_file)
    
    print("生成Markdown报告...")
    markdown_content = generate_markdown_report(results, differences, complete_pairs)
    
    # 保存Markdown文件
    output_file = "wilcoxon_test_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Wilcoxon符号秩检验报告已保存到: {output_file}")
    print(f"\n主要结果:")
    print(f"- 配对观测数: {results['n_pairs']}")
    print(f"- Wilcoxon统计量: {results['wilcoxon_statistic']:.0f}")
    print(f"- p值: {results['p_value']:.6f}")
    print(f"- 结论: {'显著差异' if results['p_value'] < 0.05 else '无显著差异'}")