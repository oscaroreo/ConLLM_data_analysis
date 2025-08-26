import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower, TTestPower

def cohen_d_paired(d1, d2):
    """Calculates Cohen's d for paired samples."""
    diff = d1 - d2
    return np.mean(diff) / np.std(diff, ddof=1)

def cohen_d_independent(d1, d2):
    """Calculates Cohen's d for independent samples."""
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / s_pooled

def get_pivoted_helpfulness_data():
    """Loads and pivots the helpfulness data from long to wide format."""
    df = pd.read_csv('Helpfulness/helpfulness_extracted.csv')
    # Pivot the table to get 'Community' and 'LLMnote' scores in columns for each user/tweet
    pivoted_df = df.pivot_table(index=['user_id', 'tweet_id'], columns='note_type', values='helpfulness_score').reset_index()
    # Drop rows with missing values for either note type to ensure pairing
    pivoted_df.dropna(subset=['Community', 'LLMnote'], inplace=True)
    return pivoted_df

def analyze_helpfulness_comparison():
    """Analyzes the paired helpfulness data using Wilcoxon signed-rank test."""
    try:
        pivoted_df = get_pivoted_helpfulness_data()
        data1 = pivoted_df['LLMnote']
        data2 = pivoted_df['Community']
        n = len(pivoted_df)
        
        wilcoxon_result = stats.wilcoxon(data1, data2)
        cohen_d = cohen_d_paired(data1, data2)
        
        power_analysis = TTestPower()
        achieved_power = power_analysis.power(effect_size=cohen_d, nobs=n, alpha=0.05, alternative='two-sided')
        required_n = power_analysis.solve_power(effect_size=cohen_d, power=0.8, alpha=0.05, alternative='two-sided')
        
        return {
            "analysis_name": "Helpfulness Comparison (Paired)",
            "test_type": "Wilcoxon Signed-Rank Test",
            "data_file": "Helpfulness/helpfulness_extracted.csv",
            "columns": "LLMnote vs Community",
            "n_samples": n,
            "statistic": wilcoxon_result.statistic,
            "p_value": wilcoxon_result.pvalue,
            "effect_size_name": "Cohen's d (approximated for non-parametric test)",
            "effect_size_value": cohen_d,
            "formula": "d = mean(D) / std(D), where D is the array of paired differences",
            "achieved_power": achieved_power,
            "required_n_80_power": int(np.ceil(required_n))
        }
    except Exception as e:
        return {"error": f"Helpfulness Comparison: {e}"}

def analyze_numerical_helpfulness():
    """Analyzes numerical helpfulness data using an independent samples t-test."""
    try:
        pivoted_df = get_pivoted_helpfulness_data()
        data1 = pivoted_df['LLMnote']
        data2 = pivoted_df['Community']
        n1, n2 = len(data1), len(data2)
        
        ttest_result = stats.ttest_ind(data1, data2)
        cohen_d = cohen_d_independent(data1, data2)
        
        power_analysis = TTestIndPower()
        achieved_power = power_analysis.power(effect_size=cohen_d, nobs1=n1, alpha=0.05, ratio=n2/n1, alternative='two-sided')
        required_n = power_analysis.solve_power(effect_size=cohen_d, power=0.8, alpha=0.05, ratio=1.0, alternative='two-sided')
        
        return {
            "analysis_name": "Numerical Helpfulness (Independent)",
            "test_type": "Independent Samples T-test",
            "data_file": "Helpfulness/helpfulness_extracted.csv",
            "columns": "LLMnote vs Community",
            "n_samples": f"n1={n1}, n2={n2}",
            "statistic": ttest_result.statistic,
            "p_value": ttest_result.pvalue,
            "effect_size_name": "Cohen's d",
            "effect_size_value": cohen_d,
            "formula": "d = (mean1 - mean2) / pooled_std_dev",
            "achieved_power": achieved_power,
            "required_n_80_power": f"~{int(np.ceil(required_n))} per group"
        }
    except Exception as e:
        return {"error": f"Numerical Helpfulness: {e}"}

def analyze_winning_rate():
    """Analyzes winning rate data using a paired samples t-test."""
    try:
        df = pd.read_csv('Winning_rate/winning_rate_data_by_participant.csv')
        col1, col2 = 'llm_rate', 'community_rate'

        # Force columns to be numeric, coercing errors
        data1 = pd.to_numeric(df[col1], errors='coerce').dropna()
        data2 = pd.to_numeric(df[col2], errors='coerce').dropna()
        
        n = len(df)
        
        ttest_result = stats.ttest_rel(data1, data2)
        cohen_d = cohen_d_paired(data1, data2)
        
        power_analysis = TTestPower()
        achieved_power = power_analysis.power(effect_size=cohen_d, nobs=n, alpha=0.05, alternative='two-sided')
        required_n = power_analysis.solve_power(effect_size=cohen_d, power=0.8, alpha=0.05, alternative='two-sided')
        
        return {
            "analysis_name": "Winning Rate Comparison (Paired)",
            "test_type": "Paired Samples T-test",
            "data_file": "Winning_rate/winning_rate_data_by_participant.csv",
            "columns": f"{col1} vs {col2}",
            "n_samples": n,
            "statistic": ttest_result.statistic,
            "p_value": ttest_result.pvalue,
            "effect_size_name": "Cohen's d",
            "effect_size_value": cohen_d,
            "formula": "d = mean(D) / std(D), where D is the array of paired differences",
            "achieved_power": achieved_power,
            "required_n_80_power": int(np.ceil(required_n))
        }
    except Exception as e:
        return {"error": f"Winning Rate: {e}"}

def get_interpretation(effect_size_name, value):
    value = abs(value)
    if "cohen" in effect_size_name.lower():
        if value >= 0.8: return "Large"
        if value >= 0.5: return "Medium"
        if value >= 0.2: return "Small"
        return "Very Small"
    return "N/A"

def format_results_to_md(results):
    md_string = """# 效应量与功效分析报告

本文档详细介绍了对三个关键指标的效应量（Effect Size）和统计功效（Power Analysis）的计算结果。

---
"""
    
    for result in results:
        if "error" in result:
            md_string += f"""## 分析出错

在分析 `{result['error']}` 时遇到问题。请检查数据文件和列名。

---
"""
            continue

        md_string += f"## {result['analysis_name']}\n\n"
        md_string += f"**数据文件**: `{result['data_file']}`\n"
        md_string += f"**比较列**: `{result['columns']}`\n\n"
        
        md_string += f"### 1. 统计检验结果\n"
        md_string += f"- **检验类型**: {result['test_type']}\n"
        md_string += f"- **样本量 (N)**: {result['n_samples']}\n"
        md_string += f"- **检验统计量**: `{result['statistic']:.4f}`\n"
        md_string += f"- **P值**: `{result['p_value']:.4f}`\n\n"
        
        md_string += f"### 2. 效应量 (Effect Size)\n"
        interpretation = get_interpretation(result['effect_size_name'], result['effect_size_value'])
        md_string += f"- **效应量类型**: {result['effect_size_name']}\n"
        md_string += f"- **计算值**: `{result['effect_size_value']:.4f}`\n"
        if result.get('formula'):
            md_string += f"- **计算公式**: `{result['formula']}`\n"
        md_string += f"- **效应量大小解释**: **{interpretation}**\n\n"
        
        md_string += f"### 3. 功效分析 (Power Analysis)\n"
        md_string += f"#### 后验功效 (Achieved Power)\n"
        md_string += f"在当前样本量和计算出的效应量下，本研究检测到真实效应的概率（功效）为 **{result['achieved_power']:.2%}**。\n\n"
        
        md_string += f"#### 前瞻性分析 (Required Sample Size)\n"
        md_string += f"为了在未来的研究中达到理想的80%统计功效，需要的样本量大约为 **{result['required_n_80_power']}**。\n\n"
        md_string += "---\n"
        
    return md_string

if __name__ == '__main__':
    all_results = [
        analyze_helpfulness_comparison(),
        analyze_numerical_helpfulness(),
        analyze_winning_rate()
    ]
    
    md_content = format_results_to_md(all_results)
    
    with open("power_and_effect_size_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print("Analysis complete. Report generated at power_and_effect_size_report.md")
