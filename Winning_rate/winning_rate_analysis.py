import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def calculate_confidence_interval(successes, total, confidence=0.95):
    """
    计算二项分布的Wilson置信区间
    """
    if total == 0:
        return 0, 0, 0
    
    p_hat = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Wilson置信区间公式
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denominator
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return p_hat, lower, upper

def analyze_winning_rates(input_csv):
    """
    分析LLMnote的获胜率，包括总体和两种笔记都被评为helpful时的情况
    """
    
    # 读取数据
    df = pd.read_csv(input_csv)
    
    print("=== LLMnote 胜率分析 ===")
    print(f"总比较次数: {len(df)}")
    
    # 1. 总体胜率分析
    total_comparisons = len(df)
    llm_wins = len(df[df['winner'] == 'LLMnote'])
    
    overall_rate, overall_lower, overall_upper = calculate_confidence_interval(
        llm_wins, total_comparisons
    )
    
    print(f"\n总体胜率:")
    print(f"- LLMnote被选中的次数: {llm_wins}/{total_comparisons}")
    print(f"- 胜率: {overall_rate:.1%} (95% CI: [{overall_lower:.1%}, {overall_upper:.1%}])")
    
    # 2. 当两条笔记都被评为"helpful"时的胜率
    both_helpful = df[(df['community_helpfulness_raw'] == 'helpful') & 
                     (df['llm_helpfulness_raw'] == 'helpful')]
    
    both_helpful_total = len(both_helpful)
    both_helpful_llm_wins = len(both_helpful[both_helpful['winner'] == 'LLMnote'])
    
    if both_helpful_total > 0:
        both_helpful_rate, both_helpful_lower, both_helpful_upper = calculate_confidence_interval(
            both_helpful_llm_wins, both_helpful_total
        )
        
        print(f"\n当两条笔记都被评为'helpful'时:")
        print(f"- 总比较次数: {both_helpful_total}")
        print(f"- LLMnote被选中的次数: {both_helpful_llm_wins}")
        print(f"- 胜率: {both_helpful_rate:.1%} (95% CI: [{both_helpful_lower:.1%}, {both_helpful_upper:.1%}])")
    else:
        both_helpful_rate = both_helpful_lower = both_helpful_upper = 0
        print(f"\n当两条笔记都被评为'helpful'时: 无数据")
    
    # 3. 统计显著性检验
    # 使用二项分布检验是否显著大于50%
    try:
        # 新版本scipy
        from scipy.stats import binomtest
        result = binomtest(llm_wins, total_comparisons, p=0.5, alternative='two-sided')
        p_value = result.pvalue
    except (ImportError, AttributeError):
        # 旧版本scipy
        p_value = stats.binom_test(llm_wins, total_comparisons, p=0.5, alternative='two-sided')
    
    print(f"\n统计显著性检验:")
    print(f"- 原假设: LLMnote被选择的概率 = 50%")
    print(f"- p值: {p_value:.4f}")
    print(f"- 结论: {'显著不同于50%' if p_value < 0.05 else '与50%无显著差异'} (α = 0.05)")
    
    # 返回结果用于绘图
    results = {
        'overall': {
            'rate': overall_rate,
            'ci_lower': overall_lower,
            'ci_upper': overall_upper,
            'n': total_comparisons,
            'wins': llm_wins
        },
        'both_helpful': {
            'rate': both_helpful_rate,
            'ci_lower': both_helpful_lower,
            'ci_upper': both_helpful_upper,
            'n': both_helpful_total,
            'wins': both_helpful_llm_wins if both_helpful_total > 0 else 0
        }
    }
    
    return results

def create_winning_rate_plot(results, output_file=None):
    """
    创建获胜率可视化图表，类似参考图的风格
    """
    
    # 设置图表样式 - 修复字体问题
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    fig, ax = plt.subplots(figsize=(10, 3.5))  # 增加宽度避免文字重叠
    
    # 准备数据
    categories = ['Overall', 'Both rated helpful']
    rates = [results['overall']['rate'], results['both_helpful']['rate']]
    ci_lowers = [results['overall']['ci_lower'], results['both_helpful']['ci_lower']]
    ci_uppers = [results['overall']['ci_upper'], results['both_helpful']['ci_upper']]
    
    # y轴位置
    y_positions = [1, 0]
    
    # 绘制背景网格
    for x in [0, 0.25, 0.5, 0.75, 1.0]:
        ax.axvline(x, color='lightgray', alpha=0.5, linewidth=0.5, zorder=0)
    
    # 绘制50%参考线（更显眼）
    ax.axvline(0.5, color='gray', alpha=0.8, linewidth=1, zorder=1)
    
    # 绘制每个类别
    for i, (cat, rate, lower, upper) in enumerate(zip(categories, rates, ci_lowers, ci_uppers)):
        # 绘制置信区间
        ax.plot([lower, upper], [y_positions[i], y_positions[i]], 
                color='darkgray', linewidth=2, solid_capstyle='round', zorder=2)
        
        # 绘制率值点
        ax.scatter(rate, y_positions[i], s=100, color='black', zorder=3)
        
        # 添加类别标签（左侧）- 增加间距
        ax.text(-0.15, y_positions[i], cat, ha='right', va='center', 
                fontsize=11, fontweight='normal')
        
        # 添加百分比标签（右侧）- 增加间距
        percentage_text = f"{rate:.1%}"
        n_text = f"(n={results['overall']['n'] if i == 0 else results['both_helpful']['n']})"
        
        ax.text(1.05, y_positions[i], percentage_text, ha='left', va='center', 
                fontsize=11, fontweight='normal')
        
        # 添加样本量（在同一行的百分比后面，而不是下方）
        if results['both_helpful']['n'] > 0 or i == 0:
            ax.text(1.15, y_positions[i], n_text, ha='left', va='center', 
                    fontsize=9, color='gray', style='italic')
    
    # 设置x轴
    ax.set_xlim(-0.2, 1.25)  # 增加左右边距，避免文字被截断
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(['0%', '50%\n(Tie)', '100%'], fontsize=10)
    
    # 添加x轴标签 - 调整位置避免重叠
    ax.text(0.5, -0.9, '%(LLMnote rated more helpful than Community Note)', 
            ha='center', va='center', fontsize=10, style='italic')
    
    # 设置y轴
    ax.set_ylim(-1.0, 1.8)  # 增加上下边距
    ax.set_yticks([])
    
    # 移除边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 添加标题 - 使用英文避免中文字体问题
    ax.text(0.5, 1.6, 'LLMnote vs Community Note Winning Rate Analysis', 
            ha='center', va='center', fontsize=13, fontweight='bold')
    
    plt.tight_layout(pad=1.5)  # 增加padding避免内容被截断
    
    # 保存图表
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"\n图表已保存到: {output_file}")
    
    # plt.show()  # 注释掉以避免交互式显示

def perform_mixed_effects_analysis(df):
    """
    执行混合效应模型分析
    """
    
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import mixedlm
        
        print("\n=== 混合效应模型分析 ===")
        
        # 创建二元因变量：LLMnote是否被选中
        df['llm_selected'] = (df['winner'] == 'LLMnote').astype(int)
        
        # 使用逻辑混合效应模型
        # 因变量: llm_selected (0/1)
        # 随机效应: participant_name
        
        # 拟合只有截距的模型
        model = mixedlm("llm_selected ~ 1", df, groups=df["participant_name"])
        fitted_model = model.fit(method='powell')
        
        print("\n模型结果:")
        print(f"截距系数 (β₀): {fitted_model.fe_params['Intercept']:.3f}")
        print(f"p值: {fitted_model.pvalues['Intercept']:.4f}")
        
        # 转换为概率
        intercept = fitted_model.fe_params['Intercept']
        probability = 1 / (1 + np.exp(-intercept))  # logistic转换
        
        print(f"\n解释:")
        print(f"- LLMnote被选择的平均概率: {probability:.1%}")
        print(f"- 统计显著性: {'显著高于50%' if fitted_model.pvalues['Intercept'] < 0.05 else '与50%无显著差异'}")
        
        return fitted_model
        
    except ImportError:
        print("\n注意: 未安装statsmodels，跳过混合效应模型分析")
        return None

def generate_markdown_report(results, output_file):
    """
    生成Markdown格式的分析报告
    """
    
    markdown_content = f"""# LLMnote vs Community Note 帮助性胜率分析

## 1. 方法详述

该方法直接分析在强制二选一的任务中，LLMnote被选为"更有帮助"的频率。

**对应数据**: 用户在"选择更有帮助的笔记"任务中的二元选择结果。

## 2. 统计计算方法

### 2.1 胜率计算
使用Wilson置信区间方法计算LLMnote被选中的百分比及其95%置信区间。

### 2.2 分层分析
- **总体胜率**: 所有比较的整体结果
- **双方都被评为"helpful"时的胜率**: 控制质量因素的影响

## 3. 分析结果

### 3.1 总体胜率
- **LLMnote被选中的比例**: {results['overall']['rate']:.1%} (95% CI: [{results['overall']['ci_lower']:.1%}, {results['overall']['ci_upper']:.1%}])
- **样本量**: {results['overall']['n']}次比较
- **LLMnote获胜次数**: {results['overall']['wins']}次

### 3.2 当两条笔记都被评为"helpful"时
"""
    
    if results['both_helpful']['n'] > 0:
        markdown_content += f"""- **LLMnote被选中的比例**: {results['both_helpful']['rate']:.1%} (95% CI: [{results['both_helpful']['ci_lower']:.1%}, {results['both_helpful']['ci_upper']:.1%}])
- **样本量**: {results['both_helpful']['n']}次比较
- **LLMnote获胜次数**: {results['both_helpful']['wins']}次
"""
    else:
        markdown_content += "- **无数据**: 没有两条笔记都被评为'helpful'的情况\n"
    
    markdown_content += f"""
## 4. 统计推断

### 4.1 假设检验
- **原假设 (H₀)**: LLMnote被选择的概率 = 50%
- **备择假设 (H₁)**: LLMnote被选择的概率 ≠ 50%

### 4.2 结果解释
"""
    
    if results['overall']['ci_lower'] > 0.5:
        interpretation = "LLMnote的胜率显著高于50%，表明用户更倾向于选择LLMnote"
    elif results['overall']['ci_upper'] < 0.5:
        interpretation = "LLMnote的胜率显著低于50%，表明用户更倾向于选择Community Note"
    else:
        interpretation = "LLMnote的胜率与50%无显著差异，表明两种笔记类型表现相当"
    
    markdown_content += f"""- {interpretation}
- 置信区间{'不' if results['overall']['ci_lower'] <= 0.5 <= results['overall']['ci_upper'] else ''}包含50%，{'不' if results['overall']['ci_lower'] <= 0.5 <= results['overall']['ci_upper'] else ''}支持存在偏好的结论

## 5. 结论

基于胜率分析：
1. LLMnote在二选一任务中的整体表现为{results['overall']['rate']:.1%}
2. {'这一结果表明LLMnote和Community Note在用户感知的帮助性方面表现相当' if 0.45 <= results['overall']['rate'] <= 0.55 else 'LLMnote表现' + ('优于' if results['overall']['rate'] > 0.55 else '劣于') + 'Community Note'}
3. {'当控制笔记质量（都被评为helpful）时，差异' + ('依然存在' if results['both_helpful']['n'] > 0 and not (0.45 <= results['both_helpful']['rate'] <= 0.55) else '不明显') if results['both_helpful']['n'] > 0 else '需要更多高质量笔记的比较数据'}

---

**分析日期**: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M')}  
**数据来源**: winning_rate_data.csv  
**分析方法**: 二项分布置信区间 (Wilson方法)
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"\n分析报告已保存到: {output_file}")

if __name__ == "__main__":
    # 文件路径
    input_file = "winning_rate_data.csv"
    plot_output = "winning_rate_analysis_plot.png"
    report_output = "winning_rate_analysis_report.md"
    
    print("开始胜率分析...")
    
    # 执行分析
    results = analyze_winning_rates(input_file)
    
    # 创建可视化
    create_winning_rate_plot(results, plot_output)
    
    # 尝试混合效应模型
    df = pd.read_csv(input_file)
    perform_mixed_effects_analysis(df)
    
    # 生成报告
    generate_markdown_report(results, report_output)
    
    print("\n胜率分析完成！")