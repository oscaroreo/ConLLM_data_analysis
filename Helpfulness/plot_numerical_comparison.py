import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def create_numerical_comparison_plot(csv_file, output_file=None):
    """
    创建数值化分析的点图，类似参考图的风格
    显示LLMnote和Community的平均helpfulness评分比较
    """
    
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 计算每种笔记类型的平均分和置信区间
    stats_by_type = df.groupby('note_type')['helpfulness_score'].agg(['mean', 'std', 'count']).reset_index()
    
    # 计算95%置信区间
    from scipy.stats import t
    confidence_intervals = {}
    
    for _, row in stats_by_type.iterrows():
        note_type = row['note_type']
        mean = row['mean']
        std = row['std']
        n = row['count']
        
        # 95%置信区间
        t_value = t.ppf(0.975, n-1)  # 双侧检验，α=0.05
        margin_error = t_value * std / np.sqrt(n)
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
        
        confidence_intervals[note_type] = {
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'margin_error': margin_error
        }
    
    print("数值化分析结果:")
    print(f"LLMnote: 均值={confidence_intervals['LLMnote']['mean']:.3f}, "
          f"95%CI=[{confidence_intervals['LLMnote']['ci_lower']:.3f}, {confidence_intervals['LLMnote']['ci_upper']:.3f}]")
    print(f"Community: 均值={confidence_intervals['Community']['mean']:.3f}, "
          f"95%CI=[{confidence_intervals['Community']['ci_lower']:.3f}, {confidence_intervals['Community']['ci_upper']:.3f}]")
    print(f"差值: {confidence_intervals['LLMnote']['mean'] - confidence_intervals['Community']['mean']:.3f}")
    
    # 进行配对t检验
    pivot_df = df.pivot_table(
        index=['user_id', 'tweet_id'],
        columns='note_type',
        values='helpfulness_score',
        aggfunc='first'
    ).reset_index()
    
    complete_pairs = pivot_df.dropna(subset=['Community', 'LLMnote'])
    community_scores = complete_pairs['Community'].values
    llm_scores = complete_pairs['LLMnote'].values
    
    # 配对t检验
    t_stat, p_value = stats.ttest_rel(llm_scores, community_scores)
    
    print(f"\n配对t检验结果:")
    print(f"t统计量: {t_stat:.4f}")
    print(f"p值: {p_value:.6f}")
    print(f"结论: {'显著差异' if p_value < 0.05 else '无显著差异'} (α = 0.05)")
    
    # 创建图表 - 调整为更窄的比例，模仿参考图
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']  # 避免中文字体问题
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # 设置y轴位置和标签
    y_positions = [1, 0]  # LLMnote在上，Community在下
    labels = ['LLMnote', 'Community']
    
    # 绘制点和置信区间
    colors = ['#2E86AB', '#A23B72']  # 蓝色和紫红色
    for i, (label, color) in enumerate(zip(labels, colors)):
        ci_data = confidence_intervals[label]
        mean_val = ci_data['mean']
        ci_lower = ci_data['ci_lower']
        ci_upper = ci_data['ci_upper']
        
        # 绘制置信区间横线 - 调整样式模仿参考图
        ax.plot([ci_lower, ci_upper], [y_positions[i], y_positions[i]], 
                color=color, linewidth=2, alpha=0.9, zorder=2)
        
        # 绘制置信区间端点 - 使用更小的端点
        ax.plot([ci_lower, ci_upper], [y_positions[i], y_positions[i]], 
                '|', color=color, markersize=6, markeredgewidth=2, zorder=2)
        
        # 绘制均值点 - 调整大小模仿参考图
        ax.scatter(mean_val, y_positions[i], s=80, color=color, zorder=3, 
                  edgecolors='white', linewidth=1)
        
        # 在点的左侧添加标签 - 调整位置适应新的x轴范围
        ax.text(0.33, y_positions[i], label, ha='right', va='center', 
               fontsize=11, fontweight='medium')
        
        # 在点的右侧显示具体数值 - 调整位置
        ax.text(mean_val + 0.02, y_positions[i], f'{mean_val:.3f}', 
               ha='left', va='center', fontsize=9, fontweight='medium')
    
    # 设置x轴 - 调整范围使图表更紧凑
    ax.set_xlim(0.35, 0.85)  # 聚焦在实际数据范围
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xticklabels(['0.4', '0.5', '0.6', '0.7', '0.8'])
    ax.set_xlabel('Helpfulness score', fontsize=11)
    
    # 在x轴上方添加语义标签
    ax.text(0.4, 1.4, 'Not helpful', ha='center', va='center', fontsize=9, alpha=0.7)
    ax.text(0.6, 1.4, 'Somewhat helpful', ha='center', va='center', fontsize=9, alpha=0.7)
    ax.text(0.8, 1.4, 'Helpful', ha='center', va='center', fontsize=9, alpha=0.7)
    
    # 设置y轴 - 调整范围使更紧凑
    ax.set_ylim(-0.3, 1.3)
    ax.set_yticks([])
    
    # 添加垂直网格线，更细
    for x in [0.4, 0.5, 0.6, 0.7, 0.8]:
        ax.axvline(x, color='gray', alpha=0.2, linewidth=0.5, zorder=0)
    
    # 移除所有边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 不添加标题，保持简洁
    
    # 在图表下方添加统计信息
    stats_text = f't = {t_stat:.3f}, p = {p_value:.3f}'
    ax.text(0.6, -0.2, stats_text, ha='center', va='center', 
           fontsize=9, style='italic', color='gray')
    
    # 置信区间数值已在上面的循环中添加，这里移除重复代码
    
    plt.tight_layout()
    
    # 保存图表
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n图表已保存到: {output_file}")
    
    plt.show()
    
    return {
        'llm_mean': confidence_intervals['LLMnote']['mean'],
        'community_mean': confidence_intervals['Community']['mean'],
        'difference': confidence_intervals['LLMnote']['mean'] - confidence_intervals['Community']['mean'],
        'llm_ci': [confidence_intervals['LLMnote']['ci_lower'], confidence_intervals['LLMnote']['ci_upper']],
        'community_ci': [confidence_intervals['Community']['ci_lower'], confidence_intervals['Community']['ci_upper']],
        't_statistic': t_stat,
        'p_value': p_value,
        'n_pairs': len(complete_pairs)
    }

def generate_numerical_analysis_summary(results):
    """
    生成数值化分析的文字总结
    """
    
    print("\n" + "="*50)
    print("数值化分析总结")
    print("="*50)
    
    print("\n方法详述:")
    print("- 将三点量表进行数值映射：")
    print("  • 'not helpful' → 0")
    print("  • 'somewhat helpful' → 0.5") 
    print("  • 'helpful' → 1")
    print("- 比较两种笔记类型的平均分")
    print("- 使用配对t检验检验显著性差异")
    
    print(f"\n主要发现:")
    print(f"- LLMnote: 均值={results['llm_mean']:.3f}, 95%CI=[{results['llm_ci'][0]:.3f}, {results['llm_ci'][1]:.3f}]")
    print(f"- Community: 均值={results['community_mean']:.3f}, 95%CI=[{results['community_ci'][0]:.3f}, {results['community_ci'][1]:.3f}]")
    print(f"- 差值: {results['difference']:.3f}")
    print(f"- 样本量: {results['n_pairs']}对")
    
    print(f"\n统计检验结果:")
    print(f"- t统计量: {results['t_statistic']:.4f}")
    print(f"- p值: {results['p_value']:.6f}")
    
    if results['p_value'] < 0.05:
        print("- 结论: 两种笔记类型存在显著差异")
        if results['difference'] > 0:
            print("- LLMnote的平均评分显著高于Community")
        else:
            print("- Community的平均评分显著高于LLMnote")
    else:
        print("- 结论: 两种笔记类型无显著差异")
        print("- 两种笔记类型在数值化评分上表现相当")

if __name__ == "__main__":
    # 设置文件路径
    input_file = "helpfulness_extracted.csv"
    output_file = "numerical_helpfulness_comparison.png"
    
    print("开始数值化分析...")
    
    # 创建图表
    results = create_numerical_comparison_plot(input_file, output_file)
    
    # 生成总结
    generate_numerical_analysis_summary(results)
    
    print("\n分析完成！")