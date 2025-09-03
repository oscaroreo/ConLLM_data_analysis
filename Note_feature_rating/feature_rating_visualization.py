import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats
import seaborn as sns

def prepare_diverging_data(feature_df):
    """
    准备分歧条形图的数据
    将1-5分制转换为百分比分布
    """
    
    # 五个维度
    dimensions = {
        'source_quality': 'Sources on note are high-quality and relevant',
        'clarity': 'Note is written in clear language',  
        'coverage': 'Note addresses all key claims in the post',
        'context': 'Note provides important context',
        'impartiality': 'Note is NOT argumentative, speculative or biased'
    }
    
    # 评分映射
    rating_labels = {
        1: 'Strongly disagree',
        2: 'Disagree',
        3: 'Neutral',
        4: 'Agree',
        5: 'Strongly agree'
    }
    
    results = []
    
    for feature, description in dimensions.items():
        # 分别计算两种笔记类型的分布
        for note_type in ['LLMnote', 'Community']:
            data = feature_df[feature_df['note_type'] == note_type][feature].dropna()
            
            # 计算每个评分的数量
            counts = data.value_counts()
            total = len(data)
            
            # 计算百分比
            percentages = {}
            for rating in range(1, 6):
                count = counts.get(rating, 0)
                percentages[rating] = (count / total) * 100
            
            results.append({
                'dimension': feature,
                'description': description,
                'note_type': note_type,
                'strongly_disagree': percentages[1],
                'disagree': percentages[2],
                'neutral': percentages[3],
                'agree': percentages[4],
                'strongly_agree': percentages[5]
            })
    
    return pd.DataFrame(results)

def perform_wilcoxon_tests(feature_df):
    """
    对每个维度进行Wilcoxon signed-rank test
    """
    
    dimensions = ['source_quality', 'clarity', 'coverage', 'context', 'impartiality']
    test_results = {}
    
    for dimension in dimensions:
        # 获取配对数据
        pivot_data = feature_df.pivot_table(
            index=['participant_name', 'post_index'],
            columns='note_type',
            values=dimension
        ).dropna()
        
        community_scores = pivot_data['Community'].values
        llm_scores = pivot_data['LLMnote'].values
        
        # Wilcoxon signed-rank test
        try:
            from scipy.stats import wilcoxon
            statistic, p_value = wilcoxon(llm_scores, community_scores, alternative='two-sided')
            
            # 计算效应大小
            n = len(llm_scores)
            z_score = statistic / np.sqrt(n * (n + 1) * (2 * n + 1) / 6)
            effect_size = abs(z_score) / np.sqrt(n)
            
            test_results[dimension] = {
                'statistic': statistic,
                'p_value': p_value,
                'n_pairs': n,
                'effect_size': effect_size,
                'mean_diff': np.mean(llm_scores - community_scores)
            }
            
        except Exception as e:
            print(f"Error in Wilcoxon test for {dimension}: {e}")
            test_results[dimension] = None
    
    return test_results

def create_diverging_bar_chart(diverging_data, test_results, output_file=None):
    """
    创建分歧条形图
    """
    
    # 设置图表样式
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['font.size'] = 18
    
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # 准备数据
    dimensions = ['source_quality', 'clarity', 'coverage', 'context', 'impartiality']
    dimension_labels = {
        'source_quality': 'Sources on note are high-quality and relevant',
        'clarity': 'Note is written in clear language',
        'coverage': 'Note addresses all key claims in the post',
        'context': 'Note provides important context',
        'impartiality': 'Note is NOT argumentative, speculative or biased'
    }
    
    # 颜色设置
    colors = {
        'strongly_disagree': '#8B0000',  # 深红
        'disagree': '#DC143C',           # 红
        'neutral': '#D3D3D3',            # 灰
        'agree': '#4682B4',              # 蓝
        'strongly_agree': '#191970'       # 深蓝
    }
    
    y_positions = []
    y_labels = []
    
    # 为每个维度创建两行（LLMnote和Community）
    for i, dimension in enumerate(dimensions):
        # 添加维度标题
        y_positions.extend([i * 4 + 1, i * 4])
        y_labels.extend(['LLM note', 'Community Note'])
    
    # 绘制条形图
    bar_height = 0.35
    
    for i, dimension in enumerate(dimensions):
        llm_data = diverging_data[(diverging_data['dimension'] == dimension) & 
                                 (diverging_data['note_type'] == 'LLMnote')].iloc[0]
        community_data = diverging_data[(diverging_data['dimension'] == dimension) & 
                                       (diverging_data['note_type'] == 'Community')].iloc[0]
        
        # 计算累积位置 - 确保neutral完全居中于0%
        for j, (data, y_pos) in enumerate([(llm_data, i * 4 + 1), (community_data, i * 4)]):
            # 中立区域以0为中心对称分布
            neutral_half = data['neutral'] / 2
            
            # 负面评价（左侧）- 从neutral左边界继续向左
            disagree_end = -neutral_half
            disagree_start = disagree_end - data['disagree']
            strongly_disagree_start = disagree_start - data['strongly_disagree']
            
            # 正面评价（右侧）- 从neutral右边界继续向右
            agree_start = neutral_half
            agree_end = agree_start + data['agree']
            strongly_agree_end = agree_end + data['strongly_agree']
            
            # 绘制条形 - 确保neutral居中
            # 强烈反对（最左侧）
            ax.barh(y_pos, data['strongly_disagree'], left=strongly_disagree_start, height=bar_height,
                   color=colors['strongly_disagree'], label='Strongly disagree' if i == 0 and j == 0 else "")
            # 反对
            ax.barh(y_pos, data['disagree'], left=disagree_start, height=bar_height,
                   color=colors['disagree'], label='Disagree' if i == 0 and j == 0 else "")
            # 中立（完全以0为中心）
            ax.barh(y_pos, data['neutral'], left=-neutral_half, height=bar_height,
                   color=colors['neutral'], label='Neutral' if i == 0 and j == 0 else "")
            # 同意（从neutral右边界开始）
            ax.barh(y_pos, data['agree'], left=agree_start, height=bar_height,
                   color=colors['agree'], label='Agree' if i == 0 and j == 0 else "")
            # 强烈同意（最右侧）
            ax.barh(y_pos, data['strongly_agree'], left=agree_end, height=bar_height,
                   color=colors['strongly_agree'], label='Strongly agree' if i == 0 and j == 0 else "")
        
        # 添加维度标题（放在每组比较的上方）
        if test_results.get(dimension):
            p_value = test_results[dimension]['p_value']
            sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
            dimension_text = dimension_labels[dimension]
            if sig_marker:
                dimension_text += f' {sig_marker}'
        else:
            dimension_text = dimension_labels[dimension]
        
        # 添加维度描述（在每组上方作为小标题）
        ax.text(0, i * 4 + 2.2, dimension_text, ha='center', va='center', 
                fontsize=24, fontweight='normal')
    
    # 设置轴
    ax.set_xlim(-100, 100)
    ax.set_ylim(-1, len(dimensions) * 4 + 0.5)  # 增加上边距给标题留空间
    ax.set_xlabel('% of responses', fontsize=20)
    
    # 设置y轴标签
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=20)
    
    # 添加中心线（强调0%位置）
    ax.axvline(0, color='black', linewidth=1, alpha=0.8)
    
    # 设置x轴刻度
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(['100', '50', '0', '50', '100'])
    
    # 添加网格
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 移除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    
    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
              ncol=5, frameon=False, fontsize=18)
    
    # 添加标题 - 注释掉
    # plt.title('Feature Rating Comparison: LLMnote vs Community Note', 
    #           fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n可视化图表已保存到: {output_file}")
    
    return fig

def generate_wilcoxon_report(test_results, output_file):
    """
    生成Wilcoxon检验报告
    """
    
    dimension_names = {
        'source_quality': '来源质量',
        'clarity': '清晰度',
        'coverage': '覆盖范围',
        'context': '上下文',
        'impartiality': '中立性'
    }
    
    dimension_descriptions = {
        'source_quality': 'Sources on note are high-quality and relevant',
        'clarity': 'Note is written in clear language',
        'coverage': 'Note addresses all key claims in the post',
        'context': 'Note provides important context',
        'impartiality': 'Note is NOT argumentative, speculative or biased'
    }
    
    report = []
    report.append("# 笔记特征评分Wilcoxon符号秩检验报告\n")
    report.append("## 统计方法")
    report.append("使用Wilcoxon符号秩检验比较LLMnote与Community Note在各维度的评分差异")
    report.append("评分量表：1-5分 (1=强烈反对, 3=中立, 5=强烈同意)\n")
    
    report.append("## 检验结果\n")
    report.append("| 评分维度 | 统计量(W) | p值 | 样本量 | 平均差异 | 效应大小 | 显著性 |")
    report.append("|---------|-----------|-----|--------|---------|----------|--------|")
    
    for dimension, result in test_results.items():
        if result:
            cn_name = dimension_names[dimension]
            sig = '***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'ns'
            effect_size = result['effect_size']
            report.append(f"| {cn_name} | {result['statistic']:.0f} | {result['p_value']:.4f} | {result['n_pairs']} | {result['mean_diff']:+.3f} | {effect_size:.3f} | {sig} |")
    
    report.append("\n*注: *** p<0.001, ** p<0.01, * p<0.05, ns = 不显著*")
    
    report.append("\n## 结果解释")
    
    # 分析整体结果
    significant_dims = [(d, r) for d, r in test_results.items() if r and r['p_value'] < 0.05]
    non_significant_dims = [(d, r) for d, r in test_results.items() if r and r['p_value'] >= 0.05]
    
    if len(significant_dims) == 0:
        report.append("\n### 主要发现")
        report.append("在所有五个评价维度上，LLMnote与Community Note的评分均无显著差异，表明两种笔记类型在用户评价中表现相当。")
    else:
        report.append("\n### 主要发现")
        report.append(f"\n在 {len(test_results)} 个评价维度中，有 {len(significant_dims)} 个维度存在显著差异：\n")
        
        # 按效应大小排序显著结果
        significant_dims.sort(key=lambda x: abs(x[1]['effect_size']), reverse=True)
        
        for dim, result in significant_dims:
            cn_name = dimension_names[dim]
            en_desc = dimension_descriptions[dim]
            diff = result['mean_diff']
            effect = result['effect_size']
            p_val = result['p_value']
            
            # 判断效应大小
            if effect < 0.1:
                effect_desc = "极小"
            elif effect < 0.3:
                effect_desc = "小"
            elif effect < 0.5:
                effect_desc = "中等"
            else:
                effect_desc = "大"
            
            if diff > 0:
                report.append(f"- **{cn_name}** ({en_desc}):")
                report.append(f"  - LLMnote评分显著高于Community Note (平均差异: {diff:+.3f}, p = {p_val:.4f})")
                report.append(f"  - 效应大小为{effect_desc} (r = {effect:.3f})")
            else:
                report.append(f"- **{cn_name}** ({en_desc}):")
                report.append(f"  - Community Note评分显著高于LLMnote (平均差异: {diff:+.3f}, p = {p_val:.4f})")
                report.append(f"  - 效应大小为{effect_desc} (r = {effect:.3f})")
        
        if non_significant_dims:
            report.append(f"\n在以下 {len(non_significant_dims)} 个维度上未发现显著差异：")
            for dim, result in non_significant_dims:
                cn_name = dimension_names[dim]
                report.append(f"- {cn_name} (p = {result['p_value']:.4f})")
    
    # 添加总结
    report.append("\n### 总体评价")
    
    # 计算各类别的数量
    llm_better = sum(1 for d, r in test_results.items() if r and r['p_value'] < 0.05 and r['mean_diff'] > 0)
    community_better = sum(1 for d, r in test_results.items() if r and r['p_value'] < 0.05 and r['mean_diff'] < 0)
    no_diff = len(test_results) - llm_better - community_better
    
    if llm_better > community_better:
        report.append(f"LLMnote在 {llm_better} 个维度上表现更好，Community Note在 {community_better} 个维度上表现更好，")
        report.append(f"{no_diff} 个维度无显著差异。总体而言，LLMnote在多数评价维度上获得了更高的用户评分。")
    elif community_better > llm_better:
        report.append(f"Community Note在 {community_better} 个维度上表现更好，LLMnote在 {llm_better} 个维度上表现更好，")
        report.append(f"{no_diff} 个维度无显著差异。总体而言，Community Note在多数评价维度上获得了更高的用户评分。")
    else:
        report.append(f"两种笔记类型表现相当：各有 {llm_better} 个维度表现更好，{no_diff} 个维度无显著差异。")
        report.append("这表明LLMnote和Community Note在整体质量上旗鼓相当。")
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Wilcoxon检验报告已保存到: {output_file}")

if __name__ == "__main__":
    import os
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 要处理的模型列表
    models = ['claude', 'gpt4o', 'grok', 'qwen']
    
    for model in models:
        # 设置输入输出路径（使用绝对路径）
        input_file = os.path.join(script_dir, f"{model}_feature_rating", f"feature_ratings_wide_{model}.csv")
        output_dir = os.path.join(script_dir, f"{model}_feature_rating")
        chart_output = os.path.join(output_dir, f"feature_rating_diverging_chart_{model}.png")
        report_output = os.path.join(output_dir, f"wilcoxon_test_feature_report_{model}.md")
        
        print(f"\n{'='*50}")
        print(f"处理 {model.upper()} 模型数据")
        print(f"{'='*50}")
        
        try:
            print("开始特征评分可视化分析...")
            
            # 读取宽格式数据
            feature_df = pd.read_csv(input_file)
            
            # 准备分歧条形图数据
            diverging_data = prepare_diverging_data(feature_df)
            
            # 进行Wilcoxon检验
            print("\n进行Wilcoxon符号秩检验...")
            test_results = perform_wilcoxon_tests(feature_df)
            
            # 打印检验结果
            print("\nWilcoxon检验结果:")
            for dimension, result in test_results.items():
                if result:
                    print(f"{dimension}: p={result['p_value']:.4f}, 平均差异={result['mean_diff']:+.3f}")
            
            # 创建可视化
            print("\n创建分歧条形图...")
            create_diverging_bar_chart(diverging_data, test_results, chart_output)
            
            # 生成报告
            generate_wilcoxon_report(test_results, report_output)
            
            print("\n特征评分可视化分析完成！")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {input_file}")
            print(f"请先运行 extract_feature_ratings.py 生成 {model} 的数据")
        except Exception as e:
            print(f"处理 {model} 过程中出现错误: {str(e)}")