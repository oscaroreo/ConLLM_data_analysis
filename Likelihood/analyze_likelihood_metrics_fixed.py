import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

def extract_likelihood_data():
    """Extract likelihood data from all JSON files."""
    data = []
    base_path = Path("../829_result")
    
    for model_dir in base_path.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name.replace("v0_", "").replace("v1_", "").replace("v2_", "").replace("v3_", "")
            
            for json_file in model_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                    
                    participant_name = content.get('participantName', 'unknown')
                    session_id = content.get('sessionId', 'unknown')
                    
                    # Extract data from userResponses
                    if 'userResponses' in content:
                        for response_id, response_data in content['userResponses'].items():
                            # Extract Community Note data
                            if 'communityNote' in response_data and 'likelihood' in response_data['communityNote']:
                                likelihood = response_data['communityNote']['likelihood']
                                data.append({
                                    'participant_id': participant_name,
                                    'session_id': session_id,
                                    'response_id': response_id,
                                    'model': model_name,
                                    'note_category': 'Community Note',
                                    'share': int(likelihood.get('share', 0)),
                                    'comment': int(likelihood.get('comment', 0)),
                                    'like': int(likelihood.get('like', 0))
                                })
                            
                            # Extract LLM Note data
                            if 'llmNote' in response_data and 'likelihood' in response_data['llmNote']:
                                likelihood = response_data['llmNote']['likelihood']
                                data.append({
                                    'participant_id': participant_name,
                                    'session_id': session_id,
                                    'response_id': response_id,
                                    'model': model_name,
                                    'note_category': 'LLM Note',
                                    'share': int(likelihood.get('share', 0)),
                                    'comment': int(likelihood.get('comment', 0)),
                                    'like': int(likelihood.get('like', 0))
                                })
                                
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
    
    return pd.DataFrame(data)

def analyze_likelihood_metrics(df):
    """Analyze likelihood metrics by note type."""
    # Group by note category and calculate statistics
    stats_by_category = df.groupby('note_category')[['share', 'comment', 'like']].agg(['mean', 'std', 'count'])
    
    # Calculate mean values for each metric
    mean_values = df.groupby('note_category')[['share', 'comment', 'like']].mean()
    
    return stats_by_category, mean_values

def perform_statistical_tests(df):
    """Perform statistical tests between LLM and Community notes."""
    results = {}
    
    llm_data = df[df['note_category'] == 'LLM Note']
    community_data = df[df['note_category'] == 'Community Note']
    
    for metric in ['share', 'comment', 'like']:
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(
            llm_data[metric], 
            community_data[metric], 
            alternative='two-sided'
        )
        
        # Calculate effect size (Cohen's d)
        llm_mean = llm_data[metric].mean()
        community_mean = community_data[metric].mean()
        pooled_std = np.sqrt((llm_data[metric].std()**2 + community_data[metric].std()**2) / 2)
        cohens_d = (llm_mean - community_mean) / pooled_std if pooled_std > 0 else 0
        
        results[metric] = {
            'statistic': statistic,
            'p_value': p_value,
            'llm_mean': llm_mean,
            'community_mean': community_mean,
            'difference': llm_mean - community_mean,
            'cohens_d': cohens_d
        }
    
    return results

def create_comprehensive_visualization(df, mean_values, test_results):
    """Create comprehensive visualization with fixed layout."""
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid spec for better layout control
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.95, bottom=0.05, top=0.95, 
                         wspace=0.3, hspace=0.4)
    
    # 1. Bar chart comparing mean values
    ax1 = fig.add_subplot(gs[0, :2])  # Span first two columns
    mean_values.plot(kind='bar', ax=ax1)
    ax1.set_title('Average Likelihood Scores by Note Type', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Note Type')
    ax1.set_ylabel('Average Score (1-7)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.legend(title='Metric', labels=['Share', 'Comment', 'Like'])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.2f')
    
    # 2. Heatmap of mean values
    ax2 = fig.add_subplot(gs[0, 2])
    sns.heatmap(mean_values.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Average Score'}, ax=ax2)
    ax2.set_title('Heatmap of Average Likelihood Scores', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Note Type')
    ax2.set_ylabel('Metric')
    
    # 3-5. Box plots for each metric
    metrics = ['share', 'comment', 'like']
    
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[1, i])
        # Create box plot data
        community_data = df[df['note_category'] == 'Community Note'][metric]
        llm_data = df[df['note_category'] == 'LLM Note'][metric]
        
        # Create box plot
        bp = ax.boxplot([community_data, llm_data], 
                       labels=['Community Note', 'LLM Note'],
                       patch_artist=True)
        
        # Customize box plot colors
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        metric_title = metric.capitalize()
        ax.set_title(f'{metric_title} Likelihood Distribution', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('Score (1-7)')
        ax.set_ylim(0.5, 7.5)
        
        # Add statistical significance
        p_value = test_results[metric]['p_value']
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'n.s.'
        
        ax.text(0.5, 0.95, f'p = {p_value:.4f} {sig_text}', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('LLM Note vs Community Note Likelihood Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('likelihood_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_violin_plots(df):
    """Create violin plots for distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['share', 'comment', 'like']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.violinplot(data=df, x='note_category', y=metric, ax=ax)
        ax.set_title(f'{metric.capitalize()} Likelihood Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Note Type')
        ax.set_ylabel('Score (1-7)')
        
        # Add mean lines
        for j, category in enumerate(['Community Note', 'LLM Note']):
            mean_val = df[df['note_category'] == category][metric].mean()
            ax.hlines(mean_val, j-0.4, j+0.4, colors='red', linestyles='dashed', 
                     label='Mean' if i == 0 and j == 0 else '')
        
        if i == 0:
            ax.legend()
    
    plt.suptitle('Likelihood Score Distributions by Note Type', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('likelihood_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(stats_by_category, mean_values, test_results, df):
    """Generate a comprehensive analysis report in Chinese."""
    report = []
    report.append("# 意愿度分析报告：LLM笔记 vs 社区笔记\n")
    report.append(f"## 数据概览\n")
    report.append(f"- 分析样本总数：{len(df)}\n")
    report.append(f"- LLM笔记样本数：{len(df[df['note_category'] == 'LLM Note'])}\n")
    report.append(f"- 社区笔记样本数：{len(df[df['note_category'] == 'Community Note'])}\n\n")
    
    report.append("## 汇总统计\n")
    report.append("### 平均意愿评分（评分范围：1-7）\n")
    report.append("```\n")
    report.append(f"                Share  Comment   Like\n")
    report.append(f"Community Note  {mean_values.loc['Community Note', 'share']:.3f}   {mean_values.loc['Community Note', 'comment']:.3f}  {mean_values.loc['Community Note', 'like']:.3f}\n")
    report.append(f"LLM Note        {mean_values.loc['LLM Note', 'share']:.3f}   {mean_values.loc['LLM Note', 'comment']:.3f}  {mean_values.loc['LLM Note', 'like']:.3f}\n")
    report.append("```\n")
    
    report.append("\n## 统计检验结果（Mann-Whitney U检验）\n")
    
    metric_names = {'share': '分享', 'comment': '评论', 'like': '点赞'}
    for metric, results in test_results.items():
        report.append(f"\n### {metric_names[metric]}意愿\n")
        report.append(f"- LLM笔记平均分：{results['llm_mean']:.3f}\n")
        report.append(f"- 社区笔记平均分：{results['community_mean']:.3f}\n")
        report.append(f"- 差值：{results['difference']:.3f}\n")
        report.append(f"- U统计量：{results['statistic']:.2f}\n")
        report.append(f"- p值：{results['p_value']:.4f}\n")
        report.append(f"- Cohen's d效应量：{results['cohens_d']:.3f}\n")
        
        # Interpretation
        if results['p_value'] < 0.05:
            direction = "更高" if results['difference'] > 0 else "更低"
            report.append(f"- **显著差异**：LLM笔记的{metric_names[metric]}意愿{direction}\n")
        else:
            report.append(f"- {metric_names[metric]}意愿无显著差异\n")
    
    report.append("\n## 主要发现\n")
    
    # Identify significant metrics
    sig_metrics = [m for m, r in test_results.items() if r['p_value'] < 0.05]
    
    if sig_metrics:
        report.append(f"1. **在以下维度发现显著差异**：{', '.join([metric_names[m] for m in sig_metrics])}\n")
        for metric in sig_metrics:
            diff = test_results[metric]['difference']
            direction = "高" if diff > 0 else "低"
            report.append(f"   - LLM笔记的{metric_names[metric]}意愿比社区笔记{direction} {abs(diff):.3f} 分\n")
    else:
        report.append("1. LLM笔记和社区笔记之间没有发现显著差异\n")
    
    # Overall pattern analysis
    overall_llm = mean_values.loc['LLM Note'].mean()
    overall_community = mean_values.loc['Community Note'].mean()
    
    report.append(f"\n2. **整体互动意愿**：\n")
    report.append(f"   - LLM笔记：{overall_llm:.3f}（所有指标平均）\n")
    report.append(f"   - 社区笔记：{overall_community:.3f}（所有指标平均）\n")
    report.append(f"   - 差异：{((overall_llm - overall_community) / overall_community * 100):.1f}%\n")
    
    # Specific metric insights
    report.append("\n3. **各指标具体表现**：\n")
    for metric in ['share', 'comment', 'like']:
        llm_val = mean_values.loc['LLM Note', metric]
        comm_val = mean_values.loc['Community Note', metric]
        diff_pct = (llm_val - comm_val) / comm_val * 100
        report.append(f"   - {metric_names[metric]}：LLM笔记 ({llm_val:.2f}) vs 社区笔记 ({comm_val:.2f})，差异 {diff_pct:.1f}%\n")
    
    # Effect size interpretation
    report.append("\n4. **效应量分析**（Cohen's d）：\n")
    for metric in ['share', 'comment', 'like']:
        d = test_results[metric]['cohens_d']
        if abs(d) < 0.2:
            effect = "微小"
        elif abs(d) < 0.5:
            effect = "小"
        elif abs(d) < 0.8:
            effect = "中等"
        else:
            effect = "大"
        report.append(f"   - {metric_names[metric]}：{d:.3f}（{effect}效应）\n")
    
    return ''.join(report)

def main():
    print("Extracting likelihood data...")
    df = extract_likelihood_data()
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"Extracted {len(df)} records")
    
    # Save raw data
    df.to_csv('likelihood_raw_data.csv', index=False)
    
    print("Analyzing likelihood metrics...")
    stats_by_category, mean_values = analyze_likelihood_metrics(df)
    
    print("Performing statistical tests...")
    test_results = perform_statistical_tests(df)
    
    print("Creating visualizations...")
    create_comprehensive_visualization(df, mean_values, test_results)
    create_violin_plots(df)
    
    print("Generating report...")
    report = generate_report(stats_by_category, mean_values, test_results, df)
    
    with open('likelihood_分析报告.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save summary statistics
    mean_values.to_csv('likelihood_mean_values.csv')
    
    print("\nAnalysis complete! Files generated:")
    print("- likelihood_analysis_comprehensive.png")
    print("- likelihood_violin_plots.png") 
    print("- likelihood_分析报告.md")
    print("- likelihood_raw_data.csv")
    print("- likelihood_mean_values.csv")

if __name__ == "__main__":
    main()