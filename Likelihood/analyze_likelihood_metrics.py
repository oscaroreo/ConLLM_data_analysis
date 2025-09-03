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

def create_visualizations(df, mean_values, test_results):
    """Create comprehensive visualizations."""
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Bar chart comparing mean values
    ax1 = plt.subplot(2, 2, 1)
    mean_values.plot(kind='bar', ax=ax1)
    ax1.set_title('Average Likelihood Scores by Note Type', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Note Type')
    ax1.set_ylabel('Average Score (1-7)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.legend(title='Metric')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.2f')
    
    # 2. Box plots for each metric
    metrics = ['share', 'comment', 'like']
    for i, metric in enumerate(metrics):
        ax = plt.subplot(2, 3, i+4)
        df.boxplot(column=metric, by='note_category', ax=ax)
        ax.set_title(f'{metric.capitalize()} Likelihood Distribution', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('Score (1-7)')
        plt.suptitle('')  # Remove default suptitle
        
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
    
    # 3. Heatmap of mean values
    ax3 = plt.subplot(2, 2, 2)
    sns.heatmap(mean_values.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Average Score'}, ax=ax3)
    ax3.set_title('Heatmap of Average Likelihood Scores', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Note Type')
    ax3.set_ylabel('Metric')
    
    plt.tight_layout()
    plt.savefig('likelihood_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second figure for violin plots
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    
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
    """Generate a comprehensive analysis report."""
    report = []
    report.append("# Likelihood Analysis Report: LLM Notes vs Community Notes\n")
    report.append(f"## Data Overview\n")
    report.append(f"- Total samples analyzed: {len(df)}\n")
    report.append(f"- LLM Note samples: {len(df[df['note_category'] == 'LLM Note'])}\n")
    report.append(f"- Community Note samples: {len(df[df['note_category'] == 'Community Note'])}\n\n")
    
    report.append("## Summary Statistics\n")
    report.append("### Mean Likelihood Scores (Scale: 1-7)\n")
    report.append(mean_values.round(3).to_string())
    report.append("\n\n### Detailed Statistics\n")
    report.append(stats_by_category.round(3).to_string())
    
    report.append("\n\n## Statistical Test Results (Mann-Whitney U Test)\n")
    for metric, results in test_results.items():
        report.append(f"\n### {metric.capitalize()} Likelihood\n")
        report.append(f"- LLM Note Mean: {results['llm_mean']:.3f}\n")
        report.append(f"- Community Note Mean: {results['community_mean']:.3f}\n")
        report.append(f"- Difference: {results['difference']:.3f}\n")
        report.append(f"- U-statistic: {results['statistic']:.2f}\n")
        report.append(f"- p-value: {results['p_value']:.4f}\n")
        report.append(f"- Cohen's d: {results['cohens_d']:.3f}\n")
        
        # Interpretation
        if results['p_value'] < 0.05:
            direction = "higher" if results['difference'] > 0 else "lower"
            report.append(f"- **Significant difference**: LLM notes have {direction} {metric} likelihood\n")
        else:
            report.append(f"- No significant difference in {metric} likelihood\n")
    
    report.append("\n## Key Findings\n")
    
    # Identify which metrics show significant differences
    sig_metrics = [m for m, r in test_results.items() if r['p_value'] < 0.05]
    
    if sig_metrics:
        report.append(f"1. **Significant differences found in**: {', '.join(sig_metrics)}\n")
        for metric in sig_metrics:
            diff = test_results[metric]['difference']
            direction = "higher" if diff > 0 else "lower"
            report.append(f"   - LLM notes show {abs(diff):.3f} points {direction} {metric} likelihood\n")
    else:
        report.append("1. No significant differences found between LLM and Community notes\n")
    
    # Overall pattern analysis
    overall_llm = mean_values.loc['LLM Note'].mean()
    overall_community = mean_values.loc['Community Note'].mean()
    
    report.append(f"\n2. **Overall engagement likelihood**:\n")
    report.append(f"   - LLM Notes: {overall_llm:.3f} (average across all metrics)\n")
    report.append(f"   - Community Notes: {overall_community:.3f} (average across all metrics)\n")
    
    # Metric-specific insights
    report.append("\n3. **Metric-specific insights**:\n")
    for metric in ['share', 'comment', 'like']:
        llm_val = mean_values.loc['LLM Note', metric]
        comm_val = mean_values.loc['Community Note', metric]
        report.append(f"   - {metric.capitalize()}: LLM ({llm_val:.2f}) vs Community ({comm_val:.2f})\n")
    
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
    create_visualizations(df, mean_values, test_results)
    
    print("Generating report...")
    report = generate_report(stats_by_category, mean_values, test_results, df)
    
    with open('likelihood_analysis_report.md', 'w') as f:
        f.write(report)
    
    # Save summary statistics
    mean_values.to_csv('likelihood_mean_values.csv')
    
    print("\nAnalysis complete! Files generated:")
    print("- likelihood_analysis_comprehensive.png")
    print("- likelihood_violin_plots.png")
    print("- likelihood_analysis_report.md")
    print("- likelihood_raw_data.csv")
    print("- likelihood_mean_values.csv")

if __name__ == "__main__":
    main()