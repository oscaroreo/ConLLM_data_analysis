import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

def load_data():
    """Load the processed data."""
    df = pd.read_csv('likelihood_raw_data.csv')
    mean_values = pd.read_csv('likelihood_mean_values.csv', index_col=0)
    return df, mean_values

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

def create_bar_chart(mean_values):
    """Create bar chart comparing mean values."""
    plt.figure(figsize=(10, 8))
    
    mean_values.plot(kind='bar')
    plt.title('Average Likelihood Scores by Note Type', fontsize=16, fontweight='bold')
    plt.xlabel('Note Type', fontsize=12)
    plt.ylabel('Average Score (1-7)', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Metric', labels=['Share', 'Comment', 'Like'], fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('likelihood_1_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_heatmap(mean_values):
    """Create heatmap of mean values."""
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(mean_values.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Average Score'}, 
                square=True,
                linewidths=0.5)
    plt.title('Heatmap of Average Likelihood Scores', fontsize=16, fontweight='bold')
    plt.xlabel('Note Type', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('likelihood_2_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_boxplot(df, metric, test_results, plot_num):
    """Create individual boxplot for a specific metric."""
    plt.figure(figsize=(8, 6))
    
    # Create box plot data
    community_data = df[df['note_category'] == 'Community Note'][metric]
    llm_data = df[df['note_category'] == 'LLM Note'][metric]
    
    # Create box plot
    bp = plt.boxplot([community_data, llm_data], 
                    tick_labels=['Community Note', 'LLM Note'],
                    patch_artist=True,
                    notch=True,  # Add notches to show confidence intervals
                    showmeans=True)  # Show mean points
    
    # Customize box plot colors
    colors = ['#87CEEB', '#90EE90']  # Sky blue and light green
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Customize other elements
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(bp[element], color='black')
    plt.setp(bp['medians'], color='darkred', linewidth=2)
    plt.setp(bp['means'], marker='o', markerfacecolor='red', markeredgecolor='black', markersize=8)
    
    metric_title = metric.capitalize()
    plt.title(f'{metric_title} Likelihood Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('Score (1-7)', fontsize=12)
    plt.ylim(0.5, 7.5)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
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
    
    plt.text(0.5, 0.95, f'p = {p_value:.4f} {sig_text}', 
            transform=plt.gca().transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=12)
    
    # Add mean values text
    plt.text(1, 7.2, f'Mean: {test_results[metric]["community_mean"]:.2f}', 
            ha='center', va='top', fontsize=10)
    plt.text(2, 7.2, f'Mean: {test_results[metric]["llm_mean"]:.2f}', 
            ha='center', va='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'likelihood_{plot_num}_boxplot_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_boxplot(df, test_results):
    """Create a single figure with all three boxplots side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['share', 'comment', 'like']
    colors = ['#87CEEB', '#90EE90']  # Sky blue and light green
    
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        # Create box plot data
        community_data = df[df['note_category'] == 'Community Note'][metric]
        llm_data = df[df['note_category'] == 'LLM Note'][metric]
        
        # Create box plot
        bp = ax.boxplot([community_data, llm_data], 
                       tick_labels=['Community Note', 'LLM Note'],
                       patch_artist=True,
                       notch=True,
                       showmeans=True)
        
        # Customize box plot colors
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        # Customize other elements
        for element in ['whiskers', 'fliers', 'caps']:
            plt.setp(bp[element], color='black')
        plt.setp(bp['medians'], color='darkred', linewidth=2)
        plt.setp(bp['means'], marker='o', markerfacecolor='red', markeredgecolor='black', markersize=8)
        
        metric_title = metric.capitalize()
        ax.set_title(f'{metric_title} Likelihood Distribution', fontsize=14)
        ax.set_ylabel('Score (1-7)')
        ax.set_ylim(0.5, 7.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
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
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Likelihood Distributions Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('likelihood_6_boxplots_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Loading data...")
    df, mean_values = load_data()
    
    print("Performing statistical tests...")
    test_results = perform_statistical_tests(df)
    
    print("Creating separate plots...")
    
    # 1. Bar chart
    print("1. Creating bar chart...")
    create_bar_chart(mean_values)
    
    # 2. Heatmap
    print("2. Creating heatmap...")
    create_heatmap(mean_values)
    
    # 3-5. Individual boxplots
    metrics = ['share', 'comment', 'like']
    for i, metric in enumerate(metrics, start=3):
        print(f"{i}. Creating {metric} boxplot...")
        create_boxplot(df, metric, test_results, i)
    
    # 6. Combined boxplots (bonus)
    print("6. Creating combined boxplots...")
    create_combined_boxplot(df, test_results)
    
    print("\nAll plots created successfully!")
    print("Files generated:")
    print("- likelihood_1_bar_chart.png")
    print("- likelihood_2_heatmap.png")
    print("- likelihood_3_boxplot_share.png")
    print("- likelihood_4_boxplot_comment.png")
    print("- likelihood_5_boxplot_like.png")
    print("- likelihood_6_boxplots_combined.png (bonus)")

if __name__ == "__main__":
    main()