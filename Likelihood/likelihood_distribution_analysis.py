import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

def analyze_likelihood_distributions(csv_path):
    """
    Analyze likelihood distributions for 60 post IDs across share, comment, and like ratings.
    Separate analysis by note category (Community Note vs LLM Note).
    """
    
    # Read the CSV file
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    print(f"Total records: {len(df)}")
    print(f"Unique post IDs: {df['response_id'].nunique()}")
    print(f"Note categories: {df['note_category'].unique()}")
    print(f"Models: {df['model'].unique()}")
    
    # Get unique post IDs
    unique_posts = sorted(df['response_id'].unique())
    print(f"\nAnalyzing {len(unique_posts)} unique posts")
    
    # Rating columns to analyze
    rating_cols = ['share', 'comment', 'like']
    note_categories = df['note_category'].unique()
    
    # Initialize results dictionary
    results = {
        'summary': {
            'total_posts': len(unique_posts),
            'note_categories': list(note_categories),
            'rating_metrics': rating_cols
        },
        'post_analysis': {}
    }
    
    # Analyze each post
    for post_id in unique_posts:
        post_data = df[df['response_id'] == post_id]
        results['post_analysis'][post_id] = {}
        
        # Analyze by note category
        for category in note_categories:
            category_data = post_data[post_data['note_category'] == category]
            
            if len(category_data) == 0:
                continue
                
            results['post_analysis'][post_id][category] = {
                'n_responses': len(category_data),
                'metrics': {}
            }
            
            # Analyze each rating metric
            for metric in rating_cols:
                ratings = category_data[metric].values
                
                # Calculate frequency distribution (1-5 scale)
                freq_dist = {}
                for rating in range(1, 6):
                    freq_dist[rating] = int(np.sum(ratings == rating))
                
                # Calculate statistics
                metric_stats = {
                    'frequency_distribution': freq_dist,
                    'mean': float(np.mean(ratings)),
                    'std': float(np.std(ratings, ddof=1)) if len(ratings) > 1 else 0.0,
                    'median': float(np.median(ratings)),
                    'min': int(np.min(ratings)),
                    'max': int(np.max(ratings)),
                    'total_responses': len(ratings)
                }
                
                results['post_analysis'][post_id][category]['metrics'][metric] = metric_stats
    
    return results, df

def identify_notable_posts(results):
    """
    Identify posts with notably high or low ratings.
    """
    
    rating_cols = ['share', 'comment', 'like']
    note_categories = results['summary']['note_categories']
    
    # Collect all means for comparison
    all_means = {metric: {category: [] for category in note_categories} for metric in rating_cols}
    
    for post_id, post_data in results['post_analysis'].items():
        for category in note_categories:
            if category in post_data:
                for metric in rating_cols:
                    if metric in post_data[category]['metrics']:
                        mean_val = post_data[category]['metrics'][metric]['mean']
                        all_means[metric][category].append((post_id, mean_val))
    
    # Find notable posts (top and bottom 10%)
    notable_posts = {
        'high_ratings': {metric: {category: [] for category in note_categories} for metric in rating_cols},
        'low_ratings': {metric: {category: [] for category in note_categories} for metric in rating_cols}
    }
    
    for metric in rating_cols:
        for category in note_categories:
            if len(all_means[metric][category]) > 0:
                sorted_means = sorted(all_means[metric][category], key=lambda x: x[1], reverse=True)
                n_posts = len(sorted_means)
                
                # Top 10% (high ratings)
                top_n = max(1, n_posts // 10)
                notable_posts['high_ratings'][metric][category] = sorted_means[:top_n]
                
                # Bottom 10% (low ratings)
                bottom_n = max(1, n_posts // 10)
                notable_posts['low_ratings'][metric][category] = sorted_means[-bottom_n:]
    
    return notable_posts

def create_summary_statistics(results):
    """
    Create overall summary statistics across all posts.
    """
    
    rating_cols = ['share', 'comment', 'like']
    note_categories = results['summary']['note_categories']
    
    summary_stats = {}
    
    for category in note_categories:
        summary_stats[category] = {}
        
        for metric in rating_cols:
            all_means = []
            all_stds = []
            total_responses = 0
            
            # Collect data across all posts
            for post_id, post_data in results['post_analysis'].items():
                if category in post_data and metric in post_data[category]['metrics']:
                    all_means.append(post_data[category]['metrics'][metric]['mean'])
                    all_stds.append(post_data[category]['metrics'][metric]['std'])
                    total_responses += post_data[category]['metrics'][metric]['total_responses']
            
            if len(all_means) > 0:
                summary_stats[category][metric] = {
                    'overall_mean': float(np.mean(all_means)),
                    'mean_std': float(np.std(all_means, ddof=1)) if len(all_means) > 1 else 0.0,
                    'avg_post_std': float(np.mean(all_stds)),
                    'posts_analyzed': len(all_means),
                    'total_responses': total_responses
                }
    
    return summary_stats

def save_results(results, notable_posts, summary_stats, output_dir):
    """
    Save all analysis results to structured files.
    """
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    with open(os.path.join(output_dir, 'detailed_post_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save notable posts
    with open(os.path.join(output_dir, 'notable_posts_analysis.json'), 'w') as f:
        json.dump(notable_posts, f, indent=2)
    
    # Save summary statistics
    with open(os.path.join(output_dir, 'summary_statistics.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create CSV summary for easy viewing
    create_csv_summary(results, summary_stats, output_dir)

def create_csv_summary(results, summary_stats, output_dir):
    """
    Create CSV summaries for easy analysis.
    """
    
    import os
    
    # Summary by post and category
    post_summary_data = []
    
    for post_id, post_data in results['post_analysis'].items():
        for category, category_data in post_data.items():
            row = {
                'post_id': post_id,
                'note_category': category,
                'n_responses': category_data['n_responses']
            }
            
            # Add metrics
            for metric in ['share', 'comment', 'like']:
                if metric in category_data['metrics']:
                    metric_data = category_data['metrics'][metric]
                    row[f'{metric}_mean'] = metric_data['mean']
                    row[f'{metric}_std'] = metric_data['std']
                    row[f'{metric}_median'] = metric_data['median']
                    
                    # Add frequency distribution
                    for rating in range(1, 6):
                        row[f'{metric}_freq_{rating}'] = metric_data['frequency_distribution'].get(rating, 0)
                else:
                    # Fill with NaN if no data
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_median'] = np.nan
                    for rating in range(1, 6):
                        row[f'{metric}_freq_{rating}'] = 0
            
            post_summary_data.append(row)
    
    # Save post summary CSV
    post_summary_df = pd.DataFrame(post_summary_data)
    post_summary_df.to_csv(os.path.join(output_dir, 'post_summary_analysis.csv'), index=False)
    
    # Create overall summary CSV
    overall_summary_data = []
    for category, category_data in summary_stats.items():
        for metric, metric_data in category_data.items():
            row = {
                'note_category': category,
                'metric': metric,
                'overall_mean': metric_data['overall_mean'],
                'mean_std': metric_data['mean_std'],
                'avg_post_std': metric_data['avg_post_std'],
                'posts_analyzed': metric_data['posts_analyzed'],
                'total_responses': metric_data['total_responses']
            }
            overall_summary_data.append(row)
    
    overall_summary_df = pd.DataFrame(overall_summary_data)
    overall_summary_df.to_csv(os.path.join(output_dir, 'overall_summary_statistics.csv'), index=False)

def create_visualizations(results, summary_stats, output_dir):
    """
    Create visualizations for the rating distributions.
    """
    
    import os
    plt.style.use('default')
    
    # 1. Overall mean comparison by category and metric
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Mean Ratings by Note Category and Metric', fontsize=16)
    
    metrics = ['share', 'comment', 'like']
    categories = list(summary_stats.keys())
    
    for i, metric in enumerate(metrics):
        means = [summary_stats[cat][metric]['overall_mean'] for cat in categories]
        stds = [summary_stats[cat][metric]['mean_std'] for cat in categories]
        
        bars = axes[i].bar(categories, means, yerr=stds, capsize=5, alpha=0.7)
        axes[i].set_title(f'{metric.capitalize()} Ratings')
        axes[i].set_ylabel('Mean Rating')
        axes[i].set_ylim(1, 5)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                        f'{mean:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_ratings_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Distribution of post means
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribution of Post Mean Ratings', fontsize=16)
    
    for cat_idx, category in enumerate(categories):
        for met_idx, metric in enumerate(metrics):
            # Collect all post means for this category and metric
            post_means = []
            for post_id, post_data in results['post_analysis'].items():
                if category in post_data and metric in post_data[category]['metrics']:
                    post_means.append(post_data[category]['metrics'][metric]['mean'])
            
            if len(post_means) > 0:
                axes[cat_idx, met_idx].hist(post_means, bins=20, alpha=0.7, edgecolor='black')
                axes[cat_idx, met_idx].set_title(f'{category} - {metric.capitalize()}')
                axes[cat_idx, met_idx].set_xlabel('Mean Rating')
                axes[cat_idx, met_idx].set_ylabel('Number of Posts')
                axes[cat_idx, met_idx].axvline(np.mean(post_means), color='red', linestyle='--', 
                                             label=f'Overall Mean: {np.mean(post_means):.2f}')
                axes[cat_idx, met_idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'post_means_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Frequency distribution heatmap
    for category in categories:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Rating Frequency Distributions - {category}', fontsize=16)
        
        for met_idx, metric in enumerate(metrics):
            # Create frequency matrix
            posts_with_data = []
            freq_matrix = []
            
            for post_id, post_data in results['post_analysis'].items():
                if category in post_data and metric in post_data[category]['metrics']:
                    freq_dist = post_data[category]['metrics'][metric]['frequency_distribution']
                    freq_row = [freq_dist.get(rating, 0) for rating in range(1, 6)]
                    freq_matrix.append(freq_row)
                    posts_with_data.append(str(post_id))
            
            if len(freq_matrix) > 0:
                freq_matrix = np.array(freq_matrix)
                
                # Create heatmap
                im = axes[met_idx].imshow(freq_matrix, aspect='auto', cmap='YlOrRd')
                axes[met_idx].set_title(f'{metric.capitalize()} Frequency Distribution')
                axes[met_idx].set_xlabel('Rating (1-5)')
                axes[met_idx].set_ylabel('Post ID')
                axes[met_idx].set_xticks(range(5))
                axes[met_idx].set_xticklabels(['1', '2', '3', '4', '5'])
                
                # Add colorbar
                plt.colorbar(im, ax=axes[met_idx], label='Frequency')
                
                # Show only every nth post ID to avoid crowding
                if len(posts_with_data) > 20:
                    step = len(posts_with_data) // 10
                    tick_positions = range(0, len(posts_with_data), step)
                    tick_labels = [posts_with_data[i] for i in tick_positions]
                    axes[met_idx].set_yticks(tick_positions)
                    axes[met_idx].set_yticklabels(tick_labels)
                else:
                    axes[met_idx].set_yticks(range(len(posts_with_data)))
                    axes[met_idx].set_yticklabels(posts_with_data)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'frequency_heatmap_{category.lower().replace(" ", "_")}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Main analysis function.
    """
    
    # File paths
    csv_path = "/Users/shidaidemac/Documents/CHI_25/Data_Analysis_829/Likelihood/likelihood_raw_data.csv"
    output_dir = "/Users/shidaidemac/Documents/CHI_25/Data_Analysis_829/Likelihood"
    
    print("=== Likelihood Distribution Analysis ===")
    print("Analyzing 60 post IDs with share, comment, and like ratings")
    print("Separating by note category (Community Note vs LLM Note)\n")
    
    # Run analysis
    results, df = analyze_likelihood_distributions(csv_path)
    
    print("\n=== Identifying Notable Posts ===")
    notable_posts = identify_notable_posts(results)
    
    print("\n=== Creating Summary Statistics ===")
    summary_stats = create_summary_statistics(results)
    
    # Print summary
    print("\n=== SUMMARY STATISTICS ===")
    for category, category_data in summary_stats.items():
        print(f"\n{category}:")
        for metric, metric_data in category_data.items():
            print(f"  {metric.capitalize()}:")
            print(f"    Overall Mean: {metric_data['overall_mean']:.3f}")
            print(f"    Standard Dev: {metric_data['mean_std']:.3f}")
            print(f"    Posts Analyzed: {metric_data['posts_analyzed']}")
            print(f"    Total Responses: {metric_data['total_responses']}")
    
    # Print notable posts
    print("\n=== NOTABLE POSTS ===")
    for rating_type in ['high_ratings', 'low_ratings']:
        print(f"\n{rating_type.replace('_', ' ').title()}:")
        for metric in ['share', 'comment', 'like']:
            print(f"  {metric.capitalize()}:")
            for category in summary_stats.keys():
                posts = notable_posts[rating_type][metric][category]
                if posts:
                    print(f"    {category}: {[(p[0], f'{p[1]:.2f}') for p in posts]}")
    
    print("\n=== Saving Results ===")
    save_results(results, notable_posts, summary_stats, output_dir)
    
    print("\n=== Creating Visualizations ===")
    create_visualizations(results, summary_stats, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print("Files created:")
    print("- detailed_post_analysis.json: Complete analysis for each post")
    print("- notable_posts_analysis.json: Posts with highest/lowest ratings")
    print("- summary_statistics.json: Overall statistics")
    print("- post_summary_analysis.csv: Summary data in CSV format")
    print("- overall_summary_statistics.csv: Overall statistics in CSV")
    print("- mean_ratings_comparison.png: Mean ratings comparison chart")
    print("- post_means_distribution.png: Distribution of post means")
    print("- frequency_heatmap_*.png: Frequency distribution heatmaps")

if __name__ == "__main__":
    main()