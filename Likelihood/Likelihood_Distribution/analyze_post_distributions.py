import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

# Read the data from parent directory
df = pd.read_csv('../likelihood_raw_data.csv')

# Get unique post IDs (response_id) - should be 60
unique_posts = sorted(df['response_id'].unique())
print(f"Total unique posts: {len(unique_posts)}")
print(f"Post IDs: {unique_posts}")

# Initialize storage for distributions
distributions = {}

# Metrics to analyze
metrics = ['share', 'comment', 'like']
note_categories = ['Community Note', 'LLM Note']

# Analyze distribution for each post
for post_id in unique_posts:
    distributions[int(post_id)] = {}
    
    for category in note_categories:
        distributions[int(post_id)][category] = {}
        
        # Filter data for this post and category
        post_data = df[(df['response_id'] == post_id) & (df['note_category'] == category)]
        
        for metric in metrics:
            # Get value counts for ratings 1-5
            value_counts = post_data[metric].value_counts().sort_index()
            
            # Create frequency distribution for ratings 1-5
            freq_dist = {}
            for rating in range(1, 6):
                freq_dist[rating] = int(value_counts.get(rating, 0))
            
            # Calculate statistics
            values = post_data[metric].values
            stats = {
                'frequency': freq_dist,
                'mean': float(np.mean(values)) if len(values) > 0 else 0,
                'std': float(np.std(values)) if len(values) > 0 else 0,
                'median': float(np.median(values)) if len(values) > 0 else 0,
                'total_responses': len(values)
            }
            
            distributions[int(post_id)][category][metric] = stats

# Save detailed distributions
with open('post_rating_distributions.json', 'w') as f:
    json.dump(distributions, f, indent=2)

# Create summary statistics
summary_stats = []

for post_id in unique_posts:
    for category in note_categories:
        row = {'post_id': int(post_id), 'note_category': category}
        
        for metric in metrics:
            if category in distributions[int(post_id)] and metric in distributions[int(post_id)][category]:
                stats = distributions[int(post_id)][category][metric]
                row[f'{metric}_mean'] = stats['mean']
                row[f'{metric}_std'] = stats['std']
                row[f'{metric}_median'] = stats['median']
                row[f'{metric}_responses'] = stats['total_responses']
                
                # Add frequency distribution
                for rating in range(1, 6):
                    row[f'{metric}_rating_{rating}'] = stats['frequency'][rating]
        
        summary_stats.append(row)

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('post_rating_summary_stats.csv', index=False)

# Identify posts with extreme ratings
print("\n=== Posts with Highest Average Ratings ===")
for category in note_categories:
    print(f"\n{category}:")
    category_df = summary_df[summary_df['note_category'] == category]
    
    for metric in metrics:
        top_posts = category_df.nlargest(5, f'{metric}_mean')[['post_id', f'{metric}_mean']]
        print(f"\nTop 5 posts for {metric}:")
        print(top_posts.to_string(index=False))

print("\n=== Posts with Lowest Average Ratings ===")
for category in note_categories:
    print(f"\n{category}:")
    category_df = summary_df[summary_df['note_category'] == category]
    
    for metric in metrics:
        bottom_posts = category_df.nsmallest(5, f'{metric}_mean')[['post_id', f'{metric}_mean']]
        print(f"\nBottom 5 posts for {metric}:")
        print(bottom_posts.to_string(index=False))

# Create visualization showing distribution patterns
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Rating Distributions by Post ID', fontsize=16)

for i, category in enumerate(note_categories):
    for j, metric in enumerate(metrics):
        ax = axes[i][j]
        
        # Extract mean ratings for all posts
        category_df = summary_df[summary_df['note_category'] == category]
        means = category_df[f'{metric}_mean'].values
        post_ids = category_df['post_id'].values
        
        # Create bar plot
        ax.bar(range(len(post_ids)), means)
        ax.set_xlabel('Post Index')
        ax.set_ylabel(f'Mean {metric} Rating')
        ax.set_title(f'{category} - {metric}')
        ax.set_ylim(1, 5)
        
        # Add horizontal line for overall mean
        overall_mean = np.mean(means)
        ax.axhline(y=overall_mean, color='r', linestyle='--', alpha=0.5, label=f'Mean: {overall_mean:.2f}')
        ax.legend()

plt.tight_layout()
plt.savefig('post_rating_distributions_overview.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a heatmap showing rating patterns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

for idx, category in enumerate(note_categories):
    ax = [ax1, ax2][idx]
    
    # Create matrix for heatmap
    heatmap_data = []
    
    for post_id in unique_posts:
        row = []
        for metric in metrics:
            if category in distributions[int(post_id)] and metric in distributions[int(post_id)][category]:
                row.append(distributions[int(post_id)][category][metric]['mean'])
            else:
                row.append(0)
        heatmap_data.append(row)
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                xticklabels=metrics,
                yticklabels=unique_posts,
                cmap='YlOrRd',
                ax=ax,
                vmin=1, vmax=5,
                cbar_kws={'label': 'Mean Rating'})
    
    ax.set_title(f'{category} - Mean Ratings by Post ID')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Post ID')

plt.tight_layout()
plt.savefig('post_rating_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Create detailed distribution plots for selected posts
# Select posts with high variance in ratings
high_variance_posts = []
for post_id in unique_posts:
    variances = []
    for category in note_categories:
        for metric in metrics:
            if category in distributions[int(post_id)] and metric in distributions[int(post_id)][category]:
                variances.append(distributions[int(post_id)][category][metric]['std'])
    if variances:
        high_variance_posts.append((post_id, np.mean(variances)))

high_variance_posts.sort(key=lambda x: x[1], reverse=True)
selected_posts = [p[0] for p in high_variance_posts[:6]]  # Top 6 high variance posts

# Create distribution plots for selected posts
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Rating Frequency Distributions for Selected Posts', fontsize=16)

for idx, post_id in enumerate(selected_posts[:6]):
    ax = axes[idx // 3][idx % 3]
    
    # Prepare data for grouped bar chart
    x = np.arange(5)  # Ratings 1-5
    width = 0.35
    
    # Community Note frequencies
    cn_freqs = []
    llm_freqs = []
    
    for rating in range(1, 6):
        cn_total = 0
        llm_total = 0
        for metric in metrics:
            if 'Community Note' in distributions[int(post_id)]:
                cn_total += distributions[int(post_id)]['Community Note'][metric]['frequency'][rating]
            if 'LLM Note' in distributions[int(post_id)]:
                llm_total += distributions[int(post_id)]['LLM Note'][metric]['frequency'][rating]
        cn_freqs.append(cn_total)
        llm_freqs.append(llm_total)
    
    # Create grouped bar chart
    ax.bar(x - width/2, cn_freqs, width, label='Community Note')
    ax.bar(x + width/2, llm_freqs, width, label='LLM Note')
    
    ax.set_xlabel('Rating')
    ax.set_ylabel('Total Frequency')
    ax.set_title(f'Post {post_id}')
    ax.set_xticks(x)
    ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.legend()

plt.tight_layout()
plt.savefig('selected_posts_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete! Files created in Likelihood_Distribution folder:")
print("- post_rating_distributions.json (detailed distributions)")
print("- post_rating_summary_stats.csv (summary statistics)")
print("- post_rating_distributions_overview.png (bar charts)")
print("- post_rating_heatmap.png (heatmap visualization)")
print("- selected_posts_distributions.png (detailed distributions for high-variance posts)")