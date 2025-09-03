# Post Rating Distribution Analysis Report

## Overview
This report analyzes the rating distributions for 60 unique posts across three metrics (share, comment, like) and two note categories (Community Note, LLM Note).

## Key Findings

### 1. Overall Statistics
- **Total unique posts analyzed**: 60
- **Post IDs**: 1, 2, 3, 7, 12, 16, 19, 21, 23, 24, 28, 29, 30, 31, 32, 34, 35, 36, 46, 49, 50, 52, 57, 59, 60, 62, 65, 68, 69, 70, 72, 74, 75, 76, 77, 79, 81, 82, 83, 85, 86, 91, 92, 93, 94, 98, 99, 101, 102, 106, 107, 109, 110, 111, 112, 114, 122, 123, 124, 142

### 2. Highest Rated Posts

#### Community Notes
- **Top Share Posts**: Post 3 (3.92), Post 99 (3.75), Post 7 (3.60)
- **Top Comment Posts**: Post 109 (3.90), Post 3 (3.88), Post 29 (3.78)
- **Top Like Posts**: Post 3 (4.28), Post 29 (3.88), Post 122 (3.88)

**Key Insight**: Post 3 consistently receives high ratings across all metrics for Community Notes.

#### LLM Notes
- **Top Share Posts**: Post 76 (4.08), Post 98 (4.00), Post 109 (3.96)
- **Top Comment Posts**: Post 98 (4.20), Post 76 (4.18), Post 109 (4.13)
- **Top Like Posts**: Post 75 (4.23), Post 98 (4.15), Post 57 (4.10)

**Key Insight**: Posts 76, 98, and 109 are highly rated for LLM Notes across multiple metrics.

### 3. Lowest Rated Posts

#### Community Notes
- **Bottom Share Posts**: Post 114 (2.59), Post 94 (2.63), Post 49 (2.80)
- **Bottom Comment Posts**: Post 114 (2.57), Post 94 (2.73), Post 77 (2.91)
- **Bottom Like Posts**: Post 114 (2.87), Post 49 (2.91), Post 76 (3.00)

**Key Insight**: Post 114 consistently receives the lowest ratings for Community Notes.

#### LLM Notes
- **Bottom Share Posts**: Post 74 (3.00), Post 24 (3.08), Post 1 (3.09)
- **Bottom Comment Posts**: Post 1 (3.02), Post 24 (3.04), Post 74 (3.15)
- **Bottom Like Posts**: Post 24 (3.14), Post 21 (3.20), Post 91 (3.24)

**Key Insight**: LLM Notes generally receive higher ratings than Community Notes, with the lowest ratings still above 3.0.

### 4. Rating Patterns

1. **LLM Notes vs Community Notes**: LLM Notes tend to receive higher average ratings across all metrics compared to Community Notes.

2. **Metric Correlations**: Posts that receive high ratings in one metric (e.g., share) tend to receive high ratings in other metrics (comment, like) as well.

3. **High Variance Posts**: Some posts show significant variance in their ratings, indicating disagreement among users about their quality.

### 5. Topic Categories (from topics.json)

Posts cover various topics including:
- Finance/Business
- Politics
- Entertainment
- Emergency Events
- Science/Technology
- Ongoing News Stories

The rating patterns may be influenced by topic categories, with some topics potentially receiving more favorable ratings than others.

## Files Generated

1. **post_rating_distributions.json**: Detailed frequency distributions for each post
2. **post_rating_summary_stats.csv**: Summary statistics including mean, median, and standard deviation
3. **post_rating_distributions_overview.png**: Bar charts showing mean ratings by post
4. **post_rating_heatmap.png**: Heatmap visualization of ratings across posts and metrics
5. **selected_posts_distributions.png**: Detailed frequency distributions for high-variance posts

## Recommendations

1. **Focus on Low-Rated Posts**: Posts like 114, 94, and 49 (Community Notes) may need quality improvements
2. **Learn from High-Rated Posts**: Analyze characteristics of posts 3, 99, 76, and 98 to understand what makes them successful
3. **Address Rating Disparities**: Investigate why Community Notes generally receive lower ratings than LLM Notes
4. **Topic Analysis**: Consider conducting deeper analysis on how topic categories influence ratings