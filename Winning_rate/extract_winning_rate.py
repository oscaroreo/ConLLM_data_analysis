import pandas as pd
import numpy as np

def extract_winning_rate_data(input_csv):
    """
    从consolidated_results.csv中提取获胜率(winning rate)数据
    分析Community Note与LLM Note的有效性比较
    """
    
    # 读取数据
    df = pd.read_csv(input_csv)
    
    print("=== 获胜率分析数据提取 ===")
    print(f"原始数据: {len(df)}条记录")
    print(f"参与者数量: {df['participant_name'].nunique()}人")
    print(f"推文数量: {df['post_index'].nunique()}条")
    
    # 创建清理后的数据框
    winning_data = []
    
    for index, row in df.iterrows():
        # 提取基本信息
        participant_name = row['participant_name']
        post_index = row['post_index']
        
        # 获取比较结果 (comparison字段)
        comparison = row['comparison']
        
        # 创建获胜者标识
        winner = None
        if comparison == 'community_note':
            winner = 'Community'
        elif comparison == 'llm_note':
            winner = 'LLMnote'
        else:
            winner = 'tie'  # 平局或无效数据
        
        # 获取帮助性评分用于分析
        community_helpfulness = row['community_note_helpfulness']
        llm_helpfulness = row['llm_note_helpfulness']
        
        # 数值化帮助性评分
        helpfulness_mapping = {
            'not helpful': 0,
            'somewhat helpful': 0.5,
            'helpful': 1
        }
        
        community_score = helpfulness_mapping.get(community_helpfulness, np.nan)
        llm_score = helpfulness_mapping.get(llm_helpfulness, np.nan)
        
        # 计算评分差异 (LLM - Community)
        score_difference = None
        if not pd.isna(community_score) and not pd.isna(llm_score):
            score_difference = llm_score - community_score
        
        # 添加到结果列表
        winning_data.append({
            'participant_name': participant_name,
            'post_index': post_index,
            'winner': winner,
            'community_helpfulness_raw': community_helpfulness,
            'llm_helpfulness_raw': llm_helpfulness,
            'community_helpfulness_score': community_score,
            'llm_helpfulness_score': llm_score,
            'score_difference': score_difference,
            'note_mapping': row['note_mapping']  # 记录展示顺序
        })
    
    # 转换为DataFrame
    winning_df = pd.DataFrame(winning_data)
    
    # 数据清理：移除无效记录
    valid_data = winning_df.dropna(subset=['community_helpfulness_score', 'llm_helpfulness_score'])
    
    print(f"有效记录: {len(valid_data)}条")
    print(f"无效记录: {len(winning_df) - len(valid_data)}条")
    
    # 输出获胜率统计
    print(f"\n=== 获胜率统计 ===")
    winner_counts = valid_data['winner'].value_counts()
    total_valid = len(valid_data)
    
    for winner_type, count in winner_counts.items():
        percentage = (count / total_valid) * 100
        print(f"{winner_type}: {count}次 ({percentage:.1f}%)")
    
    # 按参与者统计获胜率
    participant_stats = valid_data.groupby('participant_name')['winner'].value_counts().unstack(fill_value=0)
    if 'tie' not in participant_stats.columns:
        participant_stats['tie'] = 0
    
    # 计算每个参与者的获胜率
    participant_stats['total'] = participant_stats.sum(axis=1)
    participant_stats['community_rate'] = participant_stats['Community'] / participant_stats['total']
    participant_stats['llm_rate'] = participant_stats['LLMnote'] / participant_stats['total']
    participant_stats['tie_rate'] = participant_stats['tie'] / participant_stats['total']
    
    print(f"\n=== 按参与者的获胜率分布 ===")
    print(f"Community Note获胜率 - 平均: {participant_stats['community_rate'].mean():.3f}, 标准差: {participant_stats['community_rate'].std():.3f}")
    print(f"LLM Note获胜率 - 平均: {participant_stats['llm_rate'].mean():.3f}, 标准差: {participant_stats['llm_rate'].std():.3f}")
    print(f"平局率 - 平均: {participant_stats['tie_rate'].mean():.3f}, 标准差: {participant_stats['tie_rate'].std():.3f}")
    
    # 评分差异分析
    print(f"\n=== 评分差异分析 ===")
    score_diffs = valid_data['score_difference'].dropna()
    print(f"平均评分差异 (LLM - Community): {score_diffs.mean():.4f}")
    print(f"评分差异标准差: {score_diffs.std():.4f}")
    print(f"LLM评分更高的比例: {(score_diffs > 0).mean():.3f}")
    print(f"Community评分更高的比例: {(score_diffs < 0).mean():.3f}")
    print(f"评分相等的比例: {(score_diffs == 0).mean():.3f}")
    
    return valid_data, participant_stats

def generate_winning_rate_summary(winning_df, participant_stats):
    """
    生成获胜率分析的详细汇总
    """
    
    total_comparisons = len(winning_df)
    
    # 总体获胜率
    winner_counts = winning_df['winner'].value_counts()
    community_wins = winner_counts.get('Community', 0)
    llm_wins = winner_counts.get('LLMnote', 0)
    ties = winner_counts.get('tie', 0)
    
    # 创建汇总报告
    summary = {
        'total_comparisons': total_comparisons,
        'community_wins': community_wins,
        'llm_wins': llm_wins,
        'ties': ties,
        'community_win_rate': community_wins / total_comparisons,
        'llm_win_rate': llm_wins / total_comparisons,
        'tie_rate': ties / total_comparisons,
        'participant_community_mean': participant_stats['community_rate'].mean(),
        'participant_llm_mean': participant_stats['llm_rate'].mean(),
        'participant_tie_mean': participant_stats['tie_rate'].mean(),
        'score_difference_mean': winning_df['score_difference'].mean(),
        'score_difference_std': winning_df['score_difference'].std()
    }
    
    print(f"\n" + "="*60)
    print(f"获胜率分析汇总报告")
    print("="*60)
    
    print(f"\n总体统计:")
    print(f"- 总比较次数: {summary['total_comparisons']}")
    print(f"- Community Note获胜: {summary['community_wins']}次 ({summary['community_win_rate']:.1%})")
    print(f"- LLM Note获胜: {summary['llm_wins']}次 ({summary['llm_win_rate']:.1%})")
    print(f"- 平局: {summary['ties']}次 ({summary['tie_rate']:.1%})")
    
    print(f"\n参与者层面平均获胜率:")
    print(f"- Community Note: {summary['participant_community_mean']:.1%}")
    print(f"- LLM Note: {summary['participant_llm_mean']:.1%}")
    print(f"- 平局: {summary['participant_tie_mean']:.1%}")
    
    print(f"\n评分差异 (LLM - Community):")
    print(f"- 平均差异: {summary['score_difference_mean']:.4f}")
    print(f"- 标准差: {summary['score_difference_std']:.4f}")
    
    # 判断显著性倾向
    if abs(summary['llm_win_rate'] - summary['community_win_rate']) < 0.05:
        conclusion = "两种笔记类型表现相当，无明显偏好"
    elif summary['llm_win_rate'] > summary['community_win_rate']:
        conclusion = f"LLM Note表现略优，领先{summary['llm_win_rate'] - summary['community_win_rate']:.1%}"
    else:
        conclusion = f"Community Note表现略优，领先{summary['community_win_rate'] - summary['llm_win_rate']:.1%}"
    
    print(f"\n结论: {conclusion}")
    
    return summary

def save_winning_rate_data(winning_df, participant_stats, output_file):
    """
    保存获胜率数据到CSV文件
    """
    
    # 保存详细数据
    winning_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n获胜率详细数据已保存到: {output_file}")
    
    # 保存参与者统计数据
    participant_file = output_file.replace('.csv', '_by_participant.csv')
    participant_stats.to_csv(participant_file, encoding='utf-8')
    print(f"按参与者统计数据已保存到: {participant_file}")
    
    return output_file, participant_file

if __name__ == "__main__":
    # 文件路径
    input_file = "../consolidated_results.csv"
    output_file = "winning_rate_data.csv"
    
    print("开始提取获胜率数据...")
    
    # 提取数据
    winning_df, participant_stats = extract_winning_rate_data(input_file)
    
    # 生成汇总报告
    summary = generate_winning_rate_summary(winning_df, participant_stats)
    
    # 保存数据
    save_winning_rate_data(winning_df, participant_stats, output_file)
    
    print("\n获胜率数据提取完成！")