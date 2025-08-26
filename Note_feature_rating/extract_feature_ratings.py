import pandas as pd
import numpy as np

def extract_feature_ratings(input_csv):
    """
    从consolidated_results.csv中提取笔记特征评分数据
    包含五个原则：来源质量、清晰度、覆盖范围、上下文、中立性(争议性)
    """
    
    # 读取数据
    df = pd.read_csv(input_csv)
    
    print("=== 笔记特征评分数据提取 ===")
    print(f"原始数据: {len(df)}条记录")
    print(f"参与者数量: {df['participant_name'].nunique()}人")
    print(f"推文数量: {df['post_index'].nunique()}条")
    
    # 创建特征评分数据列表
    feature_data = []
    
    # 五个评分维度的映射
    feature_dimensions = {
        'source_quality': '来源质量',
        'clarity': '清晰度',
        'coverage': '覆盖范围',
        'context': '上下文',
        'impartiality': '中立性'
    }
    
    for index, row in df.iterrows():
        # 基础信息
        base_info = {
            'participant_name': row['participant_name'],
            'participant_id': row['session_id'],
            'post_index': row['post_index'],
            'note_mapping': row['note_mapping'],
            'response_timestamp': row['response_timestamp']
        }
        
        # Community Note评分
        community_entry = base_info.copy()
        community_entry['note_type'] = 'Community'
        community_entry['helpfulness'] = row['community_note_helpfulness']
        community_entry['source_quality'] = row['community_note_source_quality']
        community_entry['clarity'] = row['community_note_clarity']
        community_entry['coverage'] = row['community_note_coverage']
        community_entry['context'] = row['community_note_context']
        community_entry['impartiality'] = row['community_note_impartiality']
        community_entry['winner'] = 1 if row['comparison'] == 'community_note' else 0
        feature_data.append(community_entry)
        
        # LLM Note评分
        llm_entry = base_info.copy()
        llm_entry['note_type'] = 'LLMnote'
        llm_entry['helpfulness'] = row['llm_note_helpfulness']
        llm_entry['source_quality'] = row['llm_note_source_quality']
        llm_entry['clarity'] = row['llm_note_clarity']
        llm_entry['coverage'] = row['llm_note_coverage']
        llm_entry['context'] = row['llm_note_context']
        llm_entry['impartiality'] = row['llm_note_impartiality']
        llm_entry['winner'] = 1 if row['comparison'] == 'llm_note' else 0
        feature_data.append(llm_entry)
    
    # 转换为DataFrame
    feature_df = pd.DataFrame(feature_data)
    
    # 数据清理和验证
    print(f"\n提取完成: {len(feature_df)}条评分记录")
    
    # 检查缺失值
    missing_counts = {}
    for col in ['source_quality', 'clarity', 'coverage', 'context', 'impartiality']:
        missing = feature_df[col].isna().sum()
        if missing > 0:
            missing_counts[col] = missing
    
    if missing_counts:
        print("\n发现缺失值:")
        for col, count in missing_counts.items():
            print(f"- {feature_dimensions[col]}: {count}个缺失")
    
    # 转换数值型数据
    for col in ['source_quality', 'clarity', 'coverage', 'context', 'impartiality']:
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
    
    return feature_df

def calculate_feature_statistics(feature_df):
    """
    计算各个特征维度的统计信息
    """
    
    print("\n=== 特征评分统计分析 ===")
    
    # 按笔记类型分组统计
    feature_cols = ['source_quality', 'clarity', 'coverage', 'context', 'impartiality']
    
    # 计算每个维度的平均分
    stats_by_type = feature_df.groupby('note_type')[feature_cols].agg(['mean', 'std', 'count'])
    
    print("\n各维度平均评分 (1-5分制):")
    print("-" * 80)
    
    feature_names = {
        'source_quality': '来源质量',
        'clarity': '清晰度',
        'coverage': '覆盖范围',
        'context': '上下文',
        'impartiality': '中立性'
    }
    
    for feature in feature_cols:
        print(f"\n{feature_names[feature]}:")
        for note_type in ['Community', 'LLMnote']:
            mean = stats_by_type.loc[note_type, (feature, 'mean')]
            std = stats_by_type.loc[note_type, (feature, 'std')]
            count = stats_by_type.loc[note_type, (feature, 'count')]
            print(f"  {note_type}: {mean:.3f} ± {std:.3f} (n={count:.0f})")
        
        # 计算差异
        diff = stats_by_type.loc['LLMnote', (feature, 'mean')] - stats_by_type.loc['Community', (feature, 'mean')]
        print(f"  差异 (LLM - Community): {diff:+.3f}")
    
    # 计算总体评分（五个维度的平均值）
    feature_df['overall_score'] = feature_df[feature_cols].mean(axis=1)
    
    overall_stats = feature_df.groupby('note_type')['overall_score'].agg(['mean', 'std', 'count'])
    print(f"\n总体评分 (五个维度平均):")
    for note_type in ['Community', 'LLMnote']:
        mean = overall_stats.loc[note_type, 'mean']
        std = overall_stats.loc[note_type, 'std']
        print(f"  {note_type}: {mean:.3f} ± {std:.3f}")
    
    # 返回统计结果
    return stats_by_type, overall_stats

if __name__ == "__main__":
    # 文件路径
    input_file = "../consolidated_results.csv"
    
    print("开始提取笔记特征评分数据...")
    
    # 提取数据
    feature_df = extract_feature_ratings(input_file)
    
    # 计算统计信息
    stats_by_type, overall_stats = calculate_feature_statistics(feature_df)
    
    # 保存数据
    feature_df.to_csv("feature_ratings_wide.csv", index=False, encoding='utf-8')
    print(f"\n宽格式数据已保存到: feature_ratings_wide.csv")
    
    print("\n特征评分数据提取完成！")