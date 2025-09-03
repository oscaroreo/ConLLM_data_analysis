import pandas as pd

def extract_helpfulness_data(input_csv_path, output_csv_path):
    """
    从consolidated_data.csv中提取helpfulness相关数据
    
    提取字段:
    - user_id: 用户ID（用于配对统计）
    - tweet_id: 推文/测试条目的唯一ID  
    - note_type: 两类笔记（LLMnote / Community）
    - helpfulness_raw: 三档评分（原始字符串）
    - helpfulness_score: 数值化评分（0, 0.5, 1）
    """
    
    # helpfulness评分映射
    helpfulness_mapping = {
        "not helpful": 0,
        "somewhat helpful": 0.5,
        "helpful": 1
    }
    
    # 读取数据
    print(f"正在读取文件: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    print(f"原始数据行数: {len(df)}")
    print(f"原始数据列名: {list(df.columns)}")
    
    # 准备提取的数据
    extracted_data = []
    
    # 假设数据结构包含以下字段（请根据实际情况调整）:
    # participant_name 或 user_id
    # post_index 作为 tweet_id
    # community_note_helpfulness 和 llm_note_helpfulness
    
    for index, row in df.iterrows():
        # 获取用户ID（优先使用user_id，如果没有则使用participant_name或session_id）
        if 'user_id' in df.columns:
            user_id = row['user_id']
        elif 'participant_name' in df.columns:
            user_id = row['participant_name']
        elif 'session_id' in df.columns:
            user_id = row['session_id']
        else:
            user_id = f"user_{index}"
        
        # 获取推文ID
        if 'tweet_id' in df.columns:
            tweet_id = row['tweet_id']
        elif 'post_index' in df.columns:
            tweet_id = row['post_index']
        else:
            tweet_id = f"tweet_{index}"
        
        # 提取Community Note的helpfulness
        if 'community_note_helpfulness' in df.columns:
            community_helpfulness = row['community_note_helpfulness']
            extracted_data.append({
                'user_id': user_id,
                'tweet_id': tweet_id,
                'note_type': 'Community',
                'helpfulness_raw': community_helpfulness,
                'helpfulness_score': helpfulness_mapping.get(community_helpfulness, None)
            })
        
        # 提取LLM Note的helpfulness
        if 'llm_note_helpfulness' in df.columns:
            llm_helpfulness = row['llm_note_helpfulness']
            extracted_data.append({
                'user_id': user_id,
                'tweet_id': tweet_id,
                'note_type': 'LLMnote',
                'helpfulness_raw': llm_helpfulness,
                'helpfulness_score': helpfulness_mapping.get(llm_helpfulness, None)
            })
    
    # 创建新的DataFrame
    result_df = pd.DataFrame(extracted_data)
    
    # 保存结果
    result_df.to_csv(output_csv_path, index=False)
    
    # 输出统计信息
    print(f"\n提取完成!")
    print(f"提取的记录数: {len(result_df)}")
    print(f"唯一用户数: {result_df['user_id'].nunique()}")
    print(f"唯一推文数: {result_df['tweet_id'].nunique()}")
    
    print(f"\nHelpfulness评分分布:")
    print(result_df['helpfulness_raw'].value_counts())
    
    print(f"\nHelpfulness数值分布:")
    print(result_df['helpfulness_score'].value_counts())
    
    print(f"\n笔记类型分布:")
    print(result_df['note_type'].value_counts())
    
    print(f"\n前5行数据:")
    print(result_df.head())
    
    print(f"\n数据已保存到: {output_csv_path}")
    
    return result_df

if __name__ == "__main__":
    import os
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # 要处理的模型列表
    models = ['claude', 'gpt4o', 'grok', 'qwen']
    
    for model in models:
        # 设置输入输出路径（使用绝对路径）
        input_file = os.path.join(parent_dir, f"{model}_results_829.csv")
        output_dir = os.path.join(script_dir, f"{model}_helpfulness")
        output_file = os.path.join(output_dir, f"helpfulness_extracted_829_{model}.csv")
        
        print(f"\n{'='*50}")
        print(f"处理 {model.upper()} 模型数据")
        print(f"{'='*50}")
        
        try:
            extract_helpfulness_data(input_file, output_file)
        except FileNotFoundError:
            print(f"错误: 找不到文件 {input_file}")
            print("请确认文件路径是否正确")
        except Exception as e:
            print(f"处理 {model} 过程中出现错误: {str(e)}")