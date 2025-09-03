import json
import csv
import os
from datetime import datetime
from pathlib import Path

def consolidate_json_to_csv(json_dir, output_csv, max_files=None):
    """
    将多个JSON文件合并为一个CSV文件
    
    Args:
        json_dir: 包含JSON文件的目录路径
        output_csv: 输出CSV文件的路径
        max_files: 每个文件夹处理的最大文件数，None表示处理所有文件
    """
    
    # 准备CSV的列名
    csv_columns = [
        'participant_name',
        'session_id',
        'start_time',
        'completion_time',
        'total_items_assigned',
        'post_index',
        'note_mapping',
        'community_note_helpfulness',
        'community_note_source_quality',
        'community_note_clarity',
        'community_note_coverage',
        'community_note_context',
        'community_note_impartiality',
        'llm_note_helpfulness',
        'llm_note_source_quality',
        'llm_note_clarity',
        'llm_note_coverage',
        'llm_note_context',
        'llm_note_impartiality',
        'comparison',
        'response_timestamp'
    ]
    
    # 收集所有数据
    all_rows = []
    
    # 遍历目录中的所有JSON文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # 如果设置了最大文件数限制，只处理前max_files个文件
    if max_files is not None and max_files > 0:
        json_files = json_files[:max_files]
        print(f"找到 {len([f for f in os.listdir(json_dir) if f.endswith('.json')])} 个JSON文件，限制处理 {len(json_files)} 个文件")
    else:
        print(f"找到 {len(json_files)} 个JSON文件")
    
    for filename in json_files:
        file_path = os.path.join(json_dir, filename)
        print(f"处理文件: {filename}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取基本信息
            participant_name = data.get('participantName', '')
            session_id = data.get('sessionId', '')
            start_time = data.get('startTime', '')
            completion_time = data.get('completionTime', '')
            total_items = data.get('totalItemsAssigned', '')
            
            # 遍历每个用户响应
            user_responses = data.get('userResponses', {})
            for post_id, response in user_responses.items():
                row = {
                    'participant_name': participant_name,
                    'session_id': session_id,
                    'start_time': start_time,
                    'completion_time': completion_time,
                    'total_items_assigned': total_items,
                    'post_index': response.get('postIndex', ''),
                    'note_mapping': response.get('noteMapping', ''),
                    'community_note_helpfulness': response.get('communityNote', {}).get('helpfulness', ''),
                    'community_note_source_quality': response.get('communityNote', {}).get('details', {}).get('source_quality', ''),
                    'community_note_clarity': response.get('communityNote', {}).get('details', {}).get('clarity', ''),
                    'community_note_coverage': response.get('communityNote', {}).get('details', {}).get('coverage', ''),
                    'community_note_context': response.get('communityNote', {}).get('details', {}).get('context', ''),
                    'community_note_impartiality': response.get('communityNote', {}).get('details', {}).get('impartiality', ''),
                    'llm_note_helpfulness': response.get('llmNote', {}).get('helpfulness', ''),
                    'llm_note_source_quality': response.get('llmNote', {}).get('details', {}).get('source_quality', ''),
                    'llm_note_clarity': response.get('llmNote', {}).get('details', {}).get('clarity', ''),
                    'llm_note_coverage': response.get('llmNote', {}).get('details', {}).get('coverage', ''),
                    'llm_note_context': response.get('llmNote', {}).get('details', {}).get('context', ''),
                    'llm_note_impartiality': response.get('llmNote', {}).get('details', {}).get('impartiality', ''),
                    'comparison': response.get('comparison', ''),
                    'response_timestamp': response.get('timestamp', '')
                }
                all_rows.append(row)
        
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            continue
    
    # 写入CSV文件
    print(f"\n总共处理了 {len(all_rows)} 条记录")
    print(f"写入CSV文件: {output_csv}")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print("完成！")

if __name__ == "__main__":
    # 设置基础路径
    base_path = "/Users/shidaidemac/Documents/CHI_25/Data_Analysis_829"
    result_dir = os.path.join(base_path, "829_result")
    
    # 设置每个文件夹处理的最大文件数
    # 设置为None处理所有文件，或设置具体数字限制文件数
    MAX_FILES_PER_FOLDER = 36  # 可以根据需要修改这个值
    
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, f))]
    
    print(f"找到 {len(subfolders)} 个子文件夹: {subfolders}")
    print(f"每个文件夹最多处理 {MAX_FILES_PER_FOLDER} 个文件" if MAX_FILES_PER_FOLDER else "处理每个文件夹中的所有文件")
    
    # 为每个子文件夹生成对应的CSV文件
    for folder in subfolders:
        json_directory = os.path.join(result_dir, folder)
        
        # 从文件夹名称提取模型名称
        # v0_claude -> claude, v1_gpt4o -> gpt4o, etc.
        model_name = folder.split('_')[1] if '_' in folder else folder
        output_file = os.path.join(base_path, f"{model_name}_results_829.csv")
        
        print(f"\n处理文件夹: {folder}")
        print(f"输出文件: {output_file}")
        
        # 执行转换
        consolidate_json_to_csv(json_directory, output_file, max_files=MAX_FILES_PER_FOLDER)