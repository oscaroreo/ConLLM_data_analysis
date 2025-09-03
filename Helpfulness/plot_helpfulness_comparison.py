import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_helpfulness_comparison_plot(csv_file, output_file=None):
    """
    创建类似参考图的helpfulness对比图表
    
    Args:
        csv_file: helpfulness_extracted.csv文件路径
        output_file: 输出图片文件路径（可选）
    """
    
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 计算每种笔记类型的helpfulness分布
    note_type_counts = df.groupby(['note_type', 'helpfulness_raw']).size().unstack(fill_value=0)
    
    # 计算百分比
    note_type_percentages = note_type_counts.div(note_type_counts.sum(axis=1), axis=0) * 100
    
    # 重新排序列，确保顺序为：not helpful, somewhat helpful, helpful
    column_order = ['not helpful', 'somewhat helpful', 'helpful']
    note_type_percentages = note_type_percentages.reindex(columns=column_order, fill_value=0)
    
    print("Helpfulness分布百分比:")
    print(note_type_percentages.round(1))
    
    # 设置图表样式
    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 定义颜色：红色系为负面，橙色为中性，蓝色系为正面
    colors = ['#d62728', '#ff7f0e', '#1f77b4']  # 红色(not helpful), 橙色(somewhat helpful), 蓝色(helpful)
    
    # 重新组织数据：左边显示not helpful，中间显示somewhat helpful，右边显示helpful
    
    # 准备数据
    llm_data = note_type_percentages.loc['LLMnote']
    community_data = note_type_percentages.loc['Community']
    
    # 创建水平条形图的数据
    categories = ['LLM note', 'Community note']
    
    # 获取原始数据
    not_helpful_data = [llm_data['not helpful'], community_data['not helpful']]
    somewhat_helpful_data = [llm_data['somewhat helpful'], community_data['somewhat helpful']]
    helpful_data = [llm_data['helpful'], community_data['helpful']]
    
    # 重新计算位置：
    # somewhat helpful 居中显示（左右各一半）
    somewhat_left = [-x/2 for x in somewhat_helpful_data]  # 左半部分
    somewhat_right = [x/2 for x in somewhat_helpful_data]  # 右半部分
    
    # not helpful 从 somewhat helpful 的左边界开始向左延伸
    not_helpful_left = [-x for x in not_helpful_data]  # not helpful的宽度
    not_helpful_start = somewhat_left  # 起始位置是somewhat helpful的左边界
    
    # helpful 从 somewhat helpful 的右边界开始向右延伸
    helpful_right = helpful_data  # helpful的宽度
    helpful_start = somewhat_right  # 起始位置是somewhat helpful的右边界
    
    # 创建堆叠的水平条形图
    y_pos = np.arange(len(categories))
    
    # 绘制not helpful（从somewhat helpful的左边界开始向左延伸）
    ax.barh(y_pos, not_helpful_left, left=not_helpful_start, height=0.6, 
            color=colors[0], label='Not helpful', alpha=0.9)
    
    # 绘制中间左半部分的条形图（somewhat helpful左半）
    ax.barh(y_pos, somewhat_left, height=0.6, color=colors[1], alpha=0.9)
    
    # 绘制中间右半部分的条形图（somewhat helpful右半）
    ax.barh(y_pos, somewhat_right, height=0.6, color=colors[1], 
            label='Somewhat helpful', alpha=0.9)
    
    # 绘制helpful（从somewhat helpful的右边界开始向右延伸）
    ax.barh(y_pos, helpful_right, left=helpful_start, height=0.6, 
            color=colors[2], label='Helpful', alpha=0.9)
    
    # 设置标签和标题
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel('% of responses')
    # ax.set_title('Is this note helpful?', fontsize=14, fontweight='bold', pad=20)
    
    # 设置x轴范围和刻度
    ax.set_xlim(-60, 120)  # 扩展右边界为图例留出空间
    ax.set_xticks(range(-50, 101, 25))
    ax.set_xticklabels([str(abs(x)) for x in range(-50, 101, 25)])
    
    # 添加垂直分割线
    ax.axvline(x=0, color='black', linewidth=1.5)
    
    # 添加图例（确保顺序正确）
    handles, labels = ax.get_legend_handles_labels()
    # 重新排序图例，确保顺序为：Not helpful, Somewhat helpful, Helpful
    legend_order = [0, 1, 2]  # 对应绘制的顺序
    ax.legend([handles[i] for i in legend_order], [labels[i] for i in legend_order], 
             loc='upper left', bbox_to_anchor=(0.72, 0.99), frameon=True)
    
    # 添加网格
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)
    
    # 在条形图上添加百分比标签
    for i, y in enumerate(y_pos):
        # Not helpful标签（左边红色区域）
        not_helpful_value = not_helpful_data[i]
        if not_helpful_value > 0:  # 显示所有非零值的标签
            # 标签位置在not helpful条形图的中心
            center_x = not_helpful_start[i] + not_helpful_left[i]/2
            
            # 如果区域太小（小于8%），使用黑色文字
            if not_helpful_value < 8:
                ax.text(center_x, y, f'{not_helpful_value:.1f}%', 
                       ha='center', va='center', fontweight='bold', color='black', 
                       fontsize=14)
            else:
                # 区域足够大时使用白色文字
                ax.text(center_x, y, f'{not_helpful_value:.1f}%', 
                       ha='center', va='center', fontweight='bold', color='white', fontsize=14)
        
        # Somewhat helpful标签（居中显示在0点，用白色背景黑色文字）
        if somewhat_helpful_data[i] > 10:  # 只有当值足够大时才显示标签
            ax.text(0, y, f'{somewhat_helpful_data[i]:.1f}%', 
                   ha='center', va='center', fontweight='bold', color='black', 
                   fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='none'))
        
        # Helpful标签（右边蓝色区域，用白色文字）
        if helpful_data[i] > 5:
            center_x = helpful_start[i] + helpful_right[i]/2
            ax.text(center_x, y, f'{helpful_data[i]:.1f}%', 
                   ha='center', va='center', fontweight='bold', color='white', fontsize=14)
    
    # 保存或显示图表
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存到: {output_file}")
    
    plt.show()
    
    return fig, ax

def print_summary_statistics(csv_file):
    """
    打印详细的统计信息
    """
    df = pd.read_csv(csv_file)
    
    print("=== 详细统计信息 ===")
    print(f"总记录数: {len(df)}")
    print(f"用户数: {df['user_id'].nunique()}")
    print(f"推文数: {df['tweet_id'].nunique()}")
    
    print("\n按笔记类型分组的helpfulness分布:")
    for note_type in ['LLMnote', 'Community']:
        subset = df[df['note_type'] == note_type]
        print(f"\n{note_type}:")
        counts = subset['helpfulness_raw'].value_counts()
        percentages = subset['helpfulness_raw'].value_counts(normalize=True) * 100
        
        for category in ['helpful', 'somewhat helpful', 'not helpful']:
            count = counts.get(category, 0)
            pct = percentages.get(category, 0)
            print(f"  {category}: {count} ({pct:.1f}%)")
    
    print("\n按用户统计的平均helpfulness分数:")
    user_avg = df.groupby(['user_id', 'note_type'])['helpfulness_score'].mean().unstack()
    print(user_avg.round(3))

if __name__ == "__main__":
    import os
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 要处理的模型列表
    models = ['claude', 'gpt4o', 'grok', 'qwen']
    
    for model in models:
        # 设置输入输出路径（使用绝对路径）
        input_file = os.path.join(script_dir, f"{model}_helpfulness", f"helpfulness_extracted_829_{model}.csv")
        output_file = os.path.join(script_dir, f"{model}_helpfulness", f"helpfulness_comparison_chart_{model}.png")
        
        print(f"\n{'='*50}")
        print(f"处理 {model.upper()} 模型数据")
        print(f"{'='*50}")
        
        try:
            # 打印统计信息
            print_summary_statistics(input_file)
            
            print("\n" + "="*50)
            print(f"创建 {model} helpfulness对比图表...")
            
            # 创建图表
            create_helpfulness_comparison_plot(input_file, output_file)
        except FileNotFoundError:
            print(f"错误: 找不到文件 {input_file}")
            print(f"请先运行 extract_helpfulness.py 生成 {model} 的数据")
        except Exception as e:
            print(f"处理 {model} 过程中出现错误: {str(e)}")