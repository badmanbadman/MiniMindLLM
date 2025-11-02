import re
import json
import os
from pathlib import Path
import random

def check_environment():
    """检查环境和路径"""
    print("=== 环境检查 ===")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python版本: {os.sys.version}")
    print("================\n")

def split_sentences(text):
    """将文本分割成句子"""
    # 使用更全面的句子分割规则
    sentence_endings = r'[。！？!?\.\n]+\s*'
    sentences = re.split(sentence_endings, text)
    
    # 过滤空句子和空白句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    print(f"分割出 {len(sentences)} 个句子")
    return sentences

def create_sample_conversations():
    """创建示例对话数据"""
    samples = [
        {
            "user": "鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。",
            "assistant": "好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？"
        },
        {
            "user": "打开闹钟功能，定一个明天早上七点的闹钟。",
            "assistant": "好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。"
        },
        {
            "user": "为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。",
            "assistant": "一位孤独的老人坐在公园长椅上凝视远方。"
        },
        {
            "user": "非常感谢你的回答。请告诉我，这些数据是关于什么主题的？",
            "assistant": "这些数据是关于不同年龄段的男女人口比例分布的。"
        },
        {
            "user": "帮我想一个有趣的标题。",
            "assistant": "这个挺有趣的：\"如何成为一名成功的魔术师\" 调皮的标题往往会吸引读者的注意力。"
        },
        {
            "user": "回答一个问题，地球的半径是多少？",
            "assistant": "地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。"
        },
        {
            "user": "识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本：\"今天是我的生日！\"",
            "assistant": "这个文本的语气是喜悦。"
        }
    ]
    return samples

def process_single_txt_file(input_file, output_file):
    """处理单个txt文件"""
    print(f"\n=== 处理文件: {input_file} ===")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在 - {input_file}")
        return 0
    
    # 尝试不同的编码读取文件
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
    content = None
    used_encoding = None
    
    for encoding in encodings:
        try:
            with open(input_file, 'r', encoding=encoding) as f:
                content = f.read()
            used_encoding = encoding
            print(f"成功使用编码: {encoding}")
            break
        except UnicodeDecodeError as e:
            print(f"编码 {encoding} 失败: {e}")
            continue
        except Exception as e:
            print(f"读取文件时出错 ({encoding}): {e}")
            continue
    
    if content is None:
        print(f"无法读取文件: {input_file}，所有编码都失败了")
        return 0
    
    print(f"文件长度: {len(content)} 字符")
    
    # 基础文本清洗
    content = re.sub(r'\s+', ' ', content)  # 合并空白字符
    content = content.strip()
    
    if len(content) < 50:
        print(f"文件内容过短，跳过: {len(content)} 字符")
        return 0
    
    # 分割句子
    sentences = split_sentences(content)
    
    if len(sentences) < 2:
        print("句子数量不足，跳过")
        return 0
    
    # 创建对话数据
    sample_conversations = create_sample_conversations()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 生成多个样本
    num_samples = min(10, len(sentences) // 2)  # 根据句子数量决定生成多少样本
    print(f"将生成 {num_samples} 个样本")
    
    created_count = 0
    with open(output_file, 'a', encoding='utf-8') as f:
        for i in range(num_samples):
            # 随机选择3-6个对话
            num_conv = random.randint(3, 6)
            selected_conv = random.sample(sample_conversations, num_conv)
            
            # 构建格式化的文本
            formatted_parts = []
            for conv in selected_conv:
                formatted_parts.append(
                    f"<|im_start|>{conv['user']}<|im_end|> "
                    f"<|im_start|>{conv['assistant']}<|im_end|>"
                )
            
            formatted_text = " ".join(formatted_parts)
            
            # 写入文件
            json_line = json.dumps({"text": formatted_text}, ensure_ascii=False)
            f.write(json_line + '\n')
            created_count += 1
    
    print(f"成功生成 {created_count} 个样本到 {output_file}")
    return created_count

def process_directory(input_dir, output_file):
    """处理目录中的所有txt文件"""
    print("=== 开始处理目录 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_file}")
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在 - {input_dir}")
        return
    
    # 查找所有txt文件
    txt_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    if not txt_files:
        print("没有找到txt文件，创建示例数据...")
        create_sample_data(output_file, 50)
        return
    
    # 清空或创建输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('')  # 清空文件
    
    total_processed = 0
    for txt_file in txt_files:
        processed = process_single_txt_file(txt_file, output_file)
        total_processed += processed
    
    print(f"\n=== 处理完成 ===")
    print(f"总共处理了 {total_processed} 个样本")
    print(f"输出文件: {output_file}")
    
    # 验证输出文件
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"输出文件包含 {len(lines)} 行数据")
    else:
        print("错误: 输出文件未创建")

def create_sample_data(output_file, num_samples=50):
    """创建示例数据"""
    print(f"=== 创建示例数据 ===")
    print(f"将创建 {num_samples} 个样本到 {output_file}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    sample_conversations = create_sample_conversations()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # 随机选择3-6个对话
            num_conv = random.randint(3, 6)
            selected_conv = random.sample(sample_conversations, num_conv)
            
            formatted_parts = []
            for conv in selected_conv:
                formatted_parts.append(
                    f"<|im_start|>{conv['user']}<|im_end|> "
                    f"<|im_start|>{conv['assistant']}<|im_end|>"
                )
            
            formatted_text = " ".join(formatted_parts)
            json_line = json.dumps({"text": formatted_text}, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"成功创建 {num_samples} 个示例样本到 {output_file}")
    
    # 显示前3行作为示例
    print("\n前3行示例:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 3:
                print(f"行 {i+1}: {line.strip()}")
            else:
                break

def main():
    """主函数"""
    check_environment()
    
    # 配置路径 - 根据您的实际情况修改
    input_dir = "../dataset/raw_texts"  # 包含txt文件的目录
    output_file = "./processed_data/pretrain_data.jsonl"  # 输出文件
    
    # 如果输入目录不存在，创建示例数据
    if not os.path.exists(input_dir):
        print(f"输入目录 {input_dir} 不存在，创建示例数据...")
        create_sample_data(output_file, 50)
    else:
        # 处理真实数据
        process_directory(input_dir, output_file)
    
    print("\n=== 脚本执行完成 ===")

if __name__ == "__main__":
    main() 