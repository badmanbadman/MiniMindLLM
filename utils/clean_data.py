import re
import os
import json
from pathlib import Path

def read_file_with_encoding(file_path):
    """
    尝试用多种编码读取文件
    """
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"成功用 {encoding} 编码读取文件")
            return content, encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"读取文件时出错: {e}")
            continue
    
    return None, None

def basic_text_cleaning(text):
    """基础文本清洗"""
    text = re.sub(r'[^\w\s，。！？；：""\'\'.,!?;:]', '', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def quality_filter(text, min_length=50, max_length=500000):  # 增加max_length到500,000
    """质量过滤函数"""
    if len(text) < min_length:
        print(f"  文本过短: {len(text)} < {min_length}")
        return False
    if len(text) > max_length:
        print(f"  文本过长: {len(text)} > {max_length}")
        return False
    
    # 中文比例检查
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    chinese_ratio = chinese_chars / len(text) if len(text) > 0 else 0
    if chinese_ratio < 0.3:
        print(f"  中文比例不足: {chinese_ratio:.2f} < 0.3")
        return False
    
    # 重复内容检查
    lines = text.split('\n')
    unique_ratio = len(set(lines)) / len(lines) if len(lines) > 0 else 1
    if unique_ratio < 0.8:
        print(f"  重复行过多: {unique_ratio:.2f} < 0.8")
        return False
    
    return True

def format_for_lora(texts, output_file, chunk_size=512):
    """将文本分块并格式化为训练数据"""
    formatted_data = []
    
    for i, text in enumerate(texts):
        print(f"处理第 {i+1}/{len(texts)} 个文本，长度: {len(text)}")
        
        # 按句子分割
        sentences = re.split(r'[。！？!?]', text)
        current_chunk = ''
        
        for j, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 如果当前句子本身就超过块大小，需要特殊处理
            if len(sentence) > chunk_size:
                # 对长句子进行进一步分割
                words = list(sentence)
                temp_chunk = ''
                for word in words:
                    if len(temp_chunk) + len(word) <= chunk_size:
                        temp_chunk += word
                    else:
                        if temp_chunk:
                            formatted_data.append({'text': temp_chunk.strip()})
                        temp_chunk = word
                if temp_chunk:
                    current_chunk += temp_chunk
            elif len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += sentence + '。'
            else:
                if current_chunk and len(current_chunk) > 20:
                    formatted_data.append({'text': current_chunk.strip()})
                current_chunk = sentence + '。'
        
        # 处理最后一个块
        if current_chunk and len(current_chunk) > 20:
            formatted_data.append({'text': current_chunk.strip()})
    
    print(f"生成了 {len(formatted_data)} 个训练数据块")
    
    # 保存为JSONL格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_dataset(raw_dir, output_dir):
    """完整的数据处理流程"""
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 支持多种文本文件扩展名
    supported_extensions = ['*.txt', '*.TXT', '*.text', '*.md']
    txt_files = []
    for ext in supported_extensions:
        txt_files.extend(raw_path.glob(ext))
    
    # 去重，避免重复处理同一文件
    txt_files = list(set(txt_files))
    
    print(f"找到 {len(txt_files)} 个文件: {[f.name for f in txt_files]}")
    
    all_cleaned_texts = []
    
    for txt_file in txt_files:
        print(f"\n处理文件: {txt_file.name}")
        try:
            # 尝试多种编码读取文件
            original_text, used_encoding = read_file_with_encoding(txt_file)
            
            if original_text is None:
                print(f"无法读取文件 {txt_file.name}，跳过")
                continue
                
            print(f"原始文件长度: {len(original_text)} 字符")
            print(f"使用的编码: {used_encoding}")
            
            # 清洗文本
            cleaned_text = basic_text_cleaning(original_text)
            print(f"清洗后长度: {len(cleaned_text)} 字符")
            
            # 质量过滤 - 现在使用更大的max_length
            if quality_filter(cleaned_text, min_length=50, max_length=500000):
                all_cleaned_texts.append(cleaned_text)
                print(f"✓ 文件通过质量过滤")
            else:
                print(f"✗ 文件未通过质量过滤")
                
        except Exception as e:
            print(f"处理文件 {txt_file} 时出错: {e}")
    
    if all_cleaned_texts:
        train_file = output_path / "lora_training_data.jsonl"
        print(f"\n开始格式化 {len(all_cleaned_texts)} 个文本为训练数据...")
        format_for_lora(all_cleaned_texts, train_file)
        print(f"训练数据已保存到: {train_file}")
        
        # 验证生成的文件
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"生成的JSONL文件包含 {len(lines)} 行数据")
        except Exception as e:
            print(f"验证生成文件时出错: {e}")
    else:
        print("没有通过质量过滤的文本，跳过训练数据生成")
    
    stats = {
        "total_files": len(txt_files),
        "cleaned_files": len(all_cleaned_texts),
        "total_chars": sum(len(text) for text in all_cleaned_texts),
        "avg_length": sum(len(text) for text in all_cleaned_texts) / len(all_cleaned_texts) if all_cleaned_texts else 0
    }
    
    print(f"\n处理完成！统计信息: {stats}")
    return stats

if __name__ == "__main__":
    raw_dir = "../dataset/raw_texts"
    output_dir = "../dataset/processed_data"
    
    # 确保目录存在
    Path(raw_dir).mkdir(exist_ok=True)
    Path(output_dir).mkdir(exist_ok=True)
    
    stats = process_dataset(raw_dir, output_dir)
    
    with open(Path(output_dir) / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)