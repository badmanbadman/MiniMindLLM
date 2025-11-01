import torch
import torch.distributed as dist

# class DistributedSampler:
#     def __init__(self, dataset, num_replicas=None, rank=None):
#         self.dataset = dataset
#         self.num_replicas = dist.get_world_size()  # 总GPU数
#         self.rank = rank                # 当前GPU编号
#         self.epoch = 0
        
#     def __iter__(self):
#         # 1. 根据epoch设置随机种子，确保可重现性
#         g = torch.Generator()
#         g.manual_seed(self.epoch)
        
#         # 2. 生成所有数据的随机排列
#         indices = torch.randperm(len(self.dataset), generator=g).tolist()
        
#         # 3. 根据rank分配数据片段
#         indices = indices[self.rank::self.num_replicas]
#         return iter(indices)
    

from transformers import AutoTokenizer
import os
import json

def comprehensive_tokenizer_analysis(tokenizer_path):
    print("🔍 开始分词器分析...")
    
    # 1. 检查文件
    print("\n1. 📁 文件检查:")
    for file in os.listdir(tokenizer_path):
        file_path = os.path.join(tokenizer_path, file)
        size = os.path.getsize(file_path)
        print(f"   {file}: {size:,} bytes")
    
    # 2. 加载分词器
    print("\n2. 🚀 加载分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("   ✅ 加载成功")
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return
    
    # 3. 基本信息
    print("\n3. 📊 基本信息:")
    print(f"   - 类型: {type(tokenizer).__name__}")
    print(f"   - 词汇表大小: {tokenizer.vocab_size:,}")
    print(f"   - 模型最大长度: {tokenizer.model_max_length}")
    
    # 4. 特殊令牌
    print("\n4. 🎯 特殊令牌:")
    special_tokens = [
        ('bos_token', '开始令牌'),
        ('eos_token', '结束令牌'), 
        ('pad_token', '填充令牌'),
        ('unk_token', '未知令牌'),
        ('mask_token', '掩码令牌')
    ]
    
    for token_name, desc in special_tokens:
        token = getattr(tokenizer, token_name, None)
        token_id = getattr(tokenizer, token_name + '_id', None)
        status = "✅" if token else "❌"
        print(f"   - {desc}({token_name}): {status} {token} (ID: {token_id})")
    
    # 5. 测试分词
    print("\n5. 🧪 分词测试:")
    test_samples = [
        "Hello world!",
        "你好，世界！",
        "I love programming.",
        "这是一个测试句子。"
    ]
    
    for sample in test_samples:
        tokens = tokenizer.tokenize(sample)
        input_ids = tokenizer.encode(sample, add_special_tokens=False)
        print(f"   '{sample}'")
        print(f"     → Tokens: {tokens}")
        print(f"     → IDs: {input_ids}")
    
    # 6. 词汇表样本
    print("\n6. 📚 词汇表样本:")
    vocab = tokenizer.get_vocab()
    print(f"   - 总token数: {len(vocab):,}")
    
    # 前5个
    print("   - 前5个token:")
    for i, (token, token_id) in enumerate(list(vocab.items())[:5]):
        print(f"     {token_id}: '{token}'")
    
    # 最后5个  
    print("   - 最后5个token:")
    for i, (token, token_id) in enumerate(list(vocab.items())[-5:]):
        print(f"     {token_id}: '{token}'")

# 运行分析
comprehensive_tokenizer_analysis('../model/')

# 使用诊断
if __name__ == "__main__":
        comprehensive_tokenizer_analysis('../model/')
    