import json
from torch.utils.data import Dataset
import torch
class NovelLoRADataset(Dataset):
    """基础小说LoRA训练数据集"""
    def __init__(self, data_path, tokenizer, max_length=512, instruction_template=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template or "请继续写下面的小说内容：\n{}"
        self.samples = self.load_novel_data(data_path)
        
        # 确保tokenizer有必要的特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def load_novel_data(self, path):
        """加载小说数据"""
        samples = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '').strip()
                    
                    if len(text) < 50:  # 过滤太短的文本
                        continue
                    
                    samples.append({'text': text})
                    
                except Exception as e:
                    print(f"解析数据时出错: {e}")
                    continue
        
        print(f"加载了 {len(samples)} 个小说样本")
        return samples
    
    def create_instruction_prompt(self, text):
        """为小说创建指令提示"""
        # 方法1: 续写任务 - 取前半部分作为输入，后半部分作为输出
        if len(text) > 100:
            split_point = len(text) // 2
            input_text = text[:split_point]
            output_text = text[split_point:]
            
            # 构建对话格式
            conversation = [
                {"role": "user", "content": self.instruction_template.format(input_text)},
                {"role": "assistant", "content": output_text}
            ]
        else:
            # 方法2: 对于短文本，直接作为回复
            conversation = [
                {"role": "user", "content": "请写一段小说内容："},
                {"role": "assistant", "content": text}
            ]
        
        return conversation
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # 创建指令提示
        conversation = self.create_instruction_prompt(sample['text'])
        
        # 应用chat template
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        # 生成loss mask (只对assistant回复计算loss)
        loss_mask = self._generate_response_mask(input_ids, conversation)
        
        X = input_ids[:-1]
        Y = input_ids[1:]
        loss_mask = loss_mask[1:]
        
        return X, Y, loss_mask
    
    def _generate_response_mask(self, input_ids, conversation):
        """生成只对assistant回复计算损失的掩码"""
        # 构建完整的对话文本
        full_text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        
        # 找到assistant回复的开始
        # 这取决于您的chat template格式
        assistant_markers = ["assistant", "Assistant", "ASSISTANT"]
        assistant_start = -1
        
        for marker in assistant_markers:
            pos = full_text.find(f"{marker}\n")
            if pos != -1:
                assistant_start = pos + len(f"{marker}\n")
                break
        
        if assistant_start == -1:
            # 如果找不到明确的assistant标记，对后半部分计算loss
            assistant_start = len(full_text) // 2
        
        # 将字符位置转换为token位置
        tokens_before_assistant = self.tokenizer.encode(
            full_text[:assistant_start], add_special_tokens=False
        )
        assistant_start_token = len(tokens_before_assistant)
        
        # 创建掩码
        loss_mask = [0] * len(input_ids)
        for i in range(assistant_start_token, len(input_ids)):
            if input_ids[i] != self.tokenizer.pad_token_id:
                loss_mask[i] = 1
        
        return torch.tensor(loss_mask, dtype=torch.long)