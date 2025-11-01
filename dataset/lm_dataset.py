"""
操作	                对应的方法	       作用
Python -> JSON 字符串	json.dumps()	  序列化，用于传输或展示
JSON 字符串 -> Python	json.loads()	  反序列化，用于解析和处理
Python -> JSON 文件	    json.dump()	      将数据序列化并直接保存到文件
JSON 文件 -> Python	    json.load()	      从文件读取并直接反序列化为数据

"""
import json
import random
"""
导入正则表达式模块
    搜索 文本中的特定模式
    匹配 符合规则的字符串
    替换 文本内容
    分割 字符串
"""
import re

import pandas as pd
import numpy as np
"""
Dataset
    一个抽象类，所有自定义的数据集都应该继承这个类，并重写__len__和__getitem__方法
    __len__ 返回数据集的大小
    __getItem__ 通过索引返回一个样本
DataLoader 数据加载器，
    提供对数据集的
    批量加载，
    多进程加载，
    打乱顺序
    自动分批和填充
    等功能
数据预处理封装在Dataset中，批量加载和并行处理有DataLoader自动处理
"""
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

"""
设置一个环境变量，禁用 tokenizers 库的并行处理功能
是什么：
    TOKENIZERS_PARALLELISM是Hugging Face tokenizers 库识别的环境变量
    这个库是transformer，datasets等流行NLP库的基础组件
为什么要设置这个：
    主要目的是避免警告和潜在冲突
使用 Hugging Face 生态系统时，经常会看到这样的警告：
The current process just got forked. Disabling parallelism to avoid deadlocks...
产生原因：
    多进程冲突：当Python使用多进程（如multiprocessing）时，tokenizers的并行处理可能与进程fork机制冲突
    死锁风险：某些情况下，tokenizers的内部线程与多进程结合可能导致程序卡死
    性能问题：在小型任务中，并行化的开销可能超过收益
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 预训练阶段
class PretrainDataset(Dataset):
    def __init__(self, data_path,tokenizer,max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self,path):
        samples = []
        with open(path,'r',encoding='utf-8') as f:
        
            # enumerate(f, 1): 遍历文件的每一行，同时记录行号（从1开始）
            for line_num,line in enumerate(f,1):
                # 调试用的
                if line_num > 100:
                    break
                if line_num < 2:
                    print(line)

                # line.strip():去除行首尾的空白字符（包括换行符）
                # json.loads():将JSON字符串解析为Python字典
                # 注意：数据集中，每一行应该是一个完整的JSON对象，
                data = json.loads(line.strip())
                samples.append(data)
        return samples
 
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        sample = self.samples[index]

        # 构建输入文本
        """
        str(sample[text]):确保输入是字符串
        max_length:序列最大长度
        padding='max_length'：填充到最大长度
        truncation=True: 超过最大长度时截断
        return_tensors='pt'：返回PyTorch张量
        """
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )


        # squeeze(): 去除批次维度，形状从[1,max_length]变为[max_length]
        # loss_mask: 创建损失掩码，标记哪些位置是真实文本（True），哪些是填充位置（False）
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        """
        自回归训练：
        X: 从第1个到倒数第2个token input_ids[:-1]
        Y: 从第2个到最后一个token input_ids[1:]
        loss_mask: 对应Y位置的掩码 loss_mask[1:]
        理解：第一个id对应的要预测的下一个词就是第二个id，第二个要预测的就是第三个词
        
        核心机制：自回归语言建模
        数据转换示例
        假设原始文本token序列为：[A, B, C, D, E, PAD, PAD]（PAD是填充token）
        input_ids = [A, B, C, D, E, PAD, PAD]        # 原始序列
        loss_mask = [1,  1,  1,  1,  1,   0,    0]   # 1表示真实token，0表示填充

        X = [A, B, C, D, E, PAD]     # 输入：从第一个到倒数第二个
        Y = [B, C, D, E, PAD, PAD]   # 目标：从第二个到最后一个
        loss_mask = [1, 1, 1, 1, 0, 0]  # 与Y对齐的掩码

        训练时的预测过程
        时间步 | 输入(X) | 预测目标(Y) | 是否计算损失
        ----------------------------------------
        t=0   | A      | B          | 是(loss_mask[0]=1)
        t=1   | B      | C          | 是(loss_mask[1]=1)  
        t=2   | C      | D          | 是(loss_mask[2]=1)
        t=3   | D      | E          | 是(loss_mask[3]=1)
        t=4   | E      | PAD        | 否(loss_mask[4]=0)
        t=5   | PAD    | PAD        | 否(loss_mask[5]=0)

        Loss Mask的作用
        # 如果没有loss mask，计算损失时会包括填充位置
        总损失 = 有效token损失 + 填充token损失（无意义）

        # 有loss mask后，只计算有效token的损失
        总损失 = 有效token损失
        """
        X = torch.tensor(input_ids[:-1], dtype=torch.long) # 输入序列：去掉最后一个token
        Y = torch.tensor(input_ids[1:], dtype=torch.long) # 目标序列：去掉第一个token 
        loss_mask = torch.tensor(loss_mask[1:],dtype=torch.long)  # 损失掩码：与Y对齐
        return X, Y, loss_mask

# 有监督微调阶段
class SFTDataset(Dataset):
    def __init__(self,jsonl_path,tokenizer,max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        """ bos_id序列的开始标记
        bos_id是Beginning Of Sequence的缩写，即序列的开始标记，在自然语言处理中，我们通常会在输入序列的前面加上一个特殊的token，表示序列的来时
        bos_token: 是tokenizer中用来表示序列开始的特殊token，例如在GPT系类模型中通常用<s>作为bos_token
        
        在对话模型中，我们通常会有多种角色，比如用户（user）和助手(assistant),在微调时候，我们只希望计算助手回答部分的损失，
        因此我们需要一个方法来表示助手回答开始，助手回答结束，在实际的数据处理中，_generate_loss_mask会使用bos_id来搜索输入序列中助手回答开始的位置然后计算损失掩码
        使得只有助手回答的部分（直到eos_token）的损失被计算
        """
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)
    
    def load_data(self,path):
        samples = []
        with open(path,'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                    # 调试用的
                if line_num > 100:
                    break
                if line_num < 2:
                    print(line)
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    def _create_chat_prompt(self,cs):
        """ 工具函数
        cs： conversation的缩写,对话的意思
        cs.copy(): 复制一份对话数据，避免修改原始数据
        tools 工具函数提取
        逻辑如下
            检测对话是否非空(cs)
            检查第一条消息角色是否为 "system"
            检查system消息中是否包含"functions"字段
            如果满足所有条件就提取functions作为tools，否则将tools设为None
        示例：
            # 有工具调用的对话
            cs = [
                {
                    "role": "system", 
                    "content": "你是助手",
                    "functions": [
                        {"name": "get_weather", "description": "获取天气"}
                    ]
                },
                {"role": "user", "content": "今天天气怎么样？"}
            ]
            # tools = [{"name": "get_weather", "description": "获取天气"}]

            # 无工具调用的对话  
            cs = [
                {"role": "user", "content": "你好"}
            ]
            # tools = None
        """
        messages = cs.copy()
        tools = cs[0]['functions'] if (cs and cs[0]['role'] == 'system' and cs[0].get('function')) else None
        """apply_chat_template参数注释
        messages: 对话消息列表，格式通常为：
            [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "用户输入"},
                {"role": "assistant", "content": "助手回复"}
            ]
        tokenize=False
            返回字符串  而不是token IDs
            如果设为True，会直接返回分词后的input_ids
        add_generation_prompt=False
            不添加生成提示，意味着模板不会再末尾添加引导模型生成的特殊标记
            如果设为True，通常会再末尾添加如<|assistant|>等标记
        tools=tools
            传递工具函数定义，让模板知道可用的工具
            则会个对支持函数调用的模型很重要
        """
        """不同模型的模板差异
        LLaMA风格模板
            <|system|>
            你是助手</s>
            <|user|>
            你好</s>
            <|assistant|>
            你好！有什么可以帮助你的吗？</s>
        ChatML风格模板
            <|im_start|>system
            你是助手<|im_end|>
            <|im_start|>user
            你好<|im_end|>
            <|im_start|>assistant
            你好！有什么可以帮助你的吗？<|im_end|>
        带有工具调用的模板
            # 当tools不为None时，可能生成：
            <|system|>
            你是助手，可用工具：get_weather
            </s>
            <|user|>
            今天天气怎么样？</s>
            <|assistant|>
        """
        """完整工作流程示例
        输入数据
            cs = [
                {"role": "system", "content": "你是一个有用的助手"},
                {"role": "user", "content": "请介绍一下AI"},
                {"role": "assistant", "content": "AI是人工智能..."}
            ]
        处理过程
            messages = cs.copy() - 复制数据
            tools = None - 无工具函数
            apply_chat_template - 应用模板格式化
        输出结果
            <|system|>
            你是一个有用的助手</s>
            <|user|>
            请介绍一下AI</s>
            <|assistant|>
            AI是人工智能...</s>
        SFT中的作用
            这个方法确保了：
            1、格式一致性：所有的对话都按照统一的模板格式化
            2、角色区分：清晰的标记不同的发言者
            3、工具支持：正确处理函数调用的场景
            4、模型兼容性：使用模型预训练时熟悉的格式
        最终输出的格式化的字符串会再__getitem__方法中被分词，然后用于训练，这种设计让数据预处理更加灵活，可以轻松去适配不同的模型对话格式要求

        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )
    
    def _generate_loss_mask(self,input_ids):
        # 初始化构建一个长度为input_ids的长度的全为0的掩码矩阵
        loss_mask = [0]*len(input_ids)
        i = 0
        """
        <|im_start|>system
        你是助手<|im_end|>
        <|im_start|>user
        你好<|im_end|>
        <|im_start|>assistant
        你好！有什么可以帮助你的吗？<|im_end|>
        """
        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                # 找到序列开始的位置
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    # 如果找到序列结束的位置，就退出while循环，没找到就给end进行累加1
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end +=1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    # 对开始位置 + 1 后开始循环，对初始化的掩码矩阵进行掩码设置
                    loss_mask[j] = 1
                # 对i进行赋值（用来） 结束位置index + 结束位置标识长度（因为是个字符串）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1

        return loss_mask
    
    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建对话提示模板
        """
        生成如下格式字符串
        <|im_start|>system
        你是助手<|im_end|>
        <|im_start|>user
        你好<|im_end|>
        <|im_start|>assistant
        你好！有什么可以帮助你的吗？<|im_end|>
        """
        prompt = self._create_chat_prompt(sample['conversations'])
        """
        分词和截断
            self.tonkenizer(prompt) 将文本转化为token
            例如"我爱你 汪婷婷" -> [100,234,455,453,3434,3434]
            获取input_ids # 提取token id列表，
            长度截断：[:self.max_lenght] #只保留前max_length个token
            如果生成的token数量超过max_length,只保留max_length个
        示例
            # 假设 max_length = 5
            prompt = "This is a long text that will be truncated"
            tokenizer(prompt).input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 分词结果
            # 截断后: [1, 2, 3, 4, 5]
        """
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        """
        不够的做填充处理
            计算填充长度
            self.max_length - len(input_ids) #需要填充的token数量
            创建填充序列
            [self.tokenizer.pad_token_id]
            拼接序列
            input_ids += 填充序列  # 在末尾添加padding tokens
        """
        input_ids += [self.tokenizer.pad_token_id]* (self.max_length - len(input_ids))

        # 动态生成损失掩码
        loss_mask = self._generate_loss_mask(input_ids)
        #  构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:],dtype=torch.long)  #与预测位置对其

        return X,Y,loss_mask




    