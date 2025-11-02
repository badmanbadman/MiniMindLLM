import argparse
import random
import warnings
import numbers as np
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))

        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        # strict 为true代表加载的state_dict必须与模型的state_dict完全匹配，否则报错
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight !='None':
            apply_lora(model)
            load_lora(model, f"./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth")
    else:
        # 自动检测并加载合适的因果语言模型（causal lm）
        # 因果语言模型，用于文本生成任务，根据前文预测下个token
        """
        args.load_from
        模型标识符 ,可以是：
            hugging face模型仓库名，如（gpt2，meta-llama/lllama-2-7B等）
            本地模型目录路径
        trust_remote_code = True
        安全相关参数，代表：
        1 信任远程代码
         允许下载和执行模型自定的Python代码
         包括自定义的模型架构，前向传播逻辑等
        2 使用场景
          当模型不再Transformer官方支持列表中时，
          模型使用了自定义或特殊处理
          社区模型，研究模型等
        """
        model=AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f"MiniMind模型参数： {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out/pth', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()

    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]

    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0]自动测试\n[1]手动输入\n'))
    """
    实现类似chatGPT打字机效果
    tokenizer
        用于将生成的token ID实时解码为可读文本
    skip_prompt=True
        跳过提示文本，不重复显示用户输入的问题
        只显示模型新生成的内容
    skip_special_tokens = True
        跳过特殊标记： 不显示如<|endoftext|>,<pad>,<s>等特殊token
        让输出更加干净自然
    """
    streamer = TextStreamer(tokenizer,skip_prompt=True, skip_special_token=True)
    """
    iter()接受2个参数
    第一个参数 可调用对象（函数/lambda)
    第二个参数  哨兵值，当可调用对象返回这个值时，迭代停止

    lambda: Python中的匿名函数，它允许我们在一行内定义简单函数（只能由一个表达式不能包含复杂逻辑），不需要def关键字
    """
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('👶: '), '')
    for prompt in prompt_iter:
        setup_seed(2026)
        if input_mode == 0: print(f'👶: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role":"user", 'content': prompt})

        templates = {'conversation': conversation, 'tokenize': False, 'add_generation_prompt': True}
        if args.weight == 'reason': templates["enalbe_thinking"] = True
        """
        (**templates) 是python中的字典解包语法，也称为关键字参数解包
        templates = {
            "conversation": conversation,
            "tokenize": False, 
            "add_generation_prompt": True
        }

        # 这两种写法完全等价：
        inputs = tokenizer.apply_chat_template(**templates)
        # 等价于：
        inputs = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        单个星号 * - 列表/元组解包
        def func(a, b, c):
            print(a, b, c)

        args = [1, 2, 3]
        func(*args)  # 输出: 1 2 3
        """
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        """
        tokenizer(inputs,...)
        inputs 是之前通过apply_chat_template或者拼接得到的文本字符串
        作用： 将文本字符串进行分词，转换成token ids
        return_tensors='pt'
        作用： 返回Python张量格式，而不是python列表或者NumPy
        输出示例
        {
            "input_ids": tensor([[ 101, 123, 456, 789, 102]]),      # token IDs
            "attention_mask": tensor([[1, 1, 1, 1, 1]])            # 注意力掩码
        }
        truncation = True
        作用： 如果输入超长，自动截断
        """
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖️: ', end='')
        generated_ids = model.generate(
            inputs=inputs['input_ids'],  # 输入序列的tokenIDs，作为生成的起点
            attention_mask = inputs['attention_mask'], #注意力掩码，指示哪些token是有效内容，哪些是填充的0
            max_new_tokens=args.max_new_tokens, #生成的最大token数量，
            do_sample=True,#是否采样，如果额外iTrue则使用采用策略如top-p,温度，如为False，则使用贪心搜索
            streamer=streamer, #用于流式输出生成结果的流式处理器，可以实时显示生成的token
            pad_token_id=tokenizer.pad_token_id, #填充token的ID，用于生成过程中忽略填充token。
            eos_token_id=tokenizer.eos_token_id, #结束token的ID，当生成此token时，停止生成            
            top_p=args.top_p, #nucleus采样的参数，仅保留概率累积达到top_p的最小token集合，然后从中采样。             
            temperature=args.temperature,#温度参数，用于调整采样分布的尖锐程度。温度越高，分布越平，生成越多样；温度越低，分布越尖，生成越确定               
            repetition_penalty=1.0   #重复惩罚因子，避免生成重复内容。1.0表示没有惩罚。            
                                          )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        print('\n\n')

if __name__ == "__main__":
    main()