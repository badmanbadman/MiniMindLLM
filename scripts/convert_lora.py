import os
import torch
import sys

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def main():
    # 配置路径
    lora_pth_path = "./out/lora/lora_training_data_512.pth"
    
    # 使用绝对路径指向同级别的 MiniMind2 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_path = os.path.join(current_dir, "MiniMind2")
    
    output_lora_path = "./out/lora/training_data_lora_converted"
    
    print("开始转换 LoRA 格式...")
    print(f"基础模型路径: {base_model_path}")
    
    # 检查文件
    if not os.path.exists(lora_pth_path):
        print(f"错误: LoRA .pth 文件不存在: {lora_pth_path}")
        return
    
    if not os.path.exists(base_model_path):
        print(f"错误: 基础模型路径不存在: {base_model_path}")
        print("请确保 MiniMind2 文件夹与 convert_lora.py 在同一目录下")
        return
    
    # 加载基础模型
    print("加载基础模型...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True  # 如果模型需要自定义代码
        )
    except Exception as e:
        print(f"加载基础模型失败: {e}")
        return
    
    # 其余代码保持不变...
    print("加载 LoRA 权重...")
    state_dict = torch.load(lora_pth_path, map_location='cpu')
    print(state_dict)
    
    # 创建 LoRA 配置
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=True,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # 根据您的模型调整
    )
    
    # 创建 PEFT 模型
    print("创建 PEFT 模型...")
    peft_model = get_peft_model(base_model, lora_config)
    
    # 加载权重
    print("加载权重到 PEFT 模型...")
    peft_model.load_state_dict(state_dict, strict=False)
    
    # 保存
    print(f"保存到: {output_lora_path}")
    peft_model.save_pretrained(output_lora_path)
    
    print("转换完成!")

if __name__ == "__main__":
    main()