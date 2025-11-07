import random
import re
import os
from threading import Thread
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 尝试导入 PEFT 相关库
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    st.warning("PEFT 库未安装，LoRA 功能将不可用。请运行: pip install peft")

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

# 样式定义
st.markdown("""
    <style>
        .stButton button {
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;
            margin: 5px 10px 5px 0 !important;
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .user-message {
            display: flex; 
            justify-content: flex-end;
            margin: 10px 0;
        }
        .user-bubble {
            display: inline-block;
            padding: 8px 12px;
            background-color: #007bff;
            border-radius: 10px;
            color: white;
            max-width: 70%;
        }
    </style>
""", unsafe_allow_html=True)

# 全局变量
device = "cuda" if torch.cuda.is_available() else "cpu"
image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"

def validate_model_path(model_path):
    """验证模型路径是否存在且有效"""
    if not os.path.exists(model_path):
        return False, f"路径不存在: {model_path}"
    
    # 检查必要的模型文件
    required_files = ['config.json']
    optional_files = ['pytorch_model.bin', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']
    
    missing_required = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_required:
        return False, f"缺少必要文件: {missing_required}"
    
    return True, "路径有效"

def validate_lora_path(lora_path):
    """验证 LoRA 路径是否存在且有效"""
    if not os.path.exists(lora_path):
        return False, f"LoRA 路径不存在: {lora_path}"
    
    # 检查 LoRA 适配器文件
    lora_files = ['adapter_config.json', 'adapter_model.safetensors', 'adapter_model.bin']
    has_lora_files = any(os.path.exists(os.path.join(lora_path, f)) for f in lora_files)
    
    if not has_lora_files:
        return False, f"LoRA 路径不包含适配器文件: {lora_path}"
    
    return True, "LoRA 路径有效"

@st.cache_resource
def load_base_model_tokenizer(model_path):
    """加载基础模型和tokenizer"""
    is_valid, message = validate_model_path(model_path)
    if not is_valid:
        st.error(f"基础模型路径无效: {message}")
        return None, None
    
    try:
        st.info(f"正在加载基础模型: {model_path}")
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        model = model.eval()
        st.success("基础模型加载成功!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"基础模型加载失败: {str(e)}")
        return None, None

@st.cache_resource
def load_lora_model(_base_model, lora_path):
    """加载 LoRA 适配器到基础模型"""
    if not PEFT_AVAILABLE:
        st.error("PEFT 库未安装，无法加载 LoRA 模型")
        return None
    
    is_valid, message = validate_lora_path(lora_path)
    if not is_valid:
        st.error(f"LoRA 路径无效: {message}")
        return None
    
    try:
        st.info(f"正在加载 LoRA 适配器: {lora_path}")
        
        # 使用 PEFT 加载 LoRA 适配器
        model = PeftModel.from_pretrained(_base_model, lora_path)
        st.success("LoRA 适配器加载成功!")
        return model
        
    except Exception as e:
        st.error(f"LoRA 适配器加载失败: {str(e)}")
        return None

def process_assistant_content(content, is_r1_model=True):
    """处理助手回复内容"""
    if not is_r1_model:
        return content

    if '<think>' in content and '</think>' in content:
        content = re.sub(
            r'<think>(.*?)</think>',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px; margin: 5px 0;"><summary style="font-weight:bold; cursor: pointer;">🤔 推理过程（点击展开）</summary>\1</details>',
            content,
            flags=re.DOTALL
        )
    elif '<think>' in content and '</think>' not in content:
        content = re.sub(
            r'<think>(.*?)$',
            r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px; margin: 5px 0;"><summary style="font-weight:bold;">🧠 推理中...</summary>\1</details>',
            content,
            flags=re.DOTALL
        )
    
    return content

def main():
    """主函数"""
    st.sidebar.title("模型设定调整")
    
    # 模型路径配置
    MODEL_PATHS = {
        "MiniMind2": [os.path.join(PROJECT_ROOT, "out/transformer/MiniMind2"), "MiniMind2"],
    }
    
    # LoRA 路径配置（根据您的实际路径修改）
    LORA_PATHS = {
        "无 LoRA": None,
        # "h_lora": os.path.join(PROJECT_ROOT, "out/lora/training_data_lora_converted"), 
    }
    
    # 侧边栏配置
    st.session_state.history_chat_num = st.sidebar.slider("历史对话轮数", 0, 6, 0, step=2)
    st.session_state.max_new_tokens = st.sidebar.slider("最大生成长度", 256, 8192, 2048, step=256)
    st.session_state.temperature = st.sidebar.slider("温度参数", 0.6, 1.2, 0.85, step=0.01)
    
    model_source = st.sidebar.radio("选择模型来源", ["本地模型", "API"], index=1)
    
    if model_source == "API":
        api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8998/v1")
        api_model_id = st.sidebar.text_input("Model ID", value="minimind")
        api_model_name = st.sidebar.text_input("Model Name", value="MiniMind2")
        api_key = st.sidebar.text_input("API Key", value="none", type="password")
        slogan = f"Hi, I'm {api_model_name}"
        model_config = {"api_config": {"url": api_url, "model_id": api_model_id, "model_name": api_model_name, "key": api_key}}
    else:
        selected_model = st.sidebar.selectbox('选择基础模型', list(MODEL_PATHS.keys()), index=0)
        model_path = MODEL_PATHS[selected_model][0]
        
        # LoRA 选择
        selected_lora = st.sidebar.selectbox('选择 LoRA 适配器', list(LORA_PATHS.keys()), index=0)
        lora_path = LORA_PATHS[selected_lora]
        
        slogan = f"Hi, I'm {MODEL_PATHS[selected_model][1]}"
        if selected_lora != "无 LoRA":
            slogan += f" + {selected_lora}"
            
        model_config = {
            "model_path": model_path, 
            "selected_model": selected_model,
            "lora_path": lora_path,
            "selected_lora": selected_lora
        }
    
    # 显示标题
    st.markdown(
        f'''
        <div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">
            <div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">
                <img src="{image_url}" style="width: 45px; height: 45px;">
                <span style="font-size: 26px; margin-left: 10px;">{slogan}</span>
            </div>
            <span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">
                内容完全由AI生成，请务必仔细甄别<br>Content AI-generated, please discern with care
            </span>
        </div>
        ''',
        unsafe_allow_html=True
    )
    
    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # 加载模型
    if model_source == "本地模型":
        # 先加载基础模型
        base_model, tokenizer = load_base_model_tokenizer(model_config["model_path"])
        
        if base_model is None or tokenizer is None:
            st.error("基础模型加载失败，请检查模型路径和文件完整性")
            return
        
        # 如果选择了 LoRA，加载 LoRA 适配器
        if model_config["lora_path"]:
            model = load_lora_model(base_model, model_config["lora_path"])
            if model is None:
                st.warning("LoRA 加载失败，将使用基础模型")
                model = base_model
        else:
            model = base_model
    else:
        model, tokenizer = None, None
    
    # 显示聊天历史
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=image_url):
                is_r1_model = 'R1' in model_config.get('selected_model', '')
                st.markdown(process_assistant_content(message["content"], is_r1_model), unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="user-message"><div class="user-bubble">{message["content"]}</div></div>',
                unsafe_allow_html=True
            )
    
    # 聊天输入
    prompt = st.chat_input("给 MiniMind 发送消息...")
    
    if prompt:
        # 显示用户消息
        st.markdown(
            f'<div class="user-message"><div class="user-bubble">{prompt}</div></div>',
            unsafe_allow_html=True
        )
        
        # 保存用户消息
        truncated_prompt = prompt[-st.session_state.max_new_tokens:]
        st.session_state.messages.append({"role": "user", "content": truncated_prompt})
        st.session_state.chat_messages.append({"role": "user", "content": truncated_prompt})
        
        # 生成回复
        with st.chat_message("assistant", avatar=image_url):
            placeholder = st.empty()
            
            if model_source == "API":
                # API调用逻辑
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key, base_url=api_url)
                    
                    history_num = st.session_state.history_chat_num
                    conversation_history = st.session_state.chat_messages[-(history_num * 2):] if history_num > 0 else []
                    conversation_history.append({"role": "user", "content": truncated_prompt})
                    
                    answer = ""
                    response = client.chat.completions.create(
                        model=api_model_id,
                        messages=conversation_history,
                        stream=True,
                        temperature=st.session_state.temperature
                    )
                    
                    for chunk in response:
                        content = chunk.choices[0].delta.content or ""
                        answer += content
                        placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)
                    
                    # 保存助手回复
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"API调用出错: {str(e)}")
            else:
                # 本地模型生成逻辑
                try:
                    random_seed = random.randint(0, 2**32 - 1)
                    torch.manual_seed(random_seed)
                    
                    # 构建对话历史
                    history_num = st.session_state.history_chat_num
                    chat_history = st.session_state.chat_messages[-(history_num * 2):] if history_num > 0 else []
                    chat_history.append({"role": "user", "content": truncated_prompt})
                    
                    # 应用聊天模板
                    new_prompt = tokenizer.apply_chat_template(
                        chat_history,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    inputs = tokenizer(
                        new_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=8192
                    ).to(device)
                    
                    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    
                    generation_kwargs = {
                        "input_ids": inputs.input_ids,
                        "max_new_tokens": st.session_state.max_new_tokens,
                        "temperature": st.session_state.temperature,
                        "top_p": 0.85,
                        "do_sample": True,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                        "streamer": streamer,
                    }
                    
                    # 使用模型生成
                    thread = Thread(target=model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    answer = ""
                    for new_text in streamer:
                        answer += new_text
                        is_r1_model = 'R1' in model_config.get('selected_model', '')
                        placeholder.markdown(
                            process_assistant_content(answer, is_r1_model), 
                            unsafe_allow_html=True
                        )
                    
                    # 保存助手回复
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"生成回复时出错: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()