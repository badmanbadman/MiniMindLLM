import argparse
import json
import os
import sys

__package__ = 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import time
import torch
import warnings
"""
uvicorn: 是一个基于uvloop和httptools构建的ASGI（Asynchronous Server Gateway）
【异步服务网关】，专门用于运行Python的异步Web应用
uvicorn.run (app, host: "0.0.0.0,port=8998)
启动了一个ASGI服务器，运行FastAPI应用，
监听地址：0.0.0.0（所有网络接口）
端口：8998
vuicorn的优势：：
    高性能：使用uvloop(基于libuv)，性能接近Go语言
    ASGI兼容：完美支持FastAPI的异步特性
    开发友好：支持重载、调试信息
    生产就绪：支持多进程，SSL等
开发环境：
    uvicorn.run(app, host="0.0.0.0", port=8998,reload=True)
生产环境：
    uvicorn.run(app,host="0.0.0.0",port=8998, workers=4,//多进程，log_level='warning)
其他常见参数
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8998,
        reload=True,           # 开发时自动重载
        log_level="info",      # 日志级别
        access_log=True,       # 访问日志
        workers=1,            # 进程数
        loop="auto",          # 事件循环类型
    )
"""
import uvicorn
"""
treading：
    Thread 是Python中用于多线程编程的类，它允许我们在同一个进程中并发执行多个任务
Thread用于在后台生成文本的同时，同时将已经生成的部分通过流式传输发送给客户端
"""
from threading import Thread
"""Queue
    提供了一个线程安全的队列，用于在生产者和消费者之间安全的传递消息
    这段代码中Queue被用于生成文本的线程和流失相应生成器之间传递数据

具体来说，在流式生成响应时，我们创建了一个自定义的TextStreamer（CusomStreamer），
它将在生成每个新的文本片段时候被调用
我们使用一个Queue来存储这些文本片段，然后再另外一个线程中运行模型生成，而主线程（异步线程）
则从Queue中读取文本并通过StreamingResponse返回给客户端

工作流程
    1、创建Queue实例
    2、将实例传递给CustomStreamer
    3、在单独的线程中运行model.generate,并指定streamer为CustomStreamer
    4、在生成过程中，模型每生成一个片段，CustomStreamer的on_finalized_text方法就会被调用，将文本放入队列
    5、同时在主线程，我们通过一个循环，从队列中读取文本，直至遇到None（标识生成结束）
    这样就实现了生成文本的实时流式传输
为什么需要Queue？
    问题背景：
    模型生成是阻塞操作，：model.generate()会一直运行直到生成完成
    HTTP请求需要及时响应，不能让客户端一直等待
    需要实时传输，希望生成一个词就立即发送一个词

解决方案：
使用生产者-消费者模式
    生产者线程： 运行model.generate() ,将生产的文本放入队列
    消费者线程： 从队列中取出文本发送给客户端
Queue 的线程安全性
Queue 是线程安全的，这意味着：

多个线程可以同时访问队列而不会出现数据竞争

put() 和 get() 操作是原子的

内部有锁机制确保数据一致性
"""
from queue import Queue
"""
FastAPI web应用框架
作用： 创建FastAPI应用实例，是整个Web服务核心
特点： 高性能基于StarLette 和Pydantic
自动文档： 自动生成OpenAPI文档和SwaggerUI

HTTPException异常处理，用于给客户端返回标准化的HTTP错误响应

StreamingResponse 流式响应
作用：实现服务器向客户端的实时数据传输
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

"""
BaseModel是Pydantic库的核心类，用于数据验证，序列化，设置管理，
1、数据验证：
    自动验证输入数据的类型和格式
    request = ChatRequest(
        model="minimind"
        messages=[{"role": "user","content": "hello"}]
        temperature=1.5 #如果超过合理范围就会被验证
    )
2、类型注解
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float=0.7
    stream: bool=False 
3、自动文档生成（Auto Documentation）
    FastAPI会自动基于Pydantic模型生成OpenAPI文档
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):  # FastAPI 使用 ChatRequest 生成文档
        return {"response": "..."}

"""
from pydantic import BaseModel
"""
AutoTokenizer, 这个是Hugging Face Transformers库中的一个类，用于自动根据模型名称或路径加载对应的分词器（tokenizer），
分词器负责将文本转化为模型可以理解的token ID序列，以及将模型输出的token ID序列转换回文本

AutoModelForCausaLM 用于自动加载因果语言模型，这类模型通常用于文本生成任务，例如GPT系列，它会根据模型名称或则和路径加载对应的模型架构和权重

TextStreamer 用于实时流式输出生成文本的类，在模型生成过程中，每当生成一个token就可以通过TextStreamer实时解码输出不需要等整个生成结束

"""
from transformers import AutoTokenizer, AutoModelForCausalLM,TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')
app = FastAPI()


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'../{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(device), tokenizer


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = False
    tools: list = []


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def generate_stream_response(messages, temperature, top_p, max_tokens):
    try:
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[-max_tokens:]
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )

        Thread(target=_generate).start()

        while True:
            text = queue.get()
            if text is None:
                yield json.dumps({
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }, ensure_ascii=False)
                break

            yield json.dumps({
                "choices": [{"delta": {"content": text}}]
            }, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
                )),
                media_type="text/event-stream"
            )
        else:
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
            )[-request.max_tokens:]
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop"
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='pth/full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    device = args.device
    model, tokenizer = init_model(args)
    uvicorn.run(app, host="0.0.0.0", port=8998)