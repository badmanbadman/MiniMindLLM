import os
import sys


"""
代码明确声明当前模块属于名为"trainer"的包。
为什么要这个声明
这两行代码配合工作：
__package__ 定义包名
sys.path.append 将父目录添加到Python路径，使得可以找到"trainer"包

在这个训练脚本中的必要性
考虑倒这是一个分布式训练脚本，需要确保：
    模块导入绝对可靠
    避免因路径问题导致的训练中断
    在多个GPU环境中保持一致性
"""
__package__  = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import math
import warnings
import torch
"""
PyTorch的分布式通信包(distributed)
常用的功能包括：
    dist.init_process_group: 初始化进程组，设置后端通信方式等
    dist.get_rank: 获取当前进程的排名，（一个整数，从0开始）
    dist.get_world_size: 获取总进程数（即总GPU数）
    dist.broadcast: 将张量从一个进程广播倒其他所有进程
    dist.all_reduce: 对所有进程中的张量进行规约操作（如求和，求平均等），并将结果同步到所有进程
    dist.barrier: 同步所有进程，直到所有进程都到这个点
    dist.is_intialized() 用于检查Pytorch的分布式进程组是否已经初始化
用于分布式数据并行（DDP）训练
提供了多进程并行训练的底层通信原语
支持多种后端：NCCL(NVIDIA GPU),GLOO(CPU/GPU),MPI
    nccl：NVIDIA GPU间高速通信，推荐用于GPU训练
    gloo：支持CPU和GPU，但GPU性能不如NCCL
    mpi：高性能计算环境

为什么需要分布式训练？
单机多GPU场景：
数据并行：每个GPU处理不同的数据批次
模型并行：模型太大，拆分到多个GPU
混合并行：数据+模型并行

优势：
加速训练：线性加速比
处理更大批次：梯度累积+多GPU
训练更大模型：内存聚合

分布式训练中，我们通常使用以下概念：
    Node（节点）：一般指一台机器。
    Rank：每个进程的全局唯一标识。在一个多机训练中，rank 0 通常为主进程。
    Local Rank：在一台机器上的进程本地编号。例如，一台有8个GPU的机器，local rank从0到7。
    World Size：所有进程的总数，即所有GPU数量。
"""
import torch.distributed as dist
from torch import optim, nn
"""
DistributedDataParallel
DDP是pytorch提供的分布式数据并行训练包装器（单机多卡，多机多卡）
DDP 的工作原理
    同一批次数据被拆分到不同GPU：
    GPU0: [样本0, 样本1, 样本2] → 计算梯度
    GPU1: [样本3, 样本4, 样本5] → 计算梯度  
    GPU2: [样本6, 样本7, 样本8] → 计算梯度
    GPU3: [样本9, 样本10, 样本11] → 计算梯度

    DDP自动同步所有GPU的梯度 → 更新模型
 DDP 的核心优势
 # 理论加速效果
GPU数量: 1 → 训练时间: 100小时
GPU数量: 2 → 训练时间: ~50小时 (2倍加速)
GPU数量: 4 → 训练时间: ~25小时 (4倍加速)
GPU数量: 8 → 训练时间: ~12.5小时 (8倍加速)
"""
from torch.nn.parallel import DistributedDataParallel
"""
DistributedSampler分布式采样器
    在分布式环境中确保每个进程获得不同的数据子集
DistributedSampler 的工作原理
    数据分片算法
    # 假设有8个样本，2个GPU
    数据集: [A, B, C, D, E, F, G, H]

    # DistributedSampler分片：
    epoch 0:
    GPU0 (rank=0): [A, C, E, G]  # 索引 0,2,4,6
    GPU1 (rank=1): [B, D, F, H]  # 索引 1,3,5,7

    epoch 1: (调用set_epoch(1)后重新打乱)
    GPU0: [D, A, H, B]  # 新的随机排列
    GPU1: [F, C, E, G]
DataLoader 重要参数
    DataLoader(
        batch_size=32,           # 每个GPU的批次大小
        num_workers=4,           # 数据加载进程数  # 经验法则：通常设置为CPU核心数或GPU数量的2-4倍
        pin_memory=True,         # 将数据固定到锁页内存，加速CPU→GPU传输
        drop_last=False,         # 保留不完整批次
        persistent_workers=True  # 保持worker进程 alive（可选）
    ) 
DistributedSampler 重要参数
    DistributedSampler(
        dataset,                 # 数据集
        num_replicas=world_size, # 总进程数（自动获取）
        rank=global_rank,        # 当前进程排名（自动获取）
        shuffle=True,            # 是否打乱数据
        seed=42                  # 随机种子
    )
没有这个组合，分布式训练就会面临数据重复、加载瓶颈等问题，严重影响训练效果和效率。
"""
from torch.utils.data import DataLoader, DistributedSampler
"""
nullcontext 是一个"什么都不做"的上下文管理器，它提供了一种统一的方式来处理"有时需要上下文，有时不需要"的情况。
在混合精度训练中的关键作用
    # 混合精度：部分操作使用FP16，部分使用FP32
    # 优点：减少显存使用，加快计算速度
    # 缺点：需要小心数值稳定性

    # GPU混合精度上下文
    with torch.cuda.amp.autocast():
        # 模型前向传播自动使用合适的精度
        outputs = model(inputs)  # 可能使用FP16
        loss = criterion(outputs, targets)
    CPU训练的特殊性
    # CPU不支持CUDA AMP混合精度
    # 如果强行使用会报错：
    # with torch.cuda.amp.autocast():  # ❌ CPU上会报错
    #     outputs = model(inputs)

    # 解决方案：使用nullcontext
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    with ctx:  # ✅ CPU和GPU都适用
        outputs = model(inputs)
"""
from contextlib import nullcontext
"""
Hugging Face Transformers库中的一个类，用于1、自动加载，2预训练模型对应的分词器
在自然语言处理（NLP）中，分词器负责将原始文本分割成模型能够处理的令牌（tokens），并将这些令牌转换为模型所需的数字ID

另外，分词器通常包括以下功能：
    分词（将文本拆分为单词、子词或字符）
    将令牌转换为ID（encode）
    将ID转换回文本（decode）
    添加特殊令牌（如[CLS]、[SEP]等）
在训练脚本中，我们使用分词器对文本进行编码，生成模型输入所需的input_ids、attention_mask等。
"""
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

warnings.filterwarnings('ignore')

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_lr(current_step,total_steps,lr):
    """
    余弦退火学习率调度器结合了常数学习和余弦衰减
    根据训练进度动态调整学习率
    
    这种调度策略的优势
    1. 平滑衰减
    # 对比不同调度策略
    恒定学习率: lr → lr → lr → lr ❌ 可能无法收敛到最优
    阶梯下降:   lr → lr → lr/2 → lr/2 ❌ 突变可能破坏训练
    余弦退火:   lr → 平滑下降 → 小lr ✅ 平滑过渡，稳定收敛
    2. 避免局部最优
    # 开始阶段相对较高的学习率有助于：
    - 快速逃离较差的局部最优点
    - 探索更广的参数空间
    - 加速训练初期进展
    3. 精细调优
    # 结束阶段较低的学习率有助于：
    - 精细调整模型参数
    - 稳定收敛到好的最优点
    - 避免在最优解附近震荡
    """
    return lr / 10 + 0.5*lr*(1+math.cos(math.pi*current_step/total_steps))

def train_epoch(epoch,wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step,(X,Y,loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch*iter_per_epoch+step, args.epochs*iter_per_epoch,args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            # 前向传播 
            res = model(X)
            # 计算总的损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)), # 模型推断得分
                Y.view(-1) # 
            ).view(Y.size())

            """
            # 假设词汇表片段：
            词汇表 = {
                0: "<pad>", 1: "<eos>", 2: "<unk>",
                100: "I", 101: "love", 102: "machine", 103: "learning",
                ... # 30000个token
            }

            # 训练批次示例：
            X = [
                [100, 101, 102, 103,   1,   0],  # "I love machine learning <eos> <pad>"
                [104, 105, 106,   1,   0,   0]   # "This is good <eos> <pad> <pad>"
            ]

            Y = [
                [101, 102, 103,   1,   0,   0],  # "love machine learning <eos> <pad> <pad>"
                [105, 106,   1,   0,   0,   0]   # "is good <eos> <pad> <pad> <pad>"
            ]

            # 损失掩码（只计算有效位置）：
            loss_mask = [
                [1, 1, 1, 1, 0, 0],  # 样本1：前4个位置计算损失
                [1, 1, 1, 0, 0, 0]   # 样本2：前3个位置计算损失
            ]
            """
            loss = (loss*loss_mask).sum() / loss_mask.sum()

        scaler.scale(loss).backward()

        #  每隔accumulation_steps次进行一次优化器步进
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放
            scaler.unscale_(optimizer)
            # 进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新参数（带缩放）
            scaler.step(optimizer)
              # 混合精度训练的"智能调节器"
            """
            主要作用:
            1 动态调整缩放因子
            2 防止梯度溢出
                FP16数值范围较小(~5.96e-8到65504)
                小梯度在FP16中可能下溢成为0
                大梯度可能上溢为inf
                scaler自动检测并动态调整缩放比例
            3 确保稳定 避免训练奔溃
            4 性能优化 找到最佳的缩放比例
            """
            scaler.update()
            # 清空梯度（使用内存优化方式）
            """
            # 没有 set_to_none=True 的内存模式
            迭代1: 分配梯度内存 → 计算梯度 → 更新参数 → 梯度置零
            迭代2: 重用梯度内存 → 计算梯度 → 更新参数 → 梯度置零
            迭代3: 重用梯度内存 → 计算梯度 → 更新参数 → 梯度置零
            # ✅ 内存稳定，但一直占用梯度内存

            # 使用 set_to_none=True 的内存模式  
            迭代1: 分配梯度内存 → 计算梯度 → 更新参数 → 释放梯度内存
            迭代2: 分配梯度内存 → 计算梯度 → 更新参数 → 释放梯度内存  
            迭代3: 分配梯度内存 → 计算梯度 → 更新参数 → 释放梯度内存
            # ✅ 峰值内存更低，但分配开销稍大
            """
            optimizer.zero_grad(set_to_none=True)
        
        if step % args.log_interval == 0 or step == iter_per_epoch -1:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.6f} lr:{:.12f} epoch_Time:{}min'.format(
                    epoch + 1, #当前epoch(从1开始计数))
                    args.epochs, #总epoch数
                    step,   # 当前epoch内的步数（当前迭代次数）
                    iter_per_epoch, # 每个epoch的总步数（即每个epoch有多少个batch）
                    loss.item() * args.accumulation_steps, #由于在梯度累积中，损失被除以了累积步数，这里要乘回来显示原始损失规模
                    optimizer.param_groups[-1]['lr'], #当前的学习率(取优化器数组中最后一个参数组的学习率，通常只有一个参数组，所以这样取没有问题)
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60)) # 用于估计剩余的epoch时间（以分钟为单位）

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                """困惑度
                # 假设模型要预测句子 "I love ___" 的下一个词
                    模型A的预测：
                    - machine: 概率 0.4
                    - deep: 概率 0.3  
                    - Python: 概率 0.2
                    - ...其他词概率很小
                    困惑度 ≈ 3.0  # 相当于平均有3个等可能的候选词

                    模型B的预测：
                    - machine: 概率 0.9
                    - learning: 概率 0.05
                    - ...其他词概率很小  
                    困惑度 ≈ 1.1  # 模型很确定应该是"machine"

                    # 结论：困惑度越低，模型预测越准确！ 
                    # LLM 预训练中困惑度的典型范围
                    训练阶段        | 损失值 | 困惑度     | 含义
                    -----------------------------------------------
                    初始阶段        | 8-10   | 3000-22026 | 几乎随机猜测
                    早期收敛        | 4-6    | 55-403     | 开始学习语言规律
                    中期训练        | 2-3    | 7-20       | 掌握基本语法
                    后期训练        | 1-2    | 3-7        | 流利的语言生成
                    优秀模型        | 0.5-1  | 1.6-2.7    | 接近人类水平
                    # 不同困惑度对应的模型能力
                    困惑度 > 100: 模型基本不会说人话
                    困惑度 20-50: 能生成基本通顺的句子  
                    困惑度 10-20: 语言流畅，逻辑基本通顺
                    困惑度 5-10:  高质量文本生成
                    困惑度 < 5:   接近人类写作水平

                    损失 vs 困惑度关系
                    损失值   | 困惑度
                    -----------------
                    10.0    | 22026.5
                    5.0     | 148.4
                    3.0     | 20.1
                    2.0     | 7.4
                    1.5     | 4.5
                    1.0     | 2.7
                    0.5     | 1.6
                    0.1     | 1.1

                    # 在相同数据上比较不同架构
                    模型A (Transformer): 困惑度=15.2
                    模型B (LSTM):       困惑度=28.5
                    模型C (你的架构):    困惑度=14.8 ✅

                    # 困惑度不是万能的
                    低困惑度 ≠ 高质量生成

                    # 可能出现：
                    - 模型过拟合训练数据风格
                    - 生成文本缺乏创造性
                    - 重复模式问题

                    # 需要配合人工评估
                """
                wandb.log({
                    "loss": loss.item() * args.accumulation_steps, #损失下降趋势
                    "lr": optimizer.param_groups[-1]['lr'], # 学习率曲线
                    "epoch_Time": spend_time /(step + 1) * iter_per_epoch // 60 -spend_time // 60, # 时间预估
                    "perplexity": math.exp(loss.item()),  # 困惑度（语言模型）【越小越好】  模型在预测下一个词时的平均选择困难程度
                        
                })
            
        if((step +1)%args.save_interval == 0 or step == iter_per_epoch -1) and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            """
            # 保存的 state_dict 包含：
            state_dict = {
                # 模型参数
                'embedding.weight': tensor(...),
                'transformer.0.attention.wq.weight': tensor(...),
                'transformer.0.attention.wk.weight': tensor(...),
                'transformer.0.attention.wv.weight': tensor(...),
                'transformer.0.attention.wo.weight': tensor(...),
                'transformer.0.feed_forward.w1.weight': tensor(...),
                # ... 所有可学习参数
                
                # 注意：不包含优化器状态、学习率调度器状态等
            }
            """
            state_dict = {k: v.half() for k , v in state_dict.items()} # 半精度保留
            torch.save(state_dict,ckp)
            model.train()

            



def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config).to(args.device)
    Logger(f"LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}百万")
    return model, tokenizer

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    # 如果为True，则初始化wandb并记录训练过程。
    parser.add_argument("--use_wandb", action="store_true")
    # 用于设置wandb项目的名称。
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    # 如果为True，则使用分布式训练，初始化进程组，并使用DistributedDataParallel包装模型。
    parser.add_argument("--ddp", action="store_true")
    # 在优化器步进之前，累积梯度的步数。这用于模拟更大的批次大小。例如，如果实际批次大小为8，累积步数为4，则有效批次大小为32。
    parser.add_argument("--accumulation_steps", type=int, default=8)
    #  在优化器步进之前，对梯度进行裁剪，防止梯度爆炸。
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # 学习率预热的迭代次数。在训练开始时，学习率从0线性增加到初始学习率，持续warmup_iters步。
    parser.add_argument("--warmup_iters", type=int, default=0)
    #  控制打印训练日志的频率，例如每100个迭代打印一次损失和学习率。
    parser.add_argument("--log_interval", type=int, default=100)
    #  控制保存模型检查点的频率，例如每100个迭代保存一次模型。
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/processed_data/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe)
    
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir,exist_ok=True)
    os.makedirs(args.out_dir,exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize={args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK",-1)) != -1 #声明全局ddp设置
    ddp_local_rank,DEVICE = 0,"cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置CUDA的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        """SwanLab 的核心功能
        1. 实验跟踪
            # 记录训练指标
            wandb.log({
                "train_loss": loss,
                "learning_rate": lr,
                "accuracy": acc,
                "epoch": epoch
            })
        2. 超参数管理
            # 自动记录所有超参数
            wandb.config.update({
                "batch_size": 32,
                "learning_rate": 5e-4,
                "model_architecture": "Transformer",
                "optimizer": "AdamW"
            })
        3. 模型版本控制
            # 保存和跟踪模型检查点
            torch.save(model.state_dict(), "model.pth")
            wandb.save("model.pth")  # 上传到SwanLab服务器
        4. 可视化分析
            # 自动生成丰富的图表
            - 损失曲线
            - 学习率变化
            - 梯度分布
            - 训练进度
        """
        import swanlab as wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None
    
    model,tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer,max_length=args.max_seq_len)
    """train_ds
    # train_ds 是 PretrainDataset 类的一个实例对象
    print(type(train_ds))  # <class '__main__.PretrainDataset'>

    # 它包含了：
    print(f"数据集大小: {len(train_ds)}")        # 通过 __len__ 方法
    print(f"第一个样本: {train_ds[0]}")         # 通过 __getitem__ 方法
    """
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds, #   ← 这里传入的是数据集实例
        batch_size=args.batch_size,
        pin_memory=True, # 将数据固定到锁页内存，加速CPU→GPU传输
        drop_last=False,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    """初始化Pytorch的梯度缩放器
    GradScaler是什么？
        GradScaler 是PyTorch自动混合精度（AMP）训练的核心工具，用于防止FP16/BF16时的梯度下溢问题
    为什么需要梯度缩放？
        FP16训练的数值问题
            # FP16 的数值范围有限
            FP16 范围: 5.96e-8 ~ 65504
            FP32 范围: 1.4e-45 ~ 3.4e38

            # 问题：小梯度在 FP16 中会变成 0
            梯度值: 1e-7 → 在 FP16 中: 0.0 ❌
            梯度值: 1e-6 → 在 FP16 中: 0.000001 ✅
        梯度缩放解决方案
            # GradScaler 的工作方式：
            原始梯度: 1e-7 (太小，在FP16中为0)
            缩放因子: 1000
            缩放后梯度: 1e-4 (在FP16中可表示)
            反向传播后，再除回去
    梯度裁剪的基本思想是：在反向传播后，如果梯度的范数超过了某个阈值，就将梯度按比例缩小，使得梯度的范数回到阈值之内。
    
    通常使用按梯度范数裁剪
        计算所有参数的梯度的L2范数（整体范数）。
        如果这个范数大于max_norm，则按比例缩放梯度，使得范数变为max_norm。
        如果范数没有超过阈值，则不做改变。
    注意：梯度裁剪不影响梯度的方向，只改变大小，因此它不会改变梯度下降的方向，只是控制了步长。
    在混合精度训练中，由于使用了梯度缩放（GradScaler），我们需要在梯度裁剪之前先取消缩放，因为GradScaler会缩放梯度值，所以我们要在原始梯度上进行裁剪。
    具体步骤：
        1. 反向传播：scaler.scale(loss).backward()
        2. 取消缩放：scaler.unscale_(optimizer)
        3. 梯度裁剪：torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        4. 优化器步骤：scaler.step(optimizer)
        5. 更新缩放因子：scaler.update()
    # 梯度裁剪的数学表达
    if gradient_norm > max_norm:
        gradient = gradient * (max_norm / gradient_norm)
    """
    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=(args.dtype in ['float16','bfloat16'])
        )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {'freqs_cos', 'freqs_sin'}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        train_epoch(epoch,wandb)

    
