"""
训练工具函数集合
"""
import os
import random
import math
import numpy as np
import torch 
import torch.distributed as dist
from torch.utils.data import Sampler

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

def get_lr(current_step, total_steps,lr):
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
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def init_distributed_mode():
    if int(os.environ.get("RANK",-1)) == -1:
        return 0 # 非DDP模式
    
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def setup_seed(seed:int):
    random.seed(seed)
    np.random.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft',model=None,optimizer=None,epoch=0,step=0,wandb=None,save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    # 保存文件名组装
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    # 每层的名字一样，意味着是对检查点直接覆盖
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None: #保存模式
        #  引入数据并行
        from torch.nn.parallel import DistributedDataParallel
        # 获取缓存中的权重字典
        """
        根据模型是否是分布式训练来决定如何获取状态字典，如果是分布式就用model.module来获取如果不是分布式直接通过model.states_dict()来获取
        作用：
        1、统一处理，无论是否使用分布式，都能获取到格式统一的模型状态字典
        2、除去DDP包装，只保存模型本身的信息
        3、提高兼容性：保存的检查点可以在单机和分布式环境中灵活切换
        4、标准化参数名：确保参数名称不包含多余的module 前缀
        PyTorch分布式训练中保存检查点的最佳实践，确保了模型保存和加载的一致性和兼容性
        """
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        # 组装缓存文件名  
        ckp_tmp = ckp_path + '.tmp'
        """保存
        1、转化为半精度
        {k: v.half() for k,v in state_dict.items()}
        .half() 将浮点数从float32转化为float16
        目的：减少文件大小，加速加载速度
        效果：文件大小减少约50%
        2、保存到临时文件
        torch.save(...,ckp_tmp)
        ckp_tmp 是临时文件名，如：checkpoint.pth.tmp
        先保存到临时文件，而不是直接覆盖目标文件
        原子重重命名
        os.replate(ckp_tmp, ckp_path)
        作用： 
        原子操作：重命名是原子性的，要么完全成功，要么完全失败
        避免损坏：防止在保存过程中程序奔溃导致检查点文件损坏

        如果直接保存到目标文件
        如： torch.save(state_dict, 'checkpoint.pth')
        如果保存过程中：程序奔溃，断电，磁盘空间不足，会导致检查点文件损坏，无法加载
        总结
        这两行代码实现了：
            空间优化：通过半精度减少50%文件大小
            安全性：通过临时文件机制防止文件损坏
            原子性：确保检查点要么完整存在，要么完全不存在
            可靠性：特别适合长时间运行的大模型训练任务
        """
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp,ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb,'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', "None") if run else None
            else:
                wandb_id = getattr(wandb, 'id',None)
        # 重新加载的配置数据
        resume_data = {
            "model": state_dict,
            "optimmizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1, #cup个数发现变化
            "wandb_id": wandb_id
        }

        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'): #检查是否有状态字典方法
                    if isinstance(value, DistributedDataParallel): #如果是分布式训练
                        resume_data[key] = value.module.state_dict() #从module中获取状态字典
                    else:
                        resume_data[key] = value.state_dict() #普通值，直接获取状态字典
                else:
                    resume_data[key] = value
            
        resume_tmp = resume_path + '.tmp'
        # 先保存为缓存文件，然后直接重命名
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path) 

         
    else: #加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path,map_location='cpu')
            save_ws = ckp_data.get('world_size',1) # world_size: GPU数量
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if save_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * save_ws // current_ws
                Logger(f"GPU数量变化({save_ws}->{current_ws}),step已自动转化为{ckp_data['step']}")
            return ckp_data
        return None

def init_model(lm_config,from_weight='pretrain',tokenizer_path='../model',save_dir="../out",device='cuda'):
    from transformers import AutoTokenizer
    from model.model_minimind import MiniMindForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        weights = torch.load(weight_path,map_location=device)
        model.load_state_dict(weights, strict=False)

    Logger(f"加载Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}百万")
    return model.to(device), tokenizer

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        """
        self.sampler是基础采样器，它负责生成单个样本的索引
        """
        for idx in self.sampler:
            # 将每个样本的索引收集到数组中
            batch.append(idx)
            # 判断数组长度是否等于批次大小，如果等于就相当于一个批次数据收集完成
            if len(batch) == self.batch_size:
                #  判断是否跳过这个批次数据
                if skipped < self.skip_batches:
                    skipped +=1
                    # 跳过该批次样本
                    batch = []
                    continue
                # 输出批次数据（可能需要跳过该批次，然后返回一个空列表，也可能是真正收集好的数据的input_ids）
                yield batch
        if len(batch) > 0 and skipped >= self.skip_batches:
            #  如果批次中的样本不为空，并且批次已经大于要跳过的批次了就将这个批次样本 收集，等待被执行这个方法的地方获取（next）
            yield batch

    def __len__(self):
        # 计算公式
        """
        总批次计算： (总样本数 + 批次大小 -1) // 批次大小
        有效批次：总批次 - 跳过的批次
        示例计算：
            总样本数 = 10, batch_size = 3,skip_batches = 1
            总批次 = (10 +3 -1) // 3 = 4
            有效批次 = max(0, 4-1) =  3
            加3(批次大小)减 1，这种做法是标准的向上取整的方法
            向下取整: 10 // 3 = 3
            这个公式的核心思想是：
                在除法前先加上 (除数 - 1)
                这样就能确保任何余数都会使结果进位
        """
        total_batches = (len(self.sampler) + self.batch_size -1) // self.batch_size
        return max(0,total_batches - self.skip_batches)