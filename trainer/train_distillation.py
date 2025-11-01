import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse 
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
# optim模块提供了很多常见的优化器，例如随机梯度下降（SGD）、Adam、RMSprop等
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr, 
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler
)

warnings.filterwarnings("ignore")

def distillation_loss(student_logits, teacher_logits,temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits/temperature, dim=-1).detach()
    
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    k1 = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    return (temperature ** 2) * k1

def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0,wandb=None, alpha=0.0,temperature=1.0):
    start_time = time.time()

    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)
    
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step+1):
        X=X.to(args.device)
        Y=Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 前向传播（学生模型）
        with autocast_ctx:
            res = model(X)
            student_logits = res.logits

        # 老师模型前向传播（只是评估eval，没有梯度更新）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X).logits
                # 学生模型词汇表大小
                vocab_size_student = student_logits.size(-1)
                """对齐词汇表维度
                使用切片操作，:vocab_size_student截取教师logits的前面部分
                ... 是省略号，表示保留所有前面的维度（batch和sequence维度）
                为什么要这样做？：
                场景假设
                教师模型：词汇表大小50000
                学生模型：词汇表大小30000
                目标：让学生模型学习教师模型在前30000个token上的预测分布

                数学原理
                在计算KL散度时，两个分布必须具有相同的支持集
                词汇表设计的最佳实践：
                推荐学生词汇表是教师词汇表的子集
                    # 推荐：学生词汇表是教师词汇表的子集
                    teacher_vocab = ["token1", "token2", ..., "token50000"]  
                    student_vocab = ["token1", "token2", ..., "token30000"]  # 前30k个相同

                    # 不推荐：词汇表顺序不一致
                    teacher_vocab = ["a", "b", "c", "d"]
                    student_vocab = ["c", "a", "b"]  # 顺序混乱，截取无意义
                """
                teacher_probs = F.softmax(teacher_logits[..., :vocab_size_student] / temperature, dim=-1)
        
        # 计算损失
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            Y.view(-1),
            ignore_index=0,
            reduction='none'
        )

        ce_loss = torch.sum(ce_loss*loss_mask_flat) / loss_mask_flat.sum()
        if lm_config_student.use_moe:
            ce_loss += res.aux_loss

        # 蒸馏损失
        if teacher_model is not None:
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 总损失 = alpha * CE + (1-alpha)*Distill
        # alpha 是交叉熵损失 的权重，本次默认设置0.5
        loss = (alpha * ce_loss + (1-alpha)*distill_loss) / args.accumulation_steps

        # 反向传播更新梯度
        scaler.scale(loss).backward()

        # 每隔 accumulation_steps进行一次缩放器检查，进行梯度裁剪，优化器参数重新设置，累计梯度清理等
        if (step + 1) % args.accumulation_steps == 0:
            # 取消缩放
            scaler.unscale_(optimizer)
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters -1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr'] #当前的学习率(取优化器数组中最后一个参数组的学习率，通常只有一个参数组，所以这样取没有问题)
            eta_min = spend_time / (step + 1)*iters // 60 - spend_time // 60

            Logger(f"Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss: {current_loss:.6f} ce:{ce_loss.item():.4f} distill: {distill_loss.item():.4f} lr:{current_lr:.12f} epoch_Time: {eta_min}min")

            if wandb:
                wandb.log({
                    'loss': current_loss,
                    "ce_loss": ce_loss.item(),
                    'distill_loss': distill_loss.item() if teacher_model is not None else 0.0,
                    'lr': current_lr,
                    'epoch_Time': eta_min
                })

        if(step % args.save_interval == 0 or step == iters -1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth"
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()} #半精度保存
            torch.save(state_dict, ckp)
            lm_checkpoint(
                lm_config_student,
                weight=args.save_weight, 
                model=model, 
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints"
                )
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=512, help="训练的最大截断长度")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
   
    parser.add_argument('--student_hidden_size', default=512, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=16, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    args = parser.parse_args()

    # 1 初始化环境和随机种子
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 配置目录、模型参数，检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_layers,use_moe=bool(args.use_moe))
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layer, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir="../checkpoints")

    # 3 设置混合精度 
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=dtype)

    # 4 配置wandb
    wandb = None
    if args.use_Wandb and is_main_process():
        import swanlab as wandb
        wandb_id= ckp_data.get("wandb_id") if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id,resume=resume)

    # 定义学生和教师模型
    model, tokenizer = init_model(lm_config=lm_config_student, from_weight=args.from_student_weight, device=args.device)
    Logger(f"学生模型总参数:{sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")
    teacher_model,_ = init_model(lm_config=lm_config_teacher,from_weight=args. )
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f"教师模型的总参数：{sum(p.num() for p in model.parameters()) / 1e6:.3f}M")
    train_ds = SFTDataset(args.data_path, tokenizer,max_length=args.max_seq_len)
    train_sampler = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 6 从检查点（ckp -> check point）恢复状态
    start_epoch,start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step',0)
    
    # DDP包模型
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {'freqs_cos', 'freqs_sin'}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 8 开始训练
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_worker=args.num_worker)
            Logger(f"Epoch [{epoch+1}/{args.epochs}]: 跳过前{start_step}个step,从step{start_step + 1}开始")
            train_epoch(
                epoch,
                loader,
                len(loader)+start_step+1,
                teacher_model, 
                wandb,
                args.alpha,
                args.temperature
                )
        else: # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size,shuffle=(train_sampler is None), sampler=train_sampler,num_workers=args.num_workers, pin_memory=True)
            train_epoch(
                epoch,
                loader,
                len(loader),
                teacher_model,
                lm_config_student,
                0, 
                wandb,
                args.aplha,
                args.temperature
            )


