import torch
from typing import Optional
import math


"""此函数预计算旋转位置编码的余弦和正弦值，支持动态的序列长度扩展"""
def precompute_freqs_cis(
        dim: int,
        end: int = int(32*1024),
        rope_base: float=1e6,
        rope_scaling: Optional[dict] = None
):
    # 1、计算基础频率
    # 只使用一半的维度，因为旋转编码是成对应用的
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 2、应用YaRN缩放（如果启用）
    if rope_scaling is not None:
        # 提取缩放参数
        orig_max = rope_scaling.get('original_max_position_embeddings',2048)
        factor  = rope_scaling.get('factor',4)
        beta_fast = rope_scaling.get('beta_fast',4.0)
        beta_slow = rope_scaling.get('beta_slow', 1.0)

        # 需要扩展时应用缩放
        if end / orig_max > 1.0:
            # 找到需要修正的维度（那些波长超过原始最大长度的维度）
            corr_dim = next(
                (i for i in range(dim//2) if 2* math.pi/freqs[i] > orig_max),
                dim // 2 # 默认值
            )

            # 计算每个维度的插值权重[0,1]范围
            power = torch.arange(0,dim//2, device=freqs.device).float()
            power = power/max(dim//2-1,1) #归一化到[0,1]

            # 线性插值计算每个维度的beta值
            beta = beta_slow + (beta_fast - beta_slow)*power

            # 应用YaRN缩放公式
            scale = torch.where(
                torch.arange(dim // 2, device=freqs.device) < corr_dim,
                (beta * factor - beta +1) / (beta * factor),#低频维度缩放
                1.0 / factor #g高频维度缩放
            )

            freqs = freqs * scale

    # 3、生成位置索引并计算外积
    t = torch.arange(end, device=freqs.device) #位置[0,1,2,....end -1]
    freqs = torch.outer(t, freqs).float() # 外积：[end,dim//2]

    # 4、计算余弦和正弦值
    freqs_cos = torch.cos(freqs) # [end,dim//2]
    freqs_sin = torch.sin(freqs) # [end,dim//2]

    # 5、复制并拼接以匹配完整维度
    # 因为旋转编码应用于成对的维度
    freqs_cos = torch.cat([freqs_cos,freqs_cos],dim=-1) # [end,dim]
    freqs_sin = torch.cat([freqs_sin,freqs_sin],dim=-1) # [end,dim]

    return freqs_cos, freqs_sin