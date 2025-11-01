import torch

"""旋转位置编码应用函数
输入q,k,输出旋转位置编码后的q,k
"""
def apply_rotary_pos_emb(q,k,cos,sin, position_ids=None,unsqueeze_dim=1):
    # 旋转操作的核心实现
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 应用旋转位置编码
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed