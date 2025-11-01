import torch

def repeat_kv(x: torch.Tensor,n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x,dim=2,repeats=n_rep)"""
    bs,slen,num_key_value_heads,head_dim = x.shape

    if n_rep == 1:
        return x
    # 键值头的数量扩展 n_rep 倍，以匹配查询头的数量
    return(
        x[:,:,:,None,:]  # 添加新维度: [bs, slen, num_kv_heads, 1, head_dim]
        .expand(bs,slen,num_key_value_heads,n_rep,head_dim)  # 扩展: [bs, slen, num_kv_heads, n_rep, head_dim]
        .reshape(bs,slen,num_key_value_heads*n_rep,head_dim) # 重塑: [bs, slen, num_kv_heads * n_rep, head_dim]
    )
