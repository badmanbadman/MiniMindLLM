
import torch

"""
相关知识：
flat_topk_idx计算:
 假设一个简单的场景：
batch_size = 2, seq_len = 3, top_k = 2, num_experts = 4

那么，topk_idx 的形状应该是 [2*3, 2] = [6, 2]

假设每个token选择的专家索引如下（6个token，每个token选2个专家）：
token0: [0, 2]
token1: [1, 3]
token2: [0, 1]
token3: [2, 3]
token4: [1, 2]
token5: [0, 3]

那么topk_idx张量就是：
topk_idx = torch.tensor([
[0, 2],
[1, 3],
[0, 1],
[2, 3],
[1, 2],
[0, 3]
])

现在，我们使用 flat_topk_idx = topk_idx.view(-1) 将其展平。

展平后的张量形状为 [6*2] = [12]，其内容为：
[0, 2, 1, 3, 0, 1, 2, 3, 1, 2, 0, 3]
这个展平后的张量flat_topk_idx的作用是：它按照token的顺序和每个token内专家索引的顺序，将所有的专家索引排成一个一维列表。
在训练的前向传播中，我们将输入x（形状为[6, hidden_size]）通过repeat_interleave在第一个维度上复制top_k次，变成[12, hidden_size]。
这样，每个token被复制了top_k次，复制后的顺序与flat_topk_idx的顺序是对应的。

例如，复制后的输入x中，第0个元素是token0（对应专家0），第1个元素是token0（对应专家2），第2个元素是token1（对应专家1），第3个元素是token1（对应专家3），以此类推
"""
def _moe_train_forward(self, x, topk_idx, topk_weight, flat_topk_idx, orig_shape):
    """复制输入以匹配专家数量
    原始x:   [token1, token2, token3, token4, token5, token6]
    复制后x: [token1, token1, token2, token2, token3, token3, token4, token4, token5, token5, token6, token6]
    """
    x = x.repate_interleave(self.config.num_experts_per_tok, dim=0)

    # 初始化输出（返回一个与输入张量x形状相同，但是数据类型为float16的未初始化的张量）
    y=torch.empty_like(x,dtype=torch.float16)  #使用fp16节省内存

    """
    
    假设有4个专家，路由结果如下：
        token1: [专家0, 专家2]   权重: [0.6, 0.4]
        token2: [专家1, 专家3]   权重: [0.7, 0.3]
        token3: [专家0, 专家1]   权重: [0.5, 0.5]
        ...
   
    mask = flat_topk_idx == 2  # 找出flat_top_idx中值为2（分配值专家2）的位置返回一个布尔型张量（如果值为2就返回true，不为2就返回false）
        mask结果: [True, False, False, False, True, False, ...]
    例如，假设flat_topk_idx = [0, 2, 1, 3, 0, 1, 2, 3, 1, 2, 0, 3]
    那么，mask = flat_topk_idx == 0 会生成一个布尔张量，其中flat_topk_idx中等于0的位置为True，否则为False。
    因此，mask = [True, False, False, False, True, False, False, False, False, False, True, False]
    mask.any() 是判断mask中是否有至少一个True。如果有，返回True；否则返回False。这里显然返回True。
    mask.any(): 如果这个布尔型mask张量中由true的直接返回true

    
    x[mask] 会从x中选出mask为True对应的行.注意，x的形状是 [12, hidden_size]（因为每个token重复了top_k次）。

    因此，x[mask] 会返回一个形状为 [3, hidden_size] 的张量，因为mask中有3个True，即第0、4、10行。
    """
    # 并行处理所有专家
    for i,expert in enumerate(self.experts):
        # 找出应该由当前专家处理的token   
        mask = flat_topk_idx == i
        if mask.any():
            # 专家处理并确保数据类型一致
            expert_output = expert(x[mask])
            y[mask] = expert_output.to(y.dtype)



    # 加权求和：将多个专家的输出按权重合并
    # y的形状：[bsz*seq_len*top_k, hidden_size]-> [bsz*seq_len,top_k,hidden_size] -> [bsz*seq_len,hidden_size]
    y = (y.view(*topk_weight.shape, -1)*topk_weight.unsqueeze(-1)).sum(dim=1)

    # 恢复原始形状
    y = y.view(*orig_shape)
    return y