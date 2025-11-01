import torch

"""
假设：
batch_size * seq_len = 4（4个token）
top_k = 2
num_experts = 3

假设flat_expert_indices（已经展平）为：
[0, 2, 1, 0, 2, 1, 0, 1] # 长度为4*2=8

假设flat_expert_weights为：
[0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.5, 0.5]

步骤1：对flat_expert_indices排序，得到idxs。
flat_expert_indices: [0, 2, 1, 0, 2, 1, 0, 1]
排序后，idxs为： [0, 3, 6, 2, 5, 7, 1, 4] （这是flat_expert_indices中元素的索引，按照专家编号从小到大排列）
作用：将token按照它们分配的专家进行分组，使相同专家的token连续排列。
示例：
flat_expert_indices = [2, 0, 1, 0, 3, 1, 2, 0]  # 原始专家分配
idxs = [1, 3 , 7, 2, 5, 0, 6, 4]  # 排序后的索引
# 这意味着：
# idxs[0]对应专家0，idxs[1]对应专家0，idxs[2]对应专家0（因为flat_expert_indices[1,3,7]都是0）
# idxs[3,4]对应专家1，idxs[5,6]对应专家2，idxs[7]对应专家3


步骤2：计算每个专家处理的token数量（在排序后的列表中）：
flat_expert_indices[idxs] = [0,0,0,1,1,1,2,2] -> 专家0有3个，专家1有3个，专家2有2个。
# bincount() 统计每个专家出现的次数
bincount = [3,3,2]  # 专家0出现3次，专家1出现3次，专家2出现1次
所以bincount = [3,3,2]，累积和tokens_per_expert = [3,6,8]  (等同于[3,3+3,3+3+2])

步骤3：计算token_idxs = idxs // top_k = [0,1,3,1,2,3,0,2] （因为top_k=2，所以除以2取整）
解释：idxs是展平后的索引，每个token有top_k个，所以除以top_k得到的是原始token的索引。
排序后的位置(idx) | 对应的原始位置 | 计算: idx//2 | 对应的token
----------------------------------------------------------------
       1         |   位置1      |    1//2=0    |   token0
       3         |   位置3      |    3//2=1    |   token1
       2         |   位置2      |    2//2=1    |   token1  
       5         |   位置5      |    5//2=2    |   token2
       0         |   位置0      |    0//2=0    |   token0
       4         |   位置4      |    4//2=2    |   token2

步骤4：遍历专家：
专家0：处理token_idxs[0:3] = [0,1,3] -> 这些是原始输入x中的token索引。
从x中取出这三个token，用专家0处理，然后乘以权重（权重为flat_expert_weights[idxs[0:3]] = [0.6, 0.3, 0.5]）
然后将结果累加到expert_cache的对应位置（0,1,3）上。

专家1：处理token_idxs[3:6] = [1,2,3] -> 权重为flat_expert_weights[idxs[3:6]] = [0.7, 0.2, 0.5]
处理并累加到expert_cache的1,2,3位置。

专家2：处理token_idxs[6:8] = [0,2] -> 权重为flat_expert_weights[idxs[6:8]] = [0.4, 0.8]
处理并累加到expert_cache的0,2位置。

最终，expert_cache中每个位置的值就是所有专家输出的加权和。

"""
@torch.no_grad()
def moe_infer(self,x,flat_expert_idices, flat_expert_weights):
    """推理优化的MoE前向传播"""
    expert_cache=torch.zeros_like(x)

    idxs = flat_expert_idices.argsort()

    # 计算每个专家处理的token数量
    """
    flat_expert_idices 一维的张量
    bincount():  torch的张量方法，用于计算每个整数值出现的次数，返回一个张量，长度是flat_expert_idices的中的最大值加1
    .cup(): 将张量从当前设备（如gpu），移动到cpu，如果张量已经在cpu上，则这个操作不会改变什么
    numpy(): 将pytorch张量转化未numpy数组
    cumsum(0):对Numpy数组进行累加求和，沿着第一个轴（0轴，对于一维数组就是沿着数组顺序累加）
    """
    tokens_per_expert = flat_expert_idices.bincount().cpu.numpy().cumsum(0)

    # 获取token索引，（去除专家重复维度）
    token_idxs = idxs // self.config.num_experts_per_tok

    # 批量处理每个专家的token
    for i, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if i == 0 else tokens_per_expert[i-1]
        if start_idx == end_idx: # 该专家没有token要处理
            continue

        expert = self.experts[i]
        exp_token_idx = token_idxs[start_idx: end_idx]
        expert_tokens = x[exp_token_idx]

        # 专家处理并加权
        """
        expert: 是一个专家模型（通常是一个神经网络模块）
        expert_tokens: 是输入给专家的令牌（数据）
        首先将expert_tokens输入奥expert模型中得到输出，然后将输出转为
            expert_cache.dtype指定的数据类型，（如半精度浮点数float16或者bfloat16）
        结果储存在expert_out中

        flat_expert_weights是一个一维张量，包含了每个令牌对应的权重，(门控网络产生的权重)
        start_idx和end_idx定义了当前专家处理的令牌在idxs中的范围
        flat_expert_weight[idxs[start_idx:end_idx]]会选取出一个权重张量，其形状与expert_out的第一维相同，
        """
        expert_out = expert(expert_tokens).to(expert_cache.dtype)
        expert_out.mul_(flat_expert_weights[idxs[start_idx: end_idx]])

        # 累加到输出缓存
        expert_cache.scatter_add_(
            0,
            exp_token_idx.view(-1,1).repeat(1,x.shape[-1]),
            expert_out
        )

        return expert_cache

