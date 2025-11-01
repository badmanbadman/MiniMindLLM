import torch


# 创建一个 2×2×3×4 的四维张量
# 可以理解为：2个三维张量，每个三维张量有2个矩阵，每个矩阵是3行4列
tensor_4d = torch.tensor([
    # 第一个三维张量
    [
        # 第一个矩阵 (3×4)
        [
            [-0.2345,  1.5678, -0.4321,  0.9876],
            [ 0.1234, -0.8765,  1.2345, -0.5678],
            [-1.3456,  0.6543, -0.7890,  0.3210]
        ],
        # 第二个矩阵 (3×4)
        [
            [ 0.1111, -0.2222,  0.3333, -0.4444],
            [ 0.5555, -0.6666,  0.7777, -0.8888],
            [ 0.9999, -0.0000,  0.1111, -0.2222]
        ]
    ],
    
    # 第二个三维张量
    [
        # 第一个矩阵 (3×4)
        [
            [ 0.4567, -0.3456,  1.0987, -0.7654],
            [-0.2345,  0.8888, -1.1111,  0.4444],
            [ 0.6666, -0.7777,  0.5555, -0.3333]
        ],
        # 第二个矩阵 (3×4)
        [
            [-0.1111,  0.2222, -0.3333,  0.4444],
            [-0.5555,  0.6666, -0.7777,  0.8888],
            [-0.9999,  0.0000, -0.1111,  0.2222]
        ]
    ]
])

print("四维张量形状:", tensor_4d.shape)
print("四维张量:")
print(tensor_4d)
def apply_rotary_pos_emb(q,k,cos,sin, position_ids=None,unsqueeze_dim=1):
    # 旋转操作的核心实现
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 应用旋转位置编码
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def test_rotary_position_embedding_detailed():
    # 输入值说明：
    # batch_size=2: 2个样本
    # seq_len=4: 序列长度为4  
    # num_heads=3: 3个注意力头
    # head_dim=8: 每个头的维度为8
    batch_size, seq_len, num_heads, head_dim = 2, 4, 3, 8  # 创建查询和键张量
    # q形状: [2, 4, 3, 8] - [batch_size, seq_len, num_heads, head_dim]
    # 使用固定种子以便重现
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)


    # 创建cos和sin位置编码
    # cos形状: [4, 8] - [seq_len, head_dim]
    # sin形状: [4, 8] - [seq_len, head_dim]
    cos = torch.randn(seq_len, head_dim)
    sin = torch.randn(seq_len, head_dim)

    # 期望值: 应用旋转位置编码后，输出形状应与输入相同
    expected_q_shape = q.shape  # 期望: [2, 4, 3, 8]
    expected_k_shape = k.shape  # 期望: [2, 4, 3, 8]

    # 执行函数
    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

      # 验证结果
    assert q_embed.shape == expected_q_shape, f"q形状改变: 期望{expected_q_shape}, 实际{q_embed.shape}"
    assert k_embed.shape == expected_k_shape, f"k形状改变: 期望{expected_k_shape}, 实际{k_embed.shape}"
    print("✓ 基本功能测试通过 - 输出形状符合预期")
    

     # 输入值: 一个简单的6维向量
    # 输入: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    test_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    print(f"输入张量: {test_tensor.tolist()}")
    
    # 期望值: rotate_half应该将后半部分取负并放到前面，前半部分放到后面
    # 前半部分: [1.0, 2.0, 3.0] -> 移动到后半部分
    # 后半部分: [4.0, 5.0, 6.0] -> 取负后移动到前半部分: [-4.0, -5.0, -6.0]
    # 期望输出: [-4.0, -5.0, -6.0, 1.0, 2.0, 3.0]
    expected_rotated = torch.tensor([-4.0, -5.0, -6.0, 1.0, 2.0, 3.0])
    print(f"期望输出: {expected_rotated.tolist()}")