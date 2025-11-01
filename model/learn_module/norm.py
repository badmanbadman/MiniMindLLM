import torch
"""归一化函数解读：
# 1. x.pow(2) - 计算每个元素的平方
#    input: x = [a, b, c, ...]
#    output: [a², b², c², ...]

# 2. .mean(-1, keepdim=True) - 在最后一个维度计算均值
#    input: [a², b², c², ...] 
#    output: mean = (a² + b² + c² + ...) / n

# 3. + self.eps - 添加小的常数防止除零
#    mean_square + ε

# 4. torch.rsqrt() - 计算平方根的倒数
#    rsqrt(z) = 1 / sqrt(z)

# 5. x * ... - 将归一化因子应用到原始输入
#    result = x / sqrt(mean_square + ε)
"""
def norm(x,eps):
    return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)

def test_norm():
    """测试RMS归一化函数"""
    # 测试用例1: 基本功能
    print('测试用例1： 基本功能测试')
    batch_size,seq_len,hidden_size = 2, 3, 4
    """ torch.randn:
    生成服从标准正态分布（均值为0，标准差为1）的随机张量的函数 ，(下为生成三维张量)
    # 创建一个 2×3×4 的三维张量
    # 可以理解为：2个矩阵，每个矩阵是3行4列
    tensor_3d = torch.randn(2, 3, 4)
    三维张量：(2个矩阵，每个矩阵是3行4列)
    tensor([[[-0.2345,  1.5678, -0.4321,  0.9876],
        [ 0.1234, -0.8765,  1.2345, -0.5678],
        [-1.3456,  0.6543, -0.7890,  0.3210]],

        [[ 0.4567, -0.3456,  1.0987, -0.7654],
        [-0.2345,  0.8888, -1.1111,  0.4444],
        [ 0.6666, -0.7777,  0.5555, -0.3333]]])

    NLP:
        # 批次序列数据：（batch_size,seq_len,embedding_dim）
        batch_size = 4
        seq_len = 10
        embedding_dim = 128
        # 模拟词嵌入序列
        word_embeddings = torch.randn(batch_size,seq_len,embedding_dim)
        print("词嵌入序列形状："， word_embeddings.shape) # torch.Size([4,10,128])

        # 访问第一个样本的第5个词的嵌入向量
        first_sample_fifth_word = word_embedding[0,4,:]
        print('第一个样本的第5个词向量形状：' first_sample_fifth_word)

    """
    x = torch.randn(batch_size, seq_len, hidden_size)
    eps = 1e-5
    # 原始值
    original_result = norm(x,eps)
    
    # 手动计算验证
    variance = x.pow(2).mean(dim=-1,keepdim=True) #计算方差
    std = torch.rsqrt(variance + eps) # 计算标准差倒数
    manual_result = x*std

    assert torch.allclose(original_result,manual_result, rtol=1e-6) #基本功能测试失败
    print("✓ 基本功能测试通过")

