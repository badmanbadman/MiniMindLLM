import torch
from torch import nn

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        """nn.Parameter是PyTorch中用于定义可学习参数的类，
        继承自torch.Tensor,但是拥有特殊的属性：
            自动梯度计算和参数注册
        注意：
            创建可学习参数
            self.weight = nn.Parameter(torch.ones(dim))
            等价于（不推荐）
            self.weight - torch.ones(dim, requires_grad=True)
            #使用nn.Parameter会自动将其注册到参数列表中
        为什么初始化为全1？
            数学原理：保持初始状态为恒等变化
            好处：
                训练开始的时候，归一化层几乎不会改变输入
                梯度稳定，避免初始阶段大幅变化
                模型可以逐渐学习到合适的缩放因子
        """
        # 创建可学习的缩放参数，初始化全部为1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self,x):
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
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self, x):
        """x.float()：将x转化为float类型进行计算，保持数值稳定性
        type_as(x)：再转化为x原来的类型
        这是为了在混合精度训练中保持数值稳定性，因为归一化操作在float32下更稳定
            # 问题场景：混合精度训练
            x = torch.randn(2, 512, dtype=torch.float16)  # 半精度输入

            # 如果没有 .float() 转换：
            x_normalized = self._norm(x)  # 在 float16 中计算

            # 风险：
            # - 小数值的平方可能下溢为0
            # - 除法操作可能数值不稳定
            # - 梯度计算可能不准确
        """
        return self.weight*self._norm(x.float()).type_as(x)
