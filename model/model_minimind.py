"""📘📘📘📘📘MiniMind config📘📘📘📘📘"""
"""
transformers: Hugging Face 的Transforms库，提供了各种与训练模型和工具
PretrainedConfig: 所有与训练模型配置的基类，提供了序列化，保存，加载等通用的功能
"""
from transformers import PretrainedConfig

class MiniMindConfig(PretrainedConfig):
    # 根据这个字段知道应该加载哪个具体的模型类
    model_type = "minimind"

    """
    MoE 工作流程比喻
    想象一个医院分诊系统：
    n_routed_experts = 4 → 有4个专科医生(心内科、神经科等)
    num_experts_per_tok = 2 → 每个病人同时看2个相关专科
    n_shared_experts = 1 → 还有1个全科医生处理所有病例
    scoring_func = 'softmax' → 根据症状评分决定分诊到哪个科室
    """
    def __init__(
            self,
            # 核心Transformer参数
            dropout: float = 0.0,           # Dropout概率，防止过拟合
            bos_token_id: int = 1,          #序列开始标记的ID
            eos_token_id: int = 2,          #序列结束标记的ID
            hidden_act: str = 'silu',       #隐藏层的激活函数，silu比RelU更加平滑
            hidden_size: int = 512,         #隐藏层维度，每个token的向量大小
            intermediate_size: int = None,  #FFN中间层维度（前馈网络维度），通常师hidden_size的4倍数
            max_position_embeddings: int = 32768, #模型能处理的最大序列长度
            num_attention_heads: int = 8,   #注意力头数
            num_hidden_layers: int = 8,     #Transformer层数
            num_key_value_heads: int = 2,   #用于分组查询注意力（GQA）的key和value的注意力头数。如果与num_attention_heads不同，则使用GQA
            vocab_size: int =6400,          #词汇表的大小
            # 归一化和位置编码
            rms_norm_eps: float = 1e-05,    #RMSNorm的epsilon值，用于数值稳定性 防止除零错误 梯度稳定性
            rope_theta: int = 1000000.0,    #RoPE(旋转位置编码)的theta参数， 确保在长序列中不同位置有足够的区分度
            inference_rope_scaling: bool = False,# 是否在推理时候使用RoPE缩放（用于扩展上下文长度），如果额外iTrue，则设置一个RoPE缩放配置字典
            flash_attn: bool = True,        #是否使用Flash Attention，*(一种高效的注意力实现，更快的内存访问)
            # MoE（混合专家）配置  use_moe为false的时候下面配置不生效
            use_moe: bool = False,          #是否启用MoE
            num_experts_per_tok: int = 2,   #每个token使用几个专家
            n_routed_experts: int = 4,      #路由专家的总数
            n_shared_experts: int = 1,      # 共享专家的数量（所有token都会使用）
            scoring_func: str = 'softmax',  #专家选择评分函数
            aux_loss_alpha: float = 0.1,    #辅助损失权重（平衡专家使用）
            seq_aux: bool = True,           #序列级辅助损失计算
            norm_topk_prob: bool = True,    #是否对top-k概率归一化
            **kwargs
    ):      
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attn = flash_attn
        # 外推长度 = factor*original_max_position_embeddings
        self.rope_scaling = {
            'beta_fast': 4, # 快速beta参数(YaRN方法)
            'beta_slow': 1, # 慢速beta参数
            'factor': 4,    # 缩放因子
            'original_max_position_embeddings': 2048, # 原始训练长度
            'type': 'yarn'  # 缩放类型：YaRN方法
        } if self.inference_rope_scaling else None

        # MoE（混合专家）配置  use_moe为false的时候下面配置不生效
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts        #总的专家数量
        self.n_shared_experts = n_shared_experts        #共享专家
        self.scoring_func = scoring_func                #评分函数， 默认为‘softmax’
        self.aux_loss_alpha = aux_loss_alpha            #辅助损失函数的alpha参数
        self.seq_aux = seq_aux                          #是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob            #是否标准化top-k概率


"""📘📘📘📘📘MiniMind Model📘📘📘📘📘"""
import math
import torch
"""torch.nn.init提供了：
1、多种初始化的方法:适应不同的网络机构和激活函数
2、训练稳定性:防止梯度消失/爆炸
3、收敛加速: 适合的初始化让模型更快收敛
4、专业配置: 针对Transformer，MoE等特定结构的优化初始化
"""
import torch.nn.init as init
"""torch.nn.functional提供了大量的无状态函数如：
1、激活函数
2、注意力机制
3、dropout和归一化
4、损失函数
5、高效计算操作，如top-k选择，矩阵操作等
6、优化函数：Flash Attention，
这个模块提供了构建现代Transformer和MoE模型所需的所有基础函数式组件
"""
import torch.nn.functional as F
"""torch.nn神经网络模块，提供了构建神经网络所需的所有基础组件，包含了：
1、模块构架基础：nn.module基类
2、丰富的层类型：线性层，嵌入层，归一化层等等
3、组织容器： moduleList，Sequential，ModuleDict
4、参数管理：自动跟踪，设备移动，状态保存
5、训练工具：模式切换，梯度管理
"""
from torch import nn
"""ACT2FN是一个字典，将字符串标识符映射到对应的激活函数
# 查看所有可用的激活函数
print("Available activation functions:")
for act_name in ACT2FN.keys():
    print(f"  - {act_name}")

# 典型输出：
# Available activation functions:
#   - gelu
#   - gelu_new
#   - gelu_fast
#   - quick_gelu
#   - gelu_python
......
eg：使用配置中的激活函数
self.activation = ACT2FN[config.hidden_act]
"""
from transformers.activations  import ACT2FN
"""Union类型: 联合类型,  用于表示一个值可以是几种类型中的一种。
例如，Union[int, str]表示一个值可以是整数或字符串
"""
from typing import Optional, Tuple, List, Union
"""PreTrainedModel是Hugging Face Transformers库中国所有预训练模型的基类，提供了模型管理的基础设施
1、模型序列化功能（将模型保存为文件，加载模型文件）
2、参数管理
3、与配置类集成
GeneerationMixin混入类
文本生成功能： 为模型添加了各种文本生成方法，是构建生成式语言模型的关键
"""
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
"""CausalLMOutputWithPast提供了：
1、标准化的输出格式：与hugging Face生态系统兼容
2、生成优化：包含past_key_values用于高效的自回归生成
3、训练支出：包含损失计算和梯度反向传播所需的所有信息
4、调试能力：提供隐藏状态和注意力权重的访问
5、扩展性：支持MoE等特殊架构的额外输出字段
这个输出类可以让因果语言模型能够无缝集成大Transformer的训练、评估和生成管道中
"""
from transformers.modeling_outputs import CausalLMOutputWithPast
from .model_minimind_module import RMSNorm,  precompute_freqs_cis,apply_rotary_pos_emb, repeat_kv

class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        #分组查询注意力（GQA）
        """
        分组查询注意力GQA
        目的：检查用户是否显示的指定了num_key_value_heads(KV)头数量
        如果为None：使用与query头相同的数量（qkv头数相等）（传统多头注意力）
        如果有值：使用用户指定的KV头数量（分组查询注意力）
        """
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0  # 必须整除

        self.n_local_heads = args.num_attention_heads #查询头数
        self.n_local_kv_heads = self.num_key_value_heads # 键值头数


        self.n_rep = self.n_local_heads // self.n_local_kv_heads # 重复次数

        #维度配置
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 投影层
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim,bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim,bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # flash支持
        self.flash = hasattr(torch.nn.functional,'scaled_dot_product_attention') and args.flash_attn

        # 输入投影和重塑
    def forward(self,x,position_embeddings, past_key_value=None,use_cache= False, attention_mask=None):
            bsz, seq_len,_ = x.shape

            # 投影到Q、K、V

            xq,xk,xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

            # 重塑为多头格式
            xq = xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)
            xk = xk.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)
            xv = xv.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)

            # 旋转位置编码应用
            # 获取预计算的cos和sin
            cos,sin = position_embeddings
            # 应用旋转位置编码（只对前seq_len个位置）
            xq,xk = apply_rotary_pos_emb(xq,xk,cos[:seq_len], sin[:seq_len])

            # 键值缓存机制
            # KV缓存实现（用于生成任务）
            if past_key_value is not None:
                # 拼接历史缓存和当前键值
                xk = torch.cat([past_key_value[0],xk],dim=1) #在序列维度拼接
                xv = torch.cat([past_key_value[1],xv],dim=1)

            past_kv = (xk, xv) if use_cache else None 

            # GQA 处理和转置
            # 转置并重复键值头以匹配查询头数量
            xq,xk,xv = (
                xq.transpose(1,2),
                repeat_kv(xk, self.n_rep).transpose(1,2),
                repeat_kv(xv, self.n_rep).transpose(1,2)
            )

            # 双路径注意力计算
            # 1、Flash Attention路径
            if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
                # 准备Flash Attention需要的掩码
                attn_mask = (
                    None
                    if attention_mask is None
                    else attention_mask.view(bsz,1,1,-1).expand(bsz,self.n_local_heads,seq_len,-1).bool()
                )
                # 使用pytorch的高效注意力（qkv之间的矩阵计算逻辑，原理与标准注意力计算一样，但是做了很多优化）
                output = F.scaled_dot_product_attention(
                    xq,xk,xv,
                    attn_mask=attn_mask, 
                    dropout_p=self.dropout if self.training else 0.0, 
                    is_causal=True # 自动应用因果掩码
                    )
            else:
                # 标准注意力路径
                # 手动计算注意力分数   转置是为了重组K的维度，使其能与Q进行有意义的矩阵乘法
                scores = (xq @ xk.transpose(-2,-1)) / math.sqrt(self.head_dim)
                # 应用因果掩码（防止关注未来位置）【创建一个下三角矩阵】
                scores = scores + torch.triu(
                    torch.full((seq_len,seq_len), float('-inf'),device=scores.device),
                    diagonal=1
                ).unsqueeze(0).unsqueeze(0)  #scores + mask

                # 应用额外的注意力掩码（如填充码）
                if attention_mask is not None:
                    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                    scores = scores + extended_attention_mask
                
                # 计算注意力权重
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                scores = self.attn_dropout(scores)

                # 应用注意力权重到值
                output = scores @ xv
            
            # 输出处理
            output = output.transpose(1,2).reshape(bsz,seq_len, -1)

            # 输出投影和残差dropout
            output = self.resid_dropout(self.o_proj(output))
            """
            Attention的输出是：每个token的上下文感知的新表示
            对比输入输出
            输入：
            # 每个token的原始表示（不考虑上下文）
            token0_vec = [原始特征0, 原始特征1, ...]  # 只包含token自身信息
            token1_vec = [原始特征0, 原始特征1, ...]
            输出：
            # 每个token的上下文感知表示
            token0_new_vec = [上下文特征0, 上下文特征1, ...]  # 融合了序列中所有相关信息
            token1_new_vec = [上下文特征0, 上下文特征1, ...]
            """

            return output, past_kv

class FeedForward(nn.Module):
    # 门控前馈网络
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 动态计算中间层维度
        """
        8/3 ≈ 2.666 是经验性的比例因子
        意味着中间层维度大约是隐藏层的2.67倍
        config.intermediate_size = 64*((intermeditate_size + 64 -1) // 64)
        向上取整到64倍数的经典算法
        具体计算示例：
        示例1：hidden_size = 512
        步骤1-计算基础值
        intermediate_size = int(512*8/3) = int(1365.33) = 1365
        步骤2-向上取整到64的倍数
        (1365 + 64 -1) // 64 = (1428)//64 = 22
        64*22 = 1408
        """
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 /3)
            config.intermediate_size = 64*((intermediate_size + 64 -1) // 64)
        
        # 三个投影层
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size,bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size,bias=False)
        self.up_proj = nn.Linear(config.hidden_size,config.intermediate_size,bias=False)

        # dropout正则化
        self.dropout = nn.Dropout(config.dropout)
        # 激活函数
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self,x):
        """门控FFN
        传统FFN vs 门控FFN
        传统FFN: output = activation(linear1(x))  #所有特征同等处理
        门控FFN: output = activation(linear1(x)) * linear2(x) # 动态控制信息流
        门控的优势：
        动态特征选择：
        1、gate_output决定哪些特征应该被强调或者抑制
        类似与注意力机制，但是在特征级别
        2、更加丰富的表达能力
        可以学习复杂的特征交互
        比简单的前馈网络有更强的非线性能力
        3、训练稳定性
        门控机制可以防止梯度消失（参考理解LSTM门控机制）
        让网络更容易学习恒等映射（当gate≈1时），输出的信息基本保持不变
        """
        # return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
        # 1、门控投影,x的维度是512，是由注意力层的8个64维的矩阵拼接而成的512维 
        gate = self.gate_proj(x)

        # 2、上投影（升高维度），这个x也是512维，是由注意力层的8个头的84维拼接而成，这个up用来将x升高维度，保存更加丰富的信息，
        up = self.up_proj(x)

        # 3、激活函数应用到门控信号，这个是将门控矩阵进行激活函数计算，判断哪些信息需要保留哪些不需要，
        # 产生0-1之间的门控权重
        # 决定哪些特征应该被保留/抑制
        activated_gate = self.act_fn(gate) 

        # 4、元素级乘法（门控机制，升维后与门控投影激活后计算），将门控线性矩阵经过激活函数计算后的值与升维后的值进行计算，值就是要保留的值，
        gate_output = activated_gate * up

        # 5、下投影回原始维度，将要保留的值进行降维，
        down = self.down_proj(gate_output)

        # 6、Dropout正则化
        output = self.dropout(down)
        return output

class MoEGate(nn.Module): 
    def __init__(self, config: MiniMindConfig):
        super().__init__() 
        self.config = config

        # 专家选择配置
        self.top_k = config.num_experts_per_tok #每个token选择的专家数量
        self.n_routed_experts = config.n_routed_experts #路由专家的总数

        # 评分和损失配置
        self.scoring_func = config.scoring_func # 专家评分函数类型
        self.aplha = config.aux_loss_alpha #辅助损失权重
        self.seq_aux = config.seq_aux #是否计算序列级辅助损失

        # 概率归一化
        self.norm_topk_prob = config.norm_topk_prob # 是否归一化top—k概率
        self.gating_dim = config.hidden_size #门控维度512

        # 门控权重
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.gating_dim))
        self.reset_parameters()
    
    def reset_parameters(self)-> None:
        """Kaiming初始化：
        也称为He初始化
        由何凯明在2015年提出，专门针对ReLU族激活函数优化
        解决了深层网络中梯度消失/爆炸问题

        标准KaiMing初始化
        对于均匀分布U(-bound,bound)
        bound = gain*sqrt(3/fan_in)

        初始化策略的核心目标是：确保信号在网络中传播时方差保持稳定
        # 前向传播: y = x @ W
        # 如果 x 的方差是 Var(x)，W 的方差是 Var(W)
        # 那么 y 的方差 ≈ fan_in * Var(x) * Var(W)

        # 为了保持方差稳定，需要：
        fan_in * Var(W) = 1
        # 因此：Var(W) = 1 / fan_in

        # 内部计算过程：
        # gain = sqrt(2.0 / (1 + a^2))  # 对于LeakyReLU
        # 由于 a = sqrt(5)，所以：
        # gain = sqrt(2.0 / (1 + 5)) = sqrt(2/6) = sqrt(1/3) ≈ 0.577

        # 然后计算边界：
        # bound = gain * sqrt(3 / fan_in)
        #       = sqrt(1/3) * sqrt(3 / fan_in)
        #       = sqrt(1 / fan_in)

        信息输入量
        # fan_in 越大，意味着：
        # - 每个神经元接收更多输入信息
        # - 需要更小的权重来避免输出爆炸
        # - 因此初始化范围应该更小

        梯度流动
        # 反向传播时：
        # ∂Loss/∂W = ∂Loss/∂y * x^T

        # fan_in 影响梯度的大小：
        # 更大的 fan_in → 更多项求和 → 可能梯度更大
        # 需要通过初始化来平衡

        网络深度的影响
        # 在深层网络中：
        # 每层的 fan_in 通常是前一层的 fan_out
        # 不恰当的初始化会导致梯度指数级变化（消失或爆炸）
        """
        init.kaiming_uniform(self.weight, a=math.sqrt(5)) 
    
    def forward(self,hidden_states):
        # 输入处理和评分计算
        bsz, seq_len, h = hidden_states.shape

        hidden_states = hidden_states.view(-1,h)
        """展平输入 [batch_size, seq_len,hidden_size]-> [batch_size*seq_len,hidden_size]
        # 要理解批次：承载的序列的个数，
        # 序列：承载的是token的个数
        # view: 要求内存连续，速度快
        hidden_states.view(-1, h)

        # reshape: 自动处理内存连续性，更安全但稍慢  
        hidden_states.reshape(-1, h)


        # 假设原始形状：
        hidden_states.shape = [2, 8, 512]  # [bsz, seq_len, h]
        变换前
        # 数据在内存中的逻辑结构：
        [
            # 批次0
            [[token0_0, token0_1, ..., token0_511],  # 序列位置0
            [token1_0, token1_1, ..., token1_511],  # 序列位置1
            ...,
            [token7_0, token7_1, ..., token7_511]], # 序列位置7
            
            # 批次1  
            [[token8_0, token8_1, ..., token8_511],
            [token9_0, token9_1, ..., token9_511],
            ...,
            [token15_0, token15_1, ..., token15_511]]
        ]
        变换后
        hidden_states = hidden_states.view(-1, 512)  # [16, 512]

        # 现在的逻辑结构：
        [
            [token0_0, token0_1, ..., token0_511],   # 批次0, 位置0
            [token1_0, token1_1, ..., token1_511],   # 批次0, 位置1
            ...,
            [token7_0, token7_1, ..., token7_511],   # 批次0, 位置7
            [token8_0, token8_1, ..., token8_511],   # 批次1, 位置0
            [token9_0, token9_1, ..., token9_511],   # 批次1, 位置1
            ...,
            [token15_0, token15_1, ..., token15_511] # 批次1, 位置7
        ]

        根据可用内存计算最优的批次和序列配置
        def calculate_optimal_config(available_memory):
        # 根据可用内存计算最优的批次和序列配置
        max_tokens = available_memory / bytes_per_token
        
        # 在批次和序列间权衡
        possible_configs = []
        for seq_len in [512, 1024, 2048, 4096]:
            batch_size = max_tokens // seq_len
            if batch_size >= 1:
                possible_configs.append((batch_size, seq_len))
        
        return possible_configs
        """
      
        logits = F.linear(hidden_states, self.weight, None) 
        #[batch_size*seq_len,n_routed_experts]
        # 计算hidden_states对应的，每个专家的分数 hidden_states @ weigth.T
        """ logits 
        注意权重的形状：
        self.weight.shape = [n_routed_experts,h] # [4,512]
        而不是传统的[h,n_routed_experts]

        这样设计是因为
        logits = hidden_states @ self.weight.T # [16,512] @ [512,4] = [16,4]
        #每个token得到4个专家的分数
        
        与普通全连接层的区别
            普通全连接层（线性层）：
                参考门控前馈网络层和传统前馈网络层
                用于特征变化
                self.fc = nn.Linear(512,256)  # 512 -> 256维度
                output = self.fc(hidden_states) # 特征的升维/降维
            MoE门控线性层
                #用于专家选择（一个专家就是一个全连接层（前馈网络层））
                logits = F.linear(hidden_states,self.weight,None) # 512 -> 4维
                #目的不是特征变化，而是为了计算选择分数
        计算复杂度：
            计算量 = bsz * seq_len* h* n_routed_experts
            示例：16 * 512 * 4 = 32768次乘加运算
            
            相对于整个模型来说计算量很小
            但是却是MoE路由决策的核心
        总结：
        专家评分：为每个token计算对所有专家的"偏好分数"
        路由决策：基于这些分数决定每个token使用哪些专家
        可学习路由：通过训练优化权重，让专家专业化
        高效计算：批量处理所有token，充分利用硬件并行性
        """
        if self.scoring_func == 'softmax':
            # 将专家分数矩阵转化为概率矩阵（走一下激活函数）
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(f'不支持评分函数，{self.scoring_func}')
        
        # 专家选择  
        """
        torch.topk() :  根据top_k来设置具体的个数
        这个函数返回一个张量中指定维度上最大（最小）的k个值，
        以及这些值对应的索引位置
        """
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        # topk_weight: [batch_size*seq_len,top_k] -选中的专家权重
        # topk_idk:    [batch_size*seq_len,top_k] -选中的专家索引
        """具体计算过程
        输入scores示例
        假设有4个token，4个专家，top_k=2：
        scores = [
            [0.10, 0.60, 0.25, 0.05],  # 专家1最高(0.6), 专家2次高(0.25) # token0对4个专家的概率
            [0.40, 0.30, 0.20, 0.10],  # 专家0(0.4), 专家1(0.3) # token1对4个专家的概率  
            [0.05, 0.10, 0.70, 0.15],  # 专家2(0.7), 专家3(0.15)# token2对4个专家的概率
            [0.25, 0.25, 0.25, 0.25]   # 任意两个专家(各0.25)# token3对4个专家的概率
        ]
        Top-K操作结果
        topk_weight, topk_idx = torch.topk(scores, k=2, dim=-1, sorted=False)

        # topk_idx (选择的专家索引):
        [
            [1, 2],  # token0: 选择专家1和专家2
            [0, 1],  # token1: 选择专家0和专家1
            [2, 3],  # token2: 选择专家2和专家3
            [0, 1]   # token3: 选择专家0和专家1 (任意两个)
        ]

        # topk_weight (对应的权重):
        [
            [0.60, 0.25],  # token0: 专家1权重0.6, 专家2权重0.25
            [0.40, 0.30],  # token1: 专家0权重0.4, 专家1权重0.3
            [0.70, 0.15],  # token2: 专家2权重0.7, 专家3权重0.15
            [0.25, 0.25]   # token3: 两个专家各0.25
        ]
        """

        # 归一化top-k概率（如果启用）
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1,keepdim=True) + 1e-20 #防止除零
            topk_weight = topk_weight/denominator #归一化，使得选中的k个专家权重和为1

        # 训练时计算辅助损失，确保专家负载均衡
        if self.training and self.aplha > 0.0:
            scores_for_aux = scores
            aux_topx = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz,-1)
            """形状变换详解
            变换前: 
            topk_idx.shape = [bsz*seq_len, top_k]  # [16, 2]
            # 16个token，每个token选择2个专家
            变换后
            topk_idx_for_aux_loss.shape = [bsz, seq_len * top_k]  # [2, 16]
            # 2个批次，每个批次有16个专家选择（8个token × 每个选2个专家）
            具体数值示例
            假设：
            bsz = 2 (2个序列)
            seq_len = 8 (每个序列8个token)
            top_k = 2 (每个token选2个专家)

            变换前数据:
                # topk_idx: [16, 2] - 16个token的专家选择
                topk_idx = [
                    # 批次0的8个token
                    [1, 2], [0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [2, 1], [3, 0],
                    # 批次1的8个token  
                    [0, 2], [1, 3], [2, 0], [3, 1], [0, 1], [2, 3], [1, 0], [3, 2]
                ]
            变换后数据
                topk_idx_for_aux_loss = topk_idx.view(2, -1)
                # 形状: [2, 16]

                topk_idx_for_aux_loss = [
                    # 批次0: 所有16个专家选择（8token×2专家）
                    [1, 2, 0, 1, 2, 3, 0, 2, 1, 3, 0, 3, 2, 1, 3, 0],
                    
                    # 批次1: 所有16个专家选择
                    [0, 2, 1, 3, 2, 0, 3, 1, 0, 1, 2, 3, 1, 0, 3, 2]
                ]
            """

            if self.seq_aux:
                #序列级辅助损失
                aux_loss = self._compute_sequence_aux_loss(
                    scores_for_aux,
                    topk_idx_for_aux_loss,
                    bsz,
                    seq_len,
                    aux_topx,
                    hidden_states.device
                )
            else:
                # Token级辅助损失
                aux_loss = self._compute_token_aux_loss(
                    scores_for_aux, 
                    topk_idx_for_aux_loss,
                    bsz, 
                    seq_len,
                    aux_topx,
                    hidden_states.device
                )
        else:
            aux_topx = 0
        return topk_idx, topk_weight, aux_loss

    # 计算序列级辅助损失
    def _compute_sequence_aux_loss(
            self, 
            scores_for_aux,
            topk_idx_for_aux_loss, 
            bsz, 
            seq_len, 
            aux_topk, 
            device
    ):
       
        scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
        """形状变化详解：
        变化前：
            scores_for_aux.shape = [bsz*seq_len,n_routed_experts] # [16*4]
            # 展平后的所有token对专家的原始分数
        变化后：
            scores_for_seq_aux.shape = [bsz,seq_len,n_routed_experts] # [2,8,4]
            #恢复批次和序列维度，保持专家维度 
        具体数值示例
        假设：
        bsz = 2 (2个序列)
        seq_len = 8 (每个序列8个token)
        n_routed_experts = 4 (4个专家)
        变换前数据
        # scores_for_aux: [16, 4] - 16个token对4个专家的原始分数
        scores_for_aux = [
            # 批次0的8个token
            [0.10, 0.60, 0.25, 0.05],  # token0
            [0.40, 0.30, 0.20, 0.10],  # token1
            [0.05, 0.10, 0.70, 0.15],  # token2
            [0.25, 0.25, 0.25, 0.25],  # token3
            [0.15, 0.50, 0.20, 0.15],  # token4
            [0.30, 0.20, 0.40, 0.10],  # token5
            [0.20, 0.30, 0.10, 0.40],  # token6
            [0.10, 0.20, 0.30, 0.40],  # token7
            
            # 批次1的8个token
            [0.50, 0.20, 0.20, 0.10],  # token8
            [0.25, 0.25, 0.25, 0.25],  # token9
            [0.10, 0.60, 0.20, 0.10],  # token10
            [0.30, 0.30, 0.20, 0.20],  # token11
            [0.15, 0.25, 0.35, 0.25],  # token12
            [0.40, 0.30, 0.15, 0.15],  # token13
            [0.20, 0.40, 0.25, 0.15],  # token14
            [0.10, 0.15, 0.35, 0.40]   # token15
        ]
        变换后数据
        scores_for_seq_aux = scores_for_aux.view(2, 8, 4)
        # 形状: [2, 8, 4]

        scores_for_seq_aux = [
            # 批次0
            [
                [0.10, 0.60, 0.25, 0.05],  # 位置0
                [0.40, 0.30, 0.20, 0.10],  # 位置1
                [0.05, 0.10, 0.70, 0.15],  # 位置2
                [0.25, 0.25, 0.25, 0.25],  # 位置3
                [0.15, 0.50, 0.20, 0.15],  # 位置4
                [0.30, 0.20, 0.40, 0.10],  # 位置5
                [0.20, 0.30, 0.10, 0.40],  # 位置6
                [0.10, 0.20, 0.30, 0.40]   # 位置7
            ],
            
            # 批次1
            [
                [0.50, 0.20, 0.20, 0.10],  # 位置0
                [0.25, 0.25, 0.25, 0.25],  # 位置1
                [0.10, 0.60, 0.20, 0.10],  # 位置2
                [0.30, 0.30, 0.20, 0.20],  # 位置3
                [0.15, 0.25, 0.35, 0.25],  # 位置4
                [0.40, 0.30, 0.15, 0.15],  # 位置5
                [0.20, 0.40, 0.25, 0.15],  # 位置6
                [0.10, 0.15, 0.35, 0.40]   # 位置7
            ]
        ]

        （回顾下topK_idx_for_aux_loss）
        topk_idx_for_aux_loss = [
            # 批次0: 所有16个专家选择（8token×2专家）
            [1, 2, 0, 1, 2, 3, 0, 2, 1, 3, 0, 3, 2, 1, 3, 0],
            
            # 批次1: 所有16个专家选择
            [0, 2, 1, 3, 2, 0, 3, 1, 0, 1, 2, 3, 1, 0, 3, 2]
        ]
        """

        # 计算每个batch中每个专家的使用频率
        ce = torch.zeros(bsz, self.n_routed_experts,device=device)
        """
        target.scatter_add_(dim, index, source)
        在dim维度上，将source的值按照index指定的位置累加到target中
        具体计算规则
        对于每个位置(i, j)：
        target[i][index[i][j]] += source[i][j]
        最终结果：
        ce = [
            [3, 3, 2],  # 批次0: 专家0被选3次，专家1被选3次，专家2被选2次
            [2, 4, 2]   # 批次1: 专家0被选2次，专家1被选4次，专家2被选2次
        ]
        物理意义：专家选择统计
        负载分布分析
        # 理想情况（完全均衡）：
        期望选择次数 = (seq_len * aux_topk) / n_routed_experts
                    = (4 * 2) / 3 ≈ 2.67次/专家

        # 实际结果：
        批次0: [3, 3, 2] → 相对均衡
        批次1: [2, 4, 2] → 专家1过载，专家0和2欠载
        """
        ce.scatter_add_(
            1,
            topk_idx_for_aux_loss, #索引
            torch.ones(bsz, seq_len*aux_topk, device=device) #值
        )
        ce.div_(seq_len*aux_topk/self.n_routed_experts) #归一化

        # 计算损失：使用频率*平均分数
        aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean()*self.aplha
        return aux_loss
   
    # Token级辅助损失实现
    def _compute_token_aux_loss(self, scores_for_aux, topk_idx_for_aux_loss, bsz, seq_len, aux_topk, device):
        # 创建one-hot编码的掩码
        mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1),num_classes=self.n_routed_experts)

        # 计算每个专家的实际使用频率
        ce = mask_ce.float().mean()

        # 计算每个专家的平均选择频率
        Pi = scores_for_aux.mean(0)

        # 计算理想使用频率的偏差
        fi = ce*self.n_routed_experts #缩放以匹配期望值

        # 计算辅助损失

        aux_loss = (Pi * fi).sum() * self.aplha

        return aux_loss

class MOEFeedForward(nn.Module):
    def __init__(self, config:MiniMindConfig):
        super().__init__()
        self.config = config
        # 专家网络列表
        # nn.ModuleList 是PyTorch中用于存储子模块的特殊容器
        # 与普通Python列表不同，它能确保其中的模块被正确注册到模型中
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 门控网络（计算出来的需要参与计算的专家）
        self.gate = MoEGate(config)
        # 共享专家（可选）
        if config.n_shared_experts > 0:
            self.share_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])
    
    # 训练和推理的双路径设计
    def forward(self,x):
        identity = x # 保存原始输入，用于残差链接（恒等映射）
        orig_shape = x.shape #原始输入的形状
        bsz,seq_len,_ = x.shape #解构原始输入，批次大小，序列长度

        # 1、门控选择专家
        topk_idx,topk_weight,aux_loss = self.gate(x)

        # 2、展平输入以便处理
        x = x.view(-1,x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        """ 展平详解
        输入张量x展平：
            变化前:
                x.shape = [bsz, seq_len, hidden_size] # [2,8,512]
                #2个批次，每个批次8个token，每个token 512维
            变化后:
                x = x.view(-1,x.shape[-1])  # [16,512]
                #16个token，每个token 512维
                #-1 表示自动极端： 2*8 = 16
        专家索引展平
            变化前：
                topk_idx.shape = [bsz*seq_len,top_k] # [16,2]
                #16个token，每个token选择2个专家
            变化后：
                flat_topk_idx = topk_idx.view(-1)  #[32]
                #32个专家选择（16个token  x  每个选2个专家）
                #-1表示自动计算：16*2 = 32
        具体数值示例：
        假设：
        bsz = 2, seq_len = 4, hidden_size = 512, top_k = 2
        输入数据
        # x (原始输入)
        x = [
            # 批次0
            [[t0_0, t0_1, ..., t0_511],  # token0
            [t1_0, t1_1, ..., t1_511],  # token1
            [t2_0, t2_1, ..., t2_511],  # token2
            [t3_0, t3_1, ..., t3_511]], # token3
            
            # 批次1
            [[t4_0, t4_1, ..., t4_511],  # token4
            [t5_0, t5_1, ..., t5_511],  # token5
            [t6_0, t6_1, ..., t6_511],  # token6
            [t7_0, t7_1, ..., t7_511]]  # token7
        ]

        # topk_idx (专家选择)
        topk_idx = [
            # 批次0的token专家选择
            [1, 2],  # token0: 专家1, 专家2
            [0, 1],  # token1: 专家0, 专家1
            [2, 3],  # token2: 专家2, 专家3
            [0, 2],  # token3: 专家0, 专家2
            
            # 批次1的token专家选择  
            [1, 3],  # token4: 专家1, 专家3
            [0, 2],  # token5: 专家0, 专家2
            [1, 0],  # token6: 专家1, 专家0
            [3, 2]   # token7: 专家3, 专家2
        ]

        变换后数据
        # x展平后
        x_flat = [
            [t0_0, t0_1, ..., t0_511],  # token0
            [t1_0, t1_1, ..., t1_511],  # token1
            [t2_0, t2_1, ..., t2_511],  # token2
            [t3_0, t3_1, ..., t3_511],  # token3
            [t4_0, t4_1, ..., t4_511],  # token4
            [t5_0, t5_1, ..., t5_511],  # token5
            [t6_0, t6_1, ..., t6_511],  # token6
            [t7_0, t7_1, ..., t7_511]   # token7
        ]  # [8, 512]

        # flat_topk_idx展平后
        flat_topk_idx = [
            1, 2,  # token0的2个专家
            0, 1,  # token1的2个专家  
            2, 3,  # token2的2个专家
            0, 2,  # token3的2个专家
            1, 3,  # token4的2个专家
            0, 2,  # token5的2个专家
            1, 0,  # token6的2个专家
            3, 2   # token7的2个专家
        ]  # [16] (8个token × 2个专家 = 16个选择)
        """

        # 训练和推理使用不同的路径
        if self.training:
            y = self._moe_train_forward(
                x,
                topk_idx,
                topk_weight,
                flat_topk_idx,
                orig_shape
            )
        else:
            y = self.moe_infer(
                x,
                flat_topk_idx,
                topk_weight.view(-1,1).view(*orig_shape)
            )
        
        # 4、添加共享专家输出
        if self.config.n_shared_experts > 0:
            for expert in self.share_experts:
                y = y + expert(identity)
        
        self.aux_loss = aux_loss

        return y

    def _moe_train_forward(self, x, topk_idx, topk_weight, flat_topk_idx, orig_shape):
        """训练模式下的前向传播"""
        # 将每个token重复top_k次，以便并行处理所有选中的专家（！！！因为每一个token都是要top_k个专家来进行处理的）
        # num_experts_per_tok 每个token由几个专家处理
        x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
        """x_flat_repeat = [
                [t0_0, t0_1, ..., t0_511],  # token0
                [t0_0, t0_1, ..., t0_511],  # token0
                [t1_0, t1_1, ..., t1_511],  # token1
                [t1_0, t1_1, ..., t1_511],  # token1
                [t2_0, t2_1, ..., t2_511],  # token2
                [t2_0, t2_1, ..., t2_511],  # token2
                [t3_0, t3_1, ..., t3_511],  # token3
                [t3_0, t3_1, ..., t3_511],  # token3
                [t4_0, t4_1, ..., t4_511],  # token4
                [t4_0, t4_1, ..., t4_511],  # token4
                [t5_0, t5_1, ..., t5_511],  # token5
                [t5_0, t5_1, ..., t5_511],  # token5
                [t6_0, t6_1, ..., t6_511],  # token6
                [t6_0, t6_1, ..., t6_511],  # token6
                [t7_0, t7_1, ..., t7_511]   # token7
                [t7_0, t7_1, ..., t7_511]   # token7
            ]  # [16, 512]
            与flat_topk_idx一 一对应
            flat_topk_idx = [
                1, 2,  # token0的2个专家
                0, 1,  # token1的2个专家  
                2, 3,  # token2的2个专家
                0, 2,  # token3的2个专家
                1, 3,  # token4的2个专家
                0, 2,  # token5的2个专家
                1, 0,  # token6的2个专家
                3, 2   # token7的2个专家
            ]  # [16] (8个token × 2个专家 = 16个选择)
        """
        # 初始化输出（返回一个与输入张量x形状相同，但是数据类型为float16的未初始化的张量）
        y=torch.empty_like(x,dtype=torch.float16)  #使用fp16节省内存

        # 并行处理所有专家
        for i, expert  in enumerate(self.experts):
            # 找出应该由当前专家处理的token   
            """
            比如i = 0,mask会返回一个长度与flat_topk_idx相等的布尔掩码矩阵
            mask = (flat_topk_idx == 1)
            # 结果: [True, False, False, True, False, False, False, False, True, False, False, False, True, False, False, False]
            """
            mask = flat_topk_idx == i
            """ 注释
            mask.any(): # 至少有一个True
            # 提取专家i的输入
            x[mask]:映射找出相应的x（token），
            # 专家计算
            expert(x[mask]): 送入这个专家去进行计算（其实也就是进行了一次普通的门控前馈，然后输出）
                1. 稀疏处理
                    # 只对需要处理的token进行计算，跳过其他token
                    # 避免了if-else判断，利用向量化操作
                2. 内存局部性
                    # mask创建后，x[mask]会提取连续的内存块
                    # 专家计算时获得连续的输入，提高缓存效率
                3. 自动批量处理
                    # 即使一个专家处理多个不连续的token副本
                    # x[mask]会自动将它们收集为连续批次
                    # 专家仍然可以一次性处理所有分配的任务
            梯度流动
                # mask也用于确保梯度正确回传：
                # - 只有被处理的token副本的梯度会更新专家i的参数
                # - 其他专家的梯度不受影响
            计算图连接
                # 通过mask索引建立的连接是可微分的
                # 梯度可以从y[mask]流回expert_output，再流回专家网络
            # 将结果存回输出张量
            y[mask]=expert_output.to(y.dtype)
            """
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
    
    @torch.no_grad()
    def moe_infer(self,x,flat_expert_idices, flat_expert_weights):
        """推理优化的MoE前向传播"""
        # 初始化输出缓存
        expert_cache=torch.zeros_like(x)

        # 按专家索引排序，以便批量处理同一专家的token
        """argsort()方法的作用
        argsort()返回的是排序后的索引，而不是排序后的值
        即：返回一个索引的数组，使得flat_expert_indices[idxs]是升序排列的

        具体计算过程
        排序前的数据
            # 原始 专家索引和位置
            位置:   0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
            值:    [1, 2, 0, 1, 2, 3, 0, 2, 1, 3, 0, 2, 1, 0, 3, 2]
        argsort()操作
            #对值进行升序排序，但是返回的是值的索引
            idxs = flat_expert_indices.argsort()
            # 结果: [2, 6, 10, 13, 0, 3, 8, 12, 1, 4, 7, 11, 15, 5, 9, 14]
        
        排序后，相同专家的token副本被聚集在一起
        分组结果 = [
            # 专家0的token副本: 位置2,6,10,13 (前4个)
            # 专家1的token副本: 位置0,3,8,12 (接下来4个)  
            # 专家2的token副本: 位置1,4,7,11,15 (接下来5个)
            # 专家3的token副本: 位置5,9,14 (最后3个)
        ]

        # 排序后，可以按专家顺序批量处理：
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 结果: [4, 8, 13, 16]  # 每个专家的结束位置

        # 处理专家0: idxs[0:4]   → 位置2,6,10,13
        # 处理专家1: idxs[4:8]   → 位置0,3,8,12  
        # 处理专家2: idxs[8:13]  → 位置1,4,7,11,15
        # 处理专家3: idxs[13:16] → 位置5,9,14

        完整推理流程数据
        # 假设数据：
        flat_expert_indices =
            [ 1, 2,  0,  1,  2,  3,  0,  2,  1,  3,   0,   2,   1,    0,   3,  2 ]
        x = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15]  # 16个token副本

        # argsort() 后：
        idxs = [ 2, 6, 10, 13,    0, 3, 8, 12,     1, 4, 7, 11, 15,      5, 9, 14]

        # 对应的token顺序（按专家分组）：
        专家0: [t2, t6, t10, t13]    # 来自位置2,6,10,13
        专家1: [t0, t3, t8, t12]     # 来自位置0,3,8,12  
        专家2: [t1, t4, t7, t11, t15] # 来自位置1,4,7,11,15
        专家3: [t5, t9, t14]         # 来自位置5,9,14
        """
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
        """
        仔细看规律：
        在上面的训练模式中，明确地复制每个token：
        x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
        假设有4个原始token，top_k=2：
        原始token: [t0, t1, t2, t3]  # 4个token
        复制后: [t0, t0, t1, t1, t2, t2, t3, t3]  # 8个副本
        索引:     0,  1,  2,  3,  4,  5,  6,  7

        2. 复制模式的数学规律
        观察复制后的索引模式：
            原始token0 → 副本0,1
            原始token1 → 副本2,3  
            原始token2 → 副本4,5
            原始token3 → 副本6,7
        发现规律：
        原始token索引 = 副本索引 // 2

        验证：
        副本0: 0//2=0 → token0
        副本1: 1//2=0 → token0  
        副本2: 2//2=1 → token1
        副本3: 3//2=1 → token1
        副本4: 4//2=2 → token2
        副本5: 5//2=2 → token2
        副本6: 6//2=3 → token3
        副本7: 7//2=3 → token3

        如果我们把推理优化的方案用于训练就是如下效果：
        假设我们有8个原始token，每个选2个专家：
        原始token: [t0, t1, t2, t3, t4, t5, t6, t7]
        复制后: [t0,t0, t1,t1, t2,t2, t3,t3, t4,t4, t5,t5, t6,t6, t7,t7]
        副本索引: 0,1,  2,3,  4,5,  6,7,  8,9,  10,11, 12,13, 14,15
        专家选择后的混乱局面
        经过门控和排序后，副本顺序被打乱：
        排序后副本索引: [2,6,10,13,0,3,8,12,1,4,7,11,15,5,9,14]
        对应的专家:     [0,0, 0, 0, 1,1,1, 1, 2,2,2,2, 2, 3,3,3]

        映射回原始token
        副本索引: 2,6,10,13,0,3,8,12,1,4,7,11,15,5,9,14
        //2操作: 1,3, 5, 6,0,1,4, 6,0,2,3,5, 7,2,4,7

        得到原始token: [t1,t3,t5,t6, t0,t1,t4,t6, t0,t2,t3,t5,t7,t2,t4,t7]


        然后我们看推理优化
        1. 数据没有实际复制
        2. 通过索引模拟复制
        # 不复制数据，而是通过索引来达到相同效果
        expert_input = x[original_indices]  # 直接访问原始数据
        x_copied[copy_idx] = x[copy_idx // top_k]
        """
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
    
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        """注意力头参数设置
        计算多头注意力种每个头的维度
        示例：hidden_size= 768,num_attention_heads=12,head_dim=64
        """
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size

        # 基础传递的属性（这些是我看引用这个类的地方没有类型提示加到这里的，在这个类里没有用）
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.dropout = config.dropout
        self.rms_norm_eps = config.rms_norm_eps
        self.rope_scaling = config.rope_scaling
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings

        self.head_dim = config.hidden_size // config.num_attention_heads

        # 自注意力机制
        self.self_attn = Attention(config)
        # 将创建初始的归一化层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)

        self.layer_id = layer_id
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states,position_embeddings, past_key_value=None,use_cache=False, attention_mask=None):
        """残差连接详解
        也就是说第一次（ residual = hidden_states）的所谓'残差'，是原始的数据（我记为a），
        第二次（  hidden_states +=residual）
        变化后的残差是原始的数据加上注意力层输出的数据（变化的部分）（我记为b），
        再将这个原始输入加上注意力层变化后的值（我记为a+b值为c），
        进行一次线性变化（c+mlp(c)）输出给下个块的输入值（本次我记为d,)
        这个d就是下一个块的a了，
        由于初始的信息很大部分都是线性变化传递的，所以到最后面最后面的输出层的时候
        和预期值比较的时候很大一部分信息就可能保留下来，
        所以我理解的就是学习的是所谓改变了什么而不是完全变化

        a (初始输入) → 
        b (注意力输出) → 
        c = a + b (第一次残差连接) → 
        d = c + mlp(c) (第二次残差连接，也是本块输出) → 
        下一个块的 a
        """
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # Pre-Norm: 先归一化上一层传递下来的数据
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        ) 
        hidden_states +=residual   # 残差连接
        # 前馈网络子层（带残差连接）
        """
        处理流程：
        归一化当前状态
        通过MLP/MoE前馈网络
        残差连接加到原始状态
        """
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        # 核心组件
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        """词嵌入层
        作用：将输入的token id转换为密集向量表示
        本质：
        它是一个可查询的字典，其键是整数（索引，代表单词ID），值是固定大小的向量（词向量）。
        config.vocab_size： 字典的大小。比如 50000，表示模型认识 50000 个不同的单词/子词。
        config.hidden_size： 每个词向量的维度。比如 512，表示每个单词用一个 512 维的向量来表示。
        
        输入：[batch_size,seq_len](整数token ID)
        输出：[batch_size,seq_len,hidden_size](浮点数向量)
        """
        self.dropout = nn.Dropout(config.dropout) 
        """嵌入后的dropout"""
        self.layers = nn.ModuleList([
            # 8个transformer块
            MiniMindBlock(l,config) for l in range(self.num_hidden_layers)
        ])
        """Transformer层堆叠
        作用：创建多个Transformer块，形成深度网络，
        一般而言每个Transformer块配置相同，共同使用同一个配置
        """
        self.norm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        """最终层归一化
        生成初始化的可训练归一化层
        """

        freqs_cos,freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        """旋转位置编码预计算
        预计算优化： 在初始化时候计算所有位置的位置编码，避免重复计算
        register_buffer: 将位置编码注册为模型缓冲区（不参与梯度更新）
        persistent=False: 不保存到 state_dict,节省存储空间
        """  

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer('freqs_sin',freqs_sin,persistent=False)
    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor]=None,
                past_key_values: Optional[List[Tuple[torch.Tensor,torch.Tensor]]]=None,
                use_cache: bool = False,
                **kwargs ):
          
          batch_size,seq_lenth = input_ids.shape
          if hasattr(past_key_values,'layers'): past_key_values = None #兼容性处理
          #   默认值设置，如果past_key_values为None（上面兼容处理后的情况），重新创建一个新的列表
          #   列表长度 = 模型层数（len(self.layers)）
          #   每个元素初始化为欸None

          """ 检查past_key_values是否具有layers属性
          检查past_key_values是否具有layers属性
          背景：某些版本的Transformer实现可能将past_key_values封装子啊对象中而不是直接使用列表
          数据状态：如果past_key_values是对象而非列表，说明数据结构不兼容
          遇到不认识的past_key_values格式时，放弃使用历史缓存，重新开始

          执行示例：
          假设模型有8层，可能会有一下执行路径
          情况1：首次调用（无历史缓存）
          past_key_values = None
            ->执行后： past_key_values = [None, None, None, None, None, None, None, None]
          情况2：正常续传（有正确格式的缓存）
          past_key_values = [(k1,v1),(k2,v2),(k3,v3),(k4,v4)(k5,v5),(k6,v6),(k7,v7)]
            ->执行后： → 保持不变：past_key_values = 原始缓存列表
          情况3：格式不兼容的缓存
            past_key_values = 某个具有layers属性的对象
              ->先重置为None，再重新创建新列表
          """
          past_key_values = past_key_values or [None] * len(self.layers)
          start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
          # 计算起始位置（用于生成任务  ）   
          """计算起始位置
          条件判断: past_key_values[0] is None
          目的：检查第一层是否存在KV缓存
          逻辑：如果第一层有缓存，说明不是第一次处理，需要续传，否则从头开始
          
          2、KV缓存数据结构
          假设past_key_values的结构：
            past_key_values = [
                (key_tensor_layer0, value_tensor_layer0), # 第0层
                (key_tensor_layer1, value_tensor_layer1), # 第1层
                ...其他层
            ]
            数值示例
            假设：batch_size =1
                  已处理的序列长度为 = 3
                  注意力头2
                  每个头的维度=4
                  模型层数=3
            第0层的KV缓存
            # Key 张量 (形状: [1, 2, 3, 4])
            key_layer0 = torch.tensor([[
                [[0.1, 0.2, 0.3, 0.4],   # 头0，token0的特征
                [0.5, 0.6, 0.7, 0.8],   # 头0，token1的特征  
                [0.9, 1.0, 1.1, 1.2]],  # 头0，token2的特征
                
                [[1.1, 1.2, 1.3, 1.4],   # 头1，token0的特征
                [1.5, 1.6, 1.7, 1.8],   # 头1，token1的特征
                [1.9, 2.0, 2.1, 2.2]]   # 头1，token2的特征
            ]])

            # Value 张量 (形状: [1, 2, 3, 4])  
            value_layer0 = torch.tensor([[
                [[2.1, 2.2, 2.3, 2.4],   # 头0，token0的特征
                [2.5, 2.6, 2.7, 2.8],   # 头0，token1的特征
                [2.9, 3.0, 3.1, 3.2]],  # 头0，token2的特征
                
                [[3.1, 3.2, 3.3, 3.4],   # 头1，token0的特征
                [3.5, 3.6, 3.7, 3.8],   # 头1，token1的特征
                [3.9, 4.0, 4.1, 4.2]]   # 头1，token2的特征
            ]])
          形状解析
          # 访问第0层的key张量
            key_tensor = past_key_values[0][0]  # 形状: [1, 2, 3, 4]

            # 各维度含义：
            print(f"batch_size: {key_tensor.shape[0]}")      # 1
            print(f"num_heads: {key_tensor.shape[1]}")       # 2  
            print(f"seq_len: {key_tensor.shape[2]}")         # 3 ← 这就是start_pos要的值!
            print(f"head_dim: {key_tensor.shape[3]}")        # 4


          3、形状分析
          past_key_values[0][0].shape[1]  #取第一层第一个元素key的第二维

          past_key_values[0]   第0层的(key,value)元组
          past_key_values[0][0]   第0层的key张量
          .shape[1]  key张量的第二维（序列长度）
          张量形状(shape):(batch_size,seq_len,num_heads,head_dim）
            第一维: batch_size,
            第二维：seq_len(!!这个就是我们需要的)
            第三维：注意力头数
            第四维： 每个头的维度
        
          计算示例
          情况1：首次推理（无缓存）
            past_key_values[0]=None
            -> start_pos = 0
          情况2：续传推理（有缓存）
            past_key_values[0] = (key_tensor,value_tensor)
            key_tensor.shape = (1, 128, 8, 64) #batch = 1，已处理128个token
            -> start_pos = 128

          """
          
          hidden_states = self.dropout(self.embed_tokens(input_ids))
          """最初的hidden_states"""
         
          position_embeddings = (
              self.freqs_cos[start_pos:start_pos+seq_lenth],
              self.freqs_sin[start_pos:start_pos+seq_lenth]  
          ) 
          # 从start_pos开始截取
          """
          实际场景
          输入："Hello"
          start_pos = 0
          位置编码：cos[0:5], sin[0,5] #处理5个token
          续传生成：
          已处理：Hello world (11个token)
          新输入：how are (3个token)
          start_pos = 11
          位置编码：cos[11:14] sin[11,14]#从第11个位置开始
          """

          presents = []
          for layer_idx, (layer,past_key_value) in enumerate(zip(self.layers, past_key_values)):
              #   layer就是每一个MiniMindBlock构建的层
              """hidden_states是被layer层更新后的"""
              hidden_states,present = layer(
                  hidden_states,
                  position_embeddings,
                  past_key_value=past_key_value,
                  use_cache=use_cache,
                  attention_mask=attention_mask
              )

              presents.append(present)

          hidden_states = self.norm(hidden_states)

          aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp,MOEFeedForward)
          )

          return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel,GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig=None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size,bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor]=None,
                attention_mask: Optional[torch.Tensor]=None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None,
                use_cache:bool=False,
                logits_to_keep: Union[int,torch.Tensor]=0,
                **args  
                ):
        h, past_kvs,aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep,int) else logits_to_keep
        logits = self.lm_head(h[:,slice_indices,:])
        self.OUT.__setitem__('last_hidden_state',h)
        self.OUT.__setitem__('logits',logits)
        self.OUT.__setitem__('aux_loss',aux_loss)
        self.OUT.__setitem__('past_key_values',past_kvs)
        return self.OUT
