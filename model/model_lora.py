import torch
from torch import optim, nn

# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features,out_features, rank):
        super().__init__()
        self.rank = rank #LoRA的秩,控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False) #低秩矩阵A
        self.B = nn.Linear(rank, out_features,bias=False) #低秩矩阵
        # 矩阵A的高斯初始化
        self.A.weight.data.normal_(mean=0,std=0.02)
        # 矩阵B全部初始化为0
        self.B.weight.data.zero_()

    def forward(self, x):
        # 把x经过矩阵A进行降维,再将结果进行升维
        return self.B(self.A(x))

def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            # 这里给每个module都价格lora属性
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显示绑定
            def forward_with_lora(x, layer1=original_forward,layer2=lora):
                return layer1(x) + layer2(x)
            
            module.forward = forward_with_lora

def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_name = f"{name}.lora."
            # 这里将前缀去掉从新塞给lora属性()或者可以说新加一个lora属性,
            lora_state = {k.replace(lora_name,""): v for k,v in state_dict.items() if lora_name in k}
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 这里将lora属性提取出来加上前缀
            lora_state = {f'{name}.lora.{k}': v for k,v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
        torch.save(state_dict, path)