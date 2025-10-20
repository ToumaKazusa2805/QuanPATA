import torch
import torch.nn as nn
import torch.nn.functional as F
from quanlib.adaround.quantizer import AdaRoundQuantizer, AsymmetricQuantizer, SymmetricQuantizer
from quanlib.adaround.observer import EMAMinMaxObserver, MinMaxObserver

class QLinear(nn.Linear):
    def __init__(self, bit: int, ptq: bool, in_features: int, out_features: int, bias: bool = False, first = False, symmetric: bool = False, device='cuda'):

        super(QLinear, self).__init__(in_features, out_features, bias, device)
        self.ptq = ptq
        self.bit = bit
        self.origin = False
        self.first = first
        rank = 2
        
        assert self.bit <= 32, "bit must be less than 32"
        assert self.bit > 0, "bit must be greater than 0"
        self.weight_quantizer = AdaRoundQuantizer(bit = self.bit, observer= MinMaxObserver(mode = 'scalar'), ptq = self.ptq, mode = 'scalar')
        self.max_threshold = 4.0
        if self.first:
            if not symmetric:
                self.input_quantizer = AsymmetricQuantizer(bit = self.bit, observer= EMAMinMaxObserver(), ptq = self.ptq)
            else:
                # self.input_quantizer = SymmetricQuantizer(bit = self.bit, observer= EMAMinMaxObserver(), ptq = self.ptq)
                self.input_quantizer = SymmetricQuantizer(bit = self.bit, observer= MinMaxObserver(), ptq = self.ptq)
        else:
            self.input_quantizer = nn.Identity()  # 非首层不进行输入量化, 因为输入全是01脉冲

        # self.mem_threshold = torch.nn.Parameter((torch.tensor(4., dtype=torch.float32)).to(device='cuda'), requires_grad=False)
        self.coefficient_weight = torch.nn.Parameter(torch.zeros(self.weight.shape[0], device=self.weight.device))
        self.raw_grad = None
        
        # LoRA
        # rank = 2 * min(rank, in_features, out_features)
        # self.lora_A = nn.Linear(in_features, rank, bias=False, device='cuda')
        # nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.01 / rank**0.5)
        
        # self.lora_B = nn.Linear(rank, out_features, bias=False, device='cuda')
        # nn.init.zeros_(self.lora_B.weight)
        
        
        self.col_min = self.weight.min(dim=1, keepdim=True)[0].detach()  # [out_features, 1]
        self.col_max = self.weight.max(dim=1, keepdim=True)[0].detach()  # [out_features, 1]
        self.sub_weight = None

        # self.register_buffer('epsilon', torch.tensor(1e-4), device=self.weight.device)
        # self.col_min_coff = nn.Parameter(torch.zeros_like(self.col_min))
        # self.col_max_coff = nn.Parameter(torch.zeros_like(self.col_min))
        # print(f'[INFO] Initializing QLinear: {self.col_min.device}, {self.col_max.device}, {self.weight.device}')
        # self.channel_scale_weight = self.weight.clone()
        # self.sign_eq = True
    def forward(self, x):
        # 量化后的Linear层
        if not self.origin:
            
            self.input = x.clone()
            input = self.input_quantizer(x)
            self.weight_quant = self.weight_quantizer(self.weight)
            output = F.linear(input, self.weight_quant)
            return output
        # 原始Linear层
        else:
            self.input = x.clone()
            if self.sub_weight is None:
                self.sub_weight = self.weight.clone().detach()
            output = F.linear(x, self.sub_weight)
            return output    

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None):
        raw_grad = self.raw_grad.reshape_as(tensor_raw)
        similarity = (raw_grad * (tensor_raw - tensor_sim)) ** 2
        similarity = torch.mean(similarity, dim = -1)
        return similarity


if __name__ == "__main__":
    tmp = QLinear(8, True, 3, 4)
    print(tmp.parameters())
    # print(tmp.state_dict())
    for name, param in tmp.named_parameters():
        print(name, param)