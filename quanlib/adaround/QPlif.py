import torch
import torch.nn as nn
from quanlib.adaround.quantizer import AdaRoundQuantizer, AsymmetricQuantizer, SymmetricQuantizer, floor_ste
from quanlib.adaround.observer import EMAMinMaxObserver, MinMaxObserver
from spikingjelly.clock_driven.neuron import ParametricLIFNode as PLIF

def round_ste(input):
    return torch.round(input) - input.detach() + input

def floor_ste(input):
    return torch.floor(input) - input.detach() + input

class QPlif(PLIF):
    def __init__(self, bit: int = 8, ptq: bool = True, v_threshold: float = 0.5, symmetric: bool = False):
        super().__init__() 
        self.bit = bit
        self.ptq = ptq
        self.origin = False
        self.v_threshold = v_threshold
        self.set_reset_value('v_threshold', v_threshold)
        self.v_q, self.v_o = [], []
        self.register_buffer("quant_min",torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32).to(device='cuda'),)
        self.register_buffer("quant_max",torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32).to(device='cuda'),)
        self.mem_threshold_max = torch.nn.Parameter((torch.tensor(0., dtype=torch.float32)).to(device='cuda'), requires_grad=True)
        self.mem_threshold_min = torch.nn.Parameter((torch.tensor(0., dtype=torch.float32)).to(device='cuda'), requires_grad=True)
        self.max_threshold = 4.0
        
        if not symmetric:
            self.Mem_quantizer = AsymmetricQuantizer(bit = self.bit, observer= EMAMinMaxObserver(), ptq = self.ptq)
            self.input_quantizer = AsymmetricQuantizer(bit = self.bit, observer= EMAMinMaxObserver(), ptq = self.ptq)
        else:
            self.Mem_quantizer = SymmetricQuantizer(bit = self.bit, observer=  EMAMinMaxObserver(), ptq = self.ptq)
            self.input_quantizer = SymmetricQuantizer(bit = self.bit, observer=  EMAMinMaxObserver(), ptq = self.ptq)
        self.vth_alpha = nn.Parameter(torch.tensor(2.5, dtype=torch.float32).to(device='cuda'), requires_grad=True)
        
    def quan_fwd(self, x):
        self.input = x.clone()
        if len(self.mem_threshold_max.shape) == 0:
            self.mem_threshold_max = nn.Parameter(torch.zeros(size = (x.shape[-1], ), device= x.device))
            self.mem_threshold_min = nn.Parameter(torch.zeros(size = (x.shape[-1], ), device= x.device))

        self.v_threshold_quan = self.v_threshold * torch.sigmoid(self.vth_alpha)
        if self.training:
            if hasattr(self.Mem_quantizer, 'scale_exp'):
                s = self.Mem_quantizer.scale_exp
                s_l = floor_ste(s)
                s_h = s_l + 1
                self.v_threshold_quan = (round_ste(self.v_threshold_quan / 2 ** s_l)* (2 ** s_l)) * (1 - s + s_l)  + (round_ste(self.v_threshold_quan / 2 **s_h) * 2 ** s_h) * (s - s_l)
        else:
                s = 2 ** torch.round(self.Mem_quantizer.scale_exp)
                self.v_threshold_quan = round_ste(self.v_threshold_quan / s) * s
        inputs = self.input_quantizer(x)
        self.quan_charge(inputs)
        self.v_q = self.v.clone()#.detach()
        output = self.quan_spike()
        self.quan_reset(output)
        
        if self.training:
            self.v = self.v.clamp(-torch.sigmoid(self.mem_threshold_min) * self.max_threshold , torch.sigmoid(self.mem_threshold_max) * self.max_threshold)
        self.v = self.Mem_quantizer(self.v)
        return output
    
    def origin_fwd(self, x):
        self.input = x.clone()
        self.neuronal_charge(x)
        self.v_o = self.v.clone()#.detach()
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike
    
    def forward(self, x):
        if not self.origin:
            return self.quan_fwd(x)
        else:
            return self.origin_fwd(x)
    
    # 需要推导膜电位重置
    def quan_reset(self, spike):
        # 量化膜电位
        if not self.origin:
            self.v -= self.v_threshold_quan * spike
            # self.v -= self.v_threshold * spike 
        else:
            self.neuronal_reset(spike)
    
    # 需要判断阈值是否要处理
    def quan_spike(self):
        if not self.origin:
            return self.surrogate_function(self.v - self.v_threshold_quan)
        else:
            return self.neuronal_fire()
    
    # 需要推导膜电位累积
    def quan_charge(self, x: torch.Tensor):
        if self.decay_input:
            self.v = self.v + (x - self.v) * self.w.sigmoid() # self.w.sigmoid() = 1 / tau
        else:
            self.v = self.v * (1. - self.w.sigmoid()) + x
        
        self.v = self.Mem_quantizer(self.v)
    
    def __str__(self):
        info = super().__str__()
        info += ' QPlif: bit: %d ' % self.bit
        return info
    
    
    # def quantize_tau(x, num_bits):
    #     """Quantize a tensor to a specified number of bits."""
    #     # scale = (2 ** num_bits - 1) / (x.max() - x.min())
    #     # x_q = torch.round((x - x.min()) * scale)
    #     scale = (2 ** num_bits - 1) / (x - 1)
    #     x_q = torch.round((x - 1) * scale)
    #     return x_q.int(), scale

    # def dequantize_tau(x_q, scale, min_val):
    #     """Dequantize a tensor using the scale and min value."""
    #     x = (x_q.float() / scale) + min_val
    #     return x
