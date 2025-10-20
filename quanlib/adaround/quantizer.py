import torch
import torch.nn as nn

from torch.autograd import Function

class FloorSTE(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def floor_ste(input):
    return FloorSTE.apply(input)

class Quantizer(nn.Module):
    def __init__(self, bit, observer, ptq):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq
        
    def update_qparams(self, tensor):
        raise NotImplementedError
    
    def forward(self, tensor):
        if self.ptq:
            
            self.observer(tensor)   #EMAMinMaxObserver   记录了下input的最大最小值
            self.update_qparams(tensor)   #调整 scale 和 zero point参数
            # if tensor.shape[0] > 128:
            #     print('Quantizer: ', torch.max(tensor), torch.min(tensor), self.observer.max_val, self.observer.min_val)
                

        if self.training:
            if not hasattr(self, 'scale_exp'):
                quant_tensor = (torch.round(tensor / self.scale.item()) - tensor / self.scale.item()).detach() + tensor / self.scale.item() + self.zero_point
                quant_tensor = quant_tensor.clamp(self.quant_min,self.quant_max)
                fake_quant_tensor = self.scale.item() * (quant_tensor - self.zero_point)    #dequantization
            else:
                
                scale_exp_low = floor_ste(self.scale_exp)
                scale_exp_high = scale_exp_low + 1 
                scale_low = 2 ** scale_exp_low
                scale_high = 2 ** scale_exp_high

                x_low = tensor / scale_low
                x_high = tensor / scale_high

                fake_quant_tensor_low = scale_low * (
                    ((torch.round(x_low) - x_low).detach() + x_low + self.zero_point)
                    .clamp(self.quant_min, self.quant_max)
                )

                fake_quant_tensor_high = scale_high * (
                    ((torch.round(x_high) - x_high).detach() + x_high + self.zero_point)
                    .clamp(self.quant_min, self.quant_max)
                )
                fake_quant_tensor = (1 - self.scale_exp + scale_exp_low) * fake_quant_tensor_low + (self.scale_exp - scale_exp_low) * fake_quant_tensor_high

                       
        else:
            if not hasattr(self, 'scale_exp'):
                quant_tensor = (torch.round(tensor / self.scale.item()) - tensor / self.scale.item()).detach() + tensor / self.scale.item() + self.zero_point
                quant_tensor = quant_tensor.clamp(self.quant_min,self.quant_max)
                fake_quant_tensor = self.scale.item() * (quant_tensor - self.zero_point)    #dequantization
            else:
                self.scale = 2 ** torch.round(self.scale_exp)
                quant_tensor = ((torch.round(tensor / self.scale) - tensor / self.scale).detach() + tensor / self.scale + self.zero_point).clamp(self.quant_min,self.quant_max)   
                fake_quant_tensor = self.scale * (quant_tensor - self.zero_point)
        return fake_quant_tensor

class AsymmetricQuantizer(Quantizer):
    def __init__(self, bit, observer, ptq):
        super(AsymmetricQuantizer, self).__init__(bit, observer, ptq)
        self.bit = bit
        self.observer = observer
        self.ptq = ptq

        self.register_buffer("scale", torch.ones((1), dtype=torch.float32).to(device='cuda'))
        self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32).to(device='cuda'))
        self.register_buffer("quant_min",torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32).to(device='cuda'),)
        self.register_buffer("quant_max",torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32).to(device='cuda'),)
    
    def update_qparams(self, inputs):

        scale = (self.observer.max_val - self.observer.min_val) / (self.quant_max - self.quant_min)  
        zero_point = (torch.round(self.quant_min - self.observer.min_val / scale) - (self.quant_min - self.observer.min_val / scale)).detach() \
                   + (self.quant_min - self.observer.min_val / scale)
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)

    def __str__(self):
        info =  ' scale: %.3f ' % self.scale
        info += ' min: %.3f ' % self.observer.min_val
        info += ' max: %.3f ' % self.observer.max_val
        return info

class AdaRoundQuantizer(Quantizer):
    def __init__(self, bit, observer, ptq, mode = 'scalar' ):
        super(AdaRoundQuantizer, self).__init__(bit, observer, ptq )
        self.bit = bit
        self.observer = observer  # MinMaxObserver()
        self.ptq = ptq
        self.alpha = None
        self.ada_init = None
        self.soft_targets = True
        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.mode = mode

        self.register_buffer("scale", torch.ones((1), dtype=torch.float32).to(device='cuda'))
        self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32).to(device='cuda'))
        self.register_buffer("quant_min",torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32).to(device='cuda'),)
        self.register_buffer("quant_max",torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32).to(device='cuda'),)
    
    def update_qparams(self, inputs):

        max_abs = torch.max(torch.abs(self.observer.min_val), torch.abs(self.observer.max_val)) * 2
        scale = max_abs / (self.quant_max - self.quant_min)
        self.scale.copy_(scale)

    def forward(self, tensor):

        if self.ptq:
            self.observer(tensor)  
            if self.mode == 'tensor':
                self.scale = torch.zeros_like(self.observer.min_val)
            self.update_qparams(tensor)
            

        if not self.ada_init:   
            self.init_alpha(tensor.clone())   #初始化 Vi,j
            self.ada_init = True
            
        #第二个batch就走这了
        # if hasattr(self, 'scale_exp'):
        #     self.scale = 2 ** self.scale_exp
        # quant_tensor = self.quant(tensor)   
        # fake_quant_tensor = self.dequantize(quant_tensor)  
        fake_quant_tensor = self.quan_fwd(tensor)
        return fake_quant_tensor
    
    def quant(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
            
        x_floor = torch.floor(inputs / scale)

        if self.soft_targets:
            x_ada = x_floor + self.get_soft_targets()  # 完整的是公式22的一部分clamp在底下 #self.get_soft_targets()是标准的公式23 h(V i,j)
        else:
            x_ada = x_floor + (self.alpha >= 0).float()    # (self.alpha >= 0).float() torch.float32  全是0，1

        outputs = x_ada.clamp(self.quant_min,self.quant_max)
        return outputs  #得到Wint

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
            
        outputs = inputs * scale
        return outputs
    
    def quan_fwd(self, tensor):
        
        if self.training:
            if not hasattr(self, 'scale_exp'):
                quant_tensor = self.quant(tensor)   
                fake_quant_tensor = self.dequantize(quant_tensor)  
            else:
                if self.soft_targets:
                    adds = self.get_soft_targets()
                else:
                    adds = (self.alpha >= 0).float()
                self.scale_exp_low = floor_ste(self.scale_exp)
                self.scale_exp_high = floor_ste(self.scale_exp) + 1 
                self.scale_low = 2 ** self.scale_exp_low
                self.scale_high = 2 ** self.scale_exp_high
                quant_tensor_low = ((torch.floor(tensor / self.scale_low) - tensor / self.scale_low).detach() + tensor / self.scale_low + self.zero_point + adds).clamp(self.quant_min,self.quant_max)   
                fake_quant_tensor_low = self.scale_low * (quant_tensor_low - self.zero_point)    #dequantization
            
                quant_tensor_high = ((torch.floor(tensor / self.scale_high) - tensor / self.scale_high).detach() + tensor / self.scale_high + self.zero_point + adds).clamp(self.quant_min,self.quant_max)
                fake_quant_tensor_high = self.scale_high * (quant_tensor_high - self.zero_point)
            
                # interpolate to dequantize
                # if torch.round(self.scale_exp) == torch.floor(self.scale_exp):
                if torch.equal( torch.round(self.scale_exp), torch.floor(self.scale_exp)):
                    fake_quant_tensor = (self.scale_exp - self.scale_exp_low) * fake_quant_tensor_high + (1 - self.scale_exp + self.scale_exp_low) * fake_quant_tensor_low
                else:
                    fake_quant_tensor = (self.scale_exp_high - self.scale_exp) * fake_quant_tensor_low + (1 - self.scale_exp_high + self.scale_exp) * fake_quant_tensor_high
        else:
                self.scale = 2 ** torch.round(self.scale_exp)
                quant_tensor = self.quant(tensor)   
                fake_quant_tensor = self.dequantize(quant_tensor) 
        
        return fake_quant_tensor
    
    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)  #h(V i,j)  self.alpha 就是Vi,j  1.2 * sigmoid(alpha) - 0.1

    def init_alpha(self, x: torch.Tensor):
        scale = self.scale
        x_floor = torch.floor(x / scale)
    
        rest = (x / scale) - x_floor  # rest of rounding [0, 1)
        denominator = rest - self.gamma
        denominator = denominator.clamp(min=1e-8)  # 避免分母为0
        alpha = -torch.log((self.zeta - self.gamma) / denominator - 1)  # 初始化的时候 使得h(Vi,j)= w/s - floor(w/s)=rest 解这个方程得到 sigmoid（Vi,j)的表达式   公式23 省去clip   # => sigmoid(alpha) = rest
        self.alpha = nn.Parameter(alpha)
        
    def __str__(self):
        info =  ' scale: %.3f ' % self.scale
        info += ' min: %.3f ' % self.observer.min_val
        info += ' max: %.3f ' % self.observer.max_val
        info += ' quant_max: %3f ' % self.quant_max
        info += ' quant_min: %.3f ' % self.quant_min
        return info
    
class SymmetricQuantizer(Quantizer):
    def __init__(self, bit, observer, ptq):
        super(SymmetricQuantizer, self).__init__(bit, observer, ptq)
        self.bit = bit
        self.observer = observer
        self.ptq = ptq
        # self.scale_exp = torch.tensor(-1.0, dtype=torch.float32).to(device='cuda')  # 初始值为1.0
        self.register_buffer("scale", torch.ones((1), dtype=torch.float32).to(device='cuda'))
        self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32).to(device='cuda'))
        self.register_buffer("quant_min",torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32).to(device='cuda'),)
        self.register_buffer("quant_max",torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32).to(device='cuda'),)
    
    def update_qparams(self, inputs):
        scale = (self.observer.max_val - self.observer.min_val) / (self.quant_max - self.quant_min)  
        self.scale.copy_(scale)

    def __str__(self):
        info =  ' scale: %.3f ' % self.scale
        info += ' min: %.3f ' % self.observer.min_val
        info += ' max: %.3f ' % self.observer.max_val
        info += ' zero_point: %d ' % self.zero_point
        return info