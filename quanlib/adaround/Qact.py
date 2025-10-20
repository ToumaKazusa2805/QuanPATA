import torch
import torch.nn as nn
import torch.nn.functional as F
from quanlib.adaround.quantizer import AdaRoundQuantizer, AsymmetricQuantizer, SymmetricQuantizer
from quanlib.adaround.observer import EMAMinMaxObserver, MinMaxObserver

class Qexp(nn.Module):
    def __init__(self, bit: int, ptq: bool, symmetric: bool = False):
        super(Qexp, self).__init__()
        self.ptq = ptq
        self.bit = bit
        self.origin = False
        
        assert self.bit <= 32, "bit must be less than 32"
        assert self.bit > 0, "bit must be greater than 0"
        if not symmetric:
            self.input_quantizer = AsymmetricQuantizer(bit = self.bit, observer= EMAMinMaxObserver(), ptq = self.ptq)
        else:
            self.input_quantizer = SymmetricQuantizer(bit = self.bit, observer= EMAMinMaxObserver(), ptq = self.ptq)
            # self.input_quantizer = SymmetricQuantizer(bit = self.bit, observer= MinMaxObserver(), ptq = self.ptq)
    def forward(self, x):
        if not self.origin:
            input = self.input_quantizer(x)
            output = torch.exp(input)
            return output
        
        else:
            output = torch.exp(x)
            return output  
        
class Qsigmoid(nn.Module):
    def __init__(self, bit: int, ptq: bool, symmetric: bool = False):
        super(Qsigmoid, self).__init__()
        self.ptq = ptq
        self.bit = bit
        self.origin = False
        
        assert self.bit <= 32, "bit must be less than 32"
        assert self.bit > 0, "bit must be greater than 0"
        
        if not symmetric:
            self.input_quantizer = AsymmetricQuantizer(bit = self.bit, observer= EMAMinMaxObserver(), ptq = self.ptq)
        else:
            self.input_quantizer = SymmetricQuantizer(bit = self.bit, observer= EMAMinMaxObserver(), ptq = self.ptq)
            # self.input_quantizer = SymmetricQuantizer(bit = self.bit, observer= MinMaxObserver(), ptq = self.ptq)
    def forward(self, x):
        if not self.origin:
            input = self.input_quantizer(x)
            output = torch.sigmoid(input)
            return output

        else:
            output = torch.sigmoid(x)
            return output  