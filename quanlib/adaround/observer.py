import torch
import torch.nn as nn

class ObserverBase(nn.Module):
    def __init__(self, mode='scalar'):
        super(ObserverBase, self).__init__()
        self.mode = mode  # 'scalar' or 'tensor'
    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):  #input:128,3,224,224  # layer级(activation/weight)
        if self.mode == 'scalar':
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.mode == 'tensor':
            min_val = input.min(dim=1, keepdim=True)[0]
            max_val = input.max(dim=1, keepdim=True)[0]
        else:
            raise ValueError("Unsupported mode: {}".format(self.mode))
        self.update_range(min_val, max_val)
        return input


class MinMaxObserver(ObserverBase):
    def __init__(self, mode = 'scalar'):
        super(MinMaxObserver, self).__init__(mode)
        self.num_flag = 0

        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32).to(device='cuda'))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32).to(device='cuda'))


    def update_range(self, min_val_cur, max_val_cur):
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
            self.min_val = torch.zeros_like(min_val_cur)
            self.max_val = torch.zeros_like(max_val_cur)
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class EMAMinMaxObserver(ObserverBase):
    def __init__(self, momentum=0.1, mode = 'scalar'):

        super(EMAMinMaxObserver, self).__init__(mode)
        self.momentum = momentum
        self.num_flag = 0

        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32).to(device='cuda'))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32).to(device='cuda'))


    def update_range(self, min_val_cur, max_val_cur):
        
        if self.num_flag == 0:  
            self.num_flag += 1
            min_val = min_val_cur  
            max_val = max_val_cur 
        # Introduce momentum item
        else:  
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        # print(self.min_val, self.max_val)

