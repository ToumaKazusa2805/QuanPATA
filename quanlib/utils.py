import torch
from torch import nn
from quanlib.adaround.QLinear import QLinear
from quanlib.adaround.QPlif import QPlif
from quanlib.adaround.Qact import Qexp, Qsigmoid
from quanlib.adaround.quantizer import AdaRoundQuantizer, Quantizer
import logging
import time
import tqdm
from spikingjelly.clock_driven.neuron import ParametricLIFNode as PLIF
from collections import OrderedDict

def enable_calibrate(module):
    for name, child in module.named_children():
        if isinstance(child,Quantizer):
            child.ptq = True
        else:
            enable_calibrate(child)
    return module

def disable_calibrate(module):
    for name, child in module.named_children():
        if isinstance(child, Quantizer):  #AdaRoundQuantizer((observer): MinMaxObserver())  #AsymmetricQuantizer((observer): EMAMinMaxObserver())
            child.ptq = False     #别的都pass
        else:
            disable_calibrate(child)
    return module

def inplace_linear(module: nn.Module, ptq: bool = True, bit = 8, sign = True, symmetric = False, Mem_bit = 8) -> nn.Module:
    if isinstance(module, nn.Linear):
        new_layer = QLinear(bit, ptq, module.in_features, module.out_features,True if module.bias is not None else False, sign, symmetric=symmetric)
        new_layer.weight = module.weight
        if module.bias is not None:
            new_layer.bias = module.bias
    elif isinstance(module, PLIF):
        new_layer = QPlif(bit = Mem_bit, ptq = ptq, symmetric = symmetric)
        new_layer.w = module.w
        new_layer.decay_input = module.decay_input
        new_layer.v_threshold = module.v_threshold
        new_layer.v_reset = module.v_reset
        new_layer.surrogate_function = module.surrogate_function
        new_layer.detach_reset = module.detach_reset
    return new_layer

# INGP-NeRF只有linear层
def inplace_net(net: nn.Module, ptq: bool = True, bit = 8, net_name = 'qsnn', sub_module = False, symmetric = False, Mem_bit = 8) -> nn.Module:
    search_res = OrderedDict((k, v) for k, v in OrderedDict(net.named_modules()).items())
    del search_res['']
    modules = []
    sub_module_list = []
    sigma_sign, color_sign = True, True
    for name, module in search_res.items():
        if isinstance(module, torch.nn.Linear) or isinstance(module, PLIF) or isinstance(module, nn.ReLU):
            modules.append(name)
            
        # 注意到key中的顺序是先出现sequential,再出现sequential中的层
        elif isinstance(module, torch.nn.modules.Sequential) or isinstance(module, nn.ModuleList):
            sub_module_list.append(name)
            inplace_net(module, ptq, bit, name, sub_module = True, symmetric = symmetric, Mem_bit = Mem_bit)
            
    for name in modules:
        if isinstance(search_res[name], nn.ReLU):
            # new_layer = getattr(net, module_name)[int(name.split('.')[-1])]
            # setattr(net, name, new_layer)
            sigma_sign, color_sign = True, True
            continue
        
        if isinstance(search_res[name], nn.Linear) or isinstance(search_res[name], PLIF):
            if not sub_module:
                module_name = name.split('.')[0]
                if module_name in sub_module_list:
                    # 默认所有层的名字都是module_name.id
                    new_layer = getattr(net, module_name)[int(name.split('.')[-1])]
                else:
                    new_layer = inplace_linear(search_res[name], ptq, bit, symmetric = symmetric, Mem_bit = Mem_bit)
                continue
            
            # 如果是sigma_net或者color_net,那么先将其进行转换
            if 'sigma_net' in name or 'sigma' in net_name:   
                new_layer = inplace_linear(search_res[name], ptq, bit, sigma_sign, symmetric = symmetric, Mem_bit = Mem_bit)
                sigma_sign = False
            elif 'color_net' in name or 'color' in net_name:
                new_layer = inplace_linear(search_res[name], ptq, bit, color_sign, symmetric = symmetric, Mem_bit = Mem_bit)
                color_sign = False
            else:
                print(f'{name}')
            setattr(net, name, new_layer)
            
        elif isinstance(search_res[name], PLIF):
            new_layer = inplace_linear(search_res[name], ptq, bit, symmetric = symmetric, Mem_bit = Mem_bit)
            setattr(net, name, new_layer)
    return net

def enable_origin(module: nn.Module):
    module.origin = True
    for name, child in module.named_children():
        
        if isinstance(child, QLinear) or isinstance(child, QPlif) or isinstance(child, Qexp) or isinstance(child, Qsigmoid):
            child.origin = True
        else:
            enable_origin(child)
    return module

def disable_origin(module: nn.Module):
    module.origin = False
    for name, child in module.named_children():
        if isinstance(child, QLinear) or isinstance(child, QPlif) or isinstance(child, Qexp) or isinstance(child, Qsigmoid):
            child.origin = False
        else:
            disable_origin(child)
    return module

def is_leaf_module(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, QLinear):
            return True
        else:
            return False

def get_spike_layer(model: nn.Module):
    """
    Get the activation layer in the model
    :param model: Model
    :return: List of spike layer
    """
    spike_layers = {}
    spike_after_sign = 0
    fc_name = None
    snn = True
    search_res = OrderedDict((k, v) for k, v in OrderedDict(model.named_modules()).items())
    # print(search_res.keys())
    keys = list(search_res.keys())
    for name, module in model.named_modules():
        if isinstance(module, QLinear) :
            spike_after_sign = 1
            fc_name = name
        elif isinstance(module, QPlif) and spike_after_sign:
            spike_layers[fc_name] = module
            spike_after_sign = 0
            continue
        elif isinstance(module, nn.ReLU): # 原始INGP
            spike_layers[fc_name] = module
            spike_after_sign = 0
            continue
            
        # if snn:
        if name == 'sigma_net.2':
            spike_layers[name] = search_res['quan_exp']
        elif name == 'color_net.4':
            spike_layers[name] = search_res['quan_sigmoid']
        # else:
            
        
    if len(spike_layers) == 0:
        raise ValueError("No spike layer found in the model")
    # print(spike_layers)
    return spike_layers

def cal_recon_loss(orig, quan):
    return (torch.norm(quan - orig, p="fro", dim=1) ** 2).mean()

def cal_round_loss(round_vals, b):
    return (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()

class StopForwardException(Exception):
    """
    Dummy exception to early-terminate forward-pass
    """

class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Linear annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

def prepare_inout(model, module, model_input, collect_input=False, collect_output = False, fwd = None):
    def _hook_to_collect_inp_out_data(_, inp, out):
        """
        hook to collect input and output data
        """
        if collect_input:
            inp_data_list.append(inp[0])

        if collect_output:
            out_data_list.append(out)

        raise StopForwardException

    inp_data_list = []
    out_data_list = []

    handle = module.register_forward_hook(_hook_to_collect_inp_out_data)

    # get the model's device placement information
    device = torch.device("cuda")
    model.eval()
    # place the input to appropriate device
    if isinstance(model_input, torch.Tensor):
        model_input = model_input.to(device)
    else:
        for idx,img in enumerate(model_input):
            model_input[idx] = img.to(device)

    # Custom injected exception is raised when the activations data from desired module is collected.
    try:
        if not fwd:
            model(model_input)
        else:
            fwd(model_input)
    except StopForwardException:
        pass
    except AssertionError as e:
        print(str(e))
        pass
    model.train()
    handle.remove()
    inp_data, out_data = None, None

    if inp_data_list and isinstance(inp_data_list[0], torch.Tensor):
        # 多个time step
        if len(inp_data_list) > 1:
            inp_data = torch.stack(inp_data_list).detach()
        else:
            inp_data = inp_data_list[0].detach()

    if out_data_list and isinstance(out_data_list[0], torch.Tensor):
        if len(out_data_list) > 1:
            out_data = torch.stack(out_data_list).detach()
        else:
            out_data = out_data_list[0].detach()
    return inp_data, out_data

def prepare_data_for_layer(inputx, model, module, fwd):
    origin = [] # output
    quant = [] # input
    enable_origin(model)        
    _ , origin_out = prepare_inout(model, module, inputx, collect_output = True, fwd = fwd)
    origin.append(origin_out)
    disable_origin(model)
    quant_in, _ = prepare_inout(model, module, inputx, collect_input = True, fwd = fwd)
    quant.append(quant_in)
    return quant, origin

def run_hook_for_layers_with_given_input(model: torch.nn.Module, input_tensor,
                                         hook, module_type_for_attaching_hook=None, leaf_node_only=True):
    """
    Register the given hook function for all layers in the model
    :param model: Model
    :param input_tensor: Input tensor to the model. If more than one model inputs, use a tuple
    :param hook: Hook function to register
    :param module_type_for_attaching_hook: Tuple of torch.nn module types for which hook has to be attached
    :param leaf_node_only: Set to False if all modules are required
    :return: None
    """

    # ------------------------
    # Register hook function
    # ------------------------
    hooks = []
    
    if module_type_for_attaching_hook:
        # if needed, filter by module types specified by caller
        modules = [module for module in modules if isinstance(module, module_type_for_attaching_hook)]
    else:
        # All leaf modules
        modules = [module for module in model.modules() if not leaf_node_only or is_leaf_module(module)]
    # print('modules: ', modules)
    for module in modules:
        if isinstance(module, nn.ModuleList):
            for sub_module in module:
                if isinstance(sub_module, QLinear):
                    hooks.append(sub_module.register_forward_hook(hook))
        else:
            hooks.append(module.register_forward_hook(hook))
    # print('modules', modules)
    # ------------------------------------------------
    # Run forward pass to execute the hook functions
    # ------------------------------------------------
    model.eval()
    with torch.no_grad():
        if isinstance(input_tensor, (list, tuple)):
            _ = model(*input_tensor)
        else:
            _ = model(input_tensor[0], input_tensor[1], rays =torch.randn(4096, 2)*100, shading='full')
    model.train()
    # --------------------------
    # Remove all hooks we added
    # --------------------------
    for h in hooks:
        h.remove()

def get_ordered_list_of_modules(model: torch.nn.Module, dummy_input = torch.randn((3,3)).cuda(), t = 2) :
    """
    Finds ordered modules in given model.
    :param model: PyTorch model.
    :param dummy_input: Dummy input to the model. Used to parse model graph.
    :return: List of module name, module in order.
    """
    def _hook_to_collect_name_of_module(module, input, output):
        """
        hook to find name of module
        """
        if isinstance(module, QLinear):
            module_name = module_to_name_dict[module]
            list_modules.append([module_name, module])
        elif isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            for name, child in module.named_children():

                if isinstance(child, QLinear):
                    module_name = module_to_name_dict[child]
                    list_modules.append([module_name, child])   
    
    module_to_name_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, QLinear):
            module_to_name_dict[module] = name

    list_modules = []
    
    enable_origin(model)
    run_hook_for_layers_with_given_input(model, dummy_input, hook=_hook_to_collect_name_of_module)
    disable_origin(model)
    model.clear_embedding()
    return list_modules[:len(list_modules) // t]

def calibrate_adaround(model_name, model, adaround_iter, b_start, b_end, warmup, trainloader, device, logger=None, load_ckpt=False):
    logger.info("adaround_iter: {}".format(adaround_iter))   #这一部分就是在ti梯度下降求 公式25 21的 argmin

    for name, child in model.named_modules():
        if isinstance(child, AdaRoundQuantizer):
            child.alpha.requires_grad = False

        if isinstance(child, QLinear):    
            child.weight.requires_grad=False
            if child.bias is not None:
                child.bias.requires_grad=False

    list_modules = get_ordered_list_of_modules(model)

    if model_name == 'snn':
        module_act_fundict = get_spike_layer(model)
    else:
        logger.error('unknown model')
        exit()
        
    disable_origin(model)
    if load_ckpt:
        thatall = torch.load(f"./ckpt/round_{model_name}.pth")
        that = thatall['net']
        dicts = model.state_dict()
        dicts.update(that)
        model.load_state_dict(dicts)
        allready_train = thatall['module_name']
        assert adaround_iter==thatall['adaround_iter']
        for idx, (name, module) in enumerate(list_modules):
            if is_leaf_module(module):
                module.weight_quantizer.alpha.requires_grad = False
            if name == allready_train:
                list_modules = list_modules[idx+1:]
                break
    
    # Adaround quantization is layer by layer
    for name, module in list_modules:
        module.weight_quantizer.alpha.requires_grad = True
        starttime = time.time()
        reconlist = []
        model = model.to(device)
        pre_inputdata, pre_outputdata = prepare_data_for_layer(trainloader, model, module)
        module.origin = False
        optimizer = torch.optim.Adam([module.weight_quantizer.alpha])
        logger.info('start {}'.format(name))
        logger.info('alpha: {}'.format(module.weight_quantizer.alpha))
        temp_decay = LinearTempDecay(adaround_iter, rel_start_decay=warmup,start_b=b_start, end_b=b_end)
        # total_loss
        with tqdm(total = adaround_iter, leave=False, desc='adaround') as pbar:
            for j in range(adaround_iter):
                b = temp_decay(j)
                recon_loss_iter = []
                for idx in range(len(pre_inputdata)):                   
                    pre_input = pre_inputdata[idx].to(device)
                    orig_output = pre_outputdata[idx].to(device)
                    optimizer.zero_grad()
                    quan_output = module(pre_input)
                    act_func = module_act_fundict[name] if name in module_act_fundict.keys() else None
                    if act_func is not None:
                        # SNN的话注重输出脉冲的一致性，不管膜电位相差如何，输出脉冲一致就行
                        quan_output = act_func(quan_output)
                        orig_output = act_func(orig_output)
                    recon_loss = cal_recon_loss(orig_output, quan_output)
                    that = recon_loss.detach().data.item() 
                    recon_loss_iter.append(that)
                    round_loss = cal_round_loss(module.weight_quantizer.get_soft_targets(), b)
                    total_loss = recon_loss + round_loss

                    # Back propagate and Update the parameter 'alpha'
                    total_loss.backward()
                    optimizer.step()
                recon_loss_iter = sum(recon_loss_iter)
                reconlist.append(recon_loss_iter)
                pbar.set_postfix({'recon_loss': recon_loss_iter, 'round_loss': round_loss.data.item()})
                pbar.update(1)
                
        logger.info('{} over! time:{} recon_loss_init:{}, recon_loss_last:{}'.format(name, time.time()-starttime, reconlist[0], reconlist[-1]))
        torch.save({'net':model.state_dict(), 'adaround_iter':adaround_iter, 'module_name':name}, f'./ckpt/round_{model_name}_wwq.pth')
        module.weight_quantizer.alpha.requires_grad = False
        module = module.to('cpu')
        logger.info('alpha: {}'.format(module.weight_quantizer.alpha))

    for name, child in model.named_modules():
        if isinstance(child, AdaRoundQuantizer):
            child.soft_targets = False  
            
def close_to_2(x, mode = 'round'):
    if mode == 'round':
        ids = torch.round(torch.log2(x))
    elif mode == 'floor':
        ids = torch.floor(torch.log2(x))
    elif mode == 'ceil':
        ids = torch.ceil(torch.log2(x))
    else:
        raise ValueError("mode should be 'round' or 'floor'")
    return torch.tensor(2 ** ids), ids

def grad_hook(module, grad_input, grad_output):
    if module.raw_grad is None:
        module.raw_grad = []
    module.raw_grad.append(grad_output[0].detach())
