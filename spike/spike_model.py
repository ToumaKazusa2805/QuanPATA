import torch
import torch.nn as nn
import spikingjelly.clock_driven.neuron as neuron
from spikingjelly.clock_driven import surrogate
from encoding import get_encoder
from nerf import renderer  
from activation import trunc_exp
import raymarching
import sympy as sp
from functools import reduce
import numpy as np
import sys
sys.path.append("..")
from quanlib.adaround import Qact

def hook_fn(module, input, output, record, key:str):
    record[key] += output.sum().item()
    return output

class SMLP(renderer.NeRFRenderer):
    def __init__(self, 
                opt,
                device = 'cuda'):
        super().__init__(opt)
        self.num_layers_sigma = opt.num_layers_sigma
        self.hidden_dim_sigma = opt.hidden_dim_sigma
        self.hidden_dim_color = opt.hidden_dim_color
        self.num_layers_color = opt.num_layers_color
        self.output_dim_sigma = opt.output_dim_sigma    
        self.Init_Time_step = int(opt.Time_step)   # Record the initial value of the time step
        self.Time_step = int(max(opt.Time_step // 4, 1)) # Avoid OOM
        self.device = device
        assert opt.forward_type in ['pure', 'skip', 'hybrid'], 'forward_type must be in [pure, skip, hybrid], while got {}'.format(opt.forward_type)
        self.forward_type = opt.forward_type
        self.neuron_type = opt.neuron_type
        self.encoder, self.in_dim = get_encoder(encoding="hashgrid", desired_resolution=2048 * opt.bound)
        self.global_step = 0
        self.origin = False
        self.vth = opt.vth
        out_dim = 3
        
        if opt.D or opt.ptq:
            # Trainable time step
            self.init_w = nn.Parameter(torch.tensor(self.Time_step - 1, device = device))
        else:
            self.init_w = torch.tensor(self.Time_step, device = device)
        
        if self.neuron_type == 'LIF':
            self.tau = opt.tau
            spike_neuron_param = {'v_threshold': self.vth, 'v_reset': None, 'decay_input': True, 'tau': self.tau, 'detach_reset': False}
            neurons = neuron.LIFNode 
            self.init_b = nn.Parameter(-torch.log(torch.tensor(opt.tau - 1., device = device)))
            self.learned_tau = 1 / self.init_b.sigmoid()
            self.iterators = lambda x, y : (1 - 1 / self.learned_tau) * x + 1 / self.learned_tau * y
            self.avg_time = 1
        elif self.neuron_type == 'IF':
            spike_neuron_param = {'v_threshold': self.vth, 'v_reset': None, 'detach_reset': False}
            neurons = neuron.IFNode
            self.iterators = lambda x, y : x + y
            self.avg_time = opt.Time_step
        elif self.neuron_type == 'Custom_LIF':
            self.lambda_tau_time = 2
            self.Time_step = max(1, self.Time_step)
            
            self.init_b = nn.Parameter(-torch.log(torch.tensor(opt.tau - 1., device = device)))
            self.learned_tau = 1 / self.init_b.sigmoid()
            self.tau_time = torch.clamp(self.init_w, min=1, max=self.Time_step)
            spike_neuron_param = {'v_threshold': self.vth, 'v_reset': None, 'decay_input': False, 'init_tau':opt.tau, 'detach_reset': False}
            neurons = neuron.ParametricLIFNode
            self.iterators = lambda x, y : (1 - 1 / self.learned_tau) * x + 1 / self.learned_tau * y
            self.avg_time = 1
        else:
            raise ValueError('neuron_type must be in [LIF, IF, Custom_LIF], while got {}'.format(opt.neuron_type))

        sigma_net = []
        color_net = []

        for i in range(self.num_layers_sigma):
            if i == 0:
                sigma_net.extend([nn.Linear(self.in_dim, self.hidden_dim_sigma, bias=False), neurons(**spike_neuron_param)])
            elif i == self.num_layers_sigma - 1:
                sigma_net.append(nn.Linear(self.hidden_dim_sigma, 1 + self.output_dim_sigma, bias=False))
            else:
                sigma_net.extend([nn.Linear(self.hidden_dim_sigma, self.hidden_dim_sigma, bias=False), neurons(**spike_neuron_param)])

        self.sigma_net = nn.Sequential(*sigma_net)

        self.encoder_dir, self.in_dim_dir = get_encoder(encoding="sh")

        for i in range(self.num_layers_color):
            if i == 0:
                color_net.extend([nn.Linear(self.in_dim_dir + self.output_dim_sigma, self.hidden_dim_color, bias=False), neurons(**spike_neuron_param)])
            elif i == self.num_layers_color - 1:
                color_net.append(nn.Linear(self.hidden_dim_color, out_dim, bias=False))
            else:
                color_net.extend([nn.Linear(self.hidden_dim_color, self.hidden_dim_color, bias=False), neurons(**spike_neuron_param)])
                
        self.color_net = nn.Sequential(*color_net)

        # Recode the spike number of each spike layer 
        self.record = {'num':0, 'sigma_1':0,'color_1':0, 'color_3':0}
        
        self.sigma_hook = self.sigma_net[1].register_forward_hook(lambda module, input, output: hook_fn(module, input, output, self.record, 'sigma_1'))
        self.color_hook = self.color_net[1].register_forward_hook(lambda module, input, output: hook_fn(module, input, output, self.record, 'color_1'))
        self.color3_hook = self.color_net[3].register_forward_hook(lambda module, input, output: hook_fn(module, input, output, self.record, 'color_3'))

        self.reset_state(self.sigma_net)
        self.reset_state(self.color_net)
        
        # Quantization
        self.dirs = None
        self.xyz_embedding = None
        if opt.ptq:
            self.quan_sigmoid = Qact.Qsigmoid(opt.bit, opt.ptq, opt.symmetric)
            self.quan_exp = Qact.Qexp(opt.bit, opt.ptq, opt.symmetric)
            self.quan_sigma = nn.Identity()
            self.quan_color = nn.Identity()
            
    def pure_forward(self, x, d):
        # denstiy
        final_sigma = 0
        # RGB
        final_color = 0
        x = self.encoder(x, bound = self.bound)
        d = self.encoder_dir(d)
        for i in range(self.Time_step):
            sigma = self.sigma_net(x)
            exp_sigma = trunc_exp(sigma[..., 0])

            sigma = sigma[..., 1:]
            view_input = torch.concat([d, sigma], dim=-1)

            color_out = self.color_net(view_input)
            color = torch.sigmoid(color_out)

            final_sigma = self.iterators(final_sigma, exp_sigma)
            final_color = self.iterators(final_color, color)

        return {
            'sigma': final_sigma / self.avg_time,
            'color': final_color / self.avg_time,
        }
    
    def weight_time_step(self, T, tau):
            max = self.ste_round(T).data.int().item()
            round_T = self.ste_round(T)
            delta = torch.tensor(1.) - torch.tensor(1.) / tau
            acc = torch.tensor(1.) / tau
            if max == 1:
                return [delta ** (round_T - torch.tensor(1.))]
            delta_item = list(map(lambda x: delta ** (round_T - torch.tensor(1 + x)), range(0, max)))
            acc_item = list(map(lambda x: acc ** torch.tensor(int(x>=1)), range(max)))
            weight = list(map(lambda x, y: x * y, delta_item, acc_item))
            return weight
    
    # 
    def hybrid_forward(self, x, d):
        sigma_list = []
        color_list = []

        self.learned_tau = 1 / self.init_b.sigmoid()
        self.tau_time = torch.clamp(self.init_w, min=1, max=self.Init_Time_step)
        self.xyz_embedding = self.encoder(x, bound = self.bound)
        self.dirs = self.encoder_dir(d)
        # First time step: No decay mode
        self.reassignment(self.sigma_net, decay_input = False, item = 'decay_input')
        self.reassignment(self.color_net, decay_input = False, item = 'decay_input')

        sigma = self.sigma_net(self.xyz_embedding)        
        exp_sigma = trunc_exp(sigma[..., 0])
                      
        view_input = torch.concat([self.dirs, sigma[..., 1:]], dim=-1)
        color_out = self.color_net(view_input)
        color = torch.sigmoid(color_out)

        sigma_list.append(exp_sigma)
        color_list.append(color)
        
        self.record['num'] += self.dirs.shape[0]

        # Remaining time step: decay mode
        self.reassignment(self.sigma_net, decay_input = True, item = 'decay_input')
        self.reassignment(self.color_net, decay_input = True, item = 'decay_input')

        for i in range(1, self.Time_step):
            sigma = self.sigma_net(self.xyz_embedding)
            exp_sigma = trunc_exp(sigma[..., 0])

            view_input = torch.concat([self.dirs, sigma[..., 1:]], dim=-1)
            color_out = self.color_net(view_input)
            color = torch.sigmoid(color_out)

            sigma_list.append(exp_sigma)
            color_list.append(color)

        self.reset_state(self.sigma_net)
        self.reset_state(self.color_net)
        
        return {
            'sigma': sigma_list,
            'color': color_list,
        }

    def forward(self, x, d, **kwargs):
        # if not self.origin and self.opt.ptq:
        if self.opt.ptq: 
            return self.quan_skip_fwd(x, d)
        else:
            if self.forward_type == 'pure':
                return self.pure_forward(x, d)
            else:
                return self.hybrid_forward(x, d)
    
    def density(self, x, **kwargs):
        final_sigma = 0
        final_geo = 0
        h = self.encoder(x, bound=self.bound)
        if self.forward_type == 'skip':
            self.reassignment(self.sigma_net, decay_input = False, item = 'decay_input')
            out = self.sigma_net(h)
            exp_sigma = trunc_exp(out[..., 0])

            final_sigma =  exp_sigma
            final_geo = out[..., 1:]
            self.reassignment(self.sigma_net, decay_input = True, item = 'decay_input')
            
            if self.opt.D:
                iter_time = self.ste_round(self.tau_time).int().item()
            else:
                iter_time = self.Time_step
            
            for t in range(1, iter_time):
                out = self.sigma_net(h)
                exp_sigma = trunc_exp(out[..., 0])

                final_sigma = self.iterators(final_sigma, exp_sigma)
                final_geo = self.iterators(final_geo, out[..., 1:])
        else:
            for t in range(self.Time_step):
                out = self.sigma_net(h)
                exp_sigma = trunc_exp(out[..., 0])

                final_sigma = self.iterators(final_sigma, exp_sigma)
                final_geo = self.iterators(final_geo, out[..., 1:])

        self.reset_state(self.sigma_net)
        return {
            'sigma': final_sigma / self.avg_time,
            'geo_feat': final_geo / self.avg_time,
        }

    def apply_total_variation(self, w):
        self.grid_encoder.grad_total_variation(w)

    def apply_weight_decay(self, w):
        self.grid_encoder.grad_weight_decay(w)

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]

        return params

    def reset_state(self, net = None):
        
        if net is not None:
            for m in net:
                if isinstance(m, (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode)):
                    m.reset()
        else:
            for m in [self.sigma_net, self.color_net]:
                for layers in m:
                    if isinstance(layers, (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode)):
                        layers.reset()

    def reassignment(self, net = None, tau = 4.0, decay_input = True, item = 'decay_input'):
        if net is not None:
            if isinstance(net, nn.Sequential):
                for m in net:
                    if isinstance(m, (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode)):
                        setattr(m, item, eval(item))
            elif isinstance(net, (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode)):
                setattr(net, item, eval(item))
        else:
            for m in [self.sigma_net, self.color_net]:
                for layers in m:
                    if isinstance(layers, (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode)):
                        setattr(layers, item, eval(item))
                        
    def run_cuda(self, rays_o, rays_d, bg_color=None, perturb=False, cam_near_far=None, shading='full', **kwargs):
        # rays_o, rays_d: [N, 3]
        # return: image: [N, 3], depth: [N]
        
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        N = rays_o.shape[0]
        device = rays_o.device
        
        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,
                                                     self.aabb_train if self.training else self.aabb_infer,
                                                     self.min_near)
        if cam_near_far is not None:
            nears = torch.maximum(nears, cam_near_far[:, 0])
            fars = torch.minimum(fars, cam_near_far[:, 1])
        
        # mix background color
        if bg_color is None:
            bg_color = 1

        results = {}
        # torch.cuda.empty_cache()
        if self.training:

            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.real_bound,
                                                                self.opt.contract, self.density_bitfield,
                                                                self.cascade, self.grid_size, nears, fars, perturb,
                                                                self.opt.dt_gamma, self.opt.max_steps)

            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
            if self.forward_type == 'pure':
                with torch.amp.autocast('cuda', enabled=self.opt.fp16):
                    outputs = self(xyzs, dirs, shading=shading)
                    sigmas = outputs['sigma']
                    rgbs = outputs['color']

                weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays,
                                                                                    self.opt.T_thresh)
                results['num_points'] = xyzs.shape[0]
                results['weights'] = weights
                results['weights_sum'] = weights_sum
            else:
                results['weights'] = []
                results['weights_sum'] = []
                results['depth'] = []
                results['image'] = []
                results['extra_out'] = 0
                results['First_out'] = []
                images = []
                with torch.amp.autocast('cuda', enabled=self.opt.fp16):
                    self.outputs = self(xyzs, dirs, shading=shading)
                    sigmas_list = self.outputs['sigma']
                    rgbs_list = self.outputs['color']
                    clip_time = self.ste_round(self.tau_time).int().item()
                    
                    # clip_time <= self.Time_step
                    if self.opt.D:
                        # t* - 1
                        if clip_time > 1:
                            time_weights = self.weight_time_step(self.tau_time - 1, self.learned_tau)
                            sigma_out = reduce(lambda x, y: x + y, [sigmas_list[j] * time_weights[j] for j in range(clip_time-1)])
                            rgb_out = reduce(lambda x, y: x + y, [rgbs_list[j] * time_weights[j] for j in range(clip_time-1)])
                            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigma_out, rgb_out, ts, rays,
                                                                                                self.opt.T_thresh)
                            results['num_points'] = xyzs.shape[0]
                            results['weights'].append(weights)
                            results['weights_sum'].append(weights_sum)
                            results['depth'].append(depth)
                            images.append(image)
                            
                            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas_list[0], rgbs_list[0], ts, rays,
                                                                                                self.opt.T_thresh)
                            results['First_out'].append(image)
                            
                        else:
                            results['num_points'] = torch.zeros(1, device=self.device)
                            results['weights'].append(torch.zeros(1, device=self.device))
                            results['weights_sum'].append(torch.zeros(1, device=self.device))
                            results['depth'].append(torch.zeros(1, device=self.device))
                            images.append(torch.zeros(1, device=self.device))

                        # t*
                        time_weights = self.weight_time_step(self.tau_time, self.learned_tau)
                        sigma_out = reduce(lambda x, y: x + y, [sigmas_list[j] * time_weights[j] for j in range(clip_time)])
                        rgb_out = reduce(lambda x, y: x + y, [rgbs_list[j] * time_weights[j] for j in range(clip_time)])
                        weights, weights_sum, depth, image = raymarching.composite_rays_train(sigma_out, rgb_out, ts, rays,
                                                                                            self.opt.T_thresh)
                        results['num_points'] = xyzs.shape[0]
                        results['weights'].append(weights)
                        results['weights_sum'].append(weights_sum)
                        results['depth'].append(depth)
                        images.append(image)
                        
                        # if clip_time != self.Time_step:
                        if clip_time + 1 == self.Time_step:
                            time_weights = self.weight_time_step(self.tau_time+1, self.learned_tau)
                            sigma_out = reduce(lambda x, y: x + y, [sigmas_list[j] * time_weights[j] for j in range(clip_time+1)])
                            rgb_out = reduce(lambda x, y: x + y, [rgbs_list[j] * time_weights[j] for j in range(clip_time+1)])
                            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigma_out, rgb_out, ts, rays,
                                                                                                self.opt.T_thresh)
                            results['num_points'] = xyzs.shape[0]
                            results['weights'].append(weights)
                            results['weights_sum'].append(weights_sum)
                            results['depth'].append(depth)
                            images.append(image)
                            
                            extra_out = [torch.tensor(0.), torch.tensor(0., )]
                            
                        elif clip_time + 1 < self.Time_step:
                            
                            time_weights = self.weight_time_step(self.tau_time+1, self.learned_tau)
                            sigma_out = reduce(lambda x, y: x + y, [sigmas_list[j] * time_weights[j] for j in range(clip_time+1)])
                            rgb_out = reduce(lambda x, y: x + y, [rgbs_list[j] * time_weights[j] for j in range(clip_time+1)])
                            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigma_out, rgb_out, ts, rays,
                                                                                                self.opt.T_thresh)
                            results['num_points'] = xyzs.shape[0]
                            results['weights'].append(weights)
                            results['weights_sum'].append(weights_sum)
                            results['depth'].append(depth)
                            images.append(image)
                            
                            time_weights = self.weight_time_step(self.Time_step, self.learned_tau)
                            sigma_out = reduce(lambda x, y: x + y, [sigmas_list[j] * time_weights[j] for j in range(self.Time_step)])
                            rgb_out = reduce(lambda x, y: x + y, [rgbs_list[j] * time_weights[j] for j in range(self.Time_step)])
                            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigma_out, rgb_out, ts, rays,
                                                                                                self.opt.T_thresh)
                            results['num_points'] = xyzs.shape[0]
                            results['weights'].append(weights)
                            results['weights_sum'].append(weights_sum)
                            results['depth'].append(depth)
                            images.append(image)

                            extra_out = [torch.norm(reduce(lambda x, y: x + y, [sigmas_list[j] * time_weights[j] for j in range(self.Time_step)][clip_time:]), p = 2) / torch.norm(results['weights'][-1], p = 2), \
                                        torch.norm(reduce(lambda x, y: x + y, [rgbs_list[j] * time_weights[j] for j in range(self.Time_step)][clip_time:]), p = 2)]

                    else:
                        time_weights = self.weight_time_step(torch.tensor(self.Time_step), self.learned_tau)
                        sigma_out = reduce(lambda x, y: x + y, [sigmas_list[j] * time_weights[j] for j in range(self.Time_step)])
                        rgb_out = reduce(lambda x, y: x + y, [rgbs_list[j] * time_weights[j] for j in range(self.Time_step)])
                        weights, weights_sum, depth, image = raymarching.composite_rays_train(sigma_out, rgb_out, ts, rays,
                                                                                            self.opt.T_thresh)
                        results['num_points'] = xyzs.shape[0]
                        results['weights'].append(weights)
                        results['weights_sum'].append(weights_sum)
                        results['depth'].append(depth)
                        images.append(image)
                        extra_out = [torch.tensor(0.), torch.tensor(0., )]
                        
                    results['extra_out'] = extra_out
                    self.sigma_cauchy_loss = self.cauchy_loss(sigma_out)
        else:
            dtype = torch.float32
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]
            
            step = 0
            while step < self.opt.max_steps:
                n_alive = rays_alive.shape[0]
                
                if n_alive <= 0:
                    break

                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d,
                                                        self.real_bound, self.opt.contract, self.density_bitfield,
                                                        self.cascade, self.grid_size, nears, fars,
                                                        perturb if step == 0 else False,
                                                        self.opt.dt_gamma, self.opt.max_steps)
                
                dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
                with torch.amp.autocast('cuda', enabled=self.opt.fp16):
                    self.outputs = self(xyzs, dirs, shading=shading)
                    if self.forward_type == 'skip':
                        sigmas_list = self.outputs['sigma']
                        rgbs_list = self.outputs['color']
                        time_weights = self.weight_time_step(torch.tensor(self.Time_step), self.learned_tau)
                        sigmas = reduce(lambda x, y: x + y, [sigmas_list[j] * time_weights[j] for j in range(self.Time_step)])
                        rgbs = reduce(lambda x, y: x + y, [rgbs_list[j] * time_weights[j] for j in range(self.Time_step)])
                    else:
                        sigmas = outputs['sigma']
                        rgbs = outputs['color']

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum,
                                           depth, image, self.opt.T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step
                
        if self.forward_type == 'skip' and self.training:
            for i in range(len(images)):
                image = images[i] + (1 - results['weights_sum'][i]).unsqueeze(-1) * bg_color
                results['image'].append(image)

        else:
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            # [4096, 3], [4096, 3]
            results['depth'] = depth
            results['image'] = image

        return results
    
    def ste_round(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return torch.round(x) - x.detach() + x
    
    def cauchy_loss(self, sigma):
        return torch.log(1 + 2 * sigma ** 2).mean()
    
    #################################Quantization#####################################
    
    def quan_skip_fwd(self, x, d):
        sigma_list = []
        color_list = []
        self.soft_data = {}
        
        self.learned_tau = 1 / self.init_b.sigmoid()
        self.tau_time = torch.clamp(self.init_w, min=1, max=self.Init_Time_step)

        if self.training:
            if self.xyz_embedding is None:
                self.xyz_embedding = self.encoder(x, bound = self.bound)
                
            if self.dirs is None:
                dirs = self.encoder_dir(d)
                self.dirs = dirs
            else:
                dirs = self.dirs
                
        else:
            self.xyz_embedding = self.encoder(x, bound = self.bound)
            dirs = self.encoder_dir(d)
        # print(self.training)
        # First time step: No decay mode
        self.reassignment(self.sigma_net, decay_input = False, item = 'decay_input')
        self.reassignment(self.color_net, decay_input = False, item = 'decay_input')

        # print(f'Memory Usage before quant forward: {torch.cuda.memory_reserved() / 1024 ** 3} GB')
        sigma = self.sigma_net(self.xyz_embedding)    
        exp_sigma = self.quan_exp(sigma[..., 0])    # 注意这里的exp是直接调用的torch.exp，不是修改后的exp
        # print(f'Memory Usage after quant sigma: {torch.cuda.memory_reserved() / 1024 ** 3} GB')    
        view_input = torch.concat([dirs, sigma[..., 1:]], dim=-1)
        color_out = self.color_net(view_input)
        color = self.quan_sigmoid(color_out)
        # print(f'Memory Usage after quant color: {torch.cuda.memory_reserved() / 1024 ** 3} GB')   
        sigma_list.append(exp_sigma)
        color_list.append(color)
        
        self.record['num'] += dirs.shape[0]

        # Remaining time step: decay mode
        self.reassignment(self.sigma_net, decay_input = True, item = 'decay_input')
        self.reassignment(self.color_net, decay_input = True, item = 'decay_input')

        # if self.encoder.quant:
        # print(f'[INFO] Memory Usage During 0-time step Quant Forward: {torch.cuda.memory_allocated() / 1024**2}MB, shape = {self.xyz_embedding.shape}')

        # from pathlib import Path
        # save_path = Path('/home/liuwh/lrx/PATA_code/workspace/quan/chair_symmetric_8_8bit22')
        # end_path = save_path / 'sigmoid_t1.pt'
        # if not self.training and len(self.xyz_embedding.shape) >= 2 and not end_path.exists():
            
        #     # torch.cuda.empty_cache()
        #     for i in range(len(self.sigma_net)):
        #         if isinstance(self.sigma_net[i], nn.Linear):
        #             torch.save(self.sigma_net[i].input, save_path / f'sigma_linear_{i}_t0.pt')
        #         else:
        #             torch.save(self.sigma_net[i].input, save_path / f'sigma_Qplif_input_{i}_t0.pt')
        #             torch.save(self.sigma_net[i].v_q, save_path / f'sigma_Qplif_Mem_{i}_t0.pt')
            
        #     for i in range(len(self.color_net)):
        #         if isinstance(self.color_net[i], nn.Linear):
        #             torch.save(self.color_net[i].input, save_path / f'color_linear_{i}_t0.pt')
        #         else:
        #             torch.save(self.color_net[i].input, save_path / f'color_Qplif_input_{i}_t0.pt')
        #             torch.save(self.color_net[i].v_q, save_path / f'color_Qplif_Mem_{i}_t0.pt')

        #     torch.save(exp_sigma, save_path / 'Qexp_t0.pt')
        #     torch.save(color, save_path / 'sigmoid_t0.pt')
        #     print(f'Time step 0 finished!')
        for i in range(1, self.Time_step):
            # print(f'Memory Usage before second quant forward: {torch.cuda.memory_reserved() / 1024 ** 3} GB, input shape = {self.xyz_embedding.shape}')
            sigma = self.sigma_net(self.xyz_embedding)
            exp_sigma = self.quan_exp(sigma[..., 0])

            view_input = torch.concat([dirs, sigma[..., 1:]], dim=-1)
            color_out = self.color_net(view_input)
            color = self.quan_sigmoid(color_out)

            sigma_list.append(exp_sigma)
            color_list.append(color)
            
            # if self.encoder.quant:
            #     print(f'[INFO] Memory Usage During {i}-time step Quant Forward: {torch.cuda.memory_allocated() / 1024**2}MB')
            
            # if not self.training and len(self.xyz_embedding.shape) >= 2 and not end_path.exists():
            #     for id in range(len(self.sigma_net)):
            #         if isinstance(self.sigma_net[id], (nn.Linear)):
            #             torch.save(self.sigma_net[id].input, save_path / f'sigma_linear_{id}_t{i}.pt')
            #         else:
            #             torch.save(self.sigma_net[id].input, save_path / f'sigma_Qplif_input_{id}_t{i}.pt')
            #             torch.save(self.sigma_net[id].v_q, save_path / f'sigma_Qplif_Mem_{id}_t{i}.pt')
                
            #     for id in range(len(self.color_net)):
            #         if isinstance(self.color_net[id], (nn.Linear)):
            #             torch.save(self.color_net[id].input, save_path / f'color_linear_{id}_t{i}.pt')
            #         else:
            #             torch.save(self.color_net[id].input, save_path / f'color_Qplif_input_{id}_t{i}.pt')
            #             torch.save(self.color_net[id].v_q, save_path / f'color_Qplif_Mem_{id}_t{i}.pt')

            #     torch.save(exp_sigma, save_path / f'Qexp_t{i}.pt')
            #     torch.save(color, save_path / f'sigmoid_t{i}.pt')
            # print(f'Time step {i} finished!')
        # torch.save(self.soft_data, '/home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/chair_symmetric_8_8bit25/soft_data.pt')
        self.reset_state(self.sigma_net)
        self.reset_state(self.color_net)

        return {
            'sigma': sigma_list,
            'color': color_list,
        }
    
    def quan_run_cuda(self, rays_o, rays_d, bg_color=None, perturb=False, cam_near_far=None, shading='full', **kwargs):
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        N = rays_o.shape[0]
        device = rays_o.device
        
        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,
                                                     self.aabb_train if self.training else self.aabb_infer,
                                                     self.min_near)
        if cam_near_far is not None:
            nears = torch.maximum(nears, cam_near_far[:, 0])
            fars = torch.minimum(fars, cam_near_far[:, 1])

        # mix background color
        if bg_color is None:
            bg_color = 1

        results = {}
        if self.training:

            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.real_bound,
                                                                self.opt.contract, self.density_bitfield,
                                                                self.cascade, self.grid_size, nears, fars, perturb,
                                                                self.opt.dt_gamma, self.opt.max_steps)
            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
            # print('XYZs: ', xyzs.shape, dirs.shape)
            with torch.amp.autocast('cuda', enabled=self.opt.fp16):
                self.outputs = self(xyzs, dirs, shading=shading)
                sigmas_list = self.outputs['sigma']
                rgbs_list = self.outputs['color']

                time_weights = self.weight_time_step(self.Time_step, self.learned_tau)
                quan_sigmas_list = list(map(lambda x: self.quan_sigma(x), sigmas_list))
                quan_rgbs_list = list(map(lambda x: self.quan_color(x), rgbs_list))
                sigmas = reduce(lambda x, y: x + y, [quan_sigmas_list[j] * time_weights[j] for j in range(self.Time_step)])
                rgbs = reduce(lambda x, y: x + y, [quan_rgbs_list[j] * time_weights[j] for j in range(self.Time_step)])
                    
            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays,
                                                                                self.opt.T_thresh)
            results['num_points'] = xyzs.shape[0]
            results['weights'] = weights
            results['weights_sum'] = weights_sum
            
        else:
            dtype = torch.float32
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]
            
            step = 0
            while step < self.opt.max_steps:
                n_alive = rays_alive.shape[0]
                
                if n_alive <= 0:
                    break

                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d,
                                                        self.real_bound, self.opt.contract, self.density_bitfield,
                                                        self.cascade, self.grid_size, nears, fars,
                                                        perturb if step == 0 else False,
                                                        self.opt.dt_gamma, self.opt.max_steps)
                
                dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
                with torch.amp.autocast('cuda', enabled=self.opt.fp16):
                    self.outputs = self(xyzs, dirs, shading=shading)
                    if self.forward_type == 'skip':
                        if self.outputs == 0:
                            return 0
                        sigmas_list = self.outputs['sigma']
                        rgbs_list = self.outputs['color']
                        quan_sigmas_list = list(map(lambda x: self.quan_sigma(x), sigmas_list))
                        quan_rgbs_list = list(map(lambda x: self.quan_color(x), rgbs_list))
                        time_weights = self.weight_time_step(torch.tensor(self.Time_step), self.learned_tau)
                        sigmas = reduce(lambda x, y: x + y, [quan_sigmas_list[j] * time_weights[j] for j in range(self.Time_step)])
                        rgbs = reduce(lambda x, y: x + y, [quan_rgbs_list[j] * time_weights[j] for j in range(self.Time_step)])
                        
                    else:
                        sigmas = self.outputs['sigma']
                        rgbs = self.outputs['color']

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum,
                                           depth, image, self.opt.T_thresh)
                # ids = 
                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step
        
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        # [4096, 3], [4096, 3]
        results['depth'] = depth
        results['image'] = image
        return results
    
    def render(self, rays_o, rays_d, **kwargs):
        
        if self.opt.ptq:
            return self.quan_run_cuda(rays_o, rays_d, **kwargs)
        else:
            return self.run_cuda(rays_o, rays_d, **kwargs)
        
        
    def clear_embedding(self, ):
        self.dirs = None
        self.xyz_embedding = None
        for m in [self.sigma_net, self.color_net]:
            for layers in m:
                if isinstance(layers, (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode)):
                    layers.v_o = []
                    layers.v_q = []
        