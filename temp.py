from spike import spike_model
import torch
import torch.nn.functional as F
from config.GetConfig import get_config
from spikingjelly.clock_driven.neuron import ParametricLIFNode as PLIF
from collections import OrderedDict
from nerf.provider import NeRFDataset
import math
import raymarching
from encoding import get_encoder
from quanlib.utils import *
from nerf.utils import Trainer, PSNRMeter, SSIMMeter, LPIPSMeter, seed_everything

# torch.random.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

seed_everything(42)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt = get_config()
snn = spike_model.SMLP(opt).to(device)


# for name, module in snn.named_modules():
#     if isinstance(module, torch.nn.Module):
#         print(f"Module Name: {name}, Module Type: {type(module)}")
# Module Name: , Module Type: <class 'spike.spike_model.SMLP'>
# Module Name: encoder, Module Type: <class 'gridencoder.grid.GridEncoder'>
# Module Name: sigma_net, Module Type: <class 'torch.nn.modules.container.Sequential'>
# Module Name: sigma_net.0, Module Type: <class 'torch.nn.modules.linear.Linear'>
# Module Name: sigma_net.1, Module Type: <class 'spikingjelly.clock_driven.neuron.ParametricLIFNode'>
# Module Name: sigma_net.1.surrogate_function, Module Type: <class 'spikingjelly.clock_driven.surrogate.Sigmoid'>
# Module Name: sigma_net.2, Module Type: <class 'torch.nn.modules.linear.Linear'>
# Module Name: encoder_dir, Module Type: <class 'shencoder.sphere_harmonics.SHEncoder'>
# Module Name: color_net, Module Type: <class 'torch.nn.modules.container.Sequential'>
# Module Name: color_net.0, Module Type: <class 'torch.nn.modules.linear.Linear'>
# Module Name: color_net.1, Module Type: <class 'spikingjelly.clock_driven.neuron.ParametricLIFNode'>
# Module Name: color_net.2, Module Type: <class 'torch.nn.modules.linear.Linear'>
# Module Name: color_net.3, Module Type: <class 'spikingjelly.clock_driven.neuron.ParametricLIFNode'>
# Module Name: color_net.4, Module Type: <class 'torch.nn.modules.linear.Linear'>
# modules = []
# search_res = OrderedDict((k, v) for k, v in OrderedDict(snn.named_modules()).items())
# print('=' * 20, 'search_res' , '=' * 20)
# print(search_res)
# print('=' * 20, 'search_res' , '=' * 20)
# for name, module in snn.named_modules():
#     if isinstance(module, torch.nn.Linear) or isinstance(module, PLIF):
#         modules.append(name)
#         print(f"Module Name: {name}, Module Type: {type(module)}")
#     for i in range(len(modules)):
#         print(f'Module [{i}]: {search_res[modules[i]]}')
# Module Name: sigma_net.0, Module Type: <class 'torch.nn.modules.linear.Linear'>
# Module Name: sigma_net.1, Module Type: <class 'spikingjelly.clock_driven.neuron.ParametricLIFNode'>
# Module Name: sigma_net.2, Module Type: <class 'torch.nn.modules.linear.Linear'>
# Module Name: color_net.0, Module Type: <class 'torch.nn.modules.linear.Linear'>
# Module Name: color_net.1, Module Type: <class 'spikingjelly.clock_driven.neuron.ParametricLIFNode'>
# Module Name: color_net.2, Module Type: <class 'torch.nn.modules.linear.Linear'>
# Module Name: color_net.3, Module Type: <class 'spikingjelly.clock_driven.neuron.ParametricLIFNode'>
# Module Name: color_net.4, Module Type: <class 'torch.nn.modules.linear.Linear'>

from quanlib.utils import *

print('=' * 20, 'snn' , '=' * 20)
# for name, module in snn.named_modules():
#     # if isinstance(module, torch.nn.Linear):
#         print(f"Module Name: {name}, Module Type: {type(module)}")
# search_res = OrderedDict((k, v) for k, v in OrderedDict(snn.named_modules()).items())
# print(search_res.keys())

print('=' * 20, 'qsnn' , '=' * 20)
# qsnn = inplace_net(snn)
fc = {}
# for name, module in qsnn.named_modules():
#     # if isinstance(module, torch.nn.Linear) or isinstance(module, PLIF):
#     # if isinstance(module, nn.Module):
#         fc[name] = module
#         print(f"Module Name: {name}, Module Type: {type(module)}")
# print(fc)
# search_res_q = OrderedDict((k, v) for k, v in OrderedDict(qsnn.named_modules()).items())
# print(search_res_q.keys())

# spike_id = get_spike_layer(qsnn)
# print(spike_id)

dataloaders = NeRFDataset(opt, device=device, type='train').dataloader()
# print(len(dataloaders)) # 100
# for i, data in enumerate(dataloaders):
#     rays_o = data['rays_o']
#     rays_d = data['rays_d']
#     index = data['index']
#     cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None
#     cascade = max(1 + math.ceil(math.log2(opt.bound)), 1)
#     aabb_train = qsnn.aabb_train
#     nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,
#                                                      aabb_train ,
#                                                      opt.min_near)
#     xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, opt.bound,
#                                                                 opt.contract, qsnn.density_bitfield,
#                                                                 cascade, opt.grid_size, nears, fars, True,
#                                                                 opt.dt_gamma, opt.max_steps)
#     encoder, in_dim = get_encoder(encoding="hashgrid", desired_resolution=2048 * opt.bound)
#     encoder = encoder.to(device)
    
#     # features = encoder(xyzs)
#     dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
#     # outputs = qsnn(xyzs, dirs, rays = rays, shading='full')
#     print(f'xyzs.shape: {xyzs.shape}')
#     print(f'dirs.shape: {dirs.shape}')
#     print(f'rays.shape: {rays.shape}')
#     print(f'rays_o.shape: {rays_o.shape}')
#     print(f'rays_d.shape: {rays_d.shape}')

#     break
# qsnn.eval()
# qsnn.to(device)
# enable_calibrate(qsnn)
opt.not_quan_model_path = '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic_dynamic/chair2/checkpoints/ngp_ep0294.pth'
criterion = torch.nn.SmoothL1Loss(reduction='none')
trainer = Trainer('ngp', opt, snn, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, use_checkpoint=opt.ckpt, world_size= 1)

# trainer.calibrate(dataloaders, 1)

# trainer.quan_train(dataloaders)
trainer.load_alreay_quan_model('/home/linranxi/Code/PATA/workspace/quan/test/chair.pth')
disable_calibrate(trainer.model)  # disable calibrate, so that the model will not be quantized again
trainer.model.eval()
valid_loader = NeRFDataset(opt, device=device, type='val').dataloader()
trainer.metrics = [PSNRMeter(),]

trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]
trainer.evaluate(valid_loader)

test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
if test_loader.has_gt:
    trainer.evaluate(test_loader) # blender has gt, so evaluate it.

trainer.test(test_loader, write_video=False) # test and save video
