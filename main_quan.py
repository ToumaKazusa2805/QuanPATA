from spike import spike_model
import torch
from config.GetConfig import get_config
from spikingjelly.clock_driven.neuron import ParametricLIFNode as PLIF
from collections import OrderedDict
from nerf.provider import NeRFDataset
from quanlib.utils import *
from nerf.utils import Trainer, PSNRMeter, SSIMMeter, LPIPSMeter, seed_everything
from quanlib.utils import *
from nerf.network import NeRFNetwork
import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")

seed_everything(42)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt = get_config()
if opt.net == 'snn':
    net = spike_model.SMLP(opt).to(device)
else:
    net = NeRFNetwork(opt).to(device)

if opt.data_format == 'colmap':
    from nerf.colmap_provider import ColmapDataset as NeRFDataset
elif opt.data_format == 'dtu':
    from nerf.dtu_provider import NeRFDataset
else: # nerf
    from nerf.provider import NeRFDataset

dataloaders = NeRFDataset(opt, device=device, type='train').dataloader()
criterion = torch.nn.SmoothL1Loss(reduction='none')
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, eps=1e-15)
trainer = Trainer('ngp', opt, model = net, device=device, workspace=opt.workspace, optimizer = optimizer,\
                  criterion=criterion, fp16=opt.fp16, use_checkpoint=opt.ckpt, world_size= 1, dataloader = dataloaders,
                  )
# trainer.model.Time_step = int(opt.Time_step * 4)
print(f'[INFO] Time Step for PTQ: {trainer.model.Time_step}')

trainer.quan_train(dataloaders)

valid_loader = NeRFDataset(opt, device=device, type='val').dataloader()
trainer.metrics = [PSNRMeter(),]

trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]
trainer.evaluate(valid_loader)
test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

# id = 0
# trainer.model.eval()
# for data in test_loader:
#     if id == 0:
#         id += 1
#         continue
#     _ = trainer.test_step(data)
#     break

if test_loader.has_gt:
    trainer.evaluate(test_loader) # blender has gt, so evaluate it.

trainer.test(test_loader, write_video=True) # test and save video
