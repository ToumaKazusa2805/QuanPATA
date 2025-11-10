import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np

import time

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function
try:
    from torchmetrics.functional import structural_similarity_index_measure
except: # old versions
    from torchmetrics.functional import ssim as structural_similarity_index_measure

import trimesh
from rich.console import Console
from torch_ema import ExponentialMovingAverage
from functools import reduce

from packaging import version as pver
import lpips

from quanlib.adaround.quantizer import AdaRoundQuantizer, AsymmetricQuantizer, Quantizer
from quanlib.adaround.observer import MinMaxObserver
import sys
sys.path.append("..")

from quanlib.utils import *
from pathlib import Path

class StopForwardException(Exception):
    """
    Dummy exception to early-terminate forward-pass
    """

class RoundToNearest(Function):

    @staticmethod
    def forward(ctx, x):
        return torch.round(x.data)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def create_dodecahedron_cameras(radius=1, center=np.array([0, 0, 0])):

    vertices = np.array([
        -0.57735,  -0.57735,  0.57735,
        0.934172,  0.356822,  0,
        0.934172,  -0.356822,  0,
        -0.934172,  0.356822,  0,
        -0.934172,  -0.356822,  0,
        0,  0.934172,  0.356822,
        0,  0.934172,  -0.356822,
        0.356822,  0,  -0.934172,
        -0.356822,  0,  -0.934172,
        0,  -0.934172,  -0.356822,
        0,  -0.934172,  0.356822,
        0.356822,  0,  0.934172,
        -0.356822,  0,  0.934172,
        0.57735,  0.57735,  -0.57735,
        0.57735,  0.57735,  0.57735,
        -0.57735,  0.57735,  -0.57735,
        -0.57735,  0.57735,  0.57735,
        0.57735,  -0.57735,  -0.57735,
        0.57735,  -0.57735,  0.57735,
        -0.57735,  -0.57735,  -0.57735,
        ]).reshape((-1,3), order="C")

    length = np.linalg.norm(vertices, axis=1).reshape((-1, 1))
    vertices = vertices / length * radius + center

    # construct camera poses by lookat
    def normalize(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    # forward is simple, notice that it is in fact the inversion of camera direction!
    forward_vector = normalize(vertices - center)
    # pick a temp up_vector, usually [0, 1, 0]
    up_vector = np.array([0, 1, 0], dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    # cross(up, forward) --> right
    right_vector = normalize(np.cross(up_vector, forward_vector, axis=-1))
    # rectify up_vector, by cross(forward, right) --> up
    up_vector = normalize(np.cross(forward_vector, right_vector, axis=-1))

    ### construct c2w
    poses = np.eye(4, dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    poses[:, :3, :3] = np.stack((right_vector, up_vector, forward_vector), axis=-1)
    poses[:, :3, 3] = vertices

    return poses


@torch.amp.autocast('cuda', enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1, coords=None, image=None):
    ''' get rays
    Args:
        poses: [N/1, 4, 4], cam2world
        intrinsics: [N/1, 4] tensor or [4] ndarray
        H, W, N: int
    Returns:
        rays_o, rays_d: [N, 3]
        i, j: [N]
    '''

    device = poses.device
    
    if isinstance(intrinsics, np.ndarray):
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2], intrinsics[:, 3]

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    results = {}

    if N > 0:
       
        if coords is not None:
            inds = coords[:, 0] * W + coords[:, 1]

        elif patch_size > 1:

            # random sample left-top cores.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten


        else: # random sampling
            # inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            if image is None:  # no input image used as ray selection reference.
                inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            else:
                # for synthetic dataset, valid pixel is the priority selection
                images = image[..., :3].view(1, -1, 3)
                inds_valid = torch.nonzero(torch.all(images != 0, dim=2))  # [B, N]
                # TODO: only B = 1 is considered here.
                inds_valid = inds_valid[:, 1]

                weights = torch.ones(H * W, dtype=torch.float32, device=device)
                weights[inds_valid] = 2.0
                inds = torch.multinomial(weights, N, replacement=False)

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['i'] = i.long()
        results['j'] = j.long()

    else:
        inds = torch.arange(H*W, device=device)

    zs = -torch.ones_like(i) # z is flipped
    xs = (i - cx) / fx
    ys = -(j - cy) / fy # y is flipped
    directions = torch.stack((xs, ys, zs), dim=-1) # [N, 3]
    # do not normalize to get actual depth, ref: https://github.com/dunbar12138/DSNeRF/issues/29
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True) 
    rays_d = (directions.unsqueeze(1) @ poses[:, :3, :3].transpose(-1, -2)).squeeze(1) # [N, 1, 3] @ [N, 3, 3] --> [N, 1, 3]

    rays_o = poses[:, :3, 3].expand_as(rays_d) # [N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    # visualize_rays(rays_o[0].detach().cpu().numpy(), rays_d[0].detach().cpu().numpy())

    return results


def visualize_rays(rays_o, rays_d):
    
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for i in range(0, rays_o.shape[0], 10):
        ro = rays_o[i]
        rd = rays_d[i]

        segs = np.array([[ro, ro + rd * 3]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [N, 3] or [H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

        return psnr

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='vgg', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous() # [3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [H, W, 3] --> [3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1

        return v
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous() # [3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [H, W, 3] --> [3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'


class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 save_interval=1, # save once every $ epoch (independently from eval)
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 dataloader = None
                 ):
        self.dataloader = dataloader
        self.opt = opt
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.record = {'num':0, 'sigma_1':0,'color_1':0, 'color_3':0}

        # try out torch 2.0
        # if torch.__version__[0] == '2':
        #     model = torch.compile(model)

        model.to(self.device)
        if self.world_size > 1:
            print('Using DDP ......')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # classify hash parameters
        # for name, param in self.model.encoder.named_parameters():
        #     self.hash_table = param
        # self.hash_grad = torch.zeros(opt.iters, self.hash_table.shape[0], self.hash_table.shape[1])
        # self.hash_grad = []

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        if not opt.D or opt.pth_path == None:
            if ema_decay is not None:
                self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
            else:
                self.ema = None


        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        # self.scaler = torch.amp.GradScaler('cuda', enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

            # grad saving path
            # self.hash_path = os.path.join(self.workspace, 'hash_grad_record')
            # os.makedirs(self.hash_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        self.log(opt)
        
        if self.workspace is not None:
            
            if opt.ptq and opt.net == 'snn':
                self.quaned_module = {}
                self.dicts = {'quaned_module_list':{}}
                self.log(f"[INFO] PTQ Mode, Loading original SNN model {opt.not_quan_model_path} ...")
                self.quan_model_dirs = os.path.join(self.workspace, 'checkpoints')
                if os.path.exists(self.quan_model_dirs) and len(os.listdir(self.quan_model_dirs)) > 0:
                    
                    quan_model_path = os.path.join(self.quan_model_dirs, os.listdir(self.quan_model_dirs)[-1])
                    self.log(f"[INFO] Found quantization model in {quan_model_path}")
                    self.load_alreay_quan_model(quan_model_path)
                else:
                    self.log(f"[INFO] Loading original model ...")
                    self.quan_load_model(opt.not_quan_model_path, model_only = True)
            elif opt.ptq and opt.net == 'ann':
                self.dicts = {'quaned_module_list':{}}
                self.log(f'[INFO] PTQ Mode, Loading original ANN model {opt.not_quan_model_path} ...')
                self.load_checkpoint(opt.not_quan_model_path)
                self.new_model = inplace_net(self.model, bit = self.opt.bit, symmetric=opt.symmetric, Mem_bit=opt.Mem_bit)
                self.model = self.new_model
                self.quan_model_dirs = os.path.join(self.workspace, 'checkpoints')
                setattr(self.model, 'Time_step', 1)
                print(self.model.Time_step)
            else:
                if not opt.D:
                    if self.use_checkpoint == "scratch":
                        self.log("[INFO] Training from scratch ...")
                    elif self.use_checkpoint == "latest":
                        self.log("[INFO] Loading latest checkpoint ...")
                        self.load_checkpoint()
                    elif self.use_checkpoint == "latest_model":
                        self.log("[INFO] Loading latest checkpoint (model only)...")
                        self.load_checkpoint(model_only=True)
                    elif self.use_checkpoint == "best":
                        if os.path.exists(self.best_path):
                            self.log("[INFO] Loading best checkpoint ...")
                            self.load_checkpoint(self.best_path)
                        else:
                            self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                            self.load_checkpoint()
                    else: # path to ckpt
                        self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                        self.load_checkpoint(self.use_checkpoint)
                else:
                    from pathlib import Path
                    if opt.pth_path != None and os.path.exists(Path(opt.pth_path)):
                        self.log(f'[INFO] Dynamic Time-step Training, Loading Pre-trained model...')
                        self.tit_load_checkpoint(opt.pth_path, model_only = True)
                    else:
                        self.log(f'[INFO] Dynamic Time-step Training, No Pre-trained model Found...')
                        
                    if os.path.exists(Path(self.workspace) / 'checkpoints') and len(os.listdir(Path(self.workspace) / 'checkpoints')) > 0:
                        best_model_file = os.path.join(Path(self.workspace) / 'checkpoints', os.listdir(Path(self.workspace) / 'checkpoints')[-1])
                        self.log(f'[INFO] Dynamic Time-step Training, Find Pre-trained TIT model, located in {best_model_file}')
                        self.load_checkpoint(best_model_file)

                    else:
                        self.log(f'[INFO] Dynamic Time-step Training, No Pre-trained TIT model Found...')
                
        self.log(self.model)
        
        
    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):

        # if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
        self.model.global_step = self.global_step
        self.local_step += 1
        self.global_step += 1

        rays_o = data['rays_o'] # [N, 3]
        rays_d = data['rays_d'] # [N, 3]
        index = data['index'] # [1/N]
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None # [1/N, 2] or None
        
        images = data['images'] # [N, 3/4]

        N, C = images.shape

        if self.opt.background == 'random':
            bg_color = torch.rand(N, 3, device=self.device) # [N, 3], pixel-wise random.
        else: # white / last_sample
            bg_color = 1

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        shading = 'diffuse' if self.global_step < self.opt.diffuse_step else 'full'
        update_proposal = self.global_step <= 3000 or self.global_step % 5 == 0
        
        outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=True,
                                    cam_near_far=cam_near_far, shading=shading, update_proposal=update_proposal)

        # MSE loss
        if  not hasattr(self.model, 'forward_type') or self.model.forward_type == 'pure' or self.model.opt.ptq:
            pred_rgb = outputs['image']
            loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [N, 3] --> [N]
            loss = loss.mean()
        else:
            loss = 0
            loss_extra = torch.tensor(0, device=self.device)
            loss_pernalty = torch.tensor(0, device=self.device)
            loss_distill = torch.tensor(0, device=self.device)
            loss_adv = torch.tensor(0, device=self.device)
            loss_cauchy = torch.tensor(0, device=self.device)
            pred_rgb = outputs['image'][-1]
            
            ray_num = pred_rgb.shape[0]
            Tar = torch.tensor([1]*ray_num, device=self.device)
            
            gamma_0, gamma_1, gamma_2 = 5e-5, 5e-2, 1e-6
            alpha, beta = 1e-6, 1e-8
            
            loss_exp_tau = torch.exp(self.model.learned_tau) * alpha
            loss_final_time = self.criterion(outputs['image'][-1], gt_rgb).mean(-1).mean() # (N, 3) -> (N)
            loss_cauchy = self.model.sigma_cauchy_loss * gamma_2

            if self.opt.D:
                if self.model.Time_step == self.model.Time_step:
                    loss_clip_time = self.weight_sum(outputs['image'], self.model.tau_time, gt_rgb).mean(-1).mean()   # (N, 3) -> (N)
                    loss_pernalty = loss_final_time.data / loss_clip_time.data * torch.exp(self.model.tau_time) * beta#1e-6# 5e-7  * 1e-7
                    loss_extra = (outputs['extra_out'][0] + outputs['extra_out'][1] / torch.norm(outputs['image'][-1], p = 2)).mean() * alpha + loss_exp_tau
                    loss_consis = torch.norm(outputs['image'][1] - gt_rgb, p = 2) * gamma_0 + torch.cosine_embedding_loss(outputs['image'][1], gt_rgb, Tar).mean() * gamma_1

                    loss_adv = loss_extra + loss_pernalty
                    loss_render = loss_clip_time + loss_cauchy + loss_consis
                    loss_distill = self.criterion(outputs['image'][-1], outputs['image'][1]).mean()

                else:
                    loss_clip_time = loss_final_time
                    loss_consis = torch.norm(outputs['image'][-1] - gt_rgb, p = 2) * gamma_0 + torch.cosine_embedding_loss(outputs['image'][-1], gt_rgb, Tar).mean() * gamma_1
                    loss_render = loss_clip_time + loss_cauchy + loss_consis
                    loss_distill = torch.tensor(0, device = self.device)
                    loss_adv = torch.tensor(0, device = self.device)
                    
                # 所有loss的总和
                loss = loss_adv + loss_render + loss_distill
                
            else:
                loss_consis = torch.norm(outputs['image'][-1] - gt_rgb, p = 2) * gamma_0 + torch.cosine_embedding_loss(outputs['image'][-1], gt_rgb, Tar).mean() * gamma_1
                loss = loss_final_time.mean(-1) + loss_cauchy + loss_consis
                
            loss = loss.mean()
            
        # extra loss
        if 'proposal_loss' in outputs and self.opt.lambda_proposal > 0:
            loss = loss + self.opt.lambda_proposal * outputs['proposal_loss']

        if 'distort_loss' in outputs and self.opt.lambda_distort > 0:
            loss = loss + self.opt.lambda_distort * outputs['distort_loss']

        if self.opt.lambda_entropy > 0:
            w = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)
            entropy = - w * torch.log2(w) - (1 - w) * torch.log2(1 - w)
            loss = loss + self.opt.lambda_entropy * (entropy.mean())

        # adaptive num_rays
        if self.opt.adaptive_num_rays:
            outputs['num_points'] = max(outputs['num_points'], 1)
            self.opt.num_rays = int(round((self.opt.num_points / outputs['num_points']) * self.opt.num_rays))
            
        return pred_rgb, gt_rgb, loss#, loss_item

    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # the new inplace TV loss
        if self.opt.lambda_tv > 0:
            self.model.apply_total_variation(self.opt.lambda_tv)
        
        if self.opt.lambda_wd > 0:
            self.model.apply_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data):
        images = data['images'] # [H, W, 3/4]

        if len(images.shape) == 2:  # on training set
            self.train_step(data)

        else:
            rays_o = data['rays_o']  # [N, 3]
            rays_d = data['rays_d']  # [N, 3]
            index = data['index']  # [1/N]

            H, W, C = images.shape

            cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None # [1/N, 2] or None

            # eval with fixed white background color
            bg_color = 1
            if C == 4:
                gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
            else:
                gt_rgb = images

            outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=False, cam_near_far=cam_near_far)
            pred_rgb = outputs['image'].reshape(H, W, 3)
            pred_depth = outputs['depth'].reshape(H, W)

            loss = self.criterion(pred_rgb, gt_rgb).mean()
            # self.model.reset_state()
            return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False, shading='full'):  

        rays_o = data['rays_o'] # [N, 3]
        rays_d = data['rays_d'] # [N, 3]
        index = data['index'] # [1/N]
        H, W = data['H'], data['W']

        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None # [1/N, 2] or None

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=perturb, cam_near_far=cam_near_far, shading=shading)
        for key in self.record.keys():
            self.record[key] += self.model.record[key]

        self.model.record = {'num':0, 'sigma_1':0,'color_1':0, 'color_3':0}
        pred_rgb = outputs['image'].reshape(H, W, 3)
        pred_depth = outputs['depth'].reshape(H, W)
        # self.model.reset_state()
        return pred_rgb, pred_depth

    def save_mesh(self, save_path=None, resolution=128, decimate_target=1e5, dataset=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, resolution=resolution, decimate_target=decimate_target, dataset=dataset)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.opt.mark_untrained:
            self.model.mark_untrained_grid(train_loader._data)
        start_t = time.time()
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            if hasattr(self.model, 'forward_type'): 
                if self.model.Time_step < self.model.Init_Time_step:
                    self.model.Time_step = int(min(self.model.Time_step + (epoch-1) * 6, self.model.Init_Time_step))
                    self.model.Time_step = max(self.model.Time_step, 1)
                self.train_one_epoch(train_loader)
                
                if self.opt.D:
                    self.log(f'Epoch {epoch} finished, the time step is {self.model.init_w.data : .4f}, total time step is {self.model.Time_step} \
                            the tau is {1 / torch.sigmoid(self.model.init_b) : .4f}')
                else:
                    self.log(f'Epoch {epoch} finished, the time step is {self.model.Time_step}')
                
            else:
                self.train_one_epoch(train_loader)

            if (self.epoch % self.save_interval == 0 or self.epoch == max_epochs) and self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                # self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)
            
        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.6f} minutes.")
        # self.model.Time_step = self.model.tau_time.round().int().item()
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        ave_loss = self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX
        return ave_loss

    def test(self, loader, save_path=None, name=None, write_video=True, offsets = None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
        self.model.record =  {'num':0, 'sigma_1':0,'color_1':0, 'color_3':0}
        if offsets is not None:
            print(self.model.encoder.embeddings[offsets:offsets+4, :])
            assert torch.allclose(self.model.encoder.embeddings[offsets:offsets+524288, :], torch.zeros(524288, 2, device=self.device))
        with torch.no_grad():
            for i, data in enumerate(loader):
                
                preds, preds_depth = self.test_step(data)
                pred = preds.detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth.detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)
                
                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
            
            for key in self.record.keys():
                    self.record[key] /= len(loader)            
    
            self.log(self.record)
        if write_video:
            all_preds = np.stack(all_preds, axis=0) # [N, H, W, 3]
            all_preds_depth = np.stack(all_preds_depth, axis=0) # [N, H, W]

            # fix ffmpeg not divisible by 2
            all_preds = np.pad(all_preds, ((0, 0), (0, 1 if all_preds.shape[1] % 2 != 0 else 0), (0, 1 if all_preds.shape[2] % 2 != 0 else 0), (0, 0)))
            all_preds_depth = np.pad(all_preds_depth, ((0, 0), (0, 1 if all_preds_depth.shape[1] % 2 != 0 else 0), (0, 1 if all_preds_depth.shape[2] % 2 != 0 else 0)))

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=24, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=24, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
    
    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0 and self.opt.mark_untrained:
            self.model.mark_untrained_grid(train_loader._data)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                self.model.update_extra_state()            
            
            self.global_step += 1

            self.optimizer.zero_grad()

            preds, truths, loss_net = self.train_step(data)
            
            loss = loss_net
         
            self.scaler.scale(loss).backward()

            self.post_train_step() # for TV loss...
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss_net.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, mvp, W, H, bg_color=None, spp=1, downscale=1, shading='full'):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'mvp': mvp,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
            'index': [0],
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            # here spp is used as perturb random seed! (but not perturb the first sample)
            preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp, shading=shading)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.unsqueeze(0).permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).squeeze(0).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(0).unsqueeze(1), size=(H, W), mode='nearest').squeeze(0).squeeze(1)

        pred = preds.detach().cpu().numpy()
        pred_depth = preds_depth.detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader):

        # self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                self.model.update_extra_state()
                
            self.optimizer.zero_grad()    

            # if self.opt.D:
            #     self.model.tau_time = torch.clamp(self.model.init_w, min=1, max=self.model.Time_step)      

            preds, truths, loss_net = self.train_step(data)
            # break
            loss = loss_net
         
            self.scaler.scale(loss).backward()
            self.post_train_step() # for TV loss...

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss_net.item()
            total_loss += loss_val
            
            # self.model.reset_state()
            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})")
                    
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}, loss={average_loss:.6f}.")
        # self.log(f"==>Loss item: loss: {loss_item[0]:.6f}, loss_clip: {loss_item[1]:.6f}, loss_extra: {loss_item[2]:.8f}, loss_credit: {loss_item[3]:.8f}, loss_distill: {loss_item[4]:.6f}")

        return average_loss

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")
        
        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()
        # Quantize the embedding
        offsets = self.model.encoder.offsets
        if self.model.encoder.quant:
            for i in range(1, self.model.encoder.offsets.shape[0]):
                self.quantizer_list[i-1].training = False
                self.quantizer_list[i-1].soft_targets = False
                self.model.encoder.quant_embeddings[offsets[i-1] : offsets[i]] = self.quantizer_list[i-1](self.model.encoder.embeddings[offsets[i-1] : offsets[i]]).detach()
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()
            
        if self.local_rank == 0:
            
            # pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            pbar = tqdm.tqdm(total=len(loader) * 1, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        with torch.no_grad():
            self.local_step = 0
            
            for data in loader:    
                self.local_step += 1
                preds, preds_depth, truths, loss = self.eval_step(data)
                
                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    
                    metric_vals = []
                    for metric in self.metrics:
                        metric_val = metric.update(preds, truths)
                        metric_vals.append(metric_val)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')
                    save_path_error = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_error_{metric_vals[0]:.2f}.png') # metric_vals[0] should be the PSNR

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds.detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth.detach().cpu().numpy()
                    pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    truth = truths.detach().cpu().numpy()
                    truth = (truth * 255).astype(np.uint8)
                    error = np.abs(truth.astype(np.float32) - pred.astype(np.float32)).mean(-1).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)
                    cv2.imwrite(save_path_error, error)

                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

        return average_loss

    def evaluate_on_trainset(self, loader):
        # get bitwidth learning reference loss_fp
        total_loss = 0
        self.model.train()
        for data in loader:
            preds, truths, loss_net = self.train_step(data)
            loss_val = loss_net.item()
            total_loss += loss_val
        self.model.eval()
        return total_loss / len(loader)

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    # if 'density_grid' in state['model']:
                    #     del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):

        if checkpoint is None: # load latest
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth')) 

            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, abort loading latest model.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        
        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
    
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

    def tit_load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None: # load latest
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth')) 

            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, abort loading latest model.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        checkpoint_dict['ema']['shadow_params'].insert(0, self.model.init_w.detach().clone())
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        if self.model.density_grid.shape != checkpoint_dict['model']['density_grid'].shape:
            self.model.density_grid = torch.zeros_like(checkpoint_dict['model']['density_grid'])
            
        if self.model.density_bitfield.shape != checkpoint_dict['model']['density_bitfield'].shape:
            self.model.density_bitfield = torch.zeros_like(checkpoint_dict['model']['density_bitfield'])
            
        if self.model.encoder.embeddings.shape != checkpoint_dict['model']['encoder.embeddings'].shape:
            self.model.encoder.embeddings = nn.Parameter(torch.zeros_like(checkpoint_dict['model']['encoder.embeddings']))

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.opt.D:
            if self.ema_decay is not None:
                self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_decay)
            else:
                self.ema = None

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
    
        if model_only:
            return
    
    def weight_sum(self, images, t, gt_rgb):
        ct = torch.ceil(t)
        rt = torch.round(t)
        if ct == rt:
            return self.criterion(images[1], gt_rgb) * (1 + t - rt) + self.criterion(images[0], gt_rgb) * (rt - t)
        else:
            return self.criterion(images[1], gt_rgb) * (ct - t) + self.criterion(images[2], gt_rgb) * (t - rt)
    
    def ste_round(self, x):
        return torch.round(x) - x.detach() + x    
    
    ################################### Quantization ######################################
    
    def enable_calibrate(self, module):
        for name, child in module.named_children():
            if isinstance(child,Quantizer):
                child.ptq = True
            else:
                self.enable_calibrate(child)
        return module
    
    def disable_calibrate(self, module):
        for name, child in module.named_children():
            if isinstance(child, Quantizer):  #AdaRoundQuantizer((observer): MinMaxObserver())  #AsymmetricQuantizer((observer): EMAMinMaxObserver())
                child.ptq = False     #别的都pass
            else:
                self.disable_calibrate(child)
        return module

    def fwd_process(self, data):
        rays_o = data['rays_o'] # [N, 3]
        rays_d = data['rays_d'] # [N, 3]
        index = data['index'] # [1/N]
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None # [1/N, 2] or None

        images = data['images'] # [N, 3/4]

        N, C = images.shape

        if self.opt.background == 'random':
            bg_color = torch.rand(N, 3, device=self.device) # [N, 3], pixel-wise random.
        else:
            bg_color = 1
        
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        #
        shading = 'diffuse' if self.global_step < self.opt.diffuse_step else 'full'
        update_proposal = self.global_step <= 3000 or self.global_step % 5 == 0
        
        # outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=True,
        #                             cam_near_far=cam_near_far, shading='full')
        outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=True,
                            cam_near_far=cam_near_far, shading=shading, update_proposal=update_proposal)
        return outputs, gt_rgb
    
    def calibrate(self, train_loader, max_epochs):
        self.log(f"==> Start Calibrating the model ...")
        self.model.train()
        
        # Reduce the size of input embedding to save memory, only for colmap data.
        if self.opt.data_format == 'colmap':
            print('Reduce the size of input embedding to save memory.')
            self.model.opt.ptq = False
            enable_origin(self.model)
            
            # Training the model to reduce the number of samples.
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr * 1e-5, eps=1e-15)
            self.global_step = 0
            for data in train_loader:
                if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                    self.model.update_extra_state()
                optimizer.zero_grad()    
                preds, truths, loss = self.train_step(data)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

            self.model.opt.ptq = True
            disable_origin(self.model)
        
        # for name, para in self.model.named_parameters():
        #     para.requires_grad = False
        self.enable_calibrate(self.model)
        max_data_num = len(train_loader)

        count = 0
        starts = time.time()

        with torch.no_grad():
            for data in train_loader:
                
                count += 1
                if count > max_data_num:
                    break
                out = self.fwd_process(data)
                del out
                del data
                self.model.clear_embedding()
        ends = time.time()
        self.disable_calibrate(self.model)
        self.log(f"==> Calibrated the model, cost {ends - starts}s")
        
        list_module = get_ordered_list_of_modules(self.model, t = self.model.Time_step)
        act_dict = get_spike_layer(self.model)
        # state = self.model.state_dict()
        
        for name, module in list_module:
            loss_w = [0, 0]# Weight [floor, ceil]
            loss_v = [0, 0]# Membrane
            loss_im = [0, 0]# Input of Module
            loss_is = [0, 0]# Input of Spike Layer
            
            module_w = module.weight.clone().detach()
            scale_w = [close_to_2(module.weight_quantizer.scale, 'floor')[0], close_to_2(module.weight_quantizer.scale, 'ceil')[0]]
            loss_w[0] = torch.norm(module_w - self.fake_quant(module_w, scale_w[0], self.opt.bit), p = 2) ** 2
            loss_w[1] = torch.norm(module_w - self.fake_quant(module_w, scale_w[1], self.opt.bit), p = 2) ** 2
            
            for data in train_loader:
                enable_origin(self.model)      
                with torch.no_grad():
                    
                    origin_input, origin_out, _ = self.prepare_inout(self.model, module, data, collect_input= True, collect_output=True)
                    if module.first:
                        scale_im = [close_to_2(module.input_quantizer.scale, 'floor')[0], close_to_2(module.input_quantizer.scale, 'ceil')[0]]
                        loss_im[0] += torch.norm(origin_input - self.fake_quant(origin_input, scale_im[0], self.opt.bit), p = 2) ** 2
                        loss_im[1] += torch.norm(origin_input - self.fake_quant(origin_input, scale_im[1], self.opt.bit), p = 2) ** 2
                    
                    scale_is = [close_to_2(act_dict[name].input_quantizer.scale, 'floor')[0], close_to_2(act_dict[name].input_quantizer.scale, 'ceil')[0]]
                    loss_is[0] += torch.norm(origin_out - self.fake_quant(origin_out, scale_is[0], self.opt.bit), p = 2) ** 2
                    loss_is[1] += torch.norm(origin_out - self.fake_quant(origin_out, scale_is[1], self.opt.bit), p = 2) ** 2
                    if isinstance(act_dict[name], QPlif):
                        scale_v = [close_to_2(act_dict[name].Mem_quantizer.scale, 'floor')[0], close_to_2(act_dict[name].Mem_quantizer.scale, 'ceil')[0]]
                        for i in range(self.model.Time_step):
                            if i == 0:
                                self.model.reassignment(act_dict[name], decay_input = False, item = 'decay_input')
                                act_dict[name](origin_out[0, i, : ,:])
                                self.model.reassignment(act_dict[name], decay_input = True, item = 'decay_input')
                                loss_v[0] += torch.norm(act_dict[name].v_o - self.fake_quant(act_dict[name].v_o, scale_v[0], self.opt.bit), p = 2) ** 2
                                loss_v[1] += torch.norm(act_dict[name].v_o - self.fake_quant(act_dict[name].v_o, scale_v[1], self.opt.bit), p = 2) ** 2
                            else:
                                act_dict[name](origin_out[0, i, :, :])
                                loss_v[0] += torch.norm(act_dict[name].v_o - self.fake_quant(act_dict[name].v_o, scale_v[0], self.opt.bit), p = 2) ** 2
                                loss_v[1] += torch.norm(act_dict[name].v_o - self.fake_quant(act_dict[name].v_o, scale_v[1], self.opt.bit), p = 2) ** 2
                        
            weight_scale_exp = torch.log2(scale_w[torch.argmin(torch.tensor(loss_w))])
            lif_is = torch.log2(scale_is[torch.argmin(torch.tensor(loss_is))])
            if isinstance(act_dict[name], QPlif):
                lif_v = torch.log2(scale_v[torch.argmin(torch.tensor(loss_v))])
            target_module = self.model
            parts= name.split('.')
            module_path = parts
            
            module_path.append('weight_quantizer')
            for part in module_path:
                if part.isdigit():
                    target_module = target_module[int(part)]
                else:
                    target_module = getattr(target_module, part)
            target_module.register_parameter('scale_exp', nn.Parameter(weight_scale_exp))
            self.log(f'[INFO] {name}.weight_quantizer.scale_exp is calibrated to {weight_scale_exp}')
            
            if module.first:
                module_path[-1] = 'input_quantizer'
                input_scale_exp = torch.log2(scale_w[torch.argmin(torch.tensor(loss_im))])
                target_module = self.model
                for part in module_path:
                    if part.isdigit():
                        target_module = target_module[int(part)]
                    else:
                        target_module = getattr(target_module, part)
                target_module.register_parameter('scale_exp', nn.Parameter(input_scale_exp))
                self.log(f'[INFO] {name}.input_quantizer.scale_exp is calibrated to {input_scale_exp}')
            
            lif_path = parts[:-1]
            if isinstance(act_dict[name], QPlif):
                lif_path[-1] = str(int(lif_path[-1]) + 1)
                
            elif isinstance(act_dict[name], Qexp):
                lif_path = ['quan_exp']
                
            elif isinstance(act_dict[name], Qsigmoid):
                lif_path = ['quan_sigmoid']
                
            lif_name = '.'.join(lif_path)
            lif_path.append('input_quantizer')
            target_module = self.model
            for part in lif_path:
                if part.isdigit():
                    target_module = target_module[int(part)]
                else:
                    target_module = getattr(target_module, part)
            target_module.register_parameter('scale_exp', nn.Parameter(lif_is))
            self.log(f'[INFO] {lif_name}.input_quantizer.scale_exp is calibrated to {lif_is}')
            if isinstance(act_dict[name], QPlif):
                lif_path[-1] = 'Mem_quantizer'
                target_module = self.model
                for part in lif_path:
                    if part.isdigit():
                        target_module = target_module[int(part)]
                    else:
                        target_module = getattr(target_module, part)
                target_module.register_parameter('scale_exp', nn.Parameter(lif_v))
            
                self.log(f'[INFO] {lif_name}.Mem_quantizer.scale_exp is calibrated to {lif_v}')

    def quan_train(self, train_loader):
        if self.opt.mark_untrained:
            self.model.mark_untrained_grid(train_loader._data)
        list_module = get_ordered_list_of_modules(self.model, t = self.model.Time_step)
        self.calibrate(train_loader, 1)
                
        start_t = time.time()
        self.quan_train_one_epoch(train_loader, list_module)

        end_t = time.time()
        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.6f} minutes.")
        self.log(f'[INFO] Ready to evaluate, time step is {self.model.Time_step}')
    
    def prepare_fwd_eval(self, data):
        images = data['images']
        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]

        N, C = images.shape

        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None # [1/N, 2] or None

        if self.opt.background == 'random':
            bg_color = torch.rand(N, 3, device=self.device) # [N, 3], pixel-wise random.
        else: # white / last_sample
            bg_color = 1
        outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=False, cam_near_far=cam_near_far)
        return outputs
    
    def prepare_inout(self, model, module, model_input, collect_input=False, collect_output = False):
        def _hook_to_collect_inp_out_data(_, inp, out):
            """
            hook to collect input and output data
            """
            if collect_input:
                inp_data_list.append(inp[0])

            if collect_output:
                out_data_list.append(out)

        inp_data_list = []
        out_data_list = []

        handle = module.register_forward_hook(_hook_to_collect_inp_out_data)

        with torch.no_grad():
        # if model.eval, the dirs will be all nan.
            model.train()
            outputs, _ = self.fwd_process(model_input)

        handle.remove()
        inp_data, out_data = None, None

        if inp_data_list and isinstance(inp_data_list[0], torch.Tensor):
            # print('length of inp list: ', len(inp_data_list)) # 4
            # 多个time step
            if len(inp_data_list) > 1:
                for i in range(len(inp_data_list) // self.model.Time_step):
                    inp_data_list[i] = torch.stack(inp_data_list[i * self.model.Time_step: (i + 1) * self.model.Time_step])
                del inp_data_list[len(inp_data_list) // self.model.Time_step : ]
                inp_data = torch.stack(inp_data_list).detach()
            else:
                inp_data = inp_data_list[0].detach()
                
        if out_data_list and isinstance(out_data_list[0], torch.Tensor):
            if len(out_data_list) > 1:
                for i in range(len(out_data_list) // self.model.Time_step):
                    out_data_list[i] = torch.stack(out_data_list[i * self.model.Time_step: (i + 1) * self.model.Time_step])
                del out_data_list[len(out_data_list) // self.model.Time_step : ]
                out_data = torch.stack(out_data_list).detach()
            else:
                out_data = out_data_list[0].detach()
        # print(f'[INFO] inp_data shape: {inp_data.shape if inp_data is not None else None}, out_data shape: {out_data.shape if out_data is not None else None}')
        # out shape: [1, 4, 67854, 32] or [1, 4, 67854, 64]
        return inp_data, out_data, outputs

    def prepare_data_for_layer(self, inputx, model, module):
        origin = [] # output, original
        quant = [] # input, quant
        enable_origin(model)       
        _, origin_out, pred_rgb_origin = self.prepare_inout(model, module, inputx, collect_output = True) # get Wx
        # print(f'Memory Usage after origin out: {torch.cuda.memory_allocated() / 1024 ** 3} GB')
        origin.append(origin_out)
        disable_origin(model)
        quant_in, _, _ = self.prepare_inout(model, module, inputx, collect_input = True) # get \hat W * \hat x
        quant.append(quant_in)
        return quant, origin, pred_rgb_origin#, origin_input
    
    def quan_embedding(self):
        bit = self.opt.bit
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model.encoder.embeddings.requires_grad = False
        embeddings = self.model.encoder.embeddings
        self.model.encoder.quant_embeddings = torch.zeros_like(self.model.encoder.embeddings)
        offsets = self.model.encoder.offsets
        
        print(f'[INFO] Init Quant Embeddings')
        for i in range(1, self.model.encoder.offsets.shape[0]):
            tmp_embedding = embeddings[offsets[i-1] : offsets[i]]
            maxs = torch.max(tmp_embedding)
            mins = torch.min(tmp_embedding)
            scale = torch.min(torch.abs(maxs), torch.abs(mins)) / (2 ** bit - 1)
            pot_scale, _ = close_to_2(scale, 'round')
            quant_tmp = torch.clamp(torch.round(tmp_embedding / pot_scale), - 2 ** (bit - 1), 2 ** (bit - 1) -1)
            fake_tmp = quant_tmp * pot_scale
            self.model.encoder.quant_embeddings[offsets[i-1] : offsets[i]] = fake_tmp
        print(f'[INFO] Init Quant Embeddings over')
    
    def quan_train_embedding(self, dataloader):
        bit = 4
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        self.model.encoder.embeddings.requires_grad = False
        embeddings = self.model.encoder.embeddings
        offsets = self.model.encoder.offsets
        print(f'[INFO] Init Quant Embeddings')
        self.quantizer_list = []
        params_list = []
        level_num_list = []
        temp_decay = LinearTempDecay(self.opt.adaround_iter // len(dataloader), rel_start_decay=0.2,start_b=20, end_b=2)
        for i in range(1, self.model.encoder.offsets.shape[0]):
            quantizer = AdaRoundQuantizer(bit = bit, observer = MinMaxObserver(), ptq = self.opt.ptq)
            quantizer.ptq = True
            quantizer.training =True
            self.quantizer_list.append(quantizer)
            tmp_embedding = embeddings[offsets[i-1] : offsets[i]]
            fake_quant = quantizer(tmp_embedding)   # float scale
            pot_scale, scale_exp = close_to_2(quantizer.scale, 'round')
            quantizer.register_parameter('scale_exp', nn.Parameter(scale_exp, requires_grad=True))
            quantizer.ptq = False
            params_list.extend([quantizer.scale_exp, quantizer.alpha])
            level_num_list.append(offsets[i] - offsets[i-1])
            
        print(f'[INFO] Init Quant Embeddings over')
        optimizer = torch.optim.Adam(params_list, lr = 1e-3)
        self.model.opt.ptq = False
        self.model.opt.adaptive_num_rays = False # disable adaptive num rays to save memory
        for iter in range(self.opt.adaround_iter  // len(dataloader)):
            b = temp_decay(iter)
            self.log(f"\n ==> For Encoder's Embeddings, Start Training Epoch {iter} ...")
            begin = time.time()
            mse_avg = 0
            recon_avg = 0
            round_avg = 0
            for data in dataloader:

                recon_loss = 0
                round_loss = 0
                loss = 0
                
                optimizer.zero_grad() 
                
                # original embedding forward
                self.model.clear_embedding()
                with torch.no_grad():
                    self.model.encoder.quant = False
                    original_preds, _, _ = self.train_step(data)
                self.model.clear_embedding()
                
                # quant embedding forward
                self.model.encoder.quant = True
                self.model.encoder.quant_embeddings = torch.zeros_like(self.model.encoder.embeddings)
                for i in range(1, self.model.encoder.offsets.shape[0]):
                    self.model.encoder.quant_embeddings[offsets[i-1] : offsets[i]] = self.quantizer_list[i-1](self.model.encoder.embeddings[offsets[i-1] : offsets[i]])
                    
                quant_preds, truths, mse_loss = self.train_step(data)
                
                recon_loss = cal_recon_loss(original_preds, quant_preds)
                for ids in range(len(self.quantizer_list)):
                    round_loss += cal_round_loss(self.quantizer_list[ids].get_soft_targets(), b) / len(self.quantizer_list) / level_num_list[ids]
                
                loss = round_loss + recon_loss + mse_loss
                loss.backward()
                optimizer.step()
                
                mse_avg += mse_loss
                recon_avg += recon_loss
                round_avg += round_loss
                
            end = time.time() 
            self.log(f'Recon loss: {recon_avg / len(dataloader) : .4f}, Round loss: {round_avg / len(dataloader) : .4f}, MSE loss: {mse_avg / len(dataloader) : .6f}, cost {end - begin : .2f}s')
        optimizer.zero_grad()
        self.model.opt.ptq = True
        self.model.opt.adaptive_num_rays = True
        # PTQ模式下使用quant_embeddings进行推理以及训练
        for i in range(1, self.model.encoder.offsets.shape[0]):
            self.quantizer_list[i-1].scale_exp.requires_grad = False
            self.quantizer_list[i-1].alpha.requires_grad = False
            self.model.encoder.quant_embeddings[offsets[i-1] : offsets[i]] = self.quantizer_list[i-1](self.model.encoder.embeddings[offsets[i-1] : offsets[i]]).detach()
                
    def quan_train_one_epoch(self, loader, list_module):
        
        self.model.train()
        self.model.encoder.embeddings.requires_grad = False
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.quan_embedding()
        act_dict = get_spike_layer(self.model)
        module_start = time.time()
        self.quaned_module = {}
        state = {}
        name_list = []
        y = list(map(lambda x: name_list.append(x[0]), list_module))
        for sub_module in self.dicts['quaned_module_list'].keys():
            if sub_module in name_list and self.dicts['quaned_module_list'][sub_module] < self.opt.adaround_iter // len(loader):
                list_module.pop(name_list.index(sub_module))
                name_list.pop(name_list.index(sub_module))
                
        # save_path = Path(f'/home/liuwh/lrx/PATA_code/workspace/intermediate_data/W{self.opt.bit}U{self.opt.Mem_bit}_without_Clamp')       
        # if not os.path.exists(save_path / 'LIF'):
        #     os.makedirs(save_path / 'LIF')
        # if not os.path.exists(save_path / 'QLIF'):
        #     os.makedirs(save_path / 'QLIF')
            
        id_module = 1
        torch.cuda.empty_cache()
        
        for name, module in list_module:

            params_list = []
            name_list = []
            for names, param in act_dict[name].named_parameters():
                if 'scale_exp' in names or 'mem_threshold' in names or 'vth_alpha' in names:
                    param.requires_grad = True
                    params_list.append(param)
                    name_list.append(names)
            
            for names, param in module.named_parameters():
                if 'scale_exp' in names or 'alpha' in names:
                    param.requires_grad = True
                    params_list.append(param)
                    name_list.append(names)

            self.log('[INFO] Parameters List: ', name_list)
            reconlist = []
            max_count = len(loader)
            temp_decay = LinearTempDecay(self.opt.adaround_iter *0.5 // max_count, rel_start_decay=0.2,start_b=20, end_b=2)
            optimizer = torch.optim.Adam(params_list, lr = 1e-3)

            for iter in range(self.opt.adaround_iter // max_count):
                
                self.log(f"\n ==> For Module {name}, Start Training Epoch {iter} ...")
                recon_loss_iter = []
                iter_begin = time.time()
                self.model.clear_embedding()
                count = 0
                original_out = 0
                quan_out = 0
                b = temp_decay(iter)
                
                for data in loader: 
                    
                    count += 1
                    recon_loss = 0
                    if count >= max_count:
                        break
                    
                    optimizer.zero_grad()  
                    with torch.no_grad():
                        pre_inputdata, pre_outputdata, pred_rgb_origin = self.prepare_data_for_layer(data, self.model, module)
                    
                    original_mem = []
                    quan_mem = []
                    
                    if isinstance(act_dict[name], QPlif):
                        act_dict[name].origin = True
                        origin_actout = []
                        original_mem = []
                        for i in range(self.model.Time_step):
                            if i == 0:
                                self.model.reassignment(act_dict[name], decay_input = False, item = 'decay_input')
                                origin_actout.append(act_dict[name](pre_outputdata[0][0, i, : ,:]))
                                self.model.reassignment(act_dict[name], decay_input = True, item = 'decay_input')
                                original_mem.append(act_dict[name].v_o)
                            else:
                                origin_actout.append(act_dict[name](pre_outputdata[0][0, i, :, :]))
                                original_mem.append(act_dict[name].v_o)

                        act_dict[name].reset()
                        act_dict[name].origin = False

                    else:
                        origin_actout = []
                        
                        if len(pre_outputdata[0].shape) == 2:
                            
                            if 'sigma' in name and not isinstance(act_dict[name], nn.ReLU):
                                act_dict[name].origin = True
                                origin_actout.append(torch.cat((act_dict[name](pre_outputdata[0])[:, 0].reshape(-1, 1), pre_outputdata[0][:, 1:]), dim = 1))
                                act_dict[name].origin = False
                            else:
                                origin_actout.append(act_dict[name](pre_outputdata[0]))
                        else:
                            if 'sigma' in name and not isinstance(act_dict[name], nn.ReLU):
                                act_dict[name].origin = True
                                for i in range(self.model.Time_step):
                                    origin_actout.append(torch.cat((act_dict[name](pre_outputdata[0][0, i, :, 0].reshape(-1, 1)), pre_outputdata[0][0, i, :, 1:]), dim = 1))# * time_weight[i]
                                act_dict[name].origin = False
                            else:
                                act_dict[name].origin = True
                                for i in range(self.model.Time_step):
                                    origin_actout.append(act_dict[name](pre_outputdata[0][0, i, :, :]) )#* time_weight[i]
                                act_dict[name].origin = False
                    
                    # PSQ
                    if iter <= int(0.5 * self.opt.adaround_iter // max_count ):
                        act_dict[name].origin = True
                    else:
                        for names, param in module.named_parameters():
                            param.requires_grad = False
                    
                    quan_output = module(pre_inputdata[0])
                    
                    if isinstance(act_dict[name], QPlif):
                        quan_actout = []
                        # quan_actinput = []
                        quan_mem = []
                        for i in range(self.model.Time_step):
                            if i == 0:
                                self.model.reassignment(act_dict[name], decay_input = False, item = 'decay_input')
                                quan_actout.append(act_dict[name](quan_output[0, i, :, :]))
                                self.model.reassignment(act_dict[name], decay_input = True, item = 'decay_input')
                                quan_mem.append(act_dict[name].v_q)
                            else:
                                quan_actout.append(act_dict[name](quan_output[0, i, :, :]))
                                quan_mem.append(act_dict[name].v_q)
                        act_dict[name].reset()
                        origin_actout = torch.stack(origin_actout).detach()
                        quan_actout = torch.stack(quan_actout)
                        
                        # if iter == self.opt.adaround_iter // max_count - 1 and count == max_count-1:
                        #     print(save_path / 'QLIF' / f'{name.replace(".", "_")}_quant_Mem_t{i}.pt')
                        #     torch.save(quan_mem, save_path / 'QLIF' / f'{name.replace(".", "_")}_quant_Mem.pt')
                        #     torch.save(original_mem, save_path / 'LIF' / f'{name.replace(".", "_")}_original_Mem.pt')
                        #     torch.save([act_dict[name].Mem_quantizer.scale, act_dict[name].vth_alpha], save_path / 'QLIF' / f'{name.replace(".", "_")}_Mem_scale_vth.pt')
                        
                    else:
                        quan_actout = []
                        if len(pre_outputdata[0].shape) == 2:
                            if 'sigma' in name and not isinstance(act_dict[name], nn.ReLU):
                                quan_actout.append(torch.cat((act_dict[name](quan_output[:, 0]).reshape(-1, 1), quan_output[:, 1:]), dim = 1))
                            else:
                                quan_actout.append(act_dict[name](quan_output))
                            origin_actout = origin_actout[0].detach()
                            quan_actout = quan_actout[0]
                        else:
                            if 'sigma' in name and not isinstance(act_dict[name], nn.ReLU):
                                for i in range(self.model.Time_step):
                                    quan_actout.append(torch.cat((act_dict[name](quan_output[0, i, :, 0].reshape(-1, 1)), quan_output[0, i, :, 1:]), dim = 1))# * time_weight[i]
                            else:
                                for i in range(self.model.Time_step):
                                    quan_actout.append(act_dict[name](quan_output[0, i, :, :]) )#* time_weight[i]
                                    
                            origin_actout = torch.stack(origin_actout).detach()
                            quan_actout = torch.stack(quan_actout)
                    
                    if len(pre_outputdata[0].shape) == 2:
                        coeff = torch.rand_like(quan_actout)
                        recon_loss = cal_recon_loss(origin_actout, quan_actout)
                    else:
                        for i in range(self.model.Time_step):
                                recon_loss += cal_recon_loss(origin_actout[i], quan_actout[i])
                                
                    self.model.clear_embedding()
                    
                    mse_loss = 0
                    # PD-loss
                    enable_origin(self.model)
                    
                    for i in range(id_module):
                        list_module[i][1].origin = False
                        act_dict[list_module[i][0]].origin=False
                    
                    _, _, mse_loss = self.train_step(data)

                    recon_loss = recon_loss / self.model.Time_step
                    round_loss = cal_round_loss(module.weight_quantizer.get_soft_targets(), b) #* 1e-4
                    
                    total_loss = recon_loss + round_loss + mse_loss# + mem_loss
                    total_loss.backward()
                    optimizer.step()
                    recon_loss_iter.append(recon_loss.detach().data.item())
                    self.quaned_module[name] = iter
                    quan_out += quan_actout.mean().detach()
                    original_out += origin_actout.mean().detach()

                    disable_origin(self.model)
                recon_loss_iter = sum(recon_loss_iter)  / len(recon_loss_iter)
                reconlist.append(recon_loss_iter)
                iter_end = time.time()

                self.log(f'recon loss: {recon_loss_iter : .4f}, round loss: {round_loss : .4f}, Mse loss: {mse_loss * 100 : .4f}\
                         alpha: {module.weight_quantizer.alpha.shape}, Qoutput: {quan_out / count:.4f}, output: {original_out / count:.4f}, Time: {iter_end - iter_begin:.2f}s')
            id_module += 1
            self.log('==> Module {} over! time:{:.2f}s, recon_loss_init:{}, recon_loss_last:{}'.format(name, time.time()-module_start, reconlist[0], reconlist[-1]))

            for param in params_list:
                param.requires_grad = False
        
        for name, child in self.model.named_modules():
            if isinstance(child, AdaRoundQuantizer):
                child.soft_targets = False  
        state['quaned_module_list'] = self.quaned_module
        state['model'] = self.model.state_dict()
        torch.save(state, os.path.join(self.quan_model_dirs, f'quan.pth'))
    
    def quan_load_model(self, checkpoint=None, model_only=False):
        self.tit_load_checkpoint(checkpoint, model_only)
        self.new_model = inplace_net(self.model, ptq = self.opt.ptq, bit = self.opt.bit, symmetric=self.opt.symmetric, Mem_bit = self.opt.Mem_bit)
        self.model = self.new_model
        self.model.Time_step = int(torch.round(self.model.init_w.data).data)
        
        for name, param in self.new_model.named_parameters():
            if '.w' in name and 'weight' not in name or 'init_b' in name:
                old_w = param.data
                near_2, _ = close_to_2(torch.sigmoid(old_w), mode = 'round')
                
                if near_2 == 1.:
                    new_w = torch.tensor(10.0)
                else:
                    new_w = -torch.log(near_2 ** (-1) - 1)
                self.log(f'Change Para: {name}: {old_w : .4f} -> {new_w : .4f}, 2^N type: {near_2}')
                param.data.copy_(new_w)
                
                
    def load_alreay_quan_model(self, pth):
        self.dicts = torch.load(pth)
        
        self.new_model = inplace_net(self.model, ptq = self.opt.ptq, bit = self.opt.bit, symmetric=self.opt.symmetric, Mem_bit = self.opt.Mem_bit)
        self.model = self.new_model
        self.model.load_state_dict(self.dicts['model'], strict=False)
        self.model.Time_step = int(torch.round(self.model.init_w.data).data)
        # self.model.Time_step = 4
        
    def quant(self, x, scale, bit):
        return torch.clamp(torch.round(x /scale), -2 ** (bit - 1), 2 ** (bit - 1) - 1)
        
    def dequant(self,x, scale):
        return x * scale
    
    def fake_quant(self, x, scale, bit):
        return self.dequant(self.quant(x, scale, bit), scale)