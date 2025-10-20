
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
from quanlib.adaround.Qact import Qexp, Qsigmoid
class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 num_layers=2, hidden_dim=64, geo_feat_dim=15,  # sigma net
                 num_layers_color=3, hidden_dim_color=64,  # color net
                 ):

        super().__init__(opt)

        # ======== position encoder ========
        self.encoder, self.in_dim = get_encoder(encoding="hashgrid", desired_resolution=2048 * self.bound)
        # ======== sigma network ========
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
                # sigma is activated by exp (see forward)
            else:
                out_dim = hidden_dim
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
                sigma_net.append(nn.ReLU())
        self.sigma_net = nn.ModuleList(sigma_net)

        # ======== direction encoder =========
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding="sh")

        # ======== color network ========
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
                color_net.append(nn.Linear(in_dim, out_dim, bias=False))
                # rgb is activated by sigmoid (see forward)
            else:
                out_dim = hidden_dim_color
                color_net.append(nn.Linear(in_dim, out_dim, bias=False))
                color_net.append(nn.ReLU())
        self.color_net = nn.ModuleList(color_net)

        # ======== quantization network ========
        self.qencoder = None
        self.qsigma_net = None
        self.qencoder_dir = None
        self.qcolor_net = None

        if opt.ptq:
            self.quan_exp = Qexp(bit=32, ptq=opt.ptq, symmetric=opt.symmetric)
            self.quan_sigmoid = Qsigmoid(bit=opt.bit, ptq=opt.ptq, symmetric=opt.symmetric)
            self.xyz_embedding = None
            self.dirs = None

        # proposal network  (??? Weihang Liu)
        # if not self.opt.cuda_ray:
        #     self.prop_encoders = nn.ModuleList()
        #     self.prop_mlp = nn.ModuleList()
        #
        #     # hard coded 2-layer prop network
        #     prop0_encoder, prop0_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=5, log2_hashmap_size=17, desired_resolution=128)
        #     prop0_mlp = MLP(prop0_in_dim, 1, 16, 2, bias=False)
        #     self.prop_encoders.append(prop0_encoder)
        #     self.prop_mlp.append(prop0_mlp)
        #
        #     prop1_encoder, prop1_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=5, log2_hashmap_size=17, desired_resolution=256)
        #     prop1_mlp = MLP(prop1_in_dim, 1, 16, 2, bias=False)
        #     self.prop_encoders.append(prop1_encoder)
        #     self.prop_mlp.append(prop1_mlp)

    # ===== full precision part =====
    def Origin_forward(self, x, d, **kwargs):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # print("TestShape:", x.shape, d.shape)
        # ======== position encoder ========
        x = self.encoder(x, bound=self.bound)
        # ======== sigma net ========
        h = x
        for l in range(len(self.sigma_net)):
            h = self.sigma_net[l](h)
        # exp activation for sigma
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # ======== direction encoder ========
        d = self.encoder_dir(d)
        # print('Dirs: ',  torch.max(d), torch.min(d))
        # ======== color net ========
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(len(self.color_net)):
            h = self.color_net[l](h)
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return {
            'sigma': sigma,
            'color': color,
        }

    def density(self, x, **kwargs):

        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(len(self.sigma_net)):
            h = self.sigma_net[l](h)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
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

    def clear_embedding(self):
        self.xyz_embedding = None
        self.dirs = None

    def quan_fwd(self, x, d, **kwargs):
        # x = self.encoder(x, bound=self.bound)
        
        if self.training:
            if self.xyz_embedding is not None:
                h = self.xyz_embedding
            else:
                self.xyz_embedding = self.encoder(x, bound=self.bound)
                h = self.xyz_embedding
                
            if self.dirs is not None:
                d = self.dirs
            else:
                self.dirs = self.encoder_dir(d)
                d = self.dirs
        else:
            h = self.encoder(x, bound=self.bound)
            d = self.encoder_dir(d)

        # ======== sigma net ========
        # h = x
        for l in range(len(self.sigma_net)):
            h = self.sigma_net[l](h)
        # exp activation for sigma
        sigma = self.quan_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # ======== direction encoder ========
        # d = self.encoder_dir(d)
        # print('Dirs: ',  torch.max(d), torch.min(d))
        # ======== color net ========
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(len(self.color_net)):
            h = self.color_net[l](h)
        # sigmoid activation for rgb
        color = self.quan_sigmoid(h)

        return {
            'sigma': sigma,
            'color': color,
        }
    
    def forward(self, x, d, **kwargs):
        if self.opt.ptq:
            return self.quan_fwd(x, d, **kwargs)
        else:
            return self.Origin_forward(x, d, **kwargs)