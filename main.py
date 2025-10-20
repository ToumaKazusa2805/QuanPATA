from nerf.utils import Trainer, PSNRMeter, SSIMMeter, LPIPSMeter, seed_everything
from nerf.network import NeRFNetwork
from config.GetConfig import get_config
import torch
from nerf.gui import NeRFGUI
from nerf.provider import NeRFDataset
import numpy as np
from spike import spike_model
import yaml
from pathlib import Path
import os

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    seed_everything(42)
    opt = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.net == 'snn':
        model = spike_model.SMLP(opt).to(device)
    else:
        model = NeRFNetwork(opt).to(device)

    if not os.path.exists(opt.workspace):
        os.makedirs(opt.workspace)
    log_path = os.path.join(opt.workspace, f"log_ngp.txt")
    log_ptr = open(log_path, "a+")
    criterion = torch.nn.SmoothL1Loss(reduction='none')
   
    if opt.test:

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, use_checkpoint=opt.ckpt, world_size= 1)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            if not opt.test_no_video:
                test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

                if test_loader.has_gt:
                    trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)] # set up metrics
                    trainer.evaluate(test_loader) # blender has gt, so evaluate it.

                trainer.test(test_loader, write_video=True) # test and save video
        
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, eps=1e-15)

        if opt.data_format == 'colmap':
            from nerf.colmap_provider import ColmapDataset as NeRFDataset
        elif opt.data_format == 'dtu':
            from nerf.dtu_provider import NeRFDataset
        else: # nerf
            from nerf.provider import NeRFDataset
        
        train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()
        
        # iter = 30000, epoch = 
        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        save_interval = max(1, max_epoch // max(1, opt.save_cnt)) # save ~50 times during the training
        eval_interval = max(1, max_epoch // max(1, opt.eval_cnt))
        print(f'[INFO] max_epoch {max_epoch}, eval every {eval_interval}, save every {save_interval}.')

        # colmap can estimate a more compact AABB
        if not opt.contract and opt.data_format == 'colmap':
            model.update_aabb(train_loader._data.pts_aabb)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, 
                          criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, 
                          scheduler_update_every_step=True, use_checkpoint=opt.ckpt, eval_interval=eval_interval, save_interval=save_interval)

        valid_loader = NeRFDataset(opt, device=device, type='val').dataloader()
        # training
        trainer.metrics = [PSNRMeter(),]
        trainer.train(train_loader, valid_loader, max_epoch)

        if opt.D:
            trainer.model.Time_step = model.init_w.round().int().data.item()
            trainer.log(f'[INFO] Time step: {trainer.model.Time_step}')
        
        # last validation
        trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]
        trainer.evaluate(valid_loader)
        
        test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
        if test_loader.has_gt:
            trainer.evaluate(test_loader) # blender has gt, so evaluate it.
        
        trainer.test(test_loader, write_video=False) # test and save video