import argparse
from pathlib import Path

def get_config():
    parser = argparse.ArgumentParser()
    ### pre-defined configuration
    parser.add_argument('path',type=str)
    parser.add_argument('-O', action='store_true', help="recommended settings")
    parser.add_argument('-O2', action='store_true', help="recommended settings")

    ###
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    ### testing options
    parser.add_argument('--save_cnt', type=int, default=50, help="save checkpoints for $ times during training")
    parser.add_argument('--eval_cnt', type=int, default=1, help="perform validation for $ times during training")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--test_no_video', action='store_true', help="test mode: do not save video")
    parser.add_argument('--test_no_mesh', action='store_true', help="test mode: do not save mesh")
    parser.add_argument('--camera_traj', type=str, default='interp',
                        help="interp for interpolation, circle for circular camera")

    ### dataset options
    parser.add_argument('--data_format', type=str, default='colmap', choices=['nerf', 'colmap', 'dtu'])
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'all'])
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--random_image_batch', action='store_true',
                        help="randomly sample rays from all images per step in training")
    parser.add_argument('--downscale', type=int, default=1, help="downscale training images")
    parser.add_argument('--selfbound', action='store_true', help="self-defined bound")
    parser.add_argument('--bound', type=float, default=2,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=-1,
                        help="scale camera location into box[-bound, bound]^3, -1 means automatically determine based on camera poses..")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--enable_cam_near_far', action='store_true',
                        help="colmap mode: use the sparse points to estimate camera near far per view.")
    parser.add_argument('--enable_cam_center', action='store_true',
                        help="use camera center instead of sparse point center (colmap dataset only)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--T_thresh', type=float, default=1e-4, help="minimum transmittance to continue ray marching")

    ### GUI options
    parser.add_argument('--vis_pose', action='store_true', help="visualize the poses")
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1000, help="GUI width")
    parser.add_argument('--H', type=int, default=1000, help="GUI height")
    parser.add_argument('--radius', type=float, default=1, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    ### training options
    parser.add_argument('--log2_hashmap_size', type=int, default=19, help="max hash table size for each level")
    parser.add_argument('--iters', type=int, default=60000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, nargs='*', default=[256, 96, 48],
                        help="num steps sampled per ray for each proposal level (only valid when NOT using --cuda_ray)")
    parser.add_argument('--contract', action='store_true',
                        help="apply spatial contraction as in mip-nerf 360, only work for bound > 1, will override bound to 2.")
    parser.add_argument('--background', type=str, default='last_sample', choices=['white', 'random', 'last_sample'],
                        help="training background mode")

    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096 * 4,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--grid_size', type=int, default=128, help="density grid resolution")
    parser.add_argument('--mark_untrained', action='store_true', help="mark_untrained grid")
    parser.add_argument('--dt_gamma', type=float, default=1 / 256,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--diffuse_step', type=int, default=0,
                        help="training iters that only trains diffuse color for better initialization")

    # batch size related
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--adaptive_num_rays', action='store_true',
                        help="adaptive num rays for more efficient training")
    parser.add_argument('--num_points', type=int, default=2 ** 18,
                        help="target num points for each training step, only work with adaptive num_rays")

    # regularizations
    parser.add_argument('--lambda_entropy', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_wd', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_proposal', type=float, default=1, help="loss scale (only for non-cuda-ray mode)")
    parser.add_argument('--lambda_distort', type=float, default=0.002, help="loss scale (only for non-cuda-ray mode)")

    ### mesh options
    parser.add_argument('--mcubes_reso', type=int, default=512, help="resolution for marching cubes")
    parser.add_argument('--env_reso', type=int, default=256, help="max layers (resolution) for env mesh")
    parser.add_argument('--decimate_target', type=int, default=3e5,
                        help="decimate target for number of triangles, <=0 to disable")
    parser.add_argument('--mesh_visibility_culling', action='store_true',
                        help="cull mesh faces based on visibility in training dataset")
    parser.add_argument('--visibility_mask_dilation', type=int, default=5, help="visibility dilation")
    parser.add_argument('--clean_min_f', type=int, default=8, help="mesh clean: min face count for isolated mesh")
    parser.add_argument('--clean_min_d', type=int, default=5, help="mesh clean: min diameter for isolated mesh")

    ### Spike Neuron Config
    parser.add_argument('--num_layers_sigma', type=int, default=2, help="Number of FC layers for sigma")
    parser.add_argument('--hidden_dim_sigma', type=int, default=64, help="Hidden dimension of sigma")
    parser.add_argument('--output_dim_sigma', type=int, default=15, help="Output dimension of sigma")
    parser.add_argument('--Time_step', type=int, default=4., help="Max time step")
    parser.add_argument('--num_layers_color', type=int, default=3, help="Number of FC layers for color")
    parser.add_argument('--hidden_dim_color', type=int, default=64, help="Hidden dimension of color")
    parser.add_argument('--forward_type', type=str, default='skip', help="Forward type")
    parser.add_argument('--neuron_type', type=str, default='Custom_LIF', help="Type of neuron", choices=['Custom_LIF', 'LIF', 'IF'])
    parser.add_argument('--tau', type=float, default=2.0, help="Initial value of tau")
    parser.add_argument('--vth', type=float, default=0.5, help="Initial value of threshold")

    ### Quantization Config
    parser.add_argument('--ptq', type=bool, default=False, help="enable quantization")
    parser.add_argument('--bit', type=int, default=8, help="")
    parser.add_argument('--Mem_bit', type=int, default=32, help="")
    parser.add_argument('--adaround_iter', type=int, default=4000, help="")
    parser.add_argument('--not_quan_model_path', type=str, help="")
    parser.add_argument('--symmetric', type=bool, default= True, help="")
    

    ### Dynamic Time Step
    parser.add_argument('-D', action='store_true', help="enable dynamic time step") 
    parser.add_argument('--pth_path', type=str, default=None, help="path to the pre-trained model")
    parser.add_argument('--net',type=str, default='ann', help=" Model type", choices=['snn', 'ann'])
    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = False # TODO: This is not supported by QAT
        opt.preload = True
        opt.cuda_ray = True
        opt.mark_untrained = True
        opt.adaptive_num_rays = True
        opt.random_image_batch = True  # TODO: make it compatibility to object-nerf ray selection (avoid ray of free space)

    if opt.O2:
        opt.fp16 = False  # TODO: This is not supported by QAT
        opt.bound = 128  # large enough
        opt.preload = True
        opt.contract = True
        opt.adaptive_num_rays = True
        opt.random_image_batch = True # TODO: make it compatibility to object-nerf ray selection (avoid ray of free space)

    if opt.D:
        opt.iters = 30000
        opt.ptq = False
        

    if opt.contract:
        # mark untrained is not correct in contraction mode...
        opt.mark_untrained = False

    if opt.data_format == 'colmap':  # default setting
        opt.background = 'random'
        opt.enable_cam_center = True
        if not opt.selfbound:
            opt.bound = 8.0
        opt.dt_gamma = 0
        opt.random_image_batch = True

    if opt.data_format == 'nerf':  # default setting
        opt.background = 'random'
        opt.enable_cam_center = False
        opt.dt_gamma = 0
        if not opt.selfbound:
            opt.bound = 1.0
        if opt.scale == -1:
            opt.scale = 0.8
    return opt
