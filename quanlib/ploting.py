import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import os
from pathlib import Path
from torch.nn.functional import relu

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def spike(x, y):
    # return torch.nn.functional.relu(x - y) / (x - y + 1e-9)
    return (x > y).float()

def plot_stats_comparison(o_stats, q_stats, stat_name, title=None, 
                         save_path=None, show=True, figsize=(12, 6),
                         channel_range=None, highlight_errors=True):
    """
    绘制原始模型和量化模型的统计量对比图
    
    参数:
    o_stats (torch.Tensor): 原始模型的统计量，形状为(通道数,)
    q_stats (torch.Tensor): 量化模型的统计量，形状为(通道数,)
    stat_name (str): 统计量名称，如"最大值"、"最小值"、"平均值"
    title (str, optional): 图表标题，默认为"原始模型与量化模型的{stat_name}对比"
    save_path (str, optional): 图像保存路径，不提供则不保存
    show (bool): 是否显示图像
    figsize (tuple): 图像大小
    channel_range (tuple, optional): 只显示指定范围的通道，如(0, 100)
    highlight_errors (bool): 是否高亮显示差异较大的通道
    """
    # 转换为numpy数组
    o_stats = o_stats.cpu().numpy() if isinstance(o_stats, torch.Tensor) else np.array(o_stats)
    q_stats = q_stats.cpu().numpy() if isinstance(q_stats, torch.Tensor) else np.array(q_stats)
    
    # 确保输入是一维数组
    o_stats = o_stats.flatten()
    q_stats = q_stats.flatten()
    
    # 检查输入是否有效
    if o_stats.shape != q_stats.shape:
        raise ValueError("原始模型和量化模型的统计量形状必须相同")
    
    # 计算差异
    diff = np.abs(o_stats - q_stats)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    # 确定显示的通道范围
    total_channels = o_stats.shape[0]
    if channel_range is None:
        start_idx, end_idx = 0, total_channels
    else:
        start_idx, end_idx = channel_range
        start_idx = max(0, start_idx)
        end_idx = min(total_channels, end_idx)
    
    channels = np.arange(start_idx, end_idx)
    o_plot = o_stats[start_idx:end_idx]
    q_plot = q_stats[start_idx:end_idx]
    diff_plot = diff[start_idx:end_idx]
    
    # 设置默认标题
    if title is None:
        title = f"原始模型与量化模型的{stat_name}对比"
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # 绘制原始模型和量化模型的统计量
    ax1.plot(channels, o_plot, 'b-', label='原始模型', linewidth=1.5, alpha=0.7)
    ax1.plot(channels, q_plot, 'r-', label='量化模型', linewidth=1.5, alpha=0.7)
    
    # 高亮显示差异较大的通道
    if highlight_errors:
        threshold = mean_diff + 2 * std_diff  # 定义差异阈值
        high_diff_mask = diff_plot > threshold
        ax1.scatter(channels[high_diff_mask], o_plot[high_diff_mask], 
                   c='cyan', s=30, edgecolors='k', alpha=0.8, 
                   label=f'高差异通道(差异>{threshold:.4f})')
    
    ax1.set_ylabel(stat_name)
    ax1.set_title(title)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    ax1.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=8))
    
    # 绘制差异
    ax2.bar(channels, diff_plot, color='gray', alpha=0.7, width=0.8)
    ax2.axhline(y=mean_diff, color='g', linestyle='--', label=f'平均差异: {mean_diff:.4f}')
    ax2.axhline(y=max_diff, color='m', linestyle='--', label=f'最大差异: {max_diff:.4f}')
    
    ax2.set_xlabel('通道索引')
    ax2.set_ylabel('绝对差异')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    ax2.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=5))
    
    # 设置x轴
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    # 显示图像
    if show:
        plt.show()
    
    return fig, (ax1, ax2)


if __name__ == '__main__':
    
    import warnings
    warnings.filterwarnings("ignore")
    
    bit = 4
    mem_bit = 4
    o_path = f'/home/linranxi/Code/PATA/workspace/intermediate_data/W{bit}U{mem_bit}/LIF'
    q_path = f'/home/linranxi/Code/PATA/workspace/intermediate_data/W{bit}U{mem_bit}/QLIF'

    o_data_list = os.listdir(f'/home/linranxi/Code/PATA/workspace/intermediate_data/W{bit}U{mem_bit}/LIF')
    q_data_list = os.listdir(f'/home/linranxi/Code/PATA/workspace/intermediate_data/W{bit}U{mem_bit}/QLIF')

    model = torch.load(f'/home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/chair_symmetric_{bit}_{mem_bit}bit/checkpoints/quan.pth')
    # q_data_list, o_data_list
    # for key in model['model'].keys():
    #     if 'scale' in key:
    #         print(f'key {key}: ', model['model'][key])
    
    model_state_dict = {'sigma_0': {'path' : 'sigma_0_fig', 'scale' : 'sigma_net.1.Mem_quantizer.scale_exp', 'max' : 'sigma_net.1.Mem_quantizer.observer.max_val', 'min' : 'sigma_net.1.Mem_quantizer.observer.min_val', 'time' : 'sigma_net.0_timestep'},
                        'color_0': {'path' : 'color_0_fig', 'scale' : 'color_net.1.Mem_quantizer.scale_exp', 'max' : 'color_net.1.Mem_quantizer.observer.max_val', 'min' : 'color_net.1.Mem_quantizer.observer.min_val', 'time' : 'color_net.0_timestep'},
                        'color_2': {'path' : 'color_2_fig', 'scale' : 'color_net.3.Mem_quantizer.scale_exp', 'max' : 'color_net.3.Mem_quantizer.observer.max_val', 'min' : 'color_net.3.Mem_quantizer.observer.min_val', 'time' : 'color_net.2_timestep'}}
    
    # scale_exp = model['model']['color_net.3.Mem_quantizer.scale_exp']
    # maxs = model['model']['color_net.3.Mem_quantizer.observer.max_val']
    # mins = model['model']['color_net.3.Mem_quantizer.observer.min_val']
    # scale = 2 ** torch.round(scale_exp)
    # print(f'scale_exp: {scale_exp}, scale: {scale}')
    # v_thq = torch.round(torch.tensor(0.5) / scale) * scale
    
    # o_spike = 0
    # q_spike = 0
    # save_par_pth = 'color_2_fig'
    print(f'Case: W{bit}U{mem_bit}')
    for key in model_state_dict.keys():
        save_par_pth = model_state_dict[key]['path']
        scale_exp = model['model'][model_state_dict[key]['scale']]
        maxs = model['model'][model_state_dict[key]['max']]
        mins = model['model'][model_state_dict[key]['min']]
        o_spike = 0
        q_spike = 0
        scale = 2 ** torch.round(scale_exp)
        if '0' in key:
            spike_key = key.replace('0', 'net.1')
        elif '2' in key:
            spike_key = key.replace('2', 'net.3')
        for mkey in model['model'].keys():
            # print(spike_key, mkey)
            if spike_key in mkey and 'vth' in mkey:
                v_thq = 0.5*torch.sigmoid(model['model'][mkey])
                print(mkey, ': ', v_thq)
        # v_thq = 0.5 * torch.sigmoid(torch.tensor(2.0))
        print('*' * 20)
        print(model_state_dict[key]['time'])
        for t in range(4):
            # suffix = f'color_net.2_timestep{t}.pth'
            suffix = f'{model_state_dict[key]['time']}{t}.pth'
            o_data_path = Path(o_path) / suffix
            q_data_path = Path(q_path) / suffix
            o = torch.load(o_data_path)
            q = torch.load(q_data_path)
            
            o_max = torch.max(o, dim = 0)[0]
            o_min = torch.min(o, dim = 0)[0]
            o_mean = torch.mean(o, dim = 0)

            q_max = torch.max(q, dim = 0)[0]
            q_min = torch.min(q, dim = 0)[0]
            q_mean = torch.mean(q, dim = 0)
            
            o_out = spike(o, 0.5)
            q_out = spike(q, v_thq)
            o_spike += o_out
            q_spike += q_out
            
            gap = torch.mean(torch.abs(o_out - q_out))
            possitive_gap = torch.mean(((o_out - q_out) > 0).float())
            negative_gap = torch.mean(((o_out - q_out) < 0).float())
            
            # print(f'The {t}-th Time Step, max value is {torch.max(o_max)}, min value is {torch.min(o_min)}, while in the quantized model, max value is {maxs}, min value is {mins}')
            
            print(f'The {t}-th Time Step, gap: {gap}, possitive_gap: {possitive_gap}, negative_gap: {negative_gap}')
            
            n_channels = q_max.shape[0]
            # print(os.path.join(**o_path.split('/')[:-1]))
            save_path = Path( '/' + os.path.join(*o_path.split('/')[:-1]))
        
            if not os.path.exists(save_path / save_par_pth ):
                os.makedirs(save_path / save_par_pth)
                
            suffix = suffix.replace('.', '_') 
            
            max_suffix = suffix + 'max.png'
            
        #     plot_stats_comparison(
        #         o_max, q_max, "最大值", 
        #         title="原始模型与量化模型的最大值对比",
        #         channel_range=(0, n_channels),
        #         save_path=save_path / save_par_pth/ max_suffix  # 改为"path/to/save/max_comparison.png"以保存图像
        #     )

        #     # 绘制最小值对比图
        #     min_suffix = suffix + 'min.png'
        #     plot_stats_comparison(
        #         o_min, q_min, "最小值", 
        #         title="原始模型与量化模型的最小值对比",
        #         channel_range=(0, n_channels),
        #         save_path=save_path / save_par_pth/ min_suffix  # 改为"path/to/save/min_comparison.png"以保存图像
        #     )

        #     # 绘制平均值对比图
        #     mean_suffix = suffix  + 'mean.png'
        #     plot_stats_comparison(
        #         o_mean, q_mean, "平均值", 
        #         title="原始模型与量化模型的平均值对比",
        #         channel_range=(0, n_channels),
        #         save_path=save_path / save_par_pth/ mean_suffix # 改为"path/to/save/mean_comparison.png"以保存图像
        #     )
            
            
        # # 绘制输出平均值对比图
        # o_spike= torch.mean(o_spike, dim=0)
        # q_spike= torch.mean(q_spike, dim=0)
        # mean_suffix = suffix  + 'spike_mean.png'
        # plot_stats_comparison(
        #     o_spike, q_spike, "脉冲平均值", 
        #     title="原始模型与量化模型的平均值对比",
        #     channel_range=(0, n_channels),
        #     save_path=save_path / save_par_pth/ mean_suffix # 改为"path/to/save/mean_comparison.png"以保存图像
        # )