import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from matplotlib.colors import Normalize
import torch

def visualize_feature_grids(feature_grids, output_dir="feature_visualizations", 
                           titles=None, figsize=(16, 16), dpi=300, 
                           show_individual=True, show_combined=True):
    # 确保有16个层级
    assert len(feature_grids) == 16, "需要提供16个层级的特征网格"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为所有层级计算统一的颜色映射范围（如果是2D特征）
    vmin, vmax = None, None
    for grid in feature_grids:
        if len(grid.shape) == 2:
            current_min = np.min(grid)
            current_max = np.max(grid)
            if vmin is None or current_min < vmin:
                vmin = current_min
            if vmax is None or current_max > vmax:
                vmax = current_max
    
    # 单个层级可视化
    if show_individual:
        for i in range(16):
            grid = feature_grids[i]
            fig, ax = plt.subplots(figsize=(8, 8))
            
            if len(grid.shape) == 2:
                # 2D特征网格，显示热图
                norm = Normalize(vmin=vmin, vmax=vmax) if vmin is not None else None
                im = ax.imshow(grid, cmap='viridis', norm=norm)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"Level {i+1}" if titles is None else titles[i])
            elif len(grid.shape) == 3 and grid.shape[-1] == 3:
                # 3D特征网格且最后一维为3，视为RGB图像
                ax.imshow(grid)
                ax.set_title(f"Level {i+1}" if titles is None else titles[i])
            else:
                # 其他维度，显示直方图
                ax.hist(grid.flatten(), bins=50, alpha=0.7, color='blue')
                ax.set_yscale('log')  # 使用对数刻度更好地展示分布
                ax.set_title(f"Level {i+1} - Feature Distribution" if titles is None else titles[i])
                ax.set_xlabel("Feature Value")
                ax.set_ylabel("Frequency (log scale)")
            
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"feature_level_{i+1}.png"), dpi=dpi, bbox_inches='tight')
            plt.close(fig)
    
    # 组合可视化（4x4网格）
    if show_combined:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(4, 4, figure=fig, wspace=0.3, hspace=0.4)
        
        for i in range(16):
            grid = feature_grids[i]
            ax = fig.add_subplot(gs[i])
            
            if len(grid.shape) == 2:
                # 2D特征网格，显示热图
                norm = Normalize(vmin=vmin, vmax=vmax) if vmin is not None else None
                im = ax.imshow(grid, cmap='viridis', norm=norm)
                if i == 15:  # 在最后一个子图添加颜色条
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                    fig.colorbar(im, cax=cbar_ax)
            elif len(grid.shape) == 3 and grid.shape[-1] == 3:
                # 3D特征网格且最后一维为3，视为RGB图像
                ax.imshow(grid)
            else:
                # 其他维度，显示直方图
                ax.hist(grid.flatten(), bins=30, alpha=0.7, color='blue')
                ax.set_yscale('log')
                ax.tick_params(axis='both', which='major', labelsize=6)
            
            # 设置标题
            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=10)
            else:
                ax.set_title(f"Level {i+1}", fontsize=10)
            
            # 简化坐标轴
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle("Multi-resolution Feature Grids Visualization", fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # 为右侧颜色条留出空间
        fig.savefig(os.path.join(output_dir, "all_feature_levels.png"), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    print(f"可视化完成，图像已保存至 {os.path.abspath(output_dir)}")

# 示例使用
if __name__ == "__main__":

    # feature_grids = []
    # embeddings = torch.load('/home/liuwh/lrx/PATA_code/workspace/quan/chair_symmetric_8_8bit/checkpoints/quan.pth')['model']['encoder.embeddings'].cpu().numpy()
    # offsets = torch.load('/home/liuwh/lrx/PATA_code/workspace/quan/chair_symmetric_8_8bit/checkpoints/quan.pth')['model']['encoder.offsets'].cpu().numpy()
    
    # for i in range(1, offsets.shape[0]):
    #     grid = embeddings[offsets[i-1]:offsets[i]]
    #     print(f"Grid {i}: shape={grid.shape}, mean={grid.mean():.4f}, std={grid.std():.4f}, min={grid.min():.4f}, max={grid.max():.4f}")
    #     feature_grids.append(grid.flatten())
    
    feature_grids = torch.load('/home/liuwh/lrx/PATA_code/workspace/quan/chair_symmetric_8_8bit/checkpoints/feature_grids_4bit.pt')#.cpu().numpy()
    for i in range(len(feature_grids)):
        # feature_grids[i] = feature_grids[i].cpu().numpy()
        print(feature_grids[i].shape, feature_grids[i].device, type(feature_grids[i]))
        
    # 可视化特征网格
    visualize_feature_grids(
        feature_grids,
        output_dir="/home/liuwh/lrx/PATA_code/snn_instantngp_feature_vis/features_4bit",
        # show_individual=True,
        show_combined=True
    )
