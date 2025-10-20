### Synthetic-NeRF Dataset
# python  main.py  /home/linranxi/Code/DATA/nerf_synthetic/chair  --workspace /home/linranxi/Code/PATA/workspace/original/nerf_synthetic/chair_LIF -O --data_format nerf
# python  main.py  /home/linranxi/Code/DATA/nerf_synthetic/chair  --workspace /home/linranxi/Code/PATA/workspace/original/nerf_synthetic_dynamic/chair2  --pth_path /home/linranxi/Code/PATA/workspace/original/nerf_synthetic/chair2/checkpoints/ngp_ep0100.pth  -O -D --data_format nerf
# python temp.py /home/linranxi/Code/DATA/nerf_synthetic/chair  --workspace /home/linranxi/Code/PATA/workspace/quan/test/chair -O --data_format nerf
# python  main.py  /home/linranxi/Code/DATA/nerf_synthetic/chair  --workspace /home/linranxi/Code/PATA/workspace/ann/nerf_synthetic/chair -O --data_format nerf

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/chair \
#        --bit 4 --Mem_bit 8\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/chair_symmetric_4_8bit \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic/chair_t4/checkpoints/ngp_ep0300.pth'\
#        -O --data_format nerf 

# BIT=8
# MEM_BIT=8

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/chair  --ptq True \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/chair_symmetric_8_8bit11 \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic/chair_t4/checkpoints/ngp_ep0300.pth'\
#        -O --data_format nerf 

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/chair --ptq True \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/chair_symmetric_8_8bit16 \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic/chair_t4/checkpoints/ngp_ep0300.pth'\
#        -O --data_format nerf 

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/chair --ptq True \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/chair_symmetric_8_8bit25 \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic/chair_t4/checkpoints/ngp_ep0300.pth'\
#        -O --data_format nerf 

# python main_quan.py /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/chair --ptq True \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/chair_symmetric_8_8bit21 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/chair_t4/checkpoints/ngp_ep0300.pth\
#        -O --data_format nerf 

# python main_quan.py /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/materials --ptq True \
#        --bit 4 --Mem_bit 4\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/materials_symmetric_4_4bit25 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/materials_t4/checkpoints/ngp_ep0300.pth\
#        -O --data_format nerf 

# python main_quan.py /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/chair --ptq True \
#        --bit 4 --Mem_bit 4\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/chair_symmetric_4_4bit10 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/chair128_t4/checkpoints/ngp_ep0300.pth\
#        -O --data_format nerf 

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/chair --ptq True \
#        --bit 4 --Mem_bit 4\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/chair_symmetric_4_4bit8 \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic/chair_t4/checkpoints/ngp_ep0300.pth'\
#        -O --data_format nerf 

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/chair \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/chair_symmetric_8_8bit5 \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic_dynamic/chair2/checkpoints/ngp_ep0294.pth'\
#        -O --data_format nerf 

# python  main.py  /home/linranxi/Code/DATA/nerf_synthetic/chair  --workspace /home/linranxi/Code/PATA/workspace/ann/nerf_synthetic/chair -O --data_format nerf

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/chair \
#        --workspace /home/linranxi/Code/PATA/workspace/ann_quan/nerf_synthetic/chair_12_4 \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/ann/nerf_synthetic/chair/checkpoints/ngp_ep0294.pth'\
#        -O --data_format nerf 


# python  main.py  /home/linranxi/Code/DATA/nerf_synthetic/mic  --workspace /home/linranxi/Code/PATA/workspace/original/nerf_synthetic/mic_t4 -O --data_format nerf
# python  main.py  /home/linranxi/Code/DATA/nerf_synthetic/ship  --workspace /home/linranxi/Code/PATA/workspace/original/nerf_synthetic/ship_t4 -O --data_format nerf --selfbound --bound 1.3
# python  main.py  /home/linranxi/Code/DATA/nerf_synthetic/drums  --workspace /home/linranxi/Code/PATA/workspace/original/nerf_synthetic/drums_t4 -O --data_format nerf
# python  main.py  /home/linranxi/Code/DATA/nerf_synthetic/ficus  --workspace /home/linranxi/Code/PATA/workspace/original/nerf_synthetic/ficus_t4 -O --data_format nerf
# python  main.py  /home/linranxi/Code/DATA/nerf_synthetic/lego  --workspace /home/linranxi/Code/PATA/workspace/original/nerf_synthetic/lego_t4 -O --data_format nerf
# python  main.py  /home/linranxi/Code/DATA/nerf_synthetic/materials  --workspace /home/linranxi/Code/PATA/workspace/original/nerf_synthetic/materials_t4 -O --data_format nerf
# python  main.py  /home/linranxi/Code/DATA/nerf_synthetic/hotdog  --workspace /home/linranxi/Code/PATA/workspace/original/nerf_synthetic/hotdog_t4 -O --data_format nerf --selfbound --bound 1.3


# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/mic --ptq True \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/multi_scene/mic_symmetric_8_8bit \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic/mic_t4/checkpoints/ngp_ep0300.pth'\
#        -O --data_format nerf 

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/lego --ptq True \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/multi_scene/lego_symmetric_8_8bit \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic/lego_t4/checkpoints/ngp_ep0300.pth'\
#        -O --data_format nerf 

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/materials --ptq True \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/multi_scene/materials_symmetric_8_8bit \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic/materials_t4/checkpoints/ngp_ep0300.pth'\
#        -O --data_format nerf 

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/hotdog --ptq True \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/multi_scene/hotdog_symmetric_8_8bit \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic/hotdog_t4/checkpoints/ngp_ep0300.pth'\
#        -O --data_format nerf  --selfbound --bound 1.3

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/drums --ptq True \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/multi_scene/drums_symmetric_8_8bit \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic/drums_t4/checkpoints/ngp_ep0300.pth'\
#        -O --data_format nerf 

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/ship --ptq True \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/multi_scene/ship_symmetric_8_8bit \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic/ship_t4/checkpoints/ngp_ep0300.pth'\
#        -O --data_format nerf --selfbound --bound 1.3

# python main_quan.py /home/linranxi/Code/DATA/nerf_synthetic/ficus --ptq True \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/linranxi/Code/PATA/workspace/quan/nerf_synthetic/multi_scene/ficus_symmetric_8_8bit \
#        --not_quan_model_path '/home/linranxi/Code/PATA/workspace/original/nerf_synthetic/ficus_t4/checkpoints/ngp_ep0300.pth'\
#        -O --data_format nerf 

# python  main.py  /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/materials  --workspace /home/liuwh/lrx/PATA_code/workspace/original/materials_t4 -O --data_format nerf
# python  main.py  /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/chair  --workspace /home/liuwh/lrx/PATA_code/workspace/original/chair128_t4 -O --data_format nerf


# python  main.py  /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/mic  --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time_128/nerf_synthetic/mic_t42 -O --data_format nerf
# python  main.py  /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/ship  --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time_128/nerf_synthetic/ship_t4 -O --data_format nerf --selfbound --bound 1.3
# python  main.py  /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/drums  --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time_128/nerf_synthetic/drums_t4 -O --data_format nerf
# python  main.py  /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/ficus  --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/ficus_t4 -O --data_format nerf
# python  main.py  /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/lego  --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/lego_t4 -O --data_format nerf
# python  main.py  /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/materials  --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/materials_t4 -O --data_format nerf
# python  main.py  /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/hotdog  --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/hotdog_t4 -O --data_format nerf --selfbound --bound 1.3
# python  main.py  /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/chair  --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/chair_t4 -O --data_format nerf

# dirsname=mipnerf_t2
# python main.py /home/liuwh/lrx/Data/360_v2/bonsai      --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time/$dirsname/bonsai2      -O --data_format colmap --downscale 4
# python main.py /home/liuwh/lrx/Data/360_v2/bicycle     --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time/$dirsname/bicycle     -O --data_format colmap --downscale 4
# python main.py /home/liuwh/lrx/Data/360_v2/counter     --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time/$dirsname/counter     -O --data_format colmap --downscale 4
# python main.py /home/liuwh/lrx/Data/360_v2/garden      --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time/$dirsname/garden      -O --data_format colmap --downscale 4
# python main.py /home/liuwh/lrx/Data/360_v2/kitchen     --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time/$dirsname/kitchen     -O --data_format colmap --downscale 4
# python main.py /home/liuwh/lrx/Data/360_v2/stump       --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time/$dirsname/stump       -O --data_format colmap --downscale 4
# python main.py /home/liuwh/lrx/Data/360_v2/room        --workspace /home/liuwh/lrx/PATA_code/workspace/original/fix_time/$dirsname/room        -O --data_format colmap --downscale 4

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/lego --ptq True \
#        --bit 8 --Mem_bit 8\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/adaround/lego_8 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/lego_t4/checkpoints/ngp_ep0400.pth\
#        -O --data_format nerf 

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/chair --ptq True \
#        --bit 6 --Mem_bit 6\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/adaround/chair_6 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/chair_t4/checkpoints/ngp_ep0400.pth\
#        -O --data_format nerf 

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/chair --ptq True \
#        --bit 4 --Mem_bit 4\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/adaround/chair_4 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/chair_t4/checkpoints/ngp_ep0400.pth\
#        -O --data_format nerf 

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/ficus --ptq True \
#        --bit 8 --Mem_bit 4\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G8W8A4/PSQ_PMC/ficus \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/ficus_t4/checkpoints/ngp_ep0400.pth\
#        -O --data_format nerf 

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/lego  --ptq True \
#        --bit 8 --Mem_bit 4\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G8W8A4/PSQ_PMC/lego \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/lego_t4/checkpoints/ngp_ep0400.pth\
#        -O --data_format nerf 

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/materials --ptq True \
#        --bit 8 --Mem_bit 4\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G8W8A4/PSQ_PMC/materials \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/materials_t4/checkpoints/ngp_ep0400.pth\
#        -O --data_format nerf 

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/hotdog --ptq True \
#        --bit 8 --Mem_bit 4\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G8W8A4/PSQ_PMC/hotdog \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/hotdog_t4/checkpoints/ngp_ep0400.pth\
#        -O --data_format nerf  --selfbound --bound 1.3

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/mic --ptq True \
#        --bit 8 --Mem_bit 4\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G8W8A4/128/mic \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time_128/nerf_synthetic/mic_t4/checkpoints/ngp_ep0600.pth\
#        -O --data_format nerf 


# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/360_v2/bonsai  --ptq True \
#        --bit 6 --Mem_bit 6\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G6W6A6/mipnerf/bonsai3 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/mipnerf/bonsai/checkpoints/ngp_ep0157.pth\
#        -O --data_format colmap --downscale 4

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/360_v2/bicycle3  --ptq True \
#        --bit 6 --Mem_bit 6\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G6W6A6/mipnerf/bicycle3 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/mipnerf/bicycle/checkpoints/ngp_ep0237.pth\
#        -O --data_format colmap --downscale 4

# CUDA_VISIBLE_DEVICES=2 python main_quan.py /home/liuwh/lrx/Data/360_v2/counter --ptq True \
#        --bit 6 --Mem_bit 6\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G6W6A6/mipnerf/counter2 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/mipnerf/counter/checkpoints/ngp_ep0191.pth\
#        -O --data_format colmap --downscale 4

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/360_v2/garden --ptq True \
#        --bit 6 --Mem_bit 6\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G6W6A6/mipnerf/garden2 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/mipnerf/garden/checkpoints/ngp_ep0249.pth\
#        -O --data_format colmap --downscale 4  # 20685112

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/360_v2/kitchen  --ptq True \
#        --bit 6 --Mem_bit 6\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G6W6A6/mipnerf/kitchen2 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/mipnerf/kitchen/checkpoints/ngp_ep0164.pth\
#        -O --data_format colmap --downscale 4

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/360_v2/stump --ptq True \
#        --bit 6 --Mem_bit 6\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G6W6A6/mipnerf/stump2 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/mipnerf/stump/checkpoints/ngp_ep0367.pth\
#        -O --data_format colmap --downscale 4

# CUDA_VISIBLE_DEVICES=1 python main_quan.py /home/liuwh/lrx/Data/360_v2/room --ptq True \
#        --bit 6 --Mem_bit 6\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G6W6A6/mipnerf/room2 \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/mipnerf/room/checkpoints/ngp_ep0148.pth\
#        -O --data_format colmap --downscale 4 

# python main_quan.py /home/liuwh/lrx/Data/nerf_synthetic/nerf_synthetic/mic --ptq True \
#        --bit 7 --Mem_bit 7\
#        --workspace /home/liuwh/lrx/PATA_code/workspace/quan/fix_time/G8W8A4/nerf_synthetic/mic \
#        --not_quan_model_path /home/liuwh/lrx/PATA_code/workspace/original/fix_time/nerf_synthetic/mic_t4/checkpoints/ngp_ep0400.pth\
#        -O --data_format nerf 


# python  main.py  /home/linrx/Data/nerf_synthetic/lego  --workspace /home/linrx/Code/PATA_Quan/workspace/synthetic_nerf/ann/lego -O --data_format nerf
python feature_grid_visual.py  /home/linrx/Data/nerf_synthetic/lego  --workspace /home/linrx/Code/PATA_Quan/workspace/synthetic_nerf/ann/lego -O --data_format nerf