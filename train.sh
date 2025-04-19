# For real scenes, add this command to eliminate the impact of background. 
# env_scope_center and env_scope_radius represent a sphere domain where the environment light takes effect.
# https://github.com/gapszju/3DGS-DR
#############################################################################################
#### For better results, more training steps (e.g., 60k steps, like 3DGS-DR) can be set,  ###
#### although we observe that the model already yields good results after 30k steps.      ###
#############################################################################################

# python train-real.py -s data/ref_real/sedan -r 8 --eval --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --env_scope_center -0.032 0.808 0.751 --env_scope_radius 2.138 --init_until_iter 700 --xyz_axis 2.0 1.0 0.0
# python train-real.py -s data/ref_real/gardenspheres -r 6 --eval --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --env_scope_center -0.2270 1.9700 1.7740 --env_scope_radius 0.974 --init_until_iter 700 --xyz_axis 2.0 1.0 0.0
# python train-real.py -s data/ref_real/toycar -r 6 --eval --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --env_scope_center 0.486 1.108 3.72 --env_scope_radius 2.507 --init_until_iter 1500 --xyz_axis 0.0 2.0 1.0

# python3 train.py -s data/refnerf/helmet --eval --run_dim 256 --albedo_bias 0
# python3 train.py -s data/refnerf/car --eval --run_dim 256 --albedo_bias 0
# python3 train.py -s data/refnerf/ball --eval --run_dim 256 --albedo_bias 0
# python3 train.py -s data/refnerf/teapot --eval --run_dim 256 --albedo_bias 0
# python3 train.py -s data/refnerf/coffee --eval --run_dim 256 --albedo_bias 0 --albedo_lr 0.002
# python train.py -s data/refnerf/toaster --eval --run_dim 256 --albedo_bias 0

# concurrently run the following commands in different terminals
log_dir="logs/refnerf/"
gpus="1 2 3 4 5 6 7"  # 改用空格分隔的字符串，而不是数组
mkdir -p "$log_dir"

# 任务列表（用换行符或分号分隔）
tasks="
-s data/refnerf/helmet --eval --run_dim 256 --albedo_bias 0
-s data/refnerf/car --eval --run_dim 256 --albedo_bias 0
-s data/refnerf/ball --eval --run_dim 256 --albedo_bias 0
-s data/refnerf/teapot --eval --run_dim 256 --albedo_bias 0
-s data/refnerf/coffee --eval --run_dim 256 --albedo_bias 0 --albedo_lr 0.002
"

# 临时修改IFS为换行符，确保按行读取任务
IFS='
'
i=0
for task in $tasks; do
    if [ -z "$task" ]; then continue; fi  # 跳过空行
    gpu_id=$(echo "$gpus" | cut -d " " -f $((i % 8 + 1)))
    task_name=$(echo "$task" | awk '{print $2}' | awk -F/ '{print $NF}')
    log_file="${log_dir}${task_name}.log"
    
    echo "Starting task on GPU python3 train.py $gpu_id: $task"
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id python3 train.py $task > \"$log_file\" 2>&1 &"
    eval $cmd

    i=$((i + 1))
done
IFS=' '  # 恢复默认IFS

wait
echo "All tasks completed."

# python train-NeRF.py -s data/nerf_synthetic/ship --eval --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002 
# python train-NeRF.py -s data/nerf_synthetic/ficus --eval --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002 
# python train-NeRF.py -s data/nerf_synthetic/lego --eval --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002 
# python train-NeRF.py -s data/nerf_synthetic/mic --eval --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002 
# python train-NeRF.py -s data/nerf_synthetic/hotdog --eval --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002 
# python train-NeRF.py -s data/nerf_synthetic/chair --eval --run_dim 64 --albedo_bias 0 --gsrgb_loss --albedo_lr 0.002 
# python train-NeRF.py -s data/nerf_synthetic/materials --eval --run_dim 256 --albedo_bias 0 
# python train-NeRF.py -s data/nerf_synthetic/drums --eval --run_dim 64 --albedo_bias 0 --albedo_lr 0.002 

# python train-NeRO.py -s data/GlossySynthetic/bell_blender --eval --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000 
# python train-NeRO.py -s data/GlossySynthetic/tbell_blender --eval --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000
# python train-NeRO.py -s data/GlossySynthetic/potion_blender --eval --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000
# python train-NeRO.py -s data/GlossySynthetic/teapot_blender --eval --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000
# python train-NeRO.py -s data/GlossySynthetic/luyu_blender --eval --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000
# python train-NeRO.py -s data/GlossySynthetic/cat_blender --eval --run_dim 256 --albedo_bias 2 --albedo_lr 0.0005 --init_until_iter 3000