#!/bin/bash

# 本脚本对每个数据集进行 性能上限测试

datasets_list=("mya1" "mya2" "mya3" "mya4" "mya5" "mya6")
num_runs=${#datasets_list[@]}
echo "执行共 ${num_runs} 次"
# 循环执行脚本多次
for ((i=0; i<$num_runs; i++)); do
    echo "执行第 $((i+1)) 次，参数为: ${datasets_list[i]}"

    export TORCH_DISTRIBUTED_DEBUG=INFO
    CUDA_VISIBLE_DEVICES=7,6,5 \
        python -m torch.distributed.launch \
                --master_port=1145 \
                --nproc_per_node=3 \
                --use_env main.py \
                mydatasets_s_dualprompt \
                --model vit_base_patch16_224 \
                --batch-size 160 \
                --data-path /media/dataset/myDataset \
                --output_dir ./output/a_s_dual_MC_010 \
                --epochs 200 \
                --length 10 \
                --head_type token \
                --no_log \
                --datasets_list "${datasets_list[i]}"\
                --no_checkpoint
                
    echo "第 $((i+1)) 次执行完成。"
done




