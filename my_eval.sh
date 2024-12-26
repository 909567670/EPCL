#!/bin/bash

# 2024/05/12 测试数据集
# 在普通测试中 测试数据集的任务选中率结果无效, 我们无法得知当前输入属于哪个任务
# 若测试的是训练数据集且数量对应 则选中率有效

export TORCH_DISTRIBUTED_DEBUG=INFO

CUDA_VISIBLE_DEVICES=7 \
        python -m torch.distributed.launch \
                --nproc_per_node=1 \
                --use_env my_eval.py \
                --batch_size 100 \
                --data_path /media/dataset/myDataset \
                --output_dir ./output/eval/conti \
                --datasets_list myb1 myb2 myb3 myb4 \
                --load_checkpoint ./output/mc_diff_PrefixT_P10_K15/task1_checkpoint.pth \
                \
                # --pre_k_means \
                # --k_means 1


# export TORCH_DISTRIBUTED_DEBUG=INFO

# for k in {1..29..2}
# do
#     CUDA_VISIBLE_DEVICES=6 \
#     python -m torch.distributed.launch \
#             --nproc_per_node=1 \
#             --master_port=11344 \
#             --use_env my_eval.py \
#             --batch_size 100 \
#             --data_path /media/dataset/myDataset \
#             --output_dir ./output/eval/p0 \
#             --datasets_list mya1 mya2 mya3 mya4 mya5 \
#             --load_checkpoint ./output/mc_diff_PrefixT_P0_K1/task5_checkpoint.pth \
#             --pre_k_means \
#             --k_means $k
# done