#!/bin/bash

    export TORCH_DISTRIBUTED_DEBUG=INFO
    CUDA_VISIBLE_DEVICES=7 \
        python -m torch.distributed.launch \
                --master_port=11452 \
                --nproc_per_node=1 \
                --use_env main.py \
                mydatasets_s_dualprompt \
                --model vit_base_patch16_224 \
                --batch_size 100 \
                --data_path /media/dataset/myDataset \
                --output_dir ./output/mc_diff_PrefixT_P10_K15 \
                --epochs 100 \
                --length 5 \
                --no_log \
                --datasets_list mya1 mya2 mya3 mya4 mya5 \
                --k_means 15 \
                --eval_all_tasks \
#                --e_prompt_layer_idx 0 \
#                --no_checkpoint
              

