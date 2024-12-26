#!/bin/bash
# 绘制特征分布图
        python  my_featuresMap.py \
                mydatasets_s_dualprompt \
                --model vit_base_patch16_224 \
                --batch_size 100 \
                --data_path /media/dataset/myDataset \
                --output_dir ./feature_map \
                --head_type token \
                --datasets_list mya1 mya2 mya3 mya4 mya5 \
                --plot_mode task \
                --method tsne \

                





