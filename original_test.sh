# 2024/05/12 测试数据集
# 在普通测试中 测试数据集的任务选中率结果无效, 我们无法得知当前任务属于哪个任务
# 若测试的是训练数据集且数量对应 则选中率有效

export TORCH_DISTRIBUTED_DEBUG=INFO
CUDA_VISIBLE_DEVICES=4 \
        python original_test.py \
        mydatasets_s_dualprompt\
                --batch_size 100 \
                --data_path /media/dataset/myDataset \
                --output_dir ./output/vit/ \
                --datasets_list mya1 mya2 mya3 mya4 mya5\
                \
                