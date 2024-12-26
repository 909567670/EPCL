# 自定义训练 接续训练
#   1. 从指定的checkpoint开始训练
#   2. 必须指定数据集datasets_list [1,2,3,4,5] 只训练最后一个数据集
#   3. 将评估所有数据集eval_all_tasks

        python -m torch.distributed.launch \
                --nproc_per_node=8 \
                --use_env main.py \
                mydatasets_dualprompt \
                --model vit_base_patch16_224 \
                --batch-size 160 \
                --data-path /home/lyx/l2p-pytorch-my/l2p-pytorch-main/myDatasets \
                --output_dir ./output/mydatasets_dual_NM_topk6_000 \
                --epochs 25 \
                --size 10 \
                --length 5 \
                --top_k 6 \
                --head_type token+prompt \
                --no_log \
                --eval_all_tasks \
                --datasets_list my1 my2 my3 my4 my5 my3+  \
                --my_train \
                --checkpoint ./output/mydatasets_dual_NM_topk6_000/task5_checkpoint.pth

# done


