# @Time    : 2024/5/12 下午12:01
# @Author  : yxL
# @File    : my_eval.py
# @Software: PyCharm
# @Description : 评估 sdp 模型

import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import *
import models
import utils


def main(opts):
    print(f"开始测试模型: {opts.load_checkpoint}\n数据集: {opts.datasets_list}")
    utils.init_distributed_mode(opts)

    device = torch.device(opts.device)

    # fix the seed for reproducibility
    seed = opts.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    '''
    评估模式下   忽略train与val子文件夹需求 仅读取 opt.data_path/opt.datasets_list 下的图片
               并且不构建训练data_loader 仅构建验证data_loader
    '''
    opts.eval = False  # 设置为评估模式 与 数据集加载有关

    data_loader, _, datasets_list = build_continual_dataloader(opts)


    # 读取检查点信息
    checkpoint = torch.load(opts.load_checkpoint)
    args = checkpoint['args']  # 读取训练时的参数

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
        use_multi_classifier=args.multi_classifier,
        num_tasks=args.num_tasks,
    )
    original_model.to(device)
    model.to(device)


    if opts.pre_k_means:
        args.device = opts.device
        args.k_means = opts.k_means
        pre_diff_clustering(original_model=original_model,data_loader=data_loader,args=args)
    else:
        opts.k_means = args.k_means

    # 设置评估模式
    original_model.eval()
    model.eval()

    # 加载模型参数
    model.load_state_dict(checkpoint['model'])


    args.output_dir = opts.output_dir
    args.eval = True
    # 写入部分参数信息到文件
    selected_args = [
                     # 'eval',
                     'num_tasks',  # 任务数量
                     'load_checkpoint',
                     'k_means',
                    ]  # 使用的checkpoint
    if opts.output_dir and utils.is_main_process():  # 主线程写入
        with open(os.path.join(opts.output_dir, 'eval_result.log'), 'a') as f:
            f.write(f"\n@ log time: {datetime.datetime.now()}\n")
            for k, v in opts.__dict__.items():  # 将args转为字典
                if k in selected_args:  # 若key在输出列表中
                    f.write(f"{k}: {v}\n")
            f.write(f"center_size:{args.diff_all_keys.size()}\n")
            f.write(f'datasets: {str(datasets_list)}\n')
            f.write('================================================\n\n')

    acc_list=[]
    select_list=[]

    for i in range(len(datasets_list)):
        res: dict = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                             device=device, task_id=i, class_mask=None, args=args)
        print('\n================================================')
        print(f"({i}) 数据集: {datasets_list[i]} Acc: {res['Acc@1']}")

        # with open(os.path.join(opts.output_dir, 'eval_result.log'), 'a') as f:
        #     f.write(f"({i}) 数据集: {datasets_list[i]} \nResult: {str(res)}\n")
        acc_list.append(res['Acc@1'])
        select_list.append(res['Select_Acc'])

    # 统计平均准确率
    ave_acc = sum(acc_list)/len(acc_list)
    ave_select = sum(select_list)/len(select_list)
    print('\n================================================')
    print(f"数据集: {datasets_list} ave_acc: {ave_acc:.3f} select_acc: {ave_select:.3f}\n")

    with open(os.path.join(opts.output_dir, 'eval_result.log'), 'a') as f:
            f.write(f"数据集: {datasets_list} ave_acc: {ave_acc:.3f} select_acc: {ave_select*100:.3f}\n")



if __name__ == '__main__':

    parser = argparse.ArgumentParser('sdp eval')
    from configs.mydatasets_s_dualprompt import get_args_parser

    get_args_parser(parser)
    parser.add_argument('--pre_k_means',action='store_true',help = '测试前聚类, 即不使用保存的聚类数量')

    opts = parser.parse_args()

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    main(opts)

    sys.exit(0)
