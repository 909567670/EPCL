# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
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

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask, datasets_list = build_continual_dataloader(args)

    # # 显示数据集中的一张图片
    # import matplotlib.pyplot as plt
    # for i, (images, targets) in enumerate(data_loader[0]["train"]):
    #     img = images[0]
    #     img = img.permute(1, 2, 0).numpy()
    #     plt.imshow(img)
    #     plt.show()
    #     # break

    print("数据集:", datasets_list)
    print("num_tasks:", args.num_tasks)


    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print(f"Creating model: {args.model}")
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

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False
        
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False
    
    print(args)

    # 评估模式
    if args.eval:
        # 写入部分参数信息到文件
        selected_args = ['eval',
                         'num_tasks',  # 任务数量
                         'checkpoint'] # 使用的checkpoint
        if args.output_dir and utils.is_main_process():  # 主线程写入
            with open(os.path.join(args.output_dir, 'eval_result.log'), 'a') as f:
                for k, v in args.__dict__.items():  # 将args转为字典
                    if k in selected_args:  # 若key在输出列表中
                        f.write(f"{k}: {v}\n")
                f.write(f'datasets: {str(datasets_list)}\n')
                f.write('================================================\n\n')


        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        if args.checkpoint: # 指定评估的checkpoint
            checkpoint_path = os.path.join(args.checkpoint)
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)

                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return

            args.eval_all_tasks = True # 强制评估所有的数据集
            _ = evaluate_till_now(model, original_model, data_loader, device,
                                            args.num_tasks - 1, class_mask, acc_matrix, args,)
        else: # 评估所有的checkpoint
            for task_id in range(args.num_tasks):
                checkpoint_path = os.path.join(args.output_dir, 'task{}_checkpoint.pth'.format(task_id+1))
                if os.path.exists(checkpoint_path):
                    print('Loading checkpoint from:', checkpoint_path)
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['model'])
                else:
                    print('No checkpoint found at:', checkpoint_path)
                    return
                _ = evaluate_till_now(model, original_model, data_loader, device,
                                                task_id, class_mask, acc_matrix, args,)

        return

    # 自定义的训练模式
    #   1. 从指定的checkpoint开始训练
    #   2. 必须指定数据集 [1,2,3,4,5] 只训练最后一个数据集
    #   3. 评估所有数据集
    if args.my_train:
        print('\n*** my_train mode ***\n')

        if not args.checkpoint:
            print('训练失败!!   未提供--checkpoint参数')
            return

        # 加载指定的checkpoint
        checkpoint_path = os.path.join(args.checkpoint)
        if os.path.exists(checkpoint_path):
            print('Loading checkpoint from:', checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model']) # 加载模型参数

        else:
            print('No checkpoint found at:', checkpoint_path)
            return

        # 自定义训练模式 参数设置 (保留)



    # 配置超参数
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],)
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    # 输出训练配置参数到文件
    selected_args = ['num_tasks', #任务数量
                     'model',       
                     'epochs',
                     'train_mask',
                     'sched',
                    #  'prompt_pool',
                    #  'size',
                     # 'top_k',
                     # 'pull_constraint_coeff',
                     'use_g_prompt',
                     'g_prompt_length',
                     'use_e_prompt',
                     'length',
                     'e_prompt_layer_idx',
                     'multi_classifier',
                     "head_type",
                     # 'my_train',
                     'checkpoint',
                     'use_prefix_tune_for_e_prompt',
                     'mean_idx',
                     'k_means'
                     ] # 指定要输出的参数名称
    # 写入文件
    if args.output_dir and utils.is_main_process(): # 主线程写入
        with open(os.path.join(args.output_dir, 'result.log'), 'a') as f:
            for k,v in args.__dict__.items(): # 将args转为字典
                if k in selected_args:  # 若key在输出列表中
                    f.write(f"{k}: {v}\n")
            f.write(f'number of params: {n_parameters}\n')
            f.write(f'datasets: {str(datasets_list)}\n')
            f.write('================================================\n\n')



    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)




    # 开始训练
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")





if __name__ == '__main__':

    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    
    # config = parser.parse_known_args()[-1][0]
    config = 'mydatasets_s_dualprompt'

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_dualprompt':
        from configs.cifar100_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 DualPrompt configs')
    elif config == 'imr_dualprompt':
        from configs.imr_dualprompt import get_args_parser
        config_parser = subparser.add_parser('imr_dualprompt', help='Split-ImageNet-R DualPrompt configs')
    elif config == 'mydatasets_s_dualprompt':
        from configs.mydatasets_s_dualprompt import get_args_parser
        config_parser = subparser.add_parser('mydatasets_s_dualprompt', help='My-Datasets DualPrompt configs')
    else:
        raise NotImplementedError
        
    get_args_parser(config_parser)

    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
    
    sys.exit(0)