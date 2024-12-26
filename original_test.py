# @Time    : 2023/12/27 10:23
# @Author  : yxL
# @File    : original_test.py
# @Software: PyCharm
# @Description :

# @Time    : 2023/12/23 12:06
# @Author  : yxL
# @File    : my_featuresMap.py
# @Software: PyCharm
# @Description : 绘制所有数据集的特征图

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


    original_model.to(device)


    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False


    print(args)

    args.multi_classifier = False

    original_model.eval()
    for task_id in range(args.num_tasks):
        # for i in range(args.num_tasks if args.eval_all_tasks else task_id+1):  # eval_all_task 判断是否 每个task只测试已学过的数据集
        eval_loader = data_loader[task_id]['val']
        criterion = torch.nn.CrossEntropyLoss()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(task_id + 1)

        with torch.no_grad():
            for input, target in metric_logger.log_every(eval_loader, args.print_freq, header):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output = original_model(input)
                logits=output.logits
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        result_str = '[{task_id}]* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}' \
            .format(task_id=task_id, top1=metric_logger.meters['Acc@1'], losses=metric_logger.meters['Loss'])

        print(result_str)

    return 0
if __name__ == '__main__':

    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')

    config = parser.parse_known_args()[-1][0]

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