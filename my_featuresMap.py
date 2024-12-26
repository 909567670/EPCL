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

import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')
# import matplotlib
# matplotlib.use('TkAgg')
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import umap.umap_ as umap

from tqdm import tqdm  # 导入tqdm库

def visualize_features_2d(data_loader, device, original_model, args, output_path, plot_by=None):
    if args.method == 'umap':
        reducer = umap.UMAP(n_components=2)  # 初始化UMAP，用于2D可视化
    elif args.method == 'tsne':
        reducer = TSNE(n_components=2)  # 初始化t-SNE，用于2D可视化
    else:
        reducer = PCA(n_components=2)  # 初始化PCA，用于2D可视化

    color_map = plt.colormaps.get_cmap('Set2')  # 获取颜色映射

    features_all = []
    labels_all = []
    task_labels = []

    # 使用tqdm添加进度条
    for task_id in tqdm(range(args.num_tasks), desc='Processing tasks'):  # 遍历任务
        fake_features = []
        real_features = []
        # 在数据加载循环中也添加进度条
        for i, (inputs, targets) in enumerate(tqdm(data_loader[task_id]['train'], desc=f'Task {task_id}')):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                if original_model is not None:
                    output = original_model(inputs)
                    feature = output.pre_logits

            ones_indices = torch.nonzero(targets, as_tuple=True)[0]
            zeros_indices = torch.nonzero(torch.logical_not(targets), as_tuple=True)[0]

            fake_features.append(feature[zeros_indices].cpu().numpy())
            real_features.append(feature[ones_indices].cpu().numpy())

        # 合并当前任务的所有特征
        fake_features_all = np.concatenate(fake_features, axis=0)
        real_features_all = np.concatenate(real_features, axis=0)

        # 添加到总列表
        features_all.append(fake_features_all)
        features_all.append(real_features_all)
        labels_all.extend([0] * len(fake_features_all))  # 0代表fake
        labels_all.extend([1] * len(real_features_all))  # 1代表real
        task_labels.extend([task_id] * (len(fake_features_all) + len(real_features_all)))

    # 合并所有特征并进行降维
    features_all = np.concatenate(features_all, axis=0)
    reduced_features_all = reducer.fit_transform(features_all)

    # 绘图
    s = 15  # 点的大小
    plt.figure(figsize=(10, 8))
    if plot_by == 'task':
        for task_id in range(args.num_tasks):
            task_indices = np.where(np.array(task_labels) == task_id)[0]
            task_features = reduced_features_all[task_indices]
            plt.scatter(task_features[:, 0], task_features[:, 1], marker='o', color=color_map(task_id), label=f'Task {task_id+1}',
                        s=s, )
    elif plot_by == 'label':
        real_indices = np.where(np.array(labels_all) == 1)[0]
        fake_indices = np.where(np.array(labels_all) == 0)[0]
        plt.scatter(reduced_features_all[real_indices, 0], reduced_features_all[real_indices, 1], marker='^', color='green', label='Real',
                    s=s, )
        plt.scatter(reduced_features_all[fake_indices, 0], reduced_features_all[fake_indices, 1], marker='o', color='red', label='Fake',
                    s=s, )
    else:
        # 原来的绘图方式
        for task_id in range(args.num_tasks):
            task_indices = np.where(np.array(task_labels) == task_id)[0]
            task_features = reduced_features_all[task_indices]
            task_labels_filtered = np.array(labels_all)[task_indices]
            plt.scatter(task_features[task_labels_filtered == 0, 0], task_features[task_labels_filtered == 0, 1], marker='o', color=color_map(task_id), label=f'Task {task_id+1} Fake',
                        s=s, )
            plt.scatter(task_features[task_labels_filtered == 1, 0], task_features[task_labels_filtered == 1, 1], marker='^', color=color_map(task_id), label=f'Task {task_id+1} Real',
                        s=s, )

    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.show()


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

    # if args.freeze:
    #     # all parameters are frozen for original vit model
    #     for p in original_model.parameters():
    #         p.requires_grad = False

    print(args)

    original_model.eval()

    # 如果提供了输出目录，则保存图形
    if args.output_dir:
        output_directory = Path(args.output_dir)
        output_directory.mkdir(parents=False, exist_ok=True)
        time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_directory / f"{time}_{args.method}_{datasets_list}.jpg"
    else:
        print('未提供保存路径!!')
        return 0

    visualize_features_2d(data_loader, device, original_model, args, output_path=output_path, plot_by=args.plot_mode)

    print(f"Features visualization saved to {output_path}")


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

    config_parser.add_argument('--method', type=str, default='pca', choices=['pca', 'umap', 'tsne'], help='降维方法')
    config_parser.add_argument('--plot_mode', type=str, default=None, help='绘图方式 task: 按任务绘制, label: 按标签绘制, None: 按任务标签绘制')

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)