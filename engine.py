# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np
from sklearn.cluster import KMeans

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import utils
import re


def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None, ):
    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)  # 多卡训练时，设置epoch，保证每个epoch的数据不同

    metric_logger = utils.MetricLogger(delimiter="  ")  # 用于记录训练过程中的指标
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.9f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

    # t=-1
    # prompt_param=[]

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # t+=1
        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output.pre_logits
            else:
                cls_features = None
        s_idx, _ = get_sprompt_idx(task_id, True, cls_features, args)
        output = model(input, s_idx=s_idx, train=set_training_mode)
        logits = output.logits

        # here is the trick to mask out classes of non-current tasks 下面是屏蔽非当前任务类的技巧
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]  # mask存放当前任务类的logits索引
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)  # not_mask存放非当前任务类的logits索引
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))  # 将非当前任务类的logits置为负无穷 便于softmax后概率为0

        loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # 梯度裁剪
        optimizer.step()  # 更新参数

        # ↓ 已证实 未使用的提示不会被更新
        # if t>0:
        #     print(f'train s_idx: {s_idx[0]}')
        #     for i in range(args.size):
        #         element_sum = torch.sum(prompt_param[i] - model.e_prompt.prompt.data[:,:,i])
        #         print(f'prompt[{i}]',element_sum)
        #     prompt_param=[]
        #
        # for i in range(args.size):
        #     prompt_param.append(model.e_prompt.prompt.data[:,:,i].clone())

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        # metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()  # 多卡训练时，同步指标
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
             device, task_id=-1, class_mask=None, args=None, ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output.pre_logits
            else:
                cls_features = None

            s_idx, batch_select_acc = get_sprompt_idx(task_id, False, cls_features, args,)
            output = model(input, s_idx=s_idx, train=False)
            logits = output.logits

            if args.task_inc and class_mask is not None:
                #adding mask to output logits 添加掩码到输出logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            # metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            metric_logger.meters['Select_Acc'].update(batch_select_acc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    result_str = '* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} Select_Acc {select_acc.global_avg:.3f}' \
        .format(top1=metric_logger.meters['Acc@1'], losses=metric_logger.meters['Loss'],
                select_acc=metric_logger.meters['Select_Acc'])

    print(result_str)

    # 将结果写入文件
    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, 'eval_result.log' if args.eval else 'result.log'), 'a') as f:
            f.write(f'[task {task_id + 1}]' + result_str + '\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, acc_matrix=None, args=None, ):
    stat_matrix = np.zeros((2, args.num_tasks))  # 2 for Acc@1, Loss

    for i in range(args.num_tasks if args.eval_all_tasks else task_id + 1):  # eval_all_task 判断是否 每个task只测试已学过的数据集
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                              device=device, task_id=i, class_mask=class_mask, args=args)

        if i < task_id + 1 or args.my_train:  # 保证 eval_all_task 时 只记录已学过(i<task_id+1)的准确率信息  自定义训练时 记录所有数据集的准确率信息
            stat_matrix[0, i] = test_stats['Acc@1']
            # stat_matrix[1, i] = test_stats['Acc@5']
            stat_matrix[1, i] = test_stats['Loss']

            acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1 if not args.my_train else args.num_tasks)  # 计算平均准确率

    diagonal = np.diag(acc_matrix)

    result_str = f"[Average accuracy till task{task_id + 1}]\tAcc@1: {avg_stat[0]:.4f}\tLoss: {avg_stat[1]:.4f}\tSelect_Acc: {test_stats['Select_Acc']:.3f}"
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    # 将结果写入文件
    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, 'eval_result.log' if args.eval else 'result.log'), 'a') as f:
            f.write(result_str + '\n\n')

    return test_stats


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module,
                       criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler,
                       device: torch.device,
                       class_mask=None, args=None, ):
    # 自定义训练模式下 只会执行1次 训练与评估
    # 若是自定义训练 则只训练一个任务
    num_tasks = 1 if args.my_train else args.num_tasks

    # create matrix to save end-of-task accuracies  创建矩阵以保存任务结束时的准确性
    acc_matrix = np.zeros((args.num_tasks, num_tasks))  # 数据集数量 * 训练任务
    loss_list = []  # 保存每个任务的损失
    acc_list = []  # 保存每个任务的准确率
    # select_acc_list = []  # 保存每个任务的选择准确率

    for task_id in range(num_tasks):
        # # Transfer previous learned prompt params to the new prompt 转移以前学习的提示参数到新的提示
        #  if args.prompt_pool and args.shared_prompt_pool:
        #      if task_id > 0:
        #          prev_start = (task_id - 1) * args.top_k
        #          prev_end = task_id * args.top_k
        #
        #          cur_start = prev_end
        #          cur_end = (task_id + 1) * args.top_k
        #
        #          if (prev_end > args.size) or (cur_end > args.size):
        #              pass
        #          else:
        #              cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
        #              prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))
        #
        #              with torch.no_grad(): # 无梯度
        #                  if args.distributed:
        #                      model.module.e_prompt.prompt.grad.zero_()
        #                      model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
        #                      optimizer.param_groups[0]['params'] = model.module.parameters()
        #                  else:
        #                      model.e_prompt.prompt.grad.zero_()
        #                      model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
        #                      optimizer.param_groups[0]['params'] = model.parameters()
        #
        #  # Transfer previous learned prompt param keys to the new prompt 将以前学习的提示符参数键转移到新的提示符上
        #  if args.prompt_pool and args.shared_prompt_key:
        #      if task_id > 0:
        #          prev_start = (task_id - 1) * args.top_k
        #          prev_end = task_id * args.top_k
        #
        #          cur_start = prev_end
        #          cur_end = (task_id + 1) * args.top_k
        #
        #          with torch.no_grad():
        #              if args.distributed:
        #                  model.module.e_prompt.prompt_key.grad.zero_()
        #                  model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
        #                  optimizer.param_groups[0]['params'] = model.module.parameters()
        #              else:
        #                  model.e_prompt.prompt_key.grad.zero_()
        #                  model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
        #                  optimizer.param_groups[0]['params'] = model.parameters()

        # Create new optimizer for each task to clear optimizer status 为每个任务创建新的优化器，清除优化器状态
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)

        # 学习率调度器
        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizer)
        elif args.sched == 'constant':
            lr_scheduler = None

        # 冻结相应参数
        if args.multi_classifier:
            if args.distributed:
                # 冻结所有分类头
                model.module.freeze_all_classifier()
                # 解冻当前任务的分类头
                model.module.unfreeze_selected_classifier(task_id)
                model.module.print_updated_params()
            else:
                model.freeze_all_classifier()
                model.unfreeze_selected_classifier(task_id)
                model.print_updated_params()

        # 聚类 使用无提示池的模型提取特征 (训练前后聚类都行)
        print(f'start clustering for task:{task_id} ({args.datasets_list[task_id]})!')
        if args.diff_clustering is not None and args.diff_clustering:
            diff_clustering(original_model, data_loader[task_id]['train'], args=args)  # 保存聚类中心 有异物和无异物聚类都得到k个中心
        else:
            clustering(original_model, data_loader[task_id]['train'], args=args)  # 保存聚类中心 每次聚类都得到5个中心

        task_loss = []  # 当前任务训练的损失
        task_acc = []  # 当前任务训练的准确率
        # 训练模型
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                          data_loader=data_loader[task_id if not args.my_train else -1]['train'],
                                          optimizer=optimizer,
                                          device=device, epoch=epoch, max_norm=args.clip_grad,
                                          set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args, )
            if lr_scheduler:
                lr_scheduler.step(epoch)

            task_loss.append(train_stats['Loss'])
            task_acc.append(train_stats['Acc@1'])

        loss_list.append(task_loss)
        acc_list.append(task_acc)
        # 测试模型
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)

        # 保存检查点
        if args.output_dir and utils.is_main_process() and not args.no_checkpoints:
            #Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir,
                                           'task{}_checkpoint.pth'.format(
                                               task_id + 1 if not args.my_train else get_latest_checkpoint(args) + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and utils.is_main_process() and not args.no_log:
            with open(os.path.join(args.output_dir,
                                   '{}_stats.log'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))),
                      'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    # 绘制损失曲线 和 准确率曲线
    if args.output_dir and utils.is_main_process():
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(len(loss_list)):
            plt.plot(loss_list[i], label=f'task {i + 1}({args.datasets_list[i]})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'loss.png'), dpi=500)

        plt.figure()
        for i in range(len(acc_list)):
            plt.plot(acc_list[i], label=f'task {i + 1}({args.datasets_list[i]})')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'acc.png'), dpi=500)


# 获取output_dir下的最新检查点的编号 int
def get_latest_checkpoint(args) -> int:
    cp_num = 0
    if args.output_dir and utils.is_main_process():
        checkpoint_dir = os.path.join(args.output_dir)
        if os.path.exists(checkpoint_dir):
            # 正则表达式匹配出所有检查点的编号
            checkpoint_list = [int(re.search(r'\d+', f).group()) for f in os.listdir(checkpoint_dir) if
                               f.endswith('.pth')]
            cp_num = max(checkpoint_list) if len(checkpoint_list) > 0 else 0
    return cp_num


# 获取 sprompt 索引
def get_sprompt_idx(task_id, train=False, cls_features=None, args=None,):
    idx = []
    batch_select_acc = 1.0
    if train:
        idx = torch.tensor(task_id).unsqueeze(0).expand(cls_features.shape[0])  # [B]
    else:

        if args.diff_clustering is not None and args.diff_clustering:
            # 计算cls_features与diff_all_keys的L1距离
            # batch_l1 = args.diff_all_keys - cls_features
            # 扩展 cls_features 的形状为 (batch, 1, 1, 1, 768)
            cls_features_expanded = cls_features.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            # 扩展 args.diff_all_keys 的形状为 (1, task_num, 2, k, 768)
            diff_all_keys_expanded = args.diff_all_keys.unsqueeze(0)
            # 执行减法操作 (batch, task_num, 2, k, 768)
            batch_l1 = diff_all_keys_expanded - cls_features_expanded
            # l1 距离
            batch_l1 = torch.abs(batch_l1).sum(dim=-1)  # (batch, task_num, 2, k)

            # 取平均最小或者直接取最小
            if args.mean_idx:
                # 取平均值  效果并不好
                mean_k = batch_l1.mean(dim=-1) # (batch, task_num, 2)
                mean_type = mean_k.min(dim=-1)[0]  # (batch, task_num)
                idx = mean_type.min(dim=-1)[1]  # (batch)
            else:
                # 取最小值
                min_k = batch_l1.min(dim=-1)[0]  # (batch, task_num, 2)
                min_type = min_k.min(dim=-1)[0]  # (batch, task_num)
                idx = min_type.min(dim=-1)[1]  # (batch)

        else:
            # 计算cls_features与all_keys的L1距离
            features = cls_features  # [B,C]
            taskselection = []
            for task_centers in args.all_keys:
                tmpcentersbatch = []
                for center in task_centers:
                    tmpcentersbatch.append((((features - center) ** 2) ** 0.5).sum(1))  # [k,bs] 计算样本到5个聚类中心的L1距离
                taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])  # [task_num,bs] 一批数据的每个样本到每次任务的5个聚类中心的最小值

            idx = torch.vstack(taskselection).min(0)[1]  # [B] 一批的每个样本的任务索引 即确定了输入图片是哪个任务的

        batch_select_acc = (idx == task_id).sum().item() / idx.shape[0]
        # print(f'task:{task_id}一批数据选中率:', batch_select_acc)

    # 2025/03/31 优化:将idx移动至device
    idx = idx.to(torch.device(args.device))
    return idx, batch_select_acc


# 数据集聚类(不区分 有无异物) model提取特征 使用original_model
def clustering(original_model, data_loader, args=None):
    if not hasattr(args, 'all_keys'):
        args.all_keys = []  # 存储聚类中心

    original_model.eval()
    device = torch.device(args.device)
    features = []
    for i, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            if original_model is not None:
                output = original_model(inputs)
                feature = output.pre_logits  # 提取特征 (B,C)

        # feature = feature / feature.norm(dim=-1, keepdim=True) # 在extract_vector中已经归一化
        features.append(feature)
    features = torch.cat(features, 0).cpu().detach().numpy()
    clustering = KMeans(n_clusters=5, random_state=0).fit(features)
    args.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))  # 保存聚类中心 每次聚类都得到5个中心
    # all_keys [task_num,5,768] 使用list存储 每个任务的聚类中心tensor [5,768]


# # 聚类(区分 有无异物) model提取特征 使用model
# def diff_clustering(original_model, data_loader, args=None):
#     original_model.eval()
#     device = torch.device(args.device)

#     k = 3  # 聚类中心数量
#     if hasattr(args,'k_means'):
#         k = args.k_means


#     if not hasattr(args, 'diff_all_keys'):
#         # 4 维 空tensor
#         args.diff_all_keys = torch.empty((0, args.nb_classes, k, 768)).to(device)  # 存储聚类中心 [task_num, fake/real, k, 768]

#     real_features = []  # 有异物特征
#     fake_features = []  # 无异物特征
#     for i, (inputs, labels) in enumerate(data_loader):
#         inputs, labels = inputs.to(device), labels.to(device)
#         with torch.no_grad():
#             if original_model is not None:
#                 output = original_model(inputs)
#                 feature = output.pre_logits  # 提取特征 (B,C)

#         # feature = feature / feature.norm(dim=-1, keepdim=True) # 在extract_vector中已经归一化

#         # 找到所有1的索引
#         ones_indices = torch.nonzero(labels, as_tuple=True)[0]
#         # print("Indices of ones:", ones_indices)
#         # 找到所有0的索引
#         zeros_indices = torch.nonzero(torch.logical_not(labels), as_tuple=True)[0]
#         # print("Indices of zeros:", zeros_indices)

#         fake_features.append(feature[zeros_indices])
#         real_features.append(feature[ones_indices])

#     fake_features = torch.cat(fake_features, 0).cpu().detach().numpy()
#     fake_clustering = KMeans(n_clusters=k, random_state=0).fit(fake_features)
#     real_features = torch.cat(real_features, 0).cpu().detach().numpy()
#     real_clustering = KMeans(n_clusters=k, random_state=0).fit(real_features)

#     # 保存聚类中心 有异物和无异物聚类都得到k个中心
#     # args.diff_all_keys.append(torch.tensor(fake_clustering.cluster_centers_).to(feature.device))  # 无异物聚类中心
#     # args.diff_all_keys.append(torch.tensor(real_clustering.cluster_centers_).to(feature.device))  # 有异物聚类中心
#     center_tensor = torch.stack((torch.tensor(fake_clustering.cluster_centers_).to(feature.device),
#                                  torch.tensor(real_clustering.cluster_centers_).to(feature.device)))
#     # 在第0维的位置添加一个新的维度
#     center_tensor = center_tensor.unsqueeze(0)

#     args.diff_all_keys = torch.cat((args.diff_all_keys, center_tensor), 0)  # 保存聚类中心 有异物和无异物聚类都得到k个中心

# 2024/10/8 : 
# 更新 适配多类别数据集不仅仅是2分类
def diff_clustering(original_model, data_loader, args=None):
    original_model.eval()
    device = torch.device(args.device)

    k = 3  # 聚类中心数量
    if hasattr(args, 'k_means'):
        k = args.k_means

    if not hasattr(args, 'diff_all_keys'):
        # 4 维 空tensor
        args.diff_all_keys = torch.empty((0, args.nb_classes, k, 768)).to(device)  # 存储聚类中心 [task_num, nb_classes, k, 768]

    # 初始化一个包含 nb_classes 个列表的列表来存储每个类别的特征
    features_list = [[] for _ in range(args.nb_classes)]

    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            if original_model is not None:
                output = original_model(inputs)
                feature = output.pre_logits  # 提取特征 (B,C)

        # 将特征按类别存储
        for class_idx in range(args.nb_classes):
            class_indices = torch.nonzero(labels == class_idx, as_tuple=True)[0]
            features_list[class_idx].append(feature[class_indices])

    # 对每个类别进行聚类
    cluster_centers = []
    for class_idx in range(args.nb_classes):

        if not features_list[class_idx]: # 若有类别没有特征
            raise Exception(f"class:{class_idx} has no feature clustering!")
        
        class_features = torch.cat(features_list[class_idx], 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=k, random_state=0).fit(class_features)
        cluster_centers.append(torch.tensor(clustering.cluster_centers_).to(device))

    # 将聚类中心堆叠成一个张量
    center_tensor = torch.stack(cluster_centers).unsqueeze(0)

    # 更新 args.diff_all_keys
    args.diff_all_keys = torch.cat((args.diff_all_keys, center_tensor), 0)  # 保存聚类中心



# 先完成所有数据集的聚类 (仅在测试k对模型影响时使用)
def pre_diff_clustering(original_model, data_loader, args=None):
    # 清空 diff_all_keys
    args.diff_all_keys = torch.empty((0, args.nb_classes, args.k_means, 768)).to(args.device)  # 存储聚类中心 [task_num, fake/real, k, 768]
    
    for i in range(len(data_loader)):
        diff_clustering(original_model,data_loader[i]['train'],args)