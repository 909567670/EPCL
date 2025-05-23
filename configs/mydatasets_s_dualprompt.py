# @Time    : 2023/10/13 15:20
# @Author  : yxL
# @File    : mydatasets_dualprompt.py
# @Software: PyCharm
# @Description : 自定义数据集参数配置

import argparse

def get_args_parser(subparsers):
    subparsers.add_argument('--batch_size', default=24, type=int, help='Batch size per device')
    subparsers.add_argument('--epochs', default=5, type=int)

    # Model parameters
    subparsers.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    subparsers.add_argument('--input-size', default=224, type=int, help='images input size')
    subparsers.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    subparsers.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    subparsers.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    subparsers.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    subparsers.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    subparsers.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    subparsers.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    subparsers.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    subparsers.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    subparsers.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    subparsers.add_argument('--sched', default='step', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    subparsers.add_argument('--lr', type=float, default=0.03, metavar='LR', help='learning rate (default: 0.03)')
    subparsers.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    subparsers.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    subparsers.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    subparsers.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    subparsers.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    subparsers.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    subparsers.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    subparsers.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    subparsers.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    subparsers.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    subparsers.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

    # Augmentation parameters
    subparsers.add_argument('--color-jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (default: 0.3)')
    subparsers.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    subparsers.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    subparsers.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    subparsers.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.25)')
    subparsers.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    subparsers.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    subparsers.add_argument('--data_path', default='/local_datasets/', type=str, help='dataset path')
    subparsers.add_argument('--dataset', default='mydatasets', type=str, help='dataset name')
    subparsers.add_argument('--shuffle', default=False, help='shuffle the data order') # 打乱数据集顺序 按照numpy.random
    subparsers.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    subparsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparsers.add_argument('--seed', default=42, type=int)
    subparsers.add_argument('--eval', action='store_true', help='Perform evaluation only')
    subparsers.add_argument('--num_workers', default=4, type=int)
    subparsers.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    subparsers.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    subparsers.set_defaults(pin_mem=True)

    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    subparsers.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Continual learning parameters
    subparsers.add_argument('--num_tasks', default=5, type=int, help='number of sequential tasks')
    subparsers.add_argument('--train_mask', default=False, type=bool, help='if using the class mask at training') #掩码
    subparsers.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')

    # G-Prompt parameters
    subparsers.add_argument('--use_g_prompt', default=False, type=bool, help='if using G-Prompt')
    subparsers.add_argument('--g_prompt_length', default=5, type=int, help='length of G-Prompt') # G-Prompt的长度
    subparsers.add_argument('--g_prompt_layer_idx', default=None, type=int, nargs = "+", help='the layer index of the G-Prompt') # 2个G-Prompt插入的层 1~2 [0,1]
    subparsers.add_argument('--use_prefix_tune_for_g_prompt', default=True, type=bool, help='if using the prefix tune for G-Prompt')


    # E-Prompt parameters
    subparsers.add_argument('--use_e_prompt', default=True, type=bool, help='if using the E-Prompt')
    subparsers.add_argument('--e_prompt_layer_idx', default=[2, 3, 4], type=int, nargs = "+", help='the layer index of the E-Prompt') # E-Prompt插入的层 3~5
    subparsers.add_argument('--use_prefix_tune_for_e_prompt', default=True, type=bool, help='if using the prefix tune for E-Prompt')

    # Use prompt pool in L2P to implement E-Prompt 
    subparsers.add_argument('--prompt_pool', default=True, type=bool,)
    subparsers.add_argument('--size', default=10, type=int,)    # ※该参数无需设置 提示池大小 在s-dual中 = 任务数量  
    subparsers.add_argument('--length', default=5,type=int, )   # 每个提示的token长度 其实际提示长度为length2倍 (pk+pv)
    subparsers.add_argument('--top_k', default=6, type=int, )   # ※该参数无需设置 仅作用于L2P和DualPrompt 选择前topk个提示用于PT
    subparsers.add_argument('--initializer', default='uniform', type=str,)
    subparsers.add_argument('--prompt_key', default=True, type=bool,)
    subparsers.add_argument('--prompt_key_init', default='uniform', type=str)
    subparsers.add_argument('--use_prompt_mask', default=True, type=bool)
    subparsers.add_argument('--mask_first_epoch', default=False, type=bool)
    subparsers.add_argument('--shared_prompt_pool', default=True, type=bool)
    subparsers.add_argument('--shared_prompt_key', default=False, type=bool)
    subparsers.add_argument('--batchwise_prompt', default=True, type=bool)
    subparsers.add_argument('--embedding_key', default='cls', type=str)
    subparsers.add_argument('--predefined_key', default='', type=str)
    subparsers.add_argument('--pull_constraint', default=True)
    subparsers.add_argument('--pull_constraint_coeff', default=1.0, type=float)
    subparsers.add_argument('--same_key_value', default=False, type=bool)

    # ViT parameters
    subparsers.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
    subparsers.add_argument('--head_type', default='token', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
    subparsers.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

    # Misc parameters
    subparsers.add_argument('--print_freq', type=int, default=10, help = 'The frequency of printing')

    # my para
    subparsers.add_argument('--no_log',action='store_true',help = 'not gen log txt file') # 不生成L2P 原生log文件, 已经实现了新的log生成
    subparsers.add_argument('--no_checkpoints',action='store_true',help = 'not gen checkpoints pth file') # 不生成checkpoints文件
    subparsers.add_argument('--eval_all_tasks',action='store_true',help = 'each task stage eval all tasks') # 每个任务阶段都评估所有任务

    subparsers.add_argument('--datasets_list',type=str,nargs="*",help="训练的数据集列表") # *代表0个或多个参数

    subparsers.add_argument('--my_train',action='store_true',help = '自定义训练模式') # !暂未实现

    subparsers.add_argument('--load_checkpoint',type=str,help = '指定checkpoint用于训练/评估, 存放于output_dir下')
    
    subparsers.add_argument('--multi_classifier',type=bool, default=True, help = '每个任务单独使用分类头') # 默认使用 多分类头

    subparsers.add_argument('--diff_clustering',type=bool, default=True, help = '有异物,无异物 单独聚类') # 默认使用 单独聚类
    subparsers.add_argument('--mean_idx',type=bool, default=False, help = '使用均值计算最近聚类中心, 否则使用最小值') 
    subparsers.add_argument('--k_means',type=int, default=3, help = 'K均值聚类 默认3聚类')
