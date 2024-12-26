# @Time    : 2023/12/15 15:03
# @Author  : yxL
# @File    : test.py
# @Software: PyCharm
# @Description :   导出onnx模型


import torch
import torch.utils.data
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import my_dataset
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

# 显式写出才 导入本地的models模块 而不是库中同名的models模块
import models
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# 读取 pth 文件
checkpoint = torch.load(r'output/a_s_dual_MC_100/task6_checkpoint.pth')
# print(checkpoint['epoch'])
# print(checkpoint['args'])
# print(checkpoint['lr_scheduler'])
for layer_name, parameters in checkpoint.items():
    print(f"Layer: {layer_name}")
    # print(f"Parameters: {parameters}")
args = checkpoint['args']

# 导出聚类中心为npy文件 @已将all_keys并入onnx中
# all_keys_np = [key.cpu().numpy() for key in args.all_keys]  # 将每个 PyTorch 张量转换为 NumPy 数组
# all_keys_np = np.array(all_keys_np)  # 将列表转换为 NumPy 数组
# np.save('output/all_keys.npy', all_keys_np)

original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        all_keys=args.all_keys, # 所有任务的key
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
    num_tasks=args.num_tasks, # 任务数量

)

# last = model.e_prompt.prompt.clone()


model.load_state_dict(checkpoint['model'])
model.eval()
original_model.eval()


# new = model.e_prompt.prompt
d=2
device= 'cuda'

# torch.manual_seed(42)
img = torch.rand(1, 3, 224, 224)
sid = torch.tensor([d],dtype=torch.int32)

'''
    导出onnx模型 SDP
'''
#
# if 0:
#     model.to(device)
#     dataset = my_dataset.myDataset(rf'/media/dataset/myDataset/mya{d+1}', train=False)
#     dataloaders = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
#
#     pbar = tqdm(enumerate(dataloaders), total=len(dataloaders))
#
#     right = 0
#     model.eval()
#     with torch.no_grad():
#         for idx, (inputs, labels) in pbar:
#             res = model(inputs.to(device), sid.to(device))
#             r = torch.argmax(res.logits, dim=1)
#             if torch.equal(r, labels.to(device)):
#                 right += 1
#             pbar.set_postfix({'accuracy': f'{right / len(dataloaders):.5f}'})
# else:
#     # model = torch.jit.script(model)
#
#     model.eval()
#     with torch.no_grad():
#         res = model(img, sid)
#

# model.eval()
# with torch.no_grad():
#     res = model(img, sid)

x = (img, sid)
traced_model = torch.jit.trace(model, x)
#
#
torch.onnx.export(traced_model,  # 模型
                  x,  # 伪输入 提供模型输入的形状
                  r"output/sdp_17.onnx",  # 保存的文件名
                  export_params=True,  # 是否保存模型参数
                  opset_version=17,  # onnx版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input','s_id'],  # 输入名称
                  output_names=['logits','pre_logits'],  # 输出名称
                  # dynamic_axes={'input': {0: 'batch_size'},  # 指定动态轴 dynamic_axes 参数的格式为一个字典，
                  #               's_id': {0: 'batch_size'},
                  #               'logits': {0: 'batch_size'},  # 其中键是输入或输出张量的名称，值是另一个字典，指定了动态轴的索引及其对应的名称。
                  #               }  # 其中键是输入或输出张量的名称，值是另一个字典，指定了动态轴的索引及其对应的名称。
                  # verbose=True,               # 是否打印详细信息
                  )



'''
导出 onnx 模型 Vit
'''
import onnxruntime

# dataset = my_dataset.myDataset(rf'/media/dataset/myDataset/mya{d+1}', train=False)
# dataloaders = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
#
# pbar = tqdm(enumerate(dataloaders), total=len(dataloaders))
#
# for idx, (inputs, labels) in pbar:
#     with torch.no_grad():
#         res = original_model(inputs)
#
#
#     # 加载模型
#     sess = onnxruntime.InferenceSession(r'output/vit_sid.onnx')
#     # 设置会话配置
#     sess.set_providers(['CUDAExecutionProvider'])
#
#     input = sess.get_inputs()[0].name
#     result = sess.run(["logits","pre_logits",'s_idx','l1_mat'], {input: inputs.cpu().numpy()})
#
#     print(np.abs(result[3]-res.l1_mat.cpu().numpy()).max())


torch.onnx.export(original_model,  # 模型
                  img,  # 伪输入 提供模型输入的形状
                  r"output/vit_sid.onnx",  # 保存的文件名
                  export_params=True,  # 是否保存模型参数
                  opset_version=17,  # onnx版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input'],  # 输入名称
                  output_names=['logits','pre_logits','s_id','l1_mat'],  # 输出名称
                  # dynamic_axes={'input': {0: 'batch_size'},  # 指定动态轴 dynamic_axes 参数的格式为一个字典，
                  #               's_id': {0: 'batch_size'},
                  #               'logits': {0: 'batch_size'},  # 其中键是输入或输出张量的名称，值是另一个字典，指定了动态轴的索引及其对应的名称。
                  #               }  # 其中键是输入或输出张量的名称，值是另一个字典，指定了动态轴的索引及其对应的名称。
                  # verbose=True,               # 是否打印详细信息
                  )