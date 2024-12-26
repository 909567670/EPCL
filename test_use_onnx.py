# @Time    : 2024/3/26 13:42
# @Author  : yxL
# @File    : test_use_onnx.py
# @Software: PyCharm
# @Description :
import os

import numpy as np
import onnxruntime
import torch.utils.data
from tqdm import tqdm

import my_dataset
from torchvision import transforms
import PIL.Image

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 加载模型
sess = onnxruntime.InferenceSession(r'output/sdp_17.onnx')
# 设置会话配置
sess.set_providers(['CUDAExecutionProvider'])

# 获取输入和输出的名称
input = sess.get_inputs()[0].name
sid = sess.get_inputs()[1].name


# # 假设我们有一个输入数据x
x = PIL.Image.open(r'E:/师兄代码/已归档数据集/myDataset/mya1/train/fake/1100.bmp')
transform = transforms.ToTensor()
tensor_image = transform(x)
x = tensor_image.unsqueeze(0).numpy()


result = sess.run(["logits","pre_logits"], {input: x, sid: np.array([0],dtype=np.int32)})


d = 1
dataset = my_dataset.myDataset(rf'/media/dataset/myDataset/mya{d+1}', train=False)
dataloaders = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
print("开始运行")
right = 0

# 创建一个tqdm对象
pbar = tqdm(enumerate(dataloaders), total=len(dataloaders))

for idx, (inputs, labels) in pbar:
    x = inputs.numpy()
    # 运行模型
    result = sess.run(["logits","pre_logits"], {input: x, sid: np.array([d])})
    res = np.argmax(result[0], axis=1)

    if res == labels.numpy():
        right += 1

    # 更新进度条的后缀
    pbar.set_postfix({'accuracy': f'{right/len(dataloaders):.5f}'})

print("正确率:",right/len(dataloaders))