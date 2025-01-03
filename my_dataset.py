# @Time    : 2023/10/10 17:02
# @Author  : yxL
# @File    : my_dataset.py
# @Software: PyCharm
# @Description : 实现密集架数据集
#               数据集文件结构要求：(与标签获取相关)
#                   数据集根目录
#                       ├─  train
#                       │     ├─无异物
#                       │     │     x.bmp
#                       │     └─有异物
#                       │           x.bmp
#                       └─  val
#                             ├─无异物
#                             │     x.bmp
#                             └─有异物
#                                   x.bmp


import torch
import torch.utils.data
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt

# from datasets import build_transform

# labelsMap = {0: "无异物", 1: "有异物"}
from torchvision.datasets.vision import VisionDataset


class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, train=True, transform=None, target_transform=None, need_sub=True):
        """
        初始化数据集
        dataPath: 数据集路径
        train: 是否为训练集
        transform: 数据集变换
        target_transform: 标签变换
        need_sub(bool): 是否需要 train/val 文件夹(区分训练集与验证集) 若为False则直接读取dataPath下的图片
        """
        # super(myDataset, self).__init__(dataPath, transform=transform, target_transform=target_transform)
        self.train = train  # 是否为训练集
        self.need_sub = need_sub  # 是否需要 train/val 文件夹 (评估模型效果时 可以不使用子文件夹)
        self.transform = transform  # 数据集变换
        self.target_transform = target_transform  # 标签变换
        self.dataPath = self.getFiles(dataPath)  # 数据集路径
        self.classes = ["fake", "real"]  # fake: 无异物 real: 有异物

    def __getitem__(self, index):
        imgPath = self.dataPath[index]  # 图片路径
        img = Image.open(imgPath)  # 读取图片
        label = self.classes.index(imgPath.split(os.path.sep)[-2])  # 获取标签

        # 数据集变换 (数据增强)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.dataPath)

    def getFiles(self, dataPath, need_suf='.jpg'):  # 查找根目录，文件后缀
        res = []
        if not os.path.exists(dataPath):  # 判断路径是否存在
            raise Exception(f'{dataPath} 数据集文件夹不存在')

        if self.need_sub:
            # 判断路径下是否存在train和val文件夹
            if not os.path.exists(os.path.join(dataPath, 'train')) or not os.path.exists(os.path.join(dataPath, 'val')):
                raise Exception(f'{dataPath}下, 数据集文件结构错误 要包含train和val文件夹')

            path = os.path.join(dataPath, 'train' if self.train else 'val')
        else:
            path = dataPath

        for root, directory, files in os.walk(path):  # =>当前根,根下目录,目录下的文件
            for filename in files:
                name, suf = os.path.splitext(filename)  # =>文件名,文件后缀
                if suf == need_suf or suf == '.png' or suf == '.bmp' or suf == '.jpg':
                    res.append(os.path.join(root, filename))  # =>吧一串字符串组合成路径
        # res = sorted(res) # 排序 (无效)
        return res


def my_build_transform(resize=False):
    t = []
    if resize:
        t.append(transforms.Resize((224, 224)))

    t.extend([
        # 依概率p水平翻转  依概率p垂直翻转
        transforms.RandomHorizontalFlip(p=0.5),  # p表示概率
        transforms.RandomVerticalFlip(p=0.2),  # p表示概率
        transforms.RandomRotation(10),  # 随机旋转 10度
        # 色度、亮度、饱和度、对比度的变化,数值为0-1之间
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.3, hue=0.2)], p=0.5),
        # 图像噪声 todo

        transforms.RandomGrayscale(p=0.2),  #进行随机的灰度化
        transforms.ToTensor(),
        # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor /255.操作
    ])
    return transforms.Compose(t)


if __name__ == '__main__':
    class MyArgs:
        def __init__(self):
            self.batch_size = 32
            self.learning_rate = 0.001
            self.model_name = "my_model"
            self.input_size = 224


    args = MyArgs()
    args.nb_classes = 20

    data = myDataset(r'C:\Users\Lenovo\Desktop\数据集', train=True)
    print('数据集大小：', len(data))
    # for img, label in data:
    #     print(img, label)
    #
    # print(data[0])
    # 获取数据集
    # dataset_train = data_loader[0]['train'].dataset # 训练集

    # 显示数据集
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()
