import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import rcParams

# # 设置字体为 Noto Sans CJK JP 以支持中文显示
# rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
# rcParams['axes.unicode_minus'] = False

# 创建数据矩阵
data = [
    [96.691, 96.601, 96.656, 96.227, 96.296],
    [97.263, 97.001, 97.157, 97.349, 97.277],
    [98.520, 98.407, 98.417, 98.506, 98.413],
    [98.791, 98.672, 98.730, 98.667, 98.624],
    [98.775, 98.634, 98.674, 98.563, 98.673],
    [98.832, 98.852, 98.907, 98.761, 98.664],
    [98.991, 98.888, 98.931, 98.744, 98.819],
    [99.074, 98.821, 99.092, 98.709, 98.780],
    [99.239, 98.731, 99.133, 98.749, 98.767],
    [99.189, 98.711, 99.107, 98.721, 98.902],
    [99.091, 98.868, 99.055, 98.775, 98.802],
    [99.234, 98.802, 99.009, 98.777, 98.812],
    [99.201, 98.756, 98.985, 98.771, 98.882],
    [99.188, 98.756, 98.976, 98.787, 98.847],
    [99.262, 98.747, 99.002, 98.767, 98.859]
]

# 创建 DataFrame 并设置行列索引
df = pd.DataFrame(data, index=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], columns=[50, 40, 30, 20, 10])
# 绘制热力图
plt.figure(figsize=(9, 4), dpi=300,)
ax = sns.heatmap(df.T, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
# cbar = ax.collections[0].colorbar
# cbar.set_label("average_acc(%)", fontsize=15)  # 为颜色条添加标签
plt.xlabel('K',)
plt.ylabel('Lp',)

# plt.tight_layout()

plt.show()
plt.savefig("./klp_mat.jpg")