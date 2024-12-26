import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Data
l2p = {"平均准确率": [0.9724, 0.7797, 0.8574, 0.7213, 0.7440],
       "平均遗忘率": [0, 0.4020, 0.1935, 0.3490, 0.2949]}

dual_p = {"平均准确率": [0.9975, 0.9481, 0.9702, 0.9512, 0.8997],
          "平均遗忘率": [0, 0.0917, 0.0368, 0.0574, 0.1127]}

sdp = {"平均准确率": [1., 1., 0.9978, 0.9869, 0.9885],
       "平均遗忘率": [0, 0., 0., 0.0050, 0.0030]}

# Create a figure with two subplots side by side with a larger space in between
fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300, figsize=(14, 4))
fig.subplots_adjust(wspace=0.4)  # Increase the space between the plots

# Plot 平均准确率
ax1.plot(range(1, len(sdp["平均准确率"]) + 1), sdp["平均准确率"], label='Ours', linestyle='-', marker='o')
ax1.plot(range(1, len(l2p["平均准确率"]) + 1), l2p["平均准确率"], label='L2P', linestyle='-.', marker='^')
ax1.plot(range(1, len(dual_p["平均准确率"]) + 1), dual_p["平均准确率"], label='DualPrompt', linestyle='--', marker='s')

ax1.set_xlabel('task')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.set_xticks(range(1, len(l2p["平均准确率"]) + 1))
ax1.grid(True)  # 添加网格线

# Plot 平均遗忘率
ax2.plot(range(1, len(sdp["平均遗忘率"]) + 1), sdp["平均遗忘率"], label='Ours', linestyle='-', marker='o')
ax2.plot(range(1, len(l2p["平均遗忘率"]) + 1), l2p["平均遗忘率"], label='L2P', linestyle='-.', marker='^')
ax2.plot(range(1, len(dual_p["平均遗忘率"]) + 1), dual_p["平均遗忘率"], label='DualPrompt', linestyle='--', marker='s')

ax2.set_xlabel('task')
ax2.set_ylabel('Forgetting')
ax2.legend()
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
# y 轴范围0~30%
# ax2.set_ylim([0, 0.20])
ax2.set_xticks(range(1, len(l2p["平均遗忘率"]) + 1))
ax2.grid(True)  # 添加网格线

# 调整布局
fig.tight_layout()

# 保存图像为 test.jpg，DPI 设置为 300
plt.savefig("./test.jpg", dpi=300)

plt.show()