import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Data
l2p = {"平均准确率": [0.9295, 0.8870, 0.8213, 0.8284, 0.8072],
       "平均遗忘率": [0, 0.1423, 0.1706, 0.1255, 0.1596]}

dual_p = {"平均准确率": [0.9436, 0.8866, 0.9175, 0.9355, 0.9011],
          "平均遗忘率": [0, 0.1628, 0.0807, 0.0430, 0.0921]}

sdp = {"平均准确率": [0.9967, 0.9910, 0.9570, 0.9564, 0.9621],
       "平均遗忘率": [0, 0.0180, 0.0090, 0.0068, 0.0068]}

# Plot 平均准确率
fig1, ax1 = plt.subplots(dpi=300)
ax1.plot(sdp["平均准确率"], label='Ours', linestyle='-', marker='o')
ax1.plot(l2p["平均准确率"], label='L2P', linestyle='-.', marker='^')
ax1.plot(dual_p["平均准确率"], label='DualPrompt', linestyle='--', marker='s')

ax1.set_xlabel('task')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.set_xticks(range(len(l2p["平均准确率"])))

# 调整布局
fig1.tight_layout()

# Plot 平均遗忘率
fig2, ax2 = plt.subplots(dpi=300)
ax2.plot(sdp["平均遗忘率"], label='Ours', linestyle='-', marker='o')
ax2.plot(l2p["平均遗忘率"], label='L2P', linestyle='-.', marker='^')
ax2.plot(dual_p["平均遗忘率"], label='DualPrompt', linestyle='--', marker='s')

ax2.set_xlabel('task')
ax2.set_ylabel('Forgetting')
ax2.legend()
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
# y 轴范围0~30%
ax2.set_ylim([0, 0.20])
ax2.set_xticks(range(len(l2p["平均遗忘率"])))

# 调整布局
fig2.tight_layout()

plt.show()