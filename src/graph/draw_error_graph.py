import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# 读取数据
data = pd.read_csv("dataset_comparison.csv")

# 获取唯一的数据集名称
datasets = data["Dataset Name"].unique()

# 设置全局样式
sns.set(style="white")

# 创建单独的图例
fig_legend, ax_legend = plt.subplots(figsize=(12, 1))  # 调整图例尺寸
handles = [
    plt.Rectangle((0, 0), 1, 1, fc="blue", label="Best Combination Score"),
    plt.Rectangle((0, 0), 1, 1, fc="green", label="Best Deviation Combination Score"),
    plt.Rectangle((0, 0), 1, 1, fc="#FF0000", label="Reference Score (100%)"),  # 纯红色
    plt.Line2D([0], [0], color="orange", linestyle="--", marker="o", linewidth=2.5, label="Missing Value Ratio (%)")  # 加粗折线
]
ax_legend.legend(handles=handles, loc='center', ncol=4, prop={"family": "Times New Roman"})
ax_legend.axis('off')
plt.savefig("legend_plot.png", dpi=600, bbox_inches='tight')
plt.close(fig_legend)

# 遍历每个数据集并绘制图像
for dataset in datasets:
    df = data[data["Dataset Name"] == dataset]

    fig, ax1 = plt.subplots(figsize=(12, 5))  # 调整横纵比为 12:5
    ax2 = ax1.twinx()

    # 仅使用错误率作为横坐标
    x_values = df["Error Rate (%)"].astype(str)
    indices = range(len(df))

    # 绘制柱状图（评分）
    width = 0.2  # 使柱子变细
    ax1.bar([i - width for i in indices], df["Best Combination Score (%)"], width=width, color="blue")
    ax1.bar(indices, df["Best Deviation Combination Score (%)"], width=width, color="green")
    ax1.bar([i + width for i in indices], [100] * len(df), width=width, color="#FF0000")  # 纯红色

    # 绘制折线图（缺失值比率），加粗折线
    ax2.plot(indices, df["Missing Value Ratio"] * 100, color="orange", linestyle="--", marker="o", linewidth=2.5)

    # 设置标题和轴标签
    ax1.set_xticks(indices)
    ax1.set_xticklabels(x_values, fontname="Times New Roman", fontsize=20)
    ax1.set_xlabel("Error Rate (%)", fontname="Times New Roman", fontsize=21.5)

    ax1.set_ylabel("", labelpad=10, fontname="Times New Roman", fontsize=20)
    ax2.set_ylabel("", labelpad=10, fontname="Times New Roman", fontsize=20)

    # 调整纵坐标范围，使顶部留白更大
    ax1.set_ylim(0, 210)  # 原本是 200，调整到 210 增加顶部留白
    ax2.set_ylim(0, 55)   # 原本是 50，调整到 55 以防数值 50 被裁剪
    ax2.set_yticks(range(0, 56, 10))

    # 调整刻度向内，并增加字体大小
    ax1.tick_params(direction='in', labelsize=20)
    ax2.tick_params(direction='in', labelsize=20)

    # 设置纵坐标字体
    for label in ax1.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(20)

    for label in ax2.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(20)

    # 移除图像中间的横竖线
    ax1.grid(False)
    ax2.grid(False)

    # 上移标题（调整 pad 值）
    ax1.set_title(f"Comparison of Scores and Missing Value Ratio for {dataset.capitalize()} Dataset (Capped at 200%)",
                  fontname="Times New Roman", fontsize=21.5, pad=15)  # 增加 pad 让标题上移

    # 保存图片，并留出额外的边距
    plt.savefig(f"{dataset}_error.png", dpi=600, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
