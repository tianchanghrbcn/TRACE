import os
import pandas as pd
import matplotlib.pyplot as plt


def save_txt_data(error_rates, scores_dict, output_dir, algo1, dataset):
    """将错误率和聚类得分数据保存为 .txt 文件"""
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"{algo1}_{dataset}_scores.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset}, Cleaning Method: {algo1}\n")
        f.write("=" * 60 + "\n")
        for i, er in enumerate(error_rates):
            scores = ", ".join([f"{method}: {scores_dict[method][i]:.2f}" for method in scores_dict])
            f.write(f"Error Rate: {er:.2f} | {scores}\n")

    print(f"Saved TXT data: {txt_path}")


def plot_absolute_scores(
        algo1,
        dataset,
        base_dir=r"D:\algorithm paper\data_experiments\results\3_analyzed_data\analysis_original_results"
):
    dataset_path = os.path.join(base_dir, dataset)

    # 确保路径存在
    if not os.path.exists(dataset_path):
        print(f"Error: Path does not exist - {dataset_path}")
        return

    # 仅收集以“_relative.csv”结尾的文件
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith("_relative.csv")]
    if not csv_files:
        print(f"Warning: No '_relative.csv' files found in {dataset_path}")
        return

    # 定义需要绘制的聚类算法
    clustering_methods = ["HC", "AffinityPropagation", "GMM", "KMeans", "DBSCAN", "OPTICS"]
    method_markers = {
        "HC": "o",
        "AffinityPropagation": "^",
        "GMM": "s",
        "KMeans": "D",
        "DBSCAN": "x",
        "OPTICS": "+"
    }

    # 存储错误率和不同聚类算法的得分
    scores_dict = {m: [] for m in clustering_methods}
    error_rates = []

    # 读取各个 CSV 文件
    for csv_file in csv_files:
        file_path = os.path.join(dataset_path, csv_file)

        # 解析文件名，获取错误率
        file_name_noext = csv_file.replace("_relative.csv", "")
        parts = file_name_noext.split("_")
        err_str = parts[-1]
        err_float = float(err_str.replace("%", ""))
        error_rates.append(err_float)

        # 读取 CSV
        df = pd.read_csv(file_path)

        # 仅保留当前清洗算法 (algo1) 的行
        subset_df = df[df["Cleaning Algorithm"] == algo1]

        # 如果该清洗算法无数据，则填充 0.0
        if subset_df.empty:
            for method in clustering_methods:
                scores_dict[method].append(0.0)
            continue

        for method in clustering_methods:
            row = subset_df[subset_df["Clustering Method"] == method]
            score = row["Score"].values[0] if not row.empty else 0.0
            scores_dict[method].append(min(score, 3.0))  # 最高分截断为 3.0

    # 对错误率进行升序排序
    sorted_indices = sorted(range(len(error_rates)), key=lambda i: error_rates[i])
    sorted_error_rates = [error_rates[i] for i in sorted_indices]
    for method in clustering_methods:
        scores_dict[method] = [scores_dict[method][i] for i in sorted_indices]

    # 保存数据到 TXT
    txt_output_dir = r"D:\algorithm paper\data_experiments\results\4_final_results\txt_data"
    save_txt_data(sorted_error_rates, scores_dict, txt_output_dir, algo1, dataset)

    # x 轴离散等分
    x_positions = list(range(len(sorted_error_rates)))

    # **保证比例不变，增大清晰度**
    plt.figure(figsize=(16, 5), dpi=600)  # **DPI 提高到 600，增强清晰度**
    plt.rcParams["font.family"] = "Times New Roman"  # 设置字体为 Times New Roman

    # 绘制每种聚类算法的折线图
    for method in clustering_methods:
        plt.plot(
            x_positions,
            scores_dict[method],
            marker=method_markers[method],
            markersize=12,
            linewidth=2,
        )

    # **调整字体大小**
    plt.title(
        f"Combined Scores of Clustering Algorithms vs. Error Rates on '{dataset}'\nDataset after '{algo1}' Cleaning",
        fontsize=30  # **增大标题字体**
    )

    # **加粗图像边框**
    ax = plt.gca()  # 获取当前坐标轴
    for spine in ax.spines.values():
        spine.set_linewidth(3)  # 设定边框宽度

    # **加粗 X/Y 轴线**
    ax.xaxis.set_tick_params(width=3)
    ax.yaxis.set_tick_params(width=3)

    # **调整刻度方向与长度**
    ax.tick_params(
        axis='both',       # 作用于 x 和 y 轴
        direction='in',    # 刻度线朝内
        length=10,         # 增大刻度长度
        width=3            # 加粗刻度线
    )

    # 调整 x 轴和 y 轴刻度字体（但去掉轴标签）
    plt.xticks(x_positions, [f"{er:.2f}" for er in sorted_error_rates], fontsize=30)  # **保留小数**
    plt.yticks(fontsize=30)

    # **删除横纵坐标标签**
    plt.xlabel("")
    plt.ylabel("")

    # 设置 y 轴范围
    plt.ylim([0, 3.3])

    # **不画图例**
    plt.grid(False)

    # 画错误率 25% 参考线
    target_value = 25.0
    vertical_line_x = None
    if target_value in sorted_error_rates:
        i_25 = sorted_error_rates.index(target_value)
        vertical_line_x = i_25 + 0.5 if i_25 < len(sorted_error_rates) - 1 else i_25 + 0.25
    else:
        for i in range(len(sorted_error_rates) - 1):
            if sorted_error_rates[i] < target_value < sorted_error_rates[i + 1]:
                vertical_line_x = i + 0.5
                break
    if vertical_line_x is not None:
        plt.axvline(vertical_line_x, linestyle='--', color='gray', linewidth=2)
        plt.scatter([vertical_line_x], [3.0], marker='*', s=250, color='red', zorder=5)

    # **保持比例不变**
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # **保存图片**
    img_output_dir = r"D:\algorithm paper\data_experiments\results\4_final_results\graphs\error_rate_graph"
    os.makedirs(img_output_dir, exist_ok=True)
    save_path = os.path.join(img_output_dir, f"{algo1}_{dataset}_combined_scores.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # **增加 DPI 并确保边界完整**
    print(f"Saved plot: {save_path}")

    plt.close()


if __name__ == "__main__":
    for algo in ["mode", "raha-baran"]:
        for ds in ["beers", "flights", "hospital", "rayyan"]:
            plot_absolute_scores(algo, ds)
