import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import FixedLocator
from matplotlib import rcParams
import matplotlib.ticker as mtick

config = {
    "font.family": 'serif',
    "font.size": 11,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}

mpl.rc('pdf', fonttype=42)
ALLWIDTH = 1.5
FONTSIZE = 20
Marker = ['o', 'v', '8', 's', 'p', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X']
HATCH = ['+', 'x', '/', 'o', '|', '\\', '-', 'O', '.', '*']
Line_Style = ['-', '--', '-.', ':']
COLORS = sns.color_palette("Paired")
rcParams.update(config)

width = 0.2

# 各指标对应的数值，分别是 ACC、F1、Precision、Recall
metrics_data = {
    "IDS": np.array([

    ]),
    "IoT": np.array([

    ]),
    "NB15": np.array([

    ]),
}

# 数据集名称
datasets = ['unsw-nb15', 'ton-iot', 'iscx', 'cicids-2018']
models = ['Helios', 'Mousikav2', 'Netbeacon', 'Planter', 'IIsy']

# Plot each dataset
for dataset_idx, dataset_name in enumerate(datasets):
    fig, ax = plt.subplots(figsize=(6.35, 3.65))  # 设置每个数据集的图大小

    
    x = np.arange(len(models))  # 模型作为横轴
    width = 0.167  # 每个柱子的宽度
    colors_choose_isx = [1, 5,  3, 9]
    hatch_choose_idx = [0, 1, 2, 4, 5]
    for j, (metric, color) in enumerate(zip(metrics_data.keys(), COLORS)):
        # 获取每个指标在当前数据集上的模型表现
        y = metrics_data[metric][dataset_idx, :]
        ax.bar(x + (j - 1.5) * width, y, width=width,label=metric, color='white',
        ec=COLORS[colors_choose_isx[j]], hatch=HATCH[hatch_choose_idx[j]] * 2, linewidth=ALLWIDTH)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)  # 设置X轴标签为模型名称
    if dataset_name == "unsw-nb15":
        ax.set_ylim(60, 100)
        ax.yaxis.set_major_locator(FixedLocator([60,70, 80, 90, 100]))

    else:
        ax.set_ylim(80, 100)
        ax.yaxis.set_major_locator(FixedLocator([80,85, 90,95, 100]))

    ax.set_ylabel("Metrics (%)",fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    # ax.set_title(f'Dataset: {dataset_name}', fontsize=FONTSIZE)  # 设置图标题为数据集名称
    plt.tick_params(axis='both', which='both', length=0)
    ax.grid(linestyle=':', axis='y')
    

    fig.legend(fontsize=FONTSIZE - 5, loc='upper right', ncol=5, handleheight=1,
               handlelength=ALLWIDTH, handletextpad=0.1, columnspacing=0.5, frameon=True, bbox_to_anchor=((0.94, 0.94)))
    plt.tight_layout()

    # 保存为PDF文件
    pp = PdfPages(f"./appendix_train_time/train_time.pdf")
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()
    plt.show()
