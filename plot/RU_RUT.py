import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import FixedLocator
from matplotlib import rcParams
import matplotlib.ticker as mtick

# config = {
#     "font.family": 'serif',
#     "font.size": 15,
#     "mathtext.fontset": 'stix',
#     "font.serif": ['STIXGeneral'],
# }
config = {
    "font.family": 'serif',
    "font.size": 20,
    # "mathtext.fontset": 'stix',
    "mathtext.fontset": 'custom',  # 使用自定义字体
    "font.serif": ['Linux Libertine O'],  # 设置为 LinLibertineT 字体
    "mathtext.rm": 'Linux Libertine',  # 数学文本的罗马字体
    "mathtext.it": 'Linux Libertine:italic',  # 数学文本的斜体
    "mathtext.bf": 'Linux Libertine:bold',  # 数学文本的粗体
}

mpl.rc('pdf', fonttype=42)
ALLWIDTH = 1.5
FONTSIZE = 20
Marker = ['o', 'v', '8', 's', 'p', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X']
HATCH = ['+', 'x', '/', 'o', '|', '\\', '-', 'O', '.', '*']
Line_Style = ['-', '--', '-.', ':']
COLORS = sns.color_palette("Paired")
rcParams.update(config)

width = 0.25  # 每个柱子的宽度

# Planter Netbeacon Mousikav2 IIsy Helios
metrics_data = {

    "RUT": np.array([
        [ 1.04946411,1.48329222,0.91995867,0.96613967,0.80089656],
        [ 2.27994625,4.88169375,1.9703475,2.1386175,1.76955625],
        [ 6.97652375,13.972525,2.9365975,4.2009575,3.0970625 ],
    ]),
    "RN": np.array([
        [1596, 2415, 37, 86, 42],         # IDS (previously 67 / 42)
        [1371, 5674, 337, 360, 278],      # IoT (previously 321 / 278)
        [11624, 17177, 365, 3958, 1884],  # NB15 (previously 2538 / 1884)
    ]),
    "TCAM": np.array([
        [4.861111, 7.638889, 1.388889, 2.777778, 2.777778],  # IDS
        [6.597222, 25.347222, 1.388889, 5.208333, 6.25],     # IoT
        [44.79166667, 31.94444444, 1.041666667, 19.44444444, 41.66666667],  # NB15
    ]),
    "SRAM": np.array([
        [9.583333, 9.270833, 8.541667, 9.166667, 8.541667],  # IDS
        [9.375, 9.270833, 8.125, 9.0625, 8.125],             # IoT
        [2.083333333, 1.25, 0.104166667, 1.145833333, 0.833333333],  # NB15
    ]),
    "Training_Time": np.array([
        [26.12, 26.57, 225, 21.97, 180.91],  # IDS
        [35.25, 36.78, 410.03, 33.62, 462.05],  # IoT
        [181.92, 75.15, 559, 86.6, 8902.33],  # NB15
    ]),
    "Throughput":np.array([]),
    "Latency":np.array([]),
}


# 数据集名称
datasets = [ 'CICIDS2018', 'TON-IoT','UNSW-NB15',]
# models = ['Helios', 'Mousikav2', 'Netbeacon', 'Planter', 'IIsy']
models = [ 'Planter','Netbeacon', 'Mousikav2',  'IIsy','Helios']


for metri_name in ['RN','RUT']:
    colors_choose_isx = [1, 5,  3, 9]
    hatch_choose_idx = [0, 1, 2, 4, 5]
    # fig, ax = plt.subplots(figsize=(6.35, 3.65))  # 设置每个数据集的图大小
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.35*1.5, 3.65), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    x = np.arange(len(models))  # 模型作为横轴

    for dataset_idx, dataset_name in enumerate(datasets):
        y = metrics_data[metri_name][dataset_idx, :]
        ax.bar(x +(dataset_idx - 1) *width, y, width=width, color='white',
        ec=COLORS[colors_choose_isx[dataset_idx]], hatch=HATCH[hatch_choose_idx[dataset_idx]] * 2, linewidth=ALLWIDTH)
        ax2.bar(x +(dataset_idx - 1) *width, y, width=width,label=dataset_name, color='white',
        ec=COLORS[colors_choose_isx[dataset_idx]], hatch=HATCH[hatch_choose_idx[dataset_idx]] * 2, linewidth=ALLWIDTH)

    ax.set_xticks(x)
    # ax.set_xticklabels(models, rotation=15)  # 设置X轴标签为模型名称

    if metri_name == "RN":
        # 绘制增量部分（虚线柱子）
        for i in range(3):  
            height = metrics_data['RN'][i, -1]
            increment = [67,321,2538][i]
            increment_text = [37.3,13.4,25.8]

            # 在当前柱子顶部绘制虚线增量部分
            ax2.bar(4 + (i - 1)*width, increment-height, width=width, bottom=height, color='none',  # 透明填充
                    edgecolor=COLORS[colors_choose_isx[i]], linestyle='--', linewidth=1,)  # 用虚线边框表示增量部分
                # 在当前柱子顶部绘制虚线增量部分
            ax.bar(4 + (i - 1)*width, increment-height, width=width, bottom=height, color='none',  # 透明填充
                    edgecolor=COLORS[colors_choose_isx[i]], linestyle='--', linewidth=1,)  # 用虚线边框表示增量部分
            if i <=1:
                # 在柱子上方标注增量百分比
                ax2.text(4 + (i - 1)*width-0.05, increment+10, f"{increment_text[i]}%",  # 调整 y 坐标以适应增量标注$\downarrow$
                        ha='center', va='bottom', fontsize=FONTSIZE-5, color=COLORS[colors_choose_isx[i]])
            else:
                # 在柱子上方标注增量百分比
                ax.text(4 + (i - 1)*width ,increment+100, f"{increment_text[i]}%",  # 调整 y 坐标以适应增量标注$\downarrow$
                        ha='center', va='bottom', fontsize=FONTSIZE-5, color=COLORS[colors_choose_isx[i]])
            
        ax2.set_ylim(0, 400)
        ax2.yaxis.set_major_locator(FixedLocator([0,200,400]))
        ax.set_ylim(600, 18000)
        ax.yaxis.set_major_locator(FixedLocator([600,9400,18000]))
        metri_full_name = "Number of Rules"

    elif metri_name == "RUT":
        ax2.set_ylim(0, 5)
        ax2.yaxis.set_major_locator(FixedLocator([0,2.5,5]))
        ax.set_ylim(5,15)
        ax.yaxis.set_major_locator(FixedLocator([5,10,15]))
        metri_full_name = "Reconfiguration Time (s)"


    ax.spines['bottom'].set_visible(False)
    ax.xaxis.tick_bottom()

    ax2.spines['top'].set_visible(False)
    ax2.xaxis.tick_bottom()

    ax.tick_params(left=False)
    ax.tick_params(bottom=False)
    ax.tick_params(top=False)
    ax.tick_params(right=False)
    ax.tick_params(labelsize=FONTSIZE)
    ax.tick_params(pad=2)
    ax.grid(linestyle=':', axis='y')

    ax2.tick_params(left=False)
    ax2.tick_params(bottom=False)
    ax2.tick_params(top=False)
    ax2.tick_params(right=False)
    ax2.tick_params(labelsize=FONTSIZE)
    ax2.tick_params(pad=2)
    ax2.grid(linestyle=':', axis='y')

    # ax2.set_xticklabels(models, rotation=15,ha='right', rotation_mode='anchor')

    ax2.set_xticklabels(models)

    d = .5
    
    # "/" in y axis , we set makersize to 0. so it is unvisble
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=0,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
    # ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    
    
    aax1 = fig.add_subplot(111, frameon=False)
    aax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    aax1.set_ylabel(metri_full_name, labelpad=23, fontsize=FONTSIZE)

    # aax2 = fig.add_subplot(111, frafull_meon=False)
    # aax2.yaxis.set_label_position("right")
    # aax2.yaxis.tick_right()
    # aax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # aax2.set_ylabel('Transmission Time(s)', labelpad=labelpad, fontsize=FONTSIZE)

    # trans = ax2.get_xaxis_transform()

    plt.subplots_adjust(hspace=0.3)

    # fig.legend(fontsize=FONTSIZE, loc='upper right', ncol=3, handleheight=0.7,
    #             handlelength=ALLWIDTH-0.3, handletextpad=0.2, columnspacing=1, frameon=True, bbox_to_anchor=((0.5, 1.5)))
    
    
    # plt.tight_layout()

    # 保存为PDF文件
    pp = PdfPages(f"./C2_TCAM/"+metri_name+".pdf")
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()
    # plt.show()
