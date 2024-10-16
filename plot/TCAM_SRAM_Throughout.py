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


# Planter Netbeacon Mousikav2 IIsy Helios
metrics_data = {

    "RUT": np.array([
        [9.445177, 13.34963, 8.279628, 8.695257, 7.208069],  # IDS
        [18.23957, 39.05355, 15.76278, 17.10894, 14.15645],  # IoT
        [55.81219, 111.7802, 23.49278, 33.60766, 24.7765],   # NB15
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
    "Throughput":np.array([[10,8,6,4,10], # 10
                          [50,60,40,20,50], # 50
                          [100,80,60,40,100], # 100
                          ]),
    "Latency":np.array([[9,7,5,3,0.66],# 10
                       [9,7,5,3,0.67],# 50
                       [9,7,5,3,0.65]]# 100
                       ),
}


# 数据集名称
datasets = [ 'IDS', 'IoT','NB15',]
# datasets = [ 'CICIDS2018', 'TON-IoT','UNSW-NB15',]

# models = ['Helios', 'Mousikav2', 'Netbeacon', 'Planter', 'IIsy']
models = [ 'Planter','Netbeacon', 'Mousikav2',  'IIsy','Helios']


for metri_name in [['TCAM','SRAM'],["Throughput","Latency"]]:




    y = metrics_data[metri_name[0]][:,-1]
    # cheng 7 red 5  purple 9
    if "TCAM" in metri_name:
        fig, ax = plt.subplots(figsize=(6.35* 3/4, 3.65))  # 设置每个数据集的图大小
        x = np.arange(len(datasets))  # 模型作为横轴
        xlabel = datasets
        colo = 9
        colo2 = 11
    else:
        fig, ax = plt.subplots(figsize=(6.35* 1.16*3/4 , 3.65))  # 设置每个数据集的图大小
        x = np.arange(len(datasets))
        xlabel = ["10Gps","50Gbps","100Gbps"]
        colo = 1
        colo2 = 3


    plt.xlim(-0.6,2.6)
    ax2 = ax.twinx()
    width = 0.3 

    ax.bar(x - 0.5 * width, y, width=width,label=metri_name[0], color='white',
    ec=COLORS[colo], hatch=HATCH[4] * 2, linewidth=ALLWIDTH)

    y2 = metrics_data[metri_name[1]][:,-1]
    ax2.bar(x + 0.5 * width, y2, width=width,label=metri_name[1], color='white',
    ec=COLORS[colo2], hatch=HATCH[5] * 2, linewidth=ALLWIDTH)


    ax.set_xticks(x)
    ax.set_xticklabels(xlabel)  # 设置X轴标签为模型名称

    if "TCAM" in metri_name:
        ax.set_ylim(0, 60)
        ax.yaxis.set_major_locator(FixedLocator([0,20,40,60]))
        ax2.set_ylim(0, 60)
        ax2.yaxis.set_major_locator(FixedLocator([]))
        ax.set_ylabel("Rate (%)",fontsize=FONTSIZE)
        ax2.tick_params(right=False)
        ax2.spines['right'].set_visible(False)   # 隐藏右边的 y 轴线
        legeng_pos = ((0.27, 0.9))
        # ax2.set_ylabel("SRAM Rate (%)",fontsize=FONTSIZE)
        ax.set_xlabel("Dataset",fontsize=FONTSIZE)

    else:

        ax.set_ylim(0, 150)
        ax.yaxis.set_major_locator(FixedLocator([0,50,100,150]))
        ax2.set_ylim(0, 1.2)
        ax2.yaxis.set_major_locator(FixedLocator([0,0.4, 0.8, 1.2]))
        ax.set_ylabel("Throughput (Gbps)",fontsize=FONTSIZE)

        ax2.set_ylabel("Latency (µs)",fontsize=FONTSIZE)
        ax.set_xlabel("Traffic rate",fontsize=FONTSIZE)

        legeng_pos = ((0.18, 0.9))

    ax.tick_params(labelsize=FONTSIZE)
    ax2.tick_params(labelsize=FONTSIZE)
    # ax.set_title(f'Dataset: {dataset_name}', fontsize=FONTSIZE)  # 设置图标题为数据集名称
    plt.tick_params(axis='both', which='both', length=0)
    ax.grid(linestyle=':', axis='y')
    

    fig.legend(fontsize=FONTSIZE , loc='upper left', ncol=3, handleheight=0.7,
               handlelength=ALLWIDTH-0.3, handletextpad=0.1, columnspacing=0.5, frameon=True,bbox_to_anchor=legeng_pos)
    # fig.legend(fontsize=FONTSIZE, loc='upper left', ncol=3, handleheight=0.7,
                # handlelength=ALLWIDTH-0.3, handletextpad=0.2, columnspacing=1, frameon=True, bbox_to_anchor=((0.19, 0.9)))
    
    plt.tight_layout()

    # 保存为PDF文件
    pp = PdfPages(f"./C2_TCAM/{metri_name[0]}.pdf")
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()