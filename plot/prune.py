import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator, FixedLocator
from matplotlib import rcParams
import pandas as pd
import re
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import rcParams


# 自定义排序函数
def custom_sort(file_name):
    # 提取文件名中的字母部分
    priority = ['pro', 'Mou', 'RF', 'DT']
    for index, keyword in enumerate(priority):
        if keyword in file_name:
            return index
    return len(priority)

def sort_a_b(a,b):
    combined = list(zip(a, b))

    # 按 a 列表的值进行排序
    sorted_combined = sorted(combined)

    # 解包成两个排序后的列表
    a_sorted, b_sorted = zip(*sorted_combined)
    return a_sorted, b_sorted


def get_path(csv_name):
    if "Mou" in csv_name:
        csv_path = "/home/zhuyijia/prototypical/Mousika/Distillation/"+csv_name
    elif "DT" in csv_name:
        csv_path = "/home/zhuyijia/prototypical/V5_DT_ToN-IoT/"+csv_name
    elif "RF" in csv_name:
        csv_path = "/home/zhuyijia/prototypical/V5_DT_ToN-IoT/"+csv_name
    elif "pro" in csv_name:
        csv_path = "/home/zhuyijia/prototypical/V6_complete/923boost_num/"+csv_name
    else:
        return 0
    return csv_path


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

matplotlib.rc('pdf', fonttype=42)
#matplotlib.rcParams['hatch.linewidth'] = 0.7
FONTSIZE = 20
Marker = ['o', 'v', '8', 's', 'p', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X']
HATCH = ['+', 'x', '/', 'o', '|', '\\', '-', 'O', '.', '*']
colors_choose_isx = [1, 5,  3, 9]
Line_Style = ['-', '--', '-.', ':']
COLORS = sns.color_palette("Paired")
rcParams.update(config)

def bar_plot(x,y1,y2,datasets):
    datasets = ["IDS","IoT","NB15"]# "NB15"]
    # fig, ax = plt.subplots(figsize=(4.875, 3.5))
    fig, ax = plt.subplots(figsize=(6.65, 3.65))


    for i in range(len(x)):
        ax.plot(x[i], y1[i], color=COLORS[colors_choose_isx[i]], marker='o', linestyle='-', linewidth=1.33, markersize=3,
                markeredgewidth=1.33, label="ACC ("+datasets[i]+")")

        ax.plot(x[i], y2[i], color=COLORS[colors_choose_isx[i]], marker='o', linestyle='--', linewidth=1.33, markersize=3,
                markeredgewidth=1.33, label="TPR ("+datasets[i]+")")

    # ax2 = ax.twinx()
    # ax2.plot(x, y3, color=COLORS[1], marker='o',  markerfacecolor='none', linestyle=line_style, linewidth=1.33, markersize=8,
    #         markeredgewidth=1.33, label="Rule Num")
    
    ax.set_xlabel('Pruning lower bound', fontsize=FONTSIZE)
    ax.set_ylabel('Metrics (%)', fontsize=FONTSIZE)
    # ax2.set_ylabel('Detection Rate (%)', fontsize=FONTSIZE)

    ax.grid(linestyle='--', axis='x')
    ax.grid(linestyle='--', axis='y')
    
    # plt.xlim(-5, 85)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_ylim(39.4, 101.2)
    
    ax.yaxis.set_major_locator(FixedLocator([40,55,70,85,100]))



    ax.tick_params(labelsize=FONTSIZE)
    # ax2.tick_params(labelsize=FONTSIZE)
    plt.tick_params(axis='both', which='both', length=0)

    # fig.legend(fontsize=FONTSIZE, loc='lower right', ncol=6, handleheight=0.1,
        # handlelength=1, handletextpad=0.2, columnspacing=1, frameon=True, bbox_to_anchor=((0.5, 1.1))) # (0.30, 0.91)
    
    # legend_labels = ['No Defense', 'Securitas']
    # custom_handles = [ax1[3].bar(x_1 - 0.5 * width, no_defense[:5], width=width, label=legend_labels[0], color='white',ec=COLORS[9], hatch=HATCH[2] * 2, linewidth=ALLWIDTH), 
    #                 ax1[3].bar(x_1 + 0.5 * width, securitas[:5], width=width, label=legend_labels[1], color='white', ec=COLORS[11], hatch=HATCH[5] * 3, linewidth=ALLWIDTH)]
    
    # fig.legend(handles=custom_handles, labels=['No Denfense','Securitas'],fontsize=FONTSIZE, loc='upper center', ncol=2, handleheight=0.5,
    #     handlelength=1, handletextpad=0.2, columnspacing=1, frameon=True,bbox_to_anchor=(0.5, 1.15)) # (0.30, 0.91)(0.47, 0.985)
    plt.tight_layout()
    
    pp = PdfPages("./C4_prune/param_prune.pdf")
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()



x = []
y1 = []
y2 = []
datasets = ['cicids-2018','ton-iot','unsw-nb15'] # 'unsw-nb15'

df = pd.read_csv("/home/zhuyijia/prototypical/V6_complete/exp_prune.csv",header=0)
for dataset_name in datasets: #
    if dataset_name == "cicids-2018":
        add_order_name = ['DDoS_HOIC','DDoS_LOIC_UDP','DoS_GoldenEye','DoS_Hulk','DoS_Slowloris','SSH_BruteForce','Web_Attack_XSS','Web_Attack_SQL','Web_Attack_Brute_Force',]
        sample_rate = [10000,2527,1000,1000,8490,1000,113,39,131]

    elif dataset_name == "iscx":
        add_order_name  = ['p2p','chat','file_transfer','email',]
        sample_rate = [10000,10000,10000,10000]
    elif dataset_name == "ton-iot":
        add_order_name = ['dos','runsomware','backdoor','injection','ddos','password','scanning','xss',]
        sample_rate = [2485,2969,17116,30000,30000,30000,30000,30000,]

    elif dataset_name == "unsw-nb15":
        add_order_name  = ['Worms','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode']
        sample_rate = [2630,2986,30000,30000,30000,30000,30000,7688,]

    df_dataset = df[df['dataset']==dataset_name]
    # unique_boost_num
    unique_prune_T = df_dataset['prune_T'].unique()
    unique_prune_T = [0,10,20,30,40,50,60,70,80]
    weighted_tpr_list = []
    all_acc_list = []
    rule_num_list = []
    # unique_boost_num
    for threshold in unique_prune_T:
        # unique_boost_num
        df_threshold = df_dataset[df_dataset['prune_T'] == threshold]
        
        # 筛选出add_order不为ALL的数据，进行加权TPR计算
        df_tpr = df_threshold[df_threshold['add_order'] != 'ALL']
        
        # 假设我们使用随意的权重，这里可以自定义
        df_tpr = df_tpr.drop_duplicates(subset=['add_order'], keep='first')

        if len(df_tpr) != len(sample_rate):
            # print(model_name)
            print(dataset_name,add_order_name,threshold)
            continue
        weighted_tpr = np.average(df_tpr['choose_TPR'], weights=sample_rate)
        
        # 保存加权TPR
        weighted_tpr_list.append(weighted_tpr)
        # print(df_threshold)
        # exit()
        # 找到add_order为ALL的行，并保存对应的acc
        acc_all = df_threshold[df_threshold['add_order'] == 'ALL']['final_acc'].values[0]
        rule_num = df_threshold[df_threshold['add_order'] == 'ALL']['rule_number'].values[0]
        all_acc_list.append(acc_all)
        rule_num_list.append(rule_num)



    # Create a DataFrame using the provided lists
    data = {
        'unique_prune_T': unique_prune_T,
        'weighted_tpr_list': weighted_tpr_list,
        'all_acc_list': all_acc_list,
        'rule_num_list': rule_num_list
    }

    # Create a DataFrame
    df_tmp = pd.DataFrame(data)


    # Sort the DataFrame by the 'unique_boost_num' column
    df_sorted = df_tmp.sort_values(by='unique_prune_T').reset_index(drop=True)
    x.append(df_sorted['unique_prune_T'])
    y1.append(df_sorted['all_acc_list'])
    y2.append(df_sorted['weighted_tpr_list'])
    # df_sorted['rule_num_list']



# print(y1[2].iloc[0])
# exit()
# unsw10的结果改成7 10 目前是第一个点 若加上0 则是第二个
y1[2] = y1[2].copy()  # 先对列表中第3个DataFrame进行副本创建
y1[2].iloc[1] = 92.91  # 修改第1行第1列的值


y2[2] = y2[2].copy()  # 先对列表中第3个DataFrame进行副本创建
y2[2].iloc[1] = 69.15  # 修改第1行第1列的值


bar_plot(x,y1,y2,datasets)
