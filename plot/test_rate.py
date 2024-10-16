import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator, FixedLocator
from matplotlib import rcParams
import pandas as pd
import re

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
        csv_path = "/home/zhuyijia/prototypical/V6_complete/"+csv_name
    else:
        return 0
    return csv_path


config = {
    "font.family": 'serif',
    "font.size": 11,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
matplotlib.rc('pdf', fonttype=42)
#matplotlib.rcParams['hatch.linewidth'] = 0.7
FONTSIZE = 20
Marker = ['o', 'v', '8', 's', 'p', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X']
HATCH = ['+', 'x', '/', 'o', '|', '\\', '-', 'O', '.', '*']
Line_Style = ['-', '--', '-.', ':']
COLORS = sns.color_palette("Paired")
rcParams.update(config)

def bar_plot(x,y1,y2,y3,dataset_name):

    # fig, ax = plt.subplots(figsize=(4.875, 3.5))
    fig, ax = plt.subplots(figsize=(6.65, 3.65))

    line_style = '--'

    # COLORS 1(blue) 2(light green) 3 (green) 5 (red) 7 (orange) 9 (purple) 11(brown)
    # (3, 5, 1)
    # (7, 11, 9) 
    # print(x[0],y[0])
    # exit()
    # marker='^'
    ax.plot(x, y1, color=COLORS[5], marker='^', markerfacecolor='none', linestyle=line_style, linewidth=1.33, markersize=8,
            markeredgewidth=1.33, label="Accuracy")
    ax.plot(x, y2, color=COLORS[3], marker='^', markerfacecolor='none', linestyle=line_style, linewidth=1.33, markersize=8,
            markeredgewidth=1.33, label="Detection Rate (%)")

    ax2 = ax.twinx()
    ax2.plot(x, y3, color=COLORS[1], marker='o',  markerfacecolor='none', linestyle=line_style, linewidth=1.33, markersize=8,
            markeredgewidth=1.33, label="Rule Num")
    
    ax.set_xlabel('Threshold', fontsize=FONTSIZE)
    ax.set_ylabel('Rate (%)', fontsize=FONTSIZE)
    # ax2.set_ylabel('Detection Rate (%)', fontsize=FONTSIZE)

    ax.grid(linestyle='--', axis='x')
    ax.grid(linestyle='--', axis='y')
    
    plt.xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylim(0, 101)
    # ax2.set_ylim(0, 101)
    ax.yaxis.set_major_locator(FixedLocator([20,40,60,80,100]))
    # ax2.yaxis.set_major_locator(FixedLocator([20,40,60,80,100]))

    # ax2.yaxis.set_major_locator(FixedLocator([30,60,90,120,150]))
    

    ax.tick_params(labelsize=FONTSIZE)
    # ax2.tick_params(labelsize=FONTSIZE)
    plt.tick_params(axis='both', which='both', length=0)

    fig.legend(fontsize=FONTSIZE, loc='lower right', ncol=1, handleheight=0.7, labelspacing=0.2,
               handlelength=1.2, handletextpad=0.4, columnspacing=1, borderpad=0.2, frameon=True, bbox_to_anchor=((0.8, 0.26))) # (0.30, 0.91)
    
    # legend_labels = ['No Defense', 'Securitas']
    # custom_handles = [ax1[3].bar(x_1 - 0.5 * width, no_defense[:5], width=width, label=legend_labels[0], color='white',ec=COLORS[9], hatch=HATCH[2] * 2, linewidth=ALLWIDTH), 
    #                 ax1[3].bar(x_1 + 0.5 * width, securitas[:5], width=width, label=legend_labels[1], color='white', ec=COLORS[11], hatch=HATCH[5] * 3, linewidth=ALLWIDTH)]
    
    # fig.legend(handles=custom_handles, labels=['No Denfense','Securitas'],fontsize=FONTSIZE, loc='upper center', ncol=2, handleheight=0.5,
        # handlelength=1, handletextpad=0.2, columnspacing=1, frameon=True,bbox_to_anchor=(0.5, 1.15)) # (0.30, 0.91)(0.47, 0.985)
    plt.tight_layout()
    
    pp = PdfPages("./C4_test_rate/test_rate_"+dataset_name+".pdf")
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()




df = pd.read_csv("/home/zhuyijia/prototypical/V6_complete/exp_test_rate_proto.csv",header=0)
for dataset_name in ["iscx",'cicids-2018','ton-iot','unsw-nb15']:
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
    # test_rate
    unique_test_rates = df_dataset['test_rate'].unique()
    weighted_tpr_list = []
    all_acc_list = []
    rule_num_list = []
    # test_rate
    for threshold in unique_test_rates:
        # test_rate
        df_threshold = df_dataset[df_dataset['test_rate'] == threshold]
        
        # 筛选出add_order不为ALL的数据，进行加权TPR计算
        df_tpr = df_threshold[df_threshold['add_order'] != 'ALL']
        
        # 假设我们使用随意的权重，这里可以自定义
        weights = sample_rate
        weighted_tpr = np.average(df_tpr['choose_TPR'], weights=weights)
        
        # 保存加权TPR
        weighted_tpr_list.append(weighted_tpr)
        
        # 找到add_order为ALL的行，并保存对应的acc
        acc_all = df_threshold[df_threshold['add_order'] == 'ALL']['final_acc'].values[0]
        rule_num = df_threshold[df_threshold['add_order'] == 'ALL']['rule_number'].values[0]
        all_acc_list.append(acc_all)
        rule_num_list.append(rule_num)

    # print(weighted_tpr_list)
    # print(all_acc_list)
    # print(len(weighted_tpr_list))
    # print(len(all_acc_list))
    # print(unique_thresholds)

    unique_test_rates,weighted_tpr_list = sort_a_b(unique_test_rates,weighted_tpr_list)
    unique_test_rates,all_acc_list = sort_a_b(unique_test_rates,all_acc_list)
    unique_test_rates,rule_num_list = sort_a_b(unique_test_rates,rule_num_list)


    bar_plot(unique_test_rates,all_acc_list,weighted_tpr_list,rule_num_list,dataset_name)
