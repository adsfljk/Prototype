import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator, FixedLocator
from sklearn.metrics import auc
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
Line_Style = ['-', '--', '-.', ':']
COLORS = sns.color_palette("Paired")
rcParams.update(config)

def bar_plot(x,y,dataset_name):

    # fig, ax = plt.subplots(figsize=(4.875, 3.5))
    fig, ax = plt.subplots(figsize=(6.65, 3.65))

    line_style = '-'
    model_names = ["Helios","Mousikav2","Netbeacon","Planter","IIsy"]
    # COLORS 1(blue) 2(light green) 3 (green) 5 (red) 7 (orange) 9 (purple) 11(brown)
    # (3, 5, 1)
    # (7, 11, 9) 
    # print(x[0],y[0])
    # exit()
    # marker='^'
    if dataset_name=="unsw-nb15":
        plt.xlim(-1, 40)
        ax.xaxis.set_major_locator(FixedLocator([0,10,20,30,40]))
        ax.set_ylim(-1, 102)
        ax.yaxis.set_major_locator(FixedLocator([0,25,50,75,100]))
    elif dataset_name=="cicids-2018":
        plt.xlim(-1, 40)
        ax.xaxis.set_major_locator(FixedLocator([0,10,20,30,40]))
        ax.set_ylim(-1, 102)
        ax.yaxis.set_major_locator(FixedLocator([0,25,50,75,100]))
    elif dataset_name=="ton-iot":
        plt.xlim(-1, 40)
        ax.xaxis.set_major_locator(FixedLocator([0,10,20,30,40]))
        ax.set_ylim(-1, 102)
        ax.yaxis.set_major_locator(FixedLocator([0,25,50,75,100]))

    COLORS_choose_idx = [3,5,1,7,11]

    for i in range(len(x)):
        # marker='o',markersize=3, 
        if i == 0:
            linewidth = 3
            zorder = 10
        else:
            linewidth = 3
            zorder = 1
        ax.plot(x[i], y[i], color=COLORS[COLORS_choose_idx[i]],  linestyle=line_style, linewidth=linewidth,label=model_names[i],zorder=zorder)#marker=Marker[i],


    
    ax.set_xlabel('FPR (%)', fontsize=FONTSIZE)
    ax.set_ylabel('TPR (%)', fontsize=FONTSIZE)
    # ax2.set_ylabel('Detection Rate (%)', fontsize=FONTSIZE)

    ax.grid(linestyle='--', axis='x')
    ax.grid(linestyle='--', axis='y')
    

    # plt.xlim(-1, 101)
    # ax.xaxis.set_major_locator(FixedLocator([0,25,50,75,100]))

    ax.tick_params(labelsize=FONTSIZE)
    # ax2.tick_params(labelsize=FONTSIZE)
    plt.tick_params(axis='both', which='both', length=0)

    # fig.legend(fontsize=FONTSIZE, loc='upper right', ncol=5, handleheight=0.7,
    #             handlelength=1.5, handletextpad=0.2, columnspacing=1, frameon=True, bbox_to_anchor=((0.5, 1.5)))
    

    plt.tight_layout()
    
    pp = PdfPages("./C2_ROC/ROC_legend.pdf")
    # pp = PdfPages("./C2_ROC/ROC_legend.pdf")

    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()



tpr_rate = 1



auc_value_list_data = pd.DataFrame()
for dataset_name in ['cicids-2018','ton-iot','unsw-nb15']:#

    if dataset_name == "cicids-2018":
        add_order_name = ['DDoS_HOIC','DDoS_LOIC_UDP','DoS_GoldenEye','DoS_Hulk','DoS_Slowloris','SSH_BruteForce','Web_Attack_XSS','Web_Attack_SQL','Web_Attack_Brute_Force',]
        sample_rate = [10000,2527,1000,1000,8490,1000,113,39,131]

    elif dataset_name == "iscx":
        add_order_name  = ['p2p','chat','file_transfer','email']
        sample_rate = [10000,10000,10000,10000]
    elif dataset_name == "ton-iot":
        add_order_name = ['dos','runsomware','backdoor','injection','ddos','password','scanning','xss',]
        sample_rate = [2485,2969,17116,30000,30000,30000,30000,30000,]

    elif dataset_name == "unsw-nb15":
        add_order_name  = ['Worms','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode']
        sample_rate = [2630,2986,30000,30000,30000,30000,30000,7688,]



    weighted_tpr_list_dataset = []
    weighted_fpr_list_dataset = []
    auc_value_list = []
    for model_name in ["proto",'mou','netbeacon','planter','iisy']:
        if model_name == "proto":
            df = pd.read_csv("/home/zhuyijia/prototypical/V6_complete/exp_threshold_proto.csv",header=0)
        elif model_name == "mou":
            df = pd.read_csv("/home/zhuyijia/prototypical/Mousika/Distillation/C2_ROC_mou.csv",header=0)
        elif model_name == "netbeacon":
            df = pd.read_csv("/home/zhuyijia/prototypical/V5_DT_ToN-IoT/C2_ROC_0.8.csv",header=0)
        elif model_name == "planter":
            df = pd.read_csv("/home/zhuyijia/prototypical/V5_DT_ToN-IoT/C2_ROC_0.8.csv",header=0)
        elif model_name == "iisy":
            df = pd.read_csv("/home/zhuyijia/prototypical/V5_DT_ToN-IoT/C2_ROC_0.8.csv",header=0)
    
        df_dataset = df[df['dataset']==dataset_name]

        if model_name in ['netbeacon','planter','iisy']:
            df_dataset = df_dataset[df_dataset['model']==model_name]
        
        # 获取所有唯一的choose_threshold
        unique_thresholds = df_dataset['choose_threshold'].unique()

        weighted_tpr_list = []
        weighted_fpr_list = []
        score_max = -999
        # 对每个choose_threshold进行处理
        for threshold in unique_thresholds:
            # 筛选出相同choose_threshold下的数据
            df_threshold = df_dataset[np.isclose(df_dataset['choose_threshold'],threshold)]
            
            # 筛选出add_order不为ALL的数据，进行加权TPR计算
            df_tpr = df_threshold[df_threshold['add_order'] != 'ALL']
            # print(threshold, model_name,dataset_name)

            # 可能多跑了几次，有重复项
            df_tpr = df_tpr.drop_duplicates(subset=['add_order'], keep='first')
            # print(df_tpr)
            # print(len(df_tpr))
            if len(df_tpr) != len(sample_rate):
                print(model_name)
                print(dataset_name,add_order_name,threshold)
                continue

            # 假设我们使用随意的权重，这里可以自定义
            weighted_tpr = np.average(df_tpr['choose_TPR'], weights=sample_rate)
            weighted_fpr = np.average(df_tpr['choose_FPR'], weights=sample_rate)

            # if weighted_tpr>60 and weighted_fpr<25 and model_name=="proto" and dataset_name=="ton-iot":
                # print(add_order_name,threshold,weighted_fpr,weighted_tpr)
                # exit()
            # score = weighted_tpr * (ALL_acc > 80) + tpr_rate * ALL_acc
            # if score > score_max:
            #     score_max = score
            #     choose_cls_threshold = threshold
            # 保存加权TPR
            weighted_tpr_list.append(weighted_tpr)
            # print(df_threshold)
            weighted_fpr_list.append(weighted_fpr)
            weighted_tpr_list.append(0)
            weighted_fpr_list.append(0)
            weighted_tpr_list.append(100)
            weighted_fpr_list.append(100)

        weighted_fpr_list ,weighted_tpr_list= sort_a_b(weighted_fpr_list,weighted_tpr_list,)

        auc_value_list.append(auc(np.array(weighted_fpr_list)/100, np.array(weighted_tpr_list)/100))

        weighted_tpr_list_dataset.append(weighted_tpr_list)
        weighted_fpr_list_dataset.append(weighted_fpr_list)

    bar_plot(weighted_fpr_list_dataset,weighted_tpr_list_dataset,dataset_name)
    auc_value_list_data[dataset_name] = auc_value_list

auc_value_list_data.to_csv("./AUC.csv",index=None)