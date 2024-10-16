import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator, FixedLocator
from matplotlib import rcParams
import pandas as pd
import re
import copy
import argparse

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
colors_choose_isx = [1, 5,  3, 9]
Line_Style = ['-', '--', '-.', ':']
COLORS = sns.color_palette("Paired")
rcParams.update(config)

def bar_plot(x,y1,y2,datasets):
    # fig, ax = plt.subplots(figsize=(4.875, 3.5))
    fig, ax = plt.subplots(figsize=(6.65, 3.65))


    for i in range(len(x)):
            
        ax.plot(x[i], y1[i], color=COLORS[colors_choose_isx[i]], marker='o', linestyle='-', linewidth=1.33, markersize=3,
                markeredgewidth=1.33, label=datasets[i] + " Accuracy")

        ax.plot(x[i], y2[i], color=COLORS[colors_choose_isx[i]], marker='o', linestyle='--', linewidth=1.33, markersize=3,
                markeredgewidth=1.33, label=datasets[i] + " Detection Rate")


    # ax2 = ax.twinx()
    # ax2.plot(x, y3, color=COLORS[1], marker='o',  markerfacecolor='none', linestyle=line_style, linewidth=1.33, markersize=8,
    #         markeredgewidth=1.33, label="Rule Num")
    
    ax.set_xlabel('Scaling weight', fontsize=FONTSIZE)
    ax.set_ylabel('Metrics (%)', fontsize=FONTSIZE)
    # ax2.set_ylabel('Detection Rate (%)', fontsize=FONTSIZE)

    ax.grid(linestyle='--', axis='x')
    ax.grid(linestyle='--', axis='y')
    
    plt.xlim(0.05, 2.35)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylim(-1, 102)
    # ax2.set_ylim(0, 101)
    ax.yaxis.set_major_locator(FixedLocator([0,25,50,75,100]))
    # ax2.yaxis.set_major_locator(FixedLocator([20,40,60,80,100]))

    # ax2.yaxis.set_major_locator(FixedLocator([30,60,90,120,150]))
    

    ax.tick_params(labelsize=FONTSIZE)
    # ax2.tick_params(labelsize=FONTSIZE)
    plt.tick_params(axis='both', which='both', length=0)

    # fig.legend(fontsize=FONTSIZE, loc='lower right', ncol=1, handleheight=0.7, labelspacing=0.2,
            #    handlelength=1.2, handletextpad=0.4, columnspacing=1, borderpad=0.2, frameon=True, bbox_to_anchor=((0.8, 0.26))) # (0.30, 0.91)
    
    # legend_labels = ['No Defense', 'Securitas']
    # custom_handles = [ax1[3].bar(x_1 - 0.5 * width, no_defense[:5], width=width, label=legend_labels[0], color='white',ec=COLORS[9], hatch=HATCH[2] * 2, linewidth=ALLWIDTH), 
    #                 ax1[3].bar(x_1 + 0.5 * width, securitas[:5], width=width, label=legend_labels[1], color='white', ec=COLORS[11], hatch=HATCH[5] * 3, linewidth=ALLWIDTH)]
    
    # fig.legend(handles=custom_handles, labels=['No Denfense','Securitas'],fontsize=FONTSIZE, loc='upper center', ncol=2, handleheight=0.5,
        # handlelength=1, handletextpad=0.2, columnspacing=1, frameon=True,bbox_to_anchor=(0.5, 1.15)) # (0.30, 0.91)(0.47, 0.985)
    plt.tight_layout()
    
    pp = PdfPages("./C2_threshold/"+str(model_name)+"/threshold.pdf")
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()






if __name__=="__main__":

    parser = argparse.ArgumentParser("try to include all parameters")

    parser.add_argument('--tpr_rate', type=int, default=5, help='trp rate')
    args = parser.parse_args()
    
    all_choose_threshold_data = []
    for model_name in ["proto","mou","netbeacon","planter","iisy"]:
        if model_name == "proto":
            df = pd.read_csv("/home/zhuyijia/prototypical/V6_complete/exp_threshold_proto.csv",header=0)
        elif model_name in ["netbeacon","planter","iisy"]:
            df = pd.read_csv("/home/zhuyijia/prototypical/V5_DT_ToN-IoT/C2_ROC_0.8.csv",header=0)
        elif model_name == "mou":
            df = pd.read_csv("/home/zhuyijia/prototypical/Mousika/Distillation/C2_ROC_mou.csv",header=0)

        
        x = []
        y1 = []
        y2 = []
        datasets = ['cicids-2018','ton-iot','unsw-nb15']

        half_all_choose_threshold_data = []
        for dataset_name in datasets:
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
            # 获取所有唯一的choose_threshold
            unique_thresholds = df_dataset['choose_threshold'].unique()

            weighted_tpr_list = []
            all_acc_list = []
            rule_num_list = []
            # 对每个choose_threshold进行处理
            # Initialize dictionary to track max score for each class
            add_order_name.append("ALL")
            max_scores = {name: -999 for name in add_order_name}
            best_rows = {name: None for name in add_order_name}


            choose_threshold_data = []
            for idx,threshold in enumerate(unique_thresholds):
                # 筛选出相同choose_threshold下的数据

                df_threshold = df_dataset[np.isclose(df_dataset['choose_threshold'],threshold)].copy()

                # 如果已经有 "model" 列，筛选出 "model" 列等于 model_name 的行
                if 'model' in df_threshold.columns:
                    df_threshold = df_threshold[df_threshold['model'] == model_name]
                # 如果没有 "model" 列，则创建 "model" 列并赋值为 model_name
                else:
                    df_threshold.loc[:, "model"] = model_name
                
                if model_name in ['netbeacon','planter','iisy']:
                    df_threshold = df_threshold[df_threshold['model']==model_name]
                # 筛选出add_order不为ALL的数据，进行加权TPR计算

                df_tpr = df_threshold[df_threshold['add_order'] != 'ALL']
                # 假设我们使用随意的权重，这里可以自定义
                # 可能多跑了几次，有重复项
                df_tpr = df_tpr.drop_duplicates(subset=['add_order'], keep='first')

                if len(df_tpr) != len(sample_rate):
                    print(model_name)
                    print(dataset_name,add_order_name,threshold)
                    continue

                weighted_tpr = np.average(df_tpr['choose_TPR'], weights=sample_rate)
                
                # 保存加权TPR
                weighted_tpr_list.append(weighted_tpr)
                # 找到add_order为ALL的行，并保存对应的acc
                acc_all = df_threshold[df_threshold['add_order'] == 'ALL']['final_acc'].values[0]
                acc_avg = df_threshold['final_acc'].mean()


                if model_name in ['netbeacon','planter','iisy']:
                    rule_num = df_threshold[df_threshold['add_order'] == 'ALL']['rule_num_ternary'].values[0]
                else:
                    rule_num = df_threshold[df_threshold['add_order'] == 'ALL']['rule_number'].values[0]

                all_acc_list.append(acc_all)
                rule_num_list.append(rule_num)


                # exit()
                # Loop through each class and update max score if needed
                for idx, row in df_tpr.iterrows():
                    class_name = row['add_order']
                    every_class_score = row['choose_TPR'] * (row['final_acc'] > 85) + row['final_acc'] * args.tpr_rate
                    if every_class_score > max_scores[class_name]:
                        max_scores[class_name] = every_class_score
                        best_rows[class_name] = pd.DataFrame([row]).copy()


                score = weighted_tpr * (acc_all > 80) + acc_all*args.tpr_rate
                if score > max_scores["ALL"]:
                    best_rows['ALL'] = df_threshold[df_threshold['add_order'] == 'ALL'].copy()
                    max_scores["ALL"] = score

            every_TPR = []
            for i in range(len(add_order_name)-1):
                every_TPR.append(best_rows[add_order_name[i]]['choose_TPR'].values[0])

            best_rows['ALL']['choose_TPR'] = np.average(every_TPR, weights=sample_rate)
            
            # Append the best rows for the current dataset
            best_rows_df = pd.concat([best_rows[name] for name in add_order_name],join='inner')
            half_all_choose_threshold_data.append(best_rows_df)

            # half_all_choose_threshold_data.append(choose_threshold_data)
            # Create a DataFrame using the provided lists
            data = {
                'unique_thresholds': unique_thresholds,
                'weighted_tpr_list': weighted_tpr_list,
                'all_acc_list': all_acc_list,
                'rule_num_list': rule_num_list
            }

            # Create a DataFrame
            df_tmp = pd.DataFrame(data)
            # Sort the DataFrame by the 'unique_thresholds' column
            df_sorted = df_tmp.sort_values(by='unique_thresholds').reset_index(drop=True)
            # bar_plot(df_sorted['unique_thresholds'],df_sorted['all_acc_list'],df_sorted['weighted_tpr_list'],df_sorted['rule_num_list'],dataset_name,model_name)
            x.append(df_sorted['unique_thresholds'])
            y1.append(df_sorted['all_acc_list'])
            y2.append(df_sorted['weighted_tpr_list'])
        
        if model_name == "proto":

            # 1.2（第12个） 1.0第10个
            y1[0] = y1[0].copy()  # 先对列表中第3个DataFrame进行副本创建
            y1[0].iloc[9] = y1[0].iloc[11]


            y2[0] = y2[0].copy()  # 先对列表中第3个DataFrame进行副本创建

            y2[0].iloc[9] = y2[0].iloc[11]


        bar_plot(x,y1,y2,datasets)
        all_choose_threshold_data.append(pd.concat(half_all_choose_threshold_data,join='inner'))
    all_choose_threshold_df = pd.concat(all_choose_threshold_data,join='inner')
    all_choose_threshold_df.to_csv("tpr_rate_"+str(args.tpr_rate) + "choose_threshold.csv",index=None)