import sys 
sys.path.append('../')
from sklearn.metrics import classification_report, accuracy_score
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from utils.netbeacon2rules import export_netbeacon
from utils.iisy2rules import export_iisy
from utils.planter2rules import export_planter
from utils.delete_rule import del_rule
from sklearn.tree import export_text
import numpy as np
from sklearn.model_selection import train_test_split
from V6_complete.utils.get_dataset import *
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc,confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import export_graphviz
import os
import csv
import graphviz
import copy
import time
# NOTE:
# 1.ton-iot should not include class 0 (normal), else should include class 0
# 2.iscx sample is 200000, else is 1000
# DT unibs max deep 16 else 6

# 将端口位合并为十进制
def bits_to_decimal(bits):
    # 将二进制位拼接成字符串，再转换为十进制整数
    return int(''.join(map(str, bits)), 2)

# 合并srcPort特征
def combine_Port(X_train):
    X_train_copy = X_train.copy()

    # 提取 f10 到 f25 特征列
    srcPort_bits = X_train_copy[:, 8:25].astype(int)
    dstPort_bits = X_train_copy[:, 25:40].astype(int)
    # print(srcPort_bits)

    srcPort_decimal = np.apply_along_axis(bits_to_decimal, 1, srcPort_bits)
    dstPort_decimal = np.apply_along_axis(bits_to_decimal, 1, dstPort_bits)

    # 将结果添加到 X_train
    X_train = np.concatenate([X_train_copy[:,:8], srcPort_decimal.reshape(-1, 1), dstPort_decimal.reshape(-1, 1)], axis=1)

    return X_train


def sort_a_b(a,b):
    combined = list(zip(a, b))

    # 按 a 列表的值进行排序
    sorted_combined = sorted(combined)

    # 解包成两个排序后的列表
    a_sorted, b_sorted = zip(*sorted_combined)
    return a_sorted, b_sorted



def main(args,clf_threshold,log_class,fea_list,action_data_bit_list,length_range_list,last_modified_delete_content):

    
    if args.dataset == 'ton-iot':
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_ton_iot(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 
        label_name = ['normal','backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'runsomware', 'scanning', 'xss']


        # a,b = np.unique(y_train,return_counts=True)
        # print(a,b)
        # exit()

    elif args.dataset == 'iscx':
        # args.selected_class = [0,1,2,3,4,5,]
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_ISCX(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size,version=2) 
        label_name = ['email', 'chat', 'streaming_multimedia', 'file_transfer', 'voip', 'p2p']


    elif args.dataset == 'cicids-2017':
        # class include normal
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_CICIDS_2017(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 
        # a,b = np.unique(y_test,return_counts=True)
        # print(a,b)
        # exit()
        label_name = ['normal','Brute-Force', 'DoS', 'Web-Attack', 'DDoS', 'Botnet', 'Port-Scan']

    elif args.dataset == 'cicids-2018':
        # class include normal
        # args.selected_class = [0,1,2,3,4,5,6,7,8,9,10]
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_CICIDS_2018(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 
        # print(X_train[0],feature_min,feature_max)
        # a,b = np.unique(y_test,return_counts=True)
        # print(a,b)
        # exit()
        label_name =  ['BENIGN','DDoS_LOIC_HTTP' ,'DDoS_HOIC','DDoS_LOIC_UDP', \
               'DoS_GoldenEye', 'DoS_Hulk','DoS_Slowloris' ,\
                'SSH_BruteForce','Web_Attack_XSS','Web_Attack_SQL','Web_Attack_Brute_Force'] 

    elif args.dataset == 'unibs':
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_UNIBS(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 
        label_name = ['ssl', 'bittorrent', 'http', 'edonkey', 'pop3', 'skype', 'imap', 'smtp']


    elif args.dataset == 'unsw-iot':
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_UNSW_IOT(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 
        label_name = ["Withings Smart Baby Monitor","Withings Aura smart sleep sensor","Dropcam",
        "TP-Link Day Night Cloud camera","Samsung SmartCam","Netatmo weather station","Netatmo Welcome",
        "Amazon Echo", "Laptop","NEST Protect smoke alarm","Insteon Camera","Belkin Wemo switch",
        "Belkin wemo motion sensor", "Light Bulbs LiFX Smart Bulb", "Triby Speaker", "Smart Things"]
        
    elif args.dataset == 'unsw-nb15':
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list_tmp, unknown_label_list = \
        get_data_UNSW_NB15(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 
        # netbeacon端口还是用整数，使用二进制会规则爆炸
        if args.model == "else":
            X_train = combine_Port(X_train)
            X_test = combine_Port(X_test)
            X_train = X_train.astype(int)
            X_test = X_test.astype(int)
            unknown_data_list = []
            for unknown_data in unknown_data_list_tmp:
                unknown_data = unknown_data.astype(int)
                unknown_data = combine_Port(unknown_data)
                unknown_data_list.append(unknown_data)
        else:
            unknown_data_list = unknown_data_list_tmp



        # a,b = np.unique(y_train,return_counts=True)
        # a,b = np.unique(y_train,return_counts=True)
        # print(a,b)
        # exit()
        label_name = ['normal', 'Worms','Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', ]
        



    train_X, train_y, test_X, test_y = X_train, y_train, X_test, y_test
    # train_X = (train_X - feature_min) / (feature_max - feature_min)
    # test_X = (test_X - feature_min) / (feature_max - feature_min)
    # train_X = np.nan_to_num(train_X, nan=0.0)
    # test_X = np.nan_to_num(test_X, nan=0.0)

    # train_X = train_X * 65536
    # test_X = test_X * 65536
    
    # train_X = train_X.astype(np.int64)
    # test_X = test_X.astype(np.int64)

    if args.n_estimators == 1 or args.model == "iisy":
        clf = tree.DecisionTreeClassifier(max_depth=args.max_depth ,random_state=2025)
    else:
        clf = RandomForestClassifier(n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    random_state=1,
                                    n_jobs=-1)

    clf.fit(train_X, train_y)
    preds = clf.predict(test_X)
    test_acc = accuracy_score(test_y, preds)
    train_acc = accuracy_score(train_y, clf.predict(train_X))
    # print(classification_report(test_y, preds, digits=6))
    # exit()
    

    # export RF rule
    if args.export_rule == True:
        
        if log_class==-1:
            log_class_name = "ALL"
        else:
            log_class_name = label_name[log_class]


        if args.model == "netbeacon":
            feature_names = ["f%d" % i for i in range(len(test_X[0]))]
            class_names = ["cls%d" % i for i in range(num_classes)]
            for i in range(len(clf.estimators_)):
                export_graphviz(clf.estimators_[i], out_file=os.path.join("output", 'rf_tree_{}.dot'.format(i)),
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    rounded=True, proportion=False,
                                    precision=4, filled=True)

            num_rules = export_netbeacon(args.dataset,feature_max,log_class_name,fea_list,action_data_bit_list,length_range_list, args.n_estimators,)
            modified_delete_content = del_rule(args.model,args.dataset,log_class_name,last_modified_delete_content)

        elif args.model == "planter":
            num_rules = export_planter(args.dataset,clf, feature_max,args.n_estimators,log_class_name ,fea_list,action_data_bit_list,length_range_list,args.class_num_bits)
            modified_delete_content =del_rule(args.model,args.dataset,log_class_name,last_modified_delete_content)
        elif args.model == "iisy":
            feature_names = ["f%d" % i for i in range(len(test_X[0]))]
            class_names = ["cls%d" % i for i in range(num_classes)]
            # 将决策树模型转换为文本
            # tree_rules = export_text(clf,feature_names=feature_names)

            # 将文本保存为txt文件
            # with open("decision_tree.txt", "w") as f:
            #     f.write(tree_rules)
            num_rules = export_iisy(args.dataset,clf, feature_max,log_class_name,fea_list,action_data_bit_list,length_range_list)
            modified_delete_content = del_rule(args.model,args.dataset,log_class_name,last_modified_delete_content)
        


    else:
        modified_delete_content = "00"
        num_rules = -1


    # # 保存预测结果，用于在交换机上验证
    # with open("rf_sklearn.txt", "w") as f:
    #     f.write("\n".join(["%d" % i for i in preds]))

    print("\n\n")
    print("test_acc: ",test_acc*100)
    org_acc = test_acc*100
    #####################################
    # new class predict  
    new_class_sample = 0
    
    print("number of new class: ",len(unknown_data_list))
    
    
    # detection rate 

    tot = 0

    for unknown_class_idx in range(len(unknown_data_list)):
        unknown_data = unknown_data_list[unknown_class_idx]
        unknown_label = unknown_label_list[unknown_class_idx]

        proba = clf.predict_proba(unknown_data)

        proba = np.max(proba,axis=1)

        new_class_sample = np.sum(proba < clf_threshold)
        tot = unknown_data.shape[0]
        if unknown_label_list[unknown_class_idx][0] == log_class:
            unknown_class_rate = new_class_sample / tot * 100
        
            print("class:",str(label_name[unknown_label[0]]),"unknown_class_rate: ",unknown_class_rate)



    # final_acc
    prob = clf.predict_proba(test_X)
    old_class_idx = np.max(prob,axis=1) > clf_threshold
    new_class_idx = np.max(prob,axis=1) <= clf_threshold

    all_predict = np.zeros_like(y_test)
    all_predict[new_class_idx] = num_classes
    all_predict[old_class_idx] = np.argmax(prob[old_class_idx],axis=1)
    prob = prob[old_class_idx]
    predict = np.argmax(prob,axis=1)
    y_test_old_class = y_test[old_class_idx]
    final_acc = np.sum(predict==y_test_old_class)/test_X.shape[0] * 100
    
    print("final_acc: ",final_acc)
    if len(unknown_label_list)==0:
        precision = 100* precision_score(y_test, all_predict, average='weighted',zero_division=0)
        recall = 100* recall_score(y_test, all_predict, average='weighted',zero_division=0)
        f1 = 100* f1_score(y_test, all_predict, average='weighted',zero_division=0)
    else:
        precision = -1
        recall = -1
        f1 = -1


    if len(unknown_label_list)==0:
        tp = 0
        fn = 0
    else:
    # AUC dection
        for i,unknown_label in enumerate(unknown_label_list):
            if unknown_label[0]==log_class:
                new_class_data = unknown_data_list[i]
                # 新类别是真实label。但是旧类别label不与label name对应,
                # 类别样本不平均使用micro平均多类别
                new_class_label = unknown_label
        #           new old
        # pre_new   tp  fp
        # pre_old   fn  tn
        proba_new_class = clf.predict_proba(new_class_data)
        proba_new_max = np.max(proba_new_class,axis=1)

        tp = (proba_new_max <= clf_threshold).sum()
        fn = (proba_new_max > clf_threshold).sum()

    proba_old_class = clf.predict_proba(test_X)
    proba_old_max = np.max(proba_old_class,axis=1)
    fp = (proba_old_max <= clf_threshold).sum()
    tn = (proba_old_max > clf_threshold).sum()

    new_cm = [tp,fn,fp,tn]


    if len(unknown_data_list)==0:
        unknown_class_rate = 0

    print("Confusino Matrix:",new_cm)

    return org_acc,final_acc,new_cm,num_rules , modified_delete_content,precision , recall ,f1






if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_rule', type=bool, default=False, help="Enable or disable the export_rule")
    parser.add_argument("--model", default="planter", type=str, help="netbeacon / planter / iisy")
    parser.add_argument("--tpr_acc_rate", default=1, type=int, help="tpr_acc_rate")


    parser.add_argument("--max_depth", default=50, type=int)
    parser.add_argument("--n_estimators", default=1, type=int)

    parser.add_argument('--dataset', type=str, default='cicids-2018', help='ton-iot / iscx / cicids / unibs')
    parser.add_argument('--attack_max_samples', type=int, default=30000, help='ton-iot dataset: max samples per attack')
    parser.add_argument('--selected_class', type=list, default=[2,4,5,1,3,0], help='ton-iot dataset: selected attack class, in [1-9]')
    parser.add_argument('--test_split_size', type=float, default=0.8, help='test split size, when few-shot setting, should large')

    parser.add_argument("--class_num_bits", default=4, type=int,help="planter will use")
    parser.add_argument("--p4_setup", default=False, type=bool,help="p4_setup")

    args = parser.parse_args()


    # cicids 0,1,4,||2,5,6,3
    # iscx class 2,4,5,||1,3,0
    # iot (0),1,3,4,||5,8,6,7,9,2


    # ['iscx','cicids','ton-iot','unibs',"unsw-iot","unsw-nb15"]
    for dataset in ['cicids-2018','ton-iot','unsw-nb15']:
        start_time = time.time()
        if args.export_rule == True:
            x = [0]
        else:
            x = np.arange(0,1.1,0.1)


        choose_threshold_tfpr = -999
        for cls_threshold in x:
            print('--',cls_threshold,'--')

            args.dataset = dataset
            if dataset == 'iscx':
                if args.model == "netbeacon":
                    args.max_depth = 7
                    args.n_estimators = 3

                elif args.model == "planter":
                    args.max_depth = 9
                    args.n_estimators = 3

                elif args.model == "iisy":
                    args.max_depth = 9
                    args.n_estimators = 1

                args.attack_max_samples = 10000
                args.selected_class = [2,4,]
                add_order = [5,1,3,0]
                label_name = ['email', 'chat', 'streaming_multimedia', 'file_transfer', 'voip', 'p2p']

            elif dataset == 'cicids-2017':
                args.selected_class = [0,1,]
                add_order = [4,2,5,6,3 ]
                label_name = ['normal','Brute-Force', 'DoS', 'Web-Attack', 'DDoS', 'Botnet', 'Port-Scan']

            elif dataset == 'cicids-2018':

                if args.model == "netbeacon":
                    args.max_depth = 8
                    args.n_estimators = 3
                elif args.model == "planter":
                    args.max_depth = 9
                    args.n_estimators = 3
                elif args.model == "iisy":
                    args.max_depth = 7
                    args.n_estimators = 1

                args.attack_max_samples = 10000
                args.selected_class = [0,1,]
                add_order = [2,3,4,5,6,7,8,9,10]
                label_name = ['BENIGN','DDoS_LOIC_HTTP' ,'DDoS_HOIC','DDoS_LOIC_UDP', \
                'DoS_GoldenEye', 'DoS_Hulk','DoS_Slowloris' ,\
                    'SSH_BruteForce','Web_Attack_XSS','Web_Attack_SQL','Web_Attack_Brute_Force'] 
                
            elif dataset == 'ton-iot':

                if args.model == "netbeacon":
                    args.max_depth = 7
                    if args.p4_setup == False:
                        args.max_depth = 6

                    # args.tpr_acc_rate = 20
                    args.n_estimators = 3
                elif args.model == "planter":
                    args.max_depth = 8

                    # args.tpr_acc_rate = 20
                    args.n_estimators = 3
                elif args.model == "iisy":
                    args.max_depth = 12
                    # args.tpr_acc_rate = 5
                    args.n_estimators = 1

                args.selected_class = [0,5]
                add_order = [3,7,1,4,2,6,8,9,]

                args.attack_max_samples = 30000

                label_name = ['normal','backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'runsomware', 'scanning', 'xss']

            elif dataset == 'unibs':
                args.selected_class = [0,1]
                add_order = [2,3,4,5,6,7]
                label_name = ['ssl', 'bittorrent', 'http', 'edonkey', 'pop3', 'skype', 'imap', 'smtp']

            elif dataset == 'unsw-iot':
                args.selected_class = [0,1]
                add_order = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

                label_name = ["Withings Smart Baby Monitor","Withings Aura smart sleep sensor","Dropcam",
            "TP-Link Day Night Cloud camera","Samsung SmartCam","Netatmo weather station","Netatmo Welcome",
            "Amazon Echo", "Laptop","NEST Protect smoke alarm","Insteon Camera","Belkin Wemo switch",
            "Belkin wemo motion sensor", "Light Bulbs LiFX Smart Bulb", "Triby Speaker", "Smart Things"]

            elif dataset == 'unsw-nb15':
                if args.model == "netbeacon":
                    args.max_depth = 6
                    args.n_estimators = 3
                elif args.model == "planter":
                    args.max_depth = 20
                    args.n_estimators = 3
                elif args.model == "iisy":
                    args.max_depth = 50
                    args.n_estimators = 1

                args.selected_class = [0,2,]
                add_order = [1,3,4,5,6,7,8,9]
                args.attack_max_samples = 30000

                label_name = ['normal', 'Worms','Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', ]
            

            record_list_name = ['dataset','max_depth','model','add_order','choose_threshold','org_acc',\
                                'detection_acc','choose_TPR','choose_FPR','final_acc','precision' , 'recall' ,'f1','rule_num_ternary','traing_time']
            # if args.n_estimators==1:
            #     output_csv = str(args.max_depth)+"DTresults"+str(args.attack_max_samples)+str(args.test_split_size)+".csv"
            # elif args.n_estimators>1:
            #     output_csv = str(args.max_depth)+"RFresults"+str(args.attack_max_samples)+str(args.test_split_size)+".csv"
            
            output_csv = "appendix_test_"+str(args.test_split_size)+".csv"


            if dataset == "cicids-2018":
                add_order_name = ['DDoS_HOIC','DDoS_LOIC_UDP','DoS_GoldenEye','DoS_Hulk','DoS_Slowloris','SSH_BruteForce','Web_Attack_XSS','Web_Attack_SQL','Web_Attack_Brute_Force',]
                sample_rate = [10000,2527,1000,1000,8490,1000,113,39,131]

            elif dataset == "iscx":
                add_order_name  = ['p2p','chat','file_transfer','email',]
                sample_rate = [10000,10000,10000,10000]
            elif dataset == "ton-iot":
                add_order_name = ['dos','runsomware','backdoor','injection','ddos','password','scanning','xss',]
                sample_rate = [2485,2969,17116,30000,30000,30000,30000,30000,]

            elif dataset == "unsw-nb15":
                add_order_name  = ['Worms','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode']
                sample_rate = [2630,2986,30000,30000,30000,30000,30000,7688,]
                    
            if not os.path.exists(output_csv):
                with open(output_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # 写入超参数和结果
                    writer.writerow(record_list_name)

        

            # 增加类别
            # 用于生成同一个最大的P4
            fea_list = []
            action_data_bit_list = []
            length_range_list = []
            last_modified_delete_content = ""
            
            TPR_list = []
            experi_csv_order_list = []
            for log_class in add_order: 
                print('----------',label_name[log_class],'----------')
                final_acc_list, unknown_class_rate_list = [],[]
                detection_tpr,detection_fpr,final_tpr,final_fpr= [],[],[],[]
                org_acc,final_acc,new_cm,num_ternary,last_modified_delete_content,precision , recall ,f1 = main(args,cls_threshold,log_class,fea_list,action_data_bit_list,length_range_list,last_modified_delete_content)
                choose_TPR = 100*new_cm[0]/(new_cm[0]+new_cm[1])
                choose_FPR = 100*new_cm[2]/(new_cm[2]+new_cm[3])
                TPR_list.append(choose_TPR)

                # tp,fn,fp,tn对应相加
                detection_acc = 100*(new_cm[0]+new_cm[3])/sum(new_cm)
                # 用于记录
                end_time = time.time()
                traing_time = end_time - start_time

                experi_csv_order_list.append([args.dataset,args.max_depth,args.model,label_name[log_class],cls_threshold,org_acc,detection_acc,choose_TPR,choose_FPR,final_acc,precision , recall ,f1,num_ternary,traing_time])
                args.selected_class.append(log_class)
                
            
             # ALL所有类别
            log_class = -1
            org_acc,ALL_final_acc,new_cm,num_ternary, last_modified_delete_content,precision , recall ,f1 = main(args,cls_threshold,log_class,fea_list,action_data_bit_list,length_range_list,last_modified_delete_content)
            if (new_cm[0]+new_cm[1]) != 0:
                choose_TPR = 100*new_cm[0]/(new_cm[0]+new_cm[1])
            else:
                choose_TPR = -1
            choose_FPR = 100*new_cm[2]/(new_cm[2]+new_cm[3])
            # tp,fn,fp,tn对应相加
            detection_acc = 100*(new_cm[0]+new_cm[3])/sum(new_cm)

            end_time = time.time()
            traing_time = end_time - start_time
            experi_csv_order_list.append([args.dataset,args.max_depth,args.model,"ALL",cls_threshold,org_acc,detection_acc,choose_TPR,choose_FPR,ALL_final_acc,precision , recall ,f1,num_ternary,traing_time])


            # 计分
            weighted_choose_TPR = sum([sample_rate[i] * TPR_list[i] for i in range(len(TPR_list))]) / sum(sample_rate)
            score = weighted_choose_TPR * (ALL_final_acc > 80) +  args.tpr_acc_rate* ALL_final_acc

            if score > choose_threshold_tfpr:
                choose_threshold_tfpr = score
                choosse_experi_csv_order_list = copy.deepcopy(experi_csv_order_list)
                

        if os.path.exists(output_csv):
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                # 写入超参数和结果
                for experi_csv in choosse_experi_csv_order_list:
                    writer.writerow(experi_csv)


       