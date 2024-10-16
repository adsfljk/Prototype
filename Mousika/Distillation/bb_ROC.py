# encoding: utf-8
import sys 
sys.path.append('../../')

import warnings
warnings.filterwarnings('ignore')
import imp
from Tree import *
from utils import *
from RandomForest import *
import torch
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import argparse
from train_teacher import *
from SoftTree import *
from models import *
import time
from sklearn.metrics import classification_report,roc_curve,auc,confusion_matrix,accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

import torch.nn.functional as F
from V6_complete.utils.get_dataset import *
from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz
import csv
import os
import re

seed = 2025
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def parse_rules(file_path):
    rules = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'then' in line:
                conditions, label = line.split(' then ')
                label = int(label.strip())
                # 使用正则表达式解析条件
                conditions = re.findall(r'feature_(\d+)([=!]+)([0-9.]+)', conditions)
                parsed_conditions = [(int(feature), operation, float(value)) for feature, operation, value in conditions]
                rules.append((parsed_conditions, label))
    return rules


# 检查样本是否满足规则条件
def match_rule(conditions, sample):
    for feature, operation, value in conditions:
        sample_value = sample[feature]
        if operation == "!=" and sample_value == value:
            return False
        elif operation == "=" and sample_value != value:
            return False
    return True

# 根据规则进行预测
def rule_predict(rules, X):
    predictions = []
    for sample in X:
        predicted = None
        for conditions, label in rules:
            if match_rule(conditions, sample):
                predicted = label
                break
        if predicted is None:
            predicted = -1  # 如果没有匹配到任何规则，返回默认值
        predictions.append(predicted)
    return predictions

# 计算准确率
def calculate_accuracy(y_true, y_pred):
    correct = np.mean(np.array(y_true) == np.array(y_pred))
    return correct *100



def softmax(x):
    """
    对输入x的每一行计算softmax。

    该函数对于输入是向量（将向量视为单独的行）或者矩阵（M x N）均适用。

    代码利用softmax函数的性质: softmax(x) = softmax(x + c)

    参数:
    x -- 一个N维向量，或者M x N维numpy矩阵.

    返回值:
    x -- 在函数内部处理后的x
    """
    orig_shape = x.shape

    # 根据输入类型是矩阵还是向量分别计算softmax
    if len(x.shape) > 1:
        # 矩阵
        tmp = np.max(x, axis=1)  # 得到每行的最大值，用于缩放每行的元素，避免溢出。 shape为(x.shape[0],)
        x -= tmp.reshape((x.shape[0], 1))  # 利用性质缩放元素
        x = np.exp(x)  # 计算所有值的指数
        tmp = np.sum(x, axis=1)  # 每行求和
        x /= tmp.reshape((x.shape[0], 1))  # 求softmax
    else:
        # 向量
        tmp = np.max(x)  # 得到最大值
        x -= tmp  # 利用最大值缩放数据
        x = np.exp(x)  # 对所有元素求指数
        tmp = np.sum(x)  # 求元素和
        x /= tmp  # 求somftmax
    return x

def min_max_normalize(data,minf,maxf):

    # 检查每个特征列
    for i in range(data.shape[1]):
        min_val = minf[i]
        max_val = maxf[i]
        # 处理特征只有一个值的情况 
        if min_val == max_val:
            if min_val == 0:  # 如果特征值为0，将其归一化为0
                data[:, i] = 0.0
            else:  # 否则将其归一化为1
                data[:, i] = 1.0
        else:
            # 正常的min-max归一化
            data[:, i] = (data[:, i] - min_val) / (max_val - min_val)
    
    return data

def sort_a_b(a,b):
    combined = list(zip(a, b))

    # 按 a 列表的值进行排序
    sorted_combined = sorted(combined)

    # 解包成两个排序后的列表
    a_sorted, b_sorted = zip(*sorted_combined)
    return a_sorted, b_sorted



def unsw_nb15_to_binary(X_train):
    header_names = {
        'f0': ['hdr.ipv4.protocol', 8],
        'f1': ['hdr.ipv4.flags', 3],
        'f2': ['hdr.ipv4.ttl', 8],
        'f3': ['hdr.ipv4.totalLen', 16],
        'f4': ['meta.dataOffset', 4],
        'f5': ['meta.flags', 8],
        'f6': ['meta.window', 16],
        'f7': ['meta.udp_length', 16],
        'f8': ['meta.srcPort_0', 1],
        'f9': ['meta.srcPort_1', 1],
        'f10': ['meta.srcPort_2', 1],
        'f11': ['meta.srcPort_3', 1],
        'f12': ['meta.srcPort_4', 1],
        'f13': ['meta.srcPort_5', 1],
        'f14': ['meta.srcPort_6', 1],
        'f15': ['meta.srcPort_7', 1],
        'f16': ['meta.srcPort_8', 1],
        'f17': ['meta.srcPort_9', 1],
        'f18': ['meta.srcPort_10', 1],
        'f19': ['meta.srcPort_11', 1],
        'f20': ['meta.srcPort_12', 1],
        'f21': ['meta.srcPort_13', 1],
        'f22': ['meta.srcPort_14', 1],
        'f23': ['meta.srcPort_15', 1],
        'f24': ['meta.dstPort_0', 1],
        'f25': ['meta.dstPort_1', 1],
        'f26': ['meta.dstPort_2', 1],
        'f27': ['meta.dstPort_3', 1],
        'f28': ['meta.dstPort_4', 1],
        'f29': ['meta.dstPort_5', 1],
        'f30': ['meta.dstPort_6', 1],
        'f31': ['meta.dstPort_7', 1],
        'f32': ['meta.dstPort_8', 1],
        'f33': ['meta.dstPort_9', 1],
        'f34': ['meta.dstPort_10', 1],
        'f35': ['meta.dstPort_11', 1],
        'f36': ['meta.dstPort_12', 1],
        'f37': ['meta.dstPort_13', 1],
        'f38': ['meta.dstPort_14', 1],
        'f39': ['meta.dstPort_15', 1]
    }

    # 初始化空的列表用于存储新特征
    binary_features = []

    # 遍历每个特征
    for i, (key, value) in enumerate(header_names.items()):
        feature_name = value[0]
        bit_width = value[1]
        
        if 'Port_' not in feature_name:
            # 对其他特征进行二进制转换
            feature_values = X_train[:, i].astype(int)
            # 将特征值转换为对应的二进制表示，并展平成位数组
            binary_rep = np.array([list(format(val, f'0{bit_width}b')) for val in feature_values])
            binary_rep = np.array(binary_rep)

            binary_rep = binary_rep.astype(int)  # 转换为int类型
            # 将二进制的每一位作为新的特征
            binary_features.append(binary_rep)
               # 跳过srcPort和dstPort的相关特征
        else:
            binary_features.append(X_train[:, i].reshape(-1, 1))  # 保留这些特征原始格式4



    # 将所有新特征连接起来
    new_X_train = np.hstack(binary_features)
    
    return new_X_train


def count_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines)


def decimal_to_binary(value, bits):
    """
    Converts a decimal value to its binary representation with the specified number of bits.
    """
    binary_str = bin(int(value))[2:]  # Convert to binary and remove the '0b' prefix
    if len(binary_str) > bits:
        raise ValueError(f"Value {value} cannot be represented with {bits} bits")
    # Add leading zeros to match the required bit length
    return binary_str.zfill(bits)


# ISCX
def fea2bit(args, X_train):
    """
    Transforms each feature in X_train to its binary representation based on the specified bit lengths.
    """
    if args.dataset == "iscx":
        header_names = {
        'hdr.ipv4.protocol': 8,
        'hdr.ipv4.ihl': 4,
        'hdr.ipv4.tos': 8,
        'hdr.ipv4.flags': 3,
        'hdr.ipv4.ttl': 8,
        'meta.dataOffset': 4,
        'meta.flags': 8,
        'meta.window': 16,
        'meta.udp_length': 16,
        'hdr.ipv4.totalLen': 16
        }



    trans_X_train = []

    for sample in X_train:
        transformed_sample = []
        for feature_value, (feature_name, bits) in zip(sample, header_names.items()):
            binary_representation = decimal_to_binary(feature_value, bits)
            transformed_sample.extend([int(bit) for bit in binary_representation])
        trans_X_train.append(transformed_sample)

    return np.array(trans_X_train)


def convert_to_binary(X_train, int_bits=16, frac_bits=0):
    def float_to_binary_array(x, int_bits, frac_bits):
        # 分离整数和小数部分
        int_part = int(x)
        frac_part = x - int_part

        # 整数部分转二进制
        int_bin = np.array(list(np.binary_repr(int_part, width=int_bits)), dtype=np.uint8)

        # 小数部分转二进制
        frac_bin = np.zeros(frac_bits, dtype=np.uint8)
        for i in range(frac_bits):
            frac_part *= 2
            bit = int(frac_part)
            frac_bin[i] = bit
            frac_part -= bit
        


        # 将整数部分和小数部分合并
        return np.concatenate([int_bin, frac_bin]).astype(int)

    # 对每个特征进行转换
    X_binary = []
    for row in X_train:
        binary_row = np.concatenate([float_to_binary_array(feature, int_bits, frac_bits) for feature in row])
        # print(len(binary_row))
        # exit()
        X_binary.append(binary_row)

    return np.array(X_binary)




def produce_soft_labels(data, num_classes,round_num, fold_num, k=1,model='rf'):

    soft_label = np.zeros([data.shape[0], num_classes])

    for i in range(round_num):
        kf = KFold(n_splits=fold_num)
        for train_index, test_index in kf.split(X=data[:, :-1], y=data[:, -1], groups=data[:, -1]):
            train_set, test_set = data[train_index], data[test_index]
            train_X,train_Y=train_set[:, :-1],train_set[:, -1].astype(int)
            test_X=test_set[:, :-1]
            if model=='rf':
                clf = RandomForestClassifier(300, min_samples_leaf=5, criterion="gini",random_state=2025)
            clf.fit(train_X, train_Y)

            pred_prob = clf.predict_proba(test_X)
            soft_label[test_index] += pred_prob

    soft_label /= round_num

    hard_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    for i in range(np.shape(data)[0]):
        hard_label[i][int(data[i, -1])] = 1

    soft_label = (soft_label + hard_label*k) / (k+1)

    # predict =  clf.predict(X_test)
    # final_acc = np.sum(predict==y_test)/X_test.shape[0] * 100
    # print(final_acc)
    # exit()

    return soft_label

def NN_produce_soft_kf(data, round_num, fold_num, k=1,T=1,model='mlp',data_name='iot'):
    #input_size = data.shape[1] - 1
    output_size = len(np.unique(data[:, -1]))
    soft_label = np.zeros([data.shape[0], output_size])


    for i in range(round_num):
        kf = KFold(n_splits=fold_num)
        for train_index, test_index in kf.split(X=data[:, :-1], y=data[:, -1], groups=data[:, -1]):
            train_set, test_set = data[train_index], data[test_index]
            train_X, train_Y = train_set[:, :-1], train_set[:, -1].astype(int)
            test_X = test_set[:, :-1]
            test_X = torch.tensor(test_X, dtype=torch.float32)


            # train_X = min_max_normalize(train_X,feature_min, feature_max)
            # test_X = min_max_normalize(test_X,feature_min, feature_max)
            NN,best_acc=Train_Teacher(train_X,train_Y,model,args.dataset,args.min_sample_leaf,args)

            

            temp = NN(test_X[0:1000].cuda())
            pred_prob = temp.detach().cpu().numpy()
            for i in range(1000, test_X.shape[0], 1000):
                temp = NN(test_X[i:i + 1000].cuda())
                pred_prob = np.append(pred_prob, temp.detach().cpu().numpy(), axis=0)

            soft_label[test_index] += pred_prob

    soft_label /= round_num

    hard_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    for i in range(np.shape(data)[0]):
        hard_label[i][int(data[i, -1])] = 1

    soft_label = softmax(soft_label / T)
    soft_label = (soft_label + hard_label*k) / (k+1)
    # soft_label= F.softmax(torch.tensor(soft_label),dim=1)

    return soft_label



def predict(args,clf,X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list,\
         soft_label,clf_threshold,log_class):


    # pred = clf.predict(data_eval[:, :-1])
    # 此处改为预测概率使用node.label_dict样本占比来表示概率

    pred_proba = clf.predict_proba(X_test)
    pred = np.argmax(pred_proba,axis=1)
    
    acc = accuracy(pred, y_test)


    print("soft decision tree:", acc*100)

    


    # final_acc
    old_class_idx = np.max(pred_proba,axis=1) > clf_threshold
    new_class_idx = np.max(pred_proba,axis=1) <= clf_threshold

    all_predict = np.zeros_like(y_test)
    all_predict[new_class_idx] = len(args.selected_class)
    all_predict[old_class_idx] = np.argmax(pred_proba[old_class_idx],axis=1)
    pred_proba = pred_proba[old_class_idx]
    predict = np.argmax(pred_proba,axis=1)
    y_test_old_class = y_test[old_class_idx]
    final_acc = np.sum(predict==y_test_old_class)/X_test.shape[0] * 100
    
    if len(unknown_label_list)==0:
        precision = 100* precision_score(y_test, all_predict, average='weighted',zero_division=0)
        recall = 100* recall_score(y_test, all_predict, average='weighted',zero_division=0)
        f1 = 100* f1_score(y_test, all_predict, average='weighted',zero_division=0)
    else:
        precision = -1
        recall = -1
        f1 = -1



    print("final_acc: ",final_acc)

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

    proba_old_class = clf.predict_proba(X_test)
    proba_old_max = np.max(proba_old_class,axis=1)
    fp = (proba_old_max <= clf_threshold).sum()
    tn = (proba_old_max > clf_threshold).sum()

    new_cm = [tp,fn,fp,tn]



    print("Confusino Matrix:",new_cm)


    if args.export_rule == True:
        if log_class == -1:
            log_label_name = "ALL"
        else:
            log_label_name = label_name[log_class]
        model_path = './rule_tree/{}/mou_{}.txt'.format(args.dataset,log_label_name)
        # 先清空txt
        with open(model_path, "w") as f:
            pass
        clf.show_tree(model_path)

        rule_num = count_lines_in_file(model_path)
        # txt 2 entity        
        # rules = parse_rules(model_path)
        # 预测
        # y_pred = rule_predict(rules, X_test)

        # 计算准确率
        # rule_acc = calculate_accuracy(y_test, y_pred)
        # print(f'Rule based Accuracy: {rule_acc:.4f}')
            

    else:
        rule_num = -1

    rule_acc = final_acc
    return acc*100,final_acc,new_cm,rule_num,rule_acc,precision , recall ,f1
            




def preprocess(args):
    # 读取数据 softlabel
    if args.dataset == 'ton-iot':
        label_name = ['normal','backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'runsomware', 'scanning', 'xss']

        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list_tmp, unknown_label_list = \
        get_data_ton_iot(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size)
        # CUSTOM_FEAT_COLS = ['count','ps_mean', 'ps_max', 'ps_min','iat_mean', 'iat_max', 'iat_min','l4_proto', 'service_port']
        # feature_attr = ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'd', 'd'] 
        # feature_attr = ['d' for i in range(X_train.shape[1])]
        X_train = convert_to_binary(X_train)
        X_test = convert_to_binary(X_test)
        unknown_data_list = []
        for unknown_data in unknown_data_list_tmp:
            unknown_data = convert_to_binary(unknown_data)
            unknown_data_list.append(unknown_data)

        feature_attr = ['d' for i in range(X_train.shape[1])]
        tmp_y_train = y_train.reshape(-1,1)
        tmp_y_test = y_test.reshape(-1,1)


        data_train = np.hstack([X_train,tmp_y_train])
        data_eval = np.hstack([X_test,tmp_y_test])

    elif args.dataset == 'iscx':
        label_name = ['email', 'chat', 'streaming_multimedia', 'file_transfer', 'voip', 'p2p']

        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list_tmp, unknown_label_list = \
        get_data_ISCX(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size,version=2)

        X_train = fea2bit(args,X_train)
        X_test = fea2bit(args,X_test)
        unknown_data_list = []
        for unknown_data in unknown_data_list_tmp:
            unknown_data = fea2bit(args,unknown_data)
            unknown_data_list.append(unknown_data)

        feature_attr = ['d' for i in range(X_train.shape[1])]
        # feature_attr = ['d', 'd', 'd', 'd', 'c', 'c', 'd', 'c', 'c', 'c']
        tmp_y_train = y_train.reshape(-1,1)
        tmp_y_test = y_test.reshape(-1,1)
        data_train = np.hstack([X_train,tmp_y_train])
        data_eval = np.hstack([X_test,tmp_y_test])


    elif args.dataset == 'cicids-2017':
        label_name = ['normal','Brute-Force', 'DoS', 'Web-Attack', 'DDoS', 'Botnet', 'Port-Scan']

        # class include normal
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list_tmp, unknown_label_list = \
        get_data_CICIDS_2017(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size)
        
        X_train = convert_to_binary(X_train)
        X_test = convert_to_binary(X_test)
        unknown_data_list = []
        for unknown_data in unknown_data_list_tmp:
            unknown_data = convert_to_binary(unknown_data)
            unknown_data_list.append(unknown_data)

        feature_attr = ['d' for i in range(X_train.shape[1])]

        # feature_attr = ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'd', 'd']
        tmp_y_train = y_train.reshape(-1,1)
        tmp_y_test = y_test.reshape(-1,1)
        data_train = np.hstack([X_train,tmp_y_train])
        data_eval = np.hstack([X_test,tmp_y_test])

    elif args.dataset == 'cicids-2018':
        label_name = ['BENIGN','DDoS_LOIC_HTTP' ,'DDoS_HOIC','DDoS_LOIC_UDP', \
               'DoS_GoldenEye', 'DoS_Hulk','DoS_Slowloris' ,\
                'SSH_BruteForce','Web_Attack_XSS','Web_Attack_SQL','Web_Attack_Brute_Force'] 

        # class include normal
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list_tmp, unknown_label_list = \
        get_data_CICIDS_2018(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size)

        X_train = convert_to_binary(X_train)
        X_test = convert_to_binary(X_test)
        unknown_data_list = []
        for unknown_data in unknown_data_list_tmp:
            unknown_data = convert_to_binary(unknown_data)
            unknown_data_list.append(unknown_data)

        feature_attr = ['d' for i in range(X_train.shape[1])]

        # feature_attr = ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'd', 'd']
        tmp_y_train = y_train.reshape(-1,1)
        tmp_y_test = y_test.reshape(-1,1)
        data_train = np.hstack([X_train,tmp_y_train])
        data_eval = np.hstack([X_test,tmp_y_test])



    elif args.dataset == 'unibs':
        label_name =  ['ssl', 'bittorrent', 'http', 'edonkey', 'pop3', 'skype', 'imap', 'smtp']
            
        # class include normal
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_UNIBS(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 
        # select_feats = ['Destination Port', 'Max Packet Length', 'Packet Length Total', 'Current Packet Length', 'ACK Flag Count']

        feature_attr = ['c', 'c', 'c', 'c', 'c']
        tmp_y_train = y_train.reshape(-1,1)
        tmp_y_test = y_test.reshape(-1,1)
        data_train = np.hstack([X_train,tmp_y_train])
        data_eval = np.hstack([X_test,tmp_y_test])



    elif args.dataset == 'unsw-iot':
        label_name = ["Withings Smart Baby Monitor","Withings Aura smart sleep sensor","Dropcam",
        "TP-Link Day Night Cloud camera","Samsung SmartCam","Netatmo weather station","Netatmo Welcome",
        "Amazon Echo", "Laptop","NEST Protect smoke alarm","Insteon Camera","Belkin Wemo switch",
        "Belkin wemo motion sensor", "Light Bulbs LiFX Smart Bulb", "Triby Speaker", "Smart Things"]
        
        # class include normal
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list_tmp, unknown_label_list = \
        get_data_UNSW_IOT(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 


        X_train = convert_to_binary(X_train)
        X_test = convert_to_binary(X_test)

        unknown_data_list = []
        for unknown_data in unknown_data_list_tmp:
            unknown_data = fea2bit(args,unknown_data)
            unknown_data_list.append(unknown_data)

        feature_attr = ['d' for i in range(X_train.shape[1])]
        # feature_attr = ['c', 'c', 'c', 'd', 'd', 'd']
        tmp_y_train = y_train.reshape(-1,1)
        tmp_y_test = y_test.reshape(-1,1)
        data_train = np.hstack([X_train,tmp_y_train])
        data_eval = np.hstack([X_test,tmp_y_test])



    elif args.dataset == 'unsw-nb15':

        label_name = ['normal', 'Worms','Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', ]
        
        # class include normal
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list_tmp, unknown_label_list = \
        get_data_UNSW_NB15(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 


        X_train = unsw_nb15_to_binary(X_train)

        X_test = unsw_nb15_to_binary(X_test)
        unknown_data_list = []
        for unknown_data in unknown_data_list_tmp:
            unknown_data = unsw_nb15_to_binary(unknown_data)
            unknown_data_list.append(unknown_data)

        feature_attr = ['d' for i in range(X_train.shape[1])]
        # feature_attr = ['c', 'c', 'c', 'd', 'd', 'd']
        tmp_y_train = y_train.reshape(-1,1)
        tmp_y_test = y_test.reshape(-1,1)
        data_train = np.hstack([X_train,tmp_y_train])
        data_eval = np.hstack([X_test,tmp_y_test])


    if args.teacher == 'rf':
        soft_label = produce_soft_labels(data_train, num_classes,round_num=1, fold_num=2, k=0, model='rf')
    else:
        soft_label = NN_produce_soft_kf(data_train, round_num=1, fold_num=2, k=0, T=T, model=args.teacher,
                                        data_name=args.dataset)
        
    # # 不过反正是对比方法了
    # soft_label = np.eye(len(set(data_train[:, -1].astype(int))))[data_train[:, -1].astype(int)]


    return X_train, X_test, y_train, y_test, feature_min, feature_max, \
        unknown_data_list, unknown_label_list, soft_label, feature_attr


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='PyTorch SDT Training')
    parser.add_argument('--export_rule', type=bool, default=False, help="Enable or disable the export_rule")
    parser.add_argument("--tpr_acc_rate", default=1, type=int, help="tpr_acc_rate")

    parser.add_argument(
        '--teacher', default='gru', choices=['rf', 'gru','mlp','lstm'], type=str, help='teacher model selection')
    parser.add_argument(
        '--cuda', default=6, type=int, help='cuda selection')
    parser.add_argument(
        '--K', default=1, type=int, help='the proportion of hard label')
    parser.add_argument('--T', default=1, type=int, help='the temperature of soft label')
    parser.add_argument('--dataset', type=str, default='cicids-2018', help='ton-iot / iscx / cicids / unibs')

    parser.add_argument('--attack_max_samples', type=int, default=10000, help='ton-iot dataset: max samples per attack')
    parser.add_argument('--selected_class', type=list, default=[0,1,4,2,5,6,3], help='ton-iot dataset: selected attack class, in [1-9]')
    parser.add_argument('--test_split_size', type=float, default=0.8, help='test split size, when few-shot setting, should large')
    parser.add_argument('--min_sample_leaf', type=int, default=5, help='SoftRF pram')


    args = parser.parse_args()
    torch.cuda.set_device(args.cuda)

    TEST_SIZE = args.test_split_size
    #MAX_T=[1,2,3,4,5,6]
    T=1


    # cicids 0,1,4,||2,5,6,3
    # iscx   2,4,5,||1,3,0
    # iot    (0),1,3,4,||5,8,6,7,9,2
    # 'iscx','cicids','ton-iot','unibs',"unsw-iot","unsw-nb15"


    



    for dataset in ["cicids-2018","iscx","ton-iot","unsw-nb15"]:
        args.dataset = dataset


        output_csv = "C2_ROC_mou.csv"




        print("Dataset: ",args.dataset)
        if dataset == 'iscx':
            args.selected_class = [2,4,]
            add_order = [5,1,3,0]
            args.attack_max_samples = 10000    
            label_name = ['email', 'chat', 'streaming_multimedia', 'file_transfer', 'voip', 'p2p']

        elif dataset == 'cicids-2017':
            args.selected_class = [0,1,]
            add_order = [4,2,5,6,3]
            label_name = ['normal','Brute-Force', 'DoS', 'Web-Attack', 'DDoS', 'Botnet', 'Port-Scan']

        elif dataset == 'cicids-2018':
            args.min_sample_leaf = 5
            args.attack_max_samples = 10000
            args.selected_class = [0,1,]
            add_order = [2,3,4,5,6,7,8,9,10]
            label_name = ['BENIGN','DDoS_LOIC_HTTP' ,'DDoS_HOIC','DDoS_LOIC_UDP', \
            'DoS_GoldenEye', 'DoS_Hulk','DoS_Slowloris' ,\
                'SSH_BruteForce','Web_Attack_XSS','Web_Attack_SQL','Web_Attack_Brute_Force'] 

        elif dataset == 'ton-iot':
            args.selected_class = [0,5]
            add_order = [3,7,1,4,2,6,8,9,]
            args.attack_max_samples = 30000      
            label_name = ['normal','backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'runsomware', 'scanning', 'xss']

        elif dataset == 'unibs':
            args.selected_class = [0,1]
            add_order = [2,3,4,5,6,7]
            label_name = ['ssl', 'bittorrent', 'http', 'edonkey', 'pop3', 'skype', 'imap', 'smtp']

        elif dataset == 'unsw-iot':
            args.selected_class = [0,1,]
            add_order = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            label_name = ["Withings Smart Baby Monitor","Withings Aura smart sleep sensor","Dropcam",
        "TP-Link Day Night Cloud camera","Samsung SmartCam","Netatmo weather station","Netatmo Welcome",
        "Amazon Echo", "Laptop","NEST Protect smoke alarm","Insteon Camera","Belkin Wemo switch",
        "Belkin wemo motion sensor", "Light Bulbs LiFX Smart Bulb", "Triby Speaker", "Smart Things"]
            
        elif dataset == 'unsw-nb15':
            args.attack_max_samples = 30000
            args.selected_class = [0,2,]
            add_order = [1,3,4,5,6,7,8,9]
            label_name = ['normal', 'Worms','Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', ]
        

        record_list_name = ['dataset','add_order','choose_threshold','org_acc',\
                            'detection_acc','choose_TPR','choose_FPR','final_acc','rule_acc','precision' , 'recall' ,'f1','rule_number']



        # 检查CSV文件是否存在，如果不存在则创建并写入表头
        if not os.path.exists(output_csv):
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                # 写入超参数和结果
                writer.writerow(record_list_name)



        for log_class in add_order:
            print('----------',label_name[log_class],'----------')
            final_acc_list, unknown_class_rate_list = [],[]
            detection_tpr,detection_fpr,final_tpr,final_fpr= [],[],[],[]


            # 为了选出阈值求出ROC，遍历阈值
            # the_first ############
            X_train, X_test, y_train, y_test, feature_min, feature_max,\
            unknown_data_list, unknown_label_list,soft_label, feature_attr  = preprocess(args)


            clf = SoftTreeClassifier(n_features="all", min_sample_leaf=args.min_sample_leaf)
            clf.fit(X_train, soft_label, feature_attr)
            # 阈值搜索


            if args.export_rule == False:
                x = np.arange(0,1.01,0.01)
            else:
                x = [0]

            for cls_threshold in x:
                args.cls_threshold = cls_threshold
                print('--',cls_threshold,'--')


                org_acc,final_acc,new_cm,rule_num,rule_acc,precision , recall ,f1 = predict(args,clf,X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list,\
                                                        soft_label , cls_threshold,log_class)


                final_acc_list.append(final_acc)
                choose_TPR = 100*new_cm[0]/(new_cm[0]+new_cm[1])
                choose_FPR = 100*new_cm[2]/(new_cm[2]+new_cm[3])
                # tp,fn,fp,tn对应相加
                detection_acc = 100*(new_cm[0]+new_cm[3])/sum(new_cm)


                experi_csv = [args.dataset,label_name[log_class],cls_threshold,org_acc,detection_acc,choose_TPR,choose_FPR,final_acc,rule_acc,precision , recall ,f1,rule_num]

                if os.path.exists(output_csv):
                    with open(output_csv, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        # 写入超参数和结果
                        writer.writerow(experi_csv)

            args.selected_class.append(log_class)


        # 所有类
        X_train, X_test, y_train, y_test, feature_min, feature_max,\
        unknown_data_list, unknown_label_list,soft_label,feature_attr  = preprocess(args)
        clf = SoftTreeClassifier(n_features="all", min_sample_leaf=args.min_sample_leaf)
        clf.fit(X_train, soft_label, feature_attr)

        log_class = -1
        args.log_class = -1

        for cls_threshold in x:
            org_acc,final_acc,new_cm,rule_num,rule_acc,precision , recall ,f1 = predict(args,clf,X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list,\
                                                        soft_label,cls_threshold,log_class)
            if (new_cm[0]+new_cm[1]) != 0:
                choose_TPR = 100*new_cm[0]/(new_cm[0]+new_cm[1])
            else:
                choose_TPR = -1
            choose_FPR = 100*new_cm[2]/(new_cm[2]+new_cm[3])
            # tp,fn,fp,tn对应相加
            detection_acc = 100*(new_cm[0]+new_cm[3])/sum(new_cm)
            experi_csv = [args.dataset,"ALL",cls_threshold,org_acc,detection_acc,choose_TPR,choose_FPR,final_acc,rule_acc,precision , recall ,f1,rule_num]

            if os.path.exists(output_csv):
                with open(output_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # 写入超参数和结果
                    writer.writerow(experi_csv)

            
            
