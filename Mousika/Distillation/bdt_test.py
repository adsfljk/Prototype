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
import time
from train_teacher import *
from SoftTree import *
from models import *
import time
from sklearn.metrics import classification_report
import torch.nn.functional as F
from V6_complete.utils.get_dataset import *

parser = argparse.ArgumentParser()
parser.add_argument("--max_depth", default=6, type=int)
parser.add_argument("--n_estimators", default=1, type=int)

parser.add_argument('--dataset', type=str, default='iscx', help='ton-iot / iscx / cicids / unibs')
parser.add_argument('--attack_max_samples', type=int, default=1000, help='ton-iot dataset: max samples per attack')
parser.add_argument('--selected_class', type=list, default=[2,4,5,1,3,0], help='ton-iot dataset: selected attack class, in [1-9]')
parser.add_argument('--test_split_size', type=float, default=0.8, help='test split size, when few-shot setting, should large')

args = parser.parse_args()
final_acc_list, unknown_class_rate_list = [],[]
detection_tpr,detection_fpr,final_tpr,final_fpr= [],[],[],[]
log_class = 0


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


def produce_soft_labels(data, num_classes,round_num, fold_num, k=1,model='rf'):

    soft_label = np.zeros([data.shape[0], num_classes])

    for i in range(round_num):
        kf = KFold(n_splits=fold_num)
        for train_index, test_index in kf.split(X=data[:, :-1], y=data[:, -1], groups=data[:, -1]):
            train_set, test_set = data[train_index], data[test_index]
            train_X,train_Y=train_set[:, :-1],train_set[:, -1].astype(int)
            test_X=test_set[:, :-1]
            if model=='rf':
                clf = RandomForestClassifier(300, min_samples_leaf=5, criterion="gini")
            clf.fit(train_X, train_Y)

            pred_prob = clf.predict_proba(test_X)
            soft_label[test_index] += pred_prob

    soft_label /= round_num

    hard_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    for i in range(np.shape(data)[0]):
        hard_label[i][int(data[i, -1])] = 1

    soft_label = (soft_label + hard_label*k) / (k+1)

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
            NN,best_acc=Train_Teacher(train_X,train_Y,model,args.dataset)

            test_X = torch.tensor(test_X, dtype=torch.float32)

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

    return soft_label



def main(args,clf_threshold):

        
    if args.dataset == 'ton-iot':
        label_name = ['normal','backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'runsomware', 'scanning', 'xss']

        num_classes = len(args.selected_class) + 1 # include normal
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_ton_iot(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size)
        feature_attr = ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'd', 'd'] 
        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        data_train = np.hstack([X_train,y_train])
        data_eval = np.hstack([X_test,y_test])

    elif args.dataset == 'iscx':
        label_name = ['email', 'chat', 'streaming_multimedia', 'file_transfer', 'voip', 'p2p']

        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_ISCX(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size,version=2) 
        feature_attr = ['d', 'd', 'd', 'd', 'c', 'c', 'd', 'c', 'c', 'c']
        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        data_train = np.hstack([X_train,y_train])
        data_eval = np.hstack([X_test,y_test])
        a,b = np.unique(y_train,return_counts=True)


    elif args.dataset == 'cicids':
        label_name = ['normal','Brute-Force', 'DoS', 'Web-Attack', 'DDoS', 'Botnet', 'Port-Scan']

        # class include normal
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_CICIDS(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 
        feature_attr = ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'd', 'd']
        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        data_train = np.hstack([X_train,y_train])
        data_eval = np.hstack([X_test,y_test])




    elif args.dataset == "univ":
        # 此数据集2分类112个特征，类别样本量差不多.特征0 1
        # dtype 是 int64
        data_train, feature_attr = load_data('univ')
        data_eval, feature_attr=load_data('univ_test')

        num_classes = 2
        # d 表示离散特征（discrete feature）。
        # c 表示连续特征（continuous feature）。
        feature_attr = ['d']*112

        data_train = np.array(data_train).astype(np.float16)
        data_eval = np.array(data_eval).astype(np.float16)




    print('training data:')

    #kk=args.K
    
    output = []
    teacher_output = []
    sdt_output = []
    for kk in range(MAX_K):
        # print("--------------", kk, "--------------")
        acc_sdt, acc_dt, acc_Teacher, sdt_time, dt_time,   sdt_test_time,  NN_time = [], [], [], [], [], [], []
        teacher_report, sdt_report = [], []
        for i in range(ROUND_NUM):
            print("ROUND:", str(i))


            begin_time = time.time()
            if args.teacher == 'rf':
                soft_label = produce_soft_labels(data_train, num_classes,round_num=1, fold_num=2, k=kk, model='rf')
            else:
                soft_label = NN_produce_soft_kf(data_train, round_num=1, fold_num=2, k=kk, T=T, model=args.teacher,
                                                data_name=args.dataset)
            end_time = time.time()
            # print('produce soft label needs {:}s'.format(end_time - begin_time))

            if args.teacher == 'rf':
                # random forest
                clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, criterion="gini")
                clf.fit(data_train[:, :-1], data_train[:, -1].astype(int))
                pred = clf.predict(data_eval[:, :-1])

                acc_Teacher.append(accuracy(pred, data_eval[:, -1].astype(int)))
            else:  # NN
                begin_time = time.time()
                NN, best_acc = Train_Teacher(data_train[:, :-1], data_train[:, -1].astype(int), args.teacher,
                                                args.dataset)
                end_time = time.time()
                t1 = end_time - begin_time
                NN_time.append(t1)
                print('training {:} needs {:}s'.format(args.teacher, t1))
                test_X = data_eval[:, :-1]
                test_X = torch.tensor(test_X, dtype=torch.float32)
                test_Y = data_eval[:, -1].astype(int)
                test_Y = torch.tensor(test_Y)

                test_datasets = Data.TensorDataset(test_X, test_Y)
                test_loader = Data.DataLoader(dataset=test_datasets, batch_size=128, shuffle=False, num_workers=2)
                correct = 0
                pred = np.array([])
                for batch_idx, (data_X, target) in enumerate(test_loader):
                    data_X = data_X.cuda()
                    test_output = NN(data_X)
                    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
                    pred = np.concatenate([pred, pred_y], axis=0)
                    correct += (pred_y == np.array(target.view(-1).data)).sum()
                acc = float(correct) / len(test_loader.dataset)
                print(args.teacher + '| test accuracy: %.4f' % acc)
                train_acc = 0.0
                acc_Teacher.append(acc)
            teacher_round = get_c_avg(classification_report(data_eval[:, -1], pred, digits=4, output_dict=True))
            teacher_report.append(teacher_round)


            # 尝试hard label编码.发现效果貌似要好些？？
            # 不过反正是对比方法了
            # soft_label = np.eye(len(set(data_train[:, -1].astype(int))))[data_train[:, -1].astype(int)]


            # soft decision tree
            clf = SoftTreeClassifier(n_features="all", min_sample_leaf=args.min_sample_leaf)
            clf.fit(data_train[:, :-1], soft_label, feature_attr)


            # pred = clf.predict(data_eval[:, :-1])
            # 此处改为预测概率使用node.label_dict样本占比来表示概率

            pred_proba = clf.predict_proba(data_eval[:, :-1])



            pred = np.argmax(pred_proba,axis=1)

            
            acc = accuracy(pred, data_eval[:, -1])

            end_time = time.time()
            t = end_time - begin_time
            sdt_test_time.append(t)

            print("  soft decision tree:", acc)
            acc_sdt.append(acc)





            model_path = './rule_tree/{}_{}_kk{}_round{}.txt'.format(args.dataset, args.teacher, str(kk),str(i))
            clf.show_tree(model_path)
            # print("teacher_report", teacher_round)
            # print("sdt_report", sdt_round)


            






if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='PyTorch SDT Training')
    parser.add_argument(
        '--teacher', default='rf', choices=['rf', 'gru'], type=str, help='teacher model selection')
    parser.add_argument(
        '--cuda', default=0, type=int, help='cuda selection')
    parser.add_argument(
        '--K', default=1,  type=int, help='the proportion of hard label')
    parser.add_argument(
        '--T', default=1, type=int, help='the temperature of soft label')
    parser.add_argument('--dataset', type=str, default='ton-iot', help='ton-iot / iscx / cicids / unibs')

    parser.add_argument('--attack_max_samples', type=int, default=1000, help='ton-iot dataset: max samples per attack')
    parser.add_argument('--selected_class', type=list, default=[2,4,5,1,3,6,7,8,9], help='ton-iot dataset: selected attack class, in [1-9]')
    parser.add_argument('--log_class', type=int, default=0, help='The next new class index')

    parser.add_argument('--test_split_size', type=float, default=0.2, help='test split size, when few-shot setting, should large')
    # parser.add_argument("--max_depth", default=6, type=int)
    parser.add_argument("--n_estimators", default=1, type=int)
    parser.add_argument("--min_sample_leaf", default=5, type=int)

    args = parser.parse_args()
    # torch.cuda.set_device(args.cuda)

    ROUND_NUM = 1
    TEST_SIZE = args.test_split_size
    #MAX_T=[1,2,3,4,5,6]
    MAX_K = 1
    T=1
    x = np.arange(0,1.2,0.1)

    choose_threshold_tfpr = -999
    
    for clf_threshold in [0.5]:
        main(args,clf_threshold)




