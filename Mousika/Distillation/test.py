from models import *
# encoding: utf-8
import sys 
sys.path.append('/home/zhuyijia/prototypical')

import warnings
warnings.filterwarnings('ignore')
import random
import torch
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import torch.utils.data as Data

import time
from sklearn.metrics import classification_report,roc_curve,auc,confusion_matrix,accuracy_score
import torch.nn.functional as F
from V6_complete.utils.get_dataset import *
from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz
import csv
import os
import re

import numpy as np
from models import *

seed = 2025
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
np.random.seed(seed)
random.seed(seed)


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.out = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.5),  # 添加 Dropout 层
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(0.5),  # 添加 Dropout 层
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.5),  # 添加 Dropout 层
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.5),  # 添加 Dropout 层
            nn.Linear(128, output_size)
        )

    def forward(self, x):

        out = self.out(x)
        return out


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




label_name = ["Withings Smart Baby Monitor","Withings Aura smart sleep sensor","Dropcam",
"TP-Link Day Night Cloud camera","Samsung SmartCam","Netatmo weather station","Netatmo Welcome",
"Amazon Echo", "Laptop","NEST Protect smoke alarm","Insteon Camera","Belkin Wemo switch",
"Belkin wemo motion sensor", "Light Bulbs LiFX Smart Bulb", "Triby Speaker", "Smart Things"]


num_classes = len([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
# 参数样本数，类别，测试集比例
X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
get_data_ISCX(1000, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],  0.8) 


# 对比一下随机森林
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
pre = clf.predict(X_test)
acc = np.mean(pre==y_test)

print("RamdomFroest acc :",acc*100)

# exit()
# MLP
model = LSTM(X_train.shape[1],num_classes).cuda()

X_train = min_max_normalize(X_train,feature_min, feature_max,)
X_test = min_max_normalize(X_test,feature_min, feature_max,)
# X_train = torch.tensor(X_train)
train_datasets = Data.TensorDataset(torch.tensor(X_train, dtype=torch.float32).cuda(), torch.tensor(y_train).cuda())
test_datasets = Data.TensorDataset(torch.tensor(X_test, dtype=torch.float32).cuda(), torch.tensor(y_test).cuda())
train_loader = Data.DataLoader(dataset=train_datasets, batch_size=256, shuffle=True)
test_loader = Data.DataLoader(dataset=test_datasets, batch_size=256, shuffle=True)
LR = 0.001
train_acc = []
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all model parameters
loss_func = nn.CrossEntropyLoss()        
for epoch in range(1000):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x=b_x.cuda()
        b_y=b_y.long().cuda()

        output = model(b_x)               # model output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        train_pred_y = torch.max(output, 1)[1].cpu().data.numpy()
        train_acc.append(100.0 * float((train_pred_y == np.array(b_y.cpu().view(-1).data)).sum())/ len(b_y))

        if step % 10 == 0:
            correct = 0.
            for batch_idx, (data, target) in enumerate(test_loader):
                data=data.cuda()
                test_output = model(data)
                pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()

                correct +=  (pred_y==np.array(target.view(-1).cpu().data)).sum()

            accuracy = 100.0 * float(correct) / len(test_loader.dataset)

            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),'| train acc: %.4f' % np.mean(train_acc), '| test accuracy: %.2f' % accuracy)




