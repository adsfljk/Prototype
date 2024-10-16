import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torchvision.models as models

class GRU(nn.Module):
    def __init__(self,INPUT_SIZE, output_size):
        super(GRU, self).__init__()
        # self.LN1=nn.LayerNorm(INPUT_SIZE)
        #self.WINDOW_SIZE=WINDOW_SIZE
        self.INPUT_SIZE=INPUT_SIZE
        self.gru = nn.GRU(input_size=self.INPUT_SIZE,
                             hidden_size=256,
                             num_layers=2,
                             batch_first=True,
                            #  dropout=0.5
                             )

        self.out = nn.Sequential(nn.Linear(256, output_size))

    def forward(self, x):
        x = x[:, np.newaxis, :]
        r_out, self.hidden = self.gru(x, None)  # x(batch,time_step,input_size)
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out
    
class LSTM(nn.Module):
    def __init__(self, INPUT_SIZE, output_size, n_layers=2, hidden_dim=256):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(INPUT_SIZE, hidden_dim, n_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1, :, :]
        x = self.classifier(x)
        return x


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.out = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 Dropout 层
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 Dropout 层
            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(0.5),  # 添加 Dropout 层
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Dropout(0.5),  # 添加 Dropout 层
            nn.Linear(128, output_size)
        )

    def forward(self, x):

        out = self.out(x)
        return out



 # -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2020-08-30 15:58:51
# @Last Modified by:   xiegr
# @Last Modified time: 2020-10-09 15:53:26
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math


torch.manual_seed(2021)
torch.cuda.manual_seed_all(2021)
np.random.seed(2021)
random.seed(2021)
torch.backends.cudnn.deterministic = True


class deep_packet(nn.Module):
# Deep packet: a novel approach for encrypted traffic classification using deep learning 

    def __init__(self):
        super(deep_packet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=200, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(200),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=200, out_channels=200, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm1d(200),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(               
            nn.Linear(in_features=200, out_features=100),
            nn.Dropout(p=0.05),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(               
            nn.Linear(in_features=100, out_features=50),
            nn.Dropout(p=0.05),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(               
            nn.Linear(in_features=50, out_features=6),
            nn.Dropout(p=0.05),
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.reshape(-1, 1, 50).float()
        out = self.conv1(x)
        out = self.conv2(out)
        #out = out.transpose(-2, -1)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        if not self.training:
            return F.softmax(out, dim=-1).max(1)[1]
        return out


class CNN_LSTM_FC_LSTM_CNN(nn.Module):
# Network traffic classification using deep convolutional recurrent autoencoder neural networks for spatial–temporal features extraction
    def __init__(self):
        super(CNN_LSTM_FC_LSTM_CNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=300, embedding_dim=128)
        
        self.lstm1 = nn.LSTM(128, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128)
        )

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc_mid = nn.Linear(in_features=128, out_features=128)
        self.fc_output = nn.Linear(in_features=128, out_features=6)

    def forward(self, x):
        h = self.embedding(x)
        h = h.transpose(-2, -1)
        h = self.conv1(h)
        h = h.transpose(-2, -1)
        h, _ = self.lstm1(h)
        h = self.fc_mid(h)
        h = torch.nn.functional.dropout(h, p = 0.5) # auto cal according to self.training 
        h, _ = self.lstm2(h)
        h = h.transpose(-2, -1)
        h = self.conv2(h)
        h = self.pooling(h).view(h.size(0), -1)
        out = self.fc_output(h)

        if not self.training:
            return F.softmax(out, dim=-1).max(1)[1]
        return out


class APP_net(nn.Module):
# App-Net: A Hybrid Neural Network for Encrypted Mobile Traffic Classification 
    def __init__(self):
        super(APP_net, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=1600, embedding_dim=128)
        
        self.lstm1 = nn.LSTM(128, 128, batch_first=True, bidirectional = True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True, bidirectional = True)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256)
        )

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features=256, out_features=6)

    def forward(self, x):
        h = self.embedding(x)
        h1, _ = self.lstm1(h)
        h1, _ = self.lstm2(h1)
        h1 = h1.transpose(-2, -1)
        h1 = self.pooling(h1).view(h1.size(0), -1)

        h2 = h.transpose(-2, -1)
        h2 = self.conv1(h2)
        h2 = self.conv2(h2)
        h2 = self.pooling(h2).view(h2.size(0), -1)

        out = h1 + h2
        out = self.fc(out)

        if not self.training:
            return F.softmax(out, dim=-1).max(1)[1]
        return out


class FS_net(nn.Module):
# FS-Net: A Flow Sequence Network For Encrypted Traffic Classification 
    def __init__(self):
        super(FS_net, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=1600, embedding_dim=128)
        
        self.lstm1 = nn.Sequential(
            nn.LSTM(128, 128, batch_first=True, bidirectional = True),
        )
        self.lstm2 = nn.Sequential(
            nn.LSTM(256, 128, batch_first=True, bidirectional = True),
        )
        self.lstm3 = nn.Sequential(
            nn.LSTM(512, 128, batch_first=True, bidirectional = True),
        )
        self.lstm4 = nn.Sequential(
            nn.LSTM(256, 128, batch_first=True, bidirectional = True),
        )
        self.fc1 = nn.Sequential(               
            nn.Linear(in_features=512, out_features=64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(               
            nn.Linear(in_features=64, out_features=6)
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        h1, _ = self.lstm1(x)
        h2, _ = self.lstm2(h1)
        h3 = torch.cat([h1, h2], 2)
        
        h4, _ = self.lstm3(h3)
        h5, _ = self.lstm4(h4)
        out = torch.cat([h4, h5], 2)

        out = out.transpose(-2, -1)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        if not self.training:
            return F.softmax(out, dim=-1).max(1)[1]
        return out


class DNN(nn.Module):
# A deep learning method with wrapper based feature extraction for wireless intrusion detection system

    def __init__(self):
        super(DNN, self).__init__()
        # input is N C H W
        self.num_class = 6
        self.conv2_filters = 100
        self.embedding = nn.Embedding(num_embeddings=300, embedding_dim=100)
        self.fc1 = nn.Sequential(
            nn.Linear(self.conv2_filters, out_features=100),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(100, out_features=100),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(100, out_features=100),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=100, out_features=1),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(in_features=50, out_features=6),
            #nn.ReLU()
        )

    def forward(self, x):
        # x: B 1 H W
        # flatten
        #out = x.view(x.shape[0], -1, self.conv2_filters)
        out = self.embedding(x)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = out.view(out.shape[0], -1)
        out = self.fc5(out)

        if not self.training:
            return F.softmax(out, dim=-1).max(1)[1]
        return out


class RNN(nn.Module):
# Novel Deep Learning-Enabled LSTM Autoencoder Architecture for Discovering Anomalous Events From Intelligent Transportation Systems

    def __init__(self):
        super(RNN, self).__init__()
        # input is N C H W
        self.embedding = nn.Embedding(num_embeddings=300, embedding_dim=128)
        self.lstm1 = nn.Sequential(
            nn.LSTM(128, 128, batch_first=True),
        )
        self.lstm2 = nn.Sequential(
            nn.LSTM(128, 64, batch_first=True),
        )
        self.lstm3 = nn.Sequential(
            nn.LSTM(64, 32, batch_first=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=32, out_features=6),
            #nn.ReLU()
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: B 1 H W
        # flatten
        out = self.embedding(x)
        out, _ = self.lstm1(out)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out = out.transpose(-2, -1)
        out = self.pooling(out).view(out.size(0), -1)
        out = self.fc(out)

        if not self.training:
            return F.softmax(out, dim=-1).max(1)[1]
        return out
    

class LuNet(nn.Module):
# LuNet: A Deep Neural Network for Network Intrusion Detection 

    def __init__(self):
        super(LuNet, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=300, embedding_dim=64)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5,
                      stride=1, padding=2),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.BatchNorm1d(32)
        )
        self.lstm1 = nn.Sequential(
            nn.LSTM(32, 100, batch_first=True),
        )
        
        self.fc1 = nn.Sequential(               
            nn.Linear(in_features=100, out_features=6),
            #nn.ReLU()
        )

    def forward(self, x):
        # x: B 1 H W
        # flatten
        out = self.embedding(x)
        out = out.transpose(-2, -1)
        out = self.conv1(out)
        out = out.transpose(-2, -1)
        out, _ = self.lstm1(out)
        # only use the last state in LSTM
        out = out.mean([1])
        out = self.fc1(out)

        if not self.training:
            return F.softmax(out, dim=-1).max(1)[1]
        return out
    
# from utils.Models import Encoder

# class transformer(nn.Module):
#     def __init__(self):
#         super(transformer, self).__init__()
#         self.encoder = Encoder(1600, 8, 8, 8, 0.1)
#         self.linear1 = nn.Linear(8, 1)
#         self.linear2 = nn.Linear(512, 6)
#         self.fc = nn.Softmax(dim=1)
    
#     def forward(self, x):
#         x = self.encoder(x, None)
#         x = self.linear1(x)
#         x = x.squeeze(-1)
#         x = self.linear2(x)
#         #x = self.fc(x)
#         if not self.training:
#             return F.softmax(x, dim=-1).max(1)[1]
#         return x
