import torch.nn as nn
import torch
import numpy as np
from sklearn.cluster import DBSCAN

#cal_dis in ['l_2', 'l_n']
#cal_mode in ['trans_abs', 'abs_trans']

class prototype(nn.Module):
    
    def __init__(self, num_classes, feature_num, prototype_num_classes, temperature, cal_dis, cal_mode):
        super(prototype, self).__init__()
        
        self.num_classes = num_classes
        self.feature_num = feature_num
        self.temperature = temperature
        self.cal_dis = cal_dis
        self.cal_mode = cal_mode
        
        #self.class_prototype = nn.ModuleList()
        self.class_prototype = []
        self.class_transform = []
        for i in range(self.num_classes):
            # set .cuda() first, otherwise not leaf node (cannot be optimized) 
            self.class_prototype.append(nn.Parameter(torch.FloatTensor(1, prototype_num_classes[i], self.feature_num).cuda()))
            self.class_transform.append(nn.Parameter(torch.zeros(1, prototype_num_classes[i], self.feature_num).cuda() + 1))
        
        for i in range(self.num_classes):
            torch.nn.init.uniform_(self.class_prototype[i], a=0, b=1)
            # torch.nn.init.uniform_(self.class_transform[i], a=0, b=1)
        
    def forward(self, train_batch):
        
        train_batch = train_batch.view(train_batch.shape[0], 1, self.feature_num)
       
        
        min_dist_class = []
        
        if self.cal_dis == 'l_2':
            
            for idx in range(self.num_classes):
                
                if self.class_prototype[idx].shape[1] == 0:
                    inf_dist = torch.zeros(train_batch.shape[0]).cuda()
                    min_dist_class.append(inf_dist + 10000)
                    continue
                
                dist_class = (train_batch - self.class_prototype[idx]) ** 2
                dist_class = torch.sum(dist_class, dim=2)
                min_dist, _ = torch.min(dist_class, dim=1)
                min_dist_class.append(min_dist)
            
        elif self.cal_dis == 'l_n':
            
            for idx in range(self.num_classes):
                
                if self.class_prototype[idx].shape[1] == 0:
                    inf_dist = torch.zeros(train_batch.shape[0]).cuda()
                    min_dist_class.append(inf_dist + 10000)
                    continue
                
                if self.cal_mode == 'trans_abs':
                    train_batch = train_batch * torch.abs(self.class_transform[idx])
                    dist_class = torch.abs(train_batch - self.class_prototype[idx])
                
                elif self.cal_mode == 'abs_trans':
                    dist_class = torch.abs(train_batch - self.class_prototype[idx])
                    dist_class = dist_class * torch.abs(self.class_transform[idx])

                
                min_dist, _ = torch.max(dist_class, dim=2)
                min_dist, _ = torch.min(min_dist, dim=1)
                min_dist_class.append(min_dist)
        
        class_similar_score = []
            
        for idx in range(self.num_classes):
            score = 1 / (min_dist_class[idx]+ 1e-6) * self.temperature 
            class_similar_score.append(score)
        
        for idx in range(self.num_classes):
            class_similar_score[idx] = class_similar_score[idx].view(-1, 1)
        
        min_dist = torch.cat(class_similar_score, dim=1)
        #min_dist.register_hook(save_grad('test'))
        logit = nn.functional.softmax(min_dist, dim=1)
        #logit.register_hook(save_grad('test'))
        
        
        class_inter_dist = 0

        return logit
    
    def save_parameter(self, dir):
        torch.save(self.class_prototype + self.class_transform, dir)
    
    def load_parameter(self, dir):
        
        state = torch.load(dir, map_location='cpu')
        for i in range(min(self.num_classes, len(state) // 2)):
            
            # num_classes 可能比加载的模型更大（增量时）
            # 这种情况下，新类别一定要加在最后面才行（标签是最后一个）
            
            if self.class_prototype[i].shape[1] != state[i].shape[1]:
                self.class_prototype[i] = nn.Parameter(torch.FloatTensor(1, state[i].shape[1], self.feature_num).cuda())
            
            self.class_prototype[i].data = state[i].cuda().data
        
        for i in range(min(self.num_classes, len(state) // 2)):
            
            if self.class_transform[i].shape[1] != state[i + len(state) // 2].shape[1]:
                self.class_transform[i] = nn.Parameter(torch.FloatTensor(1, state[i + len(state) // 2].shape[1], self.feature_num).cuda())
            
            self.class_transform[i].data = state[i + len(state) // 2].cuda().data


def init_model(model, init_type,dbscan_eps,min_samples, X_train, y_train, feature_min, feature_max):
    
    if init_type == 'NONE':
        return model
    
    if init_type == 'DBSCAN':
        
        # 初始化结果字典
        results = {}
        centers = {}
        noise_counts = {}

        # 对输入样本进行分类
        unique_labels = np.unique(y_train)
        
        for label in unique_labels:
            # 获取当前类别的样本
            class_mask = (y_train == label)
            X_class = X_train[class_mask]

            # 对当前类别的样本进行min-max正则化

            X_class_scaled = (X_class - feature_min) / (feature_max - feature_min)

            # 使用DBSCAN进行聚类
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_samples) # 不要太小（比如0.003），会导致梯度过小模型不更新
            # (0.01, 2) 96.165%
            # (0.01, 5) 94.592%
            # (0.01, 10) 92.281%
            
            # (0.009, 2) 93.117%
            # (0.02, 2) 94.690%
            # (0.03, 2) 92.035%
            # 
            
            cluster_labels = dbscan.fit_predict(X_class_scaled)

            # 保存聚类结果
            results[label] = cluster_labels

            # 计算每个聚类的中心点及其接收的样本数量
            unique_clusters = set(cluster_labels)
            centers[label] = {}
            for cluster in unique_clusters:
                if cluster != -1:  # 忽略噪声点
                    cluster_mask = (cluster_labels == cluster)
                    cluster_points = X_class_scaled[cluster_mask]
                    cluster_center = np.mean(cluster_points, axis=0)
                    centers[label][cluster] = {
                        'center': cluster_center,
                        'count': cluster_points.shape[0]
                    }

            # 计算没有被任何聚类接收的样本数量
            noise_count = np.sum(cluster_labels == -1)
            noise_counts[label] = noise_count
        
        # print("\n聚类中心及样本数量:")
        # for label, cluster_info in centers.items():
            # print(f"类别 {label}:")
            # for cluster, info in cluster_info.items():
                # print(f"  聚类 {cluster}:", end="")
                #print(f"    中心: {info['center']}")
                # print(f"    样本数量: {info['count']}")

        # print("\n噪声点数量:")
        # for label, count in noise_counts.items():
            # print(f"类别 {label}: {count}")
        
        
        model.class_prototype = []
        model.class_transform = []


        for i in range(model.num_classes):
            # set .cuda() first, otherwise not leaf node (cannot be optimized) 
            if i not in list(centers.keys()):
                model.class_prototype.append(nn.Parameter(torch.FloatTensor(1, 0, model.feature_num).cuda()))
                model.class_transform.append(nn.Parameter(torch.zeros(1, 0, model.feature_num).cuda() + 1))
                continue
        
            model.class_prototype.append(nn.Parameter(torch.FloatTensor(1, len(centers[i]), model.feature_num).cuda()))
            model.class_transform.append(nn.Parameter(torch.zeros(1, len(centers[i]), model.feature_num).cuda() + 1))
        
        for i in range(model.num_classes):
            if i not in list(centers.keys()):
                continue
            for idx, cluster in enumerate(centers[i]):
                # 不要用.data直接赋值，会产生数值错误
                with torch.no_grad():
                    model.class_prototype[i][0][idx].copy_(torch.tensor(centers[i][cluster]['center']))
        
        return model