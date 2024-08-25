import numpy as np
import torch
import torch.nn as nn

batch_size = 512

def model_prune(model, num_classes, X_train, y_train, feature_min, feature_max, max_proto_num, prune_T):
    support_num = np.zeros((num_classes, max_proto_num))
    support_num = support_num.astype(np.int64)

    feature_num = len(feature_min)

    idx = 0
    while idx < X_train.shape[0]:
        train_batch = X_train[idx:min(idx + batch_size, X_train.shape[0])]
        train_label = y_train[idx:min(idx + batch_size, y_train.shape[0])]
            
        train_batch = (train_batch - feature_min) / (feature_max - feature_min)
            
        train_batch = torch.from_numpy(train_batch).float().cuda()
        train_label = torch.from_numpy(train_label).long().cuda()
        
        logits = model(train_batch)
        _, predictions = torch.max(logits, dim=1)

        for dim1 in range(num_classes):
            
            train_batch = train_batch.view(train_batch.shape[0], 1, feature_num)
            
            if model.cal_dis == 'l_n':
                
                if model.cal_mode == 'trans_abs':
                    train_batch = train_batch * torch.abs(model.class_transform[dim1])
                    dist_class = torch.abs(train_batch - model.class_prototype[dim1])
                elif model.cal_mode == 'abs_trans':
                    dist_class = torch.abs(train_batch - model.class_prototype[dim1])
                    dist_class = dist_class * torch.abs(model.class_transform[dim1])


                min_dist, _ = torch.max(dist_class, dim=2)
                min_dist, indice = torch.min(min_dist, dim=1)
            
                for dim2 in range(predictions.shape[0]):
                    #if predictions[dim2] == dim1:
                    if predictions[dim2] == dim1 and predictions[dim2] == train_label[dim2]:
                        support_num[dim1][indice[dim2]] += 1
            
            else:
                raise ValueError('not support cal_dis type!')
                
        idx += batch_size

    print('train set num = %d, all support num = %d, distance-based train acc before prune = %.3f%%' % (X_train.shape[0], np.sum(support_num), np.sum(support_num) / X_train.shape[0] * 100))

    print('prune_T = %d' % (prune_T))
    print('support num of each selected prototype(order is random):')
    for dim1 in range(num_classes):
        num_sort = {}
        
        for dim2 in range(support_num.shape[1]):
            num_sort[dim2] = support_num[dim1][dim2]
        num_sort = dict(sorted(num_sort.items(), key=lambda x: x[1]))

        del_list = []
        support_list = []
        
        for dim2 in num_sort:
            
            # new added code
            # notice: dim1, not dim2
            if dim2 >= model.class_prototype[dim1].shape[1]:
                continue
            
            if num_sort[dim2] <= prune_T:
                del_list.append(dim2)
            else:
                support_list.append(num_sort[dim2])
        
        print(support_list)
        
        proto_pre = model.class_prototype[dim1].detach().cpu().numpy()
        proto_new = np.delete(proto_pre, del_list, axis=1)
        model.class_prototype[dim1] = nn.Parameter(torch.from_numpy(proto_new).cuda())
        
        trans_pre = model.class_transform[dim1].detach().cpu().numpy()
        trans_new = np.delete(trans_pre, del_list, axis=1)
        model.class_transform[dim1] = nn.Parameter(torch.from_numpy(trans_new).cuda())
        
        assert model.class_prototype[dim1].shape == model.class_transform[dim1].shape

    print('prototype num of each class after prune:')
    num_list = []
    for dim1 in range(num_classes):
        num_list.append(model.class_prototype[dim1].shape[1])
    print(num_list)
    
    return model
