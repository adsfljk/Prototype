import numpy as np
import torch
import torch.nn as nn
from utils.proto import prototype, init_model
from utils.general import test_model_acc, set_global_seed
from utils.prune import model_prune
from utils.get_dataset import get_data_ton_iot
import statistics
import argparse
import copy

parser = argparse.ArgumentParser("try to include all parameters")

parser.add_argument('--gpu', type=int, default=0, help='single GPU device, should in [0-7]')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', type=int, default=15, help='training epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--temperature', type=float, default=0.1, help='training temperature, very important')
parser.add_argument('--max_proto_num', type=int, default=2000, help='initial prototype number') # 没用了，但是还可以保留
parser.add_argument('--prune_T', type=int, default=10, help='model prune parameter')
parser.add_argument('--cls_threshold', type=float, default=0.07, help='class distance threshold, very important') # 0.01
parser.add_argument('--init_type', type=str, default='DBSCAN', help='k-means / DBSCAN / NONE')
parser.add_argument('--cal_dis', type=str, default='l_n', help='only support l_n now')
parser.add_argument('--cal_mode', type=str, default='abs_trans', help='only support abs_trans now')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--dataset', type=str, default='ton-iot', help='ton-iot / ISCX')
parser.add_argument('--ton_iot_attack_max_samples', type=int, default=30000, help='ton-iot dataset: max samples per attack') # 1000
parser.add_argument('--ton_iot_selected_class', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9], help='ton-iot dataset: selected attack class, in [1-9]') # [1, 2, 3, 4, 6]
parser.add_argument('--test_split_size', type=float, default=0.2, help='test split size, when few-shot setting, should large')
parser.add_argument('--solve_conflict_on', type=str, default='test_set', help='train_set / test_set')
parser.add_argument('--save_model', type=str, default='./save_model', help='path to save model')
args = parser.parse_args()

# 改进的点：
# （1）用神经网络生成原型向量，看看精度会不会变好
# （2）可以自学习的温度参数
# （3）原型向量的初始值用k-means的结果 / 与k-means进行对比 / DBSCAN
# （4）先有高分类精度，再尽可能地设计规则提取算法：聚类会比直接用原型+距离的方式更好，理论上不输决策树的
# （5）看看数据面能不能改成range匹配，这样更新就完全是增量的了

# 2024.6.17记录：
# V2版本代码使用根据分类结果的上下界设定匹配范围，发现没有最开始的原型+距离的效果好，因此去掉V2，在V3中保留原始设置
# 在二者中都修改了的点，同时也进一步明确了一个问题：基于range匹配的表项；产生了表项优先级确定问题
# 与对比方法在资源消耗上的优势：只占用1张表和1个stage；对比方法使用9+3+1张表，至少3个stage
# 温度自学习效果很差，不采用这个trick；学到的原型数量特别多，并且实际分类效果也并不好

# todo:
# (1) 构建神经网络，生成原型向量
# (2) 原型的增量学习：原型保留的算法设计
# 有时间可以写一下p4代码，目的是对比硬件资源占用
# 再有时间可以实现k-means初始化，对比一下结果
# 都确定下来了之后，整理一下代码规范性，方便实习生对接
# 其余就是做实验了：AUC指标，训练集的比例，其他便于模型精度的调参
# 争取我这边把ton-iot这个数据集搞定，然后实习生做另外两个就可以了，也有个参照

# 决策树和随机森林的分类概率，自己先研究一下

# 2024.8.7
# todo1：函数模块封装，整理代码逻辑：包括训练，剪枝，规则处理
# todo2：规则的上下界校准，不能仅仅用中心点和范围，以减少重叠
# todo3：将错误分类的样本提取，进行迭代更新和规则整合

# todo4：和实习生对接，把EVT估计加进去，看看结果
# todo5：考虑如何进行纯的增量更新
# todo6：整理新数据集和对比方法，完成实验结果

torch.cuda.set_device(args.gpu)

X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
    get_data_ton_iot(args.ton_iot_attack_max_samples, 
                     args.ton_iot_selected_class, 
                     args.test_split_size)

if args.dataset == 'ton-iot':
    num_classes = len(args.ton_iot_selected_class) + 1 # include normal 
feature_num = X_train.shape[1]
assert feature_num == len(feature_min)
assert feature_num == len(feature_max)

set_global_seed(args.seed)

# boosting之前的第一次初始化
this_X_train = X_train.copy()
this_y_train = y_train.copy()

class_rules = []

for boost_iter in range(5):
    # step0: init model
    print('step0: init model')

    model = prototype(num_classes, feature_num, [args.max_proto_num for _ in range(num_classes)], args.temperature, args.cal_dis, args.cal_mode).cuda()
    model = init_model(model, args.init_type, this_X_train, this_y_train, feature_min, feature_max)

    for idx in range(num_classes):
        assert model.class_prototype[idx].shape[1] == model.class_transform[idx].shape[1]
    # 先不考虑CE loss的label smoothing
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.class_prototype + model.class_transform, lr=args.learning_rate)

    train_acc = test_model_acc(model, this_X_train, this_y_train, feature_min, feature_max)
    print('before training, train acc = ', train_acc)

    # step1: train model
    print('step1: train model')
    #best_test_acc = 0
    best_train_acc = 0
    for epoch in range(args.epochs):
        
        train_acc_num = 0
        idx = 0
        
        while idx < this_X_train.shape[0]:
            train_batch = this_X_train[idx:min(idx + args.batch_size, this_X_train.shape[0])]
            train_label = this_y_train[idx:min(idx + args.batch_size, this_y_train.shape[0])]
            
            train_batch = (train_batch - feature_min) / (feature_max - feature_min)
            
            train_batch = torch.from_numpy(train_batch).float().cuda()
            train_label = torch.from_numpy(train_label).long().cuda()
            
            optimizer.zero_grad()
            logits = model(train_batch)
            _, predictions = torch.max(logits, dim=1)
            
            loss = loss_func(logits, train_label)
            loss.backward()
            optimizer.step()
            
            train_acc_num += torch.sum(predictions == train_label).item()
            idx += args.batch_size
        
        train_acc = train_acc_num / this_X_train.shape[0] * 100
        #test_acc = test_model_acc(model, X_test, y_test, feature_min, feature_max)
        
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            model.save_parameter(args.save_model + '/model_best.pth')

        print('epoch = %d/%d, train_acc = %.3f%%, best_train_acc = %.3f%%' % (epoch + 1, args.epochs, train_acc, best_train_acc))


    # step2: model prune
    # 注意要在训练集上剪枝，不要搞成测试集，并且要和训练时的数据保持一致（不要新增新类别）
    print('step2: model prune')
    model.load_parameter(args.save_model + '/model_best.pth')
    #test_acc = test_model_acc(model, X_test, y_test, feature_min, feature_max)
    #print('distance-based test acc before prune: %.3f%%' % (test_acc))
    model = model_prune(model, num_classes, this_X_train, this_y_train, feature_min, feature_max, args.max_proto_num, args.prune_T)
    #test_acc = test_model_acc(model, X_test, y_test, feature_min, feature_max)
    #print('prune_T = %d, distance-based test acc after prune: %.3f%%' % (args.prune_T, test_acc))
    print('prune done')
    model.save_parameter(args.save_model + '/model_prune.pth')


    # step3: 在训练集上计算分类阈值，并根据距离计算的方式，得到新类别识别率
    print('step3: distance and threshold calculate')
    # 8.8 修改：增加了具体的sample值，目的是进一步校准范围
    dist_list = [[[] for _ in range(model.class_prototype[idx].shape[1])] for idx in range(num_classes)]

    idx = 0
    tot = 0
    while idx < this_X_train.shape[0]:
        train_batch = this_X_train[idx:min(idx + args.batch_size, this_X_train.shape[0])]
        train_label = this_y_train[idx:min(idx + args.batch_size, this_y_train.shape[0])]
        
        train_before_norm = train_batch.copy()
        
        train_batch = (train_batch - feature_min) / (feature_max - feature_min)
        
        train_batch = torch.from_numpy(train_batch).float().cuda()
        train_label = torch.from_numpy(train_label).long().cuda()
        
        logits = model(train_batch)
        _, predictions = torch.max(logits, dim=1)
        for dim1 in range(num_classes):
            
            if model.class_prototype[dim1].shape[1] == 0:
                continue
            
            train_batch = train_batch.view(train_batch.shape[0], 1, feature_num)
            
            if model.cal_mode == 'trans_abs':
                train_batch = train_batch * model.class_transform[dim1]
                dist_class = torch.abs(train_batch - model.class_prototype[dim1])
            elif model.cal_mode == 'abs_trans':
                dist_class = torch.abs(train_batch - model.class_prototype[dim1])
                dist_class = dist_class * torch.abs(model.class_transform[dim1])
            min_dist, _ = torch.max(dist_class, dim=2)
            min_dist, indice = torch.min(min_dist, dim=1)
            
            for dim2 in range(predictions.shape[0]):
                #if predictions[dim2] == dim1: # 想一下为什么这里改了？如果只用这行是很不准确的
                if predictions[dim2] == dim1 and predictions[dim2] == train_label[dim2]:
                    dist_list[dim1][indice[dim2]].append((min_dist[dim2].item(), train_before_norm[dim2]))
                    tot += 1
                    
        idx += args.batch_size

    print('distance-based train acc after prune = %.3f%%' % (tot / this_X_train.shape[0] * 100))

    for dim1 in range(num_classes):
        for dim2 in range(model.class_prototype[dim1].shape[1]):
            dist_list[dim1][dim2].sort(reverse=True, key=lambda x: x[0])
            
            # notice may 0 sample (in theory), so add '0'
            # 8.8：但是应该又不会出现这种情况了，因为已经剪枝掉了，所以不用考虑这种情况
            #if len(dist_list[dim1][dim2]) == 0:
            #    dist_list[dim1][dim2].append(0)

    print('distance list of each prototype(sorted and print top 10)')
    for dim1 in range(num_classes):
        for dim2 in range(model.class_prototype[dim1].shape[1]):
            # print(dist_list[dim1][dim2][:10])
            print([t[0] for t in dist_list[dim1][dim2][:10]])
            
    # 从输出中我们可以看出，cls_threshold设置为0.01~0.02之间比较好，不会损失太多正确分类的样本
    # 我们的第一优先级是保证正确分类，其次才是未知类别的识别，所以这里阈值不能设置的太小，要不然太影响分类精度了

    cost_cnt = 0
    for dim1 in range(num_classes):
        for dim2 in range(model.class_prototype[dim1].shape[1]):
            # 要保证pop之后至少还剩一个元素
            #while dist_list[dim1][dim2][0][0] > args.cls_threshold and len(dist_list[dim1][dim2]) > 1:
            simple_threshold = dist_list[dim1][dim2][10][0]
            while dist_list[dim1][dim2][0][0] > simple_threshold and len(dist_list[dim1][dim2]) > 1:
                dist_list[dim1][dim2].pop(0)
                cost_cnt += 1
                
                #if len(dist_list[dim1][dim2]) == 0:
                #    dist_list[dim1][dim2].append(0)
                #    break

    print('cost due to distance threshold = %d/%d, rate = %.3f%%' % (cost_cnt, tot, cost_cnt / tot * 100))

    '''
    for unknown_class_idx in range(len(unknown_data_list)):
        
        unknown_data = unknown_data_list[unknown_class_idx]
        unknown_label = unknown_label_list[unknown_class_idx]
        
        idx = 0
        tot = 0
        while idx < unknown_data.shape[0]:
            train_batch = unknown_data[idx:min(idx + args.batch_size, unknown_data.shape[0])]
            train_batch = (train_batch - feature_min) / (feature_max - feature_min)
            train_batch = torch.from_numpy(train_batch).float().cuda()
            
            logits = model(train_batch)
            _, predictions = torch.max(logits, dim=1)
            for dim1 in range(num_classes):
                
                if model.class_prototype[dim1].shape[1] == 0:
                    continue
                
                train_batch = train_batch.view(train_batch.shape[0], 1, feature_num)
                
                if model.cal_mode == 'trans_abs':
                    train_batch = train_batch * model.class_transform[dim1]
                    dist_class = torch.abs(train_batch - model.class_prototype[dim1])
                elif model.cal_mode == 'abs_trans':
                    dist_class = torch.abs(train_batch - model.class_prototype[dim1])
                    dist_class = dist_class * torch.abs(model.class_transform[dim1])
                
                min_dist, _ = torch.max(dist_class, dim=2)
                min_dist, indice = torch.min(min_dist, dim=1)
                for dim2 in range(predictions.shape[0]):
                    if predictions[dim2] == dim1:
                        #dist_list[dim1][indice[dim2]].append(min_dist[dim2].item())
                        if min_dist[dim2] > dist_list[dim1][indice[dim2]][0][0]: # distance list经过筛选之后，第一个元素就是分类阈值了
                            tot += 1
                        
            idx += args.batch_size

        if args.dataset == 'ton-iot':
            attack_classes = ['backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'runsomware', 'scanning', 'xss']
            print('unknown class = %d, name = %s, distance-based detect count = %d/%d, rate = %.3f%%' % (unknown_label[0], attack_classes[unknown_label[0] - 1], tot, unknown_data.shape[0], tot / unknown_data.shape[0] * 100))
    '''

    # step4: 转化成匹配规则，得到最终结果
    print('change prototype to match rules and get final results')

    temp_class_rules = []
    
    for dim1 in range(num_classes):
        temp_1 = []
        for dim2 in range(model.class_prototype[dim1].shape[1]):
            temp_2 = []
            for dim3 in range(feature_num):
                temp_2.append([1e10, -1e10])
            
            for dim4 in range(len(dist_list[dim1][dim2])):
                sample = dist_list[dim1][dim2][dim4][1]
                for dim5 in range(feature_num):
                    temp_2[dim5][0] = min(temp_2[dim5][0], sample[dim5])
                    temp_2[dim5][1] = max(temp_2[dim5][1], sample[dim5])
                
                '''
                fmin_i = feature_min[dim3]
                fmax_i = feature_max[dim3]
                trans_i = torch.abs(model.class_transform[dim1])[0][dim2][dim3].item()
                proto_i = model.class_prototype[dim1][0][dim2][dim3].item()
                thres_i = dist_list[dim1][dim2][0] * 1.1
                
                if model.cal_mode == 'trans_abs':
                    threshold_1 = fmin_i + (fmax_i - fmin_i) * (proto_i - thres_i) / trans_i
                    threshold_2 = fmin_i + (fmax_i - fmin_i) * (proto_i + thres_i) / trans_i
                elif model.cal_mode == 'abs_trans':
                    threshold_1 = fmin_i + (fmax_i - fmin_i) * (proto_i - thres_i / trans_i)
                    threshold_2 = fmin_i + (fmax_i - fmin_i) * (proto_i + thres_i / trans_i)
                
                if threshold_1 < threshold_2:
                    temp_2.append([threshold_1, threshold_2])
                else:
                    temp_2.append([threshold_2, threshold_1])
                '''
                    
            temp_1.append(temp_2)
        temp_class_rules.append(temp_1)

    # step5: 新旧模型boosting
    # temp_class_rules的合并
    
    if boost_iter == 0:
        class_rules = temp_class_rules
    else:
        for dim1 in range(num_classes):
            for dim2 in range(len(temp_class_rules[dim1])):
                class_rules[dim1].append(temp_class_rules[dim1][dim2])

    def check_rule(sample, thres_list):
        
        for dim1 in range(sample.shape[0]):
            if sample[dim1] < thres_list[dim1][0] or sample[dim1] > thres_list[dim1][1]:
                return False
        
        return True


    for unknown_class_idx in range(len(unknown_data_list)):
        
        unknown_data = unknown_data_list[unknown_class_idx]
        unknown_label = unknown_label_list[unknown_class_idx]
        
        tot = 0
        for idx in range(unknown_data.shape[0]):
            
            sample = unknown_data[idx]
            cls_res = []
            
            for dim1 in range(num_classes):
                for dim2 in range(len(class_rules[dim1])):
                    if check_rule(sample, class_rules[dim1][dim2]):
                        
                        if dim1 not in cls_res:
                            cls_res.append(dim1)
            
            if len(cls_res) >= 1:
                tot += 1
        
        if args.dataset == 'ton-iot':
            attack_classes = ['backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'runsomware', 'scanning', 'xss']
            print('unknown class = %d, name = %s, final rule-based detect count = %d/%d, rate = %.3f%%' % (unknown_label[0], attack_classes[unknown_label[0] - 1], unknown_data.shape[0] - tot, unknown_data.shape[0], 100 - tot / unknown_data.shape[0] * 100))
            
    '''
    idx = 0
    tot_1 = 0
    tot_2 = 0
    while idx < X_test.shape[0]:
        test_batch = X_test[idx:min(idx + args.batch_size, X_test.shape[0])]
        test_label = y_test[idx:min(idx + args.batch_size, y_test.shape[0])]
        
        test_batch = (test_batch - feature_min) / (feature_max - feature_min)
        
        test_batch = torch.from_numpy(test_batch).float().cuda()
        test_label = torch.from_numpy(test_label).long().cuda()
        
        logits = model(test_batch)
        _, predictions = torch.max(logits, dim=1)
        for dim1 in range(num_classes):
            
            if model.class_prototype[dim1].shape[1] == 0:
                continue
            
            test_batch = test_batch.view(test_batch.shape[0], 1, feature_num)
            
            if model.cal_mode == 'trans_abs':
                test_batch = test_batch * model.class_transform[dim1]
                dist_class = torch.abs(test_batch - model.class_prototype[dim1])
            elif model.cal_mode == 'abs_trans':
                dist_class = torch.abs(test_batch - model.class_prototype[dim1])
                dist_class = dist_class * torch.abs(model.class_transform[dim1])
            min_dist, _ = torch.max(dist_class, dim=2)
            min_dist, indice = torch.min(min_dist, dim=1)
            
            for dim2 in range(predictions.shape[0]):
                if predictions[dim2] == dim1 and predictions[dim2] == test_label[dim2]:
                    tot_1 += 1
                    if min_dist[dim2] <= dist_list[dim1][indice[dim2]][0][0]:
                        tot_2 += 1
                    
        idx += args.batch_size

    print('distance-based test acc after prune = %.3f%%' % (tot_1 / X_test.shape[0] * 100))
    print('distance-based test acc after threshold setting = %.3f%%' % (tot_2 / X_test.shape[0] * 100))
    '''
    
    # 在有向图中，给定两个点，判断是否存在有向路径
    def have_path(restrict, st, ed):
        # 构建邻接表，注意允许重边
        graph = {}
        for u, v in restrict:
            if u not in graph:
                graph[u] = []
            graph[u].append(v)
        
        # 深度优先搜索判断是否有路径
        def dfs(node, target, visited):
            if node == target:
                return True
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, target, visited):
                        return True
            return False
    
        # 用集合 visited 来避免重复访问节点
        visited = set()
        return dfs(st, ed, visited)

    
    # 9.4 使用原型规则优先级排序，减少冲突数量
    # 舍弃表项
    def cal_rule_num(cls_fix): # 输入的cls_fix已经是排好序了的
        
        res = 0
        for dim1 in range(num_classes):
            for dim2 in range(len(class_rules[dim1])):
                res += 1
        
        # 定义约束条件
        restrict = []
        
        for item in cls_fix:
            gt = cls_fix[item]
            proto_list = [s for s in item.split('#') if s] # 过滤掉空字符串
            for idx in range(len(proto_list)):
                proto_list[idx] = int(proto_list[idx])
            
            # 判断所有原型是否都属于同一类别
            flag = 1
            for idx in range(len(proto_list)):
                if proto_list[idx] // 100 != proto_list[0] // 100:
                    flag = 0
                    break
            if flag == 1: # 全相同
                if gt != proto_list[0] // 100: # 这种情况必须加一，否则无需处理
                    res += 1
                continue
            
            # 判断gt类别是否存在于原型中，不存在则直接加一
            flag = 1
            for idx in range(len(proto_list)):
                if gt == proto_list[idx] // 100:
                    flag = 0
                    break
            if flag == 1:
                res += 1
                continue
            
            # 开始增加约束条件（类似拓扑排序）
            gt_proto = []
            other_proto = []
            for idx in range(len(proto_list)):
                if proto_list[idx] // 100 == gt:
                    gt_proto.append(proto_list[idx])
                else:
                    other_proto.append(proto_list[idx])
            
            # 约束条件尝试
            flag = 1
            add_edge_num = 0
            for dim1 in range(len(gt_proto)):
                if flag == 0:
                    break
                for dim2 in range(len(other_proto)):
                    if have_path(restrict, other_proto[dim2], gt_proto[dim1]): # 反向判断，如果存在边
                        flag = 0
                        break
                    else:
                        add_edge_num += 1
                        restrict.append((gt_proto[dim1], other_proto[dim2]))
            
            if flag == 0: # 当前规则不能被满足
                res += 1
                # 回退，删掉增加的有向边
                for _ in range(add_edge_num):
                    restrict.pop()
            # 否则无需进行任何操作，既不用回退边，也无需对res计数
            
        return res

    # 如果有多个表项都匹配成功，则返回具有最高优先级的表项结果（交换机上也是这样的）
    def solve_conflict(old_X, old_y):
        from collections import Counter
        
        cls_fix = {}
        
        for idx in range(old_X.shape[0]):
            
            sample = old_X[idx]
            cls_res = []
            
            for dim1 in range(num_classes):
                for dim2 in range(len(class_rules[dim1])):
                    if check_rule(sample, class_rules[dim1][dim2]):
                        
                        if (dim1 * 100 + dim2) not in cls_res:
                            cls_res.append(dim1 * 100 + dim2)
            
            if len(cls_res) <= 1:
                continue
            
            # cls_res has more than 1 class
            cls_res.sort(reverse=False)
            cls_res_hash = '#'
            for p in cls_res:
                cls_res_hash += str(p)
                cls_res_hash += '#'
            
            if cls_res_hash not in cls_fix:
                cls_fix[cls_res_hash] = []
            
            cls_fix[cls_res_hash].append(old_y[idx])
        
        # 9.4 按照规则的最优类别接收样本数量排序，由chatgpt生成，效率很高
        cls_fix = dict(sorted(cls_fix.items(), key=lambda item: Counter(item[1]).most_common(1)[0][1], reverse=True))
        
        for p in cls_fix:
            cls_fix[p] = statistics.mode(cls_fix[p])
        
        proto_num = 0
        for dim1 in range(num_classes):
            for dim2 in range(len(class_rules[dim1])):
                proto_num += 1
        print('prototype number = %d' % (proto_num))
        print('total conflix number = %d' % (len(cls_fix)))
        reduced_rule_num = cal_rule_num(cls_fix)
        print('total rule number = %d, save %d/%d rules' % (reduced_rule_num, proto_num + len(cls_fix) - reduced_rule_num, len(cls_fix)))
        print(cls_fix)
        exit(0)
        
        return cls_fix


    # 这里可以灵活点，训练集和测试集都试试，看结果有区别没
    if args.solve_conflict_on == 'train_set':
        cls_fix = solve_conflict(X_train, y_train)
    elif args.solve_conflict_on == 'test_set':
        cls_fix = solve_conflict(X_test, y_test)
    print('solve match conflict:')
    print('conflict number = ', len(cls_fix))
    print(cls_fix)

    no_match = 0
    tot = 0
    tot_overlap = 0
    overlap_success = 0

    for idx in range(X_test.shape[0]):
        
        sample = X_test[idx]
        cls_res = []
        
        for dim1 in range(num_classes):
            for dim2 in range(len(class_rules[dim1])):
                if check_rule(sample, class_rules[dim1][dim2]):
                    
                    if (dim1 * 100 + dim2) not in cls_res:
                        cls_res.append(dim1 * 100 + dim2)
        
        # fix important bug: cls_res[0], not cls_res
        if len(cls_res) == 0:
            no_match += 1
        if len(cls_res) == 1 and cls_res[0] // 100 == y_test[idx]:
            tot += 1
        elif len(cls_res) >= 2:
            tot_overlap += 1
            cls_res.sort(reverse=False)
            cls_res_hash = '#'
            for p in cls_res:
                cls_res_hash += str(p)
                cls_res_hash += '#'
            if cls_fix[cls_res_hash] == y_test[idx]:
                tot += 1
                overlap_success += 1
        
    print('global rule-based accuracy on test_data = %.3f%%, samples have overlap = %d/%d, rate = %.3f%%, overlap success rate = %.3f%%, no match rate = %.3f%%' % (tot / X_test.shape[0] * 100, tot_overlap, X_test.shape[0], tot_overlap / X_test.shape[0] * 100, overlap_success / tot_overlap * 100, no_match / X_test.shape[0] * 100))

    # step6: 过滤出分类错误的训练样本，以供下一次迭代学习
    
    cls_fix = solve_conflict(X_train, y_train)
    
    no_match = 0
    tot = 0
    tot_overlap = 0
    overlap_success = 0
    
    fault_list = []

    for idx in range(X_train.shape[0]):
        
        sample = X_train[idx]
        cls_res = []
        
        for dim1 in range(num_classes):
            for dim2 in range(len(class_rules[dim1])):
                if check_rule(sample, class_rules[dim1][dim2]):
                    
                    if (dim1 * 100 + dim2) not in cls_res:
                        cls_res.append(dim1 * 100 + dim2)
        
        # fix important bug: cls_res[0], not cls_res
        if len(cls_res) == 0:
            no_match += 1
            fault_list.append(idx)
        if len(cls_res) == 1:
            if cls_res[0] // 100 == y_train[idx]:
                tot += 1
            else:
                fault_list.append(idx)
        elif len(cls_res) >= 2:
            tot_overlap += 1
            cls_res.sort(reverse=False)
            cls_res_hash = '#'
            for p in cls_res:
                cls_res_hash += str(p)
                cls_res_hash += '#'
            if cls_fix[cls_res_hash] == y_train[idx]:
                tot += 1
                overlap_success += 1
            else:
                fault_list.append(idx)
        
    print('global rule-based accuracy on train_data = %.3f%%, samples have overlap = %d/%d, rate = %.3f%%, overlap success rate = %.3f%%, no match rate = %.3f%%' % (tot / X_train.shape[0] * 100, tot_overlap, X_train.shape[0], tot_overlap / X_train.shape[0] * 100, overlap_success / tot_overlap * 100, no_match / X_train.shape[0] * 100))

    this_X_train = X_train[fault_list]
    this_y_train = y_train[fault_list]
    