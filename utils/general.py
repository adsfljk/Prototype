import torch
import torch.nn as nn
import numpy as np
import random
import statistics
from collections import Counter
import copy

batch_size = 512
def merge_rules(a, b):
    # 确定两个列表中最小的长度
    min_length = min(len(a), len(b))
    result = []

    # 合并相同索引的元素
    for i in range(min_length):
        result.append(a[i] + b[i])
    
    # 添加多余的元素
    if len(a) > min_length:
        result.extend(a[min_length:])
    elif len(b) > min_length:
        result.extend(b[min_length:])

    return result


def test_model_acc(model, X_test, y_test, feature_min, feature_max):
    test_acc_num = 0
    idx = 0
    while idx < X_test.shape[0]:
        test_batch = X_test[idx:min(idx + batch_size, X_test.shape[0])]
        test_label = y_test[idx:min(idx + batch_size, y_test.shape[0])]
        
        test_batch = (test_batch - feature_min) / (feature_max - feature_min)
        
        test_batch = torch.from_numpy(test_batch).float().cuda()
        test_label = torch.from_numpy(test_label).long().cuda()
        
        logits = model(test_batch)
        
        _, predictions = torch.max(logits, dim=1)
        test_acc_num += torch.sum(predictions == test_label).item()
        
        idx += batch_size

    return test_acc_num / X_test.shape[0] * 100


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def sort_a_b(a,b):
    combined = list(zip(a, b))

    # 按 a 列表的值进行排序
    sorted_combined = sorted(combined)

    # 解包成两个排序后的列表
    a_sorted, b_sorted = zip(*sorted_combined)
    return a_sorted, b_sorted 

def three_sigma(data,sigma_time):
    mean = np.mean(data)

    # 计算标准差
    std_dev = np.std(data)

    # 计算上限阈值 (仅考虑上限)
    upper_bound = mean + sigma_time * std_dev
    return upper_bound

def boxplot(data):
  # 计算 Q1 (25%) 和 Q3 (75%)
  Q1 = np.percentile(data, 25)
  Q3 = np.percentile(data, 75)
  # 计算 IQR
  IQR = Q3 - Q1
  # 计算上下边界
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  return upper_bound


def set_prioity(cls_fix):

    # 存储图形ID及其优先级的字典
    priority_dict = {}
    # 遍历原始字典并提取图形ID及其优先级
    for key, value in cls_fix.items():
        ids = key.strip('#').split('#')
        # 跳过重叠的都是同一类的原型
        if len(set([int(i)//10000 for i in ids]))==1:
            continue

        for id in ids:

            if id not in priority_dict:
                priority_dict[id] = 1

            if int(id)//10000 == value:       
                priority_dict[id] += 1
    # 优先级升序排序
    priority_dict = sorted(priority_dict.items(), key=lambda x: (x[1], int(x[0])))

    # 将优先级替换为排序后的排位序号+1
    priority_dict = {item[0]: index + 1 for index, item in enumerate(priority_dict)}

    # 检查不满足优先级的加入class_rules
    add_class_rules_keys = []
    for key, value in cls_fix.items():
        ids = key.strip('#').split('#')
        # 跳过重叠的都是同一类的原型
        if len(set([int(i)//10000 for i in ids]))==1:
            continue

        priori_list = [priority_dict[i] for i in ids]

        prior_id = ids[np.argmax(priori_list)]
        if int(prior_id)//10000 != value:
            add_class_rules_keys.append(key)


        
    return priority_dict , add_class_rules_keys



def check_rule(sample, thres_list):

    for dim1 in range(sample.shape[0]):
        if sample[dim1] < thres_list[dim1][0] or sample[dim1] > thres_list[dim1][1]:
            return False
    
    return True

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



def cal_rule_num(cls_fix,num_classes,class_rules): # 输入的cls_fix已经是排好序了的
        
        res = 0

        # 定义约束条件
        restrict = []
        reduced_cls_fix = {}
        # print(cls_fix)
        # exit()
        for item in cls_fix:
            gt = cls_fix[item]
            proto_list = [s for s in item.split('#') if s] # 过滤掉空字符串
            for idx in range(len(proto_list)):
                proto_list[idx] = int(proto_list[idx])
            
            # 判断所有原型是否都属于同一类别
            flag = 1
            for idx in range(len(proto_list)):
                if proto_list[idx] // 10000 != proto_list[0] // 10000:
                    flag = 0
                    break
            if flag == 1: # 全相同
                if gt != proto_list[0] // 10000: # 这种情况必须加一，否则无需处理
                    reduced_cls_fix[item] = gt
                    res += 1
                continue
            
            # 判断gt类别是否存在于原型中，不存在则直接加一
            flag = 1
            for idx in range(len(proto_list)):
                if gt == proto_list[idx] // 10000:
                    flag = 0
                    break
            if flag == 1:
                reduced_cls_fix[item] = gt
                res += 1
                continue
            
            # 开始增加约束条件（类似拓扑排序）
            gt_proto = []
            other_proto = []
            for idx in range(len(proto_list)):
                if proto_list[idx] // 10000 == gt:
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
                reduced_cls_fix[item] = gt
                res += 1
                # 回退，删掉增加的有向边
                for _ in range(add_edge_num):
                    restrict.pop()
            # 否则无需进行任何操作，既不用回退边，也无需对res计数
            
        return res,reduced_cls_fix

# 如果有多个表项都匹配成功，则返回具有最高优先级的表项结果（交换机上也是这样的）
def solve_conflict(old_X, old_y,num_classes,class_rules):
    
 
    cls_fix = {}
    
    for idx in range(old_X.shape[0]):
        
        sample = old_X[idx]
        cls_res = []
        
        for dim1 in range(num_classes):
            for dim2 in range(len(class_rules[dim1])):
                if check_rule(sample, class_rules[dim1][dim2]):
                    
                    if (dim1 * 10000 + dim2) not in cls_res:
                        cls_res.append(dim1 * 10000 + dim2)
        
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
    
    cls_fix = dict(sorted(cls_fix.items(), key=lambda item: Counter(item[1]).most_common(1)[0][1], reverse=True))

    for p in cls_fix:
        cls_fix[p] = statistics.mode(cls_fix[p])
    # confilct rule num
    conflict_rule_num,reduced_cls_fix = cal_rule_num(cls_fix,num_classes,class_rules)

    return cls_fix,reduced_cls_fix,conflict_rule_num


def intersect_rules(list1, list2):
    # 创建一个空列表存储交集后的规则
    intersected_rules = []

    # 确保两个列表长度相同才能逐项比较
    if len(list1) != len(list2):
        raise ValueError("两个列表的长度应相同")

    # 遍历两个列表的规则，并计算交集
    for i in range(len(list1)):
        lower_bound = max(list1[i][0], list2[i][0])  # 取较大的下界
        upper_bound = min(list1[i][1], list2[i][1])  # 取较小的上界
        
        # 如果下界小于上界，表示有交集
        if lower_bound <= upper_bound:
            intersected_rules.append([lower_bound, upper_bound])
        else:
            return -1
    return intersected_rules

def conflict_2rule(cls_fix,class_rules,num_classes):
    conflict_rules = [[] for i in range(num_classes)]
    conflict_rules_pri = [[] for i in range(num_classes)]
    for key, value in cls_fix.items():
        ids = key.strip('#').split('#')
        for i,id in enumerate(ids):
            dim1 = int(id) // 10000
            dim2 = int(id) % 10000
            if i == 0:
                tmp = copy.deepcopy(class_rules[dim1][dim2])
            else:
                tmp2 = copy.deepcopy(class_rules[dim1][dim2])
                tmp = intersect_rules(tmp,tmp2)
                if tmp == -1:
                    break
        if tmp == -1:
            continue
        conflict_rules_pri[value].append(10000 + len(ids))
        conflict_rules[value].append(tmp)
    return conflict_rules,conflict_rules_pri
        


                


def export_range(num_classes,log_class,\
                 class_rules,priority_dict,\
                 data_plane_remain_rule,
                 conflict_rules,conflict_rules_pri,\
                 remain_conflict_rules):
    param_names = [
        "ipv4_protocol", "ipv4_ihl", "ipv4_tos", "ipv4_flags", 
        "ipv4_ttl", "meta_dataoffset", "meta_flags", 
        "meta_window", "meta_udp_length", "ipv4_totallen"
    ]
    param_info = {
        "ipv4_protocol": 255,  # IPv4 protocol range
        "ipv4_ihl": 15,  # IPv4 IHL range
        "ipv4_tos": 255,  # IPv4 TOS range
        "ipv4_flags": 7,  # IPv4 flags range
        "ipv4_ttl": 255,  # IPv4 TTL range
        "meta_dataoffset":15,  # Data offset range
        "meta_flags":255,  # Meta flags range
        "meta_window": 65535,  # Window size range
        "meta_udp_length": 65535,  # UDP length range
        "ipv4_totallen": 65535  # Total length range
    }

    if log_class == -1:
        log_label_name = "ALL"
    else:
        log_label_name = attack_classes[log_class]
    
    # cal entry priority
    # cls_fix = solve_conflict(X_train, y_train,num_classes,class_rules)
    # priority_dict = set_prioity(cls_fix)
    # TODO 检查rule是否矛盾

    # 打开文件时使用 'w' 模式清空文件内容
    with open("pro_entity/pro_iscx_"+log_label_name+"_setup.py", "w") as f:

        f.write(
"""p4 = bfrt.pro_iscx.pipe

def clear_all(verbose=True, batching=True):
global p4
global bfrt
for table_types in (['MATCH_DIRECT', 'MATCH_INDIRECT_SELECTOR'],
                    ['SELECTOR'],
                    ['ACTION_PROFILE']):
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in table_types:
            if verbose:
                print("Clearing table {:<40} ... ".
                    format(table['full_name']), end='', flush=True)
            table['node'].clear(batch=batching)
            if verbose:
                print('Done')
clear_all(verbose=True)\n
tb_packet_cls = p4.Ingress.tb_packet_cls\n
""")

        # 形状（类别，原型，特征数目，2）
        for dim1 in range(num_classes):
            # 生成插入表项代码并写入文件
                # features
                for dim2 in range(len(class_rules[dim1])):
                    params = []
                    for param_name, (start, end) in zip(param_names, class_rules[dim1][dim2]):
                        if start<0:
                            start = 0
                        if end > param_info[param_name]:
                            end = param_info[param_name]

                        params.append(f"{param_name}_start={start}, {param_name}_end={end}")

                    params.append(f"match_priority={priority_dict[dim1*10000 + dim2]}")
                    params.append(f"port={dim1}")  # 假设每个类别对应不同的端口

                    params_str = ", ".join(params)
                    f.write(f"tb_packet_cls.add_with_ac_packet_forward({params_str})\n")


        f.write("bfrt.complete_operations()")

