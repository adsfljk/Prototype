import torch
import torch.nn as nn
import numpy as np
import random
import statistics
from collections import Counter
import copy

batch_size = 512


def merge_rules(c, d):
    """合并两个规则列表，确保没有重复的规则，并且第二个参数规则优先"""
    a = copy.deepcopy(c)
    b = copy.deepcopy(d)
    result = []

    # 将所有类别下的规则都存储为一个集合用于去重
    seen_rules = set()

    # 优先处理第二个参数 `b`，并保留这些规则
    for rules_b in b:
        merged_result_b = []
        for rule in rules_b:
            rule_tuple = tuple(map(tuple, rule))  # 将规则转换为元组，便于去重
            if rule_tuple not in seen_rules:
                seen_rules.add(rule_tuple)
                merged_result_b.append(rule)
        result.append(merged_result_b)

    # 处理第一个参数 `a`，删除与第二个参数 `b` 中重复的规则
    for i, rules_a in enumerate(a):
        if i < len(result):  # 如果已经有合并的结果，则继续在这个类别添加
            for rule in rules_a:
                rule_tuple = tuple(map(tuple, rule))  # 转换为元组便于比较
                if rule_tuple not in seen_rules:  # 仅保留不重复的规则
                    result[i].append(rule)
        else:
            merged_result_a = []
            for rule in rules_a:
                rule_tuple = tuple(map(tuple, rule))  # 转换为元组便于比较
                if rule_tuple not in seen_rules:
                    seen_rules.add(rule_tuple)
                    merged_result_a.append(rule)
            result.append(merged_result_a)

    # 确保输出维度与第一个参数 `a` 的类别数目一致
    while len(result) < len(a):
        result.append([])

    return result

def remove_rules(class_rules, temp_class_rules):
    class_rules_new = copy.deepcopy(class_rules)
    temp_class_rules_new = copy.deepcopy(temp_class_rules)
    # 创建一个新的列表，用于存放去除后的规则
    new_class_rules = []
    
    # 遍历所有类别
    for class_idx in range(len(class_rules_new)):
        class_rule = class_rules_new[class_idx]
        temp_rule = temp_class_rules_new[class_idx]
        
        # 新的规则存储空间
        updated_rules = []
        
        # 遍历当前类别下的所有规则
        for rule in class_rule:
            # 如果当前规则不在 temp_class_rules 中，则保留
            if rule not in temp_rule:
                updated_rules.append(rule)
        
        # 如果该类别所有规则都被移除，则保留空列表
        new_class_rules.append(updated_rules)
    
    return new_class_rules

def is_contained(delete_class_rules, delete_conflict_rules):
    # Helper function to check if one rule is contained within another
    def is_rule_contained(rule, candidate):
        if len(rule) != len(candidate):
            return False
        for r, c in zip(rule, candidate):
            if not (r[0] >= c[0] and r[1] <= c[1]):
                return False
        return True

    # Iterate through each category
    for class_idx in range(len(delete_conflict_rules)):
        conflict_rules = delete_conflict_rules[class_idx]
        class_rules = delete_class_rules[class_idx]
        
        # For each conflict rule, check if it is contained in any of the class rules
        for c_rule in conflict_rules:
            contained = any(is_rule_contained(c_rule, d_rule) for d_rule in class_rules)
            if not contained:
                return False
    
    return True

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

def convert_bounds_to_integers(rules):
    new_rules = copy.deepcopy(rules)
    # 遍历规则列表的每一个元素
    for category in range(len(new_rules)):
        for rule in range(len(new_rules[category])):
            for feature in range(len(new_rules[category][rule])):
                # 将上下界转换为整数
                new_rules[category][rule][feature][0] = int(new_rules[category][rule][feature][0])  # 下界
                new_rules[category][rule][feature][1] = int(new_rules[category][rule][feature][1])  # 上界
    return new_rules

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
        # print("ids",ids,len(ids),value)
        conflict_rules_pri[int(value)].append(10000 + len(ids))
        conflict_rules[int(value)].append(tmp)
    return conflict_rules,conflict_rules_pri
    
def add_rules(args,dim1,param_names,param_info,f,class_rules):
    s = 0
    for dim2 in range(len(class_rules[dim1])):
        params = []
        src_port_value = 0
        src_port_mask = 0
        dst_port_value = 0
        dst_port_mask = 0
        for param_name, (start, end) in zip(param_names, class_rules[dim1][dim2]):
            if start<0:
                start = 0
            if end > param_info[param_name]:
                end = param_info[param_name]

            # 处理 srcPort_0 到 srcPort_15
            if 'srcport' in param_name:

                bit_position = int(param_name.split('_')[-1])  # 提取 bit 位置 (0 到 15)
                if start == 0 and end == 1:
                    src_port_mask &= ~(1 << bit_position)  # mask 的这一位设为 0（不关心该位）
                elif start == 0 and end == 0:
                    src_port_value &= ~(1 << bit_position)  # value 的这一位设为 0
                    src_port_mask |= (1 << bit_position)  # mask 的这一位设为 1（明确匹配 0）
                elif start == 1 and end == 1:
                    src_port_value |= (1 << bit_position)  # value 的这一位设为 1
                    src_port_mask |= (1 << bit_position)  # mask 的这一位设为 1（明确匹配 1）

            # 处理 dstPort_0 到 dstPort_15
            elif 'dstport' in param_name:
                bit_position = int(param_name.split('_')[-1])  # 提取 bit 位置 (0 到 15)
                if start == 0 and end == 1:
                    dst_port_mask &= ~(1 << bit_position)  # mask 的这一位设为 0（不关心该位）
                elif start == 0 and end == 0:
                    dst_port_value &= ~(1 << bit_position)  # value 的这一位设为 0
                    dst_port_mask |= (1 << bit_position)  # mask 的这一位设为 1（明确匹配 0）
                elif start == 1 and end == 1:
                    dst_port_value |= (1 << bit_position)  # value 的这一位设为 1
                    dst_port_mask |= (1 << bit_position)  # mask 的这一位设为 1（明确匹配 1）

            else:
                params.append(f"{param_name}_start={start}, {param_name}_end={end}")
                # TODO 9.9暂时还没有加优先级
                # params.append(f"match_priority={priority_dict[dim1*10000 + dim2]}")
        if args.dataset == "unsw-nb15":
            # 将合并后的 srcPort 和 dstPort 写入参数
            params.append(f"meta_srcport={src_port_value}, meta_srcport_mask={src_port_mask}")
            params.append(f"meta_dstport={dst_port_value}, meta_dstport_mask={dst_port_mask}")
        # TODO 9.9暂时还没有加优先级
        # params.append(f"match_priority={priority_dict[dim1*10000 + dim2]}")
        params.append(f"port={dim1}")  # 假设每个类别对应不同的端口
        params_str = ", ".join(params)

        f.write(f"tb_packet_cls.add_with_ac_packet_forward({params_str})\n")
        s += 1
    return s

def delete_rules(args,dim1,param_names,param_info,f,class_rules):

    s = 0
    for dim2 in range(len(class_rules[dim1])):
        params = []
        src_port_value = 0
        src_port_mask = 0
        dst_port_value = 0
        dst_port_mask = 0
        for param_name, (start, end) in zip(param_names, class_rules[dim1][dim2]):
            if start<0:
                start = 0
            if end > param_info[param_name]:
                end = param_info[param_name]

            # 处理 srcPort_0 到 srcPort_15
            if 'srcport' in param_name:
                bit_position = int(param_name.split('_')[-1])  # 提取 bit 位置 (0 到 15)
                if start == 0 and end == 1:
                    src_port_mask &= ~(1 << bit_position)  # mask 的这一位设为 0（不关心该位）
                elif start == 0 and end == 0:
                    src_port_value &= ~(1 << bit_position)  # value 的这一位设为 0
                    src_port_mask |= (1 << bit_position)  # mask 的这一位设为 1（明确匹配 0）
                elif start == 1 and end == 1:
                    src_port_value |= (1 << bit_position)  # value 的这一位设为 1
                    src_port_mask |= (1 << bit_position)  # mask 的这一位设为 1（明确匹配 1）

            # 处理 dstPort_0 到 dstPort_15
            elif 'dstport' in param_name:
                bit_position = int(param_name.split('_')[-1])  # 提取 bit 位置 (0 到 15)
                if start == 0 and end == 1:
                    dst_port_mask &= ~(1 << bit_position)  # mask 的这一位设为 0（不关心该位）
                elif start == 0 and end == 0:
                    dst_port_value &= ~(1 << bit_position)  # value 的这一位设为 0
                    dst_port_mask |= (1 << bit_position)  # mask 的这一位设为 1（明确匹配 0）
                elif start == 1 and end == 1:
                    dst_port_value |= (1 << bit_position)  # value 的这一位设为 1
                    dst_port_mask |= (1 << bit_position)  # mask 的这一位设为 1（明确匹配 1）

            else:
                params.append(f"{param_name}_start={start}, {param_name}_end={end}")
        if args.dataset == "unsw-nb15":
            # 将合并后的 srcPort 和 dstPort 写入参数
            params.append(f"meta_srcport={src_port_value}, meta_srcport_mask={src_port_mask}")
            params.append(f"meta_dstport={dst_port_value}, meta_dstport_mask={dst_port_mask}")

        params_str = ", ".join(params)
        f.write(f"tb_packet_cls.delete({params_str})\n")
        s += 1
    return s

def check(dim1,class_rules,debug_list,param_names,param_info):
    for dim2 in range(len(class_rules[dim1])):
        params = []
        src_port_value = 0
        src_port_mask = 0
        dst_port_value = 0
        dst_port_mask = 0
        for param_name, (start, end) in zip(param_names, class_rules[dim1][dim2]):
            if start<0:
                start = 0
            if end > param_info[param_name]:
                end = param_info[param_name]

            # 处理 srcPort_0 到 srcPort_15
            if 'srcport' in param_name:
                bit_position = int(param_name.split('_')[-1])  # 提取 bit 位置 (0 到 15)
                if start == 0 and end == 1:
                    src_port_mask &= ~(1 << bit_position)  # mask 的这一位设为 0（不关心该位）
                elif start == 0 and end == 0:
                    src_port_value &= ~(1 << bit_position)  # value 的这一位设为 0
                    src_port_mask |= (1 << bit_position)  # mask 的这一位设为 1（明确匹配 0）
                elif start == 1 and end == 1:
                    src_port_value |= (1 << bit_position)  # value 的这一位设为 1
                    src_port_mask |= (1 << bit_position)  # mask 的这一位设为 1（明确匹配 1）

            # 处理 dstPort_0 到 dstPort_15
            elif 'dstport' in param_name:
                bit_position = int(param_name.split('_')[-1])  # 提取 bit 位置 (0 到 15)
                if start == 0 and end == 1:
                    dst_port_mask &= ~(1 << bit_position)  # mask 的这一位设为 0（不关心该位）
                elif start == 0 and end == 0:
                    dst_port_value &= ~(1 << bit_position)  # value 的这一位设为 0
                    dst_port_mask |= (1 << bit_position)  # mask 的这一位设为 1（明确匹配 0）
                elif start == 1 and end == 1:
                    dst_port_value |= (1 << bit_position)  # value 的这一位设为 1
                    dst_port_mask |= (1 << bit_position)  # mask 的这一位设为 1（明确匹配 1）

            else:
                params.append([start,end])

        # 将合并后的 srcPort 和 dstPort 写入参数
        params.append([src_port_value,src_port_mask])

        params.append([dst_port_value,dst_port_mask])
        if params == debug_list:
            return 1


    return 0


def export_range(idx_log_class,args,num_classes,log_class,attack_classes,\
                class_rules,last_class_rules,last_remain_pro_rule,\
                conflict_rules,conflict_rules_pri,\
                last_conflict_rules,last_conflict_rules_pri,\
                remain_conflict_rules,remain_conflict_rules_pri):
    '''
    class_rules  conflict_rules 
    last_class_rules    last_remain_pro_rule 
    last_conflict_rules remain_conflict_rules
    注意，对于生成的python下发表项文件，名称一定得小写
    '''
    if args.dataset == "cicids-2018":
        # 若位数没有固定 流级别，则使用此函数
        pkt_feat_bits =  [5, 7, 11, 9, 9, 12, 1, 11]
        param_info = {
            'f0': 2**5-1,
            'f1': 2**7-1,
            'f2': 2**11-1,
            'f3': 2**9-1,
            'f4': 2**9-1,
            'f5': 2**12-1,
            'f6': 2**1-1,
            'f7': 2**11-1,


        }



    elif args.dataset == "ton-iot":
        # 若位数没有固定 流级别，则使用此函数
        
        pkt_feat_bits = [11, 11, 13, 11, 4, 4, 4, 5, 16]
        param_info = {
            'f0': 2**11-1,
            'f1': 2**11-1,
            'f2': 2**13-1,
            'f3': 2**11-1,
            'f4': 2**4-1,
            'f5': 2**4-1,
            'f6': 2**4-1,
            'f7': 2**5-1,
            'f8': 2**16-1,
        }
        # bit<11> f0;
        # bit<11> f1;
        # bit<13> f2;
        # bit<11> f3;
        # bit<4> f4;
        # bit<4> f5;
        # bit<4> f6;
        # bit<5> f7;
        # bit<16> f8;


    elif args.dataset == "unsw-nb15":
        ## 这里先将端口分开写 ，后面会合并为ternary
        param_info = {
            "ipv4_protocol": 255,     # hdr.ipv4.protocol, 8 bits -> max value 2^8 - 1
            "ipv4_flags": 7,          # hdr.ipv4.flags, 3 bits -> max value 2^3 - 1
            "ipv4_ttl": 255,          # hdr.ipv4.ttl, 8 bits -> max value 2^8 - 1
            "ipv4_totallen": 65535,   # hdr.ipv4.totalLen, 16 bits -> max value 2^16 - 1
            "meta_dataoffset": 15,    # meta.dataOffset, 4 bits -> max value 2^4 - 1
            "meta_flags": 255,        # meta.flags, 8 bits -> max value 2^8 - 1
            "meta_window": 65535,     # meta.window, 16 bits -> max value 2^16 - 1
            "meta_udp_length": 65535, # meta.udp_length, 16 bits -> max value 2^16 - 1
            # "meta_srcPort": 65535,      # meta.srcPort_0, 1 bit -> max value 2^1 - 1
            # "meta_dstPort": 65535,      # meta.srcPort_1, 1 bit -> max value 2^1 - 1
            "meta_srcport_0": 1,
            "meta_srcport_1": 1,
            "meta_srcport_2": 1,
            "meta_srcport_3": 1,
            "meta_srcport_4": 1,
            "meta_srcport_5": 1,
            "meta_srcport_6": 1,
            "meta_srcport_7": 1,
            "meta_srcport_8": 1,
            "meta_srcport_9": 1,
            "meta_srcport_10": 1,
            "meta_srcport_11": 1,
            "meta_srcport_12": 1,
            "meta_srcport_13": 1,
            "meta_srcport_14": 1,
            "meta_srcport_15": 1,

            "meta_dstport_0": 1,
            "meta_dstport_1": 1,
            "meta_dstport_2": 1,
            "meta_dstport_3": 1,
            "meta_dstport_4": 1,
            "meta_dstport_5": 1,
            "meta_dstport_6": 1,
            "meta_dstport_7": 1,
            "meta_dstport_8": 1,
            "meta_dstport_9": 1,
            "meta_dstport_10": 1,
            "meta_dstport_11": 1,
            "meta_dstport_12": 1,
            "meta_dstport_13": 1,
            "meta_dstport_14": 1,
            "meta_dstport_15": 1,
        }

    elif args.dataset == "iscx":
        pkt_feat_bits = [8,4,8,3,8,4,8,16,16,16]
        param_info = {
            "ipv4_protocol": 255,     # hdr.ipv4.protocol, 8 bits -> max value 2^8 - 1
            'ipv4_ihl': 15,
            'ipv4_tos': 255,
            'ipv4_flags': 7,
            "ipv4_ttl": 255,          # hdr.ipv4.ttl, 8 bits -> max value 2^8 - 1
            "meta_dataoffset": 15,    # meta.dataOffset, 4 bits -> max value 2^4 - 1
            'meta_flags': 255,
            "meta_window": 65535,     # meta.window, 16 bits -> max value 2^16 - 1
            "meta_udp_length":65535,
            "ipv4_totallen": 65535,   # hdr.ipv4.totalLen, 16 bits -> max value 2^16 - 1
        }


    param_names = list(param_info.keys())

    if log_class == -1:
        log_label_name = "ALL"
    else:
        log_label_name = attack_classes[log_class]
    
    if args.dataset == "unsw-nb15":
        p4_na = "unsw"
    elif args.dataset == "cicids-2018":
        p4_na = "cicids"
    elif args.dataset == "ton-iot":
        p4_na = "ton_iot"
    elif args.dataset == "iscx":
        p4_na = "iscx"

    # 打开文件时使用 'w' 模式清空文件内容
    with open("pro_entity/pro_"+str(p4_na)+"_"+log_label_name+"_setup.py", "w") as f:
        f.write("p4 = bfrt.pro_"+str(p4_na)+".pipe")
        f.write("""
tb_packet_cls = p4.Ingress.tb_packet_cls
""")
        if idx_log_class == 0:
            f.write("""
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
clear_all(verbose=False)
""")

    
        
        # 第一次
        add_rule_num = 0
        delete_rule_num = 0
        # now_rule_num = 0

        if idx_log_class == 0:
            if args.export_int == True:
                class_rules = convert_bounds_to_integers(class_rules)
                conflict_rules = convert_bounds_to_integers(conflict_rules)

            adding_class_rules = merge_rules(class_rules,conflict_rules)

            for dim1 in range(num_classes):
                add_rule_num += add_rules(args,dim1,param_names,param_info,f,adding_class_rules)
        
            # now_rule_num += add_rule_num            


        
        # 形状（类别，原型，特征数目，2）
        else:

            last_all_rules = merge_rules(last_class_rules,last_conflict_rules)
                        # 增加规则 生成插入表项代码并写入文件
                        

            remian_all_rules = merge_rules(last_remain_pro_rule,remain_conflict_rules)

            delete_all_rules = remove_rules(last_all_rules, remian_all_rules)

            remian_all_rules.append([])
            now_all_rules = merge_rules(class_rules,conflict_rules)
            adding_all_rules = remove_rules(now_all_rules, remian_all_rules)


            # 删除上一次规则
            for dim1 in range(num_classes-1):

                delete_rule_num += delete_rules(args,dim1,param_names,param_info,f,delete_all_rules)
            # 增加规则 生成插入表项代码并写入文件
            for dim1 in range(num_classes):
                add_rule_num += add_rules(args,dim1,param_names,param_info,f,adding_all_rules)

        print("rule num",add_rule_num ,delete_rule_num )

        f.write("""bfrt.complete_operations()
""")

