import torch
import torch.nn as nn
import numpy as np
import random
import statistics

batch_size = 512

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
    
    for p in cls_fix:
        cls_fix[p] = statistics.mode(cls_fix[p])

    
        
    return cls_fix



def export_range(X_train, y_train,attack_classes,num_classes,class_rules,log_class):
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
    cls_fix = solve_conflict(X_train, y_train,num_classes,class_rules)
    priority_dict = set_prioity(cls_fix)
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

