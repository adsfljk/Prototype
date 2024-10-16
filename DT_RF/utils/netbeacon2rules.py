import numpy as np
import re,time
from itertools import product
import fileinput
from shutil import copyfile

def find_next_split(minz, maxz):
    count = 0
    while (minz >> count) & 1 == 0 and (minz + (1 << count)) < maxz:
        count += 1
    if (minz + (1 << count)) > maxz:
        return 1 << (count - 1)
    return 1 << count


def add_to_template(outname, placeholder, code):
    with fileinput.FileInput(outname, inplace=True) as file:
        for line in file:
            print(line.replace(placeholder, code), end='')


def split_codes(code, used_features, add=""):
    code = code[2:]  # 删除"0b"
    idx = 0
    content = ""
    for fea in used_features:
        fea_n = fea[0].lower()
        content += "codes_" + fea_n + add + "=" + "0b" + code[idx:idx + fea[1]] + ","
        idx = idx + fea[1]
    return content


# 前缀方式Prefix
# 返回值 range_to_tenary(0,48) return ([0, 32], [32, 16]),表示从值0开始，长32，下一段从值32开始，长16
def range_to_tenary(minz, maxz):  # [minz,maxz)
    if maxz <= minz:
        return [[], []]
    start_num = []
    bcount = []
    while True:
        a = find_next_split(minz, maxz)
        start_num.append(minz)
        bcount.append(a)
        if minz + a == maxz:
            break
        minz += a
    return start_num, bcount


# 根据位数和长度，得到mask
# length 是特征位数，即mask位数，num是mask覆盖的数目,即range_to_tenary返回的bcount
def get_mask(length, num):
    a = int(np.log2(num))
    result = '0b'
    for i in range(length - a):
        result += '1'
    for i in range(a):  # mask位置为0
        result += '0'
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_value_mask(s, bits):  # 将1**00转变为10000和mask 10011,bits是位数
    value = s.copy()
    mask = ['1'] * bits
    for i in range(len(s)):
        if s[i] == '*':
            value[i] = '0'  # 把*替代，随便0，1都行
            mask[i] = '0'  # 数据面mask中 0代表任意
    return ''.join(value), ''.join(mask)


# 得到feature table中基区间的range mark
# length是range mark需要的位数，valid是当前特征阈值个数，num是基区间是第几个，
def get_feature_table_range_mark(length, num, valid):
    result = '0b'
    for i in range(num):
        result += '0'
    for i in range(valid - num):
        result += '1'
    for i in range(length - valid):  # 无效的位默认为0
        result += '0'
    return result


# 得到model table中相连区间的range mark
# length是range mark位数，valid是当前特征阈值个数，a,b分别代表相连区间的左端和右端，
def get_model_table_range_mark(length, a, b, valid):
    result = ['0'] * length  # 初始化为0
    for n in range(valid):
        if n < abs(b):
            result[n] = '0'
        elif n >= a - 1:
            result[n] = '1'
        else:
            result[n] = '*'
    return result


trans = lambda x: list(map(float, x.strip('[').strip(']').split(',')))


def list_to_proba(ll):  # 将rf叶子节点中的value 变成概率
    # if "\\n" in ll:
    #     ll = ll.replace('\\n', ', ')

    ll = trans(ll)
    re = []
    for i in ll:
        re.append(i / np.sum(ll))
    return re


# 得到feature table表项，包括key和action的参数
# feat_dict 各个特征的阈值，key_bits特征占的位数，key_encode_bits range mark占的位数，pkts第几个包,可选，
# 返回字典，key为特征，value为列表，列表中每一项是对应特征表中一条表项，包括：[优先级, value, mask, action参数,（包数）]
def get_feature_table_entries(feat_dict, key_bits, key_encode_bits, pkts=None):
    feat_table_datas = {}

    for key in feat_dict.keys():
        thres = feat_dict[key]
        if "avg" in key and pkts & (pkts - 1) != 0:  # pkts 不是2的幂次时
            print("avg in key, pkts ", pkts & (pkts - 1))
            thres = list(np.array(thres) * pkts)
        feat_table = []
        sum1 = 0
        priority = 1
        for i in range(len(thres)):
            best_start = 0
            best_end = 0
            min_entries = 100000
            end = thres[i]
            start_time = time.time()

            while (i != len(thres) - 1 and end < thres[i + 1]) or (i == len(thres) - 1 and end < (2 ** key_bits[key])):
                right_entries = len(range_to_tenary(thres[i], end)[0])
                start = thres[i - 1] if i > 0 else 0
                while start >= 0:
                    temp = range_to_tenary(start, end)
                    now_entries = len(temp[0]) + right_entries
                    if now_entries < min_entries:
                        min_entries = now_entries
                        best_start = start
                        best_end = end
                    if len(temp[0]) > 1 and temp[1][0] - temp[1][1] < 0:
                        start += temp[1][0] - temp[1][1]
                    else:
                        break
                # if time.time()-start_time>1: #控制时间如果有必要
                #     break
                if len(temp[0]) > 1 and temp[1][-1] - temp[1][-2] < 0:
                    end += temp[1][-2] - temp[1][-1]
                else:
                    break
            # print(min_entries,thres[i-1],thres[i],best_start,best_end)
            temp = range_to_tenary(thres[i], best_end)
            for j in range(len(temp[0])):
                feat_table.append([priority, temp[0][j], int(get_mask(key_bits[key], temp[1][j]), 2),
                                   int(get_feature_table_range_mark(key_encode_bits[key], i + 1, len(thres)), 2)])
                if pkts is not None:
                    feat_table[-1].append(pkts)
            priority += 1
            temp = range_to_tenary(best_start, best_end)
            for j in range(len(temp[0])):
                feat_table.append([priority, temp[0][j], int(get_mask(key_bits[key], temp[1][j]), 2),
                                   int(get_feature_table_range_mark(key_encode_bits[key], i, len(thres)), 2)])
                if pkts is not None:
                    feat_table[-1].append(pkts)
            priority += 1

            sum1 += min_entries
        print("The entries of {} is {}.".format(key, len(feat_table)))
        feat_table_datas[key] = feat_table
    return feat_table_datas


def get_bin_table(keys, bin_count_bits, QL=4):  # bin_count_bits是bin count变量位数
    bin_table_data = []
    mask_value = (((2 ** bin_count_bits) - 1) >> QL) << QL
    for key in keys:
        if "bin" in key:
            start_value = int(key[4:]) * 2 ** QL
            bin_table_data.append([start_value, mask_value])
    return bin_table_data


# Get all the thresholds that appear in the tree
def get_rf_feature_thres(model_file, keys, tree_num):
    # model_file: tree model file, keys: the list of features, tree_num: the number of trees
    feat_dict = {}
    for key in keys:
        feat_dict[key] = []
    for i in range(tree_num):
        with open(model_file + '_{}.dot'.format(i), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if "[" in line:
                m = re.search(r".*\[label=\"(.*?) <= (.*?)\\n.*", line.strip(), re.M | re.I)
                if m:
                    feat_dict[m.group(1)].append(float(m.group(2)))
    for key in feat_dict.keys():
        for i in range(len(feat_dict[key])):
            feat_dict[key][i] = int(
                feat_dict[key][i]) + 1  # rounding down, then adding 1 , because the node in rf is f<=a
        feat_dict[key] = list(np.unique(np.array(feat_dict[key])))
    return feat_dict


# Get the model table table entries
def get_rf_trees_table_entries(model_file, keys, feat_dict, key_encode_bits, tree_num, pkts=None):
    # model_file: model file，feat_dict: the thresholds of each feature, key_encode_bits: range mark,
    # tree_num: the number of trees in the forest, pkts: the first few packets, optional
    # The return value is a list, each element represents a table item, the content is the range mark of each feature and the classification result
    tree_data = []
    tree_leaves = []  # Each row is a leaf node, recording that smallest threshold index in left subtree and smallest threshold index (negative) in right subtree on the path of that leaf node
    trees = []
    leaf_index = []
    leaf_info = []
    trees.append(len(tree_leaves))
    for i in range(tree_num):
        with open(model_file + '_{}.dot'.format(i), 'r') as f:
            lines = f.readlines()
        nodes = {}
        for j in range(len(lines)):
            line = lines[j]
            if "label=\"" in line and "->" not in line:
                if "[label=\"gini" in line or "[label=\"entropy" in line:
                    m = re.search(r"(.*?) \[label=.*value = (.*?)\\nclass.*", line.strip(), re.M | re.I)
                    nodes[m.group(1)] = {}
                    nodes[m.group(1)]['path'] = [1000, 0] * len(keys)  # assumption that there are no more than 1000 different feature thresholds.
                    leaf_info.append(list_to_proba(m.group(2)))
                    leaf_index.append(int(m.group(1)))
                else:
                    m = re.search(r"(.*?) \[label=\"(.*?) <= (.*?)\\n.*", line.strip(), re.M | re.I)
                    nodes[m.group(1)] = {}
                    nodes[m.group(1)]['info'] = [m.group(2), m.group(3)]  # feat and threshold
                    nodes[m.group(1)]['path'] = [1000, 0] * len(keys)
                    nodes[m.group(1)]['have_left'] = False
            if "->" in line:
                m = re.search(r"(.*?) -> (.*?) ", line.strip(), re.M | re.I)
                [feat, thre] = nodes[m.group(1)]['info']
                thre = int(float(thre)) + 1
                nodes[m.group(2)]['path'] = nodes[m.group(1)]['path'].copy()
                if not nodes[m.group(1)]['have_left']:  # left subtree
                    nodes[m.group(2)]['path'][keys.index(feat) * 2] = min(
                        nodes[m.group(2)]['path'][keys.index(feat) * 2],
                        feat_dict[feat].index(thre) + 1)
                    nodes[m.group(1)]['have_left'] = True
                else:
                    nodes[m.group(2)]['path'][keys.index(feat) * 2 + 1] = min(
                        nodes[m.group(2)]['path'][keys.index(feat) * 2 + 1],
                        -feat_dict[feat].index(thre) - 1)
                if 'have_left' not in nodes[m.group(2)].keys():  # leaf node
                    tree_leaves.append(nodes[m.group(2)]['path'])
        trees.append(len(tree_leaves))

    print("trees: ", trees, len(leaf_info))
    print("judge leaf conflict ...")
    loop_val = []
    for i in range(len(trees))[:-1]:
        loop_val.append(range(trees[i], trees[i + 1]))
    print(loop_val)
    for tup in product(*loop_val):

        flag = 0
        for f in range(len(keys)):  # Check for conflicting feature values
            a = 1000;
            b = 1000;
            for i in tup:
                a = min(tree_leaves[i][f * 2], a)
                b = min(tree_leaves[i][f * 2 + 1], b)
            if a + b <= 0:
                flag = 1
                break
        # Semantic conflict check can be added here
        if flag == 0:
            # print("-- ",tup,sigmoid(leafs[i]+leafs[j]))
            if pkts is None:
                tree_data.append([])  #
            else:
                tree_data.append([pkts])
            for f in range(len(keys)):
                a = 1000;
                b = 1000;
                for i in tup:
                    a = min(tree_leaves[i][f * 2], a)
                    b = min(tree_leaves[i][f * 2 + 1], b)
                key = keys[f]
                te = get_model_table_range_mark(key_encode_bits[key], a, b, len(feat_dict[key]))
                tree_data[-1].extend([int(get_value_mask(te, key_encode_bits[key])[0], 2),
                                      int(get_value_mask(te, key_encode_bits[key])[1],
                                          2)])  # The value and mask of each feature
            leaf_sum = leaf_info[tup[0]].copy()
            for i in tup[1:]:
                for j in range(len(leaf_sum)):
                    leaf_sum[j] += leaf_info[i][j]
            tree_data[-1].append(np.array(leaf_sum) / len(tup))  # classification probabilities list
            # print(tup,np.max(leaf_sum)/len(tup))
    return tree_data,len(leaf_info)

def binary_bit_count_list(numbers):
    # 对输入列表中的每个数字进行处理
    bit_counts = []
    for n in numbers:
        # 取整数部分
        integer_part = int(n)
        
        # 将整数部分转为二进制字符串并去掉前缀 "0b"
        binary_representation = bin(integer_part)[2:]
        
        # 计算二进制位数并将其添加到结果列表中
        bit_count = len(binary_representation)
        bit_counts.append(bit_count)
    return bit_counts

def export_netbeacon(dataset,feature_max, log_class_name,fea_list,action_data_bit_list,length_range_list,class_pkt_tree_num = 1):
    # 这里feature max是当前类别的，不是总体的
    source_file = "./utils/netbeacon_tmpl.p4"
    destination_file = "./entity/netbeacon/"+str(dataset)+'/'+log_class_name+".p4"
    setup_file = "./entity/netbeacon/"+str(dataset)+'/'+log_class_name+"_setup.py"
    copyfile(source_file, destination_file)
    num_rules = 0
    class_pkt_model_file = "output/rf_tree"
    pkt_feat = ["f%d" % i for i in range(len(feature_max))]
    ############################ 自定义内容 #############################
    # 特征占用的bits
    # a = binary_bit_count_list(feature_max)
    # print(a)
    # exit()
    if dataset == "cicids-2018":
       # 若位数没有固定 流级别，则使用此函数
        pkt_feat_bits =  [5, 7,  11, 9, 9, 12, 1, 11]
        header_names = {
            'f0': ['meta.f0', 5],
            'f1': ['meta.f1', 7],
            'f2': ['meta.f2', 11],
            'f3': ['meta.f3', 9],
            'f4': ['meta.f4', 9],
            'f5': ['meta.f5', 12],
            'f6': ['meta.f6', 1],
            'f7': ['meta.f7', 11],
        }

        # bit<5> f0;
        # bit<7> f1;
        # bit<11> f2;
        # bit<9> f3;
        # bit<9> f4;
        # bit<12> f5;
        # bit<1> f6;
        # bit<11> f7;
    elif dataset == "ton-iot":
        # 若位数没有固定 流级别，则使用此函数
        
        pkt_feat_bits = [11, 11, 13, 11, 4, 4, 4, 5, 16]
        header_names = {
            'f0': ['meta.f0', 11],
            'f1': ['meta.f1', 11],
            'f2': ['meta.f2', 13],
            'f3': ['meta.f3', 11],
            'f4': ['meta.f4', 4],
            'f5': ['meta.f5', 4],
            'f6': ['meta.f6', 4],
            'f7': ['meta.f7', 5],
            'f8': ['meta.f8', 16],
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



    elif dataset == "unsw-nb15":

        pkt_feat_bits = [
            8,   # hdr.ipv4.protocol
            # 4,   # hdr.ipv4.ihl
            # 8,   # hdr.ipv4.tos
            3,   # hdr.ipv4.flags
            8,   # hdr.ipv4.ttl
            16,  # hdr.ipv4.totalLen
            4,   # meta.dataOffset
            8,   # meta.flags
            16,  # meta.window
            16,  # meta.udp_length

            # 16,16
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # meta.srcPort_0 to meta.srcPort_15
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1   # meta.dstPort_0 to meta.dstPort_15
        ]

        
        # 参考tmpl.p4&../P4/headers.p4, 修改特征对应的PHV字段
        header_names = {
            'f0': ['hdr.ipv4.protocol', 8],
            'f1': ['hdr.ipv4.flags', 3],
            'f2': ['hdr.ipv4.ttl', 8],
            'f3': ['hdr.ipv4.totalLen', 16],
            'f4': ['meta.dataOffset', 4],
            'f5': ['meta.flags', 8],
            'f6': ['meta.window', 16],
            'f7': ['meta.udp_length', 16],
            # 'f8': ['meta.srcPort', 16],
            # 'f9': ['meta.dstPort', 16],
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

    elif dataset == "iscx":
        pkt_feat_bits = [8,4,8,3,8,4,8,16,16,16]
        header_names = {
                'f0': ['hdr.ipv4.protocol', 8],
                'f1': ['hdr.ipv4.ihl', 4],
                'f2': ['hdr.ipv4.tos', 8],
                'f3': ['hdr.ipv4.flags', 3],
                'f4': ['hdr.ipv4.ttl', 8],
                'f5': ['meta.dataOffset', 4],
                'f6': ['meta.flags', 8],
                'f7': ['meta.window', 16],
                'f8': ['meta.udp_length', 16],
                'f9': ['hdr.ipv4.totalLen', 16]}

    ##################################################################
    key_bits = {}
    for i in range(len(pkt_feat)):
        key_bits[pkt_feat[i]] = pkt_feat_bits[i]
    feat_dict_class_pkt = get_rf_feature_thres(class_pkt_model_file, pkt_feat, class_pkt_tree_num)

    # 去除重复
    feat_dict = {}
    for key in pkt_feat:
        feat_dict[key] = list(set(feat_dict_class_pkt[key]))
        feat_dict[key].sort()


    ignored_fea = []
    pkt_feat_mark_bit = []
    for k in feat_dict.keys():
        if len(feat_dict[k]) == 0:
            ignored_fea.append(k)
        pkt_feat_mark_bit.append(len(feat_dict[k]))

    range_mark_bits = {}
    for i in range(len(pkt_feat)):
        range_mark_bits[pkt_feat[i]] = pkt_feat_mark_bit[i]

    for key in ignored_fea:
        pkt_feat.remove(key)
        del feat_dict[key]
        del range_mark_bits[key]
        del key_bits[key]
        print("delete %s" % key)

    feat_table_datas = get_feature_table_entries(feat_dict, key_bits, range_mark_bits)

    tree_data_datas,num_range = get_rf_trees_table_entries(class_pkt_model_file, pkt_feat, feat_dict, range_mark_bits,
                                                     class_pkt_tree_num)

    num_rules += len(tree_data_datas)
    for v in feat_table_datas.values():
        num_rules += len(v)

    # 生成P4
    fea_tbl = ""
    tbl_apply = ""
    tbl_model_key = ""
    meta_code = ""

    for fea in pkt_feat:
        if dataset == "unsw-nb15":
            if int(fea[1:]) >= 8 and int(fea[1:]) <= 39:
                continue

        if fea not in fea_list:
            fea_list.append(fea)
            action_data_bit_list.append(range_mark_bits[fea])
            length_range_list.append(len(feat_table_datas[fea]))

        max_action_data_bit = action_data_bit_list[fea_list.index(fea)]
        max_length_range_list = length_range_list[fea_list.index(fea)]

        if range_mark_bits[fea] > max_action_data_bit:
            action_data_bit_list[fea_list.index(fea)] = range_mark_bits[fea]
        if len(feat_table_datas[fea]) > max_length_range_list:
            length_range_list[fea_list.index(fea)] = len(feat_table_datas[fea])

    for idx,fea in enumerate(fea_list):
        max_forest_action_data_bits = action_data_bit_list[idx]
        max_length_range_list = length_range_list[idx]

        meta_code += "bit<%d> codes_%s;\n\t" % (max_forest_action_data_bits, fea)
        tbl_model_key += "meta.codes_%s : ternary;\n\t\t" % fea

        fea_tbl += "action ac_fea_%s(bit<%d> code){\n" % (fea, max_forest_action_data_bits)
        fea_tbl += "\t\tmeta.codes_%s = code;\n" % fea
        fea_tbl += "\t}\n\n\t"

        fea_tbl += "table tbl_fea_%s{\n" % fea
        fea_tbl += "\t\tkey= {%s : ternary;}\n" % header_names[fea][0]
        fea_tbl += "\t\tactions = {ac_fea_%s;}\n" % fea
        fea_tbl += "\t\tsize=%d;\n" % max_length_range_list
        fea_tbl += "\t}\n\n\t"
        tbl_apply += "tbl_fea_%s.apply();\n\t\t" % fea
    if dataset == "unsw-nb15":
        tbl_model_key += "meta.srcPort : ternary;\n\t\t"
        tbl_model_key += "meta.dstPort : ternary;\n\t\t"

    key = "==model_size=="
    add_to_template(destination_file, key, str(len(tree_data_datas)))
    key = "==fea_tbl=="
    add_to_template(destination_file, key, fea_tbl)
    key = "==apply_tbl=="
    add_to_template(destination_file, key, tbl_apply)
    key = "==codes_ternary=="
    add_to_template(destination_file, key, tbl_model_key)
    key = "==codes=="
    add_to_template(destination_file, key, meta_code)

    # 生成 python
    # setup python file
    if dataset == "unsw-nb15":
        p4_na = "unsw"
    elif dataset == "cicids-2018":
        p4_na = "cicids"
    elif dataset == "ton-iot":
        p4_na = "ton_iot"
    elif dataset == "iscx":
        p4_na = "iscx"
    f = open(setup_file, "w")
    print("p4 = bfrt.netbeacon_"+str(p4_na)+".pipe\n", file=f)
    clear_tables = """

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
clear_all(verbose=False)\n
        """
    print(clear_tables, file=f)
    for fea in pkt_feat:
        if dataset == "unsw-nb15":
            if int(fea[1:]) >= 8 and int(fea[1:]) <= 39:
                continue

        tbl_name = "tbl_fea_%s" % fea
        fea_tbl_data = feat_table_datas[fea]
        fea_name = header_names[fea][0].split(".")[-1].lower()
        print(tbl_name + " = p4.Ingress."+tbl_name, file=f)
        print('', file=f)
        for rule in fea_tbl_data:
            print(tbl_name+".add_with_ac_fea_"+fea+"("+fea_name+"="+"%d, " % rule[1] + fea_name+"_mask="+"%d, " % rule[2] + "match_priority=%d, " % rule[0] + "code=%d" % rule[3] + ")", file=f)
        print('', file=f)

    code_tbl_name = "tb_packet_cls"
    print(code_tbl_name + " = p4.Ingress." + code_tbl_name, file=f)
    print('', file=f)

    for rule in tree_data_datas:
        cla = np.argmax(rule[-1])
        if dataset == "unsw-nb15":
            srcPort = 0
            srcPort_mask = 0
            dstPort = 0
            dstPort_mask = 0
            code_else_str = []


            for i , fea in enumerate(pkt_feat):
                fea_idx = int(fea[1:])
                if 8 <= fea_idx <= 23:  # 源端口特征范围
                    # 拼接 srcPort
                    srcPort = (srcPort << 1) | rule[i * 2]
                    # 拼接 srcPort_mask
                    srcPort_mask = (srcPort_mask << 1) | rule[i * 2 + 1]
                elif 24 <= fea_idx <= 39:  # 目标端口特征范围
                    # 拼接 dstPort
                    dstPort = (dstPort << 1) | rule[i * 2]
                    # 拼接 dstPort_mask
                    dstPort_mask = (dstPort_mask << 1) | rule[i * 2 + 1]
                else:
                    code_else_str.append("codes_%s=%d, codes_%s_mask=%d" % (pkt_feat[i], rule[i * 2], pkt_feat[i], rule[i * 2 + 1]))

            code_else_str.append("srcport=%d, srcport_mask=%d" % (srcPort,srcPort_mask))
            code_else_str.append("dstport=%d, dstport_mask=%d" % (dstPort,dstPort_mask))
            code_str = ", ".join(code_else_str)

        else:    
            code_str = ", ".join(
                ["codes_%s=%d, codes_%s_mask=%d" % (pkt_feat[i], rule[i * 2], pkt_feat[i], rule[i * 2 + 1]) for i in
                range(len(pkt_feat))])

        print(code_tbl_name + ".add_with_ac_packet_forward(" + code_str + ", port=%d" % cla + ")", file=f)
    print('', file=f)
    print("bfrt.complete_operations()", file=f)
    f.close()

    return num_rules





