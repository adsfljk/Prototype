# !/usr/bin/env python
# -*-coding:utf-8 -*-

import pandas as pd
import numpy as np
from shutil import copyfile
import fileinput
def sort_a_b(a,b):
    combined = list(zip(a, b))

    # 按 a 列表的值进行排序
    sorted_combined = sorted(combined)

    # 解包成两个排序后的列表
    a_sorted, b_sorted = zip(*sorted_combined)
    return a_sorted, b_sorted 


def add_to_template(outname, placeholder, code):
    with fileinput.FileInput(outname, inplace=True) as file:
        for line in file:
            print(line.replace(placeholder, code), end='')


def split_codes(dataset,code, used_features, add="", tree_id=0):
    code = code[2:]  # 删除"0b"
    idx = 0
    content = ""

    if dataset == "unsw-nb15":
        srcport_bits = ""
        dstport_bits = ""
        for fea in used_features:
            fea_n = fea[0].lower()

            fea_idx = int(fea_n[1:])  # 提取特征的编号

            if 8 <= fea_idx <= 23:  # 源端口特征范围
                srcport_bits += code[idx:idx + fea[1]]
            elif 24 <= fea_idx <= 39:  # 目标端口特征范围
                dstport_bits += code[idx:idx + fea[1]]
            else:
                content += "codes_%d_" % tree_id + fea_n + add + "=" + "0b" + code[idx:idx + fea[1]] + ","
            
            idx += fea[1]
        
        # 合并源端口和目标端口
        if srcport_bits:
            content += "srcport" + add + "=" + "0b" + srcport_bits + ","
        if dstport_bits:
            content += "dstport" + add + "=" + "0b" + dstport_bits + ","
        

    else:
        for fea in used_features:
            fea_n = fea[0].lower()
            content += "codes_%d_" % tree_id + fea_n + add + "=" + "0b" + code[idx:idx + fea[1]] + ","
            idx = idx + fea[1]
    return content


def comb_tree_preds(comb, forest_domain):
    new_comb = []
    if len(forest_domain) == 1:
        current_tree = forest_domain.pop()
        if len(comb) == 0:
            for f in current_tree:
                new_comb.append([f])
        else:
            for f in current_tree:
                for c in comb:
                    new_comb.append([*c, f])
    else:
        current_tree = forest_domain.pop()
        if len(comb) == 0:
            for f in current_tree:
                new_comb.append([f])
        else:
            for f in current_tree:
                for c in comb:
                    new_comb.append([*c, f])
        new_comb = comb_tree_preds(new_comb, forest_domain)
    return new_comb


## get list of splits crossed to get to leaves
def retrieve_branches(estimator):
    number_nodes = estimator.tree_.node_count
    children_left_list = estimator.tree_.children_left
    children_right_list = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    # Calculate if a node is a leaf
    is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left_list, children_right_list)]
    # Store the branches paths
    paths = []
    for i in range(number_nodes):
        if is_leaves_list[i]:
            # Search leaf node in previous paths
            end_node = [path[-1] for path in paths]
            # If it is a leave node yield the path
            if i in end_node:
                output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                yield output
        else:
            # Origin and end nodes
            origin, end_l, end_r = i, children_left_list[i], children_right_list[i]
            # Iterate over previous paths to add nodes
            for index, path in enumerate(paths):
                if origin == path[-1]:
                    paths[index] = path + [end_l]
                    paths.append(path + [end_r])
            # Initialize path in first iteration
            if i == 0:
                paths.append([i, children_left_list[i]])
                paths.append([i, children_right_list[i]])


## get classes and certainties
def get_classes(clf):
    leaves = []
    classes = []
    certainties = []
    for branch in list(retrieve_branches(clf)):
        leaves.append(branch[-1])
    for leaf in leaves:
        if clf.tree_.n_outputs == 1:
            value = clf.tree_.value[leaf][0]
        else:
            value = clf.tree_.value[leaf].T[0]
        class_name = np.argmax(value)
        certainty = int(round(max(value) / sum(value), 2) * 100)
        classes.append(class_name)
        certainties.append(certainty)
    return classes, certainties


## get the codes corresponging to the branches followed
def get_leaf_paths(clf):
    depth = clf.max_depth
    branch_codes = []
    for branch in list(retrieve_branches(clf)):
        code = [0] * len(branch)
        for i in range(1, len(branch)):
            if (branch[i] == clf.tree_.children_left[branch[i - 1]]):
                code[i] = 0
            elif (branch[i] == clf.tree_.children_right[branch[i - 1]]):
                code[i] = 1
        branch_codes.append(list(code[1:]))
    return branch_codes


def get_splits_per_tree(clf, feature_names):
    data = []
    n_nodes = clf.tree_.node_count
    # set feature names
    features = [feature_names[i] for i in clf.tree_.feature]
    # generate dataframe with all thresholds and features
    for i in range(0, n_nodes):
        node_id = i
        left_child_id = clf.tree_.children_left[i]
        right_child_id = clf.tree_.children_right[i]
        threshold = clf.tree_.threshold[i]
        feature = features[i]
        if threshold != -2.0:
            data.append([node_id, left_child_id,
                         right_child_id, threshold, feature])
    data = pd.DataFrame(data)
    data.columns = ["NodeID", "LeftID", "RightID", "Threshold", "Feature"]
    return data


def get_splits(forest, feature_names):
    data = []
    for t in range(len(forest.estimators_)):
        clf = forest[t]
        n_nodes = clf.tree_.node_count
        features = [feature_names[i] for i in clf.tree_.feature]  # features used in nodes
        for i in range(0, n_nodes):
            node_id = i
            left_child_id = clf.tree_.children_left[i]
            right_child_id = clf.tree_.children_right[i]
            threshold = clf.tree_.threshold[i]
            feature = features[i]
            if threshold != -2.0:
                data.append([t, node_id, left_child_id,
                             right_child_id, threshold, feature])
    data = pd.DataFrame(data)
    data.columns = ["Tree", "NodeID", "LeftID", "RightID", "Threshold", "Feature"]
    return data


## gets the feature table of each feature from the splits
def get_feature_table(splits_data, feature_name):
    feature_data = splits_data[splits_data["Feature"] == feature_name]
    feature_data = feature_data.sort_values(by="Threshold")
    feature_data = feature_data.reset_index(drop=True)
    
    # 9.14 zyj 由于重复的分支导致bit位数超过524b 不能部署问题
    # 这里不需要所有相同分支，相同阈值的第一个分支即可    
    feature_data = feature_data.drop_duplicates(subset=['Tree', 'Threshold'], keep='first').reset_index(drop=True)
    # 按 Threshold 分组，并将 NodeID 合并为列表
    # feature_data = feature_data.groupby('Threshold').agg({
    #     'NodeID': lambda x: list(x),  # 将相同 Threshold 的 NodeID 合并为列表
    #     'Tree': 'first',
    #     'LeftID': 'first',
    #     'RightID': 'first',
    #     'Feature': 'first'
    # }).reset_index()

    ##
    # feature_data["Threshold"] = (feature_data["Threshold"]).astype(int)
    feature_data["Threshold"] = feature_data["Threshold"].astype(int)
    ##

    code_table = pd.DataFrame()
    code_table["Threshold"] = feature_data["Threshold"]

    # create a column for each split in each tree
    for tree_id, node in zip(list(feature_data["Tree"]), list(feature_data["NodeID"])):
        colname = "s" + str(tree_id) + "_" + str(node)

        code_table[colname] = np.where((code_table["Threshold"] <=
                                        feature_data[(feature_data["NodeID"] == node) &
                                                     (feature_data["Tree"] == tree_id)]["Threshold"].values[0]), 0, 1)

    # add a row to represent the values above the largest threshold
    temp = [max(code_table["Threshold"]) + 1]
    temp.extend(list([1] * (len(code_table.columns) - 1)))

    code_table.loc[len(code_table)] = temp
    code_table = code_table.drop_duplicates(subset=['Threshold'])
    code_table = code_table.reset_index(drop=True)

    return code_table

## get feature tables with ranges and codes only
def get_feature_codes_with_ranges(feature_table, num_of_trees):
    Codes = pd.DataFrame()
    for tree_id in range(num_of_trees):
        colname = "code" + str(tree_id)
        Codes[colname] = feature_table[
            feature_table[[col for col in feature_table.columns if ('s' + str(tree_id) + '_') in col]].columns[
            0:]].apply(lambda x: ''.join(x.dropna().astype(str)), axis=1)
        Codes[colname] = ["0b" + x for x in Codes[colname]]
    feature_table["Range"] = [0] * len(feature_table)
    feature_table["Range"].loc[0] = "0," + str(feature_table["Threshold"].loc[0])
    for i in range(1, len(feature_table)):
        if (i == (len(feature_table)) - 1):
            feature_table["Range"].loc[i] = str(feature_table["Threshold"].loc[i]) + "," + str(
                feature_table["Threshold"].loc[i])
        else:
            feature_table["Range"].loc[i] = str(feature_table["Threshold"].loc[i - 1] + 1) + "," + str(
                feature_table["Threshold"].loc[i])
    Ranges = feature_table["Range"]
    return Ranges, Codes



## get the order of the splits to enable code generation
def get_order_of_splits(data, feature_names):
    splits_order = []
    for feature_name in feature_names:
        feature_data = data[data.iloc[:, 4] == feature_name]
        feature_data = feature_data.sort_values(by="Threshold")

        # 9.14 zyj 由于重复的分支导致bit位数超过524b 不能部署问题
        # 按 Threshold 分组，并将 NodeID 合并为列表
        feature_data = feature_data.groupby('Threshold').agg({
            'NodeID': lambda x: list(x),  # 将相同 Threshold 的 NodeID 合并为列表
            'LeftID': 'first',
            'RightID': 'first',
            'Feature': 'first'
        }).reset_index()

        # 增加列表含有所有NodeID
        for node in list(feature_data['NodeID']):
            splits_order.append(node)
    return splits_order


def get_codes_and_masks(clf, feature_names):
    # 9.14 变为二维数组 ,相同阈值合并为一个数组
    splits = get_order_of_splits(get_splits_per_tree(clf, feature_names), feature_names)

    depth = clf.max_depth
    codes = []
    masks = []
    for branch, coded in zip(list(retrieve_branches(clf)), get_leaf_paths(clf)):
        code = [0] * len(splits)
        mask = [0] * len(splits)
        for index, split in enumerate(splits):
            for split_one in split:
                if split_one in branch:
                    mask[index] = 1
        masks.append(mask)
        codes.append(code)
    masks = pd.DataFrame(masks)
    masks['Mask'] = masks[masks.columns[0:]].apply(lambda x: ''.join(x.dropna().astype(str)), axis=1)
    masks = ["0b" + x for x in masks['Mask']]

    # indices = range(0, len(splits))
    temp = pd.DataFrame(columns=["split", "index"], dtype=object)
    # temp["split"] = splits
    # temp["index"] = indices

    final_codes = []
    for branch, code, coded in zip(list(retrieve_branches(clf)), codes, get_leaf_paths(clf)):
        indices_to_use = []
        temp_split = []
        for idx, split in enumerate(splits):
            for split_one in split:
                if split_one in branch:
                    temp_split.append(split_one)
                    indices_to_use.append(idx)
        _,indices_to_use = sort_a_b(temp_split,indices_to_use)

        # indices_to_use = temp[temp["split"].isin(branch)].sort_values(by="split")["index"]
        for i, j in zip(range(0, len(coded)), list(indices_to_use)):
            code[j] = coded[i]
        final_codes.append(code)
    final_codes = pd.DataFrame(final_codes)
    final_codes["Code"] = final_codes[final_codes.columns[0:]].apply(lambda x: ''.join(x.dropna().astype(str)), axis=1)
    final_codes = ["0b" + x for x in final_codes["Code"]]

    return final_codes, masks

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

def export_planter(dataset,clf, feature_max, class_pkt_tree_num,log_class_name,fea_list,action_data_bit_list,length_range_list,class_num_bits=1):
    """
    class_num_bits: 用于存储每个树预测类别的phv的bits.
    """
    source_file = "./utils/planter_tmpl.p4"
    # destination_file = "rf.p4"
    # setup_file = "rf_setup.py"
    destination_file = "./entity/planter/"+str(dataset)+'/'+log_class_name+".p4"
    setup_file = "./entity/planter/"+str(dataset)+'/'+log_class_name+"_setup.py"
    copyfile(source_file, destination_file)
    n_tree = clf.n_estimators

    num_rules = 0
    feature_names = ["f%d" % i for i in range(len(feature_max))]

    a =   binary_bit_count_list(feature_max)
    print(a)
    ############################ 自定义内容 #############################
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
    used_features = []

    forest_domain = [[] for _ in range(n_tree)]
    model_dict = {}
    for tree_id in range(n_tree):
        Final_Codes, Final_Masks = get_codes_and_masks(clf.estimators_[tree_id], feature_names)
        Classe, Certain = get_classes(clf.estimators_[tree_id])
        forest_domain[tree_id] = list(set(Classe))
        model_dict["code%d" % tree_id] = [Final_Codes, Final_Masks, Classe]

    fea_tbl = ""
    tbl_apply = ""
    tbl_model_key = dict(("code%d" % k, "") for k in range(n_tree))
    meta_code = ""
    tree_tbl = ""
    vote_table_key = ""

    data = get_splits(clf, feature_names)
    used_features_ = data["Feature"].unique()
    used_features = dict(("code%d" % k, []) for k in range(n_tree))
    # setup python file
    if dataset == "unsw-nb15":
        p4_na = "unsw"
    elif dataset == "cicids-2018":
        p4_na = "cicids"
    elif dataset == "ton-iot":
        p4_na = "ton_iot"
    elif dataset == "iscx":
        p4_na = "iscx"
    # setup python file
    f = open(setup_file, "w")
    print("p4 = bfrt.planter_"+str(p4_na)+".pipe\n", file=f)
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

    for fea in feature_names:
        if fea not in used_features_:
            continue
        fea_space_split = get_feature_table(data, fea)
        # Get feature table
        Ranges, Codes = get_feature_codes_with_ranges(fea_space_split, n_tree)
        forest_action_data_bits = []
        for k in Codes.columns:
            Code = Codes[k]
            forest_action_data_bits.append(len(Code[0]) - 2)
            if len(Code[0]) - 2 != 0:
                used_features[k].append([fea, len(Code[0]) - 2])
        num_rules += len(Ranges)


        if dataset == "unsw-nb15":
            if int(fea[1:]) >= 8 and int(fea[1:]) <= 39:
                continue
        # python
        tbl_name = "tbl_fea_%s" % fea
        print(tbl_name + " = p4.Ingress." + tbl_name, file=f)
        print('', file=f)

        ub = 2 ** header_names[fea][1] - 1
        fea_name = header_names[fea][0].split(".")[-1].lower()
        for ids in range(len(Codes)):
            ran = Ranges.iloc[ids]
            code = Codes.iloc[ids, :]
            code_str = ",".join(
                ["code%d=%s" % (i, code["code%d" % i]) for i in range(n_tree) if forest_action_data_bits[i] != 0])
            if (ran == Ranges[len(Ranges) - 1]):
                print(tbl_name + ".add_with_ac_fea_" + fea + "(" + fea_name + "_start=" + str(ran.split(",")[0]) + \
                      ", " + fea_name + "_end=" + str(ub) + "," + code_str + ")", file=f)
            else:
                print(tbl_name + ".add_with_ac_fea_" + fea + "(" + fea_name + "_start=" + str(ran.split(",")[0]) + \
                      ", " + fea_name + "_end=" + str(ran.split(",")[1]) + "," + code_str + ")", file=f)
        print('', file=f)

        # P4   fea   forest_action_data_bits
        if fea not in fea_list:
            fea_list.append(fea)
            action_data_bit_list.append(forest_action_data_bits)
            length_range_list.append(len(Ranges))

        max_action_data_bit = action_data_bit_list[fea_list.index(fea)]
        max_length_range_list = length_range_list[fea_list.index(fea)]

        for i in range(n_tree):
            if forest_action_data_bits[i] > max_action_data_bit[i]:
                action_data_bit_list[fea_list.index(fea)][i] = forest_action_data_bits[i]

        if len(Ranges) > max_length_range_list:
            length_range_list[fea_list.index(fea)] = len(Ranges)

    for idx,fea in enumerate(fea_list):
        max_forest_action_data_bits = action_data_bit_list[idx]
        max_length_range_list = length_range_list[idx]
        # action 
        fea_tbl += "action ac_fea_%s(" % fea + \
                   ", ".join(["bit<%d> code%d" % (max_forest_action_data_bits[i], i) for i in range(n_tree) if
                              max_forest_action_data_bits[i] != 0]) + \
                   "){\n"
        for i in range(n_tree):
            if max_forest_action_data_bits[i] == 0:
                continue
            fea_tbl += "\t\tmeta.codes_%d_%s = code%d;\n" % (i, fea, i)
        fea_tbl += "\t}\n\n\t"

        # tbl
        fea_tbl += "table tbl_fea_%s{\n" % fea
        fea_tbl += "\t\tkey= {%s : range;}\n" % header_names[fea][0]
        fea_tbl += "\t\tactions = {ac_fea_%s;}\n" % fea
        fea_tbl += "\t\tsize=%d;\n" % max_length_range_list
        fea_tbl += "\t}\n\n\t"
        tbl_apply += "tbl_fea_%s.apply();\n\t\t" % fea

        # phv
        for i in range(n_tree):
            if max_forest_action_data_bits[i] == 0:
                continue
            meta_code += "bit<%d> codes_%d_%s;\n\t" % (max_forest_action_data_bits[i], i, fea)
            # model_tbl
            tbl_model_key["code%d" % i] += "meta.codes_%d_%s : ternary;\n\t\t" % (i, fea)
        

    for i in range(n_tree):
        meta_code += "bit<%d> pred_%d;\n\t" % (class_num_bits, i)
        vote_table_key += "meta.pred_%d : exact;\n\t\t" % i
        tree_tbl += "action ac_tree_%d(bit<%d> cls) {\n" % (i, class_num_bits)
        tree_tbl += "\t\tmeta.pred_%d = cls;\n" % i
        tree_tbl += "\t}\n\n\t"

        tree_tbl += "table tbl_tree_%d{\n" % i
        tree_tbl += "\t\tkey={\n\t"
        tree_tbl += tbl_model_key["code%d" % i]
        if dataset == "unsw-nb15":
            tree_tbl += "meta.srcPort : ternary;\n\t\t"
            tree_tbl += "meta.dstPort : ternary;\n\t\t"

        tree_tbl += "}\n"
        tree_tbl += "\t\tactions = {ac_tree_%d;}\n" % i
        tree_tbl += "\t\tsize=%d;\n" % len(model_dict["code%d" % i][0])
        tree_tbl += "\t}\n\n\t"
        tbl_apply += "tbl_tree_%d.apply();\n\t\t" % i

        tbl_name = "tbl_tree_%d" % i
        print(tbl_name + " = p4.Ingress." + tbl_name, file=f)
        print('', file=f)

        [Final_Codes, Final_Masks, Classe] = model_dict["code%d" % i]
        num_rules += len(Classe)
        for cod, mas, cla in zip(Final_Codes, Final_Masks, Classe):
            print(tbl_name + ".add_with_ac_tree_%d(" % i + split_codes(dataset,cod, used_features["code%d" % i], tree_id=i) +
                  split_codes(dataset,mas, used_features["code%d" % i], add="_mask", tree_id=i) + "cls=", cla, ")", file=f)
        print('', file=f)

    code_tbl_name = "tb_packet_cls"
    print(code_tbl_name + " = p4.Ingress." + code_tbl_name, file=f)
    print('', file=f)

    vote_table_size = 0
    for comb in comb_tree_preds([], forest_domain):
        voted_class = max(set(comb), key=comb.count)
        vote_table_size += 1
        print(code_tbl_name + ".add_with_ac_packet_forward(" + \
              ",".join(["pred_%d=%d" % (i, comb[i]) for i in range(n_tree)]) + "," + \
              "port=", voted_class, ")", file=f)
    num_rules += vote_table_size
    print('', file=f)
    print("bfrt.complete_operations()", file=f)
    f.close()





    key = "==codes=="
    add_to_template(destination_file, key, meta_code)
    key = "==fea_tbl=="
    add_to_template(destination_file, key, fea_tbl)
    key = "==apply_tbl=="
    add_to_template(destination_file, key, tbl_apply)
    key = "==tree_tbl=="
    add_to_template(destination_file, key, tree_tbl)
    key = "==model_size=="
    add_to_template(destination_file, key, str(vote_table_size))
    key = "==codes_ternary=="
    add_to_template(destination_file, key, vote_table_key)

    return num_rules



