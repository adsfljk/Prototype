import numpy as np
from sklearn.model_selection import train_test_split

def get_data(args):
    
    if args.dataset == 'ton-iot':
        attack_classes = ['normal','backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'runsomware', 'scanning', 'xss']

        num_classes = len(args.selected_class) + 1 # include normal
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_ton_iot(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 


    elif args.dataset == 'iscx':
        attack_classes =  ['email', 'chat', 'streaming_multimedia', 'file_transfer', 'voip', 'p2p']

        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_ISCX(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size,version=2) 
        
        

    elif args.dataset == 'cicids':
        attack_classes =  ['normal','Brute-Force', 'DoS', 'Web-Attack', 'DDoS', 'Botnet', 'Port-Scan']

        # class include normal
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_CICIDS(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 
    elif args.dataset == 'unibs':
        attack_classes = ['ssl', 'bittorrent', 'http', 'edonkey', 'pop3', 'skype', 'imap', 'smtp']
        
        
        # class include normal
        num_classes = len(args.selected_class)

        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_UNIBS(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 
        


    elif args.dataset == 'unsw':
        attack_classes = ["Withings Smart Baby Monitor","Withings Aura smart sleep sensor","Dropcam",
           "TP-Link Day Night Cloud camera","Samsung SmartCam","Netatmo weather station","Netatmo Welcome",
          "Amazon Echo", "Laptop","NEST Protect smoke alarm","Insteon Camera","Belkin Wemo switch",
           "Belkin wemo motion sensor", "Light Bulbs LiFX Smart Bulb", "Triby Speaker", "Smart Things"]
        
        
        # class include normal
        num_classes = len(args.selected_class)

        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_UNSW(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 
    
    return X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list,num_classes,attack_classes



# return dataset and feature min-max value

def get_data_ton_iot(ton_iot_attack_max_samples, ton_iot_selected_class, test_split_size):
    
    # attack_classes = ['backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'runsomware', 'scanning', 'xss']
    # CUSTOM_FEAT_COLS = ['count', 'ps_mean', 'ps_max', 'ps_min', 'iat_mean', 'iat_max', 'iat_min', 'l4_proto', 'service_port']
    # 上面是9个特征，先都用上吧
    # label of 'normal' is 0，normal的数量大约是5000，ton_iot_attack_max_samples只是限制了每个attack类别的样本数量
    data = np.load('/home/zhuyijia/prototypical/V6_complete/data_ToN-IoT/ToN-IoT_X.npy').astype(np.float64)
    label = np.load('/home/zhuyijia/prototypical/V6_complete/data_ToN-IoT/ToN-IoT_y.npy').astype(np.int64)


    assert data.shape[0] == label.shape[0]
    
    num_cnt = [0 for _ in range(10)]
    select_list = []
    new_label = []
    for idx in range(label.shape[0]):
        if label[idx] == 0:
            select_list.append(idx)
            new_label.append(0)
        elif label[idx] in ton_iot_selected_class:
            num_cnt[label[idx]] += 1
            if num_cnt[label[idx]] <= ton_iot_attack_max_samples:
                select_list.append(idx)
                new_label.append(ton_iot_selected_class.index(label[idx]) + 1)
    
    new_data = data[select_list]
    new_label = np.array(new_label)
    
    # 这里test_size和random_state也是可以加到参数列表里的，目前没有这个需求所以没加，先用这个吧
    X_train, X_test, y_train, y_test = train_test_split(new_data, new_label, test_size=test_split_size, random_state=2023)
    
    # feature_min = [1e9 for _ in range(new_data.shape[1])]
    # feature_max = [-1e9 for _ in range(new_data.shape[1])]
    
    # for dim1 in range(new_data.shape[0]):
    #     for dim2 in range(new_data.shape[1]):
    #         feature_min[dim2] = min(feature_min[dim2], new_data[dim1][dim2])
    #         feature_max[dim2] = max(feature_max[dim2], new_data[dim1][dim2])

    
    
    feature_max = np.max(new_data,axis=0)
    feature_min = np.min(new_data,axis=0)


    if 0 in (feature_max-feature_min):
        idx0 = (feature_max-feature_min)==0
        feature_min[idx0] = 0
        feature_max[idx0] = 1



    # 未知类别的数据也一起返回了，标签用原始的就行（只是方便提示，标签不参与训练与测试）
    unknown_data_list = []
    unknown_label_list = []
    
    for dim1 in range(10):
        if dim1 == 0 or dim1 in ton_iot_selected_class:
            continue
        new_class_index = []
        cnt = 0
        for dim2 in range(label.shape[0]):
            if label[dim2] == dim1:
                new_class_index.append(dim2)
                cnt += 1
                if cnt == ton_iot_attack_max_samples:
                    break
        
        unknown_data_list.append(data[new_class_index])
        unknown_label_list.append(label[new_class_index])
    
    return X_train, X_test, y_train, y_test, np.array(feature_min), np.array(feature_max), unknown_data_list, unknown_label_list

def get_data_ISCX(iscx_attack_max_samples, iscx_selected_class, test_split_size,version = 2):
    if version == 1:
        data = np.load("/home/zhuyijia/prototypical/V6_complete/ISCX_dataset/iscx_data_6class.npy").astype(np.float64)
        label = np.load("/home/zhuyijia/prototypical/V6_complete/ISCX_dataset/iscx_label_6class.npy").astype(np.int64)
    elif version == 2:
        data = np.load("/home/zhuyijia/prototypical/V6_complete/ISCX_dataset/iscx_v2_data_6class.npy").astype(np.float64)
        label = np.load("/home/zhuyijia/prototypical/V6_complete/ISCX_dataset/iscx_v2_label_6class.npy").astype(np.int64)

    selected_data = []
    selected_label = []
    for idx,i in enumerate(iscx_selected_class):
        selected_data.append(data[label==i][:iscx_attack_max_samples])
        # selected_label.append(label[label==i][:iscx_attack_max_samples])
        leng = data[label==i][:iscx_attack_max_samples].shape[0]
        selected_label.append(np.ones(leng)*idx)
    selected_data = np.vstack(selected_data)
    selected_label = np.hstack(selected_label)


    feature_max = np.max(selected_data,axis=0)
    feature_min = np.min(selected_data,axis=0)


    if 0 in (feature_max-feature_min):
        idx0 = (feature_max-feature_min)==0
        feature_min[idx0] = 0
        feature_max[idx0] = 1



    X_train, X_test, y_train, y_test = train_test_split(selected_data, selected_label, test_size=test_split_size, random_state=2023)
    unknown_data_list = []
    unknown_label_list = []
    for dim1 in range(len(np.unique(label))):
        if dim1 in iscx_selected_class:
            continue
        new_class_index = []
        cnt = 0
        for dim2 in range(label.shape[0]):
            if label[dim2] == dim1:
                new_class_index.append(dim2)
                cnt += 1
                if cnt == iscx_attack_max_samples:
                    break
        unknown_data_list.append(data[new_class_index])
        unknown_label_list.append(label[new_class_index])
    
    return X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list


def get_data_CICIDS(iscx_attack_max_samples, iscx_selected_class, test_split_size):

    data = np.load("/home/zhuyijia/prototypical/V6_complete/CICIDS_dataset/CICIDS_2017_X.npy").astype(np.float64)
    label = np.load("/home/zhuyijia/prototypical/V6_complete/CICIDS_dataset/CICIDS_2017_y.npy").astype(np.int64)


    selected_data = []
    selected_label = []
    for idx,i in enumerate(iscx_selected_class):
        selected_data.append(data[label==i][:iscx_attack_max_samples])
        # selected_label.append(label[label==i][:iscx_attack_max_samples])
        leng = data[label==i][:iscx_attack_max_samples].shape[0]
        selected_label.append(np.ones(leng)*idx)
    selected_data = np.vstack(selected_data)
    selected_label = np.hstack(selected_label)


    feature_max = np.max(selected_data,axis=0)
    feature_min = np.min(selected_data,axis=0)


    if 0 in (feature_max-feature_min):
        idx0 = (feature_max-feature_min)==0
        feature_min[idx0] = 0
        feature_max[idx0] = 1




    X_train, X_test, y_train, y_test = train_test_split(selected_data, selected_label, test_size=test_split_size, random_state=2023)
    unknown_data_list = []
    unknown_label_list = []
    for dim1 in range(len(np.unique(label))):
        if dim1 in iscx_selected_class:
            continue
        new_class_index = []
        cnt = 0
        for dim2 in range(label.shape[0]):
            if label[dim2] == dim1:
                new_class_index.append(dim2)
                cnt += 1
                if cnt == iscx_attack_max_samples:
                    break
        unknown_data_list.append(data[new_class_index])
        unknown_label_list.append(label[new_class_index])
    
    return X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list

def get_data_UNIBS(iscx_attack_max_samples, iscx_selected_class, test_split_size):

    data = np.load("/home/zhuyijia/prototypical/V6_complete/Flowrest/UNIBS_X.npy").astype(np.float64)
    label_str = np.load("/home/zhuyijia/prototypical/V6_complete/Flowrest/UNIBS_y.npy",allow_pickle=True)


    label_name = ['ssl', 'bittorrent', 'http', 'edonkey', 'pop3', 'skype', 'imap', 'smtp']


    # 创建映射字典
    label_to_int = {name: index for index, name in enumerate(label_name)}

    # 对标签进行编码
    label = np.array([label_to_int[i] for i in label_str]).astype(np.int64)
    #  [10000  1266   962 10000 10000 10000 10000 10000  2632   914]

    selected_data = []
    selected_label = []
    for idx,i in enumerate(iscx_selected_class):
        selected_data.append(data[label==i][:iscx_attack_max_samples])
        # selected_label.append(label[label==i][:iscx_attack_max_samples])
        leng = data[label==i][:iscx_attack_max_samples].shape[0]
        selected_label.append(np.ones(leng)*idx)
    selected_data = np.vstack(selected_data)
    selected_label = np.hstack(selected_label)


    feature_max = np.max(selected_data,axis=0)
    feature_min = np.min(selected_data,axis=0)


    if 0 in (feature_max-feature_min):
        idx0 = (feature_max-feature_min)==0
        feature_min[idx0] = 0
        feature_max[idx0] = 1




    X_train, X_test, y_train, y_test = train_test_split(selected_data, selected_label, test_size=test_split_size, random_state=2023)
    unknown_data_list = []
    unknown_label_list = []
    for dim1 in range(len(np.unique(label))):
        if dim1 in iscx_selected_class:
            continue
        new_class_index = []
        cnt = 0
        for dim2 in range(label.shape[0]):
            if label[dim2] == dim1:
                new_class_index.append(dim2)
                cnt += 1
                if cnt == iscx_attack_max_samples:
                    break
        unknown_data_list.append(data[new_class_index])
        unknown_label_list.append(label[new_class_index])
    
    return X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list


def get_data_UNSW(iscx_attack_max_samples, iscx_selected_class, test_split_size):

    data = np.load("/home/zhuyijia/prototypical/V6_complete/Flowrest/UNSW_X.npy").astype(np.float64)
    label_str = np.load("/home/zhuyijia/prototypical/V6_complete/Flowrest/UNSW_y.npy",allow_pickle=True)


    label_name = ["Withings Smart Baby Monitor","Withings Aura smart sleep sensor","Dropcam",
           "TP-Link Day Night Cloud camera","Samsung SmartCam","Netatmo weather station","Netatmo Welcome",
          "Amazon Echo", "Laptop","NEST Protect smoke alarm","Insteon Camera","Belkin Wemo switch",
           "Belkin wemo motion sensor", "Light Bulbs LiFX Smart Bulb", "Triby Speaker", "Smart Things"]

    # 创建映射字典
    label_to_int = {name: index for index, name in enumerate(label_name)}

    # 对标签进行编码
    label = np.array([label_to_int[i] for i in label_str]).astype(np.int64)
    #  [10000  1266   962 10000 10000 10000 10000 10000  2632   914]

    selected_data = []
    selected_label = []
    for idx,i in enumerate(iscx_selected_class):
        selected_data.append(data[label==i][:iscx_attack_max_samples])
        # selected_label.append(label[label==i][:iscx_attack_max_samples])
        leng = data[label==i][:iscx_attack_max_samples].shape[0]
        selected_label.append(np.ones(leng)*idx)
    selected_data = np.vstack(selected_data)
    selected_label = np.hstack(selected_label)


    feature_max = np.max(selected_data,axis=0)
    feature_min = np.min(selected_data,axis=0)


    if 0 in (feature_max-feature_min):
        idx0 = (feature_max-feature_min)==0
        feature_min[idx0] = 0
        feature_max[idx0] = 1




    X_train, X_test, y_train, y_test = train_test_split(selected_data, selected_label, test_size=test_split_size, random_state=2023)
    unknown_data_list = []
    unknown_label_list = []
    for dim1 in range(len(np.unique(label))):
        if dim1 in iscx_selected_class:
            continue
        new_class_index = []
        cnt = 0
        for dim2 in range(label.shape[0]):
            if label[dim2] == dim1:
                new_class_index.append(dim2)
                cnt += 1
                if cnt == iscx_attack_max_samples:
                    break
        unknown_data_list.append(data[new_class_index])
        unknown_label_list.append(label[new_class_index])
    
    return X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list