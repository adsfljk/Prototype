import numpy as np
from sklearn.model_selection import train_test_split

def get_data(args):
    
    if args.dataset == 'ton-iot':
        attack_classes = ['normal','backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'runsomware', 'scanning', 'xss']

        num_classes = len(args.selected_class) 
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
        
        

    elif args.dataset == 'cicids-2017':
        attack_classes =  ['normal','Brute-Force', 'DoS', 'Web-Attack', 'DDoS', 'Botnet', 'Port-Scan']

        # class include normal
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_CICIDS_2017(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size)
        
    elif args.dataset == 'cicids-2018':
        attack_classes = ['BENIGN','DDoS-LOIC-HTTP' ,'DDoS-HOIC','DDoS-LOIC-UDP', \
               'DoS GoldenEye', 'DoS Hulk','DoS Slowloris' ,\
                'SSH-BruteForce','Web Attack - XSS','Web Attack - SQL','Web Attack - Brute Force'] 
        # [   9776  289328 1082293    2527   22560 1803160    8490   94197     113  39     131]
        # class include normal
        num_classes = len(args.selected_class)
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_CICIDS_2018(args.attack_max_samples, 
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
        


    elif args.dataset == 'unsw-iot':
        attack_classes = ["Withings Smart Baby Monitor","Withings Aura smart sleep sensor","Dropcam",
           "TP-Link Day Night Cloud camera","Samsung SmartCam","Netatmo weather station","Netatmo Welcome",
          "Amazon Echo", "Laptop","NEST Protect smoke alarm","Insteon Camera","Belkin Wemo switch",
           "Belkin wemo motion sensor", "Light Bulbs LiFX Smart Bulb", "Triby Speaker", "Smart Things"]
        
        
        # class include normal
        num_classes = len(args.selected_class)

        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_UNSW_IOT(args.attack_max_samples, 
                        args.selected_class, 
                        args.test_split_size) 
        

    elif args.dataset == 'unsw-nb15':
        attack_classes = ['normal',  'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms','Analysis', 'Backdoor',]
        
        # class include normal
        num_classes = len(args.selected_class)

        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
        get_data_UNSW_NB15(args.attack_max_samples, 
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


    selected_data = []
    selected_label = []
    for idx,i in enumerate(ton_iot_selected_class):
        selected_data.append(data[label==i][:ton_iot_attack_max_samples])
        # selected_label.append(label[label==i][:iscx_attack_max_samples])
        leng = data[label==i][:ton_iot_attack_max_samples].shape[0]
        selected_label.append(np.ones(leng)*idx)
    selected_data = np.vstack(selected_data)
    selected_label = np.hstack(selected_label)


    feature_max = np.max(selected_data,axis=0)
    feature_min = np.min(selected_data,axis=0)


    if 0 in (feature_max-feature_min):
        idx0 = (feature_max-feature_min)==0
        feature_min[idx0] = 0
        feature_max[idx0] = 1




    X_train, X_test, y_train, y_test = train_test_split(selected_data, selected_label, test_size=test_split_size, random_state=2024)
    
    unknown_data_list = []
    unknown_label_list = []
    for dim1 in range(len(np.unique(label))):
        if dim1 in ton_iot_selected_class:
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


def get_data_CICIDS_2017(iscx_attack_max_samples, iscx_selected_class, test_split_size):

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



def get_data_CICIDS_2018(iscx_attack_max_samples, iscx_selected_class, test_split_size):

    data = np.load("/home/zhuyijia/prototypical/V6_complete/CICIDS_dataset/CICIDS_2018_X.npy").astype(np.float64)
    label = np.load("/home/zhuyijia/prototypical/V6_complete/CICIDS_dataset/CICIDS_2018_y.npy").astype(np.int64)


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


def get_data_UNSW_IOT(iscx_attack_max_samples, iscx_selected_class, test_split_size):

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



def get_data_UNSW_NB15(iscx_attack_max_samples, iscx_selected_class, test_split_size):

    data = np.load("/home/zhuyijia/prototypical/V6_complete/UNSW_dataset/UNSW_X_1.npy").astype(np.float32)
    label_str = np.load("/home/zhuyijia/prototypical/V6_complete/UNSW_dataset/UNSW_y_1.npy")

    label_name = ['normal',  'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms','Analysis', 'Backdoor',]


    # 创建映射字典
    label_to_int = {name: index for index, name in enumerate(label_name)}

    # 对标签进行编码
    label = np.array([label_to_int[i] for i in label_str]).astype(np.int8)
    selected_data = []
    selected_label = []
    for i in iscx_selected_class:
        selected_data.append(data[label==i][:iscx_attack_max_samples])
        selected_label.append(label[label==i][:iscx_attack_max_samples])
    selected_data = np.vstack(selected_data)
    selected_label = np.hstack(selected_label)


    feature_max = np.max(selected_data,axis=0)
    feature_min = np.min(selected_data,axis=0)


    if 0 in (feature_max-feature_min):
        idx0 = (feature_max-feature_min)==0
        feature_min[idx0] = -1e6




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
