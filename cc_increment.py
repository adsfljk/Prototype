import numpy as np
import torch
import torch.nn as nn
from utils.proto import prototype, init_model
from utils.general import *
from utils.prune import model_prune
from utils.get_dataset import *
import argparse
import wandb
import os
import csv
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc,confusion_matrix,accuracy_score
from pot_utils.pot import pot,pot_min
from sklearn.model_selection import train_test_split
from collections import Counter


def main(args,add_order,output_csv,export_entry = False):

    torch.cuda.set_device(args.gpu)


    first_flag = 1
    for idx_log_class,log_class in enumerate(add_order):

        X_train, X_test, y_train, y_test, feature_min, feature_max,\
        unknown_data_list, unknown_label_list,num_classes,attack_classes = get_data(args)

        epochs = args.epochs
        if log_class==-1:
            print('----------ALL----------')
        else:
            print('----------',attack_classes[log_class],'----------')
            
        a,b = np.unique(y_train,return_counts=True)
        print(a,b)
        # exit()
        
        feature_num = X_train.shape[1]
        assert feature_num == len(feature_min)
        assert feature_num == len(feature_max)

        set_global_seed(args.seed)

        if first_flag == 1:
            # boosting之前的第一次初始化
            this_X_train = X_train.copy()
            this_y_train = y_train.copy()
            class_rules = []
            if args.incremental_train == True:
                first_flag = 0

        else:
            class_rules = old_class_rules

        for boost_iter in range(args.boost_num):
            # step0: init model
            print(boost_iter,'step0: init model')

            a,b = np.unique(this_y_train,return_counts=True)
            print(a,b)
            model = prototype(num_classes, feature_num, [args.max_proto_num for _ in range(num_classes)], args.temperature, args.cal_dis, args.cal_mode).cuda()
            if args.init_methed == "boost_train":
                model = init_model(model, args.init_type,args.dbscan_eps,args.min_samples, this_X_train, this_y_train, feature_min, feature_max)
            else:
                model = init_model(model, args.init_type,args.dbscan_eps,args.min_samples, X_train, y_train, feature_min, feature_max)



            for idx in range(num_classes):
                assert model.class_prototype[idx].shape[1] == model.class_transform[idx].shape[1]
            # 先不考虑CE loss的label smoothing
            loss_func = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.class_prototype + model.class_transform, lr=args.learning_rate)

            train_acc = test_model_acc(model, this_X_train, this_y_train, feature_min, feature_max)
            print('before training, train acc = ', train_acc)



            print(boost_iter,'step1: train model')
            best_train_acc = 0
            for epoch in range(epochs):
                
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
                    # print(logits)
                    _, predictions = torch.max(logits, dim=1)
                    loss = loss_func(logits, train_label)
                    loss.backward()
                    optimizer.step()
                    
                    train_acc_num += torch.sum(predictions == train_label).item()
                    idx += args.batch_size
                
                train_acc = train_acc_num / this_X_train.shape[0] * 100
                # test_acc = test_model_acc(model, X_test, y_test, feature_min, feature_max)
                
                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                    model.save_parameter(args.save_model+'_model_best.pth')

            print('epoch = %d/%d, train_acc = %.3f%%, best_train_acc = %.3f%%' % (epoch + 1, args.epochs, train_acc, best_train_acc))



            # step2: model prune
            # 注意要在训练集上剪枝，不要搞成测试集，并且要和训练时的数据保持一致（不要新增新类别）
            print(boost_iter,'step2: model prune')

            model.load_parameter(args.save_model + '_model_best.pth')
            # test_acc = test_model_acc(model, X_test, y_test, feature_min, feature_max)
            # print('distance-based test acc before prune: %.3f%%' % (test_acc))

            
            model = model_prune(model, num_classes, this_X_train, this_y_train, feature_min, feature_max, args.max_proto_num, args.prune_T)

            # if boost_iter==0:
            #     prune_acc = test_model_acc(model, X_test, y_test, feature_min, feature_max)
            #     print('prune_T = %d, distance-based test acc after prune: %.3f%%' % (args.prune_T, prune_acc))
            model.save_parameter(args.save_model + '_model_prune.pth')





            # step3: 在训练集上计算分类阈值，并根据距离计算的方式，得到新类别识别率
            print(boost_iter,'step3: distance and threshold calculate')
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
                        train_batch = train_batch * torch.abs(model.class_transform[dim1])
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
            # 剪枝后没有满足剪枝条件的原型，则停止boost
            if tot == 0:
                print("no proto, stop boosting")
                continue
            prune_train_acc = tot / this_X_train.shape[0] * 100
            print('distance-based train acc after prune = %.3f%%' % (prune_train_acc))


            for dim1 in range(num_classes):
                for dim2 in range(model.class_prototype[dim1].shape[1]):
                    dist_list[dim1][dim2].sort(reverse=True, key=lambda x: x[0])
                    

            # print('distance list of each prototype(sorted and print top 10)')
            # for dim1 in range(num_classes):
            #     for dim2 in range(model.class_prototype[dim1].shape[1]):
            #         tmp = [t[0] for t in dist_list[dim1][dim2][:10]]
            #         print(tmp)

            
            # 从输出中我们可以看出，cls_threshold设置为0.01~0.02之间比较好，不会损失太多正确分类的样本
            # 我们的第一优先级是保证正确分类，其次才是未知类别的识别，所以这里阈值不能设置的太小，要不然太影响分类精度了
            # EVT POT
            # Parameters

            epsilon = 1e-8

            cost_cnt = 0
            for dim1 in range(num_classes):
                for dim2 in range(model.class_prototype[dim1].shape[1]):
                    if args.thr_method == "pot":
                        EVT_threshold, t = pot(np.array([x[0] for x in dist_list[dim1][dim2]]), args.risk, args.cls_threshold, args.num_candidates, epsilon)
                    elif args.thr_method == "three_sigma":
                        EVT_threshold = three_sigma(np.array([x[0] for x in dist_list[dim1][dim2]]), args.cls_threshold)
                    elif args.thr_method== "mean":
                        EVT_threshold = np.mean(np.array([x[0] for x in dist_list[dim1][dim2]])) * args.cls_threshold
                    elif args.thr_method== "median":
                        EVT_threshold = np.median(np.array([x[0] for x in dist_list[dim1][dim2]])) * args.cls_threshold
                    elif args.thr_method== "percentile":
                        EVT_threshold = np.percentile(np.array([x[0] for x in dist_list[dim1][dim2]]),args.cls_threshold)

                    # print("list: ",np.array([x[0] for x in dist_list[dim1][dim2][:10]]),"thr: ",EVT_threshold)

                    # EVT_threshold = dist_list[dim1][dim2][10][0]
                    while dist_list[dim1][dim2][0][0] > EVT_threshold and len(dist_list[dim1][dim2]) > 1:
                        dist_list[dim1][dim2].pop(0)
                        cost_cnt += 1

            print('cost due to distance threshold = %d/%d, rate = %.3f%%' % (cost_cnt, tot, cost_cnt / tot * 100))

            # step4: 转化成匹配规则，得到最终结果
            print(boost_iter,'step4:change prototype to match rules and get final results')

            temp_class_rules = []
            
            for dim1 in range(num_classes):
                temp_1 = []
                for dim2 in range(model.class_prototype[dim1].shape[1]):
                    temp_2 = [[1e10, -1e10] for dim3 in range(feature_num)]
                    
                    for dim4 in range(len(dist_list[dim1][dim2])):
                        sample = dist_list[dim1][dim2][dim4][1]
                        for dim5 in range(feature_num):
                            temp_2[dim5][0] = min(temp_2[dim5][0], sample[dim5])
                            temp_2[dim5][1] = max(temp_2[dim5][1], sample[dim5])


                    temp_1.append(temp_2)
                temp_class_rules.append(temp_1)

            # step5: 新旧模型boosting
            if boost_iter == 0:                
                class_rules = merge_rules(class_rules,temp_class_rules)
            else:
                for dim1 in range(num_classes):
                    for dim2 in range(len(temp_class_rules[dim1])):
                        class_rules[dim1].append(temp_class_rules[dim1][dim2])

            # # 将字典保存为txt文件
            # with open("./output"+str(boost_iter)+".txt", "w") as file:
            #     for dim1 in range(num_classes):
            #         for dim2 in range(len(class_rules[dim1])):
            #             for dim3 in range(feature_num):
            #                 a = class_rules[dim1][dim2][dim3].copy()

            #                 tmp = f"[{a[0]},{a[1]}]"

            #                 file.write(f"{dim1},{dim2},{dim3}: {tmp}\n")

            # # 这里可以灵活点，训练集和测试集都试试，看结果有区别没
            # if args.solve_conflict_on == 'train_set':
            #     cls_fix = solve_conflict(X_train, y_train,num_classes,class_rules)
            # elif args.solve_conflict_on == 'test_set':
            #     cls_fix = solve_conflict(X_test, y_test,num_classes,class_rules)
            # print('solve match conflict:')
            # print('conflict number = ', len(cls_fix))
            # print(cls_fix)

            # no_match = 0
            # tot = 0
            # tot_overlap = 0
            # overlap_success = 0

            # for idx in range(X_test.shape[0]):
                
            #     sample = X_test[idx]
            #     cls_res = []
                
            #     for dim1 in range(num_classes):
            #         for dim2 in range(len(class_rules[dim1])):
            #             if check_rule(sample, class_rules[dim1][dim2]):
                            
            #                 if (dim1 * 10000 + dim2) not in cls_res:
            #                     cls_res.append(dim1 * 10000 + dim2)

            #     # fix important bug: cls_res[0], not cls_res
            #     if len(cls_res) == 0:
            #         no_match += 1
            #     if len(cls_res) == 1 and cls_res[0] // 10000 == y_test[idx]:
            #         tot += 1
            #     elif len(cls_res) >= 2:
            #         tot_overlap += 1
            #         cls_res.sort(reverse=False)
            #         cls_res_hash = '#'
            #         for p in cls_res:
            #             cls_res_hash += str(p)
            #             cls_res_hash += '#'
            #         if cls_fix[cls_res_hash] == y_test[idx]:
            #             tot += 1
            #             overlap_success += 1
                
            # print('global rule-based accuracy on test_data = %.3f%%, samples have overlap = %d/%d, rate = %.3f%%, overlap success rate = %.3f%%, no match rate = %.3f%%' % (tot / X_test.shape[0] * 100, tot_overlap, X_test.shape[0], tot_overlap / X_test.shape[0] * 100, overlap_success / X_test.shape[0] * 100, no_match / X_test.shape[0] * 100))

            # step6: 过滤出分类错误的训练样本，以供下一次迭代学习
            
            cls_fix,_,_ = solve_conflict(X_train, y_train,num_classes,class_rules)
            
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
                            
                            if (dim1 * 10000 + dim2) not in cls_res:
                                cls_res.append(dim1 * 10000 + dim2)
                
                # fix important bug: cls_res[0], not cls_res
                if len(cls_res) == 0:
                    no_match += 1
                    fault_list.append(idx)
                if len(cls_res) == 1:
                    if cls_res[0] // 10000 == y_train[idx]:
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
                
            print('global rule-based accuracy on train_data = %.3f%%, samples have overlap = %d/%d, rate = %.3f%%, overlap success rate = %.3f%%, no match rate = %.3f%%' % (tot / X_train.shape[0] * 100, tot_overlap, X_train.shape[0], tot_overlap / X_train.shape[0] * 100, overlap_success / X_train.shape[0] * 100, no_match / X_train.shape[0] * 100))

            this_X_train = X_train[fault_list]
            this_y_train = y_train[fault_list]
            # boost end 
    #######################################################################################

        



        # step7 increment rule :save rules that not include new sample
                
        old_class_rules = []
        if log_class != -1:
            for unknown_class_idx in range(len(unknown_data_list)):
                # 只记录log_class
                if unknown_label_list[unknown_class_idx][0]==log_class:

                    unknown_data = unknown_data_list[unknown_class_idx]

                    unknown_label = unknown_label_list[unknown_class_idx]
                    unknown_data, _, unknown_label, _ = train_test_split(unknown_data, unknown_label, test_size=args.test_split_size, random_state=2023)

                    this_X_train = unknown_data.copy()
                    this_y_train = np.ones_like(unknown_label) * num_classes

                    for dim1 in range(num_classes):
                        tmp_old_class_rules = []
                        for dim2 in range(len(class_rules[dim1])):
                            flag = 0
                            for sample in unknown_data:
                                if check_rule(sample, class_rules[dim1][dim2]):
                                    flag += 1
                                
                            if flag==0:

                                tmp_old_class_rules.append(class_rules[dim1][dim2])

                        old_class_rules.append(tmp_old_class_rules)


            cls_fix,_,_ = solve_conflict(X_train, y_train,num_classes,old_class_rules)

            # data plane rule remain for this incremental class
            # 增量没有被删 ,没有参加拓步排序的规则可以保留
            # 留给下个类别的
            all_conflict_id = [] 
            for key, value in cls_fix.items():
                ids = key.strip('#').split('#')
                all_conflict_id.extend(ids)

            data_plane_remain_rule = []
            remain_num = 0
            for dim1 in range(num_classes):
                data_plane_remain_rule_tmp = []
                for dim2 in range(len(old_class_rules[dim1])):
                    if str(dim1 * 10000 + dim2) not in all_conflict_id:
                        data_plane_remain_rule_tmp.append(old_class_rules[dim1][dim2])
                        remain_num += 1
                data_plane_remain_rule.append(data_plane_remain_rule_tmp)
                        

            # reload X
            no_match = 0
            tot = 0
            tot_overlap = 0
            overlap_success = 0
            
            fault_list = []

            for idx in range(X_train.shape[0]):
                
                sample = X_train[idx]
                cls_res = []
                
                for dim1 in range(num_classes):
                    for dim2 in range(len(old_class_rules[dim1])):
                        if check_rule(sample, old_class_rules[dim1][dim2]):
                            
                            if (dim1 * 10000 + dim2) not in cls_res:
                                cls_res.append(dim1 * 10000 + dim2)
                
                # fix important bug: cls_res[0], not cls_res
                if len(cls_res) == 0:
                    no_match += 1
                    fault_list.append(idx)
                if len(cls_res) == 1:
                    if cls_res[0] // 10000 == y_train[idx]:
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
                
            print('incremental old rule-based accuracy on train_data = %.3f%%, samples have overlap = %d/%d, rate = %.3f%%, overlap success rate = %.3f%%, no match rate = %.3f%%' % (tot / X_train.shape[0] * 100, tot_overlap, X_train.shape[0], tot_overlap / X_train.shape[0] * 100, overlap_success / X_train.shape[0] * 100, no_match / X_train.shape[0] * 100))
            # for next new class
            this_X_train = np.concatenate((this_X_train, X_train[fault_list]), axis=0)
            this_y_train = np.concatenate((this_y_train, y_train[fault_list]), axis=0)

        else:
            remain_num = -1
            data_plane_remain_rule = [[] for i in range(num_classes)]


        # Detection Rate
        #           new old
        # pre_new   tp  fp
        # pre_old   fn  tn
        if len(unknown_data_list)==0:
            tp = 0
            fn = 0
            
        else:
            for unknown_class_idx in range(len(unknown_data_list)):
                # 只记录log_class
                if unknown_label_list[unknown_class_idx][0]==log_class:

                    unknown_data = unknown_data_list[unknown_class_idx]
                    # 新类别是真实label。但是旧类别label不与label name对应,
                    # 类别样本不平均使用micro平均多类别
                    unknown_label = unknown_label_list[unknown_class_idx]


                    fn = 0
                    for idx in range(unknown_data.shape[0]):
                        
                        sample = unknown_data[idx]
                        cls_res = []
                        for dim1 in range(num_classes):
                            for dim2 in range(len(class_rules[dim1])):
                                if check_rule(sample, class_rules[dim1][dim2]):
                                    
                                    if dim1 not in cls_res:
                                        cls_res.append(dim1)
                        # 新类中识别为旧类别数目fn
                        if len(cls_res) >= 1:
                            fn += 1

            tp = unknown_data.shape[0] - fn

            print('unknown class = %d, name = %s, final rule-based detect count = %d/%d, rate = %.3f%%' % (unknown_label[0], attack_classes[unknown_label[0]], unknown_data.shape[0] - fn, unknown_data.shape[0], 100 - fn / unknown_data.shape[0] * 100))

    
        # final acc
        #           new old
        # pre_new   tp  fp
        # pre_old   fn  tn
        # 这里可以灵活点，训练集和测试集都试试，看结果有区别没
        if args.solve_conflict_on == 'train_set':
            cls_fix,reduced_cls_fix,conflict_rule_num = solve_conflict(X_train, y_train,num_classes,class_rules)
        elif args.solve_conflict_on == 'test_set':
            cls_fix,reduced_cls_fix,conflict_rule_num = solve_conflict(X_test, y_test,num_classes,class_rules)
        

        conflict_rules,conflict_rules_pri = conflict_2rule(reduced_cls_fix,class_rules,num_classes)

        # 增量保留重叠：保留下来的原型，同时出现上次规则中
        remain_conflict_rules = [[] for i in range(num_classes)]
        remain_conflict_rules_num = 0

        if idx_log_class != 0:
            
            for dim1 in range(num_classes-1):
                for dim2 in range(len(conflict_rules[dim1])):
                    if conflict_rules[dim1][dim2] in old_conflict_rules[dim1]:
                        remain_conflict_rules[dim1].append(conflict_rules[dim1][dim2])
                        remain_conflict_rules_num += 1

        old_conflict_rules = conflict_rules.copy()

       
        print('solve match conflict:')
        print(cls_fix)
        
        save_num = len(cls_fix) - conflict_rule_num
        # priority_dict , add_class_rules_keys = set_prioity(cls_fix)
        # print("-----priority_dict------",priority_dict)
        # conflict_rule_num = len(add_class_rules_keys)
        rule_num = sum([len(i) for i in class_rules]) + conflict_rule_num
        if log_class != -1:
            old_rule_num = sum([len(i) for i in old_class_rules])
        else:
            old_rule_num = -1


        tot = 0
        tot_overlap = 0
        overlap_success = 0
        fp = 0 #预测为新类别，实际为旧类别
        tmp_1 = 0

        for idx in range(X_test.shape[0]):
            
            sample = X_test[idx]
            cls_res = []
            
            for dim1 in range(num_classes):
                for dim2 in range(len(class_rules[dim1])):
                    if check_rule(sample, class_rules[dim1][dim2]):
                        if (dim1 * 10000 + dim2) not in cls_res:
                            cls_res.append(dim1 * 10000 + dim2)



            # 预测类别只有1个，且旧类别预测正确
            if len(cls_res) == 1:
                if cls_res[0] // 10000 == y_test[idx]:
                    tot += 1
                else:
                    tmp_1+=1

            # 预测类别有多个，使用矛盾解决后旧类别预测正确
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
                
            elif len(cls_res) == 0:
                fp += 1


        tn = X_test.shape[0] - fp
        # 注意：混淆矩阵是对于新旧类别分类的矩阵
        new_cm = [tp,fn,fp,tn]
        final_acc = tot / X_test.shape[0] * 100
        print('final rule-based accuracy on test_data = %.3f%%, samples have overlap = %d/%d, rate = %.3f%%, overlap success rate = %.3f%%' % (tot / X_test.shape[0] * 100, tot_overlap, X_test.shape[0], tot_overlap / X_test.shape[0] * 100, overlap_success / X_test.shape[0] * 100))
        print("1_class but wrong rate: ",tmp_1/X_test.shape[0] * 100)

        print("Confusino Matrix:",new_cm)

        # use_SNE
        if use_T_SNE ==True and unknown_label[0] == log_class:
            # 拼接测试集与下一个新类别
            M =np.vstack([X_train,X_test,unknown_data])
            M_class_num = len(np.unique(y_test))
            unknown_label_new_idx = np.ones(unknown_data.shape[0])*M_class_num
            M_label = np.hstack([y_train,y_test,unknown_label_new_idx])
            print(X_test.shape,M.shape)

            # arg TSNE
            perplexity = 300 # 5-50
            lr = 0.1 # 50-1000
            tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=1000, n_jobs=-1, learning_rate=lr,random_state=2024)
            x_dr = tsne.fit_transform(M)
            np.save("./fig_tsne/x_dr.npy",x_dr)

            plt.figure()  # 创建一个画布
            for i in range(M_class_num+1):
                plt.scatter(x_dr[M_label == i, 0], x_dr[M_label == i, 1], s=10, edgecolors='none',label=str(i))
            plt.legend()
            plt.title(args.dataset + '_tsne_lr' + str(lr) + 'per' + str(perplexity))  # 显示标题
            plt.savefig('./fig_tsne/'+args.dataset + '_lr_'+ str(lr) + 'per' + str(perplexity) + '.png', dpi=300)




        if export_entry==True:
            export_range()


        choose_FPR = 100*new_cm[2]/(new_cm[2]+new_cm[3])
        # tp,fn,fp,tn对应相加
        detection_acc = 100*(new_cm[0]+new_cm[3])/sum(new_cm)
        if log_class == -1:
            choose_TPR = -1
            log_class_name = "ALL"
        else:
            
            choose_TPR = 100*new_cm[0]/(new_cm[0]+new_cm[1])
            log_class_name = attack_classes[log_class]
        experi_csv = [args.dataset,log_class_name,args.cls_threshold,args.boost_num,args.temperature,args.prune_T,args.dbscan_eps,detection_acc,choose_TPR,choose_FPR,final_acc,\
                      rule_num,old_rule_num,remain_num,conflict_rule_num,remain_conflict_rules_num,save_num]


                
        if os.path.exists(output_csv):
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                # 写入超参数和结果
                writer.writerow(experi_csv)

        args.selected_class.append(log_class)
    
























if __name__=="__main__":

    use_T_SNE = False

    parser = argparse.ArgumentParser("try to include all parameters")

    parser.add_argument('--gpu', type=int, default=1, help='single GPU device, should in [0-7]')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--epochs', type=int, default=300, help='training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--temperature', type=float, default=0.1, help='training temperature, very important')
    parser.add_argument('--max_proto_num', type=int, default=2000, help='initial prototype number')
    parser.add_argument('--prune_T', type=int, default=10, help='model prune parameter')
    parser.add_argument('--cls_threshold', type=float, default=1.2, help='class distance threshold, very important')
    parser.add_argument('--init_type', type=str, default='DBSCAN', help='k-means / DBSCAN / NONE')
    parser.add_argument('--init_methed', type=str, default='boost_train', help='boost_train/ X_train')
    parser.add_argument('--incremental_train', type=bool, default=True, help="class incremental train")



    parser.add_argument('--dbscan_eps', type=float, default=0.01, help='training temperature, very important, not too small')
    parser.add_argument('--min_samples', type=int, default=2, help='initial prototype number')
    parser.add_argument('--boost_num', type=int, default=3, help='boost iter number')


    parser.add_argument('--cal_dis', type=str, default='l_n', help='only support l_n now')
    parser.add_argument('--cal_mode', type=str, default='abs_trans', help='only support abs_trans now')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--dataset', type=str, default='unsw-nb15', help='ton-iot / ISCX / cicids')
    parser.add_argument('--attack_max_samples', type=int, default=30000, help='ton-iot dataset: max samples per attack')
    parser.add_argument('--selected_class', type=list, default=[0, 1], help='ton-iot dataset: selected attack class, in [1-9]')
    parser.add_argument('--test_split_size', type=float, default=0.8, help='test split size, when few-shot setting, should large')
    parser.add_argument('--add_order_id', type=int, default=1, help='add_order_id')

    


    parser.add_argument('--thr_method', type=str, default="mean", help='method of choose thres: mean /pot / three_sigma')



    parser.add_argument('--solve_conflict_on', type=str, default='test_set', help='train_set / test_set')
    parser.add_argument('--save_model', type=str, default='./final_model/ton-iot', help='path to save model')


    args = parser.parse_args()


    # cicids 0,1,||4,2,5,6,3
    # iscx   2,4,||5,1,3,0
    # iot    (0),1,3,4,||5,8,6,7,9,2

    for dataset in [args.dataset]:#'iscx','cicids','ton-iot','unibs',"unsw-iot"
        args.dataset = dataset
        if dataset == 'iscx':
            args.selected_class = [2,4,]
            add_order = [5,1,3,0,-1]
            args.batch_size = 256
            args.learning_rate = 0.001
            # args.temperature = 0.3
            # args.prune_T = 7
            # args.dbscan_eps = 0.005
            args.min_samples = 2


        elif dataset == 'cicids-2017':
            args.selected_class = [0,1,]
            add_order = [4,2,5,6,3,-1]
            args.batch_size = 512
            args.learning_rate = 0.001
            # args.temperature = 0.1
            # args.prune_T = 5
            # args.dbscan_eps = 0.005
            args.min_samples = 2

        elif dataset == 'cicids-2018':
            args.selected_class = [0,1,]
            add_order = [2,3,4,5,6,7,8,9,10,-1]
            args.batch_size = 256
            args.learning_rate = 0.001
            args.attack_max_samples = 10000
            args.boost_num = 5
            # args.temperature = 0.315973553
            # args.prune_T = 10
            # args.dbscan_eps = 0.202365925
            
            # args.temperature = 0.1757949
            # args.prune_T = 10
            # args.dbscan_eps = 0.340206712



            args.min_samples = 2

        elif dataset == 'ton-iot':
            args.selected_class = [0,8]
            add_order = [4,6,9,2,1,7,3,5,-1]
 

            args.attack_max_samples = 30000
            args.batch_size = 256
            args.learning_rate = 0.001
            args.boost_num = 5
            # args.temperature = 0.10411489
            # args.prune_T = 15
            # args.dbscan_eps = 0.136310726



            args.min_samples = 2


        elif dataset == 'unibs':
            args.selected_class = [0,1,]
            add_order = [2,3,5,6,7,4,-1]
            args.batch_size = 1024
            args.learning_rate = 0.001
            args.temperature = 0.03
            args.prune_T = 10
            args.dbscan_eps = 0.06
            args.min_samples = 2

        elif dataset == 'unsw-iot':
            args.selected_class = [0,1]
            add_order = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,-1]
            args.batch_size = 512
            args.learning_rate = 0.001
            args.temperature = 0.1
            args.prune_T = 5
            args.dbscan_eps = 0.06
            args.min_samples = 2


        elif dataset == 'unsw-nb15':
            args.selected_class = [0,1]
            add_order = [2,3,4,5,6,7,8,9,-1]
            args.batch_size = 512
            args.learning_rate = 0.001
            args.attack_max_samples = 30000
            args.cls_threshold = 1

            # args.boost_num = 5
            # args.temperature = 0.510702046
            # args.prune_T = 10
            # args.dbscan_eps = 0.017891946
            
            args.min_samples = 2


        record_list_name = ['dataset','add_order','choose_threshold','boost_num','temperature','prune_T','dbscan_eps',\
                            'detection_acc','choose_TPR','choose_FPR','final_acc','rule_number',"next_rule_num","remain_num","conflict_rule_num","remain_conflict_rules_num","save_num"]

        output_csv = "unsw2_2_proto_"+str(args.attack_max_samples)+str(args.test_split_size)+".csv"

        # 检查CSV文件是否存在，如果不存在则创建并写入表头
        if not os.path.exists(output_csv):
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                # 写入超参数和结果
                writer.writerow(record_list_name)


        main(args,add_order,output_csv)
