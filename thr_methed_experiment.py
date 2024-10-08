import subprocess
import csv
import os
import random 
from multiprocessing import Pool
from multiprocessing import Process
import itertools

# iscx 6 class
# iot 10 class
# cicids 7 class


def run_command(gpu_index, batch,num_gpus):
    # combinations = [list(combo) for combo in itertools.combinations([0,1,2,3,4,5,6,7,8,9], 4)]
    # combination = combinations[(gpu_index + 7*batch)%len(combinations)]
    # combination = [1,2,3,4,5,6,7,8,9]
    # combo = ""
    # combo2 = ""
    # for i in combination:
    #     combo += str(i)+' '
    #     combo2 += str(i) 

    # 不能有空格
    command = [
        "python", "cc_increment.py",
        "--gpu",str(gpu_index % 8),
        "--temperature", str(random.uniform(0.001,1)),
        # "--temperature", str(0.5825),
        # "--thr_method",["mean","pot","three_sigma","median","percentile"][int(gpu_index % 5)],
        # "--prune_T",str(random.choice([5,8,10,15,5,20,30])),
        # "--prune_T",str(15),
        # "--dataset",str(["ton-iot","cicids-2018"][int(gpu_index % 2)]),
        "--dataset","unsw-nb15",
        "--boost_num",str(5),
        "--prune_T",str([5,10,15,20][int(gpu_index % 4)]),
        # "--add_order_id",str(int(gpu_index % 8)),
        "--dbscan_eps",str(random.uniform(0.0001,0.1)),
        # "--dbscan_eps",str(0.025918019),
        "--save_model","./save_model/tmp" + str(gpu_index + num_gpus*batch),
        # "--selected_class",#后面添加
    ]
    # for i in combination:
    #     command.append(str(i))
    
    # 运行train.py并捕获输出
    result = subprocess.run(command, capture_output=True, text=True)
    return command, result

if __name__ == '__main__':
    
    num_gpus = 16

    # # 输出CSV文件的路径
    # output_csv = "results.csv"
    # hyperparams = ["selected_class","batch_size","learning_rate","temperature","max_proto_num","prune_T","dbscan_eps","cls_threshold"]
    # # 检查CSV文件是否存在，如果不存在则创建并写入表头
    # if not os.path.exists(output_csv):
    #     with open(output_csv, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["selected_class","batch_size","learning_rate","temperature","max_proto_num","prune_T","cls_threshold",\
    #                         "best_test_acc","prune_test_acc","unknown_class_rate","final_rule-based_acc"])

    # command, result = run_command(0+1,0)
    # print(result)
    # exit()

    for batch in range(1000):
        processes = []
        print(f"Starting batch {batch + 1}")

        for gpu_index in range(num_gpus):
            p = Process(target=run_command, args=(gpu_index,batch,num_gpus))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f"Batch {batch + 1} completed")