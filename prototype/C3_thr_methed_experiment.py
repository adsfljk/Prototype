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

    command = [
        "python", "cc_increment.py",
        "--gpu",str(gpu_index % 8),
        "--dataset",str(['iscx',"cicids-2018","ton-iot","unsw-nb15",][gpu_index % 4]),
        "--save_model","./save_model/final" + str(gpu_index + num_gpus*batch),
    ]
    
    # 运行train.py并捕获输出
    result = subprocess.run(command, capture_output=True, text=True)
    # print(result)
    # exit()
    return command, result

if __name__ == '__main__':
    
    num_gpus = 4

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

    for batch in range(1):
        processes = []
        print(f"Starting batch {batch + 1}")

        for gpu_index in range(num_gpus):
            p = Process(target=run_command, args=(gpu_index,batch,num_gpus))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f"Batch {batch + 1} completed")