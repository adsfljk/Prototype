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

    # prune_T, temperature = param_combinations[gpu_index % len(param_combinations)]
    command = [
        "python", "C0_serch.py",
        "--gpu",str(gpu_index % 8),
        "--temperature", str([0.05,0.5,0.9][gpu_index%3]),
        "--dbscan", str([0.005,0.01,0.05,0.1,0.15,0.2][int(gpu_index/5)]),

        "--prune_T", str([7,10,12,15][batch%5]),
        # "--dataset",str(["cicids-2018","iscx","ton-iot","unsw-nb15"][int(batch/5)]),
        # "--temperature", str(0.9),
        # "--dbscan", str(0.01),
        # "--prune_T", str(10),
        "--dataset",str("unsw-nb15"),

        "--prune_rule",str(0),
        "--boost_num",str(4),

        "--save_model","./save_model/C0" + str(10000+gpu_index + num_gpus*batch),
    ]
    
    # 运行train.py并捕获输出
    result = subprocess.run(command, capture_output=True, text=True)
    # print(result)
    # exit()
    return command, result

if __name__ == '__main__':
    
    num_gpus = 15


    for batch in range(20):
        processes = []
        print(f"Starting batch {batch + 1}")

        for gpu_index in range(num_gpus):

            p = Process(target=run_command, args=(gpu_index,batch,num_gpus))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f"Batch {batch + 1} completed")