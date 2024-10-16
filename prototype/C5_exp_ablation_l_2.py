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
# python C5_ablation.py --dataset cicids-2018 --cal_dis l_2 --cls_threshold 1.2 --output_csv exp_ablation_l_2.csv

def run_command(gpu_index, batch,num_gpus):

    command = [
        "python", "C5_ablation.py",
        "--gpu",str(gpu_index % 8),
        "--cal_dis","l_2",

        # "--cls_threshold","1.2",
        # "--dataset","cicids-2018",
        # "--temperature","0.1",


        "--cls_threshold","1",
        "--dataset","ton-iot",
        "--temperature","0.05",

        "--output_csv","exp_ablation_l_2.csv",

        "--save_model","./save_model/ablation" + str(1000+gpu_index + num_gpus*batch),
    ]
    
    # 运行train.py并捕获输出

    result = subprocess.run(command, capture_output=True, text=True)
    # print(result)
    # exit()
    return command, result

if __name__ == '__main__':
    
    num_gpus = 1

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