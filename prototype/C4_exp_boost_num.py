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
        "python", "C4_boost_num.py",
        "--gpu",str(gpu_index % 8),
        "--dataset",str(["cicids-2018","iscx","ton-iot","unsw-nb15"][int(batch/8)]),
        "--prune_T", str([10,7,10,7][int(batch/8)]),
        "--cls_threshold",str([1.2,1,1,1][int(batch/8)]),
        "--test_split_size",str(0.8),
        "--temperature",str([0.1,0.5,0.05,0.5][int(batch/8)]),


        "--boost_num",str([1,2,3,4,5,6,7,8,9,10][gpu_index%10]),
        "--output_csv","exp_boost_num.csv",
        "--save_model","./save_model/tmp" + str(20000+gpu_index + num_gpus*batch),
    ]
    
    # 运行train.py并捕获输出
    result = subprocess.run(command, capture_output=True, text=True)
    
    return command, result

if __name__ == '__main__':
    
    num_gpus = 10


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