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
# python C4_boost_num.py --dataset ton-iot --temperature 0.05 --cls_threshold 1 --test_split_size 0.8 --boost_num 4 --output_csv exp_prune.csv --prune_T 0

def run_command(gpu_index, batch,num_gpus):

    command = [
        "python", "C4_boost_num.py",
        "--gpu",str(gpu_index % 8),
        # "--dataset",str(["cicids-2018","iscx","ton-iot","unsw-nb15"][int(gpu_index/8)]),
        # "--temperature",str([0.1,0.5,0.05,0.5][int(gpu_index/8)]),
        # "--cls_threshold",str([1.2,1,1,1][int(gpu_index/8)]),
        "--dataset",str(["cicids-2018","iscx","ton-iot","unsw-nb15"][gpu_index%4]),
        "--temperature",str([0.1,0.5,0.05,0.5][gpu_index%4]),
        "--cls_threshold",str([1.2,1,1,1][gpu_index%4]),

        # "--prune_T",str([30,40,50,60,70,80,90,100][gpu_index%8]),
        "--prune_T",str(0),

        "--test_split_size",str(0.8),
        "--boost_num",str(4),
        "--output_csv","exp_prune.csv",
        "--save_model","./save_model/prune" + str(20000+gpu_index + num_gpus*batch),
    ]
    
    # 运行train.py并捕获输出
    result = subprocess.run(command, capture_output=True, text=True)
    
    return command, result

if __name__ == '__main__':
    
    num_gpus = 4


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