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
        "python", "C3_cc_increment.py",
        "--gpu",str(gpu_index % 8),
        "--dataset",str(["cicids-2018","ton-iot","unsw-nb15"][batch%3]),
        # "--dataset","unsw-nb15",
        "--test_split_size",str(0.8),
        "--boost_num",str(4),
        "--cls_threshold",str(0.05 + [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,][gpu_index%23]),
        "--save_model","./save_model/tmp" + str(gpu_index + num_gpus*batch),
    ]
    
    # 运行train.py并捕获输出
    result = subprocess.run(command, capture_output=True, text=True)
    
    return command, result

if __name__ == '__main__':
    
    num_gpus = 23


    for batch in range(3):
        processes = []
        print(f"Starting batch {batch + 1}")

        for gpu_index in range(num_gpus):
            p = Process(target=run_command, args=(gpu_index,batch,num_gpus))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f"Batch {batch + 1} completed")