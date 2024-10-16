import os
import time
import copy
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='iscx', help='ton-iot / iscx / cicids / unibs')
args = parser.parse_args()

args.dataset = "cicids-2018"



if args.dataset == "unsw-nb15":
    p4_na = "p4 = bfrt.mou_unsw.pipe"   
    entry_paths = "./output/ternary_entry/unsw-nb15/"
    # 指定文件处理顺序
    file_order = ["mou_Worms.txt","mou_Backdoor.txt","mou_DoS.txt",
        "mou_Exploits.txt","mou_Fuzzers.txt","mou_Generic.txt","mou_Reconnaissance.txt",
        "mou_Shellcode.txt","mou_ALL.txt"
    ]
elif args.dataset == "ton-iot":
    p4_na = "p4 = bfrt.mou_ton_iot.pipe"      
    entry_paths = "./output/ternary_entry/ton-iot/"

    file_order = [
        "mou_dos.txt",
        "mou_runsomware.txt",
        "mou_backdoor.txt",
        "mou_injection.txt",
        "mou_ddos.txt",
        "mou_password.txt",
        "mou_scanning.txt",
        "mou_xss.txt",
        "mou_ALL.txt"
    ]
elif args.dataset == "cicids-2018":
    p4_na = "p4 = bfrt.mou_cicids.pipe"     
    entry_paths = "./output/ternary_entry/cicids-2018/"

    file_order = [
        "mou_DDoS_HOIC.txt",
        "mou_DDoS_LOIC_UDP.txt",
        "mou_DoS_GoldenEye.txt",
        "mou_DoS_Hulk.txt",
        "mou_DoS_Slowloris.txt",
        "mou_SSH_BruteForce.txt",
        "mou_Web_Attack_XSS.txt",
        "mou_Web_Attack_SQL.txt",
        "mou_Web_Attack_Brute_Force.txt",
        "mou_ALL.txt"
    ]
elif args.dataset == "iscx":
    p4_na = "p4 = bfrt.mou_iscx.pipe"      
    entry_paths = "./output/ternary_entry/iscx/"

    file_order = [
        "mou_p2p.txt",
        "mou_chat.txt",
        "mou_file_transfer.txt",
        "mou_email.txt",
        "mou_ALL.txt"
    ]


last_ternary_list = []
for idx, file_name in enumerate(file_order):

    with open("./entry/"+args.dataset+'_'+file_name.split(".")[0]+".py",mode="w") as fa:
        print(p4_na,file=fa)
        if idx == 0:
            print('''
def clear_all(p4,verbose=True, batching=True):  
    tb_packet_cls = p4.Ingress.tb_packet_cls
    for table_types in (['MATCH_DIRECT', 'MATCH_INDIRECT_SELECTOR'],
                        ['SELECTOR'],
                        ['ACTION_PROFILE']):
        for table in p4.info(return_info=True, print_info=False):
            if table['type'] in table_types:
                if verbose:
                    print("Clearing table {:<40} ... ".
                        format(table['full_name']), end='', flush=True)
                table['node'].clear(batch=batching)
                if verbose:
                    print('Done')

# 清除所有表项
clear_all(p4,verbose=False)
''',file=fa)
            
        ternary_list = []

        file_path = os.path.join(entry_paths, file_name)

        with open(file_path, "r") as f:
            for line in f:
                ternary_list.append([int(x) for x in line.strip().split()])

        print("tb_packet_cls = p4.Ingress.tb_packet_cls",file = fa)

        # 删除上次的表项

        for value, mask, port in last_ternary_list:
            print('''tb_packet_cls.delete(bin_feature=%s, bin_feature_mask=%s)''' % (value, mask),file = fa)

        for value, mask, port in ternary_list:
            print('''tb_packet_cls.add_with_ac_packet_forward(bin_feature=%s, bin_feature_mask=%s, port=%s)''' % (value, mask,port),file = fa)

        print("bfrt.complete_operations()",file=fa)


        # 更新last_ternary_list为当前的列表
        last_ternary_list = copy.deepcopy(ternary_list)
