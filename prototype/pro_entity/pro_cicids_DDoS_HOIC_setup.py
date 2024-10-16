p4 = bfrt.pro_cicids.pipe
tb_packet_cls = p4.Ingress.tb_packet_cls

def clear_all(verbose=True, batching=True):
    global p4
    global bfrt
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
clear_all(verbose=False)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=0, f1_end=8, f2_start=0, f2_end=64, f3_start=0, f3_end=0, f4_start=0, f4_end=12, f5_start=0, f5_end=41, f6_start=0, f6_end=0, f7_start=0, f7_end=27, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=8, f1_end=21, f2_start=272, f2_end=808, f3_start=0, f3_end=0, f4_start=52, f4_end=171, f5_start=976, f5_end=1104, f6_start=0, f6_end=0, f7_start=106, f7_end=256, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=1, f1_end=2, f2_start=0, f2_end=0, f3_start=0, f3_end=0, f4_start=0, f4_end=0, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=0, f7_end=0, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=5, f1_end=5, f2_start=20, f2_end=32, f3_start=0, f3_end=0, f4_start=2, f4_end=7, f5_start=976, f5_end=976, f6_start=0, f6_end=0, f7_start=203, f7_end=211, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=17, f0_end=17, f1_start=0, f1_end=0, f2_start=500, f2_end=500, f3_start=500, f3_end=500, f4_start=500, f4_end=500, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=0, f7_end=0, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=9, f1_end=15, f2_start=536, f2_end=808, f3_start=0, f3_end=0, f4_start=53, f4_end=139, f5_start=976, f5_end=976, f6_start=0, f6_end=0, f7_start=116, f7_end=238, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=4, f1_end=4, f2_start=20, f2_end=20, f3_start=0, f3_end=0, f4_start=4, f4_end=5, f5_start=964, f5_end=964, f6_start=0, f6_end=0, f7_start=241, f7_end=241, port=1)
bfrt.complete_operations()
