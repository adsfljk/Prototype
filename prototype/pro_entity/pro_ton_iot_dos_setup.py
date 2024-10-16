p4 = bfrt.pro_ton_iot.pipe
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
tb_packet_cls.add_with_ac_packet_forward(f0_start=155, f0_end=448, f1_start=110, f1_end=244, f2_start=318, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=2, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=282, f0_end=512, f1_start=132, f1_end=267, f2_start=1022, f2_end=2948, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=1, f5_end=9, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=426, f0_end=499, f1_start=164, f1_end=185, f2_start=920, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=1, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=366, f0_end=396, f1_start=193, f1_end=206, f2_start=1500, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=530, f0_end=574, f1_start=149, f1_end=159, f2_start=1500, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=397, f0_end=425, f1_start=184, f1_end=195, f2_start=1500, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=192, f0_end=293, f1_start=159, f1_end=349, f2_start=991, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=1, f5_end=2, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=352, f0_end=430, f1_start=183, f1_end=213, f2_start=1500, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=1, f5_end=1, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=460, f0_end=517, f1_start=161, f1_end=174, f2_start=1500, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=374, f0_end=437, f1_start=121, f1_end=181, f2_start=379, f2_end=871, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=503, f0_end=612, f1_start=143, f1_end=165, f2_start=1500, f2_end=1702, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=2, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
bfrt.complete_operations()
