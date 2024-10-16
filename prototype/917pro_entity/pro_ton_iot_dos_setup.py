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
tb_packet_cls.add_with_ac_packet_forward(f0_start=548, f0_end=612, f1_start=143, f1_end=156, f2_start=1500, f2_end=1702, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=1, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=406, f0_end=533, f1_start=121, f1_end=191, f2_start=473, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=1, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=531, f0_end=579, f1_start=148, f1_end=159, f2_start=1500, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=367, f0_end=516, f1_start=161, f1_end=267, f2_start=1500, f2_end=2948, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=5, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=374, f0_end=409, f1_start=123, f1_end=159, f2_start=379, f2_end=723, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=293, f0_end=369, f1_start=139, f1_end=245, f2_start=1022, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=1, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=283, f0_end=329, f1_start=110, f1_end=159, f2_start=318, f2_end=991, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=1, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=410, f0_end=517, f1_start=160, f1_end=191, f2_start=923, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=1, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=370, f0_end=433, f1_start=132, f1_end=206, f2_start=1237, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=2, f5_end=2, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=220, f0_end=263, f1_start=267, f1_end=312, f2_start=1500, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=1, f5_end=1, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=293, f0_end=381, f1_start=140, f1_end=245, f2_start=1500, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=2, f5_end=4, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=389, f0_end=437, f1_start=122, f1_end=191, f2_start=528, f2_end=949, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=370, f0_end=462, f1_start=175, f1_end=206, f2_start=1500, f2_end=1500, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=1, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=302, f0_end=315, f1_start=111, f1_end=114, f2_start=402, f2_end=528, f3_start=52, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=1883, f8_end=1883, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=7, f0_end=15, f1_start=48, f1_end=282, f2_start=71, f2_end=765, f3_start=40, f3_end=52, f4_start=0, f4_end=0, f5_start=1, f5_end=3, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=80, f8_end=443, port=1)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=70, f1_end=82, f2_start=86, f2_end=98, f3_start=40, f3_end=52, f4_start=1, f4_end=1, f5_start=5, f5_end=5, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=443, f8_end=443, port=1)
tb_packet_cls.add_with_ac_packet_forward(f0_start=12, f0_end=60, f1_start=203, f1_end=415, f2_start=976, f2_end=1500, f3_start=40, f3_end=52, f4_start=0, f4_end=0, f5_start=0, f5_end=1, f6_start=0, f6_end=0, f7_start=6, f7_end=6, f8_start=443, f8_end=443, port=1)
bfrt.complete_operations()
