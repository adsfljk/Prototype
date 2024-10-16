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
tb_packet_cls.add_with_ac_packet_forward(f0_start=0.0, f0_end=0.0, f1_start=1.0, f1_end=21.0, f2_start=0.0, f2_end=21.0, f3_start=0.0, f3_end=304.0, f4_start=0.0, f4_end=1968.0, f5_start=0.0, f5_end=0.0, f6_start=0.0, f6_end=303.0, f7_start=0.0, f7_end=655.0, f8_start=0.0, f8_end=303.0, f9_start=8.0, f9_end=65498.5, f10_start=8.0, f10_end=65441.0, f11_start=2.0, f11_end=38045.0, f12_start=1.0, f12_end=6.0, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=0.0, f0_end=0.0, f1_start=1.0, f1_end=17.0, f2_start=1.0, f2_end=17.0, f3_start=0.0, f3_end=186.28571428571428, f4_start=0.0, f4_end=976.0, f5_start=0.0, f5_end=0.0, f6_start=0.0, f6_end=38.0, f7_start=0.0, f7_end=178.0, f8_start=0.0, f8_end=0.0, f9_start=1.0, f9_end=65210.0, f10_start=1.0, f10_end=65205.0, f11_start=1.0, f11_end=45.0, f12_start=6.0, f12_end=6.0, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=0.0, f0_end=0.0, f1_start=6.0, f1_end=6.0, f2_start=0.0, f2_end=0.0, f3_start=500.0, f3_end=500.0, f4_start=500.0, f4_end=500.0, f5_start=500.0, f5_end=500.0, f6_start=273.0, f6_end=273.0, f7_start=640.0, f7_end=640.0, f8_start=61.0, f8_end=61.0, f9_start=4459.800000000745, f9_end=5212.199999999255, f10_start=43844.0, f10_end=50220.0, f11_start=1924.0, f11_end=3564.0, f12_start=17.0, f12_end=17.0, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=0.0, f0_end=0.0, f1_start=5.0, f1_end=5.0, f2_start=4.0, f2_end=4.0, f3_start=109.33333333333334, f3_end=109.33333333333334, f4_start=964.0, f4_end=964.0, f5_start=0.0, f5_end=0.0, f6_start=158.0, f6_end=192.0, f7_start=1260.0, f7_end=1543.0, f8_start=0.0, f8_end=0.0, f9_start=167.75, f9_end=65506.375, f10_start=401.0, f10_end=65178.0, f11_start=1.0, f11_end=23.0, f12_start=6.0, f12_end=6.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(f0_start=0.0, f0_end=0.0, f1_start=5.0, f1_end=6.0, f2_start=4.0, f2_end=5.0, f3_start=89.45454545454545, f3_end=109.33333333333334, f4_start=964.0, f4_end=964.0, f5_start=0.0, f5_end=0.0, f6_start=99.0, f6_end=194.0, f7_start=864.0, f7_end=1558.0, f8_start=0.0, f8_end=0.0, f9_start=7866.25, f9_end=58592.625, f10_start=7908.0, f10_end=59301.0, f11_start=1.0, f11_end=13.0, f12_start=6.0, f12_end=6.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(f0_start=0.0, f0_end=0.0, f1_start=4.0, f1_end=5.0, f2_start=4.0, f2_end=4.0, f3_start=109.33333333333334, f3_end=123.0, f4_start=964.0, f4_end=964.0, f5_start=0.0, f5_end=0.0, f6_start=1.0, f6_end=158.0, f7_start=8.0, f7_end=1255.0, f8_start=0.0, f8_end=0.0, f9_start=92.75, f9_end=65438.75, f10_start=18.0, f10_end=65519.0, f11_start=1.0, f11_end=14.0, f12_start=6.0, f12_end=6.0, port=1)
bfrt.complete_operations()
