p4 = bfrt.pro_iscx.pipe

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
clear_all(verbose=True)

tb_packet_cls = p4.Ingress.tb_packet_cls

tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=16.0, meta_flags_end=16.0, meta_window_start=256.0, meta_window_end=256.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=1500.0, ipv4_totallen_end=1500.0, port=0)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=16.0, meta_flags_end=16.0, meta_window_start=25397.0, meta_window_end=32768.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=40.0, ipv4_totallen_end=40.0, port=0)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=24.0, meta_flags_end=24.0, meta_window_start=256.0, meta_window_end=256.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=688.0, ipv4_totallen_end=688.0, port=0)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=101.0, meta_udp_length_end=120.0, ipv4_totallen_start=121.0, ipv4_totallen_end=140.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=1067.0, meta_udp_length_end=1189.0, ipv4_totallen_start=1087.0, ipv4_totallen_end=1209.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=64.0, ipv4_ttl_end=64.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=42.0, meta_udp_length_end=86.0, ipv4_totallen_start=62.0, ipv4_totallen_end=106.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=64.0, ipv4_ttl_end=64.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=1106.0, meta_udp_length_end=1189.0, ipv4_totallen_start=1126.0, ipv4_totallen_end=1209.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=619.0, meta_udp_length_end=1065.0, ipv4_totallen_start=639.0, ipv4_totallen_end=1085.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=64.0, ipv4_ttl_end=64.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=707.0, meta_udp_length_end=1105.0, ipv4_totallen_start=727.0, ipv4_totallen_end=1125.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=104.0, ipv4_ttl_end=128.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=91.0, meta_udp_length_end=180.0, ipv4_totallen_start=111.0, ipv4_totallen_end=200.0, port=2)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=10.0, meta_udp_length_end=10.0, ipv4_totallen_start=30.0, ipv4_totallen_end=30.0, port=2)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=1.0, ipv4_ttl_end=4.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=30.0, meta_udp_length_end=181.0, ipv4_totallen_start=50.0, ipv4_totallen_end=201.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=44.0, ipv4_ttl_end=64.0, meta_dataoffset_start=5.0, meta_dataoffset_end=8.0, meta_flags_start=16.0, meta_flags_end=18.0, meta_window_start=42900.0, meta_window_end=65535.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=40.0, ipv4_totallen_end=52.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=128.0, ipv4_ttl_end=255.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=16.0, meta_udp_length_end=653.0, ipv4_totallen_start=36.0, ipv4_totallen_end=673.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=1.0, ipv4_ttl_end=64.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=122.0, meta_udp_length_end=522.0, ipv4_totallen_start=142.0, ipv4_totallen_end=542.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=43.0, ipv4_ttl_end=44.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=4.0, meta_flags_end=17.0, meta_window_start=0.0, meta_window_end=809.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=40.0, ipv4_totallen_end=40.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=124.0, ipv4_ttl_end=128.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=16.0, meta_flags_end=24.0, meta_window_start=16201.0, meta_window_end=32550.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=0.0, ipv4_totallen_end=1390.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=64.0, ipv4_ttl_end=64.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=11.0, meta_udp_length_end=58.0, ipv4_totallen_start=31.0, ipv4_totallen_end=78.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=86.0, meta_udp_length_end=97.0, ipv4_totallen_start=106.0, ipv4_totallen_end=117.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=254.0, ipv4_ttl_end=255.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=50.0, meta_udp_length_end=255.0, ipv4_totallen_start=70.0, ipv4_totallen_end=275.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=43.0, ipv4_ttl_end=44.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=24.0, meta_flags_end=24.0, meta_window_start=344.0, meta_window_end=367.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=70.0, ipv4_totallen_end=99.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=16.0, meta_flags_end=17.0, meta_window_start=16202.0, meta_window_end=16493.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=0.0, ipv4_totallen_end=40.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=43.0, ipv4_ttl_end=46.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=16.0, meta_flags_end=24.0, meta_window_start=344.0, meta_window_end=365.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=103.0, ipv4_totallen_end=1390.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=64.0, ipv4_ttl_end=64.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=24.0, meta_flags_end=24.0, meta_window_start=65535.0, meta_window_end=65535.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=42.0, ipv4_totallen_end=2960.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=64.0, ipv4_ttl_end=64.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=24.0, meta_flags_end=24.0, meta_window_start=29200.0, meta_window_end=45440.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=42.0, ipv4_totallen_end=557.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=127.0, ipv4_ttl_end=128.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=122.0, meta_udp_length_end=220.0, ipv4_totallen_start=142.0, ipv4_totallen_end=240.0, port=3)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=45.0, ipv4_ttl_end=49.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=16.0, meta_flags_end=16.0, meta_window_start=185.0, meta_window_end=250.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=1390.0, ipv4_totallen_end=1390.0, port=4)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=8.0, meta_dataoffset_end=14.0, meta_flags_start=16.0, meta_flags_end=16.0, meta_window_start=50625.0, meta_window_end=65475.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=52.0, ipv4_totallen_end=76.0, port=4)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=16.0, meta_flags_end=17.0, meta_window_start=63112.0, meta_window_end=65475.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=40.0, ipv4_totallen_end=40.0, port=4)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=39.0, ipv4_ttl_end=128.0, meta_dataoffset_start=5.0, meta_dataoffset_end=8.0, meta_flags_start=16.0, meta_flags_end=16.0, meta_window_start=127.0, meta_window_end=23962.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=40.0, ipv4_totallen_end=52.0, port=4)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=48.0, ipv4_ttl_end=49.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=24.0, meta_flags_end=24.0, meta_window_start=237.0, meta_window_end=250.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=391.0, ipv4_totallen_end=1390.0, port=4)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=5.0, meta_dataoffset_end=8.0, meta_flags_start=16.0, meta_flags_end=16.0, meta_window_start=41512.0, meta_window_end=56362.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=40.0, ipv4_totallen_end=52.0, port=4)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=16.0, meta_flags_end=16.0, meta_window_start=57037.0, meta_window_end=61762.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=40.0, ipv4_totallen_end=40.0, port=4)
bfrt.complete_operations()