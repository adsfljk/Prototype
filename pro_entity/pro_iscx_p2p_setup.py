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
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=16.0, meta_flags_end=16.0, meta_window_start=18192.0, meta_window_end=32768.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=40.0, ipv4_totallen_end=40.0, port=0)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=6.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=5.0, meta_dataoffset_end=5.0, meta_flags_start=16.0, meta_flags_end=24.0, meta_window_start=256.0, meta_window_end=16201.0, meta_udp_length_start=0.0, meta_udp_length_end=0.0, ipv4_totallen_start=40.0, ipv4_totallen_end=688.0, port=0)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=1126.0, meta_udp_length_end=1189.0, ipv4_totallen_start=1146.0, ipv4_totallen_end=1209.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=64.0, ipv4_ttl_end=64.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=42.0, meta_udp_length_end=86.0, ipv4_totallen_start=62.0, ipv4_totallen_end=106.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=6.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=53.0, ipv4_ttl_end=76.0, meta_dataoffset_start=0.0, meta_dataoffset_end=10.0, meta_flags_start=0.0, meta_flags_end=24.0, meta_window_start=0.0, meta_window_end=29200.0, meta_udp_length_start=0.0, meta_udp_length_end=1189.0, ipv4_totallen_start=52.0, ipv4_totallen_end=2728.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=875.0, meta_udp_length_end=1060.0, ipv4_totallen_start=895.0, ipv4_totallen_end=1080.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=95.0, meta_udp_length_end=127.0, ipv4_totallen_start=115.0, ipv4_totallen_end=147.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=0.0, ipv4_flags_end=0.0, ipv4_ttl_start=128.0, ipv4_ttl_end=128.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=1061.0, meta_udp_length_end=1125.0, ipv4_totallen_start=1081.0, ipv4_totallen_end=1145.0, port=1)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=17.0, ipv4_protocol_end=17.0, ipv4_ihl_start=5.0, ipv4_ihl_end=5.0, ipv4_tos_start=0.0, ipv4_tos_end=0.0, ipv4_flags_start=2.0, ipv4_flags_end=2.0, ipv4_ttl_start=64.0, ipv4_ttl_end=64.0, meta_dataoffset_start=0.0, meta_dataoffset_end=0.0, meta_flags_start=0.0, meta_flags_end=0.0, meta_window_start=0.0, meta_window_end=0.0, meta_udp_length_start=707.0, meta_udp_length_end=1103.0, ipv4_totallen_start=727.0, ipv4_totallen_end=1123.0, port=1)
bfrt.complete_operations()