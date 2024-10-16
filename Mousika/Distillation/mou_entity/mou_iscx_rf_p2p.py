
p4 = bfrt.mou_iscx.pipe
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

tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=0, ipv4_protocol_end=255, ipv4_ihl_start=0, ipv4_ihl_end=15, ipv4_tos_start=0, ipv4_tos_end=255, ipv4_flags_start=0, ipv4_flags_end=7, ipv4_ttl_start=0, ipv4_ttl_end=255, meta_dataoffset_start=0, meta_dataoffset_end=15, meta_flags_start=0, meta_flags_end=255, meta_window_start=0, meta_window_end=65535, meta_udp_length_start=0, meta_udp_length_end=65535, ipv4_totallen_start=0, ipv4_totallen_end=65535, port=0)
tb_packet_cls.add_with_ac_packet_forward(ipv4_protocol_start=0, ipv4_protocol_end=255, ipv4_ihl_start=0, ipv4_ihl_end=15, ipv4_tos_start=0, ipv4_tos_end=255, ipv4_flags_start=0, ipv4_flags_end=7, ipv4_ttl_start=0, ipv4_ttl_end=255, meta_dataoffset_start=0, meta_dataoffset_end=15, meta_flags_start=0, meta_flags_end=255, meta_window_start=0, meta_window_end=65535, meta_udp_length_start=0, meta_udp_length_end=65535, ipv4_totallen_start=0, ipv4_totallen_end=65535, port=1)
bfrt.complete_operations()