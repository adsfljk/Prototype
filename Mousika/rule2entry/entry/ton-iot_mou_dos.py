p4 = bfrt.mou_ton_iot.pipe

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

tb_packet_cls = p4.Ingress.tb_packet_cls
tb_packet_cls.add_with_ac_packet_forward(bin_feature=0, bin_feature_mask=0, port=0)
bfrt.complete_operations()
