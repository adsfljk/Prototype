p4 = bfrt.mou_iscx.pipe

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
tb_packet_cls.add_with_ac_packet_forward(bin_feature=309485009965460256800636928, bin_feature_mask=309485009821345068724781056, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=309485009965460256800636944, bin_feature_mask=16, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=309485009965460566038282256, bin_feature_mask=274877906944, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=309485009965460566038282256, bin_feature_mask=309237645312, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=309485009965460531678543888, bin_feature_mask=0, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=144115188075855872, bin_feature_mask=144115188075855872, port=0)
bfrt.complete_operations()
