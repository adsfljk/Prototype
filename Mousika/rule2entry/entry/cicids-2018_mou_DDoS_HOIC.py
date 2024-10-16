p4 = bfrt.mou_cicids.pipe

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
tb_packet_cls.add_with_ac_packet_forward(bin_feature=77371252455336301540933632, bin_feature_mask=77371252455336301540933632, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=77371252456462201447776256, bin_feature_mask=34359738368, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=77371252456462201447776256, bin_feature_mask=1125934266580992, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=34359738370, bin_feature_mask=0, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=9671406556917067757387810, bin_feature_mask=9671406556917033397649410, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=9671406556917067757387810, bin_feature_mask=9671406556917033397649442, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=9671406556917067757387778, bin_feature_mask=2, port=0)
bfrt.complete_operations()
