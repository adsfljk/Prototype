p4 = bfrt.mou_unsw.pipe

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
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825356782512495184095215682, bin_feature_mask=56668397794435742564354, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825356782512495184095215682, bin_feature_mask=56668397794435743612930, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825356782512495184094167106, bin_feature_mask=56668397794435742564418, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825356782512495184094167042, bin_feature_mask=37778931862957161709570, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825337893046563705513312274, bin_feature_mask=37778931862957161709584, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=643731597179889670303275548690, bin_feature_mask=9906278176309038072014569472, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=643731597179889670303275548690, bin_feature_mask=2757862025995872821575680, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633828076865606628104082554898, bin_feature_mask=2455630571092215527899136, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633827774634151724446864375826, bin_feature_mask=37778931862957245595648, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633828983559971339076039082002, bin_feature_mask=37778931862957178486784, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633829134675698790904685920274, bin_feature_mask=1397820478929415000031232, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633829134712592279052240289810, bin_feature_mask=1246704751477586487410688, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633829134712592279052240289810, bin_feature_mask=1246704751477586488459264, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633829134712592279052239241234, bin_feature_mask=1246704751477586353192960, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633829134712592279052105023506, bin_feature_mask=1246741644965733772296192, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=643731294948434766645991309330, bin_feature_mask=37778931862957187923968, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=643731294948434766645991309330, bin_feature_mask=9903558093214905156380917760, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633827774634151724446798315538, bin_feature_mask=37778931862957186875392, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825961245422302503026819090, bin_feature_mask=37778931862957212041216, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825961245422302503026819090, bin_feature_mask=642241841670271799394304, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825961245422302502994837522, bin_feature_mask=642241841670271750635520, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825961245422302502994837522, bin_feature_mask=642241841670271749586944, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825961245422302502993788946, bin_feature_mask=642241841670271749062656, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825961245422302502993264658, bin_feature_mask=37778931862957161709568, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825356782512495188389134354, bin_feature_mask=56668397794435742564352, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633830778059234829541094457362, bin_feature_mask=5477945120128792742854656, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633830778059234829543242989586, bin_feature_mask=642241841670276044029952, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633830778059234829543242989586, bin_feature_mask=642241841670276045078528, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633830778059234829543241941010, bin_feature_mask=642241841670278191513600, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825942355956371024395632658, bin_feature_mask=37778931862961456676864, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825337893046563707660796418, bin_feature_mask=633825337893046563707660796416, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633864023519274231841251394054, bin_feature_mask=633864023519274231841251393536, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633864325750729135498545070598, bin_feature_mask=633825640124501467364954472448, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=643770263916651406956087476742, bin_feature_mask=643728858207329605906853789696, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=643770263916651406956087476742, bin_feature_mask=643731276058968835165203202048, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=643767846065012177697905836550, bin_feature_mask=633825337893046563707694350336, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=643767846065012177697905836550, bin_feature_mask=633825337893046563707828568064, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=643767846065012177697771618822, bin_feature_mask=633825337893046563707660795904, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825337893046563707660796422, bin_feature_mask=633825337893046563707660795908, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825337893046563707660795906, bin_feature_mask=633825337893046563707660795906, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825337893046563707929231360, bin_feature_mask=633825337893046563705513312256, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825337893046563707929231366, bin_feature_mask=633825337893046563705781747716, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825338483342374066634883078, bin_feature_mask=633825337893046563705781747712, port=1)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825338483342374066634883078, bin_feature_mask=633825338483342374064487399424, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=633825337893046563707929231362, bin_feature_mask=633825337893046563705781747714, port=0)
tb_packet_cls.add_with_ac_packet_forward(bin_feature=37778931862957161709568, bin_feature_mask=0, port=0)
bfrt.complete_operations()
