p4 = bfrt.pro_cicids.pipe
tb_packet_cls = p4.Ingress.tb_packet_cls
tb_packet_cls.delete(f0_start=6, f0_end=6, f1_start=0, f1_end=8, f2_start=0, f2_end=64, f3_start=0, f3_end=0, f4_start=0, f4_end=12, f5_start=0, f5_end=41, f6_start=0, f6_end=0, f7_start=0, f7_end=27)
tb_packet_cls.delete(f0_start=6, f0_end=6, f1_start=8, f1_end=21, f2_start=272, f2_end=808, f3_start=0, f3_end=0, f4_start=52, f4_end=171, f5_start=976, f5_end=1104, f6_start=0, f6_end=0, f7_start=106, f7_end=256)
tb_packet_cls.delete(f0_start=6, f0_end=6, f1_start=1, f1_end=2, f2_start=0, f2_end=0, f3_start=0, f3_end=0, f4_start=0, f4_end=0, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=0, f7_end=0)
tb_packet_cls.delete(f0_start=6, f0_end=6, f1_start=9, f1_end=15, f2_start=536, f2_end=808, f3_start=0, f3_end=0, f4_start=53, f4_end=139, f5_start=976, f5_end=976, f6_start=0, f6_end=0, f7_start=116, f7_end=238)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=0, f1_end=8, f2_start=0, f2_end=64, f3_start=0, f3_end=0, f4_start=0, f4_end=12, f5_start=0, f5_end=41, f6_start=0, f6_end=0, f7_start=0, f7_end=27, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=8, f1_end=21, f2_start=272, f2_end=808, f3_start=0, f3_end=0, f4_start=52, f4_end=171, f5_start=976, f5_end=1104, f6_start=0, f6_end=0, f7_start=106, f7_end=256, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=1, f1_end=2, f2_start=0, f2_end=0, f3_start=0, f3_end=0, f4_start=0, f4_end=0, f5_start=0, f5_end=0, f6_start=0, f6_end=0, f7_start=0, f7_end=0, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=9, f1_end=15, f2_start=536, f2_end=808, f3_start=0, f3_end=0, f4_start=53, f4_end=139, f5_start=976, f5_end=976, f6_start=0, f6_end=0, f7_start=116, f7_end=238, port=0)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=5, f1_end=5, f2_start=73, f2_end=73, f3_start=0, f3_end=0, f4_start=14, f4_end=14, f5_start=935, f5_end=935, f6_start=0, f6_end=0, f7_start=187, f7_end=187, port=2)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=5, f1_end=5, f2_start=224, f2_end=365, f3_start=0, f3_end=0, f4_start=44, f4_end=73, f5_start=935, f5_end=935, f6_start=0, f6_end=0, f7_start=187, f7_end=187, port=2)
tb_packet_cls.add_with_ac_packet_forward(f0_start=6, f0_end=6, f1_start=5, f1_end=5, f2_start=259, f2_end=326, f3_start=0, f3_end=0, f4_start=51, f4_end=65, f5_start=935, f5_end=935, f6_start=0, f6_end=0, f7_start=187, f7_end=187, port=2)
bfrt.complete_operations()
