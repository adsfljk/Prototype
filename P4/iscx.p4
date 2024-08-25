 /* -*- P4_16 -*- */
#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif
#include "headers.p4"
#include "egress.p4"

const bit<16> TYPE_IPV4 = 0x800;
const bit<8> PROTO_TCP = 6;
const bit<8> PROTO_UDP = 17;
// feature_name = {"protocol": 8, "ip_ihl": 4, "ip_tos": 8, "ip_flags": 8, "ip_ttl": 8,  "tcp_dataofs": 4, "tcp_flags": 8, "tcp_window": 16, "udp_len": 16, "length": 16}

struct my_ingress_metadata_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> udp_length;
    bit<4>  dataOffset;
    bit<16> window;
    bit<8>  flags;
    bit<33> codes_f0;
	bit<31> codes_f1;
	bit<34> codes_f2;
	bit<1> codes_f3;
	bit<1> codes_f4;
	bit<1> codes_f5;
	
}

struct my_ingress_headers_t {
    ethernet_t  ethernet;
    ipv4_t      ipv4;
    tcp_t       tcp;
    udp_t       udp;
}


parser IngressParser(packet_in        pkt,
    out my_ingress_headers_t          hdr,
    out my_ingress_metadata_t         meta,
    out ingress_intrinsic_metadata_t  ig_intr_md)
{

    state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition parse_ipv4;
        }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            PROTO_TCP   : parse_tcp;
            PROTO_UDP   : parse_udp;
            // default: accept;
        }
   }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        meta.dataOffset = hdr.tcp.dataOffset;
        meta.window = hdr.tcp.window;
        meta.flags = hdr.tcp.flags;
        meta.udp_length = 0x0;
        meta.srcPort=hdr.tcp.srcPort;
        meta.dstPort=hdr.tcp.dstPort;
        transition accept;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        meta.dataOffset = 0x0;
        meta.window = 0x0;
        meta.flags = 0x0;
        meta.udp_length = hdr.udp.udp_length;
        meta.srcPort=hdr.udp.srcPort;
        meta.dstPort=hdr.udp.dstPort;
        transition accept;
    }
}


control Ingress(
    /* User */
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{

    action ac_packet_forward(PortId_t port) {
        ig_tm_md.ucast_egress_port = port;
#ifdef BYPASS_EGRESS
        ig_tm_md.bypass_egress = 1;
#endif
    }

    action default_forward() {
        ig_tm_md.ucast_egress_port = 2;
#ifdef BYPASS_EGRESS
        ig_tm_md.bypass_egress = 1;
#endif
    }

    table tb_packet_cls {
        key = {
            meta.codes_f0 : ternary;
		meta.codes_f1 : ternary;
		meta.codes_f2 : ternary;
		meta.codes_f3 : ternary;
		meta.codes_f4 : ternary;
		meta.codes_f5 : ternary;
		
        }
        actions = {
            ac_packet_forward;
            default_forward;
        }
        default_action = default_forward();
        size=1683;
    }
    action ac_fea_f0(bit<33> code){
		meta.codes_f0 = code;
	}

	table tbl_fea_f0{
		key= {hdr.ipv4.protocol : ternary;}
		actions = {ac_fea_f0;}
		size=146;
	}

	action ac_fea_f1(bit<31> code){
		meta.codes_f1 = code;
	}

	table tbl_fea_f1{
		key= {hdr.ipv4.ihl : ternary;}
		actions = {ac_fea_f1;}
		size=120;
	}

	action ac_fea_f2(bit<34> code){
		meta.codes_f2 = code;
	}

	table tbl_fea_f2{
		key= {hdr.ipv4.tos : ternary;}
		actions = {ac_fea_f2;}
		size=138;
	}

	action ac_fea_f3(bit<1> code){
		meta.codes_f3 = code;
	}

	table tbl_fea_f3{
		key= {hdr.ipv4.flags : ternary;}
		actions = {ac_fea_f3;}
		size=0;
	}

	action ac_fea_f4(bit<1> code){
		meta.codes_f4 = code;
	}

	table tbl_fea_f4{
		key= {hdr.ipv4.ttl : ternary;}
		actions = {ac_fea_f4;}
		size=0;
	}

	action ac_fea_f5(bit<1> code){
		meta.codes_f5 = code;
	}

	table tbl_fea_f5{
		key= {meta.dataOffset : ternary;}
		actions = {ac_fea_f5;}
		size=0;
	}

	
    ==tree_tbl==

    apply {
        tbl_fea_f0.apply();
		tbl_fea_f1.apply();
		tbl_fea_f2.apply();
		tbl_fea_f3.apply();
		tbl_fea_f4.apply();
		tbl_fea_f5.apply();
		
        tb_packet_cls.apply();
    }

}

control IngressDeparser(
    packet_out pkt,
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{

    apply {
        pkt.emit(hdr);
    }
}


Pipeline(
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()
) pipe;

Switch(pipe) main;

