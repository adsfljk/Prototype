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

struct my_ingress_metadata_t {
    // bit<16> srcPort;
    // bit<16> dstPort;
    bit<16> udp_length;
    bit<4>  dataOffset;
    bit<16> window;
    bit<8>  flags;
    bit<1> srcPort_0;
    bit<1> srcPort_1;
    bit<1> srcPort_2;
    bit<1> srcPort_3;
    bit<1> srcPort_4;
    bit<1> srcPort_5;
    bit<1> srcPort_6;
    bit<1> srcPort_7;
    bit<1> srcPort_8;
    bit<1> srcPort_9;
    bit<1> srcPort_10;
    bit<1> srcPort_11;
    bit<1> srcPort_12;
    bit<1> srcPort_13;
    bit<1> srcPort_14;
    bit<1> srcPort_15;

    bit<1> dstPort_0;
    bit<1> dstPort_1;
    bit<1> dstPort_2;
    bit<1> dstPort_3;
    bit<1> dstPort_4;
    bit<1> dstPort_5;
    bit<1> dstPort_6;
    bit<1> dstPort_7;
    bit<1> dstPort_8;
    bit<1> dstPort_9;
    bit<1> dstPort_10;
    bit<1> dstPort_11;
    bit<1> dstPort_12;
    bit<1> dstPort_13;
    bit<1> dstPort_14;
    bit<1> dstPort_15;

    ==codes==
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
        // meta.srcPort=hdr.tcp.srcPort;
        // meta.dstPort=hdr.tcp.dstPort;
        meta.srcPort_0 = hdr.tcp.srcPort[0:0];
        meta.srcPort_1 = hdr.tcp.srcPort[1:1];
        meta.srcPort_2 = hdr.tcp.srcPort[2:2];
        meta.srcPort_3 = hdr.tcp.srcPort[3:3];
        meta.srcPort_4 = hdr.tcp.srcPort[4:4];
        meta.srcPort_5 = hdr.tcp.srcPort[5:5];
        meta.srcPort_6 = hdr.tcp.srcPort[6:6];
        meta.srcPort_7 = hdr.tcp.srcPort[7:7];
        meta.srcPort_8 = hdr.tcp.srcPort[8:8];
        meta.srcPort_9 = hdr.tcp.srcPort[9:9];
        meta.srcPort_10 = hdr.tcp.srcPort[10:10];
        meta.srcPort_11 = hdr.tcp.srcPort[11:11];
        meta.srcPort_12 = hdr.tcp.srcPort[12:12];
        meta.srcPort_13 = hdr.tcp.srcPort[13:13];
        meta.srcPort_14 = hdr.tcp.srcPort[14:14];
        meta.srcPort_15 = hdr.tcp.srcPort[15:15];
        meta.dstPort_0 = hdr.tcp.dstPort[0:0];
        meta.dstPort_1 = hdr.tcp.dstPort[1:1];
        meta.dstPort_2 = hdr.tcp.dstPort[2:2];
        meta.dstPort_3 = hdr.tcp.dstPort[3:3];
        meta.dstPort_4 = hdr.tcp.dstPort[4:4];
        meta.dstPort_5 = hdr.tcp.dstPort[5:5];
        meta.dstPort_6 = hdr.tcp.dstPort[6:6];
        meta.dstPort_7 = hdr.tcp.dstPort[7:7];
        meta.dstPort_8 = hdr.tcp.dstPort[8:8];
        meta.dstPort_9 = hdr.tcp.dstPort[9:9];
        meta.dstPort_10 = hdr.tcp.dstPort[10:10];
        meta.dstPort_11 = hdr.tcp.dstPort[11:11];
        meta.dstPort_12 = hdr.tcp.dstPort[12:12];
        meta.dstPort_13 = hdr.tcp.dstPort[13:13];
        meta.dstPort_14 = hdr.tcp.dstPort[14:14];
        meta.dstPort_15 = hdr.tcp.dstPort[15:15];

        transition accept;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        meta.dataOffset = 0x0;
        meta.window = 0x0;
        meta.flags = 0x0;
        meta.udp_length = hdr.udp.udp_length;
        // meta.srcPort=hdr.udp.srcPort;
        // meta.dstPort=hdr.udp.dstPort;
        meta.srcPort_0 = hdr.udp.srcPort[0:0];
        meta.srcPort_1 = hdr.udp.srcPort[1:1];
        meta.srcPort_2 = hdr.udp.srcPort[2:2];
        meta.srcPort_3 = hdr.udp.srcPort[3:3];
        meta.srcPort_4 = hdr.udp.srcPort[4:4];
        meta.srcPort_5 = hdr.udp.srcPort[5:5];
        meta.srcPort_6 = hdr.udp.srcPort[6:6];
        meta.srcPort_7 = hdr.udp.srcPort[7:7];
        meta.srcPort_8 = hdr.udp.srcPort[8:8];
        meta.srcPort_9 = hdr.udp.srcPort[9:9];
        meta.srcPort_10 = hdr.udp.srcPort[10:10];
        meta.srcPort_11 = hdr.udp.srcPort[11:11];
        meta.srcPort_12 = hdr.udp.srcPort[12:12];
        meta.srcPort_13 = hdr.udp.srcPort[13:13];
        meta.srcPort_14 = hdr.udp.srcPort[14:14];
        meta.srcPort_15 = hdr.udp.srcPort[15:15];
        meta.dstPort_0 = hdr.udp.dstPort[0:0];
        meta.dstPort_1 = hdr.udp.dstPort[1:1];
        meta.dstPort_2 = hdr.udp.dstPort[2:2];
        meta.dstPort_3 = hdr.udp.dstPort[3:3];
        meta.dstPort_4 = hdr.udp.dstPort[4:4];
        meta.dstPort_5 = hdr.udp.dstPort[5:5];
        meta.dstPort_6 = hdr.udp.dstPort[6:6];
        meta.dstPort_7 = hdr.udp.dstPort[7:7];
        meta.dstPort_8 = hdr.udp.dstPort[8:8];
        meta.dstPort_9 = hdr.udp.dstPort[9:9];
        meta.dstPort_10 = hdr.udp.dstPort[10:10];
        meta.dstPort_11 = hdr.udp.dstPort[11:11];
        meta.dstPort_12 = hdr.udp.dstPort[12:12];
        meta.dstPort_13 = hdr.udp.dstPort[13:13];
        meta.dstPort_14 = hdr.udp.dstPort[14:14];
        meta.dstPort_15 = hdr.udp.dstPort[15:15];
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
            ==codes_ternary==
        }
        actions = {
            ac_packet_forward;
            default_forward;
        }
        default_action = default_forward();
        size===model_size==;
    }
    ==fea_tbl==

    apply {
        ==apply_tbl==
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

