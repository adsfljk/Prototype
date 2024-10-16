 /* -*- P4_16 -*- */
#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif
#include "utils/headers.p4"
#include "utils/egress.p4"

const bit<16> TYPE_IPV4 = 0x800;
const bit<8> PROTO_TCP = 6;
const bit<8> PROTO_UDP = 17;

struct my_ingress_metadata_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> udp_length;
    bit<4>  dataOffset;
    bit<16> window;
    bit<8>  flags;
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
    ==tree_tbl==

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

