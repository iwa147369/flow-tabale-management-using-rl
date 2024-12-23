from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, raw

class BaseController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(BaseController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}

    def extract_flow_type(self, pkt):
        """Extract flow type and priority from packet payload"""
        for p in pkt.protocols:
            if isinstance(p, raw.Raw):
                flow_type = p.load.decode()
                # Get flow requirements from traffic generator's flow types
                if flow_type in ['realtime', 'streaming', 'background']:
                    return flow_type
        return 'background'  # default to background if not specified

    def get_flow_priority(self, flow_type):
        """Get priority based on flow type"""
        priorities = {
            'realtime': 3,
            'streaming': 2,
            'background': 1
        }
        return priorities.get(flow_type, 1) * 100  # Scale priorities for OpenFlow

    def get_flow_timeout(self, flow_type):
        """Get timeout based on flow type"""
        timeouts = {
            'realtime': 5,
            'streaming': 30,
            'background': 60
        }
        return timeouts.get(flow_type, 60)

    def add_flow(self, datapath, priority, match, actions, timeout=0):
        """Add a flow entry with specified priority and timeout"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            hard_timeout=timeout
        )
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Parse packet
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        
        # Extract flow type and get corresponding priority and timeout
        flow_type = self.extract_flow_type(pkt)
        priority = self.get_flow_priority(flow_type)
        timeout = self.get_flow_timeout(flow_type)

        # Create match and actions
        match = parser.OFPMatch(
            eth_src=eth.src,
            eth_dst=eth.dst
        )
        
        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        
        # Add flow with appropriate priority and timeout
        self.add_flow(datapath, priority, match, actions, timeout)

        # Send packet out
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=msg.match['in_port'],
            actions=actions,
            data=msg.data
        )
        datapath.send_msg(out) 