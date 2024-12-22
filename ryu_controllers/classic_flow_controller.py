from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
from collections import deque
import time

class ClassicFlowController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, mechanism="FIFO", table_size=100, **kwargs):
        super(ClassicFlowController, self).__init__(*args, **kwargs)
        self.mechanism = mechanism
        self.table_size = table_size
        self.flow_table = deque(maxlen=table_size) if mechanism == "FIFO" else []
        self.flow_timestamps = {}  # For LRU
        self.mac_to_port = {}
        self.logger.info(f"Using {mechanism} flow management mechanism")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                        ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                           actions)]
        
        mod = parser.OFPFlowMod(datapath=datapath,
                               priority=priority,
                               match=match,
                               instructions=inst,
                               hard_timeout=timeout)
        datapath.send_msg(mod)

    def remove_flow(self, datapath, match):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            match=match
        )
        datapath.send_msg(mod)

    def manage_flow_table(self, datapath, new_flow):
        if self.mechanism == "FIFO":
            if len(self.flow_table) >= self.table_size:
                old_flow = self.flow_table.popleft()
                self.remove_flow(datapath, old_flow['match'])
            self.flow_table.append(new_flow)

        elif self.mechanism == "LRU":
            if len(self.flow_table) >= self.table_size:
                # Find least recently used flow
                lru_time = float('inf')
                lru_flow = None
                for flow in self.flow_table:
                    flow_key = str(flow['match'])
                    if self.flow_timestamps[flow_key] < lru_time:
                        lru_time = self.flow_timestamps[flow_key]
                        lru_flow = flow
                
                # Remove LRU flow
                self.flow_table.remove(lru_flow)
                self.remove_flow(datapath, lru_flow['match'])
                del self.flow_timestamps[str(lru_flow['match'])]

            self.flow_table.append(new_flow)
            self.flow_timestamps[str(new_flow['match'])] = time.time()

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            
            new_flow = {
                'match': match,
                'actions': actions
            }

            # Manage flow table using the selected mechanism
            self.manage_flow_table(datapath, new_flow)
            
            # Install new flow
            self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                 in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out) 